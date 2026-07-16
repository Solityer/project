#include "gatzk/protocol/quotients.hpp"

#include <chrono>
#include <cctype>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "gatzk/protocol/challenges.hpp"
#include "gatzk/protocol/schema.hpp"

namespace gatzk::protocol
{
    // 实现各工作域 quotient 恒等式的求值逻辑，
    // prove 会在hidden_point或挑战点上构造相应承诺，verify 会按同一公式重算并检查 opening。
    namespace
    {

        using algebra::FieldElement;
        using Clock = std::chrono::steady_clock;

        double elapsed_ms(const Clock::time_point &start, const Clock::time_point &end)
        {
            return std::chrono::duration<double, std::milli>(end - start).count();
        }

        std::vector<FieldElement> powers(const FieldElement &base, std::size_t max_exponent)
        {
            std::vector<FieldElement> out(max_exponent + 1, FieldElement::one());
            for (std::size_t i = 1; i <= max_exponent; ++i)
            {
                out[i] = out[i - 1] * base;
            }
            return out;
        }

        FieldElement lagrange_first(const algebra::RootOfUnityDomain &domain, const FieldElement &z)
        {
            return domain.lagrange_basis_eval(0, z);
        }

        FieldElement lagrange_last(const algebra::RootOfUnityDomain &domain, const FieldElement &z)
        {
            return domain.lagrange_basis_eval(domain.size - 1, z);
        }

        FieldElement lagrange_at(const algebra::RootOfUnityDomain &domain, std::size_t index, const FieldElement &z)
        {
            return domain.lagrange_basis_eval(index, z);
        }

        // 单个 lookup 链的标准 LogUp 约束项，供 feat / range / exp / ELU 等表复用。
        FieldElement c_lookup_1(
            const EvalFn &eval,
            const std::string &prefix,
            const FieldElement &beta,
            const FieldElement &z,
            const FieldElement &omega_z)
        {
            return (eval("P_R_" + prefix, omega_z) - eval("P_R_" + prefix, z)) * (eval("P_Table_" + prefix, z) + beta) * (eval("P_Query_" + prefix, z) + beta) - eval("P_Q_tbl_" + prefix, z) * eval("P_m_" + prefix, z) * (eval("P_Query_" + prefix, z) + beta) + eval("P_Q_qry_" + prefix, z) * (eval("P_Table_" + prefix, z) + beta);
        }

        bool starts_with(const std::string &value, std::string_view prefix)
        {
            return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
        }

        std::optional<std::string> hidden_head_suffix(const std::string &label)
        {
            if (!starts_with(label, "P_h"))
            {
                return std::nullopt;
            }
            std::size_t pos = 3;
            while (pos < label.size() && std::isdigit(static_cast<unsigned char>(label[pos])))
            {
                ++pos;
            }
            if (pos >= label.size() || label[pos] != '_')
            {
                return std::nullopt;
            }
            return label.substr(pos + 1);
        }

        std::optional<std::string> output_head_suffix(const std::string &label)
        {
            if (!starts_with(label, "P_out"))
            {
                return std::nullopt;
            }
            std::size_t pos = 5;
            while (pos < label.size() && std::isdigit(static_cast<unsigned char>(label[pos])))
            {
                ++pos;
            }
            if (pos >= label.size() || label[pos] != '_')
            {
                return std::nullopt;
            }
            return label.substr(pos + 1);
        }

        bool hidden_concat_binding_label(const ProtocolContext &context, const std::string &label, std::string_view suffix)
        {
            for (std::size_t layer_index = 0; layer_index < context.model.hidden_layers.size(); ++layer_index)
            {
                const bool is_final_layer = layer_index + 1 == context.model.hidden_layers.size();
                if (label == hidden_layer_cat_prefix(layer_index, is_final_layer) + std::string(suffix))
                {
                    return true;
                }
            }
            return false;
        }

        // bias 在 quotient 中只以量化后域元素形式出现，这里与 trace / verify 保持同一规则。
        FieldElement quantize_bias_value(double value)
        {
            const auto scaled = value >= 0.0 ? value * 16.0 + 0.5 : value * 16.0 - 0.5;
            return FieldElement::from_signed(static_cast<std::int64_t>(scaled));
        }

        FieldElement bias_fold_vector(
            const std::vector<double> &bias,
            std::size_t node_count,
            const FieldElement &y_out)
        {
            if (bias.empty() || node_count == 0)
            {
                return FieldElement::zero();
            }

            FieldElement column_fold = FieldElement::zero();
            FieldElement column_power = FieldElement::one();
            for (std::size_t col = 0; col < bias.size(); ++col)
            {
                column_fold += quantize_bias_value(bias[col]) * column_power;
                column_power *= y_out;
            }
            const auto row_stride = y_out.pow(static_cast<std::uint64_t>(bias.size()));
            FieldElement row_sum = FieldElement::zero();
            if (row_stride == FieldElement::one())
            {
                row_sum = FieldElement(static_cast<std::uint64_t>(node_count));
            }
            else
            {
                row_sum = (row_stride.pow(static_cast<std::uint64_t>(node_count)) - FieldElement::one()) / (row_stride - FieldElement::one());
            }
            return column_fold * row_sum;
        }

        std::optional<std::string> dynamic_label_domain_name(const ProtocolContext &context, const std::string &label)
        {
            if (!context.model.has_real_multihead)
            {
                return std::nullopt;
            }
            if (label == "P_H" || label == "P_Table_feat" || label == "P_Query_feat"
                || label == "P_m_feat" || label == "P_J_tbl_feat"
                || label == "P_J_qry_feat" || label == "P_R_feat")
            {
                return std::string("FH");
            }
            if (label == "P_Y_lin" || label == "P_Y")
                return std::string("coef");
            for (std::size_t head = 0; head < context.model.output_layer.heads.size(); ++head)
            {
                if (label == output_y_lin_label(head, false)
                    || label == output_y_label(head, false))
                    return std::string("coef");
                for (const auto *suffix : {"Table_can", "m_can", "J_can", "R_can"})
                {
                    if (label == output_bias_view_label(head, suffix))
                        return std::string("C");
                }
                for (const auto *suffix : {"Query", "J_qry", "R_qry"})
                {
                    if (label == output_bias_view_label(head, suffix))
                        return std::string("edge");
                }
            }
            for (std::size_t layer_index = 0; layer_index < context.model.hidden_layers.size(); ++layer_index)
            {
                const bool is_final_layer = layer_index + 1 == context.model.hidden_layers.size();
                if (label == hidden_layer_concat_label(layer_index, is_final_layer))
                    return std::string("coef");
                if (label == hidden_layer_concat_star_label(layer_index, is_final_layer))
                {
                    return std::string("N");
                }
            }
            if (hidden_concat_binding_label(context, label, "_a") || hidden_concat_binding_label(context, label, "_b") || hidden_concat_binding_label(context, label, "_Acc"))
            {
                return std::string("cat");
            }
            if (const auto suffix = output_head_suffix(label); suffix.has_value())
            {
                const auto &name = *suffix;
                if (name == "Y_prime") return std::string("coef");
                if (name == "ScoreMask_src" || name == "ScoreMask_dst")
                    return std::string("N");
                if (name.rfind("J_tbl_", 0) == 0 || name.rfind("J_qry_", 0) == 0)
                {
                    return std::string("edge");
                }
                if (name == "a_proj" || name == "b_proj" || name == "Acc_proj")
                {
                    return std::string("cat");
                }
                if (name == "ModelBind_proj_R") return std::string("cat");
                if (name == "ModelBind_src_R" || name == "ModelBind_dst_R") return std::string("C");
                if (name == "a_src" || name == "b_src" || name == "Acc_src" || name == "a_dst" || name == "b_dst" || name == "Acc_dst" || name == "a_y" || name == "b_y" || name == "Acc_y")
                {
                    return std::string("C");
                }
                if (name == "E_src" || name == "E_dst" || name == "M" || name == "Sum" || name == "inv" || name == "Y_prime_star" || name == "Table_src" || name == "m_src" || name == "J_src_node" || name == "R_src_node" || name == "Y_star" || name == "T" || name == "Table_dst" || name == "m_dst" || name == "J_dst_node" || name == "R_dst_node" || name == "Table_t" || name == "m_t" || name == "J_t_node" || name == "R_t_node")
                {
                    return std::string("N");
                }
                if (name.rfind("Div_", 0) == 0 || name.rfind("Bind_", 0) == 0 || name == "Y_pre_flat" || name == "Y_pre_field_flat" || name == "Y_agg_num_flat" || name == "Bias_flat" || name == "Y_flat" || name == "E_src_edge" || name == "E_dst_edge" || name == "Query_src" || name == "J_src_edge" || name == "R_src" || name == "S" || name == "Z" || name == "Table_L" || name == "Query_L" || name == "m_L" || name == "R_L" || name == "M_edge" || name == "Delta" || name == "U" || name == "Sum_flat" || name == "s_max" || name == "C_max" || name == "Table_R" || name == "Query_R" || name == "m_R" || name == "R_R" || name == "Sum_edge" || name == "inv_edge" || name == "alpha" || name == "Table_exp" || name == "Query_exp" || name == "m_exp" || name == "R_exp" || name == "Y_prime_star_edge" || name == "widehat_y_star" || name == "w" || name == "T_edge" || name == "PSQ" || name == "Y_star_edge" || name == "Query_dst" || name == "J_dst_edge" || name == "R_dst" || name == "Query_t" || name == "J_t_edge" || name == "R_t")
                {
                    return std::string("edge");
                }
            }
            if (label == "P_H_cat_star")
            {
                return std::string("N");
            }
            if (label == "P_cat_a" || label == "P_cat_b" || label == "P_cat_Acc")
            {
                return std::string("cat");
            }
            if (const auto suffix = hidden_head_suffix(label); suffix.has_value())
            {
                const auto &name = *suffix;
                if (name == "H_prime" || name == "H_agg_pre" || name == "H_agg")
                    return std::string("coef");
                if (name == "ScoreMask_src" || name == "ScoreMask_dst")
                    return std::string("N");
                if (name.rfind("J_tbl_", 0) == 0 || name.rfind("J_qry_", 0) == 0)
                {
                    return std::string("edge");
                }
                if (name == "a_proj" || name == "b_proj" || name == "Acc_proj")
                {
                    return std::string("in");
                }
                if (name == "ModelBind_proj_R") return std::string("in");
                if (name == "ModelBind_src_R" || name == "ModelBind_dst_R") return std::string("d");
                if (name == "a_src" || name == "b_src" || name == "Acc_src" || name == "a_dst" || name == "b_dst" || name == "Acc_dst" || name == "a_star" || name == "b_star" || name == "Acc_star" || name == "a_agg_pre" || name == "b_agg_pre" || name == "Acc_agg_pre" || name == "a_agg" || name == "b_agg" || name == "Acc_agg")
                {
                    return std::string("d");
                }
                if (name == "E_src" || name == "E_dst" || name == "M" || name == "Sum" || name == "inv" || name == "H_star" || name == "Table_src" || name == "m_src" || name == "J_src_node" || name == "R_src_node" || name == "H_agg_pre_star" || name == "T_psq" || name == "H_agg_star" || name == "Table_dst" || name == "m_dst" || name == "J_dst_node" || name == "R_dst_node" || name == "Table_t" || name == "m_t" || name == "J_t_node" || name == "R_t_node")
                {
                    return std::string("N");
                }
                if (name.rfind("Div_", 0) == 0 || name == "E_src_edge" || name == "E_dst_edge" || name == "H_src_star_edge" || name == "Query_src" || name == "J_src_edge" || name == "R_src" || name == "S" || name == "Z" || name == "Table_L" || name == "Query_L" || name == "m_L" || name == "R_L" || name == "M_edge" || name == "Delta" || name == "U" || name == "Sum_flat" || name == "Sum_edge" || name == "s_max" || name == "C_max" || name == "Table_R" || name == "Query_R" || name == "m_R" || name == "R_R" || name == "inv_edge" || name == "alpha" || name == "H_agg_pre_star_edge" || name == "H_agg_pre_flat" || name == "H_agg_num_flat" || name == "H_ELU_in_flat" || name == "H_agg_flat" || name == "Table_exp" || name == "Query_exp" || name == "m_exp" || name == "R_exp" || name == "Table_ELU" || name == "Query_ELU" || name == "m_ELU" || name == "R_ELU" || name == "widehat_v_pre_star" || name == "w_psq" || name == "T_psq_edge" || name == "PSQ" || name == "H_agg_star_edge" || name == "Query_dst" || name == "J_dst_edge" || name == "R_dst" || name == "Query_t" || name == "J_t_edge" || name == "R_t")
                {
                    return std::string("edge");
                }
            }
            if (label.rfind("P_avg_Div_", 0) == 0)
            {
                return std::string("edge");
            }
            if (label.rfind("P_avg_Bind_", 0) == 0)
            {
                return std::string("edge");
            }
            return std::nullopt;
        }

        class QuotientAccumulator
        {
        public:
            explicit QuotientAccumulator(
                const FieldElement &alpha,
                bool track_first_nonzero = false)
                : alpha_(alpha.native()),
                  track_first_nonzero_(track_first_nonzero)
            {
                power_ = 1;
                sum_.clear();
            }

            void add(const FieldElement &term)
            {
                if (track_first_nonzero_
                    && !first_nonzero_term_.has_value()
                    && !term.is_zero())
                {
                    first_nonzero_term_ = term_count_;
                    first_nonzero_value_ = term;
                }
                mcl::Fr product;
                mcl::Fr::mul(product, power_, term.native());
                mcl::Fr::add(sum_, sum_, product);
                mcl::Fr::mul(power_, power_, alpha_);
                ++term_count_;
            }

            FieldElement value() const
            {
                return FieldElement::from_native(sum_);
            }

            std::optional<std::size_t> first_nonzero_term() const
            {
                return first_nonzero_term_;
            }

            const FieldElement &first_nonzero_value() const
            {
                return first_nonzero_value_;
            }

        private:
            mcl::Fr alpha_;
            mcl::Fr power_;
            mcl::Fr sum_;
            bool track_first_nonzero_ = false;
            std::size_t term_count_ = 0;
            std::optional<std::size_t> first_nonzero_term_;
            FieldElement first_nonzero_value_ = FieldElement::zero();
        };

        FieldElement route_node_step(
            const FieldElement &next_value,
            const FieldElement &current_value,
            const FieldElement &table_value,
            const FieldElement &beta,
            const FieldElement &q_valid,
            const FieldElement &multiplicity)
        {
            return (next_value - current_value) * (table_value + beta) - q_valid * multiplicity;
        }

        FieldElement route_edge_step(
            const FieldElement &next_value,
            const FieldElement &current_value,
            const FieldElement &query_value,
            const FieldElement &beta,
            const FieldElement &q_edge_valid)
        {
            return (next_value - current_value) * (query_value + beta) - q_edge_valid;
        }

        FieldElement acc_step(
            const FieldElement &next_value,
            const FieldElement &current_value,
            const FieldElement &a_value,
            const FieldElement &b_value)
        {
            return next_value - current_value - a_value * b_value;
        }

        FieldElement psq_step(
            const FieldElement &psq_next,
            const FieldElement &psq_current,
            const FieldElement &w_next,
            const FieldElement &q_valid_next,
            const FieldElement &q_new_next)
        {
            return psq_next - ((FieldElement::one() - q_valid_next) * psq_current + q_valid_next * (q_new_next * w_next + (FieldElement::one() - q_new_next) * (psq_current + w_next)));
        }

        FieldElement lookup_step(
            const FieldElement &r_next,
            const FieldElement &r_current,
            const FieldElement &table_value,
            const FieldElement &query_value,
            const FieldElement &multiplicity,
            const FieldElement &q_tbl,
            const FieldElement &q_qry,
            const FieldElement &beta)
        {
            return (r_next - r_current) * (table_value + beta) * (query_value + beta) - q_tbl * multiplicity * (query_value + beta) + q_qry * (table_value + beta);
        }

        FieldElement zero_on_unselected(const FieldElement &selector, const FieldElement &value)
        {
            return (FieldElement::one() - selector) * value;
        }

    }

    std::optional<std::string> dynamic_commitment_domain_name(
        const ProtocolContext &context,
        const std::string &label)
    {
        if (label.ends_with("_ExternalMask"))
            return std::string("N");
        return dynamic_label_domain_name(context, label);
    }

    FieldElement evaluate_geometric_power_column(
        const algebra::RootOfUnityDomain &domain,
        const FieldElement &gamma,
        const FieldElement &point)
    {
        if (const auto index = domain.rotation_shift(
                FieldElement::one(), point);
            index.has_value())
            return gamma.pow(*index);

        const auto weights = domain.barycentric_weights(point);
        auto value = FieldElement::zero();
        auto gamma_power = FieldElement::one();
        for (const auto &weight : weights)
        {
            value += weight * gamma_power;
            gamma_power *= gamma;
        }
        return value;
    }

    std::vector<std::string> domain_opening_labels(const ProtocolContext &context, const std::string &domain_name)
    {
        // opening bundle 的列集合由标签到工作域的映射唯一决定。
        // 这既约束 prover / verifier 的 bundle 形状，也直接影响 proof_size_bytes。
        std::vector<std::string> labels;
        for (const auto &label : dynamic_commitment_labels(context))
        {
            if (label.find("_ScoreMask_") != std::string::npos
                || label.ends_with("_ExternalMask"))
                continue;
            if (const auto mapped = dynamic_label_domain_name(context, label); mapped.has_value() && *mapped == domain_name)
            {
                labels.push_back(label);
            }
        }
        // Pi_bind-owned indexed-view claims share the same domain point groups
        // and witnesses as ordinary trace claims, but their values are encoded
        // by Eval_bind rather than duplicated in Eval_dom.
        for (std::size_t head = 0; head < context.model.output_layer.heads.size(); ++head)
        {
            if (domain_name == "C")
                labels.push_back(model_proof_local_label(
                    output_bias_label(head, false)));
            const auto suffixes = domain_name == "C"
                ? std::vector<std::string>{"Table_can", "m_can", "J_can", "R_can"}
                : (domain_name == "edge"
                    ? std::vector<std::string>{"Query", "J_qry", "R_qry"}
                    : std::vector<std::string>{});
            for (const auto &suffix : suffixes)
                labels.push_back(output_bias_view_label(head, suffix));
            if (domain_name == "edge")
            {
                const auto metadata = build_binding_relation_metadata(context);
                const auto relation_id =
                    "bind:output:" + std::to_string(head);
                const auto relation = std::find_if(
                    metadata.begin(), metadata.end(),
                    [&](const BindingRelationMeta &candidate)
                    { return candidate.relation_id == relation_id; });
                if (relation == metadata.end()
                    || relation->coefficient_basis_relations.size() != 1)
                    throw std::runtime_error(
                        "domain opening registry is missing coefficient basis metadata");
                const auto &basis =
                    relation->coefficient_basis_relations.front();
                for (const auto &chunk : basis.chunks)
                {
                    labels.push_back(chunk.object_id);
                    labels.push_back(chunk.accumulator_object);
                }
            }
        }
        if (domain_name == context.domains.n->name)
        {
            for (const auto &binding : build_binding_relation_metadata(context))
                for (const auto &matrix : binding.canonical_matrices)
                {
                    if (matrix.canonical_domain != domain_name)
                        throw std::runtime_error(
                            "canonical matrix registered on unsupported domain: "
                            + matrix.relation_id);
                    labels.insert(
                        labels.end(), matrix.column_objects.begin(),
                        matrix.column_objects.end());
                    labels.push_back(matrix.compressed_object);
                }
        }
        for (const auto &binding : build_binding_relation_metadata(context))
        {
            for (const auto &relation : binding.aggregation_relations)
            {
                if (domain_name == relation.node_domain)
                {
                    labels.push_back(relation.source_node_object);
                    labels.push_back(relation.target_node_object);
                    for (const auto *route : {
                             &relation.source_route, &relation.target_route})
                    {
                        labels.push_back(route->table_object);
                        labels.push_back(route->node_inverse_object);
                        labels.push_back(route->node_accumulator_object);
                    }
                }
                if (domain_name == relation.edge_domain)
                {
                    for (const auto *object : {
                             &relation.source_edge_object,
                             &relation.target_edge_object,
                             &relation.product_object,
                             &relation.psq_object})
                        labels.push_back(*object);
                    for (const auto *route : {
                             &relation.source_route, &relation.target_route})
                    {
                        labels.push_back(route->query_object);
                        labels.push_back(route->edge_inverse_object);
                        labels.push_back(route->edge_accumulator_object);
                    }
                }
            }
        }
        return labels;
    }

    FieldElement evaluate_t_fh(
        const ProtocolContext &context,
        const std::map<std::string, FieldElement> &challenges,
        const EvalFn &eval,
        const FieldElement &z,
        FHQuotientProfile *profile)
    {
        const auto total_start = Clock::now();
        const auto dependency_before = profile != nullptr ? profile->dependency_eval_ms : 0.0;
        auto profiled_eval = [&](const std::string &name, const FieldElement &point)
        {
            const auto eval_start = Clock::now();
            const auto value = eval(name, point);
            if (profile != nullptr)
            {
                profile->dependency_eval_ms += elapsed_ms(eval_start, Clock::now());
            }
            return value;
        };
        const auto eta_feat = challenges.at("eta_feat");
        const auto omega_z = z * context.domains.fh->omega;
        const auto l0 = lagrange_first(*context.domains.fh, z);
        const auto l_last = lagrange_last(*context.domains.fh, z);
        const auto not_last = FieldElement::one() - l_last;
        const auto zero_eval = context.domains.fh->zero_polynomial_eval(z);
        const auto q_table = profiled_eval("P_Q_tbl_feat", z);
        const auto q_query = profiled_eval("P_Q_qry_feat", z);
        const auto table = profiled_eval("P_Table_feat", z);
        const auto query = profiled_eval("P_Query_feat", z);
        const auto multiplicity = profiled_eval("P_m_feat", z);
        const auto j_table = profiled_eval("P_J_tbl_feat", z);
        const auto j_query = profiled_eval("P_J_qry_feat", z);
        const auto accumulator = profiled_eval("P_R_feat", z);
        const auto accumulator_next = profiled_eval("P_R_feat", omega_z);
        const auto beta = challenges.at("beta_feat");
        const auto one = FieldElement::one();
        QuotientAccumulator acc(challenges.at("alpha_quot"));
        acc.add(l0 * accumulator);
        acc.add(not_last * (
            accumulator_next - accumulator
            - q_table * multiplicity * j_table
            + q_query * j_query));
        acc.add(l_last * accumulator);
        acc.add(q_table * ((table + beta) * j_table - one));
        acc.add((one - q_table) * j_table);
        acc.add(q_query * ((query + beta) * j_query - one));
        acc.add((one - q_query) * j_query);
        acc.add(
            table - (profiled_eval("P_Row_feat_tbl", z) + eta_feat * profiled_eval("P_Col_feat_tbl", z) + eta_feat.pow(2) * profiled_eval("P_T_H", z)));
        acc.add(
            query - (profiled_eval("P_I_feat_qry", z) + eta_feat * profiled_eval("P_Col_feat_qry", z) + eta_feat.pow(2) * profiled_eval("P_H", z)));
        const auto result = acc.value() / zero_eval;
        if (profile != nullptr)
        {
            const auto total_ms = elapsed_ms(total_start, Clock::now());
            profile->assembly_ms += total_ms - (profile->dependency_eval_ms - dependency_before);
        }
        return result;
    }

    FieldElement evaluate_t_edge(
        const ProtocolContext &context,
        const std::map<std::string, FieldElement> &challenges,
        const std::map<std::string, FieldElement> &witness_scalars,
        const std::map<std::string, FieldElement> &external_evaluations,
        const EvalFn &eval,
        const FieldElement &z,
        bool return_numerator,
        std::size_t *first_nonzero_term,
        FieldElement *first_nonzero_value)
    {
        if (context.model.has_real_multihead)
        {
            // edge 域承载大部分“边上对象”约束：softmax 链、lookup、route 和 PSQ 都在这里闭合。
            const auto omega_z = z * context.domains.edge->omega;
            const auto first = lagrange_first(*context.domains.edge, z);
            const auto last = lagrange_last(*context.domains.edge, z);
            const auto not_last = FieldElement::one() - last;
            const auto zero_eval = context.domains.edge->zero_polynomial_eval(z);
            const auto one = FieldElement::one();
            auto eval_z = [&](const std::string &label)
            {
                return eval(label, z);
            };
            auto eval_omega = [&](const std::string &label)
            {
                return eval(label, omega_z);
            };

            const auto q_edge = eval_z("P_Q_edge_valid");
            const auto q_edge_next = eval_omega("P_Q_edge_valid");
            const auto q_new_next = eval_omega("P_Q_grp_new_edge");
            const auto q_end = eval_z("P_Q_grp_end_edge");
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
            QuotientAccumulator acc(
                challenges.at("alpha_quot"), return_numerator);
            auto add_signed_division_constraints = [&](
                const std::string &prefix,
                const std::string &public_selector_label)
            {
                const auto selector = eval_z(public_selector_label);
                auto value = [&](std::string_view suffix)
                {
                    return eval_z(prefix + std::string(suffix));
                };
                const auto active = value("Active");
                const auto x = value("X");
                const auto d = value("D");
                const auto y = value("Y");
                const auto sigma = value("Sigma");
                const auto zero = value("Zero");
                const auto a = value("A");
                const auto a_inv = value("A_inv");
                const auto d_inv = value("D_inv");
                const auto q = value("Q");
                const auto r = value("R");
                const auto delta = value("Delta");
                const auto sign = one - FieldElement(2) * sigma;

                acc.add(active - selector);
                acc.add(selector * sigma * (sigma - one));
                acc.add(selector * (x - sign * a));
                acc.add(selector * zero * (zero - one));
                acc.add(selector * zero * a);
                acc.add(selector * (a * a_inv - (one - zero)));
                acc.add(selector * zero * sigma);
                acc.add(selector * zero * a_inv);
                acc.add(selector * (d * d_inv - one));
                acc.add(selector * (y - sign * q));
                acc.add(selector * (FieldElement(2) * a + d - FieldElement(2) * d * q - r));
                acc.add(selector * (r + delta - FieldElement(2) * d + one));

                // Canonical padding is part of the public column-padding
                // registry: no auxiliary value may survive outside Q_div.
                for (const auto &item : {x, d, y, sigma, zero, a, a_inv, d_inv, q, r, delta})
                {
                    acc.add((one - selector) * item);
                }

                const auto limb_count = signed_division_limb_count(context.config.range_bits);
                const auto limb_base = FieldElement(std::uint64_t{1} << context.config.range_bits);
                for (const auto *value_name : {"A", "DMinus1", "Q", "R", "Delta"})
                {
                    const FieldElement source =
                        std::string_view(value_name) == "A" ? a
                        : std::string_view(value_name) == "DMinus1" ? d - one
                        : std::string_view(value_name) == "Q" ? q
                        : std::string_view(value_name) == "R" ? r
                        : delta;
                    auto reconstructed = FieldElement::zero();
                    auto place = FieldElement::one();
                    for (std::size_t limb = 0; limb < limb_count; ++limb)
                    {
                        const auto limb_label = signed_division_limb_label(
                            prefix, value_name, limb);
                        const auto limb_value = eval_z(limb_label);
                        reconstructed += place * limb_value;
                        place *= limb_base;

                        auto range_value = [&](std::string_view column)
                        {
                            return eval_z(signed_division_range_label(
                                prefix, value_name, limb, column));
                        };
                        auto range_omega = [&](std::string_view column)
                        {
                            return eval_omega(signed_division_range_label(
                                prefix, value_name, limb, column));
                        };
                        const auto table = range_value("Table");
                        const auto query = range_value("Query");
                        const auto multiplicity = range_value("m");
                        const auto j_table = range_value("J_tbl");
                        const auto j_query = range_value("J_qry");
                        const auto accumulator = range_value("R");
                        const auto accumulator_next = range_omega("R");
                        const auto beta = challenges.at(signed_division_beta_name(
                            prefix, value_name, limb));
                        acc.add(first * accumulator);
                        acc.add(not_last * (
                            accumulator_next - accumulator
                            - q_tbl_r_z * multiplicity * j_table
                            + selector * j_query));
                        acc.add(last * accumulator);
                        acc.add(q_tbl_r_z * ((table + beta) * j_table - one));
                        acc.add((one - q_tbl_r_z) * j_table);
                        acc.add(selector * ((query + beta) * j_query - one));
                        acc.add((one - selector) * j_query);
                        acc.add(table - t_range_z);
                        acc.add(query - limb_value);
                        acc.add((one - q_tbl_r_z) * table);
                        acc.add((one - q_tbl_r_z) * multiplicity);
                        acc.add((one - selector) * query);
                        acc.add((one - selector) * limb_value);
                    }
                    acc.add(selector * (source - reconstructed));
                }
            };
            auto add_logup_constraints = [&](
                const std::string &prefix,
                const std::string &suffix,
                const FieldElement &q_table,
                const FieldElement &q_query,
                const FieldElement &beta)
            {
                const auto table = eval_z(prefix + "Table_" + suffix);
                const auto query = eval_z(prefix + "Query_" + suffix);
                const auto multiplicity = eval_z(prefix + "m_" + suffix);
                const auto j_table = eval_z(prefix + "J_tbl_" + suffix);
                const auto j_query = eval_z(prefix + "J_qry_" + suffix);
                const auto accumulator = eval_z(prefix + "R_" + suffix);
                const auto accumulator_next = eval_omega(prefix + "R_" + suffix);
                acc.add(first * accumulator);
                acc.add(not_last * (
                    accumulator_next - accumulator
                    - q_table * multiplicity * j_table
                    + q_query * j_query));
                acc.add(last * accumulator);
                acc.add(q_table * ((table + beta) * j_table - one));
                acc.add((one - q_table) * j_table);
                acc.add(q_query * ((query + beta) * j_query - one));
                acc.add((one - q_query) * j_query);
            };
            auto add_coefficient_basis_binding = [&](
                const std::string &prefix,
                const std::string &public_selector_label,
                const std::string &source_label,
                const FieldElement &target)
            {
                const auto selector = eval_z(public_selector_label);
                const auto selector_next = eval_omega(public_selector_label);
                const auto a = eval_z(prefix + "a");
                const auto b = eval_z(prefix + "b");
                const auto b_next = eval_omega(prefix + "b");
                const auto current = eval_z(prefix + "Acc");
                const auto next = eval_omega(prefix + "Acc");
                const auto source = eval_z(source_label);
                acc.add(first * current);
                acc.add(first * (b - one));
                acc.add(not_last * selector_next
                    * (b_next - challenges.at("y_out") * b));
                acc.add(not_last * selector * (next - current - a * b));
                acc.add(not_last * (one - selector) * (next - current));
                acc.add(selector * (a - source));
                acc.add((one - selector) * a);
                acc.add((one - selector) * b);
                acc.add(last * (current - target));
            };

            for (std::size_t head_index = 0; head_index < context.model.hidden_heads.size(); ++head_index)
            {
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

                auto z_head = [&](std::string_view suffix)
                {
                    return eval_z(prefix + std::string(suffix));
                };
                auto omega_head = [&](std::string_view suffix)
                {
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
                const auto sum_flat_z = z_head("Sum_flat");
                const auto h_agg_pre_flat_z = z_head("H_agg_pre_flat");
                const auto h_agg_num_flat_z = z_head("H_agg_num_flat");
                const auto h_elu_in_flat_z = z_head("H_ELU_in_flat");
                const auto h_agg_flat_z = z_head("H_agg_flat");
                const auto r_dst_z = z_head("R_dst");
                const auto r_dst_omega = omega_head("R_dst");
                const auto query_dst_z = z_head("Query_dst");
                const auto sum_edge_z = z_head("Sum_edge");
                const auto h_agg_star_edge_z = z_head("H_agg_star_edge");

                acc.add(first * r_src_z);
                acc.add(q_edge * ((query_src_z + beta_src) * z_head("J_src_edge") - one));
                acc.add(not_last * route_edge_step(r_src_omega, r_src_z, query_src_z, beta_src, q_edge));
                acc.add(query_src_z - (src_z + eta_src * e_src_edge_z + eta_src_sq * h_src_star_edge_z));
                acc.add(zero_on_unselected(q_edge, query_src_z));

                acc.add(q_edge * (s_z - e_src_edge_z - e_dst_edge_z));
                add_logup_constraints(prefix, "L", q_tbl_l_z, q_qry_l_z, beta_l);
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
                    q_edge_next * (c_max_omega - q_new_next * s_max_omega - (one - q_new_next) * (c_max_z + s_max_omega)) + (one - q_edge_next) * (c_max_omega - c_max_z));
                acc.add(q_end * (c_max_z - one));

                add_logup_constraints(prefix, "R", q_tbl_r_z, q_qry_r_z, beta_r);
                acc.add(table_r_z - t_range_z);
                acc.add(query_r_z - delta_z);
                acc.add(zero_on_unselected(q_tbl_r_z, table_r_z));
                acc.add(zero_on_unselected(q_tbl_r_z, m_r_z));
                acc.add(zero_on_unselected(q_qry_r_z, query_r_z));

                acc.add(q_edge * (alpha_z - u_z * inv_edge_z));
                add_logup_constraints(prefix, "exp", q_tbl_exp_z, q_qry_exp_z, beta_exp);
                acc.add(table_exp_z - (t_exp_x_z + eta_exp * t_exp_y_z));
                acc.add(query_exp_z - (delta_z + eta_exp * u_z));
                acc.add(zero_on_unselected(q_tbl_exp_z, table_exp_z));
                acc.add(zero_on_unselected(q_tbl_exp_z, m_exp_z));
                acc.add(zero_on_unselected(q_qry_exp_z, query_exp_z));

                acc.add(first * (psq_z - q_edge * w_psq_z));
                acc.add(psq_step(psq_omega, psq_z, w_psq_omega, q_edge_next, q_new_next));
                acc.add(q_end * (psq_z - t_psq_edge_z));
                acc.add(first * r_t_z);
                acc.add(q_edge * ((query_t_z + beta_t) * z_head("J_t_edge") - one));
                acc.add(not_last * route_edge_step(r_t_omega, r_t_z, query_t_z, beta_t, q_edge));
                acc.add(query_t_z - (dst_z + eta_t * t_psq_edge_z));
                acc.add(zero_on_unselected(q_edge, query_t_z));

                add_logup_constraints(prefix, "ELU", q_tbl_elu_z, q_qry_elu_z, beta_elu);
                acc.add(table_elu_z - (t_elu_x_z + eta_elu * t_elu_y_z));
                acc.add(query_elu_z - (h_elu_in_flat_z + eta_elu * h_agg_flat_z));
                acc.add(zero_on_unselected(q_tbl_elu_z, table_elu_z));
                acc.add(zero_on_unselected(q_tbl_elu_z, m_elu_z));
                acc.add(zero_on_unselected(q_qry_elu_z, query_elu_z));

                acc.add(first * r_dst_z);
                acc.add(q_edge * ((query_dst_z + beta_dst) * z_head("J_dst_edge") - one));
                acc.add(not_last * route_edge_step(r_dst_omega, r_dst_z, query_dst_z, beta_dst, q_edge));
                acc.add(query_dst_z - (dst_z + eta_dst * e_dst_edge_z + eta_dst_sq * m_edge_z + eta_dst_cu * sum_edge_z + eta_dst_4 * inv_edge_z + eta_dst_5 * h_agg_star_edge_z));
                acc.add(zero_on_unselected(q_edge, query_dst_z));
                add_signed_division_constraints(
                    prefix + "Div_",
                    "P_Q_div_h" + std::to_string(head_index));
                const auto q_div = eval_z("P_Q_div_h" + std::to_string(head_index));
                acc.add(q_div * (h_agg_num_flat_z - sum_flat_z * h_agg_pre_flat_z));
                acc.add(q_div * (z_head("Div_X") - h_agg_num_flat_z));
                acc.add(q_div * (z_head("Div_D") - sum_flat_z));
                acc.add(q_div * (z_head("Div_Y") - h_elu_in_flat_z));
                for (const auto &padding_value : {
                         sum_flat_z, h_agg_pre_flat_z, h_agg_num_flat_z,
                         h_elu_in_flat_z, h_agg_flat_z})
                {
                    acc.add((one - q_div) * padding_value);
                }
            }

            const bool legacy_single_output = false;
            for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index)
            {
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

                auto z_out = [&](std::string_view suffix)
                {
                    return eval_z(prefix + std::string(suffix));
                };
                auto omega_out = [&](std::string_view suffix)
                {
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
                acc.add(q_edge * ((query_src_z + beta_src_out) * z_out("J_src_edge") - one));
                acc.add(not_last * route_edge_step(r_src_omega, r_src_z, query_src_z, beta_src_out, q_edge));
                acc.add(query_src_z - (src_z + eta_src_out * e_src_edge_z));
                acc.add(zero_on_unselected(q_edge, query_src_z));

                acc.add(q_edge * (s_z - e_src_edge_z - e_dst_edge_z));
                add_logup_constraints(prefix, "L", q_tbl_l_z, q_qry_l_z, beta_l_out);
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
                    q_edge_next * (c_max_omega - q_new_next * s_max_omega - (one - q_new_next) * (c_max_z + s_max_omega)) + (one - q_edge_next) * (c_max_omega - c_max_z));
                acc.add(q_end * (c_max_z - one));

                add_logup_constraints(prefix, "R", q_tbl_r_z, q_qry_r_z, beta_r_out);
                acc.add(table_r_z - t_range_z);
                acc.add(query_r_z - delta_z);
                acc.add(zero_on_unselected(q_tbl_r_z, table_r_z));
                acc.add(zero_on_unselected(q_tbl_r_z, m_r_z));
                acc.add(zero_on_unselected(q_qry_r_z, query_r_z));

                acc.add(q_edge * (alpha_z - u_z * inv_edge_z));
                add_logup_constraints(prefix, "exp", q_tbl_exp_z, q_qry_exp_z, beta_exp_out);
                acc.add(table_exp_z - (t_exp_x_z + eta_exp_out * t_exp_y_z));
                acc.add(query_exp_z - (delta_z + eta_exp_out * u_z));
                acc.add(zero_on_unselected(q_tbl_exp_z, table_exp_z));
                acc.add(zero_on_unselected(q_tbl_exp_z, m_exp_z));
                acc.add(zero_on_unselected(q_qry_exp_z, query_exp_z));

                acc.add(first * (psq_z - q_edge * w_z));
                acc.add(psq_step(psq_omega, psq_z, w_omega, q_edge_next, q_new_next));
                acc.add(q_end * (psq_z - t_edge_z));
                acc.add(first * r_t_z);
                acc.add(q_edge * ((query_t_z + beta_t_out) * z_out("J_t_edge") - one));
                acc.add(not_last * route_edge_step(r_t_omega, r_t_z, query_t_z, beta_t_out, q_edge));
                acc.add(query_t_z - (dst_z + eta_t_out * t_edge_z));
                acc.add(zero_on_unselected(q_edge, query_t_z));

                acc.add(first * r_dst_z);
                acc.add(q_edge * ((query_dst_z + beta_dst_out) * z_out("J_dst_edge") - one));
                acc.add(not_last * route_edge_step(r_dst_omega, r_dst_z, query_dst_z, beta_dst_out, q_edge));
                acc.add(query_dst_z - (dst_z + eta_dst_out * e_dst_edge_z + eta_dst_out_sq * m_edge_z + eta_dst_out_cu * sum_edge_z + eta_dst_out_4 * inv_edge_z + eta_dst_out_5 * y_star_edge_z));
                acc.add(zero_on_unselected(q_edge, query_dst_z));
                add_signed_division_constraints(
                    prefix + "Div_",
                    "P_Q_div_out" + std::to_string(head_index));
                const auto q_div = eval_z("P_Q_div_out" + std::to_string(head_index));
                const auto y_pre_flat = z_out("Y_pre_flat");
                const auto sum_flat = z_out("Sum_flat");
                const auto y_pre_field_flat = z_out("Y_pre_field_flat");
                const auto y_agg_num_flat = z_out("Y_agg_num_flat");
                const auto bias_flat = z_out("Bias_flat");
                const auto y_flat = z_out("Y_flat");
                acc.add(q_div * (y_agg_num_flat - sum_flat * y_pre_field_flat));
                acc.add(q_div * (z_out("Div_X") - y_agg_num_flat));
                acc.add(q_div * (z_out("Div_D") - FieldElement(16) * sum_flat));
                acc.add(q_div * (y_pre_flat - z_out("Div_Y")));
                acc.add(q_div * (y_flat - y_pre_flat - bias_flat));
                acc.add((one - q_div) * sum_flat);
                acc.add((one - q_div) * y_pre_field_flat);
                acc.add((one - q_div) * y_agg_num_flat);
                acc.add((one - q_div) * y_pre_flat);
                acc.add((one - q_div) * bias_flat);
                acc.add((one - q_div) * y_flat);
                add_coefficient_basis_binding(
                    prefix + "Bind_",
                    "P_Q_div_out" + std::to_string(head_index),
                    prefix + "Y_flat",
                    external_evaluations.at(output_external_eval_name("y", head_index, false)));
                const auto view_prefix = output_bias_view_prefix(head_index);
                const auto q_view = eval_z(
                    "P_Q_div_out" + std::to_string(head_index));
                const auto query_view = eval_z(view_prefix + "Query");
                const auto j_view = eval_z(view_prefix + "J_qry");
                const auto r_view = eval_z(view_prefix + "R_qry");
                const auto r_view_next = eval_omega(view_prefix + "R_qry");
                const auto beta_view = challenges.at(
                    output_bias_view_beta_label(head_index));
                const auto eta_view = challenges.at(
                    output_bias_view_eta_label(head_index));
                acc.add(first * r_view);
                acc.add(not_last * (
                    r_view_next - r_view - q_view * j_view));
                acc.add(q_view * ((query_view + beta_view) * j_view - one));
                acc.add((one - q_view) * j_view);
                acc.add(query_view - q_view * (
                    eval_z("P_BiasIndex_edge")
                    + eta_view * bias_flat));

                const auto basis_chunks = coefficient_chunk_layout(
                    context.local.num_classes
                        + registered_blinder_coefficient_count(
                            context,
                            output_bias_basis_label(head_index, "Coeff")),
                    context.domains.edge->size);
                const auto gamma = challenges.at(
                    output_bias_basis_gamma_label(head_index));
                const auto power_label = output_bias_basis_label(
                    head_index, "Powers");
                const auto power = eval_z(power_label);
                const auto power_next = eval_omega(power_label);
                acc.add(first * (power - one));
                acc.add(not_last * (power_next - gamma * power));
                for (const auto &chunk : basis_chunks)
                {
                    const auto chunk_value = eval_z(
                        output_bias_basis_chunk_label(
                            head_index, chunk.chunk_index));
                    const auto selector = [&](std::string_view region)
                    {
                        return eval_z(
                            output_bias_basis_region_selector_label(
                                head_index, chunk.chunk_index, region));
                    };
                    const auto q_active = selector("active");
                    const auto q_semantic = selector("semantic");
                    const auto q_gap = selector("gap");
                    const auto q_tail = selector("tail");
                    acc.add(q_semantic + q_gap + q_tail - q_active);
                    acc.add((one - q_active) * chunk_value);
                    acc.add(q_gap * chunk_value);
                    acc.add(q_semantic * (chunk_value - bias_flat));
                    const auto accumulator_label =
                        output_bias_basis_chunk_accumulator_label(
                            head_index, chunk.chunk_index);
                    const auto basis_accumulator = eval_z(accumulator_label);
                    const auto basis_accumulator_next =
                        eval_omega(accumulator_label);
                    acc.add(first * basis_accumulator);
                    acc.add(not_last * (
                        basis_accumulator_next - basis_accumulator
                        - q_active * power * chunk_value));
                }
            }

            for (const auto &binding :
                 build_binding_relation_metadata(context))
            {
                for (const auto &relation : binding.aggregation_relations)
                {
                    const auto source_edge = eval_z(
                        relation.source_edge_object);
                    const auto target_edge = eval_z(
                        relation.target_edge_object);
                    const auto product = eval_z(relation.product_object);
                    const auto product_next = eval_omega(
                        relation.product_object);
                    const auto psq = eval_z(relation.psq_object);
                    const auto psq_next = eval_omega(relation.psq_object);
                    acc.add(q_edge * (
                        product
                        - eval_z(relation.alpha_object) * source_edge));
                    acc.add(first * (psq - q_edge * product));
                    acc.add(not_last * psq_step(
                        psq_next, psq, product_next,
                        q_edge_next, q_new_next));
                    acc.add(q_end * (psq - target_edge));
                    acc.add(zero_on_unselected(q_edge, source_edge));
                    acc.add(zero_on_unselected(q_edge, target_edge));
                    acc.add(zero_on_unselected(q_edge, product));
                    for (const auto *route : {
                             &relation.source_route,
                             &relation.target_route})
                    {
                        const auto beta = challenges.at(
                            route->beta_challenge_label);
                        const auto eta = challenges.at(
                            route->eta_challenge_label);
                        const auto accumulator = eval_z(
                            route->edge_accumulator_object);
                        const auto accumulator_next = eval_omega(
                            route->edge_accumulator_object);
                        const auto query = eval_z(route->query_object);
                        const auto inverse = eval_z(
                            route->edge_inverse_object);
                        acc.add(first * accumulator);
                        acc.add(q_edge * (
                            (query + beta) * inverse - one));
                        acc.add((one - q_edge) * inverse);
                        acc.add(not_last * route_edge_step(
                            accumulator_next, accumulator, query, beta,
                            q_edge));
                        acc.add(query - (
                            eval_z(route->index_object)
                            + eta * eval_z(route->edge_value_object)));
                        acc.add(zero_on_unselected(q_edge, query));
                    }
                }
            }

            add_signed_division_constraints("P_avg_Div_", "P_Q_div_avg");
            {
                const auto q_avg = eval_z("P_Q_div_avg");
                auto summed_heads = FieldElement::zero();
                for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index)
                {
                    summed_heads += eval_z(
                        output_head_prefix(head_index, false) + "_Y_flat");
                }
                acc.add(q_avg * (eval_z("P_avg_Div_X") - summed_heads));
                acc.add(q_avg * (
                    eval_z("P_avg_Div_D")
                    - FieldElement(context.model.output_layer.heads.size())));
                add_coefficient_basis_binding(
                    "P_avg_Bind_",
                    "P_Q_div_avg",
                    "P_avg_Div_Y",
                    external_evaluations.at("mu_out"));
            }

            if (return_numerator && acc.first_nonzero_term().has_value())
            {
                if (first_nonzero_term != nullptr)
                    *first_nonzero_term = *acc.first_nonzero_term();
                if (first_nonzero_value != nullptr)
                    *first_nonzero_value = acc.first_nonzero_value();
            }
            return return_numerator ? acc.value() : acc.value() / zero_eval;
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
        const auto c_src_1 = eval("P_Q_qry_src", z) * ((eval("P_R_src", omega_z) - eval("P_R_src", z)) * (eval("P_Query_src", z) + beta_src) - FieldElement::one());
        const auto c_src_2 = (FieldElement::one() - eval("P_Q_qry_src", z)) * (eval("P_R_src", omega_z) - eval("P_R_src", z));
        const auto c_src_3 = l_last * (eval("P_R_src", z) - witness_scalars.at("S_src"));
        const auto c_src_bind = eval("P_Query_src", z) - (eval("P_src", z) + eta_src * eval("P_E_src_edge", z) + eta_src.pow(2) * eval("P_H_src_star_edge", z));

        const auto c_dst_0 = l0 * eval("P_R_dst", z);
        const auto c_dst_1 = eval("P_Q_qry_dst", z) * ((eval("P_R_dst", omega_z) - eval("P_R_dst", z)) * (eval("P_Query_dst", z) + beta_dst) - FieldElement::one());
        const auto c_dst_2 = (FieldElement::one() - eval("P_Q_qry_dst", z)) * (eval("P_R_dst", omega_z) - eval("P_R_dst", z));
        const auto c_dst_3 = l_last * (eval("P_R_dst", z) - witness_scalars.at("S_dst"));
        const auto c_dst_bind = eval("P_Query_dst", z) - (eval("P_dst", z) + eta_dst * eval("P_E_dst_edge", z) + eta_dst.pow(2) * eval("P_M_edge", z) + eta_dst.pow(3) * eval("P_Sum_edge", z) + eta_dst.pow(4) * eval("P_inv_edge", z) + eta_dst.pow(5) * eval("P_H_agg_star_edge", z));

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
            eval("P_Q_edge_valid", omega_z) * (eval("P_C_max", omega_z) - eval("P_Q_grp_new_edge", omega_z) * eval("P_s_max", omega_z) - (FieldElement::one() - eval("P_Q_grp_new_edge", omega_z)) * (eval("P_C_max", z) + eval("P_s_max", omega_z))) + (FieldElement::one() - eval("P_Q_edge_valid", omega_z)) * (eval("P_C_max", omega_z) - eval("P_C_max", z));
        const auto c_max_5 = eval("P_Q_grp_end_edge", z) * (eval("P_C_max", z) - FieldElement::one());

        const auto c_inv_1 = eval("P_alpha", z) - eval("P_U", z) * eval("P_inv_edge", z);
        const auto c_vhat_0 = eval("P_v_hat", z) - eval("P_alpha", z) * eval("P_H_src_star_edge", z);
        const auto c_psq_0 = l0 * (eval("P_PSQ", z) - eval("P_w_psq", z));
        const auto c_psq_1 =
            eval("P_PSQ", omega_z) - eval("P_Q_grp_new_edge", omega_z) * eval("P_w_psq", omega_z) - (FieldElement::one() - eval("P_Q_grp_new_edge", omega_z)) * (eval("P_PSQ", z) + eval("P_w_psq", omega_z));
        const auto c_psq_2 = eval("P_Q_grp_end_edge", z) * (eval("P_PSQ", z) - eval("P_T_psq_edge", z));

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
        return return_numerator ? n_edge : n_edge / zero_eval;
    }

    FieldElement evaluate_t_n(
        const ProtocolContext &context,
        const std::map<std::string, FieldElement> &challenges,
        const std::map<std::string, FieldElement> &witness_scalars,
        const EvalFn &eval,
        const FieldElement &z)
    {
        if (context.model.has_real_multihead)
        {
            // N 域负责把边上广播的量重新收回到目标点，并补上分母倒数、PSQ 目标表和 route 节点侧约束。
            const auto omega_z = z * context.domains.n->omega;
            const auto first = lagrange_first(*context.domains.n, z);
            const auto last = lagrange_last(*context.domains.n, z);
            const auto not_last = FieldElement::one() - last;
            const auto zero_eval = context.domains.n->zero_polynomial_eval(z);
            const auto q_n = eval("P_Q_N", z);
            QuotientAccumulator acc(challenges.at("alpha_quot"));

            for (std::size_t head_index = 0; head_index < context.model.hidden_heads.size(); ++head_index)
            {
                const auto prefix = "P_h" + std::to_string(head_index) + "_";
                const auto beta_src = challenges.at("beta_src_h" + std::to_string(head_index));
                const auto beta_dst = challenges.at("beta_dst_h" + std::to_string(head_index));
                const auto eta_src = challenges.at("eta_src_h" + std::to_string(head_index));
                const auto eta_dst = challenges.at("eta_dst_h" + std::to_string(head_index));
                const auto beta_t = challenges.at("beta_t_h" + std::to_string(head_index));
                const auto eta_t = challenges.at("eta_t_h" + std::to_string(head_index));
                const auto lambda_psq = challenges.at("lambda_psq_h" + std::to_string(head_index));

                acc.add(first * eval(prefix + "R_src_node", z));
                acc.add(q_n * ((eval(prefix + "Table_src", z) + beta_src) * eval(prefix + "J_src_node", z) - FieldElement::one()));
                acc.add(not_last * route_node_step(eval(prefix + "R_src_node", omega_z), eval(prefix + "R_src_node", z), eval(prefix + "Table_src", z), beta_src, q_n, eval(prefix + "m_src", z)));
                acc.add(first * eval(prefix + "R_dst_node", z));
                acc.add(q_n * ((eval(prefix + "Table_dst", z) + beta_dst) * eval(prefix + "J_dst_node", z) - FieldElement::one()));
                acc.add(not_last * route_node_step(eval(prefix + "R_dst_node", omega_z), eval(prefix + "R_dst_node", z), eval(prefix + "Table_dst", z), beta_dst, q_n, eval(prefix + "m_dst", z)));
                acc.add(first * eval(prefix + "R_t_node", z));
                acc.add(q_n * ((eval(prefix + "Table_t", z) + beta_t) * eval(prefix + "J_t_node", z) - FieldElement::one()));
                acc.add(not_last * route_node_step(eval(prefix + "R_t_node", omega_z), eval(prefix + "R_t_node", z), eval(prefix + "Table_t", z), beta_t, q_n, eval(prefix + "m_t", z)));
                acc.add(q_n * (eval(prefix + "Sum", z) * eval(prefix + "inv", z) - FieldElement::one()));
                acc.add(q_n * (eval(prefix + "T_psq", z) - eval(prefix + "Sum", z) - lambda_psq * eval(prefix + "H_agg_pre_star", z)));
                acc.add(eval(prefix + "Table_src", z) - (eval("P_I", z) + eta_src * eval(prefix + "E_src", z) + eta_src.pow(2) * eval(prefix + "H_star", z)));
                acc.add(eval(prefix + "Table_dst", z) - (eval("P_I", z) + eta_dst * eval(prefix + "E_dst", z) + eta_dst.pow(2) * eval(prefix + "M", z) + eta_dst.pow(3) * eval(prefix + "Sum", z) + eta_dst.pow(4) * eval(prefix + "inv", z) + eta_dst.pow(5) * eval(prefix + "H_agg_star", z)));
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

            const bool legacy_single_output = false;
            for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index)
            {
                const auto prefix = output_head_prefix(head_index, legacy_single_output) + "_";
                const auto beta_src_out = challenges.at(output_challenge_name("beta_src_out", head_index, legacy_single_output));
                const auto beta_dst_out = challenges.at(output_challenge_name("beta_dst_out", head_index, legacy_single_output));
                const auto eta_src_out = challenges.at(output_challenge_name("eta_src_out", head_index, legacy_single_output));
                const auto eta_dst_out = challenges.at(output_challenge_name("eta_dst_out", head_index, legacy_single_output));
                const auto beta_t_out = challenges.at(output_challenge_name("beta_t_out", head_index, legacy_single_output));
                const auto eta_t_out = challenges.at(output_challenge_name("eta_t_out", head_index, legacy_single_output));
                const auto lambda_out = challenges.at(output_challenge_name("lambda_out", head_index, legacy_single_output));

                acc.add(first * eval(prefix + "R_src_node", z));
                acc.add(q_n * ((eval(prefix + "Table_src", z) + beta_src_out) * eval(prefix + "J_src_node", z) - FieldElement::one()));
                acc.add(not_last * route_node_step(eval(prefix + "R_src_node", omega_z), eval(prefix + "R_src_node", z), eval(prefix + "Table_src", z), beta_src_out, q_n, eval(prefix + "m_src", z)));
                acc.add(first * eval(prefix + "R_dst_node", z));
                acc.add(q_n * ((eval(prefix + "Table_dst", z) + beta_dst_out) * eval(prefix + "J_dst_node", z) - FieldElement::one()));
                acc.add(not_last * route_node_step(eval(prefix + "R_dst_node", omega_z), eval(prefix + "R_dst_node", z), eval(prefix + "Table_dst", z), beta_dst_out, q_n, eval(prefix + "m_dst", z)));
                acc.add(first * eval(prefix + "R_t_node", z));
                acc.add(q_n * ((eval(prefix + "Table_t", z) + beta_t_out) * eval(prefix + "J_t_node", z) - FieldElement::one()));
                acc.add(not_last * route_node_step(eval(prefix + "R_t_node", omega_z), eval(prefix + "R_t_node", z), eval(prefix + "Table_t", z), beta_t_out, q_n, eval(prefix + "m_t", z)));
                acc.add(q_n * (eval(prefix + "Sum", z) * eval(prefix + "inv", z) - FieldElement::one()));
                acc.add(q_n * (eval(prefix + "T", z) - eval(prefix + "Sum", z) - lambda_out * eval(prefix + "Y_star", z)));
                acc.add(eval(prefix + "Table_src", z) - (eval("P_I", z) + eta_src_out * eval(prefix + "E_src", z)));
                acc.add(eval(prefix + "Table_dst", z) - (eval("P_I", z) + eta_dst_out * eval(prefix + "E_dst", z) + eta_dst_out.pow(2) * eval(prefix + "M", z) + eta_dst_out.pow(3) * eval(prefix + "Sum", z) + eta_dst_out.pow(4) * eval(prefix + "inv", z) + eta_dst_out.pow(5) * eval(prefix + "Y_star", z)));
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

            for (const auto &binding :
                 build_binding_relation_metadata(context))
            {
                for (const auto &matrix : binding.canonical_matrices)
                {
                    if (matrix.canonical_domain != context.domains.n->name
                        || matrix.column_objects.size()
                            != matrix.column_count
                        || matrix.row_count != context.local.num_nodes)
                        throw std::runtime_error(
                            "invalid canonical matrix quotient registry: "
                            + matrix.relation_id);
                    const auto compression = challenges.at(
                        matrix.compression_challenge_label);
                    auto expected = FieldElement::zero();
                    auto power = FieldElement::one();
                    for (const auto &column : matrix.column_objects)
                    {
                        expected += power * eval(column, z);
                        power *= compression;
                        acc.add(zero_on_unselected(
                            q_n, eval(column, z)));
                    }
                    acc.add(q_n * (
                        eval(matrix.compressed_object, z) - expected));
                }
                for (const auto &relation : binding.aggregation_relations)
                {
                    const auto find_matrix = [&](const std::string &object)
                        -> const CanonicalMatrixRelationMeta &
                    {
                        const auto it = std::find_if(
                            binding.canonical_matrices.begin(),
                            binding.canonical_matrices.end(),
                            [&](const auto &matrix)
                            { return matrix.matrix_object == object; });
                        if (it == binding.canonical_matrices.end())
                            throw std::runtime_error(
                                "aggregation quotient canonical matrix missing: "
                                + object);
                        return *it;
                    };
                    const auto compression = challenges.at(
                        relation.compression_challenge_label);
                    const auto compressed_columns = [&](
                        const CanonicalMatrixRelationMeta &matrix)
                    {
                        auto value = FieldElement::zero();
                        auto power = FieldElement::one();
                        for (const auto &column : matrix.column_objects)
                        {
                            value += power * eval(column, z);
                            power *= compression;
                        }
                        return value;
                    };
                    acc.add(q_n * (
                        eval(relation.source_node_object, z)
                        - compressed_columns(find_matrix(
                            relation.source_matrix_object))));
                    acc.add(q_n * (
                        eval(relation.target_node_object, z)
                        - compressed_columns(find_matrix(
                            relation.target_matrix_object))));
                    acc.add(zero_on_unselected(
                        q_n, eval(relation.source_node_object, z)));
                    acc.add(zero_on_unselected(
                        q_n, eval(relation.target_node_object, z)));
                    for (const auto *route : {
                             &relation.source_route,
                             &relation.target_route})
                    {
                        const auto beta = challenges.at(
                            route->beta_challenge_label);
                        const auto eta = challenges.at(
                            route->eta_challenge_label);
                        const auto accumulator = eval(
                            route->node_accumulator_object, z);
                        const auto accumulator_next = eval(
                            route->node_accumulator_object, omega_z);
                        const auto table = eval(route->table_object, z);
                        const auto inverse = eval(
                            route->node_inverse_object, z);
                        const auto multiplicity = eval(
                            route->multiplicity_object, z);
                        acc.add(first * accumulator);
                        acc.add(q_n * ((table + beta) * inverse
                                     - FieldElement::one()));
                        acc.add((FieldElement::one() - q_n) * inverse);
                        acc.add(not_last * route_node_step(
                            accumulator_next, accumulator, table, beta,
                            q_n, multiplicity));
                        acc.add(table - (
                            eval("P_AggBind_index_N", z)
                            + eta * eval(route->node_value_object, z)));
                        acc.add(zero_on_unselected(q_n, table));
                    }
                }
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
        const auto c_src_node_1 = q_n * ((eval("P_R_src_node", omega_z) - eval("P_R_src_node", z)) * (eval("P_Table_src", z) + beta_src) - eval("P_m_src", z));
        const auto c_src_node_2 = q_invalid * (eval("P_R_src_node", omega_z) - eval("P_R_src_node", z));
        const auto c_src_node_3 = l_last * (eval("P_R_src_node", z) - witness_scalars.at("S_src"));
        const auto c_src_node_bind = eval("P_Table_src", z) - (eval("P_I", z) + eta_src * eval("P_E_src", z) + eta_src.pow(2) * eval("P_H_star", z));

        const auto c_dst_node_0 = l0 * eval("P_R_dst_node", z);
        const auto c_dst_node_1 = q_n * ((eval("P_R_dst_node", omega_z) - eval("P_R_dst_node", z)) * (eval("P_Table_dst", z) + beta_dst) - eval("P_m_dst", z));
        const auto c_dst_node_2 = q_invalid * (eval("P_R_dst_node", omega_z) - eval("P_R_dst_node", z));
        const auto c_dst_node_3 = l_last * (eval("P_R_dst_node", z) - witness_scalars.at("S_dst"));
        const auto c_dst_node_bind = eval("P_Table_dst", z) - (eval("P_I", z) + eta_dst * eval("P_E_dst", z) + eta_dst.pow(2) * eval("P_M", z) + eta_dst.pow(3) * eval("P_Sum", z) + eta_dst.pow(4) * eval("P_inv", z) + eta_dst.pow(5) * eval("P_H_agg_star", z));

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
        const ProtocolContext &context,
        const std::map<std::string, FieldElement> &challenges,
        const std::map<std::string, FieldElement> &external_evaluations,
        const EvalFn &eval,
        const FieldElement &z,
        bool return_numerator,
        std::size_t *first_nonzero_term,
        FieldElement *first_nonzero_value)
    {
        if (context.model.has_real_multihead)
        {
            const auto omega_z = z * context.domains.in->omega;
            const auto first = lagrange_first(*context.domains.in, z);
            const auto last = lagrange_last(*context.domains.in, z);
            const auto not_last = FieldElement::one() - last;
            const auto zero_eval = context.domains.in->zero_polynomial_eval(z);
            const auto q_valid = eval("P_Q_proj_valid", z);
            const auto q_invalid = FieldElement::one() - q_valid;
            QuotientAccumulator acc(
                challenges.at("alpha_quot"), return_numerator);
            for (std::size_t head_index = 0; head_index < context.model.hidden_heads.size(); ++head_index)
            {
                const auto prefix = "P_h" + std::to_string(head_index) + "_";
                acc.add(first * eval(prefix + "Acc_proj", z));
                acc.add(q_valid * acc_step(eval(prefix + "Acc_proj", omega_z), eval(prefix + "Acc_proj", z), eval(prefix + "a_proj", z), eval(prefix + "b_proj", z)));
                acc.add(not_last * q_invalid * (eval(prefix + "Acc_proj", omega_z) - eval(prefix + "Acc_proj", z)));
                acc.add(zero_on_unselected(q_valid, eval(prefix + "a_proj", z)));
                acc.add(zero_on_unselected(q_valid, eval(prefix + "b_proj", z)));
                acc.add(last * (eval(prefix + "Acc_proj", z) - external_evaluations.at("mu_h" + std::to_string(head_index) + "_proj")));
            }
            if (return_numerator && acc.first_nonzero_term().has_value())
            {
                if (first_nonzero_term != nullptr)
                    *first_nonzero_term = *acc.first_nonzero_term();
                if (first_nonzero_value != nullptr)
                    *first_nonzero_value = acc.first_nonzero_value();
            }
            return return_numerator ? acc.value() : acc.value() / zero_eval;
        }

        const auto omega_z = z * context.domains.in->omega;
        const auto l0 = lagrange_first(*context.domains.in, z);
        const auto not_last = FieldElement::one() - lagrange_last(*context.domains.in, z);
        const auto ld = lagrange_at(*context.domains.in, context.local.num_features, z);
        const auto zero_eval = context.domains.in->zero_polynomial_eval(z);
        const auto c_proj_0 = l0 * eval("P_Acc_proj", z);
        const auto c_proj_1 = eval("P_Q_proj_valid", z) * (eval("P_Acc_proj", omega_z) - eval("P_Acc_proj", z) - eval("P_a_proj", z) * eval("P_b_proj", z));
        const auto c_proj_2 = (FieldElement::one() - eval("P_Q_proj_valid", z)) * not_last * (eval("P_Acc_proj", omega_z) - eval("P_Acc_proj", z));
        const auto c_proj_3 = ld * (eval("P_Acc_proj", z) - external_evaluations.at("mu_proj"));
        const auto alpha = challenges.at("alpha_quot");
        const auto alpha_powers = powers(alpha, 33);
        const auto numerator =
            alpha_powers[30] * c_proj_0 + alpha_powers[31] * c_proj_1 + alpha_powers[32] * c_proj_2 + alpha_powers[33] * c_proj_3;
        return numerator / zero_eval;
    }

    FieldElement evaluate_t_d(
        const ProtocolContext &context,
        const std::map<std::string, FieldElement> &challenges,
        const std::map<std::string, FieldElement> &external_evaluations,
        const EvalFn &eval,
        const FieldElement &z,
        bool return_numerator,
        std::size_t *first_nonzero_term,
        FieldElement *first_nonzero_value)
    {
        if (context.model.has_real_multihead)
        {
            const auto omega_z = z * context.domains.d->omega;
            const auto first = lagrange_first(*context.domains.d, z);
            const auto last = lagrange_last(*context.domains.d, z);
            const auto not_last = FieldElement::one() - last;
            const auto zero_eval = context.domains.d->zero_polynomial_eval(z);
            const auto q_valid = eval("P_Q_d_valid", z);
            const auto q_invalid = FieldElement::one() - q_valid;
            QuotientAccumulator acc(
                challenges.at("alpha_quot"), return_numerator);
            const auto semantic_external =
                [&](const std::string &eval_name,
                    const std::string &challenge_name)
            {
                return external_evaluations.at(eval_name)
                    - context.domains.n->zero_polynomial_eval(
                        challenges.at(challenge_name))
                        * external_evaluations.at(
                            external_mask_eval_name(eval_name));
            };

            for (std::size_t head_index = 0; head_index < context.model.hidden_heads.size(); ++head_index)
            {
                const auto prefix = "P_h" + std::to_string(head_index) + "_";
                const auto src_semantic_eval =
                    external_evaluations.at(
                        "mu_h" + std::to_string(head_index) + "_src")
                    - context.domains.n->zero_polynomial_eval(
                        challenges.at("y_src_h" + std::to_string(head_index)))
                        * external_evaluations.at(
                            hidden_score_mask_eval_name(head_index, "src"));
                const auto dst_semantic_eval =
                    external_evaluations.at(
                        "mu_h" + std::to_string(head_index) + "_dst")
                    - context.domains.n->zero_polynomial_eval(
                        challenges.at("y_dst_h" + std::to_string(head_index)))
                        * external_evaluations.at(
                            hidden_score_mask_eval_name(head_index, "dst"));
                acc.add(first * eval(prefix + "Acc_src", z));
                acc.add(q_valid * acc_step(eval(prefix + "Acc_src", omega_z), eval(prefix + "Acc_src", z), eval(prefix + "a_src", z), eval(prefix + "b_src", z)));
                acc.add(not_last * q_invalid * (eval(prefix + "Acc_src", omega_z) - eval(prefix + "Acc_src", z)));
                acc.add(zero_on_unselected(q_valid, eval(prefix + "a_src", z)));
                acc.add(zero_on_unselected(q_valid, eval(prefix + "b_src", z)));
                acc.add(last * (eval(prefix + "Acc_src", z) - src_semantic_eval));

                acc.add(first * eval(prefix + "Acc_dst", z));
                acc.add(q_valid * acc_step(eval(prefix + "Acc_dst", omega_z), eval(prefix + "Acc_dst", z), eval(prefix + "a_dst", z), eval(prefix + "b_dst", z)));
                acc.add(not_last * q_invalid * (eval(prefix + "Acc_dst", omega_z) - eval(prefix + "Acc_dst", z)));
                acc.add(zero_on_unselected(q_valid, eval(prefix + "a_dst", z)));
                acc.add(zero_on_unselected(q_valid, eval(prefix + "b_dst", z)));
                acc.add(last * (eval(prefix + "Acc_dst", z) - dst_semantic_eval));

                acc.add(first * eval(prefix + "Acc_star", z));
                acc.add(q_valid * acc_step(eval(prefix + "Acc_star", omega_z), eval(prefix + "Acc_star", z), eval(prefix + "a_star", z), eval(prefix + "b_star", z)));
                acc.add(not_last * q_invalid * (eval(prefix + "Acc_star", omega_z) - eval(prefix + "Acc_star", z)));
                acc.add(zero_on_unselected(q_valid, eval(prefix + "a_star", z)));
                acc.add(zero_on_unselected(q_valid, eval(prefix + "b_star", z)));
                acc.add(last * (eval(prefix + "Acc_star", z) - semantic_external(
                    "mu_h" + std::to_string(head_index) + "_star",
                    "y_star_h" + std::to_string(head_index))));

                acc.add(first * eval(prefix + "Acc_agg_pre", z));
                acc.add(q_valid * acc_step(eval(prefix + "Acc_agg_pre", omega_z), eval(prefix + "Acc_agg_pre", z), eval(prefix + "a_agg_pre", z), eval(prefix + "b_agg_pre", z)));
                acc.add(not_last * q_invalid * (eval(prefix + "Acc_agg_pre", omega_z) - eval(prefix + "Acc_agg_pre", z)));
                acc.add(zero_on_unselected(q_valid, eval(prefix + "a_agg_pre", z)));
                acc.add(zero_on_unselected(q_valid, eval(prefix + "b_agg_pre", z)));
                acc.add(last * (eval(prefix + "Acc_agg_pre", z) - semantic_external(
                    "mu_h" + std::to_string(head_index) + "_agg_pre",
                    "y_agg_pre_h" + std::to_string(head_index))));

                acc.add(first * eval(prefix + "Acc_agg", z));
                acc.add(q_valid * acc_step(eval(prefix + "Acc_agg", omega_z), eval(prefix + "Acc_agg", z), eval(prefix + "a_agg", z), eval(prefix + "b_agg", z)));
                acc.add(not_last * q_invalid * (eval(prefix + "Acc_agg", omega_z) - eval(prefix + "Acc_agg", z)));
                acc.add(zero_on_unselected(q_valid, eval(prefix + "a_agg", z)));
                acc.add(zero_on_unselected(q_valid, eval(prefix + "b_agg", z)));
                acc.add(last * (eval(prefix + "Acc_agg", z) - semantic_external(
                    "mu_h" + std::to_string(head_index) + "_agg",
                    "y_agg_h" + std::to_string(head_index))));
            }

            if (return_numerator && acc.first_nonzero_term().has_value())
            {
                if (first_nonzero_term != nullptr)
                    *first_nonzero_term = *acc.first_nonzero_term();
                if (first_nonzero_value != nullptr)
                    *first_nonzero_value = acc.first_nonzero_value();
            }
            return return_numerator ? acc.value() : acc.value() / zero_eval;
        }

        const auto omega_z = z * context.domains.d->omega;
        const auto q_valid = eval("P_Q_d_valid", z);
        const auto q_invalid = FieldElement::one() - q_valid;
        const auto l0 = lagrange_first(*context.domains.d, z);
        const auto not_last = FieldElement::one() - lagrange_last(*context.domains.d, z);
        const auto ld = lagrange_at(*context.domains.d, context.model.a_src.size(), z);
        const auto zero_eval = context.domains.d->zero_polynomial_eval(z);

        auto c0 = [&](const std::string &acc)
        { return l0 * eval(acc, z); };
        auto c1 = [&](const std::string &a, const std::string &b, const std::string &acc)
        {
            return q_valid * (eval(acc, omega_z) - eval(acc, z) - eval(a, z) * eval(b, z));
        };
        auto c2 = [&](const std::string &acc)
        { return not_last * q_invalid * (eval(acc, omega_z) - eval(acc, z)); };
        auto c3 = [&](const std::string &acc, const std::string &mu_name)
        {
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
        const ProtocolContext &context,
        const std::map<std::string, FieldElement> &challenges,
        const std::map<std::string, FieldElement> &external_evaluations,
        const EvalFn &eval,
        const FieldElement &z)
    {
        const auto omega_z = z * context.domains.cat->omega;
        const auto first = lagrange_first(*context.domains.cat, z);
        const auto last = lagrange_last(*context.domains.cat, z);
        const auto not_last = FieldElement::one() - last;
        const auto zero_eval = context.domains.cat->zero_polynomial_eval(z);
        const auto q_valid = eval("P_Q_cat_valid", z);
        const auto q_invalid = FieldElement::one() - q_valid;
        // cat 域只约束多头拼接与输出投影输入，不重复承担 edge 域上的 softmax/lookup 主链。
        QuotientAccumulator acc(challenges.at("alpha_quot"));

        for (std::size_t layer_index = 0; layer_index < context.model.hidden_layers.size(); ++layer_index)
        {
            const bool is_final_layer = layer_index + 1 == context.model.hidden_layers.size();
            const auto prefix = hidden_layer_cat_prefix(layer_index, is_final_layer);
            const auto mu_name = is_final_layer ? std::string("mu_cat") : "mu_cat_l" + std::to_string(layer_index);
            const auto semantic_mu =
                external_evaluations.at(mu_name)
                - context.domains.n->zero_polynomial_eval(challenges.at(
                    hidden_concat_y_name(layer_index, is_final_layer)))
                    * external_evaluations.at(
                        external_mask_eval_name(mu_name));
            acc.add(first * eval(prefix + "_Acc", z));
            acc.add(q_valid * acc_step(eval(prefix + "_Acc", omega_z), eval(prefix + "_Acc", z), eval(prefix + "_a", z), eval(prefix + "_b", z)));
            acc.add(not_last * q_invalid * (eval(prefix + "_Acc", omega_z) - eval(prefix + "_Acc", z)));
            acc.add(zero_on_unselected(q_valid, eval(prefix + "_a", z)));
            acc.add(zero_on_unselected(q_valid, eval(prefix + "_b", z)));
            acc.add(last * (eval(prefix + "_Acc", z) - semantic_mu));
        }

        const bool legacy_single_output = false;
        for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index)
        {
            const auto prefix = output_head_prefix(head_index, legacy_single_output) + "_";
            const auto mu_name =
                legacy_single_output ? std::string("mu_out_proj") : output_external_eval_name("proj", head_index, false);
            acc.add(first * eval(prefix + "Acc_proj", z));
            acc.add(q_valid * acc_step(eval(prefix + "Acc_proj", omega_z), eval(prefix + "Acc_proj", z), eval(prefix + "a_proj", z), eval(prefix + "b_proj", z)));
            acc.add(not_last * q_invalid * (eval(prefix + "Acc_proj", omega_z) - eval(prefix + "Acc_proj", z)));
            acc.add(zero_on_unselected(q_valid, eval(prefix + "a_proj", z)));
            acc.add(zero_on_unselected(q_valid, eval(prefix + "b_proj", z)));
            acc.add(last * (eval(prefix + "Acc_proj", z) - external_evaluations.at(mu_name)));
        }

        return acc.value() / zero_eval;
    }

    FieldElement evaluate_t_c(
        const ProtocolContext &context,
        const std::map<std::string, FieldElement> &challenges,
        const std::map<std::string, FieldElement> &external_evaluations,
        const EvalFn &eval,
        const FieldElement &z)
    {
        const auto omega_z = z * context.domains.c->omega;
        const auto first = lagrange_first(*context.domains.c, z);
        const auto last = lagrange_last(*context.domains.c, z);
        const auto not_last = FieldElement::one() - last;
        const auto zero_eval = context.domains.c->zero_polynomial_eval(z);
        const auto q_valid = eval("P_Q_C_valid", z);
        const auto q_invalid = FieldElement::one() - q_valid;
        // C 域主要对齐输出层每头的源/目标打分绑定，以及 bias 后输出和多头平均的关系。
        QuotientAccumulator acc(challenges.at("alpha_quot"));

        const bool legacy_single_output = false;
        for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index)
        {
            const auto prefix = output_head_prefix(head_index, legacy_single_output) + "_";
            const auto mu_src =
                legacy_single_output ? std::string("mu_out_src") : output_external_eval_name("src", head_index, false);
            const auto mu_dst =
                legacy_single_output ? std::string("mu_out_dst") : output_external_eval_name("dst", head_index, false);
            const auto mu_star =
                legacy_single_output ? std::string("mu_out_star") : output_external_eval_name("star", head_index, false);
            const auto star_semantic_eval =
                external_evaluations.at(mu_star)
                - context.domains.n->zero_polynomial_eval(challenges.at(
                    output_challenge_name(
                        "y_out_star", head_index, legacy_single_output)))
                    * external_evaluations.at(
                        external_mask_eval_name(mu_star));
            const auto src_semantic_eval =
                external_evaluations.at(mu_src)
                - context.domains.n->zero_polynomial_eval(challenges.at(
                    output_challenge_name(
                        "y_src_out", head_index, legacy_single_output)))
                    * external_evaluations.at(
                        output_score_mask_eval_name(head_index, "src"));
            const auto dst_semantic_eval =
                external_evaluations.at(mu_dst)
                - context.domains.n->zero_polynomial_eval(challenges.at(
                    output_challenge_name(
                        "y_dst_out", head_index, legacy_single_output)))
                    * external_evaluations.at(
                        output_score_mask_eval_name(head_index, "dst"));

            acc.add(first * eval(prefix + "Acc_src", z));
            acc.add(q_valid * acc_step(eval(prefix + "Acc_src", omega_z), eval(prefix + "Acc_src", z), eval(prefix + "a_src", z), eval(prefix + "b_src", z)));
            acc.add(not_last * q_invalid * (eval(prefix + "Acc_src", omega_z) - eval(prefix + "Acc_src", z)));
            acc.add(zero_on_unselected(q_valid, eval(prefix + "a_src", z)));
            acc.add(zero_on_unselected(q_valid, eval(prefix + "b_src", z)));
            acc.add(last * (eval(prefix + "Acc_src", z) - src_semantic_eval));

            acc.add(first * eval(prefix + "Acc_dst", z));
            acc.add(q_valid * acc_step(eval(prefix + "Acc_dst", omega_z), eval(prefix + "Acc_dst", z), eval(prefix + "a_dst", z), eval(prefix + "b_dst", z)));
            acc.add(not_last * q_invalid * (eval(prefix + "Acc_dst", omega_z) - eval(prefix + "Acc_dst", z)));
            acc.add(zero_on_unselected(q_valid, eval(prefix + "a_dst", z)));
            acc.add(zero_on_unselected(q_valid, eval(prefix + "b_dst", z)));
            acc.add(last * (eval(prefix + "Acc_dst", z) - dst_semantic_eval));

            acc.add(first * eval(prefix + "Acc_y", z));
            acc.add(q_valid * acc_step(eval(prefix + "Acc_y", omega_z), eval(prefix + "Acc_y", z), eval(prefix + "a_y", z), eval(prefix + "b_y", z)));
            acc.add(not_last * q_invalid * (eval(prefix + "Acc_y", omega_z) - eval(prefix + "Acc_y", z)));
            acc.add(zero_on_unselected(q_valid, eval(prefix + "a_y", z)));
            acc.add(zero_on_unselected(q_valid, eval(prefix + "b_y", z)));
            acc.add(last * (eval(prefix + "Acc_y", z) - star_semantic_eval));

            const auto view_prefix = output_bias_view_prefix(head_index);
            const auto eta_view = challenges.at(output_bias_view_eta_label(head_index));
            const auto beta_view = challenges.at(output_bias_view_beta_label(head_index));
            const auto table = eval(view_prefix + "Table_can", z);
            const auto multiplicity = eval(view_prefix + "m_can", z);
            const auto inverse = eval(view_prefix + "J_can", z);
            const auto accumulator = eval(view_prefix + "R_can", z);
            const auto accumulator_next = eval(view_prefix + "R_can", omega_z);
            acc.add(first * accumulator);
            acc.add(not_last * (
                accumulator_next - accumulator
                - q_valid * multiplicity * inverse));
            acc.add(q_valid * ((table + beta_view) * inverse - FieldElement::one()));
            acc.add(q_invalid * inverse);
            acc.add(table - q_valid * (
                eval("P_BiasIndex_C", z)
                + eta_view * eval(
                    model_proof_local_label(output_bias_label(head_index, false)), z)));
            acc.add(multiplicity - eval("P_BiasMultiplicity_C", z));

        }

        return acc.value() / zero_eval;
    }

} // namespace gatzk::protocol
