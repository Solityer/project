#include "gatzk/protocol/quotients.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <optional>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "gatzk/protocol/challenges.hpp"
#include "gatzk/protocol/schema.hpp"
#include "gatzk/protocol/t_edge_program.hpp"

namespace gatzk::protocol
{

    struct SignedDivisionEvaluationConstants
    {
        std::array<std::vector<algebra::FieldElement>, 5> betas;
    };

    struct HiddenHeadEvaluationConstants
    {
        std::size_t layer_index = 0;
        std::size_t local_head_index = 0;
        std::string prefix;
        std::string division_prefix;
        std::string division_selector;
        std::string elu_query_selector;
        SignedDivisionEvaluationConstants division;
        algebra::FieldElement beta_src;
        algebra::FieldElement beta_dst;
        algebra::FieldElement eta_src;
        algebra::FieldElement eta_src_sq;
        algebra::FieldElement eta_dst;
        algebra::FieldElement eta_dst_sq;
        algebra::FieldElement eta_dst_cu;
        algebra::FieldElement eta_dst_4;
        algebra::FieldElement eta_dst_5;
        algebra::FieldElement beta_l;
        algebra::FieldElement eta_l;
        algebra::FieldElement beta_r;
        algebra::FieldElement beta_exp;
        algebra::FieldElement eta_exp;
        algebra::FieldElement beta_elu;
        algebra::FieldElement eta_elu;
        algebra::FieldElement beta_t;
        algebra::FieldElement eta_t;
    };

    struct OutputHeadEvaluationConstants
    {
        struct BasisChunkLabels
        {
            std::string value;
            std::string active_selector;
            std::string semantic_selector;
            std::string gap_selector;
            std::string tail_selector;
            std::string accumulator;
        };

        std::string prefix;
        std::string division_prefix;
        std::string division_selector;
        std::string binding_prefix;
        std::string y_flat_label;
        std::string external_y_name;
        std::string view_prefix;
        std::string power_label;
        SignedDivisionEvaluationConstants division;
        algebra::FieldElement beta_src;
        algebra::FieldElement beta_dst;
        algebra::FieldElement eta_src;
        algebra::FieldElement eta_dst;
        algebra::FieldElement eta_dst_sq;
        algebra::FieldElement eta_dst_cu;
        algebra::FieldElement eta_dst_4;
        algebra::FieldElement eta_dst_5;
        algebra::FieldElement beta_l;
        algebra::FieldElement eta_l;
        algebra::FieldElement beta_r;
        algebra::FieldElement beta_exp;
        algebra::FieldElement eta_exp;
        algebra::FieldElement beta_t;
        algebra::FieldElement eta_t;
        algebra::FieldElement beta_view;
        algebra::FieldElement eta_view;
        algebra::FieldElement gamma;
        std::vector<BasisChunkLabels> basis_chunks;
    };

    struct TEdgeEvaluationPlan
    {
        algebra::FieldElement alpha_quot;
        algebra::FieldElement y_out;
        SignedDivisionEvaluationConstants average_division;

        std::vector<BindingRelationMeta> binding_metadata;
        // Formal quotient constants are keyed by (layer, local head).  Keep
        // the same nested identity as the trace and binding registries; a
        // flattened planned index is not a formal object identity.
        std::vector<std::vector<HiddenHeadEvaluationConstants>> hidden_heads;
        std::vector<OutputHeadEvaluationConstants> output_heads;
    };

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

        QuotientBoundaryValues boundary_lagrange_values(
            const algebra::RootOfUnityDomain &domain,
            const FieldElement &z)
        {
            const auto one = FieldElement::one();
            const auto last_point = domain.point_at(domain.size - 1);
            const auto zero = domain.zero_polynomial_eval(z);
            if (zero.is_zero())
            {
                return QuotientBoundaryValues{
                    .first = z == one ? one : FieldElement::zero(),
                    .last = z == last_point ? one : FieldElement::zero(),
                    .zero = zero,
                    .zero_inverse = std::nullopt,
                };
            }

            const auto first_denominator = z - one;
            const auto last_denominator = z - last_point;
            const auto denominator_product_inverse =
                (first_denominator * last_denominator).inv();
            const auto zero_scaled = zero * domain.inv_size;
            return QuotientBoundaryValues{
                .first = zero_scaled * last_denominator * denominator_product_inverse,
                .last = zero_scaled * last_point * first_denominator * denominator_product_inverse,
                .zero = zero,
                .zero_inverse = std::nullopt,
            };
        }

        FieldElement lagrange_at(const algebra::RootOfUnityDomain &domain, std::size_t index, const FieldElement &z)
        {
            return domain.lagrange_basis_eval(index, z);
        }

        bool starts_with(const std::string &value, std::string_view prefix)
        {
            return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
        }

        std::optional<std::string> hidden_head_suffix(const std::string &label)
        {
            // Formal hidden labels carry both coordinates.  Do not accept
            // the legacy flat P_h<index> form on the proof path.
            if (!starts_with(label, "P_h_l"))
            {
                return std::nullopt;
            }
            std::size_t pos = 5;
            const auto layer_begin = pos;
            while (pos < label.size() && std::isdigit(static_cast<unsigned char>(label[pos])))
            {
                ++pos;
            }
            if (pos == layer_begin || label.compare(pos, 2, "_h") != 0)
            {
                return std::nullopt;
            }
            pos += 2;
            const auto head_begin = pos;
            while (pos < label.size() && std::isdigit(static_cast<unsigned char>(label[pos])))
            {
                ++pos;
            }
            if (pos == head_begin || pos >= label.size() || label[pos] != '_')
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
                if (label == hidden_layer_cat_prefix(layer_index) + std::string(suffix))
                {
                    return true;
                }
            }
            return false;
        }

        std::optional<std::string> dynamic_label_domain_name(const ProtocolContext &context, const std::string &label)
        {
            // The null result is only for the removed legacy single-head
            // compatibility representation; formal context construction is
            // fail-closed before dynamic domain lookup.
            if (!context.model.has_real_multihead)
            {
                return std::nullopt;
            }
            if (label == "P_H" || label == "P_Table_feat" || label == "P_Query_feat" || label == "P_m_feat" || label == "P_J_tbl_feat" || label == "P_J_qry_feat" || label == "P_R_feat")
            {
                return std::string("FH");
            }
            if (label == "P_Y_lin" || label == "P_Y")
                return std::string("coef");
            for (std::size_t head = 0; head < context.model.output_layer.heads.size(); ++head)
            {
                if (label == output_y_lin_label(head) || label == output_y_label(head))
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
                if (label == hidden_layer_concat_label(layer_index))
                    return std::string("coef");
                if (label == hidden_layer_concat_star_label(layer_index))
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
                if (name == "Y_prime")
                    return std::string("coef");
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
                if (name == "ModelBind_proj_R")
                    return std::string("cat");
                if (name == "ModelBind_src_R" || name == "ModelBind_dst_R")
                    return std::string("C");
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
                if (name == "ModelBind_proj_R")
                    return std::string("in");
                if (name == "ModelBind_src_R" || name == "ModelBind_dst_R")
                    return std::string("d");
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

        template <bool UsePrecomputedAlphaPowers>
        class BasicQuotientAccumulator
        {
        public:
            explicit BasicQuotientAccumulator(
                const FieldElement &alpha,
                bool track_first_nonzero = false,
                const std::vector<FieldElement> *precomputed_alpha_powers = nullptr)
                : alpha_(alpha.native()),
                  track_first_nonzero_(track_first_nonzero),
                  precomputed_alpha_powers_(precomputed_alpha_powers)
            {
                power_ = 1;
                sum_.clear();
            }

            void add(const FieldElement &term)
            {
                if (track_first_nonzero_ && !first_nonzero_term_.has_value() && !term.is_zero())
                {
                    first_nonzero_term_ = term_count_;
                    first_nonzero_value_ = term;
                }
                mcl::Fr product;
                if constexpr (!UsePrecomputedAlphaPowers)
                {
                    mcl::Fr::mul(product, power_, term.native());
                    mcl::Fr::add(sum_, sum_, product);
                    mcl::Fr::mul(power_, power_, alpha_);
                }
                else
                {
                    const auto &power = term_count_ < precomputed_alpha_powers_->size()
                                            ? precomputed_alpha_powers_->at(term_count_).native()
                                            : power_;
                    mcl::Fr::mul(product, power, term.native());
                    mcl::Fr::add(sum_, sum_, product);
                    if (term_count_ + 1 >= precomputed_alpha_powers_->size())
                    {
                        mcl::Fr::mul(power_, power, alpha_);
                    }
                }
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
            const std::vector<FieldElement> *precomputed_alpha_powers_ = nullptr;
            std::size_t term_count_ = 0;
            std::optional<std::size_t> first_nonzero_term_;
            FieldElement first_nonzero_value_ = FieldElement::zero();
        };

        using QuotientAccumulator = BasicQuotientAccumulator<false>;

        template <class Scalar>
        Scalar route_node_step(
            const Scalar &next_value,
            const Scalar &current_value,
            const Scalar &table_value,
            const FieldElement &beta,
            const Scalar &q_valid,
            const Scalar &multiplicity)
        {
            return (next_value - current_value) *
                       (table_value + beta) -
                   q_valid * multiplicity;
        }

        template <class Scalar>
        Scalar route_edge_step(
            const Scalar &next_value,
            const Scalar &current_value,
            const Scalar &query_value,
            const FieldElement &beta,
            const Scalar &q_edge_valid)
        {
            return (next_value - current_value) *
                       (query_value + beta) -
                   q_edge_valid;
        }

        template <class Scalar>
        Scalar acc_step(
            const Scalar &next_value,
            const Scalar &current_value,
            const Scalar &a_value,
            const Scalar &b_value)
        {
            return next_value -
                   current_value -
                   a_value * b_value;
        }

        template <class Scalar>
        Scalar psq_step(
            const Scalar &psq_next,
            const Scalar &psq_current,
            const Scalar &w_next,
            const Scalar &q_valid_next,
            const Scalar &q_new_next)
        {
            const Scalar one =
                FieldElement::one();

            return psq_next -
                   ((one - q_valid_next) *
                        psq_current +
                    q_valid_next *
                        (q_new_next *
                             w_next +
                         (one - q_new_next) *
                             (psq_current +
                              w_next)));
        }

        template <class Scalar>
        Scalar zero_on_unselected(
            const Scalar &selector,
            const Scalar &value)
        {
            const Scalar one =
                FieldElement::one();

            return (one - selector) *
                   value;
        }

        struct SignedDivisionLimbLabelPlan
        {
            std::string limb;
            std::array<std::string, 6> range;
            std::string beta;
        };

        struct SignedDivisionLabelPlan
        {
            std::string prefix;
            std::size_t range_bits = 0;
            std::array<std::string, 12> values;
            std::array<std::vector<SignedDivisionLimbLabelPlan>, 5> decompositions;
        };

        const SignedDivisionLabelPlan &signed_division_label_plan(
            const std::string &prefix,
            std::size_t range_bits)
        {
            thread_local std::vector<SignedDivisionLabelPlan> cache;
            for (const auto &entry : cache)
            {
                if (entry.range_bits == range_bits && entry.prefix == prefix)
                    return entry;
            }

            constexpr std::array<std::string_view, 12> value_suffixes = {
                "Active",
                "X",
                "D",
                "Y",
                "Sigma",
                "Zero",
                "A",
                "A_inv",
                "D_inv",
                "Q",
                "R",
                "Delta",
            };
            constexpr std::array<std::string_view, 5> decomposition_names = {
                "A",
                "DMinus1",
                "Q",
                "R",
                "Delta",
            };
            constexpr std::array<std::string_view, 6> range_columns = {
                "Table",
                "Query",
                "m",
                "J_tbl",
                "J_qry",
                "R",
            };

            SignedDivisionLabelPlan plan;
            plan.prefix = prefix;
            plan.range_bits = range_bits;
            for (std::size_t index = 0; index < value_suffixes.size(); ++index)
                plan.values[index] = prefix + std::string(value_suffixes[index]);

            const auto limb_count = signed_division_limb_count(range_bits);
            for (std::size_t value_index = 0;
                 value_index < decomposition_names.size(); ++value_index)
            {
                auto &limbs = plan.decompositions[value_index];
                limbs.reserve(limb_count);
                for (std::size_t limb = 0; limb < limb_count; ++limb)
                {
                    SignedDivisionLimbLabelPlan limb_plan;
                    limb_plan.limb = signed_division_limb_label(
                        prefix, decomposition_names[value_index], limb);
                    for (std::size_t column = 0;
                         column < range_columns.size(); ++column)
                    {
                        limb_plan.range[column] = signed_division_range_label(
                            prefix,
                            decomposition_names[value_index],
                            limb,
                            range_columns[column]);
                    }
                    limb_plan.beta = signed_division_beta_name(
                        prefix, decomposition_names[value_index], limb);
                    limbs.push_back(std::move(limb_plan));
                }
            }
            cache.push_back(std::move(plan));
            return cache.back();
        }

        SignedDivisionEvaluationConstants
        bind_signed_division_challenges(
            const std::string &prefix,
            std::size_t range_bits,
            const std::map<std::string, FieldElement> &challenges)
        {
            const auto &labels =
                signed_division_label_plan(
                    prefix,
                    range_bits);

            SignedDivisionEvaluationConstants out;

            for (std::size_t value_index = 0;
                 value_index < labels.decompositions.size();
                 ++value_index)
            {
                const auto &limbs =
                    labels.decompositions[value_index];

                auto &betas = out.betas[value_index];
                betas.reserve(limbs.size());

                for (const auto &limb : limbs)
                {
                    betas.push_back(
                        challenges.at(limb.beta));
                }
            }

            return out;
        }
        class TEdgeProgramBuilder;

        thread_local TEdgeProgramBuilder
            *active_t_edge_program_builder =
                nullptr;

        class TEdgeProgramScalar
        {
        public:
            TEdgeProgramScalar() =
                default;

            TEdgeProgramScalar(
                TEdgeProgramBuilder *builder,
                std::uint32_t node)
                : builder_(builder),
                  node_(node)
            {
            }

            TEdgeProgramScalar(
                const FieldElement &value);

            std::uint32_t node() const
            {
                return node_;
            }

            TEdgeProgramBuilder *builder()
                const
            {
                return builder_;
            }

            TEdgeProgramScalar &operator+=(
                const TEdgeProgramScalar &rhs);

            TEdgeProgramScalar &operator-=(
                const TEdgeProgramScalar &rhs);

            TEdgeProgramScalar &operator*=(
                const TEdgeProgramScalar &rhs);

        private:
            TEdgeProgramBuilder *builder_ =
                nullptr;

            std::uint32_t node_ = 0;
        };

        class TEdgeProgramBuilder
        {
        public:
            TEdgeProgramScalar constant(
                const FieldElement &value)
            {
                if (program_.constants.size() >
                    std::numeric_limits<
                        std::uint32_t>::max())
                {
                    throw std::runtime_error(
                        "t_edge program constant "
                        "index overflow");
                }

                const auto constant_index =
                    static_cast<std::uint32_t>(
                        program_.constants
                            .size());

                program_.constants.push_back(
                    value);

                return make_node(
                    TEdgeProgramOpcode::
                        LoadConstant,
                    0,
                    0,
                    constant_index);
            }

            TEdgeProgramScalar access(
                const std::string &label,
                bool rotated)
            {
                const auto key =
                    label +
                    (rotated
                         ? "#rotated"
                         : "#current");

                std::uint32_t access_index = 0;

                if (const auto found =
                        access_indices_.find(
                            key);
                    found !=
                    access_indices_.end())
                {
                    access_index =
                        found->second;
                }
                else
                {
                    if (program_.accesses.size() >
                        std::numeric_limits<
                            std::uint32_t>::max())
                    {
                        throw std::runtime_error(
                            "t_edge program access "
                            "index overflow");
                    }

                    access_index =
                        static_cast<
                            std::uint32_t>(
                            program_.accesses
                                .size());

                    program_.accesses.push_back(
                        TEdgeProgramAccess{
                            .label = label,
                            .rotated =
                                rotated,
                        });

                    access_indices_.emplace(
                        key,
                        access_index);
                }

                return make_node(
                    TEdgeProgramOpcode::
                        LoadAccess,
                    0,
                    0,
                    access_index);
            }

            TEdgeProgramScalar
            boundary_first()
            {
                return access(
                    "__t_edge_boundary_l0",
                    false);
            }

            TEdgeProgramScalar
            boundary_last()
            {
                return access(
                    "__t_edge_boundary_l0",
                    true);
            }

            TEdgeProgramScalar add(
                const TEdgeProgramScalar &lhs,
                const TEdgeProgramScalar &rhs)
            {
                require_same_builder(
                    lhs,
                    rhs);

                return make_node(
                    TEdgeProgramOpcode::Add,
                    lhs.node(),
                    rhs.node(),
                    0);
            }

            TEdgeProgramScalar subtract(
                const TEdgeProgramScalar &lhs,
                const TEdgeProgramScalar &rhs)
            {
                require_same_builder(
                    lhs,
                    rhs);

                return make_node(
                    TEdgeProgramOpcode::
                        Subtract,
                    lhs.node(),
                    rhs.node(),
                    0);
            }

            TEdgeProgramScalar multiply(
                const TEdgeProgramScalar &lhs,
                const TEdgeProgramScalar &rhs)
            {
                require_same_builder(
                    lhs,
                    rhs);

                return make_node(
                    TEdgeProgramOpcode::
                        Multiply,
                    lhs.node(),
                    rhs.node(),
                    0);
            }

            TEdgeProgramScalar negate(
                const TEdgeProgramScalar &value)
            {
                require_builder(value);

                return make_node(
                    TEdgeProgramOpcode::Negate,
                    value.node(),
                    0,
                    0);
            }

            void append_term(
                const TEdgeProgramScalar &term,
                const FieldElement
                    &alpha_scale)
            {
                require_builder(term);

                TEdgeProgramTerm output;

                output.instruction_begin =
                    program_.instructions
                        .size();

                std::size_t stack_depth = 0;
                std::size_t maximum_depth = 0;

                emit_node(
                    term.node(),
                    &output.required_accesses,
                    &stack_depth,
                    &maximum_depth);

                if (stack_depth != 1)
                {
                    throw std::runtime_error(
                        "invalid t_edge postfix "
                        "program stack");
                }

                std::sort(
                    output.required_accesses
                        .begin(),
                    output.required_accesses
                        .end());

                output.required_accesses.erase(
                    std::unique(
                        output.required_accesses
                            .begin(),
                        output.required_accesses
                            .end()),
                    output.required_accesses
                        .end());

                output.instruction_end =
                    program_.instructions
                        .size();

                output.maximum_stack_depth =
                    maximum_depth;

                output.alpha_scale =
                    alpha_scale;

                program_.maximum_stack_depth =
                    std::max(
                        program_
                            .maximum_stack_depth,
                        maximum_depth);

                program_.terms.push_back(
                    std::move(output));
            }

            TEdgeBoundProgram finish()
            {
                if (program_.terms.empty())
                {
                    throw std::runtime_error(
                        "t_edge program has no "
                        "constraint terms");
                }

                return std::move(program_);
            }

        private:
            struct Node
            {
                TEdgeProgramOpcode opcode =
                    TEdgeProgramOpcode::
                        LoadConstant;

                std::uint32_t lhs = 0;
                std::uint32_t rhs = 0;
                std::uint32_t operand = 0;
            };

            TEdgeProgramScalar make_node(
                TEdgeProgramOpcode opcode,
                std::uint32_t lhs,
                std::uint32_t rhs,
                std::uint32_t operand)
            {
                if (nodes_.size() >
                    std::numeric_limits<
                        std::uint32_t>::max())
                {
                    throw std::runtime_error(
                        "t_edge program node "
                        "index overflow");
                }

                const auto index =
                    static_cast<std::uint32_t>(
                        nodes_.size());

                nodes_.push_back(
                    Node{
                        .opcode = opcode,
                        .lhs = lhs,
                        .rhs = rhs,
                        .operand = operand,
                    });

                return TEdgeProgramScalar(
                    this,
                    index);
            }

            void require_builder(
                const TEdgeProgramScalar &value)
                const
            {
                if (value.builder() != this)
                {
                    throw std::runtime_error(
                        "t_edge program mixed "
                        "different builders");
                }
            }

            void require_same_builder(
                const TEdgeProgramScalar &lhs,
                const TEdgeProgramScalar &rhs)
                const
            {
                require_builder(lhs);
                require_builder(rhs);
            }

            void emit_node(
                std::uint32_t node_index,
                std::vector<std::uint32_t>
                    *required_accesses,
                std::size_t *stack_depth,
                std::size_t *maximum_depth)
            {
                if (node_index >=
                    nodes_.size())
                {
                    throw std::runtime_error(
                        "t_edge program node "
                        "out of range");
                }

                const auto &node =
                    nodes_[node_index];

                switch (node.opcode)
                {
                case TEdgeProgramOpcode::
                    LoadAccess:
                {
                    program_.instructions
                        .push_back(
                            TEdgeProgramInstruction{
                                .opcode =
                                    node.opcode,
                                .operand =
                                    node.operand,
                            });

                    required_accesses->push_back(
                        node.operand);

                    ++*stack_depth;

                    *maximum_depth =
                        std::max(
                            *maximum_depth,
                            *stack_depth);

                    return;
                }

                case TEdgeProgramOpcode::
                    LoadConstant:
                {
                    program_.instructions
                        .push_back(
                            TEdgeProgramInstruction{
                                .opcode =
                                    node.opcode,
                                .operand =
                                    node.operand,
                            });

                    ++*stack_depth;

                    *maximum_depth =
                        std::max(
                            *maximum_depth,
                            *stack_depth);

                    return;
                }

                case TEdgeProgramOpcode::
                    Negate:
                {
                    emit_node(
                        node.lhs,
                        required_accesses,
                        stack_depth,
                        maximum_depth);

                    program_.instructions
                        .push_back(
                            TEdgeProgramInstruction{
                                .opcode =
                                    node.opcode,
                                .operand = 0,
                            });

                    return;
                }

                case TEdgeProgramOpcode::Add:
                case TEdgeProgramOpcode::
                    Subtract:
                case TEdgeProgramOpcode::
                    Multiply:
                {
                    emit_node(
                        node.lhs,
                        required_accesses,
                        stack_depth,
                        maximum_depth);

                    emit_node(
                        node.rhs,
                        required_accesses,
                        stack_depth,
                        maximum_depth);

                    if (*stack_depth < 2)
                    {
                        throw std::runtime_error(
                            "invalid t_edge "
                            "binary expression "
                            "stack");
                    }

                    --*stack_depth;

                    program_.instructions
                        .push_back(
                            TEdgeProgramInstruction{
                                .opcode =
                                    node.opcode,
                                .operand = 0,
                            });

                    return;
                }
                }
            }

            std::vector<Node> nodes_;

            std::unordered_map<
                std::string,
                std::uint32_t>
                access_indices_;

            TEdgeBoundProgram program_;
        };

        TEdgeProgramScalar::
            TEdgeProgramScalar(
                const FieldElement &value)
        {
            if (active_t_edge_program_builder ==
                nullptr)
            {
                throw std::runtime_error(
                    "t_edge symbolic constant "
                    "created without builder");
            }

            *this =
                active_t_edge_program_builder
                    ->constant(value);
        }

        TEdgeProgramScalar operator+(
            const TEdgeProgramScalar &lhs,
            const TEdgeProgramScalar &rhs)
        {
            return lhs.builder()->add(
                lhs,
                rhs);
        }

        TEdgeProgramScalar operator-(
            const TEdgeProgramScalar &lhs,
            const TEdgeProgramScalar &rhs)
        {
            return lhs.builder()->subtract(
                lhs,
                rhs);
        }

        TEdgeProgramScalar operator*(
            const TEdgeProgramScalar &lhs,
            const TEdgeProgramScalar &rhs)
        {
            return lhs.builder()->multiply(
                lhs,
                rhs);
        }

        TEdgeProgramScalar operator-(
            const TEdgeProgramScalar &value)
        {
            return value.builder()->negate(
                value);
        }

        TEdgeProgramScalar operator+(
            const TEdgeProgramScalar &lhs,
            const FieldElement &rhs)
        {
            return lhs +
                   TEdgeProgramScalar(rhs);
        }

        TEdgeProgramScalar operator+(
            const FieldElement &lhs,
            const TEdgeProgramScalar &rhs)
        {
            return TEdgeProgramScalar(lhs) +
                   rhs;
        }

        TEdgeProgramScalar operator-(
            const TEdgeProgramScalar &lhs,
            const FieldElement &rhs)
        {
            return lhs -
                   TEdgeProgramScalar(rhs);
        }

        TEdgeProgramScalar operator-(
            const FieldElement &lhs,
            const TEdgeProgramScalar &rhs)
        {
            return TEdgeProgramScalar(lhs) -
                   rhs;
        }

        TEdgeProgramScalar operator*(
            const TEdgeProgramScalar &lhs,
            const FieldElement &rhs)
        {
            return lhs *
                   TEdgeProgramScalar(rhs);
        }

        TEdgeProgramScalar operator*(
            const FieldElement &lhs,
            const TEdgeProgramScalar &rhs)
        {
            return TEdgeProgramScalar(lhs) *
                   rhs;
        }

        TEdgeProgramScalar &
        TEdgeProgramScalar::operator+=(
            const TEdgeProgramScalar &rhs)
        {
            *this = *this + rhs;
            return *this;
        }

        TEdgeProgramScalar &
        TEdgeProgramScalar::operator-=(
            const TEdgeProgramScalar &rhs)
        {
            *this = *this - rhs;
            return *this;
        }

        TEdgeProgramScalar &
        TEdgeProgramScalar::operator*=(
            const TEdgeProgramScalar &rhs)
        {
            *this = *this * rhs;
            return *this;
        }

        class TEdgeProgramQuotientAccumulator
        {
        public:
            explicit TEdgeProgramQuotientAccumulator(
                const FieldElement &alpha,
                bool,
                const std::vector<
                    FieldElement> *)
                : builder_(
                      active_t_edge_program_builder),
                  alpha_(alpha),
                  alpha_power_(
                      FieldElement::one())
            {
                if (builder_ == nullptr)
                {
                    throw std::runtime_error(
                        "t_edge symbolic "
                        "accumulator has no "
                        "builder");
                }
            }

            void add(
                const TEdgeProgramScalar &term)
            {
                builder_->append_term(
                    term,
                    alpha_power_);

                alpha_power_ *= alpha_;

                ++term_count_;
            }

            TEdgeProgramScalar value() const
            {
                return TEdgeProgramScalar(
                    FieldElement::zero());
            }

            std::optional<std::size_t>
            first_nonzero_term() const
            {
                return std::nullopt;
            }

            FieldElement
            first_nonzero_value() const
            {
                return FieldElement::zero();
            }

        private:
            TEdgeProgramBuilder *builder_ =
                nullptr;

            FieldElement alpha_;

            FieldElement alpha_power_;

            std::size_t term_count_ = 0;
        };

    }

    std::shared_ptr<const TEdgeEvaluationPlan>
    build_t_edge_evaluation_plan(
        const ProtocolContext &context,
        const std::map<std::string, FieldElement> &challenges)
    {
        auto plan = std::make_shared<TEdgeEvaluationPlan>();

        plan->alpha_quot =
            challenges.at("alpha_quot");

        plan->y_out =
            challenges.at("y_out");

        plan->binding_metadata =
            build_binding_relation_metadata(context);
        plan->hidden_heads.resize(context.model.hidden_layers.size());
        for (std::size_t layer_index = 0;
             layer_index < context.model.hidden_layers.size();
             ++layer_index)
        {
            const auto &layer = context.model.hidden_layers[layer_index];
            auto &layer_constants = plan->hidden_heads.at(layer_index);
            layer_constants.reserve(layer.heads.size());
            for (std::size_t local_head_index = 0;
                 local_head_index < layer.heads.size();
                 ++local_head_index)
            {
            HiddenHeadEvaluationConstants constants;
            constants.layer_index = layer_index;
            constants.local_head_index = local_head_index;
            const auto challenge = [&](std::string_view base)
            {
                return hidden_challenge_name(base, layer_index, local_head_index);
            };
            constants.prefix = hidden_head_prefix(layer_index, local_head_index) + "_";
            constants.division_prefix = constants.prefix + "Div_";
            constants.division_selector = hidden_q_nd_h_selector_label(layer_index);

            constants.division =
                bind_signed_division_challenges(
                    constants.division_prefix,
                    context.config.range_bits,
                    challenges);

            // Q_Nd_h is the sole layer-scoped edge selector for hidden
            // aggregation/ELU queries, including layers whose width happens
            // to equal the historical maximum.
            constants.elu_query_selector = constants.division_selector;
            constants.beta_src = challenges.at(challenge("beta_src_h"));
            constants.beta_dst = challenges.at(challenge("beta_dst_h"));
            constants.eta_src = challenges.at(challenge("eta_src_h"));
            constants.eta_src_sq = constants.eta_src * constants.eta_src;
            constants.eta_dst = challenges.at(challenge("eta_dst_h"));
            constants.eta_dst_sq = constants.eta_dst * constants.eta_dst;
            constants.eta_dst_cu = constants.eta_dst_sq * constants.eta_dst;
            constants.eta_dst_4 = constants.eta_dst_cu * constants.eta_dst;
            constants.eta_dst_5 = constants.eta_dst_4 * constants.eta_dst;
            constants.beta_l = challenges.at(challenge("beta_L_h"));
            constants.eta_l = challenges.at(challenge("eta_L_h"));
            constants.beta_r = challenges.at(challenge("beta_R_h"));
            constants.beta_exp = challenges.at(challenge("beta_exp_h"));
            constants.eta_exp = challenges.at(challenge("eta_exp_h"));
            constants.beta_elu = challenges.at(challenge("beta_ELU_h"));
            constants.eta_elu = challenges.at(challenge("eta_ELU_h"));
            constants.beta_t = challenges.at(challenge("beta_t_h"));
            constants.eta_t = challenges.at(challenge("eta_t_h"));
            layer_constants.push_back(std::move(constants));
            }
        }
        plan->output_heads.reserve(context.model.output_layer.heads.size());
        for (std::size_t head_index = 0;
             head_index < context.model.output_layer.heads.size(); ++head_index)
        {
            OutputHeadEvaluationConstants constants;
            constants.prefix =
                output_head_prefix(head_index) + "_";

            constants.division_prefix =
                constants.prefix + "Div_";

            constants.division_selector =
                "P_Q_div_out" + std::to_string(head_index);

            constants.division =
                bind_signed_division_challenges(
                    constants.division_prefix,
                    context.config.range_bits,
                    challenges);

            constants.binding_prefix =
                constants.prefix + "Bind_";
            constants.y_flat_label = constants.prefix + "Y_flat";
            constants.external_y_name =
                output_external_eval_name("y", head_index);
            constants.view_prefix = output_bias_view_prefix(head_index);
            constants.power_label =
                output_bias_basis_label(head_index, "Powers");
            constants.beta_src = challenges.at(
                output_challenge_name("beta_src_out", head_index));
            constants.beta_dst = challenges.at(
                output_challenge_name("beta_dst_out", head_index));
            constants.eta_src = challenges.at(
                output_challenge_name("eta_src_out", head_index));
            constants.eta_dst = challenges.at(
                output_challenge_name("eta_dst_out", head_index));
            constants.eta_dst_sq = constants.eta_dst * constants.eta_dst;
            constants.eta_dst_cu = constants.eta_dst_sq * constants.eta_dst;
            constants.eta_dst_4 = constants.eta_dst_cu * constants.eta_dst;
            constants.eta_dst_5 = constants.eta_dst_4 * constants.eta_dst;
            constants.beta_l = challenges.at(
                output_challenge_name("beta_L_out", head_index));
            constants.eta_l = challenges.at(
                output_challenge_name("eta_L_out", head_index));
            constants.beta_r = challenges.at(
                output_challenge_name("beta_R_out", head_index));
            constants.beta_exp = challenges.at(
                output_challenge_name("beta_exp_out", head_index));
            constants.eta_exp = challenges.at(
                output_challenge_name("eta_exp_out", head_index));
            constants.beta_t = challenges.at(
                output_challenge_name("beta_t_out", head_index));
            constants.eta_t = challenges.at(
                output_challenge_name("eta_t_out", head_index));
            constants.beta_view = challenges.at(
                output_bias_view_beta_label(head_index));
            constants.eta_view = challenges.at(
                output_bias_view_eta_label(head_index));
            constants.gamma = challenges.at(
                output_bias_basis_gamma_label(head_index));
            const auto basis_layout = coefficient_chunk_layout(
                context.local.num_classes + registered_blinder_coefficient_count(
                                                context,
                                                output_bias_basis_label(head_index, "Coeff")),
                context.domains.edge->size);
            constants.basis_chunks.reserve(basis_layout.size());
            for (const auto &chunk : basis_layout)
            {
                OutputHeadEvaluationConstants::BasisChunkLabels labels;
                labels.value = output_bias_basis_chunk_label(
                    head_index, chunk.chunk_index);
                labels.active_selector =
                    output_bias_basis_region_selector_label(
                        head_index, chunk.chunk_index, "active");
                labels.semantic_selector =
                    output_bias_basis_region_selector_label(
                        head_index, chunk.chunk_index, "semantic");
                labels.gap_selector =
                    output_bias_basis_region_selector_label(
                        head_index, chunk.chunk_index, "gap");
                labels.tail_selector =
                    output_bias_basis_region_selector_label(
                        head_index, chunk.chunk_index, "tail");
                labels.accumulator =
                    output_bias_basis_chunk_accumulator_label(
                        head_index, chunk.chunk_index);
                constants.basis_chunks.push_back(std::move(labels));
            }
            plan->output_heads.push_back(
                std::move(constants));
        }

        plan->average_division =
            bind_signed_division_challenges(
                "P_avg_Div_",
                context.config.range_bits,
                challenges);

        return plan;
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
            if (label.find("_ScoreMask_") != std::string::npos || label.ends_with("_ExternalMask"))
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
                    output_bias_label(head)));
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
                if (relation == metadata.end() || relation->coefficient_basis_relations.size() != 1)
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
                            "canonical matrix registered on unsupported domain: " + matrix.relation_id);
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

    namespace
    {
        std::array<FieldElement, 9> t_fh_terms(
            const FHQuotientParameters &parameters,
            const FHQuotientValues &values,
            const QuotientBoundaryValues &boundary,
            const FHFeatureEncodingValues *feature_encoding_values)
        {
            const auto one = FieldElement::one();
            const auto not_last = one - boundary.last;
            std::array<FieldElement, 9> terms;
            terms[0] = boundary.first * values.accumulator;
            terms[1] = not_last * (values.accumulator_next - values.accumulator - values.q_table * values.multiplicity * values.j_table + values.q_query * values.j_query);
            terms[2] = boundary.last * values.accumulator;
            terms[3] = values.q_table * ((values.table + parameters.beta_feat) * values.j_table - one);
            terms[4] = (one - values.q_table) * values.j_table;
            terms[5] = values.q_query * ((values.query + parameters.beta_feat) * values.j_query - one);
            terms[6] = (one - values.q_query) * values.j_query;
            if (feature_encoding_values != nullptr)
            {
                const auto eta_feat_squared =
                    parameters.eta_feat * parameters.eta_feat;
                terms[7] =
                    values.table - (feature_encoding_values->table_row + parameters.eta_feat * feature_encoding_values->table_column + eta_feat_squared * feature_encoding_values->table_feature);
                terms[8] =
                    values.query - (feature_encoding_values->query_row + parameters.eta_feat * feature_encoding_values->query_column + eta_feat_squared * feature_encoding_values->query_feature);
            }
            return terms;
        }

        FieldElement finish_t_fh_quotient(
            const FieldElement &numerator,
            const QuotientBoundaryValues &boundary)
        {
            return boundary.zero_inverse.has_value()
                       ? numerator * *boundary.zero_inverse
                       : numerator / boundary.zero;
        }
    }

    FieldElement evaluate_t_fh_from_values(
        const FHQuotientParameters &parameters,
        const FHQuotientValues &values,
        const QuotientBoundaryValues &boundary,
        const FHFeatureEncodingValues *feature_encoding_values)
    {
        const auto terms = t_fh_terms(
            parameters, values, boundary, feature_encoding_values);
        // The quotient alpha slots form one fixed polynomial in alpha.  Horner
        // evaluation removes the per-point alpha-power stream and its second
        // multiplication per term.  In the full-identity production route the
        // last two terms are certified zero, so they need not be traversed.
        const auto term_count = feature_encoding_values == nullptr
                                    ? std::size_t{7}
                                    : terms.size();
        auto numerator = terms[term_count - 1];
        for (std::size_t index = term_count - 1; index-- > 0;)
        {
            numerator = terms[index] + parameters.alpha_quot * numerator;
        }
        return finish_t_fh_quotient(numerator, boundary);
    }

    FieldElement evaluate_t_fh(
        const ProtocolContext &context,
        const std::map<std::string, FieldElement> &challenges,
        const EvalFn &eval,
        const FieldElement &z,
        FHQuotientProfile *profile,
        const QuotientBoundaryValues *boundary_values,
        bool feature_encoding_terms_are_identically_zero)
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
        const auto omega_z = z * context.domains.fh->omega;
        const auto boundary = boundary_values != nullptr
                                  ? *boundary_values
                                  : boundary_lagrange_values(*context.domains.fh, z);
        const FHQuotientParameters parameters{
            .alpha_quot = challenges.at("alpha_quot"),
            .beta_feat = challenges.at("beta_feat"),
            .eta_feat = feature_encoding_terms_are_identically_zero
                            ? FieldElement::zero()
                            : challenges.at("eta_feat"),
        };
        const FHQuotientValues values{
            .q_table = profiled_eval("P_Q_tbl_feat", z),
            .q_query = profiled_eval("P_Q_qry_feat", z),
            .table = profiled_eval("P_Table_feat", z),
            .query = profiled_eval("P_Query_feat", z),
            .multiplicity = profiled_eval("P_m_feat", z),
            .j_table = profiled_eval("P_J_tbl_feat", z),
            .j_query = profiled_eval("P_J_qry_feat", z),
            .accumulator = profiled_eval("P_R_feat", z),
            .accumulator_next = profiled_eval("P_R_feat", omega_z),
        };
        std::optional<FHFeatureEncodingValues> feature_encoding_values;
        if (!feature_encoding_terms_are_identically_zero)
        {
            feature_encoding_values.emplace(FHFeatureEncodingValues{
                .table_row = profiled_eval("P_Row_feat_tbl", z),
                .table_column = profiled_eval("P_Col_feat_tbl", z),
                .table_feature = profiled_eval("P_T_H", z),
                .query_row = profiled_eval("P_I_feat_qry", z),
                .query_column = profiled_eval("P_Col_feat_qry", z),
                .query_feature = profiled_eval("P_H", z),
            });
        }
        const auto result = evaluate_t_fh_from_values(
            parameters,
            values,
            boundary,
            feature_encoding_values.has_value()
                ? &*feature_encoding_values
                : nullptr);
        if (profile != nullptr)
        {
            const auto total_ms = elapsed_ms(total_start, Clock::now());
            profile->assembly_ms += total_ms - (profile->dependency_eval_ms - dependency_before);
        }
        return result;
    }

    template <
        bool UsePrecomputedAlphaPowers,
        class Scalar,
        class EvalCallable>
    Scalar evaluate_t_edge_impl(
        const ProtocolContext &context,
        const std::map<
            std::string,
            FieldElement> &challenges,
        const std::map<
            std::string,
            FieldElement>
            &external_evaluations,
        const EvalCallable &eval,
        const FieldElement &z,
        bool return_numerator,
        std::size_t *first_nonzero_term,
        FieldElement *first_nonzero_value,
        const std::vector<FieldElement>
            *precomputed_alpha_powers,
        const TEdgeEvaluationPlan
            *evaluation_plan,
        const QuotientBoundaryValues
            *boundary_values,
        const Scalar *symbolic_first =
            nullptr,
        const Scalar *symbolic_last =
            nullptr)
    {
        if (context.model.has_real_multihead)
        {
            // edge 域承载大部分“边上对象”约束：softmax 链、lookup、route 和 PSQ 都在这里闭合。
            const auto omega_z = z * context.domains.edge->omega;
            const auto boundary = boundary_values != nullptr
                                      ? *boundary_values
                                      : boundary_lagrange_values(*context.domains.edge, z);
            const Scalar first =
                symbolic_first != nullptr
                    ? *symbolic_first
                    : Scalar(
                          boundary.first);

            const Scalar last =
                symbolic_last != nullptr
                    ? *symbolic_last
                    : Scalar(
                          boundary.last);

            const Scalar one =
                FieldElement::one();

            const Scalar two =
                FieldElement(2);

            const auto not_last =
                one - last;

            const auto zero_eval =
                boundary.zero;
            const auto alpha_quot =
                evaluation_plan != nullptr
                    ? evaluation_plan->alpha_quot
                    : challenges.at("alpha_quot");

            const auto y_out =
                evaluation_plan != nullptr
                    ? evaluation_plan->y_out
                    : challenges.at("y_out");
            (void)signed_division_limb_count(context.config.range_bits);
            const auto signed_division_limb_base =
                FieldElement(std::uint64_t{1} << context.config.range_bits);
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
            const auto t_l_x_z = eval_z("P_T_L_x");
            const auto t_l_y_z = eval_z("P_T_L_y");
            const auto t_range_z = eval_z("P_T_range");
            const auto t_exp_x_z = eval_z("P_T_exp_x");
            const auto t_exp_y_z = eval_z("P_T_exp_y");
            const auto t_elu_x_z = eval_z("P_T_ELU_x");
            const auto t_elu_y_z = eval_z("P_T_ELU_y");
            using ActiveAccumulator =
                std::conditional_t<
                    std::is_same_v<
                        Scalar,
                        FieldElement>,
                    BasicQuotientAccumulator<
                        UsePrecomputedAlphaPowers>,
                    TEdgeProgramQuotientAccumulator>;

            ActiveAccumulator acc(
                alpha_quot,
                return_numerator,
                precomputed_alpha_powers);
            auto add_signed_division_constraints = [&](
                                                       const std::string &prefix,
                                                       const std::string &public_selector_label,
                                                       const SignedDivisionEvaluationConstants
                                                           *planned_constants)
            {
                const auto selector = eval_z(public_selector_label);
                const auto &labels = signed_division_label_plan(
                    prefix, context.config.range_bits);
                const auto active = eval_z(labels.values[0]);
                const auto x = eval_z(labels.values[1]);
                const auto d = eval_z(labels.values[2]);
                const auto y = eval_z(labels.values[3]);
                const auto sigma = eval_z(labels.values[4]);
                const auto zero = eval_z(labels.values[5]);
                const auto a = eval_z(labels.values[6]);
                const auto a_inv = eval_z(labels.values[7]);
                const auto d_inv = eval_z(labels.values[8]);
                const auto q = eval_z(labels.values[9]);
                const auto r = eval_z(labels.values[10]);
                const auto delta = eval_z(labels.values[11]);
                const auto sign = one - two * sigma;

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
                acc.add(selector * (two * a + d - two * d * q - r));
                acc.add(selector * (r + delta - two * d + one));

                // Canonical padding is part of the public column-padding
                // registry: no auxiliary value may survive outside Q_div.
                for (const auto &item : {x, d, y, sigma, zero, a, a_inv, d_inv, q, r, delta})
                {
                    acc.add((one - selector) * item);
                }

                for (std::size_t value_index = 0;
                     value_index < labels.decompositions.size(); ++value_index)
                {
                    const auto source =
                        value_index == 0   ? a
                        : value_index == 1 ? d - one
                        : value_index == 2 ? q
                        : value_index == 3 ? r
                                           : delta;
                    Scalar reconstructed =
                        FieldElement::zero();
                    auto place = FieldElement::one();
                    for (std::size_t limb_index = 0;
                         limb_index < labels.decompositions[value_index].size();
                         ++limb_index)
                    {
                        const auto &limb_labels =
                            labels.decompositions[value_index][limb_index];
                        const auto limb_value = eval_z(limb_labels.limb);
                        reconstructed += place * limb_value;
                        place *= signed_division_limb_base;

                        const auto table = eval_z(limb_labels.range[0]);
                        const auto query = eval_z(limb_labels.range[1]);
                        const auto multiplicity = eval_z(limb_labels.range[2]);
                        const auto j_table = eval_z(limb_labels.range[3]);
                        const auto j_query = eval_z(limb_labels.range[4]);
                        const auto accumulator = eval_z(limb_labels.range[5]);
                        const auto accumulator_next = eval_omega(limb_labels.range[5]);
                        const auto beta =
                            planned_constants != nullptr
                                ? planned_constants
                                      ->betas[value_index][limb_index]
                                : challenges.at(limb_labels.beta);
                        acc.add(first * accumulator);
                        acc.add(not_last * (accumulator_next - accumulator - q_tbl_r_z * multiplicity * j_table + selector * j_query));
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
                                             const Scalar &q_table,
                                             const Scalar &q_query,
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
                acc.add(not_last * (accumulator_next - accumulator - q_table * multiplicity * j_table + q_query * j_query));
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
                acc.add(not_last * selector_next * (b_next - y_out * b));
                acc.add(not_last * selector * (next - current - a * b));
                acc.add(not_last * (one - selector) * (next - current));
                acc.add(selector * (a - source));
                acc.add((one - selector) * a);
                acc.add((one - selector) * b);
                acc.add(last * (current - target));
            };

            for (std::size_t layer_index = 0;
                 layer_index < context.model.hidden_layers.size();
                 ++layer_index)
            {
                for (std::size_t local_head_index = 0;
                     local_head_index < context.model.hidden_layers[layer_index].heads.size();
                     ++local_head_index)
                {
                const auto *planned = evaluation_plan != nullptr
                                          ? &evaluation_plan->hidden_heads
                                                 .at(layer_index)
                                                 .at(local_head_index)
                                          : nullptr;
                const auto challenge = [&](std::string_view base)
                {
                    return hidden_challenge_name(base, layer_index, local_head_index);
                };
                const auto fallback_prefix = hidden_head_prefix(layer_index, local_head_index) + "_";
                const std::string &prefix = planned != nullptr
                                                ? planned->prefix
                                                : fallback_prefix;
                const auto beta_src = planned != nullptr
                                          ? planned->beta_src
                                          : challenges.at(challenge("beta_src_h"));
                const auto beta_dst = planned != nullptr
                                          ? planned->beta_dst
                                          : challenges.at(challenge("beta_dst_h"));
                const auto eta_src = planned != nullptr
                                         ? planned->eta_src
                                         : challenges.at(challenge("eta_src_h"));
                const auto eta_src_sq = planned != nullptr
                                            ? planned->eta_src_sq
                                            : eta_src * eta_src;
                const auto eta_dst = planned != nullptr
                                         ? planned->eta_dst
                                         : challenges.at(challenge("eta_dst_h"));
                const auto eta_dst_sq = planned != nullptr
                                            ? planned->eta_dst_sq
                                            : eta_dst * eta_dst;
                const auto eta_dst_cu = planned != nullptr
                                            ? planned->eta_dst_cu
                                            : eta_dst_sq * eta_dst;
                const auto eta_dst_4 = planned != nullptr
                                           ? planned->eta_dst_4
                                           : eta_dst_cu * eta_dst;
                const auto eta_dst_5 = planned != nullptr
                                           ? planned->eta_dst_5
                                           : eta_dst_4 * eta_dst;
                const auto beta_l = planned != nullptr
                                        ? planned->beta_l
                                        : challenges.at(challenge("beta_L_h"));
                const auto eta_l = planned != nullptr
                                       ? planned->eta_l
                                       : challenges.at(challenge("eta_L_h"));
                const auto beta_r = planned != nullptr
                                        ? planned->beta_r
                                        : challenges.at(challenge("beta_R_h"));
                const auto beta_exp = planned != nullptr
                                          ? planned->beta_exp
                                          : challenges.at(challenge("beta_exp_h"));
                const auto eta_exp = planned != nullptr
                                         ? planned->eta_exp
                                         : challenges.at(challenge("eta_exp_h"));
                const auto beta_elu = planned != nullptr
                                          ? planned->beta_elu
                                          : challenges.at(challenge("beta_ELU_h"));
                const auto eta_elu = planned != nullptr
                                         ? planned->eta_elu
                                         : challenges.at(challenge("eta_ELU_h"));
                const auto q_qry_elu_z = eval_z(
                    planned != nullptr
                        ? planned->elu_query_selector
                        : hidden_q_nd_h_selector_label(layer_index));
                const auto beta_t = planned != nullptr
                                        ? planned->beta_t
                                        : challenges.at(challenge("beta_t_h"));
                const auto eta_t = planned != nullptr
                                       ? planned->eta_t
                                       : challenges.at(challenge("eta_t_h"));

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
                const auto table_r_z = z_head("Table_R");
                const auto query_r_z = z_head("Query_R");
                const auto m_r_z = z_head("m_R");
                const auto alpha_z = z_head("alpha");
                const auto u_z = z_head("U");
                const auto inv_edge_z = z_head("inv_edge");
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
                const auto &scales = context.config.quantization;
                const auto aggregate_pre_scale =
                    FieldElement(scales.s_agg_pre.at(layer_index));
                const auto h_prime_scale =
                    FieldElement(scales.s_h_prime.at(layer_index).at(local_head_index));

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
                    planned != nullptr
                        ? planned->division_prefix
                        : prefix + "Div_",
                    planned != nullptr
                        ? planned->division_selector
                        : hidden_q_nd_h_selector_label(layer_index),
                    planned != nullptr
                        ? &planned->division
                        : nullptr);
                const auto q_div = eval_z(
                    planned != nullptr
                        ? planned->division_selector
                        : hidden_q_nd_h_selector_label(layer_index));
                acc.add(q_div * (h_agg_num_flat_z - sum_flat_z * h_agg_pre_flat_z));
                acc.add(q_div * (z_head("Div_X") - aggregate_pre_scale * h_agg_num_flat_z));
                acc.add(q_div * (z_head("Div_D") - h_prime_scale * sum_flat_z));
                acc.add(q_div * (z_head("Div_Y") - h_elu_in_flat_z));
                for (const auto &padding_value : {
                         sum_flat_z, h_agg_pre_flat_z, h_agg_num_flat_z,
                         h_elu_in_flat_z, h_agg_flat_z})
                {
                    acc.add((one - q_div) * padding_value);
                }
                }
            }

            for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index)
            {
                const auto *planned = evaluation_plan != nullptr
                                          ? &evaluation_plan->output_heads.at(head_index)
                                          : nullptr;
                const auto fallback_prefix = planned == nullptr
                                                 ? output_head_prefix(head_index) + "_"
                                                 : std::string{};
                const std::string &prefix = planned != nullptr
                                                ? planned->prefix
                                                : fallback_prefix;
                const auto beta_src_out = planned != nullptr
                                              ? planned->beta_src
                                              : challenges.at(output_challenge_name(
                                                    "beta_src_out", head_index));
                const auto beta_dst_out = planned != nullptr
                                              ? planned->beta_dst
                                              : challenges.at(output_challenge_name(
                                                    "beta_dst_out", head_index));
                const auto eta_src_out = planned != nullptr
                                             ? planned->eta_src
                                             : challenges.at(output_challenge_name(
                                                   "eta_src_out", head_index));
                const auto eta_dst_out = planned != nullptr
                                             ? planned->eta_dst
                                             : challenges.at(output_challenge_name(
                                                   "eta_dst_out", head_index));
                const auto eta_dst_out_sq = planned != nullptr
                                                ? planned->eta_dst_sq
                                                : eta_dst_out * eta_dst_out;
                const auto eta_dst_out_cu = planned != nullptr
                                                ? planned->eta_dst_cu
                                                : eta_dst_out_sq * eta_dst_out;
                const auto eta_dst_out_4 = planned != nullptr
                                               ? planned->eta_dst_4
                                               : eta_dst_out_cu * eta_dst_out;
                const auto eta_dst_out_5 = planned != nullptr
                                               ? planned->eta_dst_5
                                               : eta_dst_out_4 * eta_dst_out;
                const auto beta_l_out = planned != nullptr
                                            ? planned->beta_l
                                            : challenges.at(output_challenge_name(
                                                  "beta_L_out", head_index));
                const auto eta_l_out = planned != nullptr
                                           ? planned->eta_l
                                           : challenges.at(output_challenge_name(
                                                 "eta_L_out", head_index));
                const auto beta_r_out = planned != nullptr
                                            ? planned->beta_r
                                            : challenges.at(output_challenge_name(
                                                  "beta_R_out", head_index));
                const auto beta_exp_out = planned != nullptr
                                              ? planned->beta_exp
                                              : challenges.at(output_challenge_name(
                                                    "beta_exp_out", head_index));
                const auto eta_exp_out = planned != nullptr
                                             ? planned->eta_exp
                                             : challenges.at(output_challenge_name(
                                                   "eta_exp_out", head_index));
                const auto beta_t_out = planned != nullptr
                                            ? planned->beta_t
                                            : challenges.at(output_challenge_name(
                                                  "beta_t_out", head_index));
                const auto eta_t_out = planned != nullptr
                                           ? planned->eta_t
                                           : challenges.at(output_challenge_name(
                                                 "eta_t_out", head_index));

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
                const auto table_r_z = z_out("Table_R");
                const auto query_r_z = z_out("Query_R");
                const auto m_r_z = z_out("m_R");
                const auto alpha_z = z_out("alpha");
                const auto u_z = z_out("U");
                const auto inv_edge_z = z_out("inv_edge");
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
                    planned != nullptr
                        ? planned->division_prefix
                        : prefix + "Div_",
                    planned != nullptr
                        ? planned->division_selector
                        : "P_Q_div_out" + std::to_string(head_index),
                    planned != nullptr
                        ? &planned->division
                        : nullptr);
                const auto q_div = eval_z(
                    planned != nullptr
                        ? planned->division_selector
                        : "P_Q_div_out" + std::to_string(head_index));
                const auto y_pre_flat = z_out("Y_pre_flat");
                const auto sum_flat = z_out("Sum_flat");
                const auto y_pre_field_flat = z_out("Y_pre_field_flat");
                const auto y_agg_num_flat = z_out("Y_agg_num_flat");
                const auto bias_flat = z_out("Bias_flat");
                const auto y_flat = z_out("Y_flat");
                const auto &scales = context.config.quantization;
                const auto output_pre_scale = FieldElement(scales.s_out_pre);
                const auto output_y_prime_scale =
                    FieldElement(scales.s_y_prime.at(head_index));
                acc.add(q_div * (y_agg_num_flat - sum_flat * y_pre_field_flat));
                acc.add(q_div * (z_out("Div_X") - output_pre_scale * y_agg_num_flat));
                acc.add(q_div * (z_out("Div_D") - output_y_prime_scale * sum_flat));
                acc.add(q_div * (y_pre_flat - z_out("Div_Y")));
                acc.add(q_div * (y_flat - y_pre_flat - bias_flat));
                acc.add((one - q_div) * sum_flat);
                acc.add((one - q_div) * y_pre_field_flat);
                acc.add((one - q_div) * y_agg_num_flat);
                acc.add((one - q_div) * y_pre_flat);
                acc.add((one - q_div) * bias_flat);
                acc.add((one - q_div) * y_flat);
                add_coefficient_basis_binding(
                    planned != nullptr
                        ? planned->binding_prefix
                        : prefix + "Bind_",
                    planned != nullptr
                        ? planned->division_selector
                        : "P_Q_div_out" + std::to_string(head_index),
                    planned != nullptr
                        ? planned->y_flat_label
                        : prefix + "Y_flat",
                    external_evaluations.at(
                        planned != nullptr
                            ? planned->external_y_name
                            : output_external_eval_name("y", head_index)));
                const auto fallback_view_prefix = planned == nullptr
                                                      ? output_bias_view_prefix(head_index)
                                                      : std::string{};
                const std::string &view_prefix = planned != nullptr
                                                     ? planned->view_prefix
                                                     : fallback_view_prefix;
                const auto q_view = q_div;
                const auto query_view = eval_z(view_prefix + "Query");
                const auto j_view = eval_z(view_prefix + "J_qry");
                const auto r_view = eval_z(view_prefix + "R_qry");
                const auto r_view_next = eval_omega(view_prefix + "R_qry");
                const auto beta_view = planned != nullptr
                                           ? planned->beta_view
                                           : challenges.at(output_bias_view_beta_label(head_index));
                const auto eta_view = planned != nullptr
                                          ? planned->eta_view
                                          : challenges.at(output_bias_view_eta_label(head_index));
                acc.add(first * r_view);
                acc.add(not_last * (r_view_next - r_view - q_view * j_view));
                acc.add(q_view * ((query_view + beta_view) * j_view - one));
                acc.add((one - q_view) * j_view);
                acc.add(query_view - q_view * (eval_z("P_BiasIndex_edge") + eta_view * bias_flat));

                std::vector<CoefficientChunkLayoutEntry> fallback_basis_chunks;
                if (planned == nullptr)
                {
                    fallback_basis_chunks = coefficient_chunk_layout(
                        context.local.num_classes + registered_blinder_coefficient_count(
                                                        context,
                                                        output_bias_basis_label(head_index, "Coeff")),
                        context.domains.edge->size);
                }
                const auto gamma = planned != nullptr
                                       ? planned->gamma
                                       : challenges.at(output_bias_basis_gamma_label(head_index));
                const auto fallback_power_label = planned == nullptr
                                                      ? output_bias_basis_label(head_index, "Powers")
                                                      : std::string{};
                const std::string &power_label = planned != nullptr
                                                     ? planned->power_label
                                                     : fallback_power_label;
                const auto power = eval_z(power_label);
                const auto power_next = eval_omega(power_label);
                acc.add(first * (power - one));
                acc.add(not_last * (power_next - gamma * power));
                const auto chunk_count = planned != nullptr
                                             ? planned->basis_chunks.size()
                                             : fallback_basis_chunks.size();
                for (std::size_t chunk_index = 0;
                     chunk_index < chunk_count; ++chunk_index)
                {
                    const auto fallback_chunk = planned == nullptr
                                                    ? fallback_basis_chunks[chunk_index].chunk_index
                                                    : std::size_t{0};
                    const auto fallback_value_label = planned == nullptr
                                                          ? output_bias_basis_chunk_label(
                                                                head_index, fallback_chunk)
                                                          : std::string{};
                    const auto fallback_active_selector = planned == nullptr
                                                              ? output_bias_basis_region_selector_label(
                                                                    head_index, fallback_chunk, "active")
                                                              : std::string{};
                    const auto fallback_semantic_selector = planned == nullptr
                                                                ? output_bias_basis_region_selector_label(
                                                                      head_index, fallback_chunk, "semantic")
                                                                : std::string{};
                    const auto fallback_gap_selector = planned == nullptr
                                                           ? output_bias_basis_region_selector_label(
                                                                 head_index, fallback_chunk, "gap")
                                                           : std::string{};
                    const auto fallback_tail_selector = planned == nullptr
                                                            ? output_bias_basis_region_selector_label(
                                                                  head_index, fallback_chunk, "tail")
                                                            : std::string{};
                    const auto fallback_accumulator_label = planned == nullptr
                                                                ? output_bias_basis_chunk_accumulator_label(
                                                                      head_index, fallback_chunk)
                                                                : std::string{};
                    const auto *planned_chunk = planned != nullptr
                                                    ? &planned->basis_chunks[chunk_index]
                                                    : nullptr;
                    const auto &value_label = planned != nullptr
                                                  ? planned_chunk->value
                                                  : fallback_value_label;
                    const auto &active_selector = planned != nullptr
                                                      ? planned_chunk->active_selector
                                                      : fallback_active_selector;
                    const auto &semantic_selector = planned != nullptr
                                                        ? planned_chunk->semantic_selector
                                                        : fallback_semantic_selector;
                    const auto &gap_selector = planned != nullptr
                                                   ? planned_chunk->gap_selector
                                                   : fallback_gap_selector;
                    const auto &tail_selector = planned != nullptr
                                                    ? planned_chunk->tail_selector
                                                    : fallback_tail_selector;
                    const auto &accumulator_label = planned != nullptr
                                                        ? planned_chunk->accumulator
                                                        : fallback_accumulator_label;
                    const auto chunk_value = eval_z(value_label);
                    const auto q_active = eval_z(active_selector);
                    const auto q_semantic = eval_z(semantic_selector);
                    const auto q_gap = eval_z(gap_selector);
                    const auto q_tail = eval_z(tail_selector);
                    acc.add(q_semantic + q_gap + q_tail - q_active);
                    acc.add((one - q_active) * chunk_value);
                    acc.add(q_gap * chunk_value);
                    acc.add(q_semantic * (chunk_value - bias_flat));
                    const auto basis_accumulator = eval_z(accumulator_label);
                    const auto basis_accumulator_next =
                        eval_omega(accumulator_label);
                    acc.add(first * basis_accumulator);
                    acc.add(not_last * (basis_accumulator_next - basis_accumulator - q_active * power * chunk_value));
                }
            }

            std::vector<BindingRelationMeta> local_binding_metadata;
            const std::vector<BindingRelationMeta> *binding_metadata =
                evaluation_plan != nullptr
                    ? &evaluation_plan->binding_metadata
                    : nullptr;
            if (binding_metadata == nullptr)
            {
                local_binding_metadata = build_binding_relation_metadata(context);
                binding_metadata = &local_binding_metadata;
            }
            for (const auto &binding : *binding_metadata)
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
                    acc.add(q_edge * (product - eval_z(relation.alpha_object) * source_edge));
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
                        acc.add(q_edge * ((query + beta) * inverse - one));
                        acc.add((one - q_edge) * inverse);
                        acc.add(not_last * route_edge_step(
                                               accumulator_next, accumulator, query, beta,
                                               q_edge));
                        acc.add(query - (eval_z(route->index_object) + eta * eval_z(route->edge_value_object)));
                        acc.add(zero_on_unselected(q_edge, query));
                    }
                }
            }

            add_signed_division_constraints(
                "P_avg_Div_",
                "P_Q_div_avg",
                evaluation_plan != nullptr
                    ? &evaluation_plan->average_division
                    : nullptr);
            {
                const auto q_avg = eval_z("P_Q_div_avg");
                Scalar summed_heads =
                    FieldElement::zero();
                for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index)
                {
                    summed_heads += eval_z(
                        output_head_prefix(head_index) + "_Y_flat");
                }
                acc.add(q_avg * (eval_z("P_avg_Div_X") - summed_heads));
                acc.add(q_avg * (eval_z("P_avg_Div_D") - FieldElement(context.model.output_layer.heads.size())));
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
            const auto numerator =
                acc.value();

            if (return_numerator)
            {
                return numerator;
            }

            if constexpr (
                std::is_same_v<
                    Scalar,
                    FieldElement>)
            {
                return boundary
                               .zero_inverse
                               .has_value()
                           ? numerator *
                                 *boundary
                                      .zero_inverse
                           : numerator /
                                 zero_eval;
            }
            else
            {
                throw std::runtime_error(
                    "symbolic t_edge "
                    "evaluation requested "
                    "quotient division");
            }
        }

        throw std::runtime_error(
            "t_edge reached the removed single-output compatibility protocol");
    }
    TEdgeBoundProgram
    build_t_edge_bound_program(
        const ProtocolContext &context,
        const std::map<
            std::string,
            FieldElement> &challenges,
        const std::map<
            std::string,
            FieldElement>
            &external_evaluations,
        const TEdgeEvaluationPlan &plan)
    {
        TEdgeProgramBuilder builder;

        auto *previous_builder =
            active_t_edge_program_builder;

        active_t_edge_program_builder =
            &builder;

        try
        {
            const auto first =
                builder.boundary_first();

            const auto last =
                builder.boundary_last();

            const auto z =
                FieldElement(7);

            const auto omega_z =
                z *
                context.domains.edge->omega;

            const auto symbolic_eval =
                [&](const std::string &label,
                    const FieldElement &point)
                -> TEdgeProgramScalar
            {
                if (point == z)
                {
                    return builder.access(
                        label,
                        false);
                }

                if (point == omega_z)
                {
                    return builder.access(
                        label,
                        true);
                }

                throw std::runtime_error(
                    "symbolic t_edge "
                    "received unsupported "
                    "evaluation point");
            };

            (void)evaluate_t_edge_impl<
                false,
                TEdgeProgramScalar>(
                context,
                challenges,
                external_evaluations,
                symbolic_eval,
                z,
                true,
                nullptr,
                nullptr,
                nullptr,
                &plan,
                nullptr,
                &first,
                &last);
        }
        catch (...)
        {
            active_t_edge_program_builder =
                previous_builder;

            throw;
        }

        active_t_edge_program_builder =
            previous_builder;

        return builder.finish();
    }
    FieldElement evaluate_t_edge(
        const ProtocolContext &context,
        const std::map<std::string, FieldElement> &challenges,
        const std::map<std::string, FieldElement> &external_evaluations,
        const EvalFn &eval,
        const FieldElement &z,
        bool return_numerator,
        std::size_t *first_nonzero_term,
        FieldElement *first_nonzero_value,
        const std::vector<FieldElement> *precomputed_alpha_powers,
        const TEdgeEvaluationPlan *evaluation_plan,
        const QuotientBoundaryValues *boundary_values)
    {
        if (precomputed_alpha_powers != nullptr)
        {
            return evaluate_t_edge_impl<
                true,
                FieldElement>(
                context,
                challenges,
                external_evaluations,
                eval,
                z,
                return_numerator,
                first_nonzero_term,
                first_nonzero_value,
                precomputed_alpha_powers,
                evaluation_plan,
                boundary_values);
        }
        return evaluate_t_edge_impl<
            false,
            FieldElement>(
            context,
            challenges,
            external_evaluations,
            eval,
            z,
            return_numerator,
            first_nonzero_term,
            first_nonzero_value,
            nullptr,
            evaluation_plan,
            boundary_values);
    }

    FieldElement evaluate_t_n(
        const ProtocolContext &context,
        const std::map<std::string, FieldElement> &challenges,
        const EvalFn &eval,
        const FieldElement &z)
    {
        if (context.model.has_real_multihead)
        {
            // N 域负责把边上广播的量重新收回到目标点，并补上分母倒数、PSQ 目标表和 route 节点侧约束。
            const auto omega_z = z * context.domains.n->omega;
            const auto boundary = boundary_lagrange_values(*context.domains.n, z);
            const auto first = boundary.first;
            const auto last = boundary.last;
            const auto not_last = FieldElement::one() - last;
            const auto zero_eval = boundary.zero;
            const auto q_n = eval("P_Q_N", z);
            QuotientAccumulator acc(challenges.at("alpha_quot"));

            for (std::size_t layer_index = 0;
                 layer_index < context.model.hidden_layers.size();
                 ++layer_index)
            {
                for (std::size_t local_head_index = 0;
                     local_head_index < context.model.hidden_layers[layer_index].heads.size();
                     ++local_head_index)
                {
                const auto prefix = hidden_head_prefix(layer_index, local_head_index) + "_";
                const auto challenge = [&](std::string_view base)
                {
                    return hidden_challenge_name(base, layer_index, local_head_index);
                };
                const auto beta_src = challenges.at(challenge("beta_src_h"));
                const auto beta_dst = challenges.at(challenge("beta_dst_h"));
                const auto eta_src = challenges.at(challenge("eta_src_h"));
                const auto eta_dst = challenges.at(challenge("eta_dst_h"));
                const auto beta_t = challenges.at(challenge("beta_t_h"));
                const auto eta_t = challenges.at(challenge("eta_t_h"));
                const auto lambda_psq = challenges.at(challenge("lambda_psq_h"));

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
            }

            for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index)
            {
                const auto prefix = output_head_prefix(head_index) + "_";
                const auto beta_src_out = challenges.at(output_challenge_name("beta_src_out", head_index));
                const auto beta_dst_out = challenges.at(output_challenge_name("beta_dst_out", head_index));
                const auto eta_src_out = challenges.at(output_challenge_name("eta_src_out", head_index));
                const auto eta_dst_out = challenges.at(output_challenge_name("eta_dst_out", head_index));
                const auto beta_t_out = challenges.at(output_challenge_name("beta_t_out", head_index));
                const auto eta_t_out = challenges.at(output_challenge_name("eta_t_out", head_index));
                const auto lambda_out = challenges.at(output_challenge_name("lambda_out", head_index));

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
                    if (matrix.canonical_domain != context.domains.n->name || matrix.column_objects.size() != matrix.column_count || matrix.row_count != context.local.num_nodes)
                        throw std::runtime_error(
                            "invalid canonical matrix quotient registry: " + matrix.relation_id);
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
                    acc.add(q_n * (eval(matrix.compressed_object, z) - expected));
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
                                "aggregation quotient canonical matrix missing: " + object);
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
                    acc.add(q_n * (eval(relation.source_node_object, z) - compressed_columns(find_matrix(
                                                                              relation.source_matrix_object))));
                    acc.add(q_n * (eval(relation.target_node_object, z) - compressed_columns(find_matrix(
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
                        acc.add(q_n * ((table + beta) * inverse - FieldElement::one()));
                        acc.add((FieldElement::one() - q_n) * inverse);
                        acc.add(not_last * route_node_step(
                                               accumulator_next, accumulator, table, beta,
                                               q_n, multiplicity));
                        acc.add(table - (eval("P_AggBind_index_N", z) + eta * eval(route->node_value_object, z)));
                        acc.add(zero_on_unselected(q_n, table));
                    }
                }
            }

            return acc.value() / zero_eval;
        }

        throw std::runtime_error(
            "t_N reached the removed single-output compatibility protocol");
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
            const auto boundary = boundary_lagrange_values(*context.domains.in, z);
            const auto first = boundary.first;
            const auto last = boundary.last;
            const auto not_last = FieldElement::one() - last;
            const auto zero_eval = boundary.zero;
            QuotientAccumulator acc(
                challenges.at("alpha_quot"), return_numerator);
            for (std::size_t layer_index = 0;
                 layer_index < context.model.hidden_layers.size();
                 ++layer_index)
            {
                const auto q_valid = eval(
                    hidden_q_in_selector_label(layer_index), z);
                const auto q_invalid = FieldElement::one() - q_valid;
                for (std::size_t local_head_index = 0;
                     local_head_index < context.model.hidden_layers[layer_index].heads.size();
                     ++local_head_index)
                {
                const auto prefix = hidden_head_prefix(layer_index, local_head_index) + "_";
                const auto mu_name =
                    "mu_l" + std::to_string(layer_index) + "_h"
                    + std::to_string(local_head_index) + "_proj";
                acc.add(first * eval(prefix + "Acc_proj", z));
                acc.add(q_valid * acc_step(eval(prefix + "Acc_proj", omega_z), eval(prefix + "Acc_proj", z), eval(prefix + "a_proj", z), eval(prefix + "b_proj", z)));
                acc.add(not_last * q_invalid * (eval(prefix + "Acc_proj", omega_z) - eval(prefix + "Acc_proj", z)));
                acc.add(zero_on_unselected(q_valid, eval(prefix + "a_proj", z)));
                acc.add(zero_on_unselected(q_valid, eval(prefix + "b_proj", z)));
                acc.add(last * (eval(prefix + "Acc_proj", z) - external_evaluations.at(mu_name)));
                }
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

        // Non-formal legacy single-head compatibility branch.  The formal
        // context builder rejects has_real_multihead == false before any
        // quotient is evaluated, so these aliases cannot enter the schema.
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
            const auto boundary = boundary_lagrange_values(*context.domains.d, z);
            const auto first = boundary.first;
            const auto last = boundary.last;
            const auto not_last = FieldElement::one() - last;
            const auto zero_eval = boundary.zero;
            QuotientAccumulator acc(
                challenges.at("alpha_quot"), return_numerator);
            const auto semantic_external =
                [&](const std::string &eval_name,
                    const std::string &challenge_name)
            {
                return external_evaluations.at(eval_name) - context.domains.n->zero_polynomial_eval(
                                                                challenges.at(challenge_name)) *
                                                                external_evaluations.at(
                                                                    external_mask_eval_name(eval_name));
            };

            for (std::size_t layer_index = 0;
                 layer_index < context.model.hidden_layers.size();
                 ++layer_index)
            {
                const auto q_valid = eval(
                    hidden_q_d_h_selector_label(layer_index), z);
                const auto q_invalid = FieldElement::one() - q_valid;
                for (std::size_t local_head_index = 0;
                     local_head_index < context.model.hidden_layers[layer_index].heads.size();
                     ++local_head_index)
                {
                const auto prefix = hidden_head_prefix(layer_index, local_head_index) + "_";
                const auto challenge = [&](std::string_view base)
                {
                    return hidden_challenge_name(base, layer_index, local_head_index);
                };
                const auto suffix =
                    "l" + std::to_string(layer_index) + "_h"
                    + std::to_string(local_head_index);
                const auto src_semantic_eval =
                    external_evaluations.at(
                        "mu_" + suffix + "_src") -
                    context.domains.n->zero_polynomial_eval(
                        challenges.at(challenge("y_src_h"))) *
                        external_evaluations.at(
                            hidden_score_mask_eval_name(layer_index, local_head_index, "src"));
                const auto dst_semantic_eval =
                    external_evaluations.at(
                        "mu_" + suffix + "_dst") -
                    context.domains.n->zero_polynomial_eval(
                        challenges.at(challenge("y_dst_h"))) *
                        external_evaluations.at(
                            hidden_score_mask_eval_name(layer_index, local_head_index, "dst"));
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
                                                                   "mu_" + suffix + "_star",
                                                                   challenge("y_star_h"))));

                acc.add(first * eval(prefix + "Acc_agg_pre", z));
                acc.add(q_valid * acc_step(eval(prefix + "Acc_agg_pre", omega_z), eval(prefix + "Acc_agg_pre", z), eval(prefix + "a_agg_pre", z), eval(prefix + "b_agg_pre", z)));
                acc.add(not_last * q_invalid * (eval(prefix + "Acc_agg_pre", omega_z) - eval(prefix + "Acc_agg_pre", z)));
                acc.add(zero_on_unselected(q_valid, eval(prefix + "a_agg_pre", z)));
                acc.add(zero_on_unselected(q_valid, eval(prefix + "b_agg_pre", z)));
                acc.add(last * (eval(prefix + "Acc_agg_pre", z) - semantic_external(
                                                                      "mu_" + suffix + "_agg_pre",
                                                                      challenge("y_agg_pre_h"))));

                acc.add(first * eval(prefix + "Acc_agg", z));
                acc.add(q_valid * acc_step(eval(prefix + "Acc_agg", omega_z), eval(prefix + "Acc_agg", z), eval(prefix + "a_agg", z), eval(prefix + "b_agg", z)));
                acc.add(not_last * q_invalid * (eval(prefix + "Acc_agg", omega_z) - eval(prefix + "Acc_agg", z)));
                acc.add(zero_on_unselected(q_valid, eval(prefix + "a_agg", z)));
                acc.add(zero_on_unselected(q_valid, eval(prefix + "b_agg", z)));
                acc.add(last * (eval(prefix + "Acc_agg", z) - semantic_external(
                                                                  "mu_" + suffix + "_agg",
                                                                  challenge("y_agg_h"))));
                }
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

        // Non-formal legacy single-head compatibility branch; unreachable from
        // the fail-closed formal proof context above.
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
        const auto boundary = boundary_lagrange_values(*context.domains.cat, z);
        const auto first = boundary.first;
        const auto last = boundary.last;
        const auto not_last = FieldElement::one() - last;
        const auto zero_eval = boundary.zero;
        // cat 域只约束多头拼接与输出投影输入，不重复承担 edge 域上的 softmax/lookup 主链。
        QuotientAccumulator acc(challenges.at("alpha_quot"));

        for (std::size_t layer_index = 0; layer_index < context.model.hidden_layers.size(); ++layer_index)
        {
            const auto q_valid = eval(
                hidden_q_cat_selector_label(layer_index), z);
            const auto q_invalid = FieldElement::one() - q_valid;
            const auto prefix = hidden_layer_cat_prefix(layer_index);
            const auto mu_name = "mu_cat_l" + std::to_string(layer_index);
            const auto semantic_mu =
                external_evaluations.at(mu_name) - context.domains.n->zero_polynomial_eval(challenges.at(
                                                       hidden_concat_y_name(layer_index))) *
                                                       external_evaluations.at(
                                                           external_mask_eval_name(mu_name));
            acc.add(first * eval(prefix + "_Acc", z));
            acc.add(q_valid * acc_step(eval(prefix + "_Acc", omega_z), eval(prefix + "_Acc", z), eval(prefix + "_a", z), eval(prefix + "_b", z)));
            acc.add(not_last * q_invalid * (eval(prefix + "_Acc", omega_z) - eval(prefix + "_Acc", z)));
            acc.add(zero_on_unselected(q_valid, eval(prefix + "_a", z)));
            acc.add(zero_on_unselected(q_valid, eval(prefix + "_b", z)));
            acc.add(last * (eval(prefix + "_Acc", z) - semantic_mu));
        }

        for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index)
        {
            const auto q_valid = eval(output_q_cat_selector_label(), z);
            const auto q_invalid = FieldElement::one() - q_valid;
            const auto prefix = output_head_prefix(head_index) + "_";
            const auto mu_name = output_external_eval_name("proj", head_index);
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
        const auto boundary = boundary_lagrange_values(*context.domains.c, z);
        const auto first = boundary.first;
        const auto last = boundary.last;
        const auto not_last = FieldElement::one() - last;
        const auto zero_eval = boundary.zero;
        const auto q_valid = eval("P_Q_C_valid", z);
        const auto q_invalid = FieldElement::one() - q_valid;
        // C 域主要对齐输出层每头的源/目标打分绑定，以及 bias 后输出和多头平均的关系。
        QuotientAccumulator acc(challenges.at("alpha_quot"));

        for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index)
        {
            const auto prefix = output_head_prefix(head_index) + "_";
            const auto mu_src = output_external_eval_name("src", head_index);
            const auto mu_dst = output_external_eval_name("dst", head_index);
            const auto mu_star = output_external_eval_name("star", head_index);
            const auto star_semantic_eval =
                external_evaluations.at(mu_star) - context.domains.n->zero_polynomial_eval(challenges.at(
                                                       output_challenge_name("y_out_star", head_index))) *
                                                       external_evaluations.at(
                                                           external_mask_eval_name(mu_star));
            const auto src_semantic_eval =
                external_evaluations.at(mu_src) - context.domains.n->zero_polynomial_eval(challenges.at(
                                                      output_challenge_name("y_src_out", head_index))) *
                                                      external_evaluations.at(
                                                          output_score_mask_eval_name(head_index, "src"));
            const auto dst_semantic_eval =
                external_evaluations.at(mu_dst) - context.domains.n->zero_polynomial_eval(challenges.at(
                                                      output_challenge_name("y_dst_out", head_index))) *
                                                      external_evaluations.at(
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
            acc.add(not_last * (accumulator_next - accumulator - q_valid * multiplicity * inverse));
            acc.add(q_valid * ((table + beta_view) * inverse - FieldElement::one()));
            acc.add(q_invalid * inverse);
            acc.add(table - q_valid * (eval("P_BiasIndex_C", z) + eta_view * eval(
                                                                                 model_proof_local_label(output_bias_label(head_index)), z)));
            acc.add(multiplicity - eval("P_BiasMultiplicity_C", z));
        }

        return acc.value() / zero_eval;
    }

} // namespace gatzk::protocol
