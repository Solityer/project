#include "gatzk/protocol/schema.hpp"

#include "gatzk/protocol/challenges.hpp"
#include "gatzk/protocol/proof.hpp"

// 实现正式 proof schema 的命名与顺序规则。任何标签拼接方式变化都会影响 trace、proof、artifact 与 verifier 的一致性。

namespace gatzk::protocol
{

    // block 顺序是 proof serializer / parser 与 verifier 共享的固定契约。
    const std::vector<std::string> &proof_block_order()
    {
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

    ProofSchema build_proof_schema(const ProtocolContext &context)
    {
        ProofSchema schema;
        auto add = [&](std::string tag, std::string domain, PolynomialVisibility visibility, bool commitment, bool opening, bool quotient)
        {
            schema.entries.push_back(ProofSchemaEntry{
                .tag = std::move(tag),
                .domain = std::move(domain),
                .visibility = visibility,
                .opening_policy = context.config.opening_mode,
                .binding_mode = context.config.binding_mode,
                .enters_commitment = commitment,
                .enters_opening = opening,
                .enters_quotient = quotient,
            });
        };
        for (const auto &label : dynamic_commitment_labels(context))
        {
            const auto visibility = polynomial_visibility(context, label);
            add(label, "dynamic", visibility, true, true, true);
        }
        for (const auto &label : quotient_commitment_labels(context))
        {
            add(label, "quotient", PolynomialVisibility::PrivateWitness, true, true, true);
        }
        schema.private_plain_value_count = 0;
        return schema;
    }

    PolynomialVisibility polynomial_visibility(
        const ProtocolContext &context,
        const std::string &label)
    {
        if (context.public_commitments.contains(label) || label == "P_H")
        {
            return PolynomialVisibility::Public;
        }
        // These are the only final-output polynomial identities. Prefix matches
        // are forbidden: `P_Y_lin`, `P_Y_o0`, and head-local compressed outputs
        // contain private intermediate state.
        if (label == "P_Y" || label == "P_Y_star" || label == "P_Y_star_edge")
        {
            return PolynomialVisibility::PublicOutput;
        }
        return PolynomialVisibility::PrivateWitness;
    }

    bool schema_accepts_evaluation(PolynomialVisibility visibility, EvaluationKind kind)
    {
        if (visibility == PolynomialVisibility::PrivateWitness)
        {
            return kind == EvaluationKind::MaskedPrivateValue;
        }
        return kind == EvaluationKind::PublicValue;
    }

    std::string hidden_head_prefix(std::size_t global_head_index)
    {
        return "P_h" + std::to_string(global_head_index);
    }

    // 隐藏层的 concat 标签按层区分，方便把多层主线和笔记里的“多头拼接”步骤一一对应。
    std::string hidden_layer_concat_label(std::size_t layer_index, bool is_final_layer)
    {
        if (is_final_layer)
        {
            return "P_H_cat";
        }
        return "P_H_cat_l" + std::to_string(layer_index);
    }

    std::string hidden_layer_concat_star_label(std::size_t layer_index, bool is_final_layer)
    {
        if (is_final_layer)
        {
            return "P_H_cat_star";
        }
        return "P_H_cat_star_l" + std::to_string(layer_index);
    }

    std::string hidden_layer_cat_prefix(std::size_t layer_index, bool is_final_layer)
    {
        if (is_final_layer)
        {
            return "P_cat";
        }
        return "P_cat_l" + std::to_string(layer_index);
    }

    std::string hidden_weight_label(std::size_t global_head_index)
    {
        return "V_h" + std::to_string(global_head_index) + "_W";
    }

    std::string hidden_src_label(std::size_t global_head_index)
    {
        return "V_h" + std::to_string(global_head_index) + "_a_src";
    }

    std::string hidden_dst_label(std::size_t global_head_index)
    {
        return "V_h" + std::to_string(global_head_index) + "_a_dst";
    }

    // 输出层允许单头兼容标签，但这只是命名兼容，不是第二套协议。
    std::string output_head_prefix(std::size_t head_index, bool use_single_output_compat_labels)
    {
        if (use_single_output_compat_labels)
        {
            return "P_out";
        }
        return "P_out" + std::to_string(head_index);
    }

    std::string output_y_lin_label(std::size_t head_index, bool use_single_output_compat_labels)
    {
        if (use_single_output_compat_labels)
        {
            return "P_Y_lin";
        }
        return "P_Y_lin_o" + std::to_string(head_index);
    }

    std::string output_y_label(std::size_t head_index, bool use_single_output_compat_labels)
    {
        if (use_single_output_compat_labels)
        {
            return "P_Y";
        }
        return "P_Y_o" + std::to_string(head_index);
    }

    std::string output_weight_label(std::size_t head_index, bool use_single_output_compat_labels)
    {
        if (use_single_output_compat_labels)
        {
            return "V_out_W";
        }
        return "V_out" + std::to_string(head_index) + "_W";
    }

    std::string output_src_label(std::size_t head_index, bool use_single_output_compat_labels)
    {
        if (use_single_output_compat_labels)
        {
            return "V_out_a_src";
        }
        return "V_out" + std::to_string(head_index) + "_a_src";
    }

    std::string output_dst_label(std::size_t head_index, bool use_single_output_compat_labels)
    {
        if (use_single_output_compat_labels)
        {
            return "V_out_a_dst";
        }
        return "V_out" + std::to_string(head_index) + "_a_dst";
    }

    std::string output_challenge_name(std::string_view base, std::size_t head_index, bool use_single_output_compat_labels)
    {
        if (use_single_output_compat_labels)
        {
            return std::string(base);
        }
        return std::string(base) + "_o" + std::to_string(head_index);
    }

    std::string output_external_eval_name(std::string_view base, std::size_t head_index, bool use_single_output_compat_labels)
    {
        if (use_single_output_compat_labels)
        {
            return "mu_" + std::string(base);
        }
        return "mu_out" + std::to_string(head_index) + "_" + std::string(base);
    }

    std::string output_witness_scalar_name(std::string_view base, std::size_t head_index, bool use_single_output_compat_labels)
    {
        if (use_single_output_compat_labels)
        {
            return "S_" + std::string(base) + "_out";
        }
        return "S_" + std::string(base) + "_out" + std::to_string(head_index);
    }

    // 拼接层的挑战名单独保留层号，保证跨层复用时不会和别的头混淆。
    std::string hidden_concat_xi_name(std::size_t layer_index, bool is_final_layer)
    {
        if (is_final_layer)
        {
            return "xi_cat";
        }
        return "xi_cat_l" + std::to_string(layer_index);
    }

    std::string hidden_concat_y_name(std::size_t layer_index, bool is_final_layer)
    {
        if (is_final_layer)
        {
            return "y_cat";
        }
        return "y_cat_l" + std::to_string(layer_index);
    }

} // namespace gatzk::protocol
