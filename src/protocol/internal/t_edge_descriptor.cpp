#include "protocol/internal/t_edge_descriptor.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "gatzk/protocol/schema.hpp"

namespace gatzk::protocol::internal
{
    namespace
    {
        using algebra::FieldElement;

        FieldElement challenge_at(
            const std::map<std::string, FieldElement> &challenges,
            std::string_view base,
            bool hidden,
            std::size_t layer_index,
            std::size_t head_index)
        {
            std::string challenge_base(base);
            if (!hidden &&
                challenge_base.size() >= 2 &&
                challenge_base.ends_with("_h"))
            {
                challenge_base.replace(
                    challenge_base.size() - 2,
                    2,
                    "_out");
            }

            const auto challenge_name = hidden
                ? hidden_challenge_name(
                      challenge_base,
                      layer_index,
                      head_index)
                : output_challenge_name(
                      challenge_base,
                      head_index);
            return challenges.at(challenge_name);
        }

        struct NameMatch
        {
            std::string prefix;
            std::string suffix;
            bool hidden = false;
            std::size_t layer_index = 0;
            std::size_t head_index = 0;
        };

        std::optional<NameMatch> match_name(
            std::string_view name,
            const TEdgeDescriptorShape &shape)
        {
            for (std::size_t layer_index = 0;
                 layer_index < shape.hidden_head_counts.size();
                 ++layer_index)
            {
                for (std::size_t head_index = 0;
                     head_index < shape.hidden_head_counts[layer_index];
                     ++head_index)
                {
                    const auto prefix =
                        hidden_head_prefix(layer_index, head_index) + "_";
                    if (name.rfind(prefix, 0) == 0)
                    {
                        return NameMatch{
                            .prefix = prefix,
                            .suffix = std::string(name.substr(prefix.size())),
                            .hidden = true,
                            .layer_index = layer_index,
                            .head_index = head_index};
                    }
                }
            }

            for (std::size_t head_index = 0;
                 head_index < shape.output_head_count;
                 ++head_index)
            {
                const auto prefix = output_head_prefix(head_index) + "_";
                if (name.rfind(prefix, 0) == 0)
                {
                    return NameMatch{
                        .prefix = prefix,
                        .suffix = std::string(name.substr(prefix.size())),
                        .hidden = false,
                        .layer_index = 0,
                        .head_index = head_index};
                }
            }
            return std::nullopt;
        }

        bool collision_sensitive(
            const NameMatch &match)
        {
            return (match.hidden &&
                    (match.suffix == "w_psq" ||
                     match.suffix == "T_psq_edge")) ||
                   (!match.hidden && match.suffix == "w");
        }
    } // namespace

    std::optional<TEdgeDerivedLinearDescriptor>
    make_t_edge_derived_linear_descriptor(
        std::string_view name,
        const TEdgeDescriptorShape &shape,
        const std::map<std::string, FieldElement> &challenges,
        const std::function<bool(std::string_view)> &collision_predicate)
    {
        const auto match = match_name(name, shape);
        if (!match.has_value())
        {
            return std::nullopt;
        }
        if (collision_sensitive(*match) &&
            collision_predicate &&
            collision_predicate(name))
        {
            return std::nullopt;
        }

        TEdgeDerivedLinearDescriptor descriptor;
        descriptor.target_label = std::string(name);

        const auto add =
            [&](const std::string &source,
                const FieldElement &scale)
        {
            descriptor.sources.push_back(
                TEdgeDerivedLinearSource{
                    .label = source,
                    .scale = scale});
        };
        const auto challenge =
            [&](std::string_view base)
        {
            return challenge_at(
                challenges,
                base,
                match->hidden,
                match->layer_index,
                match->head_index);
        };

        const auto &prefix = match->prefix;
        const auto &suffix = match->suffix;
        const bool hidden = match->hidden;

        if (suffix == "Table_L")
        {
            add("P_T_L_x", FieldElement::one());
            add("P_T_L_y", challenge("eta_L_h"));
        }
        else if (suffix == "Query_L")
        {
            add(prefix + "S", FieldElement::one());
            add(prefix + "Z", challenge("eta_L_h"));
        }
        else if (suffix == "S")
        {
            add(prefix + "E_src_edge", FieldElement::one());
            add(prefix + "E_dst_edge", FieldElement::one());
        }
        else if (suffix == "Delta")
        {
            add(prefix + "M_edge", FieldElement::one());
            add(prefix + "Z", -FieldElement::one());
        }
        else if (suffix == "Table_R")
        {
            add("P_T_range", FieldElement::one());
        }
        else if (suffix == "Query_R")
        {
            add(prefix + "Delta", FieldElement::one());
        }
        else if (suffix == "Table_exp")
        {
            add("P_T_exp_x", FieldElement::one());
            add("P_T_exp_y", challenge("eta_exp_h"));
        }
        else if (suffix == "Query_exp")
        {
            add(prefix + "Delta", FieldElement::one());
            add(prefix + "U", challenge("eta_exp_h"));
        }
        else if (hidden && suffix == "w_psq")
        {
            add(prefix + "U", FieldElement::one());
            add(prefix + "widehat_v_pre_star", challenge("lambda_psq_h"));
        }
        else if (hidden && suffix == "T_psq_edge")
        {
            add(prefix + "Sum_edge", FieldElement::one());
            add(prefix + "H_agg_pre_star_edge", challenge("lambda_psq_h"));
        }
        else if (hidden && suffix == "Table_ELU")
        {
            add("P_T_ELU_x", FieldElement::one());
            add("P_T_ELU_y", challenge("eta_ELU_h"));
        }
        else if (hidden && suffix == "Query_ELU")
        {
            add(prefix + "H_ELU_in_flat", FieldElement::one());
            add(prefix + "H_agg_flat", challenge("eta_ELU_h"));
        }
        else if (suffix == "Query_src")
        {
            const auto eta_src = challenge(
                hidden ? "eta_src_h" : "eta_src_out");
            add("P_src", FieldElement::one());
            add(prefix + "E_src_edge", eta_src);
            if (hidden)
            {
                add(prefix + "H_src_star_edge", eta_src * eta_src);
            }
        }
        else if (suffix == "Query_t")
        {
            add("P_dst", FieldElement::one());
            add(prefix + (hidden ? "T_psq_edge" : "T_edge"), challenge(
                                                                   hidden ? "eta_t_h" : "eta_t_out"));
        }
        else if (suffix == "Query_dst")
        {
            const auto eta_dst = challenge(
                hidden ? "eta_dst_h" : "eta_dst_out");
            const auto eta_dst_sq = eta_dst * eta_dst;
            const auto eta_dst_cu = eta_dst_sq * eta_dst;
            const auto eta_dst_4 = eta_dst_cu * eta_dst;
            const auto eta_dst_5 = eta_dst_4 * eta_dst;
            add("P_dst", FieldElement::one());
            add(prefix + "E_dst_edge", eta_dst);
            add(prefix + "M_edge", eta_dst_sq);
            add(prefix + "Sum_edge", eta_dst_cu);
            add(prefix + "inv_edge", eta_dst_4);
            add(prefix + (hidden ? "H_agg_star_edge" : "Y_star_edge"), eta_dst_5);
        }
        else if (!hidden && suffix == "w")
        {
            add(prefix + "U", FieldElement::one());
            add(prefix + "widehat_y_star", challenge("lambda_out"));
        }
        else if (!hidden && suffix == "T_edge")
        {
            add(prefix + "Sum_edge", FieldElement::one());
            add(prefix + "Y_star_edge", challenge("lambda_out"));
        }
        else
        {
            return std::nullopt;
        }
        return descriptor;
    }

    TEdgeDerivedLinearFlattenResult
    flatten_t_edge_derived_linear_sources(
        const TEdgeDerivedLinearDescriptor &descriptor,
        const TEdgeDescriptorLookup &lookup,
        const TEdgeRegisteredSourcePredicate &is_registered_source)
    {
        constexpr std::size_t kMaximumDepth = 64;
        TEdgeDerivedLinearFlattenResult result;
        std::unordered_set<std::string> active;
        std::unordered_map<std::string, std::size_t> positions;

        std::function<void(const std::string &, const FieldElement &, std::size_t)> visit;
        visit = [&](const std::string &label,
                    const FieldElement &scale,
                    std::size_t depth)
        {
            if (result.status != TEdgeDerivedLinearFlattenStatus::Success)
            {
                return;
            }
            if (scale.is_zero())
            {
                return;
            }
            if (depth > kMaximumDepth)
            {
                result.status =
                    TEdgeDerivedLinearFlattenStatus::DepthExceeded;
                return;
            }

            const auto nested = lookup ? lookup(label) : std::nullopt;
            if (nested.has_value())
            {
                if (!active.insert(label).second)
                {
                    result.status =
                        TEdgeDerivedLinearFlattenStatus::Cycle;
                    return;
                }
                for (const auto &source : nested->sources)
                {
                    visit(
                        source.label,
                        scale * source.scale,
                        depth + 1);
                    if (result.status !=
                        TEdgeDerivedLinearFlattenStatus::Success)
                    {
                        break;
                    }
                }
                active.erase(label);
                return;
            }

            if (!is_registered_source || !is_registered_source(label))
            {
                result.status =
                    TEdgeDerivedLinearFlattenStatus::UnknownSource;
                return;
            }

            const auto position = positions.find(label);
            if (position == positions.end())
            {
                positions.emplace(label, result.sources.size());
                result.sources.push_back(
                    TEdgeDerivedLinearSource{
                        .label = label,
                        .scale = scale});
            }
            else
            {
                auto &merged = result.sources[position->second].scale;
                merged += scale;
                if (merged.is_zero())
                {
                    positions.erase(position);
                }
            }
        };

        active.insert(descriptor.target_label);
        for (const auto &source : descriptor.sources)
        {
            visit(source.label, source.scale, 1);
            if (result.status != TEdgeDerivedLinearFlattenStatus::Success)
            {
                break;
            }
        }
        active.erase(descriptor.target_label);

        result.sources.erase(
            std::remove_if(
                result.sources.begin(),
                result.sources.end(),
                [](const auto &source)
                { return source.scale.is_zero(); }),
            result.sources.end());
        return result;
    }

} // namespace gatzk::protocol::internal
