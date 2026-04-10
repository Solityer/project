#include "gatzk/protocol/challenges.hpp"

#include <cstdint>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

#include "gatzk/crypto/transcript.hpp"
#include "gatzk/protocol/schema.hpp"

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

std::string challenge_context_cache_key(const ProtocolContext& context) {
    std::ostringstream stream;
    stream << context.config.project_root << '|'
           << context.config.dataset << '|'
           << context.config.data_root << '|'
           << context.config.cache_root << '|'
           << context.config.checkpoint_bundle << '|'
           << context.config.hidden_dim << '|'
           << context.config.num_classes << '|'
           << context.config.range_bits << '|'
           << context.config.seed << '|'
           << context.config.layer_count << '|'
           << context.config.K_out << '|'
           << context.config.batch_graphs << '|'
           << context.config.task_type << '|'
           << context.config.report_unit << '|'
           << context.config.batching_rule << '|'
           << context.config.subgraph_rule << '|'
           << context.config.self_loop_rule << '|'
           << context.config.edge_sort_rule << '|'
           << context.config.chunking_rule << '|'
           << (context.config.allow_synthetic_model ? "synthetic" : "formal");
    for (const auto input_dim : context.config.d_in_profile) {
        stream << "|din=" << input_dim;
    }
    for (const auto& layer : context.config.hidden_profile) {
        stream << "|hid=" << layer.head_count << 'x' << layer.head_dim;
    }
    return stream.str();
}

std::string commitments_cache_fingerprint(
    const std::vector<std::string>& labels,
    const std::unordered_map<std::string, crypto::Commitment>& commitments) {
    std::ostringstream stream;
    for (const auto& label : labels) {
        const auto it = commitments.find(label);
        if (it == commitments.end()) {
            continue;
        }
        stream << '|' << label << '=' << it->second.tau_evaluation.to_string();
    }
    return stream.str();
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

std::size_t hidden_head_width(const model::ModelParameters& parameters, const util::AppConfig& config) {
    if (parameters.has_real_multihead) {
        return model::max_hidden_head_dim(parameters);
    }
    return config.hidden_dim;
}

std::size_t concat_width(const model::ModelParameters& parameters, const util::AppConfig& config) {
    if (parameters.has_real_multihead) {
        return model::max_hidden_concat_width(parameters);
    }
    return config.hidden_dim;
}

std::string hidden_profile_string(const model::ModelParameters& parameters, const util::AppConfig& config) {
    std::ostringstream out;
    if (parameters.has_real_multihead && !parameters.hidden_profile.empty()) {
        for (std::size_t i = 0; i < parameters.hidden_profile.size(); ++i) {
            if (i != 0) {
                out << ',';
            }
            out << parameters.hidden_profile[i].head_count << 'x' << parameters.hidden_profile[i].head_dim;
        }
        return out.str();
    }
    out << "1x" << config.hidden_dim;
    return out.str();
}

std::string d_in_profile_string(const model::ModelParameters& parameters, const ProtocolContext& context) {
    std::ostringstream out;
    const auto profile = parameters.has_real_multihead && !parameters.d_in_profile.empty()
        ? parameters.d_in_profile
        : std::vector<std::size_t>{context.local.num_features};
    for (std::size_t i = 0; i < profile.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << profile[i];
    }
    return out.str();
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
    push("Table_L");
    push("Query_L");
    push("m_L");
    push("R_L");
    push("M");
    push("M_edge");
    push("s_max");
    push("C_max");
    push("Table_R");
    push("Query_R");
    push("m_R");
    push("R_R");
    push("Delta");
    push("U");
    push("Sum");
    push("Sum_edge");
    push("inv");
    push("inv_edge");
    push("alpha");
    push("Table_exp");
    push("Query_exp");
    push("m_exp");
    push("R_exp");
    push("H_agg_pre");
    push("H_agg_pre_flat");
    push("H_agg_flat");
    push("Table_ELU");
    push("Query_ELU");
    push("m_ELU");
    push("R_ELU");
    push("H_agg_pre_star");
    push("H_agg_pre_star_edge");
    push("widehat_v_pre_star");
    push("w_psq");
    push("T_psq");
    push("T_psq_edge");
    push("Table_t");
    push("Query_t");
    push("m_t");
    push("R_t_node");
    push("R_t");
    push("PSQ");
    push("H_agg");
    push("H_agg_star");
    push("H_agg_star_edge");
    push("a_agg_pre");
    push("b_agg_pre");
    push("Acc_agg_pre");
    push("a_agg");
    push("b_agg");
    push("Acc_agg");
    push("Table_dst");
    push("Query_dst");
    push("m_dst");
    push("R_dst_node");
    push("R_dst");
}

void append_concat_labels(
    std::vector<std::string>& labels,
    const std::string& concat_label,
    const std::string& concat_star_label,
    const std::string& cat_prefix) {
    labels.push_back(concat_label);
    labels.push_back(concat_star_label);
    labels.push_back(cat_prefix + "_a");
    labels.push_back(cat_prefix + "_b");
    labels.push_back(cat_prefix + "_Acc");
}

void append_output_head_dynamic_labels(
    std::vector<std::string>& labels,
    const std::string& prefix,
    const std::string& y_lin_label,
    const std::string& y_label) {
    labels.push_back(prefix + "_Y_prime");
    labels.push_back(prefix + "_a_proj");
    labels.push_back(prefix + "_b_proj");
    labels.push_back(prefix + "_Acc_proj");
    labels.push_back(prefix + "_E_src");
    labels.push_back(prefix + "_E_dst");
    labels.push_back(prefix + "_a_src");
    labels.push_back(prefix + "_b_src");
    labels.push_back(prefix + "_Acc_src");
    labels.push_back(prefix + "_a_dst");
    labels.push_back(prefix + "_b_dst");
    labels.push_back(prefix + "_Acc_dst");
    labels.push_back(prefix + "_E_src_edge");
    labels.push_back(prefix + "_E_dst_edge");
    labels.push_back(prefix + "_Y_prime_star");
    labels.push_back(prefix + "_Y_prime_star_edge");
    labels.push_back(prefix + "_Table_src");
    labels.push_back(prefix + "_Query_src");
    labels.push_back(prefix + "_m_src");
    labels.push_back(prefix + "_R_src_node");
    labels.push_back(prefix + "_R_src");
    labels.push_back(prefix + "_S");
    labels.push_back(prefix + "_Z");
    labels.push_back(prefix + "_Table_L");
    labels.push_back(prefix + "_Query_L");
    labels.push_back(prefix + "_m_L");
    labels.push_back(prefix + "_R_L");
    labels.push_back(prefix + "_M");
    labels.push_back(prefix + "_M_edge");
    labels.push_back(prefix + "_s_max");
    labels.push_back(prefix + "_C_max");
    labels.push_back(prefix + "_Table_R");
    labels.push_back(prefix + "_Query_R");
    labels.push_back(prefix + "_m_R");
    labels.push_back(prefix + "_R_R");
    labels.push_back(prefix + "_Delta");
    labels.push_back(prefix + "_U");
    labels.push_back(prefix + "_Sum");
    labels.push_back(prefix + "_Sum_edge");
    labels.push_back(prefix + "_inv");
    labels.push_back(prefix + "_inv_edge");
    labels.push_back(prefix + "_alpha");
    labels.push_back(prefix + "_Table_exp");
    labels.push_back(prefix + "_Query_exp");
    labels.push_back(prefix + "_m_exp");
    labels.push_back(prefix + "_R_exp");
    labels.push_back(prefix + "_widehat_y_star");
    labels.push_back(prefix + "_w");
    labels.push_back(prefix + "_T");
    labels.push_back(prefix + "_T_edge");
    labels.push_back(prefix + "_Table_t");
    labels.push_back(prefix + "_Query_t");
    labels.push_back(prefix + "_m_t");
    labels.push_back(prefix + "_R_t_node");
    labels.push_back(prefix + "_R_t");
    labels.push_back(prefix + "_PSQ");
    labels.push_back(y_lin_label);
    labels.push_back(y_label);
    labels.push_back(prefix + "_Y_star");
    labels.push_back(prefix + "_Y_star_edge");
    labels.push_back(prefix + "_a_y");
    labels.push_back(prefix + "_b_y");
    labels.push_back(prefix + "_Acc_y");
    labels.push_back(prefix + "_Table_dst");
    labels.push_back(prefix + "_Query_dst");
    labels.push_back(prefix + "_m_dst");
    labels.push_back(prefix + "_R_dst_node");
    labels.push_back(prefix + "_R_dst");
}

}  // namespace

PublicMetadata canonical_public_metadata(const ProtocolContext& context) {
    PublicMetadata metadata;
    metadata.protocol_id = "gatzkml";
    metadata.dataset_name = context.dataset.name;
    metadata.task_type = context.local.task_type;
    metadata.report_unit = context.local.report_unit;
    metadata.graph_count = std::to_string(context.local.graph_count);
    metadata.L = std::to_string(context.model.has_real_multihead ? context.model.L : context.config.layer_count);
    metadata.hidden_profile = hidden_profile_string(context.model, context.config);
    metadata.d_in_profile = d_in_profile_string(context.model, context);
    metadata.K_out = std::to_string(context.model.has_real_multihead ? context.model.K_out : context.config.K_out);
    metadata.C = std::to_string(context.local.num_classes);
    metadata.batching_rule = context.local.batching_rule;
    metadata.subgraph_rule = context.local.subgraph_rule;
    metadata.self_loop_rule = context.local.self_loop_rule;
    metadata.edge_sort_rule = context.local.edge_sort_rule;
    metadata.chunking_rule = context.local.chunking_rule;
    metadata.model_arch_id = !context.config.model_arch_id.empty()
        ? context.config.model_arch_id
        : (context.model.has_real_multihead
            ? ("L=" + metadata.L + "|hidden=" + metadata.hidden_profile + "|K_out=" + metadata.K_out + "|C=" + metadata.C)
            : "legacy_single_head_debug");
    metadata.model_param_id = !context.config.model_param_id.empty()
        ? context.config.model_param_id
        : (context.model.has_real_multihead
            ? ("checkpoint_bundle:" + context.config.checkpoint_bundle)
            : ("synthetic_seed:" + std::to_string(context.config.seed)));
    metadata.static_table_id = !context.config.static_table_id.empty()
        ? context.config.static_table_id
        : "tables:lrelu+elu+exp+range";
    metadata.quant_cfg_id = !context.config.quant_cfg_id.empty()
        ? context.config.quant_cfg_id
        : ("range_bits=" + std::to_string(context.config.range_bits));
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
    metadata.degree_bound_id = !context.config.degree_bound_id.empty()
        ? context.config.degree_bound_id
        : (context.model.has_real_multihead ? "note-target:FH,edge,in,d_h,cat,C,N" : "legacy:FH,edge,in,d,N");
    return metadata;
}

void absorb_public_metadata(crypto::Transcript& transcript, const PublicMetadata& metadata) {
    absorb_text_scalar(transcript, "M_pub.protocol_id", metadata.protocol_id);
    absorb_text_scalar(transcript, "M_pub.dataset_name", metadata.dataset_name);
    absorb_text_scalar(transcript, "M_pub.task_type", metadata.task_type);
    absorb_text_scalar(transcript, "M_pub.report_unit", metadata.report_unit);
    absorb_text_scalar(transcript, "M_pub.graph_count", metadata.graph_count);
    absorb_text_scalar(transcript, "M_pub.L", metadata.L);
    absorb_text_scalar(transcript, "M_pub.hidden_profile", metadata.hidden_profile);
    absorb_text_scalar(transcript, "M_pub.d_in_profile", metadata.d_in_profile);
    absorb_text_scalar(transcript, "M_pub.K_out", metadata.K_out);
    absorb_text_scalar(transcript, "M_pub.C", metadata.C);
    absorb_text_scalar(transcript, "M_pub.batching_rule", metadata.batching_rule);
    absorb_text_scalar(transcript, "M_pub.subgraph_rule", metadata.subgraph_rule);
    absorb_text_scalar(transcript, "M_pub.self_loop_rule", metadata.self_loop_rule);
    absorb_text_scalar(transcript, "M_pub.edge_sort_rule", metadata.edge_sort_rule);
    absorb_text_scalar(transcript, "M_pub.chunking_rule", metadata.chunking_rule);
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
        append_attention_head_dynamic_labels(labels, hidden_head_prefix(head_index));
    }
    for (std::size_t layer_index = 0; layer_index < context.model.hidden_layers.size(); ++layer_index) {
        const bool is_final_layer = layer_index + 1 == context.model.hidden_layers.size();
        append_concat_labels(
            labels,
            hidden_layer_concat_label(layer_index, is_final_layer),
            hidden_layer_concat_star_label(layer_index, is_final_layer),
            hidden_layer_cat_prefix(layer_index, is_final_layer));
    }
    const bool legacy_single_output = context.model.output_layer.heads.size() == 1;
    for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index) {
        append_output_head_dynamic_labels(
            labels,
            output_head_prefix(head_index, legacy_single_output),
            output_y_lin_label(head_index, legacy_single_output),
            output_y_label(head_index, legacy_single_output));
    }
    if (!legacy_single_output) {
        labels.push_back("P_Y_lin");
        labels.push_back("P_Y");
    }
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
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, std::map<std::string, algebra::FieldElement>> cache;
    const auto dynamic_labels = dynamic_commitment_labels(context);
    const auto quotient_labels = quotient_commitment_labels(context);
    const auto cache_key =
        challenge_context_cache_key(context)
        + "|dyn" + commitments_cache_fingerprint(dynamic_labels, dynamic_commitments)
        + "|quot" + commitments_cache_fingerprint(quotient_labels, quotient_commitments);
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (const auto it = cache.find(cache_key); it != cache.end()) {
            return it->second;
        }
    }

    if (context.model.has_real_multihead) {
        crypto::Transcript transcript("gatzkml");
        std::map<std::string, algebra::FieldElement> out;

        const std::size_t d_h = model::max_hidden_head_dim(context.model);
        const std::size_t d_cat = model::max_hidden_concat_width(context.model);

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

        std::size_t global_head_index = 0;
        for (std::size_t layer_index = 0; layer_index < context.model.hidden_layers.size(); ++layer_index) {
            const bool is_first_layer = layer_index == 0;
            const auto input_label =
                is_first_layer
                ? std::string("P_H")
                : hidden_layer_concat_label(layer_index - 1, false);
            for (std::size_t head_index = 0; head_index < context.model.hidden_layers[layer_index].heads.size(); ++head_index, ++global_head_index) {
                const auto prefix = hidden_head_prefix(global_head_index);
                absorb(transcript, input_label, dynamic_commitments);
                absorb(transcript, prefix + "_H_prime", dynamic_commitments);
                absorb_static(transcript, context, hidden_weight_label(global_head_index));
                out["y_proj_h" + std::to_string(global_head_index)] = transcript.challenge("y_proj_h" + std::to_string(global_head_index));
                out["xi_h" + std::to_string(global_head_index)] = transcript.challenge("xi_h" + std::to_string(global_head_index));

                absorb(transcript, prefix + "_H_prime", dynamic_commitments);
                absorb(transcript, prefix + "_E_src", dynamic_commitments);
                absorb_static(transcript, context, hidden_src_label(global_head_index));
                out["y_src_h" + std::to_string(global_head_index)] = transcript.challenge("y_src_h" + std::to_string(global_head_index));

                absorb(transcript, prefix + "_H_prime", dynamic_commitments);
                absorb(transcript, prefix + "_E_dst", dynamic_commitments);
                absorb_static(transcript, context, hidden_dst_label(global_head_index));
                out["y_dst_h" + std::to_string(global_head_index)] = transcript.challenge("y_dst_h" + std::to_string(global_head_index));

                absorb(transcript, prefix + "_H_star", dynamic_commitments);
                out["y_star_h" + std::to_string(global_head_index)] = transcript.challenge("y_star_h" + std::to_string(global_head_index));

                absorb(transcript, prefix + "_E_src", dynamic_commitments);
                absorb(transcript, prefix + "_H_star", dynamic_commitments);
                out["eta_src_h" + std::to_string(global_head_index)] = transcript.challenge("eta_src_h" + std::to_string(global_head_index));
                out["beta_src_h" + std::to_string(global_head_index)] = transcript.challenge("beta_src_h" + std::to_string(global_head_index));

                absorb(transcript, prefix + "_S", dynamic_commitments);
                absorb(transcript, prefix + "_Z", dynamic_commitments);
                absorb_static(transcript, context, "V_T_L_x");
                absorb_static(transcript, context, "V_T_L_y");
                out["eta_L_h" + std::to_string(global_head_index)] = transcript.challenge("eta_L_h" + std::to_string(global_head_index));
                out["beta_L_h" + std::to_string(global_head_index)] = transcript.challenge("beta_L_h" + std::to_string(global_head_index));

                absorb(transcript, prefix + "_M", dynamic_commitments);
                absorb(transcript, prefix + "_M_edge", dynamic_commitments);
                absorb(transcript, prefix + "_Delta", dynamic_commitments);
                absorb_static(transcript, context, "V_T_range");
                out["beta_R_h" + std::to_string(global_head_index)] = transcript.challenge("beta_R_h" + std::to_string(global_head_index));

                absorb(transcript, prefix + "_Delta", dynamic_commitments);
                absorb(transcript, prefix + "_U", dynamic_commitments);
                absorb_static(transcript, context, "V_T_exp_x");
                absorb_static(transcript, context, "V_T_exp_y");
                out["eta_exp_h" + std::to_string(global_head_index)] = transcript.challenge("eta_exp_h" + std::to_string(global_head_index));
                out["beta_exp_h" + std::to_string(global_head_index)] = transcript.challenge("beta_exp_h" + std::to_string(global_head_index));

                out["lambda_psq_h" + std::to_string(global_head_index)] = transcript.challenge("lambda_psq_h" + std::to_string(global_head_index));
                absorb(transcript, prefix + "_T_psq", dynamic_commitments);
                absorb(transcript, prefix + "_T_psq_edge", dynamic_commitments);
                out["eta_t_h" + std::to_string(global_head_index)] = transcript.challenge("eta_t_h" + std::to_string(global_head_index));
                out["beta_t_h" + std::to_string(global_head_index)] = transcript.challenge("beta_t_h" + std::to_string(global_head_index));

                absorb(transcript, prefix + "_H_agg_pre", dynamic_commitments);
                absorb(transcript, prefix + "_H_agg_pre_star", dynamic_commitments);
                out["y_agg_pre_h" + std::to_string(global_head_index)] = transcript.challenge("y_agg_pre_h" + std::to_string(global_head_index));
                absorb(transcript, prefix + "_H_agg_pre", dynamic_commitments);
                absorb(transcript, prefix + "_H_agg", dynamic_commitments);
                absorb_static(transcript, context, "V_T_ELU_x");
                absorb_static(transcript, context, "V_T_ELU_y");
                out["eta_ELU_h" + std::to_string(global_head_index)] = transcript.challenge("eta_ELU_h" + std::to_string(global_head_index));
                out["beta_ELU_h" + std::to_string(global_head_index)] = transcript.challenge("beta_ELU_h" + std::to_string(global_head_index));

                absorb(transcript, prefix + "_H_agg", dynamic_commitments);
                absorb(transcript, prefix + "_H_agg_star", dynamic_commitments);
                out["y_agg_h" + std::to_string(global_head_index)] = transcript.challenge("y_agg_h" + std::to_string(global_head_index));

                out["eta_dst_h" + std::to_string(global_head_index)] = transcript.challenge("eta_dst_h" + std::to_string(global_head_index));
                out["beta_dst_h" + std::to_string(global_head_index)] = transcript.challenge("beta_dst_h" + std::to_string(global_head_index));
            }

            const bool is_final_layer = layer_index + 1 == context.model.hidden_layers.size();
            absorb(transcript, hidden_layer_concat_label(layer_index, is_final_layer), dynamic_commitments);
            out[hidden_concat_xi_name(layer_index, is_final_layer)] =
                transcript.challenge(hidden_concat_xi_name(layer_index, is_final_layer));
            absorb(transcript, hidden_layer_concat_star_label(layer_index, is_final_layer), dynamic_commitments);
            out[hidden_concat_y_name(layer_index, is_final_layer)] =
                transcript.challenge(hidden_concat_y_name(layer_index, is_final_layer));
        }

        const bool legacy_single_output = context.model.output_layer.heads.size() == 1;
        const auto final_hidden_label =
            hidden_layer_concat_label(context.model.hidden_layers.size() - 1, true);
        for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index) {
            const auto prefix = output_head_prefix(head_index, legacy_single_output);
            const auto y_label = output_y_label(head_index, legacy_single_output);
            absorb(transcript, final_hidden_label, dynamic_commitments);
            absorb(transcript, prefix + "_Y_prime", dynamic_commitments);
            absorb_static(transcript, context, output_weight_label(head_index, legacy_single_output));
            out[output_challenge_name("y_proj_out", head_index, legacy_single_output)] =
                transcript.challenge(output_challenge_name("y_proj_out", head_index, legacy_single_output));
            out[output_challenge_name("xi_out", head_index, legacy_single_output)] =
                transcript.challenge(output_challenge_name("xi_out", head_index, legacy_single_output));
            absorb(transcript, prefix + "_Y_prime", dynamic_commitments);
            absorb(transcript, prefix + "_E_src", dynamic_commitments);
            absorb_static(transcript, context, output_src_label(head_index, legacy_single_output));
            out[output_challenge_name("y_src_out", head_index, legacy_single_output)] =
                transcript.challenge(output_challenge_name("y_src_out", head_index, legacy_single_output));
            absorb(transcript, prefix + "_Y_prime", dynamic_commitments);
            absorb(transcript, prefix + "_E_dst", dynamic_commitments);
            absorb_static(transcript, context, output_dst_label(head_index, legacy_single_output));
            out[output_challenge_name("y_dst_out", head_index, legacy_single_output)] =
                transcript.challenge(output_challenge_name("y_dst_out", head_index, legacy_single_output));
            absorb(transcript, prefix + "_S", dynamic_commitments);
            absorb(transcript, prefix + "_Z", dynamic_commitments);
            absorb_static(transcript, context, "V_T_L_x");
            absorb_static(transcript, context, "V_T_L_y");
            out[output_challenge_name("eta_L_out", head_index, legacy_single_output)] =
                transcript.challenge(output_challenge_name("eta_L_out", head_index, legacy_single_output));
            out[output_challenge_name("beta_L_out", head_index, legacy_single_output)] =
                transcript.challenge(output_challenge_name("beta_L_out", head_index, legacy_single_output));
            absorb(transcript, prefix + "_M", dynamic_commitments);
            absorb(transcript, prefix + "_M_edge", dynamic_commitments);
            absorb(transcript, prefix + "_Delta", dynamic_commitments);
            absorb_static(transcript, context, "V_T_range");
            out[output_challenge_name("beta_R_out", head_index, legacy_single_output)] =
                transcript.challenge(output_challenge_name("beta_R_out", head_index, legacy_single_output));
            absorb(transcript, prefix + "_Delta", dynamic_commitments);
            absorb(transcript, prefix + "_U", dynamic_commitments);
            absorb_static(transcript, context, "V_T_exp_x");
            absorb_static(transcript, context, "V_T_exp_y");
            out[output_challenge_name("eta_exp_out", head_index, legacy_single_output)] =
                transcript.challenge(output_challenge_name("eta_exp_out", head_index, legacy_single_output));
            out[output_challenge_name("beta_exp_out", head_index, legacy_single_output)] =
                transcript.challenge(output_challenge_name("beta_exp_out", head_index, legacy_single_output));
            out[output_challenge_name("eta_src_out", head_index, legacy_single_output)] =
                transcript.challenge(output_challenge_name("eta_src_out", head_index, legacy_single_output));
            out[output_challenge_name("beta_src_out", head_index, legacy_single_output)] =
                transcript.challenge(output_challenge_name("beta_src_out", head_index, legacy_single_output));
            out[output_challenge_name("lambda_out", head_index, legacy_single_output)] =
                transcript.challenge(output_challenge_name("lambda_out", head_index, legacy_single_output));
            absorb(transcript, prefix + "_T", dynamic_commitments);
            absorb(transcript, prefix + "_T_edge", dynamic_commitments);
            out[output_challenge_name("eta_t_out", head_index, legacy_single_output)] =
                transcript.challenge(output_challenge_name("eta_t_out", head_index, legacy_single_output));
            out[output_challenge_name("beta_t_out", head_index, legacy_single_output)] =
                transcript.challenge(output_challenge_name("beta_t_out", head_index, legacy_single_output));
            absorb(transcript, y_label, dynamic_commitments);
            absorb(transcript, prefix + "_Y_star", dynamic_commitments);
            out[output_challenge_name("y_out_star", head_index, legacy_single_output)] =
                transcript.challenge(output_challenge_name("y_out_star", head_index, legacy_single_output));
            out[output_challenge_name("eta_dst_out", head_index, legacy_single_output)] =
                transcript.challenge(output_challenge_name("eta_dst_out", head_index, legacy_single_output));
            out[output_challenge_name("beta_dst_out", head_index, legacy_single_output)] =
                transcript.challenge(output_challenge_name("beta_dst_out", head_index, legacy_single_output));
        }

        if (!legacy_single_output) {
            for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index) {
                absorb(transcript, output_y_lin_label(head_index, false), dynamic_commitments);
                absorb(transcript, output_y_label(head_index, false), dynamic_commitments);
                absorb(transcript, output_head_prefix(head_index, false) + "_Y_star", dynamic_commitments);
                absorb(transcript, output_head_prefix(head_index, false) + "_Table_dst", dynamic_commitments);
                absorb(transcript, output_head_prefix(head_index, false) + "_Query_dst", dynamic_commitments);
            }
            absorb(transcript, "P_Y_lin", dynamic_commitments);
            absorb(transcript, "P_Y", dynamic_commitments);
        } else {
            absorb(transcript, "P_out_Y_prime", dynamic_commitments);
            absorb(transcript, "P_Y", dynamic_commitments);
            absorb(transcript, "P_out_Y_star", dynamic_commitments);
            absorb(transcript, "P_out_Table_dst", dynamic_commitments);
            absorb(transcript, "P_out_Query_dst", dynamic_commitments);
        }
        out["y_out"] = transcript.challenge("y_out");

        for (const auto& label : dynamic_labels) {
            absorb(transcript, label, dynamic_commitments);
        }
        for (const auto& label : {
                 std::string("V_T_H"),
                 std::string("V_T_L_x"),
                 std::string("V_T_L_y"),
                 std::string("V_T_ELU_x"),
                 std::string("V_T_ELU_y"),
                 std::string("V_T_exp_x"),
                 std::string("V_T_exp_y"),
                 std::string("V_T_range"),
             }) {
            absorb_static_if_present(transcript, context, label);
        }
        out["alpha_quot"] = transcript.challenge("alpha_quot");

        if (quotient_commitments.empty()) {
            std::lock_guard<std::mutex> lock(cache_mutex);
            cache.emplace(cache_key, out);
            return out;
        }

        for (const auto& label : quotient_labels) {
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
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            cache.emplace(cache_key, out);
        }
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

    for (const auto& label : dynamic_labels) {
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
        std::lock_guard<std::mutex> lock(cache_mutex);
        cache.emplace(cache_key, out);
        return out;
    }

    
    for (const auto& label : quotient_labels) {
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
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        cache.emplace(cache_key, out);
    }
    return out;
}
}
