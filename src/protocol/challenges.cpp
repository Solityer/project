#include "gatzk/protocol/challenges.hpp"

#include <cstdint>
#include <array>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

#include <openssl/sha.h>
#include <openssl/rand.h>

#include "gatzk/crypto/transcript.hpp"
#include "gatzk/protocol/schema.hpp"

namespace gatzk::protocol
{
    // 负责把 formal note 固定的 transcript 吸收顺序重放为 challenge 集合
    // 连接 public metadata、dynamic commitment、quotient commitment 与 prove/verify 共享挑战值
    // 这里的任何标签顺序变化都会直接改变 proof contract，因此只能做缓存和标签整理，不能改语义
    namespace
    {

        std::string sha256_hex(std::string_view data)
        {
            std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
            SHA256(
                reinterpret_cast<const unsigned char *>(data.data()),
                data.size(),
                digest.data());
            std::ostringstream out;
            out << std::hex << std::setfill('0');
            for (const auto byte : digest)
            {
                out << std::setw(2) << static_cast<unsigned>(byte);
            }
            return out.str();
        }

        std::string challenge_context_cache_key(const ProtocolContext &context)
        {
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
                   << context.config.srs_id << '|'
                   << context.config.srs_type << '|'
                   << context.config.privacy_policy << '|'
                   << context.config.input_privacy << '|'
                   << context.config.output_mode << '|'
                   << context.config.opening_mode << '|'
                   << context.config.binding_mode << '|'
                   << context.config.proof_task << '|'
                   << (context.config.allow_synthetic_model ? "synthetic" : "formal");
            for (const auto input_dim : context.config.d_in_profile)
            {
                stream << "|din=" << input_dim;
            }
            for (const auto &layer : context.config.hidden_profile)
            {
                stream << "|hid=" << layer.head_count << 'x' << layer.head_dim;
            }
            return stream.str();
        }

        std::string commitments_cache_fingerprint(
            const std::vector<std::string> &labels,
            const std::unordered_map<std::string, crypto::Commitment> &commitments)
        {
            std::ostringstream stream;
            for (const auto &label : labels)
            {
                const auto it = commitments.find(label);
                if (it == commitments.end())
                {
                    continue;
                }
                stream << '|' << label << '=' << it->second.point.to_string();
            }
            return stream.str();
        }

        void absorb_text_scalar(crypto::Transcript &transcript, const std::string &label, const std::string &value)
        {
            transcript.absorb_text(label, value);
        }

        void absorb(
            crypto::Transcript &transcript,
            const std::string &label,
            const std::unordered_map<std::string, crypto::Commitment> &commitments)
        {
            transcript.absorb_commitment(label, commitments.at(label).point);
        }

        void absorb_static(crypto::Transcript &transcript, const ProtocolContext &context, const std::string &label)
        {
            transcript.absorb_commitment(label, context.static_commitments.at(label).point);
        }

        void absorb_model_reblind(
            crypto::Transcript &transcript,
            const std::string &anchor_label,
            const std::unordered_map<std::string, crypto::Commitment> &commitments)
        {
            const auto proof_label = model_proof_local_label(anchor_label);
            const auto witness_label = model_reblind_witness_label(anchor_label);
            absorb(transcript, proof_label, commitments);
            absorb(transcript, witness_label, commitments);
        }

        void absorb_public(crypto::Transcript &transcript, const ProtocolContext &context, const std::string &label)
        {
            transcript.absorb_commitment(label, context.public_commitments.at(label).point);
        }

        void absorb_static_if_present(crypto::Transcript &transcript, const ProtocolContext &context, const std::string &label)
        {
            if (const auto it = context.static_commitments.find(label); it != context.static_commitments.end())
            {
                transcript.absorb_commitment(label, it->second.point);
            }
        }

        std::size_t hidden_head_width(const model::ModelParameters &parameters, const util::AppConfig &config)
        {
            if (parameters.has_real_multihead)
            {
                return model::max_hidden_head_dim(parameters);
            }
            return config.hidden_dim;
        }

        std::size_t concat_width(const model::ModelParameters &parameters, const util::AppConfig &config)
        {
            if (parameters.has_real_multihead)
            {
                return model::max_hidden_concat_width(parameters);
            }
            return config.hidden_dim;
        }

        std::string hidden_profile_string(const model::ModelParameters &parameters, const util::AppConfig &config)
        {
            std::ostringstream out;
            if (parameters.has_real_multihead && !parameters.hidden_profile.empty())
            {
                for (std::size_t i = 0; i < parameters.hidden_profile.size(); ++i)
                {
                    if (i != 0)
                    {
                        out << ',';
                    }
                    out << parameters.hidden_profile[i].head_count << 'x' << parameters.hidden_profile[i].head_dim;
                }
                return out.str();
            }
            out << "1x" << config.hidden_dim;
            return out.str();
        }

        std::string d_in_profile_string(const model::ModelParameters &parameters, const ProtocolContext &context)
        {
            std::ostringstream out;
            const auto profile = parameters.has_real_multihead && !parameters.d_in_profile.empty()
                                     ? parameters.d_in_profile
                                     : std::vector<std::size_t>{context.local.num_features};
            for (std::size_t i = 0; i < profile.size(); ++i)
            {
                if (i != 0)
                {
                    out << ',';
                }
                out << profile[i];
            }
            return out.str();
        }

        std::string size_vector_string(const std::vector<std::size_t> &values)
        {
            std::ostringstream out;
            for (std::size_t i = 0; i < values.size(); ++i)
            {
                if (i != 0)
                {
                    out << ':';
                }
                out << values[i];
            }
            return out.str();
        }

        void append_attention_head_dynamic_labels(std::vector<std::string> &labels, const std::string &prefix)
        {
            auto push = [&](const std::string &suffix)
            { labels.push_back(prefix + "_" + suffix); };
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
            push("J_src_node");
            push("J_src_edge");
            push("R_src_node");
            push("R_src");
            push("S");
            push("Z");
            push("Table_L");
            push("Query_L");
            push("m_L");
            push("J_tbl_L");
            push("J_qry_L");
            push("R_L");
            push("M");
            push("M_edge");
            push("s_max");
            push("C_max");
            push("Table_R");
            push("Query_R");
            push("m_R");
            push("J_tbl_R");
            push("J_qry_R");
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
            push("J_tbl_exp");
            push("J_qry_exp");
            push("R_exp");
            push("H_agg_pre");
            push("Sum_flat");
            push("H_agg_pre_flat");
            push("H_agg_num_flat");
            push("H_ELU_in_flat");
            push("H_agg_flat");
            push("Table_ELU");
            push("Query_ELU");
            push("m_ELU");
            push("J_tbl_ELU");
            push("J_qry_ELU");
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
            push("J_t_node");
            push("J_t_edge");
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
            push("J_dst_node");
            push("J_dst_edge");
            push("R_dst_node");
            push("R_dst");
        }

        void append_concat_labels(
            std::vector<std::string> &labels,
            const std::string &concat_label,
            const std::string &concat_star_label,
            const std::string &cat_prefix)
        {
            labels.push_back(concat_label);
            labels.push_back(concat_star_label);
            labels.push_back(cat_prefix + "_a");
            labels.push_back(cat_prefix + "_b");
            labels.push_back(cat_prefix + "_Acc");
        }

        void append_output_head_dynamic_labels(
            std::vector<std::string> &labels,
            const std::string &prefix,
            const std::string &y_lin_label,
            const std::string &y_label)
        {
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
            labels.push_back(prefix + "_J_src_node");
            labels.push_back(prefix + "_J_src_edge");
            labels.push_back(prefix + "_R_src_node");
            labels.push_back(prefix + "_R_src");
            labels.push_back(prefix + "_S");
            labels.push_back(prefix + "_Z");
            labels.push_back(prefix + "_Table_L");
            labels.push_back(prefix + "_Query_L");
            labels.push_back(prefix + "_m_L");
            labels.push_back(prefix + "_J_tbl_L");
            labels.push_back(prefix + "_J_qry_L");
            labels.push_back(prefix + "_R_L");
            labels.push_back(prefix + "_M");
            labels.push_back(prefix + "_M_edge");
            labels.push_back(prefix + "_s_max");
            labels.push_back(prefix + "_C_max");
            labels.push_back(prefix + "_Table_R");
            labels.push_back(prefix + "_Query_R");
            labels.push_back(prefix + "_m_R");
            labels.push_back(prefix + "_J_tbl_R");
            labels.push_back(prefix + "_J_qry_R");
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
            labels.push_back(prefix + "_J_tbl_exp");
            labels.push_back(prefix + "_J_qry_exp");
            labels.push_back(prefix + "_R_exp");
            labels.push_back(prefix + "_widehat_y_star");
            labels.push_back(prefix + "_w");
            labels.push_back(prefix + "_T");
            labels.push_back(prefix + "_T_edge");
            labels.push_back(prefix + "_Table_t");
            labels.push_back(prefix + "_Query_t");
            labels.push_back(prefix + "_m_t");
            labels.push_back(prefix + "_J_t_node");
            labels.push_back(prefix + "_J_t_edge");
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
            labels.push_back(prefix + "_J_dst_node");
            labels.push_back(prefix + "_J_dst_edge");
            labels.push_back(prefix + "_R_dst_node");
            labels.push_back(prefix + "_R_dst");
            labels.push_back(prefix + "_Y_pre_flat");
            labels.push_back(prefix + "_Sum_flat");
            labels.push_back(prefix + "_Y_pre_field_flat");
            labels.push_back(prefix + "_Y_agg_num_flat");
            labels.push_back(prefix + "_Bias_flat");
            labels.push_back(prefix + "_Y_flat");
        }

        void append_signed_division_labels(
            std::vector<std::string> &labels,
            const std::string &prefix,
            std::size_t range_bits)
        {
            for (const auto *suffix : {
                     "Active", "X", "D", "Y", "Sigma", "Zero",
                     "A", "A_inv", "D_inv", "Q", "R", "Delta"})
            {
                labels.push_back(prefix + suffix);
            }
            const auto limb_count = signed_division_limb_count(range_bits);
            for (const auto *value_name : {"A", "DMinus1", "Q", "R", "Delta"})
            {
                for (std::size_t limb = 0; limb < limb_count; ++limb)
                {
                    labels.push_back(signed_division_limb_label(prefix, value_name, limb));
                    for (const auto *column : {"Table", "Query", "m", "J_tbl", "J_qry", "R"})
                    {
                        labels.push_back(signed_division_range_label(prefix, value_name, limb, column));
                    }
                }
            }
        }

    }

    // 统一构造会进入 transcript、proof metadata 与 run manifest 的公开元数据。
    // 这些字段既参与 verifier 重放，也会影响 proof_size_bytes 中被计入的字符串载荷。
    PublicMetadata canonical_public_metadata(const ProtocolContext &context)
    {
        PublicMetadata metadata;
        metadata.protocol_name = "zkgat";
        metadata.proof_nonce.clear();
        metadata.proof_task = context.config.proof_task;
        metadata.srs_id = context.kzg.manifest.srs_id;
        if (context.compiler_artifact == nullptr)
        {
            throw std::runtime_error(
                "VerifierContext is missing the authenticated ProtocolCompilerArtifact");
        }
        const auto &compiler_artifact = *context.compiler_artifact;
        std::string artifact_reason;
        if (!validate_protocol_compiler_artifact(context, compiler_artifact, &artifact_reason))
        {
            throw std::runtime_error("invalid ProtocolCompilerArtifact: " + artifact_reason);
        }
        metadata.protocol_schema_id = sha256_hex(compiler_artifact.protocol_canonical_bytes);
        metadata.instance_schema_hash = sha256_hex(compiler_artifact.instance_canonical_bytes);
        if (context.kzg.manifest.protocol_schema_id.empty()
            || context.kzg.manifest.protocol_schema_id != metadata.protocol_schema_id)
        {
            throw std::runtime_error(
                "instance protocol compiler artifact is not authenticated by the SRS manifest");
        }
        metadata.srs_type = context.kzg.manifest.srs_type;
        metadata.srs_degree_bound = std::to_string(context.kzg.manifest.degree_bound);
        metadata.kzg_assumption_profile =
            context.kzg.manifest.kzg_assumption_profile_id
            + "|D_max=" + std::to_string(context.kzg.manifest.kzg_assumption_degree_bound)
            + "|security_bits=" + std::to_string(context.kzg.manifest.kzg_assumption_security_bits)
            + "|domain_catalog=" + context.kzg.manifest.kzg_assumption_domain_catalog_hash;
        metadata.privacy_policy = context.config.privacy_policy;
        metadata.input_privacy = context.config.input_privacy;
        metadata.output_mode = context.config.output_mode;
        metadata.opening_mode = context.config.opening_mode;
        metadata.binding_mode = context.config.binding_mode;
        metadata.model_arch = !context.config.model_arch_id.empty()
                                  ? context.config.model_arch_id
                                  : ("L=" + std::to_string(context.model.L)
                                     + "|hidden=" + hidden_profile_string(context.model, context.config)
                                     + "|K_out=" + std::to_string(context.model.K_out)
                                     + "|C=" + std::to_string(context.local.num_classes)
                                     + "|bias=per_output_head|average=signed_round_div");
        std::ostringstream model_registry;
        model_registry << "id="
                       << (!context.config.model_param_id.empty()
                               ? context.config.model_param_id
                               : ("checkpoint_bundle:" + context.config.checkpoint_bundle));
        for (std::size_t head = 0; head < context.model.hidden_heads.size(); ++head)
        {
            for (std::size_t column = 0;; ++column)
            {
                const auto label = hidden_weight_column_label(head, column);
                const auto it = context.static_commitments.find(label);
                if (it == context.static_commitments.end()) break;
                model_registry << '|' << label << '=' << it->second.point.to_string();
            }
            for (const auto &label : {hidden_src_label(head), hidden_dst_label(head)})
            {
                model_registry << '|' << label << '=' << context.static_commitments.at(label).point.to_string();
            }
        }
        for (std::size_t head = 0; head < context.model.output_layer.heads.size(); ++head)
        {
            for (std::size_t column = 0; column < context.local.num_classes; ++column)
            {
                const auto label = output_weight_column_label(head, column);
                model_registry << '|' << label << '=' << context.static_commitments.at(label).point.to_string();
            }
            for (const auto &label : {
                     output_src_label(head, false), output_dst_label(head, false),
                     output_bias_label(head, false)})
            {
                model_registry << '|' << label << '=' << context.static_commitments.at(label).point.to_string();
            }
        }
        metadata.model_params = "registry_sha256=" + sha256_hex(model_registry.str());

        std::ostringstream table_registry;
        table_registry << "id="
                       << (!context.config.static_table_id.empty()
                               ? context.config.static_table_id
                               : "tables=lrelu,elu,exp,range")
                       << "|generator=deterministic-v1";
        for (const auto *label : {
                 "V_T_H", "V_T_L_x", "V_T_L_y", "V_T_ELU_x", "V_T_ELU_y",
                 "V_T_exp_x", "V_T_exp_y", "V_T_range"})
        {
            if (const auto it = context.static_commitments.find(label);
                it != context.static_commitments.end())
            {
                table_registry << '|' << label << '=' << it->second.point.to_string();
            }
        }
        metadata.static_tables = "registry_sha256=" + sha256_hex(table_registry.str());
        const auto limb_count = (64U + context.config.range_bits - 1U) / context.config.range_bits;
        metadata.quant_config = !context.config.quant_cfg_id.empty()
                                    ? (context.config.quant_cfg_id
                                       + "|integer_bound_ledger="
                                       + sha256_hex(compiler_artifact.instance_canonical_bytes + "|integer_bounds"))
                                    : ("range_bits=" + std::to_string(context.config.range_bits)
                                       + "|limbs_i64=" + std::to_string(limb_count)
                                       + "|signed=balanced-field|round=nearest-ties-away"
                                       + "|output_rescale=16|output_average=signed_round_div"
                                       + "|integer_bound_ledger="
                                       + sha256_hex(compiler_artifact.instance_canonical_bytes + "|integer_bounds"));
        metadata.domain_cfg =
            "FH=" + context.domains.fh->name + ":" + std::to_string(context.domains.fh->size) + ",edge=" + context.domains.edge->name + ":" + std::to_string(context.domains.edge->size) + ",in=" + context.domains.in->name + ":" + std::to_string(context.domains.in->size) + ",d_h=" + context.domains.d->name + ":" + std::to_string(context.domains.d->size) + ",cat=" + context.domains.cat->name + ":" + std::to_string(context.domains.cat->size) + ",C=" + context.domains.c->name + ":" + std::to_string(context.domains.c->size) + ",N=" + context.domains.n->name + ":" + std::to_string(context.domains.n->size);
        metadata.dim_cfg =
            "N_total=" + std::to_string(context.dataset.num_nodes)
            + ",N=" + std::to_string(context.local.num_nodes)
            + ",E=" + std::to_string(context.local.edges.size())
            + ",G_batch=" + std::to_string(context.local.graph_count)
            + ",node_ptr=" + size_vector_string(context.local.node_ptr)
            + ",edge_ptr=" + size_vector_string(context.local.edge_ptr)
            + ",d_in=" + std::to_string(context.local.num_features)
            + ",d_h=" + std::to_string(hidden_head_width(context.model, context.config))
            + ",d_cat=" + std::to_string(concat_width(context.model, context.config))
            + ",C=" + std::to_string(context.local.num_classes)
            + ",B=" + std::to_string(context.config.range_bits)
            + ",K_out=" + std::to_string(context.model.K_out);
        metadata.encoding =
            "wire=v2|u64=le|field=canonical-le32|point=canonical-compressed"
            "|transcript=sha256-hash-chain|strings=type-tagged-u64-length"
            "|hash_suite=sha256|hash_to_field=rejection-counter-u32";
        metadata.padding_rule =
            "lookup+route+psq=hold-last|evaluation-columns=zero-pad+selector"
            "|coefficient-tail=coefficient-region-pad";
        metadata.degree_bound = !context.config.degree_bound_id.empty()
                                    ? (context.config.degree_bound_id
                                       + "|registry=" + sha256_hex(
                                           compiler_artifact.instance_canonical_bytes + "|degree_registry"))
                                    : ("D_max=" + std::to_string(context.kzg.manifest.degree_bound)
                                       + "|policy=object-ledger+quotient+blinding|registry="
                                       + sha256_hex(compiler_artifact.instance_canonical_bytes + "|degree_registry"));
        const auto security_summary = std::find_if(
            compiler_artifact.security_budget_ledger.begin(),
            compiler_artifact.security_budget_ledger.end(),
            [](const CompilerLedgerEntry &entry)
            { return entry.id == "budget:summary"; });
        if (security_summary == compiler_artifact.security_budget_ledger.end())
            throw std::runtime_error(
                "ProtocolCompilerArtifact is missing SecurityBudgetLedger summary");
        metadata.security_budget = security_summary->expression_or_certificate
            + "|ledger="
            + sha256_hex(
                compiler_artifact.instance_canonical_bytes + "|security_budget");
        metadata.public_output.clear();
        return metadata;
    }

    void absorb_public_metadata(crypto::Transcript &transcript, const PublicMetadata &metadata)
    {
        const std::array<std::pair<std::string_view, const std::string *>, 25> fields = {{
            {"protocol_name", &metadata.protocol_name}, {"proof_nonce", &metadata.proof_nonce},
            {"proof_task", &metadata.proof_task}, {"srs_id", &metadata.srs_id},
            {"protocol_schema_id", &metadata.protocol_schema_id}, {"instance_schema_hash", &metadata.instance_schema_hash},
            {"srs_type", &metadata.srs_type}, {"srs_degree_bound", &metadata.srs_degree_bound},
            {"kzg_assumption_profile", &metadata.kzg_assumption_profile}, {"privacy_policy", &metadata.privacy_policy},
            {"input_privacy", &metadata.input_privacy}, {"output_mode", &metadata.output_mode},
            {"opening_mode", &metadata.opening_mode}, {"binding_mode", &metadata.binding_mode},
            {"model_arch", &metadata.model_arch}, {"model_params", &metadata.model_params},
            {"static_tables", &metadata.static_tables}, {"quant_config", &metadata.quant_config},
            {"domain_cfg", &metadata.domain_cfg}, {"dim_cfg", &metadata.dim_cfg},
            {"encoding", &metadata.encoding}, {"padding_rule", &metadata.padding_rule},
            {"degree_bound", &metadata.degree_bound}, {"security_budget", &metadata.security_budget},
            {"public_output", &metadata.public_output},
        }};
        for (const auto &[name, value] : fields)
        {
            absorb_text_scalar(transcript, "M_pub." + std::string(name), *value);
        }
    }

    void absorb_binding_relation_metadata(
        crypto::Transcript &transcript,
        const BindingRelationMeta &metadata)
    {
        const auto base = "Pi_bind.meta:" + metadata.relation_id + ":";
        transcript.absorb_text(base + "relation_id", metadata.relation_id);
        transcript.absorb_text(base + "object_kind", metadata.object_kind);
        transcript.absorb_scalar(
            base + "layer_present",
            algebra::FieldElement(metadata.layer_id.has_value() ? 1 : 0));
        if (metadata.layer_id.has_value())
            transcript.absorb_scalar(
                base + "layer_id", algebra::FieldElement(*metadata.layer_id));
        transcript.absorb_scalar(
            base + "head_present",
            algebra::FieldElement(metadata.head_id.has_value() ? 1 : 0));
        if (metadata.head_id.has_value())
            transcript.absorb_scalar(
                base + "head_id", algebra::FieldElement(*metadata.head_id));
        auto absorb_text_vector = [&](std::string_view field,
                                      const std::vector<std::string> &values)
        {
            transcript.absorb_scalar(
                base + std::string(field) + ":count",
                algebra::FieldElement(values.size()));
            for (std::size_t i = 0; i < values.size(); ++i)
                transcript.absorb_text(
                    base + std::string(field) + ":" + std::to_string(i),
                    values[i]);
        };
        absorb_text_vector("source_objects", metadata.source_objects);
        absorb_text_vector("target_objects", metadata.target_objects);
        absorb_text_vector("work_domains", metadata.work_domains);
        transcript.absorb_scalar(
            base + "real_lengths:count",
            algebra::FieldElement(metadata.real_lengths.size()));
        for (std::size_t i = 0; i < metadata.real_lengths.size(); ++i)
            transcript.absorb_scalar(
                base + "real_lengths:" + std::to_string(i),
                algebra::FieldElement(metadata.real_lengths[i]));
        absorb_text_vector("quotient_relations", metadata.quotient_relations);
        transcript.absorb_scalar(
            base + "indexed_view_count", algebra::FieldElement(metadata.indexed_views.size()));
        for (std::size_t i = 0; i < metadata.indexed_views.size(); ++i)
        {
            const auto &view = metadata.indexed_views[i];
            const auto label = base + "indexed:" + std::to_string(i) + ":";
            for (const auto &[field, value] : std::vector<std::pair<std::string, std::string>>{
                     {"relation_id", view.relation_id},
                     {"canonical_object", view.canonical_object},
                     {"view_object", view.view_object},
                     {"canonical_domain", view.canonical_domain},
                     {"view_domain", view.view_domain},
                     {"index_map_id", view.index_map_id},
                     {"canonical_selector_id", view.canonical_selector_id},
                     {"query_selector_id", view.query_selector_id},
                     {"multiplicity_id", view.multiplicity_id},
                     {"eta_label", view.eta_challenge_label},
                     {"beta_label", view.beta_challenge_label}})
                transcript.absorb_text(label + field, value);
            transcript.absorb_scalar(
                label + "canonical_length", algebra::FieldElement(view.canonical_length));
            transcript.absorb_scalar(label + "view_length", algebra::FieldElement(view.view_length));
            transcript.absorb_scalar(
                label + "commitment_count", algebra::FieldElement(view.commitment_order.size()));
            for (std::size_t j = 0; j < view.commitment_order.size(); ++j)
                transcript.absorb_text(
                    label + "commitment:" + std::to_string(j), view.commitment_order[j]);
        }
        transcript.absorb_scalar(
            base + "shift_relation_count", algebra::FieldElement(metadata.shift_relations.size()));
        for (std::size_t i = 0; i < metadata.shift_relations.size(); ++i)
        {
            const auto &relation = metadata.shift_relations[i];
            const auto label = base + "shift:" + std::to_string(i) + ":";
            transcript.absorb_text(label + "relation_id", relation.relation_id);
            transcript.absorb_text(label + "challenge_label", relation.challenge_label);
            transcript.absorb_scalar(label + "lambda0", relation.lambda0);
            transcript.absorb_scalar(
                label + "exclusion_degree", algebra::FieldElement(relation.exclusion_degree));
            transcript.absorb_scalar(
                label + "max_difference_degree", algebra::FieldElement(relation.max_difference_degree));
            transcript.absorb_scalar(
                label + "source_count", algebra::FieldElement(relation.source_linear_maps.size()));
            for (std::size_t j = 0; j < relation.source_linear_maps.size(); ++j)
            {
                transcript.absorb_text(
                    label + "source_map:" + std::to_string(j), relation.source_linear_maps[j]);
                transcript.absorb_text(
                    label + "source_domain:" + std::to_string(j), relation.source_domains.at(j));
                transcript.absorb_text(
                    label + "source_kind:" + std::to_string(j), relation.source_value_kinds.at(j));
                transcript.absorb_scalar(
                    label + "point:" + std::to_string(j), relation.canonical_points.at(j));
                transcript.absorb_scalar(
                    label + "delta:" + std::to_string(j), relation.deltas.at(j));
            }
            for (std::size_t j = 0; j < relation.configured_degrees.size(); ++j)
            {
                transcript.absorb_scalar(
                    label + "configured_degree:" + std::to_string(j),
                    algebra::FieldElement(relation.configured_degrees[j]));
                transcript.absorb_scalar(
                    label + "shift_domain_size:" + std::to_string(j),
                    algebra::FieldElement(relation.shift_domain_sizes.at(j)));
            }
            for (std::size_t j = 0; j < relation.opening_claim_order.size(); ++j)
                transcript.absorb_text(
                    label + "opening_claim:" + std::to_string(j),
                    relation.opening_claim_order[j]);
        }
    }

    std::string canonical_output_digest(
        const std::vector<algebra::FieldElement> &public_logits,
        std::size_t node_count,
        std::size_t class_count)
    {
        const auto wide_count = static_cast<unsigned __int128>(node_count)
            * static_cast<unsigned __int128>(class_count);
        if (node_count == 0 || class_count == 0 || wide_count == 0
            || wide_count > std::numeric_limits<std::uint64_t>::max())
        {
            throw std::runtime_error("public output shape cannot be encoded as canonical u64");
        }
        const auto element_count = static_cast<std::uint64_t>(wide_count);
        if (public_logits.size() != element_count)
        {
            throw std::runtime_error("public output digest shape mismatch");
        }
        SHA256_CTX hash;
        SHA256_Init(&hash);
        constexpr std::string_view tag = "zkgat:public-output";
        SHA256_Update(&hash, tag.data(), tag.size());
        auto absorb_u64 = [&](std::uint64_t value)
        {
            std::array<std::uint8_t, 8> bytes{};
            for (std::size_t i = 0; i < bytes.size(); ++i)
            {
                bytes[i] = static_cast<std::uint8_t>((value >> (8U * i)) & 0xffU);
            }
            SHA256_Update(&hash, bytes.data(), bytes.size());
        };
        absorb_u64(static_cast<std::uint64_t>(node_count));
        absorb_u64(static_cast<std::uint64_t>(class_count));
        absorb_u64(element_count);
        std::array<std::uint8_t, 32> scalar{};
        for (const auto &value : public_logits)
        {
            value.write_little_endian(scalar.data(), scalar.size());
            SHA256_Update(&hash, scalar.data(), scalar.size());
        }
        std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
        SHA256_Final(digest.data(), &hash);
        std::ostringstream out;
        out << std::hex << std::setfill('0');
        for (const auto byte : digest) out << std::setw(2) << static_cast<unsigned>(byte);
        return out.str();
    }

    std::string fresh_proof_nonce()
    {
        std::array<unsigned char, 32> nonce{};
        if (RAND_bytes(nonce.data(), static_cast<int>(nonce.size())) != 1)
        {
            throw std::runtime_error("CSPRNG failed while sampling proof_nonce");
        }
        static constexpr char hex[] = "0123456789abcdef";
        std::string encoded(nonce.size() * 2, '0');
        for (std::size_t i = 0; i < nonce.size(); ++i)
        {
            encoded[2 * i] = hex[nonce[i] >> 4U];
            encoded[2 * i + 1] = hex[nonce[i] & 0x0fU];
        }
        return encoded;
    }

    std::string canonical_public_output_descriptor(
        const std::vector<algebra::FieldElement> &public_logits,
        std::size_t node_count,
        std::size_t class_count)
    {
        return "N=" + std::to_string(node_count)
            + "|C=" + std::to_string(class_count)
            + "|encoding=canonical-field-le32"
            + "|digest=" + canonical_output_digest(public_logits, node_count, class_count);
    }

    algebra::FieldElement derive_batch_opening_folding_challenge(
        const PublicMetadata &metadata,
        const std::string &group_id,
        const std::vector<crypto::Commitment> &commitments,
        const std::vector<algebra::FieldElement> &points,
        const std::vector<std::vector<algebra::FieldElement>> &values)
    {
        if (commitments.empty() || points.empty() || values.size() != commitments.size())
        {
            throw std::runtime_error("invalid batch opening point-group shape");
        }
        crypto::Transcript transcript("zkgat-point-group-v1");
        absorb_public_metadata(transcript, metadata);
        absorb_text_scalar(transcript, "point_group.id", group_id);
        transcript.absorb_scalar("point_group.commitment_count", algebra::FieldElement(commitments.size()));
        transcript.absorb_scalar("point_group.point_count", algebra::FieldElement(points.size()));
        for (std::size_t i = 0; i < commitments.size(); ++i)
        {
            if (values[i].size() != points.size())
            {
                throw std::runtime_error("invalid batch opening claimed-value shape");
            }
            absorb_text_scalar(transcript, "point_group.commitment_name." + std::to_string(i), commitments[i].name);
            transcript.absorb_commitment("point_group.commitment." + std::to_string(i), commitments[i].point);
        }
        for (std::size_t point_index = 0; point_index < points.size(); ++point_index)
        {
            transcript.absorb_scalar("point_group.point." + std::to_string(point_index), points[point_index]);
            for (std::size_t claim_index = 0; claim_index < commitments.size(); ++claim_index)
            {
                transcript.absorb_scalar(
                    "point_group.value." + std::to_string(point_index) + "." + std::to_string(claim_index),
                    values[claim_index][point_index]);
            }
        }
        return transcript.challenge("point_group.rho");
    }

    algebra::FieldElement derive_external_opening_folding_challenge(
        const PublicMetadata &metadata,
        const std::vector<std::pair<crypto::Commitment, algebra::FieldElement>> &claims,
        const std::vector<algebra::FieldElement> &points)
    {
        if (claims.empty() || claims.size() != points.size())
        {
            throw std::runtime_error("invalid external point-group shape");
        }
        crypto::Transcript transcript("zkgat-external-point-group-v1");
        absorb_public_metadata(transcript, metadata);
        transcript.absorb_scalar("external_group.claim_count", algebra::FieldElement(claims.size()));
        for (std::size_t i = 0; i < claims.size(); ++i)
        {
            absorb_text_scalar(transcript, "external_group.commitment_name." + std::to_string(i), claims[i].first.name);
            transcript.absorb_commitment("external_group.commitment." + std::to_string(i), claims[i].first.point);
            transcript.absorb_scalar("external_group.point." + std::to_string(i), points[i]);
            transcript.absorb_scalar("external_group.value." + std::to_string(i), claims[i].second);
        }
        return transcript.challenge("external_group.rho");
    }

    std::vector<std::string> dynamic_commitment_labels(const ProtocolContext &context)
    {
        // 这里返回的是 proof 中动态承诺的正式顺序。
        // family 形状越宽，标签数量越多，最终 proof_size_bytes 也会随之增加。
        if (!context.model.has_real_multihead)
        {
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
            "P_J_tbl_feat",
            "P_J_qry_feat",
            "P_R_feat",
        };
        for (std::size_t head_index = 0; head_index < context.model.hidden_heads.size(); ++head_index)
        {
            append_attention_head_dynamic_labels(labels, hidden_head_prefix(head_index));
            append_signed_division_labels(
                labels,
                hidden_head_prefix(head_index) + "_Div_",
                context.config.range_bits);
            labels.push_back(hidden_model_binding_blinder_label(head_index, "proj"));
            labels.push_back(hidden_model_binding_blinder_label(head_index, "src"));
            labels.push_back(hidden_model_binding_blinder_label(head_index, "dst"));
        }
        for (std::size_t layer_index = 0; layer_index < context.model.hidden_layers.size(); ++layer_index)
        {
            const bool is_final_layer = layer_index + 1 == context.model.hidden_layers.size();
            append_concat_labels(
                labels,
                hidden_layer_concat_label(layer_index, is_final_layer),
                hidden_layer_concat_star_label(layer_index, is_final_layer),
                hidden_layer_cat_prefix(layer_index, is_final_layer));
        }
        const bool legacy_single_output = false;
        for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index)
        {
            append_output_head_dynamic_labels(
                labels,
                output_head_prefix(head_index, legacy_single_output),
                output_y_lin_label(head_index, legacy_single_output),
                output_y_label(head_index, legacy_single_output));
            append_signed_division_labels(
                labels,
                output_head_prefix(head_index, legacy_single_output) + "_Div_",
                context.config.range_bits);
            labels.push_back(output_model_binding_blinder_label(head_index, "proj"));
            labels.push_back(output_model_binding_blinder_label(head_index, "src"));
            labels.push_back(output_model_binding_blinder_label(head_index, "dst"));
        }
        if (!legacy_single_output)
        {
            labels.push_back("P_Y_lin");
            labels.push_back("P_Y");
            append_signed_division_labels(labels, "P_avg_Div_", context.config.range_bits);
            for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index)
            {
                const auto prefix = output_head_prefix(head_index, false) + "_Bind_";
                labels.push_back(prefix + "a");
                labels.push_back(prefix + "b");
                labels.push_back(prefix + "Acc");
            }
            labels.push_back("P_avg_Bind_a");
            labels.push_back("P_avg_Bind_b");
            labels.push_back("P_avg_Bind_Acc");
        }
        return labels;
    }

    std::vector<std::string> quotient_commitment_labels(const ProtocolContext &context)
    {
        std::vector<std::string> quotient_labels;
        if (context.model.has_real_multihead)
        {
            quotient_labels = {"t_FH", "t_edge", "t_in", "t_d_h", "t_cat", "t_C", "t_N"};
        }
        else
        {
            quotient_labels = {"t_FH", "t_edge", "t_in", "t_d", "t_N"};
        }
        std::vector<std::string> labels;
        labels.reserve(quotient_labels.size() * 2);
        for (const auto &label : quotient_labels)
        {
            labels.push_back(label);
            labels.push_back(quotient_mask_label(label));
        }
        return labels;
    }

    std::string quotient_mask_label(const std::string &quotient_label)
    {
        if (quotient_label.rfind("t_", 0) != 0)
        {
            throw std::runtime_error("invalid quotient label for mask: " + quotient_label);
        }
        return "R_" + quotient_label;
    }

    std::map<std::string, algebra::FieldElement> derive_domain_opening_challenges(
        const ProtocolContext &context,
        crypto::Transcript &transcript)
    {
        struct DomainChallengeSpec
        {
            std::string label;
            std::shared_ptr<algebra::RootOfUnityDomain> domain;
        };
        const std::vector<DomainChallengeSpec> specs = {
            {"z_FH", context.domains.fh},
            {"z_edge", context.domains.edge},
            {"z_in", context.domains.in},
            {"z_d_h", context.domains.d},
            {"z_cat", context.domains.cat},
            {"z_C", context.domains.c},
            {"z_N", context.domains.n},
        };
        std::vector<algebra::FieldElement> prior_points;
        prior_points.reserve(specs.size() * 2);
        std::map<std::string, algebra::FieldElement> out;
        for (const auto &spec : specs)
        {
            if (spec.domain == nullptr)
                throw std::runtime_error(
                    "missing registered domain for opening challenge: "
                    + spec.label);
            const auto point = transcript.challenge_excluding(
                spec.label,
                [&](const algebra::FieldElement &candidate)
                {
                    const auto rotated = candidate * spec.domain->omega;
                    if (spec.domain->zero_polynomial_eval(candidate).is_zero()
                        || candidate == rotated)
                        return true;
                    return std::any_of(
                        prior_points.begin(), prior_points.end(),
                        [&](const algebra::FieldElement &prior)
                        { return candidate == prior || rotated == prior; });
                });
            out.emplace(spec.label, point);
            prior_points.push_back(point);
            prior_points.push_back(point * spec.domain->omega);
        }
        return out;
    }

    PublicOutputChallenges derive_public_output_challenges(
        const PublicMetadata &metadata,
        const crypto::Commitment &output_commitment,
        const std::vector<algebra::FieldElement> &public_logits,
        std::size_t node_count,
        std::size_t class_count)
    {
        if (node_count == 0 || class_count == 0
            || public_logits.size() != node_count * class_count)
        {
            throw std::runtime_error("public logits shape mismatch");
        }
        crypto::Transcript transcript("zkgat-public-output-v1");
        absorb_public_metadata(transcript, metadata);
        transcript.absorb_scalar("public_output.N", algebra::FieldElement(node_count));
        transcript.absorb_scalar("public_output.C", algebra::FieldElement(class_count));
        transcript.absorb_commitment("public_output.P_Y", output_commitment.point);
        for (std::size_t i = 0; i < public_logits.size(); ++i)
        {
            transcript.absorb_scalar("public_output.logit." + std::to_string(i), public_logits[i]);
        }
        PublicOutputChallenges out;
        out.point = transcript.challenge("public_output.z");
        return out;
    }

    // 按正式 transcript 顺序重放 challenge，供 prove 与 verify 共享同一挑战语义。
    // 当 quotient commitment 仍未生成时，函数只返回前半段 challenge，避免在 build_context 阶段改写顺序。
    std::map<std::string, algebra::FieldElement> replay_challenges(
        const ProtocolContext &context,
        const std::unordered_map<std::string, crypto::Commitment> &dynamic_commitments,
        const std::unordered_map<std::string, crypto::Commitment> &quotient_commitments,
        const PublicMetadata *proof_metadata)
    {
        const auto dynamic_labels = dynamic_commitment_labels(context);
        const auto quotient_labels = quotient_commitment_labels(context);

        if (context.model.has_real_multihead)
        {
            crypto::Transcript transcript("gatzkml");
            std::map<std::string, algebra::FieldElement> out;

            auto replay_signed_division = [&](const std::string &division_prefix)
            {
                for (const auto *suffix : {
                         "Active", "X", "D", "Y", "Sigma", "Zero",
                         "A", "A_inv", "D_inv", "Q", "R", "Delta"})
                {
                    absorb(transcript, division_prefix + suffix, dynamic_commitments);
                }
                const auto limb_count = signed_division_limb_count(context.config.range_bits);
                for (const auto *value_name : {"A", "DMinus1", "Q", "R", "Delta"})
                {
                    for (std::size_t limb = 0; limb < limb_count; ++limb)
                    {
                        absorb(
                            transcript,
                            signed_division_limb_label(division_prefix, value_name, limb),
                            dynamic_commitments);
                        for (const auto *column : {"Table", "Query", "m"})
                        {
                            absorb(
                                transcript,
                                signed_division_range_label(
                                    division_prefix, value_name, limb, column),
                                dynamic_commitments);
                        }
                        const auto beta_name = signed_division_beta_name(
                            division_prefix, value_name, limb);
                        out[beta_name] = transcript.challenge(beta_name);
                        for (const auto *column : {"J_tbl", "J_qry", "R"})
                        {
                            absorb(
                                transcript,
                                signed_division_range_label(
                                    division_prefix, value_name, limb, column),
                                dynamic_commitments);
                        }
                    }
                }
            };

            const std::size_t d_h = model::max_hidden_head_dim(context.model);
            const std::size_t d_cat = model::max_hidden_concat_width(context.model);

            const auto canonical_metadata = canonical_public_metadata(context);
            absorb_public_metadata(transcript, proof_metadata != nullptr ? *proof_metadata : canonical_metadata);
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
            absorb_public(transcript, context, "P_node_gid");
            absorb_public(transcript, context, "P_edge_gid");
            absorb_public(transcript, context, "P_Q_graph_new_edge");
            absorb_public(transcript, context, "P_Q_graph_end_edge");
            absorb_public(transcript, context, "P_Q_grp_new_edge");
            absorb_public(transcript, context, "P_Q_grp_end_edge");
            absorb_public(transcript, context, "P_Q_edge_valid");
            absorb_public(transcript, context, "P_Q_N");
            absorb_public(transcript, context, "P_Q_proj_valid");
            absorb_public(transcript, context, "P_Q_d_valid");
            absorb_public(transcript, context, "P_Q_cat_valid");
            absorb_public(transcript, context, "P_Q_C_valid");
            absorb_public(transcript, context, "P_BiasIndex_C");
            absorb_public(transcript, context, "P_BiasMultiplicity_C");
            absorb_public(transcript, context, "P_BiasIndex_edge");
            for (std::size_t head_index = 0; head_index < context.model.hidden_heads.size(); ++head_index)
            {
                absorb_public(transcript, context, "P_Q_div_h" + std::to_string(head_index));
            }
            for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index)
            {
                absorb_public(transcript, context, "P_Q_div_out" + std::to_string(head_index));
            }
            absorb_public(transcript, context, "P_Q_div_avg");
            absorb(transcript, "P_H", dynamic_commitments);
            absorb_static(transcript, context, "V_T_H");
            out["eta_feat"] = transcript.challenge("eta_feat");
            absorb(transcript, "P_Table_feat", dynamic_commitments);
            absorb(transcript, "P_Query_feat", dynamic_commitments);
            absorb(transcript, "P_m_feat", dynamic_commitments);
            out["beta_feat"] = transcript.challenge("beta_feat");
            absorb(transcript, "P_J_tbl_feat", dynamic_commitments);
            absorb(transcript, "P_J_qry_feat", dynamic_commitments);
            absorb(transcript, "P_R_feat", dynamic_commitments);

            std::size_t global_head_index = 0;
            for (std::size_t layer_index = 0; layer_index < context.model.hidden_layers.size(); ++layer_index)
            {
                const bool is_first_layer = layer_index == 0;
                const auto input_label =
                    is_first_layer
                        ? std::string("P_H")
                        : hidden_layer_concat_label(layer_index - 1, false);
                for (std::size_t head_index = 0; head_index < context.model.hidden_layers[layer_index].heads.size(); ++head_index, ++global_head_index)
                {
                    const auto prefix = hidden_head_prefix(global_head_index);
                    absorb(transcript, input_label, dynamic_commitments);
                    absorb(transcript, prefix + "_H_prime", dynamic_commitments);
                    for (std::size_t column = 0;
                         column < context.model.hidden_layers[layer_index].shape.head_dim;
                         ++column)
                    {
                        absorb_model_reblind(
                            transcript,
                            hidden_weight_column_label(global_head_index, column),
                            dynamic_commitments);
                    }
                    out["y_proj_h" + std::to_string(global_head_index)] = transcript.challenge("y_proj_h" + std::to_string(global_head_index));
                    out["xi_h" + std::to_string(global_head_index)] = transcript.challenge("xi_h" + std::to_string(global_head_index));

                    absorb(transcript, prefix + "_H_prime", dynamic_commitments);
                    absorb(transcript, prefix + "_E_src", dynamic_commitments);
                    absorb_model_reblind(
                        transcript, hidden_src_label(global_head_index), dynamic_commitments);
                    out["y_src_h" + std::to_string(global_head_index)] = transcript.challenge("y_src_h" + std::to_string(global_head_index));

                    absorb(transcript, prefix + "_H_prime", dynamic_commitments);
                    absorb(transcript, prefix + "_E_dst", dynamic_commitments);
                    absorb_model_reblind(
                        transcript, hidden_dst_label(global_head_index), dynamic_commitments);
                    out["y_dst_h" + std::to_string(global_head_index)] = transcript.challenge("y_dst_h" + std::to_string(global_head_index));

                    absorb(transcript, prefix + "_H_star", dynamic_commitments);
                    out["y_star_h" + std::to_string(global_head_index)] = transcript.challenge("y_star_h" + std::to_string(global_head_index));

                    absorb(transcript, prefix + "_E_src", dynamic_commitments);
                    absorb(transcript, prefix + "_H_star", dynamic_commitments);
                    absorb(transcript, prefix + "_H_src_star_edge", dynamic_commitments);
                    out["eta_src_h" + std::to_string(global_head_index)] = transcript.challenge("eta_src_h" + std::to_string(global_head_index));
                    absorb(transcript, prefix + "_Table_src", dynamic_commitments);
                    absorb(transcript, prefix + "_Query_src", dynamic_commitments);
                    absorb(transcript, prefix + "_m_src", dynamic_commitments);
                    out["beta_src_h" + std::to_string(global_head_index)] = transcript.challenge("beta_src_h" + std::to_string(global_head_index));
                    absorb(transcript, prefix + "_J_src_node", dynamic_commitments);
                    absorb(transcript, prefix + "_J_src_edge", dynamic_commitments);
                    absorb(transcript, prefix + "_R_src_node", dynamic_commitments);
                    absorb(transcript, prefix + "_R_src", dynamic_commitments);

                    absorb(transcript, prefix + "_S", dynamic_commitments);
                    absorb(transcript, prefix + "_Z", dynamic_commitments);
                    absorb_static(transcript, context, "V_T_L_x");
                    absorb_static(transcript, context, "V_T_L_y");
                    out["eta_L_h" + std::to_string(global_head_index)] = transcript.challenge("eta_L_h" + std::to_string(global_head_index));
                    absorb(transcript, prefix + "_Table_L", dynamic_commitments);
                    absorb(transcript, prefix + "_Query_L", dynamic_commitments);
                    absorb(transcript, prefix + "_m_L", dynamic_commitments);
                    out["beta_L_h" + std::to_string(global_head_index)] = transcript.challenge("beta_L_h" + std::to_string(global_head_index));
                    absorb(transcript, prefix + "_J_tbl_L", dynamic_commitments);
                    absorb(transcript, prefix + "_J_qry_L", dynamic_commitments);
                    absorb(transcript, prefix + "_R_L", dynamic_commitments);

                    absorb(transcript, prefix + "_M", dynamic_commitments);
                    absorb(transcript, prefix + "_M_edge", dynamic_commitments);
                    absorb(transcript, prefix + "_Delta", dynamic_commitments);
                    absorb_static(transcript, context, "V_T_range");
                    absorb(transcript, prefix + "_Table_R", dynamic_commitments);
                    absorb(transcript, prefix + "_Query_R", dynamic_commitments);
                    absorb(transcript, prefix + "_m_R", dynamic_commitments);
                    out["beta_R_h" + std::to_string(global_head_index)] = transcript.challenge("beta_R_h" + std::to_string(global_head_index));
                    absorb(transcript, prefix + "_J_tbl_R", dynamic_commitments);
                    absorb(transcript, prefix + "_J_qry_R", dynamic_commitments);
                    absorb(transcript, prefix + "_R_R", dynamic_commitments);

                    absorb(transcript, prefix + "_Delta", dynamic_commitments);
                    absorb(transcript, prefix + "_U", dynamic_commitments);
                    absorb_static(transcript, context, "V_T_exp_x");
                    absorb_static(transcript, context, "V_T_exp_y");
                    out["eta_exp_h" + std::to_string(global_head_index)] = transcript.challenge("eta_exp_h" + std::to_string(global_head_index));
                    absorb(transcript, prefix + "_Table_exp", dynamic_commitments);
                    absorb(transcript, prefix + "_Query_exp", dynamic_commitments);
                    absorb(transcript, prefix + "_m_exp", dynamic_commitments);
                    out["beta_exp_h" + std::to_string(global_head_index)] = transcript.challenge("beta_exp_h" + std::to_string(global_head_index));
                    absorb(transcript, prefix + "_J_tbl_exp", dynamic_commitments);
                    absorb(transcript, prefix + "_J_qry_exp", dynamic_commitments);
                    absorb(transcript, prefix + "_R_exp", dynamic_commitments);

                    out["lambda_psq_h" + std::to_string(global_head_index)] = transcript.challenge("lambda_psq_h" + std::to_string(global_head_index));
                    absorb(transcript, prefix + "_T_psq", dynamic_commitments);
                    absorb(transcript, prefix + "_T_psq_edge", dynamic_commitments);
                    out["eta_t_h" + std::to_string(global_head_index)] = transcript.challenge("eta_t_h" + std::to_string(global_head_index));
                    absorb(transcript, prefix + "_Table_t", dynamic_commitments);
                    absorb(transcript, prefix + "_Query_t", dynamic_commitments);
                    absorb(transcript, prefix + "_m_t", dynamic_commitments);
                    out["beta_t_h" + std::to_string(global_head_index)] = transcript.challenge("beta_t_h" + std::to_string(global_head_index));
                    absorb(transcript, prefix + "_J_t_node", dynamic_commitments);
                    absorb(transcript, prefix + "_J_t_edge", dynamic_commitments);
                    absorb(transcript, prefix + "_R_t_node", dynamic_commitments);
                    absorb(transcript, prefix + "_R_t", dynamic_commitments);

                    absorb(transcript, prefix + "_H_agg_pre", dynamic_commitments);
                    absorb(transcript, prefix + "_H_agg_pre_star", dynamic_commitments);
                    out["y_agg_pre_h" + std::to_string(global_head_index)] = transcript.challenge("y_agg_pre_h" + std::to_string(global_head_index));
                    absorb(transcript, prefix + "_H_agg_pre", dynamic_commitments);
                    absorb(transcript, prefix + "_H_agg", dynamic_commitments);
                    absorb_static(transcript, context, "V_T_ELU_x");
                    absorb_static(transcript, context, "V_T_ELU_y");
                    absorb(transcript, prefix + "_H_ELU_in_flat", dynamic_commitments);
                    absorb(transcript, prefix + "_H_agg_flat", dynamic_commitments);
                    out["eta_ELU_h" + std::to_string(global_head_index)] = transcript.challenge("eta_ELU_h" + std::to_string(global_head_index));
                    absorb(transcript, prefix + "_Table_ELU", dynamic_commitments);
                    absorb(transcript, prefix + "_Query_ELU", dynamic_commitments);
                    absorb(transcript, prefix + "_m_ELU", dynamic_commitments);
                    out["beta_ELU_h" + std::to_string(global_head_index)] = transcript.challenge("beta_ELU_h" + std::to_string(global_head_index));
                    absorb(transcript, prefix + "_J_tbl_ELU", dynamic_commitments);
                    absorb(transcript, prefix + "_J_qry_ELU", dynamic_commitments);
                    absorb(transcript, prefix + "_R_ELU", dynamic_commitments);

                    absorb(transcript, prefix + "_H_agg", dynamic_commitments);
                    absorb(transcript, prefix + "_H_agg_star", dynamic_commitments);
                    out["y_agg_h" + std::to_string(global_head_index)] = transcript.challenge("y_agg_h" + std::to_string(global_head_index));

                    for (const auto *suffix : {
                             "_E_dst", "_M", "_Sum", "_inv", "_H_agg_star",
                             "_E_dst_edge", "_M_edge", "_Sum_edge", "_inv_edge", "_H_agg_star_edge"})
                    {
                        absorb(transcript, prefix + suffix, dynamic_commitments);
                    }
                    out["eta_dst_h" + std::to_string(global_head_index)] = transcript.challenge("eta_dst_h" + std::to_string(global_head_index));
                    absorb(transcript, prefix + "_Table_dst", dynamic_commitments);
                    absorb(transcript, prefix + "_Query_dst", dynamic_commitments);
                    absorb(transcript, prefix + "_m_dst", dynamic_commitments);
                    out["beta_dst_h" + std::to_string(global_head_index)] = transcript.challenge("beta_dst_h" + std::to_string(global_head_index));
                    absorb(transcript, prefix + "_J_dst_node", dynamic_commitments);
                    absorb(transcript, prefix + "_J_dst_edge", dynamic_commitments);
                    absorb(transcript, prefix + "_R_dst_node", dynamic_commitments);
                    absorb(transcript, prefix + "_R_dst", dynamic_commitments);
                    replay_signed_division(prefix + "_Div_");
                }

                const bool is_final_layer = layer_index + 1 == context.model.hidden_layers.size();
                absorb(transcript, hidden_layer_concat_label(layer_index, is_final_layer), dynamic_commitments);
                out[hidden_concat_xi_name(layer_index, is_final_layer)] =
                    transcript.challenge(hidden_concat_xi_name(layer_index, is_final_layer));
                absorb(transcript, hidden_layer_concat_star_label(layer_index, is_final_layer), dynamic_commitments);
                out[hidden_concat_y_name(layer_index, is_final_layer)] =
                    transcript.challenge(hidden_concat_y_name(layer_index, is_final_layer));
            }

            const bool legacy_single_output = false;
            const auto final_hidden_label =
                hidden_layer_concat_label(context.model.hidden_layers.size() - 1, true);
            for (std::size_t head_index = 0;
                 head_index < context.model.output_layer.heads.size();
                 ++head_index)
            {
                absorb(
                    transcript,
                    output_head_prefix(head_index, legacy_single_output) + "_Y_prime",
                    dynamic_commitments);
            }
            out["xi_out"] = transcript.challenge("xi_out");
            for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index)
            {
                const auto prefix = output_head_prefix(head_index, legacy_single_output);
                const auto y_label = output_y_label(head_index, legacy_single_output);
                absorb(transcript, final_hidden_label, dynamic_commitments);
                absorb(transcript, prefix + "_Y_prime", dynamic_commitments);
                for (std::size_t column = 0; column < context.local.num_classes; ++column)
                {
                    absorb_model_reblind(
                        transcript,
                        output_weight_column_label(head_index, column),
                        dynamic_commitments);
                }
                out[output_challenge_name("y_proj_out", head_index, legacy_single_output)] =
                    transcript.challenge(output_challenge_name("y_proj_out", head_index, legacy_single_output));
                absorb(transcript, prefix + "_Y_prime", dynamic_commitments);
                absorb(transcript, prefix + "_E_src", dynamic_commitments);
                absorb_model_reblind(
                    transcript,
                    output_src_label(head_index, legacy_single_output),
                    dynamic_commitments);
                out[output_challenge_name("y_src_out", head_index, legacy_single_output)] =
                    transcript.challenge(output_challenge_name("y_src_out", head_index, legacy_single_output));
                absorb(transcript, prefix + "_Y_prime", dynamic_commitments);
                absorb(transcript, prefix + "_E_dst", dynamic_commitments);
                absorb_model_reblind(
                    transcript,
                    output_dst_label(head_index, legacy_single_output),
                    dynamic_commitments);
                out[output_challenge_name("y_dst_out", head_index, legacy_single_output)] =
                    transcript.challenge(output_challenge_name("y_dst_out", head_index, legacy_single_output));
                absorb(transcript, prefix + "_Y_prime_star", dynamic_commitments);
                absorb(transcript, prefix + "_Y_prime_star_edge", dynamic_commitments);
                out[output_challenge_name("eta_src_out", head_index, legacy_single_output)] =
                    transcript.challenge(output_challenge_name("eta_src_out", head_index, legacy_single_output));
                absorb(transcript, prefix + "_Table_src", dynamic_commitments);
                absorb(transcript, prefix + "_Query_src", dynamic_commitments);
                absorb(transcript, prefix + "_m_src", dynamic_commitments);
                out[output_challenge_name("beta_src_out", head_index, legacy_single_output)] =
                    transcript.challenge(output_challenge_name("beta_src_out", head_index, legacy_single_output));
                absorb(transcript, prefix + "_J_src_node", dynamic_commitments);
                absorb(transcript, prefix + "_J_src_edge", dynamic_commitments);
                absorb(transcript, prefix + "_R_src_node", dynamic_commitments);
                absorb(transcript, prefix + "_R_src", dynamic_commitments);
                absorb(transcript, prefix + "_S", dynamic_commitments);
                absorb(transcript, prefix + "_Z", dynamic_commitments);
                absorb_static(transcript, context, "V_T_L_x");
                absorb_static(transcript, context, "V_T_L_y");
                out[output_challenge_name("eta_L_out", head_index, legacy_single_output)] =
                    transcript.challenge(output_challenge_name("eta_L_out", head_index, legacy_single_output));
                for (const auto *suffix : {"_Table_L", "_Query_L", "_m_L"})
                {
                    absorb(transcript, prefix + suffix, dynamic_commitments);
                }
                out[output_challenge_name("beta_L_out", head_index, legacy_single_output)] =
                    transcript.challenge(output_challenge_name("beta_L_out", head_index, legacy_single_output));
                for (const auto *suffix : {"_J_tbl_L", "_J_qry_L", "_R_L"})
                {
                    absorb(transcript, prefix + suffix, dynamic_commitments);
                }
                absorb(transcript, prefix + "_M", dynamic_commitments);
                absorb(transcript, prefix + "_M_edge", dynamic_commitments);
                absorb(transcript, prefix + "_Delta", dynamic_commitments);
                absorb_static(transcript, context, "V_T_range");
                for (const auto *suffix : {"_Table_R", "_Query_R", "_m_R"})
                {
                    absorb(transcript, prefix + suffix, dynamic_commitments);
                }
                out[output_challenge_name("beta_R_out", head_index, legacy_single_output)] =
                    transcript.challenge(output_challenge_name("beta_R_out", head_index, legacy_single_output));
                for (const auto *suffix : {"_J_tbl_R", "_J_qry_R", "_R_R"})
                {
                    absorb(transcript, prefix + suffix, dynamic_commitments);
                }
                absorb(transcript, prefix + "_Delta", dynamic_commitments);
                absorb(transcript, prefix + "_U", dynamic_commitments);
                absorb_static(transcript, context, "V_T_exp_x");
                absorb_static(transcript, context, "V_T_exp_y");
                out[output_challenge_name("eta_exp_out", head_index, legacy_single_output)] =
                    transcript.challenge(output_challenge_name("eta_exp_out", head_index, legacy_single_output));
                for (const auto *suffix : {"_Table_exp", "_Query_exp", "_m_exp"})
                {
                    absorb(transcript, prefix + suffix, dynamic_commitments);
                }
                out[output_challenge_name("beta_exp_out", head_index, legacy_single_output)] =
                    transcript.challenge(output_challenge_name("beta_exp_out", head_index, legacy_single_output));
                for (const auto *suffix : {"_J_tbl_exp", "_J_qry_exp", "_R_exp"})
                {
                    absorb(transcript, prefix + suffix, dynamic_commitments);
                }
                out[output_challenge_name("lambda_out", head_index, legacy_single_output)] =
                    transcript.challenge(output_challenge_name("lambda_out", head_index, legacy_single_output));
                absorb(transcript, prefix + "_T", dynamic_commitments);
                absorb(transcript, prefix + "_T_edge", dynamic_commitments);
                out[output_challenge_name("eta_t_out", head_index, legacy_single_output)] =
                    transcript.challenge(output_challenge_name("eta_t_out", head_index, legacy_single_output));
                absorb(transcript, prefix + "_Table_t", dynamic_commitments);
                absorb(transcript, prefix + "_Query_t", dynamic_commitments);
                absorb(transcript, prefix + "_m_t", dynamic_commitments);
                out[output_challenge_name("beta_t_out", head_index, legacy_single_output)] =
                    transcript.challenge(output_challenge_name("beta_t_out", head_index, legacy_single_output));
                absorb(transcript, prefix + "_J_t_node", dynamic_commitments);
                absorb(transcript, prefix + "_J_t_edge", dynamic_commitments);
                absorb(transcript, prefix + "_R_t_node", dynamic_commitments);
                absorb(transcript, prefix + "_R_t", dynamic_commitments);
                absorb(transcript, y_label, dynamic_commitments);
                absorb(transcript, prefix + "_Y_star", dynamic_commitments);
                out[output_challenge_name("y_out_star", head_index, legacy_single_output)] =
                    transcript.challenge(output_challenge_name("y_out_star", head_index, legacy_single_output));
                for (const auto *suffix : {
                         "_E_dst", "_M", "_Sum", "_inv", "_Y_star",
                         "_E_dst_edge", "_M_edge", "_Sum_edge", "_inv_edge", "_Y_star_edge"})
                {
                    absorb(transcript, prefix + suffix, dynamic_commitments);
                }
                out[output_challenge_name("eta_dst_out", head_index, legacy_single_output)] =
                    transcript.challenge(output_challenge_name("eta_dst_out", head_index, legacy_single_output));
                absorb(transcript, prefix + "_Table_dst", dynamic_commitments);
                absorb(transcript, prefix + "_Query_dst", dynamic_commitments);
                absorb(transcript, prefix + "_m_dst", dynamic_commitments);
                out[output_challenge_name("beta_dst_out", head_index, legacy_single_output)] =
                    transcript.challenge(output_challenge_name("beta_dst_out", head_index, legacy_single_output));
                absorb(transcript, prefix + "_J_dst_node", dynamic_commitments);
                absorb(transcript, prefix + "_J_dst_edge", dynamic_commitments);
                absorb(transcript, prefix + "_R_dst_node", dynamic_commitments);
                absorb(transcript, prefix + "_R_dst", dynamic_commitments);
                replay_signed_division(prefix + "_Div_");
            }

            replay_signed_division("P_avg_Div_");

            if (!legacy_single_output)
            {
                for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index)
                {
                    absorb(transcript, output_y_lin_label(head_index, false), dynamic_commitments);
                absorb(transcript, output_y_label(head_index, false), dynamic_commitments);
                    absorb_model_reblind(
                        transcript,
                        output_bias_label(head_index, false),
                        dynamic_commitments);
                    absorb(transcript, output_head_prefix(head_index, false) + "_Y_star", dynamic_commitments);
                    absorb(transcript, output_head_prefix(head_index, false) + "_Table_dst", dynamic_commitments);
                    absorb(transcript, output_head_prefix(head_index, false) + "_Query_dst", dynamic_commitments);
                    absorb(transcript, output_head_prefix(head_index, false) + "_Y_pre_flat", dynamic_commitments);
                    absorb(transcript, output_head_prefix(head_index, false) + "_Bias_flat", dynamic_commitments);
                    absorb(transcript, output_head_prefix(head_index, false) + "_Y_flat", dynamic_commitments);
                    for (const auto *suffix : {
                             "Active", "X", "D", "Y", "Sigma", "Zero",
                             "A", "A_inv", "D_inv", "Q", "R", "Delta"})
                    {
                        absorb(
                            transcript,
                            output_head_prefix(head_index, false) + "_Div_" + suffix,
                            dynamic_commitments);
                    }
                    const auto binding_metadata = build_binding_relation_metadata(context);
                    const auto relation_id = "bind:output:" + std::to_string(head_index);
                    const auto meta_it = std::find_if(
                        binding_metadata.begin(), binding_metadata.end(),
                        [&](const BindingRelationMeta &meta)
                        { return meta.relation_id == relation_id; });
                    if (meta_it == binding_metadata.end()
                        || meta_it->indexed_views.size() != 1)
                    {
                        throw std::runtime_error(
                            "output bias binding transcript metadata is missing");
                    }
                    const auto &view_meta = meta_it->indexed_views.front();
                    const auto *shift_meta_ptr = find_unique_shift_relation_by_kind(
                        *meta_it, "output-bias-terminal");
                    if (shift_meta_ptr == nullptr)
                        throw std::runtime_error(
                            "output bias terminal relation is not uniquely registered during replay");
                    const auto &shift_meta = *shift_meta_ptr;
                    absorb_binding_relation_metadata(transcript, *meta_it);
                    out[view_meta.eta_challenge_label] = transcript.challenge(
                        view_meta.eta_challenge_label);
                    for (const auto *suffix : {"Table_can", "Query", "m_can"})
                        absorb(
                            transcript,
                            output_bias_view_label(head_index, suffix),
                            dynamic_commitments);
                    out[view_meta.beta_challenge_label] = transcript.challenge(
                        view_meta.beta_challenge_label);
                    for (const auto *suffix : {"J_can", "J_qry", "R_can", "R_qry"})
                        absorb(
                            transcript,
                            output_bias_view_label(head_index, suffix),
                            dynamic_commitments);
                    if (shift_meta.configured_degrees.empty())
                    {
                        out[shift_meta.challenge_label] = transcript.challenge(
                            shift_meta.challenge_label);
                    }
                    else
                    {
                        absorb(
                            transcript,
                            output_bias_view_label(head_index, "R_shifted"),
                            dynamic_commitments);
                        const auto second_domain =
                            shift_meta.source_domains.at(1) == context.domains.c->name
                                ? context.domains.c
                                : context.domains.edge;
                        out[shift_meta.challenge_label] = transcript.challenge_excluding(
                            shift_meta.challenge_label,
                            [&](const algebra::FieldElement &candidate)
                            {
                                return second_domain->zero_polynomial_eval(
                                    candidate + shift_meta.deltas.at(1)).is_zero();
                            });
                    }
                }
                absorb(transcript, "P_Y_lin", dynamic_commitments);
                absorb(transcript, "P_Y", dynamic_commitments);
                for (const auto *suffix : {
                         "Active", "X", "D", "Y", "Sigma", "Zero",
                         "A", "A_inv", "D_inv", "Q", "R", "Delta"})
                {
                    absorb(transcript, "P_avg_Div_" + std::string(suffix), dynamic_commitments);
                }
            }
            else
            {
                absorb(transcript, "P_out_Y_prime", dynamic_commitments);
                absorb(transcript, "P_Y", dynamic_commitments);
                absorb(transcript, "P_out_Y_star", dynamic_commitments);
                absorb(transcript, "P_out_Table_dst", dynamic_commitments);
                absorb(transcript, "P_out_Query_dst", dynamic_commitments);
                absorb_static(transcript, context, output_bias_label(0, true));
            }
            out["y_out"] = transcript.challenge("y_out");

            for (const auto &label : dynamic_labels)
            {
                absorb(transcript, label, dynamic_commitments);
            }
            for (const auto &label : {
                     std::string("V_T_H"),
                     std::string("V_T_L_x"),
                     std::string("V_T_L_y"),
                     std::string("V_T_ELU_x"),
                     std::string("V_T_ELU_y"),
                     std::string("V_T_exp_x"),
                     std::string("V_T_exp_y"),
                     std::string("V_T_range"),
                 })
            {
                absorb_static_if_present(transcript, context, label);
            }
            out["alpha_quot"] = transcript.challenge("alpha_quot");

            if (quotient_commitments.empty())
            {
                return out;
            }

            for (const auto &label : quotient_labels)
            {
                absorb(transcript, label, quotient_commitments);
            }
            for (const auto &[label, point] :
                 derive_domain_opening_challenges(context, transcript))
                out[label] = point;
            return out;
        }

        crypto::Transcript transcript("gatzkml");
        std::map<std::string, algebra::FieldElement> out;

        const auto canonical_metadata = canonical_public_metadata(context);
        absorb_public_metadata(transcript, proof_metadata != nullptr ? *proof_metadata : canonical_metadata);
        transcript.absorb_scalar("N", algebra::FieldElement(context.local.num_nodes));
        transcript.absorb_scalar("E", algebra::FieldElement(context.local.edges.size()));
        transcript.absorb_scalar("d_in", algebra::FieldElement(context.local.num_features));
        transcript.absorb_scalar("d", algebra::FieldElement(context.config.hidden_dim));
        transcript.absorb_scalar("C", algebra::FieldElement(context.local.num_classes));
        transcript.absorb_scalar("B", algebra::FieldElement(context.config.range_bits));
        absorb_public(transcript, context, "P_I");
        absorb_public(transcript, context, "P_src");
        absorb_public(transcript, context, "P_dst");
        absorb_public(transcript, context, "P_node_gid");
        absorb_public(transcript, context, "P_edge_gid");
        absorb_public(transcript, context, "P_Q_graph_new_edge");
        absorb_public(transcript, context, "P_Q_graph_end_edge");
        absorb_public(transcript, context, "P_Q_grp_new_edge");
        absorb_public(transcript, context, "P_Q_grp_end_edge");
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

        for (const auto &label : dynamic_labels)
        {
            absorb(transcript, label, dynamic_commitments);
        }
        for (const auto &label : {
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
             })
        {
            absorb_static(transcript, context, label);
        }
        out["alpha_quot"] = transcript.challenge("alpha_quot");

        if (quotient_commitments.empty())
        {
            return out;
        }

        for (const auto &label : quotient_labels)
        {
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
