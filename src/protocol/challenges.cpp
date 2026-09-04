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

        std::size_t hidden_head_width(const model::ModelParameters &parameters)
        {
            return model::max_hidden_head_dim(parameters);
        }

        std::size_t concat_width(const model::ModelParameters &parameters)
        {
            return model::max_hidden_concat_width(parameters);
        }

        std::string hidden_profile_string(const model::ModelParameters &parameters)
        {
            std::ostringstream out;
            if (!parameters.hidden_profile.empty())
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
            throw std::runtime_error("model hidden_profile must be non-empty");
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
            push("ScoreMask_src");
            push("ScoreMask_dst");
            push("a_src");
            push("b_src");
            push("Acc_src");
            push("a_dst");
            push("b_dst");
            push("Acc_dst");
            push("H_star");
            push("H_star_ExternalMask");
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
            push("Sum_flat"); // 注意力权重归一化分母
            push("H_agg_pre_flat");
            push("H_agg_num_flat"); // 整数特征聚合的分子
            push("H_ELU_in_flat");  // 经过除法和舍入后进入ELU的整数
            push("H_agg_flat");
            push("Table_ELU");
            push("Query_ELU");
            push("m_ELU");
            push("J_tbl_ELU");
            push("J_qry_ELU");
            push("R_ELU");
            push("H_agg_pre_star");
            push("H_agg_pre_star_ExternalMask");
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
            push("H_agg_star_ExternalMask");
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
            labels.push_back(external_mask_label(concat_star_label));
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
            labels.push_back(prefix + "_ScoreMask_src");
            labels.push_back(prefix + "_ScoreMask_dst");
            labels.push_back(prefix + "_a_src");
            labels.push_back(prefix + "_b_src");
            labels.push_back(prefix + "_Acc_src");
            labels.push_back(prefix + "_a_dst");
            labels.push_back(prefix + "_b_dst");
            labels.push_back(prefix + "_Acc_dst");
            labels.push_back(prefix + "_E_src_edge");
            labels.push_back(prefix + "_E_dst_edge");
            labels.push_back(prefix + "_Y_prime_star");
            labels.push_back(
                external_mask_label(
                    prefix + "_Y_prime_star"));
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
            labels.push_back(external_mask_label(prefix + "_Y_star"));
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
            labels.push_back(prefix + "_Y_pre_field_flat"); // 聚合辅助值
            labels.push_back(prefix + "_Y_agg_num_flat");   // 整数聚合分子
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
        if (context.compiler_artifact == nullptr || context.validated_compiler_artifact_identity != context.compiler_artifact.get())
        {
            throw std::runtime_error(
                "VerifierContext is missing the authenticated ProtocolCompilerArtifact identity");
        }
        const auto &compiler_artifact = *context.compiler_artifact;
        metadata.protocol_schema_id = sha256_hex(compiler_artifact.protocol_canonical_bytes);
        metadata.instance_schema_hash = sha256_hex(compiler_artifact.instance_canonical_bytes);
        if (context.kzg.manifest.protocol_schema_id.empty() || context.kzg.manifest.protocol_schema_id != metadata.protocol_schema_id)
        {
            throw std::runtime_error(
                "instance protocol compiler artifact is not authenticated by the SRS manifest");
        }
        metadata.srs_type = context.kzg.manifest.srs_type;
        metadata.srs_degree_bound = std::to_string(context.kzg.manifest.degree_bound);
        metadata.kzg_assumption_profile =
            context.kzg.manifest.kzg_assumption_profile_id + "|D_max=" + std::to_string(context.kzg.manifest.kzg_assumption_degree_bound) + "|security_bits=" + std::to_string(context.kzg.manifest.kzg_assumption_security_bits) + "|domain_catalog=" + context.kzg.manifest.kzg_assumption_domain_catalog_hash;
        metadata.privacy_policy = context.config.privacy_policy;
        metadata.input_privacy = context.config.input_privacy;
        metadata.output_mode = context.config.output_mode;
        metadata.opening_mode = context.config.opening_mode;
        metadata.binding_mode = context.config.binding_mode;
        metadata.model_arch = !context.config.model_arch_id.empty()
                                  ? context.config.model_arch_id
                                  : ("L=" + std::to_string(context.model.L) + "|hidden=" + hidden_profile_string(context.model) + "|K_out=" + std::to_string(context.model.K_out) + "|C=" + std::to_string(context.local.num_classes) + "|bias=per_output_head|average=signed_round_div");
        std::ostringstream model_registry;
        model_registry << "id="
                       << (!context.config.model_param_id.empty()
                               ? context.config.model_param_id
                               : ("checkpoint_bundle:" + context.config.checkpoint_bundle));
        for (std::size_t layer = 0;
             layer < context.model.hidden_layers.size();
             ++layer)
            for (std::size_t head = 0;
                 head < context.model.hidden_layers[layer].heads.size();
                 ++head)
            {
                for (std::size_t column = 0;; ++column)
                {
                    const auto label = hidden_weight_column_label(
                        layer, head, column);
                    const auto it = context.static_commitments.find(label);
                    if (it == context.static_commitments.end())
                        break;
                    model_registry << '|' << label << '=' << it->second.point.to_string();
                }
                for (const auto &label : {
                         hidden_src_label(layer, head),
                         hidden_dst_label(layer, head)})
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
                     output_src_label(head), output_dst_label(head),
                     output_bias_label(head)})
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
                       << "|generator=deterministic";
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
        const auto quantization_serialization =
            context.config.quantization.canonical_serialization();
        if (quantization_serialization.empty() || context.config.quant_cfg_id.empty())
        {
            throw std::runtime_error(
                "formal public metadata requires the validated canonical quantization scale chain");
        }
        metadata.quant_config =
            context.config.quant_cfg_id + "|" + quantization_serialization + "|integer_bound_ledger|entries=" + std::to_string(compiler_artifact.integer_bound_ledger.size()) + "|sha256=" + sha256_hex(integer_bound_ledger_canonical_bytes(compiler_artifact.integer_bound_ledger));
        metadata.domain_cfg =
            "FH=" + context.domains.fh->name + ":" + std::to_string(context.domains.fh->size) + ",edge=" + context.domains.edge->name + ":" + std::to_string(context.domains.edge->size) + ",in=" + context.domains.in->name + ":" + std::to_string(context.domains.in->size) + ",d_h=" + context.domains.d->name + ":" + std::to_string(context.domains.d->size) + ",cat=" + context.domains.cat->name + ":" + std::to_string(context.domains.cat->size) + ",C=" + context.domains.c->name + ":" + std::to_string(context.domains.c->size) + ",N=" + context.domains.n->name + ":" + std::to_string(context.domains.n->size);
        metadata.dim_cfg =
            "N_total=" + std::to_string(context.dataset.num_nodes) + ",N=" + std::to_string(context.local.num_nodes) + ",E=" + std::to_string(context.local.edges.size()) + ",G_batch=" + std::to_string(context.local.graph_count) + ",node_ptr=" + size_vector_string(context.local.node_ptr) + ",edge_ptr=" + size_vector_string(context.local.edge_ptr) + ",d_in=" + std::to_string(context.local.num_features) + ",d_h=" + std::to_string(hidden_head_width(context.model)) + ",d_cat=" + std::to_string(concat_width(context.model)) + ",C=" + std::to_string(context.local.num_classes) + ",B=" + std::to_string(context.config.range_bits) + ",K_out=" + std::to_string(context.model.K_out);
        metadata.encoding =
            "wire|u64=le|field=canonical-le32|point=canonical-compressed"
            "|transcript=sha256-hash-chain|strings=type-tagged-u64-length"
            "|hash_suite=sha256|hash_to_field=rejection-counter-u32";
        metadata.padding_rule =
            "lookup+route+psq=hold-last|evaluation-columns=zero-pad+selector"
            "|coefficient-tail=coefficient-region-pad";
        metadata.degree_bound = !context.config.degree_bound_id.empty()
                                    ? (context.config.degree_bound_id + "|registry=" + sha256_hex(compiler_artifact.instance_canonical_bytes + "|degree_registry"))
                                    : ("D_max=" + std::to_string(context.kzg.manifest.degree_bound) + "|policy=object-ledger+quotient+blinding|registry=" + sha256_hex(compiler_artifact.instance_canonical_bytes + "|degree_registry"));
        const auto security_summary = std::find_if(
            compiler_artifact.security_budget_ledger.begin(),
            compiler_artifact.security_budget_ledger.end(),
            [](const CompilerLedgerEntry &entry)
            { return entry.id == "budget:summary"; });
        if (security_summary == compiler_artifact.security_budget_ledger.end())
            throw std::runtime_error(
                "ProtocolCompilerArtifact is missing SecurityBudgetLedger summary");
        metadata.security_budget = security_summary->expression_or_certificate + "|ledger=" + sha256_hex(compiler_artifact.instance_canonical_bytes + "|security_budget");
        metadata.input_preprocess_id = context.config.input_preprocess_id;
        metadata.public_output.clear();
        return metadata;
    }

    void absorb_public_metadata(crypto::Transcript &transcript, const PublicMetadata &metadata)
    {
        const std::array<std::pair<std::string_view, const std::string *>, 26> fields = {{
            {"protocol_name", &metadata.protocol_name},
            {"proof_nonce", &metadata.proof_nonce},
            {"proof_task", &metadata.proof_task},
            {"srs_id", &metadata.srs_id},
            {"protocol_schema_id", &metadata.protocol_schema_id},
            {"instance_schema_hash", &metadata.instance_schema_hash},
            {"srs_type", &metadata.srs_type},
            {"srs_degree_bound", &metadata.srs_degree_bound},
            {"kzg_assumption_profile", &metadata.kzg_assumption_profile},
            {"privacy_policy", &metadata.privacy_policy},
            {"input_privacy", &metadata.input_privacy},
            {"output_mode", &metadata.output_mode},
            {"opening_mode", &metadata.opening_mode},
            {"binding_mode", &metadata.binding_mode},
            {"model_arch", &metadata.model_arch},
            {"model_params", &metadata.model_params},
            {"input_preprocess_id", &metadata.input_preprocess_id},
            {"static_tables", &metadata.static_tables},
            {"quant_config", &metadata.quant_config},
            {"domain_cfg", &metadata.domain_cfg},
            {"dim_cfg", &metadata.dim_cfg},
            {"encoding", &metadata.encoding},
            {"padding_rule", &metadata.padding_rule},
            {"degree_bound", &metadata.degree_bound},
            {"security_budget", &metadata.security_budget},
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
            base + "canonical_matrix_count",
            algebra::FieldElement(metadata.canonical_matrices.size()));
        for (std::size_t i = 0; i < metadata.canonical_matrices.size(); ++i)
        {
            const auto &matrix = metadata.canonical_matrices[i];
            const auto label = base + "canonical:" + std::to_string(i) + ":";
            transcript.absorb_text(label + "relation_id", matrix.relation_id);
            transcript.absorb_text(label + "matrix_object", matrix.matrix_object);
            transcript.absorb_text(
                label + "canonical_domain", matrix.canonical_domain);
            transcript.absorb_scalar(
                label + "row_count", algebra::FieldElement(matrix.row_count));
            transcript.absorb_scalar(
                label + "column_count",
                algebra::FieldElement(matrix.column_count));
            transcript.absorb_scalar(
                label + "column_object_count",
                algebra::FieldElement(matrix.column_objects.size()));
            for (std::size_t j = 0; j < matrix.column_objects.size(); ++j)
                transcript.absorb_text(
                    label + "column_object:" + std::to_string(j),
                    matrix.column_objects[j]);
            transcript.absorb_text(
                label + "compressed_object", matrix.compressed_object);
            transcript.absorb_text(
                label + "compression_challenge_label",
                matrix.compression_challenge_label);
        }
        transcript.absorb_scalar(
            base + "aggregation_relation_count",
            algebra::FieldElement(metadata.aggregation_relations.size()));
        for (std::size_t i = 0;
             i < metadata.aggregation_relations.size(); ++i)
        {
            const auto &relation = metadata.aggregation_relations[i];
            const auto label = base + "aggregation:" + std::to_string(i) + ":";
            for (const auto &[field, value] :
                 std::vector<std::pair<std::string, std::string>>{
                     {"relation_id", relation.relation_id},
                     {"source_matrix_object", relation.source_matrix_object},
                     {"target_matrix_object", relation.target_matrix_object},
                     {"alpha_object", relation.alpha_object},
                     {"node_domain", relation.node_domain},
                     {"edge_domain", relation.edge_domain},
                     {"compression_challenge_label",
                      relation.compression_challenge_label},
                     {"source_node_object", relation.source_node_object},
                     {"target_node_object", relation.target_node_object},
                     {"source_edge_object", relation.source_edge_object},
                     {"target_edge_object", relation.target_edge_object},
                     {"product_object", relation.product_object},
                     {"psq_object", relation.psq_object}})
                transcript.absorb_text(label + field, value);
            transcript.absorb_scalar(
                label + "row_count",
                algebra::FieldElement(relation.row_count));
            transcript.absorb_scalar(
                label + "column_count",
                algebra::FieldElement(relation.column_count));
            const auto absorb_route = [&](
                                          std::string_view role,
                                          const AggregationRouteRelationMeta &route)
            {
                const auto route_label = label + std::string(role) + ":";
                for (const auto &[field, value] :
                     std::vector<std::pair<std::string, std::string>>{
                         {"relation_id", route.relation_id},
                         {"index_object", route.index_object},
                         {"node_value_object", route.node_value_object},
                         {"edge_value_object", route.edge_value_object},
                         {"table_object", route.table_object},
                         {"query_object", route.query_object},
                         {"multiplicity_object", route.multiplicity_object},
                         {"node_inverse_object", route.node_inverse_object},
                         {"edge_inverse_object", route.edge_inverse_object},
                         {"node_accumulator_object",
                          route.node_accumulator_object},
                         {"edge_accumulator_object",
                          route.edge_accumulator_object},
                         {"shifted_accumulator_object",
                          route.shifted_accumulator_object},
                         {"relation_virtual_object",
                          route.relation_virtual_object},
                         {"eta_challenge_label", route.eta_challenge_label},
                         {"beta_challenge_label", route.beta_challenge_label},
                         {"terminal_challenge_label",
                          route.terminal_challenge_label}})
                    transcript.absorb_text(route_label + field, value);
                transcript.absorb_scalar(
                    route_label + "node_terminal_point",
                    route.node_terminal_point);
                transcript.absorb_scalar(
                    route_label + "edge_terminal_point",
                    route.edge_terminal_point);
                transcript.absorb_scalar(
                    route_label + "terminal_delta", route.terminal_delta);
                transcript.absorb_scalar(
                    route_label + "configured_shift_degree",
                    algebra::FieldElement(route.configured_shift_degree));
                transcript.absorb_scalar(
                    route_label + "shift_domain_size",
                    algebra::FieldElement(route.shift_domain_size));
            };
            absorb_route("source_route", relation.source_route);
            absorb_route("target_route", relation.target_route);
            absorb_text_vector(
                "aggregation:" + std::to_string(i) + ":commitment_order",
                relation.commitment_order);
        }
        transcript.absorb_scalar(
            base + "route_terminal_relation_count",
            algebra::FieldElement(metadata.route_terminal_relations.size()));
        for (std::size_t i = 0;
             i < metadata.route_terminal_relations.size(); ++i)
        {
            const auto &relation = metadata.route_terminal_relations[i];
            const auto label = base + "route_terminal:" + std::to_string(i) + ":";
            for (const auto &[field, value] :
                 std::vector<std::pair<std::string, std::string>>{
                     {"relation_id", relation.relation_id},
                     {"node_accumulator_object",
                      relation.node_accumulator_object},
                     {"edge_accumulator_object",
                      relation.edge_accumulator_object},
                     {"shifted_accumulator_object",
                      relation.shifted_accumulator_object},
                     {"relation_virtual_object",
                      relation.relation_virtual_object},
                     {"node_domain", relation.node_domain},
                     {"edge_domain", relation.edge_domain},
                     {"challenge_label", relation.challenge_label}})
                transcript.absorb_text(label + field, value);
            transcript.absorb_scalar(
                label + "node_terminal_point",
                relation.node_terminal_point);
            transcript.absorb_scalar(
                label + "edge_terminal_point",
                relation.edge_terminal_point);
            transcript.absorb_scalar(
                label + "terminal_delta", relation.terminal_delta);
            transcript.absorb_scalar(
                label + "configured_shift_degree",
                algebra::FieldElement(relation.configured_shift_degree));
            transcript.absorb_scalar(
                label + "shift_domain_size",
                algebra::FieldElement(relation.shift_domain_size));
        }
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
            base + "coefficient_basis_relation_count",
            algebra::FieldElement(metadata.coefficient_basis_relations.size()));
        for (std::size_t i = 0;
             i < metadata.coefficient_basis_relations.size(); ++i)
        {
            const auto &relation = metadata.coefficient_basis_relations[i];
            const auto label = base + "basis:" + std::to_string(i) + ":";
            for (const auto &[field, value] :
                 std::vector<std::pair<std::string, std::string>>{
                     {"relation_id", relation.relation_id},
                     {"canonical_object", relation.canonical_object},
                     {"coefficient_object", relation.coefficient_object},
                     {"carrier_domain", relation.carrier_domain},
                     {"power_object", relation.power_object},
                     {"gamma_label", relation.gamma_challenge_label},
                     {"relation_challenge_label",
                      relation.relation_challenge_label}})
                transcript.absorb_text(label + field, value);
            transcript.absorb_scalar(
                label + "low_degree_length",
                algebra::FieldElement(relation.low_degree_length));
            transcript.absorb_scalar(
                label + "tail_length",
                algebra::FieldElement(relation.tail_length));
            transcript.absorb_scalar(
                label + "total_length",
                algebra::FieldElement(relation.total_length));
            transcript.absorb_scalar(
                label + "semantic_position_count",
                algebra::FieldElement(relation.semantic_positions.size()));
            for (std::size_t j = 0; j < relation.semantic_positions.size(); ++j)
                transcript.absorb_scalar(
                    label + "semantic_position:" + std::to_string(j),
                    algebra::FieldElement(relation.semantic_positions[j]));
            transcript.absorb_scalar(
                label + "chunk_count",
                algebra::FieldElement(relation.chunks.size()));
            for (std::size_t j = 0; j < relation.chunks.size(); ++j)
            {
                const auto &chunk = relation.chunks[j];
                const auto chunk_label = label + "chunk:" + std::to_string(j) + ":";
                transcript.absorb_text(
                    chunk_label + "object_id", chunk.object_id);
                transcript.absorb_text(
                    chunk_label + "accumulator_object",
                    chunk.accumulator_object);
                transcript.absorb_text(
                    chunk_label + "shifted_accumulator_object",
                    chunk.shifted_accumulator_object);
                transcript.absorb_scalar(
                    chunk_label + "index",
                    algebra::FieldElement(chunk.chunk_index));
                transcript.absorb_scalar(
                    chunk_label + "offset",
                    algebra::FieldElement(chunk.offset));
                transcript.absorb_scalar(
                    chunk_label + "length",
                    algebra::FieldElement(chunk.length));
                transcript.absorb_scalar(
                    chunk_label + "terminal_point", chunk.terminal_point);
                transcript.absorb_scalar(
                    chunk_label + "configured_shift_degree",
                    algebra::FieldElement(chunk.configured_shift_degree));
                transcript.absorb_scalar(
                    chunk_label + "shift_domain_size",
                    algebra::FieldElement(chunk.shift_domain_size));
            }
            transcript.absorb_scalar(
                label + "commitment_count",
                algebra::FieldElement(relation.commitment_order.size()));
            for (std::size_t j = 0; j < relation.commitment_order.size(); ++j)
                transcript.absorb_text(
                    label + "commitment:" + std::to_string(j),
                    relation.commitment_order[j]);
            transcript.absorb_scalar(
                label + "opening_claim_count",
                algebra::FieldElement(relation.opening_claim_order.size()));
            for (std::size_t j = 0; j < relation.opening_claim_order.size(); ++j)
                transcript.absorb_text(
                    label + "opening_claim:" + std::to_string(j),
                    relation.opening_claim_order[j]);
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
        const auto wide_count = static_cast<unsigned __int128>(node_count) * static_cast<unsigned __int128>(class_count);
        if (node_count == 0 || class_count == 0 || wide_count == 0 || wide_count > std::numeric_limits<std::uint64_t>::max())
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
        for (const auto byte : digest)
            out << std::setw(2) << static_cast<unsigned>(byte);
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
        return "N=" + std::to_string(node_count) + "|C=" + std::to_string(class_count) + "|encoding=canonical-field-le32" + "|digest=" + canonical_output_digest(public_logits, node_count, class_count);
    }

    std::vector<std::string> dynamic_commitment_labels(const ProtocolContext &context)
    {
        // 这里返回的是 proof 中动态承诺的正式顺序。
        // family 形状越宽，标签数量越多，最终 proof_size_bytes 也会随之增加。
        std::vector<std::string> labels = {
            "P_H",
            "P_Table_feat",
            "P_Query_feat",
            "P_m_feat",
            "P_J_tbl_feat",
            "P_J_qry_feat",
            "P_R_feat",
        };
        for (std::size_t layer_index = 0;
             layer_index < context.model.hidden_layers.size();
             ++layer_index)
        {
            const auto &layer = context.model.hidden_layers[layer_index];
            for (std::size_t head_index = 0;
                 head_index < layer.heads.size();
                 ++head_index)
            {
                const auto prefix = hidden_head_prefix(layer_index, head_index);
                append_attention_head_dynamic_labels(labels, prefix);
                append_signed_division_labels(
                    labels, prefix + "_Div_", context.config.range_bits);
                labels.push_back(hidden_model_binding_blinder_label(
                    layer_index, head_index, "proj"));
                labels.push_back(hidden_model_binding_blinder_label(
                    layer_index, head_index, "src"));
                labels.push_back(hidden_model_binding_blinder_label(
                    layer_index, head_index, "dst"));
            }
            // The concat block is serialized immediately after all heads of
            // its layer and before the next layer's hidden-head family.
            append_concat_labels(
                labels,
                hidden_layer_concat_label(layer_index),
                hidden_layer_concat_star_label(layer_index),
                hidden_layer_cat_prefix(layer_index));
        }
        for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index)
        {
            append_output_head_dynamic_labels(
                labels,
                output_head_prefix(head_index),
                output_y_lin_label(head_index),
                output_y_label(head_index));
            append_signed_division_labels(
                labels,
                output_head_prefix(head_index) + "_Div_",
                context.config.range_bits);
            labels.push_back(output_model_binding_blinder_label(head_index, "proj"));
            labels.push_back(output_model_binding_blinder_label(head_index, "src"));
            labels.push_back(output_model_binding_blinder_label(head_index, "dst"));
        }
        labels.push_back("P_Y_lin");
        labels.push_back("P_Y");
        append_signed_division_labels(labels, "P_avg_Div_", context.config.range_bits);
        for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index)
        {
            const auto prefix = output_head_prefix(head_index) + "_Bind_";
            labels.push_back(prefix + "a");
            labels.push_back(prefix + "b");
            labels.push_back(prefix + "Acc");
        }
        labels.push_back("P_avg_Bind_a");
        labels.push_back("P_avg_Bind_b");
        labels.push_back("P_avg_Bind_Acc");

        for (const auto &spec :
             matrix_fold_binding_specs(context))
        {
            labels.push_back(
                spec.source_fold_object);
            labels.push_back(
                spec.source_accumulator_object +
                "_Weight");
            labels.push_back(
                spec.source_accumulator_object);
            labels.push_back(
                spec.fold_accumulator_object);
        }

        for (const auto &spec :
             flat_matrix_binding_specs(context))
        {
            labels.push_back(
                spec.row_fold_object);
            labels.push_back(
                spec.row_accumulator_object);
            labels.push_back(
                spec.flat_accumulator_object);
        }

        for (const auto &spec :
             broadcast_flat_binding_specs(context))
        {
            labels.push_back(
                spec.node_accumulator_object);
            labels.push_back(
                spec.flat_accumulator_object);
        }

        return labels;
    }

    std::vector<std::string> quotient_commitment_labels(const ProtocolContext &context)
    {
        const std::vector<std::string> quotient_labels = {
            "t_FH", "t_edge", "t_in", "t_d_h", "t_cat", "t_C", "t_N"};
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
                    "missing registered domain for opening challenge: " + spec.label);
            const auto point = transcript.challenge_excluding(
                spec.label,
                [&](const algebra::FieldElement &candidate)
                {
                    const auto rotated = candidate * spec.domain->omega;
                    if (spec.domain->zero_polynomial_eval(candidate).is_zero() || candidate == rotated)
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
        if (node_count == 0 || class_count == 0 || public_logits.size() != node_count * class_count)
        {
            throw std::runtime_error("public logits shape mismatch");
        }
        crypto::Transcript transcript("zkgat-public-output");
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

        if (proof_metadata != nullptr)
        {
            absorb_public_metadata(transcript, *proof_metadata);
        }
        else
        {
            absorb_public_metadata(transcript, canonical_public_metadata(context));
        }
        transcript.absorb_scalar("N", algebra::FieldElement(context.local.num_nodes));
        transcript.absorb_scalar("E", algebra::FieldElement(context.local.edges.size()));
        for (std::size_t layer_index = 0;
             layer_index < context.model.hidden_layers.size();
             ++layer_index)
        {
            const auto &layer = context.model.hidden_layers[layer_index];
            transcript.absorb_scalar(
                "d_in_l" + std::to_string(layer_index),
                algebra::FieldElement(layer.input_dim));
            transcript.absorb_scalar(
                "d_h_l" + std::to_string(layer_index),
                algebra::FieldElement(layer.shape.head_dim));
            transcript.absorb_scalar(
                "d_cat_l" + std::to_string(layer_index),
                algebra::FieldElement(
                    layer.shape.head_count * layer.shape.head_dim));
        }
        transcript.absorb_scalar(
            "d_in_out",
            algebra::FieldElement(context.model.output_layer.input_dim));
        transcript.absorb_scalar(
            "d_out",
            algebra::FieldElement(context.local.num_classes));
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
        absorb_public(transcript, context, "P_AggBind_m_src");
        absorb_public(transcript, context, "P_AggBind_m_dst");
        absorb_public(transcript, context, "P_AggBind_index_N");
        for (std::size_t layer_index = 0;
             layer_index < context.model.hidden_layers.size();
             ++layer_index)
        {
            absorb_public(
                transcript, context,
                hidden_q_in_selector_label(layer_index));
            absorb_public(
                transcript, context,
                hidden_q_d_h_selector_label(layer_index));
            absorb_public(
                transcript, context,
                hidden_q_cat_selector_label(layer_index));
            absorb_public(
                transcript, context,
                hidden_q_nd_h_selector_label(layer_index));
        }
        absorb_public(
            transcript, context, output_q_cat_selector_label());
        absorb_public(transcript, context, "P_Q_C_valid");
        absorb_public(transcript, context, "P_BiasIndex_C");
        absorb_public(transcript, context, "P_BiasMultiplicity_C");
        absorb_public(transcript, context, "P_BiasIndex_edge");
        constexpr std::size_t kCoefficientTailBaseObservationCount = 3;
        const auto bias_basis_chunks = coefficient_chunk_layout(
            context.local.num_classes + kCoefficientTailBaseObservationCount,
            context.domains.edge->size);
        for (std::size_t head_index = 0;
             head_index < context.model.output_layer.heads.size(); ++head_index)
            for (const auto &chunk : bias_basis_chunks)
                for (const auto *region : {
                         "active", "semantic", "gap", "tail"})
                    absorb_public(
                        transcript, context,
                        output_bias_basis_region_selector_label(
                            head_index, chunk.chunk_index, region));
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

        for (std::size_t layer_index = 0; layer_index < context.model.hidden_layers.size(); ++layer_index)
        {
            const bool is_first_layer = layer_index == 0;
            const auto input_label =
                is_first_layer
                    ? std::string("P_H")
                    : hidden_layer_concat_label(layer_index - 1);
            for (std::size_t head_index = 0;
                 head_index < context.model.hidden_layers[layer_index].heads.size();
                 ++head_index)
            {
                const auto prefix = hidden_head_prefix(layer_index, head_index);
                const auto challenge = [&](std::string_view base)
                {
                    return hidden_challenge_name(base, layer_index, head_index);
                };
                absorb(transcript, input_label, dynamic_commitments);
                absorb(transcript, prefix + "_H_prime", dynamic_commitments);

                for (std::size_t column = 0;
                     column <
                     context.model.hidden_layers[layer_index]
                         .shape.head_dim;
                     ++column)
                {
                    absorb(
                        transcript,
                        canonical_matrix_column_label(
                            prefix + "_H_prime",
                            column),
                        dynamic_commitments);
                }

                for (std::size_t column = 0;
                     column < context.model.hidden_layers[layer_index].shape.head_dim;
                     ++column)
                {
                    absorb_model_reblind(
                        transcript,
                        hidden_weight_column_label(layer_index, head_index, column),
                        dynamic_commitments);
                }
                out[challenge("y_proj_h")] = transcript.challenge(challenge("y_proj_h"));
                out[challenge("xi_h")] = transcript.challenge(challenge("xi_h"));

                absorb(transcript, prefix + "_H_prime", dynamic_commitments);
                absorb(transcript, prefix + "_E_src", dynamic_commitments);
                absorb(
                    transcript,
                    hidden_score_mask_label(layer_index, head_index, "src"),
                    dynamic_commitments);
                absorb_model_reblind(
                    transcript, hidden_src_label(layer_index, head_index), dynamic_commitments);
                out[challenge("y_src_h")] = transcript.challenge(challenge("y_src_h"));

                absorb(transcript, prefix + "_H_prime", dynamic_commitments);
                absorb(transcript, prefix + "_E_dst", dynamic_commitments);
                absorb(
                    transcript,
                    hidden_score_mask_label(layer_index, head_index, "dst"),
                    dynamic_commitments);
                absorb_model_reblind(
                    transcript, hidden_dst_label(layer_index, head_index), dynamic_commitments);
                out[challenge("y_dst_h")] = transcript.challenge(challenge("y_dst_h"));

                absorb(transcript, prefix + "_H_star", dynamic_commitments);
                absorb(
                    transcript,
                    external_mask_label(prefix + "_H_star"),
                    dynamic_commitments);

                out[challenge("rho_proj_h")] =
                    transcript.challenge(
                        challenge("rho_proj_h"));

                out[challenge("y_star_h")] =
                    transcript.challenge(
                        challenge("y_star_h"));

                absorb(transcript, prefix + "_E_src", dynamic_commitments);
                absorb(transcript, prefix + "_H_star", dynamic_commitments);
                absorb(transcript, prefix + "_H_src_star_edge", dynamic_commitments);
                out[challenge("eta_src_h")] = transcript.challenge(challenge("eta_src_h"));
                absorb(transcript, prefix + "_Table_src", dynamic_commitments);
                absorb(transcript, prefix + "_Query_src", dynamic_commitments);
                absorb(transcript, prefix + "_m_src", dynamic_commitments);
                out[challenge("beta_src_h")] = transcript.challenge(challenge("beta_src_h"));
                absorb(transcript, prefix + "_J_src_node", dynamic_commitments);
                absorb(transcript, prefix + "_J_src_edge", dynamic_commitments);
                absorb(transcript, prefix + "_R_src_node", dynamic_commitments);
                absorb(transcript, prefix + "_R_src", dynamic_commitments);

                absorb(transcript, prefix + "_S", dynamic_commitments);
                absorb(transcript, prefix + "_Z", dynamic_commitments);
                absorb_static(transcript, context, "V_T_L_x");
                absorb_static(transcript, context, "V_T_L_y");
                out[challenge("eta_L_h")] = transcript.challenge(challenge("eta_L_h"));
                absorb(transcript, prefix + "_Table_L", dynamic_commitments);
                absorb(transcript, prefix + "_Query_L", dynamic_commitments);
                absorb(transcript, prefix + "_m_L", dynamic_commitments);
                out[challenge("beta_L_h")] = transcript.challenge(challenge("beta_L_h"));
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
                out[challenge("beta_R_h")] = transcript.challenge(challenge("beta_R_h"));
                absorb(transcript, prefix + "_J_tbl_R", dynamic_commitments);
                absorb(transcript, prefix + "_J_qry_R", dynamic_commitments);
                absorb(transcript, prefix + "_R_R", dynamic_commitments);

                absorb(transcript, prefix + "_Delta", dynamic_commitments);
                absorb(transcript, prefix + "_U", dynamic_commitments);
                absorb_static(transcript, context, "V_T_exp_x");
                absorb_static(transcript, context, "V_T_exp_y");
                out[challenge("eta_exp_h")] = transcript.challenge(challenge("eta_exp_h"));
                absorb(transcript, prefix + "_Table_exp", dynamic_commitments);
                absorb(transcript, prefix + "_Query_exp", dynamic_commitments);
                absorb(transcript, prefix + "_m_exp", dynamic_commitments);
                out[challenge("beta_exp_h")] = transcript.challenge(challenge("beta_exp_h"));
                absorb(transcript, prefix + "_J_tbl_exp", dynamic_commitments);
                absorb(transcript, prefix + "_J_qry_exp", dynamic_commitments);
                absorb(transcript, prefix + "_R_exp", dynamic_commitments);

                out[challenge("lambda_psq_h")] = transcript.challenge(challenge("lambda_psq_h"));
                absorb(transcript, prefix + "_T_psq", dynamic_commitments);
                absorb(transcript, prefix + "_T_psq_edge", dynamic_commitments);
                out[challenge("eta_t_h")] = transcript.challenge(challenge("eta_t_h"));
                absorb(transcript, prefix + "_Table_t", dynamic_commitments);
                absorb(transcript, prefix + "_Query_t", dynamic_commitments);
                absorb(transcript, prefix + "_m_t", dynamic_commitments);
                out[challenge("beta_t_h")] = transcript.challenge(challenge("beta_t_h"));
                absorb(transcript, prefix + "_J_t_node", dynamic_commitments);
                absorb(transcript, prefix + "_J_t_edge", dynamic_commitments);
                absorb(transcript, prefix + "_R_t_node", dynamic_commitments);
                absorb(transcript, prefix + "_R_t", dynamic_commitments);

                absorb(transcript, prefix + "_H_agg_pre", dynamic_commitments);
                absorb(transcript, prefix + "_H_agg_pre_star", dynamic_commitments);
                absorb(
                    transcript,
                    external_mask_label(prefix + "_H_agg_pre_star"),
                    dynamic_commitments);
                out[challenge("y_agg_pre_h")] = transcript.challenge(challenge("y_agg_pre_h"));
                absorb(transcript, prefix + "_H_agg_pre", dynamic_commitments);
                absorb(transcript, prefix + "_H_agg", dynamic_commitments);
                absorb_static(transcript, context, "V_T_ELU_x");
                absorb_static(transcript, context, "V_T_ELU_y");
                absorb(transcript, prefix + "_H_ELU_in_flat", dynamic_commitments);
                absorb(transcript, prefix + "_H_agg_flat", dynamic_commitments);
                out[challenge("eta_ELU_h")] = transcript.challenge(challenge("eta_ELU_h"));
                absorb(transcript, prefix + "_Table_ELU", dynamic_commitments);
                absorb(transcript, prefix + "_Query_ELU", dynamic_commitments);
                absorb(transcript, prefix + "_m_ELU", dynamic_commitments);
                out[challenge("beta_ELU_h")] = transcript.challenge(challenge("beta_ELU_h"));
                absorb(transcript, prefix + "_J_tbl_ELU", dynamic_commitments);
                absorb(transcript, prefix + "_J_qry_ELU", dynamic_commitments);
                absorb(transcript, prefix + "_R_ELU", dynamic_commitments);

                absorb(transcript, prefix + "_H_agg", dynamic_commitments);
                absorb(transcript, prefix + "_H_agg_star", dynamic_commitments);
                absorb(
                    transcript,
                    external_mask_label(prefix + "_H_agg_star"),
                    dynamic_commitments);
                out[challenge("y_agg_h")] = transcript.challenge(challenge("y_agg_h"));

                for (const auto *suffix : {
                         "_E_dst", "_M", "_Sum", "_inv", "_H_agg_star",
                         "_E_dst_edge", "_M_edge", "_Sum_edge", "_inv_edge", "_H_agg_star_edge"})
                {
                    absorb(transcript, prefix + suffix, dynamic_commitments);
                }
                out[challenge("eta_dst_h")] = transcript.challenge(challenge("eta_dst_h"));
                absorb(transcript, prefix + "_Table_dst", dynamic_commitments);
                absorb(transcript, prefix + "_Query_dst", dynamic_commitments);
                absorb(transcript, prefix + "_m_dst", dynamic_commitments);
                out[challenge("beta_dst_h")] = transcript.challenge(challenge("beta_dst_h"));
                absorb(transcript, prefix + "_J_dst_node", dynamic_commitments);
                absorb(transcript, prefix + "_J_dst_edge", dynamic_commitments);
                absorb(transcript, prefix + "_R_dst_node", dynamic_commitments);
                absorb(transcript, prefix + "_R_dst", dynamic_commitments);
                replay_signed_division(prefix + "_Div_");
            }

            absorb(transcript, hidden_layer_concat_label(layer_index), dynamic_commitments);
            out[hidden_concat_xi_name(layer_index)] =
                transcript.challenge(hidden_concat_xi_name(layer_index));
            absorb(transcript, hidden_layer_concat_star_label(layer_index), dynamic_commitments);
            absorb(
                transcript,
                external_mask_label(hidden_layer_concat_star_label(layer_index)),
                dynamic_commitments);
            out[hidden_concat_y_name(layer_index)] =
                transcript.challenge(hidden_concat_y_name(layer_index));
        }

        const auto final_hidden_label =
            hidden_layer_concat_label(context.model.hidden_layers.size() - 1);
        for (std::size_t head_index = 0;
             head_index < context.model.output_layer.heads.size();
             ++head_index)
        {
            absorb(
                transcript,
                output_head_prefix(head_index) + "_Y_prime",
                dynamic_commitments);

            for (std::size_t column = 0;
                 column < context.local.num_classes;
                 ++column)
            {
                absorb(
                    transcript,
                    canonical_matrix_column_label(
                        output_head_prefix(head_index) +
                            "_Y_prime",
                        column),
                    dynamic_commitments);
            }
        }

        out["xi_out"] =
            transcript.challenge("xi_out");
        for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index)
        {
            const auto prefix = output_head_prefix(head_index);
            const auto y_label = output_y_label(head_index);
            absorb(transcript, final_hidden_label, dynamic_commitments);
            absorb(transcript, prefix + "_Y_prime", dynamic_commitments);
            for (std::size_t column = 0; column < context.local.num_classes; ++column)
            {
                absorb_model_reblind(
                    transcript,
                    output_weight_column_label(head_index, column),
                    dynamic_commitments);
            }
            out[output_challenge_name("y_proj_out", head_index)] =
                transcript.challenge(output_challenge_name("y_proj_out", head_index));
            absorb(transcript, prefix + "_Y_prime", dynamic_commitments);
            absorb(transcript, prefix + "_E_src", dynamic_commitments);
            absorb(
                transcript,
                output_score_mask_label(head_index, "src"),
                dynamic_commitments);
            absorb_model_reblind(
                transcript,
                output_src_label(head_index),
                dynamic_commitments);
            out[output_challenge_name("y_src_out", head_index)] =
                transcript.challenge(output_challenge_name("y_src_out", head_index));
            absorb(transcript, prefix + "_Y_prime", dynamic_commitments);
            absorb(transcript, prefix + "_E_dst", dynamic_commitments);
            absorb(
                transcript,
                output_score_mask_label(head_index, "dst"),
                dynamic_commitments);
            absorb_model_reblind(
                transcript,
                output_dst_label(head_index),
                dynamic_commitments);
            out[output_challenge_name("y_dst_out", head_index)] =
                transcript.challenge(output_challenge_name("y_dst_out", head_index));
            absorb(transcript, prefix + "_Y_prime_star", dynamic_commitments);
            absorb(
                transcript,
                external_mask_label(
                    prefix + "_Y_prime_star"),
                dynamic_commitments);

            out[output_challenge_name(
                "rho_proj_out",
                head_index)] =
                transcript.challenge(
                    output_challenge_name(
                        "rho_proj_out",
                        head_index));
            absorb(transcript, prefix + "_Y_prime_star_edge", dynamic_commitments);
            out[output_challenge_name("eta_src_out", head_index)] =
                transcript.challenge(output_challenge_name("eta_src_out", head_index));
            absorb(transcript, prefix + "_Table_src", dynamic_commitments);
            absorb(transcript, prefix + "_Query_src", dynamic_commitments);
            absorb(transcript, prefix + "_m_src", dynamic_commitments);
            out[output_challenge_name("beta_src_out", head_index)] =
                transcript.challenge(output_challenge_name("beta_src_out", head_index));
            absorb(transcript, prefix + "_J_src_node", dynamic_commitments);
            absorb(transcript, prefix + "_J_src_edge", dynamic_commitments);
            absorb(transcript, prefix + "_R_src_node", dynamic_commitments);
            absorb(transcript, prefix + "_R_src", dynamic_commitments);
            absorb(transcript, prefix + "_S", dynamic_commitments);
            absorb(transcript, prefix + "_Z", dynamic_commitments);
            absorb_static(transcript, context, "V_T_L_x");
            absorb_static(transcript, context, "V_T_L_y");
            out[output_challenge_name("eta_L_out", head_index)] =
                transcript.challenge(output_challenge_name("eta_L_out", head_index));
            for (const auto *suffix : {"_Table_L", "_Query_L", "_m_L"})
            {
                absorb(transcript, prefix + suffix, dynamic_commitments);
            }
            out[output_challenge_name("beta_L_out", head_index)] =
                transcript.challenge(output_challenge_name("beta_L_out", head_index));
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
            out[output_challenge_name("beta_R_out", head_index)] =
                transcript.challenge(output_challenge_name("beta_R_out", head_index));
            for (const auto *suffix : {"_J_tbl_R", "_J_qry_R", "_R_R"})
            {
                absorb(transcript, prefix + suffix, dynamic_commitments);
            }
            absorb(transcript, prefix + "_Delta", dynamic_commitments);
            absorb(transcript, prefix + "_U", dynamic_commitments);
            absorb_static(transcript, context, "V_T_exp_x");
            absorb_static(transcript, context, "V_T_exp_y");
            out[output_challenge_name("eta_exp_out", head_index)] =
                transcript.challenge(output_challenge_name("eta_exp_out", head_index));
            for (const auto *suffix : {"_Table_exp", "_Query_exp", "_m_exp"})
            {
                absorb(transcript, prefix + suffix, dynamic_commitments);
            }
            out[output_challenge_name("beta_exp_out", head_index)] =
                transcript.challenge(output_challenge_name("beta_exp_out", head_index));
            for (const auto *suffix : {"_J_tbl_exp", "_J_qry_exp", "_R_exp"})
            {
                absorb(transcript, prefix + suffix, dynamic_commitments);
            }
            out[output_challenge_name("lambda_out", head_index)] =
                transcript.challenge(output_challenge_name("lambda_out", head_index));
            absorb(transcript, prefix + "_T", dynamic_commitments);
            absorb(transcript, prefix + "_T_edge", dynamic_commitments);
            out[output_challenge_name("eta_t_out", head_index)] =
                transcript.challenge(output_challenge_name("eta_t_out", head_index));
            absorb(transcript, prefix + "_Table_t", dynamic_commitments);
            absorb(transcript, prefix + "_Query_t", dynamic_commitments);
            absorb(transcript, prefix + "_m_t", dynamic_commitments);
            out[output_challenge_name("beta_t_out", head_index)] =
                transcript.challenge(output_challenge_name("beta_t_out", head_index));
            absorb(transcript, prefix + "_J_t_node", dynamic_commitments);
            absorb(transcript, prefix + "_J_t_edge", dynamic_commitments);
            absorb(transcript, prefix + "_R_t_node", dynamic_commitments);
            absorb(transcript, prefix + "_R_t", dynamic_commitments);
            absorb(transcript, y_label, dynamic_commitments);
            absorb(transcript, prefix + "_Y_star", dynamic_commitments);
            absorb(
                transcript,
                external_mask_label(prefix + "_Y_star"),
                dynamic_commitments);
            out[output_challenge_name("y_out_star", head_index)] =
                transcript.challenge(output_challenge_name("y_out_star", head_index));
            for (const auto *suffix : {
                     "_E_dst", "_M", "_Sum", "_inv", "_Y_star",
                     "_E_dst_edge", "_M_edge", "_Sum_edge", "_inv_edge", "_Y_star_edge"})
            {
                absorb(transcript, prefix + suffix, dynamic_commitments);
            }
            out[output_challenge_name("eta_dst_out", head_index)] =
                transcript.challenge(output_challenge_name("eta_dst_out", head_index));
            absorb(transcript, prefix + "_Table_dst", dynamic_commitments);
            absorb(transcript, prefix + "_Query_dst", dynamic_commitments);
            absorb(transcript, prefix + "_m_dst", dynamic_commitments);
            out[output_challenge_name("beta_dst_out", head_index)] =
                transcript.challenge(output_challenge_name("beta_dst_out", head_index));
            absorb(transcript, prefix + "_J_dst_node", dynamic_commitments);
            absorb(transcript, prefix + "_J_dst_edge", dynamic_commitments);
            absorb(transcript, prefix + "_R_dst_node", dynamic_commitments);
            absorb(transcript, prefix + "_R_dst", dynamic_commitments);
            replay_signed_division(prefix + "_Div_");
        }

        replay_signed_division("P_avg_Div_");

        for (const auto &spec :
             matrix_fold_binding_specs(context))
        {
            absorb(
                transcript,
                spec.fold_object,
                dynamic_commitments);

            if (spec.public_source)
            {
                absorb(
                    transcript,
                    "P_H",
                    dynamic_commitments);
            }
            else
            {
                for (std::size_t column = 0;
                     column < spec.width;
                     ++column)
                {
                    absorb(
                        transcript,
                        canonical_matrix_column_label(
                            spec.source_matrix_object,
                            column),
                        dynamic_commitments);
                }
            }

            out[spec.lambda_challenge] =
                transcript.challenge(
                    spec.lambda_challenge);

            absorb(
                transcript,
                spec.source_fold_object,
                dynamic_commitments);

            absorb(
                transcript,
                spec.source_accumulator_object +
                    "_Weight",
                dynamic_commitments);

            absorb(
                transcript,
                spec.source_accumulator_object,
                dynamic_commitments);

            absorb(
                transcript,
                spec.fold_accumulator_object,
                dynamic_commitments);
        }

        for (const auto &spec :
             flat_matrix_binding_specs(context))
        {
            absorb(
                transcript,
                spec.flat_object,
                dynamic_commitments);

            for (std::size_t column = 0;
                 column < spec.columns;
                 ++column)
            {
                absorb(
                    transcript,
                    canonical_matrix_column_label(
                        spec.matrix_object,
                        column),
                    dynamic_commitments);
            }

            out[spec.lambda_challenge] =
                transcript.challenge(
                    spec.lambda_challenge);

            absorb(
                transcript,
                spec.row_fold_object,
                dynamic_commitments);

            absorb(
                transcript,
                spec.row_accumulator_object,
                dynamic_commitments);

            absorb(
                transcript,
                spec.flat_accumulator_object,
                dynamic_commitments);
        }

        for (const auto &spec :
             broadcast_flat_binding_specs(context))
        {
            absorb(
                transcript,
                spec.node_object,
                dynamic_commitments);

            absorb(
                transcript,
                spec.flat_object,
                dynamic_commitments);

            out[spec.lambda_challenge] =
                transcript.challenge(
                    spec.lambda_challenge);

            absorb(
                transcript,
                spec.node_accumulator_object,
                dynamic_commitments);

            absorb(
                transcript,
                spec.flat_accumulator_object,
                dynamic_commitments);
        }
        const auto binding_metadata =
            build_binding_relation_metadata(context);
        for (const auto &meta : binding_metadata)
        {
            absorb_binding_relation_metadata(transcript, meta);
            for (const auto &relation :
                 meta.route_terminal_relations)
            {
                absorb(
                    transcript,
                    relation.shifted_accumulator_object,
                    dynamic_commitments);

                std::shared_ptr<
                    algebra::RootOfUnityDomain>
                    relation_edge_domain;

                for (const auto &domain : {
                         context.domains.fh,
                         context.domains.edge,
                         context.domains.in,
                         context.domains.d,
                         context.domains.cat,
                         context.domains.c,
                         context.domains.n})
                {
                    if (domain != nullptr &&
                        domain->name ==
                            relation.edge_domain)
                    {
                        relation_edge_domain =
                            domain;
                        break;
                    }
                }

                if (relation_edge_domain == nullptr)
                {
                    throw std::runtime_error(
                        "route terminal replay domain missing: " +
                        relation.relation_id);
                }

                out[relation.challenge_label] =
                    transcript.challenge_excluding(
                        relation.challenge_label,
                        [&](const algebra::FieldElement &candidate)
                        {
                            return relation_edge_domain
                                ->zero_polynomial_eval(
                                    candidate +
                                    relation.terminal_delta)
                                .is_zero();
                        });
            }
            for (const auto &matrix : meta.canonical_matrices)
            {
                absorb(
                    transcript, matrix.matrix_object,
                    dynamic_commitments);
                for (const auto &column : matrix.column_objects)
                    absorb(transcript, column, dynamic_commitments);
                out[matrix.compression_challenge_label] =
                    transcript.challenge(
                        matrix.compression_challenge_label);
                absorb(
                    transcript, matrix.compressed_object,
                    dynamic_commitments);
            }
            for (const auto &relation : meta.aggregation_relations)
            {
                absorb(
                    transcript, relation.alpha_object,
                    dynamic_commitments);
                out[relation.compression_challenge_label] =
                    transcript.challenge(
                        relation.compression_challenge_label);
                for (const auto *object : {
                         &relation.source_node_object,
                         &relation.target_node_object,
                         &relation.source_edge_object,
                         &relation.target_edge_object,
                         &relation.product_object,
                         &relation.psq_object})
                    absorb(transcript, *object, dynamic_commitments);
                for (const auto *route : {
                         &relation.source_route,
                         &relation.target_route})
                {
                    absorb(
                        transcript, route->node_value_object,
                        dynamic_commitments);
                    absorb(
                        transcript, route->edge_value_object,
                        dynamic_commitments);
                    out[route->eta_challenge_label] =
                        transcript.challenge(
                            route->eta_challenge_label);
                    absorb(
                        transcript, route->table_object,
                        dynamic_commitments);
                    absorb(
                        transcript, route->query_object,
                        dynamic_commitments);
                    absorb_public(
                        transcript, context,
                        route->multiplicity_object);
                    out[route->beta_challenge_label] =
                        transcript.challenge(
                            route->beta_challenge_label);
                    for (const auto *object : {
                             &route->node_inverse_object,
                             &route->edge_inverse_object,
                             &route->node_accumulator_object,
                             &route->edge_accumulator_object,
                             &route->shifted_accumulator_object})
                        absorb(transcript, *object, dynamic_commitments);
                    out[route->terminal_challenge_label] =
                        transcript.challenge_excluding(
                            route->terminal_challenge_label,
                            [&](const algebra::FieldElement &candidate)
                            {
                                return context.domains.edge
                                    ->zero_polynomial_eval(
                                        candidate + route->terminal_delta)
                                    .is_zero();
                            });
                }
            }
        }

        for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index)
        {
            absorb(transcript, output_y_lin_label(head_index), dynamic_commitments);
            absorb(transcript, output_y_label(head_index), dynamic_commitments);
            absorb_model_reblind(
                transcript,
                output_bias_label(head_index),
                dynamic_commitments);
            absorb(transcript, output_head_prefix(head_index) + "_Y_star", dynamic_commitments);
            absorb(transcript, output_head_prefix(head_index) + "_Table_dst", dynamic_commitments);
            absorb(transcript, output_head_prefix(head_index) + "_Query_dst", dynamic_commitments);
            absorb(transcript, output_head_prefix(head_index) + "_Y_pre_flat", dynamic_commitments);
            absorb(transcript, output_head_prefix(head_index) + "_Bias_flat", dynamic_commitments);
            absorb(transcript, output_head_prefix(head_index) + "_Y_flat", dynamic_commitments);
            for (const auto *suffix : {
                     "Active", "X", "D", "Y", "Sigma", "Zero",
                     "A", "A_inv", "D_inv", "Q", "R", "Delta"})
            {
                absorb(
                    transcript,
                    output_head_prefix(head_index) + "_Div_" + suffix,
                    dynamic_commitments);
            }
            const auto relation_id = "bind:output:" + std::to_string(head_index);
            const auto meta_it = std::find_if(
                binding_metadata.begin(), binding_metadata.end(),
                [&](const BindingRelationMeta &meta)
                { return meta.relation_id == relation_id; });
            if (meta_it == binding_metadata.end() || meta_it->indexed_views.size() != 1)
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
                                                candidate + shift_meta.deltas.at(1))
                            .is_zero();
                    });
            }
            if (meta_it->coefficient_basis_relations.size() != 1)
                throw std::runtime_error(
                    "output bias coefficient basis metadata is not unique during replay");
            const auto &basis =
                meta_it->coefficient_basis_relations.front();
            const auto pre_gamma_commitment_count =
                basis.chunks.size() + 1;
            if (basis.commitment_order.size() != pre_gamma_commitment_count + basis.chunks.size())
                throw std::runtime_error(
                    "output bias coefficient basis commitment order is invalid");
            for (std::size_t index = 0;
                 index < basis.chunks.size(); ++index)
                if (basis.commitment_order[pre_gamma_commitment_count + index] != basis.chunks[index].accumulator_object)
                    throw std::runtime_error(
                        "output bias coefficient basis accumulator order is invalid");
            for (std::size_t index = 0;
                 index < pre_gamma_commitment_count; ++index)
                absorb(
                    transcript, basis.commitment_order[index],
                    dynamic_commitments);
            out[basis.gamma_challenge_label] =
                transcript.challenge_excluding(
                    basis.gamma_challenge_label,
                    [&](const algebra::FieldElement &candidate)
                    {
                        return candidate.is_zero() || std::any_of(
                                                          basis.chunks.begin(), basis.chunks.end(),
                                                          [&](const CoefficientBasisChunkMeta &chunk)
                                                          { return candidate == chunk.terminal_point; });
                    });
            for (const auto &chunk : basis.chunks)
                absorb(
                    transcript, chunk.accumulator_object,
                    dynamic_commitments);
            for (const auto &chunk : basis.chunks)
                absorb(
                    transcript, chunk.shifted_accumulator_object,
                    dynamic_commitments);
            const auto gamma = out.at(basis.gamma_challenge_label);
            out[basis.relation_challenge_label] =
                transcript.challenge_excluding(
                    basis.relation_challenge_label,
                    [&](const algebra::FieldElement &candidate)
                    {
                        return std::any_of(
                            basis.chunks.begin(), basis.chunks.end(),
                            [&](const CoefficientBasisChunkMeta &chunk)
                            {
                                return context.domains.edge
                                    ->zero_polynomial_eval(
                                        candidate + chunk.terminal_point - gamma)
                                    .is_zero();
                            });
                    });
        }
        absorb(transcript, "P_Y_lin", dynamic_commitments);
        absorb(transcript, "P_Y", dynamic_commitments);
        for (const auto *suffix : {
                 "Active", "X", "D", "Y", "Sigma", "Zero",
                 "A", "A_inv", "D_inv", "Q", "R", "Delta"})
        {
            absorb(transcript, "P_avg_Div_" + std::string(suffix), dynamic_commitments);
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
}
