#pragma once

// full verifier 汇总层只检查 public metadata 与 proof object 中的子证明摘要。
// 它不读取 witness、不读取完整 tensor，只确认 all-layer/all-head 必需 relation 已生成并验证。

std::string zkdl_normalized_stage_name(const std::string& stage)
{
    const std::string cpu_prefix = "cpu_";
    if (stage.rfind(cpu_prefix, 0) == 0) {
        return stage.substr(cpu_prefix.size());
    }
    return stage;
}

bool zkdl_digest_has_kv(const ZkdlFullRelationRecord& record, const std::string& key, const std::string& value)
{
    const std::string needle = key + "=" + value;
    return std::find(record.proof_digests.begin(), record.proof_digests.end(), needle) != record.proof_digests.end();
}

bool zkdl_digest_has_zero_counter(const ZkdlFullRelationRecord& record, const std::string& key)
{
    return zkdl_digest_has_kv(record, key, "0");
}

const ZkdlFullRelationRecord* zkdl_find_full_relation_record(
    const ZkdlFullProofObject& proof,
    const std::string& normalized_stage,
    const std::string& relation_scope)
{
    for (const auto& record : proof.records) {
        if (zkdl_normalized_stage_name(record.stage) == normalized_stage
            && record.relation_scope == relation_scope
            && record.proof_generated
            && record.verified) {
            return &record;
        }
    }
    return nullptr;
}

void zkdl_require_full_relation_record(
    const ZkdlFullProofObject& proof,
    const std::string& normalized_stage,
    const std::string& relation_scope,
    std::vector<std::string>& missing)
{
    const auto* record = zkdl_find_full_relation_record(proof, normalized_stage, relation_scope);
    if (record == nullptr) {
        missing.push_back("缺少或未验证 relation: " + normalized_stage + " scope=" + relation_scope);
        return;
    }
    if (record->proof_size_bytes == 0) {
        missing.push_back("relation proof_size_bytes 为 0: " + normalized_stage + " scope=" + relation_scope);
    }
    if (record->proof_digests.empty()) {
        missing.push_back("relation proof_digests 为空: " + normalized_stage + " scope=" + relation_scope);
    }
}

void zkdl_require_digest_binding(
    const ZkdlFullProofObject& proof,
    const std::string& normalized_stage,
    const std::string& relation_scope,
    const std::string& key,
    const std::string& value,
    std::vector<std::string>& missing)
{
    const auto* record = zkdl_find_full_relation_record(proof, normalized_stage, relation_scope);
    if (record == nullptr) return;
    if (!zkdl_digest_has_kv(*record, key, value)) {
        missing.push_back("public metadata binding 缺失: " + normalized_stage
            + " scope=" + relation_scope + " key=" + key);
    }
}

void zkdl_require_zero_counter(
    const ZkdlFullProofObject& proof,
    const std::string& normalized_stage,
    const std::string& relation_scope,
    const std::string& counter_key,
    std::vector<std::string>& missing)
{
    const auto* record = zkdl_find_full_relation_record(proof, normalized_stage, relation_scope);
    if (record == nullptr) return;
    if (!zkdl_digest_has_zero_counter(*record, counter_key)) {
        missing.push_back("relation mismatch counter 非 0 或缺失: " + normalized_stage
            + " scope=" + relation_scope + " counter=" + counter_key);
    }
}

std::vector<std::string> zkdl_verify_full_proof_manifest_clean(
    const ZkdlFullProofObject& proof,
    const DatasetRunSpec& spec,
    const ModelParams& model,
    const ForwardTraceHost& trace,
    const ZkdlGatPublicMetadata& metadata,
    const ZkdlGatQuantContract& quant,
    const ZkdlGatTableContext& table,
    const std::string& device)
{
    std::vector<std::string> missing;
    if (proof.protocol_id != "GAT-ZKML-zkDL") {
        missing.push_back("protocol_id 不匹配");
    }
    if (proof.proof_scope != "zkdl_gat_full" || metadata.proof_scope_id != "zkdl_gat_full") {
        missing.push_back("proof_scope 不是 zkdl_gat_full");
    }
    if (proof.dataset != spec.dataset || metadata.dataset != spec.dataset) {
        missing.push_back("dataset metadata binding 不匹配");
    }
    if (proof.device != device) {
        missing.push_back("device metadata binding 不匹配");
    }
    if (!proof.is_full_dataset || metadata.subgraph_rule != "whole_graph") {
        missing.push_back("full dataset / whole graph metadata binding 缺失");
    }
    if (proof.metadata_id != metadata.metadata_id) {
        missing.push_back("metadata_id 未绑定到 proof object");
    }
    if (proof.vk_static_id != table.context_id) {
        missing.push_back("VK_static/table context 未绑定到 proof object");
    }
    if (proof.vk_model_id != model.model_param_id || metadata.model_param_id != model.model_param_id) {
        missing.push_back("VK_model/model_param_id 未绑定到 proof object");
    }
    if (proof.quant_contract_id != quant.contract_id || metadata.quant_cfg_id != model.quant_cfg_id) {
        missing.push_back("quant contract 未绑定到 proof object");
    }
    if (proof.verifier_reads_witness) {
        missing.push_back("verifier_reads_witness 必须为 false");
    }
    if (proof.witness_included_in_proof) {
        missing.push_back("witness_included_in_proof 必须为 false");
    }
    if (trace.hidden_heads.size() != model.hidden_profile.size()) {
        missing.push_back("hidden layer trace 数量与 model profile 不一致");
    }
    if (trace.output_heads.size() != model.output_heads.size()) {
        missing.push_back("output head trace 数量与 model 不一致");
    }
    if (metadata.hidden_head_count != model.hidden_heads.size()
        || metadata.output_head_count != model.output_heads.size()) {
        missing.push_back("metadata head count 与 model 不一致");
    }

    const auto require_head = [&](const std::string& layer_kind, std::size_t layer, std::size_t head) {
        const std::string head_prefix = spec.dataset + ":" + layer_kind + "_layer" + std::to_string(layer)
            + ":head" + std::to_string(head);
        const std::string projection_scope = head_prefix + ":pre_rescale_product";
        const std::string rescale_scope = head_prefix + ":projection_rescale_rounding";
        const std::string dot_scope = head_prefix + ":attention_dot_lrelu";
        const std::string max_scope = head_prefix + ":target_max_shift";
        const std::string exp_scope = head_prefix + ":expmap_lookup";
        const std::string softmax_scope = spec.dataset + ":" + layer_kind + "_layer" + std::to_string(layer)
            + "_head" + std::to_string(head);
        zkdl_require_full_relation_record(proof, "projection_matmul", projection_scope, missing);
        zkdl_require_full_relation_record(proof, "rescale_rounding", rescale_scope, missing);
        zkdl_require_digest_binding(proof, "rescale_rounding", rescale_scope, "quant_contract_id", quant.contract_id, missing);
        zkdl_require_digest_binding(proof, "rescale_rounding", rescale_scope, "tie_policy", quant.rounding_policy, missing);
        zkdl_require_zero_counter(proof, "rescale_rounding", rescale_scope, "canonical_rounding_mismatch_count", missing);

        zkdl_require_full_relation_record(proof, "attention_dot_lrelu_lookup_range", dot_scope, missing);
        zkdl_require_digest_binding(proof, "attention_dot_lrelu_lookup_range", dot_scope, "quant_contract_id", quant.contract_id, missing);
        zkdl_require_digest_binding(proof, "attention_dot_lrelu_lookup_range", dot_scope, "table_context_id", table.context_id, missing);
        zkdl_require_digest_binding(proof, "attention_dot_lrelu_lookup_range", dot_scope, "lrelu_table_digest", table.lrelu_digest, missing);
        zkdl_require_zero_counter(proof, "attention_dot_lrelu_lookup_range", dot_scope, "attention_dot_mismatch_count", missing);
        zkdl_require_zero_counter(proof, "attention_dot_lrelu_lookup_range", dot_scope, "lrelu_mismatch_count", missing);

        zkdl_require_full_relation_record(proof, "target_max_shift_comparison", max_scope, missing);
        zkdl_require_digest_binding(proof, "target_max_shift_comparison", max_scope, "table_context_id", table.context_id, missing);
        zkdl_require_zero_counter(proof, "target_max_shift_comparison", max_scope, "max_mismatch_count", missing);

        zkdl_require_full_relation_record(proof, "expmap_lookup_range", exp_scope, missing);
        zkdl_require_digest_binding(proof, "expmap_lookup_range", exp_scope, "table_context_id", table.context_id, missing);
        zkdl_require_digest_binding(proof, "expmap_lookup_range", exp_scope, "exp_table_digest", table.exp_digest, missing);
        zkdl_require_zero_counter(proof, "expmap_lookup_range", exp_scope, "exp_mismatch_count", missing);

        zkdl_require_full_relation_record(proof, "attention_softmax_weighted_aggregation", softmax_scope, missing);
    };

    for (std::size_t layer = 0; layer < trace.hidden_heads.size(); ++layer) {
        for (std::size_t head = 0; head < trace.hidden_heads[layer].size(); ++head) {
            require_head("hidden", layer, head);
            const std::string hidden_scope = spec.dataset + ":hidden_layer" + std::to_string(layer)
                + ":head" + std::to_string(head) + ":elu_lookup";
            zkdl_require_full_relation_record(proof, "elu_lookup_range", hidden_scope, missing);
            zkdl_require_digest_binding(proof, "elu_lookup_range", hidden_scope, "table_context_id", table.context_id, missing);
            zkdl_require_digest_binding(proof, "elu_lookup_range", hidden_scope, "elu_table_digest", table.elu_digest, missing);
            zkdl_require_zero_counter(proof, "elu_lookup_range", hidden_scope, "elu_mismatch_count", missing);
        }
        const std::string concat_scope = spec.dataset + ":hidden_layer" + std::to_string(layer) + ":concat";
        zkdl_require_full_relation_record(proof, "hidden_concat_binding", concat_scope, missing);
    }

    for (std::size_t head = 0; head < trace.output_heads.size(); ++head) {
        require_head("output", 0, head);
        const std::string output_scope = spec.dataset + ":output_layer0:head" + std::to_string(head)
            + ":attention_bias_binding";
        zkdl_require_full_relation_record(proof, "output_attention_bias_scale_binding", output_scope, missing);
        zkdl_require_zero_counter(proof, "output_attention_bias_scale_binding", output_scope, "bias_mismatch_count", missing);
    }

    zkdl_require_full_relation_record(
        proof,
        "output_head_average_public_logits_binding",
        spec.dataset + ":output_layer:all_heads:public_logits",
        missing);

    for (const auto& record : proof.records) {
        if (!record.proof_generated) {
            missing.push_back("relation proof_generated=false: " + record.stage + " scope=" + record.relation_scope);
        }
        if (!record.verified) {
            missing.push_back("relation verified=false: " + record.stage + " scope=" + record.relation_scope);
        }
    }
    return missing;
}
