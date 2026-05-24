#pragma once

// zkDL-GAT full proof object 与 verifier manifest 的最小结构层。
// 这里记录真实 relation proof 的摘要、大小与 verify 结果；缺失或失败的协议子句会阻止 official 导出。

// 单个 full-chain relation proof 的摘要。
// 记录只包含 proof 摘要、timing 和 verifier 结果，不包含 witness 或完整 tensor。
struct ZkdlFullRelationRecord {
    std::string stage;
    std::string relation_scope;
    std::string layer_kind;
    std::size_t layer_index = 0;
    std::size_t head_index = 0;
    bool proof_generated = false;
    bool verified = false;
    double commitment_ms = 0.0;
    double prove_ms = 0.0;
    double verify_ms = 0.0;
    std::uintmax_t proof_size_bytes = 0;
    std::vector<std::string> proof_digests;
};

// 一个 dataset/device 的 full proof object 聚合状态。
// exporter 只读取这里生成的 manifest；proof_generated/verified 为 official gate 的核心字段。
struct ZkdlFullProofObject {
    std::string protocol_id = "GAT-ZKML-zkDL";
    std::string proof_scope = "zkdl_gat_full";
    std::string dataset;
    std::string device;
    std::string metadata_id;
    std::string vk_static_id;
    std::string vk_model_id;
    std::string quant_contract_id;
    bool is_full_dataset = false;
    bool relation_proofs_generated = false;
    bool proof_generated = false;
    bool verified = false;
    bool verifier_reads_witness = false;
    bool witness_included_in_proof = false;
    double commitment_ms = 0.0;
    double prove_ms = 0.0;
    double verify_ms = 0.0;
    std::uintmax_t proof_size_bytes = 0;
    std::vector<ZkdlFullRelationRecord> records;
    std::vector<std::string> missing_relations;
    fs::path proof_object_path;
    fs::path proof_manifest_path;
    fs::path verifier_manifest_path;
};

// 追加一个 relation record，并把三段 timing 与 proof_size 汇总到 proof object。
void append_full_relation_record(ZkdlFullProofObject& proof, ZkdlFullRelationRecord record)
{
    proof.commitment_ms += record.commitment_ms;
    proof.prove_ms += record.prove_ms;
    proof.verify_ms += record.verify_ms;
    proof.proof_size_bytes += record.proof_size_bytes;
    proof.records.push_back(std::move(record));
}

// 写 JSON 字符串数组；用于 manifest 中的 proof_digests 与 missing_relations。
void write_json_string_list_inline(std::ofstream& out, const std::vector<std::string>& values)
{
    out << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out << ", ";
        out << "\"" << json_escape(values[i]) << "\"";
    }
    out << "]";
}

// 将单个 relation record 序列化进 full proof/verifier manifest。
// 这些字段用于 clean verifier 复查 all-layer/all-head 是否完整覆盖。
void write_full_relation_record_json(std::ofstream& out, const ZkdlFullRelationRecord& record, std::size_t indent)
{
    const std::string pad(indent, ' ');
    out << pad << "{\n";
    out << pad << "  \"stage\": \"" << json_escape(record.stage) << "\",\n";
    out << pad << "  \"relation_scope\": \"" << json_escape(record.relation_scope) << "\",\n";
    out << pad << "  \"layer_kind\": \"" << json_escape(record.layer_kind) << "\",\n";
    out << pad << "  \"layer_index\": " << record.layer_index << ",\n";
    out << pad << "  \"head_index\": " << record.head_index << ",\n";
    out << pad << "  \"proof_generated\": " << (record.proof_generated ? "true" : "false") << ",\n";
    out << pad << "  \"verified\": " << (record.verified ? "true" : "false") << ",\n";
    out << pad << "  \"commitment_time_ms\": " << format_ms(record.commitment_ms) << ",\n";
    out << pad << "  \"prove_time_ms\": " << format_ms(record.prove_ms) << ",\n";
    out << pad << "  \"verify_time_ms\": " << format_ms(record.verify_ms) << ",\n";
    out << pad << "  \"proof_size_bytes\": " << record.proof_size_bytes << ",\n";
    out << pad << "  \"proof_digests\": ";
    write_json_string_list_inline(out, record.proof_digests);
    out << "\n" << pad << "}";
}

// 根据 relation record 与 missing_relations 汇总 full proof gate。
// 注意该函数不放宽语义：存在任何 missing relation 时 proof_generated 仍为 false。
void finalize_full_proof_status(ZkdlFullProofObject& proof)
{
    proof.relation_proofs_generated = proof.is_full_dataset
        && !proof.records.empty()
        && std::all_of(proof.records.begin(), proof.records.end(), [](const ZkdlFullRelationRecord& record) {
            return record.proof_generated;
        });
    proof.proof_generated = proof.relation_proofs_generated && proof.missing_relations.empty();
    proof.verified = proof.proof_generated
        && std::all_of(proof.records.begin(), proof.records.end(), [](const ZkdlFullRelationRecord& record) {
            return record.verified;
        });
}

// 写 proof object 摘要、full proof manifest 和 full verifier manifest。
// 副作用是 official exporter 的唯一输入；写出的 proof object 明确不包含完整 witness/tensor。
void write_zkdl_full_proof_object(const ZkdlFullProofObject& proof)
{
    {
        std::ofstream out(proof.proof_object_path);
        out << "ZKDLGATFULLPROOF\n";
        out << "version=1\n";
        out << "protocol_id=" << proof.protocol_id << "\n";
        out << "proof_scope=" << proof.proof_scope << "\n";
        out << "dataset=" << proof.dataset << "\n";
        out << "device=" << proof.device << "\n";
        out << "metadata_id=" << proof.metadata_id << "\n";
        out << "vk_static_id=" << proof.vk_static_id << "\n";
        out << "vk_model_id=" << proof.vk_model_id << "\n";
        out << "quant_contract_id=" << proof.quant_contract_id << "\n";
        out << "full_dataset=" << (proof.is_full_dataset ? "true" : "false") << "\n";
        out << "relation_proofs_generated=" << (proof.relation_proofs_generated ? "true" : "false") << "\n";
        out << "proof_generated=" << (proof.proof_generated ? "true" : "false") << "\n";
        out << "verified=" << (proof.verified ? "true" : "false") << "\n";
        out << "verifier_reads_witness=" << (proof.verifier_reads_witness ? "true" : "false") << "\n";
        out << "witness_included_in_proof=" << (proof.witness_included_in_proof ? "true" : "false") << "\n";
        out << "relation_record_count=" << proof.records.size() << "\n";
        out << "missing_relation_count=" << proof.missing_relations.size() << "\n";
        out << "commitment_time_ms=" << format_ms(proof.commitment_ms) << "\n";
        out << "prove_time_ms=" << format_ms(proof.prove_ms) << "\n";
        out << "verify_time_ms=" << format_ms(proof.verify_ms) << "\n";
        out << "proof_size_bytes=" << proof.proof_size_bytes << "\n";
        for (const auto& record : proof.records) {
            out << "relation=" << record.stage
                << ";scope=" << record.relation_scope
                << ";layer_kind=" << record.layer_kind
                << ";layer=" << record.layer_index
                << ";head=" << record.head_index
                << ";verified=" << (record.verified ? "true" : "false")
                << ";proof_size_bytes=" << record.proof_size_bytes << "\n";
        }
        for (const auto& missing : proof.missing_relations) {
            out << "missing=" << missing << "\n";
        }
        out << "说明=该 proof object 只保存 proof metadata、子证明摘要和 proof_size；不包含完整 witness 或完整 tensor。verified=false 时不得导出 official。\n";
    }
    {
        std::ofstream out(proof.proof_manifest_path);
        out << "{\n";
        out << "  \"protocol_id\": \"" << json_escape(proof.protocol_id) << "\",\n";
        out << "  \"proof_scope\": \"" << json_escape(proof.proof_scope) << "\",\n";
        out << "  \"dataset\": \"" << json_escape(proof.dataset) << "\",\n";
        out << "  \"device\": \"" << json_escape(proof.device) << "\",\n";
        out << "  \"metadata_id\": \"" << json_escape(proof.metadata_id) << "\",\n";
        out << "  \"vk_static_id\": \"" << json_escape(proof.vk_static_id) << "\",\n";
        out << "  \"vk_model_id\": \"" << json_escape(proof.vk_model_id) << "\",\n";
        out << "  \"quant_contract_id\": \"" << json_escape(proof.quant_contract_id) << "\",\n";
        out << "  \"is_full_dataset\": " << (proof.is_full_dataset ? "true" : "false") << ",\n";
        out << "  \"relation_proofs_generated\": " << (proof.relation_proofs_generated ? "true" : "false") << ",\n";
        out << "  \"proof_generated\": " << (proof.proof_generated ? "true" : "false") << ",\n";
        out << "  \"verified\": " << (proof.verified ? "true" : "false") << ",\n";
        out << "  \"verifier_reads_witness\": " << (proof.verifier_reads_witness ? "true" : "false") << ",\n";
        out << "  \"witness_included_in_proof\": " << (proof.witness_included_in_proof ? "true" : "false") << ",\n";
        out << "  \"benchmark_mode\": \"warm\",\n";
        out << "  \"commitment_time_ms\": " << format_ms(proof.commitment_ms) << ",\n";
        out << "  \"prove_time_ms\": " << format_ms(proof.prove_ms) << ",\n";
        out << "  \"verify_time_ms\": " << format_ms(proof.verify_ms) << ",\n";
        out << "  \"proof_size_bytes\": " << proof.proof_size_bytes << ",\n";
        out << "  \"proof_object_path\": \"" << json_escape(proof.proof_object_path.string()) << "\",\n";
        out << "  \"relations\": [\n";
        for (std::size_t i = 0; i < proof.records.size(); ++i) {
            write_full_relation_record_json(out, proof.records[i], 4);
            out << (i + 1 == proof.records.size() ? "\n" : ",\n");
        }
        out << "  ],\n";
        out << "  \"missing_relations\": ";
        write_json_string_list_inline(out, proof.missing_relations);
        out << "\n}\n";
    }
    {
        std::ofstream out(proof.verifier_manifest_path);
        out << "{\n";
        out << "  \"protocol_id\": \"" << json_escape(proof.protocol_id) << "\",\n";
        out << "  \"proof_scope\": \"" << json_escape(proof.proof_scope) << "\",\n";
        out << "  \"dataset\": \"" << json_escape(proof.dataset) << "\",\n";
        out << "  \"device\": \"" << json_escape(proof.device) << "\",\n";
        out << "  \"relation_proofs_generated\": " << (proof.relation_proofs_generated ? "true" : "false") << ",\n";
        out << "  \"proof_generated\": " << (proof.proof_generated ? "true" : "false") << ",\n";
        out << "  \"verified\": " << (proof.verified ? "true" : "false") << ",\n";
        out << "  \"verifier_reads_witness\": " << (proof.verifier_reads_witness ? "true" : "false") << ",\n";
        out << "  \"witness_included_in_proof\": " << (proof.witness_included_in_proof ? "true" : "false") << ",\n";
        out << "  \"relation_verify_count\": " << proof.records.size() << ",\n";
        out << "  \"missing_relation_count\": " << proof.missing_relations.size() << ",\n";
        out << "  \"verify_time_ms\": " << format_ms(proof.verify_ms) << ",\n";
        out << "  \"relations\": [\n";
        for (std::size_t i = 0; i < proof.records.size(); ++i) {
            write_full_relation_record_json(out, proof.records[i], 4);
            out << (i + 1 == proof.records.size() ? "\n" : ",\n");
        }
        out << "  ],\n";
        out << "  \"missing_relations\": ";
        write_json_string_list_inline(out, proof.missing_relations);
        out << "\n}\n";
    }
}
