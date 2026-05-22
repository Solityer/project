#pragma once

// artifact/evidence 写入集中在这里，避免 CPU/GPU runner 与 relation backend
// 继续夹杂结果文件格式。这里仍只写 partial_formal evidence，不写 official manifest。

struct CpuPartialArtifactPayload {
    std::size_t node_count = 0;
    std::size_t edge_count = 0;
    std::size_t padded_node_count = 0;
    std::size_t padded_edge_count = 0;
    std::size_t commitment_block_size = 1;
    std::uint64_t topology_hash = 0;
    std::size_t available_feature_columns = 0;
    std::size_t verified_feature_columns = 0;
    double commitment_ms = 0.0;
    double prove_ms = 0.0;
    double verify_ms = 0.0;
    std::uintmax_t proof_size_bytes = 0;
    std::vector<std::string> relation_details;
};

struct GpuSparseArtifactPayload {
    std::size_t node_count = 0;
    std::size_t edge_count = 0;
    std::size_t padded_node_count = 0;
    std::size_t padded_edge_count = 0;
    std::size_t commitment_block_size = 1;
    std::uint64_t topology_hash = 0;
    double commitment_ms = 0.0;
    double prove_ms = 0.0;
    double verify_ms = 0.0;
    std::uintmax_t proof_size_bytes = 0;
    std::string source_digest;
    std::string destination_digest;
    std::string grouped_sum_digest;
};

struct GpuSoftmaxArtifactPayload {
    std::size_t node_count = 0;
    std::size_t edge_count = 0;
    std::size_t padded_node_count = 0;
    std::size_t padded_edge_count = 0;
    std::size_t feature_columns = 0;
    std::size_t available_feature_columns = 0;
    std::size_t verified_feature_columns = 0;
    std::size_t commitment_block_size = 1;
    std::uint64_t topology_hash = 0;
    bool denominator_ok = false;
    bool inverse_ok = false;
    bool inverse_routing_ok = false;
    bool alpha_from_inverse_ok = false;
    bool alpha_times_denominator_ok = false;
    bool weighted_aggregation_ok = false;
    bool weighted_executed_ok = false;
    double commitment_ms = 0.0;
    double prove_ms = 0.0;
    double verify_ms = 0.0;
    std::uintmax_t proof_size_bytes = 0;
    std::vector<std::string> relation_details;
};

struct GpuOutputBindingArtifactPayload {
    std::size_t node_count = 0;
    std::size_t class_count = 0;
    std::size_t padded_node_count = 0;
    std::size_t padded_class_count = 0;
    std::size_t output_head_count = 0;
    std::size_t commitment_block_size = 1;
    bool output_average_ok = false;
    bool public_logits_binding_ok = false;
    double commitment_ms = 0.0;
    double prove_ms = 0.0;
    double verify_ms = 0.0;
    std::uintmax_t proof_size_bytes = 0;
    std::string transcript_digest;
};

std::pair<fs::path, fs::path> write_cpu_partial_artifact(
    const DatasetRunSpec& spec,
    const fs::path& output_root,
    const CpuPartialArtifactPayload& payload)
{
    const auto run_dir = device_run_dir(output_root, spec.dataset, "cpu");
    ensure_dir(run_dir / "formal_artifacts");
    ensure_dir(output_root / "evidence");
    const auto proof_path = run_dir / "formal_artifacts" / "zkdl_cpu_relation_partial.proof.txt";
    {
        std::ofstream out(proof_path);
        out << "ZKDLGATCPUFORMAL\n";
        out << "version=1\n";
        out << "dataset=" << spec.dataset << "\n";
        out << "backend=cpu_only_host_fr_g1_commitment\n";
        out << "relation_scope=hidden_layer0_head0_sparse_softmax_partial\n";
        out << "node_count=" << payload.node_count << "\n";
        out << "edge_count=" << payload.edge_count << "\n";
        out << "padded_node_count=" << payload.padded_node_count << "\n";
        out << "padded_edge_count=" << payload.padded_edge_count << "\n";
        out << "commitment_block_size=" << payload.commitment_block_size << "\n";
        out << "topology_hash=" << payload.topology_hash << "\n";
        out << "source_sparse_routing_status=PASS\n";
        out << "destination_sparse_routing_status=PASS\n";
        out << "grouped_sum_status=PASS\n";
        out << "denominator_grouped_sum_status=PASS\n";
        out << "inverse_status=PASS\n";
        out << "inverse_destination_routing_status=PASS\n";
        out << "alpha_from_inverse_status=PASS\n";
        out << "alpha_times_denominator_status=PASS\n";
        out << "weighted_aggregation_status=NOT_RUN_CPU_EXTENSION\n";
        out << "available_feature_columns=" << payload.available_feature_columns << "\n";
        out << "verified_feature_columns=" << payload.verified_feature_columns << "\n";
        out << "proof_size_bytes=" << payload.proof_size_bytes << "\n";
        out << "relation_details=" << payload.relation_details.size() << "\n";
        for (const auto& detail : payload.relation_details) out << detail << "\n";
        out << "说明=CPU proof artifact 只包含 relation metadata、证明大小和 transcript 摘要；不包含完整 witness、完整 tensor 或 debug dump。\n";
    }

    const auto evidence_path = output_root / "evidence" / ("cpu_relation_partial_" + spec.dataset + ".json");
    {
        std::ofstream out(evidence_path);
        out << "{\n";
        out << "  \"dataset\": \"" << json_escape(spec.dataset) << "\",\n";
        out << "  \"device\": \"cpu\",\n";
        out << "  \"status\": \"partial_formal\",\n";
        out << "  \"backend\": \"cpu_only_host_fr_g1_commitment\",\n";
        out << "  \"relation_scope\": \"hidden_layer0_head0_sparse_softmax_partial\",\n";
        out << "  \"node_count\": \"" << payload.node_count << "\",\n";
        out << "  \"edge_count\": \"" << payload.edge_count << "\",\n";
        out << "  \"commitment_block_size\": \"" << payload.commitment_block_size << "\",\n";
        out << "  \"topology_hash\": \"" << payload.topology_hash << "\",\n";
        out << "  \"source_sparse_routing\": \"PASS\",\n";
        out << "  \"destination_sparse_routing\": \"PASS\",\n";
        out << "  \"grouped_sum_by_dst\": \"PASS\",\n";
        out << "  \"denominator_grouped_sum\": \"PASS\",\n";
        out << "  \"inverse\": \"PASS\",\n";
        out << "  \"inverse_destination_routing\": \"PASS\",\n";
        out << "  \"alpha_from_inverse\": \"PASS\",\n";
        out << "  \"alpha_times_denominator\": \"PASS\",\n";
        out << "  \"weighted_aggregation\": \"NOT_RUN_CPU_EXTENSION\",\n";
        out << "  \"available_feature_columns\": \"" << payload.available_feature_columns << "\",\n";
        out << "  \"verified_feature_columns\": \"" << payload.verified_feature_columns << "\",\n";
        out << "  \"commitment_time_ms\": \"" << format_ms(payload.commitment_ms) << "\",\n";
        out << "  \"prove_time_ms\": \"" << format_ms(payload.prove_ms) << "\",\n";
        out << "  \"verify_time_ms\": \"" << format_ms(payload.verify_ms) << "\",\n";
        out << "  \"proof_size_bytes\": \"" << payload.proof_size_bytes << "\",\n";
        out << "  \"proof_path\": \"" << json_escape(proof_path.string()) << "\",\n";
        out << "  \"note\": \"CPU-only hidden layer0/head0 sparse + denominator/inverse/alpha relation-level formal result；weighted aggregation、max/ExpMap/lookup/range/ELU/output aggregation/rescale remain open.\"\n";
        out << "}\n";
    }
    return {proof_path, evidence_path};
}

void write_cpu_partial_debug_resume(
    const DatasetRunSpec& spec,
    const fs::path& output_root,
    const CpuPartialArtifactPayload& payload,
    const fs::path& proof_path,
    const fs::path& evidence_path)
{
    const auto run_dir = device_run_dir(output_root, spec.dataset, "cpu");
    {
        std::ofstream debug(run_dir / "proof_debug.log");
        debug << "dataset=" << spec.dataset << " device=cpu status=partial_formal official=blocked\n";
        debug << "backend=cpu_only_host_fr_g1_commitment\n";
        debug << "nodes=" << payload.node_count << " edges=" << payload.edge_count
              << " topology_hash=" << payload.topology_hash
              << " commitment_block_size=" << payload.commitment_block_size << "\n";
        debug << "commitment_ms=" << format_ms(payload.commitment_ms)
              << " prove_ms=" << format_ms(payload.prove_ms)
              << " verify_ms=" << format_ms(payload.verify_ms)
              << " proof_size_bytes=" << payload.proof_size_bytes << "\n";
        for (const auto& detail : payload.relation_details) {
            debug << detail << "\n";
        }
        debug << "note=CPU relation-level partial formal 仅用于调试结果表；不是 official Level A\n";
    }
    {
        std::ofstream resume(run_dir / "RESUME_POINT.md");
        resume << "# zkDL-GAT CPU 继续点\n\n";
        resume << "- 数据集: `" << spec.dataset << "`\n";
        resume << "- 当前状态: `partial_formal`，official 仍为 `blocked + NA`。\n";
        resume << "- CPU backend: `CpuFrTensor` + host `G1Jacobian` + `CpuCommitment::commit/open/verify_opening`。\n";
        resume << "- commitment block size: `" << payload.commitment_block_size << "`；用于减少 host G1 opening replay 的外层承诺行数。\n";
        resume << "- 已完成 relation: source/destination sparse routing、denominator grouped sum、inverse soundness、inverse destination routing、alpha from inverse 与 alpha*denominator consistency。\n";
        resume << "- 证明范围: hidden layer0/head0 full graph / full edge set。\n";
        resume << "- proof artifact: `" << proof_path.string() << "`\n";
        resume << "- evidence: `" << evidence_path.string() << "`\n";
        resume << "- runner log: `" << (run_dir / "cpu_formal_chain.log").string() << "`\n";
        resume << "- Level A blocker: max/range、ExpMap lookup/range、LeakyReLU/ELU lookup-range、output aggregation、rescale/rounding 和所有层/头 full proof object 仍未闭合。\n";
    }
}

std::pair<fs::path, fs::path> write_gpu_sparse_artifact(
    const DatasetRunSpec& spec,
    const fs::path& output_root,
    const GpuSparseArtifactPayload& payload)
{
    const auto run_dir = device_run_dir(output_root, spec.dataset);
    ensure_dir(run_dir / "formal_artifacts");
    ensure_dir(output_root / "evidence");
    const auto proof_path = run_dir / "formal_artifacts" / "zkdl_sparse_grouped_relations.proof.txt";
    {
        std::ofstream out(proof_path);
        out << "ZKDLGATS\n";
        out << "version=1\n";
        out << "dataset=" << spec.dataset << "\n";
        out << "relation_scope=hidden_layer0_head0_sparse_routing_and_one_dim_grouped_sum\n";
        out << "node_count=" << payload.node_count << "\n";
        out << "edge_count=" << payload.edge_count << "\n";
        out << "padded_node_count=" << payload.padded_node_count << "\n";
        out << "padded_edge_count=" << payload.padded_edge_count << "\n";
        out << "edge_range=0.." << payload.edge_count << "\n";
        out << "commitment_block_size=" << payload.commitment_block_size << "\n";
        out << "topology_hash=" << payload.topology_hash << "\n";
        out << "source_sparse_routing_status=PASS\n";
        out << "destination_sparse_routing_status=PASS\n";
        out << "grouped_sum_status=PASS\n";
        out << "source_transcript_digest=" << payload.source_digest << "\n";
        out << "destination_transcript_digest=" << payload.destination_digest << "\n";
        out << "grouped_sum_transcript_digest=" << payload.grouped_sum_digest << "\n";
        out << "proof_size_bytes=" << payload.proof_size_bytes << "\n";
        out << "说明=本 proof artifact 只包含 relation metadata、承诺/opening/sumcheck 证明大小与 transcript 摘要，不包含完整 witness、完整 tensor 或 debug dump。\n";
    }

    const auto evidence_path = output_root / "evidence" / ("sparse_grouped_relation_" + spec.dataset + ".json");
    {
        std::ofstream out(evidence_path);
        out << "{\n";
        out << "  \"dataset\": \"" << json_escape(spec.dataset) << "\",\n";
        out << "  \"status\": \"partial_formal\",\n";
        out << "  \"relation_scope\": \"hidden_layer0_head0_sparse_routing_and_one_dim_grouped_sum\",\n";
        out << "  \"node_count\": \"" << payload.node_count << "\",\n";
        out << "  \"edge_count\": \"" << payload.edge_count << "\",\n";
        out << "  \"edge_range\": \"0.." << payload.edge_count << "\",\n";
        out << "  \"commitment_block_size\": \"" << payload.commitment_block_size << "\",\n";
        out << "  \"topology_hash\": \"" << payload.topology_hash << "\",\n";
        out << "  \"source_sparse_routing\": \"PASS\",\n";
        out << "  \"destination_sparse_routing\": \"PASS\",\n";
        out << "  \"grouped_sum_by_dst\": \"PASS\",\n";
        out << "  \"commitment_time_ms\": \"" << format_ms(payload.commitment_ms) << "\",\n";
        out << "  \"prove_time_ms\": \"" << format_ms(payload.prove_ms) << "\",\n";
        out << "  \"verify_time_ms\": \"" << format_ms(payload.verify_ms) << "\",\n";
        out << "  \"proof_size_bytes\": \"" << payload.proof_size_bytes << "\",\n";
        out << "  \"proof_path\": \"" << json_escape(proof_path.string()) << "\",\n";
        out << "  \"note\": \"一维 grouped sum 使用 hidden layer0 head0 的 source attention edge 值作为输入；它证明 grouped sum gadget 本身，不等同于 softmax denominator 或 weighted aggregation。\"\n";
        out << "}\n";
    }
    return {proof_path, evidence_path};
}

void write_gpu_sparse_debug_resume(
    const DatasetRunSpec& spec,
    const fs::path& output_root,
    const GpuSparseArtifactPayload& payload,
    const fs::path& proof_path,
    const fs::path& evidence_path)
{
    const auto run_dir = device_run_dir(output_root, spec.dataset);
    {
        std::ofstream debug(run_dir / "proof_debug.log");
        debug << "dataset=" << spec.dataset << " stage=formal_sparse_grouped status=PASS\n";
        debug << "nodes=" << payload.node_count << " edges=" << payload.edge_count
              << " topology_hash=" << payload.topology_hash
              << " commitment_block_size=" << payload.commitment_block_size << "\n";
        debug << "source_sparse_routing_digest=" << payload.source_digest << "\n";
        debug << "destination_sparse_routing_digest=" << payload.destination_digest << "\n";
        debug << "grouped_sum_digest=" << payload.grouped_sum_digest << "\n";
        debug << "commitment_ms=" << format_ms(payload.commitment_ms)
              << " prove_ms=" << format_ms(payload.prove_ms)
              << " verify_ms=" << format_ms(payload.verify_ms)
              << " proof_size_bytes=" << payload.proof_size_bytes << "\n";
    }
    {
        std::ofstream resume(run_dir / "RESUME_POINT.md");
        resume << "# zkDL GAT 数据集继续点\n\n";
        resume << "- 数据集: `" << spec.dataset << "`\n";
        resume << "- 本轮 relation: `SparseSourceRoutingRelationProof`, `SparseDestinationRoutingRelationProof`, `GroupedSumRelationProof`\n";
        resume << "- 证明范围: full graph / full edge set，edge range `0.." << payload.edge_count << "`\n";
        resume << "- commitment block size: `" << payload.commitment_block_size << "`\n";
        resume << "- 结果: `PASS`\n";
        resume << "- proof artifact: `" << proof_path.string() << "`\n";
        resume << "- evidence: `" << evidence_path.string() << "`\n";
        resume << "- Level A 状态: blocked；softmax ExpMap/inverse、LeakyReLU/ELU lookup-range、weighted/output aggregation、real rescale chain 与 rounding sign/tie-breaking 仍未闭合。\n";
    }
}

std::pair<fs::path, fs::path> write_gpu_softmax_artifact(
    const DatasetRunSpec& spec,
    const fs::path& output_root,
    const GpuSoftmaxArtifactPayload& payload)
{
    const auto run_dir = device_run_dir(output_root, spec.dataset);
    ensure_dir(run_dir / "formal_artifacts");
    ensure_dir(output_root / "evidence");
    const auto proof_path = run_dir / "formal_artifacts" / "zkdl_softmax_aggregation_relations.proof.txt";
    {
        std::ofstream out(proof_path);
        out << "ZKDLGATA\n";
        out << "version=1\n";
        out << "dataset=" << spec.dataset << "\n";
        out << "relation_scope=hidden_layer0_head0_softmax_denominator_inverse_alpha_weighted_aggregation\n";
        out << "node_count=" << payload.node_count << "\n";
        out << "edge_count=" << payload.edge_count << "\n";
        out << "padded_node_count=" << payload.padded_node_count << "\n";
        out << "padded_edge_count=" << payload.padded_edge_count << "\n";
        out << "feature_columns=" << payload.feature_columns << "\n";
        out << "available_feature_columns=" << payload.available_feature_columns << "\n";
        out << "verified_feature_columns=" << payload.verified_feature_columns << "\n";
        out << "commitment_block_size=" << payload.commitment_block_size << "\n";
        out << "topology_hash=" << payload.topology_hash << "\n";
        out << "denominator_grouped_sum_status=" << (payload.denominator_ok ? "PASS" : "FAIL") << "\n";
        out << "inverse_status=" << (payload.inverse_ok ? "PASS" : "FAIL") << "\n";
        out << "inverse_destination_routing_status=" << (payload.inverse_routing_ok ? "PASS" : "FAIL") << "\n";
        out << "alpha_from_inverse_status=" << (payload.alpha_from_inverse_ok ? "PASS" : "FAIL") << "\n";
        out << "alpha_times_denominator_status=" << (payload.alpha_times_denominator_ok ? "PASS" : "FAIL") << "\n";
        out << "weighted_aggregation_status="
            << (payload.weighted_aggregation_ok ? "PASS" : (payload.weighted_executed_ok ? "PARTIAL_PASS" : "FAIL")) << "\n";
        out << "proof_size_bytes=" << payload.proof_size_bytes << "\n";
        out << "relation_details=" << payload.relation_details.size() << "\n";
        for (const auto& detail : payload.relation_details) out << detail << "\n";
        out << "说明=本 proof artifact 只包含 relation metadata、证明大小与 transcript 摘要，不包含完整 witness、完整 tensor 或 debug dump。\n";
    }

    const auto evidence_path = output_root / "evidence" / ("softmax_aggregation_relation_" + spec.dataset + ".json");
    {
        std::ofstream out(evidence_path);
        out << "{\n";
        out << "  \"dataset\": \"" << json_escape(spec.dataset) << "\",\n";
        out << "  \"status\": \"partial_formal\",\n";
        out << "  \"relation_scope\": \"hidden_layer0_head0_softmax_denominator_inverse_alpha_weighted_aggregation\",\n";
        out << "  \"node_count\": \"" << payload.node_count << "\",\n";
        out << "  \"edge_count\": \"" << payload.edge_count << "\",\n";
        out << "  \"feature_columns\": \"" << payload.feature_columns << "\",\n";
        out << "  \"available_feature_columns\": \"" << payload.available_feature_columns << "\",\n";
        out << "  \"verified_feature_columns\": \"" << payload.verified_feature_columns << "\",\n";
        out << "  \"commitment_block_size\": \"" << payload.commitment_block_size << "\",\n";
        out << "  \"topology_hash\": \"" << payload.topology_hash << "\",\n";
        out << "  \"denominator_grouped_sum\": \"" << (payload.denominator_ok ? "PASS" : "FAIL") << "\",\n";
        out << "  \"inverse\": \"" << (payload.inverse_ok ? "PASS" : "FAIL") << "\",\n";
        out << "  \"inverse_destination_routing\": \"" << (payload.inverse_routing_ok ? "PASS" : "FAIL") << "\",\n";
        out << "  \"alpha_from_inverse\": \"" << (payload.alpha_from_inverse_ok ? "PASS" : "FAIL") << "\",\n";
        out << "  \"alpha_times_denominator\": \"" << (payload.alpha_times_denominator_ok ? "PASS" : "FAIL") << "\",\n";
        out << "  \"weighted_aggregation\": \""
            << (payload.weighted_aggregation_ok ? "PASS" : (payload.weighted_executed_ok ? "PARTIAL_PASS" : "FAIL")) << "\",\n";
        out << "  \"commitment_time_ms\": \"" << format_ms(payload.commitment_ms) << "\",\n";
        out << "  \"prove_time_ms\": \"" << format_ms(payload.prove_ms) << "\",\n";
        out << "  \"verify_time_ms\": \"" << format_ms(payload.verify_ms) << "\",\n";
        out << "  \"proof_size_bytes\": \"" << payload.proof_size_bytes << "\",\n";
        out << "  \"proof_path\": \"" << json_escape(proof_path.string()) << "\",\n";
        out << "  \"note\": \"该 evidence 证明 hidden layer0/head0 的 U 分母、inverse、alpha 与 weighted aggregation 的 zkDL relation-level 链条；ExpMap、max/range、ELU、output aggregation、rescale 与 rounding 仍未闭合，因此不写入 official latest。\"\n";
        out << "}\n";
    }
    return {proof_path, evidence_path};
}

void write_gpu_softmax_debug_resume(
    const DatasetRunSpec& spec,
    const fs::path& output_root,
    const GpuSoftmaxArtifactPayload& payload,
    const fs::path& proof_path,
    const fs::path& evidence_path)
{
    const auto run_dir = device_run_dir(output_root, spec.dataset);
    {
        std::ofstream debug(run_dir / "proof_debug.log", std::ios::app);
        debug << "dataset=" << spec.dataset << " stage=formal_softmax_aggregation status="
              << (payload.weighted_aggregation_ok ? "PASS" : "PARTIAL") << "\n";
        debug << "nodes=" << payload.node_count << " edges=" << payload.edge_count
              << " topology_hash=" << payload.topology_hash
              << " commitment_block_size=" << payload.commitment_block_size
              << " feature_columns=" << payload.feature_columns
              << " available_feature_columns=" << payload.available_feature_columns
              << " verified_feature_columns=" << payload.verified_feature_columns << "\n";
        debug << "commitment_ms=" << format_ms(payload.commitment_ms)
              << " prove_ms=" << format_ms(payload.prove_ms)
              << " verify_ms=" << format_ms(payload.verify_ms)
              << " proof_size_bytes=" << payload.proof_size_bytes << "\n";
    }
    {
        std::ofstream resume(run_dir / "RESUME_POINT.md", std::ios::app);
        resume << "\n## softmax/aggregation relation 继续点\n\n";
        resume << "- 本轮 relation: `denominator grouped sum`, `inverse Hadamard`, `alpha multiplication`, `weighted aggregation`\n";
        resume << "- 证明范围: hidden layer0/head0 full graph / full edge set，feature columns `"
               << payload.verified_feature_columns << "/" << payload.available_feature_columns
               << "`；本次实际运行 columns `" << payload.feature_columns << "`\n";
        resume << "- commitment block size: `" << payload.commitment_block_size << "`\n";
        resume << "- 结果: `" << (payload.weighted_aggregation_ok ? "PASS" : "PARTIAL") << "`\n";
        resume << "- proof artifact: `" << proof_path.string() << "`\n";
        resume << "- evidence: `" << evidence_path.string() << "`\n";
        resume << "- Level A 状态: blocked；softmax max/ExpMap、LeakyReLU/ELU lookup-range、output aggregation、real rescale chain 与 rounding sign/tie-breaking 仍未闭合。\n";
    }
}

std::pair<fs::path, fs::path> write_gpu_output_binding_artifact(
    const DatasetRunSpec& spec,
    const fs::path& output_root,
    const GpuOutputBindingArtifactPayload& payload)
{
    const auto run_dir = device_run_dir(output_root, spec.dataset);
    ensure_dir(run_dir / "formal_artifacts");
    ensure_dir(output_root / "evidence");
    const auto proof_path = run_dir / "formal_artifacts" / "zkdl_output_binding_relations.proof.txt";
    {
        std::ofstream out(proof_path);
        out << "ZKDLGATOUTPUTBINDING\n";
        out << "version=1\n";
        out << "dataset=" << spec.dataset << "\n";
        out << "relation_scope=output_head_average_public_logits_binding\n";
        out << "node_count=" << payload.node_count << "\n";
        out << "class_count=" << payload.class_count << "\n";
        out << "padded_node_count=" << payload.padded_node_count << "\n";
        out << "padded_class_count=" << payload.padded_class_count << "\n";
        out << "output_head_count=" << payload.output_head_count << "\n";
        out << "commitment_block_size=" << payload.commitment_block_size << "\n";
        out << "output_average_status=" << (payload.output_average_ok ? "PASS" : "FAIL") << "\n";
        out << "public_logits_binding_status=" << (payload.public_logits_binding_ok ? "PASS" : "FAIL") << "\n";
        out << "linear_relation_digest=" << payload.transcript_digest << "\n";
        out << "proof_size_bytes=" << payload.proof_size_bytes << "\n";
        out << "说明=本 artifact 证明输出头平均后的承诺开值等于 public logits 的公开 MLE 取值；它不包含完整 logits tensor 或 witness dump，也不是 full GAT official proof。\n";
    }

    const auto evidence_path = output_root / "evidence" / ("output_binding_relation_" + spec.dataset + ".json");
    {
        std::ofstream out(evidence_path);
        out << "{\n";
        out << "  \"dataset\": \"" << json_escape(spec.dataset) << "\",\n";
        out << "  \"status\": \"partial_formal\",\n";
        out << "  \"relation_scope\": \"output_head_average_public_logits_binding\",\n";
        out << "  \"node_count\": \"" << payload.node_count << "\",\n";
        out << "  \"class_count\": \"" << payload.class_count << "\",\n";
        out << "  \"padded_node_count\": \"" << payload.padded_node_count << "\",\n";
        out << "  \"padded_class_count\": \"" << payload.padded_class_count << "\",\n";
        out << "  \"output_head_count\": \"" << payload.output_head_count << "\",\n";
        out << "  \"commitment_block_size\": \"" << payload.commitment_block_size << "\",\n";
        out << "  \"output_average\": \"" << (payload.output_average_ok ? "PASS" : "FAIL") << "\",\n";
        out << "  \"public_logits_binding\": \"" << (payload.public_logits_binding_ok ? "PASS" : "FAIL") << "\",\n";
        out << "  \"commitment_time_ms\": \"" << format_ms(payload.commitment_ms) << "\",\n";
        out << "  \"prove_time_ms\": \"" << format_ms(payload.prove_ms) << "\",\n";
        out << "  \"verify_time_ms\": \"" << format_ms(payload.verify_ms) << "\",\n";
        out << "  \"proof_size_bytes\": \"" << payload.proof_size_bytes << "\",\n";
        out << "  \"transcript_digest\": \"" << json_escape(payload.transcript_digest) << "\",\n";
        out << "  \"proof_path\": \"" << json_escape(proof_path.string()) << "\",\n";
        out << "  \"note\": \"该 evidence 使用 zkDL linear relation 绑定 output head average 与 public logits MLE；output attention 前序、bias 前序、rescale/rounding 和完整 verifier 仍未闭合。\"\n";
        out << "}\n";
    }
    return {proof_path, evidence_path};
}

void write_gpu_output_binding_debug_resume(
    const DatasetRunSpec& spec,
    const fs::path& output_root,
    const GpuOutputBindingArtifactPayload& payload,
    const fs::path& proof_path,
    const fs::path& evidence_path)
{
    const auto run_dir = device_run_dir(output_root, spec.dataset);
    {
        std::ofstream debug(run_dir / "proof_debug.log", std::ios::app);
        debug << "dataset=" << spec.dataset << " stage=formal_output_binding status="
              << (payload.output_average_ok && payload.public_logits_binding_ok ? "PASS" : "FAIL")
              << " output_heads=" << payload.output_head_count
              << " classes=" << payload.class_count
              << " commitment_block_size=" << payload.commitment_block_size
              << " proof_size_bytes=" << payload.proof_size_bytes
              << " digest=" << payload.transcript_digest << "\n";
    }
    {
        std::ofstream resume(run_dir / "RESUME_POINT.md", std::ios::app);
        resume << "\n## output binding relation 继续点\n\n";
        resume << "- 本轮 relation: `output_head_average_public_logits_binding`\n";
        resume << "- 证明范围: full graph / all output heads / public logits MLE。\n";
        resume << "- output heads: `" << payload.output_head_count << "`；classes `" << payload.class_count << "`\n";
        resume << "- 结果: `" << (payload.output_average_ok && payload.public_logits_binding_ok ? "PASS" : "FAIL") << "`\n";
        resume << "- proof artifact: `" << proof_path.string() << "`\n";
        resume << "- evidence: `" << evidence_path.string() << "`\n";
        resume << "- Level A 状态: blocked；output attention 前序链、bias 前序链、rescale/rounding 与 clean verifier 仍未闭合。\n";
    }
}
