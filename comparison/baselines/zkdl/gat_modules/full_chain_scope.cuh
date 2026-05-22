#pragma once

// full-chain scope 是正式实验流程的一部分：它把协议要求的完整 GAT 证明范围
// 写入每个 dataset/device 的 runs/evidence 目录。该文件不生成 official manifest，
// 也不把 relation-level partial artifact 冒充为 full proof。

struct FullChainScopeStatus
{
    std::string dataset;
    std::string device;
    std::size_t node_count = 0;
    std::size_t edge_count = 0;
    std::size_t feature_count = 0;
    std::size_t class_count = 0;
    std::size_t graph_count = 0;
    std::size_t hidden_layer_count = 0;
    std::size_t hidden_head_count = 0;
    std::size_t output_head_count = 0;
    bool is_full_dataset = false;
    bool official_ready = false;
    std::vector<std::string> required_clauses;
    std::vector<std::string> partial_artifacts;
    std::vector<std::string> missing_clauses;
    fs::path status_path;
    fs::path evidence_path;
};

std::string join_semicolon(const std::vector<std::string>& values)
{
    std::ostringstream out;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out << ';';
        out << values[i];
    }
    return out.str();
}

void write_json_string_array(std::ofstream& out, const std::string& key, const std::vector<std::string>& values, bool trailing_comma)
{
    out << "  \"" << key << "\": [";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out << ", ";
        out << "\"" << json_escape(values[i]) << "\"";
    }
    out << "]";
    if (trailing_comma) out << ",";
    out << "\n";
}

std::vector<std::string> zkdl_full_chain_required_clauses()
{
    return {
        "M_pub/Topo/VK_static/VK_model/Y public binding",
        "Com_base/Com_comp/Com_aux/Com_tbl/Com_agg zkDL proof object layout",
        "Pi_slice feature retrieval and topology slice consistency",
        "all hidden layers and all hidden heads",
        "hidden projection matmul relations",
        "hidden attention dot relations",
        "source and destination sparse routing for every head",
        "LeakyReLU lookup-range relation",
        "target-node max/shift comparison relation",
        "ExpMap table lookup and range relation",
        "softmax denominator grouped sum and nonzero inverse soundness",
        "alpha consistency and weighted aggregation",
        "ELU lookup-range relation for hidden activation",
        "hidden concat binding",
        "all output attention heads",
        "output aggregation and output bias binding",
        "head average relation",
        "public logits binding",
        "real fixed-point rescale chain",
        "rounding sign/tie-breaking public contract",
        "verifier replay without witness tensor access",
        "proof_size_bytes from serialized zkDL proof object only"
    };
}

std::vector<std::string> zkdl_full_chain_missing_clauses(const std::string& device)
{
    std::vector<std::string> missing = {
        "缺少协议定义的 pi_GAT^zkDL 完整 proof object 序列化",
        "缺少 Com_base/Com_comp/Com_aux/Com_tbl/Com_agg 的全链承诺布局",
        "lookup/range/max/comparison 仍未绑定真实 GAT VK_static/table/context",
        "ExpMap table/range 尚未接入 softmax runner",
        "LeakyReLU/ELU lookup-range 尚未接入隐藏层和输出层语义",
        "target-node max/shift 仍缺 per-destination representative selector",
        "all-layer/all-head relation runner 尚未生成全链 sumcheck/eval proof",
        "output attention aggregation 尚未进入 zkDL relation proof",
        "output bias/head average/public logits binding 尚未进入 verifier 可复验对象",
        "real rescale/rounding sign/tie-breaking public contract 尚未落到 proof",
        "clean verifier 尚未按协议重放 full-chain transcript 与所有 relation",
        "proof_size_bytes 尚无完整 zkDL proof object 可统计"
    };
    if (device == "gpu") {
        missing.push_back("GPU verifier 仍不是完整 GPU/batched verifier；host replay 不能写成 GPU official");
    } else {
        missing.push_back("CPU full-chain batched opening/verifier 仍未闭合到 all-layer/all-head proof");
    }
    return missing;
}

std::vector<std::string> existing_partial_artifacts_for_scope(
    const fs::path& output_root,
    const std::string& dataset,
    const std::string& device)
{
    const auto run_dir = device_run_dir(output_root, dataset, device);
    std::vector<std::string> paths;
    const std::vector<fs::path> candidates = device == "cpu"
        ? std::vector<fs::path>{
              run_dir / "formal_artifacts" / "zkdl_cpu_relation_partial.proof.txt"}
        : std::vector<fs::path>{
              run_dir / "formal_artifacts" / "zkdl_sparse_grouped_relations.proof.txt",
              run_dir / "formal_artifacts" / "zkdl_softmax_aggregation_relations.proof.txt"};
    for (const auto& path : candidates) {
        if (fs::exists(path)) paths.push_back(path.string());
    }
    return paths;
}

FullChainScopeStatus write_full_chain_scope_status(
    const DatasetRunSpec& spec,
    const fs::path& output_root,
    const std::string& device,
    const GraphDataset& dataset,
    const LocalGraph& graph,
    const ModelParams& model)
{
    if (device != "cpu" && device != "gpu") {
        throw std::runtime_error("full-chain scope device 必须是 cpu 或 gpu");
    }
    const auto run_dir = device_run_dir(output_root, spec.dataset, device);
    ensure_dir(run_dir / "formal_artifacts");
    ensure_dir(output_root / "evidence");

    FullChainScopeStatus status;
    status.dataset = spec.dataset;
    status.device = device;
    status.node_count = graph.num_nodes;
    status.edge_count = graph.edges.size();
    status.feature_count = graph.num_features;
    status.class_count = graph.num_classes;
    status.graph_count = dataset.graph_count;
    status.hidden_layer_count = model.hidden_profile.size();
    status.hidden_head_count = model.hidden_heads.size();
    status.output_head_count = model.output_heads.size();
    status.is_full_dataset = graph.num_nodes == dataset.num_nodes;
    status.required_clauses = zkdl_full_chain_required_clauses();
    status.partial_artifacts = existing_partial_artifacts_for_scope(output_root, spec.dataset, device);
    status.missing_clauses = zkdl_full_chain_missing_clauses(device);
    status.official_ready = status.is_full_dataset && status.missing_clauses.empty();
    status.status_path = run_dir / "formal_artifacts" / "zkdl_full_chain_scope.status.txt";
    status.evidence_path = output_root / "evidence" / ("full_chain_scope_" + device + "_" + spec.dataset + ".json");

    {
        std::ofstream out(status.status_path);
        out << "ZKDLGATFULLCHAINSCOPE\n";
        out << "version=1\n";
        out << "dataset=" << status.dataset << "\n";
        out << "device=" << status.device << "\n";
        out << "status=" << (status.official_ready ? "official_ready" : "blocked") << "\n";
        out << "official_manifest_written=false\n";
        out << "full_dataset=" << (status.is_full_dataset ? "true" : "false") << "\n";
        out << "node_count=" << status.node_count << "\n";
        out << "edge_count=" << status.edge_count << "\n";
        out << "feature_count=" << status.feature_count << "\n";
        out << "class_count=" << status.class_count << "\n";
        out << "graph_count=" << status.graph_count << "\n";
        out << "hidden_layer_count=" << status.hidden_layer_count << "\n";
        out << "hidden_head_count=" << status.hidden_head_count << "\n";
        out << "output_head_count=" << status.output_head_count << "\n";
        out << "partial_artifacts=" << join_semicolon(status.partial_artifacts) << "\n";
        out << "missing_clause_count=" << status.missing_clauses.size() << "\n";
        for (const auto& clause : status.missing_clauses) {
            out << "missing=" << clause << "\n";
        }
        out << "说明=该文件是正式 full-chain 流程的范围状态，不是 proof artifact，也不会进入 official latest。\n";
    }
    {
        std::ofstream out(status.evidence_path);
        out << "{\n";
        out << "  \"dataset\": \"" << json_escape(status.dataset) << "\",\n";
        out << "  \"device\": \"" << json_escape(status.device) << "\",\n";
        out << "  \"status\": \"" << (status.official_ready ? "official_ready" : "blocked") << "\",\n";
        out << "  \"official_manifest_written\": \"false\",\n";
        out << "  \"full_dataset\": \"" << (status.is_full_dataset ? "true" : "false") << "\",\n";
        out << "  \"node_count\": \"" << status.node_count << "\",\n";
        out << "  \"edge_count\": \"" << status.edge_count << "\",\n";
        out << "  \"feature_count\": \"" << status.feature_count << "\",\n";
        out << "  \"class_count\": \"" << status.class_count << "\",\n";
        out << "  \"graph_count\": \"" << status.graph_count << "\",\n";
        out << "  \"hidden_layer_count\": \"" << status.hidden_layer_count << "\",\n";
        out << "  \"hidden_head_count\": \"" << status.hidden_head_count << "\",\n";
        out << "  \"output_head_count\": \"" << status.output_head_count << "\",\n";
        write_json_string_array(out, "required_clauses", status.required_clauses, true);
        write_json_string_array(out, "partial_artifacts", status.partial_artifacts, true);
        write_json_string_array(out, "missing_clauses", status.missing_clauses, false);
        out << "}\n";
    }
    {
        std::ofstream debug(run_dir / "proof_debug.log", std::ios::app);
        debug << "dataset=" << status.dataset
              << " device=" << status.device
              << " stage=full_chain_scope status=" << (status.official_ready ? "official_ready" : "blocked")
              << " missing_clause_count=" << status.missing_clauses.size()
              << " status_path=" << status.status_path.string()
              << " evidence_path=" << status.evidence_path.string() << "\n";
    }
    return status;
}

FullChainScopeStatus run_formal_full_chain_scope(
    const fs::path& project_root,
    const fs::path& output_root,
    const DatasetRunSpec& spec,
    const std::string& device)
{
    std::cout << "formal_full_chain_scope dataset=" << spec.dataset
              << " device=" << device
              << " stage=load_dataset\n";
    const auto dataset = load_cached_dataset(project_root / "data/cache" / spec.cache_name);
    const auto graph = normalize_graph(dataset, spec);
    if (graph.num_nodes != dataset.num_nodes) {
        throw std::runtime_error("full-chain scope 只允许 full dataset/full graph，不能使用子图或 reduced 输入");
    }
    std::cout << "formal_full_chain_scope dataset=" << spec.dataset
              << " device=" << device
              << " stage=load_model\n";
    const auto model = load_model(project_root / "artifacts/checkpoints" / spec.checkpoint_name);
    auto status = write_full_chain_scope_status(spec, output_root, device, dataset, graph, model);
    std::cout << "formal_full_chain_scope dataset=" << spec.dataset
              << " device=" << device
              << " status=" << (status.official_ready ? "official_ready" : "blocked")
              << " hidden_heads=" << status.hidden_head_count
              << " output_heads=" << status.output_head_count
              << " missing_clause_count=" << status.missing_clauses.size()
              << " evidence=" << status.evidence_path.string() << "\n";
    return status;
}
