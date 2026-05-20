#pragma once

// GPU formal runner 只负责加载全量数据、构造 GPU trace 并调用 relation gadget。

void run_formal_sparse_grouped_gadget(
    const fs::path& project_root,
    const fs::path& output_root,
    const DatasetRunSpec& spec,
    std::size_t requested_commitment_block_size)
{
    const auto run_dir = device_run_dir(output_root, spec.dataset);
    ensure_dir(run_dir);
    std::cout << "formal_sparse_grouped dataset=" << spec.dataset << " stage=load_dataset\n";
    const auto dataset = load_cached_dataset(project_root / "data/cache" / spec.cache_name);
    std::cout << "formal_sparse_grouped dataset=" << spec.dataset << " stage=normalize_graph\n";
    const auto graph = normalize_graph(dataset, spec);
    std::cout << "formal_sparse_grouped dataset=" << spec.dataset
              << " nodes=" << graph.num_nodes
              << " edges=" << graph.edges.size()
              << " full_dataset=" << (graph.num_nodes == dataset.num_nodes ? "true" : "false")
              << "\n";
    if (graph.num_nodes != dataset.num_nodes) {
        throw std::runtime_error("formal sparse/grouped gadget 只允许 full dataset/full graph");
    }
    std::cout << "formal_sparse_grouped dataset=" << spec.dataset << " stage=load_model\n";
    const auto model = load_model(project_root / "artifacts/checkpoints" / spec.checkpoint_name);
    std::cout << "formal_sparse_grouped dataset=" << spec.dataset << " stage=gpu_forward_trace\n";
    ForwardMetrics forward;
    ForwardTraceHost trace;
    (void)run_gat_forward_gpu(graph, model, &forward, &trace);
    std::cout << "formal_sparse_grouped dataset=" << spec.dataset
              << " forward_ms=" << format_ms(forward.total_ms)
              << " stage=prove_verify\n";
    const auto result = run_sparse_grouped_relations_with_trace(
        spec, graph, trace, output_root, requested_commitment_block_size);
    std::cout << "formal_sparse_grouped dataset=" << spec.dataset
              << " status=PASS"
              << " commitment_block_size=" << result.commitment_block_size
              << " commitment_ms=" << format_ms(result.commitment_ms)
              << " prove_ms=" << format_ms(result.prove_ms)
              << " verify_ms=" << format_ms(result.verify_ms)
              << " proof_size_bytes=" << result.proof_size_bytes
              << " proof_path=" << result.proof_path.string()
              << "\n";
}

void run_formal_softmax_aggregation_gadget(
    const fs::path& project_root,
    const fs::path& output_root,
    const DatasetRunSpec& spec,
    std::size_t max_feature_columns,
    std::size_t requested_commitment_block_size)
{
    const auto run_dir = device_run_dir(output_root, spec.dataset);
    ensure_dir(run_dir);
    std::cout << "formal_softmax_aggregation dataset=" << spec.dataset << " stage=load_dataset\n";
    const auto dataset = load_cached_dataset(project_root / "data/cache" / spec.cache_name);
    std::cout << "formal_softmax_aggregation dataset=" << spec.dataset << " stage=normalize_graph\n";
    const auto graph = normalize_graph(dataset, spec);
    std::cout << "formal_softmax_aggregation dataset=" << spec.dataset
              << " nodes=" << graph.num_nodes
              << " edges=" << graph.edges.size()
              << " full_dataset=" << (graph.num_nodes == dataset.num_nodes ? "true" : "false")
              << "\n";
    if (graph.num_nodes != dataset.num_nodes) {
        throw std::runtime_error("formal softmax/aggregation gadget 只允许 full dataset/full graph");
    }
    std::cout << "formal_softmax_aggregation dataset=" << spec.dataset << " stage=load_model\n";
    const auto model = load_model(project_root / "artifacts/checkpoints" / spec.checkpoint_name);
    std::cout << "formal_softmax_aggregation dataset=" << spec.dataset << " stage=gpu_forward_trace\n";
    ForwardMetrics forward;
    ForwardTraceHost trace;
    (void)run_gat_forward_gpu(graph, model, &forward, &trace);
    std::cout << "formal_softmax_aggregation dataset=" << spec.dataset
              << " forward_ms=" << format_ms(forward.total_ms)
              << " stage=prove_verify\n";
    const auto result = run_softmax_aggregation_relations_with_trace(
        spec, graph, trace, output_root, max_feature_columns, requested_commitment_block_size);
    std::cout << "formal_softmax_aggregation dataset=" << spec.dataset
              << " status=" << (result.weighted_aggregation_ok ? "PASS" : "PARTIAL")
              << " commitment_block_size=" << result.commitment_block_size
              << " denominator=" << (result.denominator_ok ? "PASS" : "FAIL")
              << " inverse=" << (result.inverse_ok ? "PASS" : "FAIL")
              << " alpha=" << (result.alpha_from_inverse_ok && result.alpha_times_denominator_ok ? "PASS" : "FAIL")
              << " weighted_columns=" << result.verified_feature_columns << "/" << result.available_feature_columns
              << " executed_feature_columns=" << result.feature_columns
              << " commitment_ms=" << format_ms(result.commitment_ms)
              << " prove_ms=" << format_ms(result.prove_ms)
              << " verify_ms=" << format_ms(result.verify_ms)
              << " proof_size_bytes=" << result.proof_size_bytes
              << " proof_path=" << result.proof_path.string()
              << "\n";
}
