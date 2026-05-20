#pragma once

// CPU formal runner 只负责加载全量数据、构造 CPU trace 并调用 CPU-only backend。

void run_cpu_formal_chain(
    const fs::path& project_root,
    const fs::path& output_root,
    const DatasetRunSpec& spec,
    std::size_t requested_commitment_block_size)
{
    const auto run_dir = device_run_dir(output_root, spec.dataset, "cpu");
    ensure_dir(run_dir / "formal_artifacts");
    std::ofstream log(run_dir / "cpu_formal_chain.log");
    if (!log) throw std::runtime_error("failed to write CPU formal chain log");
    auto log_line = [&](const std::string& line) {
        log << line << '\n';
        log.flush();
        std::cout << line << '\n';
    };

    log_line("cpu_formal_chain dataset=" + spec.dataset + " stage=load_dataset");
    const auto dataset = load_cached_dataset(project_root / "data/cache" / spec.cache_name);
    log_line("cpu_formal_chain dataset=" + spec.dataset
        + " loaded_nodes=" + std::to_string(dataset.num_nodes)
        + " loaded_edges=" + std::to_string(dataset.edges.size())
        + " features=" + std::to_string(dataset.num_features)
        + " classes=" + std::to_string(dataset.num_classes));

    log_line("cpu_formal_chain dataset=" + spec.dataset + " stage=normalize_graph");
    const auto graph = normalize_graph(dataset, spec);
    const bool full_dataset = graph.num_nodes == dataset.num_nodes;
    log_line("cpu_formal_chain dataset=" + spec.dataset
        + " full_dataset=" + std::string(full_dataset ? "true" : "false")
        + " normalized_nodes=" + std::to_string(graph.num_nodes)
        + " normalized_edges=" + std::to_string(graph.edges.size())
        + " graphs=" + std::to_string(graph.graph_count));
    if (!full_dataset) {
        throw std::runtime_error("CPU formal chain requires full dataset");
    }

    log_line("cpu_formal_chain dataset=" + spec.dataset + " stage=load_model");
    const auto model = load_model(project_root / "artifacts/checkpoints" / spec.checkpoint_name);
    log_line("cpu_formal_chain dataset=" + spec.dataset
        + " model=" + model.model_arch_id
        + " L=" + std::to_string(model.L)
        + " K_out=" + std::to_string(model.K_out)
        + " C=" + std::to_string(model.C));

    try {
        const auto blocker_path = run_dir / "formal_artifacts" / "zkdl_cpu_formal_chain.blocker.txt";
        if (fs::exists(blocker_path)) {
            fs::remove(blocker_path);
        }
        log_line("cpu_formal_chain dataset=" + spec.dataset + " stage=cpu_forward_trace");
        ScopedTimer forward_timer;
        const HeadParams* hidden0 = nullptr;
        for (const auto& head : model.hidden_heads) {
            if (head.layer_index == 0 && head.head_index == 0) {
                hidden0 = &head;
                break;
            }
        }
        if (hidden0 == nullptr) {
            throw std::runtime_error("CPU formal relation runner requires hidden layer0/head0");
        }
        const auto head_trace = run_attention_head_cpu(
            graph.features, graph.num_nodes, graph.num_features, *hidden0, graph, false, true);
        log_line("cpu_formal_chain dataset=" + spec.dataset
            + " cpu_forward_ms=" + format_ms(forward_timer.elapsed_ms())
            + " stage=cpu_relation_prove_verify");

        const auto result = run_cpu_formal_relations_with_trace(
            spec, graph, head_trace, output_root, requested_commitment_block_size);
        {
            std::ofstream debug(run_dir / "proof_debug.log");
            debug << "dataset=" << spec.dataset << " device=cpu status=partial_formal official=blocked\n";
            debug << "backend=cpu_only_host_fr_g1_commitment\n";
            debug << "nodes=" << graph.num_nodes << " edges=" << graph.edges.size()
                  << " topology_hash=" << result.topology_hash
                  << " commitment_block_size=" << result.commitment_block_size << "\n";
            debug << "commitment_ms=" << format_ms(result.commitment_ms)
                  << " prove_ms=" << format_ms(result.prove_ms)
                  << " verify_ms=" << format_ms(result.verify_ms)
                  << " proof_size_bytes=" << result.proof_size_bytes << "\n";
            for (const auto& detail : result.relation_details) {
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
            resume << "- commitment block size: `" << result.commitment_block_size << "`；用于减少 host G1 opening replay 的外层承诺行数。\n";
            resume << "- 已完成 relation: source/destination sparse routing、denominator grouped sum、inverse soundness、inverse destination routing、alpha from inverse 与 alpha*denominator consistency。\n";
            resume << "- 证明范围: hidden layer0/head0 full graph / full edge set。\n";
            resume << "- proof artifact: `" << result.proof_path.string() << "`\n";
            resume << "- evidence: `" << result.evidence_path.string() << "`\n";
            resume << "- runner log: `" << (run_dir / "cpu_formal_chain.log").string() << "`\n";
            resume << "- Level A blocker: max/range、ExpMap lookup/range、LeakyReLU/ELU lookup-range、output aggregation、rescale/rounding 和所有层/头 full proof object 仍未闭合。\n";
        }
        log_line("cpu_formal_chain dataset=" + spec.dataset
            + " status=PARTIAL_FORMAL"
            + " backend=cpu_only_host_fr_g1_commitment"
            + " commitment_ms=" + format_ms(result.commitment_ms)
            + " prove_ms=" + format_ms(result.prove_ms)
            + " verify_ms=" + format_ms(result.verify_ms)
            + " proof_size_bytes=" + std::to_string(result.proof_size_bytes)
            + " artifact=" + result.proof_path.string());
    } catch (const std::exception& error) {
        const auto artifact_path = run_dir / "formal_artifacts" / "zkdl_cpu_formal_chain.blocker.txt";
        std::ofstream artifact(artifact_path);
        if (!artifact) throw std::runtime_error("failed to write CPU formal-chain blocker artifact");
        artifact << "ZKDLGATCPU\n";
        artifact << "version=2\n";
        artifact << "dataset=" << spec.dataset << "\n";
        artifact << "status=failed_with_evidence\n";
        artifact << "full_dataset=true\n";
        artifact << "node_count=" << graph.num_nodes << "\n";
        artifact << "edge_count=" << graph.edges.size() << "\n";
        artifact << "checkpoint=" << spec.checkpoint_name << "\n";
        artifact << "backend=cpu_only_host_fr_g1_commitment\n";
        artifact << "requested_commitment_block_size=" << requested_commitment_block_size << "\n";
        artifact << "blocker=" << error.what() << "\n";
        artifact << "attempted_files=gat_modules/cpu_forward.cuh;gat_modules/cpu_formal_backend.cuh;gat_modules/cpu_formal_runner.cuh\n";
        artifact << "proof_generation=false\n";
        artifact << "verification=false\n";
        artifact << "note=本 artifact 是 CPU-only backend 实现尝试后的最小 failure evidence，不是 proof，不含 witness、完整 tensor 或 debug dump。\n";

        std::ofstream resume(run_dir / "RESUME_POINT.md");
        resume << "# zkDL-GAT CPU 继续点\n\n";
        resume << "- 数据集: `" << spec.dataset << "`\n";
        resume << "- 当前状态: `failed_with_evidence`。\n";
        resume << "- 已完成动作: 加载 full dataset cache、规范化公开拓扑、加载 checkpoint bundle，并尝试 CPU-only formal relation prover/verifier。\n";
        resume << "- failure artifact: `" << artifact_path.string() << "`\n";
        resume << "- runner log: `" << (run_dir / "cpu_formal_chain.log").string() << "`\n";
        resume << "- blocker: `" << error.what() << "`\n";
        log_line("cpu_formal_chain dataset=" + spec.dataset
            + " status=FAILED_WITH_EVIDENCE proof_generation=false verification=false"
            + " artifact=" + artifact_path.string()
            + " reason=\"" + std::string(error.what()) + "\"");
    }
}
