#pragma once

// 单数据集 runner、旧 matmul 调度入口与 CLI 调度辅助。

RunRow run_one(const fs::path& project_root, const fs::path& output_root, const DatasetRunSpec& spec) {
    RunRow row;
    row.dataset = spec.dataset;
    const auto run_dir = device_run_dir(output_root, spec.dataset);
    ensure_dir(run_dir);
    fs::remove_all(run_dir / "proof");
    fs::remove_all(run_dir / "formal_artifacts");
    std::ofstream log(run_dir / "run.log");
    auto log_line = [&](const std::string& line) {
        log << line << '\n';
        log.flush();
        std::cout << line << '\n';
    };

    try {
        log_line("dataset=" + spec.dataset + " stage=load_dataset");
        ScopedTimer load_timer;
        const auto dataset = load_cached_dataset(project_root / "data/cache" / spec.cache_name);
        log_line("dataset=" + spec.dataset + " loaded_nodes=" + std::to_string(dataset.num_nodes)
            + " loaded_edges=" + std::to_string(dataset.edges.size())
            + " features=" + std::to_string(dataset.num_features)
            + " classes=" + std::to_string(dataset.num_classes)
            + " load_ms=" + format_ms(load_timer.elapsed_ms()));

        log_line("dataset=" + spec.dataset + " stage=normalize_graph");
        const auto graph = normalize_graph(dataset, spec);
        row.full_dataset = graph.num_nodes == dataset.num_nodes;
        log_line("dataset=" + spec.dataset + " normalized_nodes=" + std::to_string(graph.num_nodes)
            + " normalized_edges=" + std::to_string(graph.edges.size())
            + " graphs=" + std::to_string(graph.graph_count));

        log_line("dataset=" + spec.dataset + " stage=load_model");
        const auto model = load_model(project_root / "artifacts/checkpoints" / spec.checkpoint_name);
        if (model.output_average_rule != "per_head_bias_then_arithmetic_mean") {
            throw std::runtime_error("unsupported output average rule: " + model.output_average_rule);
        }
        std::ostringstream model_line;
        model_line << "dataset=" << spec.dataset
                   << " model=" << model.model_arch_id
                   << " L=" << model.L
                   << " K_out=" << model.K_out
                   << " C=" << model.C
                   << " hidden_profile=";
        for (std::size_t i = 0; i < model.hidden_profile.size(); ++i) {
            if (i != 0) model_line << ';';
            model_line << model.hidden_profile[i].first << 'x' << model.hidden_profile[i].second;
        }
        log_line(model_line.str());

        log_line("dataset=" + spec.dataset + " stage=gpu_forward");
        ForwardMetrics forward;
        ForwardTraceHost trace;
        const auto logits = run_gat_forward_gpu(graph, model, &forward, &trace);
        row.gpu_forward_ok = true;
        log_line("dataset=" + spec.dataset + " gpu_forward_ms=" + format_ms(forward.total_ms)
            + " logits_values=" + std::to_string(logits.size()));

        log_line("dataset=" + spec.dataset + " stage=zkdl_level_b_output_binding_proof");
        ProofAttempt proof;
        proof.level_b = run_level_b_output_binding_proof(spec, model, graph, trace, run_dir);
        row.partial_commitment_ok = proof.level_b.commitment_ms > 0.0;
        row.debug_level_b_status = proof.level_b.verification_ok ? "success_debug_only" : (proof.level_b.proof_generation_ok ? "blocked_debug" : "failed_debug");
        row.debug_level_b_blocker = proof.level_b.blocker;
        row.debug_level_b_benchmark_mode = proof.level_b.benchmark_mode;
        row.debug_level_b_commitment_time_ms = proof.level_b.proof_generation_ok ? format_ms(proof.level_b.commitment_ms) : "NA";
        row.debug_level_b_prove_time_ms = proof.level_b.proof_generation_ok ? format_ms(proof.level_b.prove_ms) : "NA";
        row.debug_level_b_verify_time_ms = proof.level_b.verification_ok ? format_ms(proof.level_b.verify_ms) : "NA";
        row.debug_level_b_proof_size_bytes = proof.level_b.proof_size_bytes != 0 ? std::to_string(proof.level_b.proof_size_bytes) : "NA";
        row.debug_level_b_proof_path = proof.level_b.proof_path.empty() ? "" : proof.level_b.proof_path.string();

        log_line("dataset=" + spec.dataset + " stage=formal_partial_trace_relations");
        proof.formal_partial = run_formal_partial_trace_proof(spec, model, graph, trace, run_dir);
        auto join_strings = [](const std::vector<std::string>& values) {
            std::ostringstream out;
            for (std::size_t i = 0; i < values.size(); ++i) {
                if (i != 0) out << ';';
                out << values[i];
            }
            return out.str();
        };
        row.formal_partial_status = proof.formal_partial.verification_ok ? "partial_formal" : "blocked";
        row.formal_partial_blocker = proof.formal_partial.blocker;
        row.formal_partial_commitment_time_ms = proof.formal_partial.proof_generation_ok ? format_ms(proof.formal_partial.commitment_ms) : "NA";
        row.formal_partial_prove_time_ms = proof.formal_partial.proof_generation_ok ? format_ms(proof.formal_partial.prove_ms) : "NA";
        row.formal_partial_verify_time_ms = proof.formal_partial.verification_ok ? format_ms(proof.formal_partial.verify_ms) : "NA";
        row.formal_partial_proof_size_bytes = proof.formal_partial.proof_size_bytes != 0 ? std::to_string(proof.formal_partial.proof_size_bytes) : "NA";
        row.formal_partial_proof_path = proof.formal_partial.proof_path.empty() ? "" : proof.formal_partial.proof_path.string();
        row.formal_partial_total_chunks = std::to_string(proof.formal_partial.total_chunks);
        row.formal_partial_verified_chunks = std::to_string(proof.formal_partial.verified_chunks);
        row.formal_partial_covered_relations = join_strings(proof.formal_partial.covered_relations);
        row.formal_partial_blocked_relations = join_strings(proof.formal_partial.blocked_relations);

        row.proof_generation_ok = proof.formal_partial.proof_generation_ok;
        row.verification_ok = proof.formal_partial.verification_ok;
        row.benchmark_mode = "warm";
        row.commitment_time_ms = "NA";
        row.prove_time_ms = "NA";
        row.verify_time_ms = "NA";
        row.proof_size_bytes = "NA";
        row.status = "blocked";
        row.blocker =
            "zkDL-GAT Level A proof 尚未完成。"
            + row.formal_partial_blocker
            + " official 四项保持 NA；partial relation timing 与 proof_size 只写入 formal_partial_debug.*。";
        log_line("dataset=" + spec.dataset
            + " commitment_status=" + (row.partial_commitment_ok ? "ok" : "failed")
            + " debug_level_b_status=" + row.debug_level_b_status
            + " debug_level_b_proof_size_bytes=" + row.debug_level_b_proof_size_bytes
            + " formal_partial_status=" + row.formal_partial_status
            + " formal_partial_proof_size_bytes=" + row.formal_partial_proof_size_bytes
            + " formal_partial_chunks=" + row.formal_partial_verified_chunks + "/" + row.formal_partial_total_chunks
            + " formal_status=" + row.status
            + " failure=\"" + row.blocker + "\"");
        log_line("dataset=" + spec.dataset
            + " commitment_time_ms=" + row.commitment_time_ms
            + " prove_time_ms=" + row.prove_time_ms
            + " verify_time_ms=" + row.verify_time_ms);
        log_line("dataset=" + spec.dataset
            + " debug_level_b_commitment_time_ms=" + row.debug_level_b_commitment_time_ms
            + " debug_level_b_prove_time_ms=" + row.debug_level_b_prove_time_ms
            + " debug_level_b_verify_time_ms=" + row.debug_level_b_verify_time_ms
            + " debug_level_b_proof_path=\"" + row.debug_level_b_proof_path + "\"");
        log_line("dataset=" + spec.dataset
            + " formal_partial_commitment_time_ms=" + row.formal_partial_commitment_time_ms
            + " formal_partial_prove_time_ms=" + row.formal_partial_prove_time_ms
            + " formal_partial_verify_time_ms=" + row.formal_partial_verify_time_ms
            + " formal_partial_proof_path=\"" + row.formal_partial_proof_path + "\"");
        write_manifest(run_dir, spec, dataset, graph, model, forward, proof, row);
    } catch (const std::exception& error) {
        row.status = "failed";
        row.blocker = error.what();
        log_line("dataset=" + spec.dataset + " status=FAILED reason=\"" + row.blocker + "\"");
        GraphDataset empty_dataset;
        empty_dataset.name = spec.dataset;
        LocalGraph empty_graph;
        ModelParams empty_model;
        ForwardMetrics empty_forward;
        ProofAttempt empty_proof;
        write_manifest(run_dir, spec, empty_dataset, empty_graph, empty_model, empty_forward, empty_proof, row);
    }

    std::cout
        << "dataset=" << row.dataset
        << " config=configs/" << spec.config_name
        << " backend=gpu"
        << " status=" << row.status
        << " commitment_time_ms=" << row.commitment_time_ms
        << " prove_time_ms=" << row.prove_time_ms
        << " verify_time_ms=" << row.verify_time_ms
        << " proof_size_bytes=" << row.proof_size_bytes
        << '\n';
    return row;
}

void run_cpu_backend_probe(
    const fs::path& project_root,
    const fs::path& output_root,
    const DatasetRunSpec& spec)
{
    const auto run_dir = device_run_dir(output_root, spec.dataset, "cpu");
    ensure_dir(run_dir / "formal_artifacts");
    std::ofstream log(run_dir / "cpu_backend_probe.log");
    if (!log) throw std::runtime_error("failed to write CPU backend probe log");
    auto log_line = [&](const std::string& line) {
        log << line << '\n';
        log.flush();
        std::cout << line << '\n';
    };

    log_line("cpu_backend_probe dataset=" + spec.dataset + " stage=load_dataset");
    const auto dataset = load_cached_dataset(project_root / "data/cache" / spec.cache_name);
    log_line("cpu_backend_probe dataset=" + spec.dataset
        + " loaded_nodes=" + std::to_string(dataset.num_nodes)
        + " loaded_edges=" + std::to_string(dataset.edges.size())
        + " features=" + std::to_string(dataset.num_features)
        + " classes=" + std::to_string(dataset.num_classes));

    log_line("cpu_backend_probe dataset=" + spec.dataset + " stage=normalize_graph");
    const auto graph = normalize_graph(dataset, spec);
    const bool full_dataset = graph.num_nodes == dataset.num_nodes;
    log_line("cpu_backend_probe dataset=" + spec.dataset
        + " full_dataset=" + std::string(full_dataset ? "true" : "false")
        + " normalized_nodes=" + std::to_string(graph.num_nodes)
        + " normalized_edges=" + std::to_string(graph.edges.size())
        + " graphs=" + std::to_string(graph.graph_count));
    if (!full_dataset) {
        throw std::runtime_error("CPU backend probe requires full dataset");
    }

    log_line("cpu_backend_probe dataset=" + spec.dataset + " stage=load_model");
    const auto model = load_model(project_root / "artifacts/checkpoints" / spec.checkpoint_name);
    log_line("cpu_backend_probe dataset=" + spec.dataset
        + " model=" + model.model_arch_id
        + " L=" + std::to_string(model.L)
        + " K_out=" + std::to_string(model.K_out)
        + " C=" + std::to_string(model.C));

    try {
        const auto blocker_path = run_dir / "formal_artifacts" / "zkdl_cpu_backend_probe.blocker.txt";
        if (fs::exists(blocker_path)) {
            fs::remove(blocker_path);
        }
        log_line("cpu_backend_probe dataset=" + spec.dataset + " stage=cpu_forward_trace");
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
        log_line("cpu_backend_probe dataset=" + spec.dataset
            + " cpu_forward_ms=" + format_ms(forward_timer.elapsed_ms())
            + " stage=cpu_relation_prove_verify");

        const auto result = run_cpu_formal_relations_with_trace(spec, graph, head_trace, output_root);
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
            debug << "note=CPU relation-level partial formal only; not official Level A\n";
        }
        {
            std::ofstream resume(run_dir / "RESUME_POINT.md");
            resume << "# zkDL-GAT CPU 继续点\n\n";
            resume << "- 数据集: `" << spec.dataset << "`\n";
            resume << "- 当前状态: `partial_formal`，official 仍为 `blocked + NA`。\n";
            resume << "- CPU backend: `CpuFrTensor` + host `G1Jacobian` + `CpuCommitment::commit/open/verify_opening`。\n";
            resume << "- 已完成 relation: source/destination sparse routing 与 grouped sum。\n";
            resume << "- 证明范围: hidden layer0/head0 full graph / full edge set。\n";
            resume << "- proof artifact: `" << result.proof_path.string() << "`\n";
            resume << "- evidence: `" << result.evidence_path.string() << "`\n";
            resume << "- probe log: `" << (run_dir / "cpu_backend_probe.log").string() << "`\n";
            resume << "- Level A blocker: CPU denominator/inverse/alpha extension、max/range、ExpMap lookup/range、LeakyReLU/ELU lookup-range、output aggregation、rescale/rounding 和所有层/头 full proof object 仍未闭合。\n";
        }
        log_line("cpu_backend_probe dataset=" + spec.dataset
            + " status=PARTIAL_FORMAL"
            + " backend=cpu_only_host_fr_g1_commitment"
            + " commitment_ms=" + format_ms(result.commitment_ms)
            + " prove_ms=" + format_ms(result.prove_ms)
            + " verify_ms=" + format_ms(result.verify_ms)
            + " proof_size_bytes=" + std::to_string(result.proof_size_bytes)
            + " artifact=" + result.proof_path.string());
    } catch (const std::exception& error) {
        const auto artifact_path = run_dir / "formal_artifacts" / "zkdl_cpu_backend_attempt.blocker.txt";
        std::ofstream artifact(artifact_path);
        if (!artifact) throw std::runtime_error("failed to write CPU backend blocker artifact");
        artifact << "ZKDLGATCPU\n";
        artifact << "version=2\n";
        artifact << "dataset=" << spec.dataset << "\n";
        artifact << "status=blocked_after_cpu_backend_attempt\n";
        artifact << "full_dataset=true\n";
        artifact << "node_count=" << graph.num_nodes << "\n";
        artifact << "edge_count=" << graph.edges.size() << "\n";
        artifact << "checkpoint=" << spec.checkpoint_name << "\n";
        artifact << "backend=cpu_only_host_fr_g1_commitment\n";
        artifact << "blocker=" << error.what() << "\n";
        artifact << "attempted_files=gat_modules/cpu_forward.cuh;gat_modules/cpu_formal_backend.cuh;gat_modules/run_dispatch.cuh\n";
        artifact << "proof_generation=false\n";
        artifact << "verification=false\n";
        artifact << "note=本 artifact 是 CPU-only backend 实现尝试后的最小 failure evidence，不是 proof，不含 witness、完整 tensor 或 debug dump。\n";

        std::ofstream resume(run_dir / "RESUME_POINT.md");
        resume << "# zkDL-GAT CPU 继续点\n\n";
        resume << "- 数据集: `" << spec.dataset << "`\n";
        resume << "- 当前状态: `blocked_after_cpu_backend_attempt`。\n";
        resume << "- 已完成动作: 加载 full dataset cache、规范化公开拓扑、加载 checkpoint bundle，并尝试 CPU-only formal relation prover/verifier。\n";
        resume << "- failure artifact: `" << artifact_path.string() << "`\n";
        resume << "- probe log: `" << (run_dir / "cpu_backend_probe.log").string() << "`\n";
        resume << "- blocker: `" << error.what() << "`\n";
        log_line("cpu_backend_probe dataset=" + spec.dataset
            + " status=BLOCKED_AFTER_ATTEMPT proof_generation=false verification=false"
            + " artifact=" + artifact_path.string()
            + " reason=\"" + std::string(error.what()) + "\"");
    }
}

struct FormalMatmulScheduleInputs {
    std::vector<double> a_values;
    std::vector<double> b_values;
    std::vector<double> c_values;
    std::size_t rows = 0;
    std::size_t shared = 0;
    std::size_t cols = 0;
    std::string label;
};

struct FormalMatmulScheduleJob {
    std::string relation;
    std::size_t start_chunk = 0;
    std::size_t max_chunks = 1;
};

FormalMatmulScheduleInputs resolve_formal_matmul_schedule_inputs(
    const LocalGraph& graph,
    const ModelParams& model,
    const ForwardTraceHost& trace,
    const std::string& relation)
{
    FormalMatmulScheduleInputs inputs;
    inputs.rows = graph.num_nodes;
    inputs.label = relation;
    std::smatch match;
    const std::regex output_relation(R"(output_head([0-9]+)_(projection|source_attention_dot|destination_attention_dot))");
    const std::regex hidden_relation(R"(hidden_layer([0-9]+)_head([0-9]+)_(projection|source_attention_dot|destination_attention_dot))");
    if (std::regex_match(relation, match, output_relation)) {
        const std::size_t head_index = static_cast<std::size_t>(std::stoull(match[1].str()));
        const std::string kind = match[2].str();
        if (head_index >= model.output_heads.size() || head_index >= trace.output_heads.size()) {
            throw std::runtime_error("output head index out of range for schedule relation: " + relation);
        }
        const auto& output_model = model.output_heads[head_index];
        const auto& output_trace = trace.output_heads[head_index];
        if (kind == "projection") {
            if (trace.output_layer_input.empty()) {
                throw std::runtime_error("output projection schedule requires output layer input trace");
            }
            inputs.a_values = trace.output_layer_input;
            inputs.shared = trace.output_layer_input_dim;
            inputs.b_values = output_model.W;
            inputs.cols = output_trace.width;
            inputs.c_values = output_trace.h_prime;
        } else {
            inputs.a_values = output_trace.h_prime;
            inputs.shared = output_trace.width;
            inputs.cols = 1;
            const bool source_dot = kind == "source_attention_dot";
            inputs.b_values = source_dot ? output_model.attn_src : output_model.attn_dst;
            const auto& e_values = source_dot ? output_trace.e_src : output_trace.e_dst;
            const double bias = source_dot ? output_model.attn_src_bias : output_model.attn_dst_bias;
            inputs.c_values.resize(e_values.size(), 0.0);
            for (std::size_t i = 0; i < inputs.c_values.size(); ++i) {
                inputs.c_values[i] = e_values[i] - bias;
            }
        }
    } else if (std::regex_match(relation, match, hidden_relation)) {
        const std::size_t layer_index = static_cast<std::size_t>(std::stoull(match[1].str()));
        const std::size_t head_index = static_cast<std::size_t>(std::stoull(match[2].str()));
        const std::string kind = match[3].str();
        if (layer_index >= trace.hidden_heads.size()
            || layer_index >= trace.hidden_layer_inputs.size()
            || layer_index >= trace.hidden_layer_input_dims.size()) {
            throw std::runtime_error("hidden layer index out of range for schedule relation: " + relation);
        }
        const HeadParams* hidden_model = nullptr;
        std::size_t trace_head_ordinal = 0;
        std::size_t layer_head_ordinal = 0;
        for (const auto& head : model.hidden_heads) {
            if (head.layer_index != layer_index) {
                continue;
            }
            if (head.head_index == head_index) {
                hidden_model = &head;
                trace_head_ordinal = layer_head_ordinal;
                break;
            }
            ++layer_head_ordinal;
        }
        if (hidden_model == nullptr || trace_head_ordinal >= trace.hidden_heads[layer_index].size()) {
            throw std::runtime_error("hidden head model/trace not found for schedule relation: " + relation);
        }
        const auto& hidden_trace = trace.hidden_heads[layer_index][trace_head_ordinal];
        inputs.rows = hidden_trace.rows;
        if (kind == "projection") {
            inputs.a_values = trace.hidden_layer_inputs[layer_index];
            inputs.shared = trace.hidden_layer_input_dims[layer_index];
            inputs.b_values = hidden_model->W;
            inputs.cols = hidden_trace.width;
            inputs.c_values = hidden_trace.h_prime;
        } else {
            inputs.a_values = hidden_trace.h_prime;
            inputs.shared = hidden_trace.width;
            inputs.cols = 1;
            const bool source_dot = kind == "source_attention_dot";
            inputs.b_values = source_dot ? hidden_model->attn_src : hidden_model->attn_dst;
            const auto& e_values = source_dot ? hidden_trace.e_src : hidden_trace.e_dst;
            const double bias = source_dot ? hidden_model->attn_src_bias : hidden_model->attn_dst_bias;
            inputs.c_values.resize(e_values.size(), 0.0);
            for (std::size_t i = 0; i < inputs.c_values.size(); ++i) {
                inputs.c_values[i] = e_values[i] - bias;
            }
        }
    } else {
        throw std::runtime_error("unsupported formal matmul schedule relation: " + relation);
    }
    if (inputs.a_values.size() != inputs.rows * inputs.shared
        || inputs.b_values.size() != inputs.shared * inputs.cols
        || inputs.c_values.size() != inputs.rows * inputs.cols) {
        throw std::runtime_error("formal matmul schedule shape mismatch for " + relation);
    }
    return inputs;
}

void run_formal_matmul_schedule_with_trace(
    const fs::path& output_root,
    const DatasetRunSpec& spec,
    const LocalGraph& graph,
    const ModelParams& model,
    const ForwardTraceHost& trace,
    const std::string& relation,
    std::size_t start_chunk,
    std::size_t max_chunks)
{
    const auto inputs = resolve_formal_matmul_schedule_inputs(graph, model, trace, relation);
    const std::size_t chunk_rows = matmul_chunk_rows(inputs.shared, inputs.cols);
    const std::size_t total_chunks = (inputs.rows + chunk_rows - 1U) / chunk_rows;
    if (start_chunk >= total_chunks) {
        throw std::runtime_error("start chunk is beyond total chunks for " + relation);
    }
    const std::size_t stop_chunk = std::min(total_chunks, start_chunk + std::max<std::size_t>(1, max_chunks));
    const auto run_dir = device_run_dir(output_root, spec.dataset);
    ensure_dir(run_dir / "formal_artifacts");
    std::ofstream debug(run_dir / "proof_debug.log", std::ios::app);
    auto dbg = [&](const std::string& line) {
        if (debug) {
            debug << line << '\n';
            debug.flush();
        }
        std::cout << line << '\n';
    };
    FormalPartialArtifact artifact;
    dbg("formal_matmul_schedule dataset=" + spec.dataset
        + " relation=" + relation
        + " rows=" + std::to_string(inputs.rows)
        + " shared=" + std::to_string(inputs.shared)
        + " cols=" + std::to_string(inputs.cols)
        + " start_chunk=" + std::to_string(start_chunk)
        + " stop_chunk=" + std::to_string(stop_chunk)
        + " total_chunks=" + std::to_string(total_chunks));

    std::size_t completed = 0;
    std::size_t next_chunk = start_chunk;
    for (std::size_t chunk_idx = start_chunk; chunk_idx < stop_chunk; ++chunk_idx) {
        const std::size_t row_begin = chunk_idx * chunk_rows;
        const std::size_t row_count = std::min(chunk_rows, inputs.rows - row_begin);
        const auto a_chunk = matrix_row_slice(inputs.a_values, inputs.rows, inputs.shared, row_begin, row_count);
        const auto c_chunk = matrix_row_slice(inputs.c_values, inputs.rows, inputs.cols, row_begin, row_count);
        const auto chunked_label = chunk_relation_label(inputs.label, chunk_idx, row_begin, row_count, total_chunks);
        ++artifact.total_chunks;
        const bool ok = run_formal_matmul_trace_attempt(
            spec.dataset,
            chunked_label,
            a_chunk,
            row_count,
            inputs.shared,
            inputs.b_values,
            inputs.cols,
            c_chunk,
            &artifact,
            &debug,
            row_begin,
            total_chunks);
        if (!ok) {
            artifact.relation_details.push_back(inputs.label + " schedule chunk=" + std::to_string(chunk_idx) + " status=REJECT_OR_BLOCKED");
            next_chunk = chunk_idx;
            break;
        }
        ++artifact.verified_chunks;
        ++completed;
        next_chunk = chunk_idx + 1;
        artifact.relation_details.push_back(inputs.label + " schedule chunk=" + std::to_string(chunk_idx) + " status=PASS");
    }
    artifact.proof_generation_ok = completed != 0;
    artifact.verification_ok = completed != 0;
    artifact.blocker = "targeted formal matmul schedule 已验证 "
        + std::to_string(completed)
        + " 个 matmul chunk，起始 chunk 为 "
        + std::to_string(start_chunk)
        + "；fixed-to-trace scale decomposition 与 residual bounded range 为 partial formal，完整 rounding mode、sign 与 tie-breaking 仍阻塞。";
    artifact.proof_path = run_dir / "formal_artifacts" / (
        "zkdl_formal_matmul_schedule_"
        + relation
        + "_chunks"
        + std::to_string(start_chunk)
        + "_"
        + std::to_string(stop_chunk - 1U)
        + ".proof.txt");
    serialize_formal_partial_artifact(artifact, artifact.proof_path);

    std::ofstream resume(output_root / "RESUME_POINT.md");
    resume << "# zkDL GAT formal matmul schedule 继续点\n\n";
    resume << "- 数据集: `" << spec.dataset << "`\n";
    resume << "- relation: `" << relation << "`\n";
    resume << "- 每个 chunk 的行数: `" << chunk_rows << "`\n";
    resume << "- matmul 总 chunk 数: `" << total_chunks << "`\n";
    resume << "- 本轮请求 chunk 范围: `" << start_chunk << ".." << stop_chunk << "`\n";
    resume << "- 本轮新增验证的 matmul chunk 数: `" << completed << "`\n";
    resume << "- 本轮新增验证的 formal relation chunk 数（含 scale decomposition 与 residual range bitness/decomposition）: `"
           << artifact.verified_chunks << "/" << artifact.total_chunks << "`\n";
    resume << "- 当前下一个 chunk: `" << (next_chunk >= total_chunks ? std::string("complete") : std::to_string(next_chunk)) << "`\n";
    resume << "- proof artifact: `" << artifact.proof_path.string() << "`\n";
    resume << "- proof debug log: `" << (run_dir / "proof_debug.log").string() << "`\n";
    resume << "- Level A 状态: blocked；exact scale decomposition 和 residual bounded range 已由 verifier 绑定到本轮 chunk，但 sign-specific rounding mode 与 tie-breaking 仍未证明。\n";

    std::ofstream dataset_resume(run_dir / "RESUME_POINT.md");
    dataset_resume << "# zkDL GAT 数据集继续点\n\n";
    dataset_resume << "- 数据集: `" << spec.dataset << "`\n";
    dataset_resume << "- relation: `" << relation << "`\n";
    dataset_resume << "- 每个 chunk 的行数: `" << chunk_rows << "`\n";
    dataset_resume << "- matmul 总 chunk 数: `" << total_chunks << "`\n";
    dataset_resume << "- 本轮请求 chunk 范围: `" << start_chunk << ".." << stop_chunk << "`\n";
    dataset_resume << "- 本轮新增验证的 matmul chunk 数: `" << completed << "`\n";
    dataset_resume << "- 本轮新增验证的 formal relation chunk 数（含 scale decomposition 与 residual range bitness/decomposition）: `"
                   << artifact.verified_chunks << "/" << artifact.total_chunks << "`\n";
    dataset_resume << "- 当前下一个 chunk: `" << (next_chunk >= total_chunks ? std::string("complete") : std::to_string(next_chunk)) << "`\n";
    dataset_resume << "- proof artifact: `" << artifact.proof_path.string() << "`\n";
    dataset_resume << "- proof debug log: `" << (run_dir / "proof_debug.log").string() << "`\n";
    dataset_resume << "- Level A 状态: blocked；exact scale decomposition 和 residual bounded range 已由 verifier 绑定到本轮 chunk，但 sign-specific rounding mode 与 tie-breaking 仍未证明。\n";
}

void run_formal_matmul_schedule(
    const fs::path& project_root,
    const fs::path& output_root,
    const DatasetRunSpec& spec,
    const std::string& relation,
    std::size_t start_chunk,
    std::size_t max_chunks)
{
    const auto dataset = load_cached_dataset(project_root / "data/cache" / spec.cache_name);
    const auto graph = normalize_graph(dataset, spec);
    const auto model = load_model(project_root / "artifacts/checkpoints" / spec.checkpoint_name);
    ForwardMetrics forward;
    ForwardTraceHost trace;
    (void)run_gat_forward_gpu(graph, model, &forward, &trace);
    run_formal_matmul_schedule_with_trace(output_root, spec, graph, model, trace, relation, start_chunk, max_chunks);
}

std::vector<FormalMatmulScheduleJob> read_formal_matmul_schedule_plan(const fs::path& path)
{
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open schedule plan: " + path.string());
    }
    std::vector<FormalMatmulScheduleJob> jobs;
    std::string line;
    while (std::getline(input, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (line == "relation,start_chunk,max_chunks") {
            continue;
        }
        std::vector<std::string> fields;
        std::stringstream stream(line);
        std::string field;
        while (std::getline(stream, field, ',')) {
            fields.push_back(trim(field));
        }
        if (fields.size() != 3) {
            throw std::runtime_error("invalid schedule plan row: " + line);
        }
        FormalMatmulScheduleJob job;
        job.relation = fields[0];
        job.start_chunk = static_cast<std::size_t>(std::stoull(fields[1]));
        job.max_chunks = static_cast<std::size_t>(std::stoull(fields[2]));
        jobs.push_back(std::move(job));
    }
    if (jobs.empty()) {
        throw std::runtime_error("schedule plan is empty: " + path.string());
    }
    return jobs;
}

void run_formal_matmul_schedule_batch(
    const fs::path& project_root,
    const fs::path& output_root,
    const DatasetRunSpec& spec,
    const std::vector<FormalMatmulScheduleJob>& jobs)
{
    const auto dataset = load_cached_dataset(project_root / "data/cache" / spec.cache_name);
    const auto graph = normalize_graph(dataset, spec);
    const auto model = load_model(project_root / "artifacts/checkpoints" / spec.checkpoint_name);
    ForwardMetrics forward;
    ForwardTraceHost trace;
    (void)run_gat_forward_gpu(graph, model, &forward, &trace);
    std::cout << "formal_matmul_schedule_batch dataset=" << spec.dataset
              << " jobs=" << jobs.size()
              << " trace_reused=true\n";
    for (const auto& job : jobs) {
        run_formal_matmul_schedule_with_trace(
            output_root,
            spec,
            graph,
            model,
            trace,
            job.relation,
            job.start_chunk,
            job.max_chunks);
    }
}

void run_matmul_first_failure_diagnostic(const fs::path& project_root, const DatasetRunSpec& spec)
{
    const fs::path debug_dir = "/home/pzh/project/comparison/baselines/zkdl/zkdl_gat_outputs/evidence/matmul_first_failure";
    ensure_dir(debug_dir);
    fs::remove(debug_dir / ("matmul_first_failure_" + spec.dataset + ".json"));
    fs::remove(debug_dir / ("matmul_first_failure_" + spec.dataset + ".md"));
    std::ofstream debug(debug_dir / ("matmul_first_failure_" + spec.dataset + ".diagnostic.log"));
    auto log_line = [&](const std::string& line) {
        debug << line << '\n';
        debug.flush();
        std::cout << line << '\n';
    };

    log_line("dataset=" + spec.dataset + " diagnostic=matmul_first_failure stage=load_dataset");
    const auto dataset = load_cached_dataset(project_root / "data/cache" / spec.cache_name);
    log_line("dataset=" + spec.dataset + " diagnostic=matmul_first_failure stage=normalize_graph");
    const auto graph = normalize_graph(dataset, spec);
    log_line("dataset=" + spec.dataset + " diagnostic=matmul_first_failure stage=load_model");
    const auto model = load_model(project_root / "artifacts/checkpoints" / spec.checkpoint_name);
    if (model.output_heads.empty()) {
        throw std::runtime_error("matmul diagnostic requires at least one output head");
    }
    log_line("dataset=" + spec.dataset + " diagnostic=matmul_first_failure stage=gpu_forward_full_trace");
    ForwardMetrics forward;
    ForwardTraceHost trace;
    (void)run_gat_forward_gpu(graph, model, &forward, &trace);
    if (trace.output_layer_input.empty() || trace.output_heads.empty() || trace.output_heads.front().h_prime.empty()) {
        throw std::runtime_error("matmul diagnostic output projection trace is unavailable");
    }

    const std::size_t rows = graph.num_nodes;
    const std::size_t shared = trace.output_layer_input_dim;
    const std::size_t cols = model.C;
    const std::size_t chunk_rows = matmul_chunk_rows(shared, cols);
    const std::size_t total_chunks = (rows + chunk_rows - 1U) / chunk_rows;
    const std::size_t row_count = std::min(chunk_rows, rows);
    const auto a_chunk = matrix_row_slice(trace.output_layer_input, rows, shared, 0, row_count);
    const auto c_chunk = matrix_row_slice(trace.output_heads.front().h_prime, rows, cols, 0, row_count);
    FormalPartialArtifact artifact;
    log_line("dataset=" + spec.dataset
        + " diagnostic=matmul_first_failure relation=output_head0_projection"
        + " chunk=0 rows=0.." + std::to_string(row_count)
        + " total_chunks=" + std::to_string(total_chunks));
    const bool ok = run_formal_matmul_trace_attempt(
        spec.dataset,
        chunk_relation_label("output_head0_projection", 0, 0, row_count, total_chunks),
        a_chunk,
        row_count,
        shared,
        model.output_heads.front().W,
        cols,
        c_chunk,
        &artifact,
        &debug,
        0,
        total_chunks);
    log_line("dataset=" + spec.dataset
        + " diagnostic=matmul_first_failure fixed_point_pre_rescale_matmul_status="
        + std::string(ok ? "PASS" : "FAILED_OR_BLOCKED")
        + " evidence_json=" + (debug_dir / ("matmul_first_failure_" + spec.dataset + ".json")).string());
}
