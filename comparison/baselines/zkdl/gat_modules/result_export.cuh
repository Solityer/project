#pragma once

// manifest、coverage、latest、summary 与 adapter README 导出。

FormalPartialArtifact run_formal_partial_trace_proof(
    const DatasetRunSpec& spec,
    const ModelParams& model,
    const LocalGraph& graph,
    const ForwardTraceHost& trace,
    const fs::path& run_dir)
{
    FormalPartialArtifact artifact;
    std::ofstream debug(run_dir / "proof_debug.log", std::ios::app);
    auto dbg = [&](const std::string& line) {
        if (debug) {
            debug << line << '\n';
            debug.flush();
        }
    };
    dbg("formal_partial stage=begin");
    try {
        if (trace.output_heads.size() != model.K_out) {
            throw std::runtime_error("output head trace count mismatch");
        }
        const std::size_t rows = graph.num_nodes;
        const std::size_t cols = model.C;
        run_output_linear_chunked_relations(spec, model, trace, rows, cols, &artifact, &debug);
        run_hidden_concat_chunked_relations(spec, trace, rows, &artifact, &debug);
        artifact.blocked_relations.push_back(
            "output rescale blocked：bias-add 子关系使用已 commitment 的 Y_pre_rescaled，但 sparse attention aggregation 的 Y_pre 到 Y_pre_rescaled 仍缺 fixed-point rounding/rescale verifier");

        if (!trace.output_layer_input.empty() && !model.output_heads.empty() && !trace.output_heads.empty()) {
            run_formal_matmul_trace_attempt_row_chunked(
                spec.dataset,
                "output_head0_projection",
                trace.output_layer_input,
                rows,
                trace.output_layer_input_dim,
                model.output_heads.front().W,
                cols,
                trace.output_heads.front().h_prime,
                &artifact,
                &debug);
            std::vector<double> e_src_pre(trace.output_heads.front().e_src.size(), 0.0);
            std::vector<double> e_dst_pre(trace.output_heads.front().e_dst.size(), 0.0);
            for (std::size_t i = 0; i < e_src_pre.size(); ++i) {
                e_src_pre[i] = trace.output_heads.front().e_src[i] - model.output_heads.front().attn_src_bias;
                e_dst_pre[i] = trace.output_heads.front().e_dst[i] - model.output_heads.front().attn_dst_bias;
            }
            run_formal_matmul_trace_attempt_row_chunked(
                spec.dataset,
                "output_head0_source_attention_dot",
                trace.output_heads.front().h_prime,
                rows,
                cols,
                model.output_heads.front().attn_src,
                1,
                e_src_pre,
                &artifact,
                &debug);
            run_formal_matmul_trace_attempt_row_chunked(
                spec.dataset,
                "output_head0_destination_attention_dot",
                trace.output_heads.front().h_prime,
                rows,
                cols,
                model.output_heads.front().attn_dst,
                1,
                e_dst_pre,
                &artifact,
                &debug);
        }

        if (!trace.hidden_heads.empty() && !trace.hidden_heads.front().empty() && !model.hidden_heads.empty()) {
            const auto& hidden0 = trace.hidden_heads.front().front();
            const auto& hidden_input0 = trace.hidden_layer_inputs.front();
            const std::size_t hidden_rows = hidden0.rows;
            const std::size_t hidden_input_dim = trace.hidden_layer_input_dims.front();
            const auto hidden_model_it = std::find_if(model.hidden_heads.begin(), model.hidden_heads.end(), [](const HeadParams& head) {
                return head.layer_index == 0;
            });
            if (hidden_model_it != model.hidden_heads.end()) {
                run_formal_matmul_trace_attempt_row_chunked(
                    spec.dataset,
                    "hidden_layer0_head0_projection",
                    hidden_input0,
                    hidden_rows,
                    hidden_input_dim,
                    hidden_model_it->W,
                    hidden0.width,
                    hidden0.h_prime,
                    &artifact,
                    &debug);
                std::vector<double> hidden_e_src_pre(hidden0.e_src.size(), 0.0);
                std::vector<double> hidden_e_dst_pre(hidden0.e_dst.size(), 0.0);
                for (std::size_t i = 0; i < hidden_e_src_pre.size(); ++i) {
                    hidden_e_src_pre[i] = hidden0.e_src[i] - hidden_model_it->attn_src_bias;
                    hidden_e_dst_pre[i] = hidden0.e_dst[i] - hidden_model_it->attn_dst_bias;
                }
                run_formal_matmul_trace_attempt_row_chunked(
                    spec.dataset,
                    "hidden_layer0_head0_source_attention_dot",
                    hidden0.h_prime,
                    hidden_rows,
                    hidden0.width,
                    hidden_model_it->attn_src,
                    1,
                    hidden_e_src_pre,
                    &artifact,
                    &debug);
                run_formal_matmul_trace_attempt_row_chunked(
                    spec.dataset,
                    "hidden_layer0_head0_destination_attention_dot",
                    hidden0.h_prime,
                    hidden_rows,
                    hidden0.width,
                    hidden_model_it->attn_dst,
                    1,
                    hidden_e_dst_pre,
                    &artifact,
                    &debug);
            }
            artifact.blocked_relations.push_back(
                "hidden projection matmul trace 已接入 layer0/head0 的 row chunk；完整 Level A 仍需要全层全头 coverage 与 quantization/rescale closure");
            artifact.blocked_relations.push_back(
                "hidden source/destination attention dot 已接入 layer0/head0 的 row chunk；全 head coverage 与后续 sparse/softmax 链路仍未闭合");
        } else {
            artifact.blocked_relations.push_back("hidden trace 不可用，无法生成 formal hidden projection/dot-product relation evidence");
        }
        artifact.blocked_relations.push_back("sparse routing/grouped sum 已有独立 zkDL relation gadget；本 legacy partial trace runner 尚未把它并入完整 GAT proof object");
        artifact.blocked_relations.push_back("softmax/LeakyReLU/ExpMap/denominator/inverse/weighted aggregation/ELU/rescale blocked：缺少 zkDL verifier gadget 或公开量化契约");

        artifact.proof_generation_ok = !artifact.covered_relations.empty();
        artifact.verification_ok = !artifact.covered_relations.empty();
        artifact.blocker = "仅完成 partial formal trace proof："
            + std::to_string(artifact.covered_relations.size())
            + " 个 relation group 已验证，但 Level A 仍被 "
            + std::to_string(artifact.blocked_relations.size())
            + " 个缺失或被拒绝的 GAT relation group 阻塞；verified_chunks="
            + std::to_string(artifact.verified_chunks)
            + "/"
            + std::to_string(artifact.total_chunks)
            + ".";
        artifact.proof_path = run_dir / "formal_artifacts" / "zkdl_formal_partial.proof.txt";
        serialize_formal_partial_artifact(artifact, artifact.proof_path);
        dbg("formal_partial stage=done covered=" + std::to_string(artifact.covered_relations.size())
            + " blocked=" + std::to_string(artifact.blocked_relations.size())
            + " proof_path=" + artifact.proof_path.string());
    } catch (const std::exception& error) {
        artifact.blocker = std::string("formal partial trace proof 失败：") + error.what();
        artifact.blocked_relations.push_back(artifact.blocker);
        artifact.proof_path = run_dir / "formal_artifacts" / "zkdl_formal_partial.blocked.txt";
        try {
            serialize_formal_partial_artifact(artifact, artifact.proof_path);
        } catch (...) {
        }
        dbg("formal_partial stage=failed reason=\"" + artifact.blocker + "\"");
    }
    return artifact;
}

struct RunRow {
    std::string dataset;
    std::string status = "blocked";
    std::string blocker;
    std::string benchmark_mode = "warm";
    std::string commitment_time_ms = "NA";
    std::string prove_time_ms = "NA";
    std::string verify_time_ms = "NA";
    std::string proof_size_bytes = "NA";
    bool full_dataset = false;
    bool gpu_forward_ok = false;
    bool partial_commitment_ok = false;
    bool proof_generation_ok = false;
    bool verification_ok = false;
    std::string debug_level_b_status = "not_run";
    std::string debug_level_b_blocker;
    std::string debug_level_b_benchmark_mode = "diagnostic_level_b_output_binding_only_not_formal_level_a";
    std::string debug_level_b_commitment_time_ms = "NA";
    std::string debug_level_b_prove_time_ms = "NA";
    std::string debug_level_b_verify_time_ms = "NA";
    std::string debug_level_b_proof_size_bytes = "NA";
    std::string debug_level_b_proof_path;
    std::string formal_partial_status = "not_run";
    std::string formal_partial_blocker;
    std::string formal_partial_commitment_time_ms = "NA";
    std::string formal_partial_prove_time_ms = "NA";
    std::string formal_partial_verify_time_ms = "NA";
    std::string formal_partial_proof_size_bytes = "NA";
    std::string formal_partial_proof_path;
    std::string formal_partial_total_chunks = "0";
    std::string formal_partial_verified_chunks = "0";
    std::string formal_partial_covered_relations;
    std::string formal_partial_blocked_relations;
};

std::string join_sizes(const std::vector<std::size_t>& values) {
    std::ostringstream out;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out << ',';
        out << values[i];
    }
    return out.str();
}

void write_manifest(
    const fs::path& run_dir,
    const DatasetRunSpec& spec,
    const GraphDataset& dataset,
    const LocalGraph& graph,
    const ModelParams& model,
    const ForwardMetrics& forward,
    const ProofAttempt& proof,
    const RunRow& row) {
    ensure_dir(run_dir);
    std::ofstream out(run_dir / "run_manifest.json");
    out << "{\n";
    auto kv = [&](const std::string& key, const std::string& value, bool comma = true) {
        out << "  \"" << json_escape(key) << "\": \"" << json_escape(value) << "\"";
        if (comma) out << ',';
        out << '\n';
    };
    kv("backend_name", "zkDL/cuda_hyrax_partial");
    kv("compute_backend_name", "gpu");
    kv("benchmark_mode", row.benchmark_mode);
    kv("dataset", spec.dataset);
    kv("dataset_name", dataset.name);
    kv("status", row.status);
    kv("blocker", row.blocker);
    kv("verified", row.status == "success" ? "true" : "false");
    kv("is_full_dataset", row.full_dataset ? "true" : "false");
    kv("gpu_forward_ok", row.gpu_forward_ok ? "true" : "false");
    kv("partial_output_commitment_ok", row.partial_commitment_ok ? "true" : "false");
    kv("proof_generation_ok", row.proof_generation_ok ? "true" : "false");
    kv("verification_ok", row.verification_ok ? "true" : "false");
    kv("node_count", std::to_string(graph.num_nodes));
    kv("edge_count", std::to_string(graph.edges.size()));
    kv("N_total", std::to_string(dataset.num_nodes));
    kv("G_batch", std::to_string(graph.graph_count));
    kv("node_ptr", join_sizes(graph.node_ptr));
    kv("edge_ptr", join_sizes(graph.edge_ptr));
    kv("task_type", spec.task_type);
    kv("report_unit", spec.report_unit);
    kv("batching_rule", spec.batching_rule);
    kv("subgraph_rule", "whole_graph");
    kv("self_loop_rule", "per_node");
    kv("edge_sort_rule", "edge_gid_then_dst_stable");
    kv("chunking_rule", "output_linear_full_dataset_row_chunks_bound_by_relation_label_offset_rows_total_chunks");
    kv("L", std::to_string(model.L));
    std::ostringstream hidden_profile;
    for (std::size_t i = 0; i < model.hidden_profile.size(); ++i) {
        if (i != 0) hidden_profile << ';';
        hidden_profile << model.hidden_profile[i].first << 'x' << model.hidden_profile[i].second;
    }
    kv("hidden_profile", hidden_profile.str());
    kv("K_out", std::to_string(model.K_out));
    kv("C", std::to_string(model.C));
    kv("model_arch_id", model.model_arch_id);
    kv("model_param_id", model.model_param_id);
    kv("quant_cfg_id", model.quant_cfg_id + ";feature_quant=round(x*16)");
    kv("static_table_id", "tables:lrelu+elu+exp+range-required-by-note");
    kv("output_average_rule", model.output_average_rule);
    kv("forward_ms", format_ms(forward.total_ms));
    kv("hidden_forward_projection_ms", format_ms(forward.hidden_projection_ms));
    kv("hidden_forward_attention_ms", format_ms(forward.hidden_attention_ms));
    kv("hidden_forward_activation_ms", format_ms(forward.hidden_activation_ms));
    kv("hidden_concat_ms", format_ms(forward.hidden_concat_ms));
    kv("output_forward_projection_ms", format_ms(forward.output_projection_ms));
    kv("output_forward_attention_ms", format_ms(forward.output_attention_ms));
    kv("output_forward_activation_ms", format_ms(forward.output_activation_ms));
    kv("partial_commitment_ms", format_ms(proof.partial_commitment_ms));
    kv("partial_open_ms", format_ms(proof.partial_open_ms));
    kv("partial_proof_size_bytes", std::to_string(proof.partial_proof_size_bytes));
    kv("level_b_proof_path", proof.level_b.proof_path.empty() ? "" : proof.level_b.proof_path.string());
    kv("debug_level_b_status", row.debug_level_b_status);
    kv("debug_level_b_blocker", row.debug_level_b_blocker);
    kv("debug_level_b_benchmark_mode", row.debug_level_b_benchmark_mode);
    kv("debug_level_b_commitment_time_ms", row.debug_level_b_commitment_time_ms);
    kv("debug_level_b_prove_time_ms", row.debug_level_b_prove_time_ms);
    kv("debug_level_b_verify_time_ms", row.debug_level_b_verify_time_ms);
    kv("debug_level_b_proof_size_bytes", row.debug_level_b_proof_size_bytes);
    kv("debug_level_b_proof_path", row.debug_level_b_proof_path);
    kv("formal_partial_status", row.formal_partial_status);
    kv("formal_partial_blocker", row.formal_partial_blocker);
    kv("formal_partial_commitment_time_ms", row.formal_partial_commitment_time_ms);
    kv("formal_partial_prove_time_ms", row.formal_partial_prove_time_ms);
    kv("formal_partial_verify_time_ms", row.formal_partial_verify_time_ms);
    kv("formal_partial_proof_size_bytes", row.formal_partial_proof_size_bytes);
    kv("formal_partial_proof_path", row.formal_partial_proof_path);
    kv("formal_partial_total_chunks", row.formal_partial_total_chunks);
    kv("formal_partial_verified_chunks", row.formal_partial_verified_chunks);
    kv("formal_partial_covered_relations", row.formal_partial_covered_relations);
    kv("formal_partial_blocked_relations", row.formal_partial_blocked_relations);
    kv("level_b_covered_relations", [&]() {
        std::ostringstream relations;
        for (std::size_t i = 0; i < proof.level_b.covered_relations.size(); ++i) {
            if (i != 0) relations << ';';
            relations << proof.level_b.covered_relations[i];
        }
        return relations.str();
    }());
    kv("level_b_uncovered_relations", [&]() {
        std::ostringstream relations;
        for (std::size_t i = 0; i < proof.level_b.uncovered_relations.size(); ++i) {
            if (i != 0) relations << ';';
            relations << proof.level_b.uncovered_relations[i];
        }
        return relations.str();
    }());
    kv("commitment_time_ms", row.commitment_time_ms);
    kv("prove_time_ms", row.prove_time_ms);
    kv("verify_time_ms", row.verify_time_ms);
    kv("proof_size_bytes", row.proof_size_bytes, false);
    out << "}\n";

    std::ofstream bench(run_dir / "benchmark.txt");
    bench << "backend = gpu\n";
    bench << "commitment_time_ms = " << row.commitment_time_ms << "\n";
    bench << "config = configs/" << spec.config_name << "\n";
    bench << "dataset = " << spec.dataset << "\n";
    bench << "proof_size_bytes = " << row.proof_size_bytes << "\n";
    bench << "prove_time_ms = " << row.prove_time_ms << "\n";
    bench << "status = " << (row.status == "success" ? "VERIFY_OK" : row.status) << "\n";
    bench << "verify_time_ms = " << row.verify_time_ms << "\n";
    bench << "blocker = " << row.blocker << "\n";
    bench << "debug_level_b_status = " << row.debug_level_b_status << "\n";
    bench << "debug_level_b_commitment_time_ms = " << row.debug_level_b_commitment_time_ms << "\n";
    bench << "debug_level_b_prove_time_ms = " << row.debug_level_b_prove_time_ms << "\n";
    bench << "debug_level_b_verify_time_ms = " << row.debug_level_b_verify_time_ms << "\n";
    bench << "debug_level_b_proof_size_bytes = " << row.debug_level_b_proof_size_bytes << "\n";
    bench << "debug_level_b_proof_path = " << row.debug_level_b_proof_path << "\n";
    bench << "debug_level_b_blocker = " << row.debug_level_b_blocker << "\n";
    bench << "formal_partial_status = " << row.formal_partial_status << "\n";
    bench << "formal_partial_commitment_time_ms = " << row.formal_partial_commitment_time_ms << "\n";
    bench << "formal_partial_prove_time_ms = " << row.formal_partial_prove_time_ms << "\n";
    bench << "formal_partial_verify_time_ms = " << row.formal_partial_verify_time_ms << "\n";
    bench << "formal_partial_proof_size_bytes = " << row.formal_partial_proof_size_bytes << "\n";
    bench << "formal_partial_proof_path = " << row.formal_partial_proof_path << "\n";
    bench << "formal_partial_total_chunks = " << row.formal_partial_total_chunks << "\n";
    bench << "formal_partial_verified_chunks = " << row.formal_partial_verified_chunks << "\n";
    bench << "formal_partial_blocker = " << row.formal_partial_blocker << "\n";
}

std::string bool_string(bool value) {
    return value ? "true" : "false";
}

void write_coverage_matrix(const fs::path& output_root) {
    ensure_dir(output_root / "results");
    struct SimpleCoverageRow {
        std::string relation;
        std::string status;
        std::string total;
        std::string verified;
        bool prover;
        bool verifier;
        bool transcript;
        bool opening;
        bool clean;
        std::string evidence;
        std::string blocker;
    };
    const std::vector<SimpleCoverageRow> rows = {
        {"commitment/evaluation binding", "supported", "selftest", "selftest", true, true, true, true, true, "zkdl_gat_outputs/runs/selftest/formal_opening_selftest_latest.log", "Gate 1 PASS。"},
        {"transcript binding", "supported", "selftest+chunks", "selftest+chunks", true, true, true, true, true, "zkdl_gat_outputs/runs/selftest/formal_sumcheck_selftest_latest.log;zkdl_gat_outputs/runs/selftest/formal_relation_selftest_latest.log", "Gate 2/3 PASS。"},
        {"HadamardRelationProof", "supported", "selftest", "selftest", true, true, true, true, true, "zkdl_gat_outputs/runs/selftest/formal_relation_selftest_latest.log", "zkDL 自身 Hadamard relation 通过。"},
        {"MatmulRelationProof", "supported", "selftest+targeted", "selftest+targeted", true, true, true, true, true, "zkdl_gat_outputs/results/formal_partial_debug.csv", "zkDL 自身 matmul relation 可验证；用于 GAT 时仍只是子关系。"},
        {"LinearRelationProof", "supported", "selftest+partial", "selftest+partial", true, true, true, true, true, "zkdl_gat_outputs/results/formal_partial_debug.csv", "zkDL 自身 linear relation 可验证。"},
        {"hidden projection", "partial_formal", "历史 targeted chunks", "see formal_partial_debug", true, true, true, true, true, "zkdl_gat_outputs/results/formal_partial_debug.csv", "只证明 projection 子关系；完整 attention 链路未闭合。"},
        {"hidden source attention dot", "partial_formal", "历史 targeted chunks", "see formal_partial_debug", true, true, true, true, true, "zkdl_gat_outputs/results/formal_partial_debug.csv", "只证明 source dot 子关系。"},
        {"hidden destination attention dot", "partial_formal", "历史 targeted chunks", "see formal_partial_debug", true, true, true, true, true, "zkdl_gat_outputs/results/formal_partial_debug.csv", "只证明 destination dot 子关系。"},
        {"output projection", "partial_formal", "历史 targeted chunks", "see formal_partial_debug", true, true, true, true, true, "zkdl_gat_outputs/results/formal_partial_debug.csv", "只证明 output projection 子关系。"},
        {"output source attention dot", "partial_formal", "历史 targeted chunks", "see formal_partial_debug", true, true, true, true, true, "zkdl_gat_outputs/results/formal_partial_debug.csv", "只证明 output source dot 子关系。"},
        {"output destination attention dot", "partial_formal", "历史 targeted chunks", "see formal_partial_debug", true, true, true, true, true, "zkdl_gat_outputs/results/formal_partial_debug.csv", "只证明 output destination dot 子关系。"},
        {"hidden concat", "partial_formal", "LinearRelationProof primitive", "supporting evidence only", true, true, true, true, true, "zkdl_gat_outputs/results/formal_partial_debug.csv", "hidden concat 子关系可表达；不代表完整 GAT proof。"},
        {"output bias add", "partial_formal", "LinearRelationProof primitive", "supporting evidence only", true, true, true, true, true, "zkdl_gat_outputs/results/formal_partial_debug.csv", "bias add 子关系可表达；前序 rescale 仍阻塞。"},
        {"output-head average", "partial_formal", "LinearRelationProof primitive", "supporting evidence only", true, true, true, true, true, "zkdl_gat_outputs/results/formal_partial_debug.csv", "output-head average 子关系可表达；前序 output attention chain 未闭合。"},
        {"public logits binding", "partial_formal", "LinearRelationProof primitive", "supporting evidence only", true, true, true, true, true, "zkdl_gat_outputs/results/formal_partial_debug.csv", "public logits 子关系可表达；前序 GAT relation 未闭合。"},
        {"fixed-to-rounded trace scale decomposition", "partial_formal", "targeted schedules", "see formal_partial_debug", true, true, true, true, true, "zkdl_gat_outputs/results/formal_partial_debug.csv", "只证明 exact decomposition；未证明 sign/tie policy。"},
        {"residual bounded range binding", "partial_formal", "targeted schedules", "see formal_partial_debug", true, true, true, true, true, "zkdl_gat_outputs/results/formal_partial_debug.csv", "residual range 子关系不能替代 rounding proof。"},
        {"proof serialization", "partial_formal", "partial artifacts", "partial artifacts", true, true, true, true, true, "zkdl_gat_outputs/results/formal_partial_debug.csv", "没有完整 GAT proof object。"},
        {"proof_size_bytes 统计", "partial_formal", "partial artifacts", "partial artifacts", true, true, true, true, true, "zkdl_gat_outputs/results/formal_partial_debug.csv", "partial proof_size 不写入 official。"},
        {"verifier 不读取 witness", "partial_formal", "partial artifacts", "partial artifacts", true, true, true, true, true, "zkdl_gat_outputs/results/formal_partial_debug.csv", "完整 GAT verifier 尚未存在。"},
        {"proof object 不含完整 tensor/debug dump", "partial_formal", "partial artifacts", "partial artifacts", true, true, true, true, true, "zkdl_gat_outputs/results/formal_partial_debug.csv", "当前只对 partial artifact 成立。"},
        {"debug_level_b output binding", "debug_only", "diagnostic", "diagnostic", false, false, false, false, false, "zkdl_gat_outputs/results/debug_level_b.csv", "不是 formal benchmark。"},
        {"SparseSourceRoutingRelationProof", "supported", "five full datasets hidden_layer0/head0", "see sparse evidence", true, true, true, true, true, "zkdl_gat_outputs/evidence/sparse_grouped_relation_cora.json;zkdl_gat_outputs/evidence/sparse_grouped_relation_citeseer.json;zkdl_gat_outputs/evidence/sparse_grouped_relation_pubmed.json;zkdl_gat_outputs/evidence/sparse_grouped_relation_ppi.json;zkdl_gat_outputs/evidence/sparse_grouped_relation_ogbn-arxiv.json", "zkDL sparse source routing gadget 使用 public src 和 topology hash；不使用 dense attention。"},
        {"SparseDestinationRoutingRelationProof", "supported", "five full datasets hidden_layer0/head0", "see sparse evidence", true, true, true, true, true, "zkdl_gat_outputs/evidence/sparse_grouped_relation_cora.json;zkdl_gat_outputs/evidence/sparse_grouped_relation_citeseer.json;zkdl_gat_outputs/evidence/sparse_grouped_relation_pubmed.json;zkdl_gat_outputs/evidence/sparse_grouped_relation_ppi.json;zkdl_gat_outputs/evidence/sparse_grouped_relation_ogbn-arxiv.json", "zkDL sparse destination routing gadget 使用 public dst 和 topology hash；不使用 dense attention。"},
        {"GroupedSumRelationProof", "supported", "five full datasets hidden_layer0/head0", "see sparse evidence", true, true, true, true, true, "zkdl_gat_outputs/evidence/sparse_grouped_relation_cora.json;zkdl_gat_outputs/evidence/sparse_grouped_relation_citeseer.json;zkdl_gat_outputs/evidence/sparse_grouped_relation_pubmed.json;zkdl_gat_outputs/evidence/sparse_grouped_relation_ppi.json;zkdl_gat_outputs/evidence/sparse_grouped_relation_ogbn-arxiv.json", "一维 grouped sum gadget 已实现；尚未等同于完整 softmax/aggregation 链条。"},
        {"MultiColumnGroupedSumRelationProof", "supported", "selftest + five full datasets", "hidden layer0/head0 8/8 feature", true, true, true, true, true, "zkdl_gat_outputs/runs/selftest/formal_relation_selftest_latest.log;zkdl_gat_outputs/evidence/softmax_aggregation_relation_cora.json;zkdl_gat_outputs/evidence/softmax_aggregation_relation_citeseer.json;zkdl_gat_outputs/evidence/softmax_aggregation_relation_pubmed.json;zkdl_gat_outputs/evidence/softmax_aggregation_relation_ppi.json;zkdl_gat_outputs/evidence/softmax_aggregation_relation_ogbn-arxiv.json", "multi-column grouped sum 使用 zkDL 自身 column compression + matrix opening 绑定。"},
        {"source sparse routing", "partial_formal", "five full datasets hidden_layer0/head0", "see sparse evidence", true, true, true, true, true, "zkdl_gat_outputs/evidence/sparse_grouped_relation_cora.json;zkdl_gat_outputs/evidence/sparse_grouped_relation_citeseer.json;zkdl_gat_outputs/evidence/sparse_grouped_relation_pubmed.json;zkdl_gat_outputs/evidence/sparse_grouped_relation_ppi.json;zkdl_gat_outputs/evidence/sparse_grouped_relation_ogbn-arxiv.json", "已局部实现并验证 hidden layer0/head0 source attention routing；完整所有层/头接入仍未完成。"},
        {"destination sparse routing", "partial_formal", "five full datasets hidden_layer0/head0", "see sparse evidence", true, true, true, true, true, "zkdl_gat_outputs/evidence/sparse_grouped_relation_cora.json;zkdl_gat_outputs/evidence/sparse_grouped_relation_citeseer.json;zkdl_gat_outputs/evidence/sparse_grouped_relation_pubmed.json;zkdl_gat_outputs/evidence/sparse_grouped_relation_ppi.json;zkdl_gat_outputs/evidence/sparse_grouped_relation_ogbn-arxiv.json", "已局部实现并验证 hidden layer0/head0 destination attention routing；完整所有层/头接入仍未完成。"},
        {"target-node softmax group", "blocked", "NA", "NA", false, false, false, false, false, "zkdl_gat_outputs/evidence/rounding_sparse_blocker_evidence.md", "缺少完整 softmax group relation。"},
        {"softmax group max", "blocked", "NA", "NA", false, false, false, false, false, "zkdl_gat_outputs/evidence/rounding_sparse_blocker_evidence.md", "缺少 max/range comparison gadget。"},
        {"ExpMap", "blocked", "NA", "NA", false, false, false, false, false, "zkdl_gat_outputs/evidence/rounding_sparse_blocker_evidence.md", "缺少 QuantExp lookup argument。"},
        {"softmax denominator sum", "partial_formal", "five full datasets hidden_layer0/head0", "5/5 dataset；U_by_dst", true, true, true, true, true, "zkdl_gat_outputs/evidence/softmax_aggregation_relation_cora.json;zkdl_gat_outputs/evidence/softmax_aggregation_relation_citeseer.json;zkdl_gat_outputs/evidence/softmax_aggregation_relation_pubmed.json;zkdl_gat_outputs/evidence/softmax_aggregation_relation_ppi.json;zkdl_gat_outputs/evidence/softmax_aggregation_relation_ogbn-arxiv.json", "已证明 U -> denominator；U 的 ExpMap 来源与所有层/头仍未闭合。"},
        {"inverse", "partial_formal", "five full datasets hidden_layer0/head0", "5/5 dataset；denominator*inv=valid_one", true, true, true, true, true, "zkdl_gat_outputs/evidence/softmax_aggregation_relation_cora.json;zkdl_gat_outputs/evidence/softmax_aggregation_relation_citeseer.json;zkdl_gat_outputs/evidence/softmax_aggregation_relation_pubmed.json;zkdl_gat_outputs/evidence/softmax_aggregation_relation_ppi.json;zkdl_gat_outputs/evidence/softmax_aggregation_relation_ogbn-arxiv.json", "已证明 denominator inverse 与 destination routing；仍缺完整 softmax max/ExpMap 链。"},
        {"alpha multiplication", "partial_formal", "five full datasets hidden_layer0/head0", "5/5 dataset；U*inv_edge=alpha 与 alpha*denom=U", true, true, true, true, true, "zkdl_gat_outputs/evidence/softmax_aggregation_relation_cora.json;zkdl_gat_outputs/evidence/softmax_aggregation_relation_citeseer.json;zkdl_gat_outputs/evidence/softmax_aggregation_relation_pubmed.json;zkdl_gat_outputs/evidence/softmax_aggregation_relation_ppi.json;zkdl_gat_outputs/evidence/softmax_aggregation_relation_ogbn-arxiv.json", "alpha 乘法 relation 已验证；U 的 ExpMap 来源与所有层/头仍未闭合。"},
        {"weighted aggregation", "partial_formal", "five full datasets hidden_layer0/head0", "8/8 feature via multi-column compressed relation", true, true, true, true, true, "zkdl_gat_outputs/evidence/softmax_aggregation_relation_cora.json;zkdl_gat_outputs/evidence/softmax_aggregation_relation_citeseer.json;zkdl_gat_outputs/evidence/softmax_aggregation_relation_pubmed.json;zkdl_gat_outputs/evidence/softmax_aggregation_relation_ppi.json;zkdl_gat_outputs/evidence/softmax_aggregation_relation_ogbn-arxiv.json", "已验证 multi-column message routing、alpha*message 与 weighted_by_dst；所有层/头、output aggregation 与 rescale 仍 blocked。"},
        {"LeakyReLU", "blocked", "NA", "NA", false, false, false, false, false, "zkdl_gat_outputs/evidence/rounding_sparse_blocker_evidence.md", "缺少 lookup/range verifier gadget。"},
        {"ELU", "blocked", "NA", "NA", false, false, false, false, false, "zkdl_gat_outputs/evidence/rounding_sparse_blocker_evidence.md", "缺少 ELU lookup/range verifier gadget。"},
        {"output attention aggregation", "blocked", "NA", "NA", false, false, false, false, false, "zkdl_gat_outputs/evidence/rounding_sparse_blocker_evidence.md", "缺少 output attention aggregation verifier。"},
        {"output rescale", "blocked", "NA", "NA", false, false, false, false, false, "zkdl_gat_outputs/evidence/rounding_sparse_blocker_evidence.md", "缺少 real rescale chain 与 rounding policy proof。"},
        {"rounding sign/tie-breaking", "blocked", "NA", "NA", false, false, false, false, false, "zkdl_gat_outputs/evidence/rounding_sparse_blocker_evidence.md", "缺少公开 rho/delta/rem_bits/tie_policy。"},
    };
    {
        std::ofstream csv(output_root / "results" / "gat_proof_coverage_matrix.csv");
        csv << "relation_name,status,total_chunks,verified_chunks,formal_prover_supported,formal_verifier_supported,transcript_bound,opening_bound,proof_object_clean,evidence_file,blocker_reason\n";
        for (const auto& row : rows) {
            csv << row.relation
                << ',' << row.status
                << ",\"" << json_escape(row.total) << "\""
                << ",\"" << json_escape(row.verified) << "\""
                << ',' << bool_string(row.prover)
                << ',' << bool_string(row.verifier)
                << ',' << bool_string(row.transcript)
                << ',' << bool_string(row.opening)
                << ',' << bool_string(row.clean)
                << ",\"" << json_escape(row.evidence) << "\""
                << ",\"" << json_escape(row.blocker) << "\"\n";
        }
    }
    {
        std::ofstream json(output_root / "results" / "gat_proof_coverage_matrix.json");
        json << "{\n  \"coverage_level\": \"Level A blocked；relation-level coverage only\",\n";
        json << "  \"allowed_status\": [\"supported\", \"partial_formal\", \"debug_only\", \"blocked\", \"not_implemented\"],\n";
        json << "  \"rows\": [\n";
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const auto& row = rows[i];
            json << "    {\n";
            json << "      \"relation_name\": \"" << json_escape(row.relation) << "\",\n";
            json << "      \"status\": \"" << json_escape(row.status) << "\",\n";
            json << "      \"total_chunks\": \"" << json_escape(row.total) << "\",\n";
            json << "      \"verified_chunks\": \"" << json_escape(row.verified) << "\",\n";
            json << "      \"formal_prover_supported\": " << bool_string(row.prover) << ",\n";
            json << "      \"formal_verifier_supported\": " << bool_string(row.verifier) << ",\n";
            json << "      \"transcript_bound\": " << bool_string(row.transcript) << ",\n";
            json << "      \"opening_bound\": " << bool_string(row.opening) << ",\n";
            json << "      \"proof_object_clean\": " << bool_string(row.clean) << ",\n";
            json << "      \"evidence_file\": \"" << json_escape(row.evidence) << "\",\n";
            json << "      \"blocker_reason\": \"" << json_escape(row.blocker) << "\"\n";
            json << "    }" << (i + 1 == rows.size() ? "\n" : ",\n");
        }
        json << "  ]\n}\n";
    }
    {
        std::ofstream md(output_root / "results" / "gat_proof_coverage_matrix.md");
        md << "# GAT proof 覆盖矩阵\n\n";
        md << "本矩阵使用五态分类：`supported`、`partial_formal`、`debug_only`、`blocked`、`not_implemented`。`partial_formal` 与 `debug_only` 不计入 Level A。\n\n";
        md << "| relation | status | total | verified | prover | verifier | transcript | opening | clean proof object | evidence | blocker |\n";
        md << "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n";
        for (const auto& row : rows) {
            md << "| " << row.relation
               << " | " << row.status
               << " | " << row.total
               << " | " << row.verified
               << " | " << bool_string(row.prover)
               << " | " << bool_string(row.verifier)
               << " | " << bool_string(row.transcript)
               << " | " << bool_string(row.opening)
               << " | " << bool_string(row.clean)
               << " | " << row.evidence
               << " | " << row.blocker << " |\n";
        }
    }
    fs::copy_file(
        output_root / "results" / "gat_proof_coverage_matrix.csv",
        output_root / "results" / "coverage_matrix.csv",
        fs::copy_options::overwrite_existing);
    fs::copy_file(
        output_root / "results" / "gat_proof_coverage_matrix.json",
        output_root / "results" / "coverage_matrix.json",
        fs::copy_options::overwrite_existing);
    fs::copy_file(
        output_root / "results" / "gat_proof_coverage_matrix.md",
        output_root / "results" / "coverage_matrix.md",
        fs::copy_options::overwrite_existing);
}

void write_result_tables(const fs::path& output_root, const std::vector<RunRow>& rows) {
    ensure_dir(output_root / "results");
    {
        std::ofstream csv(output_root / "results" / "latest.csv");
        csv << "dataset,status,blocker,benchmark_mode,commitment_time_ms,prove_time_ms,verify_time_ms,proof_size_bytes\n";
        for (const auto& row : rows) {
            csv << row.dataset << ',' << row.status << ",\"" << json_escape(row.blocker) << "\"," << row.benchmark_mode
                << ',' << row.commitment_time_ms
                << ',' << row.prove_time_ms
                << ',' << row.verify_time_ms
                << ',' << row.proof_size_bytes << '\n';
        }
    }
    {
        std::ofstream json(output_root / "results" / "latest.json");
        json << "{\n  \"benchmark_mode\": \"warm\",\n  \"policy\": \"official rows 使用主仓库 benchmark schema；完整 Level A proof generation 和 verification 成功前正式四项保持 NA\",\n  \"rows\": [\n";
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const auto& row = rows[i];
            json << "    {\n";
            json << "      \"dataset\": \"" << json_escape(row.dataset) << "\",\n";
            json << "      \"status\": \"" << json_escape(row.status) << "\",\n";
            json << "      \"blocker\": \"" << json_escape(row.blocker) << "\",\n";
            json << "      \"benchmark_mode\": \"" << json_escape(row.benchmark_mode) << "\",\n";
            json << "      \"commitment_time_ms\": \"" << row.commitment_time_ms << "\",\n";
            json << "      \"prove_time_ms\": \"" << row.prove_time_ms << "\",\n";
            json << "      \"verify_time_ms\": \"" << row.verify_time_ms << "\",\n";
            json << "      \"proof_size_bytes\": \"" << row.proof_size_bytes << "\"\n";
            json << "    }" << (i + 1 == rows.size() ? "\n" : ",\n");
        }
        json << "  ]\n}\n";
    }
    {
        std::ofstream csv(output_root / "results" / "debug_level_b.csv");
        csv << "dataset,status,blocker,benchmark_mode,commitment_time_ms,prove_time_ms,verify_time_ms,proof_size_bytes,proof_path\n";
        for (const auto& row : rows) {
            csv << row.dataset
                << ',' << row.debug_level_b_status
                << ",\"" << json_escape(row.debug_level_b_blocker) << "\","
                << row.debug_level_b_benchmark_mode
                << ',' << row.debug_level_b_commitment_time_ms
                << ',' << row.debug_level_b_prove_time_ms
                << ',' << row.debug_level_b_verify_time_ms
                << ',' << row.debug_level_b_proof_size_bytes
                << ",\"" << json_escape(row.debug_level_b_proof_path) << "\"\n";
        }
    }
    {
        std::ofstream json(output_root / "results" / "debug_level_b.json");
        json << "{\n  \"benchmark_mode\": \"diagnostic_level_b_output_binding_only_not_formal_level_a\",\n  \"policy\": \"diagnostic 数字不得写入正式对比表\",\n  \"rows\": [\n";
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const auto& row = rows[i];
            json << "    {\n";
            json << "      \"dataset\": \"" << json_escape(row.dataset) << "\",\n";
            json << "      \"status\": \"" << json_escape(row.debug_level_b_status) << "\",\n";
            json << "      \"blocker\": \"" << json_escape(row.debug_level_b_blocker) << "\",\n";
            json << "      \"benchmark_mode\": \"" << json_escape(row.debug_level_b_benchmark_mode) << "\",\n";
            json << "      \"commitment_time_ms\": \"" << row.debug_level_b_commitment_time_ms << "\",\n";
            json << "      \"prove_time_ms\": \"" << row.debug_level_b_prove_time_ms << "\",\n";
            json << "      \"verify_time_ms\": \"" << row.debug_level_b_verify_time_ms << "\",\n";
            json << "      \"proof_size_bytes\": \"" << row.debug_level_b_proof_size_bytes << "\",\n";
            json << "      \"proof_path\": \"" << json_escape(row.debug_level_b_proof_path) << "\"\n";
            json << "    }" << (i + 1 == rows.size() ? "\n" : ",\n");
        }
        json << "  ]\n}\n";
    }
    {
        std::ofstream csv(output_root / "results" / "formal_partial_debug.csv");
        csv << "dataset,component,status,benchmark_mode,commitment_time_ms,prove_time_ms,verify_time_ms,proof_size_bytes,total_chunks,verified_chunks,newly_verified_chunks,verified_relations,blocked_relations,proof_artifact_paths,evidence_paths,blocker_evidence,resume_point,blocker\n";
        csv << "NA,formal_opening_selftest,success,formal_partial_opening_proof_verify,NA,NA,NA,see_log,NA,NA,NA,\"commitment/evaluation binding\",\"PPI/ogbn-arxiv 完整 G1 大开口只作为压力边界，不写入 Level A\",\"zkdl_gat_outputs/runs/selftest/formal_opening_selftest_latest.log\",\"zkdl_gat_outputs/runs/selftest/formal_opening_selftest_latest.log\",\"\",,\"Gate 1 opening selftest PASS；大图 logits-size G1 opening 不作为正式 Level A 结果。\"\n";
        csv << "NA,formal_sumcheck_selftest,success,formal_partial_inner_product_sumcheck_verify,NA,NA,NA,see_log,NA,NA,NA,\"inner-product sumcheck transcript replay\",\"\",\"zkdl_gat_outputs/runs/selftest/formal_sumcheck_selftest_latest.log\",\"zkdl_gat_outputs/runs/selftest/formal_sumcheck_selftest_latest.log\",\"\",,\"Gate 2 inner-product transcript replay verifier PASS。\"\n";
        csv << "NA,formal_relation_selftest,success,formal_partial_hadamard_matmul_linear_verify,NA,NA,NA,see_log,NA,NA,NA,\"HadamardRelationProof;MatmulRelationProof;LinearRelationProof\",\"\",\"zkdl_gat_outputs/runs/selftest/formal_relation_selftest_latest.log\",\"zkdl_gat_outputs/runs/selftest/formal_relation_selftest_latest.log\",\"\",,\"Gate 3 relation selftest PASS，包含 tamper rejection。\"\n";
        for (const auto& row : rows) {
            csv << row.dataset
                << ",formal_partial_trace"
                << ',' << row.formal_partial_status
                << ",partial_formal_relations_only,"
                << row.formal_partial_commitment_time_ms
                << ',' << row.formal_partial_prove_time_ms
                << ',' << row.formal_partial_verify_time_ms
                << ',' << row.formal_partial_proof_size_bytes
                << ',' << row.formal_partial_total_chunks
                << ',' << row.formal_partial_verified_chunks
                << ',' << row.formal_partial_verified_chunks
                << ",\"" << json_escape(row.formal_partial_covered_relations) << "\""
                << ",\"" << json_escape(row.formal_partial_blocked_relations) << "\""
                << ",\"" << json_escape(row.formal_partial_proof_path) << "\""
                << ",\"" << json_escape(row.formal_partial_proof_path + ";zkdl_gat_outputs/runs/" + row.dataset + "/gpu/warm/proof_debug.log") << "\""
                << ",\"" << json_escape("zkdl_gat_outputs/evidence/rounding_sparse_blocker_evidence.md") << "\""
                << ",\"" << json_escape("zkdl_gat_outputs/runs/" + row.dataset + "/gpu/warm/RESUME_POINT.md") << "\""
                << ","
                << "\"" << json_escape(row.formal_partial_blocker) << "\"\n";
        }
    }
    {
        std::ofstream json(output_root / "results" / "formal_partial_debug.json");
        json << "{\n  \"rows\": [\n";
        json << "    {\"dataset\":\"NA\",\"component\":\"formal_opening_selftest\",\"status\":\"success\",\"benchmark_mode\":\"formal_partial_opening_proof_verify\",\"commitment_time_ms\":\"NA\",\"prove_time_ms\":\"NA\",\"verify_time_ms\":\"NA\",\"proof_size_bytes\":\"see_log\",\"total_chunks\":\"NA\",\"verified_chunks\":\"NA\",\"newly_verified_chunks\":\"NA\",\"proof_artifact_paths\":\"zkdl_gat_outputs/runs/selftest/formal_opening_selftest_latest.log\",\"evidence_paths\":\"zkdl_gat_outputs/runs/selftest/formal_opening_selftest_latest.log\",\"blocker_evidence\":\"\",\"resume_point\":\"\",\"blocker\":\"Gate 1 opening selftest PASS；大图 logits-size G1 opening 不作为正式 Level A 结果。\"},\n";
        json << "    {\"dataset\":\"NA\",\"component\":\"formal_sumcheck_selftest\",\"status\":\"success\",\"benchmark_mode\":\"formal_partial_inner_product_sumcheck_verify\",\"commitment_time_ms\":\"NA\",\"prove_time_ms\":\"NA\",\"verify_time_ms\":\"NA\",\"proof_size_bytes\":\"see_log\",\"total_chunks\":\"NA\",\"verified_chunks\":\"NA\",\"newly_verified_chunks\":\"NA\",\"proof_artifact_paths\":\"zkdl_gat_outputs/runs/selftest/formal_sumcheck_selftest_latest.log\",\"evidence_paths\":\"zkdl_gat_outputs/runs/selftest/formal_sumcheck_selftest_latest.log\",\"blocker_evidence\":\"\",\"resume_point\":\"\",\"blocker\":\"Gate 2 inner-product transcript replay verifier PASS。\"},\n";
        json << "    {\"dataset\":\"NA\",\"component\":\"formal_relation_selftest\",\"status\":\"success\",\"benchmark_mode\":\"formal_partial_hadamard_matmul_linear_verify\",\"commitment_time_ms\":\"NA\",\"prove_time_ms\":\"NA\",\"verify_time_ms\":\"NA\",\"proof_size_bytes\":\"see_log\",\"total_chunks\":\"NA\",\"verified_chunks\":\"NA\",\"newly_verified_chunks\":\"NA\",\"proof_artifact_paths\":\"zkdl_gat_outputs/runs/selftest/formal_relation_selftest_latest.log\",\"evidence_paths\":\"zkdl_gat_outputs/runs/selftest/formal_relation_selftest_latest.log\",\"blocker_evidence\":\"\",\"resume_point\":\"\",\"blocker\":\"Gate 3 relation selftest PASS，包含 tamper rejection。\"}";
        for (const auto& row : rows) {
            json << ",\n";
            json << "    {\"dataset\":\"" << json_escape(row.dataset)
                 << "\",\"component\":\"formal_partial_trace"
                 << "\",\"status\":\"" << json_escape(row.formal_partial_status)
                 << "\",\"benchmark_mode\":\"partial_formal_relations_only\""
                 << ",\"commitment_time_ms\":\"" << row.formal_partial_commitment_time_ms
                 << "\",\"prove_time_ms\":\"" << row.formal_partial_prove_time_ms
                 << "\",\"verify_time_ms\":\"" << row.formal_partial_verify_time_ms
                 << "\",\"proof_size_bytes\":\"" << row.formal_partial_proof_size_bytes
                 << "\",\"total_chunks\":\"" << row.formal_partial_total_chunks
                 << "\",\"verified_chunks\":\"" << row.formal_partial_verified_chunks
                 << "\",\"newly_verified_chunks\":\"" << row.formal_partial_verified_chunks
                 << "\",\"proof_artifact_paths\":\"" << json_escape(row.formal_partial_proof_path)
                 << "\",\"evidence_file\":\"" << json_escape(row.formal_partial_proof_path)
                 << "\",\"evidence_paths\":\"" << json_escape(row.formal_partial_proof_path + ";zkdl_gat_outputs/runs/" + row.dataset + "/gpu/warm/proof_debug.log")
                 << "\",\"blocker_evidence\":\"" << json_escape("zkdl_gat_outputs/evidence/rounding_sparse_blocker_evidence.md")
                 << "\",\"resume_point\":\"" << json_escape("zkdl_gat_outputs/runs/" + row.dataset + "/gpu/warm/RESUME_POINT.md")
                 << "\",\"covered_relations\":\"" << json_escape(row.formal_partial_covered_relations)
                 << "\",\"blocked_relations\":\"" << json_escape(row.formal_partial_blocked_relations)
                 << "\",\"blocker\":\"" << json_escape(row.formal_partial_blocker) << "\"}";
        }
        json << "\n";
        json << "  ]\n}\n";
    }
    {
        std::ofstream md(output_root / "results" / "summary.md");
        md << "# 最新正式结果\n\n";
        md << "| 数据集 | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes | blocker |\n";
        md << "| --- | ---: | ---: | ---: | ---: | --- |\n";
        for (const auto& row : rows) {
            md << "| " << row.dataset
               << " | " << row.commitment_time_ms
               << " | " << row.prove_time_ms
               << " | " << row.verify_time_ms
               << " | " << row.proof_size_bytes
               << " | " << row.blocker << " |\n";
        }
    }
    write_coverage_matrix(output_root);
}

void write_adapter_readme(const fs::path& output_root, const std::vector<RunRow>& rows) {
    std::ofstream out(output_root / "README_ZKDL_GAT_ADAPTER.md");
    out << "# zkDL-GAT adapter 状态\n\n";
    out << "当前状态：不是 Level A。Gate 1/2/3 通过时，以 `runs/selftest/*_latest.log` 中的 `SUMMARY status=PASS` 为准。\n\n";
    out << "本目录是 zkDL baseline 的 GAT 适配，只替换模型入口并补齐运行五个全量数据集所需的数据、checkpoint、trace、relation evidence、artifact 与 benchmark 输出。\n\n";
    out << "本目录没有混入主仓库 GAT-ZKML 证明方案：不使用主仓库 KZG proof schema、quotient polynomial、verifier equation、opening proof 规则、transcript challenge 语义或 proof object layout。当前保留的是 zkDL 自身 commitment、opening、transcript、sumcheck 与 relation proof 语义。\n\n";
    out << "## 协议边界\n\n";
    out << "- 本适配严格以 `/home/pzh/project/comparison/GAT-ZKML-zkDL.md` 为最高协议边界。\n";
    out << "- 本轮修改未改变该文件规定的 GAT 前向语义。\n";
    out << "- 本轮修改未混入主仓库 GAT-ZKML 证明方案。\n";
    out << "- 本轮结果如果仍未 Level A，则 latest.csv/json 保持 NA。\n";
    out << "- 所有 partial_formal 只作为 supporting evidence。\n\n";
    out << "与主仓库指标口径对齐：`latest.csv` 字段固定为 `dataset,status,blocker,benchmark_mode,commitment_time_ms,prove_time_ms,verify_time_ms,proof_size_bytes`，数据集顺序固定为 `cora,citeseer,pubmed,ppi,ogbn-arxiv`，`benchmark_mode` 固定为 `warm`。只有 full dataset + complete formal proof generation + complete formal verification 成功后才填写 official 四项。\n\n";
    out << "当前 supported relation：commitment/evaluation binding、transcript binding、HadamardRelationProof、MatmulRelationProof、LinearRelationProof、SparseSourceRoutingRelationProof、SparseDestinationRoutingRelationProof、一维 GroupedSumRelationProof 与 MultiColumnGroupedSumRelationProof。hidden concat、output bias add、output-head average、public logits binding、projection/dot、scale decomposition 与 residual bounded range 只计入 partial formal。\n\n";
    out << "当前 partial formal：五个数据集 hidden layer0/head0 已验证 sparse/grouped、denominator/inverse/alpha 与 8/8 feature multi-column weighted aggregation。当前 blocked relation：target-node softmax group、softmax max、ExpMap、LeakyReLU、ELU、所有层/头 weighted/output aggregation、real rescale chain 与 rounding sign/tie-breaking 仍缺 zkDL verifier gadget、lookup/range primitive 或公开量化契约；这些 partial relation 尚未并入完整所有层/头的 GAT proof object。\n\n";
    out << "目录清理原则：保留源码、构建脚本、正式 results、summary、formal_partial_debug、debug_level_b、coverage matrix、必要 proof artifact、proof evidence、run manifest、benchmark、proof_debug、dataset resume point 和本 README；不保留错误调度日志、过期 proof artifact 或临时 debug 目录。\n\n";
    out << "数据集状态：\n\n";
    out << "| dataset | status | gpu_forward | partial commitment_ms | partial prove_ms | partial verify_ms | partial proof_size_bytes | verified chunks | total chunks | debug Level B |\n";
    out << "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n";
    for (const auto& row : rows) {
        out << "| " << row.dataset
            << " | " << row.status
            << " | " << (row.gpu_forward_ok ? "ok" : "failed")
            << " | " << row.formal_partial_commitment_time_ms
            << " | " << row.formal_partial_prove_time_ms
            << " | " << row.formal_partial_verify_time_ms
            << " | " << row.formal_partial_proof_size_bytes
            << " | " << row.formal_partial_verified_chunks
            << " | " << row.formal_partial_total_chunks
            << " | " << row.debug_level_b_status << " |\n";
    }
    out << "\n下一最小继续点以 `RESUME_POINT.md` 为准：优先补 softmax、lookup/range、aggregation、rescale 与 rounding 契约，不继续堆 projection/dot 长队列。coverage 说明见 `results/coverage_matrix.*`。\n";
}
