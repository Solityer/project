#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/crypto/curve.hpp"
#include "gatzk/protocol/trace.hpp"
#include "gatzk/protocol/prover.hpp"
#include "gatzk/protocol/verifier.hpp"
#include "gatzk/util/config.hpp"
#include "gatzk/util/route2.hpp"

// 本文件是正式命令行入口，
// 负责读取配置、选择 full_cpu / full_gpu 执行路线，并串联 context、trace、prove、verify 与 artifact 导出。
// 它只汇总正式四项指标与 manifest，不允许在入口层改写 proof contract。
// 控制台正式输出必须收口为“每次运行一行”，避免把开发过程日志重新暴露到提交表面。

int main(int argc, char** argv) {
    try {
        auto format_ms = [](double value) {
            std::ostringstream stream;
            stream << std::fixed << std::setprecision(3) << value;
            return stream.str();
        };
        auto clamp_non_negative = [](double value) {
            return value < 0.0 ? 0.0 : value;
        };
        auto prove_accounted_ms = [](const gatzk::protocol::RunMetrics& metrics) {
            return metrics.forward_ms
                + metrics.trace_generation_ms
                + metrics.commit_dynamic_ms
                + metrics.quotient_build_ms
                + metrics.domain_opening_ms
                + metrics.external_opening_ms
                + metrics.prove_finalize_ms;
        };
        auto verify_accounted_ms = [](const gatzk::protocol::RunMetrics& metrics) {
            return metrics.verify_metadata_ms
                + metrics.verify_transcript_ms
                + metrics.verify_domain_opening_ms
                + metrics.verify_quotient_ms
                + metrics.verify_external_fold_ms
                + metrics.verify_misc_ms;
        };
        auto append_note = [](gatzk::protocol::RunMetrics& metrics, const std::string& note) {
            if (note.empty()) {
                return;
            }
            if (!metrics.notes.empty()) {
                metrics.notes += "; ";
            }
            metrics.notes += note;
        };
        auto with_export_suffix = [](const gatzk::util::AppConfig& base, const std::vector<std::string>& suffixes) {
            auto updated = base;
            auto export_dir = std::filesystem::path(base.export_dir);
            for (const auto& suffix : suffixes) {
                if (!suffix.empty()) {
                    export_dir /= suffix;
                }
            }
            updated.export_dir = export_dir.string();
            return updated;
        };
        // 全图实例先把大对象初始化和正式测量拆开，避免冷启动污染 warm 主表。
        auto memory_safe_warm_required = [](const gatzk::util::AppConfig& config) {
            return config.batching_rule == "whole_graph_single"
                && config.subgraph_rule == "whole_graph";
        };
        auto parse_switch = [](const std::string& value, const std::string& name) {
            if (value == "on" || value == "true" || value == "1") {
                return true;
            }
            if (value == "off" || value == "false" || value == "0") {
                return false;
            }
            throw std::runtime_error("invalid value for " + name + ": " + value);
        };
        auto normalize_compute_backend = [](const std::string& value) {
            if (value == "cpu") {
                return std::string("cpu");
            }
            if (value == "cuda_hotspots") {
                return std::string("cuda_hotspots");
            }
            throw std::runtime_error("unsupported --compute-backend: " + value);
        };
        auto set_process_env = [](const char* key, const char* value) {
            if (::setenv(key, value, 1) != 0) {
                throw std::runtime_error(std::string("failed to set environment override: ") + key);
            }
        };
        auto console_backend_label = [](const gatzk::protocol::RunMetrics& metrics) {
            return metrics.compute_backend_name == "cuda_hotspots" ? std::string("gpu") : std::string("cpu");
        };
        const bool progress_enabled = std::getenv("GATZK_RUN_PROGRESS") != nullptr;
        auto progress = [&](const std::string& label) {
            if (progress_enabled) {
                std::cerr << "[progress] " << label << '\n';
            }
        };

        std::string config_path;
        std::string benchmark_mode = "warm";
        std::string compute_backend = "cpu";
        std::string export_tag;
        gatzk::util::Route2Options route2;
        // 入口层只解析 formal 主线允许暴露的执行层开关，不在这里分叉协议语义。
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--config" && i + 1 < argc) {
                config_path = argv[++i];
            } else if (arg == "--benchmark-mode" && i + 1 < argc) {
                benchmark_mode = argv[++i];
            } else if (arg == "--compute-backend" && i + 1 < argc) {
                compute_backend = normalize_compute_backend(argv[++i]);
            } else if (arg == "--fast-msm" && i + 1 < argc) {
                route2.fast_msm = parse_switch(argv[++i], "--fast-msm");
            } else if (arg == "--parallel-fft" && i + 1 < argc) {
                route2.parallel_fft = parse_switch(argv[++i], "--parallel-fft");
            } else if (arg == "--fft-backend-upgrade" && i + 1 < argc) {
                route2.fft_backend_upgrade = parse_switch(argv[++i], "--fft-backend-upgrade");
            } else if (arg == "--fft-kernel-upgrade" && i + 1 < argc) {
                route2.fft_kernel_upgrade = parse_switch(argv[++i], "--fft-kernel-upgrade");
            } else if (arg == "--trace-layout-upgrade" && i + 1 < argc) {
                route2.trace_layout_upgrade = parse_switch(argv[++i], "--trace-layout-upgrade");
            } else if (arg == "--fast-verify-pairing" && i + 1 < argc) {
                route2.fast_verify_pairing = parse_switch(argv[++i], "--fast-verify-pairing");
            } else if (arg == "--export-tag" && i + 1 < argc) {
                export_tag = argv[++i];
            }
        }
        if (config_path.empty()) {
            throw std::runtime_error("usage: gatzk_run --config <path>");
        }
        if (benchmark_mode != "warm") {
            throw std::runtime_error("formal mainline only supports --benchmark-mode warm");
        }
        compute_backend = normalize_compute_backend(compute_backend);

        // 先确定执行层落点，再把 route2 选项整体交给后续 prove / verify 共用。
        set_process_env("GATZK_ALGEBRA_BACKEND", "cpu");
        route2.cuda_trace_hotspots = compute_backend == "cuda_hotspots";
        route2.cuda_field_hotspots = compute_backend == "cuda_hotspots";
        if (route2.cuda_trace_hotspots || route2.cuda_field_hotspots) {
            if (!gatzk::algebra::cuda_backend_build_enabled()) {
                throw std::runtime_error("compute backend cuda_hotspots requires a CUDA-enabled build");
            }
            if (!gatzk::algebra::cuda_backend_available()) {
                throw std::runtime_error("compute backend cuda_hotspots requires an available CUDA runtime");
            }
        }

        const auto loaded_config = gatzk::util::load_config(config_path);
        const auto absolute_config_path = std::filesystem::absolute(config_path).string();
        gatzk::util::set_route2_options(route2);

        // warm-up 与 measured run 共用这条正式执行入口，
        // 但只有 measured run 会输出正式单行结果。
        auto run_once = [&](
                            const gatzk::util::AppConfig& run_config,
                            bool is_cold_run,
                            const std::string& mode_label,
                            bool export_artifacts = true,
                            bool log_benchmark = true) {
            gatzk::protocol::RunMetrics metrics;
            metrics.crypto_backend_name = gatzk::crypto::backend_name();
            metrics.algebra_backend_name = gatzk::algebra::configured_algebra_backend_name();
            metrics.compute_backend_name = (route2.cuda_trace_hotspots || route2.cuda_field_hotspots) ? "cuda_hotspots" : "cpu";
            metrics.backend_name = metrics.crypto_backend_name + "/" + metrics.compute_backend_name;
            metrics.config = absolute_config_path;
            metrics.enabled_fast_msm = route2.fast_msm;
            metrics.enabled_parallel_fft = route2.parallel_fft;
            metrics.enabled_fft_backend_upgrade = route2.fft_backend_upgrade;
            metrics.enabled_fft_kernel_upgrade = route2.fft_kernel_upgrade;
            metrics.enabled_trace_layout_upgrade = route2.trace_layout_upgrade;
            metrics.enabled_fast_verify_pairing = route2.fast_verify_pairing;
            metrics.enabled_cuda_trace_hotspots = route2.cuda_trace_hotspots;
            metrics.enabled_cuda_field_hotspots = route2.cuda_field_hotspots;
            metrics.benchmark_mode = mode_label;
            metrics.route2_label = gatzk::util::route2_feature_label(route2);
            metrics.optimization_flags = gatzk::util::route2_feature_notes(route2);
            metrics.hardware_and_device =
                "cpu_threads=" + std::to_string(std::max(1U, std::thread::hardware_concurrency()))
                + "; " + gatzk::algebra::cuda_device_summary();
            metrics.gpu_runtime_present = gatzk::algebra::cuda_backend_available();
            metrics.gpu_acceleration_status = metrics.compute_backend_name == "cuda_hotspots"
                ? std::string("enabled_cuda_hotspots; accelerated_stages=barycentric_weights,packed_weighted_evaluation,lookup_histogram_size_aware; cpu_fallback=KZG_MSM,pairing,small_domains; runtime_present=")
                    + std::string(metrics.gpu_runtime_present ? "true" : "false")
                : std::string("not_requested_cpu_run");
            if (route2.fft_backend_upgrade && route2.fft_kernel_upgrade) {
                metrics.fft_backend_route = "packed_rotated_kernel";
            } else if (route2.fft_backend_upgrade) {
                metrics.fft_backend_route = "packed_domain";
            } else {
                metrics.fft_backend_route = "plain_domain";
            }
            metrics.is_cold_run = is_cold_run;
            append_note(metrics, "benchmark_mode=" + mode_label);
            append_note(metrics, "compute_backend=" + metrics.compute_backend_name);
            append_note(metrics, "full_backend=" + std::string(metrics.compute_backend_name == "cuda_hotspots" ? "full_gpu" : "full_cpu"));
            if (metrics.compute_backend_name == "cuda_hotspots") {
                append_note(metrics, "cuda_routing=size_aware");
            }
            append_note(metrics, gatzk::util::route2_feature_notes(route2));

            // 正式主线从这里开始：先构静态上下文，再物化 trace、生成 proof，最后复验并导出产物。
            progress(mode_label + ":context:start");
            const auto context_start = std::chrono::steady_clock::now();
            const auto context = gatzk::protocol::build_context(run_config, &metrics);
            const auto context_end = std::chrono::steady_clock::now();
            progress(mode_label + ":context:done");
            metrics.context_build_ms = std::chrono::duration<double, std::milli>(context_end - context_start).count();
            metrics.dataset = context.dataset.name;
            metrics.node_count = context.local.num_nodes;
            metrics.edge_count = context.local.edges.size();
            metrics.optimization_profile_id =
                context.dataset.name + ":" + console_backend_label(metrics) + ":" + metrics.route2_label
                + ":" + context.kzg.manifest.manifest_hash.substr(0, 12);
            metrics.is_full_dataset = context.local.num_nodes == context.dataset.num_nodes;
            if (!metrics.is_full_dataset) {
                throw std::runtime_error("formal mainline only supports full-dataset runs");
            }

            progress(mode_label + ":trace:start");
            const auto trace_start = std::chrono::steady_clock::now();
            const auto trace = gatzk::protocol::build_trace(context, &metrics);
            progress(mode_label + ":trace:done");
            progress(mode_label + ":prove:start");
            const auto pcs_start = std::chrono::steady_clock::now();
            const auto proof = gatzk::protocol::prove(context, trace, &metrics);
            const auto pcs_end = std::chrono::steady_clock::now();
            progress(mode_label + ":prove:done");
            metrics.lookup_trace_ms =
                metrics.lookup_table_pack_ms
                + metrics.lookup_query_pack_ms
                + metrics.lookup_key_build_ms
                + metrics.lookup_multiplicity_ms
                + metrics.lookup_accumulator_ms
                + metrics.lookup_state_machine_ms
                + metrics.lookup_selector_mask_ms
                + metrics.lookup_public_helper_ms
                + metrics.lookup_copy_convert_ms;
            metrics.hidden_head_trace_ms =
                metrics.hidden_projection_trace_ms
                + metrics.hidden_src_attention_trace_ms
                + metrics.hidden_dst_attention_trace_ms
                + metrics.hidden_edge_score_trace_ms
                + metrics.hidden_softmax_chain_trace_ms
                + metrics.hidden_h_star_trace_ms
                + metrics.hidden_h_agg_pre_star_trace_ms
                + metrics.hidden_h_agg_star_trace_ms
                + metrics.hidden_route_trace_ms
                + metrics.hidden_copy_convert_ms;
            metrics.trace_misc_ms =
                metrics.route_pack_residual_ms
                + metrics.selector_padding_residual_ms
                + metrics.public_poly_residual_ms
                + metrics.hidden_output_object_residual_ms
                + metrics.shared_helper_build_ms
                + metrics.field_conversion_residual_ms
                + metrics.copy_move_residual_ms
                + metrics.trace_finalize_ms;
            const auto prove_core_ms = std::chrono::duration<double, std::milli>(pcs_end - trace_start).count();
            metrics.prove_time_ms = prove_core_ms;
            metrics.commitment_time_ms =
                metrics.commit_dynamic_ms
                + metrics.quotient_bundle_pack_ms;
            metrics.prove_finalize_ms = clamp_non_negative(
                metrics.prove_time_ms
                - metrics.forward_ms
                - metrics.trace_generation_ms
                - metrics.commit_dynamic_ms
                - metrics.quotient_build_ms
                - metrics.domain_opening_ms
                - metrics.external_opening_ms);
            metrics.prove_accounted_ms = prove_accounted_ms(metrics);
            metrics.prove_accounting_gap_ms = metrics.prove_time_ms - metrics.prove_accounted_ms;
            progress(mode_label + ":verify:start");
            const auto verify_start = std::chrono::steady_clock::now();
            const auto accepted = gatzk::protocol::verify(context, proof, &metrics);
            const auto verify_end = std::chrono::steady_clock::now();
            progress(mode_label + ":verify:done");
            metrics.verify_time_ms = std::chrono::duration<double, std::milli>(verify_end - verify_start).count();
            metrics.verify_misc_ms = clamp_non_negative(
                metrics.verify_time_ms
                - metrics.verify_metadata_ms
                - metrics.verify_transcript_ms
                - metrics.verify_domain_opening_ms
                - metrics.verify_quotient_ms
                - metrics.verify_external_fold_ms);
            metrics.verify_accounted_ms = verify_accounted_ms(metrics);
            metrics.verify_accounting_gap_ms = metrics.verify_time_ms - metrics.verify_accounted_ms;
            metrics.proof_size_bytes = gatzk::protocol::proof_size_bytes(proof);

            if (log_benchmark) {
                std::cout
                    << "dataset=" << metrics.dataset
                    << " config=" << config_path
                    << " backend=" << console_backend_label(metrics)
                    << " status=" << (accepted ? "VERIFY_OK" : "VERIFY_FAIL")
                    << " commitment_time_ms=" << format_ms(metrics.commitment_time_ms)
                    << " prove_time_ms=" << format_ms(metrics.prove_time_ms)
                    << " verify_time_ms=" << format_ms(metrics.verify_time_ms)
                    << " proof_size_bytes=" << std::to_string(metrics.proof_size_bytes)
                    << '\n';
            }
            if (export_artifacts) {
                progress(mode_label + ":export:start");
                gatzk::protocol::export_run_artifacts(context, trace, proof, metrics, accepted);
                progress(mode_label + ":export:done");
            }
            return accepted;
        };

        auto warm_up_context = [&](const gatzk::util::AppConfig& run_config, const std::string& label) {
            progress(label + ":context:start");
            const auto context = gatzk::protocol::build_context(run_config, nullptr);
            (void)context;
            progress(label + ":context:done");
        };

        bool accepted = false;
        std::vector<std::string> export_segments;
        if (!export_tag.empty()) {
            export_segments.push_back(export_tag);
        }
        // warm-up 只预热执行层缓存和大对象初始化，不写正式 benchmark 结论。
        if (memory_safe_warm_required(loaded_config)) {
            warm_up_context(with_export_suffix(loaded_config, export_segments), "light-warmup");
        } else {
            warm_up_context(with_export_suffix(loaded_config, export_segments), "warmup");
        }
        auto segments = export_segments;
        segments.push_back("warm");
        // measured run 固定写到 warm 目录下，作为 formal 主表与 manifest 的真源。
        accepted = run_once(with_export_suffix(loaded_config, segments), false, "warm");
        return accepted ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
