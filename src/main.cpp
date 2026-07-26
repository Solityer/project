#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/crypto/curve.hpp"
#include "gatzk/protocol/trace.hpp"
#include "gatzk/protocol/prover.hpp"
#include "gatzk/protocol/proof_codec.hpp"
#include "gatzk/protocol/verifier.hpp"
#include "gatzk/util/config.hpp"
#include "gatzk/util/route2.hpp"

// 读取配置、选择 CPU / partial CUDA-hotspot 执行路线，并串联 context、trace、prove、verify 与 artifact 导出。
// 对外只汇总三项正式时间与 canonical proof bytes 大小，不在入口层改写 proof contract。
// 控制台正式输出必须收口为“每次运行一行”，避免把开发过程日志重新暴露到提交表面。

int main(int argc, char **argv)
{
    try
    {
        auto format_ms = [](double value)
        {
            std::ostringstream stream;
            stream << std::fixed << std::setprecision(3) << value;
            return stream.str();
        };
        auto clamp_non_negative = [](double value)
        {
            return value < 0.0 ? 0.0 : value;
        };
        auto prove_accounted_ms = [](const gatzk::protocol::RunMetrics &metrics)
        {
            return metrics.forward_ms
                + std::max(
                    0.0,
                    metrics.trace_generation_ms
                        - metrics.derived_commitment_point_ms)
                + std::max(
                    0.0,
                    metrics.commit_dynamic_ms
                        - metrics.dynamic_commit_msm_ms)
                + std::max(
                    0.0,
                    metrics.quotient_build_ms
                        - metrics.quotient_commitment_ms)
                + metrics.domain_opening_ms + metrics.external_opening_ms
                + metrics.proof_serialization_ms + metrics.prove_finalize_ms;
        };
        auto verify_accounted_ms = [](const gatzk::protocol::RunMetrics &metrics)
        {
            return metrics.proof_parse_ms + metrics.verify_metadata_ms
                + metrics.verify_transcript_ms + metrics.verify_domain_opening_ms
                + metrics.verify_quotient_ms + metrics.verify_external_opening_ms
                + metrics.verify_misc_ms;
        };
        auto append_note = [](gatzk::protocol::RunMetrics &metrics, const std::string &note)
        {
            if (note.empty())
            {
                return;
            }
            if (!metrics.notes.empty())
            {
                metrics.notes += "; ";
            }
            metrics.notes += note;
        };
        auto with_export_suffix = [](const gatzk::util::AppConfig &base, const std::vector<std::string> &suffixes)
        {
            auto updated = base;
            auto export_dir = std::filesystem::path(base.export_dir);
            for (const auto &suffix : suffixes)
            {
                if (!suffix.empty())
                {
                    export_dir /= suffix;
                }
            }
            updated.export_dir = export_dir.string();
            return updated;
        };
        // 全图实例先把大对象初始化和正式测量拆开，避免冷启动污染 warm 主表。
        auto memory_safe_warm_required = [](const gatzk::util::AppConfig &config)
        {
            return config.batching_rule == "whole_graph_single" && config.subgraph_rule == "whole_graph";
        };
        auto parse_switch = [](const std::string &value, const std::string &name)
        {
            if (value == "on" || value == "true" || value == "1")
            {
                return true;
            }
            if (value == "off" || value == "false" || value == "0")
            {
                return false;
            }
            throw std::runtime_error("invalid value for " + name + ": " + value);
        };
        auto normalize_compute_backend = [](const std::string &value)
        {
            if (value == "cpu")
            {
                return std::string("cpu");
            }
            if (value == "cuda_hotspots")
            {
                return std::string("cuda_hotspots");
            }
            throw std::runtime_error("unsupported --compute-backend: " + value);
        };
        auto set_process_env = [](const char *key, const char *value)
        {
            if (::setenv(key, value, 1) != 0)
            {
                throw std::runtime_error(std::string("failed to set environment override: ") + key);
            }
        };
        auto console_backend_label = [](const gatzk::protocol::RunMetrics &metrics)
        {
            return metrics.compute_backend_name == "cuda_hotspots" ? std::string("gpu") : std::string("cpu");
        };
        const bool progress_enabled = std::getenv("GATZK_RUN_PROGRESS") != nullptr;
        auto progress = [&](const std::string &label)
        {
            if (progress_enabled)
            {
                std::cerr << "[progress] " << label << '\n';
            }
        };

        std::string config_path;
        std::string benchmark_mode = "warm";
        std::string compute_backend = "cpu";
        std::string export_tag;
        gatzk::util::Route2Options route2;
        std::set<std::string> cli_optimization_overrides;
        auto require_option_value = [&](int &index, const std::string &option)
        {
            if (index + 1 >= argc)
            {
                throw std::runtime_error("missing value for " + option);
            }
            const std::string_view value(argv[index + 1]);
            if (value.empty() || value.starts_with("--"))
            {
                throw std::runtime_error("missing value for " + option);
            }
            return std::string(argv[++index]);
        };
        // 入口层只解析 formal 主线允许暴露的执行层开关，不在这里分叉协议语义。
        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            if (arg == "--config")
            {
                config_path = require_option_value(i, arg);
            }
            else if (arg == "--benchmark-mode")
            {
                benchmark_mode = require_option_value(i, arg);
            }
            else if (arg == "--compute-backend")
            {
                compute_backend = normalize_compute_backend(
                    require_option_value(i, arg));
            }
            else if (arg == "--fast-msm")
            {
                route2.fast_msm = parse_switch(
                    require_option_value(i, arg), arg);
                cli_optimization_overrides.insert("fast_msm");
            }
            else if (arg == "--parallel-fft")
            {
                route2.parallel_fft = parse_switch(
                    require_option_value(i, arg), arg);
                cli_optimization_overrides.insert("parallel_fft");
            }
            else if (arg == "--fft-backend-upgrade")
            {
                route2.fft_backend_upgrade = parse_switch(
                    require_option_value(i, arg), arg);
                cli_optimization_overrides.insert("fft_backend_upgrade");
            }
            else if (arg == "--fft-kernel-upgrade")
            {
                route2.fft_kernel_upgrade = parse_switch(
                    require_option_value(i, arg), arg);
                cli_optimization_overrides.insert("fft_kernel_upgrade");
            }
            else if (arg == "--trace-layout-upgrade")
            {
                route2.trace_layout_upgrade = parse_switch(
                    require_option_value(i, arg), arg);
                cli_optimization_overrides.insert("trace_layout_upgrade");
            }
            else if (arg == "--fast-verify-pairing")
            {
                route2.fast_verify_pairing = parse_switch(
                    require_option_value(i, arg), arg);
                cli_optimization_overrides.insert("fast_verify_pairing");
            }
            else if (arg == "--export-tag")
            {
                export_tag = require_option_value(i, arg);
            }
            else
            {
                throw std::runtime_error("unknown argument: " + arg);
            }
        }
        if (config_path.empty())
        {
            throw std::runtime_error("usage: gatzk_run --config <path>");
        }
        if (benchmark_mode != "warm")
        {
            throw std::runtime_error("formal mainline only supports --benchmark-mode warm");
        }
        if (!export_tag.empty())
        {
            const bool valid_export_tag =
                export_tag.size() <= 96
                && export_tag != "."
                && export_tag != ".."
                && std::all_of(
                    export_tag.begin(), export_tag.end(),
                    [](unsigned char value)
                    {
                        return (value >= 'a' && value <= 'z')
                            || (value >= 'A' && value <= 'Z')
                            || (value >= '0' && value <= '9')
                            || value == '_' || value == '-' || value == '.';
                    });
            if (!valid_export_tag)
            {
                throw std::runtime_error(
                    "invalid value for --export-tag: expected one safe path component");
            }
        }
        compute_backend = normalize_compute_backend(compute_backend);

        // 先确定执行层落点，再把 route2 选项整体交给后续 prove / verify 共用。
        set_process_env("GATZK_ALGEBRA_BACKEND", "cpu");
        route2.cuda_trace_hotspots = compute_backend == "cuda_hotspots";
        route2.cuda_field_hotspots = compute_backend == "cuda_hotspots";
        if (route2.cuda_trace_hotspots || route2.cuda_field_hotspots)
        {
            if (!gatzk::algebra::cuda_backend_build_enabled())
            {
                throw std::runtime_error("compute backend cuda_hotspots requires a CUDA-enabled build");
            }
            if (!gatzk::algebra::cuda_backend_available())
            {
                throw std::runtime_error("compute backend cuda_hotspots requires an available CUDA runtime");
            }
        }
        auto optimization_state_notes = [&]()
        {
            std::vector<std::string> entries;
            auto append = [&](const std::string &name, bool enabled,
                              bool applicable, const std::string &source)
            {
                entries.push_back(
                    name + "_state="
                    + (applicable ? (enabled ? "on" : "off") : "not_applicable")
                    + "," + name + "_source=" + source
                    + "," + name + "_applicability="
                    + (applicable ? "applicable" : "not_applicable"));
            };
            auto source = [&](const std::string &name)
            {
                return cli_optimization_overrides.contains(name)
                    ? std::string("cli_override")
                    : std::string("compile_default");
            };
            append("fast_msm", route2.fast_msm, true, source("fast_msm"));
            append("parallel_fft", route2.parallel_fft, true, source("parallel_fft"));
            append(
                "fft_backend_upgrade", route2.fft_backend_upgrade, true,
                source("fft_backend_upgrade"));
            append(
                "fft_kernel_upgrade", route2.fft_kernel_upgrade, true,
                source("fft_kernel_upgrade"));
            append(
                "trace_layout_upgrade", route2.trace_layout_upgrade, true,
                source("trace_layout_upgrade"));
            append(
                "fast_verify_pairing", route2.fast_verify_pairing, true,
                source("fast_verify_pairing"));
            const bool cuda_applicable = compute_backend == "cuda_hotspots";
            append(
                "cuda_trace_hotspots", route2.cuda_trace_hotspots,
                cuda_applicable, "backend_selection");
            append(
                "cuda_field_hotspots", route2.cuda_field_hotspots,
                cuda_applicable, "backend_selection");
            std::ostringstream out;
            for (std::size_t index = 0; index < entries.size(); ++index)
            {
                if (index != 0)
                {
                    out << "; ";
                }
                out << entries[index];
            }
            return out.str();
        };

        const auto loaded_config = gatzk::util::load_config(config_path);
        const auto absolute_config_path = std::filesystem::absolute(config_path).string();
        gatzk::util::set_route2_options(route2);

        // warm-up 与 measured run 共用这条正式执行入口，
        // 但只有 measured run 会输出正式单行结果。
        auto run_once = [&](
                            const gatzk::util::AppConfig &run_config,
                            const std::string &mode_label,
                            bool export_artifacts = true,
                            bool log_benchmark = true)
        {
            const auto row_start = std::chrono::steady_clock::now();
            const auto cuda_kernel_count_at_start =
                gatzk::algebra::cuda_kernel_invocation_count();
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
            metrics.gpu_backend_requested = metrics.compute_backend_name == "cuda_hotspots";
            metrics.benchmark_mode = mode_label;
            metrics.route2_label = gatzk::util::route2_feature_label(route2);
            metrics.optimization_flags = optimization_state_notes();
            metrics.hardware_and_device =
                "cpu_threads=" + std::to_string(std::max(1U, std::thread::hardware_concurrency())) + "; " + gatzk::algebra::cuda_device_summary();
            metrics.gpu_runtime_present = gatzk::algebra::cuda_backend_available();
            metrics.gpu_device = gatzk::algebra::cuda_device_summary();
            metrics.gpu_used_for = metrics.compute_backend_name == "cuda_hotspots"
                                       ? "barycentric_weights,packed_weighted_evaluation,lookup_histogram_size_aware"
                                       : "none";
            metrics.gpu_fallback_stages = metrics.compute_backend_name == "cuda_hotspots"
                                              ? "KZG_MSM,pairing,NTT_IFFT,quotient_materialization,opening_witness,small_domains"
                                              : "not_requested";
            metrics.gpu_fallback_reason = metrics.compute_backend_name == "cuda_hotspots"
                                              ? "current CUDA backend covers Fr algebra hotspots only; KZG curve and quotient kernels remain CPU"
                                              : "cpu backend run";
            metrics.gpu_acceleration_status = metrics.compute_backend_name == "cuda_hotspots"
                                                  ? std::string("enabled_cuda_hotspots; accelerated_stages=barycentric_weights,packed_weighted_evaluation,lookup_histogram_size_aware; cpu_fallback=KZG_MSM,pairing,small_domains; runtime_present=") + std::string(metrics.gpu_runtime_present ? "true" : "false")
                                                  : std::string("not_requested_cpu_run");
            if (route2.fft_backend_upgrade && route2.fft_kernel_upgrade)
            {
                metrics.fft_backend_route = "packed_rotated_kernel";
            }
            else if (route2.fft_backend_upgrade)
            {
                metrics.fft_backend_route = "packed_domain";
            }
            else
            {
                metrics.fft_backend_route = "plain_domain";
            }
            append_note(metrics, "benchmark_mode=" + mode_label);
            append_note(metrics, "compute_backend=" + metrics.compute_backend_name);
            append_note(
                metrics,
                "execution_scope="
                    + std::string(
                        metrics.compute_backend_name == "cuda_hotspots"
                            ? "partial_cuda_hotspots_with_cpu_fallback"
                            : "full_cpu"));
            if (metrics.compute_backend_name == "cuda_hotspots")
            {
                append_note(metrics, "cuda_routing=size_aware");
            }
            append_note(metrics, metrics.optimization_flags);

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
                context.dataset.name + ":" + console_backend_label(metrics) + ":" + metrics.route2_label + ":" + context.kzg.manifest.manifest_hash.substr(0, 12);
            metrics.is_full_dataset = context.local.num_nodes == context.dataset.num_nodes;
            if (!metrics.is_full_dataset)
            {
                throw std::runtime_error("formal mainline only supports full-dataset runs");
            }

            progress(mode_label + ":trace:start");
            const auto trace_start = std::chrono::steady_clock::now();
            const auto trace = gatzk::protocol::build_trace(context, &metrics);
            progress(mode_label + ":trace:done");
            metrics.reference_forward_ms = metrics.forward_ms;
            progress(mode_label + ":prove:start");
            const auto proof = gatzk::protocol::prove(context, trace, &metrics);
            const auto serialization_start = std::chrono::steady_clock::now();
            const auto proof_bytes = gatzk::protocol::serialize_proof(proof);
            metrics.proof_serialization_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - serialization_start).count();
            const auto prover_pipeline_end = std::chrono::steady_clock::now();
            progress(mode_label + ":prove:done");
            metrics.lookup_trace_ms =
                metrics.lookup_table_pack_ms + metrics.lookup_query_pack_ms + metrics.lookup_key_build_ms + metrics.lookup_multiplicity_ms + metrics.lookup_accumulator_ms + metrics.lookup_state_machine_ms + metrics.lookup_selector_mask_ms + metrics.lookup_public_helper_ms + metrics.lookup_copy_convert_ms;
            metrics.hidden_head_trace_ms =
                metrics.hidden_projection_trace_ms + metrics.hidden_src_attention_trace_ms + metrics.hidden_dst_attention_trace_ms + metrics.hidden_edge_score_trace_ms + metrics.hidden_softmax_chain_trace_ms + metrics.hidden_h_star_trace_ms + metrics.hidden_h_agg_pre_star_trace_ms + metrics.hidden_h_agg_star_trace_ms + metrics.hidden_route_trace_ms + metrics.hidden_copy_convert_ms;
            metrics.trace_misc_ms =
                metrics.route_pack_residual_ms + metrics.selector_padding_residual_ms + metrics.public_poly_residual_ms + metrics.hidden_output_object_residual_ms + metrics.shared_helper_build_ms + metrics.field_conversion_residual_ms + metrics.copy_move_residual_ms + metrics.trace_finalize_ms;
            // build_trace and prove interleave ordinary prover work with commitment
            // point generation.  Measure the contiguous formal prover pipeline once,
            // then partition it into two mutually exclusive metrics.
            metrics.commitment_time_ms =
                metrics.dynamic_commit_msm_ms
                + metrics.derived_commitment_point_ms
                + metrics.quotient_commitment_ms;
            const auto prover_pipeline_ms =
                std::chrono::duration<double, std::milli>(
                    prover_pipeline_end - trace_start).count();
            if (metrics.commitment_time_ms < 0.0
                || metrics.commitment_time_ms > prover_pipeline_ms + 1.0)
            {
                throw std::runtime_error(
                    "formal commitment timing escaped the prover pipeline");
            }
            metrics.prove_time_ms = clamp_non_negative(
                prover_pipeline_ms - metrics.commitment_time_ms);
            metrics.prove_finalize_ms = 0.0;
            const auto prove_without_finalize_ms = prove_accounted_ms(metrics);
            const auto prove_overlap_tolerance_ms =
                std::max(1.0, metrics.prove_time_ms * 0.02);
            if (prove_without_finalize_ms
                > metrics.prove_time_ms + prove_overlap_tolerance_ms)
            {
                throw std::runtime_error(
                    "nested prover timers overlap the exclusive formal timing boundary");
            }
            metrics.prove_finalize_ms = clamp_non_negative(
                metrics.prove_time_ms - prove_without_finalize_ms);
            metrics.prove_accounted_ms = prove_accounted_ms(metrics);
            metrics.prove_accounting_gap_ms = metrics.prove_time_ms - metrics.prove_accounted_ms;
            progress(mode_label + ":verify:start");
            const auto verify_start = std::chrono::steady_clock::now();
            const auto parse_start = std::chrono::steady_clock::now();
            const auto parsed_proof = gatzk::protocol::parse_proof(context, proof_bytes);
            metrics.proof_parse_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - parse_start).count();
            const auto accepted = gatzk::protocol::verify(context, parsed_proof, &metrics);
            const auto verify_end = std::chrono::steady_clock::now();
            progress(mode_label + ":verify:done");
            metrics.verify_time_ms = std::chrono::duration<double, std::milli>(verify_end - verify_start).count();
            metrics.verify_misc_ms = clamp_non_negative(
                metrics.verify_time_ms - metrics.proof_parse_ms
                - metrics.verify_metadata_ms - metrics.verify_transcript_ms
                - metrics.verify_domain_opening_ms - metrics.verify_quotient_ms
                - metrics.verify_external_opening_ms);
            metrics.verify_accounted_ms = verify_accounted_ms(metrics);
            metrics.verify_accounting_gap_ms = metrics.verify_time_ms - metrics.verify_accounted_ms;
            metrics.proof_size_bytes = proof_bytes.size();
            const auto audit_start = std::chrono::steady_clock::now();
            const auto private_plain_count = gatzk::protocol::private_plain_value_count(context, parsed_proof);
            const auto audit_end = std::chrono::steady_clock::now();
            metrics.payload_audit_ms = std::chrono::duration<double, std::milli>(audit_end - audit_start).count();
            const auto reference_start = std::chrono::steady_clock::now();
            const bool reference_consistent =
                !trace.reference_quant_logits_hash.empty()
                && trace.reference_quant_logits_hash == trace.secure_trace_logits_hash
                && trace.public_logits == parsed_proof.public_logits
                && private_plain_count == 0;
            metrics.cpu_gpu_reference_consistency_passed = reference_consistent;
            const auto reference_end = std::chrono::steady_clock::now();
            metrics.reference_consistency_ms =
                std::chrono::duration<double, std::milli>(reference_end - reference_start).count();
            const auto tamper_start = std::chrono::steady_clock::now();
            auto tampered = proof;
            tampered.public_metadata.privacy_policy = "tampered_privacy_policy";
            const bool tamper_rejected = !gatzk::protocol::verify(context, tampered, nullptr);
            const auto tamper_end = std::chrono::steady_clock::now();
            metrics.tamper_tests_ms = std::chrono::duration<double, std::milli>(tamper_end - tamper_start).count();
            if (!reference_consistent)
            {
                throw std::runtime_error(
                    "reference consistency check failed"
                    " (reference_hash_present="
                    + std::string(
                        trace.reference_quant_logits_hash.empty()
                            ? "false" : "true")
                    + ", reference_matches_secure="
                    + std::string(
                        trace.reference_quant_logits_hash
                                == trace.secure_trace_logits_hash
                            ? "true" : "false")
                    + ", proof_output_matches_trace="
                    + std::string(
                        trace.public_logits == parsed_proof.public_logits
                            ? "true" : "false")
                    + ", private_plain_count="
                    + std::to_string(private_plain_count) + ")");
            }
            if (!tamper_rejected)
            {
                throw std::runtime_error("tamper test was not rejected");
            }
            metrics.total_wall_clock_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - row_start).count();
            metrics.formal_time_sum_ms =
                metrics.commitment_time_ms
                + metrics.prove_time_ms
                + metrics.verify_time_ms;
            metrics.non_formal_overhead_ms = clamp_non_negative(
                metrics.total_wall_clock_ms - metrics.formal_time_sum_ms);
            const auto cuda_kernel_count_at_end =
                gatzk::algebra::cuda_kernel_invocation_count();
            if (cuda_kernel_count_at_end < cuda_kernel_count_at_start)
            {
                throw std::runtime_error(
                    "CUDA kernel invocation counter moved backwards");
            }
            metrics.gpu_kernel_invocation_count =
                cuda_kernel_count_at_end - cuda_kernel_count_at_start;
            if (metrics.compute_backend_name == "cuda_hotspots")
            {
                if (metrics.gpu_kernel_invocation_count == 0)
                {
                    throw std::runtime_error(
                        "compute backend cuda_hotspots executed zero CUDA kernels "
                        "during the measured run; refusing to label CPU fallback as GPU");
                }
                else
                {
                    metrics.gpu_used_for =
                        "partial_size_aware_hotspots; kernel_invocations="
                        + std::to_string(metrics.gpu_kernel_invocation_count);
                    metrics.gpu_acceleration_status =
                        "partial_cuda_hotspots_executed; kernels_executed="
                        + std::to_string(metrics.gpu_kernel_invocation_count)
                        + "; cpu_fallback=KZG_MSM,pairing,NTT_IFFT,quotient_materialization,opening_witness,small_domains";
                    append_note(metrics, "cuda_kernel_execution=partial_hotspots_only");
                }
            }

            if (log_benchmark)
            {
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
            if (export_artifacts)
            {
                progress(mode_label + ":export:start");
                gatzk::protocol::export_run_artifacts(context, trace, proof, metrics, accepted);
                progress(mode_label + ":export:done");
            }
            return accepted;
        };

        auto warm_up_context = [&](const gatzk::util::AppConfig &run_config, const std::string &label)
        {
            progress(label + ":context:start");
            const auto context = gatzk::protocol::build_context(run_config, nullptr);
            (void)context;
            progress(label + ":context:done");
        };

        bool accepted = false;
        std::vector<std::string> export_segments;
        if (!export_tag.empty())
        {
            export_segments.push_back(export_tag);
        }
        // warm-up 只预热执行层缓存和大对象初始化，不写正式 benchmark 结论。
        if (memory_safe_warm_required(loaded_config))
        {
            warm_up_context(with_export_suffix(loaded_config, export_segments), "light-warmup");
        }
        else
        {
            warm_up_context(with_export_suffix(loaded_config, export_segments), "warmup");
        }
        auto segments = export_segments;
        segments.push_back("warm");
        // measured run 固定写到 warm 目录下，作为 formal 主表与 manifest 的真源。
        accepted = run_once(with_export_suffix(loaded_config, segments), "warm");
        return accepted ? 0 : 1;
    }
    catch (const std::exception &error)
    {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
