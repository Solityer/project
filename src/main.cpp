#include <cstdlib>
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <filesystem>
#include <functional>
#include <iostream>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/crypto/curve.hpp"
#include "gatzk/protocol/trace.hpp"
#include "gatzk/protocol/challenges.hpp"
#include "gatzk/protocol/prover.hpp"
#include "gatzk/protocol/proof_codec.hpp"
#include "gatzk/protocol/verifier.hpp"
#include "gatzk/protocol/verifier_context.hpp"
#include "gatzk/util/config.hpp"
#include "gatzk/util/route2.hpp"

// 读取配置、选择 CPU / partial CUDA-hotspot 执行路线，并串联 context、trace、prove、verify 与 artifact 导出。
// 对外只汇总三项正式时间与 canonical proof bytes 大小，不在入口层改写 proof contract。
// 控制台正式输出必须收口为“每次运行一行”，避免把开发过程日志重新暴露到提交表面。

namespace
{
    class ExclusiveFormalRunLock
    {
    public:
        explicit ExclusiveFormalRunLock(
            const std::filesystem::path &path)
        {
            std::filesystem::create_directories(
                path.parent_path());
            descriptor_ = ::open(
                path.c_str(),
                O_CREAT | O_CLOEXEC | O_RDWR,
                0644);
            if (descriptor_ < 0)
            {
                throw std::system_error(
                    errno,
                    std::generic_category(),
                    "failed to open formal run lock");
            }
            if (::flock(
                    descriptor_,
                    LOCK_EX | LOCK_NB) != 0)
            {
                const auto lock_error = errno;
                ::close(descriptor_);
                descriptor_ = -1;
                if (lock_error == EWOULDBLOCK || lock_error == EAGAIN)
                {
                    throw std::runtime_error(
                        "another gatzk_run is active in this workspace; "
                        "formal runs must execute sequentially");
                }
                throw std::system_error(
                    lock_error,
                    std::generic_category(),
                    "failed to acquire formal run lock");
            }
            const auto owner =
                "pid=" + std::to_string(::getpid()) + "\n";
            const auto truncate_ok =
                ::ftruncate(descriptor_, 0) == 0;
            const auto written = truncate_ok
                                     ? ::write(
                                           descriptor_,
                                           owner.data(),
                                           owner.size())
                                     : static_cast<ssize_t>(-1);
            if (!truncate_ok || written != static_cast<ssize_t>(owner.size()))
            {
                const auto record_error =
                    errno != 0 ? errno : EIO;
                (void)::flock(descriptor_, LOCK_UN);
                (void)::close(descriptor_);
                descriptor_ = -1;
                throw std::system_error(
                    record_error,
                    std::generic_category(),
                    "failed to record formal run lock owner");
            }
        }

        ExclusiveFormalRunLock(
            const ExclusiveFormalRunLock &) = delete;
        ExclusiveFormalRunLock &operator=(
            const ExclusiveFormalRunLock &) = delete;

        ~ExclusiveFormalRunLock()
        {
            if (descriptor_ >= 0)
            {
                (void)::flock(descriptor_, LOCK_UN);
                (void)::close(descriptor_);
            }
        }

    private:
        int descriptor_ = -1;
    };

    class ScopedFormalTimingDiagnosticsSuppression
    {
    public:
        explicit ScopedFormalTimingDiagnosticsSuppression(
            bool enabled)
            : enabled_(
                  enabled &&
                  std::getenv(
                      "GATZK_ALLOW_TIMED_PROGRESS") ==
                      nullptr)
        {
            if (!enabled_)
            {
                return;
            }

            capture_and_unset(
                "GATZK_RUN_PROGRESS",
                &had_progress_,
                &progress_value_);

            capture_and_unset(
                "GATZK_DETAILED_PROFILE",
                &had_detailed_profile_,
                &detailed_profile_value_);
        }

        ScopedFormalTimingDiagnosticsSuppression(
            const ScopedFormalTimingDiagnosticsSuppression &) =
            delete;

        ScopedFormalTimingDiagnosticsSuppression &
        operator=(
            const ScopedFormalTimingDiagnosticsSuppression &) =
            delete;

        ~ScopedFormalTimingDiagnosticsSuppression()
        {
            restore();
        }

        void restore() noexcept
        {
            if (!enabled_ || restored_)
            {
                return;
            }

            restore_one(
                "GATZK_RUN_PROGRESS",
                had_progress_,
                progress_value_);

            restore_one(
                "GATZK_DETAILED_PROFILE",
                had_detailed_profile_,
                detailed_profile_value_);

            restored_ = true;
        }

    private:
        static void capture_and_unset(
            const char *name,
            bool *present,
            std::string *value)
        {
            const auto *current =
                std::getenv(name);

            if (current == nullptr)
            {
                return;
            }

            *present = true;
            *value = current;

            if (::unsetenv(name) != 0)
            {
                throw std::system_error(
                    errno,
                    std::generic_category(),
                    std::string(
                        "failed to suppress timed diagnostic environment: ") +
                        name);
            }
        }

        static void restore_one(
            const char *name,
            bool present,
            const std::string &value) noexcept
        {
            if (!present)
            {
                return;
            }

            (void)::setenv(
                name,
                value.c_str(),
                1);
        }

        bool enabled_ = false;
        bool restored_ = false;

        bool had_progress_ = false;
        std::string progress_value_;

        bool had_detailed_profile_ = false;
        std::string detailed_profile_value_;
    };
}

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
        auto verify_accounted_ms = [](const gatzk::protocol::RunMetrics &metrics)
        {
            return metrics.proof_parse_ms + metrics.verify_metadata_ms + metrics.verify_transcript_ms + metrics.verify_domain_opening_ms + metrics.verify_point_group_registry_ms + metrics.verify_quotient_ms + metrics.verify_external_opening_ms + metrics.verify_misc_ms;
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
        auto progress = [&](const std::string &label)
        {
            if (std::getenv(
                    "GATZK_RUN_PROGRESS") != nullptr)
            {
                std::cerr
                    << "[progress] "
                    << label
                    << '\n';
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
            else if (arg == "--packed-fft-backend")
            {
                route2.packed_fft_backend = parse_switch(
                    require_option_value(i, arg), arg);
                cli_optimization_overrides.insert("packed_fft_backend");
            }
            else if (arg == "--rotated-fft-kernel")
            {
                route2.rotated_fft_kernel = parse_switch(
                    require_option_value(i, arg), arg);
                cli_optimization_overrides.insert("rotated_fft_kernel");
            }
            else if (arg == "--compact-trace-layout")
            {
                route2.compact_trace_layout = parse_switch(
                    require_option_value(i, arg), arg);
                cli_optimization_overrides.insert("compact_trace_layout");
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
                export_tag.size() <= 96 && export_tag != "." && export_tag != ".." && std::all_of(export_tag.begin(), export_tag.end(), [](unsigned char value)
                                                                                                  { return (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') || (value >= '0' && value <= '9') || value == '_' || value == '-' || value == '.'; });
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
                    name + "_state=" + (applicable ? (enabled ? "on" : "off") : "not_applicable") + "," + name + "_source=" + source + "," + name + "_applicability=" + (applicable ? "applicable" : "not_applicable"));
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
                "packed_fft_backend", route2.packed_fft_backend, true,
                source("packed_fft_backend"));
            append(
                "rotated_fft_kernel", route2.rotated_fft_kernel, true,
                source("rotated_fft_kernel"));
            append(
                "compact_trace_layout", route2.compact_trace_layout, true,
                source("compact_trace_layout"));
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
        const ExclusiveFormalRunLock formal_run_lock(
            std::filesystem::current_path() / "runs" / ".gatzk_run.lock");
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
            ScopedFormalTimingDiagnosticsSuppression
                timed_diagnostics_suppression(
                    log_benchmark);

            const auto row_start =
                std::chrono::steady_clock::now();
            const auto cuda_kernel_count_at_start =
                gatzk::algebra::cuda_kernel_invocation_count();

            const auto cuda_stats_at_start =
                gatzk::algebra::
                    cuda_execution_stats_snapshot();

            gatzk::protocol::RunMetrics metrics;
            metrics.crypto_backend_name = gatzk::crypto::backend_name();
            metrics.algebra_backend_name = gatzk::algebra::configured_algebra_backend_name();
            metrics.compute_backend_name = (route2.cuda_trace_hotspots || route2.cuda_field_hotspots) ? "cuda_hotspots" : "cpu";
            metrics.backend_name = metrics.crypto_backend_name + "/" + metrics.compute_backend_name;
            metrics.config = absolute_config_path;
            metrics.enabled_fast_msm = route2.fast_msm;
            metrics.enabled_parallel_fft = route2.parallel_fft;
            metrics.enabled_packed_fft_backend = route2.packed_fft_backend;
            metrics.enabled_rotated_fft_kernel = route2.rotated_fft_kernel;
            metrics.enabled_compact_trace_layout = route2.compact_trace_layout;
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
            metrics.gpu_used_for =
                metrics.compute_backend_name == "cuda_hotspots"
                    ? "runtime_measured"
                    : "none";
            metrics.gpu_fallback_stages =
                metrics.compute_backend_name == "cuda_hotspots"
                    ? "runtime_measured"
                    : "not_requested";

            metrics.gpu_fallback_reason =
                metrics.compute_backend_name == "cuda_hotspots"
                    ? "workload_aware_hybrid_routing"
                    : "cpu backend run";

            metrics.gpu_acceleration_status =
                metrics.compute_backend_name == "cuda_hotspots"
                    ? "runtime_measured"
                    : "not_requested_cpu_run";
            if (route2.packed_fft_backend && route2.rotated_fft_kernel)
            {
                metrics.fft_backend_route = "packed_rotated_kernel";
            }
            else if (route2.packed_fft_backend)
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
                "execution_scope=" + std::string(
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
            if (context.compiler_artifact == nullptr)
                throw std::runtime_error(
                    "formal run is missing its compiler artifact");
            const auto security_budget =
                gatzk::protocol::evaluate_security_budget(
                    context, *context.compiler_artifact);
            metrics.security_budget_passed = security_budget.passed;
            metrics.security_budget_ledger_digest = security_budget.ledger_digest;
            metrics.security_budget_exact = security_budget.exact_budget;
            if (!security_budget.passed)
                throw std::runtime_error(
                    "formal security budget ledger reconstruction failed");
            metrics.is_full_dataset = context.local.num_nodes == context.dataset.num_nodes;
            if (!metrics.is_full_dataset)
            {
                throw std::runtime_error("formal mainline only supports full-dataset runs");
            }

            progress(mode_label + ":trace:start");
            const auto trace_start = std::chrono::steady_clock::now();
            auto trace = gatzk::protocol::build_trace(context, &metrics);
            const auto trace_elapsed_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - trace_start)
                    .count();
            metrics.lookup_trace_ms =
                metrics.lookup_table_pack_ms + metrics.lookup_query_pack_ms + metrics.lookup_key_build_ms + metrics.lookup_multiplicity_ms + metrics.lookup_inverse_ms + metrics.lookup_accumulator_ms + metrics.lookup_state_machine_ms + metrics.lookup_selector_mask_ms + metrics.lookup_public_helper_ms + metrics.lookup_copy_convert_ms;
            const auto trace_other_ms = clamp_non_negative(
                metrics.trace_generation_ms - metrics.hidden_head_trace_ms - metrics.output_head_trace_ms);
            progress(
                mode_label + ":trace:done:ms=" + format_ms(trace_elapsed_ms));
            progress(
                mode_label + ":trace:profile" + ":commit_dynamic_ms=" + format_ms(metrics.commit_dynamic_ms) + ":materialization_ms=" + format_ms(metrics.dynamic_polynomial_materialization_ms) + ":msm_ms=" + format_ms(metrics.dynamic_commit_msm_ms) + ":derived_point_ms=" + format_ms(metrics.derived_commitment_point_ms) + ":trace_generation_ms=" + format_ms(metrics.trace_generation_ms) + ":forward_ms=" + format_ms(metrics.forward_ms) + ":lookup_inverse_ms=" + format_ms(metrics.lookup_inverse_ms) + ":lookup_accumulator_ms=" + format_ms(metrics.lookup_accumulator_ms) + ":lookup_pack_ms=" + format_ms(metrics.lookup_query_pack_ms + metrics.lookup_multiplicity_ms + metrics.lookup_key_build_ms) + ":lookup_nested_total_ms=" + format_ms(metrics.lookup_trace_ms) + ":hidden_stage_ms=" + format_ms(metrics.hidden_head_trace_ms) + ":output_stage_ms=" + format_ms(metrics.output_head_trace_ms) + ":trace_other_ms=" + format_ms(trace_other_ms) + ":finalize_ms=" + format_ms(metrics.trace_finalize_ms));
            metrics.reference_forward_ms = metrics.forward_ms;
            progress(mode_label + ":prove:start");
            const auto proof = gatzk::protocol::prove(context, trace, &metrics);
            const auto serialization_start = std::chrono::steady_clock::now();
            const auto proof_bytes = gatzk::protocol::serialize_proof(proof);
            const auto serialization_end = std::chrono::steady_clock::now();
            metrics.proof_serialization_ms = std::chrono::duration<double, std::milli>(
                                                 serialization_end - serialization_start)
                                                 .count();
            gatzk::protocol::timing::finish_prove_timer(&metrics);
            gatzk::protocol::timing::assert_prove_finished(metrics);
            progress(mode_label + ":prove:done");
            metrics.lookup_trace_ms =
                metrics.lookup_table_pack_ms + metrics.lookup_query_pack_ms + metrics.lookup_key_build_ms + metrics.lookup_multiplicity_ms + metrics.lookup_inverse_ms + metrics.lookup_accumulator_ms + metrics.lookup_state_machine_ms + metrics.lookup_selector_mask_ms + metrics.lookup_public_helper_ms + metrics.lookup_copy_convert_ms;
            metrics.trace_misc_ms =
                metrics.route_pack_residual_ms + metrics.selector_padding_residual_ms + metrics.public_poly_residual_ms + metrics.hidden_output_object_residual_ms + metrics.shared_helper_build_ms + metrics.field_conversion_residual_ms + metrics.copy_move_residual_ms + metrics.trace_finalize_ms;
            // The formal verifier must be constructed from the authenticated
            // point-only VK registry; the verifier consumes only this
            // independently authenticated context.
            const auto configured_registry_path =
                run_config.verifier_registry_path.empty()
                    ? gatzk::protocol::canonical_verifier_registry_path(
                          context)
                    : run_config.verifier_registry_path;
            auto verifier_registry_path =
                std::filesystem::path(configured_registry_path);
            if (verifier_registry_path.is_relative())
            {
                verifier_registry_path =
                    std::filesystem::path(run_config.project_root) /
                    verifier_registry_path;
            }
            const auto verifier_context =
                gatzk::protocol::load_verifier_context_from_registry(
                    verifier_registry_path.string(),
                    run_config.verifier_registry_digest);
            progress(mode_label + ":verify:start");
            const auto verify_start = std::chrono::steady_clock::now();
            const auto parsed_proof = gatzk::protocol::parse_proof(verifier_context, proof_bytes);
            const auto proof_parse_end = std::chrono::steady_clock::now();
            metrics.proof_parse_ms = std::chrono::duration<double, std::milli>(
                                         proof_parse_end - verify_start)
                                         .count();
            const auto verifier_accepted =
                gatzk::protocol::verify(verifier_context, parsed_proof, &metrics);
            // The formal verify clock closes at the protocol verifier return.
            // Everything below is an experiment-level audit or acceptance
            // check and must not be folded into verify_time_ms.
            const auto verify_end = std::chrono::steady_clock::now();
            metrics.verify_time_ms = std::chrono::duration<double, std::milli>(
                                         verify_end - verify_start)
                                         .count();
            metrics.verify_misc_ms = clamp_non_negative(
                metrics.verify_time_ms - metrics.proof_parse_ms - metrics.verify_metadata_ms - metrics.verify_transcript_ms - metrics.verify_domain_opening_ms - metrics.verify_point_group_registry_ms - metrics.verify_quotient_ms - metrics.verify_external_opening_ms);
            metrics.verify_accounted_ms = verify_accounted_ms(metrics);
            metrics.verify_accounting_gap_ms = metrics.verify_time_ms - metrics.verify_accounted_ms;
            metrics.proof_size_bytes = proof_bytes.size();
            // The parsed public-output digest is an experiment-level payload
            // audit input, deliberately computed after the protocol timer.
            trace.proof_public_output_hash =
                gatzk::protocol::canonical_output_digest(
                    parsed_proof.public_logits,
                    context.local.num_nodes,
                    context.local.num_classes);
            const auto audit_start = std::chrono::steady_clock::now();
            const auto private_plain_count = gatzk::protocol::private_plain_value_count(verifier_context, parsed_proof);
            const auto audit_end = std::chrono::steady_clock::now();
            metrics.payload_audit_ms = std::chrono::duration<double, std::milli>(audit_end - audit_start).count();
            const auto reference_start = std::chrono::steady_clock::now();
            const bool reference_matrix_match = trace.reference_consistency_passed;
            const auto parsed_output_hash = trace.proof_public_output_hash;
            const bool reference_payload_audit =
                !trace.reference_quant_logits_hash.empty() &&
                !trace.secure_trace_logits_hash.empty() &&
                !trace.proof_public_output_hash.empty() &&
                trace.secure_trace_logits_hash == trace.reference_quant_logits_hash &&
                trace.secure_trace_logits_hash == trace.proof_public_output_hash &&
                trace.proof_public_output_hash == parsed_output_hash &&
                trace.public_logits == parsed_proof.public_logits &&
                private_plain_count == 0;
            const bool reference_consistent =
                reference_matrix_match && reference_payload_audit;
            metrics.reference_consistency_passed = reference_consistent;
            const auto reference_end = std::chrono::steady_clock::now();
            const bool final_accepted =
                gatzk::protocol::final_formal_run_accepted(
                    verifier_accepted,
                    reference_consistent,
                    metrics.security_budget_passed);
            metrics.reference_consistency_ms =
                std::chrono::duration<double, std::milli>(reference_end - reference_start).count();
            progress(mode_label + ":verify:done");
            const auto tamper_start = std::chrono::steady_clock::now();
            // Every case below must mutate a real proof/context field and then
            // enter the production serializer, parser, and verifier.  The
            // status records whether the canonical codec itself rejected the
            // mutation or the parsed proof reached the verifier and was
            // rejected there.
            std::vector<std::pair<std::string, std::string>> tamper_results;
            static constexpr std::array<std::string_view, 39>
                kRequiredTamperCases = {
                    "codec_tag",
                    "codec_length",
                    "codec_trailing",
                    "public_output",
                    "metadata",
                    "schema",
                    "ids",
                    "dynamic_commitment",
                    "quotient_commitment",
                    "opening",
                    "reblind_commitment",
                    "static_registry",
                    "model_registry",
                    "reference_mismatch",
                    "private_opening",
                    "route_padding",
                    "route_query",
                    "lookup_inverse",
                    "lookup_multiplicity",
                    "psq",
                    "shift",
                    "logup",
                    "aggregate",
                    "average",
                    "division",
                    "transcript_challenge",
                    "transcript_value",
                    "srs_metadata",
                    "srs_consistency",
                    "srs_resource_identity",
                    "srs_resource_compromised",
                    "proof_scope",
                    "observation_ledger",
                    "random_identity_ledger",
                    "integer_bound_ledger",
                    "integer_bound_certificate",
                    "security_budget_ledger",
                    "security_budget_exact",
                    "public_input_digest",
                };
            const auto mutate_named_value = [](
                                                gatzk::protocol::Proof &candidate,
                                                std::initializer_list<std::string_view> tokens)
            {
                const auto matches = [&](const std::string &label)
                {
                    return std::any_of(
                        tokens.begin(), tokens.end(),
                        [&](std::string_view token)
                        { return label.find(token) != std::string::npos; });
                };
                for (auto &subproof : candidate.pi_bind.subproofs)
                {
                    for (auto &evaluation : subproof.evaluations)
                    {
                        if (matches(evaluation.object_id) && !evaluation.values.empty())
                        {
                            evaluation.values.front() +=
                                gatzk::algebra::FieldElement::one();
                            return true;
                        }
                    }
                }
                for (auto &bundle_entry : candidate.domain_openings)
                {
                    for (auto &value : bundle_entry.second.values)
                    {
                        if (matches(value.first) && !value.second.empty())
                        {
                            value.second.front() +=
                                gatzk::algebra::FieldElement::one();
                            return true;
                        }
                    }
                }
                return false;
            };
            const auto mutate_any_value = [](gatzk::protocol::Proof &candidate)
            {
                if (!candidate.external_evaluations.empty())
                {
                    candidate.external_evaluations.front().second +=
                        gatzk::algebra::FieldElement::one();
                    return true;
                }
                for (auto &subproof : candidate.pi_bind.subproofs)
                {
                    for (auto &evaluation : subproof.evaluations)
                    {
                        if (!evaluation.values.empty())
                        {
                            evaluation.values.front() +=
                                gatzk::algebra::FieldElement::one();
                            return true;
                        }
                    }
                }
                return false;
            };
            const auto run_proof_case = [&](const std::string &name,
                                            const std::function<bool(gatzk::protocol::Proof &)> &mutator)
            {
                std::string status = "fail";
                try
                {
                    auto candidate = proof;
                    if (!mutator(candidate))
                    {
                        tamper_results.emplace_back(name, status);
                        return;
                    }
                    std::vector<std::uint8_t> bytes;
                    try
                    {
                        bytes = gatzk::protocol::serialize_proof(candidate);
                    }
                    catch (...)
                    {
                        status = "serialization_rejected";
                        tamper_results.emplace_back(name, status);
                        return;
                    }
                    try
                    {
                        const auto parsed =
                            gatzk::protocol::parse_proof(verifier_context, bytes);
                        status = gatzk::protocol::verify(
                                     verifier_context, parsed, nullptr)
                                     ? "fail"
                                     : "reject";
                    }
                    catch (...)
                    {
                        status = "reject";
                    }
                }
                catch (...)
                {
                    status = "fail";
                }
                tamper_results.emplace_back(name, status);
            };
            const auto run_raw_case = [&](const std::string &name,
                                          const std::function<bool(std::vector<std::uint8_t> &)> &mutator)
            {
                std::string status = "fail";
                try
                {
                    auto bytes = proof_bytes;
                    if (!mutator(bytes))
                    {
                        tamper_results.emplace_back(name, status);
                        return;
                    }
                    try
                    {
                        const auto parsed =
                            gatzk::protocol::parse_proof(verifier_context, bytes);
                        status = gatzk::protocol::verify(
                                     verifier_context, parsed, nullptr)
                                     ? "fail"
                                     : "reject";
                    }
                    catch (...)
                    {
                        status = "serialization_rejected";
                    }
                }
                catch (...)
                {
                    status = "fail";
                }
                tamper_results.emplace_back(name, status);
            };
            const auto run_context_case = [&](const std::string &name,
                                              const std::function<bool(gatzk::protocol::VerifierContext &)> &mutator)
            {
                std::string status = "fail";
                try
                {
                    auto candidate_context = verifier_context;
                    if (!mutator(candidate_context))
                    {
                        tamper_results.emplace_back(name, status);
                        return;
                    }
                    try
                    {
                        const auto parsed =
                            gatzk::protocol::parse_proof(candidate_context, proof_bytes);
                        status = gatzk::protocol::verify(
                                     candidate_context, parsed, nullptr)
                                     ? "fail"
                                     : "reject";
                    }
                    catch (...)
                    {
                        status = "reject";
                    }
                }
                catch (...)
                {
                    status = "fail";
                }
                tamper_results.emplace_back(name, status);
            };

            run_raw_case("codec_tag", [](auto &bytes)
                         {
                             if (bytes.empty()) return false;
                             bytes.front() ^= 0x01U;
                             return true; });
            run_raw_case("codec_length", [](auto &bytes)
                         {
                             if (bytes.size() < 2) return false;
                             bytes.pop_back();
                             return true; });
            run_raw_case("codec_trailing", [](auto &bytes)
                         {
                             bytes.push_back(0xa5U);
                             return true; });
            run_proof_case("public_output", [](auto &candidate)
                           {
                               if (candidate.public_logits.empty()) return false;
                               candidate.public_logits.front() +=
                                   gatzk::algebra::FieldElement::one();
                               return true; });
            run_proof_case("route_padding", [&](auto &candidate)
                           { return mutate_named_value(candidate, {"Q_", "padding"}); });
            run_proof_case("route_query", [&](auto &candidate)
                           { return mutate_named_value(candidate, {"Query_", "query"}); });
            run_proof_case("lookup_inverse", [&](auto &candidate)
                           { return mutate_named_value(candidate, {"J_", "inv"}); });
            run_proof_case("lookup_multiplicity", [&](auto &candidate)
                           { return mutate_named_value(candidate, {"m_", "multiplicity"}); });
            run_proof_case("psq", [&](auto &candidate)
                           { return mutate_named_value(candidate, {"PSQ", "psq"}); });
            run_proof_case("aggregate", [&](auto &candidate)
                           { return mutate_named_value(candidate, {"agg", "Agg"}); });
            run_proof_case("average", [&](auto &candidate)
                           { return mutate_named_value(candidate, {"Div_avg", "average", "avg"}); });
            run_proof_case("division", [&](auto &candidate)
                           { return mutate_named_value(candidate, {"Div_", "division"}); });
            run_proof_case("transcript_challenge", [](auto &candidate)
                           {
                               candidate.public_metadata.proof_nonce +=
                                   "tampered-transcript-challenge";
                               return true; });
            run_proof_case("transcript_value", [&](auto &candidate)
                           { return mutate_any_value(candidate); });
            run_proof_case("metadata", [](auto &candidate)
                           {
                               candidate.public_metadata.privacy_policy =
                                   "tampered_privacy_policy";
                               return true; });
            run_proof_case("schema", [](auto &candidate)
                           {
                               if (candidate.block_order.empty()) return false;
                               candidate.block_order.front() = "tampered-block";
                               return true; });
            run_proof_case("ids", [](auto &candidate)
                           {
                               candidate.public_metadata.protocol_schema_id =
                                   "tampered-protocol-schema";
                               return true; });
            run_proof_case("dynamic_commitment", [&verifier_context](auto &candidate)
                           {
                               if (candidate.dynamic_commitments.empty()) return false;
                               candidate.dynamic_commitments.front().second.point =
                                   gatzk::crypto::g1_add(
                                       candidate.dynamic_commitments.front().second.point,
                                       verifier_context.kzg_vk.g1_generator);
                               return true; });
            run_proof_case("quotient_commitment", [&verifier_context](auto &candidate)
                           {
                               if (candidate.quotient_commitments.empty()) return false;
                               candidate.quotient_commitments.front().second.point =
                                   gatzk::crypto::g1_add(
                                       candidate.quotient_commitments.front().second.point,
                                       verifier_context.kzg_vk.g1_generator);
                               return true; });
            run_proof_case("opening", [&verifier_context](auto &candidate)
                           {
                               if (candidate.opening_groups.empty()) return false;
                               candidate.opening_groups.front().witness =
                                   gatzk::crypto::g1_add(
                                       candidate.opening_groups.front().witness,
                                       verifier_context.kzg_vk.g1_generator);
                               return true; });
            run_proof_case("reblind_commitment", [&verifier_context](auto &candidate)
                           {
                               for (auto &subproof : candidate.pi_bind.subproofs)
                               {
                                   if (!subproof.model_reblinds.empty())
                                   {
                                       auto &commitment =
                                           subproof.model_reblinds.front().blinder_commitment;
                                       commitment.point = gatzk::crypto::g1_add(
                                           commitment.point,
                                           verifier_context.kzg_vk.g1_generator);
                                       return true;
                                   }
                               }
                               return false; });
            run_proof_case("logup", [&verifier_context](auto &candidate)
                           {
                               for (auto &subproof : candidate.pi_bind.subproofs)
                               {
                                   if (!subproof.view_logup_commitments.empty())
                                   {
                                       subproof.view_logup_commitments.front().second.point =
                                           gatzk::crypto::g1_add(
                                               subproof.view_logup_commitments.front().second.point,
                                               verifier_context.kzg_vk.g1_generator);
                                       return true;
                                   }
                               }
                               return false; });
            run_proof_case("shift", [](auto &candidate)
                           {
                               for (auto &subproof : candidate.pi_bind.subproofs)
                               {
                                   for (auto &relation : subproof.relation_meta.shift_relations)
                                   {
                                       if (!relation.deltas.empty())
                                       {
                                           relation.deltas.front() +=
                                               gatzk::algebra::FieldElement::one();
                                           return true;
                                       }
                                   }
                               }
                               return false; });
            run_context_case("static_registry", [](auto &candidate)
                             {
                                 if (candidate.registry.static_entries.empty()) return false;
                                 candidate.registry.static_entries.front().point =
                                     gatzk::crypto::g1_add(
                                         candidate.registry.static_entries.front().point,
                                         candidate.kzg_vk.g1_generator);
                                 return true; });
            run_context_case("model_registry", [](auto &candidate)
                             {
                                 if (candidate.registry.model_entries.empty()) return false;
                                 candidate.registry.model_entries.front().point =
                                     gatzk::crypto::g1_add(
                                         candidate.registry.model_entries.front().point,
                                         candidate.kzg_vk.g1_generator);
                                 return true; });
            run_proof_case("reference_mismatch", [](auto &candidate)
                           {
                               candidate.public_metadata.public_output +=
                                   "|reference-mismatch-probe";
                               return true; });
            run_context_case("srs_metadata", [](auto &candidate)
                             {
                                 candidate.srs.srs_type = "untrusted_srs";
                                 return true; });
            run_context_case("srs_consistency", [](auto &candidate)
                             {
                                 if (candidate.srs.srs_consistency_hash.empty())
                                     return false;
                                 candidate.srs.srs_consistency_hash[0] =
                                     candidate.srs.srs_consistency_hash[0] == '0' ? '1' : '0';
                                 return true; });
            run_context_case("srs_resource_identity", [](auto &candidate)
                             {
                                 if (candidate.registry.srs_manifest_hash.empty())
                                     return false;
                                 candidate.registry.srs_manifest_hash[0] =
                                     candidate.registry.srs_manifest_hash[0] == '0' ? '1' : '0';
                                 return true; });
            run_context_case("srs_resource_compromised", [](auto &candidate)
                             {
                                 candidate.kzg_vk.g2_tau = gatzk::crypto::g2_add(
                                     candidate.kzg_vk.g2_tau,
                                     candidate.kzg_vk.g2_one);
                                 return true; });
            run_proof_case("proof_scope", [](auto &candidate)
                           {
                               candidate.public_metadata.output_mode = "labels_only";
                               return true; });
            run_context_case("observation_ledger", [](auto &candidate)
                             {
                                 if (candidate.compiler_artifact == nullptr
                                     || candidate.compiler_artifact->observation_ledger.empty())
                                     return false;
                                 auto artifact = *candidate.compiler_artifact;
                                 artifact.observation_ledger.front().expression_or_certificate +=
                                     "|tamper-observation-ledger";
                                 candidate.compiler_artifact =
                                     std::make_shared<const gatzk::protocol::ProtocolCompilerArtifact>(
                                         std::move(artifact));
                                 return true; });
            run_context_case("random_identity_ledger", [](auto &candidate)
                             {
                                 if (candidate.compiler_artifact == nullptr
                                     || candidate.compiler_artifact->random_identity_ledger.empty())
                                     return false;
                                 auto artifact = *candidate.compiler_artifact;
                                 auto &entry = artifact.random_identity_ledger.front();
                                 entry.degree_bound = entry.degree_bound == 0
                                                           ? 1
                                                           : entry.degree_bound - 1;
                                 candidate.compiler_artifact =
                                     std::make_shared<const gatzk::protocol::ProtocolCompilerArtifact>(
                                         std::move(artifact));
                                 return true; });
            run_context_case("integer_bound_ledger", [](auto &candidate)
                             {
                                 if (candidate.registry.integer_bound_ledger.empty())
                                     return false;
                                 auto &entry = candidate.registry.integer_bound_ledger.front();
                                 if (entry.absolute_bound != std::numeric_limits<std::uint64_t>::max())
                                     ++entry.absolute_bound;
                                 else
                                     --entry.absolute_bound;
                                 return true; });
            run_context_case("integer_bound_certificate", [](auto &candidate)
                             {
                                 if (candidate.registry.model_entries.empty())
                                     return false;
                                 auto &certificate =
                                     candidate.registry.model_entries.front().bound_certificate;
                                 if (!certificate.present)
                                     return false;
                                 if (certificate.absolute_bound !=
                                     std::numeric_limits<std::uint64_t>::max())
                                     ++certificate.absolute_bound;
                                 else
                                     --certificate.absolute_bound;
                                 return true; });
            run_context_case("security_budget_ledger", [](auto &candidate)
                             {
                                 if (candidate.compiler_artifact == nullptr
                                     || candidate.compiler_artifact->security_budget_ledger.empty())
                                     return false;
                                 auto artifact = *candidate.compiler_artifact;
                                 artifact.security_budget_ledger.front().expression_or_certificate +=
                                     "|tamper-security-budget-ledger";
                                 candidate.compiler_artifact =
                                     std::make_shared<const gatzk::protocol::ProtocolCompilerArtifact>(
                                         std::move(artifact));
                                 return true; });
            run_context_case("security_budget_exact", [](auto &candidate)
                             {
                                 if (candidate.registry.security_budget_exact.empty())
                                     return false;
                                 candidate.registry.security_budget_exact += "/0";
                                 return true; });
            run_context_case("public_input_digest", [](auto &candidate)
                             {
                                 if (candidate.registry.public_input_digest.empty())
                                     return false;
                                 candidate.registry.public_input_digest[0] =
                                     candidate.registry.public_input_digest[0] == '0' ? '1' : '0';
                                 return true; });
            if (!tamper_results.empty())
            {
                auto &reference_case = tamper_results.back();
                try
                {
                    const auto parsed =
                        gatzk::protocol::parse_proof(verifier_context, proof_bytes);
                    const auto accepted = gatzk::protocol::verify(
                        verifier_context, parsed, nullptr);
                    reference_case.second =
                        !gatzk::protocol::final_formal_run_accepted(
                            accepted, false, metrics.security_budget_passed)
                            ? "reject"
                            : "fail";
                }
                catch (...)
                {
                    reference_case.second = "reject";
                }
            }
            run_proof_case("private_opening", [](auto &candidate)
                           {
                               if (candidate.dynamic_commitments.empty())
                                   return false;
                               candidate.dynamic_commitments.front().second.opening_polynomial =
                                   std::make_shared<gatzk::algebra::Polynomial>(
                                       gatzk::algebra::Polynomial::from_coefficients(
                                           "tamper-private-opening",
                                           {gatzk::algebra::FieldElement::one()}));
                               return true; });
            std::ostringstream tamper_map;
            bool tamper_rejected = !tamper_results.empty();
            for (std::size_t index = 0; index < tamper_results.size(); ++index)
            {
                if (index != 0)
                    tamper_map << ',';
                tamper_map << tamper_results[index].first << '='
                           << tamper_results[index].second;
                tamper_rejected = tamper_rejected && tamper_results[index].second != "fail";
            }
            for (const auto required_case : kRequiredTamperCases)
            {
                const auto found = std::find_if(
                    tamper_results.begin(), tamper_results.end(),
                    [&](const auto &entry)
                    { return entry.first == required_case; });
                if (found == tamper_results.end() || found->second == "fail")
                    tamper_rejected = false;
            }
            metrics.tamper_test_case_map = tamper_map.str();
            const auto tamper_end = std::chrono::steady_clock::now();
            metrics.tamper_tests_passed = tamper_rejected;
            metrics.tamper_tests_ms = std::chrono::duration<double, std::milli>(tamper_end - tamper_start).count();
            if (!reference_consistent)
            {
                std::cerr << "reference consistency check failed; final formal acceptance denied"
                             " (reference_hash_present=" +
                                 std::string(
                                     trace.secure_trace_logits_hash.empty()
                                         ? "false"
                                         : "true") +
                                 ", reference_matrix_match=" + std::string(reference_matrix_match ? "true" : "false") + ", reference_matches_secure=" + std::string(trace.reference_quant_logits_hash == trace.secure_trace_logits_hash ? "true" : "false") + ", proof_output_hash_match=" + std::string(trace.proof_public_output_hash == parsed_output_hash ? "true" : "false") + ", proof_output_matches_trace=" + std::string(trace.public_logits == parsed_proof.public_logits ? "true" : "false") + ", private_plain_count=" + std::to_string(private_plain_count) + ")\n";
            }
            if (!tamper_rejected)
            {
                throw std::runtime_error("tamper test was not rejected");
            }
            metrics.total_wall_clock_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - row_start).count();
            metrics.formal_time_sum_ms =
                metrics.commitment_time_ms + metrics.prove_time_ms + metrics.verify_time_ms;
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

            const auto cuda_stats_at_end =
                gatzk::algebra::
                    cuda_execution_stats_snapshot();

            metrics.gpu_h2d_bytes =
                cuda_stats_at_end.h2d_bytes -
                cuda_stats_at_start.h2d_bytes;

            metrics.gpu_d2h_bytes =
                cuda_stats_at_end.d2h_bytes -
                cuda_stats_at_start.d2h_bytes;

            metrics.gpu_h2d_operations =
                cuda_stats_at_end.h2d_operations -
                cuda_stats_at_start.h2d_operations;

            metrics.gpu_d2h_operations =
                cuda_stats_at_end.d2h_operations -
                cuda_stats_at_start.d2h_operations;

            metrics.gpu_stream_synchronizations =
                cuda_stats_at_end.stream_synchronizations -
                cuda_stats_at_start.stream_synchronizations;

            metrics.gpu_ntt_calls =
                cuda_stats_at_end.ntt_calls -
                cuda_stats_at_start.ntt_calls;

            metrics.gpu_intt_calls =
                cuda_stats_at_end.intt_calls -
                cuda_stats_at_start.intt_calls;

            metrics.gpu_quotient_calls =
                cuda_stats_at_end.quotient_calls -
                cuda_stats_at_start.quotient_calls;

            metrics.gpu_t_edge_calls =
                cuda_stats_at_end.t_edge_calls -
                cuda_stats_at_start.t_edge_calls;

            metrics.gpu_opening_fold_calls =
                cuda_stats_at_end.opening_fold_calls -
                cuda_stats_at_start.opening_fold_calls;

            metrics.gpu_msm_single_calls =
                cuda_stats_at_end.msm_single_calls -
                cuda_stats_at_start.msm_single_calls;

            metrics.gpu_msm_batch_calls =
                cuda_stats_at_end.msm_batch_calls -
                cuda_stats_at_start.msm_batch_calls;

            metrics.gpu_resident_hits =
                cuda_stats_at_end.resident_hits -
                cuda_stats_at_start.resident_hits;

            metrics.gpu_resident_misses =
                cuda_stats_at_end.resident_misses -
                cuda_stats_at_start.resident_misses;

            if (metrics.gpu_t_edge_calls >
                metrics.gpu_quotient_calls)
            {
                throw std::runtime_error(
                    "CUDA t_edge execution accounting exceeds total "
                    "CUDA quotient execution accounting");
            }

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
                        "runtime_measured;"
                        "ntt_calls=" +
                        std::to_string(
                            metrics.gpu_ntt_calls) +
                        ";intt_calls=" +
                        std::to_string(
                            metrics.gpu_intt_calls) +
                        ";quotient_calls=" +
                        std::to_string(
                            metrics.gpu_quotient_calls) +
                        ";t_edge_calls=" +
                        std::to_string(
                            metrics.gpu_t_edge_calls) +
                        ";opening_fold_calls=" +
                        std::to_string(
                            metrics.gpu_opening_fold_calls) +
                        ";msm_single_calls=" +
                        std::to_string(
                            metrics.gpu_msm_single_calls) +
                        ";msm_batch_calls=" +
                        std::to_string(
                            metrics.gpu_msm_batch_calls);

                    metrics.gpu_fallback_stages =
                        "pairing,workloads_below_cuda_threshold_or_resource_admission";

                    metrics.gpu_fallback_reason =
                        "workload_aware_hybrid_routing";

                    metrics.gpu_acceleration_status =
                        "cuda_hotspots_executed;"
                        "kernels=" +
                        std::to_string(
                            metrics.gpu_kernel_invocation_count) +
                        ";h2d_bytes=" +
                        std::to_string(
                            metrics.gpu_h2d_bytes) +
                        ";d2h_bytes=" +
                        std::to_string(
                            metrics.gpu_d2h_bytes) +
                        ";stream_syncs=" +
                        std::to_string(
                            metrics.gpu_stream_synchronizations);

                    append_note(
                        metrics,
                        "cuda_kernel_execution=workload_aware_hybrid");
                }
            }

            if (log_benchmark)
            {
                std::cout
                    << "dataset=" << metrics.dataset
                    << " config=" << config_path
                    << " backend=" << console_backend_label(metrics)
                    << " status=" << (final_accepted ? "VERIFY_OK" : "VERIFY_FAIL")
                    << " commitment_time_ms=" << format_ms(metrics.commitment_time_ms)
                    << " prove_time_ms=" << format_ms(metrics.prove_time_ms)
                    << " verify_time_ms=" << format_ms(metrics.verify_time_ms)
                    << " proof_size_bytes=" << std::to_string(metrics.proof_size_bytes)
                    << '\n';
            }
            if (export_artifacts)
            {
                progress(mode_label + ":export:start");
                gatzk::protocol::export_run_artifacts(
                    context,
                    trace,
                    proof,
                    verifier_context,
                    parsed_proof,
                    metrics,
                    verifier_accepted,
                    final_accepted);
                progress(mode_label + ":export:done");
            }
            return final_accepted;
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
