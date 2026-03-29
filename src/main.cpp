#include <chrono>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

#include "gatzk/crypto/curve.hpp"
#include "gatzk/protocol/prover.hpp"
#include "gatzk/protocol/trace.hpp"
#include "gatzk/protocol/verifier.hpp"
#include "gatzk/util/config.hpp"
#include "gatzk/util/logging.hpp"
#include "gatzk/util/route2.hpp"

int main(int argc, char** argv) {
    try {
        auto format_ms = [](double value) {
            std::ostringstream stream;
            stream << std::fixed << std::setprecision(3) << value;
            return stream.str();
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
        auto parse_switch = [](const std::string& value, const std::string& name) {
            if (value == "on" || value == "true" || value == "1") {
                return true;
            }
            if (value == "off" || value == "false" || value == "0") {
                return false;
            }
            throw std::runtime_error("invalid value for " + name + ": " + value);
        };

        std::string config_path;
        std::string benchmark_mode = "single";
        std::string export_tag;
        gatzk::util::Route2Options route2;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--config" && i + 1 < argc) {
                config_path = argv[++i];
            } else if (arg == "--benchmark-mode" && i + 1 < argc) {
                benchmark_mode = argv[++i];
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
        if (benchmark_mode != "single" && benchmark_mode != "cold" && benchmark_mode != "warm" && benchmark_mode != "both") {
            throw std::runtime_error("unsupported benchmark mode: " + benchmark_mode);
        }

        const auto loaded_config = gatzk::util::load_config(config_path);
        const auto absolute_config_path = std::filesystem::absolute(config_path).string();
        gatzk::util::set_route2_options(route2);

        auto run_once = [&](const gatzk::util::AppConfig& run_config, bool is_cold_run, const std::string& mode_label) {
            gatzk::protocol::RunMetrics metrics;
            metrics.backend_name = gatzk::crypto::backend_name();
            metrics.config = absolute_config_path;
            metrics.enabled_fast_msm = route2.fast_msm;
            metrics.enabled_parallel_fft = route2.parallel_fft;
            metrics.enabled_fft_backend_upgrade = route2.fft_backend_upgrade;
            metrics.enabled_fft_kernel_upgrade = route2.fft_kernel_upgrade;
            metrics.enabled_trace_layout_upgrade = route2.trace_layout_upgrade;
            metrics.enabled_fast_verify_pairing = route2.fast_verify_pairing;
            if (route2.fft_backend_upgrade && route2.fft_kernel_upgrade) {
                metrics.fft_backend_route = "packed_rotated_kernel";
            } else if (route2.fft_backend_upgrade) {
                metrics.fft_backend_route = "packed_domain";
            } else {
                metrics.fft_backend_route = "legacy_domain";
            }
            metrics.is_cold_run = is_cold_run;
            append_note(metrics, "benchmark_mode=" + mode_label);
            append_note(metrics, gatzk::util::route2_feature_notes(route2));

            gatzk::util::info(
                "backend_name=" + gatzk::crypto::backend_name()
                + " benchmark_mode=" + mode_label
                + " route2=" + gatzk::util::route2_feature_label(route2));
            gatzk::util::info("building protocol context");
            const auto context = gatzk::protocol::build_context(run_config, &metrics);
            metrics.dataset = context.dataset.name;
            metrics.node_count = context.local.num_nodes;
            metrics.edge_count = context.local.edges.size();
            metrics.is_full_dataset = context.local.num_nodes == context.dataset.num_nodes;
            append_note(metrics, metrics.is_full_dataset ? "dataset_scope=full" : "dataset_scope=local_subgraph");

            if (!run_config.prove_enabled) {
                std::cout << "SMOKE_OK" << '\n';
                return true;
            }

            gatzk::util::info("building trace");
            const auto trace = gatzk::protocol::build_trace(context, &metrics);
            gatzk::util::info("proving");
            const auto proof = gatzk::protocol::prove(context, trace, &metrics);
            gatzk::util::info("verifying");
            const auto verify_start = std::chrono::steady_clock::now();
            const auto accepted = gatzk::protocol::verify(context, proof);
            const auto verify_end = std::chrono::steady_clock::now();
            metrics.verify_time_ms = std::chrono::duration<double, std::milli>(verify_end - verify_start).count();
            metrics.proof_size_bytes = gatzk::protocol::proof_size_bytes(proof);

            gatzk::util::info(
                "benchmark backend_name=" + metrics.backend_name
                + " config=" + metrics.config
                + " dataset=" + metrics.dataset
                + " node_count=" + std::to_string(metrics.node_count)
                + " edge_count=" + std::to_string(metrics.edge_count)
                + " enabled_fast_msm=" + std::string(metrics.enabled_fast_msm ? "true" : "false")
                + " enabled_parallel_fft=" + std::string(metrics.enabled_parallel_fft ? "true" : "false")
                + " enabled_fft_backend_upgrade=" + std::string(metrics.enabled_fft_backend_upgrade ? "true" : "false")
                + " enabled_fft_kernel_upgrade=" + std::string(metrics.enabled_fft_kernel_upgrade ? "true" : "false")
                + " enabled_trace_layout_upgrade=" + std::string(metrics.enabled_trace_layout_upgrade ? "true" : "false")
                + " fft_backend_route=" + metrics.fft_backend_route
                + " enabled_fast_verify_pairing=" + std::string(metrics.enabled_fast_verify_pairing ? "true" : "false")
                + " is_cold_run=" + std::string(metrics.is_cold_run ? "true" : "false")
                + " is_full_dataset=" + std::string(metrics.is_full_dataset ? "true" : "false")
                + " load_static_ms=" + format_ms(metrics.load_static_ms)
                + " fft_plan_ms=" + format_ms(metrics.fft_plan_ms)
                + " srs_prepare_ms=" + format_ms(metrics.srs_prepare_ms)
                + " forward_ms=" + format_ms(metrics.forward_ms)
                + " feature_projection_ms=" + format_ms(metrics.feature_projection_ms)
                + " trace_generation_ms=" + format_ms(metrics.trace_generation_ms)
                + " witness_materialization_ms=" + format_ms(metrics.witness_materialization_ms)
                + " lookup_trace_ms=" + format_ms(metrics.lookup_trace_ms)
                + " route_trace_ms=" + format_ms(metrics.route_trace_ms)
                + " psq_trace_ms=" + format_ms(metrics.psq_trace_ms)
                + " zkmap_trace_ms=" + format_ms(metrics.zkmap_trace_ms)
                + " commit_dynamic_ms=" + format_ms(metrics.commit_dynamic_ms)
                + " dynamic_commit_input_ms=" + format_ms(metrics.dynamic_commit_input_ms)
                + " dynamic_polynomial_materialization_ms=" + format_ms(metrics.dynamic_polynomial_materialization_ms)
                + " dynamic_commit_msm_ms=" + format_ms(metrics.dynamic_commit_msm_ms)
                + " dynamic_commit_finalize_ms=" + format_ms(metrics.dynamic_commit_finalize_ms)
                + " quotient_build_ms=" + format_ms(metrics.quotient_build_ms)
                + " domain_opening_ms=" + format_ms(metrics.domain_opening_ms)
                + " external_opening_ms=" + format_ms(metrics.external_opening_ms)
                + " prove_time_ms=" + format_ms(metrics.prove_time_ms)
                + " verify_time_ms=" + format_ms(metrics.verify_time_ms)
                + " proof_size_bytes=" + std::to_string(metrics.proof_size_bytes)
                + " notes=\"" + metrics.notes + "\"");
            gatzk::protocol::export_run_artifacts(context, trace, proof, metrics, accepted);
            return accepted;
        };

        auto warm_up = [&](const gatzk::util::AppConfig& run_config) {
            gatzk::util::info("priming process-local caches for warm benchmark");
            const auto context = gatzk::protocol::build_context(run_config, nullptr);
            if (!run_config.prove_enabled) {
                return;
            }
            const auto trace = gatzk::protocol::build_trace(context, nullptr);
            const auto proof = gatzk::protocol::prove(context, trace, nullptr);
            (void)gatzk::protocol::verify(context, proof);
        };

        bool accepted = false;
        std::vector<std::string> export_segments;
        if (!export_tag.empty()) {
            export_segments.push_back(export_tag);
        }
        if (benchmark_mode == "single") {
            accepted = run_once(with_export_suffix(loaded_config, export_segments), true, "single");
        } else if (benchmark_mode == "cold") {
            auto segments = export_segments;
            segments.push_back("cold");
            accepted = run_once(with_export_suffix(loaded_config, segments), true, "cold");
        } else if (benchmark_mode == "warm") {
            warm_up(with_export_suffix(loaded_config, export_segments));
            auto segments = export_segments;
            segments.push_back("warm");
            accepted = run_once(with_export_suffix(loaded_config, segments), false, "warm");
        } else {
            auto cold_segments = export_segments;
            cold_segments.push_back("cold");
            auto warm_segments = export_segments;
            warm_segments.push_back("warm");
            const auto cold_ok = run_once(with_export_suffix(loaded_config, cold_segments), true, "cold");
            const auto warm_ok = run_once(with_export_suffix(loaded_config, warm_segments), false, "warm");
            accepted = cold_ok && warm_ok;
        }

        std::cout << (accepted ? "VERIFY_OK" : "VERIFY_FAIL") << '\n';
        return accepted ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
