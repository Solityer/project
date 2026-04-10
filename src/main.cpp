#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/crypto/curve.hpp"
#include "gatzk/protocol/prover.hpp"
#include "gatzk/protocol/challenges.hpp"
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
        auto memory_safe_warm_required = [](const gatzk::util::AppConfig& config) {
            return config.dataset == "ogbn_arxiv"
                && config.batching_rule == "whole_graph_single"
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

        std::string config_path;
        std::string benchmark_mode = "warm";
        std::string compute_backend = "cpu";
        std::string export_tag;
        gatzk::util::Route2Options route2;
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

        set_process_env("GATZK_ALGEBRA_BACKEND", "cpu");
        route2.cuda_trace_hotspots = compute_backend == "cuda_hotspots";
        if (route2.cuda_trace_hotspots) {
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

        auto run_once = [&](
                            const gatzk::util::AppConfig& run_config,
                            bool is_cold_run,
                            const std::string& mode_label,
                            bool export_artifacts = true,
                            bool log_benchmark = true) {
            gatzk::protocol::RunMetrics metrics;
            metrics.crypto_backend_name = gatzk::crypto::backend_name();
            metrics.algebra_backend_name = gatzk::algebra::configured_algebra_backend_name();
            metrics.compute_backend_name = route2.cuda_trace_hotspots ? "cuda_hotspots" : "cpu";
            metrics.backend_name = metrics.crypto_backend_name + "/" + metrics.compute_backend_name;
            metrics.config = absolute_config_path;
            metrics.enabled_fast_msm = route2.fast_msm;
            metrics.enabled_parallel_fft = route2.parallel_fft;
            metrics.enabled_fft_backend_upgrade = route2.fft_backend_upgrade;
            metrics.enabled_fft_kernel_upgrade = route2.fft_kernel_upgrade;
            metrics.enabled_trace_layout_upgrade = route2.trace_layout_upgrade;
            metrics.enabled_fast_verify_pairing = route2.fast_verify_pairing;
            metrics.enabled_cuda_trace_hotspots = route2.cuda_trace_hotspots;
            metrics.benchmark_mode = mode_label;
            metrics.route2_label = gatzk::util::route2_feature_label(route2);
            metrics.gpu_runtime_present = gatzk::algebra::cuda_backend_available();
            if (route2.fft_backend_upgrade && route2.fft_kernel_upgrade) {
                metrics.fft_backend_route = "packed_rotated_kernel";
            } else if (route2.fft_backend_upgrade) {
                metrics.fft_backend_route = "packed_domain";
            } else {
                metrics.fft_backend_route = "legacy_domain";
            }
            metrics.is_cold_run = is_cold_run;
            append_note(metrics, "benchmark_mode=" + mode_label);
            append_note(metrics, "compute_backend=" + metrics.compute_backend_name);
            append_note(metrics, gatzk::util::route2_feature_notes(route2));

            gatzk::util::info(
                "backend_name=" + metrics.backend_name
                + " crypto_backend=" + metrics.crypto_backend_name
                + " algebra_backend=" + metrics.algebra_backend_name
                + " benchmark_mode=" + mode_label
                + " route2=" + gatzk::util::route2_feature_label(route2));
            gatzk::util::info("building protocol context");
            const auto context_start = std::chrono::steady_clock::now();
            const auto context = gatzk::protocol::build_context(run_config, &metrics);
            const auto context_end = std::chrono::steady_clock::now();
            metrics.context_build_ms = std::chrono::duration<double, std::milli>(context_end - context_start).count();
            metrics.dataset = context.dataset.name;
            metrics.node_count = context.local.num_nodes;
            metrics.edge_count = context.local.edges.size();
            metrics.is_full_dataset = context.local.num_nodes == context.dataset.num_nodes;
            if (!metrics.is_full_dataset) {
                throw std::runtime_error("formal mainline only supports full-dataset runs");
            }
            const auto run_profile = gatzk::protocol::canonical_public_metadata(context);
            gatzk::util::info(
                "profile dataset_name=" + run_profile.dataset_name
                + " graph_count=" + run_profile.graph_count
                + " L=" + run_profile.L
                + " hidden_profile=" + run_profile.hidden_profile
                + " d_in_profile=" + run_profile.d_in_profile
                + " K_out=" + run_profile.K_out
                + " C=" + run_profile.C
                + " batching_rule=" + run_profile.batching_rule
                + " subgraph_rule=" + run_profile.subgraph_rule
                + " self_loop_rule=" + run_profile.self_loop_rule
                + " edge_sort_rule=" + run_profile.edge_sort_rule);

            gatzk::util::info("building trace");
            const auto trace_start = std::chrono::steady_clock::now();
            const auto trace = gatzk::protocol::build_trace(context, &metrics);
            gatzk::util::info("proving");
            const auto pcs_start = std::chrono::steady_clock::now();
            const auto proof = gatzk::protocol::prove(context, trace, &metrics);
            const auto pcs_end = std::chrono::steady_clock::now();
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
            gatzk::util::info("verifying");
            const auto verify_start = std::chrono::steady_clock::now();
            const auto accepted = gatzk::protocol::verify(context, proof, &metrics);
            const auto verify_end = std::chrono::steady_clock::now();
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
                    + " context_build_ms=" + format_ms(metrics.context_build_ms)
                    + " load_static_ms=" + format_ms(metrics.load_static_ms)
                    + " fft_plan_ms=" + format_ms(metrics.fft_plan_ms)
                    + " srs_prepare_ms=" + format_ms(metrics.srs_prepare_ms)
                    + " forward_ms=" + format_ms(metrics.forward_ms)
                    + " feature_projection_ms=" + format_ms(metrics.feature_projection_ms)
                    + " hidden_forward_projection_ms=" + format_ms(metrics.hidden_forward_projection_ms)
                    + " hidden_forward_attention_ms=" + format_ms(metrics.hidden_forward_attention_ms)
                    + " hidden_forward_activation_ms=" + format_ms(metrics.hidden_forward_activation_ms)
                    + " hidden_concat_ms=" + format_ms(metrics.hidden_concat_ms)
                    + " output_forward_projection_ms=" + format_ms(metrics.output_forward_projection_ms)
                    + " output_forward_attention_ms=" + format_ms(metrics.output_forward_attention_ms)
                    + " output_forward_activation_ms=" + format_ms(metrics.output_forward_activation_ms)
                    + " trace_generation_ms=" + format_ms(metrics.trace_generation_ms)
                    + " trace_misc_ms=" + format_ms(metrics.trace_misc_ms)
                    + " witness_materialization_ms=" + format_ms(metrics.witness_materialization_ms)
                    + " lookup_trace_ms=" + format_ms(metrics.lookup_trace_ms)
                    + " lookup_table_pack_ms=" + format_ms(metrics.lookup_table_pack_ms)
                    + " lookup_query_pack_ms=" + format_ms(metrics.lookup_query_pack_ms)
                    + " lookup_key_build_ms=" + format_ms(metrics.lookup_key_build_ms)
                    + " lookup_multiplicity_ms=" + format_ms(metrics.lookup_multiplicity_ms)
                    + " lookup_accumulator_ms=" + format_ms(metrics.lookup_accumulator_ms)
                    + " lookup_state_machine_ms=" + format_ms(metrics.lookup_state_machine_ms)
                    + " lookup_selector_mask_ms=" + format_ms(metrics.lookup_selector_mask_ms)
                    + " lookup_public_helper_ms=" + format_ms(metrics.lookup_public_helper_ms)
                    + " lookup_copy_convert_ms=" + format_ms(metrics.lookup_copy_convert_ms)
                    + " route_trace_ms=" + format_ms(metrics.route_trace_ms)
                    + " psq_trace_ms=" + format_ms(metrics.psq_trace_ms)
                    + " zkmap_trace_ms=" + format_ms(metrics.zkmap_trace_ms)
                    + " state_machine_trace_ms=" + format_ms(metrics.state_machine_trace_ms)
                    + " padding_selector_trace_ms=" + format_ms(metrics.padding_selector_trace_ms)
                    + " public_poly_trace_ms=" + format_ms(metrics.public_poly_trace_ms)
                    + " hidden_head_trace_ms=" + format_ms(metrics.hidden_head_trace_ms)
                    + " hidden_projection_trace_ms=" + format_ms(metrics.hidden_projection_trace_ms)
                    + " hidden_src_attention_trace_ms=" + format_ms(metrics.hidden_src_attention_trace_ms)
                    + " hidden_dst_attention_trace_ms=" + format_ms(metrics.hidden_dst_attention_trace_ms)
                    + " hidden_edge_score_trace_ms=" + format_ms(metrics.hidden_edge_score_trace_ms)
                    + " hidden_softmax_chain_trace_ms=" + format_ms(metrics.hidden_softmax_chain_trace_ms)
                    + " hidden_h_star_trace_ms=" + format_ms(metrics.hidden_h_star_trace_ms)
                    + " hidden_h_agg_pre_star_trace_ms=" + format_ms(metrics.hidden_h_agg_pre_star_trace_ms)
                    + " hidden_h_agg_star_trace_ms=" + format_ms(metrics.hidden_h_agg_star_trace_ms)
                    + " hidden_route_trace_ms=" + format_ms(metrics.hidden_route_trace_ms)
                    + " hidden_copy_convert_ms=" + format_ms(metrics.hidden_copy_convert_ms)
                    + " output_head_trace_ms=" + format_ms(metrics.output_head_trace_ms)
                    + " route_pack_residual_ms=" + format_ms(metrics.route_pack_residual_ms)
                    + " selector_padding_residual_ms=" + format_ms(metrics.selector_padding_residual_ms)
                    + " public_poly_residual_ms=" + format_ms(metrics.public_poly_residual_ms)
                    + " hidden_output_object_residual_ms=" + format_ms(metrics.hidden_output_object_residual_ms)
                    + " shared_helper_build_ms=" + format_ms(metrics.shared_helper_build_ms)
                    + " field_conversion_residual_ms=" + format_ms(metrics.field_conversion_residual_ms)
                    + " copy_move_residual_ms=" + format_ms(metrics.copy_move_residual_ms)
                    + " trace_finalize_ms=" + format_ms(metrics.trace_finalize_ms)
                    + " fh_table_materialization_ms=" + format_ms(metrics.fh_table_materialization_ms)
                    + " fh_query_materialization_ms=" + format_ms(metrics.fh_query_materialization_ms)
                    + " fh_multiplicity_build_ms=" + format_ms(metrics.fh_multiplicity_build_ms)
                    + " fh_accumulator_build_ms=" + format_ms(metrics.fh_accumulator_build_ms)
                    + " fh_interpolation_ms=" + format_ms(metrics.fh_interpolation_ms)
                    + " fh_lagrange_eval_ms=" + format_ms(metrics.fh_lagrange_eval_ms)
                    + " fh_barycentric_weight_fetch_ms=" + format_ms(metrics.fh_barycentric_weight_fetch_ms)
                    + " fh_point_powers_ms=" + format_ms(metrics.fh_point_powers_ms)
                    + " fh_public_poly_interp_ms=" + format_ms(metrics.fh_public_poly_interp_ms)
                    + " fh_feature_poly_interp_ms=" + format_ms(metrics.fh_feature_poly_interp_ms)
                    + " fh_fold_prep_ms=" + format_ms(metrics.fh_fold_prep_ms)
                    + " fh_opening_eval_prep_ms=" + format_ms(metrics.fh_opening_eval_prep_ms)
                    + " fh_copy_convert_ms=" + format_ms(metrics.fh_copy_convert_ms)
                    + " fh_eval_prep_ms=" + format_ms(metrics.fh_eval_prep_ms)
                    + " fh_public_eval_reuse_ms=" + format_ms(metrics.fh_public_eval_reuse_ms)
                    + " fh_quotient_assembly_ms=" + format_ms(metrics.fh_quotient_assembly_ms)
                    + " fh_open_gather_ms=" + format_ms(metrics.fh_open_gather_ms)
                    + " fh_open_witness_ms=" + format_ms(metrics.fh_open_witness_ms)
                    + " fh_open_fold_prepare_ms=" + format_ms(metrics.fh_open_fold_prepare_ms)
                    + " commit_dynamic_ms=" + format_ms(metrics.commit_dynamic_ms)
                    + " commitment_time_ms=" + format_ms(metrics.commitment_time_ms)
                    + " dynamic_commit_input_ms=" + format_ms(metrics.dynamic_commit_input_ms)
                    + " dynamic_polynomial_materialization_ms=" + format_ms(metrics.dynamic_polynomial_materialization_ms)
                    + " dynamic_commit_pack_ms=" + format_ms(metrics.dynamic_commit_pack_ms)
                    + " dynamic_fft_ms=" + format_ms(metrics.dynamic_fft_ms)
                    + " dynamic_domain_convert_ms=" + format_ms(metrics.dynamic_domain_convert_ms)
                    + " dynamic_copy_convert_ms=" + format_ms(metrics.dynamic_copy_convert_ms)
                    + " dynamic_commit_msm_ms=" + format_ms(metrics.dynamic_commit_msm_ms)
                    + " dynamic_commit_finalize_ms=" + format_ms(metrics.dynamic_commit_finalize_ms)
                    + " dynamic_bundle_finalize_ms=" + format_ms(metrics.dynamic_bundle_finalize_ms)
                    + " quotient_build_ms=" + format_ms(metrics.quotient_build_ms)
                    + " quotient_t_fh_ms=" + format_ms(metrics.quotient_t_fh_ms)
                    + " quotient_t_edge_ms=" + format_ms(metrics.quotient_t_edge_ms)
                    + " quotient_t_in_ms=" + format_ms(metrics.quotient_t_in_ms)
                    + " quotient_t_d_h_ms=" + format_ms(metrics.quotient_t_d_h_ms)
                    + " quotient_t_cat_ms=" + format_ms(metrics.quotient_t_cat_ms)
                    + " quotient_t_C_ms=" + format_ms(metrics.quotient_t_c_ms)
                    + " quotient_t_N_ms=" + format_ms(metrics.quotient_t_n_ms)
                    + " quotient_public_eval_ms=" + format_ms(metrics.quotient_public_eval_ms)
                    + " quotient_bundle_pack_ms=" + format_ms(metrics.quotient_bundle_pack_ms)
                    + " quotient_fold_prepare_ms=" + format_ms(metrics.quotient_fold_prepare_ms)
                    + " quotient_copy_convert_ms=" + format_ms(metrics.quotient_copy_convert_ms)
                    + " domain_opening_ms=" + format_ms(metrics.domain_opening_ms)
                    + " domain_eval_gather_ms=" + format_ms(metrics.domain_eval_gather_ms)
                    + " domain_open_witness_ms=" + format_ms(metrics.domain_open_witness_ms)
                    + " domain_open_FH_ms=" + format_ms(metrics.domain_open_fh_ms)
                    + " domain_open_edge_ms=" + format_ms(metrics.domain_open_edge_ms)
                    + " domain_open_in_ms=" + format_ms(metrics.domain_open_in_ms)
                    + " domain_open_d_h_ms=" + format_ms(metrics.domain_open_d_h_ms)
                    + " domain_open_cat_ms=" + format_ms(metrics.domain_open_cat_ms)
                    + " domain_open_C_ms=" + format_ms(metrics.domain_open_c_ms)
                    + " domain_open_N_ms=" + format_ms(metrics.domain_open_n_ms)
                    + " external_opening_ms=" + format_ms(metrics.external_opening_ms)
                    + " prove_finalize_ms=" + format_ms(metrics.prove_finalize_ms)
                    + " prove_accounted_ms=" + format_ms(metrics.prove_accounted_ms)
                    + " prove_accounting_gap_ms=" + format_ms(metrics.prove_accounting_gap_ms)
                    + " prove_time_ms=" + format_ms(metrics.prove_time_ms)
                    + " verify_time_ms=" + format_ms(metrics.verify_time_ms)
                    + " verify_metadata_ms=" + format_ms(metrics.verify_metadata_ms)
                    + " verify_transcript_ms=" + format_ms(metrics.verify_transcript_ms)
                    + " verify_domain_opening_ms=" + format_ms(metrics.verify_domain_opening_ms)
                    + " verify_quotient_ms=" + format_ms(metrics.verify_quotient_ms)
                    + " verify_external_fold_ms=" + format_ms(metrics.verify_external_fold_ms)
                    + " verify_misc_ms=" + format_ms(metrics.verify_misc_ms)
                    + " verify_accounted_ms=" + format_ms(metrics.verify_accounted_ms)
                    + " verify_accounting_gap_ms=" + format_ms(metrics.verify_accounting_gap_ms)
                    + " verify_FH_ms=" + format_ms(metrics.verify_fh_ms)
                    + " verify_edge_ms=" + format_ms(metrics.verify_edge_ms)
                    + " verify_in_ms=" + format_ms(metrics.verify_in_ms)
                    + " verify_d_h_ms=" + format_ms(metrics.verify_d_h_ms)
                    + " verify_cat_ms=" + format_ms(metrics.verify_cat_ms)
                    + " verify_C_ms=" + format_ms(metrics.verify_c_ms)
                    + " verify_N_ms=" + format_ms(metrics.verify_n_ms)
                    + " verify_public_eval_ms=" + format_ms(metrics.verify_public_eval_ms)
                    + " verify_bundle_lookup_ms=" + format_ms(metrics.verify_bundle_lookup_ms)
                    + " verify_fold_ms=" + format_ms(metrics.verify_fold_ms)
                    + " verify_copy_convert_ms=" + format_ms(metrics.verify_copy_convert_ms)
                    + " proof_size_bytes=" + std::to_string(metrics.proof_size_bytes)
                    + " notes=\"" + metrics.notes + "\"");
            }
            if (export_artifacts) {
                gatzk::protocol::export_run_artifacts(context, trace, proof, metrics, accepted);
            }
            return accepted;
        };

        auto warm_up = [&](const gatzk::util::AppConfig& run_config) {
            gatzk::util::info("priming process-local caches for warm benchmark");
            const auto context = gatzk::protocol::build_context(run_config, nullptr);
            const auto trace = gatzk::protocol::build_trace(context, nullptr);
            const auto proof = gatzk::protocol::prove(context, trace, nullptr);
            (void)gatzk::protocol::verify(context, proof);
        };

        bool accepted = false;
        std::vector<std::string> export_segments;
        if (!export_tag.empty()) {
            export_segments.push_back(export_tag);
        }
        if (memory_safe_warm_required(loaded_config)) {
            gatzk::util::info("using memory-safe warm path");
        } else {
            warm_up(with_export_suffix(loaded_config, export_segments));
        }
        auto segments = export_segments;
        segments.push_back("warm");
        accepted = run_once(with_export_suffix(loaded_config, segments), false, "warm");

        std::cout << (accepted ? "VERIFY_OK" : "VERIFY_FAIL") << '\n';
        return accepted ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
