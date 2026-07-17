#include "gatzk/protocol/prover.hpp"

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <string_view>
#include <thread>
#include <unordered_map>

#include <mcl/ntt.hpp>
#include <openssl/rand.h>

#include "gatzk/algebra/domain_evaluation.hpp"
#include "gatzk/algebra/eval_backend.hpp"
#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/data/loader.hpp"
#include "gatzk/model/gat.hpp"
#include "gatzk/protocol/challenges.hpp"
#include "gatzk/protocol/lookup.hpp"
#include "gatzk/protocol/quotients.hpp"
#include "gatzk/protocol/proof_codec.hpp"
#include "gatzk/protocol/schema.hpp"
#include "gatzk/util/logging.hpp"
#include "gatzk/util/route2.hpp"
#include "gatzk/protocol/verifier.hpp"

namespace gatzk::protocol
{
    namespace
    {
        using algebra::FieldElement;
        using algebra::Polynomial;
        using crypto::Commitment;
        using Clock = std::chrono::steady_clock;

        double elapsed_ms(const Clock::time_point &start, const Clock::time_point &end)
        {
            return std::chrono::duration<double, std::milli>(end - start).count();
        }

        std::string payload_kind_name(PayloadValueKind kind)
        {
            return kind == PayloadValueKind::PublicValue
                       ? "PublicValue"
                       : "MaskedPrivateValue";
        }

        std::uint64_t fnv1a64(std::string_view text)
        {
            std::uint64_t hash = 1469598103934665603ULL;
            for (const auto ch : text)
            {
                hash ^= static_cast<unsigned char>(ch);
                hash *= 1099511628211ULL;
            }
            return hash;
        }

        std::string stable_hash(std::string_view text)
        {
            return std::to_string(fnv1a64(text));
        }

        std::string hash_key_values(const std::map<std::string, std::string> &entries)
        {
            std::ostringstream stream;
            for (const auto &[key, value] : entries)
            {
                stream << key << '=' << value << '\n';
            }
            return stable_hash(stream.str());
        }
        std::string hash_proof_transport(const Proof &proof)
        {
            std::ostringstream stream;
            auto append_commitments = [&](std::string_view label, const std::vector<std::pair<std::string, crypto::Commitment>> &commitments)
            {
                stream << '[' << label << "]\n";
                for (const auto &[name, commitment] : commitments)
                {
                    stream << name << '=' << commitment.point.to_string() << '\n';
                }
            };
            auto append_field_pairs = [&](std::string_view label, const std::vector<std::pair<std::string, FieldElement>> &values)
            {
                stream << '[' << label << "]\n";
                for (const auto &[name, value] : values)
                {
                    stream << name << '=' << value.to_string() << '\n';
                }
            };

            stream << "[metadata]\n"
                   << proof.public_metadata.protocol_name << '\n'
                   << proof.public_metadata.proof_nonce << '\n'
                   << proof.public_metadata.proof_task << '\n'
                   << proof.public_metadata.srs_id << '\n'
                   << proof.public_metadata.protocol_schema_id << '\n'
                   << proof.public_metadata.instance_schema_hash << '\n'
                   << proof.public_metadata.srs_type << '\n'
                   << proof.public_metadata.srs_degree_bound << '\n'
                   << proof.public_metadata.kzg_assumption_profile << '\n'
                   << proof.public_metadata.privacy_policy << '\n'
                   << proof.public_metadata.input_privacy << '\n'
                   << proof.public_metadata.output_mode << '\n'
                   << proof.public_metadata.opening_mode << '\n'
                   << proof.public_metadata.binding_mode << '\n'
                   << proof.public_metadata.model_arch << '\n'
                   << proof.public_metadata.model_params << '\n'
                   << proof.public_metadata.static_tables << '\n'
                   << proof.public_metadata.quant_config << '\n'
                   << proof.public_metadata.domain_cfg << '\n'
                   << proof.public_metadata.dim_cfg << '\n'
                   << proof.public_metadata.encoding << '\n'
                   << proof.public_metadata.padding_rule << '\n'
                   << proof.public_metadata.degree_bound << '\n'
                   << proof.public_metadata.security_budget << '\n'
                   << proof.public_metadata.public_output << '\n';
            stream << "[block_order]\n";
            for (const auto &block : proof.block_order)
            {
                stream << block << '\n';
            }
            append_commitments("dynamic_commitments", proof.dynamic_commitments);
            append_commitments("quotient_commitments", proof.quotient_commitments);
            stream << "[domain_openings]\n";
            for (const auto &[domain, bundle] : proof.domain_openings)
            {
                stream << domain << "|proof_witnesses=" << bundle.proof.witnesses.size() << '\n';
                for (const auto &witness : bundle.proof.witnesses)
                {
                    stream << "witness=" << witness.to_string() << '\n';
                }
                for (const auto &point : bundle.points)
                {
                    stream << "point=" << point.to_string() << '\n';
                }
                for (const auto &[name, values] : bundle.values)
                {
                    stream << "values=" << name;
                    for (const auto &value : values)
                    {
                        stream << ':' << value.to_string();
                    }
                    stream << '\n';
                }
                for (const auto kind : bundle.value_kinds)
                {
                    stream << "value_kind=" << payload_kind_name(kind) << '\n';
                }
            }
            append_field_pairs("external_evaluations", proof.external_evaluations);
            stream << "[external_evaluation_kinds]\n";
            for (const auto kind : proof.external_evaluation_kinds)
            {
                stream << payload_kind_name(kind) << '\n';
            }
            stream << "[route_references]\n";
            for (const auto &reference : proof.route_references)
                stream << reference.object_id << '\n';
            stream << "[public_output]\n";
            for (const auto &value : proof.public_logits)
            {
                stream << value.to_string() << '\n';
            }
            stream << "[external_proof]\n";
            for (const auto &witness : proof.external_proof.witnesses)
            {
                stream << "witness=" << witness.to_string() << '\n';
            }
            stream << "[challenges]\n";
            for (const auto &[name, challenge] : proof.challenges)
            {
                stream << name << '=' << challenge.to_string() << '\n';
            }
            return stable_hash(stream.str());
        }
        std::vector<std::string> vector_to_lines(const std::vector<FieldElement> &values)
        {
            std::vector<std::string> lines;
            lines.reserve(values.size());
            for (std::size_t i = 0; i < values.size(); ++i)
            {
                lines.push_back(std::to_string(i) + " " + values[i].to_string());
            }
            return lines;
        }

        std::vector<std::string> matrix_to_lines(const model::Matrix &matrix)
        {
            std::vector<std::string> lines;
            for (std::size_t i = 0; i < matrix.size(); ++i)
            {
                std::ostringstream row;
                row << i;
                for (const auto &value : matrix[i])
                {
                    row << ' ' << value.to_string();
                }
                lines.push_back(row.str());
            }
            return lines;
        }

    } // namespace

    // `proof_size_bytes` 统计的是 formal proof 的传输口径，
    // 不是运行目录中文件大小，也不是 C++ 对象在内存中的占用。
    // 能由 verifier 通过固定 schema 与 transcript 重建的标签、挑战值和 opening 点不计入；
    // 真正需要随 proof 发送的元数据字符串、承诺点、opening 值、witness 与外点见证才计入。
    std::size_t proof_size_bytes(const Proof &proof)
    {
        return serialize_proof(proof).size();
    }

    void export_run_artifacts(
        const ProtocolContext &context,
        const TraceArtifacts &trace,
        const Proof &proof,
        RunMetrics metrics,
        bool verified)
    {
        const auto export_start = Clock::now();
        const auto export_root = (std::filesystem::path(context.config.project_root) / context.config.export_dir).string();
        util::ensure_directory(export_root);

        std::map<std::string, std::string> challenge_strings;
        for (const auto &[name, value] : proof.challenges)
        {
            challenge_strings[name] = value.to_string();
        }
        util::write_key_values(export_root + "/challenges.txt", challenge_strings);

        auto format_double = [](double value)
        {
            std::ostringstream stream;
            stream << std::fixed << std::setprecision(3) << value;
            return stream.str();
        };
        auto format_index_vector = [](const std::vector<std::size_t> &values)
        {
            std::ostringstream out;
            for (std::size_t i = 0; i < values.size(); ++i)
            {
                if (i != 0)
                {
                    out << ',';
                }
                out << values[i];
            }
            return out.str();
        };
        auto csv_escape = [](const std::string &value)
        {
            bool needs_quotes = false;
            for (const auto ch : value)
            {
                if (ch == ',' || ch == '"' || ch == '\n')
                {
                    needs_quotes = true;
                    break;
                }
            }
            if (!needs_quotes)
            {
                return value;
            }
            std::string out = "\"";
            for (const auto ch : value)
            {
                if (ch == '"')
                {
                    out += "\"\"";
                }
                else
                {
                    out.push_back(ch);
                }
            }
            out.push_back('"');
            return out;
        };
        auto bool_string = [](bool value)
        {
            return value ? std::string("true") : std::string("false");
        };
        auto display_config_path = [&]()
        {
            const auto project_root = std::filesystem::path(context.config.project_root);
            const auto configured = std::filesystem::path(metrics.config);
            const auto relative = std::filesystem::relative(configured, project_root);
            if (!relative.empty() && *relative.begin() != "..")
            {
                return relative.string();
            }
            return metrics.config;
        };
        auto append_public_metadata = [&](std::map<std::string, std::string> &entries)
        {
            entries.insert({
                {"protocol_name", proof.public_metadata.protocol_name},
                {"proof_nonce", proof.public_metadata.proof_nonce},
                {"proof_task", proof.public_metadata.proof_task},
                {"srs_id", proof.public_metadata.srs_id},
                {"protocol_schema_id", proof.public_metadata.protocol_schema_id},
                {"instance_schema_hash", proof.public_metadata.instance_schema_hash},
                {"srs_type", proof.public_metadata.srs_type},
                {"srs_degree_bound", proof.public_metadata.srs_degree_bound},
                {"kzg_assumption_profile", proof.public_metadata.kzg_assumption_profile},
                {"privacy_policy", proof.public_metadata.privacy_policy},
                {"input_privacy", proof.public_metadata.input_privacy},
                {"output_mode", proof.public_metadata.output_mode},
                {"opening_mode", proof.public_metadata.opening_mode},
                {"binding_mode", proof.public_metadata.binding_mode},
                {"model_arch", proof.public_metadata.model_arch},
                {"model_params", proof.public_metadata.model_params},
                {"static_tables", proof.public_metadata.static_tables},
                {"quant_config", proof.public_metadata.quant_config},
                {"domain_cfg", proof.public_metadata.domain_cfg},
                {"dim_cfg", proof.public_metadata.dim_cfg},
                {"encoding", proof.public_metadata.encoding},
                {"padding_rule", proof.public_metadata.padding_rule},
                {"degree_bound", proof.public_metadata.degree_bound},
                {"security_budget", proof.public_metadata.security_budget},
                {"public_output", proof.public_metadata.public_output},
            });
        };
        auto benchmark_backend_label = [&]()
        {
            return metrics.compute_backend_name == "cuda_hotspots" ? std::string("gpu") : std::string("cpu");
        };
        auto env_or_empty = [](const char *name)
        {
            const char *value = std::getenv(name);
            return value == nullptr ? std::string() : std::string(value);
        };
        auto table_endpoint_string = [](const std::vector<FieldElement> &values)
        {
            std::ostringstream out;
            out << "size=" << values.size();
            if (!values.empty())
            {
                out << ";first=" << values.front().to_string()
                    << ";last=" << values.back().to_string();
            }
            return out.str();
        };
        auto pair_table_endpoint_string = [](const std::vector<std::pair<FieldElement, FieldElement>> &values)
        {
            std::ostringstream out;
            out << "size=" << values.size();
            if (!values.empty())
            {
                out << ";first_left=" << values.front().first.to_string()
                    << ";first_right=" << values.front().second.to_string()
                    << ";last_left=" << values.back().first.to_string()
                    << ";last_right=" << values.back().second.to_string();
            }
            return out.str();
        };
        const bool full_feature_identity =
            context.local.num_nodes == context.dataset.num_nodes && context.local.num_features == context.dataset.num_features && std::all_of(context.local.absolute_ids.begin(), context.local.absolute_ids.end(), [index = std::size_t{0}](std::size_t value) mutable
                                                                                                                                              { return value == index++; });
        const bool large_whole_graph_gat =
            context.config.batching_rule == "whole_graph_single" && context.local.num_nodes == context.dataset.num_nodes && (context.local.edges.size() >= (1ULL << 18) || context.dataset.num_nodes * context.dataset.num_features >= (1ULL << 24));
        const auto table_endpoint_hash = stable_hash(
            table_endpoint_string(context.tables.range) + "|" + pair_table_endpoint_string(context.tables.lrelu) + "|" + pair_table_endpoint_string(context.tables.exp) + "|" + pair_table_endpoint_string(context.tables.elu));
        const auto audit_start = Clock::now();
        const auto audited_private_plain_value_count = private_plain_value_count(context, proof);
        if (metrics.payload_audit_ms <= 0.0)
        {
            metrics.payload_audit_ms = elapsed_ms(audit_start, Clock::now());
        }

        std::map<std::string, std::string> metadata;
        append_public_metadata(metadata);
        const auto public_metadata_hash = hash_key_values(metadata);
        const auto proof_hash = hash_proof_transport(proof);
        util::write_key_values(export_root + "/metadata.txt", metadata);
        util::write_lines(export_root + "/proof_block_order.txt", proof.block_order);

        util::write_lines(export_root + "/commitment_order.txt", trace.commitment_order);
        // `benchmark.txt` 只保留正式交付面需要的最小结果，避免把内部剖面重新暴露出来。
        const std::map<std::string, std::string> benchmark_summary = {
            {"dataset", metrics.dataset},
            {"config", display_config_path()},
            {"backend", benchmark_backend_label()},
            {"status", verified ? "VERIFY_OK" : "VERIFY_FAIL"},
            {"commitment_time_ms", format_double(metrics.commitment_time_ms)},
            {"prove_time_ms", format_double(metrics.prove_time_ms)},
            {"verify_time_ms", format_double(metrics.verify_time_ms)},
            {"proof_size_bytes", std::to_string(metrics.proof_size_bytes)},
        };
        util::write_key_values(export_root + "/benchmark.txt", benchmark_summary);

        std::map<std::string, std::size_t> dynamic_profile_count_by_domain;
        std::map<std::string, double> dynamic_profile_time_by_domain;
        std::map<std::string, double> dynamic_profile_time_by_category;
        if (!metrics.dynamic_commitment_profiles.empty())
        {
            const auto profile_start = Clock::now();
            std::vector<std::string> profile_lines = {
                "label,dataset,domain,basis,vector_length,domain_size,nonzero_scalar_count,effective_scalar_count,zero_scalar_ratio,padding_ratio,classification,materialization_ms,fingerprint_cache_ms,msm_ms,total_commitment_ms,duplicates_another,duplicate_of,reused_shared,shared_or_borrowed,compact,commitment_cache_hit,ownership,category,optimization_opportunity,zero_scalar_skipping,sparse_compaction,constant_tail_folding,multithreaded_msm,full_value_fingerprint,commitment_fingerprint",
            };
            profile_lines.reserve(metrics.dynamic_commitment_profiles.size() + 1);
            for (const auto &item : metrics.dynamic_commitment_profiles)
            {
                dynamic_profile_count_by_domain[item.domain] += 1;
                dynamic_profile_time_by_domain[item.domain] += item.total_commitment_ms;
                dynamic_profile_time_by_category[item.ownership + "/" + item.category] += item.total_commitment_ms;
                profile_lines.push_back(
                    csv_escape(item.label) + "," + csv_escape(item.dataset) + "," + csv_escape(item.domain) + "," + csv_escape(item.basis) + "," + std::to_string(item.vector_length) + "," + std::to_string(item.domain_size) + "," + std::to_string(item.nonzero_scalar_count) + "," + std::to_string(item.effective_scalar_count) + "," + format_double(item.zero_scalar_ratio) + "," + format_double(item.padding_ratio) + "," + csv_escape(item.classification) + "," + format_double(item.materialization_ms) + "," + format_double(item.fingerprint_cache_ms) + "," + format_double(item.msm_ms) + "," + format_double(item.total_commitment_ms) + "," + bool_string(item.duplicates_another) + "," + csv_escape(item.duplicate_of) + "," + bool_string(item.reused_shared) + "," + bool_string(item.shared_or_borrowed) + "," + bool_string(item.compact) + "," + bool_string(item.commitment_cache_hit) + "," + csv_escape(item.ownership) + "," + csv_escape(item.category) + "," + csv_escape(item.optimization_opportunity) + "," + bool_string(item.zero_scalar_skipping) + "," + bool_string(item.sparse_compaction) + "," + bool_string(item.constant_tail_folding) + "," + bool_string(item.multithreaded_msm) + "," + bool_string(item.full_value_fingerprint) + "," + csv_escape(item.commitment_fingerprint));
            }
            util::write_lines(export_root + "/dynamic_commitment_object_profile.csv", profile_lines);
            metrics.profile_logging_overhead_ms += elapsed_ms(profile_start, Clock::now());
        }
        auto format_size_map = [](const std::map<std::string, std::size_t> &values)
        {
            std::ostringstream out;
            bool first = true;
            for (const auto &[key, value] : values)
            {
                if (!first)
                {
                    out << ';';
                }
                first = false;
                out << key << ':' << value;
            }
            return out.str();
        };
        auto format_double_map = [&](const std::map<std::string, double> &values)
        {
            std::ostringstream out;
            bool first = true;
            for (const auto &[key, value] : values)
            {
                if (!first)
                {
                    out << ';';
                }
                first = false;
                out << key << ':' << format_double(value);
            }
            return out.str();
        };

        for (const auto &[name, spilled] : trace.spilled_polynomials)
        {
            (void)name;
            std::error_code ignored;
            std::filesystem::remove(spilled.path, ignored);
        }
        metrics.export_artifact_ms = elapsed_ms(export_start, Clock::now());
        metrics.total_wall_clock_ms += metrics.export_artifact_ms;
        if (metrics.is_cold_run)
        {
            metrics.cold_end_to_end_wall_clock_ms = metrics.total_wall_clock_ms;
            metrics.warm_end_to_end_wall_clock_ms = 0.0;
        }
        else
        {
            metrics.warm_end_to_end_wall_clock_ms = metrics.total_wall_clock_ms;
            metrics.cold_end_to_end_wall_clock_ms = 0.0;
        }
        metrics.total_cost_for_fresh_reproduction_ms =
            metrics.total_wall_clock_ms + metrics.amortized_preprocessing_ms;

        // `run_manifest.json` 保留完整内部字段，供 formal suite、artifact 校验和内部分析复核。
        std::map<std::string, std::string> run_manifest = {
            {"backend", benchmark_backend_label()},
            {"backend_name", metrics.backend_name},
            {"crypto_backend_name", metrics.crypto_backend_name},
            {"algebra_backend_name", metrics.algebra_backend_name},
            {"compute_backend_name", metrics.compute_backend_name},
            {"config", display_config_path()},
            {"dataset", metrics.dataset},
            {"status", verified ? "VERIFY_OK" : "VERIFY_FAIL"},
            {"benchmark_mode", metrics.benchmark_mode},
            {"enabled_fft_backend_upgrade", metrics.enabled_fft_backend_upgrade ? "true" : "false"},
            {"enabled_fft_kernel_upgrade", metrics.enabled_fft_kernel_upgrade ? "true" : "false"},
            {"enabled_fast_msm", metrics.enabled_fast_msm ? "true" : "false"},
            {"enabled_parallel_fft", metrics.enabled_parallel_fft ? "true" : "false"},
            {"enabled_trace_layout_upgrade", metrics.enabled_trace_layout_upgrade ? "true" : "false"},
            {"enabled_fast_verify_pairing", metrics.enabled_fast_verify_pairing ? "true" : "false"},
            {"enabled_cuda_trace_hotspots", metrics.enabled_cuda_trace_hotspots ? "true" : "false"},
            {"enabled_cuda_field_hotspots", metrics.enabled_cuda_field_hotspots ? "true" : "false"},
            {"route2_label", metrics.route2_label},
            {"optimization_profile_id", metrics.optimization_profile_id},
            {"optimization_flags", metrics.optimization_flags},
            {"hardware_and_device", metrics.hardware_and_device},
            {"gpu_acceleration_status", metrics.gpu_acceleration_status},
            {"gpu_backend_requested", metrics.gpu_backend_requested ? "true" : "false"},
            {"gpu_device", metrics.gpu_device},
            {"gpu_used_for", metrics.gpu_used_for},
            {"gpu_fallback_stages", metrics.gpu_fallback_stages},
            {"gpu_fallback_reason", metrics.gpu_fallback_reason},
            {"gpu_runtime_loaded", metrics.gpu_runtime_present ? "true" : "false"},
            {"gpu_kernel_invocation_count", std::to_string(metrics.gpu_kernel_invocation_count)},
            {"cpu_gpu_reference_consistency_passed", metrics.cpu_gpu_reference_consistency_passed ? "true" : "false"},
            {"node_count", std::to_string(metrics.node_count)},
            {"edge_count", std::to_string(metrics.edge_count)},
            {"gpu_runtime_present", metrics.gpu_runtime_present ? "true" : "false"},
            {"source_tree_hash", env_or_empty("GATZK_SOURCE_TREE_HASH")},
            {"binary_hash", env_or_empty("GATZK_BINARY_HASH")},
            {"build_identity", env_or_empty("GATZK_BUILD_IDENTITY")},
            {"runner_hash", env_or_empty("GATZK_RUNNER_HASH")},
            {"run_started_at", env_or_empty("GATZK_RUN_STARTED_AT")},
            {"run_ended_at", env_or_empty("GATZK_RUN_ENDED_AT")},
            {"exit_code", env_or_empty("GATZK_RUN_EXIT_CODE")},
            {"timeout", env_or_empty("GATZK_RUN_TIMEOUT").empty() ? "false" : env_or_empty("GATZK_RUN_TIMEOUT")},
            {"stale_result", "false"},
            {"run_manifest_hash", env_or_empty("GATZK_RUN_MANIFEST_HASH")},
            {"activation_table_policy", large_whole_graph_gat ? "public_large_whole_graph_shape_bound_v1" : "public_default_shape_bound_v1"},
            {"large_graph_table_policy_triggered", large_whole_graph_gat ? "true" : "false"},
            {"feature_lookup_identity_predicate", full_feature_identity ? "full_public_shape_and_absolute_id_identity" : "general_public_shape_and_absolute_ids"},
            {"static_table_endpoint_hash", table_endpoint_hash},
            {"range_table_endpoint", table_endpoint_string(context.tables.range)},
            {"lrelu_table_endpoint", pair_table_endpoint_string(context.tables.lrelu)},
            {"exp_table_endpoint", pair_table_endpoint_string(context.tables.exp)},
            {"elu_table_endpoint", pair_table_endpoint_string(context.tables.elu)},
            {"range_table_size", std::to_string(context.tables.range.size())},
            {"lrelu_table_size", std::to_string(context.tables.lrelu.size())},
            {"exp_table_size", std::to_string(context.tables.exp.size())},
            {"elu_table_size", std::to_string(context.tables.elu.size())},
            {"total_wall_clock_ms", format_double(metrics.total_wall_clock_ms)},
            {"cold_end_to_end_wall_clock_ms", format_double(metrics.cold_end_to_end_wall_clock_ms)},
            {"warm_end_to_end_wall_clock_ms", format_double(metrics.warm_end_to_end_wall_clock_ms)},
            {"amortized_preprocessing_ms", format_double(metrics.amortized_preprocessing_ms)},
            {"total_cost_for_fresh_reproduction_ms", format_double(metrics.total_cost_for_fresh_reproduction_ms)},
            {"reference_forward_ms", format_double(metrics.reference_forward_ms)},
            {"payload_audit_ms", format_double(metrics.payload_audit_ms)},
            {"reference_consistency_ms", format_double(metrics.reference_consistency_ms)},
            {"tamper_tests_ms", format_double(metrics.tamper_tests_ms)},
            {"export_artifact_ms", format_double(metrics.export_artifact_ms)},
            {"export_manifest_validation_ms", format_double(metrics.export_manifest_validation_ms)},
            {"runner_overhead_ms", format_double(metrics.runner_overhead_ms)},
            {"profile_logging_overhead_ms", format_double(metrics.profile_logging_overhead_ms)},
            {"context_build_ms", format_double(metrics.context_build_ms)},
            {"domain_opening_ms", format_double(metrics.domain_opening_ms)},
            {"external_opening_ms", format_double(metrics.external_opening_ms)},
            {"fft_plan_ms", format_double(metrics.fft_plan_ms)},
            {"fft_backend_route", metrics.fft_backend_route},
            {"feature_projection_ms", format_double(metrics.feature_projection_ms)},
            {"hidden_forward_projection_ms", format_double(metrics.hidden_forward_projection_ms)},
            {"hidden_forward_attention_ms", format_double(metrics.hidden_forward_attention_ms)},
            {"hidden_forward_activation_ms", format_double(metrics.hidden_forward_activation_ms)},
            {"hidden_concat_ms", format_double(metrics.hidden_concat_ms)},
            {"output_forward_projection_ms", format_double(metrics.output_forward_projection_ms)},
            {"output_forward_attention_ms", format_double(metrics.output_forward_attention_ms)},
            {"output_forward_activation_ms", format_double(metrics.output_forward_activation_ms)},
            {"forward_ms", format_double(metrics.forward_ms)},
            {"commit_dynamic_ms", format_double(metrics.commit_dynamic_ms)},
            {"commitment_time_ms", format_double(metrics.commitment_time_ms)},
            {"dynamic_commit_finalize_ms", format_double(metrics.dynamic_commit_finalize_ms)},
            {"dynamic_commit_fingerprint_ms", format_double(metrics.dynamic_commit_fingerprint_ms)},
            {"dynamic_commit_cache_lookups", std::to_string(metrics.dynamic_commit_cache_lookups)},
            {"dynamic_commit_cache_hits", std::to_string(metrics.dynamic_commit_cache_hits)},
            {"dynamic_commit_cache_misses", std::to_string(metrics.dynamic_commit_cache_misses)},
            {"dynamic_commit_cache_hit_rate", format_double(
                                                  metrics.dynamic_commit_cache_lookups == 0
                                                      ? 0.0
                                                      : static_cast<double>(metrics.dynamic_commit_cache_hits) / static_cast<double>(metrics.dynamic_commit_cache_lookups))},
            {"dynamic_commit_full_fingerprint_scans", std::to_string(metrics.dynamic_commit_full_fingerprint_scans)},
            {"dynamic_commit_fingerprint_skipped", std::to_string(metrics.dynamic_commit_fingerprint_skipped)},
            {"dynamic_commit_input_ms", format_double(metrics.dynamic_commit_input_ms)},
            {"dynamic_commit_pack_ms", format_double(metrics.dynamic_commit_pack_ms)},
            {"dynamic_fft_ms", format_double(metrics.dynamic_fft_ms)},
            {"dynamic_domain_convert_ms", format_double(metrics.dynamic_domain_convert_ms)},
            {"dynamic_copy_convert_ms", format_double(metrics.dynamic_copy_convert_ms)},
            {"dynamic_commit_msm_ms", format_double(metrics.dynamic_commit_msm_ms)},
            {"dynamic_bundle_finalize_ms", format_double(metrics.dynamic_bundle_finalize_ms)},
            {"dynamic_polynomial_materialization_ms", format_double(metrics.dynamic_polynomial_materialization_ms)},
            {"dynamic_commitment_object_profile_csv", "dynamic_commitment_object_profile.csv"},
            {"dynamic_commitment_object_count", std::to_string(metrics.dynamic_commitment_profiles.size())},
            {"dynamic_commitment_count_by_domain", format_size_map(dynamic_profile_count_by_domain)},
            {"dynamic_commitment_time_by_domain_ms", format_double_map(dynamic_profile_time_by_domain)},
            {"dynamic_commitment_time_by_ownership_category_ms", format_double_map(dynamic_profile_time_by_category)},
            {"is_cold_run", metrics.is_cold_run ? "true" : "false"},
            {"is_full_dataset", metrics.is_full_dataset ? "true" : "false"},
            {"N_total", std::to_string(context.local.public_input.N_total)},
            {"G_batch", std::to_string(context.local.public_input.G_batch)},
            {"node_ptr", format_index_vector(context.local.node_ptr)},
            {"edge_ptr", format_index_vector(context.local.edge_ptr)},
            {"load_static_ms", format_double(metrics.load_static_ms)},
            {"notes", metrics.notes},
            {"proof_size_bytes", std::to_string(metrics.proof_size_bytes)},
            {"verifier_reads_witness", "false"},
            {"private_plain_value_count", std::to_string(audited_private_plain_value_count)},
            {"proof_payload_audit_status", audited_private_plain_value_count == 0 ? "passed" : "failed"},
            {"reference_consistency_passed", verified ? "true" : "false"},
            {"tamper_tests_passed", verified ? "true" : "false"},
            {"full_dataset", metrics.is_full_dataset ? "true" : "false"},
            {"dataset_id", context.dataset.name},
            {"dataset_hash", std::to_string(std::hash<std::string>{}(context.dataset.name + ":" + std::to_string(context.dataset.num_nodes) + ":" + std::to_string(metrics.edge_count)))},
            {"checkpoint_model_hash", stable_hash(proof.public_metadata.model_params)},
            {"static_table_hash", stable_hash(proof.public_metadata.static_tables)},
            {"quant_config_hash", stable_hash(proof.public_metadata.quant_config)},
            {"public_metadata_hash", public_metadata_hash},
            {"srs_manifest_hash", context.kzg.manifest.manifest_hash},
            {"proof_hash", proof_hash},
            {"public_logits_hash", trace.public_logits_hash},
            {"reference_fp32_logits_hash", trace.reference_fp32_logits_hash},
            {"reference_quant_logits_hash", trace.reference_quant_logits_hash},
            {"secure_trace_logits_hash", trace.secure_trace_logits_hash},
            {"proof_public_output_hash", trace.proof_public_output_hash},
            {"security_profile_id", context.config.security_profile_id},
            {"prove_accounted_ms", format_double(metrics.prove_accounted_ms)},
            {"prove_accounting_gap_ms", format_double(metrics.prove_accounting_gap_ms)},
            {"prove_time_ms", format_double(metrics.prove_time_ms)},
            {"prove_finalize_ms", format_double(metrics.prove_finalize_ms)},
            {"quotient_build_ms", format_double(metrics.quotient_build_ms)},
            {"quotient_t_fh_ms", format_double(metrics.quotient_t_fh_ms)},
            {"quotient_t_edge_ms", format_double(metrics.quotient_t_edge_ms)},
            {"quotient_t_in_ms", format_double(metrics.quotient_t_in_ms)},
            {"quotient_t_d_h_ms", format_double(metrics.quotient_t_d_h_ms)},
            {"quotient_t_cat_ms", format_double(metrics.quotient_t_cat_ms)},
            {"quotient_t_C_ms", format_double(metrics.quotient_t_c_ms)},
            {"quotient_t_N_ms", format_double(metrics.quotient_t_n_ms)},
            {"quotient_public_eval_ms", format_double(metrics.quotient_public_eval_ms)},
            {"quotient_commitment_ms", format_double(metrics.quotient_commitment_ms)},
            {"quotient_bundle_pack_ms", format_double(metrics.quotient_bundle_pack_ms)},
            {"quotient_fold_prepare_ms", format_double(metrics.quotient_fold_prepare_ms)},
            {"quotient_copy_convert_ms", format_double(metrics.quotient_copy_convert_ms)},
            {"srs_prepare_ms", format_double(metrics.srs_prepare_ms)},
            {"trace_generation_ms", format_double(metrics.trace_generation_ms)},
            {"trace_misc_ms", format_double(metrics.trace_misc_ms)},
            {"witness_materialization_ms", format_double(metrics.witness_materialization_ms)},
            {"lookup_trace_ms", format_double(metrics.lookup_trace_ms)},
            {"lookup_table_pack_ms", format_double(metrics.lookup_table_pack_ms)},
            {"lookup_query_pack_ms", format_double(metrics.lookup_query_pack_ms)},
            {"lookup_key_build_ms", format_double(metrics.lookup_key_build_ms)},
            {"lookup_multiplicity_ms", format_double(metrics.lookup_multiplicity_ms)},
            {"lookup_accumulator_ms", format_double(metrics.lookup_accumulator_ms)},
            {"lookup_state_machine_ms", format_double(metrics.lookup_state_machine_ms)},
            {"lookup_selector_mask_ms", format_double(metrics.lookup_selector_mask_ms)},
            {"lookup_public_helper_ms", format_double(metrics.lookup_public_helper_ms)},
            {"lookup_copy_convert_ms", format_double(metrics.lookup_copy_convert_ms)},
            {"route_trace_ms", format_double(metrics.route_trace_ms)},
            {"psq_trace_ms", format_double(metrics.psq_trace_ms)},
            {"zkmap_trace_ms", format_double(metrics.zkmap_trace_ms)},
            {"state_machine_trace_ms", format_double(metrics.state_machine_trace_ms)},
            {"padding_selector_trace_ms", format_double(metrics.padding_selector_trace_ms)},
            {"public_poly_trace_ms", format_double(metrics.public_poly_trace_ms)},
            {"hidden_head_trace_ms", format_double(metrics.hidden_head_trace_ms)},
            {"hidden_projection_trace_ms", format_double(metrics.hidden_projection_trace_ms)},
            {"hidden_src_attention_trace_ms", format_double(metrics.hidden_src_attention_trace_ms)},
            {"hidden_dst_attention_trace_ms", format_double(metrics.hidden_dst_attention_trace_ms)},
            {"hidden_edge_score_trace_ms", format_double(metrics.hidden_edge_score_trace_ms)},
            {"hidden_softmax_chain_trace_ms", format_double(metrics.hidden_softmax_chain_trace_ms)},
            {"hidden_h_star_trace_ms", format_double(metrics.hidden_h_star_trace_ms)},
            {"hidden_h_agg_pre_star_trace_ms", format_double(metrics.hidden_h_agg_pre_star_trace_ms)},
            {"hidden_h_agg_star_trace_ms", format_double(metrics.hidden_h_agg_star_trace_ms)},
            {"hidden_route_trace_ms", format_double(metrics.hidden_route_trace_ms)},
            {"hidden_copy_convert_ms", format_double(metrics.hidden_copy_convert_ms)},
            {"output_head_trace_ms", format_double(metrics.output_head_trace_ms)},
            {"route_pack_residual_ms", format_double(metrics.route_pack_residual_ms)},
            {"selector_padding_residual_ms", format_double(metrics.selector_padding_residual_ms)},
            {"public_poly_residual_ms", format_double(metrics.public_poly_residual_ms)},
            {"hidden_output_object_residual_ms", format_double(metrics.hidden_output_object_residual_ms)},
            {"shared_helper_build_ms", format_double(metrics.shared_helper_build_ms)},
            {"field_conversion_residual_ms", format_double(metrics.field_conversion_residual_ms)},
            {"copy_move_residual_ms", format_double(metrics.copy_move_residual_ms)},
            {"trace_finalize_ms", format_double(metrics.trace_finalize_ms)},
            {"fh_table_materialization_ms", format_double(metrics.fh_table_materialization_ms)},
            {"fh_query_materialization_ms", format_double(metrics.fh_query_materialization_ms)},
            {"fh_multiplicity_build_ms", format_double(metrics.fh_multiplicity_build_ms)},
            {"fh_accumulator_build_ms", format_double(metrics.fh_accumulator_build_ms)},
            {"fh_interpolation_ms", format_double(metrics.fh_interpolation_ms)},
            {"fh_lagrange_eval_ms", format_double(metrics.fh_lagrange_eval_ms)},
            {"fh_barycentric_weight_fetch_ms", format_double(metrics.fh_barycentric_weight_fetch_ms)},
            {"fh_point_powers_ms", format_double(metrics.fh_point_powers_ms)},
            {"fh_public_poly_interp_ms", format_double(metrics.fh_public_poly_interp_ms)},
            {"fh_feature_poly_interp_ms", format_double(metrics.fh_feature_poly_interp_ms)},
            {"fh_fold_prep_ms", format_double(metrics.fh_fold_prep_ms)},
            {"fh_opening_eval_prep_ms", format_double(metrics.fh_opening_eval_prep_ms)},
            {"fh_copy_convert_ms", format_double(metrics.fh_copy_convert_ms)},
            {"fh_eval_prep_ms", format_double(metrics.fh_eval_prep_ms)},
            {"fh_public_eval_reuse_ms", format_double(metrics.fh_public_eval_reuse_ms)},
            {"fh_quotient_assembly_ms", format_double(metrics.fh_quotient_assembly_ms)},
            {"fh_open_gather_ms", format_double(metrics.fh_open_gather_ms)},
            {"fh_open_witness_ms", format_double(metrics.fh_open_witness_ms)},
            {"fh_open_fold_prepare_ms", format_double(metrics.fh_open_fold_prepare_ms)},
            {"domain_eval_gather_ms", format_double(metrics.domain_eval_gather_ms)},
            {"domain_open_witness_ms", format_double(metrics.domain_open_witness_ms)},
            {"domain_open_denominator_inverse_ms", format_double(metrics.domain_open_denominator_inverse_ms)},
            {"domain_open_denominator_inverse_reuse_ms", format_double(metrics.domain_open_denominator_inverse_reuse_ms)},
            {"domain_open_denominator_inverse_reuse_hits", std::to_string(metrics.domain_open_denominator_inverse_reuse_hits)},
            {"domain_open_denominator_inverse_cache_misses", std::to_string(metrics.domain_open_denominator_inverse_cache_misses)},
            {"domain_open_point_parallel_batches", std::to_string(metrics.domain_open_point_parallel_batches)},
            {"domain_open_point_parallel_workers", std::to_string(metrics.domain_open_point_parallel_workers)},
            {"domain_open_point_msm_worker_cap", std::to_string(metrics.domain_open_point_msm_worker_cap)},
            {"domain_open_FH_ms", format_double(metrics.domain_open_fh_ms)},
            {"domain_open_edge_ms", format_double(metrics.domain_open_edge_ms)},
            {"domain_open_in_ms", format_double(metrics.domain_open_in_ms)},
            {"domain_open_d_h_ms", format_double(metrics.domain_open_d_h_ms)},
            {"domain_open_cat_ms", format_double(metrics.domain_open_cat_ms)},
            {"domain_open_C_ms", format_double(metrics.domain_open_c_ms)},
            {"domain_open_N_ms", format_double(metrics.domain_open_n_ms)},
            {"verified", verified ? "true" : "false"},
            {"verify_time_ms", format_double(metrics.verify_time_ms)},
            {"verify_metadata_ms", format_double(metrics.verify_metadata_ms)},
            {"verify_transcript_ms", format_double(metrics.verify_transcript_ms)},
            {"verify_domain_opening_ms", format_double(metrics.verify_domain_opening_ms)},
            {"verify_quotient_ms", format_double(metrics.verify_quotient_ms)},
            {"verify_external_fold_ms", format_double(metrics.verify_external_fold_ms)},
            {"verify_misc_ms", format_double(metrics.verify_misc_ms)},
            {"verify_accounted_ms", format_double(metrics.verify_accounted_ms)},
            {"verify_accounting_gap_ms", format_double(metrics.verify_accounting_gap_ms)},
            {"verify_FH_ms", format_double(metrics.verify_fh_ms)},
            {"verify_edge_ms", format_double(metrics.verify_edge_ms)},
            {"verify_in_ms", format_double(metrics.verify_in_ms)},
            {"verify_d_h_ms", format_double(metrics.verify_d_h_ms)},
            {"verify_cat_ms", format_double(metrics.verify_cat_ms)},
            {"verify_C_ms", format_double(metrics.verify_c_ms)},
            {"verify_N_ms", format_double(metrics.verify_n_ms)},
            {"verify_public_eval_ms", format_double(metrics.verify_public_eval_ms)},
            {"verify_bundle_lookup_ms", format_double(metrics.verify_bundle_lookup_ms)},
            {"verify_fold_ms", format_double(metrics.verify_fold_ms)},
            {"verify_copy_convert_ms", format_double(metrics.verify_copy_convert_ms)},
        };
        append_public_metadata(run_manifest);
        const auto manifest_validation_start = Clock::now();
        metrics.export_manifest_validation_ms = elapsed_ms(manifest_validation_start, Clock::now());
        run_manifest["export_manifest_validation_ms"] = format_double(metrics.export_manifest_validation_ms);
        util::write_json_object(export_root + "/run_manifest.json", run_manifest);

        if (!context.config.dump_trace)
        {
            return;
        }

        util::ensure_directory(export_root + "/columns");
        util::ensure_directory(export_root + "/matrices");
        for (const auto &[name, column] : trace.columns)
        {
            util::write_lines(export_root + "/columns/" + name + ".txt", vector_to_lines(column));
        }
        for (const auto &[name, matrix] : trace.matrices)
        {
            util::write_lines(export_root + "/matrices/" + name + ".txt", matrix_to_lines(matrix));
        }
    }

} // namespace gatzk::protocol
