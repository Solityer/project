#include "gatzk/protocol/prover.hpp"

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
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
#include <openssl/sha.h>

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
#include "gatzk/protocol/verifier_context.hpp"
#include "gatzk/build_provenance.hpp"

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

        std::string sha256_hex(std::string_view text)
        {
            std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
            SHA256(
                reinterpret_cast<const unsigned char *>(text.data()),
                text.size(),
                digest.data());
            std::ostringstream out;
            out << std::hex << std::setfill('0');
            for (const auto byte : digest)
                out << std::setw(2) << static_cast<unsigned int>(byte);
            return out.str();
        }

        std::string hash_key_values(const std::map<std::string, std::string> &entries)
        {
            std::ostringstream stream;
            for (const auto &[key, value] : entries)
            {
                stream << key << '=' << value << '\n';
            }
            return sha256_hex(stream.str());
        }
        std::string sha256_hex(const std::vector<std::uint8_t> &bytes)
        {
            std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
            SHA256(bytes.data(), bytes.size(), digest.data());
            std::ostringstream out;
            out << std::hex << std::setfill('0');
            for (const auto byte : digest)
            {
                out << std::setw(2) << static_cast<unsigned int>(byte);
            }
            return out.str();
        }

        std::string sha256_file_hex(const std::filesystem::path &path)
        {
            std::ifstream input(path, std::ios::binary);
            if (!input)
            {
                throw std::runtime_error(
                    "failed to open artifact for SHA-256: " + path.string());
            }
            SHA256_CTX context{};
            SHA256_Init(&context);
            std::array<char, 1U << 16U> buffer{};
            while (input)
            {
                input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
                const auto count = input.gcount();
                if (count > 0)
                {
                    SHA256_Update(
                        &context,
                        buffer.data(),
                        static_cast<std::size_t>(count));
                }
            }
            if (input.bad())
            {
                throw std::runtime_error(
                    "failed while reading artifact for SHA-256: " + path.string());
            }
            std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
            SHA256_Final(digest.data(), &context);
            std::ostringstream out;
            out << std::hex << std::setfill('0');
            for (const auto byte : digest)
            {
                out << std::setw(2) << static_cast<unsigned int>(byte);
            }
            return out.str();
        }

        std::vector<std::uint8_t> read_binary_file(
            const std::filesystem::path &path)
        {
            std::ifstream input(path, std::ios::binary);
            if (!input)
            {
                throw std::runtime_error(
                    "failed to open artifact for byte comparison: " + path.string());
            }
            input.seekg(0, std::ios::end);
            const auto end = input.tellg();
            if (end < 0)
                throw std::runtime_error(
                    "failed to seek artifact for byte comparison: " + path.string());
            input.seekg(0, std::ios::beg);
            std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
            if (!bytes.empty() && !input.read(
                                      reinterpret_cast<char *>(bytes.data()),
                                      static_cast<std::streamsize>(bytes.size())))
            {
                throw std::runtime_error(
                    "failed to read artifact for byte comparison: " + path.string());
            }
            return bytes;
        }

        void write_exact_file(
            const std::filesystem::path &path,
            std::string_view bytes)
        {
            std::ofstream output(path, std::ios::binary | std::ios::trunc);
            if (!output || (!bytes.empty() && !output.write(
                                                  bytes.data(),
                                                  static_cast<std::streamsize>(bytes.size()))))
            {
                throw std::runtime_error(
                    "failed to write artifact: " + path.string());
            }
        }

        void append_u64(std::string *out, std::uint64_t value)
        {
            for (std::size_t i = 0; i < sizeof(value); ++i)
                out->push_back(static_cast<char>((value >> (8U * i)) & 0xffU));
        }

        void append_string(std::string *out, std::string_view value)
        {
            append_u64(out, value.size());
            out->append(value.data(), value.size());
        }

        std::string build_identity(
            const std::string &source_tree_hash,
            const std::string &binary_hash)
        {
            std::string bytes;
            append_string(&bytes, gatzk::build_provenance::kBuildIdentityAlgorithm);
            append_string(&bytes, source_tree_hash);
            append_string(&bytes, gatzk::build_provenance::kCompilerId);
            append_string(&bytes, gatzk::build_provenance::kCompilerVersion);
            append_string(&bytes, gatzk::build_provenance::kBuildType);
            append_string(&bytes, gatzk::build_provenance::kCompileFlags);
            append_string(&bytes, binary_hash);
            return sha256_hex(bytes);
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
        const VerifierContext &verifier_context,
        const Proof &parsed_proof,
        RunMetrics metrics,
        bool verifier_accepted,
        bool final_accepted)
    {
        const auto export_start = Clock::now();
        const auto export_root = (std::filesystem::path(context.config.project_root) / context.config.export_dir).string();
        util::ensure_directory(export_root);
        const auto proof_bytes = serialize_proof(proof);
        if (proof_bytes.size() != metrics.proof_size_bytes)
        {
            throw std::runtime_error(
                "exported canonical proof size differs from measured proof size");
        }
        {
            const auto proof_path = std::filesystem::path(export_root) / "proof.bin";
            write_exact_file(
                proof_path,
                std::string_view(
                    reinterpret_cast<const char *>(proof_bytes.data()),
                    proof_bytes.size()));
            if (read_binary_file(proof_path) != proof_bytes)
                throw std::runtime_error(
                    "canonical proof artifact is not byte-identical to serialized proof");
        }

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
        auto resolve_project_path = [&](const std::string &configured)
        {
            const auto path = std::filesystem::path(configured);
            if (path.is_absolute())
            {
                return path;
            }
            return std::filesystem::path(context.config.project_root) / path;
        };
        const auto srs_manifest_path =
            resolve_project_path(context.config.srs_path) / "srs_manifest.json";
        const auto srs_manifest_sha256 =
            sha256_file_hex(srs_manifest_path);
        if (srs_manifest_sha256 != context.kzg.manifest.manifest_hash)
        {
            throw std::runtime_error(
                "SRS manifest changed after authenticated context load");
        }
        if (proof.public_metadata.input_preprocess_id.empty() || proof.public_metadata.input_preprocess_id != context.config.input_preprocess_id)
        {
            throw std::runtime_error(
                "artifact export input_preprocess_id is not config-bound");
        }
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
                {"input_preprocess_id", proof.public_metadata.input_preprocess_id},
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
        if (!validate_verifier_context(verifier_context) || verifier_context.trusted_registry_digest.empty() || verifier_context.authenticated_registry_digest != verifier_context.trusted_registry_digest)
            throw std::runtime_error(
                "artifact export requires the independently loaded authenticated verifier context");
        if (serialize_proof(parsed_proof) != proof_bytes)
            throw std::runtime_error(
                "artifact export parsed proof is not byte-identical to proof.bin");
        const auto parsed_output_hash = canonical_output_digest(
            parsed_proof.public_logits,
            verifier_context.local.num_nodes,
            verifier_context.local.num_classes);
        if (trace.proof_public_output_hash.empty() || trace.proof_public_output_hash != parsed_output_hash)
            throw std::runtime_error(
                "artifact export proof_public_output_hash is not derived from parsed proof output");
        const auto source_tree_hash = std::string(
            gatzk::build_provenance::kSourceTreeHash);
        if (source_tree_hash.size() != 64 || !std::all_of(
                                                 source_tree_hash.begin(), source_tree_hash.end(),
                                                 [](unsigned char ch)
                                                 {
                                                     return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
                                                 }))
            throw std::runtime_error("embedded build source inventory hash is invalid");
        std::error_code binary_error;
        const auto binary_target = std::filesystem::read_symlink(
            "/proc/self/exe", binary_error);
        if (binary_error || binary_target.empty())
            throw std::runtime_error(
                "formal artifact requires the actual executable bytes via /proc/self/exe");
        const auto binary_path = std::filesystem::absolute(binary_target).lexically_normal();
        if (binary_path.empty() || std::filesystem::is_symlink(std::filesystem::symlink_status(binary_path, binary_error)) || binary_error || !std::filesystem::is_regular_file(binary_path, binary_error))
            throw std::runtime_error(
                "formal artifact executable target is not a regular non-symlink file");
        const auto binary_hash = sha256_file_hex(binary_path);
        const auto build_identity_value = build_identity(source_tree_hash, binary_hash);
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
        const auto table_endpoint_hash = sha256_hex(
            table_endpoint_string(context.tables.range) + "|" + pair_table_endpoint_string(context.tables.lrelu) + "|" + pair_table_endpoint_string(context.tables.exp) + "|" + pair_table_endpoint_string(context.tables.elu));
        const auto audit_start = Clock::now();
        const auto audited_private_plain_value_count =
            private_plain_value_count(verifier_context, parsed_proof);
        if (metrics.payload_audit_ms <= 0.0)
        {
            metrics.payload_audit_ms = elapsed_ms(audit_start, Clock::now());
        }

        std::map<std::string, std::string> metadata;
        append_public_metadata(metadata);
        const auto public_metadata_hash = hash_key_values(metadata);
        const auto proof_hash = sha256_hex(proof_bytes);
        const auto config_id =
            context.config.model_arch_id + ":" + context.config.quant_cfg_id + ":" + context.config.static_table_id + ":" + context.config.degree_bound_id;
        const auto verifier_registry_digest =
            verifier_context.authenticated_registry_digest;
        const auto source_manifest_path =
            std::filesystem::path(export_root) / "source_manifest.json";
        write_exact_file(
            source_manifest_path,
            gatzk::build_provenance::kSourceManifestJson);
        const auto source_manifest_hash = sha256_file_hex(source_manifest_path);
        const std::map<std::string, std::string> build_receipt = {
            {"format", "zkgat_build_receipt"},
            {"source_manifest_path", "source_manifest.json"},
            {"source_manifest_hash", source_manifest_hash},
            {"source_inventory_id", gatzk::build_provenance::kSourceInventoryId},
            {"source_tree_hash", source_tree_hash},
            {"compiler_id", gatzk::build_provenance::kCompilerId},
            {"compiler_version", gatzk::build_provenance::kCompilerVersion},
            {"build_type", gatzk::build_provenance::kBuildType},
            {"compile_flags", gatzk::build_provenance::kCompileFlags},
            {"binary_path", binary_path.string()},
            {"binary_hash", binary_hash},
            {"build_identity_algorithm", gatzk::build_provenance::kBuildIdentityAlgorithm},
            {"build_identity", build_identity_value},
        };
        util::write_json_object(export_root + "/build_receipt.json", build_receipt);
        const auto build_receipt_hash = sha256_file_hex(
            std::filesystem::path(export_root) / "build_receipt.json");
        // This report is the independent, binary-produced acceptance record
        // consumed by the aggregate gate.  In particular, the security and
        // tamper fields are copied from the completed production checks here,
        // then bound to the proof, authenticated registry, public-input
        // identity, and exact binary/build provenance.  The aggregate gate
        // never promotes the corresponding run-manifest fields on their own.
        const std::map<std::string, std::string> verifier_report = {
            {"format", "zkgat_verifier_report"},
            {"trusted_registry_digest", verifier_context.trusted_registry_digest},
            {"authenticated_registry_digest", verifier_context.authenticated_registry_digest},
            {"proof_bin_sha256", proof_hash},
            {"parsed_output_digest", parsed_output_hash},
            {"proof_public_output_hash", parsed_output_hash},
            {"parsed_output_count", std::to_string(parsed_proof.public_logits.size())},
            {"verify_result", verifier_accepted ? "true" : "false"},
            {"reference_consistency_passed", metrics.reference_consistency_passed ? "true" : "false"},
            {"tamper_tests_passed", metrics.tamper_tests_passed ? "true" : "false"},
            {"tamper_test_case_map", metrics.tamper_test_case_map},
            {"security_budget_passed", metrics.security_budget_passed ? "true" : "false"},
            {"security_budget_ledger_digest", metrics.security_budget_ledger_digest},
            {"security_budget_exact", metrics.security_budget_exact},
            {"public_input_digest", verifier_context.registry.public_input_digest},
            {"topology_digest", verifier_context.registry.topology_digest},
            {"source_tree_hash", source_tree_hash},
            {"binary_hash", binary_hash},
            {"build_identity", build_identity_value},
        };
        util::write_json_object(export_root + "/verifier_report.json", verifier_report);
        const auto verifier_report_hash = sha256_file_hex(
            std::filesystem::path(export_root) / "verifier_report.json");
        util::write_key_values(export_root + "/metadata.txt", metadata);
        util::write_lines(export_root + "/proof_block_order.txt", proof.block_order);

        util::write_lines(export_root + "/commitment_order.txt", trace.commitment_order);
        // `benchmark.txt` 只保留正式交付面需要的最小结果，避免把内部剖面重新暴露出来。
        const std::map<std::string, std::string> benchmark_summary = {
            {"dataset", metrics.dataset},
            {"config_id", config_id},
            {"backend", benchmark_backend_label()},
            {"status", final_accepted ? "VERIFY_OK" : "VERIFY_FAIL"},
            {"timing_schema", kFormalTimingSchema},
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
                "label,dataset,domain,basis,vector_length,domain_size,nonzero_scalar_count,effective_scalar_count,zero_scalar_ratio,padding_ratio,classification,materialization_ms,msm_ms,total_commitment_ms,duplicates_another,duplicate_of,reused_shared,shared_or_borrowed,compact,ownership,category,optimization_opportunity,zero_scalar_skipping,sparse_compaction,constant_tail_folding,multithreaded_msm,commitment_fingerprint",
            };
            profile_lines.reserve(metrics.dynamic_commitment_profiles.size() + 1);
            for (const auto &item : metrics.dynamic_commitment_profiles)
            {
                dynamic_profile_count_by_domain[item.domain] += 1;
                dynamic_profile_time_by_domain[item.domain] += item.total_commitment_ms;
                dynamic_profile_time_by_category[item.ownership + "/" + item.category] += item.total_commitment_ms;
                profile_lines.push_back(
                    csv_escape(item.label) + "," + csv_escape(item.dataset) + "," + csv_escape(item.domain) + "," + csv_escape(item.basis) + "," + std::to_string(item.vector_length) + "," + std::to_string(item.domain_size) + "," + std::to_string(item.nonzero_scalar_count) + "," + std::to_string(item.effective_scalar_count) + "," + format_double(item.zero_scalar_ratio) + "," + format_double(item.padding_ratio) + "," + csv_escape(item.classification) + "," + format_double(item.materialization_ms) + "," + format_double(item.msm_ms) + "," + format_double(item.total_commitment_ms) + "," + bool_string(item.duplicates_another) + "," + csv_escape(item.duplicate_of) + "," + bool_string(item.reused_shared) + "," + bool_string(item.shared_or_borrowed) + "," + bool_string(item.compact) + "," + csv_escape(item.ownership) + "," + csv_escape(item.category) + "," + csv_escape(item.optimization_opportunity) + "," + bool_string(item.zero_scalar_skipping) + "," + bool_string(item.sparse_compaction) + "," + bool_string(item.constant_tail_folding) + "," + bool_string(item.multithreaded_msm) + "," + csv_escape(item.commitment_fingerprint));
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
        metrics.formal_time_sum_ms =
            metrics.commitment_time_ms + metrics.prove_time_ms + metrics.verify_time_ms;
        metrics.non_formal_overhead_ms = std::max(
            0.0,
            metrics.total_wall_clock_ms - metrics.formal_time_sum_ms);
        // `run_manifest.json` is public.  Keep only opaque identifiers and
        // execution results here; checkpoint/tensor/source provenance belongs
        // to the private setup environment and must never enter the aggregate.
        std::map<std::string, std::string> run_manifest = {
            {"backend", benchmark_backend_label()},
            {"backend_name", metrics.backend_name},
            {"crypto_backend_name", metrics.crypto_backend_name},
            {"algebra_backend_name", metrics.algebra_backend_name},
            {"compute_backend_name", metrics.compute_backend_name},
            {"config_id", config_id},
            {"config", metrics.config},
            {"config_sha256", sha256_file_hex(std::filesystem::path(metrics.config))},
            {"dataset", metrics.dataset},
            {"input_preprocess_id", context.config.input_preprocess_id},
            {"status", final_accepted ? "VERIFY_OK" : "VERIFY_FAIL"},
            {"verifier_accepted", verifier_accepted ? "true" : "false"},
            {"final_formal_run_accepted", final_accepted ? "true" : "false"},
            {"benchmark_mode", metrics.benchmark_mode},
            {"timing_schema", kFormalTimingSchema},
            {"enabled_packed_fft_backend", metrics.enabled_packed_fft_backend ? "true" : "false"},
            {"enabled_rotated_fft_kernel", metrics.enabled_rotated_fft_kernel ? "true" : "false"},
            {"enabled_fast_msm", metrics.enabled_fast_msm ? "true" : "false"},
            {"enabled_parallel_fft", metrics.enabled_parallel_fft ? "true" : "false"},
            {"enabled_compact_trace_layout", metrics.enabled_compact_trace_layout ? "true" : "false"},
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
            {"gpu_h2d_bytes", std::to_string(metrics.gpu_h2d_bytes)},
            {"gpu_d2h_bytes", std::to_string(metrics.gpu_d2h_bytes)},
            {"gpu_h2d_operations", std::to_string(metrics.gpu_h2d_operations)},
            {"gpu_d2h_operations", std::to_string(metrics.gpu_d2h_operations)},
             {"gpu_stream_synchronizations", std::to_string(metrics.gpu_stream_synchronizations)},
            {"gpu_ntt_calls", std::to_string(metrics.gpu_ntt_calls)},
            {"gpu_intt_calls", std::to_string(metrics.gpu_intt_calls)},
            {"gpu_quotient_calls", std::to_string(metrics.gpu_quotient_calls)},
            {"gpu_t_edge_calls", std::to_string(metrics.gpu_t_edge_calls)},
            {"gpu_opening_fold_calls", std::to_string(metrics.gpu_opening_fold_calls)},
            {"gpu_msm_single_calls", std::to_string(metrics.gpu_msm_single_calls)},
            {"gpu_msm_batch_calls", std::to_string(metrics.gpu_msm_batch_calls)},
            {"gpu_resident_hits", std::to_string(metrics.gpu_resident_hits)},
            {"gpu_resident_misses", std::to_string(metrics.gpu_resident_misses)},
            {"reference_consistency_passed", metrics.reference_consistency_passed ? "true" : "false"},
            {"node_count", std::to_string(metrics.node_count)},
            {"edge_count", std::to_string(metrics.edge_count)},
            {"gpu_runtime_present", metrics.gpu_runtime_present ? "true" : "false"},
            {"stale_result", "false"},
            {"activation_table_policy", large_whole_graph_gat ? "public_large_whole_graph_shape_bound" : "public_default_shape_bound"},
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
            {"formal_time_sum_ms", format_double(metrics.formal_time_sum_ms)},
            {"non_formal_overhead_ms", format_double(metrics.non_formal_overhead_ms)},
            {"reference_forward_ms", format_double(metrics.reference_forward_ms)},
            {"payload_audit_ms", format_double(metrics.payload_audit_ms)},
            {"reference_consistency_ms", format_double(metrics.reference_consistency_ms)},
            {"tamper_tests_ms", format_double(metrics.tamper_tests_ms)},
            {"export_artifact_ms", format_double(metrics.export_artifact_ms)},
            {"export_manifest_validation_ms", format_double(metrics.export_manifest_validation_ms)},
            {"profile_logging_overhead_ms", format_double(metrics.profile_logging_overhead_ms)},
            {"context_build_ms", format_double(metrics.context_build_ms)},
            {"domain_opening_ms", format_double(metrics.domain_opening_ms)},
            {"point_group_registry_ms", format_double(metrics.point_group_registry_ms)},
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
            {"dynamic_commit_input_ms", format_double(metrics.dynamic_commit_input_ms)},
            {"dynamic_commit_pack_ms", format_double(metrics.dynamic_commit_pack_ms)},
            {"dynamic_fft_ms", format_double(metrics.dynamic_fft_ms)},
            {"dynamic_domain_convert_ms", format_double(metrics.dynamic_domain_convert_ms)},
            {"dynamic_copy_convert_ms", format_double(metrics.dynamic_copy_convert_ms)},
            {"dynamic_commit_msm_ms", format_double(metrics.dynamic_commit_msm_ms)},
            {"derived_commitment_point_ms", format_double(metrics.derived_commitment_point_ms)},
            {"dynamic_bundle_finalize_ms", format_double(metrics.dynamic_bundle_finalize_ms)},
            {"dynamic_polynomial_materialization_ms", format_double(metrics.dynamic_polynomial_materialization_ms)},
            {"dynamic_commitment_object_profile_csv", "dynamic_commitment_object_profile.csv"},
            {"dynamic_commitment_object_count", std::to_string(metrics.dynamic_commitment_profiles.size())},
            {"dynamic_commitment_count_by_domain", format_size_map(dynamic_profile_count_by_domain)},
            {"dynamic_commitment_time_by_domain_ms", format_double_map(dynamic_profile_time_by_domain)},
            {"dynamic_commitment_time_by_ownership_category_ms", format_double_map(dynamic_profile_time_by_category)},
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
            {"tamper_tests_passed", metrics.tamper_tests_passed ? "true" : "false"},
            {"tamper_test_case_map", metrics.tamper_test_case_map},
            {"security_budget_passed", metrics.security_budget_passed ? "true" : "false"},
            {"security_budget_ledger_digest", metrics.security_budget_ledger_digest},
            {"security_budget_exact", metrics.security_budget_exact},
            {"full_dataset", metrics.is_full_dataset ? "true" : "false"},
            {"dataset_id", context.dataset.name},
            {"static_table_hash", sha256_hex(proof.public_metadata.static_tables)},
            {"quant_config_hash", sha256_hex(proof.public_metadata.quant_config)},
            {"public_input_digest", verifier_context.registry.public_input_digest},
            {"topology_digest", verifier_context.registry.topology_digest},
            {"source_tree_hash", source_tree_hash},
            {"source_manifest_path", "source_manifest.json"},
            {"source_manifest_hash", source_manifest_hash},
            {"binary_hash", binary_hash},
            {"build_identity", build_identity_value},
            {"build_receipt_path", "build_receipt.json"},
            {"build_receipt_hash", build_receipt_hash},
            {"binary_path", binary_path.string()},
            {"verifier_registry_id", verifier_context.registry.registry_id},
            {"verifier_registry_digest", verifier_registry_digest},
            {"trusted_registry_digest", verifier_context.trusted_registry_digest},
            {"verifier_report_path", "verifier_report.json"},
            {"verifier_report_hash", verifier_report_hash},
            {"public_metadata_hash", public_metadata_hash},
            {"srs_manifest_hash", srs_manifest_sha256},
            {"proof_hash", proof_hash},
            {"reference_fp32_logits_hash", trace.reference_fp32_logits_hash},
            {"reference_quant_logits_hash", trace.reference_quant_logits_hash},
            {"secure_trace_logits_hash", trace.secure_trace_logits_hash},
            {"proof_public_output_hash", trace.proof_public_output_hash},
            {"security_profile_id", context.config.security_profile_id},
            {"prove_accounted_ms", format_double(metrics.prove_accounted_ms)},
            {"prove_accounting_gap_ms", format_double(metrics.prove_accounting_gap_ms)},
            {"prove_time_ms", format_double(metrics.prove_time_ms)},
            {"proof_serialization_ms", format_double(metrics.proof_serialization_ms)},
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
            {"verified", verifier_accepted ? "true" : "false"},
            {"verify_time_ms", format_double(metrics.verify_time_ms)},
            {"proof_parse_ms", format_double(metrics.proof_parse_ms)},
            {"verify_metadata_ms", format_double(metrics.verify_metadata_ms)},
            {"verify_transcript_ms", format_double(metrics.verify_transcript_ms)},
            {"verify_domain_opening_ms", format_double(metrics.verify_domain_opening_ms)},
            {"verify_point_group_registry_ms", format_double(metrics.verify_point_group_registry_ms)},
            {"verify_quotient_ms", format_double(metrics.verify_quotient_ms)},
            {"verify_external_opening_ms", format_double(metrics.verify_external_opening_ms)},
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

        // Trace columns/matrices are prover witness material.  `dump_trace`
        // remains an execution/debug configuration bit, but public formal
        // artifacts never materialize those values.  A separately managed
        // private debug export, if needed, must be invoked outside this public
        // artifact writer.
    }

} // namespace gatzk::protocol
