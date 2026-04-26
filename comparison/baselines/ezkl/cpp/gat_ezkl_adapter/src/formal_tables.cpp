#include "gat_ezkl/formal_tables.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "gat_ezkl/hash.hpp"
#include "gat_ezkl/metadata.hpp"

namespace gat_ezkl {
namespace {

constexpr double kTraceScalarScale = 16.0;

std::int64_t quantize_float_model_value_signed(double value) {
    return static_cast<std::int64_t>(
        value >= 0.0 ? value * kTraceScalarScale + 0.5 : value * kTraceScalarScale - 0.5);
}

double dequantize_trace_scalar(std::int64_t value) {
    return static_cast<double>(value) / kTraceScalarScale;
}

std::size_t elu_table_band() {
    return 1;
}

std::size_t exp_table_band() {
    return 2;
}

std::uint64_t mix_i64(std::uint64_t state, std::int64_t value) {
    return fnv1a64_bytes(&value, sizeof(value), state);
}

void write_i64(std::ofstream& output, std::int64_t value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!output) {
        throw std::runtime_error("failed while writing formal table binary payload");
    }
}

std::string finish_hash(std::uint64_t state) {
    return hex_u64(state);
}

std::string hash_summary_fields(
    const AppConfig& config,
    const FormalTableHashes& hashes,
    std::size_t range_bits,
    std::size_t range_size,
    std::size_t lrelu_bound,
    std::size_t elu_bound) {
    std::ostringstream out;
    out << "derived_from_main_formal_static_tables_v1\n"
        << config.quant_cfg_id << '\n'
        << config.static_table_id << '\n'
        << range_bits << '\n'
        << range_size << '\n'
        << lrelu_bound << '\n'
        << elu_bound << '\n'
        << hashes.range_hash << '\n'
        << hashes.lrelu_hash << '\n'
        << hashes.elu_hash << '\n'
        << hashes.exp_hash << '\n'
        << hashes.inv_hash << '\n';
    return hex_u64(fnv1a64_string(out.str()));
}

std::size_t effective_range_size(const AppConfig& config) {
    if (config.range_bits == 0 || config.range_bits >= 62) {
        throw std::runtime_error("range_bits must be in [1, 61] for formal table generation");
    }
    const auto configured = std::size_t{1} << config.range_bits;
    const bool large_whole_graph_gat =
        (config.dataset == "ogbn_arxiv" || config.dataset == "ogbn-arxiv")
        && config.batching_rule == "whole_graph_single";
    return std::max<std::size_t>(configured, large_whole_graph_gat ? (std::size_t{1} << 19) : configured);
}

}  // namespace

FormalTablesSummary write_formal_table_artifacts(
    const std::filesystem::path& output_dir,
    const AppConfig& config) {
    std::filesystem::create_directories(output_dir);

    FormalTablesSummary summary;
    summary.quant_cfg_id = config.quant_cfg_id;
    summary.static_table_id = config.static_table_id;
    summary.range_bits = config.range_bits;
    summary.range_size = effective_range_size(config);
    summary.range_entries = summary.range_size;
    summary.signed_bound = static_cast<std::int64_t>(std::int64_t{1} << (config.range_bits - 1));
    const bool large_whole_graph_gat =
        (config.dataset == "ogbn_arxiv" || config.dataset == "ogbn-arxiv")
        && config.batching_rule == "whole_graph_single";
    summary.lrelu_bound = std::max<std::size_t>(
        large_whole_graph_gat ? (std::size_t{1} << 20) : 4096,
        summary.range_size);
    summary.elu_bound = summary.lrelu_bound;
    summary.elu_band = elu_table_band();
    summary.exp_band = exp_table_band();
    summary.lrelu_entries = summary.lrelu_bound * 2 + 1;
    summary.elu_entries = (summary.elu_bound * 2 + 1) * (2 * summary.elu_band + 1);
    summary.exp_entries = summary.range_size * (2 * summary.exp_band + 1);
    summary.inv_entries = 0;

    const auto bin_path = output_dir / "formal_tables.bin";
    std::ofstream bin(bin_path, std::ios::binary | std::ios::trunc);
    if (!bin) {
        throw std::runtime_error("failed to write formal table binary: " + bin_path.string());
    }

    std::uint64_t range_state = 1469598103934665603ULL;
    for (std::size_t i = 0; i < summary.range_size; ++i) {
        const auto value = static_cast<std::int64_t>(i);
        write_i64(bin, value);
        range_state = mix_i64(range_state, value);
    }
    summary.hashes.range_hash = finish_hash(range_state);

    std::uint64_t lrelu_state = 1469598103934665603ULL;
    for (std::int64_t raw = -static_cast<std::int64_t>(summary.lrelu_bound);
         raw <= static_cast<std::int64_t>(summary.lrelu_bound);
         ++raw) {
        const auto input = dequantize_trace_scalar(raw);
        const auto output = input >= 0.0 ? input : 0.2 * input;
        const auto out = quantize_float_model_value_signed(output);
        write_i64(bin, raw);
        write_i64(bin, out);
        lrelu_state = mix_i64(mix_i64(lrelu_state, raw), out);
    }
    summary.hashes.lrelu_hash = finish_hash(lrelu_state);

    std::uint64_t elu_state = 1469598103934665603ULL;
    for (std::int64_t raw = -static_cast<std::int64_t>(summary.elu_bound);
         raw <= static_cast<std::int64_t>(summary.elu_bound);
         ++raw) {
        const auto input = dequantize_trace_scalar(raw);
        const auto central_signed = quantize_float_model_value_signed(
            input >= 0.0 ? input : std::exp(input) - 1.0);
        for (std::int64_t offset = -static_cast<std::int64_t>(summary.elu_band);
             offset <= static_cast<std::int64_t>(summary.elu_band);
             ++offset) {
            const auto out = central_signed + offset;
            write_i64(bin, raw);
            write_i64(bin, out);
            elu_state = mix_i64(mix_i64(elu_state, raw), out);
        }
    }
    summary.hashes.elu_hash = finish_hash(elu_state);

    std::uint64_t exp_state = 1469598103934665603ULL;
    for (std::size_t i = 0; i < summary.range_size; ++i) {
        const auto input = static_cast<std::int64_t>(i);
        const auto central_signed = quantize_float_model_value_signed(
            std::exp(-dequantize_trace_scalar(input)));
        for (std::int64_t offset = -static_cast<std::int64_t>(summary.exp_band);
             offset <= static_cast<std::int64_t>(summary.exp_band);
             ++offset) {
            const auto out = std::max<std::int64_t>(0, central_signed + offset);
            write_i64(bin, input);
            write_i64(bin, out);
            exp_state = mix_i64(mix_i64(exp_state, input), out);
        }
    }
    summary.hashes.exp_hash = finish_hash(exp_state);
    summary.hashes.inv_hash = "unused:exact_sum_times_inv_gate";

    bin.close();
    summary.hashes.bin_hash = hash_file_hex(bin_path);
    summary.hashes.table_hash = hash_summary_fields(
        config,
        summary.hashes,
        summary.range_bits,
        summary.range_size,
        summary.lrelu_bound,
        summary.elu_bound);

    write_text_file(output_dir / "formal_tables.json", formal_tables_json(summary));
    write_text_file(output_dir / "formal_table_hashes.json", formal_table_hashes_json(summary));
    return summary;
}

std::string formal_tables_json(const FormalTablesSummary& summary) {
    std::ostringstream out;
    out << "{\n"
        << "  \"source\": \"" << json_escape(summary.source) << "\",\n"
        << "  \"table_policy\": \"" << json_escape(summary.table_policy) << "\",\n"
        << "  \"quant_cfg_id\": \"" << json_escape(summary.quant_cfg_id) << "\",\n"
        << "  \"static_table_id\": \"" << json_escape(summary.static_table_id) << "\",\n"
        << "  \"range_bits\": " << summary.range_bits << ",\n"
        << "  \"signed_bound\": " << summary.signed_bound << ",\n"
        << "  \"range_size\": " << summary.range_size << ",\n"
        << "  \"lrelu_bound\": " << summary.lrelu_bound << ",\n"
        << "  \"elu_bound\": " << summary.elu_bound << ",\n"
        << "  \"elu_band\": " << summary.elu_band << ",\n"
        << "  \"exp_band\": " << summary.exp_band << ",\n"
        << "  \"range_entries\": " << summary.range_entries << ",\n"
        << "  \"lrelu_entries\": " << summary.lrelu_entries << ",\n"
        << "  \"elu_entries\": " << summary.elu_entries << ",\n"
        << "  \"exp_entries\": " << summary.exp_entries << ",\n"
        << "  \"inv_entries\": " << summary.inv_entries << ",\n"
        << "  \"scale_policy\": \"" << json_escape(summary.scale_policy) << "\",\n"
        << "  \"inverse_policy\": \"" << json_escape(summary.inverse_policy) << "\",\n"
        << "  \"fp32_weights_checkpoint_imported\": true,\n"
        << "  \"proving_semantics\": \"quantized_finite_table_formalization\",\n"
        << "  \"deterministic_from_public_config\": true,\n"
        << "  \"witness_specific_table\": false,\n"
        << "  \"table_hash\": \"" << summary.hashes.table_hash << "\",\n"
        << "  \"binary_path\": \"formal_tables.bin\",\n"
        << "  \"binary_hash\": \"" << summary.hashes.bin_hash << "\"\n"
        << "}\n";
    return out.str();
}

std::string formal_table_hashes_json(const FormalTablesSummary& summary) {
    std::ostringstream out;
    out << "{\n"
        << "  \"source\": \"" << json_escape(summary.source) << "\",\n"
        << "  \"table_policy\": \"" << json_escape(summary.table_policy) << "\",\n"
        << "  \"quant_cfg_id\": \"" << json_escape(summary.quant_cfg_id) << "\",\n"
        << "  \"static_table_id\": \"" << json_escape(summary.static_table_id) << "\",\n"
        << "  \"range_bits\": " << summary.range_bits << ",\n"
        << "  \"signed_bound\": " << summary.signed_bound << ",\n"
        << "  \"range_hash\": \"" << summary.hashes.range_hash << "\",\n"
        << "  \"lrelu_hash\": \"" << summary.hashes.lrelu_hash << "\",\n"
        << "  \"elu_hash\": \"" << summary.hashes.elu_hash << "\",\n"
        << "  \"exp_hash\": \"" << summary.hashes.exp_hash << "\",\n"
        << "  \"inv_hash\": \"" << summary.hashes.inv_hash << "\",\n"
        << "  \"table_hash\": \"" << summary.hashes.table_hash << "\",\n"
        << "  \"bin_hash\": \"" << summary.hashes.bin_hash << "\"\n"
        << "}\n";
    return out.str();
}

std::string formal_table_binding_json(const FormalTablesSummary& summary) {
    std::ostringstream out;
    out << "{\"source\":\"" << json_escape(summary.source)
        << "\",\"table_policy\":\"" << json_escape(summary.table_policy)
        << "\",\"quant_cfg_id\":\"" << json_escape(summary.quant_cfg_id)
        << "\",\"static_table_id\":\"" << json_escape(summary.static_table_id)
        << "\",\"range_bits\":" << summary.range_bits
        << ",\"signed_bound\":" << summary.signed_bound
        << ",\"range_hash\":\"" << summary.hashes.range_hash
        << "\",\"lrelu_hash\":\"" << summary.hashes.lrelu_hash
        << "\",\"elu_hash\":\"" << summary.hashes.elu_hash
        << "\",\"exp_hash\":\"" << summary.hashes.exp_hash
        << "\",\"inv_hash\":\"" << summary.hashes.inv_hash
        << "\",\"table_hash\":\"" << summary.hashes.table_hash
        << "\",\"bin_hash\":\"" << summary.hashes.bin_hash
        << "\"}";
    return out.str();
}

}  // namespace gat_ezkl
