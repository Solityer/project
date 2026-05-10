#include "gat_ezkl/formal_tables.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "gat_ezkl/hash.hpp"
#include "gat_ezkl/metadata.hpp"

namespace gat_ezkl 
{
    namespace 
    {
        // 激活函数查找表的量化缩放因子：浮点值 × 16 转成定点整数
        constexpr double kTraceScalarScale = 16.0;

        std::int64_t quantize_float_model_value_signed(double value) {
            return static_cast<std::int64_t>(
                value >= 0.0 ? value * kTraceScalarScale + 0.5 : value * kTraceScalarScale - 0.5);
        }

        // 反量化：int64 → 浮点
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
            const InverseQuantPolicy& inverse,
            const FormalTablesSummary& summary,
            std::size_t range_bits,
            std::size_t range_size,
            std::size_t lrelu_bound,
            std::size_t elu_bound) {
            std::ostringstream out;
            out << "derived_from_main_formal_static_tables_v1\n"
                << config.quant_cfg_id << '\n'
                << config.static_table_id << '\n'
                << config.degree_bound_id << '\n'
                << range_bits << '\n'
                << range_size << '\n'
                << lrelu_bound << '\n'
                << elu_bound << '\n'
                << summary.inverse_table_policy << '\n'
                << summary.degree_bound << '\n'
                << summary.exp_bound << '\n'
                << summary.derived_sum_bound << '\n'
                << summary.inverse_table_range_old << '\n'
                << summary.inverse_table_range_size << '\n'
                << hashes.range_hash << '\n'
                << hashes.lrelu_hash << '\n'
                << hashes.elu_hash << '\n'
                << hashes.exp_hash << '\n'
                << hashes.inv_hash << '\n';
            out << inverse.inverse_policy << '\n'
                << inverse.sum_scale << '\n'
                << inverse.inv_scale << '\n'
                << inverse.alpha_scale << '\n'
                << inverse.inverse_scale_product << '\n'
                << inverse.inverse_band << '\n'
                << inverse.alpha_rescale_policy << '\n'
                << inverse.alpha_rescale_divisor << '\n'
                << inverse.rounding_rule << '\n'
                << inverse.alpha_band << '\n'
                << hashes.inverse_policy_hash << '\n';
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

        // 返回不小于 value 的 2 的幂
        std::size_t next_power_of_two_at_least(std::size_t value) {
            if (value <= 1) {
                return 1;
            }
            std::size_t out = 1;
            while (out < value) {
                if (out > (std::numeric_limits<std::size_t>::max() >> 1)) {
                    throw std::runtime_error("inverse table range overflow while deriving formal table policy");
                }
                out <<= 1;
            }
            return out;
        }

        std::int64_t checked_sum_bound(std::size_t degree_bound, std::int64_t exp_bound) {
            if (exp_bound < 0) {
                throw std::runtime_error("negative exp bound is invalid for inverse table policy");
            }
            const auto degree = static_cast<unsigned long long>(degree_bound);
            const auto exp = static_cast<unsigned long long>(exp_bound);
            if (degree != 0 && exp > std::numeric_limits<unsigned long long>::max() / degree) {
                throw std::runtime_error("inverse table sum bound overflow");
            }
            const auto bound = degree * exp;
            if (bound > static_cast<unsigned long long>(std::numeric_limits<std::int64_t>::max())) {
                throw std::runtime_error("inverse table sum bound exceeds i64");
            }
            return static_cast<std::int64_t>(bound);
        }
    }  // namespace

    // 初始化逆量化策略，scale 乘积预先计算，策略整体做哈希
    InverseQuantPolicy derive_inverse_quant_policy(const AppConfig& config) {
        (void)config;
        InverseQuantPolicy policy;
        policy.inverse_scale_product = policy.sum_scale * policy.inv_scale;
        std::ostringstream serialized;
        serialized << "quantized_inverse_lookup_v1\n"
                << policy.sum_scale << '\n'
                << policy.inv_scale << '\n'
                << policy.alpha_scale << '\n'
                << policy.inverse_scale_product << '\n'
                << policy.inverse_band << '\n'
                << policy.rounding_rule << '\n'
                << policy.alpha_rescale_policy << '\n'
                << policy.alpha_rescale_divisor << '\n'
                << policy.alpha_band << '\n';
        policy.policy_hash = hex_u64(fnv1a64_string(serialized.str()));
        return policy;
    }

    // 定点逆运算
    std::int64_t quantized_inverse_for_sum(
        std::int64_t sum_q,
        const InverseQuantPolicy& policy) {
        if (sum_q <= 0) {
            return 0;
        }
        const auto numerator = policy.inverse_scale_product;
        const auto denominator = sum_q;
        return (2 * numerator + denominator) / (2 * denominator);
    }

    std::pair<std::int64_t, std::int64_t> quantized_inverse_interval_for_sum(
        std::int64_t sum_q,
        const InverseQuantPolicy& policy) {
        if (sum_q <= 0) {
            return {0, 0};
        }
        const auto doubled_numerator = 2 * policy.inverse_scale_product;
        const auto lower_denominator = 2 * sum_q - 1;
        const auto upper_denominator = 2 * sum_q + 1;
        const auto max_inv = (2 * doubled_numerator + lower_denominator) / (2 * lower_denominator);
        const auto min_inv = (2 * doubled_numerator + upper_denominator) / (2 * upper_denominator);
        return {min_inv, max_inv};
    }

    // 生成逆查找表：每个 sum_q 对应一个或多个合法 inv_q
    std::vector<InverseTableEntry> generate_inverse_table(
        const InverseQuantPolicy& policy,
        std::size_t range_size) {
        std::vector<InverseTableEntry> out;
        out.reserve(range_size + 32);
        for (std::size_t sum_q = 0; sum_q < range_size; ++sum_q) {
            const auto [min_inv, max_inv] = quantized_inverse_interval_for_sum(
                static_cast<std::int64_t>(sum_q),
                policy);
            for (auto inv_q = min_inv; inv_q <= max_inv; ++inv_q) {
                out.push_back({
                    static_cast<std::int64_t>(sum_q),
                    inv_q,
                });
            }
        }
        return out;
    }

    bool validate_inverse_witness_against_policy(
        std::int64_t sum_q,
        std::int64_t inv_q,
        const InverseQuantPolicy& policy,
        std::int64_t* expected_inv_q) {
        const auto expected = quantized_inverse_for_sum(sum_q, policy);
        if (expected_inv_q != nullptr) {
            *expected_inv_q = expected;
        }
        const auto [min_inv, max_inv] = quantized_inverse_interval_for_sum(sum_q, policy);
        return inv_q >= min_inv && inv_q <= max_inv;
    }

    bool validate_alpha_rescale(
        std::int64_t u_q,
        std::int64_t inv_q,
        std::int64_t alpha_q,
        const InverseQuantPolicy& policy,
        std::int64_t* expected_alpha_q) {
        const auto product = u_q * inv_q;
        const auto divisor = policy.alpha_rescale_divisor == 0 ? 1 : policy.alpha_rescale_divisor;
        const auto abs_product = product < 0 ? -product : product;
        const auto rounded_abs = (2 * abs_product + divisor) / (2 * divisor);
        const auto expected = product < 0 ? -rounded_abs : rounded_abs;
        if (expected_alpha_q != nullptr) {
            *expected_alpha_q = expected;
        }
        return std::llabs(alpha_q - expected) <= policy.alpha_band;
    }

    FormalTablesSummary write_formal_table_artifacts(
        const std::filesystem::path& output_dir,
        const AppConfig& config,
        std::size_t degree_bound) {
        std::filesystem::create_directories(output_dir);

        FormalTablesSummary summary;
        summary.quant_cfg_id = config.quant_cfg_id;
        summary.static_table_id = config.static_table_id;
        summary.degree_bound_id = config.degree_bound_id.empty() ? "auto" : config.degree_bound_id;
        summary.inverse = derive_inverse_quant_policy(config);
        summary.inverse_policy = summary.inverse.inverse_policy;
        summary.range_bits = config.range_bits;
        summary.range_size = effective_range_size(config);
        summary.range_entries = summary.range_size;
        summary.signed_bound = static_cast<std::int64_t>(std::int64_t{1} << (config.range_bits - 1));
        summary.degree_bound = std::max<std::size_t>(degree_bound, 1);
        summary.exp_bound = summary.inverse.sum_scale + static_cast<std::int64_t>(exp_table_band());
        summary.derived_sum_bound = checked_sum_bound(summary.degree_bound, summary.exp_bound);
        summary.inverse_table_range_old = summary.range_size;
        summary.inverse_table_range_size = next_power_of_two_at_least(
            static_cast<std::size_t>(summary.derived_sum_bound) + 1);
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
        summary.inverse_table_range_size = std::max<std::size_t>(
            std::size_t{1} << config.range_bits,
            summary.inverse_table_range_size);
        const auto inverse_table = generate_inverse_table(summary.inverse, summary.inverse_table_range_size);
        summary.inv_entries = inverse_table.size();

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

        std::uint64_t inv_state = 1469598103934665603ULL;
        for (const auto& entry : inverse_table) {
            write_i64(bin, entry.sum_q);
            write_i64(bin, entry.inv_q);
            inv_state = mix_i64(mix_i64(inv_state, entry.sum_q), entry.inv_q);
        }
        summary.hashes.inv_hash = finish_hash(inv_state);
        summary.hashes.inverse_policy_hash = summary.inverse.policy_hash;

        bin.close();
        summary.hashes.bin_hash = hash_file_hex(bin_path);
        summary.hashes.table_hash = hash_summary_fields(
            config,
            summary.hashes,
            summary.inverse,
            summary,
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
            << "  \"inverse_table_policy\": \"" << json_escape(summary.inverse_table_policy) << "\",\n"
            << "  \"quant_cfg_id\": \"" << json_escape(summary.quant_cfg_id) << "\",\n"
            << "  \"static_table_id\": \"" << json_escape(summary.static_table_id) << "\",\n"
            << "  \"degree_bound_id\": \"" << json_escape(summary.degree_bound_id) << "\",\n"
            << "  \"range_bits\": " << summary.range_bits << ",\n"
            << "  \"range_size\": " << summary.range_size << ",\n"
            << "  \"signed_bound\": " << summary.signed_bound << ",\n"
            << "  \"degree_bound\": " << summary.degree_bound << ",\n"
            << "  \"exp_bound\": " << summary.exp_bound << ",\n"
            << "  \"derived_sum_bound\": " << summary.derived_sum_bound << ",\n"
            << "  \"inverse_table_range_old\": " << summary.inverse_table_range_old << ",\n"
            << "  \"inverse_table_range_new\": " << summary.inverse_table_range_size << ",\n"
            << "  \"inverse_table_range_size\": " << summary.inverse_table_range_size << ",\n"
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
            << "  \"sum_scale\": " << summary.inverse.sum_scale << ",\n"
            << "  \"inv_scale\": " << summary.inverse.inv_scale << ",\n"
            << "  \"alpha_scale\": " << summary.inverse.alpha_scale << ",\n"
            << "  \"inverse_scale_product\": " << summary.inverse.inverse_scale_product << ",\n"
            << "  \"inverse_band\": " << summary.inverse.inverse_band << ",\n"
            << "  \"alpha_rescale_policy\": \"" << json_escape(summary.inverse.alpha_rescale_policy) << "\",\n"
            << "  \"alpha_rescale_divisor\": " << summary.inverse.alpha_rescale_divisor << ",\n"
            << "  \"alpha_rounding_rule\": \"" << json_escape(summary.inverse.rounding_rule) << "\",\n"
            << "  \"alpha_band\": " << summary.inverse.alpha_band << ",\n"
            << "  \"inverse_table_hash\": \"" << summary.hashes.inv_hash << "\",\n"
            << "  \"inverse_policy_hash\": \"" << summary.hashes.inverse_policy_hash << "\",\n"
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
            << "  \"inverse_table_policy\": \"" << json_escape(summary.inverse_table_policy) << "\",\n"
            << "  \"quant_cfg_id\": \"" << json_escape(summary.quant_cfg_id) << "\",\n"
            << "  \"static_table_id\": \"" << json_escape(summary.static_table_id) << "\",\n"
            << "  \"degree_bound_id\": \"" << json_escape(summary.degree_bound_id) << "\",\n"
            << "  \"range_bits\": " << summary.range_bits << ",\n"
            << "  \"range_size\": " << summary.range_size << ",\n"
            << "  \"signed_bound\": " << summary.signed_bound << ",\n"
            << "  \"degree_bound\": " << summary.degree_bound << ",\n"
            << "  \"exp_bound\": " << summary.exp_bound << ",\n"
            << "  \"derived_sum_bound\": " << summary.derived_sum_bound << ",\n"
            << "  \"inverse_table_range_old\": " << summary.inverse_table_range_old << ",\n"
            << "  \"inverse_table_range_new\": " << summary.inverse_table_range_size << ",\n"
            << "  \"inverse_table_range_size\": " << summary.inverse_table_range_size << ",\n"
            << "  \"range_hash\": \"" << summary.hashes.range_hash << "\",\n"
            << "  \"lrelu_hash\": \"" << summary.hashes.lrelu_hash << "\",\n"
            << "  \"elu_hash\": \"" << summary.hashes.elu_hash << "\",\n"
            << "  \"exp_hash\": \"" << summary.hashes.exp_hash << "\",\n"
            << "  \"inv_hash\": \"" << summary.hashes.inv_hash << "\",\n"
            << "  \"inverse_policy\": \"" << json_escape(summary.inverse_policy) << "\",\n"
            << "  \"sum_scale\": " << summary.inverse.sum_scale << ",\n"
            << "  \"inv_scale\": " << summary.inverse.inv_scale << ",\n"
            << "  \"alpha_scale\": " << summary.inverse.alpha_scale << ",\n"
            << "  \"inverse_scale_product\": " << summary.inverse.inverse_scale_product << ",\n"
            << "  \"inverse_band\": " << summary.inverse.inverse_band << ",\n"
            << "  \"alpha_rescale_policy\": \"" << json_escape(summary.inverse.alpha_rescale_policy) << "\",\n"
            << "  \"alpha_rescale_divisor\": " << summary.inverse.alpha_rescale_divisor << ",\n"
            << "  \"alpha_rounding_rule\": \"" << json_escape(summary.inverse.rounding_rule) << "\",\n"
            << "  \"alpha_band\": " << summary.inverse.alpha_band << ",\n"
            << "  \"inverse_table_hash\": \"" << summary.hashes.inv_hash << "\",\n"
            << "  \"inverse_policy_hash\": \"" << summary.hashes.inverse_policy_hash << "\",\n"
            << "  \"table_hash\": \"" << summary.hashes.table_hash << "\",\n"
            << "  \"bin_hash\": \"" << summary.hashes.bin_hash << "\",\n"
            << "  \"witness_specific_table\": false\n"
            << "}\n";
        return out.str();
    }

    std::string formal_table_binding_json(const FormalTablesSummary& summary) {
        std::ostringstream out;
        out << "{\"source\":\"" << json_escape(summary.source)
            << "\",\"table_policy\":\"" << json_escape(summary.table_policy)
            << "\",\"inverse_table_policy\":\"" << json_escape(summary.inverse_table_policy)
            << "\",\"quant_cfg_id\":\"" << json_escape(summary.quant_cfg_id)
            << "\",\"static_table_id\":\"" << json_escape(summary.static_table_id)
            << "\",\"degree_bound_id\":\"" << json_escape(summary.degree_bound_id)
            << "\",\"range_bits\":" << summary.range_bits
            << ",\"range_size\":" << summary.range_size
            << ",\"signed_bound\":" << summary.signed_bound
            << ",\"degree_bound\":" << summary.degree_bound
            << ",\"exp_bound\":" << summary.exp_bound
            << ",\"derived_sum_bound\":" << summary.derived_sum_bound
            << ",\"inverse_table_range_old\":" << summary.inverse_table_range_old
            << ",\"inverse_table_range_new\":" << summary.inverse_table_range_size
            << ",\"inverse_table_range_size\":" << summary.inverse_table_range_size
            << ",\"range_hash\":\"" << summary.hashes.range_hash
            << "\",\"lrelu_hash\":\"" << summary.hashes.lrelu_hash
            << "\",\"elu_hash\":\"" << summary.hashes.elu_hash
            << "\",\"exp_hash\":\"" << summary.hashes.exp_hash
            << "\",\"inv_hash\":\"" << summary.hashes.inv_hash
            << "\",\"inverse_policy\":\"" << json_escape(summary.inverse_policy)
            << "\",\"sum_scale\":" << summary.inverse.sum_scale
            << ",\"inv_scale\":" << summary.inverse.inv_scale
            << ",\"alpha_scale\":" << summary.inverse.alpha_scale
            << ",\"inverse_scale_product\":" << summary.inverse.inverse_scale_product
            << ",\"inverse_band\":" << summary.inverse.inverse_band
            << ",\"alpha_rescale_policy\":\"" << json_escape(summary.inverse.alpha_rescale_policy)
            << "\",\"alpha_rescale_divisor\":" << summary.inverse.alpha_rescale_divisor
            << ",\"alpha_rounding_rule\":\"" << json_escape(summary.inverse.rounding_rule)
            << "\",\"alpha_band\":" << summary.inverse.alpha_band
            << ",\"inverse_table_hash\":\"" << summary.hashes.inv_hash
            << "\",\"inverse_policy_hash\":\"" << summary.hashes.inverse_policy_hash
            << "\",\"table_hash\":\"" << summary.hashes.table_hash
            << "\",\"bin_hash\":\"" << summary.hashes.bin_hash
            << "\",\"witness_specific_table\":false}";
        return out.str();
    }
}