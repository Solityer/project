#include "gatzk/util/config.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <limits>

// 实现正式配置解析与约束检查
// 负责拦截过时字段和形状冲突，避免错误配置进入 proving / verifying 主链

namespace gatzk::util
{
    namespace
    {

        constexpr std::string_view kCanonicalQuantCfgId =
            "gat_l3_scale_chain_v1";

        std::string serialize_scale_vector(
            std::string_view name,
            const std::vector<std::uint64_t> &values)
        {
            std::ostringstream out;
            out << name << "=[";
            for (std::size_t i = 0; i < values.size(); ++i)
            {
                if (i != 0)
                {
                    out << ',';
                }
                out << values[i];
            }
            out << ']';
            return out.str();
        }

        std::string serialize_scale_matrix(
            std::string_view name,
            const std::vector<std::vector<std::uint64_t>> &values)
        {
            std::ostringstream out;
            out << name << "=[";
            for (std::size_t row = 0; row < values.size(); ++row)
            {
                if (row != 0)
                {
                    out << ';';
                }
                out << '[';
                for (std::size_t column = 0; column < values[row].size(); ++column)
                {
                    if (column != 0)
                    {
                        out << ',';
                    }
                    out << values[row][column];
                }
                out << ']';
            }
            out << ']';
            return out.str();
        }

        void require_positive_scale(std::uint64_t value, std::string_view name)
        {
            if (value == 0)
            {
                throw std::runtime_error(
                    "registered quantization scale must be positive: " +
                    std::string(name));
            }
        }

        void validate_scale_chain_shape(
            const QuantizationScaleChain &chain,
            const std::vector<HiddenLayerProfile> &hidden_profile,
            std::size_t K_out)
        {
            if (hidden_profile.size() != 2 || K_out != 2)
            {
                throw std::runtime_error(
                    "registered quantization scale chain only supports canonical L3/K_out=2");
            }
            const auto expected_heads = std::vector<std::size_t>{8, 8};
            // S_H carries the input scale of every hidden layer plus the
            // final hidden-concat/output input scale.  Keeping the terminal
            // entry explicit lets validation enforce every protocol
            // transition S_H^(l+1) = S_cat^(l), including the output input.
            if (chain.s_h.size() != hidden_profile.size() + 1 ||
                chain.s_agg_pre.size() != hidden_profile.size() ||
                chain.s_agg.size() != hidden_profile.size() ||
                chain.s_cat.size() != hidden_profile.size() ||
                chain.s_w_hidden.size() != hidden_profile.size() ||
                chain.s_h_prime.size() != hidden_profile.size() ||
                chain.s_a_src_hidden.size() != hidden_profile.size() ||
                chain.s_a_dst_hidden.size() != hidden_profile.size())
            {
                throw std::runtime_error(
                    "registered quantization scale chain hidden layer count mismatch");
            }
            if (chain.s_w_output.size() != K_out ||
                chain.s_y_prime.size() != K_out ||
                chain.s_a_src_output.size() != K_out ||
                chain.s_a_dst_output.size() != K_out ||
                chain.s_b_output.size() != K_out)
            {
                throw std::runtime_error(
                    "registered quantization scale chain output head count mismatch");
            }
            require_positive_scale(chain.s_input, "S_input");
            require_positive_scale(chain.s_delta, "S_Delta");
            require_positive_scale(chain.s_exp, "S_exp");
            require_positive_scale(chain.s_out_pre, "S_out_pre");
            require_positive_scale(chain.s_out_head, "S_out_head");
            require_positive_scale(chain.s_out, "S_out");
            // The canonical output protocol fixes S_out_pre=S_out_head=S_out;
            // its pre-bias alignment SignedRoundDiv is therefore an identity
            // and cannot hide an implicit rescale in the production trace.
            if (chain.s_h.front() != chain.s_input ||
                chain.s_cat != chain.s_agg ||
                chain.s_out != chain.s_out_head ||
                chain.s_out_pre != chain.s_out_head)
            {
                throw std::runtime_error(
                    "registered quantization scale chain violates H/cat/output equalities; canonical output alignment requires S_out_pre=S_out_head=S_out");
            }
            for (std::size_t layer = 0;
                 layer < hidden_profile.size();
                 ++layer)
            {
                if (chain.s_h[layer + 1] != chain.s_cat[layer])
                {
                    throw std::runtime_error(
                        "registered quantization scale chain violates S_H[l+1]=S_cat[l] at layer " +
                        std::to_string(layer));
                }
            }
            // StaticTables owns one shared authenticated ELU table.  The
            // canonical runtime therefore requires every hidden layer to use
            // the same ELU input/output scales before the table is built from
            // the first entry; a layer-specific table would need a distinct
            // formal table commitment and is intentionally not accepted.
            const auto elu_input_scale = chain.s_agg_pre.front();
            const auto elu_output_scale = chain.s_agg.front();
            for (std::size_t layer = 0;
                 layer < hidden_profile.size();
                 ++layer)
            {
                if (chain.s_agg_pre[layer] != elu_input_scale ||
                    chain.s_agg[layer] != elu_output_scale)
                {
                    throw std::runtime_error(
                        "registered quantization scale chain violates shared ELU table scale equality");
                }
            }
            for (std::size_t layer = 0; layer < hidden_profile.size(); ++layer)
            {
                if (hidden_profile[layer].head_count != expected_heads[layer])
                {
                    throw std::runtime_error(
                        "registered quantization scale chain hidden head count mismatch");
                }
                const auto head_count = hidden_profile[layer].head_count;
                for (const auto *family : {
                         &chain.s_w_hidden[layer],
                         &chain.s_h_prime[layer],
                         &chain.s_a_src_hidden[layer],
                         &chain.s_a_dst_hidden[layer]})
                {
                    if (family->size() != head_count)
                    {
                        throw std::runtime_error(
                            "registered quantization scale chain hidden head count mismatch");
                    }
                }
                for (std::size_t head = 0; head < head_count; ++head)
                {
                    require_positive_scale(chain.s_w_hidden[layer][head], "S_W_hidden");
                    require_positive_scale(chain.s_h_prime[layer][head], "S_H_prime_hidden");
                    require_positive_scale(chain.s_a_src_hidden[layer][head], "S_a_src_hidden");
                    require_positive_scale(chain.s_a_dst_hidden[layer][head], "S_a_dst_hidden");
                    if (chain.s_h_prime[layer][head] !=
                            chain.s_h[layer] * chain.s_w_hidden[layer][head] ||
                        chain.s_h_prime[layer][head] * chain.s_a_src_hidden[layer][head] != chain.s_delta ||
                        chain.s_h_prime[layer][head] * chain.s_a_dst_hidden[layer][head] != chain.s_delta)
                    {
                        throw std::runtime_error(
                            "registered quantization scale chain violates hidden H/W/a equations");
                    }
                }
                require_positive_scale(chain.s_agg_pre[layer], "S_agg_pre");
                require_positive_scale(chain.s_agg[layer], "S_agg");
                require_positive_scale(chain.s_cat[layer], "S_cat");
            }
            for (std::size_t head = 0; head < K_out; ++head)
            {
                require_positive_scale(chain.s_w_output[head], "S_W_output");
                require_positive_scale(chain.s_y_prime[head], "S_Y_prime_output");
                require_positive_scale(chain.s_a_src_output[head], "S_a_src_output");
                require_positive_scale(chain.s_a_dst_output[head], "S_a_dst_output");
                require_positive_scale(chain.s_b_output[head], "S_b_output");
                // Output projection consumes the final hidden concat scale;
                // require its explicit S_H terminal entry to agree with the
                // final S_cat transition rather than silently selecting a
                // different input scale.
                const auto output_input_scale = chain.s_cat.back();
                if (chain.s_h.back() != output_input_scale ||
                    chain.s_y_prime[head] != output_input_scale * chain.s_w_output[head] ||
                    chain.s_y_prime[head] * chain.s_a_src_output[head] != chain.s_delta ||
                    chain.s_y_prime[head] * chain.s_a_dst_output[head] != chain.s_delta ||
                    chain.s_b_output[head] != chain.s_out_head)
                {
                    throw std::runtime_error(
                        "registered quantization scale chain violates output H/W/a/b equations");
                }
            }
        }

        std::string trim(std::string value)
        {
            value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch)
                                                    { return !std::isspace(ch); }));
            value.erase(std::find_if(value.rbegin(), value.rend(), [](unsigned char ch)
                                     { return !std::isspace(ch); })
                            .base(),
                        value.end());
            return value;
        }

        bool parse_bool(const std::string &value)
        {
            return value == "true" || value == "1" || value == "yes";
        }

        std::string strip_wrappers(std::string value)
        {
            value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char ch)
                                       { return std::isspace(ch) || ch == '[' || ch == ']' || ch == '(' || ch == ')'; }),
                        value.end());
            return value;
        }

        std::vector<std::string> split_tokens(const std::string &value)
        {
            std::vector<std::string> out;
            std::string current;
            for (const auto ch : value)
            {
                if (ch == ',' || ch == ';')
                {
                    if (!current.empty())
                    {
                        out.push_back(current);
                        current.clear();
                    }
                    continue;
                }
                current.push_back(ch);
            }
            if (!current.empty())
            {
                out.push_back(current);
            }
            return out;
        }

        std::vector<std::size_t> parse_size_list(const std::string &value)
        {
            const auto compact = strip_wrappers(value);
            if (compact.empty())
            {
                return {};
            }
            std::vector<std::size_t> out;
            for (const auto &token : split_tokens(compact))
            {
                out.push_back(std::stoull(token));
            }
            return out;
        }

        std::vector<HiddenLayerProfile> parse_hidden_profile(const std::string &value)
        {
            const auto compact = strip_wrappers(value);
            if (compact.empty())
            {
                return {};
            }
            std::vector<HiddenLayerProfile> out;
            for (const auto &token : split_tokens(compact))
            {
                const auto sep = token.find_first_of("x:X:");
                if (sep == std::string::npos)
                {
                    throw std::runtime_error("invalid hidden_profile token: " + token);
                }
                out.push_back(HiddenLayerProfile{
                    .head_count = std::stoull(token.substr(0, sep)),
                    .head_dim = std::stoull(token.substr(sep + 1)),
                });
            }
            return out;
        }

        void fail_if_present(
            const std::unordered_map<std::string, std::string> &entries,
            const std::set<std::string> &keys)
        {
            for (const auto &key : keys)
            {
                if (entries.contains(key))
                {
                    throw std::runtime_error("unsupported legacy config field: " + key);
                }
            }
        }

        void validate_supported_keys(const std::unordered_map<std::string, std::string> &entries)
        {
            // formal 主线显式列出允许字段，未收敛的历史键直接拒绝进入运行时
            static const std::set<std::string> supported_keys = {
                "L",
                "K_out",
                "auto_prepare_dataset",
                "batch_graphs",
                "batching_rule",
                "cache_root",
                "checkpoint_bundle",
                "chunking_rule",
                "d_in_profile",
                "data_root",
                "dataset",
                "deduplicate_edges",
                "degree_bound_id",
                "dump_trace",
                "edge_sort_rule",
                "export_dir",
                "hidden_dim",
                "hidden_profile",
                "model_arch_id",
                "model_param_id",
                "num_classes",
                "quant_cfg_id",
                "range_bits",
                "report_unit",
                "binding_mode",
                "seed",
                "self_loop_rule",
                "security_profile_id",
                "srs_id",
                "srs_path",
                "srs_type",
                "static_table_id",
                "symmetrize_edges",
                "task_type",
                "input_privacy",
                "opening_mode",
                "output_mode",
                "privacy_policy",
                "proof_task",
            };
            for (const auto &[key, value] : entries)
            {
                (void)value;
                if (!supported_keys.contains(key))
                {
                    throw std::runtime_error("unsupported config field: " + key);
                }
            }
        }

        void require_explicit_family_shape_fields(
            const std::unordered_map<std::string, std::string> &entries)
        {
            for (const auto *key : {"L", "hidden_profile", "d_in_profile", "K_out"})
            {
                if (!entries.contains(key))
                {
                    throw std::runtime_error(
                        "formal config must set family shape field explicitly: " + std::string(key));
                }
            }
        }
        void validate_config(const AppConfig &config)
        {
            if (config.dataset.empty())
            {
                throw std::runtime_error(
                    "formal config must set dataset explicitly");
            }

            if (config.srs_id.empty() ||
                config.srs_path.empty())
            {
                throw std::runtime_error(
                    "formal config must set srs_id and srs_path explicitly");
            }

            if (config.checkpoint_bundle.empty())
            {
                throw std::runtime_error(
                    "formal config must set checkpoint_bundle explicitly");
            }

            if (config.subgraph_rule != "whole_graph")
            {
                throw std::runtime_error(
                    "formal mainline only supports subgraph_rule=whole_graph");
            }

            if (config.hidden_profile.empty())
            {
                throw std::runtime_error(
                    "hidden_profile must be set explicitly");
            }

            if (config.layer_count == 0)
            {
                throw std::runtime_error(
                    "L must be set explicitly and be >= 2");
            }

            if (config.layer_count !=
                config.hidden_profile.size() + 1)
            {
                throw std::runtime_error(
                    "L=" +
                    std::to_string(config.layer_count) +
                    " conflicts with hidden_profile size=" +
                    std::to_string(config.hidden_profile.size()));
            }

            if (config.d_in_profile.empty())
            {
                throw std::runtime_error(
                    "d_in_profile must be set explicitly");
            }

            if (config.d_in_profile.size() !=
                config.hidden_profile.size())
            {
                throw std::runtime_error(
                    "d_in_profile size=" +
                    std::to_string(config.d_in_profile.size()) +
                    " conflicts with hidden_profile size=" +
                    std::to_string(config.hidden_profile.size()));
            }

            if (std::any_of(
                    config.d_in_profile.begin(),
                    config.d_in_profile.end(),
                    [](std::size_t value)
                    {
                        return value == 0;
                    }))
            {
                throw std::runtime_error(
                    "d_in_profile entries must be non-zero");
            }

            if (config.K_out == 0)
            {
                throw std::runtime_error(
                    "K_out must be >= 1");
            }

            if (config.num_classes == 0)
            {
                throw std::runtime_error(
                    "C/num_classes must be >= 1");
            }

            if (config.srs_type != "trusted_srs")
            {
                throw std::runtime_error(
                    "secure mainline requires srs_type=trusted_srs");
            }

            if (config.privacy_policy !=
                "model_and_trace_hidden")
            {
                throw std::runtime_error(
                    "secure mainline requires privacy_policy=model_and_trace_hidden");
            }

            if (config.input_privacy !=
                "public_graph_public_features")
            {
                throw std::runtime_error(
                    "secure mainline requires input_privacy=public_graph_public_features");
            }

            if (config.output_mode != "public_logits")
            {
                throw std::runtime_error(
                    "secure mainline requires output_mode=public_logits");
            }

            if (config.opening_mode !=
                "point_grouped_batch_opening")
            {
                throw std::runtime_error(
                    "secure mainline requires opening_mode=point_grouped_batch_opening");
            }

            if (config.binding_mode !=
                "zk_subproof_for_crpc")
            {
                throw std::runtime_error(
                    "secure mainline requires binding_mode=zk_subproof_for_crpc");
            }

            if (config.proof_task !=
                "node_classification_full_gat")
            {
                throw std::runtime_error(
                    "secure mainline requires proof_task=node_classification_full_gat");
            }

            for (std::size_t layer_index = 0;
                 layer_index < config.hidden_profile.size();
                 ++layer_index)
            {
                const auto &layer =
                    config.hidden_profile[layer_index];

                if (layer.head_count == 0 ||
                    layer.head_dim == 0)
                {
                    throw std::runtime_error(
                        "hidden_profile entries must have non-zero head_count and head_dim");
                }

                if (layer_index + 1 <
                    config.hidden_profile.size())
                {
                    const auto expected =
                        layer.head_count * layer.head_dim;

                    const auto actual =
                        config.d_in_profile[layer_index + 1];

                    if (expected != actual)
                    {
                        throw std::runtime_error(
                            "d_in_profile[" +
                            std::to_string(layer_index + 1) +
                            "]=" +
                            std::to_string(actual) +
                            " conflicts with previous hidden concat width=" +
                            std::to_string(expected));
                    }
                }
            }

            // The five registered experiments intentionally share one formal
            // model family.  Rejecting a legacy shape here keeps an old
            // checkpoint/config pair out of the proving path even when a
            // caller supplies a non-canonical config file.
            struct CanonicalShape
            {
                std::vector<std::size_t> d_in;
                std::size_t classes;
                std::size_t range_bits;
                const char *model_arch_id;
            };
            const CanonicalShape *expected = nullptr;
            static const CanonicalShape cora{
                {1433, 64}, 7, 12, "gat_family_L3_hid8x8-8x8_kout2_c7"};
            static const CanonicalShape citeseer{
                {3703, 64}, 6, 12, "gat_family_L3_hid8x8-8x8_kout2_c6"};
            static const CanonicalShape pubmed{
                {500, 64}, 3, 12, "gat_family_L3_hid8x8-8x8_kout2_c3"};
            static const CanonicalShape ppi{
                {50, 64}, 121, 14, "gat_family_L3_hid8x8-8x8_kout2_c121"};
            static const CanonicalShape ogbn_arxiv{
                {128, 64}, 40, 21, "gat_family_L3_hid8x8-8x8_kout2_c40"};
            if (config.dataset == "cora")
                expected = &cora;
            else if (config.dataset == "citeseer")
                expected = &citeseer;
            else if (config.dataset == "pubmed")
                expected = &pubmed;
            else if (config.dataset == "ppi")
                expected = &ppi;
            else if (config.dataset == "ogbn_arxiv")
                expected = &ogbn_arxiv;
            if (expected != nullptr)
            {
                const bool profile_matches =
                    config.hidden_profile.size() == 2 &&
                    std::all_of(
                        config.hidden_profile.begin(),
                        config.hidden_profile.end(),
                        [](const HiddenLayerProfile &layer)
                        {
                            return layer.head_count == 8 && layer.head_dim == 8;
                        });
                if (config.layer_count != 3 || !profile_matches ||
                    config.d_in_profile != expected->d_in || config.K_out != 2 ||
                    config.num_classes != expected->classes ||
                    config.range_bits != expected->range_bits ||
                    config.model_arch_id != expected->model_arch_id)
                {
                    throw std::runtime_error(
                        "registered dataset requires canonical L=3 hidden_profile=[8x8,8x8], K_out=2 shape and metadata: " + config.dataset);
                }
            }
        }

    }

    std::string QuantizationScaleChain::canonical_serialization() const
    {
        std::ostringstream out;
        out << "schema=gat_l3_scale_chain_v1"
            << "|S_input=" << s_input
            << "|S_Delta=" << s_delta
            << "|S_exp=" << s_exp
            << '|' << serialize_scale_vector("S_H", s_h)
            << '|' << serialize_scale_matrix("S_W_hidden", s_w_hidden)
            << '|' << serialize_scale_matrix("S_H_prime_hidden", s_h_prime)
            << '|' << serialize_scale_matrix("S_a_src_hidden", s_a_src_hidden)
            << '|' << serialize_scale_matrix("S_a_dst_hidden", s_a_dst_hidden)
            << '|' << serialize_scale_vector("S_agg_pre", s_agg_pre)
            << '|' << serialize_scale_vector("S_agg", s_agg)
            << '|' << serialize_scale_vector("S_cat", s_cat)
            << "|S_out_pre=" << s_out_pre
            << "|S_out_head=" << s_out_head
            << "|S_out=" << s_out
            << '|' << serialize_scale_vector("S_W_output", s_w_output)
            << '|' << serialize_scale_vector("S_Y_prime_output", s_y_prime)
            << '|' << serialize_scale_vector("S_a_src_output", s_a_src_output)
            << '|' << serialize_scale_vector("S_a_dst_output", s_a_dst_output)
            << '|' << serialize_scale_vector("S_b_output", s_b_output)
            << "|S_inv=field_exact|S_alpha=field_exact"
            << "|round=nearest_ties_away|rescale=none|output_average=signed_round_div";
        return out.str();
    }

    QuantizationScaleChain canonical_quantization_scale_chain(
        std::string_view quant_cfg_id,
        const std::vector<HiddenLayerProfile> &hidden_profile,
        std::size_t K_out)
    {
        if (quant_cfg_id != kCanonicalQuantCfgId)
        {
            throw std::runtime_error(
                "unsupported quant_cfg_id; formal runtime requires the registered canonical scale chain: " +
                std::string(quant_cfg_id));
        }
        if (hidden_profile.size() != 2 ||
            hidden_profile[0].head_count != 8 || hidden_profile[0].head_dim != 8 ||
            hidden_profile[1].head_count != 8 || hidden_profile[1].head_dim != 8 ||
            K_out != 2)
        {
            throw std::runtime_error(
                "canonical quantization scale chain requires hidden_profile=[8x8,8x8], K_out=2");
        }

        // This is the sole declaration of the formal L3 fixed-point scales.
        QuantizationScaleChain chain;
        chain.s_input = 1;
        chain.s_delta = 256;
        chain.s_exp = 16;
        chain.s_h = {1, 16, 16};
        chain.s_w_hidden = {{16, 16, 16, 16, 16, 16, 16, 16},
                            {16, 16, 16, 16, 16, 16, 16, 16}};
        chain.s_h_prime = {{16, 16, 16, 16, 16, 16, 16, 16},
                           {256, 256, 256, 256, 256, 256, 256, 256}};
        chain.s_a_src_hidden = {{16, 16, 16, 16, 16, 16, 16, 16},
                                {1, 1, 1, 1, 1, 1, 1, 1}};
        chain.s_a_dst_hidden = {{16, 16, 16, 16, 16, 16, 16, 16},
                                {1, 1, 1, 1, 1, 1, 1, 1}};
        chain.s_agg_pre = {16, 16};
        chain.s_agg = {16, 16};
        chain.s_cat = {16, 16};
        chain.s_out_pre = 16;
        chain.s_out_head = 16;
        chain.s_out = 16;
        chain.s_w_output = {16, 16};
        chain.s_y_prime = {256, 256};
        chain.s_a_src_output = {1, 1};
        chain.s_a_dst_output = {1, 1};
        chain.s_b_output = {16, 16};
        validate_scale_chain_shape(chain, hidden_profile, K_out);
        return chain;
    }

    // 解析 `.cfg` 文件并补齐工程根路径等派生字段
    AppConfig load_config(const std::string &path)
    {
        std::ifstream input(path);
        if (!input)
        {
            throw std::runtime_error("failed to open config: " + path);
        }

        std::unordered_map<std::string, std::string> entries;
        std::string line;
        // 配置文件按 `key = value` 读取，注释与空行都在这一层先剔掉。
        while (std::getline(input, line))
        {
            const auto hash = line.find('#');
            if (hash != std::string::npos)
            {
                line = line.substr(0, hash);
            }
            line = trim(line);
            if (line.empty())
            {
                continue;
            }
            const auto eq = line.find('=');
            if (eq == std::string::npos)
            {
                throw std::runtime_error("invalid config line: " + line);
            }
            const auto key = trim(line.substr(0, eq));
            const auto value = trim(line.substr(eq + 1));
            if (!entries.emplace(key, value).second)
            {
                throw std::runtime_error("duplicate config field: " + key);
            }
        }

        AppConfig config;
        fail_if_present(entries, {"hidden_heads", "checkpoint_dir"});
        validate_supported_keys(entries);
        require_explicit_family_shape_fields(entries);
        const auto absolute = std::filesystem::absolute(path);
        config.config_dir = absolute.parent_path().string();
        config.project_root = absolute.parent_path().parent_path().string();
        if (entries.contains("dataset"))
            config.dataset = entries["dataset"];
        if (entries.contains("data_root"))
            config.data_root = entries["data_root"];
        if (entries.contains("cache_root"))
            config.cache_root = entries["cache_root"];
        if (entries.contains("export_dir"))
            config.export_dir = entries["export_dir"];
        if (entries.contains("checkpoint_bundle"))
            config.checkpoint_bundle = entries["checkpoint_bundle"];
        if (entries.contains("task_type"))
            config.task_type = entries["task_type"];
        if (entries.contains("report_unit"))
            config.report_unit = entries["report_unit"];
        if (entries.contains("batching_rule"))
            config.batching_rule = entries["batching_rule"];
        if (entries.contains("self_loop_rule"))
            config.self_loop_rule = entries["self_loop_rule"];
        if (entries.contains("edge_sort_rule"))
            config.edge_sort_rule = entries["edge_sort_rule"];
        if (entries.contains("chunking_rule"))
            config.chunking_rule = entries["chunking_rule"];
        if (entries.contains("quant_cfg_id"))
            config.quant_cfg_id = entries["quant_cfg_id"];
        if (entries.contains("model_arch_id"))
            config.model_arch_id = entries["model_arch_id"];
        if (entries.contains("model_param_id"))
            config.model_param_id = entries["model_param_id"];
        if (entries.contains("static_table_id"))
            config.static_table_id = entries["static_table_id"];
        if (entries.contains("degree_bound_id"))
            config.degree_bound_id = entries["degree_bound_id"];
        if (entries.contains("srs_type"))
            config.srs_type = entries["srs_type"];
        if (entries.contains("srs_id"))
            config.srs_id = entries["srs_id"];
        if (entries.contains("srs_path"))
            config.srs_path = entries["srs_path"];
        if (entries.contains("privacy_policy"))
            config.privacy_policy = entries["privacy_policy"];
        if (entries.contains("input_privacy"))
            config.input_privacy = entries["input_privacy"];
        if (entries.contains("output_mode"))
            config.output_mode = entries["output_mode"];
        if (entries.contains("opening_mode"))
            config.opening_mode = entries["opening_mode"];
        if (entries.contains("binding_mode"))
            config.binding_mode = entries["binding_mode"];
        if (entries.contains("proof_task"))
            config.proof_task = entries["proof_task"];
        if (entries.contains("security_profile_id"))
            config.security_profile_id = entries["security_profile_id"];
        if (entries.contains("hidden_dim"))
            config.hidden_dim = std::stoull(entries["hidden_dim"]);
        if (entries.contains("num_classes"))
            config.num_classes = std::stoull(entries["num_classes"]);
        if (entries.contains("range_bits"))
            config.range_bits = std::stoull(entries["range_bits"]);
        if (entries.contains("seed"))
            config.seed = std::stoull(entries["seed"]);
        if (entries.contains("L"))
            config.layer_count = std::stoull(entries["L"]);
        if (entries.contains("K_out"))
            config.K_out = std::stoull(entries["K_out"]);
        if (entries.contains("batch_graphs"))
            config.batch_graphs = std::stoull(entries["batch_graphs"]);
        if (entries.contains("dump_trace"))
            config.dump_trace = parse_bool(entries["dump_trace"]);
        if (entries.contains("auto_prepare_dataset"))
            config.auto_prepare_dataset = parse_bool(entries["auto_prepare_dataset"]);
        if (entries.contains("symmetrize_edges"))
            config.symmetrize_edges = parse_bool(entries["symmetrize_edges"]);
        if (entries.contains("deduplicate_edges"))
            config.deduplicate_edges = parse_bool(entries["deduplicate_edges"]);
        if (entries.contains("hidden_profile"))
            config.hidden_profile = parse_hidden_profile(entries["hidden_profile"]);
        if (entries.contains("d_in_profile"))
            config.d_in_profile = parse_size_list(entries["d_in_profile"]);
        if (config.quant_cfg_id.empty())
        {
            throw std::runtime_error(
                "formal config must set the registered quant_cfg_id explicitly");
        }
        if (config.static_table_id.empty())
        {
            config.static_table_id = "tables:lrelu+elu+exp+range";
        }
        if (config.degree_bound_id.empty())
        {
            config.degree_bound_id = "auto";
        }
        config.quantization = canonical_quantization_scale_chain(
            config.quant_cfg_id,
            config.hidden_profile,
            config.K_out);
        // 这里最后做一次 shape 和 formal 约束校验，保证 prover / verifier 看到的是同一套配置含义。
        validate_config(config);
        return config;
    }

}
