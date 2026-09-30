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

namespace gatzk::util
{
    std::string canonical_input_preprocess_id(std::string_view dataset)
    {
        if (dataset == "ogbn_arxiv")
            return "ogbn_arxiv_raw_float32";
        if (dataset == "cora" || dataset == "citeseer" || dataset == "pubmed" || dataset == "ppi")
            return "row_normalize";
        throw std::runtime_error("unsupported formal dataset for input preprocessing: " + std::string(dataset));
    }

    namespace
    {

        std::string serialize_scale_vector(std::string_view name, const std::vector<std::uint64_t> &values)
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

        std::string serialize_scale_matrix(std::string_view name, const std::vector<std::vector<std::uint64_t>> &values)
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
                throw std::runtime_error("registered quantization scale must be positive: " + std::string(name));
            }
        }

        void validate_scale_chain_shape(const QuantizationScaleChain &chain, const std::vector<HiddenLayerProfile> &hidden_profile, std::size_t K_out)
        {
            if (hidden_profile.size() != 2 || K_out != 2)
            {
                throw std::runtime_error("registered quantization scale chain only supports canonical L3/K_out=2");
            }
            const auto expected_heads = std::vector<std::size_t>{8, 8};
            if (chain.s_h.size() != hidden_profile.size() + 1 || chain.s_agg_pre.size() != hidden_profile.size() ||
                chain.s_agg.size() != hidden_profile.size() || chain.s_cat.size() != hidden_profile.size() ||
                chain.s_w_hidden.size() != hidden_profile.size() || chain.s_h_prime.size() != hidden_profile.size() ||
                chain.s_a_src_hidden.size() != hidden_profile.size() || chain.s_a_dst_hidden.size() != hidden_profile.size())
            {
                throw std::runtime_error("registered quantization scale chain hidden layer count mismatch");
            }
            if (chain.s_w_output.size() != K_out || chain.s_y_prime.size() != K_out || chain.s_a_src_output.size() != K_out ||
                chain.s_a_dst_output.size() != K_out || chain.s_b_output.size() != K_out)
            {
                throw std::runtime_error("registered quantization scale chain output head count mismatch");
            }
            require_positive_scale(chain.s_input, "S_input");
            require_positive_scale(chain.s_delta, "S_Delta");
            require_positive_scale(chain.s_exp, "S_exp");
            require_positive_scale(chain.s_out_pre, "S_out_pre");
            require_positive_scale(chain.s_out_head, "S_out_head");
            require_positive_scale(chain.s_out, "S_out");
            if (chain.s_h.front() != chain.s_input || chain.s_cat != chain.s_agg || chain.s_out != chain.s_out_head || chain.s_out_pre != chain.s_out_head)
            {
                throw std::runtime_error(
                    "registered quantization scale chain violates H/cat/output equalities; canonical output alignment requires S_out_pre=S_out_head=S_out");
            }
            for (std::size_t layer = 0; layer < hidden_profile.size(); ++layer)
            {
                if (chain.s_h[layer + 1] != chain.s_cat[layer])
                {
                    throw std::runtime_error("registered quantization scale chain violates S_H[l+1]=S_cat[l] at layer " + std::to_string(layer));
                }
            }
            const auto elu_input_scale = chain.s_agg_pre.front();
            const auto elu_output_scale = chain.s_agg.front();
            for (std::size_t layer = 0; layer < hidden_profile.size(); ++layer)
            {
                if (chain.s_agg_pre[layer] != elu_input_scale || chain.s_agg[layer] != elu_output_scale)
                {
                    throw std::runtime_error("registered quantization scale chain violates shared ELU table scale equality");
                }
            }
            for (std::size_t layer = 0; layer < hidden_profile.size(); ++layer)
            {
                if (hidden_profile[layer].head_count != expected_heads[layer])
                {
                    throw std::runtime_error("registered quantization scale chain hidden head count mismatch");
                }
                const auto head_count = hidden_profile[layer].head_count;
                for (const auto *family : {&chain.s_w_hidden[layer], &chain.s_h_prime[layer], &chain.s_a_src_hidden[layer], &chain.s_a_dst_hidden[layer]})
                {
                    if (family->size() != head_count)
                    {
                        throw std::runtime_error("registered quantization scale chain hidden head count mismatch");
                    }
                }
                for (std::size_t head = 0; head < head_count; ++head)
                {
                    require_positive_scale(chain.s_w_hidden[layer][head], "S_W_hidden");
                    require_positive_scale(chain.s_h_prime[layer][head], "S_H_prime_hidden");
                    require_positive_scale(chain.s_a_src_hidden[layer][head], "S_a_src_hidden");
                    require_positive_scale(chain.s_a_dst_hidden[layer][head], "S_a_dst_hidden");
                    if (chain.s_h_prime[layer][head] != chain.s_h[layer] * chain.s_w_hidden[layer][head] ||
                        chain.s_h_prime[layer][head] * chain.s_a_src_hidden[layer][head] != chain.s_delta ||
                        chain.s_h_prime[layer][head] * chain.s_a_dst_hidden[layer][head] != chain.s_delta)
                    {
                        throw std::runtime_error("registered quantization scale chain violates hidden H/W/a equations");
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
                const auto output_input_scale = chain.s_cat.back();
                if (chain.s_h.back() != output_input_scale || chain.s_y_prime[head] != output_input_scale * chain.s_w_output[head] ||
                    chain.s_y_prime[head] * chain.s_a_src_output[head] != chain.s_delta ||
                    chain.s_y_prime[head] * chain.s_a_dst_output[head] != chain.s_delta || chain.s_b_output[head] != chain.s_out_head)
                {
                    throw std::runtime_error("registered quantization scale chain violates output H/W/a/b equations");
                }
            }
        }

        std::string trim(std::string value)
        {
            value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch) { return !std::isspace(ch); }));
            value.erase(std::find_if(value.rbegin(), value.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), value.end());
            return value;
        }

        bool parse_bool(const std::string &value) { return value == "true" || value == "1" || value == "yes"; }

        std::string strip_wrappers(std::string value)
        {
            value.erase(std::remove_if(value.begin(), value.end(),
                                       [](unsigned char ch) { return std::isspace(ch) || ch == '[' || ch == ']' || ch == '(' || ch == ')'; }),
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

        std::unordered_map<std::string, std::string> read_canonical_scale_chain_source(std::string_view source_path)
        {
            std::filesystem::path path;
            if (source_path.empty())
            {
#ifdef GATZK_CANONICAL_SCALE_CHAIN_PATH
                path = GATZK_CANONICAL_SCALE_CHAIN_PATH;
#else
                throw std::runtime_error("canonical quantization scale-chain path is missing from the build");
#endif
            }
            else
            {
                path = std::string(source_path);
            }
            std::ifstream input(path);
            if (!input)
            {
                throw std::runtime_error("canonical quantization scale-chain source not found: " + path.string());
            }
            std::unordered_map<std::string, std::string> entries;
            std::string line;
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
                const auto equal = line.find('=');
                if (equal == std::string::npos)
                {
                    throw std::runtime_error("malformed canonical quantization scale-chain source line");
                }
                const auto key = trim(line.substr(0, equal));
                const auto value = trim(line.substr(equal + 1));
                if (key.empty() || value.empty() || !entries.emplace(key, value).second)
                {
                    throw std::runtime_error("malformed or duplicate canonical quantization scale-chain source key");
                }
            }
            return entries;
        }

        std::uint64_t parse_scale_value(const std::unordered_map<std::string, std::string> &entries, std::string_view key)
        {
            const auto it = entries.find(std::string(key));
            if (it == entries.end())
            {
                throw std::runtime_error("canonical quantization scale-chain source is missing " + std::string(key));
            }
            const auto value = std::stoull(it->second);
            if (value == 0)
            {
                throw std::runtime_error("canonical quantization scale-chain source requires positive " + std::string(key));
            }
            return static_cast<std::uint64_t>(value);
        }

        std::vector<std::uint64_t> parse_scale_vector(const std::unordered_map<std::string, std::string> &entries, std::string_view key)
        {
            const auto it = entries.find(std::string(key));
            if (it == entries.end())
            {
                throw std::runtime_error("canonical quantization scale-chain source is missing " + std::string(key));
            }
            const auto parsed = parse_size_list(it->second);
            if (parsed.empty())
            {
                throw std::runtime_error("canonical quantization scale-chain source has an empty vector " + std::string(key));
            }
            std::vector<std::uint64_t> out;
            out.reserve(parsed.size());
            for (const auto value : parsed)
            {
                if (value == 0)
                {
                    throw std::runtime_error("canonical quantization scale-chain source requires positive vector entries " + std::string(key));
                }
                out.push_back(static_cast<std::uint64_t>(value));
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

        void fail_if_present(const std::unordered_map<std::string, std::string> &entries, const std::set<std::string> &keys)
        {
            for (const auto &key : keys)
            {
                if (entries.contains(key))
                {
                    throw std::runtime_error("unsupported non-canonical config field: " + key);
                }
            }
        }

        void validate_supported_keys(const std::unordered_map<std::string, std::string> &entries)
        {
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
                "hidden_profile",
                "input_preprocess_id",
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
                "verifier_registry_path",
                "verifier_registry_digest",
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

        void require_explicit_family_shape_fields(const std::unordered_map<std::string, std::string> &entries)
        {
            for (const auto *key : {"L", "hidden_profile", "d_in_profile", "K_out"})
            {
                if (!entries.contains(key))
                {
                    throw std::runtime_error("formal config must set family shape field explicitly: " + std::string(key));
                }
            }
        }
        void validate_config(const AppConfig &config)
        {
            if (config.dataset.empty())
            {
                throw std::runtime_error("formal config must set dataset explicitly");
            }

            if (config.srs_id.empty() || config.srs_path.empty())
            {
                throw std::runtime_error("formal config must set srs_id and srs_path explicitly");
            }

            if (config.checkpoint_bundle.empty())
            {
                throw std::runtime_error("formal config must set checkpoint_bundle explicitly");
            }

            const auto expected_input_preprocess_id = canonical_input_preprocess_id(config.dataset);
            if (config.input_preprocess_id.empty() || config.input_preprocess_id != expected_input_preprocess_id)
            {
                throw std::runtime_error("formal config input_preprocess_id does not match dataset; expected " + expected_input_preprocess_id);
            }

            if (!config.verifier_registry_digest.empty() &&
                (config.verifier_registry_digest.size() != 64 ||
                 !std::all_of(config.verifier_registry_digest.begin(), config.verifier_registry_digest.end(),
                              [](unsigned char ch) { return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'); })))
            {
                throw std::runtime_error("verifier_registry_digest must be lowercase SHA-256");
            }

            if (config.subgraph_rule != "whole_graph")
            {
                throw std::runtime_error("formal mainline only supports subgraph_rule=whole_graph");
            }

            if (config.hidden_profile.empty())
            {
                throw std::runtime_error("hidden_profile must be set explicitly");
            }

            if (config.layer_count == 0)
            {
                throw std::runtime_error("L must be set explicitly and be >= 2");
            }

            if (config.layer_count != config.hidden_profile.size() + 1)
            {
                throw std::runtime_error("L=" + std::to_string(config.layer_count) +
                                         " conflicts with hidden_profile size=" + std::to_string(config.hidden_profile.size()));
            }

            if (config.d_in_profile.empty())
            {
                throw std::runtime_error("d_in_profile must be set explicitly");
            }

            if (config.d_in_profile.size() != config.hidden_profile.size())
            {
                throw std::runtime_error("d_in_profile size=" + std::to_string(config.d_in_profile.size()) +
                                         " conflicts with hidden_profile size=" + std::to_string(config.hidden_profile.size()));
            }

            if (std::any_of(config.d_in_profile.begin(), config.d_in_profile.end(), [](std::size_t value) { return value == 0; }))
            {
                throw std::runtime_error("d_in_profile entries must be non-zero");
            }

            if (config.K_out == 0)
            {
                throw std::runtime_error("K_out must be >= 1");
            }

            if (config.num_classes == 0)
            {
                throw std::runtime_error("C/num_classes must be >= 1");
            }

            if (config.range_bits < 1 || config.range_bits > 62)
            {
                throw std::runtime_error("range_bits must be between 1 and 62 for signed certificates");
            }

            if (config.srs_type != "trusted_srs")
            {
                throw std::runtime_error("secure mainline requires srs_type=trusted_srs");
            }

            if (config.privacy_policy != "model_and_trace_hidden")
            {
                throw std::runtime_error("secure mainline requires privacy_policy=model_and_trace_hidden");
            }

            if (config.input_privacy != "public_graph_public_features")
            {
                throw std::runtime_error("secure mainline requires input_privacy=public_graph_public_features");
            }

            if (config.output_mode != "public_logits")
            {
                throw std::runtime_error("secure mainline requires output_mode=public_logits");
            }

            if (config.opening_mode != "point_grouped_batch_opening")
            {
                throw std::runtime_error("secure mainline requires opening_mode=point_grouped_batch_opening");
            }

            if (config.binding_mode != "zk_subproof_for_crpc")
            {
                throw std::runtime_error("secure mainline requires binding_mode=zk_subproof_for_crpc");
            }

            if (config.proof_task != "node_classification_full_gat")
            {
                throw std::runtime_error("secure mainline requires proof_task=node_classification_full_gat");
            }

            for (std::size_t layer_index = 0; layer_index < config.hidden_profile.size(); ++layer_index)
            {
                const auto &layer = config.hidden_profile[layer_index];

                if (layer.head_count == 0 || layer.head_dim == 0)
                {
                    throw std::runtime_error("hidden_profile entries must have non-zero head_count and head_dim");
                }

                if (layer_index + 1 < config.hidden_profile.size())
                {
                    const auto expected = layer.head_count * layer.head_dim;

                    const auto actual = config.d_in_profile[layer_index + 1];

                    if (expected != actual)
                    {
                        throw std::runtime_error("d_in_profile[" + std::to_string(layer_index + 1) + "]=" + std::to_string(actual) +
                                                 " conflicts with previous hidden concat width=" + std::to_string(expected));
                    }
                }
            }

            struct CanonicalShape
            {
                std::vector<std::size_t> d_in;
                std::size_t classes;
                std::size_t range_bits;
                const char *model_arch_id;
            };
            const CanonicalShape *expected = nullptr;
            static const CanonicalShape cora{{1433, 64}, 7, 12, "gat_family_L3_hid8x8-8x8_kout2_c7"};
            static const CanonicalShape citeseer{{3703, 64}, 6, 12, "gat_family_L3_hid8x8-8x8_kout2_c6"};
            static const CanonicalShape pubmed{{500, 64}, 3, 12, "gat_family_L3_hid8x8-8x8_kout2_c3"};
            static const CanonicalShape ppi{{50, 64}, 121, 14, "gat_family_L3_hid8x8-8x8_kout2_c121"};
            static const CanonicalShape ogbn_arxiv{{128, 64}, 40, 21, "gat_family_L3_hid8x8-8x8_kout2_c40"};
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
                const bool profile_matches = config.hidden_profile.size() == 2 &&
                                             std::all_of(config.hidden_profile.begin(), config.hidden_profile.end(),
                                                         [](const HiddenLayerProfile &layer) { return layer.head_count == 8 && layer.head_dim == 8; });
                if (config.layer_count != 3 || !profile_matches || config.d_in_profile != expected->d_in || config.K_out != 2 ||
                    config.num_classes != expected->classes || config.range_bits != expected->range_bits || config.model_arch_id != expected->model_arch_id)
                {
                    throw std::runtime_error("registered dataset requires canonical L=3 hidden_profile=[8x8,8x8], K_out=2 shape and metadata: " +
                                             config.dataset);
                }
            }
        }

    }

    std::string QuantizationScaleChain::canonical_serialization() const
    {
        std::ostringstream out;
        out << "schema=gat_l3_scale_chain" << "|S_input=" << s_input << "|S_Delta=" << s_delta << "|S_exp=" << s_exp << '|'
            << serialize_scale_vector("S_H", s_h) << '|' << serialize_scale_matrix("S_W_hidden", s_w_hidden) << '|'
            << serialize_scale_matrix("S_H_prime_hidden", s_h_prime) << '|' << serialize_scale_matrix("S_a_src_hidden", s_a_src_hidden) << '|'
            << serialize_scale_matrix("S_a_dst_hidden", s_a_dst_hidden) << '|' << serialize_scale_vector("S_agg_pre", s_agg_pre) << '|'
            << serialize_scale_vector("S_agg", s_agg) << '|' << serialize_scale_vector("S_cat", s_cat) << "|S_out_pre=" << s_out_pre
            << "|S_out_head=" << s_out_head << "|S_out=" << s_out << '|' << serialize_scale_vector("S_W_output", s_w_output) << '|'
            << serialize_scale_vector("S_Y_prime_output", s_y_prime) << '|' << serialize_scale_vector("S_a_src_output", s_a_src_output) << '|'
            << serialize_scale_vector("S_a_dst_output", s_a_dst_output) << '|' << serialize_scale_vector("S_b_output", s_b_output)
            << "|S_inv=field_exact|S_alpha=field_exact" << "|round=nearest_ties_away|rescale=none|output_average=signed_round_div";
        return out.str();
    }

    QuantizationScaleChain canonical_quantization_scale_chain(std::string_view quant_cfg_id, const std::vector<HiddenLayerProfile> &hidden_profile,
                                                              std::size_t K_out, std::string_view source_path)
    {
        const auto source = read_canonical_scale_chain_source(source_path);
        const auto source_id_it = source.find("quant_cfg_id");
        const auto source_schema_it = source.find("schema");
        if (source_id_it == source.end() || source_schema_it == source.end() || source_schema_it->second != "gat_l3_scale_chain" ||
            source_id_it->second != quant_cfg_id)
        {
            throw std::runtime_error("unsupported quant_cfg_id; formal runtime requires the registered canonical scale chain: " + std::string(quant_cfg_id));
        }
        if (hidden_profile.size() != 2 || hidden_profile[0].head_count != 8 || hidden_profile[0].head_dim != 8 || hidden_profile[1].head_count != 8 ||
            hidden_profile[1].head_dim != 8 || K_out != 2)
        {
            throw std::runtime_error("canonical quantization scale chain requires hidden_profile=[8x8,8x8], K_out=2");
        }

        QuantizationScaleChain chain;
        chain.s_input = parse_scale_value(source, "s_input");
        chain.s_delta = parse_scale_value(source, "s_delta");
        chain.s_exp = parse_scale_value(source, "s_exp");
        chain.s_h = parse_scale_vector(source, "s_h");
        for (std::size_t layer = 0; layer < hidden_profile.size(); ++layer)
        {
            chain.s_w_hidden.push_back(parse_scale_vector(source, "s_w_hidden_" + std::to_string(layer)));
            chain.s_h_prime.push_back(parse_scale_vector(source, "s_h_prime_hidden_" + std::to_string(layer)));
            chain.s_a_src_hidden.push_back(parse_scale_vector(source, "s_a_src_hidden_" + std::to_string(layer)));
            chain.s_a_dst_hidden.push_back(parse_scale_vector(source, "s_a_dst_hidden_" + std::to_string(layer)));
        }
        chain.s_agg_pre = parse_scale_vector(source, "s_agg_pre");
        chain.s_agg = parse_scale_vector(source, "s_agg");
        chain.s_cat = parse_scale_vector(source, "s_cat");
        chain.s_out_pre = parse_scale_value(source, "s_out_pre");
        chain.s_out_head = parse_scale_value(source, "s_out_head");
        chain.s_out = parse_scale_value(source, "s_out");
        chain.s_w_output = parse_scale_vector(source, "s_w_output");
        chain.s_y_prime = parse_scale_vector(source, "s_y_prime");
        chain.s_a_src_output = parse_scale_vector(source, "s_a_src_output");
        chain.s_a_dst_output = parse_scale_vector(source, "s_a_dst_output");
        chain.s_b_output = parse_scale_vector(source, "s_b_output");
        const auto round_it = source.find("round");
        const auto rescale_it = source.find("rescale");
        const auto average_it = source.find("output_average");
        if (round_it == source.end() || rescale_it == source.end() || average_it == source.end() || round_it->second != "nearest_ties_away" ||
            rescale_it->second != "none" || average_it->second != "signed_round_div")
        {
            throw std::runtime_error("canonical quantization scale-chain source rounding policy mismatch");
        }
        validate_scale_chain_shape(chain, hidden_profile, K_out);
        return chain;
    }

    AppConfig load_config(const std::string &path)
    {
        std::ifstream input(path);
        if (!input)
        {
            throw std::runtime_error("failed to open config: " + path);
        }

        std::unordered_map<std::string, std::string> entries;
        std::string line;
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
        fail_if_present(entries, {"checkpoint_dir"});
        validate_supported_keys(entries);
        require_explicit_family_shape_fields(entries);
        const auto absolute = std::filesystem::absolute(path);
        config.config_dir = absolute.parent_path().string();
        std::filesystem::path canonical_scale_chain_path;
#ifdef GATZK_CANONICAL_SCALE_CHAIN_PATH
        canonical_scale_chain_path = GATZK_CANONICAL_SCALE_CHAIN_PATH;
#else
        throw std::runtime_error("canonical quantization scale-chain path is missing from the build");
#endif
        if (!canonical_scale_chain_path.is_absolute() || canonical_scale_chain_path.parent_path().filename() != "configs")
        {
            throw std::runtime_error("canonical quantization scale-chain path must be an absolute file under configs");
        }
        config.project_root = canonical_scale_chain_path.parent_path().parent_path().string();
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
        if (entries.contains("input_preprocess_id"))
            config.input_preprocess_id = entries["input_preprocess_id"];
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
        if (entries.contains("verifier_registry_path"))
            config.verifier_registry_path = entries["verifier_registry_path"];
        if (entries.contains("verifier_registry_digest"))
            config.verifier_registry_digest = entries["verifier_registry_digest"];
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
            throw std::runtime_error("formal config must set the registered quant_cfg_id explicitly");
        }
        if (config.static_table_id.empty())
        {
            config.static_table_id = "tables:lrelu+elu+exp+range";
        }
        if (config.degree_bound_id.empty())
        {
            config.degree_bound_id = "auto";
        }
        config.quantization = canonical_quantization_scale_chain(config.quant_cfg_id, config.hidden_profile, config.K_out, canonical_scale_chain_path.string());
        validate_config(config);
        return config;
    }

}
