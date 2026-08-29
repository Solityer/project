#include "gat_ezkl/config.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace gat_ezkl 
{
    namespace 
    {
        // 去除字符串首尾空白
        std::string trim(std::string value) 
        {
            value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch) {
                return !std::isspace(ch);
            }));
            value.erase(std::find_if(value.rbegin(), value.rend(), [](unsigned char ch) {
                return !std::isspace(ch);
            }).base(), value.end());
            return value;
        }

        bool parse_bool(const std::string& value) 
        {
            return value == "true" || value == "1" || value == "yes" || value == "on";
        }

        std::string strip_wrappers(std::string value) 
        {
            value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char ch) {
                return std::isspace(ch) || ch == '[' || ch == ']' || ch == '(' || ch == ')';
            }), value.end());
            return value;
        }

        std::vector<std::string> split_tokens(const std::string& value) 
        {
            std::vector<std::string> out;
            std::string current;
            for (const auto ch : value) {
                if (ch == ',' || ch == ';') {
                    if (!current.empty()) {
                        out.push_back(current);
                        current.clear();
                    }
                    continue;
                }
                current.push_back(ch);
            }
            if (!current.empty()) {
                out.push_back(current);
            }
            return out;
        }

        std::vector<std::size_t> parse_size_list(const std::string& value) 
        {
            const auto compact = strip_wrappers(value);
            std::vector<std::size_t> out;
            for (const auto& token : split_tokens(compact)) {
                if (!token.empty()) {
                    out.push_back(static_cast<std::size_t>(std::stoull(token)));
                }
            }
            return out;
        }

        // 解析隐藏层配置列表
        std::vector<HiddenLayerProfile> parse_hidden_profile(const std::string& value) 
        {
            const auto compact = strip_wrappers(value);
            std::vector<HiddenLayerProfile> out;
            for (const auto& token : split_tokens(compact)) {
                if (token.empty()) {
                    continue;
                }
                const auto sep = token.find_first_of("x:X:");
                if (sep == std::string::npos) {
                    throw std::runtime_error("invalid hidden_profile token: " + token);
                }
                out.push_back(HiddenLayerProfile{
                    static_cast<std::size_t>(std::stoull(token.substr(0, sep))),
                    static_cast<std::size_t>(std::stoull(token.substr(sep + 1))),
                });
            }
            return out;
        }

        void validate_supported_keys(const std::unordered_map<std::string, std::string>& entries) 
        {
            static const std::set<std::string> supported_keys = {
                "L",
                "K_out",
                "allow_synthetic_model",
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
                "reference_output_dir",
                "report_unit",
                "seed",
                "self_loop_rule",
                "static_table_id",
                "subgraph_rule",
                "symmetrize_edges",
                "task_type",
            };
            for (const auto& [key, value] : entries) {
                (void)value;
                if (!supported_keys.count(key)) {
                    throw std::runtime_error("unsupported config field: " + key);
                }
            }
        }

        void validate_shape_fields(const AppConfig& config) {
            constexpr std::size_t canonical_layer_count = 3;
            constexpr std::size_t canonical_hidden_layers = 2;
            constexpr std::size_t canonical_hidden_heads = 8;
            constexpr std::size_t canonical_hidden_dim = 8;
            constexpr std::size_t canonical_second_input_dim = 64;
            constexpr std::size_t canonical_output_heads = 2;
            if (config.subgraph_rule != "whole_graph") {
                throw std::runtime_error("only subgraph_rule=whole_graph is allowed");
            }
            if (config.hidden_profile.empty()) {
                throw std::runtime_error("hidden_profile must be explicit");
            }
            if (config.layer_count != canonical_layer_count) {
                throw std::runtime_error("formal ezkl configs require canonical L=3");
            }
            if (config.layer_count != config.hidden_profile.size() + 1) {
                throw std::runtime_error("L conflicts with hidden_profile size");
            }
            if (config.hidden_profile.size() != canonical_hidden_layers) {
                throw std::runtime_error("formal ezkl configs require two hidden layers");
            }
            if (config.d_in_profile.empty()) {
                throw std::runtime_error("d_in_profile must be explicit for every hidden layer");
            }
            if (config.d_in_profile.size() != config.hidden_profile.size()) {
                throw std::runtime_error("d_in_profile size conflicts with hidden_profile");
            }
            if (config.K_out != canonical_output_heads) {
                throw std::runtime_error("formal ezkl configs require canonical K_out=2");
            }
            if (config.num_classes == 0) {
                throw std::runtime_error("num_classes must be >= 1");
            }
            for (const auto& layer : config.hidden_profile) {
                if (layer.head_count != canonical_hidden_heads
                    || layer.head_dim != canonical_hidden_dim) {
                    throw std::runtime_error(
                        "formal ezkl configs require hidden_profile=[8x8,8x8]");
                }
            }
            if (config.d_in_profile[1] != canonical_second_input_dim) {
                throw std::runtime_error(
                    "formal ezkl configs require d_in_profile=[D,64]");
            }
            const auto expected_arch =
                "gat_family_L3_hid8x8-8x8_kout2_c" + std::to_string(config.num_classes);
            if (config.model_arch_id != expected_arch) {
                throw std::runtime_error("model_arch_id is not the canonical L3/K2 architecture id");
            }
            for (std::size_t layer_index = 0; layer_index < config.hidden_profile.size(); ++layer_index) {
                const auto& layer = config.hidden_profile[layer_index];
                if (layer.head_count == 0 || layer.head_dim == 0) {
                    throw std::runtime_error("hidden_profile entries must be non-zero");
                }
                if (layer_index + 1 < config.hidden_profile.size()) {
                    if (layer.head_count > std::numeric_limits<std::size_t>::max() / layer.head_dim) {
                        throw std::runtime_error("hidden head width overflows size_t");
                    }
                    const auto expected = layer.head_count * layer.head_dim;
                    if (config.d_in_profile[layer_index + 1] != expected) {
                        throw std::runtime_error("d_in_profile does not match previous hidden concat width");
                    }
                }
            }
        }

    }

    // 主解析函数：读取配置文件，构建AppConfig对象
    AppConfig load_config_file(const std::filesystem::path& path) 
    {
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error("failed to open config: " + path.string());
        }

        std::unordered_map<std::string, std::string> entries;
        std::string line;
        while (std::getline(input, line)) {
            const auto hash = line.find('#');
            if (hash != std::string::npos) {
                line = line.substr(0, hash);
            }
            line = trim(line);
            if (line.empty()) {
                continue;
            }
            const auto eq = line.find('=');
            if (eq == std::string::npos) {
                throw std::runtime_error("invalid config line: " + line);
            }
            entries[trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
        }
        validate_supported_keys(entries);

        AppConfig config;
        if (entries.count("dataset")) config.dataset = entries.at("dataset");
        if (entries.count("data_root")) config.data_root = entries.at("data_root");
        if (entries.count("cache_root")) config.cache_root = entries.at("cache_root");
        if (entries.count("export_dir")) config.export_dir = entries.at("export_dir");
        if (entries.count("checkpoint_bundle")) config.checkpoint_bundle = entries.at("checkpoint_bundle");
        if (entries.count("reference_output_dir")) config.reference_output_dir = entries.at("reference_output_dir");
        if (entries.count("task_type")) config.task_type = entries.at("task_type");
        if (entries.count("report_unit")) config.report_unit = entries.at("report_unit");
        if (entries.count("batching_rule")) config.batching_rule = entries.at("batching_rule");
        if (entries.count("subgraph_rule")) config.subgraph_rule = entries.at("subgraph_rule");
        if (entries.count("self_loop_rule")) config.self_loop_rule = entries.at("self_loop_rule");
        if (entries.count("edge_sort_rule")) config.edge_sort_rule = entries.at("edge_sort_rule");
        if (entries.count("chunking_rule")) config.chunking_rule = entries.at("chunking_rule");
        if (entries.count("quant_cfg_id")) config.quant_cfg_id = entries.at("quant_cfg_id");
        if (entries.count("model_arch_id")) config.model_arch_id = entries.at("model_arch_id");
        if (entries.count("model_param_id")) config.model_param_id = entries.at("model_param_id");
        if (entries.count("static_table_id")) config.static_table_id = entries.at("static_table_id");
        if (entries.count("degree_bound_id")) config.degree_bound_id = entries.at("degree_bound_id");
        if (entries.count("hidden_dim")) config.hidden_dim = static_cast<std::size_t>(std::stoull(entries.at("hidden_dim")));
        if (entries.count("num_classes")) config.num_classes = static_cast<std::size_t>(std::stoull(entries.at("num_classes")));
        if (entries.count("range_bits")) config.range_bits = static_cast<std::size_t>(std::stoull(entries.at("range_bits")));
        if (entries.count("seed")) config.seed = static_cast<std::uint64_t>(std::stoull(entries.at("seed")));
        if (entries.count("L")) config.layer_count = static_cast<std::size_t>(std::stoull(entries.at("L")));
        if (entries.count("K_out")) config.K_out = static_cast<std::size_t>(std::stoull(entries.at("K_out")));
        if (entries.count("batch_graphs")) config.batch_graphs = static_cast<std::size_t>(std::stoull(entries.at("batch_graphs")));
        if (entries.count("allow_synthetic_model")) config.allow_synthetic_model = parse_bool(entries.at("allow_synthetic_model"));
        if (entries.count("dump_trace")) config.dump_trace = parse_bool(entries.at("dump_trace"));
        if (entries.count("auto_prepare_dataset")) config.auto_prepare_dataset = parse_bool(entries.at("auto_prepare_dataset"));
        if (entries.count("symmetrize_edges")) config.symmetrize_edges = parse_bool(entries.at("symmetrize_edges"));
        if (entries.count("deduplicate_edges")) config.deduplicate_edges = parse_bool(entries.at("deduplicate_edges"));
        if (entries.count("hidden_profile")) config.hidden_profile = parse_hidden_profile(entries.at("hidden_profile"));
        if (entries.count("d_in_profile")) config.d_in_profile = parse_size_list(entries.at("d_in_profile"));

        if (config.model_arch_id.empty() || config.model_param_id.empty() ||
            config.quant_cfg_id.empty() || config.static_table_id.empty() ||
            config.degree_bound_id.empty()) {
            throw std::runtime_error("formal config must explicitly bind all checkpoint identity fields");
        }
        validate_shape_fields(config);
        return config;
    }

    std::vector<DatasetSpec> official_dataset_specs() {
        return {
            {"cora_full", "cora_full.cfg", "cora", "Cora"},
            {"citeseer_full", "citeseer_full.cfg", "citeseer", "Citeseer"},
            {"pubmed_full", "pubmed_full.cfg", "pubmed", "Pubmed"},
            {"ppi_full_formal", "ppi_full.cfg", "ppi", "PPI"},
            {"ogbn_arxiv_full", "ogbn_arxiv_full.cfg", "ogbn_arxiv", "ogbn-arxiv"},
        };
    }

    DatasetSpec official_dataset_spec(const std::string& run_id) {
        for (const auto& spec : official_dataset_specs()) {
            if (spec.run_id == run_id) {
                return spec;
            }
        }
        throw std::runtime_error("unsupported dataset run_id: " + run_id);
    }

    // 获取官方数据集的默认配置文件路径
    std::filesystem::path config_path_for_dataset(
        const std::filesystem::path& project_root,
        const std::string& run_id) {
        return project_root / "configs" / official_dataset_spec(run_id).config_file;
    }

    void validate_full_dataset_contract(const AppConfig& config, const DatasetSpec& spec) {
        if (config.dataset != spec.main_dataset_name) {
            throw std::runtime_error("config dataset does not match official run_id");
        }
        if (config.batching_rule != "whole_graph_single" && config.batching_rule != "multi_graph_batch") {
            throw std::runtime_error("unsupported batching_rule for full-dataset run");
        }
        if (config.self_loop_rule != "per_node") {
            throw std::runtime_error("self_loop_rule must be per_node");
        }
        if (config.edge_sort_rule != "edge_gid_then_dst_stable") {
            throw std::runtime_error("edge_sort_rule must be edge_gid_then_dst_stable");
        }
        if (config.chunking_rule != "none") {
            throw std::runtime_error("chunking_rule must be none");
        }
        if (config.allow_synthetic_model) {
            throw std::runtime_error("formal ezkl baseline cannot use synthetic model parameters");
        }
        if (config.quant_cfg_id != "fp32_torch_checkpoint"
            || config.static_table_id != "tables:lrelu+elu+exp+range") {
            throw std::runtime_error("config quantization/table profile is not canonical");
        }
        static const std::unordered_map<std::string, std::string> canonical_model_params = {
            {"cora", "cora_planetoid_seed11"},
            {"citeseer", "citeseer_planetoid_seed11"},
            {"pubmed", "pubmed_planetoid_seed11"},
            {"ppi", "ppi_full_seed11"},
            {"ogbn_arxiv", "ogbn_arxiv_seed11"},
        };
        const auto model_param = canonical_model_params.find(config.dataset);
        if (model_param == canonical_model_params.end()
            || config.model_param_id != model_param->second) {
            throw std::runtime_error("model_param_id is not the canonical dataset checkpoint id");
        }
    }
}
