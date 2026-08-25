#include "gat_ezkl/checkpoint.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <cmath>

namespace gat_ezkl 
{
    // 文件 I/O 和 JSON 解析的工具函数
    namespace 
    {
        std::string read_file(const std::filesystem::path& path) {
            std::ifstream input(path);
            if (!input) {
                throw std::runtime_error("failed to open file: " + path.string());
            }
            std::ostringstream stream;
            stream << input.rdbuf();
            return stream.str();
        }

        std::size_t parse_size_t(const std::string& text, const std::string& key) {
            const std::regex pattern("\"" + key + "\"\\s*:\\s*([0-9]+)");
            std::smatch match;
            if (!std::regex_search(text, match, pattern)) {
                throw std::runtime_error("missing numeric checkpoint key: " + key);
            }
            return static_cast<std::size_t>(std::stoull(match[1].str()));
        }

        std::string parse_string(const std::string& text, const std::string& key) {
            const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"");
            std::smatch match;
            if (!std::regex_search(text, match, pattern)) {
                throw std::runtime_error("missing string checkpoint key: " + key);
            }
            return match[1].str();
        }

        std::vector<std::size_t> parse_size_array(const std::string& text, const std::string& key) {
            const std::regex pattern("\"" + key + "\"\\s*:\\s*\\[([^\\]]*)\\]");
            std::smatch match;
            if (!std::regex_search(text, match, pattern)) {
                return {};
            }
            std::string body = match[1].str();
            std::replace(body.begin(), body.end(), ',', ' ');
            std::vector<std::size_t> out;
            std::istringstream input(body);
            std::size_t value = 0;
            while (input >> value) {
                out.push_back(value);
            }
            return out;
        }

        std::string array_body(const std::string& text, const std::string& key) {
            const auto key_pos = text.find("\"" + key + "\"");
            if (key_pos == std::string::npos) {
                throw std::runtime_error("missing checkpoint array: " + key);
            }
            const auto begin = text.find('[', key_pos);
            if (begin == std::string::npos) {
                throw std::runtime_error("malformed checkpoint array: " + key);
            }
            std::size_t depth = 0;
            for (std::size_t i = begin; i < text.size(); ++i) {
                if (text[i] == '[') ++depth;
                if (text[i] == ']') {
                    --depth;
                    if (depth == 0) {
                        return text.substr(begin + 1, i - begin - 1);
                    }
                }
            }
            throw std::runtime_error("unterminated checkpoint array: " + key);
        }

        std::vector<std::string> object_entries(const std::string& text, const std::string& key) {
            const auto body = array_body(text, key);
            std::vector<std::string> entries;
            std::size_t depth = 0;
            std::size_t object_begin = std::string::npos;
            for (std::size_t i = 0; i < body.size(); ++i) {
                if (body[i] == '{') {
                    if (depth == 0) object_begin = i;
                    ++depth;
                } else if (body[i] == '}') {
                    if (depth == 0) {
                        throw std::runtime_error("malformed checkpoint object array: " + key);
                    }
                    --depth;
                    if (depth == 0) {
                        entries.push_back(body.substr(object_begin, i - object_begin + 1));
                    }
                }
            }
            if (depth != 0) {
                throw std::runtime_error("unterminated checkpoint object array: " + key);
            }
            return entries;
        }

        void load_tensor_shapes(const std::filesystem::path& path, CheckpointBundle* bundle) {
            std::ifstream input(path);
            if (!input) {
                throw std::runtime_error("failed to open checkpoint tensor dump: " + path.string());
            }
            std::string marker;
            while (input >> marker) {
                if (marker != "TENSOR") {
                    throw std::runtime_error("malformed tensor dump marker");
                }
                std::string name;
                std::size_t rank = 0;
                input >> name >> rank;
                TensorInfo info;
                info.shape.resize(rank);
                for (std::size_t i = 0; i < rank; ++i) {
                    input >> info.shape[i];
                }
                input >> info.value_count;
                info.values.resize(info.value_count);
                for (std::size_t i = 0; i < info.value_count; ++i) {
                    input >> info.values[i];
                }
                bundle->tensors.emplace(name, std::move(info));
            }
        }

        void require_tensor(const CheckpointBundle& bundle, const std::string& name) {
            if (!bundle.tensors.count(name)) {
                throw std::runtime_error("checkpoint tensor missing from tensors.txt: " + name);
            }
        }

        void require_zero_tensor(const CheckpointBundle& bundle, const std::string& name,
                                 const char* context) {
            const auto found = bundle.tensors.find(name);
            if (found == bundle.tensors.end()) {
                throw std::runtime_error("checkpoint tensor missing from tensors.txt: " + name);
            }
            for (const double value : found->second.values) {
                if (!std::isfinite(value) || value != 0.0) {
                    throw std::runtime_error(std::string(context)
                                             + " contains a non-zero unused private bias: " + name);
                }
            }
        }

    }  // namespace

    // 从 bundle_root 目录加载完整 checkpoint（manifest.json + tensors.txt）
    CheckpointBundle load_checkpoint_bundle(const std::filesystem::path& bundle_root) 
    {
        if (!std::filesystem::exists(bundle_root)) 
        {
            throw std::runtime_error("checkpoint bundle path does not exist: " + bundle_root.string());
        }
        const auto manifest = read_file(bundle_root / "manifest.json");
        CheckpointBundle bundle;
        bundle.root = bundle_root;
        bundle.family_schema_version = parse_string(manifest, "family_schema_version");
        bundle.output_average_rule = parse_string(manifest, "output_average_rule");
        bundle.model_arch_id = parse_string(manifest, "model_arch_id");
        bundle.model_param_id = parse_string(manifest, "model_param_id");
        bundle.quant_cfg_id = parse_string(manifest, "quant_cfg_id");
        bundle.static_table_id = parse_string(manifest, "static_table_id");
        bundle.degree_bound_id = parse_string(manifest, "degree_bound_id");
        bundle.layer_count = parse_size_t(manifest, "L");
        bundle.K_out = parse_size_t(manifest, "K_out");
        bundle.class_count = parse_size_t(manifest, "C");
        bundle.d_in_profile = parse_size_array(manifest, "d_in_profile");

        for (const auto& entry : object_entries(manifest, "hidden_layers")) 
        {
            bundle.hidden_profile.push_back(HiddenLayerProfile{
                parse_size_t(entry, "head_count"),
                parse_size_t(entry, "head_dim"),
            });
        }
        for (const auto& entry : object_entries(manifest, "hidden_head_specs")) 
        {
            CheckpointHeadSpec spec;
            spec.layer_index = parse_size_t(entry, "layer_index");
            spec.local_head_index = parse_size_t(entry, "local_head_index");
            spec.global_head_index = parse_size_t(entry, "global_head_index");
            spec.seq_kernel = parse_string(entry, "seq_kernel");
            spec.attn_src_kernel = parse_string(entry, "attn_src_kernel");
            spec.attn_dst_kernel = parse_string(entry, "attn_dst_kernel");
            spec.attn_src_bias = parse_string(entry, "attn_src_bias");
            spec.attn_dst_bias = parse_string(entry, "attn_dst_bias");
            spec.output_bias = parse_string(entry, "output_bias");
            bundle.hidden_heads.push_back(std::move(spec));
        }
        for (const auto& entry : object_entries(manifest, "output_head_specs")) 
        {
            CheckpointOutputHeadSpec spec;
            spec.head_index = parse_size_t(entry, "head_index");
            spec.seq_kernel = parse_string(entry, "seq_kernel");
            spec.attn_src_kernel = parse_string(entry, "attn_src_kernel");
            spec.attn_dst_kernel = parse_string(entry, "attn_dst_kernel");
            spec.attn_src_bias = parse_string(entry, "attn_src_bias");
            spec.attn_dst_bias = parse_string(entry, "attn_dst_bias");
            spec.output_bias = parse_string(entry, "output_bias");
            bundle.output_heads.push_back(std::move(spec));
        }
        load_tensor_shapes(bundle_root / "tensors.txt", &bundle);
        for (const auto& head : bundle.hidden_heads) 
        {
            require_tensor(bundle, head.seq_kernel);
            require_tensor(bundle, head.attn_src_kernel);
            require_tensor(bundle, head.attn_dst_kernel);
            require_tensor(bundle, head.attn_src_bias);
            require_tensor(bundle, head.attn_dst_bias);
            require_tensor(bundle, head.output_bias);
            // The native formal circuit has no hidden attention/output-bias
            // argument.  Non-zero values would therefore be private freedom
            // outside H(W); reject them instead of silently dropping them.
            require_zero_tensor(bundle, head.attn_src_bias, "hidden head");
            require_zero_tensor(bundle, head.attn_dst_bias, "hidden head");
            require_zero_tensor(bundle, head.output_bias, "hidden head");
        }
        for (const auto& head : bundle.output_heads) 
        {
            require_tensor(bundle, head.seq_kernel);
            require_tensor(bundle, head.attn_src_kernel);
            require_tensor(bundle, head.attn_dst_kernel);
            require_tensor(bundle, head.attn_src_bias);
            require_tensor(bundle, head.attn_dst_bias);
            require_tensor(bundle, head.output_bias);
        }
        return bundle;
    }

    // Formal runs require an explicit, exact source identity match.  The
    // collision-resistant derived quant/table profiles are emitted separately.
    void validate_checkpoint_against_config(const CheckpointBundle& bundle, AppConfig* config) {
        if (bundle.family_schema_version.empty()) {
            throw std::runtime_error("checkpoint family_schema_version is empty");
        }
        if (bundle.output_average_rule != "per_head_bias_then_arithmetic_mean") {
            throw std::runtime_error("checkpoint output_average_rule does not match formal note");
        }
        if (bundle.layer_count != config->layer_count) {
            throw std::runtime_error("checkpoint L does not match config");
        }
        if (bundle.K_out != config->K_out) {
            throw std::runtime_error("checkpoint K_out does not match config");
        }
        if (bundle.class_count != config->num_classes) {
            throw std::runtime_error("checkpoint class count does not match config");
        }
        if (bundle.hidden_profile.size() != config->hidden_profile.size()) {
            throw std::runtime_error("checkpoint hidden_profile size does not match config");
        }
        for (std::size_t i = 0; i < bundle.hidden_profile.size(); ++i) {
            if (bundle.hidden_profile[i].head_count != config->hidden_profile[i].head_count
                || bundle.hidden_profile[i].head_dim != config->hidden_profile[i].head_dim) {
                throw std::runtime_error("checkpoint hidden_profile does not match config");
            }
        }
        if (!bundle.d_in_profile.empty() && !config->d_in_profile.empty() && bundle.d_in_profile != config->d_in_profile) {
            throw std::runtime_error("checkpoint d_in_profile does not match config");
        }
        const auto require_equal = [](const std::string& configured, const std::string& actual,
                                      const char* label) {
            if (configured.empty() || actual.empty() || configured != actual) {
                throw std::runtime_error(std::string("checkpoint ") + label + " does not match config");
            }
        };
        require_equal(config->model_arch_id, bundle.model_arch_id, "model_arch_id");
        require_equal(config->model_param_id, bundle.model_param_id, "model_param_id");
        require_equal(config->quant_cfg_id, bundle.quant_cfg_id, "quant_cfg_id");
        require_equal(config->static_table_id, bundle.static_table_id, "static_table_id");
        require_equal(config->degree_bound_id, bundle.degree_bound_id, "degree_bound_id");
    }

}
