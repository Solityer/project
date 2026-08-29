#include "gat_ezkl/checkpoint.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <regex>
#include <set>
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

        std::size_t checked_shape_product(const std::vector<std::size_t>& shape,
                                           const std::string& name) {
            std::size_t product = 1;
            for (const auto dimension : shape) {
                if (dimension != 0 && product > std::numeric_limits<std::size_t>::max() / dimension) {
                    throw std::runtime_error("checkpoint tensor shape product overflow: " + name);
                }
                product *= dimension;
            }
            return product;
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
                if (!(input >> name >> rank) || name.empty()) {
                    throw std::runtime_error("malformed checkpoint tensor header");
                }
                TensorInfo info;
                info.shape.resize(rank);
                for (std::size_t i = 0; i < rank; ++i) {
                    if (!(input >> info.shape[i])) {
                        throw std::runtime_error("malformed checkpoint tensor shape: " + name);
                    }
                }
                const auto expected_count = checked_shape_product(info.shape, name);
                if (!(input >> info.value_count) || info.value_count != expected_count) {
                    throw std::runtime_error("checkpoint tensor value_count does not equal shape product: " + name);
                }
                info.values.resize(info.value_count);
                for (std::size_t i = 0; i < info.value_count; ++i) {
                    if (!(input >> info.values[i]) || !std::isfinite(info.values[i])) {
                        throw std::runtime_error("checkpoint tensor contains a non-finite value: " + name);
                    }
                }
                if (!bundle->tensors.emplace(name, std::move(info)).second) {
                    throw std::runtime_error("duplicate checkpoint tensor name: " + name);
                }
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

        void require_shape(const CheckpointBundle& bundle, const std::string& name,
                           std::initializer_list<std::size_t> expected) {
            const auto found = bundle.tensors.find(name);
            if (found == bundle.tensors.end()) {
                throw std::runtime_error("checkpoint tensor missing from tensors.txt: " + name);
            }
            const std::vector<std::size_t> shape(expected);
            if (found->second.shape != shape
                || found->second.value_count != checked_shape_product(shape, name)
                || found->second.values.size() != found->second.value_count) {
                throw std::runtime_error("checkpoint tensor shape mismatch: " + name);
            }
        }

        void require_quantizable(const CheckpointBundle& bundle, const std::string& name,
                                 double scale) {
            if (!(scale > 0.0) || !std::isfinite(scale)) {
                throw std::runtime_error("invalid canonical tensor quantization scale");
            }
            const auto found = bundle.tensors.find(name);
            if (found == bundle.tensors.end()) {
                throw std::runtime_error("checkpoint tensor missing from tensors.txt: " + name);
            }
            const long double max_scaled =
                static_cast<long double>(std::numeric_limits<std::int64_t>::max()) - 0.5L;
            for (const auto value : found->second.values) {
                const auto scaled = static_cast<long double>(value) * static_cast<long double>(scale);
                if (!std::isfinite(value) || !std::isfinite(scaled)
                    || std::abs(scaled) > max_scaled) {
                    throw std::runtime_error("checkpoint tensor cannot be represented by canonical signed i64: " + name);
                }
            }
        }

        void add_tensor_reference(std::set<std::string>* references, const std::string& name,
                                  const char* role) {
            if (name.empty()) {
                throw std::runtime_error(std::string("empty checkpoint tensor reference: ") + role);
            }
            if (!references->insert(name).second) {
                throw std::runtime_error("checkpoint tensor is referenced more than once: " + name);
            }
        }

        void validate_head_registry_and_tensor_shapes(const CheckpointBundle& bundle,
                                                      const AppConfig& config) {
            if (bundle.hidden_profile.size() != config.hidden_profile.size()
                || bundle.hidden_profile.empty()) {
                throw std::runtime_error("checkpoint hidden layer registry is incomplete");
            }
            if (bundle.d_in_profile.size() != config.d_in_profile.size()
                || bundle.d_in_profile != config.d_in_profile) {
                throw std::runtime_error("checkpoint d_in_profile is missing or non-canonical");
            }

            std::size_t expected_hidden_heads = 0;
            for (const auto& layer : config.hidden_profile) {
                if (layer.head_count == 0 || layer.head_dim == 0
                    || expected_hidden_heads > std::numeric_limits<std::size_t>::max() - layer.head_count) {
                    throw std::runtime_error("checkpoint hidden head registry overflows or contains an empty layer");
                }
                expected_hidden_heads += layer.head_count;
            }
            if (bundle.hidden_heads.size() != expected_hidden_heads
                || bundle.output_heads.size() != config.K_out) {
                throw std::runtime_error("checkpoint head registry is incomplete or has extra heads");
            }

            std::set<std::string> references;
            std::set<std::pair<std::size_t, std::size_t>> hidden_slots;
            std::set<std::size_t> global_slots;
            for (const auto& head : bundle.hidden_heads) {
                if (head.layer_index >= config.hidden_profile.size()
                    || head.local_head_index >= config.hidden_profile[head.layer_index].head_count
                    || head.global_head_index >= expected_hidden_heads
                    || !hidden_slots.emplace(head.layer_index, head.local_head_index).second
                    || !global_slots.insert(head.global_head_index).second) {
                    throw std::runtime_error("checkpoint hidden head indices are duplicate or out of range");
                }
                add_tensor_reference(&references, head.seq_kernel, "hidden.seq_kernel");
                add_tensor_reference(&references, head.attn_src_kernel, "hidden.attn_src_kernel");
                add_tensor_reference(&references, head.attn_dst_kernel, "hidden.attn_dst_kernel");
                add_tensor_reference(&references, head.attn_src_bias, "hidden.attn_src_bias");
                add_tensor_reference(&references, head.attn_dst_bias, "hidden.attn_dst_bias");
                add_tensor_reference(&references, head.output_bias, "hidden.output_bias");
            }
            for (std::size_t layer = 0; layer < config.hidden_profile.size(); ++layer) {
                for (std::size_t local = 0; local < config.hidden_profile[layer].head_count; ++local) {
                    if (!hidden_slots.count({layer, local})) {
                        throw std::runtime_error("checkpoint hidden head registry has a missing canonical slot");
                    }
                }
            }
            for (std::size_t global = 0; global < expected_hidden_heads; ++global) {
                if (!global_slots.count(global)) {
                    throw std::runtime_error("checkpoint hidden global head indices are not contiguous");
                }
            }

            std::set<std::size_t> output_slots;
            for (const auto& head : bundle.output_heads) {
                if (head.head_index >= config.K_out || !output_slots.insert(head.head_index).second) {
                    throw std::runtime_error("checkpoint output head indices are duplicate or out of range");
                }
                add_tensor_reference(&references, head.seq_kernel, "output.seq_kernel");
                add_tensor_reference(&references, head.attn_src_kernel, "output.attn_src_kernel");
                add_tensor_reference(&references, head.attn_dst_kernel, "output.attn_dst_kernel");
                add_tensor_reference(&references, head.attn_src_bias, "output.attn_src_bias");
                add_tensor_reference(&references, head.attn_dst_bias, "output.attn_dst_bias");
                add_tensor_reference(&references, head.output_bias, "output.output_bias");
            }
            for (std::size_t index = 0; index < config.K_out; ++index) {
                if (!output_slots.count(index)) {
                    throw std::runtime_error("checkpoint output head indices are not contiguous");
                }
            }
            if (references.size() != bundle.tensors.size()) {
                for (const auto& [name, info] : bundle.tensors) {
                    (void)info;
                    if (!references.count(name)) {
                        throw std::runtime_error("checkpoint tensors.txt contains an unknown tensor: " + name);
                    }
                }
                throw std::runtime_error("checkpoint tensor registry is not exact");
            }

            for (const auto& head : bundle.hidden_heads) {
                const auto& profile = config.hidden_profile[head.layer_index];
                const auto input_dim = config.d_in_profile.at(head.layer_index);
                require_shape(bundle, head.seq_kernel, {1, input_dim, profile.head_dim});
                require_shape(bundle, head.attn_src_kernel, {1, profile.head_dim, 1});
                require_shape(bundle, head.attn_dst_kernel, {1, profile.head_dim, 1});
                require_shape(bundle, head.attn_src_bias, {1});
                require_shape(bundle, head.attn_dst_bias, {1});
                require_shape(bundle, head.output_bias, {profile.head_dim});
                require_zero_tensor(bundle, head.attn_src_bias, "hidden head");
                require_zero_tensor(bundle, head.attn_dst_bias, "hidden head");
                require_zero_tensor(bundle, head.output_bias, "hidden head");
                require_quantizable(bundle, head.seq_kernel, 16.0);
                require_quantizable(bundle, head.attn_src_kernel, 16.0);
                require_quantizable(bundle, head.attn_dst_kernel, 16.0);
            }
            std::size_t output_input_dim = 0;
            const auto& last = config.hidden_profile.back();
            if (last.head_count > std::numeric_limits<std::size_t>::max() / last.head_dim) {
                throw std::runtime_error("output layer input width overflows size_t");
            }
            output_input_dim = last.head_count * last.head_dim;
            for (const auto& head : bundle.output_heads) {
                require_shape(bundle, head.seq_kernel, {1, output_input_dim, config.num_classes});
                require_shape(bundle, head.attn_src_kernel, {1, config.num_classes, 1});
                require_shape(bundle, head.attn_dst_kernel, {1, config.num_classes, 1});
                require_shape(bundle, head.attn_src_bias, {1});
                require_shape(bundle, head.attn_dst_bias, {1});
                require_shape(bundle, head.output_bias, {config.num_classes});
                require_zero_tensor(bundle, head.attn_src_bias, "output head");
                require_zero_tensor(bundle, head.attn_dst_bias, "output head");
                require_quantizable(bundle, head.seq_kernel, 16.0);
                require_quantizable(bundle, head.attn_src_kernel, 1.0);
                require_quantizable(bundle, head.attn_dst_kernel, 1.0);
                require_quantizable(bundle, head.output_bias, 16.0);
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
        if (config == nullptr) {
            throw std::runtime_error("checkpoint validation requires a non-null config");
        }
        if (bundle.family_schema_version != "multi_layer_multi_head_v2") {
            throw std::runtime_error(
                "checkpoint family_schema_version is not the canonical multi-layer schema");
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
        validate_head_registry_and_tensor_shapes(bundle, *config);
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
