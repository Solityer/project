#pragma once

// checkpoint bundle、manifest 与 GAT 参数加载。

// checkpoint 文本张量；shape 与 values 用于重建 VK_model 对应参数。
struct DenseTensor {
    std::vector<std::size_t> shape;
    std::vector<double> values;
};

// 计算张量元素总数；输入 shape，输出用于检查 dump 完整性。
std::size_t shape_product(const std::vector<std::size_t>& shape) {
    std::size_t out = 1;
    for (const auto dim : shape) {
        out *= dim;
    }
    return out;
}

// 读取 checkpoint tensors.txt；只读主项目 checkpoint，不修改上游来源。
std::unordered_map<std::string, DenseTensor> load_text_tensors(const fs::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open tensor dump: " + path.string());
    }
    std::unordered_map<std::string, DenseTensor> tensors;
    std::string marker;
    while (input >> marker) {
        if (marker != "TENSOR") {
            throw std::runtime_error("malformed tensor marker in " + path.string());
        }
        std::string name;
        std::size_t rank = 0;
        input >> name >> rank;
        DenseTensor tensor;
        tensor.shape.resize(rank);
        for (std::size_t i = 0; i < rank; ++i) {
            input >> tensor.shape[i];
        }
        std::size_t count = 0;
        input >> count;
        tensor.values.resize(count);
        for (std::size_t i = 0; i < count; ++i) {
            input >> tensor.values[i];
        }
        if (count != shape_product(tensor.shape)) {
            throw std::runtime_error("tensor count mismatch for " + name);
        }
        tensors.emplace(name, std::move(tensor));
    }
    return tensors;
}

// 从 manifest JSON 文本中取必需字符串字段；缺失即拒绝复现。
std::string parse_required_string(const std::string& text, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error("missing string key: " + key);
    }
    return match[1].str();
}

// 从 manifest JSON 文本中取必需整数字段；输出用于模型结构绑定。
std::size_t parse_required_size_t(const std::string& text, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*([0-9]+)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error("missing numeric key: " + key);
    }
    return static_cast<std::size_t>(std::stoull(match[1].str()));
}

// 提取 manifest 中对象数组正文；用于保持 checkpoint schema 的最小解析。
std::string extract_array_body(const std::string& text, const std::string& key) {
    const auto key_pos = text.find("\"" + key + "\"");
    if (key_pos == std::string::npos) {
        throw std::runtime_error("missing array key: " + key);
    }
    const auto begin = text.find('[', key_pos);
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
    throw std::runtime_error("unterminated array: " + key);
}

// 拆分 manifest 对象数组；输出每个 head/layer 的原始 JSON 片段。
std::vector<std::string> parse_object_array(const std::string& text, const std::string& key) {
    const auto body = extract_array_body(text, key);
    std::vector<std::string> entries;
    std::size_t depth = 0;
    std::size_t begin = std::string::npos;
    for (std::size_t i = 0; i < body.size(); ++i) {
        if (body[i] == '{') {
            if (depth == 0) begin = i;
            ++depth;
        } else if (body[i] == '}') {
            --depth;
            if (depth == 0 && begin != std::string::npos) {
                entries.push_back(body.substr(begin, i - begin + 1));
            }
        }
    }
    return entries;
}

// 单个 GAT attention head 参数；hidden/output 头共用该结构。
struct HeadParams {
    std::size_t layer_index = 0;
    std::size_t head_index = 0;
    std::size_t input_dim = 0;
    std::size_t output_dim = 0;
    std::vector<double> W;
    std::vector<double> attn_dst;
    std::vector<double> attn_src;
    double attn_dst_bias = 0.0;
    double attn_src_bias = 0.0;
    std::vector<double> output_bias;
};

// 模型验证键相关参数集合；model_* id 会进入 public metadata 和 proof object。
struct ModelParams {
    std::size_t L = 0;
    std::size_t K_out = 0;
    std::size_t C = 0;
    std::vector<std::pair<std::size_t, std::size_t>> hidden_profile;
    std::vector<HeadParams> hidden_heads;
    std::vector<HeadParams> output_heads;
    std::string model_arch_id;
    std::string model_param_id;
    std::string quant_cfg_id;
    std::string static_table_id;
    std::string output_average_rule;
};

// 单个 attention head 的 forward trace；relation runner 从中读取 witness 张量。
struct HeadTraceHost {
    std::size_t rows = 0;
    std::size_t width = 0;
    std::vector<double> h_prime;
    std::vector<double> e_src;
    std::vector<double> e_dst;
    std::vector<double> z;
    std::vector<double> m;
    std::vector<double> u;
    std::vector<double> sum;
    std::vector<double> h_pre;
    std::vector<double> h_out;
};

// 完整 GAT forward trace；按隐藏层、输出头和最终 logits 组织。
struct ForwardTraceHost {
    std::vector<std::vector<double>> hidden_layer_inputs;
    std::vector<std::size_t> hidden_layer_input_dims;
    std::vector<std::vector<HeadTraceHost>> hidden_heads;
    std::vector<double> output_layer_input;
    std::size_t output_layer_input_dim = 0;
    std::vector<HeadTraceHost> output_heads;
    std::vector<double> logits;
};

// 读取必需 checkpoint 张量；缺失时拒绝构造 VK_model。
const DenseTensor& require_tensor(const std::unordered_map<std::string, DenseTensor>& tensors, const std::string& name) {
    const auto it = tensors.find(name);
    if (it == tensors.end()) {
        throw std::runtime_error("missing tensor: " + name);
    }
    return it->second;
}

// 将 [1,in,out] kernel 展平成 row-major 矩阵；输出投影参数维度。
std::vector<double> reshape_seq_kernel(const DenseTensor& tensor, std::size_t* rows_out, std::size_t* cols_out) {
    if (tensor.shape.size() != 3 || tensor.shape[0] != 1) {
        throw std::runtime_error("expected seq kernel shape [1,in,out]");
    }
    *rows_out = tensor.shape[1];
    *cols_out = tensor.shape[2];
    return tensor.values;
}

// 将 [1,width,1] attention kernel 展平成向量；用于源/目标注意力点积。
std::vector<double> reshape_attention_kernel(const DenseTensor& tensor) {
    if (tensor.shape.size() != 3 || tensor.shape[0] != 1 || tensor.shape[2] != 1) {
        throw std::runtime_error("expected attention kernel shape [1,width,1]");
    }
    return tensor.values;
}

// 读取标量 bias；输入必须只有一个元素。
double reshape_scalar(const DenseTensor& tensor) {
    if (tensor.values.size() != 1) {
        throw std::runtime_error("expected scalar tensor");
    }
    return tensor.values.front();
}

// 从 checkpoint manifest 条目加载一个 attention head；输出会进入 forward 与 proof runner。
HeadParams load_head(
    const std::unordered_map<std::string, DenseTensor>& tensors,
    const std::string& entry,
    bool output_head) {
    HeadParams head;
    head.layer_index = output_head ? 0 : parse_required_size_t(entry, "layer_index");
    head.head_index = output_head ? parse_required_size_t(entry, "head_index") : parse_required_size_t(entry, "local_head_index");
    std::size_t rows = 0;
    std::size_t cols = 0;
    head.W = reshape_seq_kernel(require_tensor(tensors, parse_required_string(entry, "seq_kernel")), &rows, &cols);
    head.input_dim = rows;
    head.output_dim = cols;
    head.attn_dst = reshape_attention_kernel(require_tensor(tensors, parse_required_string(entry, "attn_dst_kernel")));
    head.attn_src = reshape_attention_kernel(require_tensor(tensors, parse_required_string(entry, "attn_src_kernel")));
    head.attn_dst_bias = reshape_scalar(require_tensor(tensors, parse_required_string(entry, "attn_dst_bias")));
    head.attn_src_bias = reshape_scalar(require_tensor(tensors, parse_required_string(entry, "attn_src_bias")));
    head.output_bias = require_tensor(tensors, parse_required_string(entry, "output_bias")).values;
    if (head.attn_dst.size() != head.output_dim || head.attn_src.size() != head.output_dim) {
        throw std::runtime_error("attention vector width mismatch");
    }
    return head;
}

// 加载完整 GAT checkpoint；输出模型结构、参数和公开 id，不写主仓库文件。
ModelParams load_model(const fs::path& checkpoint_root) {
    const auto manifest = read_text(checkpoint_root / "manifest.json");
    const auto tensors = load_text_tensors(checkpoint_root / "tensors.txt");
    ModelParams model;
    model.L = parse_required_size_t(manifest, "L");
    model.K_out = parse_required_size_t(manifest, "K_out");
    model.C = parse_required_size_t(manifest, "C");
    model.model_arch_id = parse_required_string(manifest, "model_arch_id");
    model.model_param_id = parse_required_string(manifest, "model_param_id");
    model.quant_cfg_id = parse_required_string(manifest, "quant_cfg_id");
    model.static_table_id = parse_required_string(manifest, "static_table_id");
    model.output_average_rule = parse_required_string(manifest, "output_average_rule");
    for (const auto& entry : parse_object_array(manifest, "hidden_layers")) {
        model.hidden_profile.push_back({
            parse_required_size_t(entry, "head_count"),
            parse_required_size_t(entry, "head_dim"),
        });
    }
    for (const auto& entry : parse_object_array(manifest, "hidden_head_specs")) {
        model.hidden_heads.push_back(load_head(tensors, entry, false));
    }
    for (const auto& entry : parse_object_array(manifest, "output_head_specs")) {
        model.output_heads.push_back(load_head(tensors, entry, true));
    }
    if (model.output_heads.size() != model.K_out) {
        throw std::runtime_error("output head count mismatch");
    }
    return model;
}
