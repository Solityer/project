#include "gatzk/data/loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

// 实现数据加载、缓存准备与局部图规范化。
// 它把外部数据源整理为 formal 主线一致消费的 dataset / local graph 对象。

namespace gatzk::data
{
    namespace
    {
        using algebra::FieldElement;
        std::vector<std::vector<double>> row_normalize(
            const std::vector<std::vector<double>> &raw_features)
        {
            // 先按行归一化输入特征，让数据进入参考前向时保持稳定的节点内尺度。
            std::vector<std::vector<double>> normalized = raw_features;
            for (auto &row : normalized)
            {
                double sum = 0.0;
                for (const auto value : row)
                {
                    if (!std::isfinite(value))
                        throw std::runtime_error(
                            "input feature contains NaN or Inf");
                    sum += value;
                }
                if (!std::isfinite(sum))
                    throw std::runtime_error(
                        "input feature normalization sum is not finite");
                if (sum == 0.0)
                {
                    continue;
                }
                for (auto &value : row)
                {
                    value /= sum;
                    if (!std::isfinite(value))
                        throw std::runtime_error(
                            "input feature normalization produced NaN or Inf");
                }
            }
            return normalized;
        }

        std::unordered_map<std::string, std::string> read_meta(const std::filesystem::path &path)
        {
            std::ifstream input(path);
            if (!input)
            {
                throw std::runtime_error("failed to open meta file: " + path.string());
            }
            std::unordered_map<std::string, std::string> out;
            std::string line;
            while (std::getline(input, line))
            {
                const auto hash = line.find('#');
                if (hash != std::string::npos)
                {
                    line = line.substr(0, hash);
                }
                const auto eq = line.find('=');
                if (eq == std::string::npos)
                {
                    continue;
                }
                auto key = line.substr(0, eq);
                auto value = line.substr(eq + 1);
                key.erase(std::remove_if(key.begin(), key.end(), ::isspace), key.end());
                value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch)
                                                        { return !std::isspace(ch); }));
                value.erase(std::find_if(value.rbegin(), value.rend(), [](unsigned char ch)
                                         { return !std::isspace(ch); })
                                .base(),
                            value.end());
                out[key] = value;
            }
            return out;
        }

        std::int64_t quantize_raw_at_scale(
            double value,
            std::uint64_t scale)
        {
            if (scale == 0 || !std::isfinite(value))
                throw std::runtime_error(
                    "input feature cannot be represented by the registered quantizer");
            const auto scaled = value * static_cast<double>(scale);
            if (!std::isfinite(scaled))
                throw std::runtime_error(
                    "input feature cannot be represented by the registered quantizer");
            const auto rounded = scaled >= 0.0
                                     ? std::floor(scaled + 0.5)
                                     : std::ceil(scaled - 0.5);
            if (!std::isfinite(rounded)
                || rounded > static_cast<double>(std::numeric_limits<std::int64_t>::max())
                || rounded < static_cast<double>(std::numeric_limits<std::int64_t>::min()))
                throw std::runtime_error(
                    "input feature cannot be represented by the registered quantizer");
            return static_cast<std::int64_t>(rounded);
        }

        FieldElement quantize(
            double value,
            std::uint64_t scale)
        {
            return FieldElement::from_signed(
                quantize_raw_at_scale(value, scale));
        }

        struct NpyArray
        {
            std::vector<std::size_t> shape;
            std::vector<double> values;
        };

        NpyArray load_npy_double(const std::filesystem::path &path)
        {
            std::ifstream input(path, std::ios::binary);
            if (!input.good())
            {
                throw std::runtime_error("failed to open npy file: " + path.string());
            }

            char magic[6] = {};
            input.read(magic, sizeof(magic));
            if (std::string(magic, sizeof(magic)) != "\x93NUMPY")
            {
                throw std::runtime_error("invalid npy magic: " + path.string());
            }

            std::uint8_t major = 0;
            std::uint8_t minor = 0;
            input.read(reinterpret_cast<char *>(&major), 1);
            input.read(reinterpret_cast<char *>(&minor), 1);
            (void)minor;

            std::uint32_t header_len = 0;
            if (major == 1)
            {
                std::uint16_t short_len = 0;
                input.read(reinterpret_cast<char *>(&short_len), sizeof(short_len));
                header_len = short_len;
            }
            else if (major == 2 || major == 3)
            {
                input.read(reinterpret_cast<char *>(&header_len), sizeof(header_len));
            }
            else
            {
                throw std::runtime_error("unsupported npy version: " + path.string());
            }

            std::string header(header_len, '\0');
            input.read(header.data(), static_cast<std::streamsize>(header.size()));
            const auto descr_pos = header.find("'descr'");
            if (descr_pos == std::string::npos)
            {
                throw std::runtime_error("missing descr in npy header: " + path.string());
            }
            const auto descr_quote = header.find('\'', header.find(':', descr_pos));
            const auto descr_end = header.find('\'', descr_quote + 1);
            const auto descr = header.substr(descr_quote + 1, descr_end - descr_quote - 1);
            const bool is_float64 = descr == "<f8" || descr == "|f8";
            const bool is_float32 = descr == "<f4" || descr == "|f4";
            if (!is_float64 && !is_float32)
            {
                throw std::runtime_error("expected float32/float64 npy payload: " + path.string());
            }

            const auto fortran_pos = header.find("fortran_order");
            if (fortran_pos == std::string::npos || header.find("False", fortran_pos) == std::string::npos)
            {
                throw std::runtime_error("fortran-order arrays are not supported: " + path.string());
            }

            const auto shape_pos = header.find('(', header.find("shape"));
            const auto shape_end = header.find(')', shape_pos);
            if (shape_pos == std::string::npos || shape_end == std::string::npos)
            {
                throw std::runtime_error("missing shape in npy header: " + path.string());
            }

            NpyArray out;
            std::stringstream shape_stream(header.substr(shape_pos + 1, shape_end - shape_pos - 1));
            std::string token;
            while (std::getline(shape_stream, token, ','))
            {
                token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
                if (!token.empty())
                {
                    out.shape.push_back(std::stoull(token));
                }
            }
            if (out.shape.empty())
            {
                throw std::runtime_error("empty shape in npy header: " + path.string());
            }

            std::size_t count = 1;
            for (const auto dim : out.shape)
            {
                count *= dim;
            }
            out.values.resize(count, 0.0);
            if (is_float64)
            {
                input.read(reinterpret_cast<char *>(out.values.data()), static_cast<std::streamsize>(count * sizeof(double)));
            }
            else
            {
                std::vector<float> float_values(count, 0.0f);
                input.read(reinterpret_cast<char *>(float_values.data()), static_cast<std::streamsize>(count * sizeof(float)));
                for (std::size_t i = 0; i < count; ++i)
                {
                    out.values[i] = static_cast<double>(float_values[i]);
                }
            }
            return out;
        }

        std::vector<std::vector<double>> load_features_matrix(
            const std::filesystem::path &root,
            std::size_t rows,
            std::size_t cols)
        {
            if (std::filesystem::exists(root / "features.txt"))
            {
                std::vector<std::vector<double>> raw_features(rows, std::vector<double>(cols, 0.0));
                std::ifstream input(root / "features.txt");
                if (!input)
                {
                    throw std::runtime_error("failed to open features file");
                }
                std::size_t file_rows = 0;
                std::size_t file_cols = 0;
                input >> file_rows >> file_cols;
                std::string line;
                std::getline(input, line);
                if (file_rows != rows || file_cols != cols)
                {
                    throw std::runtime_error("feature metadata mismatch");
                }
                for (std::size_t row = 0; row < file_rows; ++row)
                {
                    std::getline(input, line);
                    std::istringstream iss(line);
                    std::string entry;
                    while (iss >> entry)
                    {
                        const auto colon = entry.find(':');
                        const auto index = static_cast<std::size_t>(std::stoull(entry.substr(0, colon)));
                        const auto value = std::stod(entry.substr(colon + 1));
                        raw_features[row][index] = value;
                    }
                }
                return raw_features;
            }

            const auto array = load_npy_double(root / "features.npy");
            if (array.shape.size() != 2 || array.shape[0] != rows || array.shape[1] != cols)
            {
                throw std::runtime_error("feature npy shape mismatch");
            }
            std::vector<std::vector<double>> out(rows, std::vector<double>(cols, 0.0));
            for (std::size_t row = 0; row < rows; ++row)
            {
                for (std::size_t col = 0; col < cols; ++col)
                {
                    out[row][col] = array.values[row * cols + col];
                }
            }
            return out;
        }

        std::vector<int> load_labels(const std::filesystem::path &root, std::size_t rows)
        {
            std::vector<int> labels;
            if (!std::filesystem::exists(root / "labels.txt"))
            {
                labels.assign(rows, 0);
                return labels;
            }
            std::ifstream input(root / "labels.txt");
            if (!input)
            {
                throw std::runtime_error("failed to open labels file");
            }
            labels.reserve(rows);
            int label = 0;
            while (input >> label)
            {
                labels.push_back(label);
            }
            if (labels.size() != rows)
            {
                throw std::runtime_error("label count mismatch");
            }
            return labels;
        }

        std::vector<std::vector<int>> load_multi_labels(
            const std::filesystem::path &root,
            std::size_t rows,
            std::size_t cols)
        {
            const auto path = root / "labels_multi.txt";
            std::ifstream input(path);
            if (!input)
            {
                throw std::runtime_error("failed to open multi-label file: " + path.string());
            }
            std::size_t file_rows = 0;
            std::size_t file_cols = 0;
            if (!(input >> file_rows >> file_cols) || file_rows != rows || file_cols != cols)
            {
                throw std::runtime_error("multi-label metadata mismatch");
            }

            std::vector<std::vector<int>> labels(rows, std::vector<int>(cols, 0));
            for (std::size_t row = 0; row < rows; ++row)
            {
                for (std::size_t col = 0; col < cols; ++col)
                {
                    int value = 0;
                    if (!(input >> value) || (value != 0 && value != 1))
                    {
                        throw std::runtime_error("malformed multi-label value");
                    }
                    labels[row][col] = value;
                }
            }
            std::string extra;
            if (input >> extra)
            {
                throw std::runtime_error("extra data in multi-label file");
            }
            return labels;
        }

        std::vector<std::size_t> load_size_vector(const std::filesystem::path &path)
        {
            std::ifstream input(path);
            if (!input)
            {
                throw std::runtime_error("failed to open vector file: " + path.string());
            }
            std::vector<std::size_t> out;
            std::size_t value = 0;
            while (input >> value)
            {
                out.push_back(value);
            }
            return out;
        }

        GraphDataset load_cached_dataset(
            const std::filesystem::path &root,
            std::uint64_t input_scale)
        {
            if (input_scale == 0)
                throw std::runtime_error(
                    "formal dataset loading requires a positive S_input");
            const auto meta = read_meta(root / "meta.cfg");
            GraphDataset dataset;
            dataset.name = meta.at("name");
            dataset.num_nodes = std::stoull(meta.at("num_nodes"));
            dataset.num_features = std::stoull(meta.at("num_features"));
            dataset.num_classes = std::stoull(meta.at("num_classes"));
            if (meta.contains("graph_count"))
            {
                dataset.graph_count = std::stoull(meta.at("graph_count"));
            }
            if (meta.contains("task_type"))
            {
                dataset.task_type = meta.at("task_type");
            }
            if (meta.contains("report_unit"))
            {
                dataset.report_unit = meta.at("report_unit");
            }
            if (dataset.name == "ppi")
            {
                if (!meta.contains("label_format") ||
                    !meta.contains("split_format") ||
                    meta.at("label_format") != "multi_hot" ||
                    meta.at("split_format") != "node_masks_v1" ||
                    dataset.num_classes != 121)
                {
                    throw std::runtime_error(
                        "PPI cache must declare the canonical 121-column multi-hot label format");
                }
            }

            // 先恢复浮点特征，再量化成 formal 主线真正消费的域元素矩阵。
            const auto raw_features = load_features_matrix(root, dataset.num_nodes, dataset.num_features);
            dataset.features_fp = row_normalize(raw_features);
            dataset.features.assign(dataset.num_nodes, std::vector<FieldElement>(dataset.num_features, FieldElement::zero()));
            for (std::size_t row = 0; row < dataset.num_nodes; ++row)
            {
                for (std::size_t col = 0; col < dataset.num_features; ++col)
                {
                    dataset.features[row][col] = quantize(
                        dataset.features_fp[row][col],
                        input_scale);
                }
            }
            if (dataset.name == "ppi")
            {
                // PPI is a 121-way multi-label task.  Requiring this file
                // prevents a missing cache from silently becoming all-zero
                // single-label targets through the legacy labels.txt path.
                dataset.labels_multi = load_multi_labels(root, dataset.num_nodes, dataset.num_classes);
            }
            else
            {
                dataset.labels = load_labels(root, dataset.num_nodes);
            }

            if (std::filesystem::exists(root / "node_ptr.txt"))
            {
                dataset.node_ptr = load_size_vector(root / "node_ptr.txt");
            }
            else
            {
                dataset.node_ptr = {0, dataset.num_nodes};
            }
            if (dataset.node_ptr.size() < 2)
            {
                dataset.node_ptr = {0, dataset.num_nodes};
            }
            dataset.graph_count = dataset.node_ptr.size() - 1;

            {
                std::ifstream input(root / "edges.txt");
                if (!input)
                {
                    throw std::runtime_error("failed to open edges file");
                }
                std::string line;
                std::size_t stable_index = 0;
                while (std::getline(input, line))
                {
                    if (line.empty())
                    {
                        continue;
                    }
                    std::istringstream iss(line);
                    Edge edge;
                    edge.graph_id = 0;
                    edge.stable_index = stable_index++;
                    std::vector<std::size_t> parts;
                    std::size_t value = 0;
                    while (iss >> value)
                    {
                        parts.push_back(value);
                    }
                    if (parts.size() == 2)
                    {
                        edge.src = parts[0];
                        edge.dst = parts[1];
                    }
                    else if (parts.size() == 3)
                    {
                        edge.graph_id = parts[0];
                        edge.src = parts[1];
                        edge.dst = parts[2];
                    }
                    else
                    {
                        throw std::runtime_error("malformed edge line in " + (root / "edges.txt").string());
                    }
                    dataset.edges.push_back(edge);
                }
            }
            dataset.edge_ptr.assign(dataset.graph_count + 1, 0);
            for (const auto &edge : dataset.edges)
            {
                if (edge.graph_id >= dataset.graph_count)
                {
                    throw std::runtime_error("edge graph_id out of range");
                }
                ++dataset.edge_ptr[edge.graph_id + 1];
            }
            std::partial_sum(dataset.edge_ptr.begin(), dataset.edge_ptr.end(), dataset.edge_ptr.begin());
            return dataset;
        }

        // Planetoid 数据集允许按需生成 cache，但生成后的缓存才是 formal 主线真正读取的对象。
        void maybe_prepare_planetoid_cache(const util::AppConfig &config, const std::filesystem::path &cache_root)
        {
            if (!config.auto_prepare_dataset || std::filesystem::exists(cache_root / "meta.cfg"))
            {
                return;
            }
            const std::filesystem::path project_root(config.project_root);
            const auto python = std::filesystem::exists(project_root / ".venv/bin/python")
                                    ? (project_root / ".venv/bin/python").string()
                                    : std::string("python3");
            const auto script = (project_root / "scripts/prepare_planetoid.py").string();
            const auto command = python + " " + script +
                                 " --data-root " + (project_root / config.data_root).string() +
                                 " --dataset " + config.dataset +
                                 " --cache-root " + (project_root / config.cache_root).string();
            if (std::system(command.c_str()) != 0)
            {
                throw std::runtime_error("failed to prepare planetoid cache for dataset " + config.dataset);
            }
        }

        // PPI 多图 batch 同样先落缓存，再进入正式 proving 主线。
        void maybe_prepare_ppi_cache(const util::AppConfig &config, const std::filesystem::path &cache_root)
        {
            const bool canonical_cache =
                std::filesystem::exists(cache_root / "meta.cfg") &&
                std::filesystem::exists(cache_root / "features.npy") &&
                std::filesystem::exists(cache_root / "edges.txt") &&
                std::filesystem::exists(cache_root / "node_ptr.txt") &&
                std::filesystem::exists(cache_root / "labels_multi.npy") &&
                std::filesystem::exists(cache_root / "labels_multi.txt") &&
                std::filesystem::exists(cache_root / "train_mask.npy") &&
                std::filesystem::exists(cache_root / "val_mask.npy") &&
                std::filesystem::exists(cache_root / "test_mask.npy") &&
                !std::filesystem::exists(cache_root / "labels.txt");
            if (!config.auto_prepare_dataset || canonical_cache)
            {
                return;
            }
            const std::filesystem::path project_root(config.project_root);
            const auto python = std::filesystem::exists(project_root / ".venv/bin/python")
                                    ? (project_root / ".venv/bin/python").string()
                                    : std::string("python3");
            const auto script = (project_root / "scripts/prepare_ppi.py").string();
            const auto command = python + " " + script +
                                 " --data-root " + (project_root / config.data_root / "ppi").string() +
                                 " --output-root " + (project_root / config.cache_root / "ppi").string();
            if (std::system(command.c_str()) != 0)
            {
                throw std::runtime_error("failed to prepare PPI cache");
            }
        }

        // ogbn-arxiv 的缓存准备与其他数据集分开处理，因为原始数据布局不同。
        void maybe_prepare_ogbn_arxiv_cache(const util::AppConfig &config, const std::filesystem::path &cache_root)
        {
            if (!config.auto_prepare_dataset || std::filesystem::exists(cache_root / "meta.cfg"))
            {
                return;
            }
            const std::filesystem::path project_root(config.project_root);
            const auto python = std::filesystem::exists(project_root / ".venv/bin/python")
                                    ? (project_root / ".venv/bin/python").string()
                                    : std::string("python3");
            const auto script = (project_root / "scripts/prepare_ogbn_arxiv.py").string();
            const auto command = python + " " + script +
                                 " --project-root " + project_root.string() +
                                 " --data-root " + config.data_root;
            if (std::system(command.c_str()) != 0)
            {
                throw std::runtime_error("failed to prepare ogbn-arxiv cache");
            }
        }

        std::vector<std::size_t> selected_graph_ids(const GraphDataset &dataset, const util::AppConfig &config)
        {
            const auto batch_graphs = std::max<std::size_t>(1, config.batch_graphs);
            const auto count = std::min(batch_graphs, dataset.graph_count);
            std::vector<std::size_t> out(count, 0);
            std::iota(out.begin(), out.end(), 0);
            return out;
        }

        void fill_local_features(
            const GraphDataset &dataset,
            const std::vector<std::size_t> &absolute_ids,
            LocalGraph *local)
        {
            local->features_fp.reserve(local->features_fp.size() + absolute_ids.size());
            local->features.reserve(local->features.size() + absolute_ids.size());
            local->labels.reserve(local->labels.size() + absolute_ids.size());
            if (!dataset.labels_multi.empty())
            {
                local->labels_multi.reserve(local->labels_multi.size() + absolute_ids.size());
            }
            for (const auto absolute_id : absolute_ids)
            {
                local->absolute_ids.push_back(absolute_id);
                local->features_fp.push_back(dataset.features_fp.at(absolute_id));
                local->features.push_back(dataset.features.at(absolute_id));
                if (!dataset.labels.empty())
                {
                    local->labels.push_back(dataset.labels.at(absolute_id));
                }
                if (!dataset.labels_multi.empty())
                {
                    local->labels_multi.push_back(dataset.labels_multi.at(absolute_id));
                }
            }
        }

        std::uint64_t edge_key(std::size_t src, std::size_t dst, std::size_t graph_id)
        {
            return (static_cast<std::uint64_t>(graph_id) << 48U) ^ (static_cast<std::uint64_t>(src) << 24U) ^ static_cast<std::uint64_t>(dst);
        }

    } // namespace

    GraphDataset load_dataset(const util::AppConfig &config)
    {
        const std::filesystem::path project_root(config.project_root);
        const auto cache_root = project_root / config.cache_root / config.dataset;
        if (config.dataset == "ppi")
        {
            maybe_prepare_ppi_cache(config, cache_root);
        }
        else if (config.dataset == "ogbn_arxiv")
        {
            maybe_prepare_ogbn_arxiv_cache(config, cache_root);
        }
        else
        {
            maybe_prepare_planetoid_cache(config, cache_root);
        }
        return load_cached_dataset(
            cache_root,
            config.quantization.s_input);
    }

    void validate_input_feature_views(
        const GraphDataset &dataset,
        const LocalGraph &local,
        std::uint64_t input_scale)
    {
        if (input_scale == 0
            || dataset.num_nodes == 0
            || dataset.num_features == 0
            || dataset.features_fp.size() != dataset.num_nodes
            || dataset.features.size() != dataset.num_nodes
            || local.num_nodes == 0
            || local.num_features != dataset.num_features
            || local.absolute_ids.size() != local.num_nodes
            || local.features_fp.size() != local.num_nodes
            || local.features.size() != local.num_nodes)
            throw std::runtime_error(
                "input feature raw/quantized view shape mismatch");

        for (std::size_t row = 0; row < dataset.num_nodes; ++row)
        {
            if (dataset.features_fp[row].size() != dataset.num_features
                || dataset.features[row].size() != dataset.num_features)
                throw std::runtime_error(
                    "dataset input feature raw/quantized view shape mismatch");
            for (std::size_t column = 0;
                 column < dataset.num_features;
                 ++column)
            {
                const auto expected = quantize(
                    dataset.features_fp[row][column],
                    input_scale);
                if (dataset.features[row][column] != expected)
                    throw std::runtime_error(
                        "dataset input feature quantization does not match S_input");
            }
        }

        std::unordered_set<std::size_t> seen_ids;
        seen_ids.reserve(local.absolute_ids.size());
        for (std::size_t row = 0; row < local.num_nodes; ++row)
        {
            const auto absolute_id = local.absolute_ids[row];
            if (absolute_id >= dataset.num_nodes
                || !seen_ids.insert(absolute_id).second
                || local.features_fp[row].size() != local.num_features
                || local.features[row].size() != local.num_features)
                throw std::runtime_error(
                    "local input feature raw/quantized view shape mismatch");
            for (std::size_t column = 0;
                 column < local.num_features;
                 ++column)
            {
                const auto expected = quantize(
                    local.features_fp[row][column],
                    input_scale);
                if (local.features[row][column] != expected
                    || local.features[row][column]
                           != dataset.features[absolute_id][column]
                    || local.features_fp[row][column]
                           != dataset.features_fp[absolute_id][column])
                    throw std::runtime_error(
                        "local input feature views are not bound to dataset views");
            }
        }
    }

    void validate_normalized_graph_input(const LocalGraph &local)
    {
        const auto &public_input = local.public_input;
        if (public_input.N != local.num_nodes || public_input.E != local.edges.size() || public_input.G_batch != local.graph_count)
        {
            throw std::runtime_error("normalized graph size metadata mismatch");
        }
        if (public_input.N_total < public_input.N || public_input.I.size() != public_input.N || public_input.src.size() != public_input.E || public_input.dst.size() != public_input.E)
        {
            throw std::runtime_error("normalized graph public vector size mismatch");
        }
        if (public_input.G_batch == 0 || public_input.node_ptr.size() != public_input.G_batch + 1 || public_input.edge_ptr.size() != public_input.G_batch + 1 || public_input.node_ptr.front() != 0 || public_input.edge_ptr.front() != 0 || public_input.node_ptr.back() != public_input.N || public_input.edge_ptr.back() != public_input.E)
        {
            throw std::runtime_error("normalized graph batch pointer shape mismatch");
        }
        if (local.node_ptr != public_input.node_ptr || local.edge_ptr != public_input.edge_ptr)
        {
            throw std::runtime_error("local/public batch pointer mismatch");
        }

        std::unordered_set<std::size_t> absolute_ids;
        absolute_ids.reserve(public_input.I.size());
        for (const auto id : public_input.I)
        {
            if (id >= public_input.N_total || !absolute_ids.insert(id).second)
            {
                throw std::runtime_error("normalized graph contains an invalid or duplicate absolute node id");
            }
        }

        std::vector<std::size_t> indegree(public_input.N, 0);
        for (std::size_t graph_id = 0; graph_id < public_input.G_batch; ++graph_id)
        {
            const auto node_begin = public_input.node_ptr[graph_id];
            const auto node_end = public_input.node_ptr[graph_id + 1];
            const auto edge_begin = public_input.edge_ptr[graph_id];
            const auto edge_end = public_input.edge_ptr[graph_id + 1];
            if (node_begin >= node_end || edge_begin >= edge_end)
            {
                throw std::runtime_error("normalized graph batch pointers must be strictly increasing");
            }
            std::size_t previous_dst = 0;
            bool have_previous = false;
            for (std::size_t edge_index = edge_begin; edge_index < edge_end; ++edge_index)
            {
                const auto src = public_input.src[edge_index];
                const auto dst = public_input.dst[edge_index];
                const auto &edge = local.edges[edge_index];
                if (src < node_begin || src >= node_end || dst < node_begin || dst >= node_end)
                {
                    throw std::runtime_error("normalized graph contains a cross-graph edge");
                }
                if (edge.graph_id != graph_id || edge.src != src || edge.dst != dst)
                {
                    throw std::runtime_error("normalized graph edge/public input mismatch");
                }
                if (have_previous && dst < previous_dst)
                {
                    throw std::runtime_error("normalized graph edges are not stably grouped by destination");
                }
                previous_dst = dst;
                have_previous = true;
                ++indegree[dst];
            }
        }
        if (std::any_of(indegree.begin(), indegree.end(), [](std::size_t value)
                        { return value == 0; }))
        {
            throw std::runtime_error("normalized graph contains a zero-indegree node");
        }
    }

    // 按 formal 约束补自环、排序、切 batch，并生成公开输入摘要。
    LocalGraph normalize_graph_input(const GraphDataset &dataset, const util::AppConfig &config)
    {
        if (dataset.num_nodes == 0)
        {
            throw std::runtime_error("cannot normalize an empty dataset");
        }

        LocalGraph local;
        local.name = dataset.name + "_normalized";
        local.num_features = dataset.num_features;
        local.num_classes = dataset.num_classes;
        local.task_type = config.task_type.empty() ? dataset.task_type : config.task_type;
        local.report_unit = config.report_unit.empty() ? dataset.report_unit : config.report_unit;
        local.batching_rule = config.batching_rule;
        local.subgraph_rule = "whole_graph";
        local.self_loop_rule = config.self_loop_rule;
        local.edge_sort_rule = config.edge_sort_rule;
        local.chunking_rule = config.chunking_rule;
        local.graph_count = 0;
        local.public_input.N_total = dataset.num_nodes;

        const auto graph_ids = selected_graph_ids(dataset, config);
        const auto total_batch_graphs = graph_ids.empty() ? 1 : graph_ids.size();

        std::vector<std::size_t> batch_node_ptr = {0};
        std::vector<std::size_t> batch_edge_ptr = {0};
        std::vector<Edge> batch_edges;
        batch_edges.reserve(dataset.edges.size() + dataset.num_nodes);

        // 逐图抽出节点、做局部重编号，并把边改写成当前 batch 下的绝对局部编号。
        for (std::size_t batch_graph_id = 0; batch_graph_id < total_batch_graphs; ++batch_graph_id)
        {
            const auto dataset_graph_id = graph_ids[batch_graph_id];
            const auto node_begin = dataset.node_ptr.at(dataset_graph_id);
            const auto node_end = dataset.node_ptr.at(dataset_graph_id + 1);
            std::vector<std::size_t> graph_nodes(node_end - node_begin, 0);
            std::iota(graph_nodes.begin(), graph_nodes.end(), node_begin);

            auto selected_nodes = graph_nodes;
            std::sort(selected_nodes.begin(), selected_nodes.end());
            fill_local_features(dataset, selected_nodes, &local);
            const auto local_offset = batch_node_ptr.back();
            batch_node_ptr.push_back(local_offset + selected_nodes.size());

            std::unordered_map<std::size_t, std::size_t> global_to_local;
            global_to_local.reserve(selected_nodes.size());
            for (std::size_t i = 0; i < selected_nodes.size(); ++i)
            {
                global_to_local[selected_nodes[i]] = local_offset + i;
            }

            std::vector<Edge> graph_edges;
            std::unordered_set<std::uint64_t> seen;
            std::size_t stable_index = 0;
            for (const auto &edge : dataset.edges)
            {
                if (edge.graph_id != dataset_graph_id)
                {
                    continue;
                }
                if (!global_to_local.contains(edge.src) || !global_to_local.contains(edge.dst))
                {
                    continue;
                }
                auto append_edge = [&](std::size_t src_global, std::size_t dst_global)
                {
                    const auto src_local = global_to_local.at(src_global);
                    const auto dst_local = global_to_local.at(dst_global);
                    const auto key = edge_key(src_local, dst_local, batch_graph_id);
                    if (config.deduplicate_edges && !seen.insert(key).second)
                    {
                        return;
                    }
                    graph_edges.push_back(Edge{
                        .src = src_local,
                        .dst = dst_local,
                        .graph_id = batch_graph_id,
                        .stable_index = stable_index++,
                    });
                };
                append_edge(edge.src, edge.dst);
                if (config.symmetrize_edges && edge.src != edge.dst)
                {
                    append_edge(edge.dst, edge.src);
                }
            }
            // 自环在规范化阶段统一补齐，后面 softmax 聚合和 route/PSQ 都默认它存在。
            for (std::size_t i = 0; i < selected_nodes.size(); ++i)
            {
                const auto node_local = local_offset + i;
                const auto key = edge_key(node_local, node_local, batch_graph_id);
                if (!seen.insert(key).second)
                {
                    continue;
                }
                graph_edges.push_back(Edge{
                    .src = node_local,
                    .dst = node_local,
                    .graph_id = batch_graph_id,
                    .stable_index = stable_index++,
                });
            }

            // 最终边序固定为“先 graph_id，再 dst，再稳定索引”，与笔记里的公开排序约定对齐。
            std::stable_sort(graph_edges.begin(), graph_edges.end(), [](const Edge &lhs, const Edge &rhs)
                             {
            if (lhs.graph_id != rhs.graph_id) {
                return lhs.graph_id < rhs.graph_id;
            }
            if (lhs.dst != rhs.dst) {
                return lhs.dst < rhs.dst;
            }
            return lhs.stable_index < rhs.stable_index; });
            batch_edge_ptr.push_back(batch_edge_ptr.back() + graph_edges.size());
            batch_edges.insert(batch_edges.end(), graph_edges.begin(), graph_edges.end());
        }

        local.num_nodes = local.absolute_ids.size();
        local.edges = std::move(batch_edges);
        local.node_ptr = std::move(batch_node_ptr);
        local.edge_ptr = std::move(batch_edge_ptr);
        local.graph_count = local.node_ptr.size() - 1;
        local.public_input.N_total = dataset.num_nodes;
        local.public_input.N = local.num_nodes;
        local.public_input.E = local.edges.size();
        local.public_input.G_batch = local.graph_count;
        local.public_input.I = local.absolute_ids;
        local.public_input.node_ptr = local.node_ptr;
        local.public_input.edge_ptr = local.edge_ptr;
        local.public_input.src.reserve(local.edges.size());
        local.public_input.dst.reserve(local.edges.size());
        // 公开输入只保留 verifier 真正需要重放的 src/dst 序列和 batch 指针。
        for (const auto &edge : local.edges)
        {
            local.public_input.src.push_back(edge.src);
            local.public_input.dst.push_back(edge.dst);
        }
        local.public_input.src.shrink_to_fit();
        local.public_input.dst.shrink_to_fit();
        validate_normalized_graph_input(local);
        return local;
    }

} // namespace gatzk::data
