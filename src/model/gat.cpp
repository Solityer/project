#include "gatzk/model/gat.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace gatzk::model
{
    namespace
    {
        // 读取和解析普通文本文件，清单文件和张量文件
        std::string load_text_file(const std::filesystem::path &path)
        {
            std::ifstream input(path);
            if (!input)
            {
                throw std::runtime_error("failed to open file: " + path.string());
            }
            std::ostringstream stream;
            stream << input.rdbuf();
            return stream.str();
        }
        std::size_t parse_required_size_t(const std::string &text, const std::string &key)
        {
            const std::regex pattern("\"" + key + "\"\\s*:\\s*([0-9]+)");
            std::smatch match;
            if (!std::regex_search(text, match, pattern))
            {
                throw std::runtime_error("missing numeric key in checkpoint manifest: " + key);
            }
            return static_cast<std::size_t>(std::stoull(match[1].str()));
        }

        std::vector<std::size_t> parse_required_size_array(
            const std::string &text,
            const std::string &key)
        {
            const std::regex pattern("\"" + key + "\"\\s*:\\s*\\[([^\\]]*)\\]");
            std::smatch match;
            if (!std::regex_search(text, match, pattern))
            {
                throw std::runtime_error(
                    "missing array key in checkpoint manifest: " + key);
            }
            std::string body = match[1].str();
            if (body.find_first_not_of("0123456789, \t\r\n") != std::string::npos)
            {
                throw std::runtime_error(
                    "malformed numeric array in checkpoint manifest: " + key);
            }
            std::replace(body.begin(), body.end(), ',', ' ');
            std::vector<std::size_t> out;
            std::istringstream input(body);
            std::size_t value = 0;
            while (input >> value)
            {
                out.push_back(value);
            }
            if (out.empty())
            {
                throw std::runtime_error(
                    "empty array in checkpoint manifest: " + key);
            }
            return out;
        }

        std::string parse_required_string(const std::string &text, const std::string &key)
        {
            const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"");
            std::smatch match;
            if (!std::regex_search(text, match, pattern))
            {
                throw std::runtime_error("missing string key in checkpoint manifest: " + key);
            }
            return match[1].str();
        }

        std::string extract_array_body(const std::string &text, const std::string &key)
        {
            const auto key_pos = text.find("\"" + key + "\"");
            if (key_pos == std::string::npos)
            {
                throw std::runtime_error("missing array key in checkpoint manifest: " + key);
            }
            const auto array_begin = text.find('[', key_pos);
            if (array_begin == std::string::npos)
            {
                throw std::runtime_error("malformed array in checkpoint manifest: " + key);
            }
            std::size_t depth = 0;
            for (std::size_t index = array_begin; index < text.size(); ++index)
            {
                if (text[index] == '[')
                {
                    ++depth;
                }
                else if (text[index] == ']')
                {
                    --depth;
                    if (depth == 0)
                    {
                        return text.substr(array_begin + 1, index - array_begin - 1);
                    }
                }
            }
            throw std::runtime_error("unterminated array in checkpoint manifest: " + key);
        }

        // 把一个对象数组拆成多个片段
        std::vector<std::string> parse_object_array_entries(const std::string &text, const std::string &key)
        {
            const auto body = extract_array_body(text, key);
            std::vector<std::string> entries;
            std::size_t depth = 0;
            std::size_t object_begin = std::string::npos;
            for (std::size_t index = 0; index < body.size(); ++index)
            {
                if (body[index] == '{')
                {
                    if (depth == 0)
                    {
                        object_begin = index;
                    }
                    ++depth;
                }
                else if (body[index] == '}')
                {
                    if (depth == 0)
                    {
                        throw std::runtime_error("malformed object array in checkpoint manifest: " + key);
                    }
                    --depth;
                    if (depth == 0 && object_begin != std::string::npos)
                    {
                        entries.push_back(body.substr(object_begin, index - object_begin + 1));
                        object_begin = std::string::npos;
                    }
                }
            }
            if (depth != 0)
            {
                throw std::runtime_error("unterminated object array in checkpoint manifest: " + key);
            }
            return entries;
        }

        // 解析每层隐藏层的形状信息
        CheckpointLayerInfo parse_checkpoint_layer_info(const std::string &entry)
        {
            CheckpointLayerInfo out;
            out.layer_index = parse_required_size_t(entry, "layer_index");
            out.input_dim = parse_required_size_t(entry, "input_dim");
            out.shape.head_count = parse_required_size_t(entry, "head_count");
            out.shape.head_dim = parse_required_size_t(entry, "head_dim");
            return out;
        }

        HiddenLayerShape parse_checkpoint_layer_shape(const std::string &entry)
        {
            return HiddenLayerShape{
                .head_count = parse_required_size_t(entry, "head_count"),
                .head_dim = parse_required_size_t(entry, "head_dim"),
            };
        }

        // 解析隐藏头张量名
        CheckpointHeadSpec parse_checkpoint_head_spec(const std::string &entry)
        {
            CheckpointHeadSpec out;
            out.layer_index = parse_required_size_t(entry, "layer_index");
            out.local_head_index = parse_required_size_t(entry, "local_head_index");
            // A flattened global index is optional legacy metadata.  Do not
            // parse it as a required field and never use it for formal lookup;
            // the layer/local pair above is the sole checkpoint identity.
            out.seq_kernel = parse_required_string(entry, "seq_kernel");
            out.attn_dst_kernel = parse_required_string(entry, "attn_dst_kernel");
            out.attn_dst_bias = parse_required_string(entry, "attn_dst_bias");
            out.attn_src_kernel = parse_required_string(entry, "attn_src_kernel");
            out.attn_src_bias = parse_required_string(entry, "attn_src_bias");
            out.output_bias = parse_required_string(entry, "output_bias");
            return out;
        }

        // 解析隐藏层每个注意力头的张量名映射
        CheckpointOutputHeadSpec parse_checkpoint_output_head_spec(const std::string &entry)
        {
            CheckpointOutputHeadSpec out;
            out.head_index = parse_required_size_t(entry, "head_index");
            out.seq_kernel = parse_required_string(entry, "seq_kernel");
            out.attn_dst_kernel = parse_required_string(entry, "attn_dst_kernel");
            out.attn_dst_bias = parse_required_string(entry, "attn_dst_bias");
            out.attn_src_kernel = parse_required_string(entry, "attn_src_kernel");
            out.attn_src_bias = parse_required_string(entry, "attn_src_bias");
            out.output_bias = parse_required_string(entry, "output_bias");
            return out;
        }

        // 把各个维度乘起来
        std::size_t shape_product(const std::vector<std::size_t> &shape)
        {
            std::size_t out = 1;
            for (const auto dim : shape)
            {
                out *= dim;
            }
            return out;
        }

        // 读取张量文件，并把每一段张量记录整理成一个稠密张量
        std::unordered_map<std::string, DenseTensor> load_text_tensors(const std::filesystem::path &path)
        {
            std::ifstream input(path);
            if (!input)
            {
                throw std::runtime_error("failed to open checkpoint tensor dump: " + path.string());
            }

            std::unordered_map<std::string, DenseTensor> out;
            std::string marker;
            while (input >> marker)
            {
                if (marker != "TENSOR")
                {
                    throw std::runtime_error("malformed tensor dump marker in " + path.string());
                }
                std::string name;
                std::size_t rank = 0;
                input >> name >> rank;
                DenseTensor tensor;
                tensor.shape.resize(rank);
                for (std::size_t i = 0; i < rank; ++i)
                {
                    input >> tensor.shape[i];
                }
                std::size_t count = 0;
                input >> count;
                tensor.values.resize(count, 0.0);
                for (std::size_t i = 0; i < count; ++i)
                {
                    input >> tensor.values[i];
                }
                if (count != shape_product(tensor.shape))
                {
                    throw std::runtime_error("tensor dump count mismatch for " + name);
                }
                out.emplace(name, std::move(tensor));
            }
            return out;
        }

        // 取一个必须存在的张量，没找到就直接报错
        const DenseTensor &require_tensor(
            const std::unordered_map<std::string, DenseTensor> &tensors,
            const std::string &name)
        {
            const auto it = tensors.find(name);
            if (it == tensors.end())
            {
                throw std::runtime_error("missing tensor in checkpoint dump: " + name);
            }
            return it->second;
        }

        // 将 [1, in, out] 的 seq kernel 张量 reshape 成二维矩阵
        FloatMatrix reshape_seq_kernel(const DenseTensor &tensor)
        {
            if (tensor.shape.size() != 3 || tensor.shape[0] != 1)
            {
                throw std::runtime_error("expected seq kernel tensor shape [1, in, out]");
            }
            const auto rows = tensor.shape[1];
            const auto cols = tensor.shape[2];
            FloatMatrix out(rows, std::vector<double>(cols, 0.0));
            for (std::size_t row = 0; row < rows; ++row)
            {
                for (std::size_t col = 0; col < cols; ++col)
                {
                    out[row][col] = tensor.values[row * cols + col];
                }
            }
            return out;
        }

        // 将 [1, width, 1] 的注意力 kernel 张量展平成向量
        std::vector<double> reshape_attention_kernel(const DenseTensor &tensor)
        {
            if (tensor.shape.size() != 3 || tensor.shape[0] != 1 || tensor.shape[2] != 1)
            {
                throw std::runtime_error("expected attention kernel tensor shape [1, width, 1]");
            }
            return tensor.values;
        }

        double reshape_scalar_bias(const DenseTensor &tensor)
        {
            if (tensor.shape.size() != 1 || tensor.shape[0] != 1 || tensor.values.size() != 1)
            {
                throw std::runtime_error("expected scalar bias tensor shape [1]");
            }
            return tensor.values.front();
        }

        std::vector<double> reshape_output_bias(const DenseTensor &tensor)
        {
            if (tensor.shape.size() != 1)
            {
                throw std::runtime_error("expected output bias tensor shape [width]");
            }
            return tensor.values;
        }

        // 按张量名字把一个注意力头需要的浮点参数封装起来
        AttentionHeadParameters load_head_parameters(
            const std::unordered_map<std::string, DenseTensor> &tensors,
            const std::string &seq_kernel_name,
            const std::string &attn_dst_kernel_name,
            const std::string &attn_dst_bias_name,
            const std::string &attn_src_kernel_name,
            const std::string &attn_src_bias_name,
            const std::string &output_bias_name)
        {
            AttentionHeadParameters out;
            out.seq_kernel_fp = reshape_seq_kernel(require_tensor(tensors, seq_kernel_name));
            out.attn_dst_kernel_fp = reshape_attention_kernel(require_tensor(tensors, attn_dst_kernel_name));
            out.attn_dst_bias_fp = reshape_scalar_bias(require_tensor(tensors, attn_dst_bias_name));
            out.attn_src_kernel_fp = reshape_attention_kernel(require_tensor(tensors, attn_src_kernel_name));
            out.attn_src_bias_fp = reshape_scalar_bias(require_tensor(tensors, attn_src_bias_name));
            out.output_bias_fp = reshape_output_bias(require_tensor(tensors, output_bias_name));
            return out;
        }
        // Leaky ReLU激活函数，默认斜率为0.2
        double leaky_relu(double value, double alpha = 0.2)
        {
            return value >= 0.0 ? value : alpha * value;
        }

        // ELU激活函数
        double elu(double value)
        {
            return value >= 0.0 ? value : std::expm1(value);
        }

        constexpr std::size_t kMaxForwardWorkerCount = 32;

        std::size_t forward_worker_budget(
            std::size_t requested_worker_count = 0)
        {
            const auto hardware_workers =
                std::max<std::size_t>(
                    1,
                    std::thread::hardware_concurrency());

            const auto requested =
                requested_worker_count == 0
                    ? hardware_workers
                    : std::min(
                          requested_worker_count,
                          hardware_workers);

            return std::max<std::size_t>(
                1,
                std::min<std::size_t>(
                    kMaxForwardWorkerCount,
                    requested));
        }

        template <typename Fn>
        void parallel_for_forward_ranges(
            std::size_t item_count,
            std::size_t min_items_per_worker,
            std::size_t max_worker_count,
            Fn &&fn)
        {
            if (item_count == 0)
            {
                return;
            }

            const auto worker_budget =
                forward_worker_budget(max_worker_count);

            const auto safe_min_items =
                std::max<std::size_t>(
                    1,
                    min_items_per_worker);

            const auto workers_by_work =
                (item_count + safe_min_items - 1) / safe_min_items;

            const auto worker_count =
                std::max<std::size_t>(
                    1,
                    std::min({
                        worker_budget,
                        item_count,
                        workers_by_work,
                    }));

            if (worker_count <= 1)
            {
                fn(0, item_count);
                return;
            }

            const auto chunk_size =
                (item_count + worker_count - 1) / worker_count;

            std::vector<std::future<void>> futures;
            futures.reserve(worker_count);

            for (std::size_t worker = 0;
                 worker < worker_count;
                 ++worker)
            {
                const auto begin =
                    worker * chunk_size;

                const auto end =
                    std::min(
                        item_count,
                        begin + chunk_size);

                if (begin >= end)
                {
                    break;
                }

                futures.push_back(
                    std::async(
                        std::launch::async,
                        [&fn, begin, end]()
                        {
                            fn(begin, end);
                        }));
            }

            for (auto &future : futures)
            {
                future.get();
            }
        }

        struct ForwardGraphLayout
        {
            std::vector<std::size_t> incoming_offsets;
            std::vector<std::size_t> incoming_sources;

            // Maps every real edge to its position in incoming_sources.
            // Synthetic self loops are present only in incoming_sources.
            std::vector<std::size_t> edge_incoming_positions;
        };

        ForwardGraphLayout build_forward_graph_layout(
            std::size_t node_count,
            const std::vector<data::Edge> &edges)
        {
            ForwardGraphLayout layout;

            layout.incoming_offsets.assign(
                node_count + 1,
                0);

            layout.edge_incoming_positions.resize(
                edges.size());

            std::vector<std::size_t>
                incoming_count(
                    node_count,
                    0);

            std::vector<unsigned char>
                has_self_loop(
                    node_count,
                    0);

            // First pass validates the graph and computes exact CSR sizes.
            for (const auto &edge : edges)
            {
                if (edge.src >= node_count || edge.dst >= node_count)
                {
                    throw std::runtime_error(
                        "edge out of bounds while running attention head forward");
                }

                ++incoming_count[edge.dst];

                if (edge.src == edge.dst)
                {
                    has_self_loop[edge.dst] = 1;
                }
            }

            for (std::size_t dst = 0;
                 dst < node_count;
                 ++dst)
            {
                const auto synthetic_self =
                    has_self_loop[dst] == 0
                        ? std::size_t{1}
                        : std::size_t{0};

                layout.incoming_offsets[dst + 1] =
                    layout.incoming_offsets[dst] + incoming_count[dst] + synthetic_self;
            }

            layout.incoming_sources.resize(
                layout.incoming_offsets.back());

            auto cursor =
                layout.incoming_offsets;

            // Preserve the original edge order inside every destination
            // group exactly.
            for (std::size_t edge_index = 0;
                 edge_index < edges.size();
                 ++edge_index)
            {
                const auto &edge =
                    edges[edge_index];

                const auto position =
                    cursor[edge.dst]++;

                layout.incoming_sources[position] =
                    edge.src;

                layout.edge_incoming_positions[edge_index] =
                    position;
            }

            // The former implementation appended a missing self loop after
            // all real incoming edges. Preserve that exact ordering.
            for (std::size_t dst = 0;
                 dst < node_count;
                 ++dst)
            {
                if (has_self_loop[dst] == 0)
                {
                    layout.incoming_sources[cursor[dst]++] =
                        dst;
                }

                if (cursor[dst] != layout.incoming_offsets[dst + 1])
                {
                    throw std::runtime_error(
                        "forward incoming-edge layout construction mismatch");
                }
            }

            return layout;
        }

        struct ForwardHeadExecutionPlan
        {
            std::size_t outer_head_workers = 1;
            std::size_t inner_workers = 1;
        };

        ForwardHeadExecutionPlan
        forward_head_execution_plan(
            std::size_t head_count)
        {
            const auto hardware_workers =
                forward_worker_budget();

            // Workload-class scheduling only.  For many independent heads,
            // parallelize heads and keep each head internally serial.  For a
            // small head count, keep heads serial and parallelize independent
            // rows/edges inside each head.  This is a generic non-nested
            // execution policy; formal model shape validation and the retained
            // flattened compatibility view are handled by their callers.
            if (head_count >= 4 && hardware_workers >= 4)
            {
                return ForwardHeadExecutionPlan{
                    .outer_head_workers =
                        std::min(
                            head_count,
                            hardware_workers),
                    .inner_workers = 1,
                };
            }

            return ForwardHeadExecutionPlan{
                .outer_head_workers = 1,
                .inner_workers =
                    hardware_workers,
            };
        }

        // Floating-point matrix multiplication.
        //
        // Parallelism is only across independent output rows. The k/j
        // accumulation order of every output scalar is unchanged.
        FloatMatrix matmul(
            const FloatMatrix &left,
            const FloatMatrix &right,
            std::size_t max_worker_count = 1)
        {
            if (left.empty() || right.empty())
            {
                return {};
            }

            const auto shared =
                left.front().size();

            if (right.size() != shared)
            {
                throw std::runtime_error(
                    "float matrix multiply dimension mismatch");
            }

            const auto rows =
                left.size();

            const auto cols =
                right.front().size();

            FloatMatrix out(
                rows,
                std::vector<double>(
                    cols,
                    0.0));

            parallel_for_forward_ranges(
                rows,
                64,
                max_worker_count,
                [&](std::size_t begin,
                    std::size_t end)
                {
                    for (std::size_t i = begin;
                         i < end;
                         ++i)
                    {
                        for (std::size_t k = 0;
                             k < shared;
                             ++k)
                        {
                            const auto lhs =
                                left[i][k];

                            for (std::size_t j = 0;
                                 j < cols;
                                 ++j)
                            {
                                out[i][j] +=
                                    lhs * right[k][j];
                            }
                        }
                    }
                });

            return out;
        }

        // Concatenate one hidden layer's head outputs in canonical head order.
        FloatMatrix concatenate_columns(
            const std::vector<FloatMatrix> &matrices)
        {
            if (matrices.empty())
            {
                return {};
            }

            const auto rows =
                matrices.front().size();

            std::size_t cols = 0;

            for (const auto &matrix : matrices)
            {
                if (matrix.size() != rows)
                {
                    throw std::runtime_error(
                        "cannot concatenate matrices with different row counts");
                }

                cols +=
                    matrix.empty()
                        ? 0
                        : matrix.front().size();
            }

            FloatMatrix out(
                rows,
                std::vector<double>(
                    cols,
                    0.0));

            std::size_t offset = 0;

            for (const auto &matrix : matrices)
            {
                const auto width =
                    matrix.empty()
                        ? 0
                        : matrix.front().size();

                for (std::size_t row = 0;
                     row < rows;
                     ++row)
                {
                    std::copy(
                        matrix[row].begin(),
                        matrix[row].end(),
                        out[row].begin() + static_cast<
                                               std::ptrdiff_t>(
                                               offset));
                }

                offset += width;
            }

            return out;
        }

        FloatMatrix concatenate_head_outputs(
            const std::vector<HeadForwardTrace> &heads,
            std::size_t max_worker_count)
        {
            if (heads.empty())
            {
                return {};
            }

            const auto rows =
                heads.front().H_agg.size();

            std::vector<std::size_t>
                offsets(
                    heads.size() + 1,
                    0);

            for (std::size_t head = 0;
                 head < heads.size();
                 ++head)
            {
                if (heads[head].H_agg.size() != rows)
                {
                    throw std::runtime_error(
                        "cannot concatenate head outputs with different row counts");
                }

                const auto width =
                    heads[head].H_agg.empty()
                        ? 0
                        : heads[head]
                              .H_agg.front()
                              .size();

                offsets[head + 1] =
                    offsets[head] + width;
            }

            FloatMatrix out(
                rows,
                std::vector<double>(
                    offsets.back(),
                    0.0));

            parallel_for_forward_ranges(
                rows,
                128,
                max_worker_count,
                [&](std::size_t begin,
                    std::size_t end)
                {
                    for (std::size_t row = begin;
                         row < end;
                         ++row)
                    {
                        for (std::size_t head = 0;
                             head < heads.size();
                             ++head)
                        {
                            const auto &source =
                                heads[head]
                                    .H_agg[row];

                            const auto expected_width =
                                offsets[head + 1] - offsets[head];

                            if (source.size() != expected_width)
                            {
                                throw std::runtime_error(
                                    "head output row width mismatch");
                            }

                            std::copy(
                                source.begin(),
                                source.end(),
                                out[row].begin() + static_cast<
                                                       std::ptrdiff_t>(
                                                       offsets[head]));
                        }
                    }
                });

            return out;
        }

        FloatMatrix average_head_outputs(
            const std::vector<HeadForwardTrace> &heads,
            bool pre_bias,
            std::size_t max_worker_count)
        {
            if (heads.empty())
            {
                return {};
            }

            const auto &first =
                pre_bias
                    ? heads.front()
                          .H_agg_pre_bias
                    : heads.front().H_agg;

            const auto rows =
                first.size();

            const auto cols =
                first.empty()
                    ? 0
                    : first.front().size();

            for (const auto &head : heads)
            {
                const auto &matrix =
                    pre_bias
                        ? head.H_agg_pre_bias
                        : head.H_agg;

                if (matrix.size() != rows || (!matrix.empty() && matrix.front().size() != cols))
                {
                    throw std::runtime_error(
                        "cannot average head outputs with different shapes");
                }
            }

            FloatMatrix out(
                rows,
                std::vector<double>(
                    cols,
                    0.0));

            const auto scale =
                1.0 / static_cast<double>(
                          heads.size());

            parallel_for_forward_ranges(
                rows,
                128,
                max_worker_count,
                [&](std::size_t begin,
                    std::size_t end)
                {
                    for (std::size_t row = begin;
                         row < end;
                         ++row)
                    {
                        for (std::size_t col = 0;
                             col < cols;
                             ++col)
                        {
                            const auto &first_matrix =
                                pre_bias
                                    ? heads.front()
                                          .H_agg_pre_bias
                                    : heads.front()
                                          .H_agg;

                            double value =
                                first_matrix[row][col];

                            for (std::size_t head = 1;
                                 head < heads.size();
                                 ++head)
                            {
                                const auto &matrix =
                                    pre_bias
                                        ? heads[head]
                                              .H_agg_pre_bias
                                        : heads[head]
                                              .H_agg;

                                value +=
                                    matrix[row][col];
                            }

                            out[row][col] =
                                value * scale;
                        }
                    }
                });

            return out;
        }

        // Full attention-head forward.
        //
        // The prepared graph layout and worker placement are execution-only.
        // Every scalar keeps the same arithmetic/reduction ordering as the
        // previous implementation.
        HeadForwardTrace attention_head_forward_impl(
            const FloatMatrix &features,
            const std::vector<data::Edge> &edges,
            const AttentionHeadParameters &parameters,
            bool apply_output_bias,
            bool apply_activation,
            HeadForwardProfile *profile,
            const ForwardGraphLayout *prepared_graph = nullptr,
            std::size_t max_worker_count = 1)
        {
            using Clock =
                std::chrono::steady_clock;

            auto stage_start =
                Clock::now();

            if (features.empty())
            {
                return {};
            }

            const auto node_count =
                features.size();

            const auto width =
                attention_head_output_width(
                    parameters);

            ForwardGraphLayout local_graph;

            const ForwardGraphLayout *graph =
                prepared_graph;

            if (graph == nullptr)
            {
                local_graph =
                    build_forward_graph_layout(
                        node_count,
                        edges);

                graph = &local_graph;
            }
            else if (
                graph->incoming_offsets.size() != node_count + 1 || graph
                                                                            ->edge_incoming_positions
                                                                            .size() != edges.size())
            {
                throw std::runtime_error(
                    "prepared forward graph layout shape mismatch");
            }

            HeadForwardTrace trace;

            // 1. H' = H * W_seq.
            trace.H_prime =
                matmul(
                    features,
                    parameters.seq_kernel_fp,
                    max_worker_count);

            if (profile != nullptr)
            {
                profile->projection_ms +=
                    std::chrono::duration<
                        double,
                        std::milli>(
                        Clock::now() - stage_start)
                        .count();
            }

            stage_start =
                Clock::now();

            // 2. E_dst / E_src.
            //
            // Both vectors scan the same H' row. Their individual summation
            // order remains identical to the former two matvec calls.
            trace.E_dst.assign(
                node_count,
                parameters.attn_dst_bias_fp);

            trace.E_src.assign(
                node_count,
                parameters.attn_src_bias_fp);

            parallel_for_forward_ranges(
                node_count,
                256,
                max_worker_count,
                [&](std::size_t begin,
                    std::size_t end)
                {
                    for (std::size_t row = begin;
                         row < end;
                         ++row)
                    {
                        for (std::size_t col = 0;
                             col < parameters
                                       .attn_dst_kernel_fp
                                       .size();
                             ++col)
                        {
                            trace.E_dst[row] +=
                                trace.H_prime[row][col] * parameters
                                                              .attn_dst_kernel_fp[col];
                        }

                        for (std::size_t col = 0;
                             col < parameters
                                       .attn_src_kernel_fp
                                       .size();
                             ++col)
                        {
                            trace.E_src[row] +=
                                trace.H_prime[row][col] * parameters
                                                              .attn_src_kernel_fp[col];
                        }
                    }
                });

            trace.S.assign(
                edges.size(),
                0.0);

            trace.Z.assign(
                edges.size(),
                0.0);

            trace.M.assign(
                node_count,
                0.0);

            trace.Delta.assign(
                edges.size(),
                0.0);

            trace.U.assign(
                edges.size(),
                0.0);

            trace.Sum.assign(
                node_count,
                0.0);

            trace.inv.assign(
                node_count,
                0.0);

            trace.alpha.assign(
                edges.size(),
                0.0);

            trace.H_agg_pre_bias.assign(
                node_count,
                std::vector<double>(
                    width,
                    0.0));

            trace.H_agg.assign(
                node_count,
                std::vector<double>(
                    width,
                    0.0));

            // One temporary value per real-or-synthetic incoming edge.
            // The previous implementation evaluated this identical exp()
            // repeatedly in Sum, aggregation and edge materialization.
            std::vector<double>
                incoming_exp(
                    graph->incoming_sources.size(),
                    0.0);

            // 3. Stable softmax maximum.
            parallel_for_forward_ranges(
                node_count,
                256,
                max_worker_count,
                [&](std::size_t begin,
                    std::size_t end)
                {
                    for (std::size_t dst = begin;
                         dst < end;
                         ++dst)
                    {
                        auto maximum =
                            -std::numeric_limits<
                                double>::infinity();

                        const auto incoming_begin =
                            graph->incoming_offsets[dst];

                        const auto incoming_end =
                            graph->incoming_offsets[dst + 1];

                        for (std::size_t position =
                                 incoming_begin;
                             position < incoming_end;
                             ++position)
                        {
                            const auto src =
                                graph->incoming_sources[position];

                            maximum =
                                std::max(
                                    maximum,
                                    leaky_relu(
                                        trace.E_dst[dst] + trace.E_src[src]));
                        }

                        trace.M[dst] =
                            maximum;
                    }
                });

            // 4/5. Evaluate every softmax numerator once, accumulate the
            // denominator in the original incoming-edge order, then invert.
            parallel_for_forward_ranges(
                node_count,
                256,
                max_worker_count,
                [&](std::size_t begin,
                    std::size_t end)
                {
                    for (std::size_t dst = begin;
                         dst < end;
                         ++dst)
                    {
                        double sum = 0.0;

                        const auto incoming_begin =
                            graph->incoming_offsets[dst];

                        const auto incoming_end =
                            graph->incoming_offsets[dst + 1];

                        for (std::size_t position =
                                 incoming_begin;
                             position < incoming_end;
                             ++position)
                        {
                            const auto src =
                                graph->incoming_sources[position];

                            const auto value =
                                std::exp(
                                    leaky_relu(
                                        trace.E_dst[dst] + trace.E_src[src]) -
                                    trace.M[dst]);

                            incoming_exp[position] =
                                value;

                            sum += value;
                        }

                        trace.Sum[dst] =
                            sum;

                        trace.inv[dst] =
                            1.0 / sum;
                    }
                });

            // 6. Neighbor aggregation. Each destination row is independent;
            // source order and per-column accumulation order are unchanged.
            parallel_for_forward_ranges(
                node_count,
                128,
                max_worker_count,
                [&](std::size_t begin,
                    std::size_t end)
                {
                    for (std::size_t dst = begin;
                         dst < end;
                         ++dst)
                    {
                        const auto incoming_begin =
                            graph->incoming_offsets[dst];

                        const auto incoming_end =
                            graph->incoming_offsets[dst + 1];

                        for (std::size_t position =
                                 incoming_begin;
                             position < incoming_end;
                             ++position)
                        {
                            const auto src =
                                graph->incoming_sources[position];

                            const auto alpha =
                                incoming_exp[position] * trace.inv[dst];

                            for (std::size_t col = 0;
                                 col < width;
                                 ++col)
                            {
                                trace
                                    .H_agg_pre_bias
                                        [dst][col] += alpha * trace
                                                                  .H_prime
                                                                      [src][col];
                            }
                        }
                    }
                });

            // Materialize the protocol-visible real-edge intermediates.
            // U and alpha reuse the already-computed identical exp value.
            parallel_for_forward_ranges(
                edges.size(),
                4096,
                max_worker_count,
                [&](std::size_t begin,
                    std::size_t end)
                {
                    for (std::size_t edge_index =
                             begin;
                         edge_index < end;
                         ++edge_index)
                    {
                        const auto &edge =
                            edges[edge_index];

                        trace.S[edge_index] =
                            trace.E_dst[edge.dst] + trace.E_src[edge.src];

                        trace.Z[edge_index] =
                            leaky_relu(
                                trace.S[edge_index]);

                        trace.Delta[edge_index] =
                            trace.M[edge.dst] - trace.Z[edge_index];

                        const auto incoming_position =
                            graph
                                ->edge_incoming_positions[edge_index];

                        trace.U[edge_index] =
                            incoming_exp[incoming_position];

                        trace.alpha[edge_index] =
                            trace.U[edge_index] * trace.inv[edge.dst];
                    }
                });

            if (profile != nullptr)
            {
                profile->attention_ms +=
                    std::chrono::duration<
                        double,
                        std::milli>(
                        Clock::now() - stage_start)
                        .count();
            }

            stage_start =
                Clock::now();

            // 7. Bias + ELU. Rows are independent.
            parallel_for_forward_ranges(
                node_count,
                256,
                max_worker_count,
                [&](std::size_t begin,
                    std::size_t end)
                {
                    for (std::size_t dst = begin;
                         dst < end;
                         ++dst)
                    {
                        for (std::size_t col = 0;
                             col < width;
                             ++col)
                        {
                            auto value =
                                trace
                                    .H_agg_pre_bias
                                        [dst][col];

                            if (apply_output_bias && col < parameters
                                                               .output_bias_fp
                                                               .size())
                            {
                                value +=
                                    parameters
                                        .output_bias_fp[col];
                            }

                            trace.H_agg[dst][col] =
                                apply_activation
                                    ? elu(value)
                                    : value;
                        }
                    }
                });

            if (profile != nullptr)
            {
                profile->activation_ms +=
                    std::chrono::duration<
                        double,
                        std::milli>(
                        Clock::now() - stage_start)
                        .count();
            }

            return trace;
        }

    }
    // 只读取 manifest 元数据，不触碰权重张量本体
    CheckpointBundleInfo inspect_checkpoint_bundle(const std::string &bundle_root)
    {
        const auto bundle_path = std::filesystem::path(bundle_root);
        if (!std::filesystem::exists(bundle_path))
        {
            throw std::runtime_error("checkpoint bundle path does not exist: " + bundle_path.string());
        }
        const auto manifest_path = bundle_path / "manifest.json";
        if (!std::filesystem::exists(manifest_path))
        {
            throw std::runtime_error("checkpoint bundle manifest not found: " + manifest_path.string());
        }
        const auto manifest = load_text_file(manifest_path);

        CheckpointBundleInfo info;
        info.bundle_root = bundle_root;
        info.family_schema_version = parse_required_string(manifest, "family_schema_version");
        info.output_average_rule = parse_required_string(manifest, "output_average_rule");
        info.model_arch_id = parse_required_string(manifest, "model_arch_id");
        info.model_param_id = parse_required_string(manifest, "model_param_id");
        info.quant_cfg_id = parse_required_string(manifest, "quant_cfg_id");
        info.static_table_id = parse_required_string(manifest, "static_table_id");
        info.degree_bound_id = parse_required_string(manifest, "degree_bound_id");
        info.layer_count = parse_required_size_t(manifest, "L");
        info.output_head_count = parse_required_size_t(manifest, "K_out");
        info.class_count = parse_required_size_t(manifest, "C");
        info.d_in_profile = parse_required_size_array(manifest, "d_in_profile");

        // 每层的形状摘要
        for (const auto &entry : parse_object_array_entries(manifest, "hidden_layers"))
        {
            info.hidden_layers.push_back(parse_checkpoint_layer_info(entry));
        }
        for (const auto &entry : parse_object_array_entries(manifest, "hidden_profile"))
        {
            info.hidden_profile.push_back(parse_checkpoint_layer_shape(entry));
        }
        // 每个头对应的张量名字
        for (const auto &entry : parse_object_array_entries(manifest, "hidden_head_specs"))
        {
            info.hidden_head_specs.push_back(parse_checkpoint_head_spec(entry));
        }
        for (const auto &entry : parse_object_array_entries(manifest, "output_head_specs"))
        {
            info.output_head_specs.push_back(parse_checkpoint_output_head_spec(entry));
        }
        info.has_output_attention_head = info.output_head_count > 0;
        return info;
    }

    // 这里检查的只是 formal 主线真正依赖的结构约束
    bool checkpoint_bundle_matches_formal_proof_shape(
        const CheckpointBundleInfo &info,
        std::string *reason)
    {
        std::vector<std::string> failures;
        if (info.layer_count < 2)
        {
            failures.push_back("L=" + std::to_string(info.layer_count) + " but checkpoint manifest must expose L >= 2");
        }
        if (info.family_schema_version.empty())
        {
            failures.push_back("checkpoint manifest must expose non-empty family_schema_version");
        }
        if (info.output_average_rule != "per_head_bias_then_arithmetic_mean")
        {
            failures.push_back(
                "checkpoint manifest output_average_rule=" + info.output_average_rule + " but expected per_head_bias_then_arithmetic_mean");
        }
        if (info.hidden_layers.empty())
        {
            failures.push_back("checkpoint manifest must expose non-empty hidden_layers");
        }
        if (info.hidden_profile.empty())
        {
            failures.push_back(
                "hidden_profile size=" + std::to_string(info.hidden_profile.size()) + " but the checkpoint manifest must expose a non-empty hidden_profile");
        }
        if (info.layer_count != info.hidden_layers.size() + 1)
        {
            failures.push_back(
                "L=" + std::to_string(info.layer_count) + " conflicts with hidden_layers size=" + std::to_string(info.hidden_layers.size()));
        }
        if (info.hidden_profile.size() != info.hidden_layers.size())
        {
            failures.push_back(
                "hidden_profile size=" + std::to_string(info.hidden_profile.size()) + " conflicts with hidden_layers size=" + std::to_string(info.hidden_layers.size()));
        }
        if (info.output_head_count == 0)
        {
            failures.push_back(
                "K_out=" + std::to_string(info.output_head_count) + " but checkpoint manifest must expose at least one output attention head");
        }
        if (info.output_head_specs.size() != info.output_head_count)
        {
            failures.push_back(
                "output_head_specs size=" + std::to_string(info.output_head_specs.size()) + " conflicts with K_out=" + std::to_string(info.output_head_count));
        }
        if (info.d_in_profile.size() != info.hidden_profile.size())
        {
            failures.push_back(
                "d_in_profile size=" + std::to_string(info.d_in_profile.size()) + " conflicts with hidden_profile size=" + std::to_string(info.hidden_profile.size()));
        }
        if (info.model_arch_id.empty())
        {
            failures.push_back("checkpoint manifest must expose non-empty model_arch_id");
        }
        if (info.model_param_id.empty())
        {
            failures.push_back("checkpoint manifest must expose non-empty model_param_id");
        }
        if (info.quant_cfg_id.empty())
        {
            failures.push_back("checkpoint manifest must expose non-empty quant_cfg_id");
        }
        // The manifest's global_head_index is a legacy serialization aid only.
        // Formal identity is always the pair (layer_index, local_head_index).
        // Validate that every layer/local pair occurs exactly once; never infer
        // a formal head from the flattened/global position.
        std::size_t flattened_hidden_heads = 0;
        for (std::size_t layer_index = 0; layer_index < info.hidden_layers.size(); ++layer_index)
        {
            const auto &layer = info.hidden_layers[layer_index];
            if (layer.layer_index != layer_index)
            {
                failures.push_back("hidden_layers must use contiguous layer_index values");
                break;
            }
            if (layer.input_dim == 0 || layer.shape.head_count == 0 || layer.shape.head_dim == 0)
            {
                failures.push_back(
                    "hidden_layers[" + std::to_string(layer_index) + "] dimensions must be non-zero");
            }
            if (layer_index < info.d_in_profile.size() && info.d_in_profile[layer_index] != layer.input_dim)
            {
                failures.push_back(
                    "hidden_layers[" + std::to_string(layer_index) + "].input_dim conflicts with d_in_profile");
            }
            if (layer_index < info.hidden_profile.size() && (info.hidden_profile[layer_index].head_count != layer.shape.head_count || info.hidden_profile[layer_index].head_dim != layer.shape.head_dim))
            {
                failures.push_back(
                    "hidden_layers[" + std::to_string(layer_index) + "] conflicts with hidden_profile");
            }
            if (layer_index != 0)
            {
                const auto &previous = info.hidden_layers[layer_index - 1];
                const auto expected_input =
                    previous.shape.head_count * previous.shape.head_dim;
                if (layer.input_dim != expected_input)
                {
                    failures.push_back(
                        "hidden_layers[" + std::to_string(layer_index) + "].input_dim conflicts with previous concat width");
                }
            }
            flattened_hidden_heads += layer.shape.head_count;

            std::vector<bool> seen_local_heads(layer.shape.head_count, false);
            std::size_t local_head_count = 0;
            for (const auto &head : info.hidden_head_specs)
            {
                if (head.layer_index != layer_index)
                {
                    continue;
                }
                if (head.local_head_index >= layer.shape.head_count
                    || seen_local_heads[head.local_head_index])
                {
                    failures.push_back(
                        "hidden_head_specs must use unique layer-local head indices");
                    break;
                }
                seen_local_heads[head.local_head_index] = true;
                ++local_head_count;
            }
            if (local_head_count != layer.shape.head_count
                || std::any_of(
                    seen_local_heads.begin(), seen_local_heads.end(),
                    [](bool seen) { return !seen; }))
            {
                failures.push_back(
                    "hidden_head_specs layer-local count conflicts with hidden_layers[" + std::to_string(layer_index) + "].head_count");
            }
        }
        if (flattened_hidden_heads != info.hidden_head_specs.size())
        {
            failures.push_back(
                "hidden_head_specs size=" + std::to_string(info.hidden_head_specs.size()) + " conflicts with hidden_layers head counts=" + std::to_string(flattened_hidden_heads));
        }
        for (std::size_t head_index = 0;
             head_index < info.output_head_specs.size(); ++head_index)
        {
            if (info.output_head_specs[head_index].head_index != head_index)
            {
                failures.push_back(
                    "output_head_specs must use contiguous head_index values");
                break;
            }
        }
        if (failures.empty())
        {
            return true;
        }
        if (reason != nullptr)
        {
            std::ostringstream stream;
            for (std::size_t i = 0; i < failures.size(); ++i)
            {
                if (i != 0)
                {
                    stream << "; ";
                }
                stream << failures[i];
            }
            *reason = stream.str();
        }
        return false;
    }

    // 正式 proving 主线的真实模型入口：manifest 校验通过后再加载权重。
    ModelParameters load_checkpoint_bundle_parameters(const std::string &bundle_root)
    {
        const auto manifest_info = inspect_checkpoint_bundle(bundle_root);
        std::string manifest_reason;
        if (!checkpoint_bundle_matches_formal_proof_shape(manifest_info, &manifest_reason))
        {
            throw std::runtime_error("checkpoint bundle family metadata is invalid: " + manifest_reason);
        }
        const auto tensor_path = std::filesystem::path(bundle_root) / "tensors.txt";
        if (!std::filesystem::exists(tensor_path))
        {
            throw std::runtime_error("checkpoint bundle tensor dump not found: " + tensor_path.string());
        }
        const auto tensors = load_text_tensors(tensor_path);

        ModelParameters out;
        out.has_real_multihead = true;
        out.model_arch_id = manifest_info.model_arch_id;
        out.model_param_id = manifest_info.model_param_id;
        out.quant_cfg_id = manifest_info.quant_cfg_id;
        out.static_table_id = manifest_info.static_table_id;
        out.degree_bound_id = manifest_info.degree_bound_id;
        out.L = manifest_info.layer_count;
        out.d_in_profile = manifest_info.d_in_profile;
        out.hidden_profile = manifest_info.hidden_profile;
        out.K_out = manifest_info.output_head_count;
        // Keep this flat vector solely for checkpoint/parity compatibility.
        // The formal model is built in hidden_layers and indexed by the
        // (layer, local_head) pair below.
        out.hidden_heads.reserve(manifest_info.hidden_head_specs.size());
        out.hidden_layers.reserve(manifest_info.hidden_layers.size());

        // Reconstruct each layer in canonical layer -> local-head order.  The
        // legacy global index is intentionally not consulted for identity.
        for (const auto &layer_info : manifest_info.hidden_layers)
        {
            HiddenLayerParameters layer;
            layer.layer_index = layer_info.layer_index;
            layer.input_dim = layer_info.input_dim;
            layer.shape = layer_info.shape;
            layer.heads.reserve(layer.shape.head_count);
            for (std::size_t local_head_index = 0;
                 local_head_index < layer.shape.head_count;
                 ++local_head_index)
            {
                const auto head_spec_it = std::find_if(
                    manifest_info.hidden_head_specs.begin(),
                    manifest_info.hidden_head_specs.end(),
                    [&](const CheckpointHeadSpec &candidate)
                    {
                        return candidate.layer_index == layer.layer_index
                            && candidate.local_head_index == local_head_index;
                    });
                if (head_spec_it == manifest_info.hidden_head_specs.end())
                {
                    throw std::runtime_error(
                        "checkpoint manifest is missing a layer/local hidden head pair");
                }
                const auto &head_spec = *head_spec_it;
                layer.heads.push_back(load_head_parameters(
                    tensors,
                    head_spec.seq_kernel,
                    head_spec.attn_dst_kernel,
                    head_spec.attn_dst_bias,
                    head_spec.attn_src_kernel,
                    head_spec.attn_src_bias,
                    head_spec.output_bias));
                out.hidden_heads.push_back(layer.heads.back());
            }
            if (layer.heads.size() != layer.shape.head_count)
            {
                throw std::runtime_error(
                    "checkpoint manifest hidden_head_specs do not match hidden_layers head_count for layer " + std::to_string(layer.layer_index));
            }
            out.hidden_layers.push_back(std::move(layer));
        }

        // 输出层同样要求索引连续读取，避免顺序混乱
        out.output_layer.head_count = manifest_info.output_head_count;
        out.output_layer.heads.reserve(manifest_info.output_head_specs.size());
        for (const auto &head_spec : manifest_info.output_head_specs)
        {
            if (head_spec.head_index != out.output_layer.heads.size())
            {
                throw std::runtime_error("checkpoint manifest output_head_specs must use contiguous head_index values");
            }
            out.output_layer.heads.push_back(load_head_parameters(
                tensors,
                head_spec.seq_kernel,
                head_spec.attn_dst_kernel,
                head_spec.attn_dst_bias,
                head_spec.attn_src_kernel,
                head_spec.attn_src_bias,
                head_spec.output_bias));
        }
        if (out.hidden_heads.empty())
        {
            throw std::runtime_error("checkpoint bundle does not contain hidden attention heads");
        }
        if (out.output_layer.heads.empty())
        {
            throw std::runtime_error("checkpoint bundle does not contain output attention heads");
        }

        out.C = manifest_info.class_count;
        out.output_layer.input_dim = out.hidden_layers.back().shape.head_count * out.hidden_layers.back().shape.head_dim;
        out.output_layer.output_dim = attention_head_output_width(out.output_layer.heads.front());
        if (out.C == 0)
        {
            out.C = out.output_layer.output_dim;
        }
        if (out.output_layer.output_dim != out.C)
        {
            throw std::runtime_error("checkpoint manifest C conflicts with output head output_dim");
        }
        return out;
    }

    // 返回注意力头输出宽度
    std::size_t attention_head_output_width(const AttentionHeadParameters &parameters)
    {
        if (!parameters.seq_kernel_fp.empty())
        {
            return parameters.seq_kernel_fp.front().size();
        }
        return parameters.output_bias_fp.size();
    }
    // 真正的主线前向实现
    MultiHeadForwardTrace forward_note_style(
        const FloatMatrix &features,
        const std::vector<data::Edge> &edges,
        const ModelParameters &parameters,
        ForwardProfile *profile)
    {
        if (!parameters.has_real_multihead)
        {
            throw std::runtime_error(
                "forward_note_style requires real multi-head checkpoint parameters");
        }

        MultiHeadForwardTrace trace;

        // H is part of the retained compatibility trace.
        trace.H = features;

        ForwardGraphLayout graph_layout;

        if (!features.empty())
        {
            // The graph is identical for every hidden/output head. Build its
            // destination-grouped execution layout exactly once.
            graph_layout =
                build_forward_graph_layout(
                    features.size(),
                    edges);
        }

        trace.hidden_layer_traces.reserve(
            parameters.hidden_layers.size());

        std::size_t formal_hidden_head_count = 0;
        for (const auto &layer : parameters.hidden_layers)
            formal_hidden_head_count += layer.heads.size();
        // Reserve the compatibility view from the nested family.  The view is
        // not consumed by formal trace, quotient, prover, verifier, or codec.
        trace.hidden_head_traces.reserve(formal_hidden_head_count);

        const FloatMatrix *layer_input =
            &trace.H;

        // 1. Hidden layers.
        for (const auto &layer :
             parameters.hidden_layers)
        {
            HiddenLayerForwardTrace
                layer_trace;

            // This retained copy is required by MultiHeadForwardTrace.
            // Use it directly for all heads instead of creating an additional
            // transient layer_input matrix.
            layer_trace.input =
                *layer_input;

            layer_trace.head_traces.resize(
                layer.heads.size());

            std::vector<HeadForwardProfile>
                head_profiles(
                    layer.heads.size());

            const auto execution_plan =
                forward_head_execution_plan(
                    layer.heads.size());

            parallel_for_forward_ranges(
                layer.heads.size(),
                1,
                execution_plan
                    .outer_head_workers,
                [&](std::size_t begin,
                    std::size_t end)
                {
                    for (std::size_t head_index =
                             begin;
                         head_index < end;
                         ++head_index)
                    {
                        layer_trace
                            .head_traces[head_index] =
                            attention_head_forward_impl(
                                layer_trace.input,
                                edges,
                                layer.heads[head_index],
                                false,
                                true,
                                &head_profiles[head_index],
                                &graph_layout,
                                execution_plan
                                    .inner_workers);
                    }
                });

            if (profile != nullptr)
            {
                for (const auto &head_profile :
                     head_profiles)
                {
                    profile
                        ->hidden_projection_ms +=
                        head_profile
                            .projection_ms;

                    profile
                        ->hidden_attention_ms +=
                        head_profile
                            .attention_ms;

                    profile
                        ->hidden_activation_ms +=
                        head_profile
                            .activation_ms;
                }
            }

            // Preserve the flattened compatibility view in canonical
            // layer/head order. No protocol consumer sees the execution order.
            for (const auto &head_trace :
                 layer_trace.head_traces)
            {
                trace.hidden_head_traces
                    .push_back(
                        head_trace);
            }

            if (profile != nullptr)
            {
                const auto concat_start =
                    std::chrono::
                        steady_clock::now();

                layer_trace.concat =
                    concatenate_head_outputs(
                        layer_trace
                            .head_traces,
                        forward_worker_budget());

                profile->hidden_concat_ms +=
                    std::chrono::duration<
                        double,
                        std::milli>(
                        std::chrono::
                            steady_clock::now() -
                        concat_start)
                        .count();
            }
            else
            {
                layer_trace.concat =
                    concatenate_head_outputs(
                        layer_trace
                            .head_traces,
                        forward_worker_budget());
            }

            trace.hidden_layer_traces
                .push_back(
                    std::move(
                        layer_trace));

            // hidden_layer_traces was reserved to its final size above, so
            // the stored concat address remains stable for the next layer.
            layer_input =
                &trace
                     .hidden_layer_traces
                     .back()
                     .concat;
        }

        if (!trace.hidden_layer_traces.empty())
        {
            // Compatibility field stores only the final hidden concat. The old
            // implementation copied every intermediate concat into this field
            // and immediately overwrote it on the next layer.
            trace.hidden_concat =
                trace
                    .hidden_layer_traces
                    .back()
                    .concat;
        }

        // 2. Output heads.
        trace.output_head_traces.resize(
            parameters.output_layer
                .heads.size());

        std::vector<HeadForwardProfile>
            output_profiles(
                parameters.output_layer
                    .heads.size());

        const auto output_execution_plan =
            forward_head_execution_plan(
                parameters.output_layer
                    .heads.size());

        parallel_for_forward_ranges(
            parameters.output_layer
                .heads.size(),
            1,
            output_execution_plan
                .outer_head_workers,
            [&](std::size_t begin,
                std::size_t end)
            {
                for (std::size_t head_index =
                         begin;
                     head_index < end;
                     ++head_index)
                {
                    trace.output_head_traces[head_index] =
                        attention_head_forward_impl(
                            *layer_input,
                            edges,
                            parameters
                                .output_layer
                                .heads[head_index],
                            true,
                            false,
                            &output_profiles[head_index],
                            &graph_layout,
                            output_execution_plan
                                .inner_workers);
                }
            });

        if (profile != nullptr)
        {
            for (const auto &output_profile :
                 output_profiles)
            {
                profile
                    ->output_projection_ms +=
                    output_profile
                        .projection_ms;

                profile
                    ->output_attention_ms +=
                    output_profile
                        .attention_ms;

                profile
                    ->output_activation_ms +=
                    output_profile
                        .activation_ms;
            }
        }

        // Preserve existing public compatibility views, but populate them
        // directly from the retained output-head traces. The former path first
        // copied every matrix into temporary output_pre_bias/output_values and
        // then copied output_values again into this field.
        trace.output_head_values.reserve(
            trace.output_head_traces.size());

        for (const auto &head_trace :
             trace.output_head_traces)
        {
            trace.output_head_values
                .push_back(
                    head_trace.H_agg);
        }

        trace.output_head_trace =
            trace.output_head_traces.front();

        // Same head order and same per-element arithmetic as before.
        trace.Y_lin =
            average_head_outputs(
                trace.output_head_traces,
                true,
                forward_worker_budget());

        trace.Y =
            average_head_outputs(
                trace.output_head_traces,
                false,
                forward_worker_budget());

        return trace;
    }

    bool supports_current_formal_proof_shape(const ModelParameters &parameters)
    {
        std::string reason;
        return parameters.has_real_multihead && hidden_family_dimension_chain_is_valid(parameters, &reason);
    }

    // 检查多层多头的维度链是不是前后对得上
    bool hidden_family_dimension_chain_is_valid(const ModelParameters &parameters, std::string *reason)
    {
        std::vector<std::string> failures;
        if (!parameters.has_real_multihead)
        {
            failures.push_back("formal family proof requires has_real_multihead=true");
        }
        if (parameters.L < 2)
        {
            failures.push_back("L must be >= 2");
        }
        if (parameters.hidden_layers.empty())
        {
            failures.push_back("hidden_layers must be non-empty");
        }
        if (parameters.hidden_profile.size() != parameters.hidden_layers.size())
        {
            failures.push_back("hidden_profile size conflicts with hidden_layers size");
        }
        if (!parameters.d_in_profile.empty() && parameters.d_in_profile.size() != parameters.hidden_layers.size())
        {
            failures.push_back("d_in_profile size conflicts with hidden_layers size");
        }
        std::size_t expected_input_dim = 0;
        for (std::size_t layer_index = 0; layer_index < parameters.hidden_layers.size(); ++layer_index)
        {
            const auto &layer = parameters.hidden_layers[layer_index];
            if (layer.layer_index != layer_index)
            {
                failures.push_back("hidden_layers must use contiguous layer_index values");
                break;
            }
            if (layer.shape.head_count == 0 || layer.shape.head_dim == 0)
            {
                failures.push_back("hidden layer shape must have non-zero head_count and head_dim");
            }
            if (layer.heads.size() != layer.shape.head_count)
            {
                failures.push_back(
                    "hidden layer " + std::to_string(layer_index) + " head_count conflicts with heads.size()");
            }
            const auto configured_input =
                !parameters.d_in_profile.empty() ? parameters.d_in_profile[layer_index] : layer.input_dim;
            if (layer.input_dim != configured_input)
            {
                failures.push_back(
                    "hidden layer " + std::to_string(layer_index) + " input_dim conflicts with d_in_profile");
            }

            // 第0层输入就是配置里给的输入宽度，后面每一层输入都应该等于上一层拼接后的总宽度。
            if (layer_index == 0)
            {
                expected_input_dim = configured_input;
            }
            else if (layer.input_dim != expected_input_dim)
            {
                failures.push_back(
                    "hidden layer " + std::to_string(layer_index) + " input_dim=" + std::to_string(layer.input_dim) + " conflicts with previous concat width=" + std::to_string(expected_input_dim));
            }

            // 每个头的参数宽度需要和这一层的输入输出对齐
            for (const auto &head : layer.heads)
            {
                if (head.seq_kernel_fp.size() != layer.input_dim)
                {
                    failures.push_back(
                        "hidden layer " + std::to_string(layer_index) + " seq kernel row count conflicts with layer input_dim");
                    break;
                }
                if (attention_head_output_width(head) != layer.shape.head_dim)
                {
                    failures.push_back(
                        "hidden layer " + std::to_string(layer_index) + " head output width conflicts with head_dim");
                    break;
                }
            }
            expected_input_dim = layer.shape.head_count * layer.shape.head_dim;
        }

        // 输出层也做同样的形状一致性检查
        if (parameters.output_layer.head_count == 0 || parameters.output_layer.heads.empty())
        {
            failures.push_back("output_layer must expose at least one output attention head");
        }
        if (parameters.output_layer.heads.size() != parameters.output_layer.head_count)
        {
            failures.push_back("output_layer head_count conflicts with heads.size()");
        }
        if (parameters.K_out != parameters.output_layer.head_count)
        {
            failures.push_back("K_out conflicts with output_layer head_count");
        }
        if (!parameters.hidden_layers.empty() && parameters.output_layer.input_dim != expected_input_dim)
        {
            failures.push_back(
                "output_layer input_dim=" + std::to_string(parameters.output_layer.input_dim) + " conflicts with final hidden concat width=" + std::to_string(expected_input_dim));
        }
        for (const auto &head : parameters.output_layer.heads)
        {
            if (head.seq_kernel_fp.size() != parameters.output_layer.input_dim)
            {
                failures.push_back("output head seq kernel row count conflicts with output_layer input_dim");
                break;
            }
            if (attention_head_output_width(head) != parameters.output_layer.output_dim)
            {
                failures.push_back("output head output width conflicts with output_layer output_dim");
                break;
            }
        }
        if (parameters.C != parameters.output_layer.output_dim)
        {
            failures.push_back("C conflicts with output_layer output_dim");
        }
        if (parameters.L != parameters.hidden_layers.size() + 1)
        {
            failures.push_back("L conflicts with hidden layer family size");
        }
        // This check covers only the retained checkpoint/parity flattening;
        // formal trace/proof code never indexes hidden_heads by this count.
        if (flattened_hidden_head_count(parameters) != parameters.hidden_heads.size())
        {
            failures.push_back("hidden_heads compatibility view must flatten all hidden layers");
        }
        if (reason != nullptr && !failures.empty())
        {
            std::ostringstream stream;
            for (std::size_t i = 0; i < failures.size(); ++i)
            {
                if (i != 0)
                {
                    stream << "; ";
                }
                stream << failures[i];
            }
            *reason = stream.str();
        }
        return failures.empty();
    }

    // Count the compatibility view; this is never used to identify a formal
    // hidden head, whose identity is always (layer, local_head).
    std::size_t flattened_hidden_head_count(const ModelParameters &parameters)
    {
        std::size_t count = 0;
        for (const auto &layer : parameters.hidden_layers)
        {
            count += layer.heads.size();
        }
        return count;
    }

    // 所有隐藏层里，最大的输入宽度
    std::size_t max_hidden_input_dim(const ModelParameters &parameters)
    {
        std::size_t value = 0;
        for (const auto &layer : parameters.hidden_layers)
        {
            value = std::max(value, layer.input_dim);
        }
        return value;
    }

    // 所有隐藏层里，单个头最大的输出宽度
    std::size_t max_hidden_head_dim(const ModelParameters &parameters)
    {
        std::size_t value = 0;
        for (const auto &layer : parameters.hidden_layers)
        {
            value = std::max(value, layer.shape.head_dim);
        }
        return value;
    }

    // 所有隐藏层里，拼接后的最大总宽度
    std::size_t max_hidden_concat_width(const ModelParameters &parameters)
    {
        std::size_t value = 0;
        for (const auto &layer : parameters.hidden_layers)
        {
            value = std::max(value, layer.shape.head_count * layer.shape.head_dim);
        }
        return value;
    }

}
