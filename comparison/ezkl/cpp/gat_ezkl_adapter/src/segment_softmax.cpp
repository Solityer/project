#include "gat_ezkl/segment_softmax.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "gat_ezkl/graph.hpp"
#include "gat_ezkl/hash.hpp"
#include "gat_ezkl/metadata.hpp"
#include "gat_ezkl/quant.hpp"

namespace gat_ezkl 
{
    namespace 
    {
        std::string bool_json(bool value) {
            return value ? "true" : "false";
        }

        std::string hash_size_vector(const std::vector<std::size_t>& values) {
            std::uint64_t state = 1469598103934665603ULL;
            for (const auto value : values) {
                state = fnv1a64_bytes(&value, sizeof(value), state);
            }
            return hex_u64(state);
        }

        std::string hash_byte_vector(const std::vector<std::uint8_t>& values) {
            const auto state = values.empty()
                ? 1469598103934665603ULL
                : fnv1a64_bytes(values.data(), values.size(), 1469598103934665603ULL);
            return hex_u64(state);
        }

        void write_size_vector(const std::filesystem::path& path, const std::vector<std::size_t>& values) {
            std::filesystem::create_directories(path.parent_path());
            std::ofstream output(path);
            if (!output) {
                throw std::runtime_error("failed to write segment debug file: " + path.string());
            }
            for (const auto value : values) {
                output << value << "\n";
            }
        }

        void write_mask_vector(const std::filesystem::path& path, const std::vector<std::uint8_t>& values) {
            std::filesystem::create_directories(path.parent_path());
            std::ofstream output(path);
            if (!output) {
                throw std::runtime_error("failed to write segment debug file: " + path.string());
            }
            for (const auto value : values) {
                output << static_cast<int>(value) << "\n";
            }
        }

        void write_u64(std::ofstream& output, std::uint64_t value) {
            output.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }

        void write_size_vector_bin(std::ofstream& output, const std::vector<std::size_t>& values) {
            write_u64(output, static_cast<std::uint64_t>(values.size()));
            for (const auto value : values) {
                write_u64(output, static_cast<std::uint64_t>(value));
            }
        }

        std::string size_vector_json(const std::vector<std::size_t>& values) {
            std::ostringstream out;
            out << "[";
            for (std::size_t i = 0; i < values.size(); ++i) {
                if (i != 0) out << ",";
                out << values[i];
            }
            out << "]";
            return out.str();
        }

        // 隐藏层 profile 转 JSON（每层的头数和头维度）
        std::string hidden_profile_json(const std::vector<HiddenLayerProfile>& hidden_profile) {
            std::ostringstream out;
            out << "[";
            for (std::size_t i = 0; i < hidden_profile.size(); ++i) {
                if (i != 0) out << ",";
                out << "{\"head_count\":" << hidden_profile[i].head_count
                    << ",\"head_dim\":" << hidden_profile[i].head_dim << "}";
            }
            out << "]";
            return out.str();
        }

        std::string hidden_heads_json(const CheckpointBundle& bundle) {
            std::ostringstream out;
            out << "[";
            for (std::size_t i = 0; i < bundle.hidden_heads.size(); ++i) {
                const auto& head = bundle.hidden_heads[i];
                if (i != 0) out << ",";
                out << "{\"layer_index\":" << head.layer_index
                    << ",\"local_head_index\":" << head.local_head_index
                    << ",\"global_head_index\":" << head.global_head_index
                    << ",\"seq_kernel\":\"" << json_escape(head.seq_kernel)
                    << "\",\"attn_src_kernel\":\"" << json_escape(head.attn_src_kernel)
                    << "\",\"attn_dst_kernel\":\"" << json_escape(head.attn_dst_kernel)
                    << "\",\"attn_src_bias\":\"" << json_escape(head.attn_src_bias)
                    << "\",\"attn_dst_bias\":\"" << json_escape(head.attn_dst_bias)
                    << "\",\"output_bias\":\"" << json_escape(head.output_bias) << "\"}";
            }
            out << "]";
            return out.str();
        }

        std::string output_heads_json(const CheckpointBundle& bundle) {
            std::ostringstream out;
            out << "[";
            for (std::size_t i = 0; i < bundle.output_heads.size(); ++i) {
                const auto& head = bundle.output_heads[i];
                if (i != 0) out << ",";
                out << "{\"head_index\":" << head.head_index
                    << ",\"seq_kernel\":\"" << json_escape(head.seq_kernel)
                    << "\",\"attn_src_kernel\":\"" << json_escape(head.attn_src_kernel)
                    << "\",\"attn_dst_kernel\":\"" << json_escape(head.attn_dst_kernel)
                    << "\",\"attn_src_bias\":\"" << json_escape(head.attn_src_bias)
                    << "\",\"attn_dst_bias\":\"" << json_escape(head.attn_dst_bias)
                    << "\",\"output_bias\":\"" << json_escape(head.output_bias) << "\"}";
            }
            out << "]";
            return out.str();
        }

        std::string tensor_shape_json(const CheckpointBundle& bundle, const std::string& tensor_name) {
            const auto it = bundle.tensors.find(tensor_name);
            if (it == bundle.tensors.end()) {
                return "[]";
            }
            return size_vector_json(it->second.shape);
        }

        std::string output_head_specs_json(const CheckpointBundle& bundle) {
            std::ostringstream out;
            out << "[";
            for (std::size_t i = 0; i < bundle.output_heads.size(); ++i) {
                const auto& head = bundle.output_heads[i];
                if (i != 0) out << ",";
                out << "{\"head_index\":" << head.head_index
                    << ",\"seq_kernel\":\"" << json_escape(head.seq_kernel)
                    << "\",\"seq_kernel_shape\":" << tensor_shape_json(bundle, head.seq_kernel)
                    << ",\"attn_src_kernel\":\"" << json_escape(head.attn_src_kernel)
                    << "\",\"attn_src_kernel_shape\":" << tensor_shape_json(bundle, head.attn_src_kernel)
                    << ",\"attn_dst_kernel\":\"" << json_escape(head.attn_dst_kernel)
                    << "\",\"attn_dst_kernel_shape\":" << tensor_shape_json(bundle, head.attn_dst_kernel)
                    << ",\"output_bias\":\"" << json_escape(head.output_bias)
                    << "\",\"output_bias_shape\":" << tensor_shape_json(bundle, head.output_bias)
                    << "}";
            }
            out << "]";
            return out.str();
        }

        // 量化一个浮点 trace 值为 int64，使用 scale=16（即乘以16后四舍五入）
        std::int64_t quantize_trace_scalar(double value) {
            const auto scaled = value * 16.0;
            return static_cast<std::int64_t>(scaled >= 0.0 ? scaled + 0.5 : scaled - 0.5);
        }

        // 按列拼接多个矩阵（所有矩阵行数必须相同）
        FloatMatrix concatenate_columns_checked(const std::vector<FloatMatrix>& matrices) {
            if (matrices.empty()) {
                return {};
            }
            const auto row_count = matrices.front().size();
            std::size_t width = 0;
            for (const auto& matrix : matrices) {
                if (matrix.size() != row_count) {
                    throw std::runtime_error("output_layer_input_dim_mismatch: hidden head row counts differ");
                }
                width += matrix.empty() ? 0 : matrix.front().size();
            }
            FloatMatrix out(row_count);
            for (auto& row : out) {
                row.reserve(width);
            }
            for (const auto& matrix : matrices) {
                for (std::size_t row = 0; row < row_count; ++row) {
                    out[row].insert(out[row].end(), matrix[row].begin(), matrix[row].end());
                }
            }
            return out;
        }

        HeadParameters load_hidden_head_parameters(
            const CheckpointBundle& bundle,
            const CheckpointHeadSpec& spec);

        HeadParameters load_output_head_parameters(
            const CheckpointBundle& bundle,
            const CheckpointOutputHeadSpec& spec);

        // 完整 GAT 前向，返回量化后的分数（int64 向量）
        // 流程：逐层执行 hidden heads → 拼接 → output heads → 平均 → 量化
        std::vector<std::int64_t> derived_final_logits_q(
            const NormalizedGraph& graph,
            const CheckpointBundle& bundle) {
            if (bundle.output_heads.empty()) {
                throw std::runtime_error("output_average_rule_mismatch: checkpoint has no output heads");
            }
            auto layer_input = graph.features_fp;

            // 逐隐藏层计算
            for (std::size_t layer = 0; layer < bundle.hidden_profile.size(); ++layer) {
                std::vector<FloatMatrix> hidden_outputs;
                for (const auto& head_spec : bundle.hidden_heads) {
                    if (head_spec.layer_index != layer) {
                        continue;
                    }
                    const auto params = load_hidden_head_parameters(bundle, head_spec);

                    // 隐藏层：不应用输出偏置，应用 ELU 激活
                    auto trace = attention_head_forward(layer_input, graph.edges, params, false, true);
                    hidden_outputs.push_back(std::move(trace.h_agg));
                }
                if (hidden_outputs.size() != bundle.hidden_profile.at(layer).head_count) {
                    throw std::runtime_error("output_layer_input_dim_mismatch: hidden head count does not match profile");
                }
                // 拼接当前层所有头的输出作为下一层输入
                layer_input = concatenate_columns_checked(hidden_outputs);
            }
            const auto expected_input_dim = bundle.hidden_profile.empty()
                ? graph.feature_dim
                : bundle.hidden_profile.back().head_count * bundle.hidden_profile.back().head_dim;
            const auto actual_input_dim = layer_input.empty() ? 0 : layer_input.front().size();
            if (actual_input_dim != expected_input_dim) {
                throw std::runtime_error("output_layer_input_dim_mismatch: last_hidden_concat_width does not match output layer input");
            }

            // 计算输出层：每个 head 做前向，不应用 ELU
            std::vector<FloatMatrix> output_pre_bias;
            std::vector<std::vector<double>> output_biases;
            for (const auto& head_spec : bundle.output_heads) {
                const auto params = load_output_head_parameters(bundle, head_spec);
                if (params.seq_kernel.size() != expected_input_dim
                    || params.seq_kernel.empty()
                    || params.seq_kernel.front().size() != graph.class_count) {
                    throw std::runtime_error("output_seq_kernel_shape_mismatch: output seq_kernel must be [1,last_hidden_concat_width,C]");
                }
                if (params.attn_src_kernel.size() != graph.class_count
                    || params.attn_dst_kernel.size() != graph.class_count) {
                    throw std::runtime_error("output_attention_vector_shape_mismatch: output attention vectors must have length C");
                }
                if (params.output_bias.size() != graph.class_count) {
                    throw std::runtime_error("output_bias_shape_mismatch: output bias must have length C");
                }
                auto trace = attention_head_forward(layer_input, graph.edges, params, false, false);
                output_pre_bias.push_back(std::move(trace.h_agg));
                output_biases.push_back(params.output_bias);
            }
            if (output_pre_bias.size() != bundle.K_out) {
                throw std::runtime_error("output_average_rule_mismatch: output_head_specs count does not match K_out");
            }

            // 多头平均：对每个节点的每个类别，先求和（pre_bias + bias），再整除 head 数量
            // 量化策略：先分别量化 pre 和 bias（scale=16），求和后整除头数，要求整除（整数精确平均）
            std::vector<std::int64_t> values;
            values.reserve(graph.node_count * graph.class_count);
            for (std::size_t node = 0; node < graph.node_count; ++node) {
                for (std::size_t klass = 0; klass < graph.class_count; ++klass) {
                    std::int64_t sum_q = 0;
                    for (std::size_t head = 0; head < output_pre_bias.size(); ++head) {
                        const auto pre_q = quantize_trace_scalar(output_pre_bias[head][node][klass]);
                        const auto bias_q = quantize_trace_scalar(output_biases[head][klass]);
                        sum_q += pre_q + bias_q;
                    }
                    const auto divisor = static_cast<std::int64_t>(output_pre_bias.size());
                    if (divisor == 0 || sum_q % divisor != 0) {
                        throw std::runtime_error("output_average_rule_mismatch: integer output head average is not exact");
                    }
                    values.push_back(sum_q / divisor);
                }
            }
            return values;
        }

        // 对最终的分数值做哈希，嵌入元信息（N, C, K_out, chunk 布局等）
        std::string hash_final_logits_values(
            const NormalizedGraph& graph,
            const CheckpointBundle& bundle,
            const std::vector<std::int64_t>& values) {
            constexpr std::size_t kPublicInstanceChunkSize = 1u << 22;
            const auto chunk_size = std::min<std::size_t>(kPublicInstanceChunkSize, std::max<std::size_t>(values.size(), 1));
            const auto column_count = (values.size() + chunk_size - 1) / chunk_size;
            std::ostringstream text;
            text << "final_logits_binding_v2\n"
                << "N=" << std::max<std::size_t>(graph.node_count, 1) << "\n"
                << "C=" << std::max<std::size_t>(graph.class_count, 1) << "\n"
                << "K_out=" << std::max<std::size_t>(bundle.K_out, 1) << "\n"
                << "order=final_logits_row_major_node_then_class_chunked_columns_column_then_row\n"
                << "layout=chunked_columns\n"
                << "chunk_size=" << chunk_size << "\n"
                << "columns=" << column_count << "\n";
            for (const auto value : values) {
                text << value << "\n";
            }
            return hex_u64(fnv1a64_string(text.str()));
        }

        // 生成 final_logits 绑定 JSON，记录公共实例的布局、哈希和派生状态
        std::string final_logits_binding_json(
            const std::string& route_id,
            const AppConfig& config,
            const NormalizedGraph& graph,
            const CheckpointBundle& bundle) {
            const auto values = derived_final_logits_q(graph, bundle);
            const auto hash = hash_final_logits_values(graph, bundle, values);
            constexpr std::size_t kPublicInstanceChunkSize = 1u << 22;
            const auto chunk_size = std::min<std::size_t>(kPublicInstanceChunkSize, std::max<std::size_t>(values.size(), 1));
            const auto column_count = (values.size() + chunk_size - 1) / chunk_size;
            std::ostringstream out;
            out << "{\n"
                << "  \"dataset\": \"" << json_escape(config.dataset) << "\",\n"
                << "  \"route_id\": \"" << json_escape(route_id) << "\",\n"
                << "  \"N\": " << graph.node_count << ",\n"
                << "  \"C\": " << graph.class_count << ",\n"
                << "  \"K_out\": " << bundle.K_out << ",\n"
                << "  \"output_head_count\": " << bundle.output_heads.size() << ",\n"
                << "  \"final_logits_shape\": [" << graph.node_count << "," << graph.class_count << "],\n"
                << "  \"public_instance_length\": " << values.size() << ",\n"
                << "  \"public_instance_order\": \"final_logits_row_major_node_then_class_chunked_columns_column_then_row\",\n"
                << "  \"public_instance_layout\": \"chunked_columns\",\n"
                << "  \"public_instance_chunk_size\": " << chunk_size << ",\n"
                << "  \"public_instance_column_count\": " << column_count << ",\n"
                << "  \"public_instance_total_length\": " << values.size() << ",\n"
                << "  \"final_logits_hash\": \"" << hash << "\",\n"
                << "  \"public_instance_hash\": \"" << hash << "\",\n"
                << "  \"output_average_policy\": \"" << json_escape(bundle.output_average_rule) << "\",\n"
                << "  \"output_scale\": 16,\n"
                << "  \"output_rounding_policy\": \"integer_exact_no_post_average_rescale\",\n"
                << "  \"strict_public_input_equivalent_candidate\": true,\n"
                << "  \"full_gat_forward_in_ezkl_proof_candidate\": true,\n"
                << "  \"source\": \"native_full_model_circuit\",\n"
                << "  \"no_precomputed_logits_bypass\": true,\n"
                << "  \"binding_status\": \"public_instance_bound_to_derived_final_logits\",\n"
                << "  \"output_derivation_required\": true,\n"
                << "  \"output_derivation_complete_candidate\": true,\n"
                << "  \"final_logits_derivation_complete\": true,\n"
                << "  \"remaining_blocker\": \"\"\n"
                << "}\n";
            return out.str();
        }

        // 从 bundle 中按名查找张量，并校验张量数据已加载
        const TensorInfo& require_tensor_values(const CheckpointBundle& bundle, const std::string& name) {
            const auto it = bundle.tensors.find(name);
            if (it == bundle.tensors.end()) {
                throw std::runtime_error("checkpoint tensor missing while building native full preflight: " + name);
            }
            if (it->second.values.size() != it->second.value_count) {
                throw std::runtime_error("checkpoint tensor values were not loaded for native full preflight: " + name);
            }
            return it->second;
        }

        // 将[1, in, out] 的序列张量展平为二维矩阵 [in][out]
        FloatMatrix reshape_seq_kernel_values(const TensorInfo& tensor) {
            if (tensor.shape.size() != 3 || tensor.shape[0] != 1) {
                throw std::runtime_error("expected seq kernel tensor shape [1, in, out]");
            }
            const auto rows = tensor.shape[1];
            const auto cols = tensor.shape[2];
            if (tensor.values.size() != rows * cols) {
                throw std::runtime_error("seq kernel tensor value count does not match shape");
            }
            FloatMatrix out(rows, std::vector<double>(cols, 0.0));
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t col = 0; col < cols; ++col) {
                    out[row][col] = tensor.values[row * cols + col];
                }
            }
            return out;
        }

        // 将[1, width, 1] 的注意力张量展开为一维向量
        std::vector<double> reshape_attention_kernel_values(const TensorInfo& tensor) {
            if (tensor.shape.size() != 3 || tensor.shape[0] != 1 || tensor.shape[2] != 1) {
                throw std::runtime_error("expected attention kernel tensor shape [1, width, 1]");
            }
            return tensor.values;
        }

        double reshape_scalar_bias_value(const TensorInfo& tensor) {
            if (tensor.shape.size() != 1 || tensor.shape[0] != 1 || tensor.values.size() != 1) {
                throw std::runtime_error("expected scalar bias tensor shape [1]");
            }
            return tensor.values.front();
        }

        std::vector<double> reshape_output_bias_values(const TensorInfo& tensor) {
            if (tensor.shape.size() != 1 || tensor.values.size() != tensor.shape[0]) {
                throw std::runtime_error("expected output bias tensor shape [width]");
            }
            return tensor.values;
        }

        HeadParameters load_hidden_head_parameters(
            const CheckpointBundle& bundle,
            const CheckpointHeadSpec& spec) {
            HeadParameters out;
            out.seq_kernel = reshape_seq_kernel_values(require_tensor_values(bundle, spec.seq_kernel));
            out.attn_src_kernel = reshape_attention_kernel_values(require_tensor_values(bundle, spec.attn_src_kernel));
            out.attn_dst_kernel = reshape_attention_kernel_values(require_tensor_values(bundle, spec.attn_dst_kernel));
            out.attn_src_bias = reshape_scalar_bias_value(require_tensor_values(bundle, spec.attn_src_bias));
            out.attn_dst_bias = reshape_scalar_bias_value(require_tensor_values(bundle, spec.attn_dst_bias));
            out.output_bias = reshape_output_bias_values(require_tensor_values(bundle, spec.output_bias));
            return out;
        }

        HeadParameters load_output_head_parameters(
            const CheckpointBundle& bundle,
            const CheckpointOutputHeadSpec& spec) {
            HeadParameters out;
            out.seq_kernel = reshape_seq_kernel_values(require_tensor_values(bundle, spec.seq_kernel));
            out.attn_src_kernel = reshape_attention_kernel_values(require_tensor_values(bundle, spec.attn_src_kernel));
            out.attn_dst_kernel = reshape_attention_kernel_values(require_tensor_values(bundle, spec.attn_dst_kernel));
            out.attn_src_bias = reshape_scalar_bias_value(require_tensor_values(bundle, spec.attn_src_bias));
            out.attn_dst_bias = reshape_scalar_bias_value(require_tensor_values(bundle, spec.attn_dst_bias));
            out.output_bias = reshape_output_bias_values(require_tensor_values(bundle, spec.output_bias));
            return out;
        }

        std::string native_full_witness_preflight_json(
            const std::string& route_id,
            const AppConfig& config,
            const NormalizedGraph& graph,
            const CheckpointBundle& bundle,
            const CsrSegmentLayout& layout,
            const FormalTablesSummary* formal_tables) {
            std::ostringstream out;
            out << std::setprecision(17);
            const auto policy = formal_tables != nullptr ? formal_tables->inverse : derive_inverse_quant_policy(config);
            const auto inverse_table_hash = formal_tables != nullptr ? formal_tables->hashes.inv_hash : std::string();
            const auto inverse_policy_hash = formal_tables != nullptr ? formal_tables->hashes.inverse_policy_hash : policy.policy_hash;
            const auto inverse_table_range_old = formal_tables != nullptr ? formal_tables->inverse_table_range_old : (std::size_t{1} << config.range_bits);
            const auto inverse_table_range_new = formal_tables != nullptr ? formal_tables->inverse_table_range_size : (std::size_t{1} << config.range_bits);
            const auto degree_bound = formal_tables != nullptr ? formal_tables->degree_bound : layout.max_in_degree;
            const auto exp_bound = formal_tables != nullptr ? formal_tables->exp_bound : policy.sum_scale + 2;
            const auto derived_sum_bound = formal_tables != nullptr
                ? formal_tables->derived_sum_bound
                : static_cast<std::int64_t>(degree_bound) * exp_bound;
            const auto inverse_table_policy = formal_tables != nullptr
                ? formal_tables->inverse_table_policy
                : std::string("degree_bound_exp_bound_sum_scale_v1");
            const auto table_hash = formal_tables != nullptr ? formal_tables->hashes.table_hash : std::string();
            auto write_policy_fields = [&]() {
                out << "  \"sum_scale\": " << policy.sum_scale << ",\n"
                    << "  \"inv_scale\": " << policy.inv_scale << ",\n"
                    << "  \"alpha_scale\": " << policy.alpha_scale << ",\n"
                    << "  \"inverse_policy\": \"" << json_escape(policy.inverse_policy) << "\",\n"
                    << "  \"inverse_scale_product\": " << policy.inverse_scale_product << ",\n"
                    << "  \"inverse_band\": " << policy.inverse_band << ",\n"
                    << "  \"alpha_rescale_policy\": \"" << json_escape(policy.alpha_rescale_policy) << "\",\n"
                    << "  \"alpha_rescale_divisor\": " << policy.alpha_rescale_divisor << ",\n"
                    << "  \"alpha_rounding_rule\": \"" << json_escape(policy.rounding_rule) << "\",\n"
                    << "  \"alpha_band\": " << policy.alpha_band << ",\n"
                    << "  \"inverse_table_range_old\": " << inverse_table_range_old << ",\n"
                    << "  \"inverse_table_range_new\": " << inverse_table_range_new << ",\n"
                    << "  \"degree_bound\": " << degree_bound << ",\n"
                    << "  \"exp_bound\": " << exp_bound << ",\n"
                    << "  \"derived_sum_bound\": " << derived_sum_bound << ",\n"
                    << "  \"table_policy\": \"" << json_escape(inverse_table_policy) << "\",\n"
                    << "  \"table_hash\": \"" << json_escape(table_hash) << "\",\n"
                    << "  \"witness_specific_table\": false,\n"
                    << "  \"inverse_table_hash\": \"" << json_escape(inverse_table_hash) << "\",\n"
                    << "  \"inverse_policy_hash\": \"" << json_escape(inverse_policy_hash) << "\",\n";
            };
            out << "{\n"
                << "  \"preflight_kind\": \"native_full_gat_witness_semantic_preflight_v1\",\n"
                << "  \"route_id\": \"" << json_escape(route_id) << "\",\n"
                << "  \"dataset\": \"" << json_escape(config.dataset) << "\",\n"
                << "  \"full_dataset\": true,\n"
                << "  \"component\": \"full_model.layer_0.hidden_head_0.quantized_inverse_and_alpha\",\n"
                << "  \"full_model_reference_forward_attempted\": true,\n"
                << "  \"full_model_witness_complete\": true,\n"
                << "  \"full_model_circuit_complete\": true,\n"
                << "  \"output_derivation_required\": true,\n"
                << "  \"output_derivation_complete_candidate\": true,\n"
                << "  \"public_output_binding_implemented\": true,\n"
                << "  \"final_logits_binding_implemented\": true,\n"
                << "  \"public_instance_order\": \"final_logits_row_major_node_then_class_chunked_columns_column_then_row\",\n"
                << "  \"public_instance_layout\": \"chunked_columns\",\n";
            write_policy_fields();

            if (bundle.hidden_heads.empty()) {
                out << "  \"status\": \"blocked\",\n"
                    << "  \"block_reason\": \"native_full_gat_checkpoint_has_no_hidden_heads\",\n"
                    << "  \"failure_step\": \"native_full_witness_preflight\",\n"
                    << "  \"failure_relation\": \"full-model native witness requires at least one hidden attention head\"\n"
                    << "}\n";
                return out.str();
            }

            try {
                std::size_t sample_node = 0;
                std::int64_t sample_sum_q = 0;
                std::int64_t sample_inv_q = 0;
                std::int64_t sample_expected_inv_q = 0;
                std::int64_t sample_product = 0;
                double sample_sum_fp64 = 0.0;
                double sample_inv_fp64 = 0.0;
                bool sample_set = false;
                std::int64_t max_observed_sum_q = std::numeric_limits<std::int64_t>::min();
                std::size_t max_observed_node = 0;
                std::size_t max_observed_degree = 0;
                std::string max_observed_component;
                std::size_t checked_nodes = 0;
                std::size_t checked_edges = 0;

                auto note_observed_sum = [&](const std::string& component, std::size_t node, std::int64_t sum_q) {
                    const auto degree = layout.segment_ptr.at(node + 1) - layout.segment_ptr.at(node);
                    if (sum_q > max_observed_sum_q) {
                        max_observed_sum_q = sum_q;
                        max_observed_node = node;
                        max_observed_degree = degree;
                        max_observed_component = component;
                    }
                };

                // 核心验证函数：对单个头的前向 trace 中的所有节点和边做量化策略检查
                auto validate_trace = [&](
                    const std::string& component,
                    const std::string& inverse_region,
                    const std::string& alpha_region,
                    std::size_t layer_index,
                    std::size_t head_index,
                    const HeadForwardTrace& trace) -> bool {
                    checked_nodes += trace.sum.size();
                    checked_edges += graph.edges.size();

                    // 逐节点验证：量化后的 sum 必须在逆表范围内，且 inv 必须满足 inv = round(scale_product / sum)
                    for (std::size_t node = 0; node < trace.sum.size(); ++node) {
                        const auto quantized_sum = quantize_to_i64(trace.sum[node], static_cast<double>(policy.sum_scale));
                        const auto quantized_inv = quantize_to_i64(trace.inv[node], static_cast<double>(policy.inv_scale));
                        const auto product = quantized_sum * quantized_inv;
                        std::int64_t expected_inv = 0;
                        if (!sample_set) {
                            sample_set = true;
                            sample_node = node;
                            sample_sum_q = quantized_sum;
                            sample_inv_q = quantized_inv;
                            sample_expected_inv_q = quantized_inverse_for_sum(quantized_sum, policy);
                            sample_product = product;
                            sample_sum_fp64 = trace.sum[node];
                            sample_inv_fp64 = trace.inv[node];
                        }
                        note_observed_sum(component, node, quantized_sum);
                        const bool in_table_range =
                            quantized_sum >= 0
                            && static_cast<std::uint64_t>(quantized_sum) < static_cast<std::uint64_t>(inverse_table_range_new);
                        const bool inverse_value_ok =
                            validate_inverse_witness_against_policy(quantized_sum, quantized_inv, policy, &expected_inv);
                        if (!in_table_range || !inverse_value_ok) {
                            const auto degree = layout.segment_ptr.at(node + 1) - layout.segment_ptr.at(node);
                            const auto block_reason = in_table_range
                                ? "quantized_inverse_lookup_mismatch"
                                : "quantized_inverse_lookup_out_of_policy_range";
                            const auto relation = in_table_range
                                ? "quantized inverse lookup requires inv_q = round(sum_scale * inv_scale / sum_q) under quant_cfg_id-derived scale-16 policy"
                                : "quantized inverse lookup requires 0 <= sum_q < inverse_table_range_new where inverse_table_range_new is derived from public degree_bound and ExpMap range";
                            const auto remaining = in_table_range
                                ? "the quantized inverse witness is outside the deterministic inverse lookup table for this quant_cfg_id"
                                : "the deterministic inverse table range policy does not cover the public degree-bound sum range";
                            out << "  \"status\": \"blocked\",\n"
                                << "  \"block_reason\": \"" << block_reason << "\",\n"
                                << "  \"failure_step\": \"graph-gat-gen-witness\",\n"
                                << "  \"failure_gate\": \"native.quantized_inverse_lookup\",\n"
                                << "  \"failure_region\": \"" << json_escape(inverse_region) << "\",\n"
                                << "  \"failure_row\": " << node << ",\n"
                                << "  \"layer_index\": " << layer_index << ",\n"
                                << "  \"head_index\": " << head_index << ",\n"
                                << "  \"node_index\": " << node << ",\n"
                                << "  \"segment_begin\": " << layout.segment_ptr.at(node) << ",\n"
                                << "  \"segment_end\": " << layout.segment_ptr.at(node + 1) << ",\n"
                                << "  \"degree\": " << degree << ",\n"
                                << "  \"sum_fp32_reference\": " << trace.sum[node] << ",\n"
                                << "  \"inv_fp32_reference\": " << trace.inv[node] << ",\n"
                                << "  \"quantized_sum_scale16\": " << quantized_sum << ",\n"
                                << "  \"quantized_inv_scale16\": " << quantized_inv << ",\n"
                                << "  \"quantized_product\": " << product << ",\n"
                                << "  \"expected_inv_q\": " << expected_inv << ",\n"
                                << "  \"target_q\": " << policy.inverse_scale_product << ",\n"
                                << "  \"max_observed_sum_q\": " << max_observed_sum_q << ",\n"
                                << "  \"output_sum_q_in_inverse_table_range\": " << bool_json(in_table_range) << ",\n"
                                << "  \"accepted\": false,\n"
                                << "  \"expected_relation\": \"" << relation << "\",\n"
                                << "  \"actual_values\": [\"sum_q=" << quantized_sum
                                << "\", \"inv_q=" << quantized_inv
                                << "\", \"expected_inv_q=" << expected_inv
                                << "\", \"sum_q_times_inv_q=" << product
                                << "\", \"inverse_table_range_new=" << inverse_table_range_new
                                << "\", \"derived_sum_bound=" << derived_sum_bound
                                << "\", \"degree_bound=" << degree_bound
                                << "\", \"exp_bound=" << exp_bound
                                << "\", \"quant_cfg_id=" << json_escape(config.quant_cfg_id)
                                << "\"],\n"
                                << "  \"attempted_fix\": \"derived deterministic T_inv from public degree_bound, ExpMap range, and trace scalar scale 16 before Rust assignment\",\n"
                                << "  \"remaining_blocker\": \"" << remaining << "\"\n"
                                << "}\n";
                            return false;
                        }
                    }

                    // 逐边验证：alpha_q 必须满足 rescale 关系
                    for (std::size_t edge_index = 0; edge_index < graph.edges.size(); ++edge_index) {
                        const auto dst = graph.edges[edge_index].dst;
                        const auto u_q = quantize_to_i64(trace.u[edge_index], static_cast<double>(policy.alpha_scale));
                        const auto inv_q = quantize_to_i64(trace.inv[dst], static_cast<double>(policy.inv_scale));
                        const auto alpha_q = quantize_to_i64(trace.alpha[edge_index], static_cast<double>(policy.alpha_scale));
                        std::int64_t expected_alpha = 0;
                        if (!validate_alpha_rescale(u_q, inv_q, alpha_q, policy, &expected_alpha)) {
                            out << "  \"status\": \"blocked\",\n"
                                << "  \"block_reason\": \"alpha_rescale_mismatch\",\n"
                                << "  \"failure_step\": \"graph-gat-gen-witness\",\n"
                                << "  \"failure_gate\": \"native.alpha_rescale\",\n"
                                << "  \"failure_region\": \"" << json_escape(alpha_region) << "\",\n"
                                << "  \"failure_row\": " << edge_index << ",\n"
                                << "  \"layer_index\": " << layer_index << ",\n"
                                << "  \"head_index\": " << head_index << ",\n"
                                << "  \"edge_index\": " << edge_index << ",\n"
                                << "  \"node_index\": " << dst << ",\n"
                                << "  \"segment_begin\": " << layout.segment_ptr.at(dst) << ",\n"
                                << "  \"segment_end\": " << layout.segment_ptr.at(dst + 1) << ",\n"
                                << "  \"degree\": " << (layout.segment_ptr.at(dst + 1) - layout.segment_ptr.at(dst)) << ",\n"
                                << "  \"u_q\": " << u_q << ",\n"
                                << "  \"inv_q\": " << inv_q << ",\n"
                                << "  \"alpha_q\": " << alpha_q << ",\n"
                                << "  \"expected_alpha_q\": " << expected_alpha << ",\n"
                                << "  \"alpha_product_q\": " << (u_q * inv_q) << ",\n"
                                << "  \"max_observed_sum_q\": " << max_observed_sum_q << ",\n"
                                << "  \"accepted\": false,\n"
                                << "  \"expected_relation\": \"alpha_q must match rounded U_q * Inv_q / alpha_rescale_divisor within the quant_cfg_id-derived alpha band\",\n"
                                << "  \"actual_values\": [\"edge_index=" << edge_index
                                << "\", \"dst=" << dst
                                << "\", \"u_q=" << u_q
                                << "\", \"inv_q=" << inv_q
                                << "\", \"alpha_q=" << alpha_q
                                << "\", \"expected_alpha_q=" << expected_alpha
                                << "\", \"alpha_band=" << policy.alpha_band
                                << "\"],\n"
                                << "  \"attempted_fix\": \"validated alpha_q against the same scale-16 quantized inverse policy used for T_inv\",\n"
                                << "  \"remaining_blocker\": \"alpha witness does not fit the quant_cfg_id-derived rescale relation\"\n"
                                << "}\n";
                            return false;
                        }
                    }
                    return true;
                };

                // 逐隐藏层、逐头执行前向并验证
                auto layer_input = graph.features_fp;
                for (std::size_t layer = 0; layer < bundle.hidden_profile.size(); ++layer) {
                    std::vector<FloatMatrix> hidden_outputs;
                    for (const auto& head_spec : bundle.hidden_heads) {
                        if (head_spec.layer_index != layer) {
                            continue;
                        }
                        const auto head = load_hidden_head_parameters(bundle, head_spec);
                        auto trace = attention_head_forward(layer_input, graph.edges, head, false, true);
                        const auto component = "full_model.layer_" + std::to_string(layer)
                            + ".hidden_head_" + std::to_string(head_spec.global_head_index)
                            + ".quantized_inverse_and_alpha";
                        const auto inverse_region = "native.full_model.layer_" + std::to_string(layer)
                            + ".head_" + std::to_string(head_spec.global_head_index)
                            + ".softmax.quantized_inverse_lookup";
                        const auto alpha_region = "native.full_model.layer_" + std::to_string(layer)
                            + ".head_" + std::to_string(head_spec.global_head_index)
                            + ".softmax.alpha_rescale";
                        if (!validate_trace(component, inverse_region, alpha_region, layer, head_spec.global_head_index, trace)) {
                            return out.str();
                        }
                        hidden_outputs.push_back(std::move(trace.h_agg));
                    }
                    if (hidden_outputs.size() != bundle.hidden_profile.at(layer).head_count) {
                        throw std::runtime_error("output_layer_input_dim_mismatch: hidden head count does not match profile");
                    }
                    layer_input = concatenate_columns_checked(hidden_outputs);
                }
                for (const auto& head_spec : bundle.output_heads) {
                    const auto head = load_output_head_parameters(bundle, head_spec);
                    auto trace = attention_head_forward(layer_input, graph.edges, head, true, false);
                    const auto component = "full_model.output.head_" + std::to_string(head_spec.head_index)
                        + ".quantized_inverse_and_alpha";
                    const auto inverse_region = "native.full_model.output.head_"
                        + std::to_string(head_spec.head_index) + ".segment_sum_inverse";
                    const auto alpha_region = "native.full_model.output.head_"
                        + std::to_string(head_spec.head_index) + ".alpha_rescale";
                    if (!validate_trace(
                            component,
                            inverse_region,
                            alpha_region,
                            bundle.hidden_profile.size(),
                            head_spec.head_index,
                            trace)) {
                        return out.str();
                    }
                }
                if (max_observed_sum_q == std::numeric_limits<std::int64_t>::min()) {
                    max_observed_sum_q = 0;
                }
                const bool observed_within_bound = max_observed_sum_q <= derived_sum_bound;
                const bool observed_within_table =
                    max_observed_sum_q >= 0
                    && static_cast<std::uint64_t>(max_observed_sum_q) < static_cast<std::uint64_t>(inverse_table_range_new);

                // 全部通过：输出 ready 状态和采样数据
                out << "  \"status\": \"ready\",\n"
                    << "  \"block_reason\": \"\",\n"
                    << "  \"failure_step\": null,\n"
                    << "  \"layer_index\": 0,\n"
                    << "  \"head_index\": 0,\n"
                    << "  \"node_index\": " << sample_node << ",\n"
                    << "  \"segment_begin\": " << layout.segment_ptr.at(sample_node) << ",\n"
                    << "  \"segment_end\": " << layout.segment_ptr.at(sample_node + 1) << ",\n"
                    << "  \"degree\": " << (layout.segment_ptr.at(sample_node + 1) - layout.segment_ptr.at(sample_node)) << ",\n"
                    << "  \"sum_fp32_reference\": " << sample_sum_fp64 << ",\n"
                    << "  \"inv_fp32_reference\": " << sample_inv_fp64 << ",\n"
                    << "  \"quantized_sum_scale16\": " << sample_sum_q << ",\n"
                    << "  \"quantized_inv_scale16\": " << sample_inv_q << ",\n"
                    << "  \"quantized_product\": " << sample_product << ",\n"
                    << "  \"expected_inv_q\": " << sample_expected_inv_q << ",\n"
                    << "  \"target_q\": " << policy.inverse_scale_product << ",\n"
                    << "  \"max_observed_sum_q\": " << max_observed_sum_q << ",\n"
                    << "  \"max_observed_sum_node\": " << max_observed_node << ",\n"
                    << "  \"max_observed_sum_degree\": " << max_observed_degree << ",\n"
                    << "  \"max_observed_sum_component\": \"" << json_escape(max_observed_component) << "\",\n"
                    << "  \"max_observed_sum_within_derived_bound\": " << bool_json(observed_within_bound) << ",\n"
                    << "  \"max_observed_sum_in_inverse_table_range\": " << bool_json(observed_within_table) << ",\n"
                    << "  \"accepted\": true,\n"
                    << "  \"inverse_all_nodes_accepted\": true,\n"
                    << "  \"alpha_all_edges_accepted\": true,\n"
                    << "  \"checked_nodes\": " << checked_nodes << ",\n"
                    << "  \"checked_edges\": " << checked_edges << "\n"
                    << "}\n";
                return out.str();
            } catch (const std::exception& error) {
                out << "  \"status\": \"blocked\",\n"
                    << "  \"block_reason\": \"native_full_witness_preflight_failed\",\n"
                    << "  \"failure_step\": \"native_full_witness_preflight\",\n"
                    << "  \"failure_relation\": \"C++ adapter must compute full-dataset hidden and output attention witness policy checks before Rust assignment\",\n"
                    << "  \"raw_error\": \"" << json_escape(error.what()) << "\"\n"
                    << "}\n";
                return out.str();
            }
        }

        // 对图的特征矩阵做 FNV-1a 哈希（用于数据完整性校验）
        std::uint64_t hash_feature_payload(const NormalizedGraph& graph) {
            std::uint64_t state = 1469598103934665603ULL;
            for (const auto& row : graph.features_fp) {
                for (const auto value : row) {
                    state = fnv1a64_bytes(&value, sizeof(value), state);
                }
            }
            return state;
        }

        // 将特征矩阵写入二进制文件（带 magic header + 维度信息）
        void write_feature_payload_bin(const std::filesystem::path& path, const NormalizedGraph& graph) {
            std::filesystem::create_directories(path.parent_path());
            std::ofstream output(path, std::ios::binary);
            if (!output) {
                throw std::runtime_error("failed to write feature payload: " + path.string());
            }
            const char magic[] = "EZKLGATFEATUREF64\n";
            output.write(magic, sizeof(magic) - 1);
            write_u64(output, static_cast<std::uint64_t>(graph.node_count));
            write_u64(output, static_cast<std::uint64_t>(graph.feature_dim));
            for (const auto& row : graph.features_fp) {
                for (const auto value : row) {
                    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
                }
            }
        }

        void write_csr_layout_bin(const std::filesystem::path& path, const CsrSegmentLayout& layout) {
            std::filesystem::create_directories(path.parent_path());
            std::ofstream output(path, std::ios::binary);
            if (!output) {
                throw std::runtime_error("failed to write CSR layout: " + path.string());
            }
            output.write("EZKLGATCSR1\n", 12);
            write_size_vector_bin(output, layout.src);
            write_size_vector_bin(output, layout.dst);
            write_size_vector_bin(output, layout.segment_ptr);
            write_size_vector_bin(output, layout.edge_ptr);
            write_size_vector_bin(output, layout.node_ptr);
            write_size_vector_bin(output, layout.segment_graph_id);
            write_u64(output, static_cast<std::uint64_t>(layout.max_in_degree));
        }

        std::string csr_layout_json(const CsrSegmentLayout& layout) {
            std::ostringstream out;
            out << "{\n"
                << "  \"layout_kind\": \"csr_dst_segment_ragged\",\n"
                << "  \"src_path\": \"csr_segment_layout.bin:src\",\n"
                << "  \"dst_path\": \"csr_segment_layout.bin:dst\",\n"
                << "  \"segment_ptr_path\": \"csr_segment_layout.bin:segment_ptr\",\n"
                << "  \"edge_ptr_path\": \"csr_segment_layout.bin:edge_ptr\",\n"
                << "  \"node_ptr_path\": \"csr_segment_layout.bin:node_ptr\",\n"
                << "  \"segment_graph_id_path\": \"csr_segment_layout.bin:segment_graph_id\",\n"
                << "  \"edge_order_rule\": \"edge_gid_then_dst_stable\",\n"
                << "  \"node_count\": " << (layout.segment_ptr.empty() ? 0 : layout.segment_ptr.size() - 1) << ",\n"
                << "  \"edge_count\": " << layout.src.size() << ",\n"
                << "  \"graph_count\": " << (layout.node_ptr.empty() ? 0 : layout.node_ptr.size() - 1) << ",\n"
                << "  \"max_in_degree\": " << layout.max_in_degree << ",\n"
                << "  \"src_hash\": \"" << layout.hashes.src_hash << "\",\n"
                << "  \"dst_hash\": \"" << layout.hashes.dst_hash << "\",\n"
                << "  \"segment_ptr_hash\": \"" << layout.hashes.segment_ptr_hash << "\",\n"
                << "  \"edge_ptr_hash\": \"" << layout.hashes.edge_ptr_hash << "\",\n"
                << "  \"node_ptr_hash\": \"" << layout.hashes.node_ptr_hash << "\",\n"
                << "  \"segment_graph_id_hash\": \"" << layout.hashes.segment_graph_id_hash << "\",\n"
                << "  \"debug_edge_to_segment_slot\": \"not_emitted_by_default\"\n"
                << "}\n";
            return out.str();
        }
    }  // namespace

    // 验证图的边顺序满足 "先按 graph_id 再按 dst 稳定排序" 的规则
    bool verify_edge_gid_then_dst_stable_for_segments(
        const NormalizedGraph& graph,
        std::string* reason) {
        return graph_blocks_are_valid(graph, reason)
            && edge_order_is_gid_then_dst_stable(graph, reason);
    }

    SegmentPackedLayout build_dst_segments(
        const NormalizedGraph& graph,
        std::string* reason) {
        if (!verify_edge_gid_then_dst_stable_for_segments(graph, reason)) {
            return {};
        }

        SegmentPackedLayout layout;
        layout.segment_ptr.assign(graph.node_count + 1, 0);
        std::size_t cursor = 0;
        for (std::size_t node = 0; node < graph.node_count; ++node) {
            layout.segment_ptr[node] = cursor;
            while (cursor < graph.edges.size() && graph.edges[cursor].dst == node) {
                const auto& edge = graph.edges[cursor];
                if (edge.src >= graph.node_count || edge.dst >= graph.node_count) {
                    if (reason) *reason = "edge endpoint out of range while building dst segments";
                    return {};
                }
                ++cursor;
            }
            layout.segment_ptr[node + 1] = cursor;
            if (cursor < graph.edges.size() && graph.edges[cursor].dst < node + 1) {
                if (reason) *reason = "dst segments are not contiguous at node " + std::to_string(node);
                return {};
            }
            const auto degree = layout.segment_ptr[node + 1] - layout.segment_ptr[node];

            // 要求每个节点至少有一条入边（自环保证）
            if (degree == 0) {
                if (reason) *reason = "node " + std::to_string(node) + " has no incoming edge after self-loop preprocessing";
                return {};
            }
            if (degree > layout.max_in_degree) {
                layout.max_in_degree = degree;
            }
        }
        if (cursor != graph.edges.size()) {
            if (reason) *reason = "not all edges were consumed by dst segment construction";
            return {};
        }
        return layout;
    }

    // 将不等长的 segment 填充（packing）为规则矩阵：每节点 pad 到 max_in_degree
    // 生成 valid_mask 标记哪些 slot 是真实的边
    SegmentPackedLayout build_segment_packed_layout(
        const NormalizedGraph& graph,
        std::string* reason) {
        auto layout = build_dst_segments(graph, reason);
        if (layout.segment_ptr.empty() && graph.node_count != 0) {
            return layout;
        }

        layout.total_packed_slots = graph.node_count * layout.max_in_degree;
        layout.padding_slots = layout.total_packed_slots - graph.edge_count;
        layout.padding_ratio = layout.total_packed_slots == 0
            ? 0.0
            : static_cast<double>(layout.padding_slots) / static_cast<double>(layout.total_packed_slots);
        layout.edge_slot_index.assign(graph.edge_count, 0);
        layout.packed_src.assign(layout.total_packed_slots, 0);
        layout.packed_edge_index.assign(layout.total_packed_slots, 0);
        layout.valid_mask.assign(layout.total_packed_slots, 0);

        for (std::size_t node = 0; node < graph.node_count; ++node) {
            const auto begin = layout.segment_ptr[node];
            const auto end = layout.segment_ptr[node + 1];
            for (std::size_t edge_index = begin; edge_index < end; ++edge_index) {
                const auto slot = edge_index - begin;
                const auto packed_index = node * layout.max_in_degree + slot;
                const auto& edge = graph.edges[edge_index];
                layout.edge_slot_index[edge_index] = slot;
                layout.packed_src[packed_index] = edge.src;
                layout.packed_edge_index[packed_index] = edge_index;
                layout.valid_mask[packed_index] = 1;
            }
        }
        return layout;
    }

    // 构建 CSR 格式的 segment 布局（不填充，保留 ragged 结构）
    CsrSegmentLayout build_csr_segment_layout(
        const NormalizedGraph& graph,
        std::string* reason) {
        auto segments = build_dst_segments(graph, reason);
        CsrSegmentLayout layout;
        if (reason != nullptr && !reason->empty()) {
            return layout;
        }
        layout.src = graph.src;
        layout.dst = graph.dst;
        layout.segment_ptr = std::move(segments.segment_ptr);
        layout.edge_ptr = graph.edge_ptr;
        layout.node_ptr = graph.node_ptr;
        layout.max_in_degree = segments.max_in_degree;
        layout.segment_graph_id.assign(graph.node_count, 0);
        for (std::size_t graph_id = 0; graph_id < graph.graph_count; ++graph_id) {
            const auto begin = graph.node_ptr.at(graph_id);
            const auto end = graph.node_ptr.at(graph_id + 1);
            for (std::size_t node = begin; node < end; ++node) {
                layout.segment_graph_id.at(node) = graph_id;
            }
        }
        layout.hashes = hash_csr_segment_layout(layout);
        return layout;
    }

    // 对 packed layout 的各数组做哈希
    SegmentLayoutHashes hash_segment_layout(const SegmentPackedLayout& layout) {
        SegmentLayoutHashes out;
        out.valid_mask_hash = hash_byte_vector(layout.valid_mask);
        out.segment_ptr_hash = hash_size_vector(layout.segment_ptr);
        out.packed_src_hash = hash_size_vector(layout.packed_src);
        out.packed_edge_index_hash = hash_size_vector(layout.packed_edge_index);
        return out;
    }

    // 对 CSR layout 的各数组做哈希
    CsrLayoutHashes hash_csr_segment_layout(const CsrSegmentLayout& layout) {
        CsrLayoutHashes out;
        out.src_hash = hash_size_vector(layout.src);
        out.dst_hash = hash_size_vector(layout.dst);
        out.segment_ptr_hash = hash_size_vector(layout.segment_ptr);
        out.edge_ptr_hash = hash_size_vector(layout.edge_ptr);
        out.node_ptr_hash = hash_size_vector(layout.node_ptr);
        out.segment_graph_id_hash = hash_size_vector(layout.segment_graph_id);
        return out;
    }

    SegmentCostEstimate estimate_segment_lowering_cost(
        const SegmentPackedLayout& layout,
        const CheckpointBundle& bundle,
        const GatSemanticSummary& summary) {
        SegmentCostEstimate out;
        out.estimated_attention_tensor_elements_per_head = layout.total_packed_slots;
        out.hidden_layer_count = bundle.hidden_profile.size();
        out.hidden_head_count_total = bundle.hidden_heads.size();
        out.output_head_count = bundle.output_heads.size();
        const auto total_heads = summary.total_attention_head_count == 0
            ? out.hidden_head_count_total + out.output_head_count
            : summary.total_attention_head_count;
        out.estimated_total_attention_elements = layout.total_packed_slots * total_heads;
        out.estimated_onnx_nodes = total_heads * 34 + out.hidden_layer_count * 6 + 10;
        out.estimated_ezkl_rows = out.estimated_total_attention_elements * 24 + total_heads * 4096;
        return out;
    }

    SegmentAnalysisReport analyze_segment_softmax(
        const std::string& dataset,
        const AppConfig& config,
        const NormalizedGraph& graph,
        const CheckpointBundle& bundle,
        const GatSemanticSummary& summary,
        const ScoreBounds& bounds,
        std::size_t max_packed_slots_limit) {
        std::string reason;
        const auto layout = build_segment_packed_layout(graph, &reason);
        SegmentAnalysisReport report;
        report.dataset = dataset;
        report.node_count = graph.node_count;
        report.edge_count = graph.edge_count;
        report.graph_count = graph.graph_count;
        report.edge_order_rule = config.edge_sort_rule;
        report.self_loop_rule = config.self_loop_rule;
        report.sentinel_value = bounds.invalid_score_sentinel;
        report.sentinel_bound_verified = bounds.sentinel_bound_verified;
        report.sentinel_bound_explanation = bounds.explanation;
        report.max_packed_slots_limit = max_packed_slots_limit;

        if (!reason.empty()) {
            report.blocker_if_any = "ezkl_segment_softmax_analysis_blocked";
            report.sentinel_bound_explanation = reason;
            return report;
        }

        report.max_in_degree = layout.max_in_degree;
        report.total_packed_slots = layout.total_packed_slots;
        report.padding_slots = layout.padding_slots;
        report.padding_ratio = layout.padding_ratio;
        report.hashes = hash_segment_layout(layout);
        report.estimate = estimate_segment_lowering_cost(layout, bundle, summary);

        std::string sentinel_reason;
        if (!verify_sentinel_bound(bounds, &sentinel_reason)) {
            report.blocker_if_any = "ezkl_segment_softmax_sentinel_bound_unproven";
            report.sentinel_bound_verified = false;
            report.sentinel_bound_explanation = sentinel_reason;
        } else if (max_packed_slots_limit != 0 && layout.total_packed_slots > max_packed_slots_limit) {
            report.blocker_if_any = "ezkl_segment_packed_softmax_resource_boundary";
        } else {
            report.route_feasible_small = true;
            report.route_feasible_full = true;
        }
        return report;
    }

    std::string segment_analysis_json(const SegmentAnalysisReport& report) {
        std::ostringstream out;
        out << std::fixed << std::setprecision(6);
        out << "{\n"
            << "  \"dataset\": \"" << json_escape(report.dataset) << "\",\n"
            << "  \"node_count\": " << report.node_count << ",\n"
            << "  \"edge_count\": " << report.edge_count << ",\n"
            << "  \"graph_count\": " << report.graph_count << ",\n"
            << "  \"max_in_degree\": " << report.max_in_degree << ",\n"
            << "  \"total_packed_slots\": " << report.total_packed_slots << ",\n"
            << "  \"padding_slots\": " << report.padding_slots << ",\n"
            << "  \"padding_ratio\": " << report.padding_ratio << ",\n"
            << "  \"estimated_attention_tensor_elements_per_head\": "
            << report.estimate.estimated_attention_tensor_elements_per_head << ",\n"
            << "  \"hidden_layer_count\": " << report.estimate.hidden_layer_count << ",\n"
            << "  \"hidden_head_count_total\": " << report.estimate.hidden_head_count_total << ",\n"
            << "  \"output_head_count\": " << report.estimate.output_head_count << ",\n"
            << "  \"estimated_total_attention_elements\": "
            << report.estimate.estimated_total_attention_elements << ",\n"
            << "  \"estimated_onnx_nodes\": " << report.estimate.estimated_onnx_nodes << ",\n"
            << "  \"estimated_ezkl_rows\": " << report.estimate.estimated_ezkl_rows << ",\n"
            << "  \"route_feasible_small\": " << bool_json(report.route_feasible_small) << ",\n"
            << "  \"route_feasible_full\": " << bool_json(report.route_feasible_full) << ",\n"
            << "  \"blocker_if_any\": \"" << json_escape(report.blocker_if_any) << "\",\n"
            << "  \"segment_lowering\": " << bool_json(report.segment_lowering) << ",\n"
            << "  \"segment_layout\": \"" << json_escape(report.segment_layout) << "\",\n"
            << "  \"padded_edge_slots\": " << report.total_packed_slots << ",\n"
            << "  \"original_edge_count\": " << report.edge_count << ",\n"
            << "  \"sentinel_value\": " << report.sentinel_value << ",\n"
            << "  \"sentinel_bound_verified\": " << bool_json(report.sentinel_bound_verified) << ",\n"
            << "  \"sentinel_bound_explanation\": \""
            << json_escape(report.sentinel_bound_explanation) << "\",\n"
            << "  \"valid_mask_hash\": \"" << report.hashes.valid_mask_hash << "\",\n"
            << "  \"segment_ptr_hash\": \"" << report.hashes.segment_ptr_hash << "\",\n"
            << "  \"packed_src_hash\": \"" << report.hashes.packed_src_hash << "\",\n"
            << "  \"packed_edge_index_hash\": \"" << report.hashes.packed_edge_index_hash << "\",\n"
            << "  \"edge_order_rule\": \"" << json_escape(report.edge_order_rule) << "\",\n"
            << "  \"self_loop_rule\": \"" << json_escape(report.self_loop_rule) << "\",\n"
            << "  \"max_packed_slots_limit\": " << report.max_packed_slots_limit << ",\n"
            << "  \"strict_public_input_equivalent_candidate\": {\n"
            << "    \"candidate\": " << bool_json(report.route_feasible_full) << ",\n"
            << "    \"requires_export_setup_prove_verify\": true,\n"
            << "    \"current_status\": \"analysis_only\"\n"
            << "  }\n"
            << "}\n";
        return out.str();
    }

    std::string segment_metadata_json(const SegmentAnalysisReport& report) {
        std::ostringstream out;
        out << std::fixed << std::setprecision(6);
        out << "{\n"
            << "  \"segment_lowering\": " << bool_json(report.segment_lowering) << ",\n"
            << "  \"segment_layout\": \"" << json_escape(report.segment_layout) << "\",\n"
            << "  \"max_in_degree\": " << report.max_in_degree << ",\n"
            << "  \"padded_edge_slots\": " << report.total_packed_slots << ",\n"
            << "  \"original_edge_count\": " << report.edge_count << ",\n"
            << "  \"padding_ratio\": " << report.padding_ratio << ",\n"
            << "  \"sentinel_value\": " << report.sentinel_value << ",\n"
            << "  \"sentinel_bound_verified\": " << bool_json(report.sentinel_bound_verified) << ",\n"
            << "  \"valid_mask_hash\": \"" << report.hashes.valid_mask_hash << "\",\n"
            << "  \"segment_ptr_hash\": \"" << report.hashes.segment_ptr_hash << "\",\n"
            << "  \"packed_src_hash\": \"" << report.hashes.packed_src_hash << "\",\n"
            << "  \"packed_edge_index_hash\": \"" << report.hashes.packed_edge_index_hash << "\",\n"
            << "  \"edge_order_rule\": \"" << json_escape(report.edge_order_rule) << "\",\n"
            << "  \"self_loop_rule\": \"" << json_escape(report.self_loop_rule) << "\",\n"
            << "  \"strict_public_input_equivalent_candidate\": "
            << bool_json(report.route_feasible_full) << "\n"
            << "}\n";
        return out.str();
    }

    void write_segment_analysis_json(
        const std::filesystem::path& path,
        const SegmentAnalysisReport& report) {
        write_text_file(path, segment_analysis_json(report));
    }

    void write_segment_debug_files(
        const std::filesystem::path& output_dir,
        const SegmentPackedLayout& layout) {
        const auto debug_dir = output_dir / "segment_debug";
        write_size_vector(debug_dir / "segment_ptr.txt", layout.segment_ptr);
        write_size_vector(debug_dir / "edge_slot_index.txt", layout.edge_slot_index);
        write_size_vector(debug_dir / "packed_src.txt", layout.packed_src);
        write_size_vector(debug_dir / "packed_edge_index.txt", layout.packed_edge_index);
        write_mask_vector(debug_dir / "valid_mask.txt", layout.valid_mask);
    }

    // 导出原生 CSR 图结构 artifacts，供 EZKL 电路生成使用
    void write_native_csr_graph_artifacts(
        const std::filesystem::path& output_dir,
        const std::string& route_id,
        const AppConfig& config,
        const NormalizedGraph& graph,
        const CheckpointBundle& bundle,
        const GatSemanticSummary& summary,
        const std::string& graph_hash,
        const std::string& checkpoint_hash,
        const std::string& metadata_hash,
        const std::string& shape_hash,
        const FormalTablesSummary* formal_tables) {
        const bool compressed_route = route_id == "ezkl_gat_native_compressed_csr_attention_full";
        const std::string native_primitive = compressed_route
            ? "GraphSegmentAttentionCompressedCSR"
            : "GraphSegmentAttention";
        const std::string native_primitive_version = compressed_route
            ? "csr_segment_attention_compressed_v1"
            : "csr_segment_attention_v1";
        std::string reason;
        const auto layout = build_csr_segment_layout(graph, &reason);
        if (!reason.empty()) {
            throw std::runtime_error("cannot export native CSR layout: " + reason);
        }

        const auto csr_bin_path = output_dir / "csr_segment_layout.bin";
        const auto features_path = output_dir / "node_features.f64.bin";
        const auto checkpoint_copy_path = output_dir / "checkpoint_tensors.txt";
        write_csr_layout_bin(csr_bin_path, layout);
        write_text_file(output_dir / "csr_segment_layout.json", csr_layout_json(layout));
        write_feature_payload_bin(features_path, graph);
        std::filesystem::copy_file(
            bundle.root / "tensors.txt",
            checkpoint_copy_path,
            std::filesystem::copy_options::overwrite_existing);

        // 生成 tensor manifest JSON
        const auto feature_hash = hex_u64(hash_feature_payload(graph));
        const auto checkpoint_tensor_hash = hash_file_hex(checkpoint_copy_path);
        std::ostringstream tensor_manifest;
        tensor_manifest << "{\n"
            << "  \"tensor_payloads_kind\": \"full_gat_native_csr_inputs\",\n"
            << "  \"node_features\": {\"path\":\"node_features.f64.bin\",\"dtype\":\"f64\",\"shape\":["
            << graph.node_count << "," << graph.feature_dim << "],\"hash\":\"" << feature_hash << "\"},\n"
            << "  \"checkpoint_tensors\": {\"path\":\"checkpoint_tensors.txt\",\"format\":\"adapter_tensor_dump\","
            << "\"tensor_count\":" << bundle.tensors.size() << ",\"hash\":\"" << checkpoint_tensor_hash << "\"},\n"
            << "  \"csr_layout\": {\"path\":\"csr_segment_layout.bin\",\"format\":\"ezkl_gat_csr_v1\","
            << "\"src_hash\":\"" << layout.hashes.src_hash
            << "\",\"dst_hash\":\"" << layout.hashes.dst_hash
            << "\",\"segment_ptr_hash\":\"" << layout.hashes.segment_ptr_hash
            << "\",\"edge_ptr_hash\":\"" << layout.hashes.edge_ptr_hash
            << "\",\"node_ptr_hash\":\"" << layout.hashes.node_ptr_hash << "\"}\n"
            << "}\n";
        write_text_file(output_dir / "tensor_payloads.json", tensor_manifest.str());
        write_text_file(
            output_dir / "native_full_witness_preflight.json",
            native_full_witness_preflight_json(route_id, config, graph, bundle, layout, formal_tables));
        write_text_file(
            output_dir / "final_logits_binding.json",
            final_logits_binding_json(route_id, config, graph, bundle));

        std::ostringstream spec;
        const auto final_logits_values = derived_final_logits_q(graph, bundle);
        const auto final_logits_hash = hash_final_logits_values(graph, bundle, final_logits_values);
        spec << "{\n"
            << "  \"route_id\": \"" << json_escape(route_id) << "\",\n"
            << "  \"native_primitive\": \"" << native_primitive << "\",\n"
            << "  \"native_primitive_version\": \"" << native_primitive_version << "\",\n"
            << "  \"full_dataset\": true,\n"
            << "  \"strict_public_input_equivalent_candidate\": true,\n"
            << "  \"forbidden_substitute_used\": false,\n"
            << "  \"precomputed_logits_binding_used\": false,\n"
            << "  \"no_dense_nxn_attention\": true,\n"
            << "  \"csr_layout\": \"csr_dst_segment_ragged\",\n"
            << "  \"dataset\": \"" << json_escape(config.dataset) << "\",\n"
            << "  \"node_count\": " << graph.node_count << ",\n"
            << "  \"edge_count\": " << graph.edge_count << ",\n"
            << "  \"graph_count\": " << graph.graph_count << ",\n"
            << "  \"feature_dim\": " << graph.feature_dim << ",\n"
            << "  \"class_count\": " << graph.class_count << ",\n"
            << "  \"L\": " << bundle.layer_count << ",\n"
            << "  \"K_out\": " << bundle.K_out << ",\n"
            << "  \"hidden_profile\": " << hidden_profile_json(bundle.hidden_profile) << ",\n"
            << "  \"d_in_profile\": " << size_vector_json(bundle.d_in_profile) << ",\n"
            << "  \"hidden_heads\": " << hidden_heads_json(bundle) << ",\n"
            << "  \"output_heads\": " << output_heads_json(bundle) << ",\n"
            << "  \"output_layer_input_source\": \"last_hidden_concat\",\n"
            << "  \"output_layer_input_dim\": "
            << (bundle.hidden_profile.empty() ? graph.feature_dim : bundle.hidden_profile.back().head_count * bundle.hidden_profile.back().head_dim) << ",\n"
            << "  \"output_head_count\": " << bundle.output_heads.size() << ",\n"
            << "  \"output_head_specs\": " << output_head_specs_json(bundle) << ",\n"
            << "  \"output_projection_policy\": \"seq_kernel_matmul_last_hidden_concat_to_C\",\n"
            << "  \"output_attention_policy\": \"a_src_a_dst_over_projected_logits_with_lrelu_segment_softmax\",\n"
            << "  \"output_bias_add_policy\": \"post_aggregation_bias_add_identity_activation\",\n"
            << "  \"output_average_policy\": \"" << json_escape(bundle.output_average_rule) << "\",\n"
            << "  \"output_derivation_required\": true,\n"
            << "  \"output_derivation_complete_candidate\": true,\n"
            << "  \"no_precomputed_logits_bypass\": true,\n"
            << "  \"output_layer_spec\": {\"output_head_count\":" << bundle.output_heads.size()
            << ",\"output_layer_input_source\":\"last_hidden_concat\","
            << "\"output_layer_input_dim\":" << (bundle.hidden_profile.empty() ? graph.feature_dim : bundle.hidden_profile.back().head_count * bundle.hidden_profile.back().head_dim) << ","
            << "\"output_projection_dims\":\"last_hidden_concat_width_to_C\","
            << "\"output_bias_dims\":[" << graph.class_count << "],"
            << "\"K_out\":" << bundle.K_out
            << ",\"C\":" << graph.class_count
            << ",\"output_average_rule\":\"" << json_escape(bundle.output_average_rule) << "\"},\n"
            << "  \"final_logits_binding_spec\": {\"public_instance_order\":\"final_logits_row_major_node_then_class_chunked_columns_column_then_row\","
            << "\"public_instance_layout\":\"chunked_columns\","
            << "\"expected_public_output_count\":" << final_logits_values.size()
            << ",\"final_logits_shape\":[" << graph.node_count << "," << graph.class_count << "],"
            << "\"output_scale\":16,"
            << "\"output_rounding_policy\":\"integer_exact_no_post_average_rescale\","
            << "\"final_logits_hash\":\"" << final_logits_hash << "\","
            << "\"public_instance_hash\":\"" << final_logits_hash << "\"},\n"
            << "  \"gat_semantics\": {\"projection\":true,\"edge_scores\":true,\"lrelu_lookup_required\":true,"
            << "\"segment_max\":true,\"exp_map_lookup_required\":true,\"inverse_policy\":\"quantized_inverse_lookup\","
            << "\"alpha_rescale_policy\":\"rounded_product_div_scale_with_quantization_band\","
            << "\"aggregation\":true,\"hidden_activation\":\"elu_lookup_required\",\"output_average_rule\":\""
            << json_escape(bundle.output_average_rule) << "\"},\n"
            << "  \"batching_rule\": \"" << json_escape(config.batching_rule) << "\",\n"
            << "  \"subgraph_rule\": \"" << json_escape(config.subgraph_rule) << "\",\n"
            << "  \"self_loop_rule\": \"" << json_escape(config.self_loop_rule) << "\",\n"
            << "  \"edge_sort_rule\": \"" << json_escape(config.edge_sort_rule) << "\",\n"
            << "  \"chunking_rule\": \"" << json_escape(config.chunking_rule) << "\",\n"
            << "  \"model_arch_id\": \"" << json_escape(config.model_arch_id) << "\",\n"
            << "  \"model_param_id\": \"" << json_escape(config.model_param_id) << "\",\n"
            << "  \"quant_cfg_id\": \"" << json_escape(config.quant_cfg_id) << "\",\n"
            << "  \"static_table_id\": \"" << json_escape(config.static_table_id) << "\",\n";
        if (formal_tables != nullptr) {
            spec << "  \"formal_tables\": " << formal_table_binding_json(*formal_tables) << ",\n"
                << "  \"formal_table_hash\": \"" << formal_tables->hashes.table_hash << "\",\n";
        }
        spec
            << "  \"graph_hash\": \"" << graph_hash << "\",\n"
            << "  \"checkpoint_hash\": \"" << checkpoint_hash << "\",\n"
            << "  \"metadata_hash\": \"" << metadata_hash << "\",\n"
            << "  \"shape_hash\": \"" << shape_hash << "\",\n"
            << "  \"feature_payload_hash\": \"" << feature_hash << "\",\n"
            << "  \"checkpoint_tensor_payload_hash\": \"" << checkpoint_tensor_hash << "\",\n"
            << "  \"total_attention_head_count\": " << summary.total_attention_head_count << ",\n"
            << "  \"csr_hashes\": {\"src\":\"" << layout.hashes.src_hash
            << "\",\"dst\":\"" << layout.hashes.dst_hash
            << "\",\"segment_ptr\":\"" << layout.hashes.segment_ptr_hash
            << "\",\"edge_ptr\":\"" << layout.hashes.edge_ptr_hash
            << "\",\"node_ptr\":\"" << layout.hashes.node_ptr_hash
            << "\",\"segment_graph_id\":\"" << layout.hashes.segment_graph_id_hash << "\"}\n"
            << "}\n";
        write_text_file(output_dir / "graph_attention_spec.json", spec.str());
    }

}
