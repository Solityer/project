#include "gatzk/data/loader.hpp"
#include "gatzk/model/gat.hpp"
#include "gatzk/util/config.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef GATZK_SOURCE_ROOT
#define GATZK_SOURCE_ROOT "."
#endif

namespace
{
    using gatzk::data::Edge;
    using gatzk::model::AttentionHeadParameters;
    using gatzk::model::FloatMatrix;
    using gatzk::model::ModelParameters;

    constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
    constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
    constexpr std::uint64_t kExpectedForwardHash = 0xc4f9d2d821b23a70ULL;

    void require(bool condition, const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error(message);
        }
    }

    void hash_bytes(std::uint64_t *hash, const void *data, std::size_t size)
    {
        const auto *bytes = static_cast<const std::uint8_t *>(data);
        for (std::size_t index = 0; index < size; ++index)
        {
            *hash ^= bytes[index];
            *hash *= kFnvPrime;
        }
    }

    template <typename T>
    void hash_value(std::uint64_t *hash, const T &value)
    {
        hash_bytes(hash, &value, sizeof(value));
    }

    void hash_double(std::uint64_t *hash, double value)
    {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        hash_value(hash, bits);
    }

    void hash_matrix(std::uint64_t *hash, const FloatMatrix &matrix)
    {
        for (const auto &row : matrix)
        {
            for (const auto value : row)
            {
                hash_double(hash, value);
            }
        }
    }

    void hash_vector(
        std::uint64_t *hash,
        const std::vector<double> &values)
    {
        for (const auto value : values)
        {
            hash_double(hash, value);
        }
    }

    std::uint64_t forward_hash(
        const gatzk::model::MultiHeadForwardTrace &trace)
    {
        // This is the pre-change semantic golden recorded by the independent
        // forward harness: every hidden per-head field, every hidden concat,
        // and both final averaged outputs.  The removed H/input members were
        // exact duplicate matrices and are deliberately excluded.  Output
        // head shapes are checked below; final Y/Y_lin cover their public
        // semantics without making the golden depend on storage details.
        std::uint64_t hash = kFnvOffset;
        for (const auto &layer : trace.hidden_layer_traces)
        {
            hash_matrix(&hash, layer.concat);
            for (const auto &head : layer.head_traces)
            {
                hash_matrix(&hash, head.H_prime);
                hash_vector(&hash, head.E_src);
                hash_vector(&hash, head.E_dst);
                hash_vector(&hash, head.S);
                hash_vector(&hash, head.Z);
                hash_vector(&hash, head.M);
                hash_vector(&hash, head.Delta);
                hash_vector(&hash, head.U);
                hash_vector(&hash, head.Sum);
                hash_vector(&hash, head.inv);
                hash_vector(&hash, head.alpha);
                hash_matrix(&hash, head.H_agg_pre_bias);
                hash_matrix(&hash, head.H_agg);
            }
        }
        hash_matrix(&hash, trace.Y_lin);
        hash_matrix(&hash, trace.Y);
        return hash;
    }

    AttentionHeadParameters make_head(
        std::size_t input_dim,
        std::size_t output_dim,
        std::size_t seed,
        bool output)
    {
        AttentionHeadParameters head;
        head.seq_kernel_fp.assign(
            input_dim,
            std::vector<double>(output_dim, 0.0));
        for (std::size_t row = 0; row < input_dim; ++row)
        {
            for (std::size_t column = 0; column < output_dim; ++column)
            {
                head.seq_kernel_fp[row][column] =
                    0.001 * static_cast<double>(
                                1 + ((row * 17 + column * 13 + seed) % 11));
            }
        }
        head.attn_src_kernel_fp.resize(output_dim);
        head.attn_dst_kernel_fp.resize(output_dim);
        for (std::size_t column = 0; column < output_dim; ++column)
        {
            head.attn_src_kernel_fp[column] =
                0.001 * static_cast<double>(1 + ((column + seed) % 7));
            head.attn_dst_kernel_fp[column] =
                0.001 * static_cast<double>(1 + ((column * 3 + seed) % 7));
        }
        if (output)
        {
            head.output_bias_fp.resize(output_dim);
            for (std::size_t column = 0; column < output_dim; ++column)
            {
                head.output_bias_fp[column] =
                    0.0001 * static_cast<double>(column + seed + 1);
            }
        }
        return head;
    }

    ModelParameters make_model()
    {
        ModelParameters parameters;
        parameters.L = 3;
        parameters.K_out = 2;
        parameters.C = 32;
        parameters.d_in_profile = {50, 64};
        parameters.hidden_profile = {{8, 8}, {8, 8}};
        parameters.hidden_layers.resize(2);
        for (std::size_t layer_index = 0; layer_index < 2; ++layer_index)
        {
            auto &layer = parameters.hidden_layers[layer_index];
            layer.layer_index = layer_index;
            layer.input_dim = layer_index == 0 ? 50 : 64;
            layer.shape = {8, 8};
            for (std::size_t head_index = 0; head_index < 8; ++head_index)
            {
                layer.heads.push_back(make_head(
                    layer.input_dim,
                    8,
                    layer_index * 8 + head_index,
                    false));
            }
        }
        parameters.output_layer.input_dim = 64;
        parameters.output_layer.output_dim = 32;
        parameters.output_layer.head_count = 2;
        for (std::size_t head_index = 0; head_index < 2; ++head_index)
        {
            parameters.output_layer.heads.push_back(make_head(
                64,
                32,
                100 + head_index,
                true));
        }
        return parameters;
    }

    void make_fixture(
        FloatMatrix *features,
        std::vector<Edge> *edges)
    {
        constexpr std::size_t node_count = 8192;
        constexpr std::size_t feature_dim = 50;
        constexpr std::size_t graph_count = 8;
        constexpr std::size_t nodes_per_graph = node_count / graph_count;

        features->assign(
            node_count,
            std::vector<double>(feature_dim, 0.0));
        for (std::size_t row = 0; row < node_count; ++row)
        {
            for (std::size_t column = 0; column < feature_dim; ++column)
            {
                features->at(row).at(column) =
                    0.01 * static_cast<double>((row + 3 * column) % 17);
            }
        }

        edges->clear();
        edges->reserve(node_count * 2);
        for (std::size_t graph = 0; graph < graph_count; ++graph)
        {
            const auto begin = graph * nodes_per_graph;
            const auto end = begin + nodes_per_graph;
            for (std::size_t node = begin; node < end; ++node)
            {
                edges->push_back(Edge{node, node, graph, edges->size()});
                const auto next = node + 1 == end ? begin : node + 1;
                edges->push_back(Edge{node, next, graph, edges->size()});
            }
        }
    }

    std::uint64_t first_input_quantized_hash(
        const FloatMatrix &features,
        std::uint64_t scale)
    {
        std::uint64_t hash = kFnvOffset;
        for (const auto &row : features)
        {
            for (const auto value : row)
            {
                const auto quantized = gatzk::data::quantize_raw_at_scale(
                    value,
                    scale);
                hash_value(&hash, quantized);
            }
        }
        return hash;
    }

    std::string read_text(const std::string &path)
    {
        std::ifstream input(path);
        if (!input)
        {
            throw std::runtime_error("unable to read source contract: " + path);
        }
        return std::string(
            std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>());
    }

    void check_consumer_mapping_contract()
    {
        const auto trace_source = read_text(
            std::string(GATZK_SOURCE_ROOT) + "/src/protocol/trace.cpp");
        const auto input_anchor = trace_source.find(
            "const RawMatrix input_raw");
        require(
            input_anchor != std::string::npos,
            "cached forward input source anchor is missing");
        const auto first_layer_source = trace_source.find(
            "context.local.features_fp",
            input_anchor);
        const auto later_layer_source = trace_source.find(
            "hidden_layer_concat_raw.back()",
            input_anchor);
        require(
            first_layer_source != std::string::npos &&
                later_layer_source != std::string::npos &&
                first_layer_source < later_layer_source,
            "quantized input consumer mapping is not canonical-first/previous-concat");
    }
}

int main()
{
    try
    {
        FloatMatrix features;
        std::vector<Edge> edges;
        make_fixture(&features, &edges);
        const auto parameters = make_model();
        const std::vector<gatzk::util::HiddenLayerProfile> scale_profile = {
            {8, 8},
            {8, 8},
        };
        const auto scales = gatzk::util::canonical_quantization_scale_chain(
            "gat_l3_scale_chain",
            scale_profile,
            parameters.K_out,
            GATZK_CANONICAL_SCALE_CHAIN_PATH);

        require(features.size() == 8192, "fixture node count changed");
        require(edges.size() == 16384, "fixture edge count changed");
        require(parameters.hidden_layers.size() == 2, "fixture layer count changed");

        const auto first_input_hash = first_input_quantized_hash(
            features,
            scales.s_input);
        constexpr std::uint64_t kExpectedFirstInputHash =
            0x88a41d9be1097e43ULL;
        require(
            first_input_hash == kExpectedFirstInputHash,
            "canonical S_input quantized input golden mismatch");

        const auto first = gatzk::model::forward_note_style(
            features,
            edges,
            parameters,
            nullptr);
        const auto second = gatzk::model::forward_note_style(
            features,
            edges,
            parameters,
            nullptr);

        require(
            first.hidden_layer_traces.size() == 2 &&
                first.hidden_layer_traces[0].head_traces.size() == 8 &&
                first.hidden_layer_traces[1].head_traces.size() == 8,
            "forward layer/head shape changed");
        require(
            first.hidden_layer_traces[0].concat.size() == features.size() &&
                first.hidden_layer_traces[1].concat.size() == features.size(),
            "forward concat row shape changed");
        require(
            first.hidden_layer_traces[0].concat.front().size() == 64 &&
                first.hidden_layer_traces[1].concat.front().size() == 64,
            "forward concat width changed");
        require(
            first.hidden_layer_traces[1].head_traces[0].H_prime.front().size() == 8,
            "second-layer projection shape changed");
        require(
            first.output_head_traces.size() == 2 &&
                first.output_head_traces[0].H_agg.size() == features.size() &&
                first.output_head_traces[0].H_agg.front().size() == 32 &&
                first.output_head_traces[1].H_agg.size() == features.size(),
            "output head trace shape changed");

        const auto first_hash = forward_hash(first);
        const auto second_hash = forward_hash(second);
        require(first_hash == second_hash, "forward trace is not repeat-stable");
        require(
            first_hash == kExpectedForwardHash,
            "forward full-field golden mismatch");

        check_consumer_mapping_contract();

        std::cout << "FORWARD_TRACE_REGRESSION_OK"
                  << " forward_hash=" << std::hex << first_hash
                  << " first_input_quantized_hash=" << first_input_hash
                  << std::dec
                  << " layers=2 heads=8x8+8x8 output_heads=2"
                  << " nodes=" << features.size()
                  << " edges=" << edges.size()
                  << " graphs=8\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "FORWARD_TRACE_REGRESSION_FAIL: " << error.what() << '\n';
        return 1;
    }
}
