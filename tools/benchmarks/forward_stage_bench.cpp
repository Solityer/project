#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gatzk/model/gat.hpp"

namespace
{
    using Clock =
        std::chrono::steady_clock;

    double elapsed_ms(
        const Clock::time_point &start)
    {
        return std::chrono::duration<
                   double,
                   std::milli>(
                   Clock::now() - start)
            .count();
    }

    gatzk::model::FloatMatrix
    make_features(
        std::size_t node_count,
        std::size_t feature_count)
    {
        gatzk::model::FloatMatrix out(
            node_count,
            std::vector<double>(
                feature_count,
                0.0));

        for (std::size_t row = 0;
             row < node_count;
             ++row)
        {
            for (std::size_t column = 0;
                 column < feature_count;
                 ++column)
            {
                const auto value =
                    static_cast<double>(
                        1 + (row * 17 + column * 13) % 101);

                out[row][column] =
                    value / 128.0;
            }
        }

        return out;
    }

    std::vector<gatzk::data::Edge>
    make_edges(
        std::size_t node_count,
        std::size_t fanout)
    {
        std::vector<gatzk::data::Edge>
            edges;

        edges.reserve(
            node_count * fanout);

        std::size_t stable_index = 0;

        for (std::size_t dst = 0;
             dst < node_count;
             ++dst)
        {
            for (std::size_t slot = 0;
                 slot < fanout;
                 ++slot)
            {
                const auto src =
                    (dst + slot * 17) % node_count;

                edges.push_back(
                    gatzk::data::Edge{
                        .src = src,
                        .dst = dst,
                        .graph_id = 0,
                        .stable_index =
                            stable_index++,
                    });
            }
        }

        return edges;
    }

    bool same_head(
        const gatzk::model::HeadForwardTrace &left,
        const gatzk::model::HeadForwardTrace &right)
    {
        return left.H_prime == right.H_prime && left.E_src == right.E_src && left.E_dst == right.E_dst && left.S == right.S && left.Z == right.Z && left.M == right.M && left.Delta == right.Delta && left.U == right.U && left.Sum == right.Sum && left.inv == right.inv && left.alpha == right.alpha && left.H_agg_pre_bias == right.H_agg_pre_bias && left.H_agg == right.H_agg;
    }

    bool same_forward_material(
        const gatzk::model::
            MultiHeadForwardTrace &reference,
        const gatzk::model::
            MultiHeadForwardTrace &optimized)
    {
        if (reference.H != optimized.H || reference.hidden_concat != optimized.hidden_concat || reference.hidden_layer_traces.size() != optimized.hidden_layer_traces.size() || reference.hidden_head_traces.size() != optimized.hidden_head_traces.size() || reference.output_head_traces.size() != optimized.output_head_traces.size() || reference.output_head_values != optimized.output_head_values || reference.Y != optimized.Y)
        {
            return false;
        }

        for (std::size_t layer = 0;
             layer < reference
                         .hidden_layer_traces
                         .size();
             ++layer)
        {
            const auto &lhs =
                reference
                    .hidden_layer_traces[layer];

            const auto &rhs =
                optimized
                    .hidden_layer_traces[layer];

            if (lhs.input != rhs.input || lhs.concat != rhs.concat || lhs.head_traces.size() != rhs.head_traces.size())
            {
                return false;
            }

            for (std::size_t head = 0;
                 head < lhs
                            .head_traces
                            .size();
                 ++head)
            {
                if (!same_head(
                        lhs.head_traces[head],
                        rhs.head_traces[head]))
                {
                    return false;
                }
            }
        }

        for (std::size_t head = 0;
             head < reference
                        .hidden_head_traces
                        .size();
             ++head)
        {
            if (!same_head(
                    reference
                        .hidden_head_traces[head],
                    optimized
                        .hidden_head_traces[head]))
            {
                return false;
            }
        }

        for (std::size_t head = 0;
             head < reference
                        .output_head_traces
                        .size();
             ++head)
        {
            if (!same_head(
                    reference
                        .output_head_traces[head],
                    optimized
                        .output_head_traces[head]))
            {
                return false;
            }
        }

        // Y_lin intentionally has different semantics in reference_style:
        // reference_style aliases the post-bias average, whereas formal
        // note_style retains the registered pre-bias average. Therefore it is
        // not an equality target here.

        return same_head(
            reference.output_head_trace,
            optimized.output_head_trace);
    }
}

int main(
    int argc,
    char **argv)
{
    try
    {
        const std::size_t node_count =
            argc > 1
                ? static_cast<std::size_t>(
                      std::stoull(argv[1]))
                : 4096;

        const std::size_t repeats =
            argc > 2
                ? static_cast<std::size_t>(
                      std::stoull(argv[2]))
                : 3;

        if (node_count < 64 || repeats == 0)
        {
            throw std::runtime_error(
                "forward benchmark requires nodes>=64 and repeats>=1");
        }

        constexpr std::size_t
            kFeatureCount = 64;

        constexpr std::size_t
            kHiddenHeads = 8;

        constexpr std::size_t
            kHiddenWidth = 16;

        constexpr std::size_t
            kOutputHeads = 4;

        constexpr std::size_t
            kClasses = 16;

        constexpr std::size_t
            kFanout = 8;

        const auto features =
            make_features(
                node_count,
                kFeatureCount);

        const auto edges =
            make_edges(
                node_count,
                kFanout);

        const auto model =
            gatzk::model::
                build_family_model_parameters(
                    {kFeatureCount},
                    {{kHiddenHeads,
                      kHiddenWidth}},
                    kOutputHeads,
                    kClasses,
                    17);

        // Warm both paths before timed samples.
        (void)gatzk::model::
            forward_reference_style(
                features,
                edges,
                model);

        (void)gatzk::model::
            forward_note_style(
                features,
                edges,
                model);

        std::cout
            << "nodes,edges,hidden_heads,output_heads,"
               "repeat,reference_ms,optimized_ms,speedup,exact\n";

        for (std::size_t repeat = 0;
             repeat < repeats;
             ++repeat)
        {
            gatzk::model::
                MultiHeadForwardTrace reference;

            gatzk::model::
                MultiHeadForwardTrace optimized;

            double reference_ms = 0.0;
            double optimized_ms = 0.0;

            if ((repeat & 1U) == 0)
            {
                auto start =
                    Clock::now();

                reference =
                    gatzk::model::
                        forward_reference_style(
                            features,
                            edges,
                            model);

                reference_ms =
                    elapsed_ms(start);

                start =
                    Clock::now();

                optimized =
                    gatzk::model::
                        forward_note_style(
                            features,
                            edges,
                            model);

                optimized_ms =
                    elapsed_ms(start);
            }
            else
            {
                auto start =
                    Clock::now();

                optimized =
                    gatzk::model::
                        forward_note_style(
                            features,
                            edges,
                            model);

                optimized_ms =
                    elapsed_ms(start);

                start =
                    Clock::now();

                reference =
                    gatzk::model::
                        forward_reference_style(
                            features,
                            edges,
                            model);

                reference_ms =
                    elapsed_ms(start);
            }

            const bool exact =
                same_forward_material(
                    reference,
                    optimized);

            if (!exact)
            {
                throw std::runtime_error(
                    "optimized forward changed retained forward material");
            }

            std::cout
                << node_count << ','
                << edges.size() << ','
                << kHiddenHeads << ','
                << kOutputHeads << ','
                << repeat << ','
                << std::fixed
                << std::setprecision(3)
                << reference_ms << ','
                << optimized_ms << ','
                << (optimized_ms > 0.0
                        ? reference_ms / optimized_ms
                        : 0.0)
                << ",true\n"
                << std::flush;
        }
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "fatal: "
            << error.what()
            << '\n';

        return 1;
    }

    return 0;
}