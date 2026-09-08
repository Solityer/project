#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gatzk/algebra/field.hpp"
#include "gatzk/protocol/schema.hpp"
#include "gatzk/protocol/t_edge_program.hpp"
#include "protocol/internal/t_edge_descriptor.hpp"

namespace
{
    using gatzk::algebra::FieldElement;
    using gatzk::protocol::TEdgeBoundProgram;
    using gatzk::protocol::TEdgeProgramAccess;
    using gatzk::protocol::TEdgeProgramInstruction;
    using gatzk::protocol::TEdgeProgramOpcode;
    using gatzk::protocol::internal::TEdgeDescriptorShape;
    using gatzk::protocol::internal::TEdgeDerivedLinearDescriptor;
    using gatzk::protocol::internal::TEdgeDerivedLinearFlattenStatus;
    using gatzk::protocol::internal::TEdgeDerivedLinearSource;

    void expect(bool condition, const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error(message);
        }
    }

    FieldElement challenge_value(
        std::size_t layer,
        std::size_t head,
        std::size_t ordinal)
    {
        return FieldElement(
            static_cast<std::uint64_t>(
                17 + layer * 31 + head * 7 + ordinal));
    }

    void add_challenges(
        std::map<std::string, FieldElement> *challenges,
        const TEdgeDescriptorShape &shape)
    {
        const std::array<std::string_view, 7> hidden_bases{
            "eta_L_h",
            "eta_exp_h",
            "lambda_psq_h",
            "eta_ELU_h",
            "eta_src_h",
            "eta_t_h",
            "eta_dst_h"};
        for (std::size_t layer = 0;
             layer < shape.hidden_head_counts.size();
             ++layer)
        {
            for (std::size_t head = 0;
                 head < shape.hidden_head_counts[layer];
                 ++head)
            {
                for (std::size_t ordinal = 0;
                     ordinal < hidden_bases.size();
                     ++ordinal)
                {
                    challenges->emplace(
                        gatzk::protocol::hidden_challenge_name(
                            hidden_bases[ordinal],
                            layer,
                            head),
                        challenge_value(layer, head, ordinal));
                }
            }
        }

        const std::array<std::string_view, 6> output_bases{
            "eta_L_out",
            "eta_exp_out",
            "lambda_out",
            "eta_src_out",
            "eta_t_out",
            "eta_dst_out"};
        for (std::size_t head = 0;
             head < shape.output_head_count;
             ++head)
        {
            for (std::size_t ordinal = 0;
                 ordinal < output_bases.size();
                 ++ordinal)
            {
                challenges->emplace(
                    gatzk::protocol::output_challenge_name(
                        output_bases[ordinal],
                        head),
                    challenge_value(9, head, ordinal));
            }
        }
    }

    const FieldElement &source_scale(
        const TEdgeDerivedLinearDescriptor &descriptor,
        const std::string &label)
    {
        const auto it = std::find_if(
            descriptor.sources.begin(),
            descriptor.sources.end(),
            [&](const auto &source)
            { return source.label == label; });
        if (it == descriptor.sources.end())
        {
            throw std::runtime_error(
                "descriptor is missing source " + label);
        }
        return it->scale;
    }

    TEdgeDerivedLinearDescriptor require_descriptor(
        const std::string &name,
        const TEdgeDescriptorShape &shape,
        const std::map<std::string, FieldElement> &challenges)
    {
        const auto descriptor =
            gatzk::protocol::internal::make_t_edge_derived_linear_descriptor(
                name,
                shape,
                challenges);
        expect(descriptor.has_value(), "missing descriptor for " + name);
        return *descriptor;
    }

    void check_hidden_mapping(
        const TEdgeDescriptorShape &shape,
        const std::map<std::string, FieldElement> &challenges,
        std::size_t layer,
        std::size_t head)
    {
        const auto prefix =
            gatzk::protocol::hidden_head_prefix(layer, head) + "_";
        const auto challenge = [&](std::string_view base)
        {
            return challenges.at(
                gatzk::protocol::hidden_challenge_name(
                    base,
                    layer,
                    head));
        };

        for (const auto suffix : {
                 "Table_L",
                 "Query_L",
                 "S",
                 "Delta",
                 "Table_R",
                 "Query_R",
                 "Table_exp",
                 "Query_exp",
                 "w_psq",
                 "T_psq_edge",
                 "Table_ELU",
                 "Query_ELU",
                 "Query_src",
                 "Query_t",
                 "Query_dst"})
        {
            const auto descriptor = require_descriptor(
                prefix + suffix,
                shape,
                challenges);
            expect(
                descriptor.target_label == prefix + suffix,
                "hidden descriptor target mismatch");
        }

        const auto table_l = require_descriptor(
            prefix + "Table_L",
            shape,
            challenges);
        expect(
            source_scale(table_l, "P_T_L_x") == FieldElement::one() &&
                source_scale(table_l, "P_T_L_y") == challenge("eta_L_h"),
            "hidden Table_L challenge scale mismatch");

        const auto query_src = require_descriptor(
            prefix + "Query_src",
            shape,
            challenges);
        const auto eta_src = challenge("eta_src_h");
        expect(
            source_scale(query_src, "P_src") == FieldElement::one() &&
                source_scale(query_src, prefix + "E_src_edge") == eta_src &&
                source_scale(query_src, prefix + "H_src_star_edge") ==
                    eta_src * eta_src,
            "hidden Query_src challenge scales mismatch");

        const auto query_dst = require_descriptor(
            prefix + "Query_dst",
            shape,
            challenges);
        const auto eta_dst = challenge("eta_dst_h");
        expect(
            source_scale(query_dst, prefix + "E_dst_edge") == eta_dst &&
                source_scale(query_dst, prefix + "M_edge") ==
                    eta_dst * eta_dst &&
                source_scale(query_dst, prefix + "H_agg_star_edge") ==
                    eta_dst.pow(5),
            "hidden Query_dst challenge scales mismatch");
    }

    void check_output_mapping(
        const TEdgeDescriptorShape &shape,
        const std::map<std::string, FieldElement> &challenges,
        std::size_t head)
    {
        const auto prefix =
            gatzk::protocol::output_head_prefix(head) + "_";
        for (const auto suffix : {
                 "Table_L",
                 "Query_L",
                 "S",
                 "Delta",
                 "Table_R",
                 "Query_R",
                 "Table_exp",
                 "Query_exp",
                 "Query_src",
                 "w",
                 "T_edge",
                 "Query_t",
                 "Query_dst"})
        {
            const auto descriptor = require_descriptor(
                prefix + suffix,
                shape,
                challenges);
            expect(
                descriptor.target_label == prefix + suffix,
                "output descriptor target mismatch");
        }

        const auto output_challenge = [&](std::string_view base)
        {
            return challenges.at(
                gatzk::protocol::output_challenge_name(base, head));
        };
        const auto weight = require_descriptor(
            prefix + "w",
            shape,
            challenges);
        expect(
            source_scale(weight, prefix + "U") == FieldElement::one() &&
                source_scale(weight, prefix + "widehat_y_star") ==
                    output_challenge("lambda_out"),
            "output w challenge scale mismatch");

        const auto query_src = require_descriptor(
            prefix + "Query_src",
            shape,
            challenges);
        expect(
            source_scale(query_src, prefix + "E_src_edge") ==
                output_challenge("eta_src_out"),
            "output Query_src challenge scale mismatch");
    }

    std::uint64_t digest_values(
        const std::vector<FieldElement> &values)
    {
        std::uint64_t digest = 1469598103934665603ULL;
        std::array<std::uint8_t, 64> bytes{};
        for (const auto &value : values)
        {
            bytes.fill(0);
            (void)value.write_little_endian(bytes.data(), bytes.size());
            for (const auto byte : bytes)
            {
                digest ^= byte;
                digest *= 1099511628211ULL;
            }
        }
        return digest;
    }

    std::vector<FieldElement> evaluate_sources(
        const TEdgeDerivedLinearDescriptor &descriptor,
        const std::unordered_map<
            std::string,
            std::vector<FieldElement>> &values)
    {
        std::size_t count = 0;
        for (const auto &source : descriptor.sources)
        {
            const auto it = values.find(source.label);
            expect(it != values.end(), "host fixture source is missing");
            count = std::max(count, it->second.size());
        }
        std::vector<FieldElement> output(count, FieldElement::zero());
        for (const auto &source : descriptor.sources)
        {
            const auto &input = values.at(source.label);
            expect(input.size() == count, "host fixture source shape mismatch");
            for (std::size_t index = 0; index < count; ++index)
            {
                output[index] += source.scale * input[index];
            }
        }
        return output;
    }

    std::vector<FieldElement> run_host_executor(
        const TEdgeBoundProgram &program,
        const std::unordered_map<
            std::string,
            std::vector<FieldElement>> &values)
    {
        expect(program.instructions.size() == 3, "unexpected host fixture program");
        expect(program.terms.size() == 1, "unexpected host fixture term count");
        const auto &left = values.at(program.accesses[0].label);
        const auto &right = values.at(program.accesses[1].label);
        expect(left.size() == right.size(), "host executor source shape mismatch");
        std::vector<FieldElement> output(left.size(), FieldElement::zero());
        for (std::size_t index = 0; index < left.size(); ++index)
        {
            FieldElement stack_value = left[index];
            stack_value += right[index];
            output[index] = stack_value * program.terms[0].alpha_scale;
        }
        return output;
    }

    void check_flatten_status(
        TEdgeDerivedLinearFlattenStatus expected,
        const TEdgeDerivedLinearDescriptor &descriptor,
        const std::unordered_map<
            std::string,
            TEdgeDerivedLinearDescriptor> &nested,
        const std::function<bool(std::string_view)> &registered)
    {
        const auto lookup = [&](std::string_view label)
            -> std::optional<TEdgeDerivedLinearDescriptor>
        {
            const auto it = nested.find(std::string(label));
            return it == nested.end()
                       ? std::nullopt
                       : std::optional<TEdgeDerivedLinearDescriptor>(it->second);
        };
        const auto result =
            gatzk::protocol::internal::flatten_t_edge_derived_linear_sources(
                descriptor,
                lookup,
                registered);
        expect(result.status == expected, "unexpected flatten status");
    }

    void check_flattening()
    {
        const auto source = [](std::string label, std::uint64_t scale)
        {
            return TEdgeDerivedLinearSource{
                .label = std::move(label),
                .scale = FieldElement(scale)};
        };

        std::unordered_map<std::string, TEdgeDerivedLinearDescriptor> nested;
        nested.emplace(
            "d1",
            TEdgeDerivedLinearDescriptor{
                .target_label = "d1",
                .sources = {source("A", 1), source("B", 2)}});
        TEdgeDerivedLinearDescriptor root{
            .target_label = "root",
            .sources = {
                source("d1", 2),
                source("A", 0),
                TEdgeDerivedLinearSource{
                    .label = "A",
                    .scale = -FieldElement(2)}}};
        const auto lookup = [&](std::string_view label)
            -> std::optional<TEdgeDerivedLinearDescriptor>
        {
            const auto it = nested.find(std::string(label));
            return it == nested.end()
                       ? std::nullopt
                       : std::optional<TEdgeDerivedLinearDescriptor>(it->second);
        };
        const auto registered = [](std::string_view label)
        {
            return label == "A" || label == "B" || label == "C";
        };
        const auto flattened =
            gatzk::protocol::internal::flatten_t_edge_derived_linear_sources(
                root,
                lookup,
                registered);
        expect(
            flattened.status == TEdgeDerivedLinearFlattenStatus::Success,
            "flatten merge returned failure");
        expect(flattened.sources.size() == 1, "flatten did not remove zero term");
        expect(
            flattened.sources.front().label == "B" &&
                flattened.sources.front().scale == FieldElement(4),
            "flatten merge/cancel result mismatch");

        const TEdgeDerivedLinearDescriptor unknown{
            .target_label = "unknown_root",
            .sources = {source("missing", 1)}};
        check_flatten_status(
            TEdgeDerivedLinearFlattenStatus::UnknownSource,
            unknown,
            {},
            registered);

        nested.emplace(
            "cycle_a",
            TEdgeDerivedLinearDescriptor{
                .target_label = "cycle_a",
                .sources = {source("cycle_b", 1)}});
        nested.emplace(
            "cycle_b",
            TEdgeDerivedLinearDescriptor{
                .target_label = "cycle_b",
                .sources = {source("cycle_a", 1)}});
        const TEdgeDerivedLinearDescriptor cycle{
            .target_label = "cycle_root",
            .sources = {source("cycle_a", 1)}};
        check_flatten_status(
            TEdgeDerivedLinearFlattenStatus::Cycle,
            cycle,
            nested,
            registered);

        std::unordered_map<std::string, TEdgeDerivedLinearDescriptor> deep;
        for (std::size_t index = 0; index <= 64; ++index)
        {
            deep.emplace(
                "deep_" + std::to_string(index),
                TEdgeDerivedLinearDescriptor{
                    .target_label = "deep_" + std::to_string(index),
                    .sources = {source(
                        index == 64
                            ? "deep_leaf"
                            : "deep_" + std::to_string(index + 1),
                        1)}});
        }
        const TEdgeDerivedLinearDescriptor deep_root{
            .target_label = "deep_root",
            .sources = {source("deep_0", 1)}};
        const auto deep_registered = [](std::string_view label)
        {
            return label == "deep_leaf";
        };
        check_flatten_status(
            TEdgeDerivedLinearFlattenStatus::DepthExceeded,
            deep_root,
            deep,
            deep_registered);
    }

    void check_host_executor_fixture(
        const TEdgeDescriptorShape &shape,
        const std::map<std::string, FieldElement> &challenges)
    {
        const auto descriptor = require_descriptor(
            gatzk::protocol::hidden_head_prefix(0, 0) + "_S",
            shape,
            challenges);
        const auto left = gatzk::protocol::hidden_head_prefix(0, 0) +
                          "_E_src_edge";
        const auto right = gatzk::protocol::hidden_head_prefix(0, 0) +
                           "_E_dst_edge";
        std::unordered_map<std::string, std::vector<FieldElement>> values{
            {left, {FieldElement(2), FieldElement(3), FieldElement(5), FieldElement(7)}},
            {right, {FieldElement(11), FieldElement(13), FieldElement(17), FieldElement(19)}}};
        const auto host_descriptor_values = evaluate_sources(descriptor, values);

        TEdgeBoundProgram program;
        program.accesses = {
            TEdgeProgramAccess{.label = left, .rotated = false},
            TEdgeProgramAccess{.label = right, .rotated = false}};
        program.instructions = {
            TEdgeProgramInstruction{
                .opcode = TEdgeProgramOpcode::LoadAccess,
                .operand = 0},
            TEdgeProgramInstruction{
                .opcode = TEdgeProgramOpcode::LoadAccess,
                .operand = 1},
            TEdgeProgramInstruction{
                .opcode = TEdgeProgramOpcode::Add,
                .operand = 0}};
        program.terms = {
            gatzk::protocol::TEdgeProgramTerm{
                .instruction_begin = 0,
                .instruction_end = program.instructions.size(),
                .alpha_scale = FieldElement::one(),
                .required_accesses = {0, 1},
                .maximum_stack_depth = 2}};
        program.maximum_stack_depth = 2;

        const auto resident_output = run_host_executor(program, values);
        expect(
            resident_output == host_descriptor_values,
            "host descriptor and resident executor output differ");
        expect(
            digest_values(resident_output) == digest_values(host_descriptor_values),
            "host descriptor and resident executor digest differ");
        expect(digest_values(resident_output) != 0, "resident digest is empty");
    }
}

int main()
{
    try
    {
        const TEdgeDescriptorShape shape{
            .hidden_head_counts = {2, 1},
            .output_head_count = 2};
        std::map<std::string, FieldElement> challenges;
        add_challenges(&challenges, shape);

        for (std::size_t layer = 0;
             layer < shape.hidden_head_counts.size();
             ++layer)
        {
            for (std::size_t head = 0;
                 head < shape.hidden_head_counts[layer];
                 ++head)
            {
                check_hidden_mapping(shape, challenges, layer, head);
            }
        }
        for (std::size_t head = 0;
             head < shape.output_head_count;
             ++head)
        {
            check_output_mapping(shape, challenges, head);
        }

        const auto collision_name =
            gatzk::protocol::hidden_head_prefix(0, 0) + "_w_psq";
        const auto collision =
            gatzk::protocol::internal::make_t_edge_derived_linear_descriptor(
                collision_name,
                shape,
                challenges,
                [&](std::string_view name)
                { return name == collision_name; });
        expect(!collision.has_value(), "collision-sensitive descriptor accepted");
        const auto non_sensitive_collision =
            gatzk::protocol::internal::make_t_edge_derived_linear_descriptor(
                gatzk::protocol::hidden_head_prefix(0, 0) + "_S",
                shape,
                challenges,
                [](std::string_view)
                { return true; });
        expect(
            non_sensitive_collision.has_value(),
            "non-sensitive derived descriptor was suppressed by collision");
        expect(
            !gatzk::protocol::internal::make_t_edge_derived_linear_descriptor(
                 "P_h_l9_h9_S",
                 shape,
                 challenges)
                 .has_value(),
            "out-of-shape descriptor was accepted");

        check_flattening();
        check_host_executor_fixture(shape, challenges);

        std::cout << "t_edge_descriptor_runtime: PASS\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "t_edge_descriptor_runtime: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
