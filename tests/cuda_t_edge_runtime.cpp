#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "gatzk/algebra/field.hpp"
#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/protocol/cuda_t_edge.hpp"
#include "gatzk/protocol/t_edge_program.hpp"

namespace
{
    using gatzk::algebra::FieldElement;
    using gatzk::protocol::TEdgeBoundProgram;
    using gatzk::protocol::TEdgeProgramAccess;
    using gatzk::protocol::TEdgeProgramInstruction;
    using gatzk::protocol::TEdgeProgramOpcode;
    using gatzk::protocol::TEdgeProgramTerm;

    TEdgeBoundProgram make_program()
    {
        TEdgeBoundProgram program;

        program.accesses = {
            TEdgeProgramAccess{
                .label = "A",
                .rotated = false,
            },
            TEdgeProgramAccess{
                .label = "B",
                .rotated = true,
            },
        };

        program.constants = {
            FieldElement(3),
        };

        program.instructions = {
            TEdgeProgramInstruction{
                .opcode =
                    TEdgeProgramOpcode::LoadAccess,
                .operand = 0,
            },
            TEdgeProgramInstruction{
                .opcode =
                    TEdgeProgramOpcode::LoadAccess,
                .operand = 1,
            },
            TEdgeProgramInstruction{
                .opcode =
                    TEdgeProgramOpcode::Add,
                .operand = 0,
            },
            TEdgeProgramInstruction{
                .opcode =
                    TEdgeProgramOpcode::LoadConstant,
                .operand = 0,
            },
            TEdgeProgramInstruction{
                .opcode =
                    TEdgeProgramOpcode::Multiply,
                .operand = 0,
            },
        };

        program.terms = {
            TEdgeProgramTerm{
                .instruction_begin = 0,
                .instruction_end =
                    program.instructions.size(),
                .alpha_scale =
                    FieldElement(5),
                .required_accesses = {
                    0,
                    1,
                },
                .maximum_stack_depth = 2,
            },
        };

        program.maximum_stack_depth = 2;

        return program;
    }

    std::vector<FieldElement>
    expected_evaluations(
        const std::vector<FieldElement> &a,
        const std::vector<FieldElement> &b,
        std::size_t domain_size,
        std::size_t target_size,
        const FieldElement &shift)
    {
        if (a.size() != target_size ||
            b.size() != target_size ||
            domain_size == 0 ||
            target_size == 0 ||
            target_size % domain_size != 0 ||
            (target_size &
             (target_size - 1U)) != 0U)
        {
            throw std::runtime_error(
                "invalid CUDA t_edge runtime fixture");
        }

        const auto root =
            FieldElement::root_of_unity(
                target_size);

        const auto rotation_step =
            target_size /
            domain_size;

        const auto one =
            FieldElement::one();

        const auto term_scale =
            FieldElement(15);

        std::vector<FieldElement>
            expected(
                target_size,
                FieldElement::zero());

        auto point = shift;

        for (std::size_t index = 0;
             index < target_size;
             ++index)
        {
            const auto rotated_index =
                (index + rotation_step) &
                (target_size - 1U);

            const auto denominator =
                point.pow(domain_size) -
                one;

            if (denominator.is_zero())
            {
                throw std::runtime_error(
                    "CUDA t_edge runtime fixture coset "
                    "overlaps the base domain");
            }

            expected[index] =
                (a[index] +
                 b[rotated_index]) *
                term_scale *
                denominator.inv();

            point *= root;
        }

        return expected;
    }

    std::vector<FieldElement>
    expected_coefficients(
        std::vector<FieldElement>
            evaluations,
        const FieldElement &shift)
    {
        if (evaluations.empty())
        {
            throw std::runtime_error(
                "CUDA t_edge coefficient fixture "
                "is empty");
        }

        auto *native =
            FieldElement::
                mutable_native_array_data(
                    evaluations.data());

        gatzk::algebra::
            ntt_plan(
                evaluations.size())
                ->intt(native);

        const auto inverse_shift =
            shift.inv();

        auto inverse_shift_power =
            FieldElement::one();

        for (auto &coefficient :
             evaluations)
        {
            coefficient *=
                inverse_shift_power;

            inverse_shift_power *=
                inverse_shift;
        }

        return evaluations;
    }

    std::unique_ptr<
        gatzk::protocol::CudaTEdgeExecutor>
    make_executor(
        std::size_t domain_size,
        std::size_t target_size,
        const FieldElement &shift)
    {
        return gatzk::protocol::
            create_cuda_t_edge_executor(
                domain_size,
                target_size,
                shift,
                FieldElement::
                    root_of_unity(
                        domain_size),
                2);
    }

    void upload_fixture(
        gatzk::protocol::CudaTEdgeExecutor
            &executor,
        const TEdgeBoundProgram &program,
        const std::vector<FieldElement> &a,
        const std::vector<FieldElement> &b)
    {
        executor.prepare_program(
            program);

        executor.upload_leaf_chunk(
            0,
            0,
            FieldElement::
                native_array_data(
                    a.data()),
            a.size());

        executor.upload_leaf_chunk(
            1,
            0,
            FieldElement::
                native_array_data(
                    b.data()),
            b.size());

        executor.accumulate_term(
            0,
            std::vector<std::uint32_t>{
                0,
                1,
            });
    }

    void require_equal(
        const std::vector<FieldElement>
            &actual,
        const std::vector<FieldElement>
            &expected,
        const std::string &label)
    {
        if (actual.size() !=
            expected.size())
        {
            throw std::runtime_error(
                label +
                " size mismatch");
        }

        for (std::size_t index = 0;
             index < actual.size();
             ++index)
        {
            if (!(actual[index] ==
                  expected[index]))
            {
                throw std::runtime_error(
                    label +
                    " mismatch at index " +
                    std::to_string(index));
            }
        }
    }
}

int main()
{
    if (!gatzk::protocol::
            cuda_t_edge_executor_available())
    {
        std::cout
            << "SKIP cuda_t_edge runtime unavailable\n";

        return 77;
    }

    constexpr std::size_t
        domain_size = 1024;

    constexpr std::size_t
        target_size = 2048;

    const FieldElement shift(7);

    std::vector<FieldElement> a;
    std::vector<FieldElement> b;

    a.reserve(target_size);
    b.reserve(target_size);

    for (std::size_t index = 0;
         index < target_size;
         ++index)
    {
        a.emplace_back(
            static_cast<std::uint64_t>(
                index + 1));

        b.emplace_back(
            static_cast<std::uint64_t>(
                index + 101));
    }

    const auto program =
        make_program();

    const auto expected =
        expected_evaluations(
            a,
            b,
            domain_size,
            target_size,
            shift);

    const auto kernel_count_before =
        gatzk::algebra::
            cuda_kernel_invocation_count();

    auto evaluation_executor =
        make_executor(
            domain_size,
            target_size,
            shift);

    if (evaluation_executor ==
        nullptr)
    {
        throw std::runtime_error(
            "CUDA t_edge executor advertised "
            "availability but factory returned null");
    }

    upload_fixture(
        *evaluation_executor,
        program,
        a,
        b);

    std::vector<FieldElement>
        actual_evaluations(
            target_size,
            FieldElement::zero());

    evaluation_executor->finalize(
        actual_evaluations.data(),
        actual_evaluations.size());

    require_equal(
        actual_evaluations,
        expected,
        "CUDA t_edge evaluation output");

    auto coefficient_executor =
        make_executor(
            domain_size,
            target_size,
            shift);

    if (coefficient_executor ==
        nullptr)
    {
        throw std::runtime_error(
            "CUDA t_edge coefficient executor "
            "factory returned null");
    }

    upload_fixture(
        *coefficient_executor,
        program,
        a,
        b);

    std::vector<FieldElement>
        actual_coefficients(
            target_size,
            FieldElement::zero());

    coefficient_executor->finalize_coefficients(
        actual_coefficients.data(),
        actual_coefficients.size());

    const auto
        expected_coefficient_values =
            expected_coefficients(
                expected,
                shift);

    require_equal(
        actual_coefficients,
        expected_coefficient_values,
        "CUDA t_edge coefficient output");

    const auto kernel_count_after =
        gatzk::algebra::
            cuda_kernel_invocation_count();

    if (kernel_count_after <=
        kernel_count_before)
    {
        throw std::runtime_error(
            "CUDA t_edge runtime regression "
            "executed zero CUDA kernels");
    }

    std::cout
        << "CUDA_T_EDGE_RUNTIME_OK kernels="
        << (kernel_count_after -
            kernel_count_before)
        << '\n';

    return 0;
}