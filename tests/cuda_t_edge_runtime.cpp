#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
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

    using gatzk::protocol::CudaTEdgeLinearSource;
    using gatzk::protocol::CudaTEdgeLeafLayout;

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

    std::vector<FieldElement> materialize_cpu_coset(
        const std::vector<FieldElement> &evaluations,
        const std::vector<FieldElement> &blinder,
        std::size_t domain_size,
        std::size_t target_size,
        const FieldElement &shift)
    {
        if (evaluations.size() != domain_size ||
            domain_size == 0 ||
            target_size == 0 ||
            target_size % domain_size != 0 ||
            (domain_size & (domain_size - 1U)) != 0 ||
            (target_size & (target_size - 1U)) != 0 ||
            blinder.size() > target_size - domain_size)
        {
            throw std::runtime_error(
                "invalid CUDA t_edge linear CPU fixture");
        }

        std::vector<FieldElement> coefficients = evaluations;
        gatzk::algebra::ntt_plan(domain_size)->intt(
            FieldElement::mutable_native_array_data(
                coefficients.data()));
        coefficients.resize(target_size, FieldElement::zero());
        for (std::size_t index = 0; index < blinder.size(); ++index)
        {
            coefficients[index] -= blinder[index];
            coefficients[domain_size + index] += blinder[index];
        }

        auto shift_power = FieldElement::one();
        for (auto &coefficient : coefficients)
        {
            coefficient *= shift_power;
            shift_power *= shift;
        }
        gatzk::algebra::ntt_plan(target_size)->ntt(
            FieldElement::mutable_native_array_data(
                coefficients.data()));
        return coefficients;
    }

    TEdgeBoundProgram make_linear_program()
    {
        TEdgeBoundProgram program;
        program.accesses = {
            TEdgeProgramAccess{
                .label = "derived",
                .rotated = false,
            },
        };
        program.instructions = {
            TEdgeProgramInstruction{
                .opcode = TEdgeProgramOpcode::LoadAccess,
                .operand = 0,
            },
        };
        program.terms = {
            TEdgeProgramTerm{
                .instruction_begin = 0,
                .instruction_end = 1,
                .alpha_scale = FieldElement::one(),
                .required_accesses = {0},
                .maximum_stack_depth = 1,
            },
        };
        program.maximum_stack_depth = 1;
        return program;
    }

    template <typename Function>
    void require_throw(Function &&function, const std::string &label)
    {
        try
        {
            function();
        }
        catch (const std::exception &)
        {
            return;
        }
        throw std::runtime_error(
            label + " unexpectedly accepted");
    }

    FieldElement evaluate_blinder(
        const std::vector<FieldElement> &blinder,
        const FieldElement &point)
    {
        FieldElement value = FieldElement::zero();
        for (auto it = blinder.rbegin(); it != blinder.rend(); ++it)
        {
            value *= point;
            value += *it;
        }
        return value;
    }

    void require_equal(
        const std::vector<FieldElement> &actual,
        const std::vector<FieldElement> &expected,
        const std::string &label);

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

    std::unique_ptr<gatzk::protocol::CudaTEdgeExecutor>
    make_linear_executor(
        std::size_t domain_size,
        std::size_t target_size,
        const FieldElement &shift)
    {
        auto executor = gatzk::protocol::create_cuda_t_edge_executor(
            domain_size,
            target_size,
            shift,
            FieldElement::root_of_unity(domain_size),
            3);
        if (executor == nullptr)
        {
            throw std::runtime_error(
                "CUDA t_edge linear executor factory returned null");
        }
        static const auto program = make_linear_program();
        executor->prepare_program(program);
        return executor;
    }

    std::vector<FieldElement> expected_linear_output(
        const std::vector<FieldElement> &source_split,
        const std::vector<FieldElement> &source_contiguous,
        const std::vector<FieldElement> &source_blinder_split,
        const std::vector<FieldElement> &source_blinder_contiguous,
        const std::vector<FieldElement> &target_blinder,
        const FieldElement &source_split_scale,
        const FieldElement &source_contiguous_scale,
        std::size_t domain_size,
        std::size_t target_size,
        const FieldElement &shift)
    {
        if (source_split.size() != target_size ||
            source_contiguous.size() != target_size)
        {
            throw std::runtime_error(
                "invalid CUDA t_edge linear expected source shape");
        }

        const auto root = FieldElement::root_of_unity(target_size);
        const auto one = FieldElement::one();
        auto point = shift;
        std::vector<FieldElement> expected(
            target_size,
            FieldElement::zero());

        for (std::size_t index = 0; index < target_size; ++index)
        {
            const auto zero = point.pow(domain_size) - one;
            if (zero.is_zero())
            {
                throw std::runtime_error(
                    "CUDA t_edge linear fixture overlaps base domain");
            }
            const auto target_value =
                source_split[index] * source_split_scale +
                source_contiguous[index] * source_contiguous_scale +
                zero *
                    (evaluate_blinder(target_blinder, point) -
                     source_split_scale *
                         evaluate_blinder(source_blinder_split, point) -
                     source_contiguous_scale *
                         evaluate_blinder(source_blinder_contiguous, point));
            expected[index] = target_value * zero.inv();
            point *= root;
        }
        return expected;
    }

    void materialize_linear_sources(
        gatzk::protocol::CudaTEdgeExecutor &executor,
        const std::vector<FieldElement> &source_evaluations,
        const std::vector<FieldElement> &source_blinder_split,
        const std::vector<FieldElement> &source_contiguous,
        std::size_t domain_size,
        std::size_t target_size,
        const FieldElement &shift)
    {
        if (!executor.materialize_evaluation_leaf(
                0,
                FieldElement::native_array_data(
                    source_evaluations.data()),
                source_evaluations.size(),
                FieldElement::native_array_data(
                    source_blinder_split.data()),
                source_blinder_split.size()))
        {
            throw std::runtime_error(
                "CUDA t_edge split source materialization declined");
        }
        executor.upload_leaf_chunk(
            1,
            0,
            FieldElement::native_array_data(
                source_contiguous.data()),
            target_size);
        (void)domain_size;
        (void)shift;
    }

    void run_linear_runtime_contracts()
    {
        constexpr std::size_t domain_size = 1024;
        constexpr std::size_t target_size = 4096;
        const FieldElement shift(7);

        std::vector<FieldElement> source_evaluations;
        source_evaluations.reserve(domain_size);
        for (std::size_t index = 0; index < domain_size; ++index)
        {
            source_evaluations.emplace_back(
                static_cast<std::uint64_t>(index + 13));
        }

        std::vector<FieldElement> source_contiguous;
        source_contiguous.reserve(target_size);
        for (std::size_t index = 0; index < target_size; ++index)
        {
            source_contiguous.emplace_back(
                static_cast<std::uint64_t>(index * 3 + 19));
        }

        const std::vector<FieldElement> source_blinder_split = {
            FieldElement(2),
            FieldElement(5),
        };
        const std::vector<FieldElement> source_blinder_contiguous = {
            FieldElement(7),
            FieldElement(11),
            FieldElement(13),
        };
        const std::vector<FieldElement> target_blinder = {
            FieldElement(17),
            FieldElement(19),
            FieldElement(23),
        };
        const auto source_split_coset = materialize_cpu_coset(
            source_evaluations,
            source_blinder_split,
            domain_size,
            target_size,
            shift);
        const auto split_scale = FieldElement(3);
        const auto contiguous_scale = FieldElement(8);
        const auto expected = expected_linear_output(
            source_split_coset,
            source_contiguous,
            source_blinder_split,
            source_blinder_contiguous,
            target_blinder,
            split_scale,
            contiguous_scale,
            domain_size,
            target_size,
            shift);

        auto executor = make_linear_executor(
            domain_size,
            target_size,
            shift);
        materialize_linear_sources(
            *executor,
            source_evaluations,
            source_blinder_split,
            source_contiguous,
            domain_size,
            target_size,
            shift);

        const std::vector<CudaTEdgeLinearSource> sources = {
            CudaTEdgeLinearSource{
                .slot = 0,
                .scale = split_scale,
                .zero_domain_blinder =
                    FieldElement::native_array_data(
                        source_blinder_split.data()),
                .blinder_count = source_blinder_split.size(),
            },
            CudaTEdgeLinearSource{
                .slot = 1,
                .scale = contiguous_scale,
                .zero_domain_blinder =
                    FieldElement::native_array_data(
                        source_blinder_contiguous.data()),
                .blinder_count = source_blinder_contiguous.size(),
            },
        };
        if (!executor->materialize_linear_leaf(
                2,
                sources,
                FieldElement::native_array_data(target_blinder.data()),
                target_blinder.size(),
                CudaTEdgeLeafLayout::Contiguous))
        {
            throw std::runtime_error(
                "CUDA t_edge linear materialization declined");
        }
        executor->accumulate_term(0, {2});
        std::vector<FieldElement> actual(
            target_size,
            FieldElement::zero());
        executor->finalize(actual.data(), actual.size());
        require_equal(
            actual,
            expected,
            "CUDA t_edge split/contiguous linear output");

        auto coefficient_executor = make_linear_executor(
            domain_size,
            target_size,
            shift);
        materialize_linear_sources(
            *coefficient_executor,
            source_evaluations,
            source_blinder_split,
            source_contiguous,
            domain_size,
            target_size,
            shift);
        if (!coefficient_executor->materialize_linear_leaf(
                2,
                sources,
                FieldElement::native_array_data(target_blinder.data()),
                target_blinder.size(),
                CudaTEdgeLeafLayout::Contiguous))
        {
            throw std::runtime_error(
                "CUDA t_edge coefficient linear materialization declined");
        }
        coefficient_executor->accumulate_term(0, {2});
        std::vector<FieldElement> actual_coefficients(
            target_size,
            FieldElement::zero());
        coefficient_executor->finalize_coefficients(
            actual_coefficients.data(),
            actual_coefficients.size());
        require_equal(
            actual_coefficients,
            expected_coefficients(expected, shift),
            "CUDA t_edge linear coefficient output");

        auto guards = make_linear_executor(
            domain_size,
            target_size,
            shift);
        materialize_linear_sources(
            *guards,
            source_evaluations,
            source_blinder_split,
            source_contiguous,
            domain_size,
            target_size,
            shift);

        require_throw(
            [&]() {
                guards->materialize_linear_leaf(
                    2,
                    {sources[0], sources[0]},
                    FieldElement::native_array_data(target_blinder.data()),
                    target_blinder.size(),
                    CudaTEdgeLeafLayout::Contiguous);
            },
            "CUDA t_edge duplicate source guard");
        require_throw(
            [&]() {
                guards->materialize_linear_leaf(
                    2,
                    {CudaTEdgeLinearSource{
                        .slot = 3,
                        .scale = split_scale,
                        .zero_domain_blinder = nullptr,
                        .blinder_count = 0,
                    }},
                    nullptr,
                    0,
                    CudaTEdgeLeafLayout::Contiguous);
            },
            "CUDA t_edge missing source guard");
        require_throw(
            [&]() {
                guards->materialize_linear_leaf(
                    2,
                    {CudaTEdgeLinearSource{
                        .slot = 0,
                        .scale = FieldElement::zero(),
                        .zero_domain_blinder =
                            FieldElement::native_array_data(
                                source_blinder_split.data()),
                        .blinder_count = source_blinder_split.size(),
                    }},
                    nullptr,
                    0,
                    CudaTEdgeLeafLayout::Contiguous);
            },
            "CUDA t_edge zero scale guard");
        require_throw(
            [&]() {
                guards->materialize_linear_leaf(
                    2,
                    {sources[0]},
                    FieldElement::native_array_data(target_blinder.data()),
                    target_size - domain_size + 1,
                    CudaTEdgeLeafLayout::Contiguous);
            },
            "CUDA t_edge blinder shape guard");
        require_throw(
            [&]() {
                guards->materialize_linear_leaf(
                    2,
                    {sources[0]},
                    nullptr,
                    0,
                    static_cast<CudaTEdgeLeafLayout>(99));
            },
            "CUDA t_edge output layout guard");
        require_throw(
            [&]() {
                guards->upload_leaf_chunk(
                    2,
                    1,
                    FieldElement::native_array_data(
                        source_contiguous.data()),
                    1);
            },
            "CUDA t_edge first offset guard");

        auto incomplete = make_linear_executor(
            domain_size,
            target_size,
            shift);
        incomplete->upload_leaf_chunk(
            2,
            0,
            FieldElement::native_array_data(source_contiguous.data()),
            domain_size);
        require_throw(
            [&]() {
                incomplete->accumulate_term(0, {2});
            },
            "CUDA t_edge incomplete binding guard");
        require_throw(
            [&]() {
                incomplete->materialize_linear_leaf(
                    2,
                    {sources[0]},
                    nullptr,
                    0,
                    CudaTEdgeLeafLayout::Contiguous);
            },
            "CUDA t_edge partial target overwrite guard");
        incomplete->upload_leaf_chunk(
            2,
            domain_size,
            FieldElement::native_array_data(
                source_contiguous.data()) + domain_size,
            target_size - domain_size);
        incomplete->accumulate_term(0, {2});
        incomplete->release_leaf(2);
        require_throw(
            [&]() {
                incomplete->accumulate_term(0, {2});
            },
            "CUDA t_edge released binding guard");
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

    constexpr std::size_t target_size_2_24 =
        std::size_t{1} << 24;
    constexpr std::size_t target_size_2_25 =
        std::size_t{1} << 25;
    constexpr std::size_t target_size_2_26 =
        std::size_t{1} << 26;

    if (gatzk::protocol::cuda_t_edge_executor_can_admit(
            0,
            1,
            false) ||
        gatzk::protocol::cuda_t_edge_executor_can_admit(
            3,
            1,
            false) ||
        gatzk::protocol::cuda_t_edge_executor_can_admit(
            std::size_t{1} << (sizeof(std::size_t) * 8 - 1),
            1,
            false) ||
        gatzk::protocol::cuda_t_edge_executor_can_admit(
            std::size_t{1} << 16,
            std::numeric_limits<std::uint32_t>::max(),
            false))
    {
        throw std::runtime_error(
            "CUDA t_edge admission accepted an invalid/overflow shape");
    }

    // Admission is deliberately exercised before executor construction.  A
    // checked shape overflow and a deterministic reduced-memory request must
    // both reject without launching a kernel or creating a partial executor.
    const auto admission_kernel_before =
        gatzk::algebra::cuda_kernel_invocation_count();

    gatzk::protocol::CudaTEdgeExecutorAdmissionShape
        overflow_shape{
            .domain_size = target_size_2_24,
            .instruction_count =
                std::numeric_limits<std::size_t>::max()};

    if (gatzk::protocol::cuda_t_edge_executor_can_admit(
            target_size_2_24,
            1,
            true,
            overflow_shape))
    {
        throw std::runtime_error(
            "CUDA t_edge admission accepted checked-arithmetic overflow");
    }

    auto reduced_memory_shape =
        gatzk::protocol::CudaTEdgeExecutorAdmissionShape{
            .domain_size = target_size_2_24,
            .instruction_count = 1,
            .constant_count = 1,
            .access_count = 2,
            .caller_host_bytes =
                std::size_t{1} << 60};

    if (gatzk::protocol::cuda_t_edge_executor_can_admit(
            target_size_2_24,
            1,
            true,
            reduced_memory_shape))
    {
        throw std::runtime_error(
            "CUDA t_edge admission accepted reduced-memory shape");
    }

    const auto admission_kernel_after =
        gatzk::algebra::cuda_kernel_invocation_count();
    if (admission_kernel_after != admission_kernel_before)
    {
        throw std::runtime_error(
            "CUDA t_edge admission launched a kernel before allocation");
    }

    std::cout
        << "CUDA_T_EDGE_ADMISSION_FAIL_CLOSED_OK overflow=1 reduced_memory=1 "
        << "kernel_delta=0\n";

    // Exact leaves and final coefficient conversion use the bounded transform
    // over target_size / 2.  The direct 2^24 lane boundary and the staged
    // 2^25/2^26 lane extensions must all be advertised, while non-powers and
    // targets beyond the bounded 2^26 lane remain rejected.
    if (!gatzk::protocol::cuda_t_edge_leaf_exact_ntt_supported(
            target_size_2_24) ||
        !gatzk::protocol::cuda_t_edge_leaf_exact_ntt_supported(
            target_size_2_25) ||
        !gatzk::protocol::cuda_t_edge_leaf_exact_ntt_supported(
            target_size_2_26) ||
        !gatzk::protocol::cuda_t_edge_leaf_exact_ntt_supported(
            std::size_t{1} << 27) ||
        gatzk::protocol::cuda_t_edge_leaf_exact_ntt_supported(
            std::size_t{1} << 28) ||
        gatzk::protocol::cuda_t_edge_leaf_exact_ntt_supported(3))
    {
        throw std::runtime_error(
            "CUDA t_edge leaf NTT capability boundary is incorrect");
    }

    // Admission must cover both exact leaf materialization and the final
    // device-resident quotient -> coefficient conversion.  A stale false
    // capability flag must still fail closed; resource-dependent true cases
    // are reported below rather than treated as unconditional pass/fail.
    const auto admit_2_24 =
        gatzk::protocol::cuda_t_edge_executor_can_admit(
            target_size_2_24,
            1,
            true);
    const auto admit_2_25 =
        gatzk::protocol::cuda_t_edge_executor_can_admit(
            target_size_2_25,
            1,
            true);
    if (gatzk::protocol::cuda_t_edge_executor_can_admit(
            target_size_2_26,
            1,
            false) ||
        gatzk::protocol::cuda_t_edge_executor_can_admit(
            std::size_t{1} << 28,
            1,
            true))
    {
        throw std::runtime_error(
            "CUDA t_edge admission accepted a stale unsupported capability");
    }

    std::cout
        << "CUDA_T_EDGE_CAPABILITY_OK target_2^24="
        << (admit_2_24 ? 1 : 0)
        << " target_2^25="
        << (admit_2_25 ? 1 : 0)
        << " target_2^26="
        << (gatzk::protocol::cuda_t_edge_executor_can_admit(
                target_size_2_26,
                1,
                true)
                ? 1
                : 0)
        << "\n";

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

    run_linear_runtime_contracts();

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
