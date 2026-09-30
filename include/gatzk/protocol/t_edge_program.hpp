#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "gatzk/algebra/field.hpp"

namespace gatzk::protocol
{
    enum class TEdgeProgramOpcode : std::uint8_t
    {
        LoadAccess,
        LoadConstant,
        Add,
        Subtract,
        Multiply,
        Negate,
    };

    inline constexpr std::size_t kMaximumProgramStackDepth = 64;

    struct TEdgeProgramInstruction
    {
        TEdgeProgramOpcode opcode = TEdgeProgramOpcode::LoadConstant;

        std::uint32_t operand = 0;
    };

    struct TEdgeProgramAccess
    {
        std::string label;
        bool rotated = false;
    };

    struct TEdgeProgramTerm
    {
        std::size_t instruction_begin = 0;
        std::size_t instruction_end = 0;

        algebra::FieldElement alpha_scale = algebra::FieldElement::one();

        std::vector<std::uint32_t> required_accesses;

        std::size_t maximum_stack_depth = 0;
    };

    struct TEdgeBoundProgram
    {
        std::vector<TEdgeProgramInstruction> instructions;

        std::vector<algebra::FieldElement> constants;

        std::vector<TEdgeProgramAccess> accesses;

        std::vector<TEdgeProgramTerm> terms;

        std::size_t maximum_stack_depth = 0;
    };
}