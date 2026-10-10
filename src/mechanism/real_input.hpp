#pragma once

#include "gatzk/mechanism/internal.hpp"

namespace gatzk::mechanism::internal
{
    RealMechanismInput read_real_mechanism_input(const std::filesystem::path &path);
    std::vector<std::uint8_t> encode_real_mechanism_input(const RealMechanismInput &input);
    std::string real_mechanism_relation_digest(const RealMechanismInput &input, Mechanism mechanism, std::size_t width);
    std::string real_mechanism_witness_digest(const RealMechanismInput &input);
}
