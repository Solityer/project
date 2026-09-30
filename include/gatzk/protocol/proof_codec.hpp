#pragma once

#include <cstdint>
#include <vector>

#include "gatzk/protocol/proof.hpp"

namespace gatzk::protocol
{
    std::vector<std::uint8_t> serialize_proof(const Proof &proof);
    Proof parse_proof(const VerifierContext &context, const std::vector<std::uint8_t> &bytes);
}
