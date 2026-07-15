#pragma once

#include <cstdint>
#include <vector>

#include "gatzk/protocol/proof.hpp"

namespace gatzk::protocol
{
    // Canonical wire codec for the nine top-level proof blocks.  Parsing is
    // strict: tags, order, lengths, scalar encodings and trailing bytes are all
    // checked before the verifier receives a Proof object.
    std::vector<std::uint8_t> serialize_proof(const Proof& proof);
    Proof parse_proof(const ProtocolContext& context, const std::vector<std::uint8_t>& bytes);
}
