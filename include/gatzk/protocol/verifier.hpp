#pragma once

#include "gatzk/protocol/proof.hpp"

namespace gatzk::protocol {

bool verify(const ProtocolContext& context, const Proof& proof);

}  // namespace gatzk::protocol
