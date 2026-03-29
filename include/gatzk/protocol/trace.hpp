#pragma once

#include "gatzk/protocol/proof.hpp"

namespace gatzk::protocol {

TraceArtifacts build_trace(const ProtocolContext& context, RunMetrics* metrics = nullptr);

}  // namespace gatzk::protocol
