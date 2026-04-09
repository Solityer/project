#pragma once

#include "gatzk/protocol/proof.hpp"

namespace gatzk::protocol {

ProtocolContext build_context(const util::AppConfig& config, RunMetrics* metrics = nullptr);
Proof prove(const ProtocolContext& context, const TraceArtifacts& trace, RunMetrics* metrics = nullptr);
std::size_t proof_size_bytes(const Proof& proof);
void export_run_artifacts(
    const ProtocolContext& context,
    const TraceArtifacts& trace,
    const Proof& proof,
    const RunMetrics& metrics,
    bool verified);

}
