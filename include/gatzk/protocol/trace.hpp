#pragma once
// 本文件声明 trace 构建入口，
// 负责把前向结果、lookup、route 与状态机对象物化为正式 witness 列。

#include <cstddef>
#include <string>
#include <vector>

#include "gatzk/protocol/proof.hpp"

namespace gatzk::protocol {

std::vector<std::size_t> stable_dynamic_commitment_chunk_order(
    const std::vector<std::vector<std::string>>& produced_labels,
    const std::vector<std::vector<std::string>>& required_labels);

// Resolve the exact prover-side evaluation polynomial registered for a trace
// commitment. Large columns may be file-backed after commitment, so consumers
// must use this registry-aware path instead of assuming the commitment keeps a
// resident opening polynomial.
std::shared_ptr<const algebra::Polynomial> resolve_trace_opening_polynomial(
    const TraceArtifacts& trace,
    const std::string& label,
    const std::shared_ptr<algebra::RootOfUnityDomain>& expected_domain);

// 构建 proving 主链需要的全部动态 trace 对象。
// 它会同时产出动态承诺顺序、opening 取值来源、外点值与 witness 标量，是 prove 的唯一上游。
// 这里若改变标签顺序，不仅会影响 prover / verifier，还会连带改变 proof_size_bytes 的统计对象。
TraceArtifacts build_trace(const ProtocolContext& context, RunMetrics* metrics = nullptr);

}  // namespace gatzk::protocol
