#pragma once
// 本文件声明 trace 构建入口，
// 负责把前向结果、lookup、route 与状态机对象物化为正式 witness 列。

#include "gatzk/protocol/proof.hpp"

namespace gatzk::protocol {

// 构建 proving 主链需要的全部动态 trace 对象。
// 它会同时产出动态承诺顺序、外点值与 witness 标量，是 prove 的唯一上游。
TraceArtifacts build_trace(const ProtocolContext& context, RunMetrics* metrics = nullptr);

}  // namespace gatzk::protocol
