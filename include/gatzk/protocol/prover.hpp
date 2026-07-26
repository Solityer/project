#pragma once
// 本文件声明 proving 主链的核心入口，
// 负责 context 构建、proof 生成、proof 大小统计与 artifact 导出。

#include "gatzk/protocol/proof.hpp"

namespace gatzk::protocol {

// 构建一次 formal run 的静态上下文，包括数据、模型、工作域、表与静态承诺。
ProtocolContext build_context(const util::AppConfig& config, RunMetrics* metrics = nullptr);
// 消费 trace 生成最终 proof。这里必须保持 commitment 顺序、challenge 顺序与 proof schema 稳定。
Proof prove(const ProtocolContext& context, const TraceArtifacts& trace, RunMetrics* metrics = nullptr);
// 统计 proof 序列化口径，是 README / manifests / tests 共用的 proof_size_bytes 真值来源。
std::size_t proof_size_bytes(const Proof& proof);
// 将一次正式运行导出为三层 artifact：
// canonical `proof.bin`、`metadata.txt` / `proof_block_order.txt` 等 formal 附件，
// `benchmark.txt` 的最小正式摘要，
// 以及供 formal suite / artifact 校验复用的完整 `run_manifest.json`。
void export_run_artifacts(
    const ProtocolContext& context,
    const TraceArtifacts& trace,
    const Proof& proof,
    RunMetrics metrics,
    bool verified);

}
