#pragma once
// 本文件声明 proving 主链的核心入口，
// 负责 context 构建、proof 生成、proof 大小统计与 artifact 导出。

#include "gatzk/protocol/proof.hpp"
#include "gatzk/algebra/domain_evaluation.hpp"

namespace gatzk::protocol {

// Execution-only diagnostics for file-backed evaluation-polynomial reads.
// These counters are not serialized and do not affect proof construction.
struct SpilledEvaluationProfile {
    std::size_t file_open_count = 0;
    std::size_t sequential_pass_count = 0;
    std::size_t direct_values_resolved = 0;
    std::size_t field_elements_streamed = 0;
};

// Evaluate one immutable file-backed evaluation polynomial at several points
// in one sequential pass. The weights must belong to `domain` and correspond
// positionally to `points`.
std::vector<algebra::FieldElement>
evaluate_spilled_evaluation_polynomial_points(
    const TraceArtifacts::SpilledEvaluationPolynomial& spilled,
    const std::shared_ptr<algebra::RootOfUnityDomain>& domain,
    const std::vector<algebra::FieldElement>& points,
    const std::vector<const algebra::DomainEvaluationWeights*>& weights,
    SpilledEvaluationProfile* profile = nullptr);

// Single-point compatibility path used by lazy cache misses and focused
// equivalence checks.
algebra::FieldElement evaluate_spilled_evaluation_polynomial(
    const TraceArtifacts::SpilledEvaluationPolynomial& spilled,
    const std::shared_ptr<algebra::RootOfUnityDomain>& domain,
    const algebra::FieldElement& point,
    const algebra::DomainEvaluationWeights& weights,
    SpilledEvaluationProfile* profile = nullptr);

struct SpilledEvaluationRequest {
    const TraceArtifacts::SpilledEvaluationPolynomial* spilled = nullptr;
    std::shared_ptr<algebra::RootOfUnityDomain> domain;
    std::vector<algebra::FieldElement> points;
    std::vector<const algebra::DomainEvaluationWeights*> weights;
};

struct SpilledEvaluationBatchProfile {
    std::size_t worker_count = 0;
    std::vector<SpilledEvaluationProfile> jobs;
};

// Run independent spill-file scans with a hard four-worker ceiling. Small or
// single-job batches remain serial so this does not create nested fine-grained
// parallelism in ordinary opening paths.
std::vector<std::vector<algebra::FieldElement>>
evaluate_spilled_evaluation_requests(
    const std::vector<SpilledEvaluationRequest>& requests,
    std::size_t max_workers = 4,
    SpilledEvaluationBatchProfile* profile = nullptr);

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
