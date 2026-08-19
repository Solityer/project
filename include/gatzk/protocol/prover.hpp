#pragma once
// 本文件声明 proving 主链的核心入口，
// 负责 context 构建、proof 生成、proof 大小统计与 artifact 导出。

#include <cstddef>
#include <memory>
#include <vector>

#include "gatzk/algebra/domain_evaluation.hpp"
#include "gatzk/protocol/proof.hpp"

namespace gatzk::protocol
{

    // Evaluate one immutable file-backed evaluation polynomial at several points
    // in one sequential pass. The weights must belong to `domain` and correspond
    // positionally to `points`.
    std::vector<algebra::FieldElement>
    evaluate_spilled_evaluation_polynomial_points(
        const TraceArtifacts::SpilledEvaluationPolynomial &spilled,
        const std::shared_ptr<algebra::RootOfUnityDomain> &domain,
        const std::vector<algebra::FieldElement> &points,
        const std::vector<const algebra::DomainEvaluationWeights *> &weights);

    // Single-point path used by lazy cache misses.
    algebra::FieldElement evaluate_spilled_evaluation_polynomial(
        const TraceArtifacts::SpilledEvaluationPolynomial &spilled,
        const std::shared_ptr<algebra::RootOfUnityDomain> &domain,
        const algebra::FieldElement &point,
        const algebra::DomainEvaluationWeights &weights);

    struct SpilledEvaluationRequest
    {
        const TraceArtifacts::SpilledEvaluationPolynomial *spilled = nullptr;
        std::shared_ptr<algebra::RootOfUnityDomain> domain;
        std::vector<algebra::FieldElement> points;
        std::vector<const algebra::DomainEvaluationWeights *> weights;
    };

    // Run independent spill-file scans with a hard four-worker ceiling.
    // Small or single-job batches remain serial.
    std::vector<std::vector<algebra::FieldElement>>
    evaluate_spilled_evaluation_requests(
        const std::vector<SpilledEvaluationRequest> &requests,
        std::size_t max_workers = 4);

    // 构建一次 formal run 的静态上下文，包括数据、模型、工作域、表与静态承诺。
    ProtocolContext build_context(
        const util::AppConfig &config,
        RunMetrics *metrics = nullptr);

    // 消费 trace 生成最终 proof。
    // 这里必须保持 commitment 顺序、challenge 顺序与 proof schema 稳定。
    Proof prove(
        const ProtocolContext &context,
        const TraceArtifacts &trace,
        RunMetrics *metrics = nullptr);

    // 统计正式 canonical proof 序列化口径。
    std::size_t proof_size_bytes(const Proof &proof);

    // 导出正式 proof、metadata、benchmark 和 run manifest。
    void export_run_artifacts(
        const ProtocolContext &context,
        const TraceArtifacts &trace,
        const Proof &proof,
        RunMetrics metrics,
        bool verified);

} // namespace gatzk::protocol