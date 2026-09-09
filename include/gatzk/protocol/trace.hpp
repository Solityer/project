#pragma once
// 本文件声明 trace 构建入口，
// 负责把前向结果、lookup、route 与状态机对象物化为正式 witness 列。

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gatzk/protocol/proof.hpp"

namespace gatzk::protocol
{

    std::vector<std::size_t> stable_dynamic_commitment_chunk_order(
        const std::vector<std::vector<std::string>> &produced_labels,
        const std::vector<std::vector<std::string>> &required_labels);

    // Compare the independently computed integer reference matrix with the
    // secure-trace matrix through the production equality seam.
    bool quantized_reference_matches_secure(
        const model::Matrix &secure_trace_logits,
        const model::Matrix &reference_quant_logits);

    // Compute the exact integer attention numerators in canonical edge order.
    // The destination groups come from the formal edge ordering and let the
    // production trace avoid rescanning every edge for every node and column.
    std::vector<std::int64_t> weighted_edge_aggregate_raw_by_destination(
        const std::vector<std::int64_t> &edge_weights,
        const std::vector<std::vector<std::int64_t>> &source_rows,
        const std::vector<std::size_t> &edge_src_indices,
        const std::vector<std::size_t> &edge_dst_indices,
        const std::vector<std::pair<std::size_t, std::size_t>> &destination_groups,
        std::size_t node_count,
        std::string_view label);

    // Resolve the exact prover-side evaluation polynomial registered for a trace
    // commitment. Large columns may be file-backed after commitment, so consumers
    // must use this registry-aware path instead of assuming the commitment keeps a
    // resident opening polynomial.
    std::shared_ptr<const algebra::Polynomial> resolve_trace_opening_polynomial(
        const TraceArtifacts &trace,
        const std::string &label,
        const std::shared_ptr<algebra::RootOfUnityDomain> &expected_domain);

    // 构建 proving 主链需要的全部动态 trace 对象。
    // 它会同时产出动态承诺顺序、opening 取值来源、外点值与 witness 标量，是 prove 的唯一上游。
    // 这里若改变标签顺序，不仅会影响 prover / verifier，还会连带改变 proof_size_bytes 的统计对象。
    TraceArtifacts build_trace(
        const ProtocolContext &context,
        RunMetrics *metrics = nullptr);

#if defined(GATZK_BUILD_TESTS) && GATZK_BUILD_TESTS
    // Test-only production seam.  It accepts the same dense evaluation-column
    // shape used by trace materialization and forwards synchronously through
    // add_dynamic_commitment_batch -> KZG::commit_batch.  The type is omitted
    // from tests-OFF builds so it cannot become a protocol ABI.
    struct DynamicCommitmentBatchTestInput
    {
        std::string name;
        std::vector<algebra::FieldElement> values;
        // Optional exact immutable evaluation view used by spill/reuse
        // regressions.  The shared_ptr may be an aliasing view; production
        // reuse is valid only when both its control block and get() pointer
        // match another view.
        std::shared_ptr<const std::vector<algebra::FieldElement>> shared_values;
        // Optional deterministic blinder used only to compare tests-only A/B
        // lanes.  Formal production callers never see this field.
        std::optional<std::vector<algebra::FieldElement>>
            zero_domain_blinder_for_tests;
        std::shared_ptr<algebra::RootOfUnityDomain> domain;
    };

    // Test-only production seam for dense coefficient-basis matrix objects.
    // The matrix is flattened by the same materialize_commitment() path used
    // by hidden concat/output matrices, then forwarded synchronously through
    // add_dynamic_commitment_batch -> KZG::commit_batch.  Keeping this type
    // behind GATZK_BUILD_TESTS prevents a diagnostic harness from becoming a
    // production protocol ABI.
    struct DynamicCommitmentMatrixBatchTestInput
    {
        std::string name;
        model::Matrix matrix;
    };

    struct DynamicCommitmentBatchTestResult
    {
        std::vector<std::string> commitment_order;
        std::vector<crypto::Commitment> commitments;
        double wall_ms = 0.0;
        double scoped_commitment_ms = 0.0;
        double dynamic_commit_ms = 0.0;
        double dynamic_commit_msm_ms = 0.0;
        std::vector<DynamicCommitmentObjectProfile> object_profiles;
    };

    // Test-only seam for the production spill materializer.  The input view
    // may be an aliasing shared_ptr; spill reuse is valid only for the exact
    // immutable view identity (control block and get() pointer).
    struct SpillEvaluationTestInput
    {
        std::string name;
        std::shared_ptr<const std::vector<algebra::FieldElement>> shared_values;
        std::vector<algebra::FieldElement> zero_domain_blinder;
        std::shared_ptr<algebra::RootOfUnityDomain> domain;
    };

    DynamicCommitmentBatchTestResult add_dynamic_commitment_batch_for_tests(
        TraceArtifacts &trace,
        const std::vector<DynamicCommitmentBatchTestInput> &inputs,
        const crypto::KZGKeyPair &key,
        bool keep_trace_payloads = false);

    DynamicCommitmentBatchTestResult add_dynamic_commitment_matrix_batch_for_tests(
        TraceArtifacts &trace,
        const std::vector<DynamicCommitmentMatrixBatchTestInput> &inputs,
        const crypto::KZGKeyPair &key,
        bool keep_trace_payloads = false);

    TraceArtifacts::SpilledEvaluationPolynomial
    spill_evaluation_polynomial_for_tests(
        TraceArtifacts &trace,
        const SpillEvaluationTestInput &input);

    // Test-only A/B control for the exact immutable-view reuse seam.  The
    // production build has no mutable switch and always keeps the canonical
    // exact-view path enabled.
    void set_exact_shared_evaluation_view_reuse_for_tests(bool enabled);
    bool exact_shared_evaluation_view_reuse_for_tests();
#endif

} // namespace gatzk::protocol
