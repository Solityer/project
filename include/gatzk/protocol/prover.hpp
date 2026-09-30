#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gatzk/algebra/domain_evaluation.hpp"
#include "gatzk/protocol/proof.hpp"
#include "gatzk/protocol/verifier_context.hpp"

namespace gatzk::protocol
{

    std::vector<algebra::FieldElement> evaluate_spilled_evaluation_polynomial_points(const TraceArtifacts::SpilledEvaluationPolynomial &spilled,
                                                                                     const std::shared_ptr<algebra::RootOfUnityDomain> &domain,
                                                                                     const std::vector<algebra::FieldElement> &points,
                                                                                     const std::vector<const algebra::DomainEvaluationWeights *> &weights);

    algebra::FieldElement evaluate_spilled_evaluation_polynomial(const TraceArtifacts::SpilledEvaluationPolynomial &spilled,
                                                                 const std::shared_ptr<algebra::RootOfUnityDomain> &domain, const algebra::FieldElement &point,
                                                                 const algebra::DomainEvaluationWeights &weights);

    struct SpilledEvaluationRequest
    {
        const TraceArtifacts::SpilledEvaluationPolynomial *spilled = nullptr;
        std::shared_ptr<algebra::RootOfUnityDomain> domain;
        std::vector<algebra::FieldElement> points;
        std::vector<const algebra::DomainEvaluationWeights *> weights;
    };

    std::vector<std::vector<algebra::FieldElement>> evaluate_spilled_evaluation_requests(const std::vector<SpilledEvaluationRequest> &requests,
                                                                                         std::size_t max_workers = 4);

    ProtocolContext build_context(const util::AppConfig &config, RunMetrics *metrics = nullptr, bool retain_static_context_for_reuse = true);

    Proof prove(const ProtocolContext &context, const TraceArtifacts &trace, RunMetrics *metrics = nullptr);

    std::size_t proof_size_bytes(const Proof &proof);

    void export_run_artifacts(const ProtocolContext &context, const TraceArtifacts &trace, const Proof &proof, const VerifierContext &verifier_context,
                              const Proof &parsed_proof, RunMetrics metrics, bool verifier_accepted, bool final_accepted);

}
