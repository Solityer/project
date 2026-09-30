#pragma once

#include <cstddef>
#include <vector>

#include "gatzk/algebra/domain_evaluation.hpp"
#include "gatzk/protocol/proof.hpp"

namespace gatzk::protocol::internal
{

    // Execution-only cache for public aggregation multiplicities.  This header
    // lives outside the installed include tree: it is shared by the verifier
    // implementation and its focused regression without becoming protocol API.
    struct AggregationMultiplicityCache
    {
        std::vector<algebra::FieldElement> src;
        std::vector<algebra::FieldElement> dst;
    };

    AggregationMultiplicityCache build_aggregation_multiplicity_cache(const data::LocalGraph &local, std::size_t n_domain_size);

    algebra::FieldElement evaluate_deterministic_public_polynomial(const ProtocolContext &context, const VerifierPublicPolynomialDescriptor &descriptor,
                                                                   const algebra::FieldElement &point, const AggregationMultiplicityCache &multiplicity,
                                                                   algebra::DomainEvaluationWeightCache &weight_cache);

} // namespace gatzk::protocol::internal
