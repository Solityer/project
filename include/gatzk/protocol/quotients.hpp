#pragma once

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "gatzk/algebra/eval_backend.hpp"
#include "gatzk/algebra/field.hpp"
#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/protocol/proof.hpp"

namespace gatzk::protocol
{

    using EvalFn = std::function<algebra::FieldElement(const std::string &, const algebra::FieldElement &)>;

    using TNPublicSourceFoldEvalFn = std::function<algebra::FieldElement(const std::string &, std::size_t)>;

    struct FHQuotientProfile
    {
        double dependency_eval_ms = 0.0;
        double assembly_ms = 0.0;
    };

    struct TEdgeEvaluationPlan;
    struct TNEvaluationPlan;

    struct QuotientBoundaryValues
    {
        algebra::FieldElement first;
        algebra::FieldElement last;
        algebra::FieldElement zero;
        std::optional<algebra::FieldElement> zero_inverse;
    };

    struct FHQuotientParameters
    {
        algebra::FieldElement alpha_quot;
        algebra::FieldElement beta_feat;
        algebra::FieldElement eta_feat;
    };

    struct FHQuotientValues
    {
        algebra::FieldElement q_table;
        algebra::FieldElement q_query;
        algebra::FieldElement table;
        algebra::FieldElement query;
        algebra::FieldElement multiplicity;
        algebra::FieldElement j_table;
        algebra::FieldElement j_query;
        algebra::FieldElement accumulator;
        algebra::FieldElement accumulator_next;
    };

    struct FHFeatureEncodingValues
    {
        algebra::FieldElement table_row;
        algebra::FieldElement table_column;
        algebra::FieldElement table_feature;
        algebra::FieldElement query_row;
        algebra::FieldElement query_column;
        algebra::FieldElement query_feature;
    };

    struct TEdgeBoundProgram;
    struct TEdgeEvaluationPlan;
    TEdgeBoundProgram build_t_edge_bound_program(const ProtocolContext &context, const std::map<std::string, algebra::FieldElement> &challenges,
                                                 const std::map<std::string, algebra::FieldElement> &external_evaluations, const TEdgeEvaluationPlan &plan);

    algebra::FieldElement evaluate_t_fh_from_values(const FHQuotientParameters &parameters, const FHQuotientValues &values,
                                                    const QuotientBoundaryValues &boundary_values,
                                                    const FHFeatureEncodingValues *feature_encoding_values = nullptr);

    std::shared_ptr<const TEdgeEvaluationPlan> build_t_edge_evaluation_plan(const ProtocolContext &context,
                                                                            const std::map<std::string, algebra::FieldElement> &challenges);

    std::shared_ptr<const TNEvaluationPlan> build_t_n_evaluation_plan(const ProtocolContext &context,
                                                                      const std::map<std::string, algebra::FieldElement> &challenges);

    using DomainOpeningLabelRegistry = std::map<std::string, std::vector<std::string>>;

    DomainOpeningLabelRegistry build_domain_opening_label_registry(const ProtocolContext &context);

    std::vector<std::string> domain_opening_labels(const ProtocolContext &context, const std::string &domain_name);

    std::optional<std::string> dynamic_commitment_domain_name(const ProtocolContext &context, const std::string &label);

    algebra::FieldElement evaluate_geometric_power_column(const algebra::RootOfUnityDomain &domain, const algebra::FieldElement &gamma,
                                                          const algebra::FieldElement &point);

    algebra::FieldElement evaluate_t_fh(const ProtocolContext &context, const std::map<std::string, algebra::FieldElement> &challenges, const EvalFn &eval,
                                        const algebra::FieldElement &z, FHQuotientProfile *profile = nullptr,
                                        const QuotientBoundaryValues *boundary_values = nullptr, bool feature_encoding_terms_are_identically_zero = false);

    algebra::FieldElement evaluate_t_edge(const ProtocolContext &context, const std::map<std::string, algebra::FieldElement> &challenges,
                                          const std::map<std::string, algebra::FieldElement> &external_evaluations, const EvalFn &eval,
                                          const algebra::FieldElement &z, bool return_numerator = false, std::size_t *first_nonzero_term = nullptr,
                                          algebra::FieldElement *first_nonzero_value = nullptr,
                                          const std::vector<algebra::FieldElement> *precomputed_alpha_powers = nullptr,
                                          const TEdgeEvaluationPlan *evaluation_plan = nullptr, const QuotientBoundaryValues *boundary_values = nullptr);

    algebra::FieldElement evaluate_t_n(const ProtocolContext &context, const std::map<std::string, algebra::FieldElement> &challenges, const EvalFn &eval,
                                       const algebra::FieldElement &z, const TNEvaluationPlan *evaluation_plan = nullptr,
                                       const QuotientBoundaryValues *boundary_values = nullptr,
                                       const TNPublicSourceFoldEvalFn *public_source_fold_eval = nullptr, std::size_t evaluation_index = 0);

    algebra::FieldElement evaluate_t_in(const ProtocolContext &context, const std::map<std::string, algebra::FieldElement> &challenges,
                                        const std::map<std::string, algebra::FieldElement> &external_evaluations, const EvalFn &eval,
                                        const algebra::FieldElement &z, bool return_numerator = false, std::size_t *first_nonzero_term = nullptr,
                                        algebra::FieldElement *first_nonzero_value = nullptr);

    algebra::FieldElement evaluate_t_d(const ProtocolContext &context, const std::map<std::string, algebra::FieldElement> &challenges,
                                       const std::map<std::string, algebra::FieldElement> &external_evaluations, const EvalFn &eval,
                                       const algebra::FieldElement &z, bool return_numerator = false, std::size_t *first_nonzero_term = nullptr,
                                       algebra::FieldElement *first_nonzero_value = nullptr);

    algebra::FieldElement evaluate_t_cat(const ProtocolContext &context, const std::map<std::string, algebra::FieldElement> &challenges,
                                         const std::map<std::string, algebra::FieldElement> &external_evaluations, const EvalFn &eval,
                                         const algebra::FieldElement &z);

    algebra::FieldElement evaluate_t_c(const ProtocolContext &context, const std::map<std::string, algebra::FieldElement> &challenges,
                                       const std::map<std::string, algebra::FieldElement> &external_evaluations, const EvalFn &eval,
                                       const algebra::FieldElement &z);

}
