#pragma once
// 本文件声明各工作域 quotient 的求值逻辑，
// verifier 会按同一接口重算这些值来检查 proof 中的 opening 结果。

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

    // 统一的“按标签在点上取值”回调接口，供 quotient 逻辑屏蔽 prove / verify 的数据来源差异。
    using EvalFn = std::function<algebra::FieldElement(const std::string &, const algebra::FieldElement &)>;

    // Prover-side t_N materialization may replace the repeated public-source
    // barycentric interpolation with the exact same public polynomial already
    // materialized on the quotient coset.  The verifier deliberately leaves
    // this callback unset and evaluates the source-of-truth equation directly.
    using TNPublicSourceFoldEvalFn =
        std::function<algebra::FieldElement(
            const std::string &,
            std::size_t)>;

    // `t_FH` 额外输出的执行侧剖面。
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

    // `t_FH` point evaluation is intentionally split into named-value collection
    // and one equation core.  The verifier collects these values through `EvalFn`;
    // the quotient materializer can bind its already-authenticated coset vectors
    // once and feed the same core without per-point string/function dispatch.
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
    TEdgeBoundProgram build_t_edge_bound_program(
        const ProtocolContext &context,
        const std::map<
            std::string,
            algebra::FieldElement> &challenges,
        const std::map<
            std::string,
            algebra::FieldElement>
            &external_evaluations,
        const TEdgeEvaluationPlan &plan);

    // Evaluate the exact formal `t_FH` alpha-slot sequence from already resolved
    // point values.  A null `feature_encoding_values` represents the production
    // full-identity optimization and still consumes the two zero alpha slots.
    algebra::FieldElement evaluate_t_fh_from_values(
        const FHQuotientParameters &parameters,
        const FHQuotientValues &values,
        const QuotientBoundaryValues &boundary_values,
        const FHFeatureEncodingValues *feature_encoding_values = nullptr);

    std::shared_ptr<const TEdgeEvaluationPlan> build_t_edge_evaluation_plan(
        const ProtocolContext &context,
        const std::map<std::string, algebra::FieldElement> &challenges);

    std::shared_ptr<const TNEvaluationPlan> build_t_n_evaluation_plan(
        const ProtocolContext &context,
        const std::map<std::string, algebra::FieldElement> &challenges);

    // 返回某个工作域 opening bundle 中必须出现的标签顺序。
    // 该顺序会同时影响 prover 组包、verifier 取值和 proof_size_bytes 统计。
    std::vector<std::string> domain_opening_labels(const ProtocolContext &context, const std::string &domain_name);
    std::optional<std::string> dynamic_commitment_domain_name(
        const ProtocolContext &context,
        const std::string &label);

    // Evaluate the verifier-local column [1, gamma, gamma^2, ...] interpolated
    // over `domain`. This quotient input is neither committed nor opened.
    algebra::FieldElement evaluate_geometric_power_column(
        const algebra::RootOfUnityDomain &domain,
        const algebra::FieldElement &gamma,
        const algebra::FieldElement &point);

    // `t_FH` 是特征 lookup 域的总约束，prove / verify 都必须按同一等式重算。
    algebra::FieldElement evaluate_t_fh(
        const ProtocolContext &context,
        const std::map<std::string, algebra::FieldElement> &challenges,
        const EvalFn &eval,
        const algebra::FieldElement &z,
        FHQuotientProfile *profile = nullptr,
        const QuotientBoundaryValues *boundary_values = nullptr,
        bool feature_encoding_terms_are_identically_zero = false);

    // 下面这些函数分别对应 edge / N / in / d / cat / C 等工作域的 quotient 组合关系。
    // 它们只允许更换实现细节，不允许改动 formal note 定下的等式结构。
    algebra::FieldElement evaluate_t_edge(
        const ProtocolContext &context,
        const std::map<std::string, algebra::FieldElement> &challenges,
        const std::map<std::string, algebra::FieldElement> &external_evaluations,
        const EvalFn &eval,
        const algebra::FieldElement &z,
        bool return_numerator = false,
        std::size_t *first_nonzero_term = nullptr,
        algebra::FieldElement *first_nonzero_value = nullptr,
        const std::vector<algebra::FieldElement> *precomputed_alpha_powers = nullptr,
        const TEdgeEvaluationPlan *evaluation_plan = nullptr,
        const QuotientBoundaryValues *boundary_values = nullptr);

    algebra::FieldElement evaluate_t_n(
        const ProtocolContext &context,
        const std::map<std::string, algebra::FieldElement> &challenges,
        const EvalFn &eval,
        const algebra::FieldElement &z,
        const TNEvaluationPlan *evaluation_plan = nullptr,
        const QuotientBoundaryValues *boundary_values = nullptr,
        const TNPublicSourceFoldEvalFn *public_source_fold_eval = nullptr,
        std::size_t evaluation_index = 0);

    algebra::FieldElement evaluate_t_in(
        const ProtocolContext &context,
        const std::map<std::string, algebra::FieldElement> &challenges,
        const std::map<std::string, algebra::FieldElement> &external_evaluations,
        const EvalFn &eval,
        const algebra::FieldElement &z,
        bool return_numerator = false,
        std::size_t *first_nonzero_term = nullptr,
        algebra::FieldElement *first_nonzero_value = nullptr);

    algebra::FieldElement evaluate_t_d(
        const ProtocolContext &context,
        const std::map<std::string, algebra::FieldElement> &challenges,
        const std::map<std::string, algebra::FieldElement> &external_evaluations,
        const EvalFn &eval,
        const algebra::FieldElement &z,
        bool return_numerator = false,
        std::size_t *first_nonzero_term = nullptr,
        algebra::FieldElement *first_nonzero_value = nullptr);

    algebra::FieldElement evaluate_t_cat(
        const ProtocolContext &context,
        const std::map<std::string, algebra::FieldElement> &challenges,
        const std::map<std::string, algebra::FieldElement> &external_evaluations,
        const EvalFn &eval,
        const algebra::FieldElement &z);

    algebra::FieldElement evaluate_t_c(
        const ProtocolContext &context,
        const std::map<std::string, algebra::FieldElement> &challenges,
        const std::map<std::string, algebra::FieldElement> &external_evaluations,
        const EvalFn &eval,
        const algebra::FieldElement &z);

} // namespace gatzk::protocol
