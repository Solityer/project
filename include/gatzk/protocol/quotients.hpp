#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "gatzk/algebra/eval_backend.hpp"
#include "gatzk/algebra/field.hpp"
#include "gatzk/protocol/proof.hpp"

namespace gatzk::protocol {

using EvalFn = std::function<algebra::FieldElement(const std::string&, const algebra::FieldElement&)>;

struct FHQuotientProfile {
    double dependency_eval_ms = 0.0;
    double assembly_ms = 0.0;
};

std::vector<std::string> domain_opening_labels(const ProtocolContext& context, const std::string& domain_name);

algebra::FieldElement evaluate_t_fh(
    const ProtocolContext& context,
    const std::map<std::string, algebra::FieldElement>& challenges,
    const EvalFn& eval,
    const algebra::FieldElement& z,
    FHQuotientProfile* profile = nullptr);

algebra::FieldElement evaluate_t_edge(
    const ProtocolContext& context,
    const std::map<std::string, algebra::FieldElement>& challenges,
    const std::map<std::string, algebra::FieldElement>& witness_scalars,
    const EvalFn& eval,
    const algebra::FieldElement& z);

algebra::FieldElement evaluate_t_n(
    const ProtocolContext& context,
    const std::map<std::string, algebra::FieldElement>& challenges,
    const std::map<std::string, algebra::FieldElement>& witness_scalars,
    const EvalFn& eval,
    const algebra::FieldElement& z);

algebra::FieldElement evaluate_t_in(
    const ProtocolContext& context,
    const std::map<std::string, algebra::FieldElement>& challenges,
    const std::map<std::string, algebra::FieldElement>& external_evaluations,
    const EvalFn& eval,
    const algebra::FieldElement& z);

algebra::FieldElement evaluate_t_d(
    const ProtocolContext& context,
    const std::map<std::string, algebra::FieldElement>& challenges,
    const std::map<std::string, algebra::FieldElement>& external_evaluations,
    const EvalFn& eval,
    const algebra::FieldElement& z);

algebra::FieldElement evaluate_t_cat(
    const ProtocolContext& context,
    const std::map<std::string, algebra::FieldElement>& challenges,
    const std::map<std::string, algebra::FieldElement>& external_evaluations,
    const EvalFn& eval,
    const algebra::FieldElement& z);

algebra::FieldElement evaluate_t_c(
    const ProtocolContext& context,
    const std::map<std::string, algebra::FieldElement>& challenges,
    const std::map<std::string, algebra::FieldElement>& external_evaluations,
    const EvalFn& eval,
    const algebra::FieldElement& z);

}  // namespace gatzk::protocol
