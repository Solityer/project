#pragma once

#include <functional>
#include <map>
#include <string>

#include "gatzk/algebra/eval_backend.hpp"
#include "gatzk/algebra/field.hpp"
#include "gatzk/protocol/proof.hpp"

namespace gatzk::protocol {

using EvalFn = std::function<algebra::FieldElement(const std::string&, const algebra::FieldElement&)>;

algebra::FieldElement evaluate_t_fh(
    const ProtocolContext& context,
    const std::map<std::string, algebra::FieldElement>& challenges,
    const EvalFn& eval,
    const algebra::FieldElement& z);

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

#if GATZK_ENABLE_CUDA_BACKEND
algebra::FieldElement evaluate_t_fh_device_cuda(
    const ProtocolContext& context,
    const std::map<std::string, algebra::FieldElement>& challenges,
    const algebra::PackedEvaluationDeviceResult& evaluations,
    const algebra::FieldElement& z);

algebra::FieldElement evaluate_t_edge_device_cuda(
    const ProtocolContext& context,
    const std::map<std::string, algebra::FieldElement>& challenges,
    const std::map<std::string, algebra::FieldElement>& witness_scalars,
    const algebra::PackedEvaluationDeviceResult& evaluations,
    const algebra::FieldElement& z);

algebra::FieldElement evaluate_t_n_device_cuda(
    const ProtocolContext& context,
    const std::map<std::string, algebra::FieldElement>& challenges,
    const std::map<std::string, algebra::FieldElement>& witness_scalars,
    const algebra::PackedEvaluationDeviceResult& evaluations,
    const algebra::FieldElement& z);

algebra::FieldElement evaluate_t_in_device_cuda(
    const ProtocolContext& context,
    const std::map<std::string, algebra::FieldElement>& challenges,
    const std::map<std::string, algebra::FieldElement>& external_evaluations,
    const algebra::PackedEvaluationDeviceResult& evaluations,
    const algebra::FieldElement& z);

algebra::FieldElement evaluate_t_d_device_cuda(
    const ProtocolContext& context,
    const std::map<std::string, algebra::FieldElement>& challenges,
    const std::map<std::string, algebra::FieldElement>& external_evaluations,
    const algebra::PackedEvaluationDeviceResult& evaluations,
    const algebra::FieldElement& z);
#endif

}  // namespace gatzk::protocol
