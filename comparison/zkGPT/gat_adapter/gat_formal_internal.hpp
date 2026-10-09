#pragma once

#include "gat_proof.hpp"
#include "gat_forward_internal.hpp"

#include <chrono>
#include <memory>

namespace zkgpt::gat::detail {

class ValidatedFormalInputs {
  public:
    ValidatedFormalInputs(const ValidatedFormalInputs&) = delete;
    ValidatedFormalInputs& operator=(const ValidatedFormalInputs&) = delete;
    ValidatedFormalInputs(ValidatedFormalInputs&&) noexcept = default;
    ValidatedFormalInputs& operator=(ValidatedFormalInputs&&) noexcept = default;

    static ValidatedFormalInputs consume(Graph&& graph, Tensor2D&& features,
                                         Model&& model, QuantConfig&& quant,
                                         ForwardResult&& forward);

    const Graph& graph() const noexcept { return graph_; }
    const Tensor2D& features() const noexcept { return features_; }
    const Model& model() const noexcept { return model_; }
    const QuantConfig& quant() const noexcept { return quant_; }
    const ForwardResult& forward() const noexcept { return forward_; }
    const GatCapacityReport& capacity() const noexcept { return capacity_; }
    const GatCircuitPlan& plan() const noexcept { return *plan_; }
    std::shared_ptr<const GatCircuitPlan> plan_owner() const noexcept {
        return plan_;
    }
    std::size_t canonical_validation_calls() const noexcept {
        return canonical_validation_calls_;
    }
    double canonical_validation_ms() const noexcept {
        return canonical_validation_ms_;
    }
    std::size_t capacity_census_calls() const noexcept {
        return capacity_census_calls_;
    }
    std::size_t capacity_restore_rebuild_calls() const noexcept {
        return capacity_restore_rebuild_calls_;
    }
    const std::string& capacity_quant_digest() const noexcept {
        return capacity_quant_digest_;
    }

  private:
    ValidatedFormalInputs(Graph&& graph, Tensor2D&& features, Model&& model,
                          QuantConfig&& quant, ForwardResult&& forward);

    Graph graph_;
    Tensor2D features_;
    Model model_;
    QuantConfig quant_;
    ForwardResult forward_;
    GatCapacityReport capacity_;
    std::string capacity_quant_digest_;
    std::shared_ptr<GatCircuitPlan> plan_;
    std::size_t canonical_validation_calls_ = 0;
    double canonical_validation_ms_ = 0.0;
    std::size_t capacity_census_calls_ = 0;
    mutable std::size_t capacity_restore_rebuild_calls_ = 0;

    friend ProveResult prove_gat_formal(
        ValidatedFormalInputs&, const ModelRegistration&,
        const VerifierContext&, const std::string&);
    friend GatCircuitPlan rebuild_formal_plan_for_restore(
        const ValidatedFormalInputs&);
};

ProveResult prove_gat_formal(ValidatedFormalInputs& inputs,
                             const ModelRegistration& registration,
                             const VerifierContext& context,
                             const std::string& privacy_profile);

bool formal_capacity_report_matches_plan(const GatCapacityReport& capacity,
                                         const Graph& graph,
                                         const Model& model,
                                         const QuantConfig& quant,
                                         const GatCircuitPlan& plan);
GatCircuitPlan rebuild_formal_plan_for_restore(
    const ValidatedFormalInputs& inputs);

}  // namespace zkgpt::gat::detail
