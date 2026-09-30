#include "gat_proof.hpp"

#include <stdexcept>

namespace zkgpt {
namespace gat {

RelationRegistry RelationRegistry::canonical() {
    RelationRegistry registry;
    registry.descriptors = {
        {"input.public", "official_input_layer", "input", "X_public and graph/PPI boundary encoding", "X_pub, M_pub, node_ptr, edge_ptr",
         "input-layer advice and public encoding"},
        {"projection.hidden", "dedicated_matmul_sumcheck", "hidden_projection", "H_prime = H * W for every hidden head", "projection claims and dimensions",
         "model weights and MatMul auxiliary"},
        {"attention.score", "ml_friendly_gkr", "attention_score", "E_src + E_dst followed by canonical LeakyReLU table", "score and table-domain claims",
         "attention vectors, score trace, lookup queries"},
        {"attention.softmax", "lasso_plus_gkr", "stable_softmax", "M = max(Z), Delta = M-Z, ExpMap(Delta), Sum and field alpha",
         "sorted sparse edge groups and static table id", "max advice, ExpMap queries, reciprocal auxiliary"},
        {"aggregation.hidden", "ml_friendly_gkr", "hidden_aggregation", "H_agg_num = sum ExpMap(Delta)*H_prime and exact field/integer consistency",
         "aggregation segment claims", "sparse accumulation advice and field openings"},
        {"aggregation.output", "ml_friendly_gkr", "output_aggregation", "Y_agg_num = sum ExpMap(Delta)*Y_prime with output scale recovery",
         "output aggregation claims", "sparse accumulation advice and reciprocal auxiliary"},
        {"rounding", "range_prover_sumcheck", "signed_round_div", "SignedRoundDiv witness equations for recovery, alignment and head average",
         "integer-bound ledger and denominator positivity", "sigma,z,a,a_inverse,d_inverse,q,r,delta"},
        {"activation.hidden", "gkr_range_lookup", "hidden_elu", "canonical ELU lookup at S_agg_pre to S_agg", "ELU table id and certified input domain",
         "ELU lookup tuples"},
        {"concat", "ml_friendly_gkr", "head_concat", "ordered hidden-head concatenation without rescale", "head order and S_cat", "concatenation trace"},
        {"output.public_logits", "hyrax_gkr_fusion", "public_logits", "output bias, explicit alignment and integer average produce Y",
         "public logits Y and commitment/opening claims", "private weights, trace and commitment randomness"}};
    return registry;
}

const RelationDescriptor* RelationRegistry::find(const std::string& id) const {
    for (const RelationDescriptor& descriptor : descriptors)
        if (descriptor.id == id) return &descriptor;
    return nullptr;
}

bool RelationRegistry::covers_all_segments() const {
    static const char* required[] = {
        "input",      "hidden_projection", "attention_score", "stable_softmax", "hidden_aggregation", "output_aggregation", "signed_round_div",
        "hidden_elu", "head_concat",       "public_logits"};
    for (const char* segment : required) {
        bool found = false;
        for (const RelationDescriptor& descriptor : descriptors)
            if (descriptor.segment == segment) {
                found = true;
                break;
            }
        if (!found) return false;
    }
    return true;
}

OfficialCircuitPlan build_official_circuit_plan(const Graph& graph, const Model& model, const QuantConfig& quant) {
    const GatCircuitPlan full = build_gat_circuit_plan(graph, model, quant);
    OfficialCircuitPlan plan;
    plan.registry = full.registry;
    plan.circuit = full.circuit;
    plan.native_api = full.native_api;
    plan.ready_for_official_consumers = full.official_layered && !full.has_placeholder;
    return plan;
}

}
}
