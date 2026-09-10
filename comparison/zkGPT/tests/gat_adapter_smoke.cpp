#include "gat_proof.hpp"

#include <cassert>
#include <functional>
#include <iostream>

using namespace zkgpt::gat;

namespace {

Tensor make_tensor(const std::string &name, const std::vector<std::size_t> &dims, double value) {
    Tensor tensor;
    tensor.name = name;
    tensor.shape.dims = dims;
    tensor.values.assign(tensor.shape.elements(), value);
    return tensor;
}

Model make_model() {
    Model model;
    model.manifest.dataset = "synthetic";
    model.manifest.model_arch_id = "gat_family_L3_hid8x8-8x8_kout2_c2";
    model.manifest.quant_cfg_id = "gat_l3_scale_chain";
    model.manifest.tensor_count = 56;
    model.manifest.layers = 3;
    model.manifest.output_heads = 2;
    model.manifest.classes = 2;
    model.manifest.model_param_id = "synthetic-model-v1";
    model.manifest.static_table_id = "tables:lrelu+elu+exp+range";
    model.manifest.input_preprocess_id = "identity";
    model.manifest.batching_rule = "single_graph_or_ppi_offsets";
    model.manifest.edge_sort_rule = "edge_gid_then_dst_stable";
    model.manifest.self_loop_rule = "per_node";
    model.manifest.label_format = "single_label";
    model.manifest.output_average_rule = "per_head_bias_then_arithmetic_mean";
    model.manifest.quantization_scale_chain = "gat_l3_scale_chain";
    for (std::size_t layer = 0; layer < 2; ++layer) for (std::size_t head = 0; head < 8; ++head) {
        std::ostringstream prefix;
        prefix << "hidden/layer" << layer << "/head" << head << "/";
        const std::size_t input_dim = layer == 0 ? 2 : 8;
        model.tensors.emplace(prefix.str() + "seq/kernel", make_tensor(prefix.str() + "seq/kernel", {1, input_dim, 1}, 0.25));
        model.tensors.emplace(prefix.str() + "attn_src/kernel", make_tensor(prefix.str() + "attn_src/kernel", {1, 1, 1}, 0.25));
        model.tensors.emplace(prefix.str() + "attn_dst/kernel", make_tensor(prefix.str() + "attn_dst/kernel", {1, 1, 1}, 0.25));
    }
    for (std::size_t head = 0; head < 2; ++head) {
        std::ostringstream prefix;
        prefix << "output/head" << head << "/";
        model.tensors.emplace(prefix.str() + "seq/kernel", make_tensor(prefix.str() + "seq/kernel", {1, 8, 2}, 0.25));
        model.tensors.emplace(prefix.str() + "attn_src/kernel", make_tensor(prefix.str() + "attn_src/kernel", {1, 2, 1}, 0.25));
        model.tensors.emplace(prefix.str() + "attn_dst/kernel", make_tensor(prefix.str() + "attn_dst/kernel", {1, 2, 1}, 0.25));
        model.tensors.emplace(prefix.str() + "output_bias", make_tensor(prefix.str() + "output_bias", {2}, 0.25));
    }
    model.validate_schema();
    return model;
}

QuantConfig make_quant() {
    QuantConfig q;
    q.schema = q.quant_cfg_id = "gat_l3_scale_chain";
    q.s_input = 16; q.s_delta = 256; q.s_exp = 16;
    q.s_h = {{16, 16, 16}};
    for (std::size_t l = 0; l < 2; ++l) {
        q.s_w_hidden[l].assign(8, 4); q.s_hprime_hidden[l].assign(8, 64);
        q.s_a_src_hidden[l].assign(8, 4); q.s_a_dst_hidden[l].assign(8, 4);
        q.s_agg_pre[l] = q.s_agg[l] = q.s_cat[l] = 16;
    }
    q.s_out_pre = q.s_out_head = q.s_out = 16;
    q.s_w_output = {{4, 4}}; q.s_yprime_output = {{64, 64}};
    q.s_a_src_output = {{4, 4}}; q.s_a_dst_output = {{4, 4}}; q.s_b_output = {{16, 16}};
    q.rounding = "nearest_ties_away"; q.rescale = "none"; q.output_average = "signed_round_div";
    q.validate();
    return q;
}

Graph make_graph() {
    Graph graph;
    graph.num_nodes = 4; graph.num_features = 2; graph.graph_count = 2; graph.num_classes = 2;
    graph.node_ptr = {0, 2, 4};
    // Two disjoint PPI-style graphs.  The ordering is the canonical
    // edge_gid-then-dst stable ordering expected by the adapter.
    graph.src = {0, 1, 0, 1, 2, 3, 2, 3};
    graph.dst = {0, 0, 1, 1, 2, 2, 3, 3};
    graph.edge_gid = {0, 0, 0, 0, 1, 1, 1, 1};
    graph.edge_ptr = {0, 4, 8};
    graph.validate();
    return graph;
}

} // namespace

int main() {
    const DivisionResult tie = signed_round_div(-5, 2);
    assert(tie.value == -3 && tie.witness.valid(RoundingMode::NEAREST_TIES_AWAY));
    const DivisionResult exact = signed_round_div(4, 2);
    assert(exact.value == 2 && exact.witness.valid(RoundingMode::NEAREST_TIES_AWAY));

    const Graph graph = make_graph();
    Tensor2D features(4, 2);
    features.values = {16, 0, 0, 16, 16, 16, 0, 0};
    const Model model = make_model();
    const QuantConfig quant = make_quant();
    const ForwardResult result = run_forward(graph, features, model, quant);
    assert(result.trace.relation_ok);
    assert(validate_relation(graph, features, model, quant, result));
    assert(result.trace.ledger.families.size() == 18);
    assert(result.trace.tables.authenticate());

    const auto expect_relation_reject = [&](const char *label,
                                            const std::function<void(ForwardResult &)> &mutate) {
        ForwardResult tampered = result;
        mutate(tampered);
        if (validate_relation(graph, features, model, quant, tampered)) {
            std::cerr << "relation tamper unexpectedly accepted: " << label << "\n";
            return false;
        }
        return true;
    };

    Model tampered_model = model;
    tampered_model.tensors.begin()->second.values.at(0) += 1.0;
    assert(!validate_relation(graph, features, tampered_model, quant, result));

    assert(expect_relation_reject("projection", [](ForwardResult &tampered) {
        tampered.trace.hidden_layers.at(0).heads.at(0).projection.at(0) += 1;
    }));
    assert(expect_relation_reject("score", [](ForwardResult &tampered) {
        tampered.trace.hidden_layers.at(0).heads.at(0).raw_score.at(0) += 1;
    }));
    assert(expect_relation_reject("group_sum", [](ForwardResult &tampered) {
        tampered.trace.hidden_layers.at(0).heads.at(0).group_sum.at(0) += 1;
    }));
    assert(expect_relation_reject("inverse", [](ForwardResult &tampered) {
        tampered.trace.hidden_layers.at(0).heads.at(0).group_inverse.at(0) = Field(0);
    }));
    assert(expect_relation_reject("integer_numerator", [](ForwardResult &tampered) {
        tampered.trace.hidden_layers.at(0).heads.at(0).integer_numerator.at(0) += 1;
    }));
    assert(expect_relation_reject("signed_round_div", [](ForwardResult &tampered) {
        tampered.trace.hidden_layers.at(0).heads.at(0).divisions.at(0).q += 1;
    }));
    assert(expect_relation_reject("ELU", [](ForwardResult &tampered) {
        tampered.trace.hidden_layers.at(0).heads.at(0).activated.at(0) += 1;
    }));
    assert(expect_relation_reject("final Y", [](ForwardResult &tampered) {
        tampered.logits.values.at(0) += 1;
    }));
    assert(expect_relation_reject("trace Y", [](ForwardResult &tampered) {
        tampered.trace.logits.values.at(0) += 1;
    }));

    const OfficialCircuitPlan plan = build_official_circuit_plan(graph, model, quant);
    assert(plan.ready_for_official_consumers && plan.registry.covers_all_segments());
    assert(plan.registry.find("projection.hidden") != nullptr);

    const GatCircuitPlan proof_plan = build_gat_circuit_plan(graph, model, quant, result);
    const ModelRegistration registration = register_model(model, quant, proof_plan.input_layout,
                                                          model.manifest.model_arch_id);
    assert(registration.status == "registered_long_lived_com_W");
    assert(registration.com_W.valid());
    assert(registration.com_W.padded_size == proof_plan.input_layout.n_in);
    assert(registration.n_in == proof_plan.input_layout.n_in);
    assert(registration.input_segments.size() == 4);
    assert(registration.selector == "00:X_pub,01:W,10:Aux,11:L_qry");
    assert(registration.opening_policy == "same_r_three_openings_v1");
    assert(registration_matches_model(model, quant, proof_plan.input_layout, registration));

    Model manifest_tampered = model;
    manifest_tampered.manifest.model_param_id += "-tampered";
    assert(!registration_matches_model(manifest_tampered, quant, proof_plan.input_layout, registration));

    Model model_tampered = model;
    model_tampered.tensors.begin()->second.values.at(0) += 1.0;
    assert(!registration_matches_model(model_tampered, quant, proof_plan.input_layout, registration));

    ModelRegistration bounds_tampered = registration;
    bounds_tampered.bounds_digest = "tampered-bounds";
    assert(!registration_matches_model(model, quant, proof_plan.input_layout, bounds_tampered));

    QuantConfig quant_tampered = quant;
    ++quant_tampered.s_input;
    assert(!registration_matches_model(model, quant_tampered, proof_plan.input_layout, registration));

    InputLayout layout_tampered = proof_plan.input_layout;
    layout_tampered.segments.at(1).offset += 1;
    assert(!registration_matches_model(model, quant, layout_tampered, registration));

    ModelRegistration selector_tampered = registration;
    selector_tampered.selector = "00:X_pub,10:W,01:Aux,11:L_qry";
    assert(!registration_matches_model(model, quant, proof_plan.input_layout, selector_tampered));

    ModelRegistration opening_tampered = registration;
    opening_tampered.opening_policy = "independent_r_openings";
    assert(!registration_matches_model(model, quant, proof_plan.input_layout, opening_tampered));

    ModelRegistration com_w_tampered = registration;
    com_w_tampered.com_W.basis_id = "tampered-com-W";
    assert(!registration_matches_model(model, quant, proof_plan.input_layout, com_w_tampered));

    ModelRegistration layout_digest_tampered = registration;
    layout_digest_tampered.layout_digest = "tampered-layout";
    assert(!registration_matches_model(model, quant, proof_plan.input_layout, layout_digest_tampered));

    const VerifierContext context = make_verifier_context(
        graph, features, quant, result.trace.tables, result.logits, registration,
        proof_plan, "masked_full");
    assert(!verifier_context_digest(context).empty());
    bool context_rejected = false;
    try {
        ModelRegistration context_registration_tampered = registration;
        context_registration_tampered.n_in += 1;
        (void)make_verifier_context(graph, features, quant, result.trace.tables,
                                    result.logits, context_registration_tampered,
                                    proof_plan, "masked_full");
    } catch (const std::exception &) {
        context_rejected = true;
    }
    assert(context_rejected);
    std::cout << "gat_adapter_smoke: PASS relation/rounding/PPI/descriptors\n";
    return 0;
}
