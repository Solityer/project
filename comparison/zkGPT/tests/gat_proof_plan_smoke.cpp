#include "gat_proof.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <set>
#include <sstream>

using namespace zkgpt::gat;

namespace {

Tensor tensor(const std::string& name, const std::vector<std::size_t>& dims) {
    Tensor result;
    result.name = name;
    result.shape.dims = dims;
    result.values.assign(result.shape.elements(), 0.25);
    return result;
}

Model model_fixture() {
    Model result;
    result.manifest.dataset = "proof-plan-smoke";
    result.manifest.model_arch_id = "gat_family_L3_hid8x8-8x8_kout2_c2";
    result.manifest.quant_cfg_id = "gat_l3_scale_chain";
    result.manifest.tensor_count = 56;
    result.manifest.layers = 3;
    result.manifest.output_heads = 2;
    result.manifest.classes = 2;
    result.manifest.model_param_id = "proof-plan-model-v1";
    result.manifest.static_table_id = "tables:lrelu+elu+exp+range";
    result.manifest.input_preprocess_id = "identity";
    result.manifest.batching_rule = "single_graph_or_ppi_offsets";
    result.manifest.edge_sort_rule = "edge_gid_then_dst_stable";
    result.manifest.self_loop_rule = "per_node";
    result.manifest.label_format = "single_label";
    result.manifest.output_average_rule = "per_head_bias_then_arithmetic_mean";
    result.manifest.quantization_scale_chain = "gat_l3_scale_chain";
    for (std::size_t layer = 0; layer < 2; ++layer)
        for (std::size_t head = 0; head < 8; ++head) {
            std::ostringstream prefix;
            prefix << "hidden/layer" << layer << "/head" << head << "/";
            const std::size_t input_dim = layer == 0 ? 2 : 8;
            result.tensors.emplace(prefix.str() + "seq/kernel", tensor(prefix.str() + "seq/kernel", {1, input_dim, 1}));
            result.tensors.emplace(prefix.str() + "attn_src/kernel", tensor(prefix.str() + "attn_src/kernel", {1, 1, 1}));
            result.tensors.emplace(prefix.str() + "attn_dst/kernel", tensor(prefix.str() + "attn_dst/kernel", {1, 1, 1}));
        }
    for (std::size_t head = 0; head < 2; ++head) {
        std::ostringstream prefix;
        prefix << "output/head" << head << "/";
        result.tensors.emplace(prefix.str() + "seq/kernel", tensor(prefix.str() + "seq/kernel", {1, 8, 2}));
        result.tensors.emplace(prefix.str() + "attn_src/kernel", tensor(prefix.str() + "attn_src/kernel", {1, 2, 1}));
        result.tensors.emplace(prefix.str() + "attn_dst/kernel", tensor(prefix.str() + "attn_dst/kernel", {1, 2, 1}));
        result.tensors.emplace(prefix.str() + "output_bias", tensor(prefix.str() + "output_bias", {2}));
    }
    result.validate_schema();
    return result;
}

QuantConfig quant_fixture() {
    QuantConfig result;
    result.schema = result.quant_cfg_id = "gat_l3_scale_chain";
    result.s_input = 16;
    result.s_delta = 256;
    result.s_exp = 16;
    result.s_h = {{16, 16, 16}};
    for (std::size_t layer = 0; layer < 2; ++layer) {
        result.s_w_hidden[layer].assign(8, 4);
        result.s_hprime_hidden[layer].assign(8, 64);
        result.s_a_src_hidden[layer].assign(8, 4);
        result.s_a_dst_hidden[layer].assign(8, 4);
        result.s_agg_pre[layer] = result.s_agg[layer] = result.s_cat[layer] = 16;
    }
    result.s_out_pre = result.s_out_head = result.s_out = 16;
    result.s_w_output = {{4, 4}};
    result.s_yprime_output = {{64, 64}};
    result.s_a_src_output = {{4, 4}};
    result.s_a_dst_output = {{4, 4}};
    result.s_b_output = {{16, 16}};
    result.rounding = "nearest_ties_away";
    result.rescale = "none";
    result.output_average = "signed_round_div";
    result.validate();
    return result;
}

Graph graph_fixture() {
    Graph result;
    result.num_nodes = 4;
    result.num_features = 2;
    result.graph_count = 2;
    result.num_classes = 2;
    result.node_ptr = {0, 2, 4};
    result.src = {0, 1, 0, 1, 2, 3, 2, 3};
    result.dst = {0, 0, 1, 1, 2, 2, 3, 3};
    result.edge_gid = {0, 0, 0, 0, 1, 1, 1, 1};
    result.edge_ptr = {0, 4, 8};
    result.validate();
    return result;
}

std::set<std::string> expected_matmul_relation_ids() {
    std::set<std::string> expected;
    for (std::size_t layer = 0; layer < 2; ++layer) {
        for (std::size_t head = 0; head < 8; ++head) {
            std::ostringstream family;
            family << "hidden/layer" << layer << "/head" << head;
            expected.insert(family.str() + "/projection");
            expected.insert(family.str() + "/attention/source_dot");
            expected.insert(family.str() + "/attention/destination_dot");
        }
    }
    for (std::size_t head = 0; head < 2; ++head) {
        const std::string family = "output/head" + std::to_string(head);
        expected.insert("projection." + family);
        expected.insert(family + "/attention/source_dot");
        expected.insert(family + "/attention/destination_dot");
    }
    return expected;
}

std::set<std::string> expected_sparse_relation_ids() {
    std::set<std::string> expected;
    const auto add_family = [&expected](const std::string& family) {
        expected.insert("attention.score/" + family);
        expected.insert("attention.softmax/" + family);
        expected.insert((family.rfind("hidden/", 0) == 0 ? "aggregation.hidden/" : "aggregation.output/") + family);
    };
    for (std::size_t layer = 0; layer < 2; ++layer)
        for (std::size_t head = 0; head < 8; ++head)
            add_family("hidden/layer" + std::to_string(layer) + "/head" + std::to_string(head));
    for (std::size_t head = 0; head < 2; ++head)
        add_family("output/head" + std::to_string(head));
    return expected;
}

bool semantic_plan_coverage_valid(const GatCircuitPlan& plan, const Model& model) {
    const auto fail = [](const std::string& reason) {
        std::cerr << "plan coverage check failed: " << reason << '\n';
        return false;
    };
    const std::set<std::string> expected_matmuls = expected_matmul_relation_ids();
    std::set<std::string> actual_matmuls;
    for (const MatMulDescriptor& descriptor : plan.matmuls)
        if (!actual_matmuls.insert(descriptor.relation_id).second) return fail("duplicate MatMul relation id");
    if (actual_matmuls != expected_matmuls || plan.fconn_claim_links.size() != plan.matmuls.size()) {
        std::ostringstream detail;
        detail << "MatMul/FCONN relation id coverage: matmuls=" << actual_matmuls.size()
               << " expected=" << expected_matmuls.size() << " links=" << plan.fconn_claim_links.size();
        for (const std::string& relation : expected_matmuls)
            if (actual_matmuls.count(relation) == 0) detail << " missing_matmul=" << relation;
        for (const std::string& relation : actual_matmuls)
            if (expected_matmuls.count(relation) == 0) detail << " extra_matmul=" << relation;
        return fail(detail.str());
    }
    for (std::size_t index = 0; index < plan.matmuls.size(); ++index) {
        const MatMulDescriptor& descriptor = plan.matmuls[index];
        const FconnClaimLink& link = plan.fconn_claim_links[index];
        if (link.descriptor_index != index || link.relation_id != descriptor.relation_id ||
            link.family_id != descriptor.family_id || link.lhs_segment != descriptor.lhs_segment ||
            link.rhs_segment != descriptor.rhs_segment || link.output_segment != descriptor.output_segment ||
            link.lhs_offset != descriptor.lhs_offset || link.rhs_offset != descriptor.rhs_offset ||
            link.output_offset != descriptor.output_offset || link.rows != descriptor.rows ||
            link.inner != descriptor.inner || link.cols != descriptor.cols)
            return fail("FCONN claim link does not match its descriptor role/offset");
    }

    const std::set<std::string> expected_sparse = expected_sparse_relation_ids();
    std::set<std::string> actual_sparse;
    for (const SparseGkrDescriptor& descriptor : plan.sparse_gkr)
        if (!actual_sparse.insert(descriptor.relation_id).second) return fail("duplicate sparse relation id");
    if (actual_sparse != expected_sparse) return fail("score/softmax/aggregation relation id coverage");

    const std::set<std::string> expected_lasso = {
        "lasso/function/lrelu", "lasso/function/exp", "lasso/function/elu", "lasso/range/all"};
    std::set<std::string> actual_lasso;
    for (const LassoFamilyDescriptor& descriptor : plan.lasso) {
        if (descriptor.padded_queries < descriptor.real_queries || descriptor.table_domain == 0 ||
            !actual_lasso.insert(descriptor.relation_id).second)
            return fail("invalid or duplicate Lasso family");
    }
    if (actual_lasso != expected_lasso) return fail("Lasso family relation id coverage");

    const std::set<std::string> expected_checkpoints = {
        "output/head_sum", "output/final_average", "output/public_logits"};
    std::set<std::string> actual_checkpoints;
    for (const CheckpointDescriptor& checkpoint : plan.checkpoints)
        if (expected_checkpoints.count(checkpoint.id) != 0) actual_checkpoints.insert(checkpoint.id);
    if (actual_checkpoints != expected_checkpoints || !plan.registry.covers_all_segments())
        return fail("canonical output checkpoints/input relation registry coverage");

    std::string reason;
    if (!validate_gat_circuit_plan(plan, &reason)) return fail("native plan validator: " + reason);
    const std::size_t expected_layer_count =
        1 + 2 * plan.matmuls.size() + (model.manifest.layers - 1) + 2;
    if (plan.circuit.circuit.size() != expected_layer_count || plan.circuit.circuit.size() <= 1)
        return fail("retained layer topology: ");

    const auto aux = std::find_if(plan.input_layout.segments.begin(), plan.input_layout.segments.end(),
                                  [](const InputSegment& segment) { return segment.name == "Aux"; });
    if (aux == plan.input_layout.segments.end()) return fail("missing canonical Aux segment");
    const std::size_t aux_end = aux->offset + aux->logical_length;
    std::size_t fconn_count = 0;
    std::size_t binding_count = 0;
    std::size_t canonical_copy_count = 0;
    for (std::size_t index = 1; index < plan.circuit.circuit.size(); ++index) {
        const layer& current = plan.circuit.circuit[index];
        if (current.ty == layerType::FCONN) {
            ++fconn_count;
            continue;
        }
        if (plan.circuit.circuit[index - 1].ty == layerType::FCONN) {
            ++binding_count;
            continue;
        }
        if (index + 1 == plan.circuit.circuit.size()) {
            if (current.size != plan.public_output_coordinates + plan.residual_coordinates)
                return fail("terminal public output/residual span");
            continue;
        }
        if (current.ty != layerType::ADD_BIAS || !current.bin_gates.empty() ||
            current.uni_gates.size() != current.size)
            return fail("unexpected noncanonical intermediate layer type or shape");
        for (std::size_t gate_index = 0; gate_index < current.uni_gates.size(); ++gate_index) {
            const uniGate& gate = current.uni_gates[gate_index];
            if (gate.g != gate_index || gate.lu != 0 || current.coefficient(gate) != 1 ||
                gate.u >= current.ori_id_u.size())
                return fail("canonical copy gate coordinates/coefficient");
            const std::size_t source_coordinate = static_cast<std::size_t>(current.ori_id_u[gate.u]);
            if (source_coordinate < aux->offset || source_coordinate >= aux_end)
                return fail("canonical copy does not source Aux");
        }
        ++canonical_copy_count;
    }
    if (fconn_count != plan.matmuls.size() || binding_count != plan.matmuls.size() ||
        canonical_copy_count != model.manifest.layers)
        return fail("FCONN/binding/canonical copy counts");
    return true;
}

}

int main() {
    const Model model = model_fixture();
    const QuantConfig quant = quant_fixture();
    const Graph graph = graph_fixture();
    const GatCircuitPlan plan = build_gat_circuit_plan(graph, model, quant);
    if (!plan.official_layered || plan.has_placeholder || plan.circuit.circuit.size() <= 1 || plan.matmuls.size() != 54 || plan.sparse_gkr.empty() ||
        plan.lasso.size() != 4 || plan.checkpoints.empty() || !plan.registry.covers_all_segments() ||
        plan.residual_targets.size() != plan.residual_coordinates ||
        std::none_of(plan.residual_targets.begin(), plan.residual_targets.end(),
                     [](Int target) { return target != 0; }) ||
        !semantic_plan_coverage_valid(plan, model)) {
        std::cerr << "canonical semantic relation or retained GKR anchor coverage changed\n";
        return 1;
    }
    const auto rejected_plan_mutation = [&plan](GatCircuitPlan mutated, const std::string& expected_reason) {
        std::string reason;
        return !validate_gat_circuit_plan(mutated, &reason) && reason.find(expected_reason) != std::string::npos;
    };
    GatCircuitPlan missing_fconn_link = plan;
    missing_fconn_link.fconn_claim_links.pop_back();
    GatCircuitPlan bad_fconn_offset = plan;
    ++bad_fconn_offset.fconn_claim_links.back().output_offset;
    GatCircuitPlan bad_fconn_relation = plan;
    bad_fconn_relation.fconn_claim_links.back().relation_id += "/tampered";
    if (!rejected_plan_mutation(std::move(missing_fconn_link), "FCONN claim link count") ||
        !rejected_plan_mutation(std::move(bad_fconn_offset), "FCONN claim link does not match") ||
        !rejected_plan_mutation(std::move(bad_fconn_relation), "FCONN claim link does not match")) {
        std::cerr << "canonical plan accepted a missing or mutated FCONN claim link\n";
        return 13;
    }
    const GatPointDomainAdmission shape = inspect_gat_point_domain_admission(graph, model, quant);
    const GatCapacityReport capacity = inspect_gat_capacity(graph, model, quant);
    const auto materialized_fconn_count = static_cast<std::uint64_t>(std::count_if(
        plan.circuit.circuit.begin(), plan.circuit.circuit.end(),
        [](const layer& current) { return current.ty == layerType::FCONN; }));
    const std::uint64_t expected_shape_touch_index_bytes =
        2 * ((capacity.gkr_subset_source_support_entries_upper_bound + 63) / 64) * sizeof(std::size_t);
    if (!shape.ready || shape.reason.size() != 0 || shape.digest != plan.gkr_point_shape_digest ||
        shape.layer_count != plan.circuit.circuit.size() || shape.fconn_count != materialized_fconn_count ||
        shape.bitmap_bytes == 0 ||
        shape.touched_index_bytes_upper_bound != expected_shape_touch_index_bytes)
        return 3;
    std::uint64_t expected_gate_count = 0;
    std::uint64_t expected_input_sources = 0;
    bool previous_lane_active = true;
    std::size_t next_matmul = plan.matmuls.size();
    for (std::size_t index = plan.circuit.circuit.size(); index-- > 1;) {
        const layer& current = plan.circuit.circuit[index];
        expected_gate_count += static_cast<std::uint64_t>(current.uni_gates.size());
        expected_gate_count += static_cast<std::uint64_t>(current.bin_gates.size());
        if (!previous_lane_active) {
            const bool fconn_residual_branch = index > 1 && next_matmul > 0 &&
                plan.circuit.circuit[index - 1].ty == layerType::FCONN;
            if (!fconn_residual_branch) ++expected_input_sources;
        }
        if (current.ty == layerType::FCONN) {
            if (next_matmul == 0) return 7;
            --next_matmul;
            expected_input_sources += 2;
            previous_lane_active = false;
        } else {
            expected_input_sources += static_cast<std::uint64_t>(current.bit_length_u[0] >= 0);
            expected_input_sources += static_cast<std::uint64_t>(current.bit_length_v[0] >= 0);
            previous_lane_active = true;
        }
    }
    if (next_matmul != 0 || capacity.ready ||
        capacity.reason != "runtime resource headroom has not been finalized" || !capacity.gkr_point_shape_ready ||
        capacity.gkr_point_shape_digest != shape.digest ||
        capacity.gkr_point_shape_layer_count != plan.circuit.circuit.size() ||
        capacity.gkr_point_shape_fconn_count != materialized_fconn_count ||
        capacity.gkr_point_shape_touched_index_scratch_bytes_upper_bound !=
            expected_shape_touch_index_bytes ||
        capacity.gkr_layer_count_upper_bound != plan.circuit.circuit.size() ||
        capacity.gkr_fconn_layer_count_upper_bound != materialized_fconn_count ||
        capacity.explicit_gate_count != expected_gate_count ||
        capacity.gkr_input_source_count_upper_bound != expected_input_sources) {
        std::cerr << "capacity visitor profile does not match materialized plan: layers="
                  << capacity.gkr_layer_count_upper_bound << "/" << plan.circuit.circuit.size()
                  << " fconn=" << capacity.gkr_fconn_layer_count_upper_bound << "/" << materialized_fconn_count
                  << " gates=" << capacity.explicit_gate_count << "/" << expected_gate_count
                  << " input_sources=" << capacity.gkr_input_source_count_upper_bound << "/" << expected_input_sources
                  << " reason=" << capacity.reason << '\n';
        return 8;
    }
    Tensor2D features(graph.num_nodes, graph.num_features);
    features.values = {3, 7, 11, 13, 17, 19, 23, 29};
    const ForwardResult retained = run_forward(graph, features, model, quant);
    if (!retained.trace.relation_ok || !validate_relation(graph, features, model, quant, retained)) return 4;
    const auto max_abs = [](const std::vector<Int>& values) {
        Wide maximum = 0;
        for (Int value : values) maximum = std::max(maximum, value < 0 ? -Wide(value) : Wide(value));
        return maximum;
    };
    const Wide raw_features_abs = max_abs(features.values);
    const Wide hidden_layer_zero_abs = max_abs(retained.trace.hidden_layers.at(0).concatenated.values);
    auto layer_one_family = std::find_if(retained.trace.ledger.families.begin(), retained.trace.ledger.families.end(),
                                         [](const BoundFamily& family) { return family.id == "hidden/layer1/head0"; });
    if (raw_features_abs == hidden_layer_zero_abs || layer_one_family == retained.trace.ledger.families.end() ||
        layer_one_family->input_abs != hidden_layer_zero_abs) {
        std::cerr << "hidden-layer admission bound does not use its current input tensor\n";
        return 14;
    }
    ForwardResult tampered_layer_bound = retained;
    auto tampered_layer_one_family = std::find_if(tampered_layer_bound.trace.ledger.families.begin(),
                                                  tampered_layer_bound.trace.ledger.families.end(),
                                                  [](const BoundFamily& family) { return family.id == "hidden/layer1/head0"; });
    if (tampered_layer_one_family == tampered_layer_bound.trace.ledger.families.end()) return 15;
    tampered_layer_one_family->input_abs = raw_features_abs;
    if (validate_relation(graph, features, model, quant, tampered_layer_bound)) {
        std::cerr << "relation validator accepted a raw-feature bound for a hidden layer input\n";
        return 16;
    }
    const GatCircuitWitness witness = build_gat_circuit_witness(plan, graph, features, model, quant, retained);
    std::string witness_reason;
    if (!witness.relation_ok || witness.public_output.size() != plan.public_output_coordinates + plan.residual_coordinates ||
        !evaluate_gat_circuit_witness(plan, witness, &witness_reason)) {
        std::cerr << "nonzero canonical layered witness failed: " << witness_reason << '\n';
        return 9;
    }
    std::size_t checked_output_bindings = 0;
    for (std::size_t index = 1; index + 1 < plan.circuit.circuit.size(); ++index) {
        if (plan.circuit.circuit[index].ty != layerType::FCONN) continue;
        const auto& binding_values = witness.values.at(index + 1);
        if (checked_output_bindings >= plan.matmuls.size() ||
            binding_values.size() != plan.matmuls[checked_output_bindings].output_coordinates ||
            std::any_of(binding_values.begin(), binding_values.end(), [](const Fr& value) { return !value.isZero(); })) {
            std::cerr << "nonzero FCONN output did not bind to its canonical Aux output span\n";
            return 10;
        }
        ++checked_output_bindings;
    }
    const auto l_query = std::find_if(plan.input_layout.segments.begin(), plan.input_layout.segments.end(),
                                      [](const InputSegment& segment) { return segment.name == "L_qry"; });
    if (checked_output_bindings != plan.matmuls.size() || l_query == plan.input_layout.segments.end()) return 11;
    for (const LassoFamilyDescriptor& family : plan.lasso) {
        if (family.query_offset < l_query->offset ||
            family.query_offset - l_query->offset > l_query->logical_length ||
            family.padded_queries > l_query->logical_length - (family.query_offset - l_query->offset)) {
            std::cerr << "Lasso query family escaped canonical L_qry input segment\n";
            return 12;
        }
    }
    const auto division_profiles = canonical_division_range_profiles(graph, model, quant);
    std::string admission_reason;
    if (!run_forward_admission(graph, features, model, quant, division_profiles, &admission_reason, &retained)) {
        std::cerr << "nonzero canonical forward admission failed: " << admission_reason << '\n';
        return 5;
    }
    Tensor2D out_of_table = features;
    out_of_table.values[0] = 1000000;
    bool retained_rejected = false;
    try {
        (void)run_forward(graph, out_of_table, model, quant);
    } catch (const std::exception&) {
        retained_rejected = true;
    }
    admission_reason.clear();
    if (!retained_rejected || run_forward_admission(graph, out_of_table, model, quant, division_profiles, &admission_reason) ||
        admission_reason.empty()) {
        std::cerr << "canonical forward admission did not reject an out-of-table input consistently\n";
        return 6;
    }
    const NativePrimitiveEvidence evidence = exercise_official_primitives("proof-plan-smoke");
    if (!evidence.sumcheck_called || !evidence.gkr_path_bound || !evidence.lasso_range_called || !evidence.hyrax_commit_called) return 2;
    std::cout << "gat_proof_plan_smoke: PASS plan_structure layers=" << plan.circuit.circuit.size()
              << " fconn_bindings=" << checked_output_bindings << " matmuls=" << plan.matmuls.size()
              << " lasso_families=" << plan.lasso.size() << " checkpoints=" << plan.checkpoints.size()
              << " capacity=NOT_FINALIZED\n";
    return 0;
}
