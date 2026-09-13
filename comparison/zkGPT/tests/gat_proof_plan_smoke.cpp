#include "gat_proof.hpp"

#include <iostream>
#include <sstream>

using namespace zkgpt::gat;

namespace {

Tensor tensor(const std::string &name, const std::vector<std::size_t> &dims) {
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
    for (std::size_t layer = 0; layer < 2; ++layer) for (std::size_t head = 0; head < 8; ++head) {
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
    std::cerr << "fixture tensors=" << result.tensors.size() << " manifest=" << result.manifest.tensor_count << "\n";
    result.validate_schema();
    return result;
}

QuantConfig quant_fixture() {
    QuantConfig result;
    result.schema = result.quant_cfg_id = "gat_l3_scale_chain";
    result.s_input = 16; result.s_delta = 256; result.s_exp = 16;
    result.s_h = {{16, 16, 16}};
    for (std::size_t layer = 0; layer < 2; ++layer) {
        result.s_w_hidden[layer].assign(8, 4);
        result.s_hprime_hidden[layer].assign(8, 64);
        result.s_a_src_hidden[layer].assign(8, 4);
        result.s_a_dst_hidden[layer].assign(8, 4);
        result.s_agg_pre[layer] = result.s_agg[layer] = result.s_cat[layer] = 16;
    }
    result.s_out_pre = result.s_out_head = result.s_out = 16;
    result.s_w_output = {{4, 4}}; result.s_yprime_output = {{64, 64}};
    result.s_a_src_output = {{4, 4}}; result.s_a_dst_output = {{4, 4}}; result.s_b_output = {{16, 16}};
    result.rounding = "nearest_ties_away";
    result.rescale = "none";
    result.output_average = "signed_round_div";
    result.validate();
    return result;
}

Graph graph_fixture() {
    Graph result;
    result.num_nodes = 4; result.num_features = 2; result.graph_count = 2; result.num_classes = 2;
    result.node_ptr = {0, 2, 4};
    result.src = {0, 1, 0, 1, 2, 3, 2, 3};
    result.dst = {0, 0, 1, 1, 2, 2, 3, 3};
    result.edge_gid = {0, 0, 0, 0, 1, 1, 1, 1};
    result.edge_ptr = {0, 4, 8};
    result.validate();
    return result;
}

} // namespace

int main() {
    const Model model = model_fixture();
    const QuantConfig quant = quant_fixture();
    const GatCircuitPlan plan = build_gat_circuit_plan(graph_fixture(), model, quant);
    if (!plan.official_layered || plan.has_placeholder || plan.circuit.circuit.size() <= 1 ||
        plan.matmuls.size() != 54 || plan.sparse_gkr.empty() || plan.lasso.size() != 4 ||
        plan.checkpoints.empty() || !plan.registry.covers_all_segments()) return 1;
    const NativePrimitiveEvidence evidence = exercise_official_primitives("proof-plan-smoke");
    if (!evidence.sumcheck_called || !evidence.gkr_path_bound || !evidence.lasso_range_called ||
        !evidence.hyrax_commit_called) return 2;
    std::cout << "gat_proof_plan_smoke: PASS layers=" << plan.circuit.circuit.size()
              << " matmuls=" << plan.matmuls.size()
              << " checkpoints=" << plan.checkpoints.size() << "\n";
    return 0;
}
