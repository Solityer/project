#include "gat_proof.hpp"

#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <limits>

using namespace zkgpt::gat;
using namespace mcl::bn;

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

int main(int argc, char **argv) {
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
    GatCircuitWitness circuit_witness = build_gat_circuit_witness(
        proof_plan, graph, features, model, quant, result);
    std::string circuit_reason;
    assert(circuit_witness.relation_ok);
    assert(circuit_witness.public_output.size() ==
           proof_plan.public_output_coordinates + proof_plan.residual_coordinates);
    assert(evaluate_gat_circuit_witness(proof_plan, circuit_witness,
                                        &circuit_reason));
    const auto expect_circuit_reject = [&](const char *label,
                                           const std::function<void(GatCircuitWitness &)> &mutate) {
        GatCircuitWitness tampered = circuit_witness;
        mutate(tampered);
        if (evaluate_gat_circuit_witness(proof_plan, tampered, &circuit_reason)) {
            std::cerr << "circuit tamper unexpectedly accepted: " << label << "\n";
            return false;
        }
        return true;
    };
    assert(expect_circuit_reject("W", [&](GatCircuitWitness &tampered) {
        tampered.input_layer.at(proof_plan.input_layout.segments.at(1).offset) += Field(1);
    }));
    assert(expect_circuit_reject("Aux", [&](GatCircuitWitness &tampered) {
        tampered.input_layer.at(proof_plan.input_layout.segments.at(2).offset + 1) += Field(1);
    }));
    assert(expect_circuit_reject("L_qry", [&](GatCircuitWitness &tampered) {
        tampered.input_layer.at(proof_plan.input_layout.segments.at(3).offset) += Field(1);
    }));
    std::size_t first_rounding_offset = 0;
    for (const CheckpointDescriptor &checkpoint : proof_plan.checkpoints) {
        if (checkpoint.id == "rounding/hidden/hidden/layer0/head0") {
            first_rounding_offset = checkpoint.coordinate_offset;
            break;
        }
    }
    assert(first_rounding_offset != 0);
    assert(expect_circuit_reject("rounding witness", [&](GatCircuitWitness &tampered) {
        tampered.input_layer.at(first_rounding_offset + 3) += Field(1);
    }));
    assert(expect_circuit_reject("Y", [](GatCircuitWitness &tampered) {
        tampered.public_output.at(0) += Field(1);
    }));
    assert(expect_circuit_reject("rounding/attention", [&](GatCircuitWitness &tampered) {
        tampered.input_layer.at(proof_plan.input_layout.segments.at(2).offset + 20) += Field(1);
    }));
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

    const std::string com_w_secret_path = "/tmp/zkgpt-gat-com-w-secret-smoke";
    std::remove(com_w_secret_path.c_str());
    std::string secret_reason;
    assert(write_prover_commitment_secret(registration, com_w_secret_path,
                                          &secret_reason));
    Fr recovered_com_w_blinder;
    assert(read_prover_commitment_secret(registration, com_w_secret_path,
                                         recovered_com_w_blinder,
                                         &secret_reason));
    assert(recovered_com_w_blinder == registration.com_W_padding_mask);
    ModelRegistration wrong_secret_identity = registration;
    wrong_secret_identity.registration_digest[0] =
        wrong_secret_identity.registration_digest[0] == '0' ? '1' : '0';
    assert(!read_prover_commitment_secret(wrong_secret_identity,
                                          com_w_secret_path,
                                          recovered_com_w_blinder,
                                          &secret_reason));
    std::remove(com_w_secret_path.c_str());

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

    if (argc > 1 && std::string(argv[1]) == "--formal") {
        Graph formal_graph;
        formal_graph.num_nodes = 1;
        formal_graph.num_features = 2;
        formal_graph.graph_count = 1;
        formal_graph.num_classes = 2;
        formal_graph.node_ptr = {0, 1};
        formal_graph.src = {0};
        formal_graph.dst = {0};
        formal_graph.edge_gid = {0};
        formal_graph.edge_ptr = {0, 1};
        formal_graph.validate();
        Tensor2D formal_features(1, 2);
        formal_features.values = {16, 0};
        const ForwardResult formal_probe = run_forward(
            formal_graph, formal_features, model, quant);
        QuantConfig formal_quant = quant;
        formal_quant.range_bits = 1;
        Int lrelu_bound = 2;
        Int exp_bound = 2;
        Int elu_bound = 2;
        const auto absorb_attention = [&](const AttentionTrace &trace) {
            for (Int value : trace.raw_score)
                lrelu_bound = std::max<Int>(lrelu_bound, std::abs(value) + 1);
            for (Int value : trace.delta)
                exp_bound = std::max<Int>(exp_bound, value + 1);
            for (Int value : trace.recovered)
                elu_bound = std::max<Int>(elu_bound, std::abs(value) + 1);
            for (const DivisionWitness &division : trace.divisions)
                formal_quant.range_bits = std::max<Int>(
                    formal_quant.range_bits,
                    static_cast<Int>(std::numeric_limits<std::uint32_t>::digits -
                                     __builtin_clz(static_cast<std::uint32_t>(
                                         std::max<Int>(1, std::abs(division.q))))));
        };
        for (const HiddenLayerTrace &layer : formal_probe.trace.hidden_layers)
            for (const AttentionTrace &trace : layer.heads) absorb_attention(trace);
        for (const OutputHeadTrace &head : formal_probe.trace.output_heads)
        {
            absorb_attention(head.attention);
            for (const DivisionWitness &division : head.alignment_divisions)
                formal_quant.range_bits = std::max<Int>(
                    formal_quant.range_bits,
                    static_cast<Int>(std::numeric_limits<std::uint32_t>::digits -
                                     __builtin_clz(static_cast<std::uint32_t>(
                                         std::max<Int>(1, std::abs(division.q))))));
        }
        for (const DivisionWitness &division : formal_probe.trace.final_average_divisions)
            formal_quant.range_bits = std::max<Int>(
                formal_quant.range_bits,
                static_cast<Int>(std::numeric_limits<std::uint32_t>::digits -
                                 __builtin_clz(static_cast<std::uint32_t>(
                                     std::max<Int>(1, std::abs(division.q))))));
        formal_quant.range_bits = std::min<Int>(formal_quant.range_bits, 12);
        formal_quant.static_lrelu_min = -lrelu_bound;
        formal_quant.static_lrelu_max = lrelu_bound;
        formal_quant.static_exp_min = 0;
        formal_quant.static_exp_max = exp_bound;
        formal_quant.static_elu_min = -elu_bound;
        formal_quant.static_elu_max = elu_bound;
        formal_quant.validate();
        const ForwardResult formal_forward = run_forward(
            formal_graph, formal_features, model, formal_quant);
        assert(formal_forward.trace.relation_ok);
        const GatCircuitPlan formal_plan = build_gat_circuit_plan(
            formal_graph, model, formal_quant, formal_forward);
        const ModelRegistration formal_registration = register_model(
            model, formal_quant, formal_plan.input_layout,
            model.manifest.model_arch_id);
        const VerifierContext formal_context = make_verifier_context(
            formal_graph, formal_features, formal_quant, formal_forward.trace.tables,
            formal_forward.logits, formal_registration, formal_plan,
            "masked_full");
        const ProveResult formal = prove_gat(
            formal_plan, formal_graph, formal_features, model, formal_quant,
            formal_forward, formal_registration, "masked_full");
        if (formal.status != ProveStatus::GENERATED || !formal.has_artifact) {
            std::cerr << "masked_full formal smoke rejected: "
                      << formal.diagnostic << "\n";
            return 1;
        }
        assert(!formal.artifact.has_private_payload());
        const std::string formal_path = "/tmp/zkgpt-masked-full-formal-smoke.artifact";
        std::string formal_reason;
        assert(formal.artifact.write(formal_path, &formal_reason));
        GatProofArtifact parsed_formal;
        assert(GatProofArtifact::read(formal_path, parsed_formal, &formal_reason));
        assert(verify_gat(formal_context, parsed_formal, &formal_reason));

        VerifierContext y_tampered = formal_context;
        y_tampered.public_logits.values.at(0) += 1;
        assert(!verify_gat(y_tampered, parsed_formal, &formal_reason));
        VerifierContext registration_tampered = formal_context;
        registration_tampered.registration.registration_digest[0] =
            registration_tampered.registration.registration_digest[0] == '0' ? '1' : '0';
        assert(!verify_gat(registration_tampered, parsed_formal, &formal_reason));
        VerifierContext table_tampered = formal_context;
        table_tampered.tables.table_id += "-tampered";
        assert(!verify_gat(table_tampered, parsed_formal, &formal_reason));
        VerifierContext resource_tampered = formal_context;
        resource_tampered.tables.digest[0] =
            resource_tampered.tables.digest[0] == '0' ? '1' : '0';
        assert(!verify_gat(resource_tampered, parsed_formal, &formal_reason));

        GatProofArtifact aux_tampered = parsed_formal;
        aux_tampered.com_aux.point_hex[0] =
            aux_tampered.com_aux.point_hex[0] == '0' ? '1' : '0';
        assert(!verify_gat(formal_context, aux_tampered, &formal_reason));
        GatProofArtifact l_tampered = parsed_formal;
        l_tampered.com_L.point_hex[0] =
            l_tampered.com_L.point_hex[0] == '0' ? '1' : '0';
        assert(!verify_gat(formal_context, l_tampered, &formal_reason));
        GatProofArtifact linear_tampered = parsed_formal;
        bool linear_message_found = false;
        for (NativeProofMessage &message : linear_tampered.native_messages) {
            if (message.label == "combine/input-linear/response") {
                message.payload = "1";
                linear_message_found = true;
                break;
            }
        }
        assert(linear_message_found);
        assert(!verify_gat(formal_context, linear_tampered, &formal_reason));
        GatProofArtifact transcript_tampered = parsed_formal;
        assert(!transcript_tampered.native_messages.empty());
        transcript_tampered.native_messages.front().payload += "tampered";
        assert(!verify_gat(formal_context, transcript_tampered, &formal_reason));

        const ProveResult fresh = prove_gat(
            formal_plan, formal_graph, formal_features, model, formal_quant,
            formal_forward, formal_registration, "masked_full");
        assert(fresh.status == ProveStatus::GENERATED && fresh.has_artifact);
        assert(fresh.artifact.transcript_digest != formal.artifact.transcript_digest);
        std::remove(formal_path.c_str());
        std::cout << "gat_adapter_formal_smoke: PASS masked_full serialize/verify/tamper/freshness\n";
        return 0;
    }

    const CommitmentWire com_aux = commit_private_segment(
        std::vector<Int>{1, 2}, "synthetic/aux", proof_plan.input_layout.n_in);
    const CommitmentWire com_L = commit_private_segment(
        std::vector<Int>{3, 4}, "synthetic/L_qry", proof_plan.input_layout.n_in);
    G1 basis_G;
    hashAndMapToG1(basis_G, "synthetic/native-basis/G");
    std::vector<G1> basis_generators(4);
    std::vector<Fr> basis_scalars{Fr(1), Fr(2), Fr(3), Fr(4)};
    for (std::size_t i = 0; i < basis_generators.size(); ++i)
        hashAndMapToG1(basis_generators[i], "synthetic/native-basis/g/" + std::to_string(i));
    const NativeProofStatement native_statement = make_native_proof_statement(
        context, com_aux, com_L, basis_G, basis_generators, basis_scalars, 4);
    assert(native_statement.valid());
    std::string expected_context_root = verifier_context_digest(context);
    expected_context_root.insert(0, 64 - expected_context_root.size(), '0');
    assert(native_statement.root() == expected_context_root);
    assert(native_statement.basis_digest ==
           NativeProofStatement::basis_digest_for(
               basis_G, basis_generators, basis_scalars, 4));
    NativeProofStatement root_tampered = native_statement;
    root_tampered.statement_root_digest[0] =
        root_tampered.statement_root_digest[0] == '0' ? '1' : '0';
    assert(!root_tampered.valid());

    // Bounded native GKR-only adapter closure.  The circuit is synthetic and
    // intentionally tiny, while Y/context/commitments come from the same
    // authenticated public context above.  This exercises the production
    // artifact wire without claiming masked_full proof completeness.
    VerifierContext synthetic_context = context;
    synthetic_context.public_logits.values.assign(result.logits.values.size(), Int(1));
    std::vector<F> gkr_output(result.logits.values.size(), Fr(1));
    prover synthetic;
    synthetic.C.init(0, 2);
    synthetic.C.circuit[0].ty = layerType::INPUT;
    synthetic.C.circuit[0].size = static_cast<u32>(gkr_output.size());
    synthetic.C.circuit[0].zero_start_id = synthetic.C.circuit[0].size;
    synthetic.C.circuit[0].bit_length = 4;
    synthetic.C.circuit[1].ty = layerType::ADD_BIAS;
    synthetic.C.circuit[1].size = static_cast<u32>(gkr_output.size());
    synthetic.C.circuit[1].zero_start_id = synthetic.C.circuit[1].size;
    synthetic.C.circuit[1].bit_length = 3;
    for (u32 index = 0; index < gkr_output.size(); ++index)
        synthetic.C.circuit[1].uni_gates.emplace_back(index, index, 0, 1);
    synthetic.C.initSubset();
    synthetic.val.resize(2);
    synthetic.val[0] = gkr_output;
    synthetic.val[1] = gkr_output;
    const GkrOnlyArtifactResult gkr_result = prove_gkr_only_artifact(
        synthetic, synthetic_context, com_aux, com_L, basis_G, basis_generators,
        basis_scalars, 4, gkr_output, {});
    assert(gkr_result.generated);
    assert(gkr_result.artifact.privacy_profile == "gkr_only_diagnostic");
    const std::string artifact_path = "/tmp/zkgpt-gkr-only-smoke.artifact";
    std::string artifact_reason;
    assert(gkr_result.artifact.write(artifact_path, &artifact_reason));
    GatProofArtifact serialized_artifact;
    assert(GatProofArtifact::read(artifact_path, serialized_artifact,
                                  &artifact_reason));
    NativeProofSession serialized_session;
    assert(serialized_artifact.native_session(serialized_session,
                                               &artifact_reason));
    assert(verify_gkr_transcript(serialized_session, synthetic.C, gkr_output,
                                 {}, &artifact_reason));
    assert(!verify_gat(synthetic_context, serialized_artifact,
                       &artifact_reason));

    GatProofArtifact tampered_artifact = serialized_artifact;
    bool tampered_claim = false;
    for (NativeProofMessage &message : tampered_artifact.native_messages) {
        if (message.kind == NativeProofMessageKind::FieldVector &&
            message.label.find("/claims") != std::string::npos &&
            message.payload.size() > sizeof(std::uint64_t) * 2) {
            const std::size_t first_value_offset = sizeof(std::uint64_t) * 2;
            message.payload[first_value_offset] =
                message.payload[first_value_offset] == '0' ? '1' : '0';
            tampered_claim = true;
            break;
        }
    }
    assert(tampered_claim);
    NativeProofSession tampered_session;
    assert(tampered_artifact.native_session(tampered_session,
                                            &artifact_reason));
    assert(!verify_gkr_transcript(tampered_session, synthetic.C, gkr_output,
                                  {}, &artifact_reason));
    assert(!tampered_artifact.valid_wire(&artifact_reason));
    std::remove(artifact_path.c_str());
    std::cout << "gat_adapter_smoke: PASS relation/rounding/PPI/descriptors\n";
    return 0;
}
