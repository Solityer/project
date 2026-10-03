#include "gat_proof.hpp"

#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

using namespace zkgpt::gat;
using namespace mcl::bn;

namespace {

Tensor make_tensor(const std::string& name, const std::vector<std::size_t>& dims, double value) {
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
    for (std::size_t layer = 0; layer < 2; ++layer)
        for (std::size_t head = 0; head < 8; ++head) {
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

Model make_formal_model() {
    Model model;
    model.manifest.dataset = "synthetic";
    model.manifest.model_arch_id = "gat_family_L3_hid1x1-1x1_kout1_c1";
    model.manifest.quant_cfg_id = "gat_l3_scale_chain";
    model.manifest.tensor_count = 10;
    model.manifest.layers = 3;
    model.manifest.output_heads = 1;
    model.manifest.classes = 1;
    model.manifest.model_param_id = "synthetic-formal-model-v1";
    model.manifest.static_table_id = "tables:lrelu+elu+exp+range";
    model.manifest.input_preprocess_id = "identity";
    model.manifest.batching_rule = "single_graph_or_ppi_offsets";
    model.manifest.edge_sort_rule = "edge_gid_then_dst_stable";
    model.manifest.self_loop_rule = "per_node";
    model.manifest.label_format = "single_label";
    model.manifest.output_average_rule = "per_head_bias_then_arithmetic_mean";
    model.manifest.quantization_scale_chain = "gat_l3_scale_chain";
    for (std::size_t layer = 0; layer < 2; ++layer)
        for (std::size_t head = 0; head < 1; ++head) {
            std::ostringstream prefix;
            prefix << "hidden/layer" << layer << "/head" << head << "/";
            const std::size_t input_dim = 1;
            model.tensors.emplace(prefix.str() + "seq/kernel", make_tensor(prefix.str() + "seq/kernel", {1, input_dim, 1}, 0.25));
            model.tensors.emplace(prefix.str() + "attn_src/kernel", make_tensor(prefix.str() + "attn_src/kernel", {1, 1, 1}, 0.25));
            model.tensors.emplace(prefix.str() + "attn_dst/kernel", make_tensor(prefix.str() + "attn_dst/kernel", {1, 1, 1}, 0.25));
        }
    for (std::size_t head = 0; head < 1; ++head) {
        std::ostringstream prefix;
        prefix << "output/head" << head << "/";
        model.tensors.emplace(prefix.str() + "seq/kernel", make_tensor(prefix.str() + "seq/kernel", {1, 1, 1}, 0.25));
        model.tensors.emplace(prefix.str() + "attn_src/kernel", make_tensor(prefix.str() + "attn_src/kernel", {1, 1, 1}, 0.25));
        model.tensors.emplace(prefix.str() + "attn_dst/kernel", make_tensor(prefix.str() + "attn_dst/kernel", {1, 1, 1}, 0.25));
        model.tensors.emplace(prefix.str() + "output_bias", make_tensor(prefix.str() + "output_bias", {1}, 0.25));
    }
    model.validate_schema();
    return model;
}

QuantConfig make_quant() {
    QuantConfig q;
    q.schema = q.quant_cfg_id = "gat_l3_scale_chain";
    q.s_input = 16;
    q.s_delta = 256;
    q.s_exp = 16;
    q.s_h = {{16, 16, 16}};
    for (std::size_t l = 0; l < 2; ++l) {
        q.s_w_hidden[l].assign(8, 4);
        q.s_hprime_hidden[l].assign(8, 64);
        q.s_a_src_hidden[l].assign(8, 4);
        q.s_a_dst_hidden[l].assign(8, 4);
        q.s_agg_pre[l] = q.s_agg[l] = q.s_cat[l] = 16;
    }
    q.s_out_pre = q.s_out_head = q.s_out = 16;
    q.s_w_output = {{4, 4}};
    q.s_yprime_output = {{64, 64}};
    q.s_a_src_output = {{4, 4}};
    q.s_a_dst_output = {{4, 4}};
    q.s_b_output = {{16, 16}};
    q.rounding = "nearest_ties_away";
    q.rescale = "none";
    q.output_average = "signed_round_div";
    q.validate();
    return q;
}

Graph make_graph() {
    Graph graph;
    graph.num_nodes = 4;
    graph.num_features = 2;
    graph.graph_count = 2;
    graph.num_classes = 2;
    graph.node_ptr = {0, 2, 4};

    graph.src = {0, 1, 0, 1, 2, 3, 2, 3};
    graph.dst = {0, 0, 1, 1, 2, 2, 3, 3};
    graph.edge_gid = {0, 0, 0, 0, 1, 1, 1, 1};
    graph.edge_ptr = {0, 4, 8};
    graph.validate();
    return graph;
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (unsigned char ch : value) {
        switch (ch) {
        case '\\':
            out << "\\\\";
            break;
        case '"':
            out << "\\\"";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            out << static_cast<char>(ch);
            break;
        }
    }
    return out.str();
}

std::string relative_to_cwd(const std::string& path) {
    char buffer[4096] = {};
    if (::getcwd(buffer, sizeof(buffer)) == nullptr) throw std::runtime_error("cannot determine standalone fixture root");
    const std::string cwd(buffer);
    const std::string prefix = cwd + "/";
    if (path.compare(0, prefix.size(), prefix) == 0) return path.substr(prefix.size());
    if (!path.empty() && path[0] != '/') return path;
    throw std::runtime_error("standalone fixture directory is outside the baseline root");
}

void write_standalone_fixture(const std::string& directory, const VerifierContext& context, const GatProofArtifact& artifact) {
    struct stat info{};
    if (::stat(directory.c_str(), &info) != 0 || !S_ISDIR(info.st_mode)) throw std::runtime_error("standalone fixture directory is not present");
    const std::string context_path = directory + "/verifier_context.bin";
    const std::string logits_path = directory + "/public_logits.txt";
    const std::string artifact_path = directory + "/proof.artifact";
    std::string reason;
    if (!write_verifier_context(context, context_path, &reason)) throw std::runtime_error("standalone verifier context write failed: " + reason);
    std::ostringstream logits;
    logits << context.public_logits.rows << ' ' << context.public_logits.cols << '\n';
    for (std::size_t row = 0; row < context.public_logits.rows; ++row) {
        for (std::size_t col = 0; col < context.public_logits.cols; ++col) {
            if (col != 0) logits << ' ';
            logits << context.public_logits.at(row, col);
        }
        logits << '\n';
    }
    {
        std::ofstream out(logits_path.c_str(), std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot write standalone public logits");
        out << logits.str();
        if (!out.good()) throw std::runtime_error("cannot finalize standalone public logits");
    }
    if (!artifact.write(artifact_path, &reason)) throw std::runtime_error("standalone proof artifact write failed: " + reason);
    struct stat artifact_info{};
    if (::stat(artifact_path.c_str(), &artifact_info) != 0 || artifact_info.st_size < 0) throw std::runtime_error("cannot stat standalone proof artifact");
    const std::string context_rel = relative_to_cwd(context_path);
    const std::string logits_rel = relative_to_cwd(logits_path);
    const std::string artifact_rel = relative_to_cwd(artifact_path);
    std::ostringstream fixture;
    fixture << "{\n"
            << "  \"schema\": \"zkgpt-gat-formal-fixture-v2\",\n"
            << "  \"dataset\": \"" << json_escape(context.registration.dataset) << "\",\n"
            << "  \"backend\": \"" << json_escape(context.backend) << "\",\n"
            << "  \"statement_digest\": \"" << json_escape(verifier_context_digest(context)) << "\",\n"
            << "  \"plan_digest\": \"" << json_escape(context.plan.plan_digest) << "\",\n"
            << "  \"circuit_id\": \"" << json_escape(context.plan.circuit_id) << "\",\n"
            << "  \"proof_digest\": \"" << json_escape(canonical_digest64(artifact.transcript_digest)) << "\",\n"
            << "  \"registration_digest\": \"" << json_escape(context.registration.registration_digest) << "\",\n"
            << "  \"registration_binding_digest\": \"" << json_escape(context.registration.binding_digest) << "\",\n"
            << "  \"registration_layout_digest\": \"" << json_escape(context.registration.layout_digest) << "\",\n"
            << "  \"com_W_point\": \"" << json_escape(context.registration.com_W.point_hex) << "\",\n"
            << "  \"full_dataset_identity\": \"" << json_escape(context.full_dataset_identity) << "\",\n"
            << "  \"checkpoint_identity\": \"" << json_escape(context.checkpoint_identity) << "\",\n"
            << "  \"cache_identity\": \"" << json_escape(context.cache_identity) << "\",\n"
            << "  \"verifier_context_path\": \"" << json_escape(context_rel) << "\",\n"
            << "  \"public_logits_path\": \"" << json_escape(logits_rel) << "\",\n"
            << "  \"artifact_path\": \"" << json_escape(artifact_rel) << "\",\n"
            << "  \"artifact_sha256\": \"" << json_escape(sha256_file_digest(artifact_path)) << "\",\n"
            << "  \"artifact_size_bytes\": " << static_cast<std::uint64_t>(artifact_info.st_size) << "\n"
            << "}\n";
    const std::string fixture_path = directory + "/fixture.json";
    std::ofstream output(fixture_path.c_str(), std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot write standalone fixture");
    output << fixture.str();
    if (!output.good()) throw std::runtime_error("cannot finalize standalone fixture");
}

void run_signed_round_div_gate_boundary_checks() {
    for (RoundingMode mode : {RoundingMode::TOWARD_ZERO, RoundingMode::NEAREST_TIES_AWAY}) {
        for (Int dividend : {-5, -4, -3, -1, 0, 1, 3, 4, 5}) {
            const DivisionResult exact = signed_round_div(dividend, 2, mode);
            std::string reason;
            if (!evaluate_signed_round_div_gate_assignment(dividend, 2, exact, mode, &reason)) throw std::runtime_error("native division boundary rejected: " + reason);
            DivisionResult corrupt = exact;
            corrupt.witness.r += 1;
            if (evaluate_signed_round_div_gate_assignment(dividend, 2, corrupt, mode, &reason)) throw std::runtime_error("native division boundary corruption accepted");
            corrupt = exact;
            corrupt.witness.delta = -1;
            if (evaluate_signed_round_div_gate_assignment(dividend, 2, corrupt, mode, &reason)) throw std::runtime_error("native negative division slack accepted");
        }
    }
    std::cout << "native_division_boundary_checks: PASS both modes positive/negative/zero/exact/ties\n";
}

int run_formal_fixture(const std::string& standalone_dir, bool proof_verify_only) {
    const auto stage = [](const char* name) { std::cerr << "gat_adapter_formal_stage: " << name << '\n'; };
    if (proof_verify_only && standalone_dir.empty()) {
        std::cerr << "formal proof/verify fixture requires a fresh standalone directory\n";
        return 1;
    }
    if (!standalone_dir.empty()) {
        struct stat info{};
        if (::stat(standalone_dir.c_str(), &info) != 0 || !S_ISDIR(info.st_mode)) {
            std::cerr << "formal standalone directory must exist before proof generation\n";
            return 1;
        }
    }
    stage("graph");
    Graph formal_graph;
    formal_graph.num_nodes = 1;
    formal_graph.num_features = 1;
    formal_graph.graph_count = 1;
    formal_graph.num_classes = 1;
    formal_graph.node_ptr = {0, 1};
    formal_graph.src = {0};
    formal_graph.dst = {0};
    formal_graph.edge_gid = {0};
    formal_graph.edge_ptr = {0, 1};
    formal_graph.validate();
    stage("probe-forward");
    Tensor2D formal_features(1, 1);
    formal_features.values = {1};
    const Model formal_model = make_formal_model();
    const QuantConfig formal_base_quant = make_quant();
    const ForwardResult formal_probe = run_forward(formal_graph, formal_features, formal_model, formal_base_quant);
    QuantConfig formal_quant = formal_base_quant;
    formal_quant.range_bits = 1;
    Int lrelu_bound = 2;
    Int exp_bound = 2;
    Int elu_bound = 2;
    const auto absorb_attention = [&](const AttentionTrace& trace) {
        for (Int value : trace.raw_score) lrelu_bound = std::max<Int>(lrelu_bound, std::abs(value) + 1);
        for (Int value : trace.delta) exp_bound = std::max<Int>(exp_bound, value + 1);
        for (Int value : trace.recovered) elu_bound = std::max<Int>(elu_bound, std::abs(value) + 1);
        for (const DivisionWitness& division : trace.divisions)
            formal_quant.range_bits =
                std::max<Int>(formal_quant.range_bits, static_cast<Int>(std::numeric_limits<std::uint32_t>::digits -
                                                                        __builtin_clz(static_cast<std::uint32_t>(std::max<Int>(1, std::abs(division.q))))));
    };
    for (const HiddenLayerTrace& layer : formal_probe.trace.hidden_layers)
        for (const AttentionTrace& trace : layer.heads) absorb_attention(trace);
    for (const OutputHeadTrace& head : formal_probe.trace.output_heads) {
        absorb_attention(head.attention);
        for (const DivisionWitness& division : head.alignment_divisions)
            formal_quant.range_bits =
                std::max<Int>(formal_quant.range_bits, static_cast<Int>(std::numeric_limits<std::uint32_t>::digits -
                                                                        __builtin_clz(static_cast<std::uint32_t>(std::max<Int>(1, std::abs(division.q))))));
    }
    for (const DivisionWitness& division : formal_probe.trace.final_average_divisions)
        formal_quant.range_bits =
            std::max<Int>(formal_quant.range_bits, static_cast<Int>(std::numeric_limits<std::uint32_t>::digits -
                                                                    __builtin_clz(static_cast<std::uint32_t>(std::max<Int>(1, std::abs(division.q))))));
    formal_quant.range_bits = std::min<Int>(formal_quant.range_bits, 12);
    formal_quant.static_lrelu_min = -lrelu_bound;
    formal_quant.static_lrelu_max = lrelu_bound;
    formal_quant.static_exp_min = 0;
    formal_quant.static_exp_max = exp_bound;
    formal_quant.static_elu_min = -elu_bound;
    formal_quant.static_elu_max = elu_bound;
    formal_quant.validate();
    stage("formal-forward");
    const ForwardResult formal_forward = run_forward(formal_graph, formal_features, formal_model, formal_quant);
    assert(formal_forward.trace.relation_ok);
    stage("plan");
    const GatCircuitPlan formal_plan = build_gat_circuit_plan(formal_graph, formal_model, formal_quant, formal_forward);
    if (!proof_verify_only) {
        stage("native-gate-negative-checks");
        const GatCircuitWitness assignment = build_gat_circuit_witness(formal_plan, formal_graph, formal_features, formal_model, formal_quant, formal_forward);
        std::string assignment_reason;
        if (!evaluate_gat_circuit_assignment(formal_plan, assignment.input_layer, formal_forward.logits, &assignment_reason)) throw std::runtime_error("honest native gate assignment rejected: " + assignment_reason);
        const auto checkpoint = [&](const std::string& suffix) { for (const CheckpointDescriptor& item : formal_plan.checkpoints) if (item.id == suffix) return item.coordinate_offset; throw std::runtime_error("missing native negative-check checkpoint: " + suffix); };
        const auto corrupt_coordinate = [&](const std::string& name, std::size_t coordinate, const Fr& increment = Fr(1)) { zkgpt::NativeFieldVector corrupted = assignment.input_layer; corrupted.at(coordinate) += increment; std::string reason; if (evaluate_gat_circuit_assignment(formal_plan, corrupted, formal_forward.logits, &reason) || reason.find("residual mismatches public target") == std::string::npos) throw std::runtime_error("native constraint corruption was not rejected: " + name + ": " + reason); };
        const std::size_t division = checkpoint("rounding/hidden/hidden/layer0/head0");
        corrupt_coordinate("public feature Aux copy", formal_plan.input_layout.segments.at(2).offset);
        corrupt_coordinate("hidden concat", checkpoint("concat/hidden/layer0"));
        corrupt_coordinate("group inverse", checkpoint("attention.softmax/hidden/layer0/head0.group_inverse"));
        corrupt_coordinate("division inverse", division + 7);
        corrupt_coordinate("zero iff", division + 1);
        corrupt_coordinate("absolute dividend", division + 2);
        corrupt_coordinate("remainder", division + 4);
        corrupt_coordinate("remainder slack", division + 5);
        corrupt_coordinate("max delta", checkpoint("attention.softmax/hidden/layer0/head0.delta"));
        for (const LassoFamilyDescriptor& family : formal_plan.lasso) {
            if (family.family_id == "function/lrelu") corrupt_coordinate("lookup tuple", family.query_offset);
            if (family.family_id == "range/all") { corrupt_coordinate("bit decomposition", family.query_offset); corrupt_coordinate("nonboolean bit", family.query_offset, Fr(2)); }
        }
        corrupt_coordinate("output fixed denominator", checkpoint("rounding/output_alignment/output/head0") + 8);
        corrupt_coordinate("final fixed denominator", checkpoint("output/final_average") + 8);
        std::cout << "native_gate_negative_checks: PASS tuple/range/inverse/division/max/fixed-denominator\n";
        run_signed_round_div_gate_boundary_checks();
    } else {
        stage("complete-relation-proof-verify");
    }
    stage("registration");
    const ModelRegistration formal_registration = register_model(formal_model, formal_quant, formal_plan.input_layout, formal_model.manifest.model_arch_id);
    stage("context");
    VerifierContext formal_context = make_verifier_context(formal_graph, formal_features, formal_quant, formal_forward.trace.tables, formal_forward.logits,
                                                           formal_registration, formal_plan, "masked_full");
    formal_context.backend = "cpu";
    formal_context.full_dataset_identity = "canonical:synthetic";
    formal_context.checkpoint_identity = "baseline-local/checkpoints/synthetic_gat/integrity.sha256";
    formal_context.cache_identity = "baseline-local/data/cache/synthetic/integrity.sha256";
    stage("masked-full-prove");
    const ProveResult formal =
        prove_gat(formal_plan, formal_graph, formal_features, formal_model, formal_quant, formal_forward, formal_registration, "masked_full");
    if (formal.status != ProveStatus::GENERATED || !formal.has_artifact) {
        std::cerr << "masked_full formal smoke rejected: " << formal.diagnostic << "\n";
        return 1;
    }
    assert(!formal.artifact.has_private_payload());
    const std::string formal_path = standalone_dir.empty() ? "/tmp/zkgpt-masked-full-formal-smoke.artifact" : standalone_dir + "/proof.artifact";
    std::string formal_reason;
    stage("artifact-write");
    if (standalone_dir.empty()) assert(formal.artifact.write(formal_path, &formal_reason));
    if (!standalone_dir.empty()) {
        try {
            write_standalone_fixture(standalone_dir, formal_context, formal.artifact);
            std::cout << "gat_adapter_formal_fixture: GENERATED for independent verification\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "standalone fixture generation failed: " << error.what() << "\n";
            return 1;
        }
    }
    GatProofArtifact parsed_formal;
    stage("independent-verify");
    if (proof_verify_only) {
        if (!GatProofArtifact::read(formal_path, parsed_formal, &formal_reason)) {
            std::cerr << "masked_full artifact readback failed: " << formal_reason << "\n";
            return 1;
        }
        if (!verify_gat(formal_context, parsed_formal, &formal_reason)) {
            std::cerr << "masked_full verifier rejected artifact: " << formal_reason << "\n";
            return 1;
        }
    } else {
        assert(GatProofArtifact::read(formal_path, parsed_formal, &formal_reason));
        assert(verify_gat(formal_context, parsed_formal, &formal_reason));
    }

    if (proof_verify_only) {
        std::remove(formal_path.c_str());
        std::cout << "gat_adapter_formal_proof_verify: PASS masked_full artifact roundtrip and verifier call\n";
        return 0;
    }

    VerifierContext y_tampered = formal_context;
    y_tampered.public_logits.values.at(0) += 1;
    assert(!verify_gat(y_tampered, parsed_formal, &formal_reason));
    VerifierContext registration_tampered = formal_context;
    registration_tampered.registration.registration_digest[0] = registration_tampered.registration.registration_digest[0] == '0' ? '1' : '0';
    assert(!verify_gat(registration_tampered, parsed_formal, &formal_reason));
    VerifierContext table_tampered = formal_context;
    table_tampered.tables.table_id += "-tampered";
    assert(!verify_gat(table_tampered, parsed_formal, &formal_reason));
    VerifierContext resource_tampered = formal_context;
    resource_tampered.tables.digest[0] = resource_tampered.tables.digest[0] == '0' ? '1' : '0';
    assert(!verify_gat(resource_tampered, parsed_formal, &formal_reason));

    GatProofArtifact aux_tampered = parsed_formal;
    aux_tampered.com_aux.point_hex[0] = aux_tampered.com_aux.point_hex[0] == '0' ? '1' : '0';
    assert(!verify_gat(formal_context, aux_tampered, &formal_reason));
    GatProofArtifact l_tampered = parsed_formal;
    l_tampered.com_L.point_hex[0] = l_tampered.com_L.point_hex[0] == '0' ? '1' : '0';
    assert(!verify_gat(formal_context, l_tampered, &formal_reason));
    GatProofArtifact linear_tampered = parsed_formal;
    bool linear_message_found = false;
    for (NativeProofMessage& message : linear_tampered.native_messages) {
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

    std::remove(formal_path.c_str());
    std::cout << "gat_adapter_formal_smoke: PASS masked_full serialize/verify/tamper\n";
    return 0;
}

}

int main(int argc, char** argv) {
    bool formal = false;
    bool proof_verify_only = false;
    bool division_boundary_only = false;
    std::string standalone_dir;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--formal") {
            formal = true;
        } else if (argument == "--formal-proof-verify") {
            formal = true;
            proof_verify_only = true;
        } else if (argument == "--division-boundary-only") {
            division_boundary_only = true;
        } else if (argument == "--standalone-dir" && index + 1 < argc) {
            standalone_dir = argv[++index];
        } else {
            std::cerr << "unknown or incomplete smoke argument: " << argument << "\n";
            return 2;
        }
    }
    if (division_boundary_only) {
        run_signed_round_div_gate_boundary_checks();
        return 0;
    }
    if (formal) return run_formal_fixture(standalone_dir, proof_verify_only);
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

    const auto expect_relation_reject = [&](const char* label, const std::function<void(ForwardResult&)>& mutate) {
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

    assert(expect_relation_reject("projection", [](ForwardResult& tampered) { tampered.trace.hidden_layers.at(0).heads.at(0).projection.at(0) += 1; }));
    assert(expect_relation_reject("score", [](ForwardResult& tampered) { tampered.trace.hidden_layers.at(0).heads.at(0).raw_score.at(0) += 1; }));
    assert(expect_relation_reject("group_sum", [](ForwardResult& tampered) { tampered.trace.hidden_layers.at(0).heads.at(0).group_sum.at(0) += 1; }));
    assert(expect_relation_reject("inverse", [](ForwardResult& tampered) { tampered.trace.hidden_layers.at(0).heads.at(0).group_inverse.at(0) = Field(0); }));
    assert(expect_relation_reject("integer_numerator",
                                  [](ForwardResult& tampered) { tampered.trace.hidden_layers.at(0).heads.at(0).integer_numerator.at(0) += 1; }));
    assert(expect_relation_reject("signed_round_div", [](ForwardResult& tampered) { tampered.trace.hidden_layers.at(0).heads.at(0).divisions.at(0).q += 1; }));
    assert(expect_relation_reject("ELU", [](ForwardResult& tampered) { tampered.trace.hidden_layers.at(0).heads.at(0).activated.at(0) += 1; }));
    assert(expect_relation_reject("final Y", [](ForwardResult& tampered) { tampered.logits.values.at(0) += 1; }));
    assert(expect_relation_reject("trace Y", [](ForwardResult& tampered) { tampered.trace.logits.values.at(0) += 1; }));

    const OfficialCircuitPlan plan = build_official_circuit_plan(graph, model, quant);
    assert(plan.ready_for_official_consumers && plan.registry.covers_all_segments());
    assert(plan.registry.find("projection.hidden") != nullptr);

    const GatCircuitPlan proof_plan = build_gat_circuit_plan(graph, model, quant, result);
    const auto is_digest64 = [](const std::string& value) {
        if (value.size() != 64) return false;
        for (char character : value)
            if (!((character >= '0' && character <= '9') || (character >= 'a' && character <= 'f'))) return false;
        return true;
    };
    assert(is_digest64(proof_plan.plan_digest));
    assert(proof_plan.circuit_id == "R_GAT-zkGPT/" + proof_plan.plan_digest);
    assert(is_digest64(proof_plan.graph_topology_digest));
    std::size_t fconn_layers = 0;
    for (const layer& candidate : proof_plan.circuit.circuit) {
        if (candidate.ty != layerType::FCONN) continue;
        ++fconn_layers;
        assert(candidate.uni_gates.empty() && candidate.bin_gates.empty());
        assert(!candidate.native_relation_id.empty());
    }
    assert(fconn_layers == proof_plan.matmuls.size());
    GatCircuitWitness circuit_witness = build_gat_circuit_witness(proof_plan, graph, features, model, quant, result);
    std::string circuit_reason;
    assert(circuit_witness.relation_ok);
    assert(circuit_witness.public_output.size() == proof_plan.public_output_coordinates + proof_plan.residual_coordinates);
    assert(evaluate_gat_circuit_witness(proof_plan, circuit_witness, &circuit_reason));
    const auto expect_circuit_reject = [&](const char* label, const std::function<void(GatCircuitWitness&)>& mutate) {
        GatCircuitWitness tampered = circuit_witness;
        mutate(tampered);
        if (evaluate_gat_circuit_witness(proof_plan, tampered, &circuit_reason)) {
            std::cerr << "circuit tamper unexpectedly accepted: " << label << "\n";
            return false;
        }
        return true;
    };
    assert(
        expect_circuit_reject("W", [&](GatCircuitWitness& tampered) { tampered.input_layer.at(proof_plan.input_layout.segments.at(1).offset) += Field(1); }));
    assert(expect_circuit_reject("Aux",
                                 [&](GatCircuitWitness& tampered) { tampered.input_layer.at(proof_plan.input_layout.segments.at(2).offset + 1) += Field(1); }));
    assert(expect_circuit_reject("L_qry",
                                 [&](GatCircuitWitness& tampered) { tampered.input_layer.at(proof_plan.input_layout.segments.at(3).offset) += Field(1); }));
    std::size_t first_rounding_offset = 0;
    for (const CheckpointDescriptor& checkpoint : proof_plan.checkpoints) {
        if (checkpoint.id == "rounding/hidden/hidden/layer0/head0") {
            first_rounding_offset = checkpoint.coordinate_offset;
            break;
        }
    }
    assert(first_rounding_offset != 0);
    assert(expect_circuit_reject("rounding witness", [&](GatCircuitWitness& tampered) { tampered.input_layer.at(first_rounding_offset + 3) += Field(1); }));
    assert(expect_circuit_reject("Y", [](GatCircuitWitness& tampered) { tampered.public_output.at(0) += Field(1); }));
    assert(expect_circuit_reject(
        "rounding/attention", [&](GatCircuitWitness& tampered) { tampered.input_layer.at(proof_plan.input_layout.segments.at(2).offset + 20) += Field(1); }));
    const ModelRegistration registration = register_model(model, quant, proof_plan.input_layout, model.manifest.model_arch_id);
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
    assert(write_prover_commitment_secret(registration, com_w_secret_path, &secret_reason));
    Fr recovered_com_w_blinder;
    assert(read_prover_commitment_secret(registration, com_w_secret_path, recovered_com_w_blinder, &secret_reason));
    assert(recovered_com_w_blinder == registration.com_W_padding_mask);
    ModelRegistration wrong_secret_identity = registration;
    wrong_secret_identity.registration_digest[0] = wrong_secret_identity.registration_digest[0] == '0' ? '1' : '0';
    assert(!read_prover_commitment_secret(wrong_secret_identity, com_w_secret_path, recovered_com_w_blinder, &secret_reason));
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

    const VerifierContext context = make_verifier_context(graph, features, quant, result.trace.tables, result.logits, registration, proof_plan, "masked_full");
    if (verifier_context_digest(context).empty()) return 1;
    const std::string context_wire_path = "/tmp/zkgpt-verifier-context-v4-smoke.bin";
    std::string context_wire_reason;
    if (!write_verifier_context(context, context_wire_path, &context_wire_reason)) {
        std::cerr << "context write failed: " << context_wire_reason << "\n";
        return 1;
    }
    VerifierContext context_roundtrip;
    if (!read_verifier_context(context_wire_path, context_roundtrip, &context_wire_reason)) {
        std::cerr << "context read failed: " << context_wire_reason << "\n";
        return 1;
    }
    if (verifier_context_digest(context_roundtrip) != verifier_context_digest(context) || context_roundtrip.graph.src != context.graph.src ||
        context_roundtrip.graph.dst != context.graph.dst || context_roundtrip.public_features.values != context.public_features.values ||
        context_roundtrip.public_logits.values != context.public_logits.values || context_roundtrip.quant.range_bits != context.quant.range_bits ||
        context_roundtrip.tables.digest != context.tables.digest || context_roundtrip.backend != context.backend ||
        context_roundtrip.full_dataset_identity != context.full_dataset_identity || context_roundtrip.checkpoint_identity != context.checkpoint_identity ||
        context_roundtrip.cache_identity != context.cache_identity ||
        context_roundtrip.registration.registration_digest != context.registration.registration_digest ||
        context_roundtrip.plan.plan_digest != context.plan.plan_digest ||
        context_roundtrip.plan.circuit.circuit.back().bin_gates.size() != context.plan.circuit.circuit.back().bin_gates.size()) {
        std::cerr << "context roundtrip mismatch\n";
        return 1;
    }
    const std::size_t recipe_bytes = compact_plan_recipe_bytes(proof_plan);
    const std::size_t payload_bytes = verifier_context_payload_bytes(context);
    struct stat context_stat{};
    if (::stat(context_wire_path.c_str(), &context_stat) != 0 || context_stat.st_size < 0 ||
        static_cast<std::size_t>(context_stat.st_size) != payload_bytes + 20 || recipe_bytes >= (1ULL << 20) || payload_bytes >= (64ULL << 20)) {
        std::cerr << "compact context exceeds bounded synthetic size\n";
        return 1;
    }
    std::cout << "gat_adapter_smoke: compact_recipe_bytes=" << recipe_bytes << " v4_payload_bytes=" << payload_bytes
              << " v4_wire_bytes=" << context_stat.st_size << " recipe_cap=67108864 wire_cap=2147483648\n";
    for (std::size_t layer_id = 0; layer_id < proof_plan.circuit.circuit.size(); ++layer_id) {
        const layer& producer_layer = proof_plan.circuit.circuit[layer_id];
        const layer& reconstructed_layer = context_roundtrip.plan.circuit.circuit.at(layer_id);
        if (producer_layer.ty != reconstructed_layer.ty || producer_layer.size != reconstructed_layer.size ||
            producer_layer.uni_gates.size() != reconstructed_layer.uni_gates.size() ||
            producer_layer.bin_gates.size() != reconstructed_layer.bin_gates.size() ||
            producer_layer.native_relation_id != reconstructed_layer.native_relation_id ||
            producer_layer.native_family_id != reconstructed_layer.native_family_id || producer_layer.native_rows != reconstructed_layer.native_rows ||
            producer_layer.native_inner != reconstructed_layer.native_inner || producer_layer.native_cols != reconstructed_layer.native_cols ||
            producer_layer.native_lhs_offset != reconstructed_layer.native_lhs_offset ||
            producer_layer.native_rhs_offset != reconstructed_layer.native_rhs_offset ||
            producer_layer.native_output_offset != reconstructed_layer.native_output_offset) {
            std::cerr << "compact context reconstructed layer metadata mismatch\n";
            return 1;
        }
        for (std::size_t i = 0; i < producer_layer.uni_gates.size(); ++i) {
            const uniGate& a = producer_layer.uni_gates[i];
            const uniGate& b = reconstructed_layer.uni_gates[i];
            if (a.g != b.g || a.u != b.u || a.lu != b.lu || a.sc != b.sc) return 1;
        }
        for (std::size_t i = 0; i < producer_layer.bin_gates.size(); ++i) {
            const binGate& a = producer_layer.bin_gates[i];
            const binGate& b = reconstructed_layer.bin_gates[i];
            if (a.g != b.g || a.u != b.u || a.v != b.v || a.l != b.l || a.sc != b.sc) return 1;
        }
    }
    const auto expect_plan_reject = [&](const char* label, const std::function<void(GatCircuitPlan&)>& mutate) {
        GatCircuitPlan tampered = context_roundtrip.plan;
        mutate(tampered);
        if (validate_gat_circuit_plan(tampered, &context_wire_reason)) {
            std::cerr << "compact plan tamper unexpectedly accepted: " << label << "\n";
            return false;
        }
        return true;
    };
    assert(expect_plan_reject("MatMul rows", [](GatCircuitPlan& tampered) { tampered.matmuls.at(0).rows += 1; }));
    assert(expect_plan_reject("MatMul inner", [](GatCircuitPlan& tampered) { tampered.matmuls.at(0).inner += 1; }));
    assert(expect_plan_reject("MatMul cols", [](GatCircuitPlan& tampered) { tampered.matmuls.at(0).cols += 1; }));
    assert(expect_plan_reject("offset", [](GatCircuitPlan& tampered) { tampered.matmuls.at(0).lhs_offset += 1; }));
    assert(expect_plan_reject("Sparse GKR", [](GatCircuitPlan& tampered) { tampered.sparse_gkr.at(0).edge_count += 1; }));
    assert(expect_plan_reject("Lasso", [](GatCircuitPlan& tampered) { tampered.lasso.at(0).query_offset += 1; }));
    assert(expect_plan_reject("model architecture", [](GatCircuitPlan& tampered) { tampered.model_arch_id += ".tampered"; }));
    assert(expect_plan_reject("model shape", [](GatCircuitPlan& tampered) { tampered.model_shape.input_dim += 1; }));
    assert(expect_plan_reject("opening policy", [](GatCircuitPlan& tampered) { tampered.opening_policy = "independent_r_openings"; }));
    const auto expect_context_reject = [&](const char* label, const std::function<void(VerifierContext&)>& mutate) {
        VerifierContext tampered = context;
        mutate(tampered);
        if (verifier_context_digest(tampered) == verifier_context_digest(context)) {
            std::cerr << "context tamper did not change canonical digest: " << label << "\n";
            return false;
        }
        const std::string path = std::string("/tmp/zkgpt-context-tamper-") + label;
        std::remove(path.c_str());
        std::string reject_reason;
        const bool accepted = write_verifier_context(tampered, path, &reject_reason);
        std::remove(path.c_str());
        if (accepted) {
            std::cerr << "context tamper unexpectedly accepted: " << label << "\n";
            return false;
        }
        return true;
    };
    assert(expect_context_reject("graph-src", [](VerifierContext& tampered) { tampered.graph.src.at(0) += 1; }));
    assert(expect_context_reject("graph-dst", [](VerifierContext& tampered) { tampered.graph.dst.at(0) += 1; }));
    assert(expect_context_reject("registration-layout", [](VerifierContext& tampered) {
        tampered.registration.layout_digest[0] = tampered.registration.layout_digest[0] == '0' ? '1' : '0';
    }));
    assert(expect_context_reject("registration", [](VerifierContext& tampered) {
        tampered.registration.registration_digest[0] = tampered.registration.registration_digest[0] == '0' ? '1' : '0';
    }));
    GatCircuitPlan plan_gate_tampered = context_roundtrip.plan;
    if (plan_gate_tampered.matmuls.empty()) return 1;
    plan_gate_tampered.matmuls.front().rows += 1;
    if (validate_gat_circuit_plan(plan_gate_tampered, &context_wire_reason)) return 1;
    {
        std::ofstream trailing(context_wire_path, std::ios::binary | std::ios::app);
        trailing.put('\0');
    }
    if (read_verifier_context(context_wire_path, context_roundtrip, &context_wire_reason)) return 1;
    if (!write_verifier_context(context, context_wire_path, &context_wire_reason)) return 1;
    std::ifstream context_input(context_wire_path, std::ios::binary);
    const std::string context_wire((std::istreambuf_iterator<char>(context_input)), std::istreambuf_iterator<char>());
    if (context_wire.size() < 21) return 1;
    {
        std::ofstream truncated(context_wire_path, std::ios::binary | std::ios::trunc);
        truncated.write(context_wire.data(), static_cast<std::streamsize>(context_wire.size() - 1));
    }
    if (read_verifier_context(context_wire_path, context_roundtrip, &context_wire_reason)) return 1;
    if (!write_verifier_context(context, context_wire_path, &context_wire_reason)) return 1;
    const std::string old_context_wire_path = "/tmp/zkgpt-verifier-context-v2-malformed-smoke.bin";
    {
        std::ofstream old_context(old_context_wire_path, std::ios::binary | std::ios::trunc);
        const std::uint32_t old_version = 2;
        const std::uint64_t old_payload_size = (2ULL << 30) + 1;
        old_context.write("ZKGPTVC2", 8);
        old_context.write(reinterpret_cast<const char*>(&old_version), sizeof(old_version));
        old_context.write(reinterpret_cast<const char*>(&old_payload_size), sizeof(old_payload_size));
    }
    if (read_verifier_context(old_context_wire_path, context_roundtrip, &context_wire_reason)) return 1;
    std::remove(old_context_wire_path.c_str());
    std::remove(context_wire_path.c_str());
    bool context_rejected = false;
    try {
        ModelRegistration context_registration_tampered = registration;
        context_registration_tampered.n_in += 1;
        (void)make_verifier_context(graph, features, quant, result.trace.tables, result.logits, context_registration_tampered, proof_plan, "masked_full");
    } catch (const std::exception&) { context_rejected = true; }
    if (!context_rejected) return 1;

    const CommitmentWire com_aux = commit_private_segment(std::vector<Int>{1, 2}, "synthetic/aux", proof_plan.input_layout.n_in);
    const CommitmentWire com_L = commit_private_segment(std::vector<Int>{3, 4}, "synthetic/L_qry", proof_plan.input_layout.n_in);
    G1 basis_G;
    hashAndMapToG1(basis_G, "synthetic/native-basis/G");
    std::vector<G1> basis_generators(4);
    std::vector<Fr> basis_scalars{Fr(1), Fr(2), Fr(3), Fr(4)};
    for (std::size_t i = 0; i < basis_generators.size(); ++i) hashAndMapToG1(basis_generators[i], "synthetic/native-basis/g/" + std::to_string(i));
    const NativeProofStatement native_statement = make_native_proof_statement(context, com_aux, com_L, basis_G, basis_generators, basis_scalars, 4);
    assert(native_statement.valid());
    std::string expected_context_root = verifier_context_digest(context);
    expected_context_root.insert(0, 64 - expected_context_root.size(), '0');
    assert(native_statement.root() == expected_context_root);
    assert(native_statement.basis_digest == NativeProofStatement::basis_digest_for(basis_G, basis_generators, basis_scalars, 4));
    NativeProofStatement root_tampered = native_statement;
    root_tampered.statement_root_digest[0] = root_tampered.statement_root_digest[0] == '0' ? '1' : '0';
    assert(!root_tampered.valid());

    VerifierContext synthetic_context = context;
    synthetic_context.public_logits.values.assign(result.logits.values.size(), Int(1));
    zkgpt::NativeFieldVector gkr_output(result.logits.values.size(), Fr(1));
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
    for (u32 index = 0; index < gkr_output.size(); ++index) synthetic.C.circuit[1].uni_gates.emplace_back(index, index, 0, 1);
    synthetic.C.initSubset();
    synthetic.val.resize(2);
    synthetic.val[0].assign(gkr_output.begin(), gkr_output.end());
    synthetic.val[1].assign(gkr_output.begin(), gkr_output.end());
    const GkrOnlyArtifactResult gkr_result =
        prove_gkr_only_artifact(synthetic, synthetic_context, com_aux, com_L, basis_G, basis_generators, basis_scalars, 4, std::vector<F>(gkr_output.begin(), gkr_output.end()), {});
    assert(gkr_result.generated);
    assert(gkr_result.artifact.privacy_profile == "gkr_only_diagnostic");
    const std::string artifact_path = "/tmp/zkgpt-gkr-only-smoke.artifact";
    std::string artifact_reason;
    assert(gkr_result.artifact.write(artifact_path, &artifact_reason));
    GatProofArtifact serialized_artifact;
    assert(GatProofArtifact::read(artifact_path, serialized_artifact, &artifact_reason));
    NativeProofSession serialized_session;
    assert(serialized_artifact.native_session(serialized_session, &artifact_reason));
    assert(verify_gkr_transcript(serialized_session, synthetic.C, gkr_output, {}, &artifact_reason));
    assert(!verify_gat(synthetic_context, serialized_artifact, &artifact_reason));

    GatProofArtifact tampered_artifact = serialized_artifact;
    bool tampered_claim = false;
    for (NativeProofMessage& message : tampered_artifact.native_messages) {
        if (message.kind == NativeProofMessageKind::FieldVector && message.label.find("/claims") != std::string::npos &&
            message.payload.size() > sizeof(std::uint64_t) * 2) {
            const std::size_t first_value_offset = sizeof(std::uint64_t) * 2;
            message.payload[first_value_offset] = message.payload[first_value_offset] == '0' ? '1' : '0';
            tampered_claim = true;
            break;
        }
    }
    assert(tampered_claim);
    NativeProofSession tampered_session;
    assert(tampered_artifact.native_session(tampered_session, &artifact_reason));
    assert(!verify_gkr_transcript(tampered_session, synthetic.C, gkr_output, {}, &artifact_reason));
    assert(!tampered_artifact.valid_wire(&artifact_reason));
    std::remove(artifact_path.c_str());
    std::cout << "gat_adapter_smoke: PASS relation/rounding/PPI/descriptors\n";
    return 0;
}
