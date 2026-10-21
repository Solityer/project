#include "gat_proof.hpp"
#include "gat_formal_internal.hpp"

#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
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

Model make_formal_model(bool full_heads) {
    if (full_heads) return make_model();
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
    for (std::size_t layer = 0; layer < 2; ++layer) {
        std::ostringstream prefix;
        prefix << "hidden/layer" << layer << "/head0/";
        const std::size_t input_dim = 1;
        model.tensors.emplace(prefix.str() + "seq/kernel", make_tensor(prefix.str() + "seq/kernel", {1, input_dim, 1}, 0.25));
        model.tensors.emplace(prefix.str() + "attn_src/kernel", make_tensor(prefix.str() + "attn_src/kernel", {1, 1, 1}, 0.25));
        model.tensors.emplace(prefix.str() + "attn_dst/kernel", make_tensor(prefix.str() + "attn_dst/kernel", {1, 1, 1}, 0.25));
    }
    const std::string prefix = "output/head0/";
    model.tensors.emplace(prefix + "seq/kernel", make_tensor(prefix + "seq/kernel", {1, 1, 1}, 0.25));
    model.tensors.emplace(prefix + "attn_src/kernel", make_tensor(prefix + "attn_src/kernel", {1, 1, 1}, 0.25));
    model.tensors.emplace(prefix + "attn_dst/kernel", make_tensor(prefix + "attn_dst/kernel", {1, 1, 1}, 0.25));
    model.tensors.emplace(prefix + "output_bias", make_tensor(prefix + "output_bias", {1}, 0.25));
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

void write_standalone_fixture(const std::string& directory, const VerifierContext& context,
                              const GatProofArtifact& artifact) {
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
    const std::uint64_t wire_limit = gat_proof_artifact_wire_limit(context, &reason);
    if (wire_limit == 0) throw std::runtime_error("standalone proof artifact has no trusted wire capacity: " + reason);
    if (!artifact.write(artifact_path, &reason))
        throw std::runtime_error("standalone proof artifact write failed: " + reason);
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
            << "  \"plan_digest\": \"" << json_escape(context.plan_ref().plan_digest) << "\",\n"
            << "  \"circuit_id\": \"" << json_escape(context.plan_ref().circuit_id) << "\",\n"
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

int run_formal_fixture(const std::string& standalone_dir, bool proof_verify_only,
                       bool full_head_fixture, bool private_formal_fixture,
                       bool owner_check_only) {
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
    stage("model");
    Model formal_model = make_formal_model(full_head_fixture);
    stage("graph");
    Graph formal_graph;
    formal_graph.num_nodes = 1;
    formal_graph.num_features = formal_model.input_dim();
    formal_graph.graph_count = 1;
    formal_graph.num_classes = formal_model.classes();
    formal_graph.node_ptr = {0, 1};
    formal_graph.src = {0};
    formal_graph.dst = {0};
    formal_graph.edge_gid = {0};
    formal_graph.edge_ptr = {0, 1};
    formal_graph.validate();
    stage("probe-forward");
    Tensor2D formal_features(1, formal_model.input_dim());
    formal_features.values.assign(formal_model.input_dim(), 1);
    if (full_head_fixture && formal_features.values.size() > 1) formal_features.values[1] = 2;
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
    ForwardResult formal_forward;
    std::unique_ptr<detail::ValidatedFormalInputs> owned_formal_inputs;
    std::shared_ptr<const GatCircuitPlan> formal_plan_owner;
    double old_reference_validation_ms = 0.0;
    std::size_t old_reference_validation_calls = 0U;
    if (private_formal_fixture) {
        stage("private-move-owned-forward-and-validation");
        if (owner_check_only) {
            ForwardResult computed_forward = detail::compute_formal_forward_unvalidated(
                formal_graph, formal_features, formal_model, formal_quant);
            const auto validation_begin = std::chrono::steady_clock::now();
            for (std::size_t call = 0; call < 4U; ++call) {
                if (!validate_relation(formal_graph, formal_features, formal_model,
                                       formal_quant, computed_forward)) {
                    std::cerr << "old-path repeated canonical reference validation failed\n";
                    return 1;
                }
                ++old_reference_validation_calls;
            }
            old_reference_validation_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - validation_begin).count();
            owned_formal_inputs = std::make_unique<detail::ValidatedFormalInputs>(
                detail::ValidatedFormalInputs::consume(
                    std::move(formal_graph), std::move(formal_features),
                    std::move(formal_model), std::move(formal_quant),
                    std::move(computed_forward)));
            if (owned_formal_inputs->canonical_validation_calls() != 1U) {
                std::cerr << "untrusted private formal fixture did not perform one canonical reference validation\n";
                return 1;
            }
        } else {
            owned_formal_inputs = std::make_unique<detail::ValidatedFormalInputs>(
                detail::ValidatedFormalInputs::compute_and_own(
                    std::move(formal_graph), std::move(formal_features),
                    std::move(formal_model), std::move(formal_quant)));
            if (owned_formal_inputs->canonical_forward_generation_calls() != 1U ||
                owned_formal_inputs->canonical_forward_value_admission_calls() != 1U ||
                owned_formal_inputs->canonical_validation_calls() != 0U) {
                std::cerr << "fresh private formal fixture did not generate once and complete one value admission without canonical recomputation\n";
                return 1;
            }
        }
        formal_plan_owner = owned_formal_inputs->plan_owner();
    } else {
        formal_forward = run_forward(formal_graph, formal_features, formal_model, formal_quant);
        if (!formal_forward.trace.relation_ok) {
            std::cerr << "18-head formal fixture forward relation is invalid\n";
            return 1;
        }
        stage("plan");
        formal_plan_owner = std::make_shared<const GatCircuitPlan>(
            build_gat_circuit_plan(formal_graph, formal_model, formal_quant, formal_forward));
    }
    const Graph& proving_graph = owned_formal_inputs ? owned_formal_inputs->graph() : formal_graph;
    const Tensor2D& proving_features = owned_formal_inputs ? owned_formal_inputs->features() : formal_features;
    const Model& proving_model = owned_formal_inputs ? owned_formal_inputs->model() : formal_model;
    const QuantConfig& proving_quant = owned_formal_inputs ? owned_formal_inputs->quant() : formal_quant;
    const ForwardResult& proving_forward = owned_formal_inputs ? owned_formal_inputs->forward() : formal_forward;
    const GatCircuitPlan& formal_plan = *formal_plan_owner;
    if (owner_check_only) {
        static_assert(!std::is_copy_constructible_v<detail::ValidatedFormalInputs>);
        ForwardResult tampered = proving_forward;
        tampered.trace.hidden_layers.at(0).heads.at(0).activated.at(0) += 1;
        bool consume_rejected = false;
        try {
            (void)detail::ValidatedFormalInputs::consume(
                Graph(proving_graph), Tensor2D(proving_features), Model(proving_model),
                QuantConfig(proving_quant), ForwardResult(tampered));
        } catch (const std::invalid_argument&) {
            consume_rejected = true;
        }
        if (!consume_rejected) {
            std::cerr << "untrusted private consume API accepted a tampered forward trace\n";
            return 1;
        }
        bool public_api_rejected = false;
        try {
            (void)build_gat_circuit_plan(proving_graph, proving_model,
                                         proving_quant, tampered);
        } catch (const std::invalid_argument&) {
            public_api_rejected = true;
        }
        if (!public_api_rejected) {
            std::cerr << "public plan API accepted a tampered formal trace\n";
            return 1;
        }
    }
    if (!proof_verify_only && !owner_check_only) {
        stage("native-gate-negative-checks");
        const GatCircuitWitness assignment = build_gat_circuit_witness(formal_plan, proving_graph, proving_features, proving_model, proving_quant, proving_forward);
        std::string assignment_reason;
        if (!evaluate_gat_circuit_assignment(formal_plan, assignment.input_layer, proving_forward.logits, &assignment_reason)) throw std::runtime_error("honest native gate assignment rejected: " + assignment_reason);
        const auto checkpoint = [&](const std::string& suffix) { for (const CheckpointDescriptor& item : formal_plan.checkpoints) if (item.id == suffix) return item.coordinate_offset; throw std::runtime_error("missing native negative-check checkpoint: " + suffix); };
        const auto corrupt_coordinate = [&](const std::string& name, std::size_t coordinate, const Fr& increment = Fr(1)) { zkgpt::NativeFieldVector corrupted = assignment.input_layer; corrupted.at(coordinate) += increment; std::string reason; if (evaluate_gat_circuit_assignment(formal_plan, corrupted, proving_forward.logits, &reason) || reason.empty()) throw std::runtime_error("native constraint corruption was not rejected: " + name + ": " + reason); };
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
            if (family.family_id == "range/all") corrupt_coordinate("bit decomposition", family.query_offset);
        }
        corrupt_coordinate("output fixed denominator", checkpoint("rounding/output_alignment/output/head0") + 8);
        corrupt_coordinate("final fixed denominator", checkpoint("output/final_average") + 8);
        std::cout << "native_gate_negative_checks: PASS tuple/range/inverse/division/max/fixed-denominator\n";
        run_signed_round_div_gate_boundary_checks();
    } else if (proof_verify_only) {
        stage("complete-relation-proof-verify");
    }
    stage("registration");
    const ModelRegistration formal_registration = register_model(proving_model, proving_quant, formal_plan.input_layout, proving_model.manifest.model_arch_id);
    stage("context");
    VerifierContext formal_context = make_verifier_context(proving_graph, proving_features, proving_quant,
                                                           proving_forward.trace.tables, proving_forward.logits,
                                                           formal_registration, formal_plan_owner, "masked_full");
    formal_context.backend = "cpu";
    formal_context.full_dataset_identity = "canonical:synthetic";
    formal_context.checkpoint_identity = "baseline-local/checkpoints/synthetic_gat/integrity.sha256";
    formal_context.cache_identity = "baseline-local/data/cache/synthetic/integrity.sha256";
    if (owned_formal_inputs) {
        const std::string owned_context_path = "/tmp/zkgpt-owned-context-" + std::to_string(static_cast<long long>(::getpid()));
        std::string owned_context_reason;
        if (!detail::write_verifier_context_formal(*owned_formal_inputs, formal_context, owned_context_path, &owned_context_reason)) {
            std::cerr << "private formal context writer rejected its move-owned context: " << owned_context_reason << '\n';
            return 1;
        }
        std::remove(owned_context_path.c_str());
    }
    if (owner_check_only) {
        const std::string owned_context_path = "/tmp/zkgpt-owned-context-" + std::to_string(static_cast<long long>(::getpid()));
        std::string owned_context_reason;
        VerifierContext mismatched_context = formal_context;
        mismatched_context.public_logits.values.at(0) += 1;
        if (detail::write_verifier_context_formal(*owned_formal_inputs, mismatched_context, owned_context_path, &owned_context_reason) ||
            owned_context_reason.empty()) {
            std::remove(owned_context_path.c_str());
            std::cerr << "private formal context writer accepted logits detached from its move-owned inputs\n";
            return 1;
        }
        ProveResult mismatch = detail::prove_gat_formal(
            *owned_formal_inputs, formal_registration, mismatched_context,
            "masked_full");
        if (mismatch.status != ProveStatus::REJECTED || mismatch.diagnostic.empty()) {
            std::cerr << "private proof API accepted a context detached from its move-owned validated inputs\n";
            return 1;
        }
        GatCapacityReport altered_report = owned_formal_inputs->capacity();
        ++altered_report.num_edges;
        QuantConfig altered_quant = proving_quant;
        ++altered_quant.s_exp;
        Graph altered_graph = proving_graph;
        ++altered_graph.num_nodes;
        if (detail::formal_capacity_report_matches_plan(
                altered_report, proving_graph, proving_model, proving_quant,
                formal_plan) ||
            detail::formal_capacity_report_matches_plan(
                owned_formal_inputs->capacity(), proving_graph, proving_model,
                altered_quant, formal_plan) ||
            detail::formal_capacity_report_matches_plan(
                owned_formal_inputs->capacity(), altered_graph, proving_model,
                proving_quant, formal_plan)) {
            std::cerr << "private capacity identity accepted a changed report, quant config, or graph shape\n";
            return 1;
        }
        if (old_reference_validation_calls != 4U ||
            owned_formal_inputs->canonical_validation_calls() != 1U ||
            owned_formal_inputs->canonical_forward_generation_calls() != 0U ||
            owned_formal_inputs->canonical_forward_value_admission_calls() != 0U ||
            owned_formal_inputs->capacity_census_calls() != 1U ||
            owned_formal_inputs->capacity_restore_rebuild_calls() != 0U) {
            std::cerr << "reference validation invocation count differs from old/new formal path contract\n";
            return 1;
        }
        std::cout << "FORMAL_REFERENCE_VALIDATION old_calls="
                  << old_reference_validation_calls << " old_total_ms="
                  << old_reference_validation_ms << " new_calls="
                  << owned_formal_inputs->canonical_validation_calls()
                  << " new_ms=" << owned_formal_inputs->canonical_validation_ms()
                  << " exact_capacity_censuses="
                  << owned_formal_inputs->capacity_census_calls()
                  << " altered_report=REJECT changed_quant=REJECT changed_graph_shape=REJECT"
                  << " untrusted_consume_tampered_trace=REJECT public_tampered_trace=REJECT private_mismatched_context=REJECT\n";
        return 0;
    }
    stage("masked-full-prove");
    ProveResult formal;
    if (owned_formal_inputs)
        formal = detail::prove_gat_formal(*owned_formal_inputs, formal_registration,
                                          formal_context, "masked_full");
    else
        formal = prove_gat(formal_plan_owner, proving_graph, proving_features,
                           proving_model, proving_quant, proving_forward,
                           formal_registration, formal_context, "masked_full");
    if (formal.status != ProveStatus::GENERATED || !formal.has_artifact) {
        std::cerr << "masked_full formal smoke rejected: " << formal.diagnostic << "\n";
        return 1;
    }
    if (owned_formal_inputs &&
        (owned_formal_inputs->capacity_census_calls() != 1U ||
         owned_formal_inputs->canonical_forward_generation_calls() != 1U ||
         owned_formal_inputs->canonical_forward_value_admission_calls() != 1U ||
         owned_formal_inputs->canonical_validation_calls() != 0U ||
         owned_formal_inputs->capacity_restore_rebuild_calls() != 0U ||
         !formal_plan_owner->native_proof_backing_is_resident())) {
        std::cerr << "private formal proof did not retain its resident native-plan backing after Lasso\n";
        return 1;
    }
    if (owned_formal_inputs)
        std::cout << "FORMAL_CAPACITY_CENSUS exact_census_calls="
                  << owned_formal_inputs->capacity_census_calls()
                  << " cached_restore_rebuilds="
                  << owned_formal_inputs->capacity_restore_rebuild_calls()
                  << " retained_plan_backing="
                  << (formal_plan_owner->native_proof_backing_is_resident() ? "PASS" : "FAIL")
                  << " report_owner_plan_identity=PASS\n";
    if (formal.artifact.has_private_payload()) {
        std::cerr << "masked_full fixture serialized private witness payload\n";
        return 1;
    }
    const std::string formal_path = standalone_dir.empty() ? "/tmp/zkgpt-masked-full-formal-smoke.artifact" : standalone_dir + "/proof.artifact";
    std::string formal_reason;
    stage("artifact-write");
    if (!standalone_dir.empty()) {
        try {
            write_standalone_fixture(standalone_dir, formal_context, formal.artifact);
            std::cout << "gat_adapter_formal_fixture: GENERATED for independent verification and mutants\n";
            if (proof_verify_only) return 0;
        } catch (const std::exception& error) {
            std::cerr << "standalone fixture generation failed: " << error.what() << "\n";
            return 1;
        }
    }
    if (standalone_dir.empty() && !formal.artifact.write(formal_path, &formal_reason)) {
        std::cerr << "masked_full artifact write failed: " << formal_reason << '\n';
        return 1;
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
        if (!GatProofArtifact::read(formal_path, parsed_formal, &formal_reason) ||
            !verify_gat(formal_context, parsed_formal, &formal_reason)) {
            std::cerr << "masked_full fixture verification failed: " << formal_reason << '\n';
            return 1;
        }
    }

    if (proof_verify_only) {
        std::remove(formal_path.c_str());
        std::cout << "gat_adapter_formal_proof_verify: PASS masked_full artifact roundtrip and verifier call\n";
        return 0;
    }

    VerifierContext y_tampered = formal_context;
    y_tampered.public_logits.values.at(0) += 1;
    formal_reason.clear();
    if (verify_gat(y_tampered, parsed_formal, &formal_reason) || formal_reason.empty()) {
        std::cerr << "masked_full verifier accepted a public output mutation\n";
        return 1;
    }
    VerifierContext registration_tampered = formal_context;
    registration_tampered.registration.registration_digest[0] = registration_tampered.registration.registration_digest[0] == '0' ? '1' : '0';
    formal_reason.clear();
    if (verify_gat(registration_tampered, parsed_formal, &formal_reason) || formal_reason.empty()) {
        std::cerr << "masked_full verifier accepted a W registration binding mutation\n";
        return 1;
    }
    VerifierContext table_tampered = formal_context;
    table_tampered.tables.table_id += "-tampered";
    formal_reason.clear();
    if (verify_gat(table_tampered, parsed_formal, &formal_reason) || formal_reason.empty()) {
        std::cerr << "masked_full verifier accepted a static table mutation\n";
        return 1;
    }
    VerifierContext resource_tampered = formal_context;
    resource_tampered.tables.digest[0] = resource_tampered.tables.digest[0] == '0' ? '1' : '0';
    formal_reason.clear();
    if (verify_gat(resource_tampered, parsed_formal, &formal_reason) || formal_reason.empty()) {
        std::cerr << "masked_full verifier accepted a static resource digest mutation\n";
        return 1;
    }

    GatProofArtifact aux_tampered = parsed_formal;
    std::string& aux_root_digest = aux_tampered.com_aux.tensor_commitment.root_digest;
    if (aux_root_digest.empty()) {
        std::cerr << "masked_full artifact has no bound Aux tensor root to tamper\n";
        return 1;
    }
    aux_root_digest[0] = aux_root_digest[0] == '0' ? '1' : '0';
    formal_reason.clear();
    if (verify_gat(formal_context, aux_tampered, &formal_reason) || formal_reason.empty()) {
        std::cerr << "masked_full verifier did not reject the bound Aux tensor-root mutation with a reason\n";
        return 1;
    }
    GatProofArtifact l_tampered = parsed_formal;
    std::string& l_root_digest = l_tampered.com_L.tensor_commitment.root_digest;
    if (l_root_digest.empty()) {
        std::cerr << "masked_full artifact has no bound L_qry tensor root to tamper\n";
        return 1;
    }
    l_root_digest[0] = l_root_digest[0] == '0' ? '1' : '0';
    formal_reason.clear();
    if (verify_gat(formal_context, l_tampered, &formal_reason) || formal_reason.empty()) {
        std::cerr << "masked_full verifier did not reject the bound L_qry tensor-root mutation with a reason\n";
        return 1;
    }
    GatProofArtifact missing_range_family = parsed_formal;
    const auto missing_range_message = std::find_if(
        missing_range_family.native_messages.begin(), missing_range_family.native_messages.end(),
        [](const NativeProofMessage& message) {
            return message.label == "lasso/range_all/commitment";
        });
    if (missing_range_message == missing_range_family.native_messages.end()) {
        std::cerr << "masked_full proof has no range/all family commitment to omit\n";
        return 1;
    }
    missing_range_family.native_messages.erase(missing_range_message);
    formal_reason.clear();
    if (verify_gat(formal_context, missing_range_family, &formal_reason) || formal_reason.empty()) {
        std::cerr << "masked_full verifier accepted an artifact with the range/all Lasso family omitted\n";
        return 1;
    }
    GatProofArtifact linear_tampered = parsed_formal;
    bool linear_message_found = false;
    for (NativeProofMessage& message : linear_tampered.native_messages) {
        if (message.label == "combine/input-linear/response") {
            message.payload = "1";
            linear_message_found = true;
            break;
        }
    }
    if (!linear_message_found) {
        std::cerr << "masked_full proof has no input linear binding response to mutate\n";
        return 1;
    }
    formal_reason.clear();
    if (verify_gat(formal_context, linear_tampered, &formal_reason) || formal_reason.empty()) {
        std::cerr << "masked_full verifier accepted an input linear binding mutation\n";
        return 1;
    }
    GatProofArtifact transcript_tampered = parsed_formal;
    if (transcript_tampered.native_messages.empty()) {
        std::cerr << "masked_full proof has no transcript messages to mutate\n";
        return 1;
    }
    transcript_tampered.native_messages.front().payload += "tampered";
    formal_reason.clear();
    if (verify_gat(formal_context, transcript_tampered, &formal_reason) || formal_reason.empty()) {
        std::cerr << "masked_full verifier accepted a transcript mutation\n";
        return 1;
    }

    if (standalone_dir.empty()) std::remove(formal_path.c_str());
    std::cout << "gat_adapter_formal_smoke: PASS masked_full heads=" << proving_model.hidden_heads() * 2 + proving_model.manifest.output_heads
              << " serialize/verify/tamper\n";
    return 0;
}

int run_existing_artifact_mutants(const std::string& standalone_dir) {
    const std::string context_path = standalone_dir + "/verifier_context.bin";
    const std::string artifact_path = standalone_dir + "/proof.artifact";
    initialize_gat_field();
    std::cerr << "existing-proof-mutants: read-context\n";
    VerifierContext context;
    std::string reason;
    if (!read_verifier_context(context_path, context, &reason)) { std::cerr << "cannot read existing verifier context: " << reason << '\n'; return 1; }
    std::cerr << "existing-proof-mutants: read-artifact\n";
    GatProofArtifact artifact;
    if (!GatProofArtifact::read(artifact_path, artifact, &reason)) { std::cerr << "cannot read existing proof artifact: " << reason << '\n'; return 1; }
    std::cerr << "existing-proof-mutants: loaded\n";
    const CanonicalModelShape& shape = context.plan_ref().model_shape;
    if (artifact.privacy_profile != "masked_full" || shape.layers != 3 || shape.tensor_count != 56 || shape.output_heads != 2 ||
        context.plan_ref().fconn_claim_links.empty() || artifact.native_messages.empty() || artifact.has_private_payload() ||
        !artifact.com_aux.has_tensor_commitment || artifact.com_aux.tensor_commitment.root_digest.empty() ||
        !artifact.com_L.has_tensor_commitment || artifact.com_L.tensor_commitment.root_digest.empty()) {
        std::cerr << "existing artifact is not the canonical full-head masked fixture required for mutation checks\n";
        return 1;
    }
    const ModelRegistration alternate_registration = register_model(make_model(), context.quant, context.plan_ref().input_layout, context.model_arch_id);
    if (!alternate_registration.com_W.valid() || alternate_registration.com_W.point_hex == context.registration.com_W.point_hex) {
        std::cerr << "cannot construct a distinct canonical W commitment for mutation checks\n";
        return 1;
    }
    const auto expect_context_rejection = [&](const char* name, const std::function<void(VerifierContext&)>& mutate) {
        std::cerr << "existing-proof-mutants: start " << name << '\n';
        VerifierContext changed = context;
        if (changed.shared_plan) { changed.plan = changed.plan_ref(); changed.shared_plan.reset(); }
        mutate(changed);
        std::string rejection;
        if (verify_gat(changed, artifact, &rejection) || rejection.empty()) { std::cerr << "existing proof accepted " << name << " mutation\n"; return false; }
        std::cout << "existing-proof-mutant: REJECTED " << name << ": " << rejection << '\n';
        return true;
    };
    const auto expect_response_rejection = [&](const char* name, const std::function<void(GatProofArtifact&)>& mutate) {
        std::cerr << "existing-proof-mutants: start " << name << '\n';
        GatProofArtifact changed = artifact;
        mutate(changed);
        std::string session_reason;
        NativeProofSession authenticated_session;
        if (!changed.native_session(authenticated_session, &session_reason)) {
            std::cerr << "response mutation did not preserve canonical Fiat-Shamir replay: " << session_reason << '\n';
            return false;
        }
        changed.transcript_digest = authenticated_session.digest();
        std::string rejection;
        if (verify_gat(context, changed, &rejection) || rejection.empty()) {
            std::cerr << "existing proof accepted " << name << " mutation\n";
            return false;
        }
        if (rejection.find("hidden-linear Schnorr equation failed") == std::string::npos) {
            std::cerr << "response mutation was rejected before the native Schnorr equation: " << rejection << '\n';
            return false;
        }
        std::cout << "existing-proof-mutant: REJECTED " << name << " by native hidden-linear equation: " << rejection << '\n';
        return true;
    };
    const auto mutate_response = [](GatProofArtifact& value) {
        for (NativeProofMessage& message : value.native_messages) {
            if (message.label != "combine/input-linear/response") continue;
            if (message.kind != NativeProofMessageKind::Field) throw std::runtime_error("hidden-linear response is not a canonical scalar message");
            Fr response;
            response.setStr(message.payload, 16);
            response += 1;
            const std::string changed_payload = response.getStr(16);
            if (changed_payload == message.payload) throw std::runtime_error("hidden-linear response mutation was a no-op");
            message.payload = changed_payload;
            std::size_t matched_summaries = 0;
            for (ProofSection* section : {&value.Pi_Lasso, &value.Pi_MatMul, &value.Pi_GKR, &value.Pi_combine, &value.Pi_open})
                for (TranscriptMessage& summary : section->messages)
                    if (summary.sequence == message.sequence && summary.label == message.label) { summary.payload = changed_payload; ++matched_summaries; }
            if (matched_summaries != 1) throw std::runtime_error("hidden-linear response does not map to exactly one proof-section summary");
            return;
        }
        throw std::runtime_error("existing proof has no hidden-linear response to mutate");
    };
    const auto expect_artifact_rejection = [&](const char* name, const std::function<void(GatProofArtifact&)>& mutate) {
        std::cerr << "existing-proof-mutants: start " << name << '\n';
        GatProofArtifact changed = artifact;
        mutate(changed);
        std::string rejection;
        if (verify_gat(context, changed, &rejection) || rejection.empty()) { std::cerr << "existing proof accepted " << name << " mutation\n"; return false; }
        std::cout << "existing-proof-mutant: REJECTED " << name << ": " << rejection << '\n';
        return true;
    };
    const auto mutate_tensor_root = [](CommitmentWire& commitment) {
        if (!commitment.has_tensor_commitment) throw std::runtime_error("tensor commitment mutation target is absent");
        std::string& digest = commitment.tensor_commitment.root_digest;
        if (digest.empty()) throw std::runtime_error("tensor commitment mutation target is empty");
        const char original = digest[0];
        digest[0] = original == '0' ? '1' : '0';
        if (digest[0] == original) throw std::runtime_error("tensor commitment root mutation was a no-op");
    };
    const auto mutate_challenge = [](GatProofArtifact& value) {
        for (NativeProofMessage& message : value.native_messages) {
            if (message.kind != NativeProofMessageKind::Challenge) continue;
            if (message.payload.empty()) throw std::runtime_error("challenge mutation target is empty");
            Fr challenge;
            challenge.setStr(message.payload, 16);
            challenge += 1;
            const std::string changed_payload = challenge.getStr(16);
            if (changed_payload == message.payload) throw std::runtime_error("canonical challenge mutation was a no-op");
            message.payload = changed_payload;
            std::size_t matched_summaries = 0;
            for (ProofSection* section : {&value.Pi_Lasso, &value.Pi_MatMul, &value.Pi_GKR, &value.Pi_combine, &value.Pi_open})
                for (TranscriptMessage& summary : section->messages)
                    if (summary.sequence == message.sequence && summary.label == message.label) { summary.payload = changed_payload; ++matched_summaries; }
            if (matched_summaries != 1) throw std::runtime_error("challenge does not map to exactly one proof-section summary");
            return;
        }
        throw std::runtime_error("existing proof has no challenge message to mutate");
    };
    const auto expect_fs_rejection = [&]() {
        std::cerr << "existing-proof-mutants: start canonical-challenge-round\n";
        GatProofArtifact changed = artifact;
        mutate_challenge(changed);
        std::string rejection;
        if (verify_gat(context, changed, &rejection) || rejection.find("Fiat-Shamir challenge mismatch") == std::string::npos) {
            std::cerr << "canonical challenge mutation did not fail native Fiat-Shamir replay: " << rejection << '\n';
            return false;
        }
        std::cout << "existing-proof-mutant: REJECTED canonical-challenge-round by native Fiat-Shamir replay: " << rejection << '\n';
        return true;
    };
    const auto expect_w_commitment_rejection = [&]() {
        std::cerr << "existing-proof-mutants: start registered-W-commitment\n";
        VerifierContext changed = context;
        if (changed.shared_plan) { changed.plan = changed.plan_ref(); changed.shared_plan.reset(); }
        changed.registration = alternate_registration;
        changed.com_W = alternate_registration.com_W;
        std::string rejection;
        if (verify_gat(changed, artifact, &rejection) || rejection.empty()) {
            std::cerr << "existing proof accepted registered-W commitment mutation\n";
            return false;
        }
        std::cout << "existing-proof-mutant: REJECTED registered-W-commitment: " << rejection << '\n';
        return true;
    };
    if (!expect_context_rejection("metadata", [](VerifierContext& value) { value.model_arch_id += "/tampered"; }) ||
        !expect_context_rejection("registered-W-binding", [](VerifierContext& value) { value.registration.model_digest[0] = value.registration.model_digest[0] == '0' ? '1' : '0'; }) ||
        !expect_w_commitment_rejection() ||
        !expect_context_rejection("public-input", [](VerifierContext& value) { value.public_features.values.at(0) += 1; }) ||
        !expect_context_rejection("public-output", [](VerifierContext& value) { value.public_logits.values.at(0) += 1; }) ||
        !expect_context_rejection("FCONN-copy-link", [](VerifierContext& value) { ++value.plan.fconn_claim_links.back().output_offset; }) ||
        !expect_artifact_rejection("Aux-commitment", [&](GatProofArtifact& value) { mutate_tensor_root(value.com_aux); }) ||
        !expect_artifact_rejection("L_qry-commitment", [&](GatProofArtifact& value) { mutate_tensor_root(value.com_L); }) ||
        !expect_fs_rejection()) return 1;
    if (!expect_response_rejection("canonical-hidden-linear-response", mutate_response)) return 1;
    std::cout << "existing-artifact-mutants: PASS full-head shape=3x(8,8)+2 output, no reproving\n";
    return 0;
}

void append_legacy_u64(std::string& out, std::uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8)
        out.push_back(static_cast<char>((value >> shift) & 0xffU));
}

void append_legacy_string(std::string& out, const std::string& value) {
    append_legacy_u64(out, static_cast<std::uint64_t>(value.size()));
    out.append(value);
}

void append_legacy_statement(std::string& out, const NativeProofStatement& statement) {
    append_legacy_string(out, "zkgpt-native-statement-v4");
    append_legacy_string(out, statement.verifier_context_digest);
    append_legacy_string(out, statement.field_id);
    append_legacy_string(out, statement.plan_digest);
    append_legacy_string(out, statement.resources_digest);
    append_legacy_string(out, statement.tables_digest);
    append_legacy_string(out, statement.com_w_digest);
    append_legacy_string(out, statement.aux_digest);
    append_legacy_string(out, statement.l_digest);
    append_legacy_string(out, statement.basis_digest);
    append_legacy_u64(out, statement.l_dimension);
    append_legacy_string(out, statement.statement_root_digest);
    append_legacy_string(out, statement.relation_schema);
    append_legacy_string(out, statement.relation_digest);
    append_legacy_u64(out, static_cast<std::uint64_t>(statement.relation_order.size()));
    for (const std::string& relation : statement.relation_order)
        append_legacy_string(out, relation);
}

void append_legacy_transcript_prefix(std::string& out, const std::string& domain,
                                     const NativeProofStatement& statement,
                                     std::size_t message_count) {
    append_legacy_string(out, "zkgpt-native-fs-v4");
    append_legacy_string(out, domain);
    append_legacy_statement(out, statement);
    append_legacy_u64(out, static_cast<std::uint64_t>(message_count));
}

void append_legacy_message(std::string& out, const NativeProofMessage& message) {
    out.push_back(static_cast<char>(message.kind));
    append_legacy_u64(out, message.sequence);
    append_legacy_string(out, message.label);
    append_legacy_string(out, message.payload);
}

int run_legacy_challenge_reference(const std::string& standalone_dir) {
    const std::string input_path = standalone_dir + "/proof.artifact";
    initialize_gat_field();
    struct stat source_info{};
    if (::stat(input_path.c_str(), &source_info) != 0 || !S_ISREG(source_info.st_mode) || source_info.st_size <= 0) {
        std::cerr << "legacy challenge reference source is not a nonempty regular file\n";
        return 1;
    }
    const std::uint64_t wire_limit = static_cast<std::uint64_t>(source_info.st_size);
    std::string wire;
    std::string reason;
    if (!GatProofArtifact::read_wire_bytes(input_path, wire, wire_limit, &reason)) {
        std::cerr << "cannot read legacy challenge reference artifact: " << reason << '\n';
        return 1;
    }
    GatProofArtifact artifact;
    if (!GatProofArtifact::parse_wire(wire, artifact, wire_limit, &reason)) {
        std::cerr << "cannot parse legacy challenge reference artifact: " << reason << '\n';
        return 1;
    }
    if (artifact.native_messages.empty()) {
        std::cerr << "legacy challenge reference artifact has no native messages\n";
        return 1;
    }

    const auto started = std::chrono::steady_clock::now();
    std::string replayed_body;
    std::uint64_t challenge_count = 0;
    std::uint64_t prior_challenge_body_bytes = 0;
    for (std::size_t index = 0; index < artifact.native_messages.size(); ++index) {
        const NativeProofMessage& message = artifact.native_messages[index];
        if (message.sequence != static_cast<std::uint64_t>(index)) {
            std::cerr << "legacy challenge reference sequence is noncanonical at " << index << '\n';
            return 1;
        }
        if (message.kind == NativeProofMessageKind::Challenge) {
            if (message.label.compare(0, sizeof("challenge/") - 1, "challenge/") != 0 ||
                message.label.size() == sizeof("challenge/") - 1) {
                std::cerr << "legacy challenge reference label is malformed at " << index << '\n';
                return 1;
            }
            prior_challenge_body_bytes += static_cast<std::uint64_t>(replayed_body.size());
            std::string encoded;
            append_legacy_transcript_prefix(encoded, artifact.native_domain,
                                            artifact.native_statement, index);
            encoded.append(replayed_body);
            append_legacy_string(encoded, message.label.substr(sizeof("challenge/") - 1));
            Fr derived;
            derived.setHashOf(encoded.data(), encoded.size());
            Fr expected;
            try {
                expected.setStr(message.payload, 16);
            } catch (const std::exception& error) {
                std::cerr << "legacy challenge reference payload is malformed at " << index
                          << ": " << error.what() << '\n';
                return 1;
            }
            if (expected.getStr(16) != message.payload || derived != expected) {
                std::cerr << "legacy one-shot challenge differs from artifact at message " << index << '\n';
                return 1;
            }
            ++challenge_count;
        }
        append_legacy_message(replayed_body, message);
    }
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - started).count();
    std::cout << "legacy_challenge_reference: PASS native_messages=" << artifact.native_messages.size()
              << " challenge_messages=" << challenge_count
              << " encoded_message_body_bytes=" << replayed_body.size()
              << " exact_challenge_prefix_copy_bytes=" << prior_challenge_body_bytes
              << " elapsed_ms=" << elapsed_ms << '\n';
    std::cout.flush();
    return 0;
}

int run_existing_artifact_io_profile(const std::string& standalone_dir) {
    const std::string input_path = standalone_dir + "/proof.artifact";
    const std::string output_path = standalone_dir + "/proof.artifact.public-write";
    initialize_gat_field();
    const auto elapsed_ms = [](const std::chrono::steady_clock::time_point begin,
                               const std::chrono::steady_clock::time_point end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };
    std::string reason;
    struct stat source_info{};
    if (::stat(input_path.c_str(), &source_info) != 0 || !S_ISREG(source_info.st_mode) || source_info.st_size <= 0) {
        std::cerr << "artifact io profile source is not a nonempty regular file\n";
        return 1;
    }
    const std::uint64_t wire_limit = static_cast<std::uint64_t>(source_info.st_size);

    std::string input_wire;
    std::cerr << "artifact_io_profile: source-bytes begin\n" << std::flush;
    auto begin = std::chrono::steady_clock::now();
    if (!GatProofArtifact::read_wire_bytes(input_path, input_wire, wire_limit, &reason)) {
        std::cerr << "artifact io profile cannot read source bytes: " << reason << '\n';
        return 1;
    }
    const double source_read_ms = elapsed_ms(begin, std::chrono::steady_clock::now());
    std::cerr << "artifact_io_profile: source-bytes complete elapsed_ms=" << source_read_ms << '\n' << std::flush;

    GatProofArtifact artifact;
    std::cerr << "artifact_io_profile: wire-parse begin\n" << std::flush;
    begin = std::chrono::steady_clock::now();
    if (!GatProofArtifact::parse_wire(input_wire, artifact, wire_limit, &reason)) {
        std::cerr << "artifact io profile cannot parse source artifact: " << reason << '\n';
        return 1;
    }
    const double parse_ms = elapsed_ms(begin, std::chrono::steady_clock::now());
    std::cerr << "artifact_io_profile: wire-parse complete elapsed_ms=" << parse_ms << '\n' << std::flush;

    std::uint64_t challenge_count = 0;
    std::uint64_t encoded_body_bytes = 0;
    std::uint64_t replay_prior_message_bytes = 0;
    for (const NativeProofMessage& message : artifact.native_messages) {
        if (message.kind == NativeProofMessageKind::Challenge) {
            ++challenge_count;
            replay_prior_message_bytes += encoded_body_bytes;
        }
        encoded_body_bytes += static_cast<std::uint64_t>(1 + sizeof(std::uint64_t) + sizeof(std::uint64_t) +
                                                           message.label.size() + sizeof(std::uint64_t) + message.payload.size());
    }

    std::remove(output_path.c_str());
    std::cerr << "artifact_io_profile: public-write begin\n" << std::flush;
    begin = std::chrono::steady_clock::now();
    if (!artifact.write(output_path, wire_limit, &reason)) {
        std::cerr << "artifact io profile public write failed: " << reason << '\n';
        return 1;
    }
    const double public_write_including_wire_validation_ms = elapsed_ms(begin, std::chrono::steady_clock::now());
    std::cerr << "artifact_io_profile: public-write complete elapsed_ms="
              << public_write_including_wire_validation_ms << '\n' << std::flush;

    GatProofArtifact readback;
    std::cerr << "artifact_io_profile: readback begin\n" << std::flush;
    begin = std::chrono::steady_clock::now();
    if (!GatProofArtifact::read(output_path, readback, wire_limit, &reason)) {
        std::cerr << "artifact io profile public-write readback failed: " << reason << '\n';
        return 1;
    }
    const double public_write_readback_and_validation_ms = elapsed_ms(begin, std::chrono::steady_clock::now());
    std::cerr << "artifact_io_profile: readback complete elapsed_ms="
              << public_write_readback_and_validation_ms << '\n' << std::flush;
    std::string source_bytes;
    std::string output_bytes;
    source_bytes.swap(input_wire);
    if (!GatProofArtifact::read_wire_bytes(output_path, output_bytes, wire_limit, &reason)) {
        std::cerr << "artifact io profile cannot read public-written bytes: " << reason << '\n';
        return 1;
    }
    if (source_bytes != output_bytes || readback.transcript_digest != artifact.transcript_digest) {
        std::cerr << "artifact io profile public write changed canonical artifact bytes\n";
        return 1;
    }
    std::cout << "artifact_io_profile: PASS source_bytes=" << source_bytes.size()
              << " source_read_ms=" << source_read_ms
              << " parse_ms=" << parse_ms
              << " native_messages=" << artifact.native_messages.size()
              << " challenge_messages=" << challenge_count
              << " encoded_message_body_bytes=" << encoded_body_bytes
              << " challenge_prior_message_bytes_per_replay=" << replay_prior_message_bytes
              << " public_write_including_wire_validation_ms=" << public_write_including_wire_validation_ms
              << " public_write_readback_and_validation_ms=" << public_write_readback_and_validation_ms
              << " byte_identical=true\n";
    std::cout.flush();
    return 0;
}

}

int main(int argc, char** argv) {
    GatProofArtifact stale_transcript_version;
    stale_transcript_version.native_domain = "zkgpt-native-gat-fs-v6";
    NativeProofSession stale_session;
    std::string stale_reason;
    if (stale_transcript_version.native_session(stale_session, &stale_reason)) {
        std::cerr << "legacy GAT transcript domain was accepted\n";
        return 30;
    }

    bool formal = false;
    bool private_formal_fixture = false;
    bool owner_check_only = false;
    bool proof_verify_only = false;
    bool full_head_fixture = false;
    bool division_boundary_only = false;
    std::string artifact_mutants_dir;
    std::string artifact_io_profile_dir;
    std::string legacy_replay_reference_dir;
    std::string standalone_dir;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--formal") {
            formal = true;
        } else if (argument == "--formal-private") {
            formal = true;
            private_formal_fixture = true;
        } else if (argument == "--formal-owner-check") {
            formal = true;
            private_formal_fixture = true;
            owner_check_only = true;
        } else if (argument == "--formal-proof-verify") {
            formal = true;
            proof_verify_only = true;
        } else if (argument == "--formal-full-heads") {
            formal = true;
            full_head_fixture = true;
        } else if (argument == "--division-boundary-only") {
            division_boundary_only = true;
        } else if (argument == "--artifact-mutants" && index + 1 < argc) {
            artifact_mutants_dir = argv[++index];
        } else if (argument == "--artifact-io-profile" && index + 1 < argc) {
            artifact_io_profile_dir = argv[++index];
        } else if (argument == "--native-replay-legacy-reference" && index + 1 < argc) {
            legacy_replay_reference_dir = argv[++index];
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
    if (!legacy_replay_reference_dir.empty()) return run_legacy_challenge_reference(legacy_replay_reference_dir);
    if (!artifact_io_profile_dir.empty()) return run_existing_artifact_io_profile(artifact_io_profile_dir);
    if (!artifact_mutants_dir.empty()) return run_existing_artifact_mutants(artifact_mutants_dir);
    if (formal) {
        if (full_head_fixture && (proof_verify_only || standalone_dir.empty())) {
            std::cerr << "full-head formal fixture requires --standalone-dir and in-process mutant checks\n";
            return 2;
        }
        return run_formal_fixture(standalone_dir, proof_verify_only,
                                  full_head_fixture, private_formal_fixture,
                                  owner_check_only);
    }
    const DivisionResult tie = signed_round_div(-5, 2);
    assert(tie.value == -3 && tie.witness.valid(RoundingMode::NEAREST_TIES_AWAY));
    const DivisionResult exact = signed_round_div(4, 2);
    assert(exact.value == 2 && exact.witness.valid(RoundingMode::NEAREST_TIES_AWAY));

    const Graph graph = make_graph();
    assert(comparison_canonical_graph_digest(graph) == "89d6b57fe8c36ef4e2d55273a1dbba7be3062617ca6ab1fab904d187f980dd11");
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
    const std::string* registration_digests[] = {
        &registration.manifest_digest, &registration.model_digest,
        &registration.bounds_digest, &registration.quant_digest,
        &registration.layout_digest, &registration.binding_digest,
        &registration.registration_digest,
    };
    for (const std::string* digest : registration_digests) {
        if (digest->size() != 64 ||
            !std::all_of(digest->begin(), digest->end(), [](char value) {
                return (value >= '0' && value <= '9') ||
                       (value >= 'a' && value <= 'f');
            })) {
            throw std::runtime_error("model registration digest is not lowercase fixed-64 hexadecimal");
        }
    }
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
        context_roundtrip.plan_ref().plan_digest != context.plan_ref().plan_digest ||
        context_roundtrip.plan_ref().circuit.circuit.back().bin_gates.size() != context.plan_ref().circuit.circuit.back().bin_gates.size()) {
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
        const layer& reconstructed_layer = context_roundtrip.plan_ref().circuit.circuit.at(layer_id);
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
            if (a.g != b.g || a.u != b.u || a.lu != b.lu || producer_layer.coefficient(a) != reconstructed_layer.coefficient(b)) return 1;
        }
        for (std::size_t i = 0; i < producer_layer.bin_gates.size(); ++i) {
            const binGate& a = producer_layer.bin_gates[i];
            const binGate& b = reconstructed_layer.bin_gates[i];
            if (a.g != b.g || a.u != b.u || a.v != b.v || a.l != b.l || producer_layer.coefficient(a) != reconstructed_layer.coefficient(b)) return 1;
        }
    }
    const auto expect_plan_reject = [&](const char* label, const std::function<void(GatCircuitPlan&)>& mutate) {
        GatCircuitPlan tampered = context_roundtrip.plan_ref();
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
    GatCircuitPlan plan_gate_tampered = context_roundtrip.plan_ref();
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
    synthetic.C.circuit[1].reserve_gate_arrays(gkr_output.size(), 0);
    synthetic.C.circuit[1].reserve_coefficients(1);
    for (u32 index = 0; index < gkr_output.size(); ++index) synthetic.C.circuit[1].add_unary(index, index, 0, 1);
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
