#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "gatzk/model/gat.hpp"
#include "gatzk/protocol/challenges.hpp"
#include "gatzk/protocol/proof.hpp"
#include "gatzk/protocol/proof_codec.hpp"
#include "gatzk/protocol/prover.hpp"
#include "gatzk/protocol/schema.hpp"
#include "gatzk/protocol/trace.hpp"
#include "gatzk/protocol/verifier.hpp"
#include "gatzk/util/config.hpp"

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Fn>
std::string require_throws(Fn&& fn, const std::string& message) {
    try {
        fn();
    } catch (const std::exception& error) {
        return error.what();
    }
    throw std::runtime_error(message);
}

std::filesystem::path source_root() {
    return GATZK_TEST_SOURCE_DIR;
}

std::filesystem::path binary_root() {
    return GATZK_TEST_BINARY_DIR;
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("failed to read " + path.string());
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void write_file(const std::filesystem::path& path, std::string_view contents) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("failed to write " + path.string());
    output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
}

std::string without_config_key(std::string text, std::string_view key) {
    const auto prefix = std::string(key) + " =";
    std::istringstream input(text);
    std::ostringstream output;
    std::string line;
    bool removed = false;
    while (std::getline(input, line)) {
        if (line.rfind(prefix, 0) == 0) {
            removed = true;
            continue;
        }
        output << line << '\n';
    }
    require(removed, "test config did not contain " + std::string(key));
    return output.str();
}

void test_formal_config_family_fields_are_explicit() {
    for (const auto* dataset : {
             "cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv"}) {
        const auto config = gatzk::util::load_config(
            (source_root() / "configs" /
             (std::string(dataset) + "_secure.cfg")).string());
        require(config.layer_count == config.hidden_profile.size() + 1,
                std::string(dataset) + " L/profile mismatch");
        require(config.d_in_profile.size() == config.hidden_profile.size(),
                std::string(dataset) + " d_in_profile mismatch");
        require(config.K_out >= 1, std::string(dataset) + " K_out mismatch");
    }

    struct ScalingConfigExpectation {
        const char* file;
        std::size_t layers;
        std::size_t hidden_layers;
        std::size_t first_hidden_heads;
        std::size_t output_heads;
    };
    for (const auto& expected : std::vector<ScalingConfigExpectation>{
             {"cora_scaling_heads4.cfg", 2, 1, 4, 1},
             {"cora_scaling_depth3.cfg", 3, 2, 8, 1},
             {"cora_scaling_kout2.cfg", 2, 1, 8, 2},
             {"cora_scaling_combined.cfg", 3, 2, 8, 2},
         }) {
        const auto config = gatzk::util::load_config(
            (source_root() / "configs" / expected.file).string());
        require(config.layer_count == expected.layers
                    && config.hidden_profile.size() == expected.hidden_layers
                    && config.hidden_profile.front().head_count
                        == expected.first_hidden_heads
                    && config.K_out == expected.output_heads,
                std::string(expected.file)
                    + " architecture-scaling shape mismatch");
        require(config.srs_id == "cora_architecture_scaling_srs",
                std::string(expected.file)
                    + " must use the shared scaling SRS envelope");
    }

    const auto canonical = read_file(
        source_root() / "configs/unit_security_regression.cfg");
    const auto temporary = binary_root() / "family_config_negative";
    std::filesystem::create_directories(temporary);
    for (const auto* key : {"L", "hidden_profile", "d_in_profile", "K_out"}) {
        const auto path = temporary / (std::string("missing_") + key + ".cfg");
        write_file(path, without_config_key(canonical, key));
        const auto error = require_throws(
            [&] { (void)gatzk::util::load_config(path.string()); },
            std::string("missing ") + key + " must reject");
        require(error.find(key) != std::string::npos,
                std::string("missing-field diagnostic must name ") + key);
    }
    std::filesystem::remove_all(temporary);
}

void write_tensor(
    std::ostream& output,
    const std::string& name,
    const std::vector<std::size_t>& shape,
    double base) {
    const auto count = std::accumulate(
        shape.begin(), shape.end(), std::size_t{1}, std::multiplies<>());
    output << "TENSOR " << name << ' ' << shape.size();
    for (const auto dim : shape) output << ' ' << dim;
    output << ' ' << count << '\n';
    for (std::size_t index = 0; index < count; ++index) {
        if (index != 0) output << ' ';
        output << base + static_cast<double>(index % 3) * 0.01;
    }
    output << '\n';
}

struct HeadNames {
    std::string seq;
    std::string dst_kernel;
    std::string dst_bias;
    std::string src_kernel;
    std::string src_bias;
    std::string output_bias;
};

HeadNames head_names(const std::string& prefix) {
    return {
        prefix + "/seq", prefix + "/dst_kernel", prefix + "/dst_bias",
        prefix + "/src_kernel", prefix + "/src_bias",
        prefix + "/output_bias"};
}

void append_head_spec_fields(std::ostream& output, const HeadNames& names) {
    output << "\"seq_kernel\":\"" << names.seq
           << "\",\"attn_dst_kernel\":\"" << names.dst_kernel
           << "\",\"attn_dst_bias\":\"" << names.dst_bias
           << "\",\"attn_src_kernel\":\"" << names.src_kernel
           << "\",\"attn_src_bias\":\"" << names.src_bias
           << "\",\"output_bias\":\"" << names.output_bias << "\"";
}

void append_head_tensors(
    std::ostream& output,
    const HeadNames& names,
    std::size_t input_dim,
    std::size_t output_dim,
    double base) {
    write_tensor(output, names.seq, {1, input_dim, output_dim}, base);
    write_tensor(output, names.dst_kernel, {1, output_dim, 1}, base + 0.1);
    write_tensor(output, names.dst_bias, {1}, base + 0.2);
    write_tensor(output, names.src_kernel, {1, output_dim, 1}, base + 0.3);
    write_tensor(output, names.src_bias, {1}, base + 0.4);
    write_tensor(output, names.output_bias, {output_dim}, base + 0.5);
}

std::filesystem::path write_family_checkpoint_fixture() {
    const auto root = binary_root() / "family_checkpoint_fixture";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    const std::vector<HeadNames> hidden = {
        head_names("hidden/l0/h0"), head_names("hidden/l0/h1"),
        head_names("hidden/l1/h0")};
    const std::vector<HeadNames> output = {
        head_names("output/h0"), head_names("output/h1")};

    std::ostringstream manifest;
    manifest
        << "{\n"
        << "\"family_schema_version\":\"multi_layer_multi_head_v2\",\n"
        << "\"output_average_rule\":\"per_head_bias_then_arithmetic_mean\",\n"
        << "\"model_arch_id\":\"unit_family_L3_2x1_1x2_kout2\",\n"
        << "\"model_param_id\":\"test_fixture_not_trained\",\n"
        << "\"quant_cfg_id\":\"unit_fp_fixture\",\n"
        << "\"static_table_id\":\"tables:lrelu+elu+exp+range\",\n"
        << "\"degree_bound_id\":\"auto\",\n"
        << "\"L\":3,\"K_out\":2,\"C\":2,\n"
        << "\"d_in_profile\":[3,2],\n"
        << "\"hidden_layers\":["
        << "{\"layer_index\":0,\"input_dim\":3,\"head_count\":2,\"head_dim\":1},"
        << "{\"layer_index\":1,\"input_dim\":2,\"head_count\":1,\"head_dim\":2}],\n"
        << "\"hidden_profile\":["
        << "{\"head_count\":2,\"head_dim\":1},"
        << "{\"head_count\":1,\"head_dim\":2}],\n"
        << "\"hidden_head_specs\":[";
    for (std::size_t index = 0; index < hidden.size(); ++index) {
        if (index != 0) manifest << ',';
        const auto layer = index < 2 ? 0 : 1;
        const auto local = index < 2 ? index : 0;
        manifest << "{\"layer_index\":" << layer
                 << ",\"local_head_index\":" << local
                 << ",\"global_head_index\":" << index << ',';
        append_head_spec_fields(manifest, hidden[index]);
        manifest << '}';
    }
    manifest << "],\n\"output_head_specs\":[";
    for (std::size_t index = 0; index < output.size(); ++index) {
        if (index != 0) manifest << ',';
        manifest << "{\"head_index\":" << index << ',';
        append_head_spec_fields(manifest, output[index]);
        manifest << '}';
    }
    manifest << "]\n}\n";
    write_file(root / "manifest.json", manifest.str());

    std::ofstream tensors(root / "tensors.txt", std::ios::trunc);
    require(static_cast<bool>(tensors), "failed to create checkpoint tensors");
    append_head_tensors(tensors, hidden[0], 3, 1, 0.10);
    append_head_tensors(tensors, hidden[1], 3, 1, 0.20);
    append_head_tensors(tensors, hidden[2], 2, 2, 0.30);
    append_head_tensors(tensors, output[0], 2, 2, 0.40);
    append_head_tensors(tensors, output[1], 2, 2, 0.50);
    return root;
}

bool matrices_near(
    const gatzk::model::FloatMatrix& left,
    const gatzk::model::FloatMatrix& right) {
    if (left.size() != right.size()) return false;
    for (std::size_t row = 0; row < left.size(); ++row) {
        if (left[row].size() != right[row].size()) return false;
        for (std::size_t column = 0; column < left[row].size(); ++column) {
            if (std::abs(left[row][column] - right[row][column]) > 1e-10)
                return false;
        }
    }
    return true;
}

gatzk::protocol::ProtocolContext family_compiler_context(
    const gatzk::model::ModelParameters& model) {
    gatzk::protocol::ProtocolContext context;
    context.config = gatzk::util::load_config(
        (source_root() / "configs/unit_family_scaling.cfg").string());
    context.model = model;
    context.local.num_nodes = 3;
    context.local.num_features = 3;
    context.local.num_classes = 2;
    context.local.graph_count = 1;
    context.local.absolute_ids = {0, 1, 2};
    context.local.node_ptr = {0, 3};
    context.local.edge_ptr = {0, 6};
    context.local.edges = {
        {0, 0, 0, 0}, {1, 0, 0, 1}, {1, 1, 0, 2},
        {2, 1, 0, 3}, {0, 2, 0, 4}, {2, 2, 0, 5}};
    context.local.public_input.N_total = 3;
    context.local.public_input.N = 3;
    context.local.public_input.E = 6;
    context.local.public_input.G_batch = 1;
    context.tables.lrelu.resize(4);
    context.tables.elu.resize(4);
    context.tables.exp.resize(4);
    context.tables.range.resize(4);
    context.domains.fh = gatzk::algebra::RootOfUnityDomain::create("FH", 16);
    context.domains.edge = gatzk::algebra::RootOfUnityDomain::create("edge", 4096);
    context.domains.in = gatzk::algebra::RootOfUnityDomain::create("in", 8);
    context.domains.d = gatzk::algebra::RootOfUnityDomain::create("d", 4);
    context.domains.cat = gatzk::algebra::RootOfUnityDomain::create("cat", 4);
    context.domains.c = gatzk::algebra::RootOfUnityDomain::create("C", 4);
    context.domains.n = gatzk::algebra::RootOfUnityDomain::create("N", 8);
    context.kzg.manifest.degree_bound = 16383;
    context.kzg.manifest.max_instance_envelope = {
        {"max_layers", 3}, {"max_hidden_heads_per_layer", 2},
        {"max_hidden_head_dim", 2}, {"max_input_dim", 3},
        {"max_output_heads", 2}, {"max_classes", 2},
        {"max_graphs", 1}, {"max_nodes", 3}, {"max_edges", 6},
        {"max_total_nodes", 3}, {"max_concat_dim", 2},
        {"max_range_bits", 10}, {"max_coefficient_object_length", 9},
        {"max_object_family_count", 8192},
        {"max_coefficient_chunks", 1}, {"security_target_bits", 120}};
    return context;
}

bool contains(const std::vector<std::string>& values, std::string_view value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

void test_l3_heterogeneous_heads_kout2_family() {
    for (const auto* dataset : {
             "cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv"}) {
        const auto production = gatzk::model::inspect_checkpoint_bundle(
            (source_root() / "artifacts/checkpoints" /
             (std::string(dataset) + "_gat")).string());
        std::string production_reason;
        require(gatzk::model::checkpoint_bundle_matches_formal_proof_shape(
                    production, &production_reason),
                std::string(dataset) + " checkpoint rejected after strict loader: "
                    + production_reason);
    }
    const auto checkpoint_root = write_family_checkpoint_fixture();
    const auto info = gatzk::model::inspect_checkpoint_bundle(
        checkpoint_root.string());
    std::string reason;
    require(gatzk::model::checkpoint_bundle_matches_formal_proof_shape(
                info, &reason),
            "L3/Kout2 checkpoint manifest rejected: " + reason);
    require(info.layer_count == 3 && info.hidden_profile.size() == 2,
            "checkpoint layer family was not preserved");
    require(info.hidden_profile[0].head_count == 2
                && info.hidden_profile[0].head_dim == 1
                && info.hidden_profile[1].head_count == 1
                && info.hidden_profile[1].head_dim == 2,
            "heterogeneous hidden profile was not preserved");
    require(info.output_head_count == 2, "K_out=2 was not preserved");

    auto nonfirst_layer_tamper = info;
    nonfirst_layer_tamper.hidden_layers[1].input_dim = 1;
    require(!gatzk::model::checkpoint_bundle_matches_formal_proof_shape(
                nonfirst_layer_tamper, &reason),
            "non-first hidden layer shape tamper must reject");
    auto truncated_profile_tamper = info;
    truncated_profile_tamper.hidden_profile.pop_back();
    require(!gatzk::model::checkpoint_bundle_matches_formal_proof_shape(
                truncated_profile_tamper, &reason),
            "truncated hidden_profile must reject without out-of-bounds access");
    auto nonfirst_output_tamper = info;
    nonfirst_output_tamper.output_head_specs[1].head_index = 0;
    require(!gatzk::model::checkpoint_bundle_matches_formal_proof_shape(
                nonfirst_output_tamper, &reason),
            "non-first output head index tamper must reject");

    const auto model = gatzk::model::load_checkpoint_bundle_parameters(
        checkpoint_root.string());
    require(gatzk::model::hidden_family_dimension_chain_is_valid(model, &reason),
            "loaded L3/Kout2 checkpoint shape is invalid: " + reason);
    require(model.hidden_heads.size() == 3,
            "global hidden-head view must enumerate all layers");
    require(model.output_layer.heads.size() == 2,
            "output layer must enumerate P_out0/P_out1 sources");

    const gatzk::model::FloatMatrix features = {
        {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9}};
    const std::vector<gatzk::data::Edge> edges = {
        {0, 0, 0, 0}, {1, 0, 0, 1}, {1, 1, 0, 2},
        {2, 1, 0, 3}, {0, 2, 0, 4}, {2, 2, 0, 5}};
    const auto note = gatzk::model::forward_note_style(features, edges, model);
    const auto reference =
        gatzk::model::forward_reference_style(features, edges, model);
    require(note.hidden_layer_traces.size() == 2,
            "forward must retain both hidden layers");
    require(note.hidden_layer_traces[0].concat[0].size() == 2
                && note.hidden_layer_traces[1].input[0].size() == 2
                && note.hidden_layer_traces[1].concat[0].size() == 2,
            "forward hidden concat/input chain is invalid");
    require(note.hidden_head_traces.size() == 3,
            "forward compatibility trace must flatten every hidden layer");
    require(matrices_near(
                note.hidden_head_traces[2].H_agg,
                note.hidden_layer_traces[1].head_traces[0].H_agg),
            "non-first hidden layer is absent from flattened head traces");
    require(note.output_head_traces.size() == 2
                && note.output_head_values.size() == 2,
            "forward must retain both output heads");
    require(matrices_near(note.Y, reference.Y),
            "note/reference L3/Kout2 logits mismatch");

    auto context = family_compiler_context(model);
    require(gatzk::protocol::instance_within_authenticated_max_envelope(
                context, &reason),
            "family instance must fit the test-only scaling envelope: " + reason);
    const auto artifact = std::make_shared<gatzk::protocol::ProtocolCompilerArtifact>(
        gatzk::protocol::build_protocol_compiler_artifact(context));
    context.compiler_artifact = artifact;
    require(gatzk::protocol::validate_protocol_compiler_artifact(
                context, *artifact, &reason),
            "family compiler artifact failed canonical validation: " + reason);

    const auto labels = gatzk::protocol::dynamic_commitment_labels(context);
    for (const auto* label : {
             "P_h0_H_prime", "P_h1_H_prime", "P_h2_H_prime",
             "P_H_cat_l0", "P_H_cat", "P_out0_Y_prime",
             "P_out1_Y_prime", "P_Y_o0", "P_Y_o1"}) {
        require(contains(labels, label),
                std::string("dynamic commitment labels missing ") + label);
    }
    const auto& schema = artifact->proof_schema.entries;
    for (const auto* label : {
             "P_H_cat_l0", "P_H_cat", "P_out0_Y_prime", "P_out1_Y_prime"}) {
        require(std::any_of(schema.begin(), schema.end(), [&](const auto& entry) {
                    return entry.tag == label;
                }),
                std::string("ProofSchema missing ") + label);
    }
    for (const auto* challenge : {
             "xi_h0", "xi_h1", "xi_h2", "xi_cat_l0", "xi_cat",
             "y_proj_out_o0", "y_proj_out_o1"}) {
        require(std::any_of(
                    artifact->random_identity_ledger.begin(),
                    artifact->random_identity_ledger.end(),
                    [&](const auto& entry) {
                        return entry.object_or_challenge == challenge;
                    }),
                std::string("compiler challenge ledger missing ") + challenge);
    }
    std::filesystem::remove_all(checkpoint_root);
}

void test_family_secure_round_trip_if_configured() {
    const auto* srs_path = std::getenv("GATZK_FAMILY_TEST_SRS_PATH");
    if (srs_path == nullptr || *srs_path == '\0') {
        std::cerr
            << "skipping opt-in L3/Kout2 secure round trip; "
            << "set GATZK_FAMILY_TEST_SRS_PATH after the independent test ceremony\n";
        return;
    }
    auto config = gatzk::util::load_config(
        (source_root() / "configs/unit_family_scaling.cfg").string());
    config.srs_path = srs_path;
    if (const auto* srs_id = std::getenv("GATZK_FAMILY_TEST_SRS_ID");
        srs_id != nullptr && *srs_id != '\0') {
        config.srs_id = srs_id;
    }
    auto context = gatzk::protocol::build_context(config, nullptr);
    auto trace = gatzk::protocol::build_trace(context, nullptr);
    auto proof = gatzk::protocol::prove(context, trace, nullptr);
    require(gatzk::protocol::verify(context, proof),
            "L3/Kout2 in-memory proof must verify before codec");
    const auto wire = gatzk::protocol::serialize_proof(proof);
    const auto parsed = gatzk::protocol::parse_proof(context, wire);
    require(gatzk::protocol::verify(context, parsed),
            "L3/Kout2 proof codec round trip must verify");

    auto nonfirst_hidden_tamper = parsed;
    auto hidden_commitment = std::find_if(
        nonfirst_hidden_tamper.dynamic_commitments.begin(),
        nonfirst_hidden_tamper.dynamic_commitments.end(),
        [](const auto& entry) { return entry.first == "P_h2_H_prime"; });
    require(hidden_commitment
                != nonfirst_hidden_tamper.dynamic_commitments.end(),
            "L3 proof must commit the non-first-layer hidden head");
    hidden_commitment->second.point = gatzk::crypto::g1_add(
        hidden_commitment->second.point, context.kzg.g1_generator);
    require(!gatzk::protocol::verify(context, nonfirst_hidden_tamper),
            "non-first-layer commitment tamper must reject");

    auto nonfirst_output_tamper = parsed;
    auto output_commitment = std::find_if(
        nonfirst_output_tamper.dynamic_commitments.begin(),
        nonfirst_output_tamper.dynamic_commitments.end(),
        [](const auto& entry) { return entry.first == "P_out1_Y_prime"; });
    require(output_commitment
                != nonfirst_output_tamper.dynamic_commitments.end(),
            "L3 proof must commit the second output head");
    output_commitment->second.point = gatzk::crypto::g1_add(
        output_commitment->second.point, context.kzg.g1_generator);
    require(!gatzk::protocol::verify(context, nonfirst_output_tamper),
            "non-first-output commitment tamper must reject");
}

}  // namespace

int main() {
    try {
        test_formal_config_family_fields_are_explicit();
        test_l3_heterogeneous_heads_kout2_family();
        test_family_secure_round_trip_if_configured();
    } catch (const std::exception& error) {
        std::cerr << "family scaling test failure: " << error.what() << '\n';
        return 1;
    }
    std::cout << "family scaling tests passed\n";
    return 0;
}
