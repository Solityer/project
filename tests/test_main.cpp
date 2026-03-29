#include <functional>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/crypto/kzg.hpp"
#include "gatzk/data/loader.hpp"
#include "gatzk/model/gat.hpp"
#include "gatzk/protocol/challenges.hpp"
#include "gatzk/protocol/prover.hpp"
#include "gatzk/protocol/trace.hpp"
#include "gatzk/protocol/verifier.hpp"
#include "gatzk/util/config.hpp"

namespace {

using gatzk::algebra::FieldElement;
using gatzk::protocol::ProtocolContext;
using gatzk::protocol::Proof;
using gatzk::protocol::TraceArtifacts;

std::filesystem::path repo_root() {
    return std::filesystem::path(__FILE__).parent_path().parent_path();
}

std::string repo_path(const std::string& relative) {
    return (repo_root() / relative).string();
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string require_throws(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::exception& error) {
        return error.what();
    }
    throw std::runtime_error("expected exception but none was thrown");
}

gatzk::util::AppConfig small_debug_config() {
    gatzk::util::AppConfig config;
    config.project_root = repo_root().string();
    config.dataset = "cora";
    config.data_root = "data";
    config.cache_root = "data/cache";
    config.export_dir = "runs/test_small_real";
    config.hidden_dim = 4;
    config.num_classes = 7;
    config.range_bits = 12;
    config.seed = 11;
    config.local_nodes = 32;
    config.center_node = 0;
    config.allow_synthetic_model = true;
    config.dump_trace = false;
    config.auto_prepare_dataset = false;
    config.prove_enabled = true;
    return config;
}

gatzk::util::AppConfig full_graph_debug_config() {
    auto config = gatzk::util::load_config(repo_path("configs/cora_full.cfg"));
    config.allow_synthetic_model = true;
    return config;
}

const ProtocolContext& full_graph_debug_context() {
    static const auto context = []() {
        const auto config = full_graph_debug_config();
        return gatzk::protocol::build_context(config);
    }();
    return context;
}

struct ProofFixture {
    ProtocolContext context;
    TraceArtifacts trace;
    Proof proof;
};

const ProofFixture& small_proof_fixture() {
    static const auto fixture = []() {
        const auto context = gatzk::protocol::build_context(small_debug_config());
        const auto trace = gatzk::protocol::build_trace(context);
        const auto proof = gatzk::protocol::prove(context, trace);
        return ProofFixture{context, trace, proof};
    }();
    return fixture;
}

std::unordered_map<std::string, gatzk::crypto::Commitment> commitment_map(
    const std::vector<std::pair<std::string, gatzk::crypto::Commitment>>& entries) {
    std::unordered_map<std::string, gatzk::crypto::Commitment> out;
    for (const auto& [name, commitment] : entries) {
        out[name] = commitment;
    }
    return out;
}

const gatzk::protocol::DomainOpeningBundle& bundle_by_name(const Proof& proof, const std::string& name) {
    for (const auto& [bundle_name, bundle] : proof.domain_openings) {
        if (bundle_name == name) {
            return bundle;
        }
    }
    throw std::runtime_error("missing bundle " + name);
}

bool bundle_contains(const gatzk::protocol::DomainOpeningBundle& bundle, const std::string& label) {
    for (const auto& [name, _] : bundle.values) {
        if (name == label) {
            return true;
        }
    }
    return false;
}

bool external_eval_contains(const Proof& proof, const std::string& label) {
    for (const auto& [name, _] : proof.external_evaluations) {
        if (name == label) {
            return true;
        }
    }
    return false;
}

void test_full_graph_config_parse() {
    const auto cora = gatzk::util::load_config(repo_path("configs/cora_full.cfg"));
    require(cora.dataset == "cora" && cora.local_nodes == 2708 && cora.prove_enabled, "cora_full.cfg mismatch");
    require(!cora.checkpoint_bundle.empty(), "cora_full.cfg must define checkpoint_bundle");
    require(!cora.allow_synthetic_model, "cora_full.cfg must not enable synthetic parameters");
}

void test_checkpoint_bundle_manifest_detects_architecture_mismatch() {
    const auto config = gatzk::util::load_config(repo_path("configs/cora_full.cfg"));
    const auto bundle = gatzk::model::inspect_checkpoint_bundle(repo_path(config.checkpoint_bundle));
    require(bundle.hidden_head_count == 8, "expected original GAT bundle to report 8 hidden heads");
    require(bundle.has_output_attention_head, "expected original GAT bundle to include output attention head");

    std::string reason;
    require(
        !gatzk::model::checkpoint_bundle_matches_single_head_protocol(bundle, &reason),
        "original GAT bundle must not be accepted by the current single-head protocol model");
    require(reason.find("hidden_head_count=8") != std::string::npos, "mismatch reason must expose hidden head count");
}

void test_formal_cora_full_refuses_synthetic_fallback() {
    const auto config = gatzk::util::load_config(repo_path("configs/cora_full.cfg"));
    const auto message = require_throws([&]() {
        (void)gatzk::protocol::build_context(config);
    });
    require(
        message.find("incompatible with the current single-head protocol model") != std::string::npos,
        "formal cora_full route must fail on incompatible real checkpoint instead of using synthetic params");
}

void test_extract_full_graph_fast_path() {
    gatzk::data::GraphDataset dataset;
    dataset.name = "synthetic";
    dataset.num_nodes = 3;
    dataset.num_features = 2;
    dataset.num_classes = 2;
    dataset.features_fp = {
        {1.0, 0.0},
        {0.0, 1.0},
        {0.5, 0.5},
    };
    dataset.features = {
        {FieldElement(1), FieldElement(0)},
        {FieldElement(0), FieldElement(1)},
        {FieldElement(1), FieldElement(1)},
    };
    dataset.labels = {0, 1, 0};
    dataset.edges = {{0, 1}, {1, 2}};

    const auto local = gatzk::data::extract_local_subgraph(dataset, 0, dataset.num_nodes);
    require(local.num_nodes == dataset.num_nodes, "full graph extraction node count mismatch");
    require(local.absolute_ids.size() == dataset.num_nodes, "full graph absolute id size mismatch");
    require(local.absolute_ids[0] == 0 && local.absolute_ids[1] == 1 && local.absolute_ids[2] == 2, "full graph extraction should preserve natural order");
    require(local.edges.size() == dataset.edges.size() + dataset.num_nodes, "full graph extraction should append self loops");
    require(local.features_fp == dataset.features_fp, "full graph extraction should preserve float features");
}

void test_is_full_dataset_context() {
    const auto& context = full_graph_debug_context();
    require(context.local.num_nodes == context.dataset.num_nodes, "full config should build full dataset context");
    require(context.local.absolute_ids.front() == 0, "full dataset order should start from node 0");
    require(context.local.absolute_ids.back() == context.dataset.num_nodes - 1, "full dataset order should preserve natural ids");
    require(!context.local.features_fp.empty(), "full dataset context should retain float features");
    require(context.local.features_fp.front().size() == context.local.num_features, "float feature width mismatch");
}

void test_no_absolute_path_dependency() {
    const auto config = gatzk::util::load_config(repo_path("configs/cora_full.cfg"));
    require(config.project_root == repo_root().string(), "project_root should derive from config location");
    require(config.export_dir.rfind("/home/", 0) != 0, "export_dir must not be an absolute home path");
    require(config.checkpoint_bundle.rfind("/home/", 0) != 0, "checkpoint_bundle must not be an absolute home path");
}

void test_selector_padding_consistency() {
    const auto& context = full_graph_debug_context();
    const auto& q_edge_valid = context.public_polynomials.at("P_Q_edge_valid").data;
    const auto& q_qry_src = context.public_polynomials.at("P_Q_qry_src").data;
    const auto& q_qry_dst = context.public_polynomials.at("P_Q_qry_dst").data;
    const auto& q_n = context.public_polynomials.at("P_Q_N").data;
    const auto& q_proj = context.public_polynomials.at("P_Q_proj_valid").data;
    const auto& q_d = context.public_polynomials.at("P_Q_d_valid").data;

    for (std::size_t i = 0; i < context.local.edges.size(); ++i) {
        require(q_edge_valid[i] == FieldElement::one(), "edge valid selector must be 1 on real edges");
        require(q_qry_src[i] == q_edge_valid[i] && q_qry_dst[i] == q_edge_valid[i], "route query selectors must match edge valid selector");
    }
    require(q_edge_valid[context.local.edges.size()] == FieldElement::zero(), "edge valid selector padding mismatch");

    for (std::size_t i = 0; i < context.local.num_nodes; ++i) {
        require(q_n[i] == FieldElement::one(), "node selector must be 1 on real nodes");
    }
    require(q_n[context.local.num_nodes] == FieldElement::zero(), "node selector padding mismatch");

    for (std::size_t i = 0; i < context.local.num_features; ++i) {
        require(q_proj[i] == FieldElement::one(), "projection selector must be 1 on real feature indices");
    }
    require(q_proj[context.local.num_features] == FieldElement::zero(), "projection selector padding mismatch");

    for (std::size_t i = 0; i < context.config.hidden_dim; ++i) {
        require(q_d[i] == FieldElement::one(), "d selector must be 1 on real hidden dimensions");
    }
    require(q_d[context.config.hidden_dim] == FieldElement::zero(), "d selector padding mismatch");
}

void test_kzg_batch_opening() {
    const auto key = gatzk::crypto::KZG::setup(7);
    const auto domain = gatzk::algebra::RootOfUnityDomain::create("tiny", 4);
    const auto poly_a = gatzk::algebra::Polynomial::from_coefficients(
        "A",
        {FieldElement(1), FieldElement(2), FieldElement(3)});
    const auto poly_b = gatzk::algebra::Polynomial::from_evaluations(
        "B",
        {FieldElement(4), FieldElement(5), FieldElement(6), FieldElement(7)},
        domain);
    const auto commitment_a = gatzk::crypto::KZG::commit("A", poly_a, key);
    const auto commitment_b = gatzk::crypto::KZG::commit("B", poly_b, key);
    const std::vector<FieldElement> points = {FieldElement(9), FieldElement(11)};
    const std::vector<std::vector<FieldElement>> values = {
        {poly_a.evaluate(points[0]), poly_a.evaluate(points[1])},
        {poly_b.evaluate(points[0]), poly_b.evaluate(points[1])},
    };
    const auto witness = gatzk::crypto::KZG::open_batch(
        {commitment_a, commitment_b},
        points,
        values,
        FieldElement(13),
        key);
    require(
        gatzk::crypto::KZG::verify_batch(
            {commitment_a, commitment_b},
            points,
            values,
            FieldElement(13),
            witness,
            key),
        "batch opening verification failed");
}

void test_transcript_order_consistency() {
    const auto& fixture = small_proof_fixture();
    const auto replayed = gatzk::protocol::replay_challenges(
        fixture.context,
        commitment_map(fixture.proof.dynamic_commitments),
        commitment_map(fixture.proof.quotient_commitments));
    require(replayed == fixture.proof.challenges, "replayed transcript must match proof challenges");
    require(fixture.proof.challenges.at("eta_dst") == fixture.trace.challenges.at("eta_dst"), "eta_dst mismatch");
    require(fixture.proof.challenges.at("lambda_psq") == fixture.trace.challenges.at("lambda_psq"), "lambda_psq mismatch");
}

void test_agg_witness_commitment_opening_consistency() {
    const auto& fixture = small_proof_fixture();
    const auto dynamic = commitment_map(fixture.proof.dynamic_commitments);
    require(dynamic.contains("P_H_agg"), "missing P_H_agg commitment");
    require(dynamic.contains("P_H_agg_star"), "missing P_H_agg_star commitment");
    require(dynamic.contains("P_a_agg"), "missing P_a_agg commitment");
    require(dynamic.contains("P_b_agg"), "missing P_b_agg commitment");
    require(dynamic.contains("P_Acc_agg"), "missing P_Acc_agg commitment");
    require(external_eval_contains(fixture.proof, "mu_agg"), "missing mu_agg external evaluation");

    const auto& bundle_d = bundle_by_name(fixture.proof, "d");
    const auto& bundle_n = bundle_by_name(fixture.proof, "N");
    require(bundle_contains(bundle_d, "P_a_agg"), "d bundle missing P_a_agg");
    require(bundle_contains(bundle_d, "P_b_agg"), "d bundle missing P_b_agg");
    require(bundle_contains(bundle_d, "P_Acc_agg"), "d bundle missing P_Acc_agg");
    require(bundle_contains(bundle_n, "P_H_agg_star"), "N bundle missing P_H_agg_star");
}

void test_prove_verify_round_trip() {
    const auto& fixture = small_proof_fixture();
    require(gatzk::protocol::verify(fixture.context, fixture.proof), "round-trip prove/verify should accept");
}

void test_tampered_witness_fails() {
    const auto& fixture = small_proof_fixture();
    auto tampered = fixture.proof;
    require(!tampered.witness_scalars.empty(), "proof should contain route witness scalars");
    tampered.witness_scalars.front().second += FieldElement::one();
    require(!gatzk::protocol::verify(fixture.context, tampered), "tampered witness scalar must be rejected");
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"full_graph_config_parse", test_full_graph_config_parse},
        {"checkpoint_bundle_manifest_detects_architecture_mismatch", test_checkpoint_bundle_manifest_detects_architecture_mismatch},
        {"formal_cora_full_refuses_synthetic_fallback", test_formal_cora_full_refuses_synthetic_fallback},
        {"extract_full_graph_fast_path", test_extract_full_graph_fast_path},
        {"is_full_dataset_context", test_is_full_dataset_context},
        {"no_absolute_path_dependency", test_no_absolute_path_dependency},
        {"selector_padding_consistency", test_selector_padding_consistency},
        {"kzg_batch_opening", test_kzg_batch_opening},
        {"transcript_order_consistency", test_transcript_order_consistency},
        {"agg_witness_commitment_opening_consistency", test_agg_witness_commitment_opening_consistency},
        {"prove_verify_round_trip", test_prove_verify_round_trip},
        {"tampered_witness_fails", test_tampered_witness_fails},
    };

    try {
        for (const auto& [name, test] : tests) {
            test();
            std::cout << "[PASS] " << name << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
    return 0;
}
