#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gatzk/algebra/domain_evaluation.hpp"
#include "gatzk/algebra/eval_backend.hpp"
#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/crypto/curve.hpp"
#include "gatzk/crypto/kzg.hpp"
#include "gatzk/data/loader.hpp"
#include "gatzk/protocol/challenges.hpp"
#include "gatzk/protocol/prover.hpp"
#include "gatzk/protocol/schema.hpp"
#include "gatzk/protocol/trace.hpp"
#include "gatzk/protocol/verifier.hpp"
#include "gatzk/util/config.hpp"

namespace {

using gatzk::algebra::FieldElement;

std::filesystem::path repo_root() {
    auto cwd = std::filesystem::current_path();
    if (std::filesystem::exists(cwd / "configs" / "cora_secure.cfg")) {
        return cwd;
    }
    if (std::filesystem::exists(cwd.parent_path() / "configs" / "cora_secure.cfg")) {
        return cwd.parent_path();
    }
    return cwd;
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Fn>
void require_throws(Fn&& fn, const std::string& message) {
    bool thrown = false;
    try {
        fn();
    } catch (const std::exception&) {
        thrown = true;
    }
    require(thrown, message);
}

struct SecureFixture {
    gatzk::protocol::ProtocolContext context;
    gatzk::protocol::TraceArtifacts trace;
    gatzk::protocol::Proof proof;
};

void strip_opening_material(gatzk::crypto::Commitment& commitment) {
    commitment.opening_polynomial.reset();
}

void strip_context_to_verifier_only(gatzk::protocol::ProtocolContext& context) {
    context.kzg.pk = {};
    for (auto& [name, commitment] : context.public_commitments) {
        (void)name;
        strip_opening_material(commitment);
    }
    for (auto& [name, commitment] : context.static_commitments) {
        (void)name;
        strip_opening_material(commitment);
    }
}

void strip_proof_to_verifier_only(gatzk::protocol::Proof& proof) {
    for (auto& [name, commitment] : proof.dynamic_commitments) {
        (void)name;
        strip_opening_material(commitment);
    }
    for (auto& [name, commitment] : proof.quotient_commitments) {
        (void)name;
        strip_opening_material(commitment);
    }
}

FieldElement choose_non_domain_point(const gatzk::algebra::RootOfUnityDomain& domain) {
    for (std::uint64_t raw = 17; raw < 10000; ++raw) {
        const FieldElement candidate(raw);
        bool overlap = false;
        for (std::size_t i = 0; i < domain.size; ++i) {
            if (domain.point_at(i) == candidate) {
                overlap = true;
                break;
            }
        }
        if (!overlap) {
            return candidate;
        }
    }
    throw std::runtime_error("failed to find a non-domain opening point");
}

const SecureFixture& cora_secure_fixture() {
    static const SecureFixture fixture = [] {
        auto config = gatzk::util::load_config((repo_root() / "configs/cora_secure.cfg").string());
        auto context = gatzk::protocol::build_context(config, nullptr);
        auto trace = gatzk::protocol::build_trace(context, nullptr);
        auto proof = gatzk::protocol::prove(context, trace, nullptr);
        return SecureFixture{std::move(context), std::move(trace), std::move(proof)};
    }();
    return fixture;
}

void test_kzg_opening_selftest() {
    const auto keys = gatzk::crypto::KZG::load_keys(
        (repo_root() / "artifacts/srs/zkgat_secure_main_srs_v1").string(),
        "trusted_srs",
        "zkgat_secure_main_srs_v1");
    auto poly = gatzk::algebra::Polynomial::from_coefficients(
        "P_test",
        {FieldElement(3), FieldElement(5), FieldElement(7)});
    const auto commitment = gatzk::crypto::KZG::commit("P_test", poly, keys);
    const std::vector<FieldElement> points{FieldElement(9)};
    const std::vector<std::vector<FieldElement>> values{{poly.evaluate(points.front())}};
    const auto proof = gatzk::crypto::KZG::open_batch({commitment}, points, values, FieldElement(13), keys);
    require(
        gatzk::crypto::KZG::verify_batch({commitment}, points, values, FieldElement(13), proof, keys.vk),
        "KZG single-point opening must verify");
    auto bad_values = values;
    bad_values[0][0] += FieldElement::one();
    require(
        !gatzk::crypto::KZG::verify_batch({commitment}, points, bad_values, FieldElement(13), proof, keys.vk),
        "KZG claimed-value tamper must reject");

    auto verifier_commitment = commitment;
    strip_opening_material(verifier_commitment);
    require(
        gatzk::crypto::KZG::verify_batch({verifier_commitment}, points, values, FieldElement(13), proof, keys.vk),
        "KZG verifier-key-only opening must verify without prover polynomial material");
}

void test_compact_eval_kzg_regression() {
    const auto keys = gatzk::crypto::KZG::load_keys(
        (repo_root() / "artifacts/srs/zkgat_secure_main_srs_v1").string(),
        "trusted_srs",
        "zkgat_secure_main_srs_v1");

    const auto exact_n_it = keys.manifest.domain_sizes.find("N_4096");
    if (keys.manifest.domain_sizes.at("N") > 4096) {
        require(exact_n_it != keys.manifest.domain_sizes.end(), "expanded SRS must include exact N_4096 material");
    }
    const auto n_size = exact_n_it != keys.manifest.domain_sizes.end()
        ? exact_n_it->second
        : keys.manifest.domain_sizes.at("N");
    auto domain = gatzk::algebra::RootOfUnityDomain::create("N", n_size);
    auto polynomial = gatzk::algebra::Polynomial::from_evaluations(
        "P_compact_eval",
        {FieldElement(3), FieldElement(5), FieldElement(7), FieldElement(11), FieldElement(13)},
        domain);
    const auto point = choose_non_domain_point(*domain);
    const auto value = polynomial.evaluate(point);
    const auto commitment = gatzk::crypto::KZG::commit("P_compact_eval", polynomial, keys);
    const std::vector<std::pair<gatzk::crypto::Commitment, FieldElement>> openings{
        {commitment, value},
    };
    const std::vector<FieldElement> points{point};
    const auto proof = gatzk::crypto::KZG::open_external_fold(openings, points, FieldElement(29), keys);

    require(
        gatzk::crypto::KZG::verify_external_fold(openings, points, FieldElement(29), proof, keys.vk),
        "compact evaluation external opening must verify");

    auto tampered = openings;
    tampered[0].second += FieldElement::one();
    require(
        !gatzk::crypto::KZG::verify_external_fold(tampered, points, FieldElement(29), proof, keys.vk),
        "compact evaluation external opening must reject a tampered value");
}

void test_packed_backend_padded_tail_regression() {
    auto domain = gatzk::algebra::RootOfUnityDomain::create("packed_pad_regression", 4096);

    {
        std::vector<FieldElement> full_values(domain->size, FieldElement(1234567));
        auto poison = gatzk::algebra::Polynomial::from_evaluations("P_poison", std::move(full_values), domain);
        gatzk::algebra::PackedEvaluationBackend poison_backend(
            domain,
            {{"P_poison", &poison}});
        (void)poison_backend;
    }

    auto polynomial = gatzk::algebra::Polynomial::from_evaluations(
        "P_pad",
        {FieldElement(3), FieldElement(5), FieldElement(7), FieldElement(11), FieldElement(13)},
        domain);
    gatzk::algebra::PackedEvaluationBackend backend(
        domain,
        {{"P_pad", &polynomial}});

    const auto direct_padding = backend.values_at_direct_index({"P_pad"}, 128);
    require(direct_padding.size() == 1 && direct_padding[0] == FieldElement::zero(), "packed backend direct padding must be zero");

    const auto point = choose_non_domain_point(*domain);
    const auto weights = gatzk::algebra::build_domain_evaluation_weights(domain, point);
    require(!weights.direct_index.has_value(), "packed backend regression point must not be in domain");
    const auto packed_values = backend.evaluate_with_native_weights({"P_pad"}, weights.native_weights);
    require(packed_values.size() == 1, "packed backend must return one value");
    require(packed_values[0] == polynomial.evaluate(point), "packed backend weighted eval must ignore padded tail");
}

void require_msm_matches_single_thread(
    const std::string& label,
    const std::vector<gatzk::crypto::G1Point>& points,
    std::size_t count,
    bool sparse) {
    require(points.size() >= count, label + " SRS input is too small");
    std::vector<FieldElement> scalars(count, FieldElement::zero());
    for (std::size_t i = 0; i < count; ++i) {
        if (sparse && i % 5 != 0) {
            continue;
        }
        scalars[i] = FieldElement(17 + (i * 131) % 65521);
    }

    const auto actual = gatzk::crypto::g1_multi_scalar_mul(points.data(), scalars.data(), count);

    std::vector<mcl::Fr> native_scalars;
    native_scalars.reserve(scalars.size());
    for (const auto& scalar : scalars) {
        native_scalars.push_back(scalar.native());
    }
    auto* native_points = const_cast<mcl::G1*>(
        reinterpret_cast<const mcl::G1*>(points.data()));
    gatzk::crypto::G1Point expected;
    expected.value.clear();
    mcl::G1::mulVec(expected.value, native_points, native_scalars.data(), native_scalars.size());
    require(actual == expected, label + " MSM must match single-thread mulVec");
}

void test_msm_chunked_equivalence_regression() {
    const auto keys = gatzk::crypto::KZG::load_keys(
        (repo_root() / "artifacts/srs/zkgat_secure_main_srs_v1").string(),
        "trusted_srs",
        "zkgat_secure_main_srs_v1");
    require(keys.pk.powers_g1 != nullptr, "MSM regression requires coefficient SRS");
    require_msm_matches_single_thread("coef_21664_dense", *keys.pk.powers_g1, 21664, false);

    const auto edge_it = keys.pk.lagrange_srs_by_domain.find("edge");
    require(edge_it != keys.pk.lagrange_srs_by_domain.end() && edge_it->second != nullptr, "MSM regression requires edge SRS");
    require_msm_matches_single_thread("edge_13264_sparse", *edge_it->second, 13264, true);
    require_msm_matches_single_thread("edge_131072_dense", *edge_it->second, 131072, false);
}

void test_srs_mismatch_gate() {
    const auto srs_path = (repo_root() / "artifacts/srs/zkgat_secure_main_srs_v1").string();
    require_throws(
        [&] { (void)gatzk::crypto::KZG::load_keys(srs_path, "mpc_srs", "zkgat_secure_main_srs_v1"); },
        "SRS type mismatch must hard fail");
    require_throws(
        [&] { (void)gatzk::crypto::KZG::load_keys(srs_path, "trusted_srs", "wrong_srs_id"); },
        "SRS id mismatch must hard fail");
}

void test_metadata_and_schema_gates() {
    const auto config = gatzk::util::load_config((repo_root() / "configs/cora_secure.cfg").string());
    require(config.proof_task == "node_classification_full_gat", "config proof_task mismatch");
    require(config.privacy_policy == "model_and_trace_hidden", "config privacy policy mismatch");
    require(config.input_privacy == "public_graph_public_features", "config input privacy mismatch");
    require(config.output_mode == "public_logits", "config output mode mismatch");
    require(config.opening_mode == "domain_grouped_batch_opening", "config opening mode mismatch");
    require(config.binding_mode == "zk_subproof_for_crpc", "config binding mode mismatch");
    require(config.srs_type == "trusted_srs", "config SRS type mismatch");
    require(config.srs_id == "zkgat_secure_main_srs_v1", "config SRS id mismatch");
    require(
        gatzk::protocol::schema_accepts_evaluation(
            gatzk::protocol::PolynomialVisibility::PrivateWitness,
            gatzk::protocol::EvaluationKind::MaskedPrivateValue),
        "private witness must accept masked values");
    require(
        !gatzk::protocol::schema_accepts_evaluation(
            gatzk::protocol::PolynomialVisibility::PrivateWitness,
            gatzk::protocol::EvaluationKind::PublicValue),
        "private witness must reject public values");

    gatzk::protocol::ProtocolContext empty_context;
    require(
        gatzk::protocol::polynomial_visibility(empty_context, "P_Y") ==
            gatzk::protocol::PolynomialVisibility::PublicOutput,
        "only the final logits polynomial is public output");
    require(
        gatzk::protocol::polynomial_visibility(empty_context, "P_Y_lin") ==
            gatzk::protocol::PolynomialVisibility::PrivateWitness,
        "pre-bias output must remain private");
    require(
        gatzk::protocol::polynomial_visibility(empty_context, "P_Y_o0") ==
            gatzk::protocol::PolynomialVisibility::PrivateWitness,
        "per-head output must remain private");
}

gatzk::data::LocalGraph valid_two_graph_fixture() {
    gatzk::data::LocalGraph local;
    local.num_nodes = 4;
    local.num_features = 1;
    local.num_classes = 2;
    local.graph_count = 2;
    local.absolute_ids = {10, 11, 12, 13};
    local.node_ptr = {0, 2, 4};
    local.edge_ptr = {0, 3, 6};
    local.edges = {
        {.src = 0, .dst = 0, .graph_id = 0, .stable_index = 0},
        {.src = 1, .dst = 0, .graph_id = 0, .stable_index = 1},
        {.src = 1, .dst = 1, .graph_id = 0, .stable_index = 2},
        {.src = 2, .dst = 2, .graph_id = 1, .stable_index = 0},
        {.src = 3, .dst = 2, .graph_id = 1, .stable_index = 1},
        {.src = 3, .dst = 3, .graph_id = 1, .stable_index = 2},
    };
    local.public_input.N_total = 20;
    local.public_input.N = 4;
    local.public_input.E = local.edges.size();
    local.public_input.G_batch = 2;
    local.public_input.I = local.absolute_ids;
    local.public_input.node_ptr = local.node_ptr;
    local.public_input.edge_ptr = local.edge_ptr;
    for (const auto& edge : local.edges) {
        local.public_input.src.push_back(edge.src);
        local.public_input.dst.push_back(edge.dst);
    }
    return local;
}

void test_multi_graph_public_boundary_validation() {
    const auto valid = valid_two_graph_fixture();
    gatzk::data::validate_normalized_graph_input(valid);

    auto cross_graph = valid;
    cross_graph.edges[3].src = 0;
    cross_graph.public_input.src[3] = 0;
    require_throws(
        [&] { gatzk::data::validate_normalized_graph_input(cross_graph); },
        "cross-graph edge must reject");

    auto bad_pointer = valid;
    bad_pointer.edge_ptr = {0, 4, 6};
    bad_pointer.public_input.edge_ptr = bad_pointer.edge_ptr;
    require_throws(
        [&] { gatzk::data::validate_normalized_graph_input(bad_pointer); },
        "edge_ptr/graph_id mismatch must reject");

    auto unsorted = valid;
    std::swap(unsorted.edges[1], unsorted.edges[2]);
    std::swap(unsorted.public_input.src[1], unsorted.public_input.src[2]);
    std::swap(unsorted.public_input.dst[1], unsorted.public_input.dst[2]);
    require_throws(
        [&] { gatzk::data::validate_normalized_graph_input(unsorted); },
        "non-monotone destination order must reject");

    auto duplicate_node = valid;
    duplicate_node.absolute_ids[3] = duplicate_node.absolute_ids[2];
    duplicate_node.public_input.I = duplicate_node.absolute_ids;
    require_throws(
        [&] { gatzk::data::validate_normalized_graph_input(duplicate_node); },
        "duplicate absolute node id must reject");
}

void test_real_ppi_representative_batch_boundary() {
    auto config = gatzk::util::load_config((repo_root() / "configs/ppi_secure.cfg").string());
    // Development validation uses two complete source graphs. No node or edge
    // inside either graph is cropped, and this is not a formal full-PPI run.
    config.batch_graphs = 2;
    const auto dataset = gatzk::data::load_dataset(config);
    require(dataset.name == "ppi", "PPI boundary test must load the real PPI cache");
    require(dataset.graph_count >= config.batch_graphs, "PPI cache must contain the representative graphs");

    const auto local = gatzk::data::normalize_graph_input(dataset, config);
    gatzk::data::validate_normalized_graph_input(local);
    require(local.graph_count == config.batch_graphs, "PPI representative batch graph count mismatch");
    require(local.node_ptr.size() == config.batch_graphs + 1, "PPI node_ptr shape mismatch");
    require(local.edge_ptr.size() == config.batch_graphs + 1, "PPI edge_ptr shape mismatch");

    for (std::size_t graph = 0; graph < config.batch_graphs; ++graph) {
        const auto source_node_begin = dataset.node_ptr.at(graph);
        const auto source_node_end = dataset.node_ptr.at(graph + 1);
        require(
            local.node_ptr[graph + 1] - local.node_ptr[graph] == source_node_end - source_node_begin,
            "PPI normalization must retain every node of each selected source graph");
        for (std::size_t node = local.node_ptr[graph]; node < local.node_ptr[graph + 1]; ++node) {
            require(
                local.absolute_ids[node] == source_node_begin + node - local.node_ptr[graph],
                "PPI normalization must preserve the complete source-graph node range");
        }
        for (std::size_t edge = local.edge_ptr[graph]; edge < local.edge_ptr[graph + 1]; ++edge) {
            const auto& item = local.edges[edge];
            require(item.graph_id == graph, "PPI edge block graph id mismatch");
            require(
                item.src >= local.node_ptr[graph] && item.src < local.node_ptr[graph + 1] &&
                    item.dst >= local.node_ptr[graph] && item.dst < local.node_ptr[graph + 1],
                "PPI representative batch contains a cross-graph edge");
        }
    }
}

void test_full_metadata_and_schema_gates() {
    const auto& fixture = cora_secure_fixture();
    const auto metadata = gatzk::protocol::canonical_public_metadata(fixture.context);
    require(metadata.protocol_name == "zkgat", "metadata protocol_name mismatch");
    require(metadata.proof_task == "node_classification_full_gat", "metadata proof_task mismatch");
    require(metadata.privacy_policy == "model_and_trace_hidden", "metadata privacy policy mismatch");
    require(metadata.input_privacy == "public_graph_public_features", "metadata input privacy mismatch");
    require(metadata.output_mode == "public_logits", "metadata output mode mismatch");
    require(metadata.opening_mode == "domain_grouped_batch_opening", "metadata opening mode mismatch");
    require(metadata.binding_mode == "zk_subproof_for_crpc", "metadata binding mode mismatch");

    const auto schema = gatzk::protocol::build_proof_schema(fixture.context);
    require(!schema.entries.empty(), "ProofSchema must register committed objects");
    require(schema.private_plain_value_count == 0, "ProofSchema private plain value count must be zero");
}

void test_reference_consistency_and_verify() {
    const auto& fixture = cora_secure_fixture();
    require(gatzk::protocol::verify(fixture.context, fixture.proof), "secure Cora proof must verify");
    require(fixture.proof.public_metadata.output_mode == "public_logits", "secure proof must expose public logits");
    require(fixture.proof.public_metadata.srs_type == "trusted_srs", "secure proof must bind trusted SRS type");
}

void test_verifier_only_boundary() {
    const auto& fixture = cora_secure_fixture();
    auto context = fixture.context;
    auto proof = fixture.proof;
    strip_context_to_verifier_only(context);
    strip_proof_to_verifier_only(proof);
    require(context.kzg.pk.powers_g1 == nullptr, "verifier-only context must not retain coefficient proving SRS");
    require(context.kzg.pk.lagrange_srs_by_domain.empty(), "verifier-only context must not retain Lagrange proving SRS");
    require(gatzk::protocol::verify(context, proof), "verifier-only proof path must verify without prover material");
    require(
        gatzk::protocol::private_plain_value_count(context, proof) == 0,
        "verifier payload audit must observe zero private plain values");
}

void require_rejects(
    const SecureFixture& fixture,
    gatzk::protocol::Proof proof,
    const std::string& label) {
    require(!gatzk::protocol::verify(fixture.context, proof), label + " must reject");
}

void test_tamper_rejection_suite() {
    const auto& fixture = cora_secure_fixture();

    auto metadata_tamper = fixture.proof;
    metadata_tamper.public_metadata.privacy_policy = "wrong_policy";
    require_rejects(fixture, metadata_tamper, "privacy_policy tamper");

    auto srs_id_tamper = fixture.proof;
    srs_id_tamper.public_metadata.srs_id = "wrong_srs";
    require_rejects(fixture, srs_id_tamper, "srs_id tamper");

    auto srs_type_tamper = fixture.proof;
    srs_type_tamper.public_metadata.srs_type = "wrong_type";
    require_rejects(fixture, srs_type_tamper, "srs_type tamper");

    auto input_privacy_tamper = fixture.proof;
    input_privacy_tamper.public_metadata.input_privacy = "hidden_graph";
    require_rejects(fixture, input_privacy_tamper, "input_privacy tamper");

    auto output_mode_tamper = fixture.proof;
    output_mode_tamper.public_metadata.output_mode = "label_only";
    require_rejects(fixture, output_mode_tamper, "output_mode tamper");

    auto opening_mode_tamper = fixture.proof;
    opening_mode_tamper.public_metadata.opening_mode = "single_opening";
    require_rejects(fixture, opening_mode_tamper, "opening_mode tamper");

    auto binding_mode_tamper = fixture.proof;
    binding_mode_tamper.public_metadata.binding_mode = "wrong_binding";
    require_rejects(fixture, binding_mode_tamper, "binding_mode tamper");

    auto proof_task_tamper = fixture.proof;
    proof_task_tamper.public_metadata.proof_task = "wrong_task";
    require_rejects(fixture, proof_task_tamper, "proof_task tamper");

    auto block_order_tamper = fixture.proof;
    block_order_tamper.block_order.push_back("unknown_field");
    require_rejects(fixture, block_order_tamper, "added unknown field tamper");

    auto commitment_order_tamper = fixture.proof;
    if (commitment_order_tamper.dynamic_commitments.size() >= 2) {
        std::swap(commitment_order_tamper.dynamic_commitments[0], commitment_order_tamper.dynamic_commitments[1]);
        require_rejects(fixture, commitment_order_tamper, "commitment order tamper");
    }

    auto dynamic_point_tamper = fixture.proof;
    dynamic_point_tamper.dynamic_commitments.front().second.point = gatzk::crypto::g1_zero();
    require_rejects(fixture, dynamic_point_tamper, "dynamic commitment point tamper");

    auto quotient_point_tamper = fixture.proof;
    quotient_point_tamper.quotient_commitments.front().second.point = gatzk::crypto::g1_zero();
    require_rejects(fixture, quotient_point_tamper, "quotient commitment tamper");

    auto domain_value_tamper = fixture.proof;
    domain_value_tamper.domain_openings.front().second.values.front().second.front() += FieldElement::one();
    require_rejects(fixture, domain_value_tamper, "domain opening value tamper");

    auto private_payload_kind_tamper = fixture.proof;
    bool changed_private_kind = false;
    for (auto& [bundle_name, bundle] : private_payload_kind_tamper.domain_openings) {
        (void)bundle_name;
        for (auto& kind : bundle.value_kinds) {
            if (kind == gatzk::protocol::PayloadValueKind::MaskedPrivateValue) {
                kind = gatzk::protocol::PayloadValueKind::PublicValue;
                changed_private_kind = true;
                break;
            }
        }
        if (changed_private_kind) {
            break;
        }
    }
    require(changed_private_kind, "fixture must contain a private typed payload value");
    require_rejects(fixture, private_payload_kind_tamper, "private payload kind tamper");

    auto external_value_tamper = fixture.proof;
    external_value_tamper.external_evaluations.front().second += FieldElement::one();
    require_rejects(fixture, external_value_tamper, "external masked value tamper");

    auto public_logits_tamper = fixture.proof;
    public_logits_tamper.public_logits.front() += FieldElement::one();
    require_rejects(fixture, public_logits_tamper, "public logits tamper");

    auto public_output_point_tamper = fixture.proof;
    public_output_point_tamper.public_output_challenge += FieldElement::one();
    require_rejects(fixture, public_output_point_tamper, "public output opening point tamper");

    auto challenge_tamper = fixture.proof;
    challenge_tamper.challenges.begin()->second += FieldElement::one();
    require_rejects(fixture, challenge_tamper, "challenge tamper");
}

}  // namespace

int main() {
    try {
        std::cerr << "running test_kzg_opening_selftest\n";
        test_kzg_opening_selftest();
        std::cerr << "running test_compact_eval_kzg_regression\n";
        test_compact_eval_kzg_regression();
        std::cerr << "running test_packed_backend_padded_tail_regression\n";
        test_packed_backend_padded_tail_regression();
        std::cerr << "running test_msm_chunked_equivalence_regression\n";
        test_msm_chunked_equivalence_regression();
        std::cerr << "running test_srs_mismatch_gate\n";
        test_srs_mismatch_gate();
        std::cerr << "running test_metadata_and_schema_gates\n";
        test_metadata_and_schema_gates();
        std::cerr << "running test_multi_graph_public_boundary_validation\n";
        test_multi_graph_public_boundary_validation();
        if (std::getenv("GATZK_RUN_PPI_BOUNDARY_TESTS") != nullptr) {
            std::cerr << "running test_real_ppi_representative_batch_boundary\n";
            test_real_ppi_representative_batch_boundary();
        } else {
            std::cerr << "skipping real PPI representative-batch boundary regression; set GATZK_RUN_PPI_BOUNDARY_TESTS=1 to run it\n";
        }
        if (std::getenv("GATZK_RUN_FULL_SECURITY_TESTS") != nullptr) {
            std::cerr << "running test_full_metadata_and_schema_gates\n";
            test_full_metadata_and_schema_gates();
            std::cerr << "running test_reference_consistency_and_verify\n";
            test_reference_consistency_and_verify();
            std::cerr << "running test_verifier_only_boundary\n";
            test_verifier_only_boundary();
            std::cerr << "running test_tamper_rejection_suite\n";
            test_tamper_rejection_suite();
        } else {
            std::cerr << "skipping full Cora security regression; set GATZK_RUN_FULL_SECURITY_TESTS=1 to run it\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
    std::cout << "secure regression tests passed\n";
    return 0;
}
