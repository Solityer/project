#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/crypto/kzg.hpp"
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

    const auto n_size = keys.manifest.domain_sizes.at("N");
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
        std::cerr << "running test_srs_mismatch_gate\n";
        test_srs_mismatch_gate();
        std::cerr << "running test_metadata_and_schema_gates\n";
        test_metadata_and_schema_gates();
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
