#include <iostream>
#include <array>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <openssl/sha.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/crypto/curve.hpp"
#include "gatzk/protocol/proof_codec.hpp"
#include "gatzk/protocol/verifier.hpp"
#include "gatzk/protocol/verifier_context.hpp"

namespace
{
    void expect(bool condition, const std::string &message)
    {
        if (!condition)
            throw std::runtime_error(message);
    }

    void expect_rejected(
        const std::vector<std::uint8_t> &bytes,
        const std::string &label)
    {
        bool rejected = false;
        try
        {
            (void)gatzk::protocol::parse_verifier_registration_artifact(bytes);
        }
        catch (const std::exception &)
        {
            rejected = true;
        }
        expect(rejected, label + " was accepted");
    }

    std::uint32_t read_u32(
        const std::vector<std::uint8_t> &bytes,
        std::size_t offset)
    {
        expect(offset + 4 <= bytes.size(), "test u32 read is out of bounds");
        std::uint32_t value = 0;
        for (std::size_t index = 0; index < 4; ++index)
            value |= static_cast<std::uint32_t>(bytes[offset + index]) << (8U * index);
        return value;
    }

    std::uint64_t read_u64(
        const std::vector<std::uint8_t> &bytes,
        std::size_t offset)
    {
        expect(offset + 8 <= bytes.size(), "test u64 read is out of bounds");
        std::uint64_t value = 0;
        for (std::size_t index = 0; index < 8; ++index)
            value |= static_cast<std::uint64_t>(bytes[offset + index]) << (8U * index);
        return value;
    }

    void write_u32(
        std::vector<std::uint8_t> *bytes,
        std::size_t offset,
        std::uint32_t value)
    {
        expect(offset + 4 <= bytes->size(), "test u32 write is out of bounds");
        for (std::size_t index = 0; index < 4; ++index)
            (*bytes)[offset + index] = static_cast<std::uint8_t>(value >> (8U * index));
    }

    void write_u64(
        std::vector<std::uint8_t> *bytes,
        std::size_t offset,
        std::uint64_t value)
    {
        expect(offset + 8 <= bytes->size(), "test u64 write is out of bounds");
        for (std::size_t index = 0; index < 8; ++index)
            (*bytes)[offset + index] = static_cast<std::uint8_t>(value >> (8U * index));
    }

    void refresh_digest(std::vector<std::uint8_t> *bytes)
    {
        constexpr std::size_t digest_size = SHA256_DIGEST_LENGTH;
        expect(bytes->size() >= digest_size, "test artifact is missing its digest");
        const auto body_size = bytes->size() - digest_size;
        std::array<unsigned char, digest_size> digest{};
        SHA256(bytes->data(), body_size, digest.data());
        std::copy(digest.begin(), digest.end(), bytes->begin() + body_size);
    }

    gatzk::protocol::VerifierRegistrationArtifact synthetic_registration_artifact()
    {
        gatzk::protocol::VerifierRegistrationArtifact artifact;
        const auto g1 = gatzk::crypto::g1_zero();
        artifact.kzg_vk.g1_generator = g1;
        artifact.kzg_vk.g2_one = gatzk::crypto::g2_zero();
        artifact.kzg_vk.g2_tau = gatzk::crypto::g2_zero();
        artifact.registry.srs_g1_generator = g1;
        artifact.registry.registry_id = "runtime-registry";
        artifact.registry.model_arch_id = "runtime-arch";
        artifact.registry.model_param_id = "runtime-params";
        artifact.registry.input_preprocess_id = "row_normalize_v1";
        artifact.registry.quant_cfg_id = "runtime-quant";
        artifact.registry.static_table_id = "runtime-tables";
        artifact.registry.srs_id = "runtime-srs";
        artifact.registry.protocol_schema_id = "runtime-schema";
        artifact.registry.srs_protocol_compiler_artifact_hash =
            std::string(64, 'a');
        artifact.config.dataset = "cora";
        artifact.config.input_preprocess_id = "row_normalize_v1";
        artifact.dataset.name = "cora";
        artifact.dataset.input_preprocess_id = "row_normalize_v1";
        artifact.local.name = "cora_normalized";
        artifact.local.input_preprocess_id = "row_normalize_v1";
        artifact.model.input_preprocess_id = "row_normalize_v1";

        artifact.domains.fh =
            gatzk::algebra::RootOfUnityDomain::create("FH", 1);
        artifact.domains.edge =
            gatzk::algebra::RootOfUnityDomain::create("edge", 1);
        artifact.domains.in =
            gatzk::algebra::RootOfUnityDomain::create("in", 1);
        artifact.domains.d =
            gatzk::algebra::RootOfUnityDomain::create("d", 1);
        artifact.domains.cat =
            gatzk::algebra::RootOfUnityDomain::create("cat", 1);
        artifact.domains.c =
            gatzk::algebra::RootOfUnityDomain::create("C", 1);
        artifact.domains.n =
            gatzk::algebra::RootOfUnityDomain::create("N", 1);

        artifact.public_polynomials.emplace(
            "runtime_public",
            gatzk::algebra::Polynomial::from_coefficients(
                "runtime_public",
                {gatzk::algebra::FieldElement::one()}));
        artifact.public_commitments.emplace(
            "runtime_public",
            gatzk::protocol::VerifierPointCommitment{"runtime_public", g1});
        return artifact;
    }
}

int main()
{
    try
    {
        const auto point = gatzk::crypto::g1_zero();

        // A public point copied from a proving context must not retain its
        // opening polynomial or alias the source map.
        gatzk::protocol::ProtocolContext proving_context;
        gatzk::crypto::Commitment public_commitment;
        public_commitment.name = "public";
        public_commitment.point = point;
        public_commitment.opening_polynomial = std::make_shared<const gatzk::algebra::Polynomial>(
            gatzk::algebra::Polynomial::from_coefficients(
                "private-opening",
                {gatzk::algebra::FieldElement::one()}));
        proving_context.public_commitments.emplace(
            public_commitment.name,
            public_commitment);
        auto verifier_inputs =
            gatzk::protocol::make_verifier_public_inputs(proving_context);
        proving_context.public_commitments.clear();
        expect(verifier_inputs.public_commitments.size() == 1,
               "public point copy was not independent");
        expect(verifier_inputs.public_commitments.at("public").point == point,
               "public point changed after proving-context mutation");

        // Canonical registry serialization binds ordered metadata and points.
        gatzk::protocol::AuthenticatedVerifierRegistry registry;
        registry.registry_id = "registry";
        registry.model_arch_id = "arch";
        registry.model_param_id = "params";
        registry.input_preprocess_id = "row_normalize_v1";
        registry.quant_cfg_id = "quant";
        registry.static_table_id = "tables";
        registry.srs_id = "srs";
        registry.protocol_schema_id = "schema";
        registry.srs_g1_generator = point;
        registry.srs_protocol_compiler_artifact_hash = std::string(64, 'a');
        const auto digest = registry.canonical_digest();
        registry.registry_id = "registry-tampered";
        expect(digest != registry.canonical_digest(),
               "registry canonical digest did not bind metadata");
        registry.registry_id = "registry";
        registry.srs_protocol_compiler_artifact_hash = std::string(64, 'b');
        expect(digest != registry.canonical_digest(),
               "registry canonical digest did not bind SRS compiler artifact hash");
        registry.srs_protocol_compiler_artifact_hash = std::string(64, 'a');
        registry.input_preprocess_id = "ogbn_arxiv_raw_float32_v1";
        expect(digest != registry.canonical_digest(),
               "registry canonical digest did not bind input preprocessing identity");
        registry.input_preprocess_id = "row_normalize_v1";
        registry.model_entries.push_back({
            .label = "model_l0_h0_w0",
            .domain = "in",
            .degree_bound = 7,
            .scale = 1,
            .range_min = -1,
            .range_max = 0,
            .point = point});
        expect(digest != registry.canonical_digest(),
               "registry canonical digest did not bind entry order/content");

        // A proof carrying an opening polynomial is prover-only material and
        // must fail at the wire codec boundary.
        gatzk::protocol::Proof proof;
        gatzk::crypto::Commitment leaking_commitment;
        leaking_commitment.name = "leaking";
        leaking_commitment.point = point;
        leaking_commitment.opening_polynomial = public_commitment.opening_polynomial;
        proof.dynamic_commitments.emplace_back(
            leaking_commitment.name,
            leaking_commitment);
        bool opening_rejected = false;
        try
        {
            (void)gatzk::protocol::serialize_proof(proof);
        }
        catch (const std::exception &)
        {
            opening_rejected = true;
        }
        expect(opening_rejected,
               "proof codec accepted non-null prover opening material");

        // Exercise the independent registration artifact codec.  A canonical
        // round trip is required before any digest/tamper checks are useful.
        auto registration = synthetic_registration_artifact();
        const auto registration_bytes =
            gatzk::protocol::serialize_verifier_registration_artifact(registration);
        const auto parsed_registration =
            gatzk::protocol::parse_verifier_registration_artifact(registration_bytes);
        expect(
            parsed_registration.canonical_digest() == registration.canonical_digest(),
            "verifier registration codec canonical roundtrip changed the artifact");
        expect(
            parsed_registration.registry.srs_protocol_compiler_artifact_hash
                == std::string(64, 'a'),
            "verifier registration codec lost SRS compiler artifact hash");
        expect(
            parsed_registration.registry.input_preprocess_id == "row_normalize_v1",
            "verifier registration codec lost input preprocessing identity");
        expect(
            parsed_registration.public_commitments.size() == 1
                && parsed_registration.public_commitments.at("runtime_public").point == point,
            "verifier registration codec lost point-only public ownership");

        auto single_byte_tamper = registration_bytes;
        single_byte_tamper[16] ^= 0x01U;
        expect_rejected(single_byte_tamper, "single-byte artifact tamper");

        auto trailing_tamper = registration_bytes;
        trailing_tamper.erase(trailing_tamper.end() - SHA256_DIGEST_LENGTH, trailing_tamper.end());
        trailing_tamper.push_back(0);
        trailing_tamper.resize(trailing_tamper.size() + SHA256_DIGEST_LENGTH);
        refresh_digest(&trailing_tamper);
        expect_rejected(trailing_tamper, "artifact trailing byte");

        auto noncanonical_tamper = registration_bytes;
        write_u32(&noncanonical_tamper, 16, 2);
        refresh_digest(&noncanonical_tamper);
        expect_rejected(noncanonical_tamper, "non-canonical section order");

        auto compiler_tamper = registration_bytes;
        std::size_t cursor = 16;
        std::size_t compiler_payload = 0;
        for (std::uint32_t section = 1; section <= 12; ++section)
        {
            expect(read_u32(compiler_tamper, cursor) == section,
                   "synthetic artifact section scan lost canonical order");
            const auto length = read_u64(compiler_tamper, cursor + 4);
            if (section == 11)
                compiler_payload = cursor + 12;
            cursor += 12 + static_cast<std::size_t>(length);
        }
        expect(compiler_payload != 0, "compiler section was not found in artifact");
        write_u64(
            &compiler_tamper,
            compiler_payload,
            std::numeric_limits<std::uint64_t>::max());
        refresh_digest(&compiler_tamper);
        expect_rejected(compiler_tamper, "compiler section tamper");

        const auto registry_path =
            std::filesystem::temp_directory_path() / "zkgat_verifier_runtime.bin";
        {
            std::ofstream output(registry_path, std::ios::binary | std::ios::trunc);
            expect(output.good(), "unable to create trusted-digest runtime fixture");
            output.write(
                reinterpret_cast<const char *>(registration_bytes.data()),
                static_cast<std::streamsize>(registration_bytes.size()));
            expect(output.good(), "unable to write trusted-digest runtime fixture");
        }
        bool trusted_digest_rejected = false;
        try
        {
            (void)gatzk::protocol::load_verifier_context_from_registry(
                registry_path.string(), std::string(64, '0'));
        }
        catch (const std::exception &)
        {
            trusted_digest_rejected = true;
        }
        std::filesystem::remove(registry_path);
        expect(trusted_digest_rejected,
               "registry loader accepted an artifact with an untrusted digest");

        auto owned_source = registration;
        auto owned_bytes =
            gatzk::protocol::serialize_verifier_registration_artifact(owned_source);
        const auto owned_copy =
            gatzk::protocol::parse_verifier_registration_artifact(owned_bytes);
        owned_source.registry.registry_id = "mutated-after-serialization";
        owned_source.public_commitments.clear();
        expect(
            owned_copy.registry.registry_id == "runtime-registry"
                && owned_copy.public_commitments.size() == 1,
            "parsed verifier registration aliases mutable source ownership");

        expect(!gatzk::protocol::validate_verifier_context(verifier_inputs),
               "incomplete verifier context was accepted");
        expect(!gatzk::protocol::verify(verifier_inputs, proof, nullptr),
               "invalid verifier context reached cryptographic verifier");
        std::cout << "verifier_privacy_runtime: PASS\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "verifier_privacy_runtime: FAIL: " << error.what() << '\n';
        return 1;
    }
}
