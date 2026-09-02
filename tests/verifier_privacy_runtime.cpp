#include <iostream>
#include <array>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <openssl/sha.h>
#include <sstream>
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

    std::string canonical_static_table_digest(
        const gatzk::protocol::StaticTables &tables)
    {
        std::ostringstream serialization;
        serialization << "static_tables|range=" << tables.range.size();
        for (const auto &value : tables.range)
            serialization << '|' << value.to_string();
        serialization << "|lrelu=" << tables.lrelu.size();
        for (const auto &[x, y] : tables.lrelu)
            serialization << '|' << x.to_string() << ',' << y.to_string();
        serialization << "|elu=" << tables.elu.size();
        for (const auto &[x, y] : tables.elu)
            serialization << '|' << x.to_string() << ',' << y.to_string();
        serialization << "|exp=" << tables.exp.size();
        for (const auto &[x, y] : tables.exp)
            serialization << '|' << x.to_string() << ',' << y.to_string();

        const auto bytes = serialization.str();
        std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
        SHA256(
            reinterpret_cast<const unsigned char *>(bytes.data()),
            bytes.size(),
            digest.data());
        std::ostringstream hex;
        hex << std::hex << std::setfill('0');
        for (const auto byte : digest)
            hex << std::setw(2) << static_cast<unsigned int>(byte);
        return hex.str();
    }

    gatzk::protocol::StaticTables expected_small_canonical_static_tables()
    {
        using gatzk::algebra::FieldElement;
        gatzk::protocol::StaticTables expected;
        expected.range = {FieldElement::zero(), FieldElement::one()};
        expected.lrelu = {
            {FieldElement::from_signed(-2), FieldElement::zero()},
            {FieldElement::from_signed(-1), FieldElement::zero()},
            {FieldElement::zero(), FieldElement::zero()},
            {FieldElement::one(), FieldElement::one()},
            {FieldElement(2), FieldElement(2)}};
        expected.elu = {
            {FieldElement::from_signed(-2), FieldElement::from_signed(-1)},
            {FieldElement::from_signed(-1), FieldElement::from_signed(-1)},
            {FieldElement::zero(), FieldElement::zero()},
            {FieldElement::one(), FieldElement::one()},
            {FieldElement(2), FieldElement(2)}};
        expected.exp = {
            {FieldElement::zero(), FieldElement::one()},
            {FieldElement::one(), FieldElement::one()},
            {FieldElement(2), FieldElement::one()},
            {FieldElement(3), FieldElement::one()},
            {FieldElement(4), FieldElement::one()}};
        return expected;
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
        artifact.registry.input_preprocess_id = "row_normalize";
        artifact.registry.quant_cfg_id = "runtime-quant";
        artifact.registry.static_table_id = "runtime-tables";
        artifact.registry.srs_id = "runtime-srs";
        artifact.registry.protocol_schema_id = "runtime-schema";
        artifact.registry.srs_protocol_compiler_artifact_hash =
            std::string(64, 'a');
        artifact.config.dataset = "cora";
        artifact.config.batching_rule = "whole_graph_single";
        artifact.config.range_bits = 1;
        artifact.config.quantization.s_delta = 1;
        artifact.config.quantization.s_exp = 1;
        artifact.config.quantization.s_agg_pre = {1};
        artifact.config.quantization.s_agg = {1};
        artifact.config.input_preprocess_id = "row_normalize";
        artifact.dataset.name = "cora";
        artifact.dataset.input_preprocess_id = "row_normalize";
        artifact.dataset.num_nodes = 1;
        artifact.dataset.num_features = 1;
        artifact.local.name = "cora_normalized";
        artifact.local.input_preprocess_id = "row_normalize";
        artifact.local.num_nodes = 1;
        artifact.local.num_features = 1;
        artifact.model.input_preprocess_id = "row_normalize";

        artifact.tables =
            gatzk::protocol::build_canonical_static_tables(
                gatzk::protocol::CanonicalStaticTableShapeInputs{
                    .range_bits = artifact.config.range_bits,
                    .quantization = artifact.config.quantization,
                    .batching_rule = artifact.config.batching_rule,
                    .dataset_num_nodes = artifact.dataset.num_nodes,
                    .dataset_num_features = artifact.dataset.num_features,
                    .local_num_nodes = artifact.local.num_nodes,
                    .local_edge_count = artifact.local.edges.size(),
                });
        artifact.registry.static_table_digest =
            canonical_static_table_digest(artifact.tables);

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

        artifact.public_polynomial_descriptors.emplace(
            "runtime_public",
            gatzk::protocol::VerifierPublicPolynomialDescriptor{
                .name = "runtime_public",
                .basis = gatzk::algebra::PolynomialBasis::Coefficient,
                .domain = {},
                .size = 1});
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

        // Canonical registry serialization binds ordered metadata and points.
        gatzk::protocol::AuthenticatedVerifierRegistry registry;
        registry.registry_id = "registry";
        registry.model_arch_id = "arch";
        registry.model_param_id = "params";
        registry.input_preprocess_id = "row_normalize";
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
        registry.input_preprocess_id = "ogbn_arxiv_raw_float32";
        expect(digest != registry.canonical_digest(),
               "registry canonical digest did not bind input preprocessing identity");
        registry.input_preprocess_id = "row_normalize";
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
        leaking_commitment.opening_polynomial = std::make_shared<const gatzk::algebra::Polynomial>(
            gatzk::algebra::Polynomial::from_coefficients(
                "private-opening",
                {gatzk::algebra::FieldElement::one()}));
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
        const auto expected_tables = expected_small_canonical_static_tables();
        constexpr const char *kExpectedStaticTableSerializationSha256 =
            "4a446d4dccda578a0eb28cec1926c66c314f0b162c89a33e0f3587c67373d0b0";
        expect(registration.tables.range == expected_tables.range
                   && registration.tables.lrelu == expected_tables.lrelu
                   && registration.tables.elu == expected_tables.elu
                   && registration.tables.exp == expected_tables.exp,
               "small canonical fixture rows differ from the independent expected rows");
        expect(
            canonical_static_table_digest(registration.tables)
                == kExpectedStaticTableSerializationSha256,
            "small canonical fixture static-table serialization digest changed");
        const auto registration_bytes =
            gatzk::protocol::serialize_verifier_registration_artifact(registration);
        std::size_t static_cursor = 12;
        std::size_t static_payload = 0;
        std::size_t static_length = 0;
        for (std::uint32_t section = 1; section <= 12; ++section)
        {
            expect(read_u32(registration_bytes, static_cursor) == section,
                   "registration section scan lost canonical order");
            const auto length = read_u64(registration_bytes, static_cursor + 4);
            if (section == 5)
            {
                static_payload = static_cursor + 12;
                static_length = static_cast<std::size_t>(length);
            }
            static_cursor += 12 + static_cast<std::size_t>(length);
        }
        expect(static_payload != 0, "static-table section was not found");
        expect(static_length < 1024,
               "compact static-table section unexpectedly contains table rows");
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
            parsed_registration.registry.input_preprocess_id == "row_normalize",
            "verifier registration codec lost input preprocessing identity");
        expect(
            parsed_registration.public_commitments.size() == 1
                && parsed_registration.public_commitments.at("runtime_public").point == point,
            "verifier registration codec lost point-only public ownership");
        expect(parsed_registration.tables.range == registration.tables.range
                   && parsed_registration.tables.lrelu == registration.tables.lrelu
                   && parsed_registration.tables.elu == registration.tables.elu
                   && parsed_registration.tables.exp == registration.tables.exp,
               "compact static-table reconstruction changed canonical rows");
        expect(parsed_registration.tables.range == expected_tables.range
                   && parsed_registration.tables.lrelu == expected_tables.lrelu
                   && parsed_registration.tables.elu == expected_tables.elu
                   && parsed_registration.tables.exp == expected_tables.exp,
               "compact static-table reconstruction differs from independent expected rows");
        expect(
            parsed_registration.registry.static_table_digest
                == canonical_static_table_digest(parsed_registration.tables)
                && parsed_registration.registry.static_table_digest
                       == registration.registry.static_table_digest,
               "static-table digest no longer binds the complete canonical tables");
        expect(
            canonical_static_table_digest(parsed_registration.tables)
                == kExpectedStaticTableSerializationSha256,
            "reconstructed static-table serialization digest changed");
        expect(parsed_registration.tables.range.front().value() == 0
                   && parsed_registration.tables.range.back().value() == 1,
               "canonical T_range ordering changed");
        expect(parsed_registration.tables.lrelu.front().first
                       == gatzk::algebra::FieldElement::from_signed(-2)
                   && parsed_registration.tables.lrelu.back().first
                          == gatzk::algebra::FieldElement::from_signed(2),
               "canonical T_LReLU field encoding/order changed");
        expect(parsed_registration.tables.elu.front().first
                       == gatzk::algebra::FieldElement::from_signed(-2)
                   && parsed_registration.tables.elu.back().first
                          == gatzk::algebra::FieldElement::from_signed(2),
               "canonical T_ELU field encoding/order changed");
        expect(parsed_registration.tables.exp.front().first.value() == 0
                   && parsed_registration.tables.exp.back().first.value() == 4,
               "canonical T_exp ordering changed");

        auto marker_tamper = registration_bytes;
        expect(read_u64(marker_tamper, static_payload) > 0,
               "static-table marker is missing");
        marker_tamper[static_payload + 8] ^= 0x01U;
        refresh_digest(&marker_tamper);
        expect_rejected(marker_tamper, "unknown static-table marker");

        auto descriptor_tamper = registration_bytes;
        const auto marker_length = read_u64(descriptor_tamper, static_payload);
        expect(marker_length < static_length,
               "static-table descriptor marker exceeds section");
        write_u64(
            &descriptor_tamper,
            static_payload + 8 + static_cast<std::size_t>(marker_length) + 8,
            0);
        refresh_digest(&descriptor_tamper);
        expect_rejected(descriptor_tamper, "static-table descriptor tamper");

        auto single_byte_tamper = registration_bytes;
        single_byte_tamper[12] ^= 0x01U;
        expect_rejected(single_byte_tamper, "single-byte artifact tamper");

        auto trailing_tamper = registration_bytes;
        trailing_tamper.erase(trailing_tamper.end() - SHA256_DIGEST_LENGTH, trailing_tamper.end());
        trailing_tamper.push_back(0);
        trailing_tamper.resize(trailing_tamper.size() + SHA256_DIGEST_LENGTH);
        refresh_digest(&trailing_tamper);
        expect_rejected(trailing_tamper, "artifact trailing byte");

        auto noncanonical_tamper = registration_bytes;
        write_u32(&noncanonical_tamper, 12, 2);
        refresh_digest(&noncanonical_tamper);
        expect_rejected(noncanonical_tamper, "non-canonical section order");

        auto compiler_tamper = registration_bytes;
        std::size_t cursor = 12;
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

        const auto trusted_registry_digest = registration.canonical_digest();
        const auto trusted_parsed_registration =
            gatzk::protocol::parse_verifier_registration_artifact(
                registration_bytes, trusted_registry_digest);
        expect(
            trusted_parsed_registration.canonical_digest() == trusted_registry_digest
                && trusted_parsed_registration.tables.range == expected_tables.range
                && trusted_parsed_registration.tables.lrelu == expected_tables.lrelu
                && trusted_parsed_registration.tables.elu == expected_tables.elu
                && trusted_parsed_registration.tables.exp == expected_tables.exp,
            "trusted registration parse did not authenticate and rebuild canonical tables");

        bool trusted_parse_rejected = false;
        try
        {
            (void)gatzk::protocol::parse_verifier_registration_artifact(
                registration_bytes, std::string(64, '0'));
        }
        catch (const std::exception &error)
        {
            trusted_parse_rejected = true;
            expect(
                std::string(error.what()).find("trusted registry root")
                    != std::string::npos,
                "trusted parse rejected the wrong digest for the wrong reason");
        }
        expect(trusted_parse_rejected,
               "trusted registration parse accepted a mismatched digest");

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

        std::cout << "verifier_privacy_runtime: PASS\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "verifier_privacy_runtime: FAIL: " << error.what() << '\n';
        return 1;
    }
}
