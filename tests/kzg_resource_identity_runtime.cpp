#include "gatzk/crypto/kzg.hpp"

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <openssl/sha.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{
    using gatzk::crypto::SRSManifest;

    struct TemporaryResourceRoot
    {
        std::filesystem::path path;

        ~TemporaryResourceRoot()
        {
            std::error_code error;
            std::filesystem::remove_all(path, error);
        }
    };

    void write_bytes(
        const std::filesystem::path &path,
        std::string_view bytes)
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output)
            throw std::runtime_error("failed to create test resource: " + path.string());
        output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        if (!output)
            throw std::runtime_error("failed to write test resource: " + path.string());
    }

    std::string sha256_hex(std::string_view bytes)
    {
        std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
        SHA256(
            reinterpret_cast<const unsigned char *>(bytes.data()),
            bytes.size(),
            digest.data());
        std::ostringstream output;
        output << std::hex << std::setfill('0');
        for (const auto value : digest)
            output << std::setw(2) << static_cast<unsigned int>(value);
        return output.str();
    }

    enum class NestedManifestMutation
    {
        unknown,
        duplicate,
        missing,
        nested_same_name,
        string_escape,
        wrong_type,
    };

    std::string nested_srs_manifest(NestedManifestMutation mutation)
    {
        std::ostringstream prefix;
        prefix << R"({
  "manifest_format": "zkgat_srs_manifest",
  "srs_id": "test-srs",
  "curve_id": "bls12_381",
  "field_modulus_id": "bls12_381_fr",
  "srs_type": "trusted_srs",
  "ceremony_transcript_hash": "ceremony",
  "setup_transcript_hash": "ceremony",
  "degree_bound": )";
        if (mutation == NestedManifestMutation::wrong_type)
            prefix << R"("8")";
        else
            prefix << "8";
        prefix << R"(,
  "domain_sizes": {"N": 8},
  "domain_generators": {)";
        if (mutation == NestedManifestMutation::unknown)
            prefix << R"("N": "generator", "unexpected": "value")";
        else if (mutation == NestedManifestMutation::duplicate)
            prefix << R"("N": "generator", "N": "duplicate")";
        else
            prefix << R"("N": "generator")";
        prefix << R"(},
  "root_derivation_id": "root",
  "row_order": "ascending",
  "protocol_compiler_artifact_path": "artifact.bin",
  "protocol_compiler_artifact_hash": "artifact-hash",
  "protocol_schema_id": "protocol",
  "max_instance_envelope_id": "envelope",
  "max_instance_envelope_hash": "envelope",
  "max_instance_envelope": {)";
        const std::array<std::string_view, 16> envelope_fields = {
            "max_layers",
            "max_hidden_heads_per_layer",
            "max_hidden_head_dim",
            "max_input_dim",
            "max_output_heads",
            "max_classes",
            "max_graphs",
            "max_nodes",
            "max_edges",
            "max_total_nodes",
            "max_concat_dim",
            "max_range_bits",
            "max_coefficient_object_length",
            "max_object_family_count",
            "max_coefficient_chunks",
            "security_target_bits",
        };
        for (std::size_t index = 0; index < envelope_fields.size(); ++index)
        {
            if (mutation == NestedManifestMutation::missing
                && envelope_fields[index] == "security_target_bits")
                continue;
            if (index != 0)
                prefix << ", ";
            prefix << '"' << envelope_fields[index] << R"(": 1)";
        }
        prefix << R"(},
  "proof_schema_template_hash": "proof",
  "observation_ledger_hash": "observation",
  "random_identity_ledger_hash": "random",
  "security_budget_ledger_hash": "security",
  "pk_coef_path": "pk_coef.bin",
  "pk_coef_hash": "pk-coef",
  "pk_lagrange_paths": {"N": ")";
        if (mutation == NestedManifestMutation::nested_same_name)
            prefix << R"(pk_lagrange_N.bin", "manifest_hash": "nested)";
        else if (mutation == NestedManifestMutation::string_escape)
            prefix << R"(pk_lagrange_\"N\".bin {)";
        else
            prefix << R"(pk_lagrange_N.bin)";
        prefix << R"("
  },
  "pk_lagrange_hashes": {"N": "pk-lagrange"},
  "vk_path": "vk.bin",
  "vk_hash": "vk",
  "vanishing_g2_hash": "vanishing",
  "srs_audit_report_path": "audit.txt",
  "srs_audit_report_hash": "audit",
  "srs_consistency_hash": "audit",
  "encoding": "encoding",
  "generator_id": "zkgat_srs_setup_tool",
  "security_level": "120-bit",
  "kzg_assumption_profile_id": "shifted_kzg_divisibility",
  "kzg_assumption_degree_bound": 8,
  "kzg_assumption_security_bits": 128,
  "kzg_assumption_domain_catalog_hash": "domain-catalog",
  "domain_generator_catalog_hash": "generator-catalog",
  "manifest_hash": ")";
        const auto body = prefix.str();
        const auto manifest_hash_field = body.rfind("  \"manifest_hash\"");
        if (manifest_hash_field == std::string::npos)
            throw std::runtime_error("test SRS manifest hash field is missing");
        return body + sha256_hex(body.substr(0, manifest_hash_field)) + "\"\n}\n";
    }

    SRSManifest make_test_manifest()
    {
        SRSManifest manifest;
        manifest.manifest_hash = "manifest-hash";
        manifest.srs_id = "test-srs";
        manifest.srs_type = "trusted_srs";
        manifest.degree_bound = 8;
        manifest.generator_id = "test-generator";
        manifest.encoding = "test-encoding";
        manifest.protocol_schema_id = "test-protocol";
        manifest.protocol_compiler_artifact_hash = "compiler-hash";
        manifest.proof_schema_template_hash = "proof-schema-hash";
        manifest.observation_ledger_hash = "observation-hash";
        manifest.random_identity_ledger_hash = "random-identity-hash";
        manifest.security_budget_ledger_hash = "security-budget-hash";
        manifest.domain_generator_catalog_hash = "domain-catalog-hash";
        manifest.pk_coef_path = "pk_coef.bin";
        manifest.vk_path = "vk.bin";
        manifest.protocol_compiler_artifact_path = "protocol_compiler_artifact.bin";
        manifest.srs_audit_report_path = "srs_audit_report.txt";
        manifest.pk_lagrange_paths.emplace("N", "pk_lagrange_N.bin");
        return manifest;
    }

    void require(bool condition, const std::string &message)
    {
        if (!condition)
            throw std::runtime_error(message);
    }
}

int main()
{
    try
    {
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        TemporaryResourceRoot temporary{
            std::filesystem::temp_directory_path()
            / ("gatzk_kzg_resource_identity_" + std::to_string(nonce))};
        std::filesystem::create_directories(temporary.path);

        write_bytes(
            temporary.path / "srs_manifest.json",
            "{\n  \"unexpected_manifest_field\": 1\n}\n");
        bool rejected_unexpected_manifest_field = false;
        try
        {
            (void)gatzk::crypto::KZG::load_keys(
                temporary.path.string(), "trusted_srs", "test-srs");
        }
        catch (const std::exception &error)
        {
            rejected_unexpected_manifest_field =
                std::string(error.what()).find(
                    "unknown/non-canonical manifest field")
                != std::string::npos;
        }
        require(
            rejected_unexpected_manifest_field,
            "SRS manifest unknown field was not rejected");

        write_bytes(
            temporary.path / "srs_manifest.json",
            "{\n  \"manifest_format\": \"a\",\n  \"manifest_format\": \"b\"\n}\n");
        bool rejected_duplicate_manifest_field = false;
        try
        {
            (void)gatzk::crypto::KZG::load_keys(
                temporary.path.string(), "trusted_srs", "test-srs");
        }
        catch (const std::exception &error)
        {
            rejected_duplicate_manifest_field =
                std::string(error.what()).find("duplicate top-level field")
                != std::string::npos;
        }
        require(
            rejected_duplicate_manifest_field,
            "SRS manifest duplicate field was not rejected");

        write_bytes(
            temporary.path / "srs_manifest.json",
            "{\n  \"manifest_format\": \"a\"\n}\n");
        bool rejected_missing_manifest_field = false;
        try
        {
            (void)gatzk::crypto::KZG::load_keys(
                temporary.path.string(), "trusted_srs", "test-srs");
        }
        catch (const std::exception &error)
        {
            rejected_missing_manifest_field =
                std::string(error.what()).find("missing canonical field")
                != std::string::npos;
        }
        require(
            rejected_missing_manifest_field,
            "SRS manifest missing field was not rejected");

        const auto require_nested_rejection =
            [&temporary](NestedManifestMutation mutation, std::string_view needle)
        {
            write_bytes(
                temporary.path / "srs_manifest.json",
                nested_srs_manifest(mutation));
            try
            {
                (void)gatzk::crypto::KZG::load_keys(
                    temporary.path.string(), "trusted_srs", "test-srs");
            }
            catch (const std::exception &error)
            {
                require(
                    std::string(error.what()).find(needle) != std::string::npos,
                    std::string("SRS nested schema rejected for an unexpected reason: ")
                        + error.what());
                return;
            }
            throw std::runtime_error("SRS nested schema mutation was not rejected");
        };
        require_nested_rejection(
            NestedManifestMutation::unknown,
            "unknown/non-canonical nested field");
        require_nested_rejection(
            NestedManifestMutation::duplicate,
            "duplicate nested field");
        require_nested_rejection(
            NestedManifestMutation::missing,
            "missing canonical nested field");
        require_nested_rejection(
            NestedManifestMutation::nested_same_name,
            "unknown/non-canonical nested field");
        require_nested_rejection(
            NestedManifestMutation::wrong_type,
            "unsigned integer");

        write_bytes(
            temporary.path / "srs_manifest.json",
            nested_srs_manifest(NestedManifestMutation::string_escape));
        bool escaped_string_reached_semantic_validation = false;
        try
        {
            (void)gatzk::crypto::KZG::load_keys(
                temporary.path.string(), "trusted_srs", "test-srs");
        }
        catch (const std::exception &error)
        {
            require(
                std::string(error.what()).find(
                    "SRS shifted-KZG assumption profile is missing or inconsistent")
                    != std::string::npos,
                "SRS escaped string fixture was rejected before semantic validation");
            escaped_string_reached_semantic_validation = true;
        }
        require(
            escaped_string_reached_semantic_validation,
            "SRS escaped string fixture unexpectedly loaded successfully");

        const auto manifest = make_test_manifest();
        const std::vector<std::string> selected_domains{"N"};
        const std::vector<std::pair<std::string, std::string>> resources{
            {"manifest", "srs_manifest.json"},
            {"pk_coef", "pk_coef.bin"},
            // vk.bin contains g2_one, g2_tau, and the complete authenticated
            // domain-zero-G2 catalogue in the production format.
            {"vk_and_domain_zero_g2", "vk.bin"},
            {"pk_lagrange", "pk_lagrange_N.bin"},
            {"protocol_compiler_artifact", "protocol_compiler_artifact.bin"},
            {"srs_audit_report", "srs_audit_report.txt"},
        };

        for (const auto &resource : resources)
            write_bytes(temporary.path / resource.second, "before!!");

        for (const auto &[label, relative_path] : resources)
        {
            const auto path = temporary.path / relative_path;
            write_bytes(path, "before!!");

            const auto initial_identity =
                gatzk::crypto::canonical_kzg_key_resource_identity(
                    temporary.path, manifest, selected_domains);
            const auto cached_payload_identity = initial_identity;

            // Keep the path and byte length unchanged.  A cache keyed by path,
            // mtime, or size alone must not accept this stale payload.
            write_bytes(path, "after!!!");
            const auto replacement_identity =
                gatzk::crypto::canonical_kzg_key_resource_identity(
                    temporary.path, manifest, selected_domains);

            require(
                initial_identity != replacement_identity,
                label + ": same-path byte replacement did not change identity");
            require(
                cached_payload_identity != replacement_identity,
                label + ": stale cache payload identity would be accepted");

            write_bytes(path, "before!!");
            const auto restored_identity =
                gatzk::crypto::canonical_kzg_key_resource_identity(
                    temporary.path, manifest, selected_domains);
            require(
                initial_identity == restored_identity,
                label + ": identity was not deterministic after restoration");
        }

        std::cout << "kzg resource identity runtime: PASS\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "kzg resource identity runtime: FAIL: " << error.what() << '\n';
        return 1;
    }
}
