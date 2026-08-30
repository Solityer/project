#include "gatzk/crypto/kzg.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
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

    SRSManifest make_test_manifest()
    {
        SRSManifest manifest;
        manifest.manifest_hash = "manifest-hash";
        manifest.srs_id = "test-srs";
        manifest.srs_type = "trusted_srs";
        manifest.degree_bound = 8;
        manifest.generator_id = "test-generator";
        manifest.generator_version = "v1";
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
