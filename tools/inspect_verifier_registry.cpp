#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <openssl/sha.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gatzk/protocol/schema.hpp"
#include "gatzk/protocol/verifier_context.hpp"

namespace
{
    std::string require_value(int &index, int argc, char **argv, const std::string &option)
    {
        if (index + 1 >= argc)
            throw std::runtime_error("missing value for " + option);
        const std::string value = argv[++index];
        if (value.empty() || value.starts_with("--"))
            throw std::runtime_error("missing value for " + option);
        return value;
    }

    bool is_lower_hex_256(const std::string &value)
    {
        return value.size() == SHA256_DIGEST_LENGTH * 2 &&
               std::all_of(value.begin(), value.end(), [](unsigned char ch) { return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'); });
    }

    std::string sha256_file(const std::filesystem::path &path)
    {
        std::ifstream input(path, std::ios::binary);
        if (!input)
            throw std::runtime_error("failed to open verifier registry: " + path.string());
        SHA256_CTX context{};
        SHA256_Init(&context);
        std::array<char, 1U << 16U> buffer{};
        while (input)
        {
            input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
            const auto count = input.gcount();
            if (count > 0)
                SHA256_Update(&context, buffer.data(), static_cast<std::size_t>(count));
        }
        if (input.bad())
            throw std::runtime_error("failed while reading verifier registry: " + path.string());
        std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
        SHA256_Final(digest.data(), &context);
        std::ostringstream out;
        out << std::hex << std::setfill('0');
        for (const auto byte : digest)
            out << std::setw(2) << static_cast<unsigned int>(byte);
        return out.str();
    }

    std::string json_escape(const std::string &value)
    {
        std::string out;
        out.reserve(value.size() + 8);
        for (const auto ch : value)
        {
            switch (ch)
            {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20U)
                    throw std::runtime_error("registry metadata contains a control character");
                out.push_back(ch);
                break;
            }
        }
        return out;
    }

    void emit_string(const char *key, const std::string &value, bool *first)
    {
        if (!*first)
            std::cout << ',';
        *first = false;
        std::cout << '"' << key << "\":\"" << json_escape(value) << '"';
    }

    void emit_size(const char *key, std::size_t value, bool *first)
    {
        if (!*first)
            std::cout << ',';
        *first = false;
        std::cout << '"' << key << "\":" << value;
    }

    void emit_bool(const char *key, bool value, bool *first)
    {
        if (!*first)
            std::cout << ',';
        *first = false;
        std::cout << '"' << key << "\":" << (value ? "true" : "false");
    }
}

int main(int argc, char **argv)
{
    try
    {
        std::string registry_path;
        std::string trusted_digest;
        for (int index = 1; index < argc; ++index)
        {
            const std::string option = argv[index];
            if (option == "--registry")
                registry_path = require_value(index, argc, argv, option);
            else if (option == "--trusted-digest")
                trusted_digest = require_value(index, argc, argv, option);
            else
                throw std::runtime_error("unknown option: " + option);
        }
        if (registry_path.empty() || !is_lower_hex_256(trusted_digest))
            throw std::runtime_error("usage: gatzk_registry_inspect --registry <artifact> "
                                     "--trusted-digest <lowercase-sha256>");

        const auto path = std::filesystem::path(registry_path);
        std::error_code status_error;
        const auto status = std::filesystem::symlink_status(path, status_error);
        if (status_error || std::filesystem::is_symlink(status) || !std::filesystem::is_regular_file(status))
            throw std::runtime_error("verifier registry must be a regular non-symlink file: " + path.string());
        const auto file_digest = sha256_file(path);
        if (file_digest != trusted_digest)
            throw std::runtime_error("verifier registry file digest does not match trusted digest");

        // The formal loader performs the complete canonical parse, digest
        // check, compiler-artifact validation, IntegerBoundLedger validation,
        // static-table validation, SecurityBudgetLedger reconstruction, and
        // model/static registry shape checks.  This utility deliberately uses
        // that independent loader instead of trusting a metadata sidecar.
        const auto verifier_context = gatzk::protocol::load_verifier_context_from_registry(path.string(), trusted_digest);
        if (!gatzk::protocol::validate_verifier_context(verifier_context))
            throw std::runtime_error("verifier registry independent context validation failed");
        if (verifier_context.compiler_artifact == nullptr)
            throw std::runtime_error("verifier registry has no compiler artifact");

        const auto &registry = verifier_context.registry;
        const auto &compiler = *verifier_context.compiler_artifact;
        const auto budget = gatzk::protocol::evaluate_security_budget(gatzk::protocol::detail::make_verifier_execution_context(verifier_context), compiler);
        if (!budget.passed || budget.ledger_digest != registry.security_budget_ledger_digest || budget.exact_budget != registry.security_budget_exact)
            throw std::runtime_error("verifier registry SecurityBudgetLedger reconstruction mismatch");

        bool first = true;
        std::cout << '{';
        emit_string("format", "zkgat_registry_inspection", &first);
        emit_string("artifact_sha256", file_digest, &first);
        emit_string("registry_digest", verifier_context.authenticated_registry_digest, &first);
        emit_string("registry_id", registry.registry_id, &first);
        emit_string("model_arch_id", registry.model_arch_id, &first);
        emit_string("model_param_id", registry.model_param_id, &first);
        emit_string("quant_cfg_id", registry.quant_cfg_id, &first);
        emit_string("static_table_id", registry.static_table_id, &first);
        emit_string("srs_id", registry.srs_id, &first);
        emit_string("protocol_schema_id", registry.protocol_schema_id, &first);
        emit_string("instance_schema_hash", registry.instance_schema_hash, &first);
        emit_string("srs_manifest_hash", registry.srs_manifest_hash, &first);
        emit_string("model_layout_digest", registry.model_layout_digest, &first);
        emit_string("quant_chain_digest", registry.quant_chain_digest, &first);
        emit_string("static_table_digest", registry.static_table_digest, &first);
        emit_string("static_table_generator_id", registry.static_table_generator_id, &first);
        emit_string("static_table_spec_hash", registry.static_table_spec_hash, &first);
        emit_string("range_and_scale_certificate", registry.range_and_scale_certificate, &first);
        emit_string("integer_bound_ledger_digest", registry.integer_bound_ledger_digest, &first);
        emit_string("public_input_digest", registry.public_input_digest, &first);
        emit_string("topology_digest", registry.topology_digest, &first);
        emit_string("security_budget_ledger_digest", registry.security_budget_ledger_digest, &first);
        emit_string("security_budget_exact", registry.security_budget_exact, &first);
        emit_size("integer_bound_ledger_count", registry.integer_bound_ledger.size(), &first);
        emit_size("security_budget_ledger_count", compiler.security_budget_ledger.size(), &first);
        emit_size("model_entry_count", registry.model_entries.size(), &first);
        emit_size("static_entry_count", registry.static_entries.size(), &first);
        emit_bool("security_budget_passed", budget.passed, &first);
        std::cout << "}\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
