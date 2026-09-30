#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace gatzk::crypto
{

    // The protocol compiler artifact is a single, line-oriented object.  The
    // parser below is intentionally shared by the ceremony producer and the
    // runtime loader so neither side can grow a second spelling of this
    // authenticated format.
    inline constexpr std::string_view kCompilerArtifactMagic = "GATZK-SRS-COMPILER-ARTIFACT\n";
    // Keep the count marker in the scalar section rather than inferring it
    // from a prefix scan.  The actual schema has 17 scalar fields.
    inline constexpr std::array<std::string_view, 17> kCompilerArtifactFields = {"protocol_template_hex",
                                                                                 "proof_schema_template_hex",
                                                                                 "observation_ledger_template_hex",
                                                                                 "random_identity_ledger_template_hex",
                                                                                 "security_budget_ledger_template_hex",
                                                                                 "protocol_schema_id",
                                                                                 "max_instance_envelope_id",
                                                                                 "max_instance_envelope_hex",
                                                                                 "max_instance_envelope_hash",
                                                                                 "degree_bound",
                                                                                 "domain_count",
                                                                                 "domain_size_catalog_hash",
                                                                                 "domain_generator_catalog_hash",
                                                                                 "proof_schema_template_hash",
                                                                                 "observation_ledger_hash",
                                                                                 "random_identity_ledger_hash",
                                                                                 "security_budget_ledger_hash"};

    inline constexpr std::array<std::string_view, 7> kCompilerArtifactDomainNames = {"C", "FH", "N", "cat", "d", "edge", "in"};

    struct CompilerArtifactDomain
    {
        std::string name;
        std::string size;
        std::string generator;
    };

    struct CompilerArtifactDocument
    {
        std::array<std::string, kCompilerArtifactFields.size()> scalars{};
        std::array<CompilerArtifactDomain, kCompilerArtifactDomainNames.size()> domains{};
        std::string canonical_prefix;
        std::string artifact_hash;

        const std::string &scalar(std::string_view key) const
        {
            for (std::size_t index = 0; index < kCompilerArtifactFields.size(); ++index)
            {
                if (kCompilerArtifactFields[index] == key)
                    return scalars[index];
            }
            throw std::runtime_error("unknown protocol compiler artifact scalar key: " + std::string(key));
        }
    };

    inline std::vector<std::string> parse_ordered_key_value_lines(std::string_view text, const std::vector<std::string_view> &keys,
                                                                  std::string_view object_name)
    {
        if (text.empty() || text.back() != '\n' || text.find('\r') != std::string_view::npos)
        {
            throw std::runtime_error(std::string(object_name) + " must use canonical LF lines and a final newline");
        }
        std::vector<std::string> values;
        values.reserve(keys.size());
        std::size_t position = 0;
        for (const auto key : keys)
        {
            const auto line_end = text.find('\n', position);
            if (line_end == std::string_view::npos)
                throw std::runtime_error(std::string(object_name) + " is missing field: " + std::string(key));
            const auto line = text.substr(position, line_end - position);
            const auto prefix = std::string(key) + '=';
            if (line.size() <= prefix.size() || line.substr(0, prefix.size()) != prefix || line.find('=', prefix.size()) != std::string_view::npos)
            {
                throw std::runtime_error(std::string(object_name) + " field order, duplicate, or encoding mismatch: " + std::string(key));
            }
            values.emplace_back(line.substr(prefix.size()));
            position = line_end + 1;
        }
        if (position != text.size())
            throw std::runtime_error(std::string(object_name) + " has trailing fields or bytes");
        return values;
    }

    inline CompilerArtifactDocument parse_compiler_artifact(std::string_view text)
    {
        if (!text.starts_with(kCompilerArtifactMagic))
            throw std::runtime_error("compiler artifact magic mismatch");
        if (text.size() <= kCompilerArtifactMagic.size() || text.find('\r') != std::string_view::npos)
        {
            throw std::runtime_error("compiler artifact must use canonical LF lines");
        }

        CompilerArtifactDocument document;
        std::size_t position = kCompilerArtifactMagic.size();
        for (std::size_t index = 0; index < kCompilerArtifactFields.size(); ++index)
        {
            const auto line_end = text.find('\n', position);
            if (line_end == std::string_view::npos)
                throw std::runtime_error("compiler artifact is missing scalar field: " + std::string(kCompilerArtifactFields[index]));
            const auto line = text.substr(position, line_end - position);
            const auto prefix = std::string(kCompilerArtifactFields[index]) + '=';
            if (line.size() <= prefix.size() || line.substr(0, prefix.size()) != prefix || line.find('=', prefix.size()) != std::string_view::npos)
            {
                throw std::runtime_error("compiler artifact scalar field order, duplicate, or encoding mismatch: " +
                                         std::string(kCompilerArtifactFields[index]));
            }
            document.scalars[index] = std::string(line.substr(prefix.size()));
            position = line_end + 1;
        }

        if (document.scalar("domain_count") != std::to_string(kCompilerArtifactDomainNames.size()))
        {
            throw std::runtime_error("compiler artifact domain count is not canonical");
        }

        for (std::size_t index = 0; index < kCompilerArtifactDomainNames.size(); ++index)
        {
            const auto line_end = text.find('\n', position);
            if (line_end == std::string_view::npos)
                throw std::runtime_error("compiler artifact is missing domain entry: " + std::string(kCompilerArtifactDomainNames[index]));
            const auto line = text.substr(position, line_end - position);
            const auto prefix = std::string("domain.") + std::string(kCompilerArtifactDomainNames[index]) + '=';
            if (line.size() <= prefix.size() || line.substr(0, prefix.size()) != prefix)
            {
                throw std::runtime_error("compiler artifact domain order, duplicate, or encoding mismatch: " +
                                         std::string(kCompilerArtifactDomainNames[index]));
            }
            const auto payload = line.substr(prefix.size());
            const auto comma = payload.find(',');
            if (comma == std::string_view::npos || comma == 0 || comma + 1 == payload.size() || payload.find(',', comma + 1) != std::string_view::npos ||
                payload.find('=') != std::string_view::npos)
            {
                throw std::runtime_error("compiler artifact domain entry encoding mismatch: " + std::string(kCompilerArtifactDomainNames[index]));
            }
            document.domains[index] = {std::string(kCompilerArtifactDomainNames[index]), std::string(payload.substr(0, comma)),
                                       std::string(payload.substr(comma + 1))};
            position = line_end + 1;
        }

        const auto hash_line_start = position;
        const auto hash_line_end = text.find('\n', position);
        if (hash_line_end == std::string_view::npos)
            throw std::runtime_error("compiler artifact is missing final artifact_hash field");
        const auto hash_line = text.substr(position, hash_line_end - position);
        constexpr std::string_view hash_prefix = "artifact_hash=";
        if (hash_line.size() <= hash_prefix.size() || !hash_line.starts_with(hash_prefix) || hash_line.find('=', hash_prefix.size()) != std::string_view::npos)
        {
            throw std::runtime_error("compiler artifact artifact_hash must be the final field");
        }
        const auto hash = hash_line.substr(hash_prefix.size());
        if (hash.size() != 64 || hash.find_first_not_of("0123456789abcdef") != std::string_view::npos)
        {
            throw std::runtime_error("compiler artifact artifact_hash is not canonical SHA-256 hex");
        }
        position = hash_line_end + 1;
        if (position != text.size())
            throw std::runtime_error("compiler artifact has trailing fields or bytes");
        document.canonical_prefix = std::string(text.substr(0, hash_line_start));
        document.artifact_hash = std::string(hash);
        return document;
    }

} // namespace gatzk::crypto
