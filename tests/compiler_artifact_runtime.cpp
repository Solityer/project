#include <iostream>
#include <stdexcept>
#include <string>

#include "gatzk/crypto/compiler_artifact.hpp"

namespace
{
    std::string canonical_fixture()
    {
        std::string text(gatzk::crypto::kCompilerArtifactMagic);
        for (const auto key : gatzk::crypto::kCompilerArtifactFields)
        {
            text += key;
            text += '=';
            text += key == "domain_count" ? "7" : "value";
            text += '\n';
        }
        for (const auto name : gatzk::crypto::kCompilerArtifactDomainNames)
        {
            text += "domain.";
            text += name;
            text += "=1,value\n";
        }
        text += "artifact_hash=";
        text += std::string(64, '0');
        text += '\n';
        return text;
    }

    void expect_rejected(const std::string &text, const char *label)
    {
        try
        {
            (void)gatzk::crypto::parse_compiler_artifact(text);
        }
        catch (const std::exception &)
        {
            return;
        }
        throw std::runtime_error(std::string("compiler artifact accepted ") + label);
    }
}

int main()
{
    const auto fixture = canonical_fixture();
    const auto parsed = gatzk::crypto::parse_compiler_artifact(fixture);
    if (parsed.scalar("domain_count") != "7"
        || parsed.domains.front().name != "C"
        || parsed.domains.back().name != "in")
    {
        throw std::runtime_error("compiler artifact canonical fixture roundtrip failed");
    }

    auto unknown = fixture;
    const auto marker = std::string("artifact_hash=");
    unknown.replace(unknown.find(marker), 0, "unknown=value\n");
    expect_rejected(unknown, "unknown field");

    auto duplicate = fixture;
    const auto first_field_end = duplicate.find('\n', gatzk::crypto::kCompilerArtifactMagic.size());
    duplicate.insert(first_field_end + 1, duplicate.substr(
        gatzk::crypto::kCompilerArtifactMagic.size(), first_field_end + 1
            - gatzk::crypto::kCompilerArtifactMagic.size()));
    expect_rejected(duplicate, "duplicate field");

    auto missing = fixture;
    missing.erase(gatzk::crypto::kCompilerArtifactMagic.size(),
                  std::string("protocol_template_hex=value\n").size());
    expect_rejected(missing, "missing field");

    auto reordered = fixture;
    const auto first_start = gatzk::crypto::kCompilerArtifactMagic.size();
    const auto first_end = reordered.find('\n', first_start) + 1;
    const auto second_end = reordered.find('\n', first_end) + 1;
    const auto first_line = reordered.substr(first_start, first_end - first_start);
    const auto second_line = reordered.substr(first_end, second_end - first_end);
    reordered.replace(first_start, second_end - first_start, second_line + first_line);
    expect_rejected(reordered, "reordered field");

    auto trailing = fixture;
    trailing += "trailing=value\n";
    expect_rejected(trailing, "trailing field");

    std::cout << "compiler artifact runtime: PASS\n";
    return 0;
}
