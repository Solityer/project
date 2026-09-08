#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <openssl/sha.h>

#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/crypto/curve.hpp"
#include "gatzk/protocol/challenges.hpp"
#include "gatzk/protocol/proof_codec.hpp"
#include "gatzk/protocol/verifier.hpp"
#include "gatzk/protocol/verifier_context.hpp"
#include "gatzk/util/logging.hpp"

namespace
{
    using gatzk::algebra::FieldElement;
    using gatzk::protocol::PayloadValueKind;
    using gatzk::protocol::Proof;
    using gatzk::protocol::VerifierContext;

    constexpr std::size_t kMaxInputBytes = 2ULL << 30U;
    constexpr std::array<std::string_view, 39> kRequiredTamperCases = {
        "codec_tag",
        "codec_length",
        "codec_trailing",
        "public_output",
        "metadata",
        "schema",
        "ids",
        "dynamic_commitment",
        "quotient_commitment",
        "opening",
        "reblind_commitment",
        "static_registry",
        "model_registry",
        "reference_mismatch",
        "private_opening",
        "route_padding",
        "route_query",
        "lookup_inverse",
        "lookup_multiplicity",
        "psq",
        "shift",
        "logup",
        "aggregate",
        "average",
        "division",
        "transcript_challenge",
        "transcript_value",
        "srs_metadata",
        "srs_consistency",
        "srs_resource_identity",
        "srs_resource_compromised",
        "proof_scope",
        "observation_ledger",
        "random_identity_ledger",
        "integer_bound_ledger",
        "integer_bound_certificate",
        "security_budget_ledger",
        "security_budget_exact",
        "public_input_digest",
    };

    struct Options
    {
        std::filesystem::path proof_path;
        std::filesystem::path registry_path;
        std::filesystem::path run_manifest_path;
        std::filesystem::path output_path;
        std::string trusted_registry_digest;
    };

    std::string require_option_value(
        int &index,
        int argc,
        char **argv,
        const std::string &option)
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
        return value.size() == SHA256_DIGEST_LENGTH * 2U
            && std::all_of(
                value.begin(), value.end(),
                [](unsigned char ch)
                {
                    return (ch >= '0' && ch <= '9')
                        || (ch >= 'a' && ch <= 'f');
                });
    }

    std::string sha256_bytes(const std::vector<std::uint8_t> &bytes)
    {
        std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
        SHA256(bytes.data(), bytes.size(), digest.data());
        std::ostringstream output;
        output << std::hex << std::setfill('0');
        for (const auto byte : digest)
            output << std::setw(2) << static_cast<unsigned int>(byte);
        return output.str();
    }

    std::vector<std::uint8_t> read_bytes(const std::filesystem::path &path)
    {
        std::error_code status_error;
        const auto status = std::filesystem::symlink_status(path, status_error);
        if (status_error || std::filesystem::is_symlink(status)
            || !std::filesystem::is_regular_file(status))
            throw std::runtime_error(
                "input must be a regular non-symlink file: " + path.string());
        const auto size = std::filesystem::file_size(path, status_error);
        if (status_error || size > kMaxInputBytes)
            throw std::runtime_error("input file is missing or exceeds the size cap: " + path.string());
        std::ifstream input(path, std::ios::binary);
        if (!input)
            throw std::runtime_error("failed to open input: " + path.string());
        std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
        if (!bytes.empty()
            && !input.read(
                reinterpret_cast<char *>(bytes.data()),
                static_cast<std::streamsize>(bytes.size())))
            throw std::runtime_error("failed to read input: " + path.string());
        return bytes;
    }

    std::string read_text(const std::filesystem::path &path)
    {
        const auto bytes = read_bytes(path);
        return std::string(
            reinterpret_cast<const char *>(bytes.data()), bytes.size());
    }

    // Run manifests are emitted by util::write_json_object as one flat object
    // whose values are all strings.  Keep this parser local to the test suite;
    // it is not a second production artifact/config parser.
    class FlatJsonStringObjectReader
    {
    public:
        explicit FlatJsonStringObjectReader(std::string_view input)
            : input_(input)
        {
        }

        std::map<std::string, std::string> parse()
        {
            skip_space();
            expect('{');
            skip_space();
            std::map<std::string, std::string> values;
            if (peek() == '}')
            {
                ++cursor_;
                finish();
                return values;
            }
            while (true)
            {
                const auto key = parse_string();
                skip_space();
                expect(':');
                const auto value = parse_string();
                if (!values.emplace(key, value).second)
                    throw std::runtime_error("run manifest contains a duplicate key: " + key);
                skip_space();
                const auto separator = peek();
                if (separator == ',')
                {
                    ++cursor_;
                    skip_space();
                    continue;
                }
                if (separator == '}')
                {
                    ++cursor_;
                    finish();
                    return values;
                }
                throw std::runtime_error("run manifest object separator is invalid");
            }
        }

    private:
        char peek() const
        {
            if (cursor_ >= input_.size())
                throw std::runtime_error("run manifest ended unexpectedly");
            return input_[cursor_];
        }

        void expect(char value)
        {
            if (peek() != value)
                throw std::runtime_error("run manifest JSON syntax is invalid");
            ++cursor_;
        }

        void skip_space()
        {
            while (cursor_ < input_.size())
            {
                const auto ch = static_cast<unsigned char>(input_[cursor_]);
                if (ch != ' ' && ch != '\n' && ch != '\r' && ch != '\t')
                    break;
                ++cursor_;
            }
        }

        std::string parse_string()
        {
            skip_space();
            expect('"');
            std::string value;
            while (true)
            {
                if (cursor_ >= input_.size())
                    throw std::runtime_error("run manifest string is unterminated");
                const auto ch = input_[cursor_++];
                if (ch == '"')
                    return value;
                if (ch == '\\')
                {
                    if (cursor_ >= input_.size())
                        throw std::runtime_error("run manifest string escape is unterminated");
                    const auto escaped = input_[cursor_++];
                    switch (escaped)
                    {
                    case '"': value.push_back('"'); break;
                    case '\\': value.push_back('\\'); break;
                    case 'n': value.push_back('\n'); break;
                    case 'r': value.push_back('\r'); break;
                    case 't': value.push_back('\t'); break;
                    default:
                        throw std::runtime_error("run manifest contains an unsupported string escape");
                    }
                    continue;
                }
                if (static_cast<unsigned char>(ch) < 0x20U)
                    throw std::runtime_error("run manifest contains a control character");
                value.push_back(ch);
            }
        }

        void finish()
        {
            skip_space();
            if (cursor_ != input_.size())
                throw std::runtime_error("run manifest has trailing data");
        }

        std::string_view input_;
        std::size_t cursor_ = 0;
    };

    std::string manifest_value(
        const std::map<std::string, std::string> &manifest,
        std::string_view key)
    {
        const auto iterator = manifest.find(std::string(key));
        if (iterator == manifest.end() || iterator->second.empty())
            throw std::runtime_error("run manifest is missing required field: " + std::string(key));
        return iterator->second;
    }

    std::size_t manifest_size(
        const std::map<std::string, std::string> &manifest,
        std::string_view key)
    {
        const auto value = manifest_value(manifest, key);
        std::size_t consumed = 0;
        std::size_t result = 0;
        try
        {
            result = std::stoull(value, &consumed, 10);
        }
        catch (const std::exception &)
        {
            throw std::runtime_error("run manifest field is not an unsigned integer: " + std::string(key));
        }
        if (consumed != value.size())
            throw std::runtime_error("run manifest field is not an unsigned integer: " + std::string(key));
        return result;
    }

    void require_manifest_hash(
        const std::map<std::string, std::string> &manifest,
        std::string_view key)
    {
        const auto value = manifest_value(manifest, key);
        if (!is_lower_hex_256(value))
            throw std::runtime_error("run manifest field is not a lowercase SHA-256 digest: " + std::string(key));
    }

    bool flip_hex_digit(std::string *value)
    {
        if (value == nullptr || value->empty())
            return false;
        (*value)[0] = (*value)[0] == '0' ? '1' : '0';
        return true;
    }

    // Mutations of protocol values must name one exact, schema-derived object.
    // A broad substring can silently select a different layer/head or a
    // similarly named helper.  Count every canonical wire occurrence and fail
    // closed unless the selector resolves to exactly one value-bearing entry.
    template <typename Selector>
    bool mutate_unique_canonical_value(Proof &candidate, Selector selector)
    {
        const auto canonical_label = selector(static_cast<const Proof &>(candidate));
        if (canonical_label.empty())
            return false;

        std::vector<FieldElement *> matches;
        for (auto &subproof : candidate.pi_bind.subproofs)
        {
            for (auto &evaluation : subproof.evaluations)
            {
                if (evaluation.object_id == canonical_label
                    && !evaluation.values.empty())
                    matches.push_back(&evaluation.values.front());
            }
        }
        for (auto &bundle_entry : candidate.domain_openings)
        {
            for (auto &value : bundle_entry.second.values)
            {
                if (value.first == canonical_label && !value.second.empty())
                    matches.push_back(&value.second.front());
            }
        }
        if (matches.size() != 1)
            return false;
        *matches.front() += FieldElement::one();
        return true;
    }

    template <typename Selector>
    std::string first_aggregation_relation_label(
        const Proof &proof,
        Selector selector)
    {
        for (const auto &subproof : proof.pi_bind.subproofs)
        {
            for (const auto &relation : subproof.relation_meta.aggregation_relations)
            {
                const auto label = selector(relation);
                if (!label.empty())
                    return label;
            }
        }
        return {};
    }

    template <typename Selector>
    std::string first_aggregation_route_label(
        const Proof &proof,
        Selector selector)
    {
        for (const auto &subproof : proof.pi_bind.subproofs)
        {
            for (const auto &relation : subproof.relation_meta.aggregation_relations)
            {
                for (const auto *route : {
                         &relation.source_route,
                         &relation.target_route})
                {
                    const auto label = selector(*route);
                    if (!label.empty())
                        return label;
                }
            }
        }
        return {};
    }

    std::string first_division_value_label(const Proof &proof)
    {
        for (const auto &subproof : proof.pi_bind.subproofs)
        {
            const auto &meta = subproof.relation_meta;
            if (meta.object_kind == "hidden"
                && meta.layer_id.has_value()
                && meta.head_id.has_value())
                return gatzk::protocol::hidden_head_prefix(
                           *meta.layer_id, *meta.head_id)
                    + "_Div_X";
            if (meta.object_kind == "output" && meta.head_id.has_value())
                return gatzk::protocol::output_head_prefix(*meta.head_id)
                    + "_Div_X";
        }
        return {};
    }

    bool mutate_any_serialized_value(Proof &candidate)
    {
        if (!candidate.external_evaluations.empty())
        {
            candidate.external_evaluations.front().second += FieldElement::one();
            return true;
        }
        for (auto &subproof : candidate.pi_bind.subproofs)
        {
            for (auto &evaluation : subproof.evaluations)
            {
                if (!evaluation.values.empty())
                {
                    evaluation.values.front() += FieldElement::one();
                    return true;
                }
            }
        }
        for (auto &bundle_entry : candidate.domain_openings)
        {
            for (auto &value : bundle_entry.second.values)
            {
                if (!value.second.empty())
                {
                    value.second.front() += FieldElement::one();
                    return true;
                }
            }
        }
        return false;
    }

    bool mutate_private_opening_value(Proof &candidate)
    {
        for (auto &subproof : candidate.pi_bind.subproofs)
        {
            for (auto &evaluation : subproof.evaluations)
            {
                if (evaluation.kind == PayloadValueKind::MaskedPrivateValue
                    && !evaluation.values.empty())
                {
                    evaluation.values.front() += FieldElement::one();
                    return true;
                }
            }
        }
        for (auto &bundle_entry : candidate.domain_openings)
        {
            auto &bundle = bundle_entry.second;
            if (bundle.value_kinds.size() != bundle.values.size())
                return false;
            for (std::size_t index = 0; index < bundle.values.size(); ++index)
            {
                if (bundle.value_kinds[index] == PayloadValueKind::MaskedPrivateValue
                    && !bundle.values[index].second.empty())
                {
                    bundle.values[index].second.front() += FieldElement::one();
                    return true;
                }
            }
        }
        // A parsed proof must not contain prover-only opening polynomials.  In
        // particular, assigning one here would not change canonical proof bytes
        // and would make this case a synthetic no-op, so missing masked values
        // fail closed instead.
        return false;
    }

    using ProofMutator = std::function<bool(Proof &)>;
    using ContextMutator = std::function<bool(VerifierContext &)>;

    class TamperRunner
    {
    public:
        TamperRunner(
            const VerifierContext &context,
            const Proof &proof,
            const std::vector<std::uint8_t> &proof_bytes,
            bool security_budget_passed)
            : context_(context), proof_(proof), proof_bytes_(proof_bytes),
              security_budget_passed_(security_budget_passed)
        {
        }

        void run_reference_mismatch_case()
        {
            std::string status = "fail";
            try
            {
                // The point-only verifier context has no independent reference
                // artifact/API to compare against.  Keep the required case name
                // and exercise the actual final acceptance gate with a
                // deliberately false reference-consistency result.  This is a
                // fail-closed acceptance test, not a proof-mutation shortcut.
                const auto parsed = gatzk::protocol::parse_proof(
                    context_, proof_bytes_);
                const bool verified = gatzk::protocol::verify(
                    context_, parsed, nullptr);
                const bool accepted = gatzk::protocol::final_formal_run_accepted(
                    verified, false, security_budget_passed_);
                status = accepted ? "fail" : "reject";
            }
            catch (const std::exception &)
            {
                status = "reject";
            }
            catch (...)
            {
                status = "reject";
            }
            results_.emplace("reference_mismatch", status);
        }

        void run_proof_case(
            std::string_view name,
            const ProofMutator &mutator)
        {
            std::string status = "fail";
            try
            {
                auto candidate = proof_;
                if (!mutator(candidate))
                {
                    results_.emplace(std::string(name), status);
                    return;
                }
                std::vector<std::uint8_t> bytes;
                try
                {
                    bytes = gatzk::protocol::serialize_proof(candidate);
                }
                catch (const std::exception &)
                {
                    status = "serialization_rejected";
                    results_.emplace(std::string(name), status);
                    return;
                }
                if (bytes == proof_bytes_)
                {
                    // The mutator touched only an execution-only field.
                    // Such a case is not a real wire/proof mutation.
                    results_.emplace(std::string(name), status);
                    return;
                }
                try
                {
                    const auto parsed = gatzk::protocol::parse_proof(context_, bytes);
                    if (gatzk::protocol::serialize_proof(parsed) != bytes)
                        throw std::runtime_error("tampered proof was not canonical after parsing");
                    status = gatzk::protocol::verify(context_, parsed, nullptr)
                                 ? "fail"
                                 : "reject";
                }
                catch (const std::exception &)
                {
                    status = "reject";
                }
                catch (...)
                {
                    status = "reject";
                }
            }
            catch (const std::exception &)
            {
                status = "fail";
            }
            catch (...)
            {
                status = "fail";
            }
            results_.emplace(std::string(name), status);
        }

        void run_raw_case(
            std::string_view name,
            const std::function<bool(std::vector<std::uint8_t> &)> &mutator)
        {
            std::string status = "fail";
            try
            {
                auto bytes = proof_bytes_;
                if (!mutator(bytes) || bytes == proof_bytes_)
                {
                    results_.emplace(std::string(name), status);
                    return;
                }
                try
                {
                    const auto parsed = gatzk::protocol::parse_proof(context_, bytes);
                    if (gatzk::protocol::serialize_proof(parsed) != bytes)
                        throw std::runtime_error("tampered proof was not canonical after parsing");
                    status = gatzk::protocol::verify(context_, parsed, nullptr)
                                 ? "fail"
                                 : "reject";
                }
                catch (const std::exception &)
                {
                    status = "reject";
                }
                catch (...)
                {
                    status = "reject";
                }
            }
            catch (const std::exception &)
            {
                status = "fail";
            }
            catch (...)
            {
                status = "fail";
            }
            results_.emplace(std::string(name), status);
        }

        void run_context_case(
            std::string_view name,
            const ContextMutator &mutator)
        {
            std::string status = "fail";
            try
            {
                auto candidate_context = context_;
                if (!mutator(candidate_context))
                {
                    results_.emplace(std::string(name), status);
                    return;
                }
                try
                {
                    // Context mutations are setup/acceptance mutations rather
                    // than proof-wire mutations.  The independent context
                    // validator is the formal fail-closed boundary before the
                    // canonical parser/verifier is allowed to consume it.
                    if (!gatzk::protocol::validate_verifier_context(candidate_context))
                    {
                        status = "reject";
                    }
                    else
                    {
                        const auto parsed =
                            gatzk::protocol::parse_proof(candidate_context, proof_bytes_);
                        status = gatzk::protocol::verify(
                                     candidate_context, parsed, nullptr)
                                     ? "fail"
                                     : "reject";
                    }
                }
                catch (const std::exception &)
                {
                    status = "reject";
                }
                catch (...)
                {
                    status = "reject";
                }
            }
            catch (const std::exception &)
            {
                status = "fail";
            }
            catch (...)
            {
                status = "fail";
            }
            results_.emplace(std::string(name), status);
        }

        const std::map<std::string, std::string> &results() const
        {
            return results_;
        }

    private:
        const VerifierContext &context_;
        const Proof &proof_;
        const std::vector<std::uint8_t> &proof_bytes_;
        bool security_budget_passed_ = false;
        std::map<std::string, std::string> results_;
    };

    std::string canonical_tamper_map(
        const std::map<std::string, std::string> &results,
        bool *passed)
    {
        bool all_rejected = true;
        std::ostringstream output;
        std::set<std::string> expected;
        for (std::size_t index = 0; index < kRequiredTamperCases.size(); ++index)
        {
            const auto name = std::string(kRequiredTamperCases[index]);
            if (!expected.insert(name).second)
                throw std::runtime_error("tamper case list contains a duplicate name");
            const auto iterator = results.find(name);
            if (iterator == results.end())
                throw std::runtime_error("tamper case did not produce a result: " + name);
            if (index != 0)
                output << ',';
            output << name << '=' << iterator->second;
            all_rejected = all_rejected
                && (iterator->second == "reject"
                    || iterator->second == "serialization_rejected");
        }
        if (results.size() != expected.size())
            throw std::runtime_error("tamper suite produced an unknown or duplicate case");
        if (passed != nullptr)
            *passed = all_rejected;
        return output.str();
    }

    bool path_has_component(
        const std::filesystem::path &path,
        std::string_view component)
    {
        for (const auto &part : path)
        {
            if (part == component)
                return true;
        }
        return false;
    }

    std::filesystem::path canonical_repo_root()
    {
        std::error_code error;
        auto cursor = std::filesystem::canonical(
            std::filesystem::current_path(), error);
        if (error)
            throw std::runtime_error("cannot resolve the repository working directory");
        while (true)
        {
            const auto cmake_status = std::filesystem::symlink_status(
                cursor / "CMakeLists.txt", error);
            if (error && error != std::make_error_code(
                              std::errc::no_such_file_or_directory))
                throw std::runtime_error("cannot inspect repository root candidates");
            error.clear();
            const auto tests_status = std::filesystem::symlink_status(
                cursor / "tests", error);
            if (error && error != std::make_error_code(
                              std::errc::no_such_file_or_directory))
                throw std::runtime_error("cannot inspect repository test directory");
            error.clear();
            if (std::filesystem::is_regular_file(cmake_status)
                && std::filesystem::is_directory(tests_status))
                return cursor;

            const auto parent = cursor.parent_path();
            if (parent == cursor)
                break;
            cursor = parent;
        }
        throw std::runtime_error(
            "run gatzk_protocol_tamper_tests from the repository or its build tree");
    }

    void reject_output_symlink_components(
        const std::filesystem::path &tests_root,
        const std::filesystem::path &relative_output)
    {
        std::filesystem::path cursor = tests_root;
        for (auto iterator = relative_output.begin();
             iterator != relative_output.end();
             ++iterator)
        {
            cursor /= *iterator;
            std::error_code error;
            const auto status = std::filesystem::symlink_status(cursor, error);
            if (error && error != std::make_error_code(
                              std::errc::no_such_file_or_directory))
                throw std::runtime_error(
                    "cannot inspect a tamper attestation output path component: "
                    + cursor.string());
            if (!error && std::filesystem::is_symlink(status))
                throw std::runtime_error(
                    "tamper attestation output cannot traverse a symlink: "
                    + cursor.string());

            const bool final_component =
                std::next(iterator) == relative_output.end();
            if (final_component)
            {
                if (!error && !std::filesystem::is_regular_file(status))
                    throw std::runtime_error(
                        "tamper attestation output must be a regular file: "
                        + cursor.string());
            }
            else if (!error && !std::filesystem::is_directory(status))
            {
                throw std::runtime_error(
                    "tamper attestation output parent is not a directory: "
                    + cursor.string());
            }
        }
    }

    std::filesystem::path validate_output_path(
        const std::filesystem::path &output)
    {
        const auto root = canonical_repo_root();
        std::error_code error;
        const auto tests_status = std::filesystem::symlink_status(
            root / "tests", error);
        if (error || std::filesystem::is_symlink(tests_status)
            || !std::filesystem::is_directory(tests_status))
            throw std::runtime_error(
                "canonical repository tests directory is unavailable or unsafe");
        const auto tests_root = std::filesystem::canonical(root / "tests", error);
        if (error)
            throw std::runtime_error("cannot resolve the canonical repository tests directory");

        const auto normalized = (output.is_absolute()
                                     ? output
                                     : root / output)
                                    .lexically_normal();
        if (path_has_component(normalized, "results")
            || path_has_component(normalized, "paper_secure")
            || path_has_component(normalized, "artifacts")
            || path_has_component(normalized, "runs"))
            throw std::runtime_error(
                "tamper attestation output is forbidden in results, artifacts, or runs");
        const auto relative = normalized.lexically_relative(tests_root);
        if (relative.empty() || relative == "." || relative.is_absolute())
            throw std::runtime_error(
                "tamper attestation output must be inside the repository tests directory");
        for (const auto &part : relative)
        {
            if (part == ".." || part == ".")
                throw std::runtime_error(
                    "tamper attestation output must be inside the repository tests directory");
        }

        // Inspect before creation so an existing symlink cannot redirect the
        // output.  Then create only the validated parent and inspect again to
        // close the check-to-use gap for newly-created components.
        reject_output_symlink_components(tests_root, relative);
        const auto parent = normalized.parent_path();
        if (!parent.empty())
        {
            std::filesystem::create_directories(parent, error);
            if (error)
                throw std::runtime_error(
                    "failed to create the tamper attestation tests directory: "
                    + parent.string());
        }
        reject_output_symlink_components(tests_root, relative);
        return normalized;
    }

    void validate_existing_run_binding(
        const Options &options,
        const std::map<std::string, std::string> &manifest,
        const VerifierContext &context,
        const Proof &proof,
        const std::vector<std::uint8_t> &proof_bytes)
    {
        for (const auto key : {
                 "proof_hash",
                 "config_sha256",
                 "source_tree_hash",
                 "source_manifest_hash",
                 "build_receipt_hash",
                 "binary_hash",
                 "build_identity",
                 "srs_manifest_hash",
                 "protocol_schema_id",
                 "instance_schema_hash",
                 "public_input_digest",
                 "topology_digest"})
            require_manifest_hash(manifest, key);

        const auto proof_hash = manifest_value(manifest, "proof_hash");
        if (sha256_bytes(proof_bytes) != proof_hash)
            throw std::runtime_error("run manifest proof_hash does not match proof.bin");
        if (manifest_value(manifest, "trusted_registry_digest")
                != options.trusted_registry_digest
            || manifest_value(manifest, "verifier_registry_digest")
                   != options.trusted_registry_digest)
            throw std::runtime_error(
                "run manifest registry digests do not match the explicit trusted registry root");
        if (manifest_value(manifest, "verifier_registry_id")
                != context.registry.registry_id)
            throw std::runtime_error("run manifest verifier registry identity mismatch");
        if (manifest_value(manifest, "srs_manifest_hash") != context.srs.manifest_hash)
            throw std::runtime_error("run manifest SRS manifest identity mismatch");
        if (manifest_value(manifest, "protocol_schema_id")
                != proof.public_metadata.protocol_schema_id
            || manifest_value(manifest, "protocol_schema_id")
                   != context.registry.protocol_schema_id)
            throw std::runtime_error("run manifest protocol schema identity mismatch");
        if (manifest_value(manifest, "instance_schema_hash")
                != proof.public_metadata.instance_schema_hash
            || manifest_value(manifest, "instance_schema_hash")
                   != context.registry.instance_schema_hash)
            throw std::runtime_error("run manifest instance schema identity mismatch");
        if (manifest_value(manifest, "public_input_digest")
                != context.registry.public_input_digest
            || manifest_value(manifest, "topology_digest")
                   != context.registry.topology_digest)
            throw std::runtime_error("run manifest public-input/topology identity mismatch");
        if (manifest_value(manifest, "status") != "VERIFY_OK"
            || manifest_value(manifest, "verifier_accepted") != "true"
            || manifest_value(manifest, "final_formal_run_accepted") != "true"
            || manifest_value(manifest, "verifier_reads_witness") != "false"
            || manifest_size(manifest, "private_plain_value_count") != 0)
            throw std::runtime_error(
                "run manifest does not describe an accepted zero-knowledge formal run");
        const auto output_digest = gatzk::protocol::canonical_output_digest(
            proof.public_logits,
            context.local.num_nodes,
            context.local.num_classes);
        if (manifest_value(manifest, "proof_public_output_hash") != output_digest)
            throw std::runtime_error("run manifest public output digest mismatch");
        if (!gatzk::protocol::validate_verifier_context(context))
            throw std::runtime_error("independent verifier context validation failed");
        if (!gatzk::protocol::verify(context, proof, nullptr))
            throw std::runtime_error("baseline proof is not accepted by the independent verifier");
        if (gatzk::protocol::private_plain_value_count(context, proof) != 0)
            throw std::runtime_error("baseline proof contains private plaintext material");
    }

    std::map<std::string, std::string> build_attestation(
        const Options &options,
        const std::map<std::string, std::string> &manifest,
        const VerifierContext &context,
        const Proof &proof,
        const std::vector<std::uint8_t> &proof_bytes,
        std::string tamper_map,
        bool tamper_passed,
        double tamper_ms)
    {
        const auto proof_hash = sha256_bytes(proof_bytes);
        const auto public_output_digest = gatzk::protocol::canonical_output_digest(
            proof.public_logits,
            context.local.num_nodes,
            context.local.num_classes);
        std::ostringstream tamper_time;
        tamper_time << std::fixed << std::setprecision(3) << tamper_ms;
        return {
            {"binary_sha256", manifest_value(manifest, "binary_hash")},
            {"build_identity", manifest_value(manifest, "build_identity")},
            {"build_receipt_sha256", manifest_value(manifest, "build_receipt_hash")},
            {"config_sha256", manifest_value(manifest, "config_sha256")},
            {"format", "zkgat_protocol_tamper_attestation"},
            {"instance_schema_hash", proof.public_metadata.instance_schema_hash},
            {"proof_sha256", proof_hash},
            {"protocol_schema_id", proof.public_metadata.protocol_schema_id},
            {"public_input_digest", context.registry.public_input_digest},
            {"source_manifest_sha256", manifest_value(manifest, "source_manifest_hash")},
            {"source_tree_hash", manifest_value(manifest, "source_tree_hash")},
            {"srs_manifest_sha256", context.srs.manifest_hash},
            {"suite_id", "zkgat_protocol_tamper_39"},
            {"tamper_test_case_map", std::move(tamper_map)},
            {"tamper_tests_ms", tamper_time.str()},
            {"tamper_tests_passed", tamper_passed ? "true" : "false"},
            {"topology_digest", context.registry.topology_digest},
            {"trusted_registry_digest", options.trusted_registry_digest},
            {"verifier_registry_digest", context.authenticated_registry_digest},
            {"public_output_digest", public_output_digest},
        };
    }

    void run_suite(
        TamperRunner &runner,
        const VerifierContext &context)
    {
        runner.run_raw_case("codec_tag", [](auto &bytes)
                            {
                                if (bytes.empty()) return false;
                                bytes.front() ^= 0x01U;
                                return true;
                            });
        runner.run_raw_case("codec_length", [](auto &bytes)
                            {
                                if (bytes.size() < 2) return false;
                                bytes.pop_back();
                                return true;
                            });
        runner.run_raw_case("codec_trailing", [](auto &bytes)
                            {
                                bytes.push_back(0xa5U);
                                return true;
                            });
        runner.run_proof_case("public_output", [](auto &candidate)
                              {
                                  if (candidate.public_logits.empty()) return false;
                                  candidate.public_logits.front() += FieldElement::one();
                                  return true;
                              });
        runner.run_proof_case("metadata", [](auto &candidate)
                              {
                                  candidate.public_metadata.privacy_policy =
                                      "tampered_privacy_policy";
                                  return true;
                              });
        runner.run_proof_case("schema", [](auto &candidate)
                              {
                                  if (candidate.block_order.empty()) return false;
                                  candidate.block_order.front() = "tampered-block";
                                  return true;
                              });
        runner.run_proof_case("ids", [](auto &candidate)
                              {
                                  candidate.public_metadata.protocol_schema_id =
                                      "tampered-protocol-schema";
                                  return true;
                              });
        runner.run_proof_case("dynamic_commitment", [&context](auto &candidate)
                              {
                                  if (candidate.dynamic_commitments.empty()) return false;
                                  candidate.dynamic_commitments.front().second.point =
                                      gatzk::crypto::g1_add(
                                          candidate.dynamic_commitments.front().second.point,
                                          context.kzg_vk.g1_generator);
                                  return true;
                              });
        runner.run_proof_case("quotient_commitment", [&context](auto &candidate)
                              {
                                  if (candidate.quotient_commitments.empty()) return false;
                                  candidate.quotient_commitments.front().second.point =
                                      gatzk::crypto::g1_add(
                                          candidate.quotient_commitments.front().second.point,
                                          context.kzg_vk.g1_generator);
                                  return true;
                              });
        runner.run_proof_case("opening", [&context](auto &candidate)
                              {
                                  if (candidate.opening_groups.empty()) return false;
                                  candidate.opening_groups.front().witness =
                                      gatzk::crypto::g1_add(
                                          candidate.opening_groups.front().witness,
                                          context.kzg_vk.g1_generator);
                                  return true;
                              });
        runner.run_proof_case("reblind_commitment", [&context](auto &candidate)
                              {
                                  for (auto &subproof : candidate.pi_bind.subproofs)
                                  {
                                      if (!subproof.model_reblinds.empty())
                                      {
                                          auto &commitment =
                                              subproof.model_reblinds.front().blinder_commitment;
                                          commitment.point = gatzk::crypto::g1_add(
                                              commitment.point,
                                              context.kzg_vk.g1_generator);
                                          return true;
                                      }
                                  }
                                  return false;
                              });
        runner.run_context_case("static_registry", [](auto &candidate)
                                {
                                    if (candidate.registry.static_entries.empty()) return false;
                                    candidate.registry.static_entries.front().point =
                                        gatzk::crypto::g1_add(
                                            candidate.registry.static_entries.front().point,
                                            candidate.kzg_vk.g1_generator);
                                    return true;
                                });
        runner.run_context_case("model_registry", [](auto &candidate)
                                {
                                    if (candidate.registry.model_entries.empty()) return false;
                                    candidate.registry.model_entries.front().point =
                                        gatzk::crypto::g1_add(
                                            candidate.registry.model_entries.front().point,
                                            candidate.kzg_vk.g1_generator);
                                    return true;
                                });
        runner.run_reference_mismatch_case();
        runner.run_proof_case("private_opening", [](auto &candidate)
                              { return mutate_private_opening_value(candidate); });
        runner.run_proof_case("route_padding", [](auto &candidate)
                              {
                                  return mutate_unique_canonical_value(
                                      candidate,
                                      [](const Proof &)
                                      { return std::string("P_Q_div_out0"); });
                              });
        runner.run_proof_case("route_query", [](auto &candidate)
                              {
                                  return mutate_unique_canonical_value(
                                      candidate,
                                      [](const Proof &proof)
                                      {
                                          return first_aggregation_route_label(
                                              proof,
                                              [](const auto &route)
                                              { return route.query_object; });
                                      });
                              });
        runner.run_proof_case("lookup_inverse", [](auto &candidate)
                              {
                                  return mutate_unique_canonical_value(
                                      candidate,
                                      [](const Proof &proof)
                                      {
                                          return first_aggregation_route_label(
                                              proof,
                                              [](const auto &route)
                                              { return route.edge_inverse_object; });
                                      });
                              });
        runner.run_proof_case("lookup_multiplicity", [](auto &candidate)
                              {
                                  return mutate_unique_canonical_value(
                                      candidate,
                                      [](const Proof &proof)
                                      {
                                          return first_aggregation_route_label(
                                              proof,
                                              [](const auto &route)
                                              { return route.multiplicity_object; });
                                      });
                              });
        runner.run_proof_case("psq", [](auto &candidate)
                              {
                                  return mutate_unique_canonical_value(
                                      candidate,
                                      [](const Proof &proof)
                                      {
                                          return first_aggregation_relation_label(
                                              proof,
                                              [](const auto &relation)
                                              { return relation.psq_object; });
                                      });
                              });
        runner.run_proof_case("shift", [](auto &candidate)
                              {
                                  for (auto &subproof : candidate.pi_bind.subproofs)
                                  {
                                      for (auto &relation : subproof.relation_meta.shift_relations)
                                      {
                                          if (!relation.deltas.empty())
                                          {
                                              relation.deltas.front() += FieldElement::one();
                                              return true;
                                          }
                                      }
                                  }
                                  return false;
                              });
        runner.run_proof_case("logup", [&context](auto &candidate)
                              {
                                  for (auto &subproof : candidate.pi_bind.subproofs)
                                  {
                                      if (!subproof.view_logup_commitments.empty())
                                      {
                                          subproof.view_logup_commitments.front().second.point =
                                              gatzk::crypto::g1_add(
                                                  subproof.view_logup_commitments.front().second.point,
                                                  context.kzg_vk.g1_generator);
                                          return true;
                                      }
                                  }
                                  return false;
                              });
        runner.run_proof_case("aggregate", [](auto &candidate)
                              {
                                  return mutate_unique_canonical_value(
                                      candidate,
                                      [](const Proof &proof)
                                      {
                                          return first_aggregation_relation_label(
                                              proof,
                                              [](const auto &relation)
                                              { return relation.product_object; });
                                      });
                              });
        runner.run_proof_case("average", [](auto &candidate)
                              {
                                  return mutate_unique_canonical_value(
                                      candidate,
                                      [](const Proof &)
                                      { return std::string("P_avg_Div_Y"); });
                              });
        runner.run_proof_case("division", [](auto &candidate)
                              {
                                  return mutate_unique_canonical_value(
                                      candidate,
                                      [](const Proof &proof)
                                      { return first_division_value_label(proof); });
                              });
        runner.run_proof_case("transcript_challenge", [](auto &candidate)
                              {
                                  candidate.public_metadata.proof_nonce +=
                                      "tampered-transcript-challenge";
                                  return true;
                              });
        runner.run_proof_case("transcript_value", [](auto &candidate)
                              { return mutate_any_serialized_value(candidate); });
        runner.run_context_case("srs_metadata", [](auto &candidate)
                                {
                                    candidate.srs.srs_type = "untrusted_srs";
                                    return true;
                                });
        runner.run_context_case("srs_consistency", [](auto &candidate)
                                {
                                    return flip_hex_digit(&candidate.srs.srs_consistency_hash);
                                });
        runner.run_context_case("srs_resource_identity", [](auto &candidate)
                                {
                                    return flip_hex_digit(&candidate.registry.srs_manifest_hash);
                                });
        runner.run_context_case("srs_resource_compromised", [](auto &candidate)
                                {
                                    candidate.kzg_vk.g2_tau = gatzk::crypto::g2_add(
                                        candidate.kzg_vk.g2_tau,
                                        candidate.kzg_vk.g2_one);
                                    return true;
                                });
        runner.run_proof_case("proof_scope", [](auto &candidate)
                              {
                                  candidate.public_metadata.output_mode = "labels_only";
                                  return true;
                              });
        runner.run_context_case("observation_ledger", [](auto &candidate)
                                {
                                    if (candidate.compiler_artifact == nullptr
                                        || candidate.compiler_artifact->observation_ledger.empty())
                                        return false;
                                    auto artifact = *candidate.compiler_artifact;
                                    artifact.observation_ledger.front().expression_or_certificate +=
                                        "|tamper-observation-ledger";
                                    candidate.compiler_artifact =
                                        std::make_shared<const gatzk::protocol::ProtocolCompilerArtifact>(
                                            std::move(artifact));
                                    return true;
                                });
        runner.run_context_case("random_identity_ledger", [](auto &candidate)
                                {
                                    if (candidate.compiler_artifact == nullptr
                                        || candidate.compiler_artifact->random_identity_ledger.empty())
                                        return false;
                                    auto artifact = *candidate.compiler_artifact;
                                    auto &entry = artifact.random_identity_ledger.front();
                                    entry.degree_bound = entry.degree_bound == 0
                                                              ? 1
                                                              : entry.degree_bound - 1;
                                    candidate.compiler_artifact =
                                        std::make_shared<const gatzk::protocol::ProtocolCompilerArtifact>(
                                            std::move(artifact));
                                    return true;
                                });
        runner.run_context_case("integer_bound_ledger", [](auto &candidate)
                                {
                                    if (candidate.registry.integer_bound_ledger.empty()) return false;
                                    auto &entry = candidate.registry.integer_bound_ledger.front();
                                    if (entry.absolute_bound != std::numeric_limits<std::uint64_t>::max())
                                        ++entry.absolute_bound;
                                    else
                                        --entry.absolute_bound;
                                    return true;
                                });
        runner.run_context_case("integer_bound_certificate", [](auto &candidate)
                                {
                                    if (candidate.registry.model_entries.empty()) return false;
                                    auto &certificate =
                                        candidate.registry.model_entries.front().bound_certificate;
                                    if (!certificate.present) return false;
                                    if (certificate.absolute_bound !=
                                        std::numeric_limits<std::uint64_t>::max())
                                        ++certificate.absolute_bound;
                                    else
                                        --certificate.absolute_bound;
                                    return true;
                                });
        runner.run_context_case("security_budget_ledger", [](auto &candidate)
                                {
                                    if (candidate.compiler_artifact == nullptr
                                        || candidate.compiler_artifact->security_budget_ledger.empty())
                                        return false;
                                    auto artifact = *candidate.compiler_artifact;
                                    artifact.security_budget_ledger.front().expression_or_certificate +=
                                        "|tamper-security-budget-ledger";
                                    candidate.compiler_artifact =
                                        std::make_shared<const gatzk::protocol::ProtocolCompilerArtifact>(
                                            std::move(artifact));
                                    return true;
                                });
        runner.run_context_case("security_budget_exact", [](auto &candidate)
                                {
                                    if (candidate.registry.security_budget_exact.empty()) return false;
                                    candidate.registry.security_budget_exact += "/0";
                                    return true;
                                });
        runner.run_context_case("public_input_digest", [](auto &candidate)
                                {
                                    return flip_hex_digit(&candidate.registry.public_input_digest);
                                });
    }

    Options parse_options(int argc, char **argv)
    {
        Options options;
        for (int index = 1; index < argc; ++index)
        {
            const std::string option = argv[index];
            if (option == "--proof")
                options.proof_path = require_option_value(index, argc, argv, option);
            else if (option == "--registry")
                options.registry_path = require_option_value(index, argc, argv, option);
            else if (option == "--trusted-registry-digest")
                options.trusted_registry_digest =
                    require_option_value(index, argc, argv, option);
            else if (option == "--run-manifest")
                options.run_manifest_path = require_option_value(index, argc, argv, option);
            else if (option == "--output")
                options.output_path = require_option_value(index, argc, argv, option);
            else
                throw std::runtime_error("unknown option: " + option);
        }
        if (options.proof_path.empty()
            || options.registry_path.empty()
            || options.run_manifest_path.empty()
            || options.output_path.empty()
            || !is_lower_hex_256(options.trusted_registry_digest))
            throw std::runtime_error(
                "usage: gatzk_protocol_tamper_tests --proof <proof.bin> "
                "--registry <point-only-registry> "
                "--trusted-registry-digest <lowercase-sha256> "
                "--run-manifest <run_manifest.json> --output <tests-attestation.json>");
        return options;
    }
}

int main(int argc, char **argv)
{
    try
    {
        auto options = parse_options(argc, argv);
        options.output_path = validate_output_path(options.output_path);
        const auto proof_bytes = read_bytes(options.proof_path);
        const auto manifest = FlatJsonStringObjectReader(
                                  read_text(options.run_manifest_path))
                                  .parse();
        const auto context =
            gatzk::protocol::load_verifier_context_from_registry(
                options.registry_path.string(),
                options.trusted_registry_digest);
        const auto proof = gatzk::protocol::parse_proof(context, proof_bytes);
        if (gatzk::protocol::serialize_proof(proof) != proof_bytes)
            throw std::runtime_error("proof.bin is not canonical after parse/serialize");
        validate_existing_run_binding(
            options, manifest, context, proof, proof_bytes);

        const bool security_budget_passed =
            context.compiler_artifact != nullptr
            && gatzk::protocol::evaluate_security_budget(
                   gatzk::protocol::detail::make_verifier_execution_context(context),
                   *context.compiler_artifact)
                   .passed;
        TamperRunner runner(
            context, proof, proof_bytes, security_budget_passed);
        const auto tamper_start = std::chrono::steady_clock::now();
        run_suite(runner, context);
        const auto tamper_end = std::chrono::steady_clock::now();
        const auto tamper_ms = std::chrono::duration<double, std::milli>(
                                   tamper_end - tamper_start)
                                   .count();
        bool tamper_passed = false;
        const auto tamper_map = canonical_tamper_map(
            runner.results(), &tamper_passed);
        if (!tamper_passed)
            throw std::runtime_error(
                "one or more canonical tamper cases was not rejected: "
                + tamper_map);

        const auto attestation = build_attestation(
            options,
            manifest,
            context,
            proof,
            proof_bytes,
            tamper_map,
            tamper_passed,
            tamper_ms);
        gatzk::util::write_json_object(
            options.output_path.string(), attestation);
        std::cout << "TAMPER_TESTS_OK cases=" << kRequiredTamperCases.size()
                  << " attestation=" << options.output_path.string()
                  << " tamper_tests_ms=" << std::fixed << std::setprecision(3)
                  << tamper_ms << '\n';
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
