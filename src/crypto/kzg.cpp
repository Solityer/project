#include "gatzk/crypto/kzg.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <deque>

#include <openssl/sha.h>
#include <openssl/rand.h>
#include <condition_variable>

#include "gatzk/algebra/domain_evaluation.hpp"
#include "gatzk/algebra/field_io.hpp"
#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/crypto/compiler_artifact.hpp"
#include "gatzk/crypto/cuda_msm.hpp"
#include "gatzk/protocol/schema.hpp"
#include "gatzk/util/logging.hpp"
#include "gatzk/util/route2.hpp"
namespace gatzk::crypto
{
    // Forward declarations for the cache identity planner, which lives in the
    // first implementation namespace while domain selection is defined near
    // the public key-loading entry point below.
    std::vector<std::string> select_lagrange_srs_domains(
        const SRSManifest &manifest,
        const std::vector<std::pair<std::string, std::size_t>> &required_lagrange_domains);

    namespace
    {
        using algebra::FieldElement;
        using Clock = std::chrono::steady_clock;

        std::string bytes_to_hex(const unsigned char *bytes, std::size_t size)
        {
            std::ostringstream out;
            out << std::hex << std::setfill('0');
            for (std::size_t i = 0; i < size; ++i)
            {
                out << std::setw(2) << static_cast<unsigned int>(bytes[i]);
            }
            return out.str();
        }

        std::string sha256_hex(const std::string &data)
        {
            unsigned char digest[SHA256_DIGEST_LENGTH];
            SHA256(reinterpret_cast<const unsigned char *>(data.data()), data.size(), digest);
            return bytes_to_hex(digest, sizeof(digest));
        }

        constexpr std::array<std::string_view, 41>
            kCanonicalSrsManifestFields = {
                "manifest_format",
                "srs_id",
                "curve_id",
                "field_modulus_id",
                "srs_type",
                "ceremony_transcript_hash",
                "setup_transcript_hash",
                "degree_bound",
                "domain_sizes",
                "domain_generators",
                "root_derivation_id",
                "row_order",
                "protocol_compiler_artifact_path",
                "protocol_compiler_artifact_hash",
                "protocol_schema_id",
                "max_instance_envelope_id",
                "max_instance_envelope_hash",
                "max_instance_envelope",
                "proof_schema_template_hash",
                "observation_ledger_hash",
                "random_identity_ledger_hash",
                "security_budget_ledger_hash",
                "pk_coef_path",
                "pk_coef_hash",
                "pk_lagrange_paths",
                "pk_lagrange_hashes",
                "vk_path",
                "vk_hash",
                "vanishing_g2_hash",
                "srs_audit_report_path",
                "srs_audit_report_hash",
                "srs_consistency_hash",
                "encoding",
                "generator_id",
                "security_level",
                "kzg_assumption_profile_id",
                "kzg_assumption_degree_bound",
                "kzg_assumption_security_bits",
                "kzg_assumption_domain_catalog_hash",
                "domain_generator_catalog_hash",
                "manifest_hash",
        };

        constexpr std::array<std::string_view, 16>
            kCanonicalSrsEnvelopeFields = {
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

        enum class SrsJsonKind
        {
            Null,
            Boolean,
            Number,
            String,
            Array,
            Object,
        };

        struct SrsJsonValue
        {
            SrsJsonKind kind = SrsJsonKind::Null;
            std::string scalar;
            std::vector<SrsJsonValue> array;
            std::vector<std::pair<std::string, SrsJsonValue>> object;
            // The parser records the raw key position so the manifest hash can
            // retain the writer's exact prefix digest without searching text.
            std::vector<std::size_t> object_key_positions;
        };

        class SrsJsonParser
        {
        public:
            explicit SrsJsonParser(const std::string &text) : text_(text) {}

            SrsJsonValue parse()
            {
                skip_whitespace();
                SrsJsonValue value = parse_value();
                skip_whitespace();
                if (position_ != text_.size())
                    fail("SRS manifest has trailing data");
                return value;
            }

        private:
            [[noreturn]] void fail(const std::string &message) const
            {
                throw std::runtime_error(
                    message + " at byte " + std::to_string(position_));
            }

            void skip_whitespace()
            {
                while (position_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[position_])))
                    ++position_;
            }

            bool consume(char expected)
            {
                skip_whitespace();
                if (position_ < text_.size() && text_[position_] == expected)
                {
                    ++position_;
                    return true;
                }
                return false;
            }

            SrsJsonValue parse_value()
            {
                skip_whitespace();
                if (position_ >= text_.size())
                    fail("SRS manifest value is missing");
                switch (text_[position_])
                {
                case '{':
                    return parse_object();
                case '[':
                    return parse_array();
                case '"':
                {
                    SrsJsonValue value;
                    value.kind = SrsJsonKind::String;
                    value.scalar = parse_string();
                    return value;
                }
                case 't':
                    parse_literal("true");
                    return SrsJsonValue{.kind = SrsJsonKind::Boolean, .scalar = "true"};
                case 'f':
                    parse_literal("false");
                    return SrsJsonValue{.kind = SrsJsonKind::Boolean, .scalar = "false"};
                case 'n':
                    parse_literal("null");
                    return SrsJsonValue{.kind = SrsJsonKind::Null};
                default:
                    if (text_[position_] == '-' || std::isdigit(static_cast<unsigned char>(text_[position_])))
                    {
                        SrsJsonValue value;
                        value.kind = SrsJsonKind::Number;
                        value.scalar = parse_number();
                        return value;
                    }
                    fail("malformed SRS manifest value");
                }
            }

            void parse_literal(std::string_view literal)
            {
                if (text_.compare(position_, literal.size(), literal) != 0)
                    fail("malformed SRS manifest literal");
                position_ += literal.size();
            }

            std::string parse_string()
            {
                skip_whitespace();
                if (position_ >= text_.size() || text_[position_] != '"')
                    fail("SRS manifest string is missing");
                ++position_;
                std::string result;
                while (position_ < text_.size())
                {
                    const unsigned char character =
                        static_cast<unsigned char>(text_[position_++]);
                    if (character == '"')
                        return result;
                    if (character < 0x20)
                        fail("SRS manifest string contains a control byte");
                    if (character != '\\')
                    {
                        result.push_back(static_cast<char>(character));
                        continue;
                    }
                    if (position_ >= text_.size())
                        fail("unterminated SRS manifest string");
                    const char escaped = text_[position_++];
                    switch (escaped)
                    {
                    case '"':
                        result.push_back('"');
                        break;
                    case '\\':
                        result.push_back('\\');
                        break;
                    case '/':
                        result.push_back('/');
                        break;
                    case 'b':
                        result.push_back('\b');
                        break;
                    case 'f':
                        result.push_back('\f');
                        break;
                    case 'n':
                        result.push_back('\n');
                        break;
                    case 'r':
                        result.push_back('\r');
                        break;
                    case 't':
                        result.push_back('\t');
                        break;
                    case 'u':
                    {
                        if (position_ + 4 > text_.size())
                            fail("truncated SRS manifest unicode escape");
                        unsigned value = 0;
                        for (std::size_t index = 0; index < 4; ++index)
                        {
                            const char digit = text_[position_++];
                            value <<= 4U;
                            if (digit >= '0' && digit <= '9')
                                value |= static_cast<unsigned>(digit - '0');
                            else if (digit >= 'a' && digit <= 'f')
                                value |= static_cast<unsigned>(digit - 'a' + 10);
                            else if (digit >= 'A' && digit <= 'F')
                                value |= static_cast<unsigned>(digit - 'A' + 10);
                            else
                                fail("malformed SRS manifest unicode escape");
                        }
                        if (value > 0x7f)
                            fail("SRS manifest must use ASCII canonical strings");
                        result.push_back(static_cast<char>(value));
                        break;
                    }
                    default:
                        fail("malformed SRS manifest escape");
                    }
                }
                fail("unterminated SRS manifest string");
            }

            std::string parse_number()
            {
                const std::size_t begin = position_;
                if (text_[position_] == '-')
                    ++position_;
                if (position_ >= text_.size())
                    fail("malformed SRS manifest number");
                if (text_[position_] == '0')
                {
                    ++position_;
                    if (position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_])))
                        fail("SRS manifest number has a leading zero");
                }
                else if (text_[position_] >= '1' && text_[position_] <= '9')
                {
                    while (position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_])))
                        ++position_;
                }
                else
                    fail("malformed SRS manifest number");
                if (position_ < text_.size() && text_[position_] == '.')
                {
                    ++position_;
                    const auto fraction_begin = position_;
                    while (position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_])))
                        ++position_;
                    if (position_ == fraction_begin)
                        fail("malformed SRS manifest fraction");
                }
                if (position_ < text_.size() && (text_[position_] == 'e' || text_[position_] == 'E'))
                {
                    ++position_;
                    if (position_ < text_.size() && (text_[position_] == '+' || text_[position_] == '-'))
                        ++position_;
                    const auto exponent_begin = position_;
                    while (position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_])))
                        ++position_;
                    if (position_ == exponent_begin)
                        fail("malformed SRS manifest exponent");
                }
                return text_.substr(begin, position_ - begin);
            }

            SrsJsonValue parse_array()
            {
                SrsJsonValue result;
                result.kind = SrsJsonKind::Array;
                ++position_;
                skip_whitespace();
                if (consume(']'))
                    return result;
                while (true)
                {
                    result.array.push_back(parse_value());
                    if (consume(']'))
                        return result;
                    if (!consume(','))
                        fail("malformed SRS manifest array");
                    skip_whitespace();
                    if (position_ < text_.size() && text_[position_] == ']')
                        fail("SRS manifest array has a trailing comma");
                }
            }

            SrsJsonValue parse_object()
            {
                SrsJsonValue result;
                result.kind = SrsJsonKind::Object;
                const bool top_level_object = object_depth_ == 0;
                ++object_depth_;
                ++position_;
                skip_whitespace();
                if (consume('}'))
                {
                    --object_depth_;
                    return result;
                }
                std::unordered_set<std::string> seen;
                while (true)
                {
                    skip_whitespace();
                    const auto key_position = position_;
                    const auto key = parse_string();
                    if (!seen.insert(key).second)
                    {
                        if (top_level_object)
                            throw std::runtime_error(
                                "SRS manifest contains duplicate top-level field: " + key);
                        throw std::runtime_error(
                            "SRS manifest contains duplicate nested field: " + key);
                    }
                    if (!consume(':'))
                        fail("SRS manifest object field lacks a value: " + key);
                    result.object.emplace_back(key, parse_value());
                    result.object_key_positions.push_back(key_position);
                    if (consume('}'))
                    {
                        --object_depth_;
                        return result;
                    }
                    if (!consume(','))
                        fail("malformed SRS manifest object");
                    skip_whitespace();
                    if (position_ < text_.size() && text_[position_] == '}')
                        fail("SRS manifest object has a trailing comma");
                }
            }

            const std::string &text_;
            std::size_t position_ = 0;
            std::size_t object_depth_ = 0;
        };

        const SrsJsonValue &require_manifest_field(
            const SrsJsonValue &root,
            std::string_view key)
        {
            if (root.kind != SrsJsonKind::Object)
                throw std::runtime_error("SRS manifest must be a JSON object");
            for (const auto &entry : root.object)
                if (entry.first == key)
                    return entry.second;
            throw std::runtime_error(
                "SRS manifest is missing canonical field: " + std::string(key));
        }

        void validate_srs_manifest_fields(const SrsJsonValue &root)
        {
            if (root.kind != SrsJsonKind::Object)
                throw std::runtime_error("SRS manifest must be a JSON object");
            std::unordered_set<std::string_view> allowed(
                kCanonicalSrsManifestFields.begin(),
                kCanonicalSrsManifestFields.end());
            std::unordered_set<std::string> fields;
            fields.reserve(root.object.size());
            for (const auto &entry : root.object)
            {
                if (!allowed.contains(entry.first))
                    throw std::runtime_error(
                        "SRS manifest contains unknown/non-canonical manifest field: " + entry.first);
                fields.insert(entry.first);
            }
            for (const auto field : kCanonicalSrsManifestFields)
            {
                if (!fields.contains(std::string(field)))
                    throw std::runtime_error(
                        "SRS manifest is missing canonical field: " + std::string(field));
            }
        }

        std::string manifest_string(
            const SrsJsonValue &value,
            std::string_view label)
        {
            if (value.kind != SrsJsonKind::String)
                throw std::runtime_error(
                    "SRS manifest field must be a string: " + std::string(label));
            return value.scalar;
        }

        std::size_t manifest_unsigned(
            const SrsJsonValue &value,
            std::string_view label)
        {
            if (value.kind != SrsJsonKind::Number || value.scalar.empty() || value.scalar.front() == '-' || !std::all_of(value.scalar.begin(), value.scalar.end(), [](unsigned char character)
                                                                                                                         { return character >= '0' && character <= '9'; }))
                throw std::runtime_error(
                    "SRS manifest field must be an unsigned integer: " + std::string(label));
            try
            {
                const auto parsed = std::stoull(value.scalar);
                if (parsed > std::numeric_limits<std::size_t>::max())
                    throw std::out_of_range("SRS manifest integer");
                return static_cast<std::size_t>(parsed);
            }
            catch (const std::exception &)
            {
                throw std::runtime_error(
                    "SRS manifest unsigned integer is out of range: " + std::string(label));
            }
        }

        std::unordered_map<std::string, std::string> parse_manifest_string_map(
            const SrsJsonValue &value,
            std::string_view label)
        {
            if (value.kind != SrsJsonKind::Object)
                throw std::runtime_error(
                    "SRS manifest nested field must be an object: " + std::string(label));
            std::unordered_map<std::string, std::string> out;
            out.reserve(value.object.size());
            for (const auto &entry : value.object)
            {
                if (entry.second.kind != SrsJsonKind::String)
                    throw std::runtime_error(
                        "SRS manifest nested field must contain strings: " + std::string(label) + "." + entry.first);
                out.emplace(entry.first, entry.second.scalar);
            }
            return out;
        }

        std::unordered_map<std::string, std::size_t> parse_manifest_size_map(
            const SrsJsonValue &value,
            std::string_view label)
        {
            if (value.kind != SrsJsonKind::Object)
                throw std::runtime_error(
                    "SRS manifest nested field must be an object: " + std::string(label));
            std::unordered_map<std::string, std::size_t> out;
            out.reserve(value.object.size());
            for (const auto &entry : value.object)
                out.emplace(
                    entry.first,
                    manifest_unsigned(entry.second, std::string(label) + "." + entry.first));
            return out;
        }

        template <typename Map>
        void validate_srs_nested_key_set(
            const Map &actual,
            const std::vector<std::string_view> &expected,
            const std::string &label)
        {
            std::unordered_set<std::string_view> expected_set(
                expected.begin(), expected.end());
            for (const auto &[key, _] : actual)
            {
                (void)_;
                if (!expected_set.contains(key))
                {
                    throw std::runtime_error(
                        "SRS manifest contains unknown/non-canonical nested field in " + label + ": " + key);
                }
            }
            for (const auto key : expected)
            {
                if (!actual.contains(std::string(key)))
                {
                    throw std::runtime_error(
                        "SRS manifest is missing canonical nested field in " + label + ": " + std::string(key));
                }
            }
        }

        void validate_srs_manifest_nested_schema(const SRSManifest &manifest)
        {
            const std::vector<std::string_view> envelope_fields(
                kCanonicalSrsEnvelopeFields.begin(),
                kCanonicalSrsEnvelopeFields.end());
            validate_srs_nested_key_set(
                manifest.max_instance_envelope,
                envelope_fields,
                "max_instance_envelope");

            const auto domain_fields = [&]
            {
                std::vector<std::string_view> names;
                names.reserve(manifest.domain_sizes.size());
                for (const auto &[name, _] : manifest.domain_sizes)
                {
                    (void)_;
                    names.push_back(name);
                }
                return names;
            }();
            if (domain_fields.empty())
            {
                throw std::runtime_error(
                    "SRS manifest domain_sizes must not be empty");
            }
            validate_srs_nested_key_set(
                manifest.domain_generators,
                domain_fields,
                "domain_generators");
            validate_srs_nested_key_set(
                manifest.pk_lagrange_paths,
                domain_fields,
                "pk_lagrange_paths");
            validate_srs_nested_key_set(
                manifest.pk_lagrange_hashes,
                domain_fields,
                "pk_lagrange_hashes");
        }

        std::string canonical_domain_catalog_hash(
            const std::unordered_map<std::string, std::size_t> &domains)
        {
            std::vector<std::pair<std::string, std::size_t>> ordered(
                domains.begin(), domains.end());
            std::sort(ordered.begin(), ordered.end());
            std::ostringstream canonical;
            canonical << "zkgat:domain-size-catalog";
            for (const auto &[name, size] : ordered)
            {
                canonical << '|' << name << '=' << size;
            }
            return sha256_hex(canonical.str());
        }

        std::string bytes_to_lower_hex(const std::uint8_t *bytes, std::size_t size)
        {
            static constexpr char alphabet[] = "0123456789abcdef";
            std::string out(size * 2, '0');
            for (std::size_t i = 0; i < size; ++i)
            {
                out[i * 2] = alphabet[bytes[i] >> 4U];
                out[i * 2 + 1] = alphabet[bytes[i] & 0x0fU];
            }
            return out;
        }

        std::string canonical_field_hex(const FieldElement &value)
        {
            std::array<std::uint8_t, 32> bytes{};
            value.write_little_endian(bytes.data(), bytes.size());
            return bytes_to_lower_hex(bytes.data(), bytes.size());
        }

        std::string canonical_domain_generator_catalog_hash(
            const std::unordered_map<std::string, std::size_t> &domains,
            const std::unordered_map<std::string, std::string> &generators,
            const std::string &root_derivation_id,
            const std::string &row_order)
        {
            std::vector<std::pair<std::string, std::size_t>> ordered(
                domains.begin(), domains.end());
            std::sort(ordered.begin(), ordered.end());
            std::ostringstream canonical;
            canonical << "zkgat:domain-generator-catalog"
                      << "|root_derivation_id=" << root_derivation_id
                      << "|row_order=" << row_order;
            for (const auto &[name, size] : ordered)
            {
                const auto generator = generators.find(name);
                if (generator == generators.end())
                {
                    throw std::runtime_error(
                        "SRS domain generator catalogue is missing domain: " + name);
                }
                canonical << '|' << name << ':' << size << ':' << generator->second;
            }
            return sha256_hex(canonical.str());
        }

        std::string read_text_file(const std::filesystem::path &path);

        std::vector<std::string> ordered_domain_names(
            const std::unordered_map<std::string, std::size_t> &domains)
        {
            std::vector<std::string> names;
            names.reserve(domains.size());
            for (const auto &[name, _] : domains)
            {
                (void)_;
                names.push_back(name);
            }
            std::sort(names.begin(), names.end());
            return names;
        }

        std::string canonical_vanishing_g2_catalog_hash(
            const SRSManifest &manifest,
            const std::unordered_map<std::string, G2Point> &domain_zero_g2)
        {
            std::ostringstream canonical;
            canonical << "zkgat:vanishing-g2-catalog";
            for (const auto &name :
                 ordered_domain_names(manifest.domain_sizes))
            {
                const auto point = domain_zero_g2.find(name);
                if (point == domain_zero_g2.end())
                    throw std::runtime_error(
                        "missing vanishing G2 catalogue entry: " + name);
                const auto bytes = serialize(point->second);
                canonical << '|' << name << ':'
                          << manifest.domain_sizes.at(name) << ':'
                          << bytes_to_lower_hex(
                                 bytes.data(), bytes.size());
            }
            return sha256_hex(canonical.str());
        }

        std::string canonical_srs_audit_report(
            const SRSManifest &manifest)
        {
            std::ostringstream report;
            report << "GATZK-SRS-CONSISTENCY-AUDIT\n";
            report << "status=passed\n";
            report << "sampling=none\n";
            report << "coefficient_relation_method=exact_full_tau_reconstruction\n";
            report << "lagrange_relation_method=exact_full_tau_reconstruction\n";
            report << "group_validation=canonical_decode,curve,subgroup,non_identity\n";
            report << "coefficient_point_count="
                   << manifest.degree_bound + 1U << '\n';
            report << "coefficient_relation_count="
                   << manifest.degree_bound << '\n';
            report << "domain_size_catalog_hash="
                   << canonical_domain_catalog_hash(
                          manifest.domain_sizes)
                   << '\n';
            report << "domain_generator_catalog_hash="
                   << canonical_domain_generator_catalog_hash(
                          manifest.domain_sizes,
                          manifest.domain_generators,
                          manifest.root_derivation_id,
                          manifest.row_order)
                   << '\n';
            report << "pk_coef_hash=" << manifest.pk_coef_hash << '\n';
            report << "vk_hash=" << manifest.vk_hash << '\n';
            report << "protocol_compiler_artifact_hash="
                   << manifest.protocol_compiler_artifact_hash << '\n';
            report << "vanishing_g2_hash="
                   << manifest.vanishing_g2_hash << '\n';
            for (const auto &name :
                 ordered_domain_names(manifest.domain_sizes))
            {
                const auto hash =
                    manifest.pk_lagrange_hashes.find(name);
                if (hash == manifest.pk_lagrange_hashes.end())
                    throw std::runtime_error(
                        "SRS audit report lacks Lagrange hash: " + name);
                report << "domain." << name << '='
                       << manifest.domain_sizes.at(name) << ','
                       << hash->second << ','
                       << manifest.domain_sizes.at(name) << '\n';
            }
            return report.str();
        }

        void validate_srs_audit_report_material(
            const std::filesystem::path &srs_root,
            const SRSManifest &manifest)
        {
            const auto report = read_text_file(
                srs_root / manifest.srs_audit_report_path);
            const auto expected = canonical_srs_audit_report(manifest);
            const auto actual_hash = sha256_hex(report);
            if (report != expected || actual_hash != manifest.srs_audit_report_hash || actual_hash != manifest.srs_consistency_hash)
            {
                throw std::runtime_error(
                    "SRS consistency audit report binding mismatch");
            }
        }

        std::string read_text_file(const std::filesystem::path &path)
        {
            std::ifstream input(path);
            if (!input)
            {
                throw std::runtime_error("failed to open SRS manifest: " + path.string());
            }
            std::ostringstream stream;
            stream << input.rdbuf();
            return stream.str();
        }

        struct CanonicalFileIdentity
        {
            std::uintmax_t size = 0;
            std::string sha256;
        };

        // Cache keys must authenticate the bytes that are actually consumed by
        // the key loader.  Filesystem path/mtime metadata is insufficient:
        // replacing a resource in place must invalidate a resident key even
        // when the path and timestamps are unchanged.
        CanonicalFileIdentity canonical_file_identity(
            const std::filesystem::path &path)
        {
            std::error_code error;
            const auto size = std::filesystem::file_size(path, error);
            if (error)
                throw std::runtime_error(
                    "failed to stat SRS key resource: " + path.string());
            std::ifstream input(path, std::ios::binary);
            if (!input)
                throw std::runtime_error(
                    "failed to open SRS key resource: " + path.string());
            SHA256_CTX digest_context{};
            SHA256_Init(&digest_context);
            std::array<char, 1U << 20> buffer{};
            std::uintmax_t consumed = 0;
            while (input)
            {
                input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
                const auto count = input.gcount();
                if (count > 0)
                {
                    SHA256_Update(
                        &digest_context,
                        buffer.data(),
                        static_cast<std::size_t>(count));
                    consumed += static_cast<std::uintmax_t>(count);
                }
            }
            if (!input.eof() || consumed != size)
                throw std::runtime_error(
                    "failed while hashing SRS key resource: " + path.string());
            std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
            SHA256_Final(digest.data(), &digest_context);
            return CanonicalFileIdentity{
                .size = size,
                .sha256 = bytes_to_hex(digest.data(), digest.size())};
        }

        std::string canonical_key_resource_identity(
            const std::filesystem::path &srs_root,
            const SRSManifest &manifest,
            const std::vector<std::string> &selected_lagrange_domains)
        {
            std::vector<std::pair<std::string, std::filesystem::path>> resources;
            resources.emplace_back("manifest", srs_root / "srs_manifest.json");
            resources.emplace_back("pk_coef", srs_root / manifest.pk_coef_path);
            resources.emplace_back("vk", srs_root / manifest.vk_path);
            resources.emplace_back(
                "protocol_compiler_artifact",
                srs_root / manifest.protocol_compiler_artifact_path);
            resources.emplace_back(
                "srs_audit_report",
                srs_root / manifest.srs_audit_report_path);
            for (const auto &domain : selected_lagrange_domains)
            {
                const auto path_it = manifest.pk_lagrange_paths.find(domain);
                if (path_it == manifest.pk_lagrange_paths.end())
                    throw std::runtime_error(
                        "SRS resource identity lacks Lagrange path: " + domain);
                resources.emplace_back(
                    "pk_lagrange:" + domain,
                    srs_root / path_it->second);
            }
            std::sort(
                resources.begin(), resources.end(),
                [](const auto &lhs, const auto &rhs)
                { return lhs.first < rhs.first; });

            std::ostringstream canonical;
            canonical << "zkgat:kzg-key-resource-identity\n"
                      << "manifest_hash=" << manifest.manifest_hash << '\n'
                      << "srs_id=" << manifest.srs_id << '\n'
                      << "srs_type=" << manifest.srs_type << '\n'
                      << "degree_bound=" << manifest.degree_bound << '\n'
                      << "generator_id=" << manifest.generator_id << '\n'
                      << "encoding=" << manifest.encoding << '\n'
                      << "protocol_schema_id=" << manifest.protocol_schema_id << '\n'
                      << "protocol_compiler_artifact_hash="
                      << manifest.protocol_compiler_artifact_hash << '\n'
                      << "proof_schema_template_hash="
                      << manifest.proof_schema_template_hash << '\n'
                      << "observation_ledger_hash="
                      << manifest.observation_ledger_hash << '\n'
                      << "random_identity_ledger_hash="
                      << manifest.random_identity_ledger_hash << '\n'
                      << "security_budget_ledger_hash="
                      << manifest.security_budget_ledger_hash << '\n'
                      << "domain_generator_catalog_hash="
                      << manifest.domain_generator_catalog_hash << '\n'
                      << "selected_lagrange_count="
                      << selected_lagrange_domains.size() << '\n';
            for (const auto &domain : selected_lagrange_domains)
                canonical << "selected_lagrange=" << domain << '\n';
            for (const auto &[label, path] : resources)
            {
                const auto identity = canonical_file_identity(path);
                std::error_code error;
                const auto absolute = std::filesystem::absolute(path, error);
                if (error)
                    throw std::runtime_error(
                        "failed to canonicalize SRS key resource: " + path.string());
                canonical << "resource=" << label
                          << "|path=" << absolute.lexically_normal().generic_string()
                          << "|size=" << identity.size
                          << "|sha256=" << identity.sha256 << '\n';
            }
            return "sha256:" + sha256_hex(canonical.str());
        }

        std::string decode_canonical_hex(std::string_view value)
        {
            if (value.empty() || (value.size() & 1U) != 0U)
                throw std::runtime_error(
                    "protocol compiler artifact hex field is malformed");
            auto nibble = [](char c) -> unsigned
            {
                if (c >= '0' && c <= '9')
                    return static_cast<unsigned>(c - '0');
                if (c >= 'a' && c <= 'f')
                    return 10U + static_cast<unsigned>(c - 'a');
                throw std::runtime_error(
                    "protocol compiler artifact hex field is not canonical lowercase hex");
            };
            std::string out(value.size() / 2, '\0');
            for (std::size_t i = 0; i < out.size(); ++i)
                out[i] = static_cast<char>((nibble(value[2 * i]) << 4U) | nibble(value[2 * i + 1]));
            return out;
        }

        SRSManifest load_manifest_file(const std::string &srs_path)
        {
            const auto manifest_path = std::filesystem::path(srs_path) / "srs_manifest.json";
            const auto text = read_text_file(manifest_path);
            const auto root = SrsJsonParser(text).parse();
            validate_srs_manifest_fields(root);
            SRSManifest manifest;
            const auto string_field =
                [&](std::string_view key)
            {
                return manifest_string(require_manifest_field(root, key), key);
            };
            manifest.manifest_format = string_field("manifest_format");
            manifest.srs_id = string_field("srs_id");
            manifest.curve_id = string_field("curve_id");
            manifest.field_modulus_id = string_field("field_modulus_id");
            manifest.srs_type = string_field("srs_type");
            manifest.ceremony_transcript_hash =
                string_field("ceremony_transcript_hash");
            manifest.setup_transcript_hash = string_field("setup_transcript_hash");
            manifest.degree_bound = manifest_unsigned(
                require_manifest_field(root, "degree_bound"), "degree_bound");
            manifest.domain_sizes = parse_manifest_size_map(
                require_manifest_field(root, "domain_sizes"), "domain_sizes");
            manifest.domain_generators = parse_manifest_string_map(
                require_manifest_field(root, "domain_generators"),
                "domain_generators");
            manifest.root_derivation_id = string_field("root_derivation_id");
            manifest.row_order = string_field("row_order");
            manifest.protocol_compiler_artifact_path =
                string_field("protocol_compiler_artifact_path");
            manifest.protocol_compiler_artifact_hash =
                string_field("protocol_compiler_artifact_hash");
            manifest.protocol_schema_id = string_field("protocol_schema_id");
            manifest.max_instance_envelope_id =
                string_field("max_instance_envelope_id");
            manifest.max_instance_envelope_hash =
                string_field("max_instance_envelope_hash");
            manifest.max_instance_envelope = parse_manifest_size_map(
                require_manifest_field(root, "max_instance_envelope"),
                "max_instance_envelope");
            manifest.proof_schema_template_hash =
                string_field("proof_schema_template_hash");
            manifest.observation_ledger_hash =
                string_field("observation_ledger_hash");
            manifest.random_identity_ledger_hash =
                string_field("random_identity_ledger_hash");
            manifest.security_budget_ledger_hash =
                string_field("security_budget_ledger_hash");
            manifest.pk_coef_path = string_field("pk_coef_path");
            manifest.pk_coef_hash = string_field("pk_coef_hash");
            manifest.pk_lagrange_paths = parse_manifest_string_map(
                require_manifest_field(root, "pk_lagrange_paths"),
                "pk_lagrange_paths");
            manifest.pk_lagrange_hashes = parse_manifest_string_map(
                require_manifest_field(root, "pk_lagrange_hashes"),
                "pk_lagrange_hashes");
            manifest.vk_path = string_field("vk_path");
            manifest.vk_hash = string_field("vk_hash");
            manifest.vanishing_g2_hash = string_field("vanishing_g2_hash");
            manifest.srs_audit_report_path =
                string_field("srs_audit_report_path");
            manifest.srs_audit_report_hash =
                string_field("srs_audit_report_hash");
            manifest.srs_consistency_hash =
                string_field("srs_consistency_hash");
            manifest.encoding = string_field("encoding");
            manifest.generator_id = string_field("generator_id");
            manifest.security_level = string_field("security_level");
            manifest.kzg_assumption_profile_id =
                string_field("kzg_assumption_profile_id");
            manifest.kzg_assumption_degree_bound = manifest_unsigned(
                require_manifest_field(root, "kzg_assumption_degree_bound"),
                "kzg_assumption_degree_bound");
            manifest.kzg_assumption_security_bits = manifest_unsigned(
                require_manifest_field(root, "kzg_assumption_security_bits"),
                "kzg_assumption_security_bits");
            manifest.kzg_assumption_domain_catalog_hash =
                string_field("kzg_assumption_domain_catalog_hash");
            manifest.domain_generator_catalog_hash =
                string_field("domain_generator_catalog_hash");
            const auto declared_manifest_hash = string_field("manifest_hash");
            std::size_t manifest_hash_key_position = std::string::npos;
            for (std::size_t index = 0; index < root.object.size(); ++index)
            {
                if (root.object[index].first == "manifest_hash")
                {
                    manifest_hash_key_position = root.object_key_positions.at(index);
                    break;
                }
            }
            if (declared_manifest_hash.empty() || manifest_hash_key_position < 2 || text[manifest_hash_key_position - 1] != ' ' || text[manifest_hash_key_position - 2] != ' ' || sha256_hex(text.substr(0, manifest_hash_key_position - 2)) != declared_manifest_hash)
            {
                throw std::runtime_error("SRS manifest canonical hash mismatch");
            }
            manifest.manifest_hash = sha256_hex(text);
            validate_srs_manifest_nested_schema(manifest);
            if (manifest.manifest_format.empty() || manifest.srs_id.empty() || manifest.srs_type.empty() || manifest.degree_bound == 0 || manifest.pk_coef_path.empty() || manifest.pk_lagrange_paths.empty() || manifest.vk_path.empty() || manifest.vanishing_g2_hash.empty() || manifest.srs_audit_report_path.empty() || manifest.srs_audit_report_hash.empty() || manifest.srs_consistency_hash.empty() || manifest.protocol_compiler_artifact_path.empty() || manifest.protocol_compiler_artifact_hash.empty() || manifest.ceremony_transcript_hash.empty() || manifest.setup_transcript_hash.empty())
            {
                throw std::runtime_error("SRS manifest is missing required material fields");
            }
            return manifest;
        }

        void validate_protocol_compiler_artifact_material(
            const std::filesystem::path &srs_root,
            const SRSManifest &manifest)
        {
            const auto artifact = read_text_file(
                srs_root / manifest.protocol_compiler_artifact_path);
            if (sha256_hex(artifact) != manifest.protocol_compiler_artifact_hash)
                throw std::runtime_error(
                    "protocol compiler artifact material hash mismatch");
            const auto parsed = parse_compiler_artifact(artifact);
            if (parsed.artifact_hash != sha256_hex(parsed.canonical_prefix))
                throw std::runtime_error(
                    "protocol compiler artifact canonical hash mismatch");
            const auto scalar = [&](std::string_view key) -> const std::string &
            { return parsed.scalar(key); };
            const auto protocol_template_hex =
                scalar("protocol_template_hex");
            const auto proof_schema_template_hex =
                scalar("proof_schema_template_hex");
            const auto observation_ledger_template_hex =
                scalar("observation_ledger_template_hex");
            const auto random_identity_ledger_template_hex =
                scalar("random_identity_ledger_template_hex");
            const auto security_budget_ledger_template_hex =
                scalar("security_budget_ledger_template_hex");
            const auto max_instance_envelope_hex =
                scalar("max_instance_envelope_hex");
            const auto max_instance_envelope_hash =
                scalar("max_instance_envelope_hash");
            if (protocol_template_hex.empty() || proof_schema_template_hex.empty() || observation_ledger_template_hex.empty() || random_identity_ledger_template_hex.empty() || security_budget_ledger_template_hex.empty() || max_instance_envelope_hex.empty() || max_instance_envelope_hash.empty() || scalar("protocol_schema_id") != manifest.protocol_schema_id || sha256_hex(decode_canonical_hex(protocol_template_hex)) != manifest.protocol_schema_id || scalar("max_instance_envelope_id") != manifest.max_instance_envelope_id || max_instance_envelope_hash != manifest.max_instance_envelope_hash || max_instance_envelope_hash != sha256_hex(decode_canonical_hex(max_instance_envelope_hex)) || static_cast<std::size_t>(std::stoull(scalar("degree_bound"))) != manifest.degree_bound || scalar("domain_size_catalog_hash") != manifest.kzg_assumption_domain_catalog_hash || scalar("domain_generator_catalog_hash") != manifest.domain_generator_catalog_hash || scalar("proof_schema_template_hash") != manifest.proof_schema_template_hash || scalar("observation_ledger_hash") != manifest.observation_ledger_hash || scalar("random_identity_ledger_hash") != manifest.random_identity_ledger_hash || scalar("security_budget_ledger_hash") != manifest.security_budget_ledger_hash)
                throw std::runtime_error(
                    "protocol compiler artifact registries do not match the SRS manifest");
            const auto envelope = decode_canonical_hex(max_instance_envelope_hex);
            const std::vector<std::string_view> envelope_keys = {
                "format", "max_layers", "max_hidden_heads_per_layer",
                "max_hidden_head_dim", "max_input_dim", "max_output_heads",
                "max_classes", "max_graphs", "max_nodes", "max_edges",
                "max_total_nodes", "max_concat_dim", "max_range_bits",
                "max_coefficient_object_length", "max_object_family_count",
                "max_coefficient_chunks", "security_target_bits"};
            std::string canonical_envelope;
            std::unordered_map<std::string, std::size_t> envelope_sizes;
            const auto envelope_values = parse_ordered_key_value_lines(
                envelope, envelope_keys, "protocol compiler max instance envelope");
            for (std::size_t index = 0; index < envelope_keys.size(); ++index)
            {
                const auto &key = envelope_keys[index];
                const auto &value = envelope_values[index];
                canonical_envelope += std::string(key) + "=" + value + "\n";
                if (key == "format")
                {
                    if (value != "zkgat_max_instance_envelope")
                        throw std::runtime_error(
                            "protocol compiler max envelope format mismatch");
                    continue;
                }
                if (value.empty() || (value.size() > 1 && value.front() == '0') || !std::all_of(value.begin(), value.end(), [](unsigned char ch)
                                                                                                { return ch >= '0' && ch <= '9'; }))
                    throw std::runtime_error(
                        "protocol compiler max envelope integer mismatch: " + std::string(key));
                const auto parsed = static_cast<std::size_t>(std::stoull(value));
                if (parsed == 0)
                    throw std::runtime_error(
                        "protocol compiler max envelope has zero bound: " + std::string(key));
                envelope_sizes.emplace(key, parsed);
            }
            if (canonical_envelope != envelope || envelope_sizes != manifest.max_instance_envelope || envelope_sizes.at("security_target_bits") != 120 || envelope_sizes.at("max_nodes") > manifest.domain_sizes.at("N") || envelope_sizes.at("max_edges") > manifest.domain_sizes.at("edge") || envelope_sizes.at("max_nodes") > std::numeric_limits<std::size_t>::max() / envelope_sizes.at("max_hidden_head_dim") || envelope_sizes.at("max_nodes") * envelope_sizes.at("max_hidden_head_dim") > manifest.domain_sizes.at("edge") || envelope_sizes.at("max_nodes") > std::numeric_limits<std::size_t>::max() / envelope_sizes.at("max_classes") || envelope_sizes.at("max_nodes") * envelope_sizes.at("max_classes") > manifest.domain_sizes.at("edge") || envelope_sizes.at("max_input_dim") > manifest.domain_sizes.at("in") || envelope_sizes.at("max_hidden_head_dim") > manifest.domain_sizes.at("d") || envelope_sizes.at("max_classes") > manifest.domain_sizes.at("C") || envelope_sizes.at("max_concat_dim") > manifest.domain_sizes.at("cat") || envelope_sizes.at("max_graphs") > envelope_sizes.at("max_nodes") || envelope_sizes.at("max_total_nodes") < envelope_sizes.at("max_nodes") || envelope_sizes.at("max_concat_dim") < envelope_sizes.at("max_hidden_head_dim") || envelope_sizes.at("max_range_bits") >= std::numeric_limits<std::size_t>::digits || manifest.domain_sizes.at("edge") < 3 || (std::size_t{1} << envelope_sizes.at("max_range_bits")) > (manifest.domain_sizes.at("edge") - 3) / 2 || envelope_sizes.at("max_coefficient_chunks") > std::numeric_limits<std::size_t>::max() / (manifest.domain_sizes.at("edge") - 1) || envelope_sizes.at("max_coefficient_object_length") > envelope_sizes.at("max_coefficient_chunks") * (manifest.domain_sizes.at("edge") - 1) || envelope_sizes.at("max_nodes") > std::numeric_limits<std::size_t>::max() / envelope_sizes.at("max_input_dim") || envelope_sizes.at("max_nodes") * envelope_sizes.at("max_input_dim") > manifest.domain_sizes.at("FH"))
                throw std::runtime_error(
                    "protocol compiler max envelope exceeds the SRS domain catalogue");
            if (decode_canonical_hex(proof_schema_template_hex) != ::gatzk::protocol::protocol_proof_schema_template_canonical_bytes() || decode_canonical_hex(observation_ledger_template_hex) != ::gatzk::protocol::protocol_observation_ledger_template_canonical_bytes() || decode_canonical_hex(random_identity_ledger_template_hex) != ::gatzk::protocol::protocol_random_identity_ledger_template_canonical_bytes() || decode_canonical_hex(security_budget_ledger_template_hex) != ::gatzk::protocol::protocol_security_budget_ledger_template_canonical_bytes() || sha256_hex(decode_canonical_hex(proof_schema_template_hex)) != manifest.proof_schema_template_hash || sha256_hex(decode_canonical_hex(observation_ledger_template_hex)) != manifest.observation_ledger_hash || sha256_hex(decode_canonical_hex(random_identity_ledger_template_hex)) != manifest.random_identity_ledger_hash || sha256_hex(decode_canonical_hex(security_budget_ledger_template_hex)) != manifest.security_budget_ledger_hash)
                throw std::runtime_error(
                    "protocol compiler artifact embedded registry bytes mismatch");
        }

        void validate_material_hash_value(
            const std::string &actual_hash,
            const std::string &expected_hash,
            const std::string &label)
        {
            if (expected_hash.empty())
            {
                throw std::runtime_error("SRS manifest is missing material hash: " + label);
            }
            if (actual_hash != expected_hash)
            {
                throw std::runtime_error("SRS material hash mismatch for " + label);
            }
        }

        std::uint64_t read_u64_le(std::istream &input, const std::string &label)
        {
            std::array<std::uint8_t, 8> bytes{};
            input.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
            if (input.gcount() != static_cast<std::streamsize>(bytes.size()))
            {
                throw std::runtime_error("truncated SRS file while reading " + label);
            }
            std::uint64_t out = 0;
            for (std::size_t i = 0; i < bytes.size(); ++i)
            {
                out |= static_cast<std::uint64_t>(bytes[i]) << (8U * i);
            }
            return out;
        }

        std::vector<G1Point> load_g1_vector_file(
            const std::filesystem::path &path,
            std::string *actual_hash = nullptr,
            std::size_t max_points = 0);

        enum class G1LoadWorkerFailure : unsigned
        {
            None = 0,
            CanonicalDeserialize,
            CanonicalIdentity,
            CanonicalSerialize,
            CanonicalEncoding,
            UnexpectedCanonical,
        };

        struct G1LoadWorkerFailureInfo
        {
            G1LoadWorkerFailure kind =
                G1LoadWorkerFailure::None;

            std::size_t index =
                std::numeric_limits<std::size_t>::max();
        };

        std::string g1_load_worker_failure_message(
            const G1LoadWorkerFailureInfo &failure,
            const std::filesystem::path &path)
        {
            switch (failure.kind)
            {
            case G1LoadWorkerFailure::CanonicalDeserialize:
                return "failed to deserialize SRS G1 point "
                       "at index " +
                       std::to_string(failure.index) + ": " + path.string();

            case G1LoadWorkerFailure::CanonicalIdentity:
                return "identity SRS G1 point at index " + std::to_string(failure.index) + ": " + path.string();

            case G1LoadWorkerFailure::CanonicalSerialize:
                return "failed to serialize SRS G1 point "
                       "at index " +
                       std::to_string(failure.index) + ": " + path.string();

            case G1LoadWorkerFailure::CanonicalEncoding:
                return "non-canonical SRS G1 encoding at index " + std::to_string(failure.index) + ": " + path.string();

            case G1LoadWorkerFailure::UnexpectedCanonical:
                return "unexpected exception while validating "
                       "canonical SRS G1 point near index " +
                       std::to_string(failure.index) + ": " + path.string();

            case G1LoadWorkerFailure::None:
                return {};
            }

            return "unknown parallel SRS G1 validation failure: " + path.string();
        }

        std::uint64_t load_u64_le_unaligned(
            const std::uint8_t *bytes)
        {
            std::uint64_t value = 0;

            for (std::size_t i = 0; i < 8; ++i)
            {
                value |=
                    static_cast<std::uint64_t>(bytes[i])
                    << (8U * i);
            }

            return value;
        }

        std::vector<G1Point>
        load_pipelined_canonical_g1_vector_file_prefix(
            const std::filesystem::path &path,
            std::size_t file_size,
            std::size_t expected_point_count,
            std::size_t point_size,
            std::size_t max_points,
            std::string *actual_hash,
            G1LoadWorkerFailureInfo &worker_failure)
        {
            worker_failure = {};

            constexpr std::size_t
                kHeaderSize = 24;

            constexpr std::size_t
                kCanonicalWorkerChunkPoints =
                    1ULL << 14;

            constexpr std::size_t
                kHashBufferBytes =
                    8ULL << 20;

            // Keep enough reusable canonical buffers to expose CPU parallelism
            // without allowing temporary payload storage to grow without a
            // deterministic bound.
            constexpr std::size_t
                kPipelineBufferBudgetBytes =
                    256ULL << 20;

            if (point_size == 0 ||
                expected_point_count >
                    (std::numeric_limits<std::size_t>::max() -
                     kHeaderSize) /
                        point_size)
            {
                throw std::runtime_error(
                    "canonical SRS G1 size overflow: " +
                    path.string());
            }

            const auto expected_size =
                kHeaderSize +
                expected_point_count * point_size;

            if (expected_size != file_size)
            {
                throw std::runtime_error(
                    "canonical SRS G1 size mismatch: " +
                    path.string());
            }

            std::ifstream input(
                path,
                std::ios::binary);

            if (!input)
            {
                throw std::runtime_error(
                    "failed to open canonical SRS G1 material: " +
                    path.string());
            }

            SHA256_CTX ctx;
            SHA256_Init(&ctx);

            auto read_hashed =
                [&](void *destination,
                    std::size_t byte_count,
                    const char *label)
            {
                if (byte_count == 0)
                {
                    return;
                }

                input.read(
                    static_cast<char *>(destination),
                    static_cast<std::streamsize>(
                        byte_count));

                if (input.gcount() !=
                    static_cast<std::streamsize>(
                        byte_count))
                {
                    throw std::runtime_error(
                        "truncated canonical SRS G1 material while reading " +
                        std::string(label) + ": " +
                        path.string());
                }

                SHA256_Update(
                    &ctx,
                    destination,
                    byte_count);
            };

            std::array<std::uint8_t, kHeaderSize>
                header{};

            read_hashed(
                header.data(),
                header.size(),
                "header");

            const std::array<std::uint8_t, 8>
                expected_magic{{'G', 'A', 'T', 'Z',
                                'K', 'G', '1', 'C'}};

            if (std::memcmp(
                    header.data(),
                    expected_magic.data(),
                    expected_magic.size()) != 0)
            {
                throw std::runtime_error(
                    "canonical SRS G1 header mismatch: " +
                    path.string());
            }

            const auto encoded_point_count =
                load_u64_le_unaligned(
                    header.data() + 8);

            const auto encoded_point_size =
                load_u64_le_unaligned(
                    header.data() + 16);

            if (encoded_point_count !=
                    static_cast<std::uint64_t>(
                        expected_point_count) ||
                encoded_point_size !=
                    static_cast<std::uint64_t>(
                        point_size))
            {
                throw std::runtime_error(
                    "canonical SRS G1 header changed while loading: " +
                    path.string());
            }

            const auto load_count =
                max_points == 0
                    ? expected_point_count
                    : std::min(
                          expected_point_count,
                          max_points);

            std::vector<G1Point> out(
                load_count);

            std::atomic<unsigned>
                worker_failure_kind{
                    static_cast<unsigned>(
                        G1LoadWorkerFailure::None)};

            std::atomic<std::size_t>
                worker_failure_index{
                    std::numeric_limits<
                        std::size_t>::max()};

            auto record_worker_failure =
                [&](G1LoadWorkerFailure kind,
                    std::size_t index) noexcept
            {
                auto expected =
                    static_cast<unsigned>(
                        G1LoadWorkerFailure::None);

                if (worker_failure_kind
                        .compare_exchange_strong(
                            expected,
                            static_cast<unsigned>(kind),
                            std::memory_order_acq_rel,
                            std::memory_order_acquire))
                {
                    worker_failure_index.store(
                        index,
                        std::memory_order_release);
                }
            };

            auto worker_failed =
                [&]() noexcept
            {
                return worker_failure_kind.load(
                           std::memory_order_acquire) !=
                       static_cast<unsigned>(
                           G1LoadWorkerFailure::None);
            };

            auto validate_chunk =
                [&](const std::uint8_t *bytes,
                    std::size_t begin,
                    std::size_t count,
                    std::vector<std::uint8_t>
                        &canonical_bytes)
            {
                try
                {
                    for (std::size_t local = 0;
                         local < count;
                         ++local)
                    {
                        if (worker_failed())
                        {
                            return;
                        }

                        const auto index =
                            begin + local;

                        const auto *point_begin =
                            bytes +
                            local * point_size;

                        const auto consumed =
                            out[index]
                                .value
                                .deserialize(
                                    point_begin,
                                    point_size);

                        if (consumed != point_size)
                        {
                            record_worker_failure(
                                G1LoadWorkerFailure::
                                    CanonicalDeserialize,
                                index);
                            return;
                        }

                        if (out[index]
                                .value
                                .isZero())
                        {
                            record_worker_failure(
                                G1LoadWorkerFailure::
                                    CanonicalIdentity,
                                index);
                            return;
                        }

                        const auto written =
                            out[index]
                                .value
                                .serialize(
                                    canonical_bytes.data(),
                                    canonical_bytes.size());

                        if (written == 0)
                        {
                            record_worker_failure(
                                G1LoadWorkerFailure::
                                    CanonicalSerialize,
                                index);
                            return;
                        }

                        if (written != point_size ||
                            std::memcmp(
                                canonical_bytes.data(),
                                point_begin,
                                point_size) != 0)
                        {
                            record_worker_failure(
                                G1LoadWorkerFailure::
                                    CanonicalEncoding,
                                index);
                            return;
                        }
                    }
                }
                catch (...)
                {
                    record_worker_failure(
                        G1LoadWorkerFailure::
                            UnexpectedCanonical,
                        begin);
                }
            };

            const auto cpu_count =
                std::max<std::size_t>(
                    1,
                    std::thread::
                        hardware_concurrency());

            const auto chunk_count =
                load_count == 0
                    ? std::size_t{0}
                    : (load_count +
                       kCanonicalWorkerChunkPoints - 1) /
                          kCanonicalWorkerChunkPoints;

            const auto desired_worker_count =
                cpu_count > 1
                    ? std::min<std::size_t>(
                          cpu_count - 1,
                          chunk_count)
                    : std::size_t{0};

            const auto maximum_chunk_points =
                std::min<std::size_t>(
                    kCanonicalWorkerChunkPoints,
                    std::max<std::size_t>(
                        1,
                        load_count));

            const auto maximum_chunk_bytes =
                maximum_chunk_points *
                point_size;

            const auto max_slots_by_memory =
                std::max<std::size_t>(
                    1,
                    kPipelineBufferBudgetBytes /
                        std::max<std::size_t>(
                            1,
                            maximum_chunk_bytes));

            const auto slot_count =
                std::max<std::size_t>(
                    1,
                    std::min<std::size_t>(
                        chunk_count,
                        std::min<std::size_t>(
                            desired_worker_count + 2,
                            max_slots_by_memory)));

            const auto background_worker_count =
                std::min<std::size_t>(
                    desired_worker_count,
                    slot_count);

            auto hash_unloaded_suffix =
                [&]()
            {
                const auto unloaded_points =
                    expected_point_count -
                    load_count;

                if (unloaded_points == 0)
                {
                    return;
                }

                std::size_t remaining =
                    unloaded_points *
                    point_size;

                auto buffer =
                    std::make_unique<
                        std::uint8_t[]>(
                        kHashBufferBytes);

                while (remaining != 0)
                {
                    const auto current =
                        std::min<std::size_t>(
                            remaining,
                            kHashBufferBytes);

                    read_hashed(
                        buffer.get(),
                        current,
                        "authenticated suffix");

                    remaining -= current;
                }
            };

            if (load_count == 0)
            {
                hash_unloaded_suffix();

                unsigned char
                    digest[SHA256_DIGEST_LENGTH];

                SHA256_Final(
                    digest,
                    &ctx);

                if (actual_hash != nullptr)
                {
                    *actual_hash =
                        bytes_to_hex(
                            digest,
                            sizeof(digest));
                }

                return out;
            }

            // A single-core host uses the same authenticated bytes but does
            // not need the producer/consumer queue.
            if (background_worker_count == 0)
            {
                std::vector<std::uint8_t>
                    payload(
                        maximum_chunk_bytes);

                std::vector<std::uint8_t>
                    canonical_bytes(
                        point_size);

                for (std::size_t begin = 0;
                     begin < load_count;
                     begin +=
                     kCanonicalWorkerChunkPoints)
                {
                    const auto current_count =
                        std::min<std::size_t>(
                            kCanonicalWorkerChunkPoints,
                            load_count - begin);

                    const auto current_bytes =
                        current_count *
                        point_size;

                    read_hashed(
                        payload.data(),
                        current_bytes,
                        "canonical G1 prefix chunk");

                    validate_chunk(
                        payload.data(),
                        begin,
                        current_count,
                        canonical_bytes);

                    if (worker_failed())
                    {
                        break;
                    }
                }

                if (!worker_failed())
                {
                    hash_unloaded_suffix();

                    unsigned char
                        digest[SHA256_DIGEST_LENGTH];

                    SHA256_Final(
                        digest,
                        &ctx);

                    if (actual_hash != nullptr)
                    {
                        *actual_hash =
                            bytes_to_hex(
                                digest,
                                sizeof(digest));
                    }
                }

                const auto recorded_failure =
                    static_cast<G1LoadWorkerFailure>(
                        worker_failure_kind.load(
                            std::memory_order_acquire));

                if (recorded_failure !=
                    G1LoadWorkerFailure::None)
                {
                    worker_failure.kind =
                        recorded_failure;

                    worker_failure.index =
                        worker_failure_index.load(
                            std::memory_order_acquire);
                }

                return out;
            }

            struct CanonicalChunkSlot
            {
                std::vector<std::uint8_t>
                    bytes;

                std::size_t begin = 0;
                std::size_t count = 0;
            };

            std::vector<CanonicalChunkSlot>
                slots(slot_count);

            for (auto &slot : slots)
            {
                slot.bytes.resize(
                    maximum_chunk_bytes);
            }

            std::deque<std::size_t>
                free_slots;

            std::deque<std::size_t>
                ready_slots;

            for (std::size_t i = 0;
                 i < slot_count;
                 ++i)
            {
                free_slots.push_back(i);
            }

            std::mutex queue_mutex;
            std::condition_variable
                free_cv;
            std::condition_variable
                ready_cv;

            bool producer_done = false;
            std::atomic<bool>
                cancel_requested{false};

            auto validation_worker =
                [&]()
            {
                std::vector<std::uint8_t>
                    canonical_bytes;

                try
                {
                    canonical_bytes.resize(
                        point_size);
                }
                catch (...)
                {
                    record_worker_failure(
                        G1LoadWorkerFailure::
                            UnexpectedCanonical,
                        0);

                    cancel_requested.store(
                        true,
                        std::memory_order_release);

                    free_cv.notify_all();
                    ready_cv.notify_all();

                    return;
                }

                while (true)
                {
                    std::size_t slot_index = 0;

                    {
                        std::unique_lock<std::mutex>
                            lock(queue_mutex);

                        ready_cv.wait(
                            lock,
                            [&]()
                            {
                                return producer_done ||
                                       !ready_slots.empty();
                            });

                        if (ready_slots.empty())
                        {
                            return;
                        }

                        slot_index =
                            ready_slots.front();

                        ready_slots.pop_front();
                    }

                    auto &slot =
                        slots[slot_index];

                    if (!cancel_requested.load(
                            std::memory_order_acquire) &&
                        !worker_failed())
                    {
                        validate_chunk(
                            slot.bytes.data(),
                            slot.begin,
                            slot.count,
                            canonical_bytes);
                    }

                    {
                        std::lock_guard<std::mutex>
                            lock(queue_mutex);

                        free_slots.push_back(
                            slot_index);
                    }

                    free_cv.notify_one();
                }
            };

            std::vector<std::jthread>
                workers;

            workers.reserve(
                background_worker_count);

            try
            {
                for (std::size_t worker = 0;
                     worker <
                     background_worker_count;
                     ++worker)
                {
                    workers.emplace_back(
                        validation_worker);
                }
            }
            catch (...)
            {
                {
                    std::lock_guard<std::mutex>
                        lock(queue_mutex);

                    producer_done = true;
                }

                cancel_requested.store(
                    true,
                    std::memory_order_release);

                ready_cv.notify_all();
                free_cv.notify_all();

                throw;
            }

            std::exception_ptr
                producer_exception;

            bool prefix_complete = true;

            try
            {
                for (std::size_t begin = 0;
                     begin < load_count;
                     begin +=
                     kCanonicalWorkerChunkPoints)
                {
                    std::size_t slot_index = 0;

                    {
                        std::unique_lock<std::mutex>
                            lock(queue_mutex);

                        free_cv.wait(
                            lock,
                            [&]()
                            {
                                return worker_failed() ||
                                       !free_slots.empty();
                            });

                        if (worker_failed())
                        {
                            prefix_complete = false;
                            break;
                        }

                        slot_index =
                            free_slots.front();

                        free_slots.pop_front();
                    }

                    auto &slot =
                        slots[slot_index];

                    slot.begin = begin;

                    slot.count =
                        std::min<std::size_t>(
                            kCanonicalWorkerChunkPoints,
                            load_count - begin);

                    const auto current_bytes =
                        slot.count *
                        point_size;

                    read_hashed(
                        slot.bytes.data(),
                        current_bytes,
                        "canonical G1 prefix chunk");

                    {
                        std::lock_guard<std::mutex>
                            lock(queue_mutex);

                        ready_slots.push_back(
                            slot_index);
                    }

                    ready_cv.notify_one();
                }

                if (prefix_complete &&
                    !worker_failed())
                {
                    // Workers consume the exact prefix bytes already entered
                    // into SHA-256 while this thread authenticates the suffix.
                    hash_unloaded_suffix();
                }
            }
            catch (...)
            {
                producer_exception =
                    std::current_exception();

                cancel_requested.store(
                    true,
                    std::memory_order_release);
            }

            {
                std::lock_guard<std::mutex>
                    lock(queue_mutex);

                producer_done = true;
            }

            ready_cv.notify_all();
            free_cv.notify_all();

            // After I/O and hashing finish, the caller also drains validation
            // work instead of leaving one CPU unused.
            validation_worker();

            for (auto &worker : workers)
            {
                if (worker.joinable())
                {
                    worker.join();
                }
            }

            if (producer_exception)
            {
                std::rethrow_exception(
                    producer_exception);
            }

            const auto recorded_failure =
                static_cast<G1LoadWorkerFailure>(
                    worker_failure_kind.load(
                        std::memory_order_acquire));

            if (recorded_failure !=
                G1LoadWorkerFailure::None)
            {
                worker_failure.kind =
                    recorded_failure;

                worker_failure.index =
                    worker_failure_index.load(
                        std::memory_order_acquire);

                return out;
            }

            if (!prefix_complete)
            {
                throw std::runtime_error(
                    "canonical SRS G1 validation stopped before prefix authentication completed: " +
                    path.string());
            }

            unsigned char
                digest[SHA256_DIGEST_LENGTH];

            SHA256_Final(
                digest,
                &ctx);

            if (actual_hash != nullptr)
            {
                *actual_hash =
                    bytes_to_hex(
                        digest,
                        sizeof(digest));
            }

            return out;
        }

        std::vector<G1Point> load_g1_vector_file(
            const std::filesystem::path &path,
            std::string *actual_hash,
            std::size_t max_points)
        {
            std::ifstream input(path, std::ios::binary | std::ios::ate);
            if (!input)
            {
                throw std::runtime_error(
                    "failed to open canonical SRS G1 material: " +
                    path.string());
            }

            const auto file_size = input.tellg();
            constexpr std::streamoff kHeaderSize = 24;
            if (file_size < kHeaderSize ||
                static_cast<std::uintmax_t>(file_size) >
                    static_cast<std::uintmax_t>(
                        std::numeric_limits<std::size_t>::max()))
            {
                throw std::runtime_error(
                    "truncated canonical SRS G1 material: " +
                    path.string());
            }

            std::array<std::uint8_t, kHeaderSize> header{};
            input.seekg(0);
            input.read(
                reinterpret_cast<char *>(header.data()),
                static_cast<std::streamsize>(header.size()));
            if (input.gcount() !=
                static_cast<std::streamsize>(header.size()))
            {
                throw std::runtime_error(
                    "truncated canonical SRS G1 material header: " +
                    path.string());
            }

            const std::array<std::uint8_t, 8> expected_magic{{'G', 'A', 'T', 'Z', 'K', 'G', '1', 'C'}};
            if (std::memcmp(
                    header.data(),
                    expected_magic.data(),
                    expected_magic.size()) != 0)
            {
                throw std::runtime_error(
                    "canonical SRS G1 header mismatch: " +
                    path.string());
            }

            const auto point_count_u64 =
                load_u64_le_unaligned(header.data() + 8);
            const auto point_size_u64 =
                load_u64_le_unaligned(header.data() + 16);
            if (point_count_u64 >
                    static_cast<std::uint64_t>(
                        std::numeric_limits<std::size_t>::max()) ||
                point_size_u64 == 0 ||
                point_size_u64 >
                    static_cast<std::uint64_t>(
                        std::numeric_limits<std::size_t>::max()))
            {
                throw std::runtime_error(
                    "invalid canonical SRS G1 header: " +
                    path.string());
            }

            const auto point_count =
                static_cast<std::size_t>(point_count_u64);
            const auto point_size =
                static_cast<std::size_t>(point_size_u64);
            if (point_count >
                    (std::numeric_limits<std::size_t>::max() -
                     static_cast<std::size_t>(kHeaderSize)) /
                        point_size ||
                static_cast<std::size_t>(file_size) !=
                    static_cast<std::size_t>(kHeaderSize) +
                        point_count * point_size)
            {
                throw std::runtime_error(
                    "canonical SRS G1 material size mismatch: " +
                    path.string());
            }

            G1LoadWorkerFailureInfo worker_failure;
            auto out =
                load_pipelined_canonical_g1_vector_file_prefix(
                    path,
                    static_cast<std::size_t>(file_size),
                    point_count,
                    point_size,
                    max_points,
                    actual_hash,
                    worker_failure);
            if (worker_failure.kind !=
                G1LoadWorkerFailure::None)
            {
                throw std::runtime_error(
                    g1_load_worker_failure_message(
                        worker_failure,
                        path));
            }
            return out;
        }

        struct LoadedVerifierKey
        {
            G2Point g2_one;
            G2Point g2_tau;
            std::unordered_map<std::string, G2Point> domain_zero_g2;
        };

        LoadedVerifierKey load_vk_file(const std::filesystem::path &path, std::string *actual_hash = nullptr)
        {
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if (!file)
            {
                throw std::runtime_error("failed to open SRS verifier key: " + path.string());
            }
            const auto file_size = file.tellg();
            file.seekg(0);
            std::string file_bytes(static_cast<std::size_t>(file_size), '\0');
            file.read(file_bytes.data(), file_size);
            if (file.gcount() != file_size)
            {
                throw std::runtime_error("failed to read SRS verifier key: " + path.string());
            }
            if (actual_hash != nullptr)
            {
                *actual_hash = sha256_hex(file_bytes);
            }
            std::istringstream input(file_bytes, std::ios::binary);
            if (!input)
            {
                throw std::runtime_error("failed to open SRS verifier key: " + path.string());
            }
            std::string header;
            std::getline(input, header);
            if (header != "GATZK-VK")
            {
                throw std::runtime_error("invalid verifier key header: " + path.string());
            }
            auto read_g2 = [&]()
            {
                const auto size = read_u64_le(input, "G2 point size");
                std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
                input.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
                if (input.gcount() != static_cast<std::streamsize>(bytes.size()))
                {
                    throw std::runtime_error("truncated SRS G2 point payload");
                }
                auto point = deserialize_g2(bytes);
                if (point.value.isZero() || serialize(point) != bytes)
                {
                    throw std::runtime_error(
                        "invalid, non-canonical, non-subgroup, or identity SRS G2 point");
                }
                return point;
            };
            auto g2_one = read_g2();
            auto g2_tau = read_g2();
            const auto domain_count = read_u64_le(input, "domain zero G2 count");
            if (domain_count == 0 || domain_count > (1ULL << 20))
            {
                throw std::runtime_error("invalid verifier key domain zero G2 count");
            }
            std::unordered_map<std::string, G2Point> domain_zero_g2;
            for (std::uint64_t i = 0; i < domain_count; ++i)
            {
                const auto name_size = read_u64_le(input, "domain zero G2 name size");
                if (name_size == 0 || name_size > 1024)
                {
                    throw std::runtime_error("invalid verifier key domain name size");
                }
                std::string name(static_cast<std::size_t>(name_size), '\0');
                input.read(name.data(), static_cast<std::streamsize>(name.size()));
                if (input.gcount() != static_cast<std::streamsize>(name.size()))
                {
                    throw std::runtime_error("truncated verifier key domain name");
                }
                if (!domain_zero_g2.emplace(std::move(name), read_g2()).second)
                {
                    throw std::runtime_error("duplicate verifier key domain zero G2 entry");
                }
            }
            if (input.peek() != std::char_traits<char>::eof())
            {
                throw std::runtime_error("trailing bytes in verifier key");
            }
            return LoadedVerifierKey{
                .g2_one = std::move(g2_one),
                .g2_tau = std::move(g2_tau),
                .domain_zero_g2 = std::move(domain_zero_g2),
            };
        }

        void validate_manifest(
            const SRSManifest &manifest,
            const std::string &expected_srs_type,
            const std::string &expected_srs_id)
        {
            if (manifest.manifest_format != "zkgat_srs_manifest" || manifest.generator_id != "zkgat_srs_setup_tool")
            {
                throw std::runtime_error("non-canonical SRS manifest format");
            }
            if (manifest.setup_transcript_hash != manifest.ceremony_transcript_hash)
            {
                throw std::runtime_error(
                    "SRS setup and ceremony transcript bindings differ");
            }
            if (manifest.curve_id != "bls12_381" || manifest.field_modulus_id != "bls12_381_fr")
            {
                throw std::runtime_error("SRS curve or field mismatch");
            }
            if (manifest.srs_type != "trusted_srs")
            {
                throw std::runtime_error("secure mainline requires srs_type=trusted_srs");
            }
            if (!expected_srs_type.empty() && manifest.srs_type != expected_srs_type)
            {
                throw std::runtime_error("SRS type mismatch");
            }
            if (!expected_srs_id.empty() && manifest.srs_id != expected_srs_id)
            {
                throw std::runtime_error("SRS id mismatch");
            }
            if (manifest.kzg_assumption_profile_id != "shifted_kzg_divisibility" || manifest.kzg_assumption_degree_bound != manifest.degree_bound || manifest.kzg_assumption_security_bits != 128 || manifest.kzg_assumption_domain_catalog_hash != canonical_domain_catalog_hash(manifest.domain_sizes))
            {
                throw std::runtime_error("SRS shifted-KZG assumption profile is missing or inconsistent");
            }
            if (manifest.root_derivation_id != "bls12_381_fr_generator_5" || manifest.row_order != "ascending_powers_from_one" || manifest.domain_generators.size() != manifest.domain_sizes.size() || manifest.domain_generator_catalog_hash != canonical_domain_generator_catalog_hash(manifest.domain_sizes, manifest.domain_generators, manifest.root_derivation_id, manifest.row_order))
            {
                throw std::runtime_error("SRS domain generator catalogue is missing or inconsistent");
            }
            if (manifest.protocol_schema_id.empty() || manifest.max_instance_envelope_id.empty() || manifest.max_instance_envelope_hash.empty() || manifest.max_instance_envelope_id != manifest.max_instance_envelope_hash || manifest.max_instance_envelope.empty() || manifest.proof_schema_template_hash.empty() || manifest.observation_ledger_hash.empty() || manifest.random_identity_ledger_hash.empty() || manifest.security_budget_ledger_hash.empty())
            {
                throw std::runtime_error(
                    "SRS protocol compiler artifact binding is incomplete");
            }
            if (manifest.security_level != "120-bit")
                throw std::runtime_error(
                    "SRS manifest security target does not match compiler target");
            if (manifest.encoding != "canonical_mcl_bls12_381_fixed" || manifest.srs_audit_report_hash.empty() || manifest.srs_consistency_hash.empty() || manifest.srs_audit_report_hash != manifest.srs_consistency_hash || manifest.vanishing_g2_hash.empty())
            {
                throw std::runtime_error(
                    "SRS canonical encoding or consistency audit binding is missing");
            }
            for (const auto &[domain_name, domain_size] : manifest.domain_sizes)
            {
                if (domain_size == 0 || domain_size > manifest.degree_bound)
                {
                    throw std::runtime_error("SRS assumption profile does not cover a registered domain size");
                }
                const auto generator = manifest.domain_generators.find(domain_name);
                if (generator == manifest.domain_generators.end() || generator->second != canonical_field_hex(FieldElement::root_of_unity(domain_size)))
                {
                    throw std::runtime_error(
                        "SRS domain generator does not match canonical root derivation: " + domain_name);
                }
            }
        }

        double elapsed_ms(const Clock::time_point &start, const Clock::time_point &end)
        {
            return std::chrono::duration<double, std::milli>(end - start).count();
        }

        std::vector<FieldElement> folding_powers(const FieldElement &challenge, std::size_t count)
        {
            std::vector<FieldElement> out(count, FieldElement::one());
            for (std::size_t i = 1; i < count; ++i)
            {
                out[i] = out[i - 1] * challenge;
            }
            return out;
        }

        template <class Fn>
        void parallel_for_chunks_capped(
            std::size_t count,
            std::size_t min_parallel_count,
            std::size_t max_worker_count,
            Fn &&fn)
        {
            const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
            const auto available_workers =
                max_worker_count == 0 ? cpu_count : std::min<std::size_t>(cpu_count, max_worker_count);
            if (count < min_parallel_count || available_workers <= 1)
            {
                fn(0, count);
                return;
            }
            const auto task_count = std::min<std::size_t>(
                available_workers,
                std::max<std::size_t>(2, count / min_parallel_count));
            const auto chunk_size = (count + task_count - 1) / task_count;
            std::vector<std::future<void>> futures;
            futures.reserve(task_count);
            for (std::size_t task = 0; task < task_count; ++task)
            {
                const auto begin = task * chunk_size;
                const auto end = std::min(count, begin + chunk_size);
                if (begin >= end)
                {
                    break;
                }
                futures.push_back(std::async(std::launch::async, [&, begin, end]()
                                             { fn(begin, end); }));
            }
            for (auto &future : futures)
            {
                future.get();
            }
        }

        template <class Fn>
        void parallel_for_chunks(std::size_t count, std::size_t min_parallel_count, Fn &&fn)
        {
            parallel_for_chunks_capped(count, min_parallel_count, 0, std::forward<Fn>(fn));
        }

        std::string srs_domain_name(const std::string &domain_name)
        {
            // The manifest key is the authenticated entry_id.  Quotient and
            // transcript labels use `d_h`, but the canonical SRS catalogue
            // entry generated by the setup tool is `d`; never rewrite one
            // entry_id into another or select material by size alone.
            return domain_name;
        }

        std::string srs_exact_domain_name(const std::string &domain_name, std::size_t domain_size)
        {
            return srs_domain_name(domain_name) + "_" + std::to_string(domain_size);
        }

        std::string polynomial_basis_name(const algebra::Polynomial &polynomial)
        {
            return polynomial.basis == algebra::PolynomialBasis::Coefficient ? "coefficient" : "evaluation";
        }

        std::string polynomial_domain_name(const algebra::Polynomial &polynomial)
        {
            if (polynomial.domain == nullptr)
            {
                return "coef";
            }
            return polynomial.domain->name;
        }

        bool progress_enabled()
        {
            return std::getenv("GATZK_RUN_PROGRESS") != nullptr;
        }

        void progress_note(const std::string &label)
        {
            if (progress_enabled())
            {
                std::cerr << "[progress] " << label << '\n';
            }
        }

        std::size_t commit_batch_outer_worker_count(
            std::size_t object_count,
            std::size_t total_scalar_count,
            std::size_t max_scalar_count,
            std::size_t cpu_count)
        {
            if (!util::route2_options().fast_msm || object_count <= 1 || cpu_count < 4)
            {
                return 1;
            }

            const auto average_scalar_count = total_scalar_count / object_count;
            if (max_scalar_count >= (1ULL << 22) && max_scalar_count * 2 >= total_scalar_count)
            {
                return 1;
            }
            if (object_count <= 8 && average_scalar_count >= (1ULL << 17))
            {
                return std::min<std::size_t>(
                    object_count,
                    std::max<std::size_t>(1, std::min<std::size_t>(3, cpu_count / 8)));
            }
            // Route batches commonly contain two N-domain MSMs followed by
            // two larger edge-domain MSMs.  Four outer workers cap every
            // object at one quarter of the machine, then leave half the CPU
            // idle once the N-domain work finishes.  Two outer workers keep
            // both edge MSMs concurrent while allowing each to use half the
            // machine; the smaller objects are picked up by the same workers.
            if (object_count == 4 && max_scalar_count >= (1ULL << 13) && max_scalar_count <= (1ULL << 15) && total_scalar_count <= max_scalar_count * 3)
            {
                return std::min<std::size_t>(2, cpu_count / 2);
            }
            if (object_count <= 20 && max_scalar_count <= (1ULL << 15))
            {
                const auto by_cpu = std::max<std::size_t>(1, cpu_count / 3);
                const auto by_work = std::max<std::size_t>(
                    2, total_scalar_count / (1ULL << 12));
                return std::min<std::size_t>(
                    object_count,
                    std::max<std::size_t>(1, std::min<std::size_t>(6, std::min(by_cpu, by_work))));
            }

            const auto by_cpu = std::max<std::size_t>(1, cpu_count / 2);
            const auto by_work = std::max<std::size_t>(1, total_scalar_count / (1ULL << 15));
            return std::min<std::size_t>(
                object_count,
                std::min<std::size_t>(10, std::min(by_cpu, by_work)));
        }

        std::vector<std::size_t> commitment_work_order(
            const std::vector<std::pair<std::string, const algebra::Polynomial *>> &named_polynomials)
        {
            std::vector<std::size_t> work_order(named_polynomials.size());
            std::iota(work_order.begin(), work_order.end(), std::size_t{0});
            auto group_key = [&](std::size_t index)
            {
                const auto *polynomial = named_polynomials[index].second;
                return std::string(polynomial_basis_name(*polynomial)) + "|" + polynomial_domain_name(*polynomial) + "|" + std::to_string(polynomial->domain != nullptr ? polynomial->domain->size : 0);
            };
            std::stable_sort(
                work_order.begin(),
                work_order.end(),
                [&](std::size_t lhs, std::size_t rhs)
                {
                    const auto lhs_key = group_key(lhs);
                    const auto rhs_key = group_key(rhs);
                    if (lhs_key != rhs_key)
                    {
                        return lhs_key < rhs_key;
                    }
                    return named_polynomials[lhs].second->size() > named_polynomials[rhs].second->size();
                });
            return work_order;
        }

        bool should_fold_constant_evaluation_tail(std::size_t size, std::size_t tail_start)
        {
            if (size < 8192 || tail_start >= size)
            {
                return false;
            }
            const auto tail_length = size - tail_start;
            const auto min_tail_length = std::max<std::size_t>(4096, size / 8);
            return tail_length >= min_tail_length;
        }

        const algebra::
            CudaResidentCoefficientBuffer *
            validated_cuda_resident_coefficients(
                const algebra::Polynomial &polynomial)
        {
            const auto *resident =
                polynomial
                    .cuda_resident_coefficients
                    .get();

            if (resident == nullptr)
            {
                return nullptr;
            }

            if (polynomial.basis !=
                    algebra::PolynomialBasis::
                        Coefficient ||
                polynomial.shared_data != nullptr ||
                resident
                        ->device_montgomery_limbs ==
                    nullptr ||
                resident->device < 0 ||
                resident->logical_count == 0 ||
                resident->logical_count >
                    polynomial.size() ||
                resident->logical_count >
                    resident->capacity ||
                resident->nonzero_count >
                    resident->logical_count)
            {
                throw std::runtime_error(
                    "invalid CUDA resident coefficient metadata");
            }

            return resident;
        }

        void finalize_cuda_resident_coefficient_material(
            algebra::Polynomial *polynomial)
        {
            if (polynomial == nullptr)
            {
                throw std::runtime_error(
                    "null CUDA resident coefficient material");
            }

            const auto *resident =
                validated_cuda_resident_coefficients(
                    *polynomial);

            if (resident == nullptr)
            {
                return;
            }

            const auto logical_count =
                resident->logical_count;

            if (resident->host_transfer !=
                nullptr)
            {
                resident->host_transfer
                    ->wait();
            }

            polynomial->data.resize(
                logical_count);

            polynomial
                ->cuda_resident_coefficients
                .reset();
        }

        std::size_t estimated_commitment_msm_work(const algebra::Polynomial &polynomial)
        {
            if (const auto *resident =
                    validated_cuda_resident_coefficients(
                        polynomial);
                resident != nullptr)
            {
                return resident->nonzero_count;
            }

            const auto &values = polynomial.values();
            if (values.empty())
            {
                return 0;
            }

            if (polynomial.basis == algebra::PolynomialBasis::Evaluation && polynomial.domain != nullptr && values.size() == polynomial.domain->size)
            {
                std::size_t tail_start = values.size();
                const auto &tail_value = values.back();
                while (tail_start > 0 && values[tail_start - 1] == tail_value)
                {
                    --tail_start;
                }
                if (should_fold_constant_evaluation_tail(values.size(), tail_start))
                {
                    std::size_t work = tail_value.is_zero() ? 0 : 1;
                    for (std::size_t i = 0; i < tail_start; ++i)
                    {
                        work += values[i] == tail_value ? 0 : 1;
                    }
                    return work;
                }
            }

            return static_cast<std::size_t>(std::count_if(
                values.begin(),
                values.end(),
                [](const FieldElement &value)
                { return !value.is_zero(); }));
        }

        bool should_compact_lagrange_prefix_delta(
            const std::vector<FieldElement> &values,
            std::size_t prefix_size,
            const FieldElement &tail_value)
        {
            if (prefix_size < 4096)
            {
                return true;
            }
            constexpr std::size_t kTargetSamples = 4096;
            const auto stride = std::max<std::size_t>(1, prefix_size / kTargetSamples);
            std::size_t sampled = 0;
            std::size_t sampled_nonzero = 0;
            const auto &tail_native = tail_value.native();
            for (std::size_t i = 0; i < prefix_size && sampled < kTargetSamples; i += stride)
            {
                mcl::Fr delta;
                mcl::Fr::sub(delta, values[i].native(), tail_native);
                sampled_nonzero += delta.isZero() ? 0 : 1;
                ++sampled;
            }
            return sampled_nonzero * 4 < sampled * 3;
        }

        G1Point commit_lagrange_prefix_delta_dense(
            const std::vector<G1Point> &lagrange_srs,
            const std::vector<FieldElement> &values,
            std::size_t prefix_size,
            const FieldElement &tail_value,
            G1MSMProfile *profile,
            const G1MSMExecutionOptions *msm_options = nullptr)
        {
            thread_local std::vector<FieldElement> prefix_delta_scratch;
            prefix_delta_scratch.resize(prefix_size);
            const auto &tail_native = tail_value.native();
            auto *scratch_native = FieldElement::mutable_native_array_data(prefix_delta_scratch.data());
            for (std::size_t i = 0; i < prefix_size; ++i)
            {
                mcl::Fr::sub(scratch_native[i], values[i].native(), tail_native);
            }
            return g1_multi_scalar_mul_profiled(
                lagrange_srs.data(),
                prefix_delta_scratch.data(),
                prefix_size,
                profile,
                msm_options);
        }

        G1Point commit_lagrange_prefix_delta(
            const std::vector<G1Point> &lagrange_srs,
            const std::vector<FieldElement> &values,
            std::size_t prefix_size,
            const FieldElement &tail_value,
            G1MSMProfile *profile,
            const G1MSMExecutionOptions *msm_options = nullptr)
        {
            if (!should_compact_lagrange_prefix_delta(values, prefix_size, tail_value))
            {
                return commit_lagrange_prefix_delta_dense(
                    lagrange_srs,
                    values,
                    prefix_size,
                    tail_value,
                    profile,
                    msm_options);
            }

            thread_local std::vector<G1Point> compact_points;
            thread_local std::vector<FieldElement> compact_scalars;
            compact_points.clear();
            compact_scalars.clear();
            const auto reserve_hint = std::min<std::size_t>(prefix_size, 1ULL << 14);
            compact_points.reserve(reserve_hint);
            compact_scalars.reserve(reserve_hint);

            const auto &tail_native = tail_value.native();
            for (std::size_t i = 0; i < prefix_size; ++i)
            {
                mcl::Fr delta;
                mcl::Fr::sub(delta, values[i].native(), tail_native);
                if (delta.isZero())
                {
                    continue;
                }
                compact_points.push_back(lagrange_srs[i]);
                compact_scalars.push_back(FieldElement::from_native(delta));
            }

            if (compact_scalars.empty())
            {
                if (profile != nullptr)
                {
                    profile->input_count = prefix_size;
                    profile->nonzero_scalar_count = 0;
                    profile->effective_scalar_count = 0;
                    profile->zero_scalar_skipping = true;
                    profile->sparse_compaction = true;
                    profile->multithreaded = false;
                }
                return g1_zero();
            }

            G1MSMProfile compact_profile;
            const G1MSMExecutionOptions compact_msm_options{
                .max_worker_count =
                    msm_options != nullptr
                        ? msm_options->max_worker_count
                        : 0,
                // The compact point vector is temporary material, not the
                // immutable Lagrange SRS storage named by the caller.
                .stable_srs_bases = false,
                .stable_srs_capacity = 0,
            };
            auto out = g1_multi_scalar_mul_profiled(
                compact_points.data(),
                compact_scalars.data(),
                compact_scalars.size(),
                profile != nullptr ? &compact_profile : nullptr,
                &compact_msm_options);
            if (profile != nullptr)
            {
                profile->input_count = prefix_size;
                profile->nonzero_scalar_count = compact_scalars.size();
                profile->effective_scalar_count = compact_profile.effective_scalar_count;
                profile->zero_scalar_skipping = compact_scalars.size() != prefix_size || compact_profile.zero_scalar_skipping;
                profile->sparse_compaction = true;
                profile->multithreaded = compact_profile.multithreaded;
            }
            return out;
        }

        G1Point commit_zero_domain_multiple_point(
            std::size_t domain_size,
            const std::vector<FieldElement> &multiplier,
            const KZGProverKey &pk,
            std::string_view label)
        {
            if (pk.powers_g1 == nullptr || domain_size + multiplier.size() > pk.powers_g1->size() || domain_size + multiplier.size() > pk.degree_bound + 1)
            {
                throw std::runtime_error(
                    "zero-domain multiple exceeds coefficient SRS degree bound: " + std::string(label));
            }
            auto out = g1_zero();
            for (std::size_t i = 0; i < multiplier.size(); ++i)
            {
                if (multiplier[i].is_zero())
                {
                    continue;
                }
                const auto shifted = g1_sub(
                    (*pk.powers_g1)[domain_size + i],
                    (*pk.powers_g1)[i]);
                out = g1_add(out, g1_mul(shifted, multiplier[i]));
            }
            return out;
        }

        const std::vector<G1Point> *evaluation_srs_points(
            const algebra::Polynomial &polynomial,
            const KZGProverKey &pk)
        {
            if (polynomial.domain == nullptr)
            {
                throw std::runtime_error(
                    "evaluation polynomial is missing its domain: " + polynomial.name);
            }
            const auto base_key =
                srs_domain_name(polynomial.domain->name);
            const auto exact_key =
                srs_exact_domain_name(
                    polynomial.domain->name,
                    polynomial.domain->size);
            auto it = pk.lagrange_srs_by_domain.find(exact_key);
            if (it == pk.lagrange_srs_by_domain.end())
            {
                it = pk.lagrange_srs_by_domain.find(base_key);
                if (it != pk.lagrange_srs_by_domain.end() && it->second != nullptr && it->second->size() != polynomial.domain->size)
                {
                    it = pk.lagrange_srs_by_domain.end();
                }
            }
            if (it == pk.lagrange_srs_by_domain.end())
            {
                throw std::runtime_error(
                    "missing exact Lagrange SRS for domain: " + base_key + ":" + std::to_string(polynomial.domain->size));
            }
            if (it->second == nullptr)
            {
                throw std::runtime_error(
                    "missing Lagrange proving SRS material for domain: " + exact_key);
            }
            return it->second.get();
        }

        struct SameSrsCommitWaveKey
        {
            const G1Point *points = nullptr;
            std::size_t storage_capacity = 0;

            bool operator==(
                const SameSrsCommitWaveKey &other) const
            {
                return points == other.points &&
                       storage_capacity == other.storage_capacity;
            }
        };

        struct SameSrsCommitWave
        {
            SameSrsCommitWaveKey key;
            std::size_t max_scalar_count = 0;
            std::vector<std::size_t> indices;
        };

        struct SameSrsCommitWaveProfile
        {
            bool used_mixed_prefix = false;
            bool used_exact_count = false;
        };

        std::optional<SameSrsCommitWaveKey>
        same_srs_commit_wave_key(
            const algebra::Polynomial &polynomial,
            const KZGProverKey &pk,
            std::size_t estimated_work)
        {
            const auto &route =
                util::route2_options();

            if (!route.fast_msm ||
                !route.cuda_field_hotspots)
            {
                return std::nullopt;
            }

            constexpr std::size_t
                kMinimumExplicitWaveScalarCount =
                    std::size_t{1} << 12;

            if (polynomial.size() <
                    kMinimumExplicitWaveScalarCount ||
                polynomial.size() == 0)
            {
                return std::nullopt;
            }

            // CUDA execution preserves the immutable SRS point vector even
            // when the logical scalar vector is sparse or has a constant
            // tail.  Keep such objects eligible for same-SRS scheduling;
            // the lower MSM execution layer owns the exact scalar route.
            (void)estimated_work;

            if (polynomial.basis ==
                algebra::PolynomialBasis::Coefficient)
            {
                if (pk.powers_g1 == nullptr)
                {
                    throw std::runtime_error(
                        "missing coefficient proving SRS");
                }

                if (polynomial.size() >
                        pk.powers_g1->size() ||
                    polynomial.size() >
                        pk.degree_bound + 1)
                {
                    throw std::runtime_error(
                        "coefficient polynomial exceeds "
                        "SRS degree bound: " +
                        polynomial.name);
                }

                return SameSrsCommitWaveKey{
                    .points =
                        pk.powers_g1->data(),
                    .storage_capacity =
                        pk.powers_g1->size(),
                };
            }

            const auto *lagrange_srs =
                evaluation_srs_points(
                    polynomial,
                    pk);

            if (polynomial.size() >
                lagrange_srs->size())
            {
                throw std::runtime_error(
                    "evaluation polynomial exceeds "
                    "Lagrange SRS domain size: " +
                    polynomial.name);
            }

            return SameSrsCommitWaveKey{
                .points =
                    lagrange_srs->data(),
                .storage_capacity =
                    lagrange_srs->size(),
            };
        }

        template <typename CommitOne>
        std::vector<bool>
        execute_cuda_same_srs_commit_waves(
            const std::vector<
                std::pair<
                    std::string,
                    const algebra::Polynomial *>>
                &named_polynomials,
            const KZGProverKey &pk,
            const std::vector<std::size_t>
                &estimated_work,
            const std::vector<bool>
                &excluded,
            CommitOne &&commit_one,
            SameSrsCommitWaveProfile *wave_profile = nullptr)
        {
            if (named_polynomials.size() !=
                    estimated_work.size() ||
                named_polynomials.size() !=
                    excluded.size())
            {
                throw std::runtime_error(
                    "same-SRS commitment wave "
                    "metadata shape mismatch");
            }

            std::vector<bool>
                completed(
                    named_polynomials.size(),
                    false);

            if (!util::route2_options()
                     .cuda_field_hotspots ||
                !util::route2_options()
                     .fast_msm)
            {
                return completed;
            }

            std::vector<SameSrsCommitWave>
                waves;

            waves.reserve(
                named_polynomials.size());

            for (std::size_t index = 0;
                 index <
                 named_polynomials.size();
                 ++index)
            {
                if (excluded[index])
                {
                    continue;
                }

                const auto *polynomial =
                    named_polynomials[index]
                        .second;

                if (polynomial == nullptr)
                {
                    throw std::runtime_error(
                        "null polynomial in "
                        "same-SRS commitment wave");
                }

                const auto key =
                    same_srs_commit_wave_key(
                        *polynomial,
                        pk,
                        estimated_work[index]);

                if (!key.has_value())
                {
                    continue;
                }

                auto found =
                    std::find_if(
                        waves.begin(),
                        waves.end(),
                        [&](const SameSrsCommitWave
                                &wave)
                        {
                            return wave.key ==
                                   *key;
                        });

                if (found ==
                    waves.end())
                {
                    waves.push_back(
                        SameSrsCommitWave{
                            .key = *key,
                            .max_scalar_count =
                                polynomial->size(),
                            .indices = {
                                index},
                        });
                }
                else
                {
                    found->indices
                        .push_back(
                            index);
                    found->max_scalar_count =
                        std::max(
                            found->max_scalar_count,
                            polynomial->size());
                }
            }

            // Large prefixes first.  Preparing the maximum useful prefix for
            // one immutable SRS storage allows later smaller prefixes to hit
            // the already-resident base cache instead of upgrading it again.
            std::stable_sort(
                waves.begin(),
                waves.end(),
                [](const SameSrsCommitWave &lhs,
                   const SameSrsCommitWave &rhs)
                {
                    return lhs.max_scalar_count >
                           rhs.max_scalar_count;
                });

            const std::size_t
                kMaximumCudaSameSrsWaveSize =
                    util::route2_options()
                                .cuda_field_hotspots &&
                            cuda_g1_msm_integrated_memory_device()
                        ? 16
                        : 8;

            for (auto &wave : waves)
            {
                if (wave.indices.size() < 2)
                {
                    continue;
                }

                // Compare the actual launch count and scalar padding before
                // using a mixed-length prefix batch.  All arithmetic here is
                // checked because this planner runs on production metadata,
                // not only the small targeted fixtures.
                const auto checked_add =
                    [](std::size_t lhs,
                       std::size_t rhs,
                       const char *what)
                {
                    if (rhs >
                        std::numeric_limits<std::size_t>::max() - lhs)
                    {
                        throw std::runtime_error(
                            std::string("same-SRS wave ") +
                            what + " overflows");
                    }
                    return lhs + rhs;
                };
                const auto checked_mul =
                    [](std::size_t lhs,
                       std::size_t rhs,
                       const char *what)
                {
                    if (lhs != 0 &&
                        rhs >
                            std::numeric_limits<std::size_t>::max() / lhs)
                    {
                        throw std::runtime_error(
                            std::string("same-SRS wave ") +
                            what + " overflows");
                    }
                    return lhs * rhs;
                };
                const auto ceil_wave_calls =
                    [](std::size_t count,
                       std::size_t wave_size)
                {
                    return count / wave_size +
                           (count % wave_size != 0 ? 1 : 0);
                };

                std::unordered_map<std::size_t, std::size_t>
                    exact_group_counts;
                exact_group_counts.reserve(wave.indices.size());
                std::size_t total_scalar_count = 0;
                for (const auto index : wave.indices)
                {
                    const auto *polynomial =
                        named_polynomials[index].second;
                    if (polynomial == nullptr)
                    {
                        throw std::runtime_error(
                            "null polynomial in same-SRS wave accounting");
                    }
                    total_scalar_count = checked_add(
                        total_scalar_count,
                        polynomial->size(),
                        "scalar-count sum");
                    auto &group_count =
                        exact_group_counts[polynomial->size()];
                    group_count = checked_add(
                        group_count,
                        1,
                        "exact-count group size");
                }

                const auto candidate_calls =
                    ceil_wave_calls(
                        wave.indices.size(),
                        kMaximumCudaSameSrsWaveSize);
                std::size_t legacy_calls = 0;
                for (const auto &[scalar_count, group_count] :
                     exact_group_counts)
                {
                    (void)scalar_count;
                    legacy_calls = checked_add(
                        legacy_calls,
                        ceil_wave_calls(
                            group_count,
                            kMaximumCudaSameSrsWaveSize),
                        "legacy call count");
                }

                const auto padded_scalar_count =
                    checked_mul(
                        wave.indices.size(),
                        wave.max_scalar_count,
                        "padded scalar count");
                if (total_scalar_count > padded_scalar_count)
                {
                    throw std::runtime_error(
                        "same-SRS wave scalar count exceeds padded capacity");
                }
                const auto padding_scalar_count =
                    padded_scalar_count - total_scalar_count;
                const bool padding_within_half =
                    padding_scalar_count <=
                        std::numeric_limits<std::size_t>::max() / 2 &&
                    checked_mul(
                        padding_scalar_count,
                        2,
                        "padding ratio") <=
                        padded_scalar_count;
                const bool use_mixed_prefix =
                    candidate_calls < legacy_calls &&
                    padding_within_half;

                if (use_mixed_prefix)
                {
                    if (wave_profile != nullptr)
                    {
                        wave_profile->used_mixed_prefix = true;
                    }

                    std::size_t offset = 0;
                    while (offset < wave.indices.size())
                    {
                        const auto batch_size =
                            std::min<std::size_t>(
                                kMaximumCudaSameSrsWaveSize,
                                wave.indices.size() - offset);

                        if (batch_size < 2)
                        {
                            break;
                        }

                        g1_parallel_msm_batch_workers(
                            batch_size,
                            [&](std::size_t worker)
                            {
                                commit_one(
                                    wave.indices[offset + worker]);
                            });

                        for (std::size_t local = 0;
                             local < batch_size;
                             ++local)
                        {
                            completed[wave.indices[offset + local]] = true;
                        }

                        offset += batch_size;
                    }
                    continue;
                }

                // The prefix candidate is not worthwhile when it saves no
                // launch, or when it would pad more than half the scalars.
                // Preserve the former exact-count wave behavior inside this
                // same immutable-SRS owner instead of sending these objects
                // to the unrelated outer fallback scheduler.
                if (wave_profile != nullptr)
                {
                    wave_profile->used_exact_count = true;
                }
                std::vector<std::size_t> exact_order = wave.indices;
                std::stable_sort(
                    exact_order.begin(),
                    exact_order.end(),
                    [&](std::size_t lhs, std::size_t rhs)
                    {
                        return named_polynomials[lhs].second->size() >
                               named_polynomials[rhs].second->size();
                    });

                std::size_t offset = 0;
                while (offset < exact_order.size())
                {
                    const auto exact_count =
                        named_polynomials[exact_order[offset]]
                            .second->size();
                    std::size_t group_end = offset + 1;
                    while (group_end < exact_order.size() &&
                           named_polynomials[exact_order[group_end]]
                                   .second->size() == exact_count)
                    {
                        ++group_end;
                    }

                    while (offset < group_end)
                    {
                        const auto batch_size =
                            std::min<std::size_t>(
                                kMaximumCudaSameSrsWaveSize,
                                group_end - offset);
                        if (batch_size >= 2)
                        {
                            g1_parallel_msm_batch_workers(
                                batch_size,
                                [&](std::size_t worker)
                                {
                                    commit_one(
                                        exact_order[offset + worker]);
                                });
                        }
                        else
                        {
                            // A singleton is still part of the exact-count
                            // plan and must not be left to outer scheduling.
                            commit_one(exact_order[offset]);
                        }

                        for (std::size_t local = 0;
                             local < batch_size;
                             ++local)
                        {
                            completed[exact_order[offset + local]] = true;
                        }
                        offset += batch_size;
                    }
                }
            }

            return completed;
        }

        // KZG多项式承诺主函数
        G1Point commit_polynomial_point(
            const algebra::Polynomial &polynomial,
            const KZGProverKey &pk,
            CommitObjectProfile *profile = nullptr,
            const G1MSMExecutionOptions *msm_options = nullptr,
            EvaluationCommitmentCache *reuse_cache = nullptr)
        {
            G1MSMProfile msm_profile;
            const auto msm_start = Clock::now();
            G1MSMExecutionOptions stable_msm_options{
                .max_worker_count =
                    msm_options != nullptr
                        ? msm_options->max_worker_count
                        : 0,
                // Both coefficient and Lagrange commitment bases below are
                // immutable proving-key SRS storage.  Individual backends
                // retain their own size-aware preparation/cache thresholds.
                .stable_srs_bases = true,
                .stable_srs_capacity = 0,
            };
            G1Point point;
            bool constant_tail_folding = false;
            bool semantic_commitment_reuse = false;
            std::size_t committed_vector_length =
                polynomial.size();

            // 系数多项式采用SRS powers + MSM
            if (polynomial.basis == algebra::PolynomialBasis::Coefficient)
            {
                if (pk.powers_g1 == nullptr)
                {
                    throw std::runtime_error("missing coefficient proving SRS");
                }

                const auto *resident =
                    validated_cuda_resident_coefficients(
                        polynomial);

                if (resident != nullptr)
                {
                    committed_vector_length =
                        resident->logical_count;
                }

                if (committed_vector_length > pk.powers_g1->size() || committed_vector_length > pk.degree_bound + 1)
                {
                    throw std::runtime_error("coefficient polynomial exceeds SRS degree bound: " + polynomial.name);
                }

                stable_msm_options.stable_srs_capacity =
                    pk.powers_g1->size();

                if (resident != nullptr)
                {
                    stable_msm_options
                        .device_montgomery_scalars =
                        resident
                            ->device_montgomery_limbs;

                    stable_msm_options
                        .device_montgomery_scalar_capacity =
                        resident->capacity;

                    stable_msm_options
                        .device_montgomery_scalar_device =
                        resident->device;

                    stable_msm_options
                        .device_montgomery_scalar_nonzero_count_known =
                        true;

                    stable_msm_options
                        .device_montgomery_scalar_nonzero_count =
                        resident->nonzero_count;
                }

                point =
                    g1_multi_scalar_mul_profiled(
                        pk.powers_g1->data(),
                        polynomial.values().data(),
                        committed_vector_length,
                        profile != nullptr
                            ? &msm_profile
                            : nullptr,
                        &stable_msm_options);
            }
            else
            {
                // 评估多项式采用 拉格朗日SRS + MSM
                const auto *lagrange_srs =
                    evaluation_srs_points(polynomial, pk);
                stable_msm_options.stable_srs_capacity =
                    lagrange_srs->size();
                if (polynomial.size() > lagrange_srs->size())
                {
                    throw std::runtime_error("evaluation polynomial exceeds Lagrange SRS domain size: " + polynomial.name);
                }
                const auto &values = polynomial.values();
                if (reuse_cache != nullptr && polynomial.shared_data != nullptr && reuse_cache->lookup_semantic(polynomial.domain.get(), lagrange_srs, polynomial.shared_data, &point))
                {
                    semantic_commitment_reuse = true;
                    if (profile != nullptr)
                    {
                        msm_profile.input_count = polynomial.size();
                        msm_profile.nonzero_scalar_count =
                            estimated_commitment_msm_work(polynomial);
                        msm_profile.effective_scalar_count = 0;
                        msm_profile.zero_scalar_skipping = true;
                        msm_profile.sparse_compaction = false;
                        msm_profile.multithreaded = false;
                    }
                }
                else
                {
                    const auto can_fold_tail =
                        values.size() == polynomial.domain->size;
                    std::size_t tail_start = values.size();
                    if (can_fold_tail && !values.empty())
                    {
                        const auto &tail_value = values.back();
                        while (tail_start > 0 && values[tail_start - 1] == tail_value)
                        {
                            --tail_start;
                        }
                    }
                    if (can_fold_tail && should_fold_constant_evaluation_tail(
                                             values.size(), tail_start))
                    {
                        constant_tail_folding = true;
                        const auto &tail_value = values.back();
                        const auto tail_commitment = tail_value.is_zero()
                                                         ? g1_zero()
                                                         : g1_mul(pk.g1_generator, tail_value);
                        if (tail_start == 0)
                        {
                            point = tail_commitment;
                            if (profile != nullptr)
                            {
                                msm_profile.input_count = polynomial.size();
                                msm_profile.nonzero_scalar_count = 0;
                                msm_profile.effective_scalar_count = 0;
                                msm_profile.zero_scalar_skipping = true;
                                msm_profile.sparse_compaction = true;
                                msm_profile.multithreaded = false;
                            }
                        }
                        else
                        {
                            const auto folded_prefix_commitment =
                                tail_value.is_zero()
                                    ? g1_multi_scalar_mul_profiled(
                                          lagrange_srs->data(),
                                          values.data(),
                                          tail_start,
                                          profile != nullptr
                                              ? &msm_profile
                                              : nullptr,
                                          &stable_msm_options)
                                    : commit_lagrange_prefix_delta(
                                          *lagrange_srs,
                                          values,
                                          tail_start,
                                          tail_value,
                                          profile != nullptr
                                              ? &msm_profile
                                              : nullptr,
                                          &stable_msm_options);
                            if (profile != nullptr)
                            {
                                msm_profile.input_count = polynomial.size();
                            }
                            point = g1_add(
                                tail_commitment,
                                folded_prefix_commitment);
                        }
                    }
                    else
                    {
                        point = g1_multi_scalar_mul_profiled(
                            lagrange_srs->data(),
                            values.data(),
                            polynomial.size(),
                            profile != nullptr ? &msm_profile : nullptr,
                            &stable_msm_options);
                    }
                    if (reuse_cache != nullptr && polynomial.shared_data != nullptr)
                    {
                        reuse_cache->store_semantic(
                            polynomial.domain.get(),
                            lagrange_srs,
                            polynomial.shared_data,
                            point);
                    }
                }
                if (polynomial.is_zero_domain_blinded())
                {
                    point = g1_add(
                        point,
                        commit_zero_domain_multiple_point(
                            polynomial.domain->size,
                            polynomial.zero_domain_blinder,
                            pk,
                            polynomial.name));
                }
            }
            if (profile != nullptr)
            {
                profile->name = polynomial.name;
                profile->basis = polynomial_basis_name(polynomial);
                profile->domain = polynomial_domain_name(polynomial);
                profile->vector_length = committed_vector_length;
                profile->nonzero_scalar_count = msm_profile.nonzero_scalar_count;
                profile->effective_scalar_count = msm_profile.effective_scalar_count;
                profile->zero_scalar_skipping = msm_profile.zero_scalar_skipping;
                profile->sparse_compaction = msm_profile.sparse_compaction;
                profile->constant_tail_folding = constant_tail_folding;
                profile->multithreaded_msm = msm_profile.multithreaded;
                profile->semantic_commitment_reuse =
                    semantic_commitment_reuse;
                profile->msm_ms = elapsed_ms(msm_start, Clock::now());
            }
            return point;
        }

        G1Point commit_borrowed_evaluation_point(
            const std::string &name,
            const std::vector<FieldElement> &values,
            const std::shared_ptr<
                algebra::RootOfUnityDomain> &domain,
            const KZGProverKey &pk,
            CommitObjectProfile *profile = nullptr,
            const G1MSMExecutionOptions *msm_options = nullptr)
        {
            if (domain == nullptr)
            {
                throw std::runtime_error(
                    "borrowed evaluation commitment is missing its domain: " +
                    name);
            }

            if (values.size() != domain->size)
            {
                throw std::runtime_error(
                    "borrowed evaluation commitment/domain size mismatch: " +
                    name);
            }

            auto borrowed_values =
                std::shared_ptr<
                    const std::vector<FieldElement>>(
                    &values,
                    [](const std::vector<FieldElement> *) {});

            auto polynomial =
                algebra::Polynomial::
                    from_shared_evaluations(
                        name,
                        std::move(
                            borrowed_values),
                        domain);

            return commit_polynomial_point(
                polynomial,
                pk,
                profile,
                msm_options);
        }

        Commitment make_commitment(
            const std::string &name,
            algebra::Polynomial polynomial,
            const KZGProverKey &pk,
            CommitObjectProfile *profile = nullptr,
            const G1MSMExecutionOptions *msm_options = nullptr,
            EvaluationCommitmentCache *reuse_cache = nullptr)
        {
            polynomial.name =
                name;

            auto material =
                std::make_shared<
                    algebra::Polynomial>(
                    std::move(
                        polynomial));

            const auto point =
                commit_polynomial_point(
                    *material,
                    pk,
                    profile,
                    msm_options,
                    reuse_cache);

            finalize_cuda_resident_coefficient_material(
                material.get());

            return Commitment{
                .name = name,
                .point = point,
                .opening_polynomial =
                    material,
            };
        }

        algebra::Polynomial constant_polynomial(const std::string &name, const FieldElement &value)
        {
            return algebra::Polynomial::from_coefficients(name, {value});
        }

        bool coefficient_single_point_quotient_coefficients(
            std::vector<FieldElement> *quotient,
            const std::vector<FieldElement> &coefficients,
            const FieldElement &point,
            const FieldElement &value,
            std::size_t worker_cap = 0);

        bool coefficient_single_point_quotient_coefficients(
            std::vector<FieldElement> *quotient,
            const std::vector<FieldElement> &coefficients,
            const FieldElement &point,
            const FieldElement &value,
            std::size_t worker_cap)
        {
            quotient->clear();
            if (coefficients.empty())
            {
                if (!(value == FieldElement::zero()))
                {
                    throw std::runtime_error("claimed value does not match committed polynomial");
                }
                return false;
            }
            std::size_t coefficient_count = coefficients.size();
            while (coefficient_count > 1 && coefficients[coefficient_count - 1] == FieldElement::zero())
            {
                --coefficient_count;
            }
            if (coefficient_count == 1)
            {
                if (!(coefficients.front() == value))
                {
                    throw std::runtime_error("claimed value does not match committed polynomial");
                }
                return false;
            }
            quotient->resize(coefficient_count - 1);
            const auto *coefficients_native = FieldElement::native_array_data(coefficients.data());
            auto *quotient_native = FieldElement::mutable_native_array_data(quotient->data());
            const auto &point_native = point.native();
            const auto quotient_size = coefficient_count - 1;
            const auto available_workers = worker_cap == 0
                                               ? std::max<std::size_t>(1, std::thread::hardware_concurrency())
                                               : std::min<std::size_t>(
                                                     std::max<std::size_t>(1, std::thread::hardware_concurrency()),
                                                     worker_cap);
            constexpr std::size_t kParallelThreshold = 1ULL << 18;
            if (quotient_size < kParallelThreshold || available_workers <= 1)
            {
                quotient_native[quotient_size - 1] =
                    coefficients_native[coefficient_count - 1];
                for (std::size_t i = quotient_size - 1; i-- > 0;)
                {
                    mcl::Fr product;
                    mcl::Fr::mul(product, point_native, quotient_native[i + 1]);
                    mcl::Fr::add(
                        quotient_native[i], coefficients_native[i + 1], product);
                }
            }
            else
            {
                const auto requested_task_count = std::min<std::size_t>(
                    available_workers,
                    std::max<std::size_t>(
                        2, quotient_size / (1ULL << 17)));
                const auto block_size =
                    (quotient_size + requested_task_count - 1) / requested_task_count;
                const auto task_count =
                    (quotient_size + block_size - 1) / block_size;
                std::vector<mcl::Fr> block_constants(task_count);
                std::vector<mcl::Fr> block_multipliers(task_count);
                std::vector<std::future<void>> futures;
                futures.reserve(task_count);
                for (std::size_t task = 0; task < task_count; ++task)
                {
                    const auto begin = task * block_size;
                    const auto end = std::min(quotient_size, begin + block_size);
                    futures.push_back(std::async(
                        std::launch::async,
                        [&, task, begin, end]
                        {
                            mcl::Fr accumulated;
                            accumulated.clear();
                            for (std::size_t index = end; index-- > begin;)
                            {
                                mcl::Fr product;
                                mcl::Fr::mul(product, point_native, accumulated);
                                mcl::Fr::add(
                                    accumulated,
                                    coefficients_native[index + 1],
                                    product);
                            }
                            block_constants[task] = accumulated;
                            block_multipliers[task] = point.pow(
                                                               static_cast<std::uint64_t>(end - begin))
                                                          .native();
                        }));
                }
                for (auto &future : futures)
                {
                    future.get();
                }

                std::vector<mcl::Fr> block_incoming(task_count);
                mcl::Fr boundary;
                boundary.clear();
                for (std::size_t task = task_count; task-- > 0;)
                {
                    block_incoming[task] = boundary;
                    mcl::Fr product;
                    mcl::Fr::mul(
                        product, block_multipliers[task], boundary);
                    mcl::Fr::add(boundary, block_constants[task], product);
                }

                futures.clear();
                futures.reserve(task_count);
                for (std::size_t task = 0; task < task_count; ++task)
                {
                    const auto begin = task * block_size;
                    const auto end = std::min(quotient_size, begin + block_size);
                    futures.push_back(std::async(
                        std::launch::async,
                        [&, task, begin, end]
                        {
                            auto accumulated = block_incoming[task];
                            for (std::size_t index = end; index-- > begin;)
                            {
                                mcl::Fr product;
                                mcl::Fr::mul(product, point_native, accumulated);
                                mcl::Fr::add(
                                    accumulated,
                                    coefficients_native[index + 1],
                                    product);
                                quotient_native[index] = accumulated;
                            }
                        }));
                }
                for (auto &future : futures)
                {
                    future.get();
                }
            }
            mcl::Fr remainder;
            mcl::Fr product;
            mcl::Fr::sub(remainder, coefficients_native[0], value.native());
            mcl::Fr::mul(product, point_native, quotient_native[0]);
            mcl::Fr::add(remainder, remainder, product);
            if (!remainder.isZero())
            {
                throw std::runtime_error("claimed value does not match committed polynomial");
            }
            return true;
        }

        G1Point commit_coefficient_single_point_quotient(
            const std::vector<FieldElement> &coefficients,
            const FieldElement &point,
            const FieldElement &value,
            const KZGProverKey &pk,
            std::size_t worker_cap,
            const G1MSMExecutionOptions *msm_options = nullptr)
        {
            if (pk.powers_g1 == nullptr)
            {
                throw std::runtime_error("missing coefficient proving SRS");
            }
            thread_local std::vector<FieldElement> quotient;
            if (!coefficient_single_point_quotient_coefficients(
                    &quotient, coefficients, point, value, worker_cap))
            {
                return g1_zero();
            }
            if (quotient.size() > pk.powers_g1->size() || quotient.size() > pk.degree_bound + 1)
            {
                throw std::runtime_error("coefficient witness quotient exceeds SRS degree bound");
            }
            const G1MSMExecutionOptions stable_msm_options{
                .max_worker_count =
                    msm_options != nullptr
                        ? msm_options->max_worker_count
                        : 0,
                .stable_srs_bases = true,
                .stable_srs_capacity = pk.powers_g1->size(),
            };
            return g1_multi_scalar_mul_profiled(
                pk.powers_g1->data(),
                quotient.data(),
                quotient.size(),
                nullptr,
                &stable_msm_options);
        }

        void accumulate_scaled_coefficients(
            std::vector<FieldElement> *target,
            const std::vector<FieldElement> &source,
            const FieldElement &scale,
            std::size_t worker_cap)
        {
            if (scale.is_zero())
            {
                return;
            }
            if (target->empty() && scale == FieldElement::one())
            {
                *target = source;
                return;
            }
            if (target->size() < source.size())
            {
                target->resize(source.size(), FieldElement::zero());
            }
            auto *target_native = FieldElement::mutable_native_array_data(target->data());
            const auto *source_native = FieldElement::native_array_data(source.data());
            const auto &scale_native = scale.native();
            parallel_for_chunks_capped(source.size(), 1ULL << 15, worker_cap, [&](std::size_t begin, std::size_t end)
                                       {
                for (std::size_t i = begin; i < end; ++i)
                {
                    mcl::Fr product;
                    mcl::Fr::mul(product, source_native[i], scale_native);
                    mcl::Fr::add(target_native[i], target_native[i], product);
                } });
        }

        std::string coefficient_opening_failure_diagnostic(
            const std::vector<Commitment> &commitments,
            const std::vector<FieldElement> &claimed_values,
            const FieldElement &point,
            const std::vector<FieldElement> &folded_coefficients,
            const FieldElement &folded_claim)
        {
            try
            {
                if (commitments.size() != claimed_values.size())
                {
                    return " [diagnostic=input_shape_mismatch]";
                }
                std::size_t unavailable = 0;
                std::size_t checked = 0;
                for (std::size_t i = 0; i < commitments.size(); ++i)
                {
                    const auto &material = commitments[i].opening_polynomial;
                    if (material == nullptr)
                    {
                        ++unavailable;
                        continue;
                    }
                    if (material->basis != algebra::PolynomialBasis::Coefficient)
                    {
                        continue;
                    }
                    ++checked;
                    if (!(material->evaluate(point) == claimed_values[i]))
                    {
                        return " [diagnostic=coefficient_claim_mismatch; object=" +
                               commitments[i].name +
                               "; claim_index=" + std::to_string(i) + "]";
                    }
                }
                const bool folded_matches =
                    algebra::horner(folded_coefficients, point) == folded_claim;
                return std::string(" [diagnostic=") +
                       (folded_matches
                            ? "fold_matches_claim_check_quotient_or_msm_error"
                            : "fold_mismatch_check_accumulation_and_claim_sources") +
                       "; checked_coefficient_objects=" + std::to_string(checked) +
                       "; unavailable_materials=" + std::to_string(unavailable) + "]";
            }
            catch (...)
            {
                return " [diagnostic=unavailable_original_failure_preserved]";
            }
        }

        bool all_zero(const std::vector<FieldElement> &values)
        {
            return std::all_of(values.begin(), values.end(), [](const FieldElement &value)
                               { return value == FieldElement::zero(); });
        }

        void trim_zero_coefficients(std::vector<FieldElement> *coefficients)
        {
            while (coefficients->size() > 1 && coefficients->back() == FieldElement::zero())
            {
                coefficients->pop_back();
            }
            if (coefficients->size() == 1 && coefficients->front() == FieldElement::zero())
            {
                coefficients->clear();
            }
        }

        struct FoldedEvaluationTerm
        {
            const std::vector<FieldElement> *values = nullptr;
            const std::vector<FieldElement> *zero_domain_blinder = nullptr;
            std::size_t value_count = 0;
            FieldElement scale = FieldElement::zero();
        };

        struct FoldedEvaluationGroup
        {
            std::shared_ptr<algebra::RootOfUnityDomain> domain;
            std::vector<FoldedEvaluationTerm> terms;
            std::vector<FieldElement> folded_values;
            std::vector<FieldElement> folded_zero_domain_blinder;
            std::vector<FieldElement> scaled_claim_sums;

            bool folded_values_pristine =
                true;

            std::unique_ptr<
                algebra::CudaScaledAccumulateSession>
                cuda_fold_session;

            std::shared_ptr<
                algebra::CudaScaledAccumulatePairSession>
                cuda_pair_fold_session;
        };

        void finish_cuda_fold_session(
            FoldedEvaluationGroup *group)
        {
            if (group == nullptr)
            {
                return;
            }

            if (group->cuda_pair_fold_session !=
                nullptr)
            {
                group->cuda_pair_fold_session
                    ->finish();

                group->cuda_pair_fold_session
                    .reset();
            }

            if (group->cuda_fold_session !=
                nullptr)
            {
                group->cuda_fold_session
                    ->finish();

                group->cuda_fold_session
                    .reset();
            }
        }

        struct FoldedOpeningPlan
        {
            std::vector<FieldElement> coefficient_fold;
            std::vector<FieldElement> coefficient_scaled_claim_sums;
            std::vector<FoldedEvaluationGroup> evaluation_groups;

            bool coefficient_fold_pristine =
                true;

            std::unique_ptr<
                algebra::CudaScaledAccumulateSession>
                cuda_coefficient_fold_session;
        };

        void finish_cuda_coefficient_fold_session(
            FoldedOpeningPlan *plan)
        {
            if (plan == nullptr ||
                plan->cuda_coefficient_fold_session ==
                    nullptr)
            {
                return;
            }

            plan->cuda_coefficient_fold_session
                ->finish();

            plan->cuda_coefficient_fold_session
                .reset();
        }

        void accumulate_coefficient_fold(
            FoldedOpeningPlan *plan,
            const std::vector<FieldElement> &source,
            const FieldElement &scale,
            std::size_t worker_cap,
            bool source_is_ephemeral = false)
        {
            if (plan == nullptr)
            {
                throw std::runtime_error(
                    "null coefficient opening fold plan");
            }

            if (scale.is_zero())
            {
                return;
            }

            if (plan->coefficient_fold.empty() &&
                scale == FieldElement::one())
            {
                plan->coefficient_fold =
                    source;

                plan->coefficient_fold_pristine =
                    false;

                return;
            }

            if (plan->coefficient_fold.size() <
                source.size())
            {
                finish_cuda_coefficient_fold_session(
                    plan);

                plan->coefficient_fold.resize(
                    source.size(),
                    FieldElement::zero());
            }

            auto *target_native =
                FieldElement::mutable_native_array_data(
                    plan->coefficient_fold.data());

            const auto *source_native =
                FieldElement::native_array_data(
                    source.data());

            if (plan->cuda_coefficient_fold_session ==
                nullptr)
            {
                plan->cuda_coefficient_fold_session =
                    algebra::
                        create_cuda_scaled_accumulate_session(
                            target_native,
                            plan->coefficient_fold.size(),
                            plan->coefficient_fold_pristine);
            }

            if (plan->cuda_coefficient_fold_session !=
                nullptr)
            {
                plan->cuda_coefficient_fold_session
                    ->accumulate_range(
                        source_native,
                        scale.native(),
                        0,
                        source.size(),
                        source_is_ephemeral);

                plan->coefficient_fold_pristine =
                    false;

                return;
            }

            accumulate_scaled_coefficients(
                &plan->coefficient_fold,
                source,
                scale,
                worker_cap);

            plan->coefficient_fold_pristine =
                false;
        }

        struct DeferredStreamedSemanticSource
        {
            OpeningSemanticSourceInfo info;
            Commitment representative;
            FieldElement scale =
                FieldElement::zero();
        };

        using StreamedSemanticSourceDefer =
            std::function<void(
                const DeferredStreamedSemanticSource &)>;

        struct PairedStreamedFoldBuild
        {
            std::array<FoldedOpeningPlan, 2> plans;
            std::size_t fused_semantic_source_count = 0;
            std::array<std::size_t, 2>
                deferred_source_count{0, 0};
        };

        FoldedEvaluationGroup &
        require_folded_evaluation_group(
            FoldedOpeningPlan *plan,
            const std::shared_ptr<
                algebra::RootOfUnityDomain> &domain)
        {
            if (plan == nullptr || domain == nullptr)
            {
                throw std::runtime_error(
                    "streamed opening is missing its folded evaluation group");
            }

            const auto it =
                std::find_if(
                    plan->evaluation_groups.begin(),
                    plan->evaluation_groups.end(),
                    [&](const FoldedEvaluationGroup &group)
                    {
                        return group.domain != nullptr &&
                               group.domain->name ==
                                   domain->name &&
                               group.domain->size ==
                                   domain->size &&
                               group.domain->omega ==
                                   domain->omega;
                    });

            if (it == plan->evaluation_groups.end())
            {
                throw std::runtime_error(
                    "streamed semantic source is missing its folded domain group: " +
                    domain->name);
            }

            return *it;
        }

        std::shared_ptr<const algebra::Polynomial>
        resolve_streamed_semantic_source_material(
            const OpeningPolynomialResolver &resolver,
            const Commitment &commitment,
            const OpeningSemanticSourceInfo &info)
        {
            auto material =
                resolver(commitment);

            if (material == nullptr)
            {
                throw std::runtime_error(
                    "commitment is missing prover-side opening polynomial: " +
                    commitment.name);
            }

            if (material->basis !=
                    algebra::PolynomialBasis::Evaluation ||
                material->domain == nullptr)
            {
                throw std::runtime_error(
                    "streamed semantic opening source is not an evaluation polynomial: " +
                    commitment.name);
            }

            if (info.domain == nullptr ||
                material->domain->name !=
                    info.domain->name ||
                material->domain->size !=
                    info.domain->size ||
                !(material->domain->omega ==
                  info.domain->omega))
            {
                throw std::runtime_error(
                    "streamed semantic opening source domain mismatch: " +
                    commitment.name);
            }

            if (material->values().size() !=
                info.value_count)
            {
                throw std::runtime_error(
                    "streamed semantic opening source size mismatch: " +
                    commitment.name);
            }

            if (material->zero_domain_blinder !=
                info.zero_domain_blinder)
            {
                throw std::runtime_error(
                    "streamed semantic opening representative blinder mismatch: " +
                    commitment.name);
            }

            return material;
        }
        void accumulate_streamed_evaluation_source(
            FoldedEvaluationGroup *group,
            const std::vector<FieldElement> &source_values,
            const FieldElement &scale,
            const std::string &label,
            std::size_t worker_cap,
            std::size_t target_offset = 0,
            bool source_is_ephemeral = false)
        {
            if (group == nullptr ||
                target_offset >
                    group->folded_values.size() ||
                source_values.size() >
                    group->folded_values.size() -
                        target_offset)
            {
                throw std::runtime_error(
                    "streamed evaluation source range "
                    "exceeds folded target: " +
                    label);
            }

            if (source_values.size() >
                group->folded_values.size())
            {
                throw std::runtime_error(
                    "streamed evaluation polynomial exceeds its domain: " +
                    label);
            }

            if (scale.is_zero())
            {
                return;
            }

            if (group->cuda_pair_fold_session !=
                nullptr)
            {
                finish_cuda_fold_session(
                    group);
            }

            auto *folded_base_native =
                FieldElement::mutable_native_array_data(
                    group->folded_values.data());

            auto *folded_native =
                folded_base_native +
                target_offset;

            const auto *source_native =
                FieldElement::native_array_data(
                    source_values.data());

            const auto scale_native =
                scale.native();

            if (group->cuda_fold_session ==
                nullptr)
            {
                group->cuda_fold_session =
                    algebra::
                        create_cuda_scaled_accumulate_session(
                            folded_base_native,
                            group->folded_values.size(),
                            group->folded_values_pristine);
            }

            if (group->cuda_fold_session !=
                nullptr)
            {
                group->cuda_fold_session
                    ->accumulate_range(
                        source_native,
                        scale_native,
                        target_offset,
                        source_values.size(),
                        source_is_ephemeral);

                group->folded_values_pristine =
                    false;

                return;
            }

            if (!algebra::try_cuda_scaled_accumulate_native(
                    folded_native,
                    source_native,
                    scale_native,
                    source_values.size()))
            {
                parallel_for_chunks_capped(
                    source_values.size(),
                    1ULL << 15,
                    worker_cap,
                    [&](std::size_t begin,
                        std::size_t end)
                    {
                        for (std::size_t row = begin;
                             row < end;
                             ++row)
                        {
                            mcl::Fr product;

                            mcl::Fr::mul(
                                product,
                                source_native[row],
                                scale_native);

                            mcl::Fr::add(
                                folded_native[row],
                                folded_native[row],
                                product);
                        }
                    });
            }

            group->folded_values_pristine =
                false;
        }

        void accumulate_streamed_evaluation_source_pair(
            FoldedEvaluationGroup *first_group,
            FoldedEvaluationGroup *second_group,
            const std::vector<FieldElement> &source_values,
            const FieldElement &first_scale,
            const FieldElement &second_scale,
            const std::string &label,
            std::size_t worker_cap,
            std::size_t target_offset = 0,
            bool source_is_ephemeral = false)
        {
            if (first_group == nullptr ||
                second_group == nullptr ||
                first_group == second_group)
            {
                throw std::runtime_error(
                    "paired streamed opening "
                    "requires two distinct "
                    "folded groups");
            }

            if (target_offset >
                    first_group->folded_values.size() ||
                target_offset >
                    second_group->folded_values.size() ||
                source_values.size() >
                    first_group->folded_values.size() -
                        target_offset ||
                source_values.size() >
                    second_group->folded_values.size() -
                        target_offset)
            {
                throw std::runtime_error(
                    "paired streamed evaluation "
                    "source exceeds its target "
                    "range: " +
                    label);
            }

            if (first_scale.is_zero())
            {
                accumulate_streamed_evaluation_source(
                    second_group,
                    source_values,
                    second_scale,
                    label,
                    worker_cap,
                    target_offset,
                    source_is_ephemeral);

                return;
            }

            if (second_scale.is_zero())
            {
                accumulate_streamed_evaluation_source(
                    first_group,
                    source_values,
                    first_scale,
                    label,
                    worker_cap,
                    target_offset,
                    source_is_ephemeral);

                return;
            }

            const auto shared_pair_session =
                first_group
                    ->cuda_pair_fold_session;

            if (shared_pair_session != nullptr &&
                shared_pair_session ==
                    second_group
                        ->cuda_pair_fold_session &&
                !shared_pair_session->finished())
            {
                shared_pair_session
                    ->accumulate_range(
                        FieldElement::
                            native_array_data(
                                source_values.data()),
                        first_scale.native(),
                        second_scale.native(),
                        target_offset,
                        source_values.size(),
                        source_is_ephemeral);

                first_group->folded_values_pristine =
                    false;

                second_group->folded_values_pristine =
                    false;

                return;
            }

            finish_cuda_fold_session(
                first_group);

            finish_cuda_fold_session(
                second_group);

            auto *first_base_native =
                FieldElement::
                    mutable_native_array_data(
                        first_group->folded_values
                            .data());

            auto *second_base_native =
                FieldElement::
                    mutable_native_array_data(
                        second_group->folded_values
                            .data());

            const auto *source_native =
                FieldElement::
                    native_array_data(
                        source_values.data());

            if (first_group->folded_values.size() ==
                second_group->folded_values.size())
            {
                auto pair_session =
                    algebra::
                        create_cuda_scaled_accumulate_pair_session(
                            first_base_native,
                            second_base_native,
                            first_group->folded_values
                                .size(),
                            first_group
                                ->folded_values_pristine,
                            second_group
                                ->folded_values_pristine);

                if (pair_session != nullptr)
                {
                    first_group
                        ->cuda_pair_fold_session =
                        pair_session;

                    second_group
                        ->cuda_pair_fold_session =
                        pair_session;

                    pair_session
                        ->accumulate_range(
                            source_native,
                            first_scale.native(),
                            second_scale.native(),
                            target_offset,
                            source_values.size(),
                            source_is_ephemeral);

                    first_group->folded_values_pristine =
                        false;

                    second_group->folded_values_pristine =
                        false;

                    return;
                }
            }

            auto *first_native =
                first_base_native +
                target_offset;

            auto *second_native =
                second_base_native +
                target_offset;

            const auto first_scale_native =
                first_scale.native();

            const auto second_scale_native =
                second_scale.native();

            if (!algebra::
                    try_cuda_scaled_accumulate_pair_native(
                        first_native,
                        second_native,
                        source_native,
                        first_scale_native,
                        second_scale_native,
                        source_values.size()))
            {
                parallel_for_chunks_capped(
                    source_values.size(),
                    1ULL << 15,
                    worker_cap,
                    [&](std::size_t begin,
                        std::size_t end)
                    {
                        for (std::size_t row =
                                 begin;
                             row < end;
                             ++row)
                        {
                            const auto source =
                                source_native[row];

                            mcl::Fr
                                first_product;

                            mcl::Fr
                                second_product;

                            mcl::Fr::mul(
                                first_product,
                                source,
                                first_scale_native);

                            mcl::Fr::mul(
                                second_product,
                                source,
                                second_scale_native);

                            mcl::Fr::add(
                                first_native[row],
                                first_native[row],
                                first_product);

                            mcl::Fr::add(
                                second_native[row],
                                second_native[row],
                                second_product);
                        }
                    });
            }

            first_group->folded_values_pristine =
                false;

            second_group->folded_values_pristine =
                false;
        }
        std::size_t validated_file_backed_semantic_source_count(
            const OpeningSemanticSourceInfo &info,
            std::string_view operation)
        {
            if (!info.file_backed || info.file_path.empty() ||
                info.domain == nullptr ||
                info.value_count > info.domain->size)
            {
                throw std::runtime_error(
                    std::string(operation) +
                    " metadata is incomplete");
            }

            // An uncompacted descriptor has a canonical full-length file.
            // Keep zero as the legacy/default sentinel, but reject any other
            // contradictory stored count instead of silently selecting a
            // different physical representation.
            if (!info.zero_tail_compacted)
            {
                if (info.stored_value_count != 0 &&
                    info.stored_value_count != info.value_count)
                {
                    throw std::runtime_error(
                        std::string(operation) +
                        " metadata has an inconsistent stored count for an uncompacted source");
                }
                return info.value_count;
            }

            if (info.stored_value_count > info.value_count)
            {
                throw std::runtime_error(
                    std::string(operation) +
                    " metadata has a physical count larger than its logical count");
            }
            return info.stored_value_count;
        }

        bool compatible_streamed_semantic_source_metadata(
            const OpeningSemanticSourceInfo &first,
            const OpeningSemanticSourceInfo &second)
        {
            if (first.domain == nullptr || second.domain == nullptr ||
                first.domain->name != second.domain->name ||
                first.domain->size != second.domain->size ||
                !(first.domain->omega == second.domain->omega) ||
                first.value_count != second.value_count ||
                first.file_backed != second.file_backed)
            {
                return false;
            }

            if (!first.file_backed)
            {
                return true;
            }

            if (first.lease == nullptr || second.lease == nullptr ||
                first.lease.get() != second.lease.get())
            {
                return false;
            }

            const auto first_stored_count =
                validated_file_backed_semantic_source_count(
                    first,
                    "streamed semantic source compatibility");
            const auto second_stored_count =
                validated_file_backed_semantic_source_count(
                    second,
                    "streamed semantic source compatibility");

            // A shared scan is valid only when both descriptors authenticate
            // the same physical representation.  In particular, a compacted
            // prefix and a full-length file (or two different paths) must be
            // scanned independently rather than silently combining a
            // representative descriptor with the other source's bytes.
            return first_stored_count == second_stored_count &&
                   first.zero_tail_compacted == second.zero_tail_compacted &&
                   first.file_path == second.file_path;
        }

        template <typename Consumer>
        void for_each_file_backed_semantic_source_chunk(
            const OpeningSemanticSourceInfo &info,
            const char *operation,
            Consumer &&consumer)
        {
            if (!info.file_backed ||
                info.file_path.empty())
            {
                throw std::runtime_error(
                    std::string(operation) +
                    " metadata is incomplete");
            }

            constexpr std::size_t kChunkSize =
                std::size_t{1} << 20;

            const auto stored_value_count =
                validated_file_backed_semantic_source_count(
                    info,
                    operation);
            if (info.lease == nullptr)
            {
                throw std::runtime_error(
                    std::string(operation) +
                    " has no authenticated trace spill lease");
            }
            const auto payload_name =
                info.lease->payload_name(info.file_path);
            if (info.lease->payload_size(payload_name) !=
                algebra::field_elements_binary_size(stored_value_count))
            {
                throw std::runtime_error(
                    std::string(operation) +
                    " field-element spill length mismatch: " +
                    info.file_path);
            }

            auto stream =
                info.lease->open_payload_stream(payload_name);
            if (!stream || !stream->good())
            {
                throw std::runtime_error(
                    "failed to open " +
                    std::string(operation) +
                    ": " +
                    info.file_path);
            }

            std::vector<FieldElement> chunk;
            chunk.reserve(
                std::min(
                    kChunkSize,
                    stored_value_count));

            for (std::size_t offset = 0;
                 offset < stored_value_count;)
            {
                const auto count =
                    std::min(
                        kChunkSize,
                        stored_value_count - offset);

                chunk.resize(count);

                try
                {
                    algebra::read_field_elements_binary_into(
                        *stream,
                        std::span<FieldElement>(
                            chunk.data(),
                            chunk.size()));
                }
                catch (const std::exception &error)
                {
                    throw std::runtime_error(
                        "failed to read " +
                        std::string(operation) +
                        ": " +
                        info.file_path +
                        ": " +
                        error.what());
                }

                consumer(
                    chunk,
                    offset);

                offset += count;
            }
        }

        void accumulate_file_backed_semantic_source(
            FoldedEvaluationGroup *group,
            const OpeningSemanticSourceInfo &info,
            const FieldElement &scale,
            std::size_t worker_cap,
            BatchOpeningProfile *profile = nullptr)
        {
            if (!info.file_backed ||
                info.file_path.empty())
            {
                throw std::runtime_error(
                    "file-backed semantic source "
                    "metadata is incomplete");
            }

            if (scale.is_zero())
            {
                return;
            }

            const auto stored_value_count =
                validated_file_backed_semantic_source_count(
                    info,
                    "semantic source");
            if (profile != nullptr)
            {
                ++profile->streamed_semantic_source_scan_count;
                profile->streamed_semantic_source_scan_bytes +=
                    algebra::field_elements_binary_size(
                        stored_value_count);
            }

            for_each_file_backed_semantic_source_chunk(
                info,
                "semantic source",
                [&](const std::vector<FieldElement> &chunk,
                    std::size_t offset)
                {
                    accumulate_streamed_evaluation_source(
                        group,
                        chunk,
                        scale,
                        info.file_path,
                        worker_cap,
                        offset,
                        true);
                });
        }

        void accumulate_file_backed_semantic_source_pair(
            FoldedEvaluationGroup *first_group,
            FoldedEvaluationGroup *second_group,
            const OpeningSemanticSourceInfo &info,
            const FieldElement &first_scale,
            const FieldElement &second_scale,
            std::size_t worker_cap,
            BatchOpeningProfile *profile = nullptr)
        {
            if (!info.file_backed ||
                info.file_path.empty())
            {
                throw std::runtime_error(
                    "paired file-backed semantic source "
                    "metadata is incomplete");
            }

            if (first_scale.is_zero() &&
                second_scale.is_zero())
            {
                return;
            }

            const auto stored_value_count =
                validated_file_backed_semantic_source_count(
                    info,
                    "paired semantic source");
            if (profile != nullptr)
            {
                ++profile->streamed_semantic_source_scan_count;
                profile->streamed_semantic_source_scan_bytes +=
                    algebra::field_elements_binary_size(
                        stored_value_count);
            }

            for_each_file_backed_semantic_source_chunk(
                info,
                "paired semantic source",
                [&](const std::vector<FieldElement> &chunk,
                    std::size_t offset)
                {
                    accumulate_streamed_evaluation_source_pair(
                        first_group,
                        second_group,
                        chunk,
                        first_scale,
                        second_scale,
                        info.file_path,
                        worker_cap,
                        offset,
                        true);
                });
        }

        struct SharedDomainWeightView
        {
            const algebra::DomainEvaluationWeights
                *weights = nullptr;

            std::size_t forward_rotation = 0;
        };

        SharedDomainWeightView
        shared_domain_weight_view(
            algebra::DomainEvaluationWeightCache
                *cache,
            const std::shared_ptr<
                algebra::RootOfUnityDomain> &domain,
            const std::vector<FieldElement> &points,
            std::size_t point_index)
        {
            if (cache == nullptr ||
                domain == nullptr ||
                point_index >= points.size())
            {
                throw std::runtime_error(
                    "invalid shared domain-weight request");
            }

            if (point_index > 0 &&
                points[point_index] ==
                    points[point_index - 1U] *
                        domain->omega)
            {
                return SharedDomainWeightView{
                    .weights =
                        &cache->get(
                            domain,
                            points[point_index - 1U]),
                    .forward_rotation = 1,
                };
            }

            return SharedDomainWeightView{
                .weights =
                    &cache->get(
                        domain,
                        points[point_index]),
                .forward_rotation = 0,
            };
        }

        FieldElement evaluate_with_shared_domain_weights(
            const algebra::Polynomial &polynomial,
            const FieldElement &point,
            const algebra::DomainEvaluationWeights &weights,
            std::size_t forward_rotation = 0)
        {
            if (polynomial.basis !=
                    algebra::PolynomialBasis::
                        Evaluation ||
                polynomial.domain == nullptr)
            {
                throw std::runtime_error(
                    "shared domain-weight evaluation requires an evaluation polynomial");
            }

            const auto &values =
                polynomial.values();

            FieldElement value =
                FieldElement::zero();

            const auto domain_size =
                polynomial.domain->size;

            const auto rotation =
                domain_size == 0
                    ? std::size_t{0}
                    : forward_rotation %
                          domain_size;

            if (weights.direct_index.has_value())
            {
                const auto rotated_index =
                    (*weights.direct_index +
                     rotation) %
                    domain_size;

                if (rotated_index <
                    values.size())
                {
                    value =
                        values[rotated_index];
                }
            }
            else
            {
                if (weights.native_weights.size() !=
                        domain_size ||
                    values.size() >
                        domain_size)
                {
                    throw std::runtime_error(
                        "evaluation polynomial exceeds its domain weight vector");
                }

                mcl::Fr sum;
                sum.clear();

                for (std::size_t index = 0;
                     index < values.size();
                     ++index)
                {
                    const auto weight_index =
                        (index +
                         domain_size -
                         rotation) %
                        domain_size;

                    mcl::Fr product;

                    mcl::Fr::mul(
                        product,
                        values[index].native(),
                        weights.native_weights[weight_index]);

                    mcl::Fr::add(
                        sum,
                        sum,
                        product);
                }

                value =
                    FieldElement::
                        from_native(
                            sum);
            }

            if (polynomial.is_zero_domain_blinded())
            {
                value +=
                    polynomial.domain
                        ->zero_polynomial_eval(
                            point) *
                    algebra::horner(
                        polynomial
                            .zero_domain_blinder,
                        point);
            }

            return value;
        }

        FoldedEvaluationGroup &folded_evaluation_group(
            FoldedOpeningPlan *plan,
            const std::shared_ptr<algebra::RootOfUnityDomain> &domain,
            std::size_t point_count)
        {
            auto it = std::find_if(
                plan->evaluation_groups.begin(),
                plan->evaluation_groups.end(),
                [&](const FoldedEvaluationGroup &group)
                {
                    return group.domain.get() == domain.get();
                });
            if (it == plan->evaluation_groups.end())
            {
                FoldedEvaluationGroup group;
                group.domain = domain;
                group.scaled_claim_sums.assign(point_count, FieldElement::zero());
                plan->evaluation_groups.push_back(std::move(group));
                it = std::prev(plan->evaluation_groups.end());
            }
            return *it;
        }

        FoldedOpeningPlan build_folded_opening_plan(
            const std::vector<Commitment> &commitments,
            const std::vector<std::vector<FieldElement>> &claimed_values,
            const std::vector<FieldElement> &powers,
            const std::vector<FieldElement> &points,
            std::size_t worker_cap,
            bool validate_claims = true)
        {
            const auto point_count = points.size();
            algebra::DomainEvaluationWeightCache claim_weight_cache;
            struct FoldedOpeningTerm
            {
                const Commitment *commitment = nullptr;
                FieldElement scale = FieldElement::zero();
                std::vector<FieldElement> scaled_claims;
            };

            std::vector<FoldedOpeningTerm> folded_terms;
            folded_terms.reserve(commitments.size());
            std::unordered_map<const algebra::Polynomial *, std::size_t>
                folded_term_index;
            folded_term_index.reserve(commitments.size());
            for (std::size_t i = 0; i < commitments.size(); ++i)
            {
                if (commitments[i].opening_polynomial == nullptr)
                {
                    throw std::runtime_error("commitment is missing prover-side opening polynomial: " + commitments[i].name);
                }
                const auto *polynomial =
                    commitments[i].opening_polynomial.get();
                const auto [existing, inserted] =
                    folded_term_index.emplace(
                        polynomial, folded_terms.size());
                if (inserted)
                {
                    FoldedOpeningTerm term;
                    term.commitment = &commitments[i];
                    term.scale = powers[i];
                    term.scaled_claims.assign(point_count, FieldElement::zero());
                    for (std::size_t point_index = 0; point_index < point_count; ++point_index)
                    {
                        term.scaled_claims[point_index] = claimed_values[i][point_index] * powers[i];
                    }
                    folded_terms.push_back(std::move(term));
                }
                else
                {
                    auto &term = folded_terms[existing->second];
                    term.scale += powers[i];
                    for (std::size_t point_index = 0; point_index < point_count; ++point_index)
                    {
                        term.scaled_claims[point_index] +=
                            claimed_values[i][point_index] * powers[i];
                    }
                }
            }

            if (validate_claims)
            {
                parallel_for_chunks_capped(
                    folded_terms.size(),
                    16,
                    worker_cap,
                    [&](std::size_t begin, std::size_t end)
                    {
                        for (std::size_t term_index = begin;
                             term_index < end; ++term_index)
                        {
                            const auto &term = folded_terms[term_index];
                            if (term.commitment == nullptr || term.commitment->opening_polynomial == nullptr || term.scale.is_zero())
                            {
                                continue;
                            }
                            const auto &polynomial =
                                *term.commitment->opening_polynomial;
                            for (std::size_t point_index = 0;
                                 point_index < point_count;
                                 ++point_index)
                            {
                                FieldElement expected = FieldElement::zero();
                                if (polynomial.basis == algebra::PolynomialBasis::Coefficient)
                                {
                                    expected = polynomial.evaluate(
                                        points[point_index]);
                                }
                                else
                                {
                                    if (polynomial.domain == nullptr)
                                    {
                                        throw std::runtime_error(
                                            "evaluation polynomial is missing its domain: " + term.commitment->name);
                                    }
                                    const auto weight_view =
                                        shared_domain_weight_view(
                                            &claim_weight_cache,
                                            polynomial.domain,
                                            points,
                                            point_index);

                                    expected =
                                        evaluate_with_shared_domain_weights(
                                            polynomial,
                                            points[point_index],
                                            *weight_view.weights,
                                            weight_view
                                                .forward_rotation);
                                }
                                const auto claimed =
                                    term.scaled_claims[point_index] / term.scale;
                                if (expected != claimed)
                                {
                                    throw std::runtime_error(
                                        (polynomial.basis == algebra::PolynomialBasis::Coefficient
                                             ? "coefficient opening claim mismatch for "
                                             : "evaluation opening claim mismatch for ") +
                                        term.commitment->name + " at point index " + std::to_string(point_index));
                                }
                            }
                        }
                    });
            }

            FoldedOpeningPlan plan;
            plan.coefficient_scaled_claim_sums.assign(point_count, FieldElement::zero());
            for (const auto &term : folded_terms)
            {
                if (term.commitment == nullptr || term.commitment->opening_polynomial == nullptr)
                {
                    throw std::runtime_error("commitment is missing prover-side opening polynomial");
                }
                if (term.scale.is_zero() && all_zero(term.scaled_claims))
                {
                    continue;
                }
                const auto &polynomial = *term.commitment->opening_polynomial;
                if (polynomial.basis == algebra::PolynomialBasis::Coefficient)
                {
                    if (!term.scale.is_zero())
                    {
                        accumulate_coefficient_fold(
                            &plan,
                            polynomial.values(),
                            term.scale,
                            worker_cap);
                    }
                    for (std::size_t point_index = 0; point_index < point_count; ++point_index)
                    {
                        plan.coefficient_scaled_claim_sums[point_index] += term.scaled_claims[point_index];
                    }
                    continue;
                }
                if (polynomial.domain == nullptr)
                {
                    throw std::runtime_error("evaluation polynomial is missing its domain: " + term.commitment->name);
                }
                auto &group = folded_evaluation_group(&plan, polynomial.domain, point_count);
                const auto &source_values = polynomial.values();
                if (!term.scale.is_zero())
                {
                    group.terms.push_back(FoldedEvaluationTerm{
                        .values = &source_values,
                        .zero_domain_blinder = polynomial.is_zero_domain_blinded()
                                                   ? &polynomial.zero_domain_blinder
                                                   : nullptr,
                        .value_count = source_values.size(),
                        .scale = term.scale,
                    });
                }
                for (std::size_t point_index = 0; point_index < point_count; ++point_index)
                {
                    group.scaled_claim_sums[point_index] += term.scaled_claims[point_index];
                }
            }
            for (auto &group : plan.evaluation_groups)
            {
                group.folded_values.assign(group.domain->size, FieldElement::zero());
                auto *folded_values_native =
                    FieldElement::mutable_native_array_data(group.folded_values.data());
                struct NativeTerm
                {
                    const std::vector<FieldElement> *values = nullptr;
                    std::size_t value_count = 0;
                    mcl::Fr scale;
                };
                std::vector<NativeTerm> native_terms;
                native_terms.reserve(group.terms.size());
                for (const auto &term : group.terms)
                {
                    native_terms.push_back(NativeTerm{
                        .values = term.values,
                        .value_count = term.value_count,
                        .scale = term.scale.native(),
                    });
                }
                const auto min_rows_per_task = std::max<std::size_t>(
                    256,
                    (1ULL << 18) / std::max<std::size_t>(1, native_terms.size()));
                parallel_for_chunks_capped(
                    group.domain->size,
                    min_rows_per_task,
                    worker_cap,
                    [&](std::size_t begin, std::size_t end)
                    {
                        for (const auto &term : native_terms)
                        {
                            const auto term_end = std::min(end, term.value_count);
                            for (std::size_t row = begin; row < term_end; ++row)
                            {
                                mcl::Fr product;
                                mcl::Fr::mul(
                                    product,
                                    (*term.values)[row].native(),
                                    term.scale);
                                mcl::Fr::add(
                                    folded_values_native[row],
                                    folded_values_native[row],
                                    product);
                            }
                        }
                    });
            }
            finish_cuda_coefficient_fold_session(
                &plan);

            trim_zero_coefficients(&plan.coefficient_fold);
            return plan;
        }

        FoldedOpeningPlan build_streamed_folded_opening_plan(
            const std::vector<Commitment> &commitments,
            const std::vector<std::vector<FieldElement>> &claimed_values,
            const std::vector<FieldElement> &powers,
            const std::vector<FieldElement> &points,
            std::size_t worker_cap,
            bool validate_claims,
            const OpeningPolynomialResolver &opening_resolver,
            const OpeningSemanticSourceResolver *semantic_source_resolver,
            const StreamedSemanticSourceDefer
                *defer_semantic_source,
            BatchOpeningProfile *profile)
        {
            if (commitments.size() != claimed_values.size() || commitments.size() != powers.size())
            {
                throw std::runtime_error(
                    "streamed folded opening input shape mismatch");
            }

            FoldedOpeningPlan plan;
            plan.coefficient_scaled_claim_sums.assign(
                points.size(), FieldElement::zero());

            algebra::DomainEvaluationWeightCache claim_weight_cache;

            struct SemanticFold
            {
                OpeningSemanticSourceInfo info;
                std::size_t representative_index = 0;
                FieldElement source_scale = FieldElement::zero();
                std::vector<FieldElement> folded_zero_domain_blinder;
                std::vector<FieldElement> scaled_claim_sums;
                std::vector<std::size_t> members;
            };

            std::vector<std::optional<std::size_t>>
                semantic_fold_for_commitment(commitments.size());

            std::vector<SemanticFold> semantic_folds;
            std::unordered_map<std::string, std::size_t>
                semantic_fold_index;

            // Production opening claims have already been computed from the exact
            // formal opening polynomials.  In that mode we may combine several
            // independently blinded evaluation polynomials that share one immutable
            // semantic source:
            //
            //   sum_i a_i (S + Z_D R_i)
            //     = (sum_i a_i) S + Z_D (sum_i a_i R_i).
            //
            // VerifyAgainstOpeningPolynomials retains the per-polynomial path
            // required by focused claim-validation tests.
            if (!validate_claims && semantic_source_resolver != nullptr)
            {
                semantic_folds.reserve(commitments.size());

                for (std::size_t i = 0;
                     i < commitments.size();
                     ++i)
                {
                    if (claimed_values[i].size() != points.size())
                    {
                        throw std::runtime_error(
                            "streamed folded opening claim shape mismatch: " + commitments[i].name);
                    }

                    // In-memory opening material retains the existing path.  This
                    // optimization is intentionally limited to file-backed evaluation
                    // material carrying explicit semantic-source metadata.
                    if (commitments[i].opening_polynomial != nullptr)
                    {
                        continue;
                    }

                    const auto semantic_info =
                        (*semantic_source_resolver)(
                            commitments[i]);

                    if (!semantic_info.has_value())
                    {
                        continue;
                    }

                    if (semantic_info->key.empty() || semantic_info->domain == nullptr || semantic_info->value_count > semantic_info->domain->size)
                    {
                        throw std::runtime_error(
                            "invalid streamed opening semantic source metadata: " + commitments[i].name);
                    }

                    if (semantic_info->zero_tail_compacted &&
                        semantic_info->stored_value_count >
                            semantic_info->value_count)
                    {
                        throw std::runtime_error(
                            "invalid streamed opening semantic source physical count: " + commitments[i].name);
                    }
                    if (!semantic_info->file_backed &&
                        !semantic_info->zero_tail_compacted &&
                        semantic_info->stored_value_count != 0 &&
                        semantic_info->stored_value_count !=
                            semantic_info->value_count)
                    {
                        throw std::runtime_error(
                            "invalid streamed opening semantic source stored count: " + commitments[i].name);
                    }

                    if (semantic_info->file_backed)
                    {
                        const auto stored_value_count =
                            validated_file_backed_semantic_source_count(
                                *semantic_info,
                                "streamed opening semantic source");
                        if (semantic_info->lease == nullptr)
                        {
                            throw std::runtime_error(
                                "streamed opening semantic source has no authenticated trace spill lease: " +
                                commitments[i].name);
                        }
                        const auto payload_name =
                            semantic_info->lease->payload_name(
                                semantic_info->file_path);
                        // Validate every descriptor before grouping by the
                        // execution-only semantic key.  A representative
                        // path may be reused for a valid shared source, but
                        // a malformed alternate descriptor must never be
                        // hidden by that reuse.
                        if (semantic_info->lease->payload_size(payload_name) !=
                            algebra::field_elements_binary_size(
                                stored_value_count))
                        {
                            throw std::runtime_error(
                                "streamed opening semantic source length mismatch: " +
                                semantic_info->file_path);
                        }
                    }

                    const auto [existing, inserted] =
                        semantic_fold_index.emplace(
                            semantic_info->key,
                            semantic_folds.size());

                    std::size_t fold_index = 0;

                    if (inserted)
                    {
                        fold_index = semantic_folds.size();

                        SemanticFold fold;
                        fold.info = *semantic_info;
                        fold.representative_index = i;
                        fold.scaled_claim_sums.assign(
                            points.size(),
                            FieldElement::zero());

                        semantic_folds.push_back(
                            std::move(fold));
                    }
                    else
                    {
                        fold_index = existing->second;

                        const auto &existing_info =
                            semantic_folds[fold_index].info;

                        if (!compatible_streamed_semantic_source_metadata(
                                existing_info,
                                *semantic_info))
                        {
                            throw std::runtime_error(
                                "conflicting streamed opening semantic source metadata/physical metadata: " + commitments[i].name);
                        }
                    }

                    semantic_fold_for_commitment[i] =
                        fold_index;

                    auto &fold =
                        semantic_folds[fold_index];

                    fold.members.push_back(i);
                    fold.source_scale += powers[i];

                    if (!semantic_info
                             ->zero_domain_blinder.empty())
                    {
                        if (fold.folded_zero_domain_blinder.size() < semantic_info
                                                                         ->zero_domain_blinder.size())
                        {
                            fold.folded_zero_domain_blinder.resize(
                                semantic_info
                                    ->zero_domain_blinder.size(),
                                FieldElement::zero());
                        }

                        for (std::size_t coefficient = 0;
                             coefficient < semantic_info
                                               ->zero_domain_blinder.size();
                             ++coefficient)
                        {
                            fold.folded_zero_domain_blinder[coefficient] += semantic_info
                                                                                ->zero_domain_blinder[coefficient] *
                                                                            powers[i];
                        }
                    }

                    for (std::size_t point_index = 0;
                         point_index < points.size();
                         ++point_index)
                    {
                        fold.scaled_claim_sums[point_index] += claimed_values[i][point_index] * powers[i];
                    }
                }

                // Singleton file-backed sources still stay on the streamed
                // path.  They have no pair to fuse, but materializing their
                // logical vector would recreate the large opening working set
                // that this resolver is intended to avoid.
                std::size_t shared_source_count = 0;
                std::size_t shared_member_count = 0;

                for (std::size_t fold_index = 0;
                     fold_index < semantic_folds.size();
                     ++fold_index)
                {
                    auto &fold =
                        semantic_folds[fold_index];

                    if (fold.members.size() < 2 &&
                        !fold.info.file_backed)
                    {
                        if (defer_semantic_source == nullptr)
                        {
                            for (const auto member :
                                 fold.members)
                            {
                                semantic_fold_for_commitment[member] =
                                    std::nullopt;
                            }

                            continue;
                        }
                    }
                    else
                    {
                        ++shared_source_count;
                        shared_member_count +=
                            fold.members.size();
                    }
                }

                if (shared_source_count != 0)
                {
                    progress_note(
                        "kzg:opening:semantic_fold:"
                        "sources=" +
                        std::to_string(
                            shared_source_count) +
                        ":members=" + std::to_string(shared_member_count));
                }
            }

            for (std::size_t i = 0;
                 i < commitments.size();
                 ++i)
            {
                if (claimed_values[i].size() != points.size())
                {
                    throw std::runtime_error(
                        "streamed folded opening claim shape mismatch: " + commitments[i].name);
                }

                if (semantic_fold_for_commitment[i]
                        .has_value())
                {
                    auto &semantic_fold =
                        semantic_folds[*semantic_fold_for_commitment[i]];

                    if (semantic_fold.representative_index != i)
                    {
                        continue;
                    }

                    auto &group =
                        folded_evaluation_group(
                            &plan,
                            semantic_fold.info.domain,
                            points.size());

                    if (group.folded_values.empty())
                    {
                        group.folded_values.assign(
                            group.domain->size,
                            FieldElement::zero());
                    }

                    if (semantic_fold.info.value_count > group.folded_values.size())
                    {
                        throw std::runtime_error(
                            "streamed semantic evaluation source exceeds its domain: " + commitments[i].name);
                    }

                    if (!semantic_fold.source_scale
                             .is_zero())
                    {
                        if (defer_semantic_source != nullptr)
                        {
                            (*defer_semantic_source)(
                                DeferredStreamedSemanticSource{
                                    .info =
                                        semantic_fold.info,
                                    .representative =
                                        commitments[i],
                                    .scale =
                                        semantic_fold.source_scale,
                                });
                        }
                        else
                        {
                            if (semantic_fold.info.file_backed)
                            {
                                accumulate_file_backed_semantic_source(
                                    &group,
                                    semantic_fold.info,
                                    semantic_fold.source_scale,
                                    worker_cap,
                                    profile);
                            }
                            else
                            {
                                const auto material =
                                    resolve_streamed_semantic_source_material(
                                        opening_resolver,
                                        commitments[i],
                                        semantic_fold.info);

                                accumulate_streamed_evaluation_source(
                                    &group,
                                    material->values(),
                                    semantic_fold.source_scale,
                                    commitments[i].name,
                                    worker_cap);
                            }
                        }
                    }

                    if (!semantic_fold
                             .folded_zero_domain_blinder
                             .empty())
                    {
                        if (group
                                .folded_zero_domain_blinder
                                .size() < semantic_fold
                                              .folded_zero_domain_blinder
                                              .size())
                        {
                            group
                                .folded_zero_domain_blinder
                                .resize(
                                    semantic_fold
                                        .folded_zero_domain_blinder
                                        .size(),
                                    FieldElement::zero());
                        }

                        for (std::size_t coefficient = 0;
                             coefficient < semantic_fold
                                               .folded_zero_domain_blinder
                                               .size();
                             ++coefficient)
                        {
                            group.folded_zero_domain_blinder[coefficient] += semantic_fold
                                                                                 .folded_zero_domain_blinder[coefficient];
                        }
                    }

                    for (std::size_t point_index = 0;
                         point_index < points.size();
                         ++point_index)
                    {
                        group.scaled_claim_sums[point_index] += semantic_fold
                                                                    .scaled_claim_sums[point_index];
                    }

                    continue;
                }

                auto material =
                    commitments[i].opening_polynomial;

                if (material == nullptr)
                {
                    material =
                        opening_resolver(
                            commitments[i]);
                }

                if (material == nullptr)
                {
                    throw std::runtime_error(
                        "commitment is missing prover-side opening polynomial: " + commitments[i].name);
                }

                if (validate_claims)
                {
                    for (std::size_t point_index = 0;
                         point_index < points.size();
                         ++point_index)
                    {
                        FieldElement expected =
                            FieldElement::zero();

                        if (material->basis == algebra::PolynomialBasis::Coefficient)
                        {
                            expected =
                                material->evaluate(
                                    points[point_index]);
                        }
                        else
                        {
                            if (material->domain == nullptr)
                            {
                                throw std::runtime_error(
                                    "evaluation polynomial is missing its domain: " + commitments[i].name);
                            }

                            const auto weight_view =
                                shared_domain_weight_view(
                                    &claim_weight_cache,
                                    material->domain,
                                    points,
                                    point_index);

                            expected =
                                evaluate_with_shared_domain_weights(
                                    *material,
                                    points[point_index],
                                    *weight_view.weights,
                                    weight_view
                                        .forward_rotation);
                        }

                        if (expected != claimed_values[i][point_index])
                        {
                            throw std::runtime_error(
                                "streamed opening claim mismatch for " + commitments[i].name + " at point index " + std::to_string(point_index));
                        }
                    }
                }

                const auto &scale =
                    powers[i];

                if (material->basis == algebra::PolynomialBasis::Coefficient)
                {
                    if (!scale.is_zero())
                    {
                        accumulate_coefficient_fold(
                            &plan,
                            material->values(),
                            scale,
                            worker_cap,
                            commitments[i].opening_polynomial == nullptr);
                    }

                    for (std::size_t point_index = 0;
                         point_index < points.size();
                         ++point_index)
                    {
                        plan.coefficient_scaled_claim_sums[point_index] += claimed_values[i][point_index] * scale;
                    }

                    continue;
                }

                if (material->domain == nullptr)
                {
                    throw std::runtime_error(
                        "evaluation polynomial is missing its domain: " + commitments[i].name);
                }

                auto &group =
                    folded_evaluation_group(
                        &plan,
                        material->domain,
                        points.size());

                if (group.folded_values.empty())
                {
                    group.folded_values.assign(
                        group.domain->size,
                        FieldElement::zero());
                }

                const auto &source_values =
                    material->values();

                accumulate_streamed_evaluation_source(
                    &group,
                    source_values,
                    scale,
                    commitments[i].name,
                    worker_cap,
                    0,
                    commitments[i].opening_polynomial == nullptr);

                if (!scale.is_zero() && material
                                            ->is_zero_domain_blinded())
                {
                    accumulate_scaled_coefficients(
                        &group
                             .folded_zero_domain_blinder,
                        material
                            ->zero_domain_blinder,
                        scale,
                        worker_cap);
                }

                for (std::size_t point_index = 0;
                     point_index < points.size();
                     ++point_index)
                {
                    group.scaled_claim_sums[point_index] += claimed_values[i][point_index] * scale;
                }
            }

            finish_cuda_coefficient_fold_session(
                &plan);

            trim_zero_coefficients(
                &plan.coefficient_fold);

            for (auto &group :
                 plan.evaluation_groups)
            {
                trim_zero_coefficients(
                    &group
                         .folded_zero_domain_blinder);

                finish_cuda_fold_session(
                    &group);
            }

            return plan;
        }

        PairedStreamedFoldBuild
        build_streamed_folded_opening_plan_pair(
            const std::array<
                std::vector<Commitment>,
                2> &commitments,
            const std::array<
                std::vector<std::vector<FieldElement>>,
                2> &claimed_values,
            const std::array<
                std::vector<FieldElement>,
                2> &powers,
            const std::array<FieldElement, 2> &points,
            const std::array<std::size_t, 2> &worker_caps,
            const OpeningPolynomialResolver &opening_resolver,
            const OpeningSemanticSourceResolver
                &semantic_source_resolver,
            const std::array<BatchOpeningProfile *, 2> *scan_profiles)
        {
            PairedStreamedFoldBuild result;

            std::array<
                std::vector<DeferredStreamedSemanticSource>,
                2>
                deferred;

            const auto build_one =
                [&](std::size_t index)
            {
                const StreamedSemanticSourceDefer
                    defer =
                        [&](const DeferredStreamedSemanticSource
                                &source)
                {
                    deferred[index].push_back(
                        source);
                };

                result.plans[index] =
                    build_streamed_folded_opening_plan(
                        commitments[index],
                        claimed_values[index],
                        powers[index],
                        std::vector<FieldElement>{
                            points[index]},
                        worker_caps[index],
                        false,
                        opening_resolver,
                        &semantic_source_resolver,
                        &defer,
                        nullptr);
            };

            auto first_future =
                std::async(
                    std::launch::async,
                    [&]()
                    {
                        build_one(0);
                    });

            auto second_future =
                std::async(
                    std::launch::async,
                    [&]()
                    {
                        build_one(1);
                    });

            first_future.get();
            second_future.get();

            result.deferred_source_count = {
                deferred[0].size(),
                deferred[1].size(),
            };

            std::unordered_map<
                std::string,
                std::size_t>
                second_by_key;

            second_by_key.reserve(
                deferred[1].size());

            for (std::size_t index = 0;
                 index < deferred[1].size();
                 ++index)
            {
                const auto [_, inserted] =
                    second_by_key.emplace(
                        deferred[1][index].info.key,
                        index);

                if (!inserted)
                {
                    throw std::runtime_error(
                        "paired streamed opening received duplicate deferred semantic source");
                }
            }

            std::vector<bool> second_consumed(
                deferred[1].size(),
                false);

            const auto shared_worker_cap =
                [&]()
            {
                if (worker_caps[0] == 0)
                {
                    return worker_caps[1];
                }

                if (worker_caps[1] == 0)
                {
                    return worker_caps[0];
                }

                return std::min(
                    worker_caps[0],
                    worker_caps[1]);
            }();

            const auto process_single =
                [&](std::size_t plan_index,
                    const DeferredStreamedSemanticSource
                        &source)
            {
                auto &group =
                    require_folded_evaluation_group(
                        &result.plans[plan_index],
                        source.info.domain);

                if (source.info.file_backed)
                {
                    accumulate_file_backed_semantic_source(
                        &group,
                        source.info,
                        source.scale,
                        worker_caps[plan_index],
                        scan_profiles != nullptr
                            ? (*scan_profiles)[plan_index]
                            : nullptr);

                    return;
                }

                const auto material =
                    resolve_streamed_semantic_source_material(
                        opening_resolver,
                        source.representative,
                        source.info);

                accumulate_streamed_evaluation_source(
                    &group,
                    material->values(),
                    source.scale,
                    source.representative.name,
                    worker_caps[plan_index]);
            };

            for (const auto &first :
                 deferred[0])
            {
                const auto second_it =
                    second_by_key.find(
                        first.info.key);

                if (second_it ==
                    second_by_key.end())
                {
                    process_single(
                        0,
                        first);

                    continue;
                }

                const auto second_index =
                    second_it->second;

                const auto &second =
                    deferred[1][second_index];

                // The semantic source key, not the formal commitment name or
                // commitment point, determines whether two point groups may
                // share a physical scan.  Their commitments and blinders are
                // intentionally still folded independently below.
                const bool same_source_with_physical_metadata =
                    first.info.key == second.info.key &&
                    compatible_streamed_semantic_source_metadata(
                        first.info,
                        second.info);

                if (!same_source_with_physical_metadata)
                {
                    process_single(
                        0,
                        first);

                    continue;
                }

                auto &first_group =
                    require_folded_evaluation_group(
                        &result.plans[0],
                        first.info.domain);

                auto &second_group =
                    require_folded_evaluation_group(
                        &result.plans[1],
                        second.info.domain);

                bool fused_scan = false;
                if (first.info.file_backed &&
                    second.info.file_backed &&
                    first.info.file_path ==
                        second.info.file_path)
                {
                    accumulate_file_backed_semantic_source_pair(
                        &first_group,
                        &second_group,
                        first.info,
                        first.scale,
                        second.scale,
                        shared_worker_cap,
                        scan_profiles != nullptr
                            ? (*scan_profiles)[0]
                            : nullptr);
                    fused_scan = true;
                }
                else if (first.info.file_backed &&
                         second.info.file_backed)
                {
                    // Distinct physical paths may still represent the same
                    // logical zero-tailed source.  Validate and consume both
                    // descriptors independently rather than treating the
                    // first path as an alias for the second one.
                    accumulate_file_backed_semantic_source(
                        &first_group,
                        first.info,
                        first.scale,
                        worker_caps[0],
                        scan_profiles != nullptr
                            ? (*scan_profiles)[0]
                            : nullptr);
                    accumulate_file_backed_semantic_source(
                        &second_group,
                        second.info,
                        second.scale,
                        worker_caps[1],
                        scan_profiles != nullptr
                            ? (*scan_profiles)[1]
                            : nullptr);
                }
                else
                {
                    const auto material =
                        resolve_streamed_semantic_source_material(
                            opening_resolver,
                            first.representative,
                            first.info);

                    accumulate_streamed_evaluation_source_pair(
                        &first_group,
                        &second_group,
                        material->values(),
                        first.scale,
                        second.scale,
                        first.representative.name,
                        shared_worker_cap);
                }

                second_consumed[second_index] =
                    true;

                if (fused_scan || !first.info.file_backed)
                {
                    ++result
                          .fused_semantic_source_count;
                }
            }

            for (std::size_t index = 0;
                 index < deferred[1].size();
                 ++index)
            {
                if (second_consumed[index])
                {
                    continue;
                }

                process_single(
                    1,
                    deferred[1][index]);
            }

            for (auto &plan :
                 result.plans)
            {
                for (auto &group :
                     plan.evaluation_groups)
                {
                    finish_cuda_fold_session(
                        &group);
                }
            }

            return result;
        }

        void build_opening_inverse_denominators(
            std::vector<FieldElement> *inverse_denominators,
            const std::shared_ptr<algebra::RootOfUnityDomain> &domain,
            const FieldElement &point,
            std::size_t worker_cap)
        {
            if (inverse_denominators == nullptr ||
                domain == nullptr ||
                inverse_denominators->size() != domain->size)
            {
                throw std::runtime_error(
                    "opening denominator buffer/domain shape mismatch");
            }

            // Small/medium domains already retain all domain points.
            // Keep their existing execution path unchanged.
            if (domain->points_precomputed)
            {
                parallel_for_chunks_capped(
                    domain->size,
                    1ULL << 15,
                    worker_cap,
                    [&](std::size_t begin, std::size_t end)
                    {
                        for (std::size_t row = begin; row < end; ++row)
                        {
                            const auto domain_point =
                                domain->point_at(row);

                            if (domain_point == point)
                            {
                                throw std::runtime_error(
                                    "opening point overlaps evaluation domain; derivative witness path is not implemented");
                            }

                            (*inverse_denominators)[row] =
                                domain_point - point;
                        }
                    });

                algebra::batch_invert_in_place(
                    inverse_denominators);
                return;
            }

            const auto cpu_count =
                std::max<std::size_t>(
                    1,
                    std::thread::hardware_concurrency());

            const auto available_workers =
                worker_cap == 0
                    ? cpu_count
                    : std::min<std::size_t>(
                          cpu_count,
                          worker_cap);

            constexpr std::size_t kMinRowsPerTask =
                1ULL << 18;

            const auto requested_task_count =
                std::max<std::size_t>(
                    1,
                    (domain->size + kMinRowsPerTask - 1) /
                        kMinRowsPerTask);

            const auto task_count =
                std::max<std::size_t>(
                    1,
                    std::min(
                        available_workers,
                        requested_task_count));

            const auto chunk_size =
                (domain->size + task_count - 1) /
                task_count;

            auto *denominator_native =
                FieldElement::mutable_native_array_data(
                    inverse_denominators->data());

            std::vector<mcl::Fr> chunk_products(
                task_count);

            const auto point_native =
                point.native();

            const auto omega_native =
                domain->omega.native();

            const auto omega_inverse_native =
                domain->omega.inv().native();

            const auto run_chunks =
                [&](const auto &fn)
            {
                if (task_count == 1)
                {
                    fn(
                        std::size_t{0},
                        std::size_t{0},
                        domain->size);
                    return;
                }

                std::vector<std::future<void>> futures;
                futures.reserve(task_count);

                for (std::size_t task = 0;
                     task < task_count;
                     ++task)
                {
                    const auto begin =
                        task * chunk_size;

                    const auto end =
                        std::min(
                            domain->size,
                            begin + chunk_size);

                    futures.push_back(
                        std::async(
                            std::launch::async,
                            [&, task, begin, end]
                            {
                                fn(
                                    task,
                                    begin,
                                    end);
                            }));
                }

                for (auto &future : futures)
                {
                    future.get();
                }
            };

            // Use the output array itself as the per-row prefix-product
            // workspace.  The original denominator does not need to remain
            // resident: it is regenerated in the reverse pass from omega^i.
            run_chunks(
                [&](std::size_t task,
                    std::size_t begin,
                    std::size_t end)
                {
                    mcl::Fr local_product;
                    local_product = 1;

                    auto domain_point =
                        domain->point_at(begin).native();

                    for (std::size_t row = begin;
                         row < end;
                         ++row)
                    {
                        mcl::Fr denominator;

                        mcl::Fr::sub(
                            denominator,
                            domain_point,
                            point_native);

                        if (denominator.isZero())
                        {
                            throw std::runtime_error(
                                "opening point overlaps evaluation domain; derivative witness path is not implemented");
                        }

                        // Temporarily store the product of every denominator
                        // preceding this row inside the eventual output slot.
                        denominator_native[row] =
                            local_product;

                        mcl::Fr::mul(
                            local_product,
                            local_product,
                            denominator);

                        mcl::Fr::mul(
                            domain_point,
                            domain_point,
                            omega_native);
                    }

                    chunk_products[task] =
                        local_product;
                });

            // Reduce the per-chunk products and perform exactly one field
            // inversion for the complete domain.
            std::vector<mcl::Fr> chunk_prefixes(
                task_count);

            mcl::Fr total_product;
            total_product = 1;

            for (std::size_t task = 0;
                 task < task_count;
                 ++task)
            {
                chunk_prefixes[task] =
                    total_product;

                mcl::Fr::mul(
                    total_product,
                    total_product,
                    chunk_products[task]);
            }

            mcl::Fr suffix_inverse;

            mcl::Fr::inv(
                suffix_inverse,
                total_product);

            for (std::size_t task = task_count;
                 task-- > 0;)
            {
                mcl::Fr chunk_inverse;

                mcl::Fr::mul(
                    chunk_inverse,
                    suffix_inverse,
                    chunk_prefixes[task]);

                mcl::Fr::mul(
                    suffix_inverse,
                    suffix_inverse,
                    chunk_products[task]);

                // Reuse the chunk-product slot for the inverse of the entire
                // chunk product required by the reverse pass.
                chunk_products[task] =
                    chunk_inverse;
            }

            // Reconstruct omega^row backwards without a full-size prefixes
            // allocation while retaining the one-inversion batch-inversion
            // result.
            run_chunks(
                [&](std::size_t task,
                    std::size_t begin,
                    std::size_t end)
                {
                    auto local_suffix =
                        chunk_products[task];

                    auto domain_point =
                        domain->point_at(end - 1).native();

                    for (std::size_t row = end;
                         row-- > begin;)
                    {
                        const mcl::Fr prefix =
                            denominator_native[row];

                        mcl::Fr inverse;

                        mcl::Fr::mul(
                            inverse,
                            local_suffix,
                            prefix);

                        mcl::Fr denominator;

                        mcl::Fr::sub(
                            denominator,
                            domain_point,
                            point_native);

                        denominator_native[row] =
                            inverse;

                        mcl::Fr::mul(
                            local_suffix,
                            local_suffix,
                            denominator);

                        if (row > begin)
                        {
                            mcl::Fr::mul(
                                domain_point,
                                domain_point,
                                omega_inverse_native);
                        }
                    }
                });
        }
        G1Point open_single_point_fold(
            const FoldedOpeningPlan &plan,
            std::size_t point_index,
            const FieldElement &point,
            const KZGProverKey &pk,
            BatchOpeningProfile *profile = nullptr,
            std::size_t worker_cap = 0,
            const G1MSMExecutionOptions *msm_options = nullptr)
        {
            auto out = g1_zero();
            const auto &coefficient_scaled_claim_sum = plan.coefficient_scaled_claim_sums[point_index];
            if (!plan.coefficient_fold.empty() || !(coefficient_scaled_claim_sum == FieldElement::zero()))
            {
                const auto coefficient_msm_start = Clock::now();
                try
                {
                    out = g1_add(
                        out,
                        commit_coefficient_single_point_quotient(
                            plan.coefficient_fold,
                            point,
                            coefficient_scaled_claim_sum,
                            pk,
                            worker_cap,
                            msm_options));
                }
                catch (const std::exception &ex)
                {
                    throw std::runtime_error("opening mismatch for folded coefficient group: " + std::string(ex.what()));
                }
                if (profile != nullptr)
                {
                    profile->witness_msm_ms += elapsed_ms(
                        coefficient_msm_start, Clock::now());
                }
            }
            std::vector<FieldElement>
                opening_evaluation_scratch;
            for (const auto &group : plan.evaluation_groups)
            {
                opening_evaluation_scratch.resize(group.domain->size);
                auto *quotient_values_native = FieldElement::mutable_native_array_data(opening_evaluation_scratch.data());
                const auto *folded_values_native = FieldElement::native_array_data(group.folded_values.data());

                const auto fused_start = Clock::now();
                const bool cuda_fused = algebra::try_cuda_opening_inverse_and_quotient_native(
                    folded_values_native,
                    quotient_values_native,
                    group.scaled_claim_sums[point_index].native(),
                    group.domain->size,
                    group.domain->omega,
                    point);

                if (!cuda_fused)
                {
                    const auto inverse_start = Clock::now();
                    build_opening_inverse_denominators(&opening_evaluation_scratch, group.domain, point, worker_cap);
                    if (profile != nullptr)
                    {
                        profile->denominator_inverse_ms += elapsed_ms(inverse_start, Clock::now());
                        profile->denominator_inverse_cache_misses += 1;
                    }

                    const auto assembly_start = Clock::now();
                    mcl::Fr zero;
                    zero.clear();
                    mcl::Fr base_numerator;
                    mcl::Fr::sub(base_numerator, zero, group.scaled_claim_sums[point_index].native());
                    parallel_for_chunks_capped(group.domain->size, 1ULL << 15, worker_cap, [&](std::size_t begin, std::size_t end)
                                               {
                        for (std::size_t row = begin; row < end; ++row)
                        {
                            mcl::Fr numerator = base_numerator;
                            mcl::Fr::add(numerator, numerator, folded_values_native[row]);
                            mcl::Fr::mul(quotient_values_native[row], numerator, quotient_values_native[row]);
                        } });
                    if (profile != nullptr)
                        profile->quotient_assembly_ms += elapsed_ms(assembly_start, Clock::now());
                }
                else if (profile != nullptr)
                {
                    const auto fused_ms = elapsed_ms(fused_start, Clock::now());
                    profile->denominator_inverse_ms += fused_ms;
                    profile->denominator_inverse_cache_misses += 1;
                }

                const auto witness_msm_start =
                    Clock::now();

                CommitObjectProfile
                    witness_commit_profile;

                out = g1_add(
                    out,
                    commit_borrowed_evaluation_point(
                        "folded_" +
                            group.domain->name +
                            "_witness",
                        opening_evaluation_scratch,
                        group.domain,
                        pk,
                        profile != nullptr
                            ? &witness_commit_profile
                            : nullptr,
                        msm_options));

                if (profile != nullptr &&
                    progress_enabled())
                {
                    progress_note(
                        "kzg:opening_witness_msm:domain=" +
                        group.domain->name +
                        ":input=" +
                        std::to_string(
                            witness_commit_profile
                                .vector_length) +
                        ":nonzero=" +
                        std::to_string(
                            witness_commit_profile
                                .nonzero_scalar_count) +
                        ":effective=" +
                        std::to_string(
                            witness_commit_profile
                                .effective_scalar_count) +
                        ":sparse=" +
                        std::string(
                            witness_commit_profile
                                    .sparse_compaction
                                ? "true"
                                : "false") +
                        ":tail=" +
                        std::string(
                            witness_commit_profile
                                    .constant_tail_folding
                                ? "true"
                                : "false") +
                        ":ms=" +
                        std::to_string(
                            witness_commit_profile
                                .msm_ms));
                }

                if (profile != nullptr)
                {
                    profile->witness_msm_ms +=
                        elapsed_ms(
                            witness_msm_start,
                            Clock::now());
                }

                // The domain-value quotient above is the unique degree < n
                // interpolant of the full quotient on H.  For
                // P^zk=P+Z_H R, the omitted high-degree part is exactly
                // Z_H * (R(X)-R(z))/(X-z); its quotient contribution is the
                // small coefficient polynomial (R(X)-R(z))/(X-z).
                const auto blinder_witness_start = Clock::now();
                std::vector<FieldElement> blinder_quotient_fold;
                if (!group.folded_zero_domain_blinder.empty())
                {
                    const auto blinder_value = algebra::horner(
                        group.folded_zero_domain_blinder,
                        point);
                    (void)coefficient_single_point_quotient_coefficients(
                        &blinder_quotient_fold,
                        group.folded_zero_domain_blinder,
                        point,
                        blinder_value);
                }
                else
                    for (const auto &term : group.terms)
                    {
                        if (term.zero_domain_blinder == nullptr || term.zero_domain_blinder->empty() || term.scale.is_zero())
                        {
                            continue;
                        }
                        std::vector<FieldElement> quotient;
                        const auto blinder_value = algebra::horner(
                            *term.zero_domain_blinder,
                            point);
                        if (coefficient_single_point_quotient_coefficients(
                                &quotient,
                                *term.zero_domain_blinder,
                                point,
                                blinder_value))
                        {
                            accumulate_scaled_coefficients(
                                &blinder_quotient_fold,
                                quotient,
                                term.scale,
                                worker_cap);
                        }
                    }
                trim_zero_coefficients(&blinder_quotient_fold);
                if (!blinder_quotient_fold.empty())
                {
                    out = g1_add(
                        out,
                        commit_zero_domain_multiple_point(
                            group.domain->size,
                            blinder_quotient_fold,
                            pk,
                            "folded_" + group.domain->name + "_blinder_witness"));
                }
                if (profile != nullptr)
                {
                    profile->blinder_witness_ms += elapsed_ms(
                        blinder_witness_start, Clock::now());
                }
            }
            return out;
        }

        G1Point folded_commitment_points(
            const std::vector<Commitment> &commitments,
            const std::vector<FieldElement> &powers)
        {
            if (powers.size() != commitments.size())
            {
                throw std::runtime_error("folded commitment power count mismatch");
            }
            if (commitments.empty())
            {
                return g1_zero();
            }
            std::vector<G1Point> points;
            points.reserve(commitments.size());
            for (const auto &commitment : commitments)
            {
                points.push_back(commitment.point);
            }
            return g1_multi_scalar_mul(
                points.data(), powers.data(), powers.size());
        }

        G1Point reference_g1_multi_scalar_mul(
            const G1Point *points,
            const FieldElement *scalars,
            std::size_t count)
        {
            if (count != 0 &&
                (points == nullptr || scalars == nullptr))
            {
                throw std::runtime_error(
                    "reference G1 MSM received null input");
            }

            auto out = g1_zero();

            for (std::size_t index = 0;
                 index < count;
                 ++index)
            {
                if (scalars[index].is_zero())
                {
                    continue;
                }

                out = g1_add(
                    out,
                    g1_mul(
                        points[index],
                        scalars[index]));
            }

            return out;
        }

        G1Point reference_folded_commitment_points(
            const std::vector<Commitment> &commitments,
            const std::vector<FieldElement> &powers)
        {
            if (commitments.size() != powers.size())
            {
                throw std::runtime_error(
                    "reference folded commitment power count mismatch");
            }

            auto out = g1_zero();

            for (std::size_t index = 0;
                 index < commitments.size();
                 ++index)
            {
                if (powers[index].is_zero())
                {
                    continue;
                }

                out = g1_add(
                    out,
                    g1_mul(
                        commitments[index].point,
                        powers[index]));
            }

            return out;
        }

        G1Point reference_folded_opening_plan_commitment(
            const FoldedOpeningPlan &plan,
            const KZGProverKey &pk)
        {
            auto out = g1_zero();

            if (!plan.coefficient_fold.empty())
            {
                if (pk.powers_g1 == nullptr ||
                    plan.coefficient_fold.size() >
                        pk.powers_g1->size() ||
                    plan.coefficient_fold.size() >
                        pk.degree_bound + 1)
                {
                    throw std::runtime_error(
                        "reference folded coefficient commitment "
                        "exceeds SRS degree bound");
                }

                out = g1_add(
                    out,
                    reference_g1_multi_scalar_mul(
                        pk.powers_g1->data(),
                        plan.coefficient_fold.data(),
                        plan.coefficient_fold.size()));
            }

            for (const auto &group :
                 plan.evaluation_groups)
            {
                if (group.domain == nullptr ||
                    group.folded_values.size() !=
                        group.domain->size)
                {
                    throw std::runtime_error(
                        "reference folded evaluation group "
                        "has invalid domain shape");
                }

                auto descriptor =
                    algebra::Polynomial::from_evaluations(
                        "kzg_fold_reference_" +
                            group.domain->name,
                        group.folded_values,
                        group.domain);

                const auto *lagrange_srs =
                    evaluation_srs_points(
                        descriptor,
                        pk);

                if (lagrange_srs == nullptr ||
                    group.folded_values.size() >
                        lagrange_srs->size())
                {
                    throw std::runtime_error(
                        "reference folded evaluation commitment "
                        "exceeds Lagrange SRS");
                }

                out = g1_add(
                    out,
                    reference_g1_multi_scalar_mul(
                        lagrange_srs->data(),
                        group.folded_values.data(),
                        group.folded_values.size()));

                auto folded_blinder =
                    group.folded_zero_domain_blinder;

                if (folded_blinder.empty())
                {
                    for (const auto &term :
                         group.terms)
                    {
                        if (term.zero_domain_blinder ==
                                nullptr ||
                            term.zero_domain_blinder
                                ->empty() ||
                            term.scale.is_zero())
                        {
                            continue;
                        }

                        accumulate_scaled_coefficients(
                            &folded_blinder,
                            *term.zero_domain_blinder,
                            term.scale,
                            1);
                    }

                    trim_zero_coefficients(
                        &folded_blinder);
                }

                if (!folded_blinder.empty())
                {
                    out = g1_add(
                        out,
                        commit_zero_domain_multiple_point(
                            group.domain->size,
                            folded_blinder,
                            pk,
                            "kzg_fold_reference_" +
                                group.domain->name));
                }
            }

            return out;
        }

        bool folded_opening_diagnostic_enabled(
            std::size_t claim_count)
        {
            const auto *requested =
                std::getenv(
                    "GATZK_KZG_FOLD_DIAGNOSTIC_CLAIMS");

            return requested != nullptr &&
                   std::string(requested) ==
                       std::to_string(claim_count);
        }

        bool verify_pairing_equation(
            const G1Point &numerator,
            const G1Point &witness,
            const FieldElement &point,
            const KZGVerifierKey &vk)
        {
            const auto rhs = g2_sub(vk.g2_tau, g2_mul(vk.g2_one, point));
            if (rhs == g2_zero())
            {
                // tau must remain outside every proof-derived opening point.  A
                // zero [tau-z]_2 makes the KZG equation degenerate and is an SRS
                // compromise, never a valid proof result.
                return false;
            }
            const bool numerator_is_zero = numerator == g1_zero();
            const bool witness_is_zero = witness == g1_zero();
            if (numerator_is_zero || witness_is_zero)
            {
                // The identically-zero folded polynomial has the valid
                // equation e(0,G2)=e(0,[tau-z]2).  Handle it explicitly since
                // the pairing backend rejects zero affine inputs.
                return numerator_is_zero && witness_is_zero;
            }
            if (util::route2_options().fast_verify_pairing && vk.g2_one_prepared != nullptr)
            {
                return pairing_equal_with_prepared_rhs_a(numerator, *vk.g2_one_prepared, witness, rhs);
            }
            return pairing_equal(numerator, vk.g2_one, witness, rhs);
        }
    }

    std::string canonical_kzg_key_resource_identity(
        const std::filesystem::path &srs_root,
        const SRSManifest &manifest,
        const std::vector<std::string> &selected_lagrange_domains)
    {
        return canonical_key_resource_identity(
            srs_root, manifest, selected_lagrange_domains);
    }

    struct EvaluationCommitmentCache::Impl
    {
        struct PointerKey
        {
            const algebra::RootOfUnityDomain *domain = nullptr;
            const std::vector<G1Point> *lagrange_srs = nullptr;
            const std::vector<algebra::FieldElement> *values = nullptr;

            bool operator==(const PointerKey &other) const
            {
                return domain == other.domain && lagrange_srs == other.lagrange_srs && values == other.values;
            }
        };

        struct PointerKeyHash
        {
            std::size_t operator()(const PointerKey &key) const
            {
                auto hash = static_cast<std::size_t>(
                    reinterpret_cast<std::uintptr_t>(key.domain));
                hash ^= static_cast<std::size_t>(
                            reinterpret_cast<std::uintptr_t>(key.lagrange_srs)) +
                        0x9e3779b9U + (hash << 6U) + (hash >> 2U);
                hash ^= static_cast<std::size_t>(
                            reinterpret_cast<std::uintptr_t>(key.values)) +
                        0x9e3779b9U + (hash << 6U) + (hash >> 2U);
                return hash;
            }
        };

        struct Entry
        {
            const algebra::RootOfUnityDomain *domain = nullptr;
            const std::vector<G1Point> *lagrange_srs = nullptr;
            std::weak_ptr<const std::vector<algebra::FieldElement>> values;
            G1Point point;
        };

        struct ScaledEntry
        {
            const algebra::RootOfUnityDomain *domain = nullptr;
            const std::vector<G1Point> *lagrange_srs = nullptr;
            std::weak_ptr<const std::vector<algebra::FieldElement>> values;
            std::size_t anchor_index = 0;
            algebra::FieldElement anchor_inverse =
                algebra::FieldElement::zero();
            G1Point point;
        };

        struct PointerEntry
        {
            std::weak_ptr<const std::vector<algebra::FieldElement>> values;
            G1Point point;
        };

        struct PointerHashEntry
        {
            std::weak_ptr<const std::vector<algebra::FieldElement>> values;
            std::uint64_t hash = 0;
        };

        std::mutex mutex;
        std::unordered_map<std::uint64_t, std::vector<Entry>> entries;
        std::unordered_map<std::uint64_t, std::vector<ScaledEntry>>
            scaled_entries;
        std::unordered_map<PointerKey, PointerEntry, PointerKeyHash>
            pointer_entries;
        std::unordered_map<PointerKey, PointerHashEntry, PointerKeyHash>
            pointer_hashes;
    };

    EvaluationCommitmentCache::EvaluationCommitmentCache()
        : impl_(std::make_unique<Impl>())
    {
    }

    EvaluationCommitmentCache::~EvaluationCommitmentCache() = default;

    namespace
    {
        std::uint64_t evaluation_commitment_cache_key(
            const algebra::RootOfUnityDomain *domain,
            const std::vector<G1Point> *lagrange_srs,
            const std::vector<algebra::FieldElement> &values)
        {
            std::uint64_t hash = 1469598103934665603ULL;
            auto absorb = [&](std::uint64_t word)
            {
                hash ^= word;
                hash *= 1099511628211ULL;
            };
            absorb(reinterpret_cast<std::uintptr_t>(domain));
            absorb(reinterpret_cast<std::uintptr_t>(lagrange_srs));
            absorb(values.size());
            for (const auto &value : values)
            {
                const auto *units = value.native().getUnit();
                for (std::size_t unit = 0;
                     unit < mcl::Fr::getUnitSize(); ++unit)
                {
                    absorb(static_cast<std::uint64_t>(units[unit]));
                }
            }
            return hash;
        }

        struct ScaledEvaluationFingerprint
        {
            std::uint64_t key = 0;
            std::size_t anchor_index = 0;
            algebra::FieldElement anchor_inverse =
                algebra::FieldElement::zero();
        };

        std::optional<ScaledEvaluationFingerprint>
        scaled_evaluation_fingerprint(
            const algebra::RootOfUnityDomain *domain,
            const std::vector<G1Point> *lagrange_srs,
            const std::vector<algebra::FieldElement> &values)
        {
            constexpr std::size_t kMinScaledReuseLength = 8192;
            constexpr std::size_t kMinScaledReuseSupport = 512;
            if (values.size() < kMinScaledReuseLength)
            {
                return std::nullopt;
            }

            std::size_t first_nonzero = 0;
            while (first_nonzero < values.size() && values[first_nonzero].is_zero())
            {
                ++first_nonzero;
            }
            if (first_nonzero == values.size())
            {
                return std::nullopt;
            }
            std::size_t last_nonzero = values.size() - 1;
            while (last_nonzero > first_nonzero && values[last_nonzero].is_zero())
            {
                --last_nonzero;
            }
            const auto support_length =
                last_nonzero - first_nonzero + 1;
            if (support_length < kMinScaledReuseSupport || support_length > values.size() - values.size() / 4)
            {
                return std::nullopt;
            }

            const auto normalizer = values[first_nonzero].inv();
            std::uint64_t hash = 1469598103934665603ULL;
            auto absorb = [&](std::uint64_t word)
            {
                hash ^= word;
                hash *= 1099511628211ULL;
            };
            absorb(reinterpret_cast<std::uintptr_t>(domain));
            absorb(reinterpret_cast<std::uintptr_t>(lagrange_srs));
            absorb(values.size());
            absorb(first_nonzero);
            absorb(last_nonzero);
            for (std::size_t i = first_nonzero;
                 i <= last_nonzero; ++i)
            {
                if (values[i].is_zero())
                {
                    return std::nullopt;
                }
                const auto normalized = values[i] * normalizer;
                const auto *units = normalized.native().getUnit();
                for (std::size_t unit = 0;
                     unit < mcl::Fr::getUnitSize(); ++unit)
                {
                    absorb(static_cast<std::uint64_t>(units[unit]));
                }
            }
            return ScaledEvaluationFingerprint{
                .key = hash,
                .anchor_index = first_nonzero,
                .anchor_inverse = normalizer,
            };
        }

        bool scaled_evaluation_values_match(
            const std::vector<algebra::FieldElement> &candidate,
            const std::vector<algebra::FieldElement> &reference,
            std::size_t anchor_index,
            const algebra::FieldElement &reference_anchor_inverse,
            algebra::FieldElement *scale)
        {
            if (candidate.size() != reference.size() || anchor_index >= candidate.size() || candidate[anchor_index].is_zero())
            {
                return false;
            }
            const auto candidate_scale =
                candidate[anchor_index] * reference_anchor_inverse;
            for (std::size_t i = 0; i < candidate.size(); ++i)
            {
                if (candidate[i] != reference[i] * candidate_scale)
                {
                    return false;
                }
            }
            if (scale != nullptr)
            {
                *scale = candidate_scale;
            }
            return true;
        }

        bool same_shared_owner(
            const std::shared_ptr<const std::vector<algebra::FieldElement>> &
                left,
            const std::shared_ptr<const std::vector<algebra::FieldElement>> &
                right)
        {
            return left != nullptr && right != nullptr && !left.owner_before(right) && !right.owner_before(left);
        }
    }

    bool EvaluationCommitmentCache::lookup_semantic(
        const algebra::RootOfUnityDomain *domain,
        const std::vector<G1Point> *lagrange_srs,
        const std::shared_ptr<const std::vector<algebra::FieldElement>> &values,
        G1Point *point)
    {
        if (domain == nullptr || lagrange_srs == nullptr || values == nullptr)
        {
            return false;
        }
        const Impl::PointerKey pointer_key{
            .domain = domain,
            .lagrange_srs = lagrange_srs,
            .values = values.get(),
        };
        std::optional<std::uint64_t> cached_hash;
        {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            if (auto it = impl_->pointer_entries.find(pointer_key);
                it != impl_->pointer_entries.end())
            {
                const auto retained = it->second.values.lock();
                if (retained != nullptr && retained.get() == values.get() && same_shared_owner(retained, values))
                {
                    if (point != nullptr)
                    {
                        *point = it->second.point;
                    }
                    return true;
                }
                impl_->pointer_entries.erase(it);
            }
            if (auto it = impl_->pointer_hashes.find(pointer_key);
                it != impl_->pointer_hashes.end())
            {
                const auto retained = it->second.values.lock();
                if (retained != nullptr && retained.get() == values.get() && same_shared_owner(retained, values))
                {
                    cached_hash = it->second.hash;
                }
                else
                {
                    impl_->pointer_hashes.erase(it);
                }
            }
        }
        const auto key = cached_hash.has_value()
                             ? *cached_hash
                             : evaluation_commitment_cache_key(domain, lagrange_srs, *values);
        {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            impl_->pointer_hashes.insert_or_assign(
                pointer_key,
                Impl::PointerHashEntry{
                    .values = values,
                    .hash = key,
                });
            if (auto it = impl_->pointer_entries.find(pointer_key);
                it != impl_->pointer_entries.end())
            {
                const auto retained = it->second.values.lock();
                if (retained != nullptr && retained.get() == values.get() && same_shared_owner(retained, values))
                {
                    if (point != nullptr)
                    {
                        *point = it->second.point;
                    }
                    return true;
                }
                impl_->pointer_entries.erase(it);
            }
            auto bucket = impl_->entries.find(key);
            if (bucket != impl_->entries.end())
            {
                auto &candidates = bucket->second;
                for (auto it = candidates.begin();
                     it != candidates.end();)
                {
                    const auto retained = it->values.lock();
                    if (retained == nullptr)
                    {
                        it = candidates.erase(it);
                        continue;
                    }
                    if (it->domain == domain && it->lagrange_srs == lagrange_srs && *retained == *values)
                    {
                        if (point != nullptr)
                        {
                            *point = it->point;
                        }
                        impl_->pointer_entries.insert_or_assign(
                            pointer_key,
                            Impl::PointerEntry{
                                .values = values,
                                .point = it->point,
                            });
                        return true;
                    }
                    ++it;
                }
                if (candidates.empty())
                {
                    impl_->entries.erase(bucket);
                }
            }
        }

        const auto scaled_fingerprint =
            scaled_evaluation_fingerprint(domain, lagrange_srs, *values);
        if (!scaled_fingerprint.has_value())
        {
            return false;
        }
        G1Point reference_point;
        algebra::FieldElement scale = algebra::FieldElement::zero();
        bool scaled_hit = false;
        {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            auto bucket =
                impl_->scaled_entries.find(scaled_fingerprint->key);
            if (bucket != impl_->scaled_entries.end())
            {
                auto &candidates = bucket->second;
                for (auto it = candidates.begin();
                     it != candidates.end();)
                {
                    const auto retained = it->values.lock();
                    if (retained == nullptr)
                    {
                        it = candidates.erase(it);
                        continue;
                    }
                    if (it->domain == domain && it->lagrange_srs == lagrange_srs && scaled_evaluation_values_match(*values, *retained, it->anchor_index, it->anchor_inverse, &scale))
                    {
                        reference_point = it->point;
                        scaled_hit = true;
                        break;
                    }
                    ++it;
                }
                if (!scaled_hit && candidates.empty())
                {
                    impl_->scaled_entries.erase(bucket);
                }
            }
        }
        if (!scaled_hit)
        {
            return false;
        }
        const auto scaled_point = g1_mul(reference_point, scale);
        {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            impl_->entries[key].push_back(Impl::Entry{
                .domain = domain,
                .lagrange_srs = lagrange_srs,
                .values = values,
                .point = scaled_point,
            });
            impl_->pointer_entries.insert_or_assign(
                pointer_key,
                Impl::PointerEntry{
                    .values = values,
                    .point = scaled_point,
                });
        }
        if (point != nullptr)
        {
            *point = scaled_point;
        }
        return true;
    }

    void EvaluationCommitmentCache::store_semantic(
        const algebra::RootOfUnityDomain *domain,
        const std::vector<G1Point> *lagrange_srs,
        std::shared_ptr<const std::vector<algebra::FieldElement>> values,
        const G1Point &point)
    {
        if (domain == nullptr || lagrange_srs == nullptr || values == nullptr)
        {
            return;
        }
        const Impl::PointerKey pointer_key{
            .domain = domain,
            .lagrange_srs = lagrange_srs,
            .values = values.get(),
        };
        std::optional<std::uint64_t> cached_hash;
        {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            if (auto it = impl_->pointer_entries.find(pointer_key);
                it != impl_->pointer_entries.end())
            {
                const auto retained = it->second.values.lock();
                if (retained != nullptr && retained.get() == values.get() && same_shared_owner(retained, values))
                {
                    return;
                }
                impl_->pointer_entries.erase(it);
            }
            if (auto it = impl_->pointer_hashes.find(pointer_key);
                it != impl_->pointer_hashes.end())
            {
                const auto retained = it->second.values.lock();
                if (retained != nullptr && retained.get() == values.get() && same_shared_owner(retained, values))
                {
                    cached_hash = it->second.hash;
                }
                else
                {
                    impl_->pointer_hashes.erase(it);
                }
            }
        }
        const auto key = cached_hash.has_value()
                             ? *cached_hash
                             : evaluation_commitment_cache_key(domain, lagrange_srs, *values);
        const auto scaled_fingerprint =
            scaled_evaluation_fingerprint(domain, lagrange_srs, *values);
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->pointer_hashes.insert_or_assign(
            pointer_key,
            Impl::PointerHashEntry{
                .values = values,
                .hash = key,
            });
        if (auto it = impl_->pointer_entries.find(pointer_key);
            it != impl_->pointer_entries.end())
        {
            const auto retained = it->second.values.lock();
            if (retained != nullptr && retained.get() == values.get() && same_shared_owner(retained, values))
            {
                return;
            }
            impl_->pointer_entries.erase(it);
        }
        auto &bucket = impl_->entries[key];
        for (auto it = bucket.begin(); it != bucket.end();)
        {
            const auto retained = it->values.lock();
            if (retained == nullptr)
            {
                it = bucket.erase(it);
                continue;
            }
            if (it->domain == domain && it->lagrange_srs == lagrange_srs && *retained == *values)
            {
                impl_->pointer_entries.insert_or_assign(
                    pointer_key,
                    Impl::PointerEntry{
                        .values = values,
                        .point = it->point,
                    });
                return;
            }
            ++it;
        }
        bucket.push_back(Impl::Entry{
            .domain = domain,
            .lagrange_srs = lagrange_srs,
            .values = values,
            .point = point,
        });
        if (scaled_fingerprint.has_value())
        {
            auto &scaled_bucket =
                impl_->scaled_entries[scaled_fingerprint->key];
            bool has_scaled_entry = false;
            for (auto it = scaled_bucket.begin();
                 it != scaled_bucket.end();)
            {
                const auto retained = it->values.lock();
                if (retained == nullptr)
                {
                    it = scaled_bucket.erase(it);
                    continue;
                }
                algebra::FieldElement scale;
                if (it->domain == domain && it->lagrange_srs == lagrange_srs && scaled_evaluation_values_match(*values, *retained, it->anchor_index, it->anchor_inverse, &scale))
                {
                    has_scaled_entry = true;
                    break;
                }
                ++it;
            }
            if (!has_scaled_entry)
            {
                scaled_bucket.push_back(Impl::ScaledEntry{
                    .domain = domain,
                    .lagrange_srs = lagrange_srs,
                    .values = values,
                    .anchor_index =
                        scaled_fingerprint->anchor_index,
                    .anchor_inverse =
                        scaled_fingerprint->anchor_inverse,
                    .point = point,
                });
            }
        }
        impl_->pointer_entries.insert_or_assign(
            pointer_key,
            Impl::PointerEntry{
                .values = values,
                .point = point,
            });
    }

    std::size_t serialized_size(const Commitment &commitment)
    {
        return serialized_size(commitment.point);
    }

    std::string required_lagrange_cache_fragment(
        const std::vector<std::pair<std::string, std::size_t>> &required_lagrange_domains)
    {
        if (required_lagrange_domains.empty())
        {
            return "all";
        }
        std::vector<std::string> parts;
        parts.reserve(required_lagrange_domains.size());
        for (const auto &[domain, size] : required_lagrange_domains)
        {
            parts.push_back(srs_domain_name(domain) + ":" + std::to_string(size));
        }
        std::sort(parts.begin(), parts.end());
        std::ostringstream out;
        for (const auto &part : parts)
        {
            out << '|' << part;
        }
        return out.str();
    }

    void add_unique_domain(std::vector<std::string> *domains, const std::string &domain)
    {
        if (domains == nullptr || domain.empty())
        {
            return;
        }
        if (std::find(domains->begin(), domains->end(), domain) == domains->end())
        {
            domains->push_back(domain);
        }
    }

    std::vector<std::string> select_lagrange_srs_domains(
        const SRSManifest &manifest,
        const std::vector<std::pair<std::string, std::size_t>> &required_lagrange_domains)
    {
        if (required_lagrange_domains.empty())
        {
            std::vector<std::string> all_domains;
            all_domains.reserve(manifest.pk_lagrange_paths.size());
            for (const auto &[domain, _] : manifest.pk_lagrange_paths)
            {
                all_domains.push_back(domain);
            }
            std::sort(all_domains.begin(), all_domains.end());
            return all_domains;
        }

        std::vector<std::string> selected;
        for (const auto &[domain_name, domain_size] : required_lagrange_domains)
        {
            const auto base_key = srs_domain_name(domain_name);
            const auto exact_key = srs_exact_domain_name(domain_name, domain_size);
            if (manifest.pk_lagrange_paths.contains(exact_key))
            {
                add_unique_domain(&selected, exact_key);
                continue;
            }
            if (manifest.pk_lagrange_paths.contains(base_key))
            {
                const auto size_it = manifest.domain_sizes.find(base_key);
                if (size_it != manifest.domain_sizes.end() && size_it->second == domain_size)
                {
                    add_unique_domain(&selected, base_key);
                    continue;
                }
            }
            const auto available_size_it = manifest.domain_sizes.find(base_key);
            if (available_size_it != manifest.domain_sizes.end())
            {
                throw std::runtime_error(
                    "SRS manifest lacks Lagrange material for required domain " + domain_name + ":" + std::to_string(domain_size) + "; authenticated " + base_key + " domain size is " + std::to_string(available_size_it->second) + ", so the derived static-table/domain capacity is not covered; "
                                                                                                                                                                                                                                     "refusing to continue or generate an SRS");
            }
            throw std::runtime_error(
                "SRS manifest lacks Lagrange material for required domain " + domain_name + ":" + std::to_string(domain_size) + "; no authenticated domain-size entry exists, refusing to continue");
        }
        std::sort(selected.begin(), selected.end());
        return selected;
    }

    KZGKeyPair KZG::load_keys(
        const std::string &srs_path,
        const std::string &expected_srs_type,
        const std::string &expected_srs_id)
    {
        return load_keys_for_domains(srs_path, expected_srs_type, expected_srs_id, {});
    }

    KZGKeyPair KZG::load_keys_for_domains(
        const std::string &srs_path,
        const std::string &expected_srs_type,
        const std::string &expected_srs_id,
        const std::vector<std::pair<std::string, std::size_t>> &required_lagrange_domains,
        std::size_t required_coefficient_count)
    {
        static std::mutex cache_mutex;
        static std::unordered_map<std::string, KZGKeyPair> cache;
        const auto cache_key =
            srs_path + "|" + expected_srs_type + "|" + expected_srs_id + "|coef=" + std::to_string(required_coefficient_count) + "|lagrange=" + required_lagrange_cache_fragment(required_lagrange_domains);
        // Resolve and authenticate the manifest/catalogue before consulting the
        // process-local cache.  The cache payload stores the resulting
        // manifest/resource identity; a same-path replacement therefore cannot
        // return stale PK/VK/domain-zero-G2 material.
        const auto srs_root = std::filesystem::path(srs_path);
        const auto manifest = load_manifest_file(srs_path);
        validate_manifest(manifest, expected_srs_type, expected_srs_id);
        const auto selected_lagrange_domains =
            select_lagrange_srs_domains(manifest, required_lagrange_domains);
        const auto resource_identity = canonical_key_resource_identity(
            srs_root, manifest, selected_lagrange_domains);
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            if (const auto it = cache.find(cache_key); it != cache.end())
            {
                if (it->second.resource_identity == resource_identity && it->second.pk.resource_identity == resource_identity && it->second.vk.resource_identity == resource_identity)
                    return it->second;
                // Erase stale material before reloading.  Do not let a
                // mismatched payload survive under the same path/config key.
                cache.erase(it);
            }
        }
        const auto srs_load_start = Clock::now();
        validate_protocol_compiler_artifact_material(srs_root, manifest);
        validate_srs_audit_report_material(srs_root, manifest);
        std::string pk_coef_hash;

        const auto pk_coef_load_start =
            Clock::now();

        progress_note(
            "kzg:srs:pk_coef:start");

        auto powers_g1 =
            load_g1_vector_file(
                srs_root / manifest.pk_coef_path,
                &pk_coef_hash,
                required_coefficient_count);

        validate_material_hash_value(
            pk_coef_hash,
            manifest.pk_coef_hash,
            "pk_coef");

        progress_note(
            "kzg:srs:pk_coef:done:ms=" + std::to_string(
                                             elapsed_ms(
                                                 pk_coef_load_start,
                                                 Clock::now())));

        std::unordered_map<std::string, std::shared_ptr<const std::vector<G1Point>>> lagrange_srs;
        for (const auto &domain : selected_lagrange_domains)
        {
            progress_note("kzg:srs:lagrange:" + domain + ":start");
            const auto lagrange_load_start = Clock::now();
            const auto path_it = manifest.pk_lagrange_paths.find(domain);
            if (path_it == manifest.pk_lagrange_paths.end())
            {
                throw std::runtime_error("missing Lagrange SRS path for domain: " + domain);
            }
            const auto hash_it = manifest.pk_lagrange_hashes.find(domain);
            if (hash_it == manifest.pk_lagrange_hashes.end())
            {
                throw std::runtime_error("missing Lagrange SRS hash for domain: " + domain);
            }
            std::string material_hash;
            auto material = load_g1_vector_file(srs_root / path_it->second, &material_hash);
            validate_material_hash_value(material_hash, hash_it->second, "pk_lagrange_" + domain);
            const auto size_it = manifest.domain_sizes.find(domain);
            if (size_it == manifest.domain_sizes.end() || material.size() < size_it->second)
            {
                throw std::runtime_error("Lagrange SRS domain size mismatch: " + domain);
            }
            lagrange_srs.emplace(domain, std::make_shared<const std::vector<G1Point>>(std::move(material)));
            progress_note(
                "kzg:srs:lagrange:" + domain + ":done:ms=" + std::to_string(elapsed_ms(lagrange_load_start, Clock::now())));
        }
        std::string vk_hash;
        progress_note("kzg:srs:vk:start");
        auto loaded_vk = load_vk_file(srs_root / manifest.vk_path, &vk_hash);
        const auto &g2_one = loaded_vk.g2_one;
        const auto &g2_tau = loaded_vk.g2_tau;
        validate_material_hash_value(vk_hash, manifest.vk_hash, "vk");
        std::vector<std::string> ceremony_domains;
        ceremony_domains.reserve(manifest.domain_sizes.size());
        for (const auto &[domain, _] : manifest.domain_sizes)
        {
            (void)_;
            ceremony_domains.push_back(domain);
        }
        std::sort(ceremony_domains.begin(), ceremony_domains.end());
        std::ostringstream public_transcript;
        public_transcript << "gatzk:trusted-srs-public-transcript|"
                          << manifest.srs_id << '|' << manifest.degree_bound << '|'
                          << manifest.pk_coef_hash << '|' << manifest.vk_hash << '|'
                          << canonical_domain_catalog_hash(manifest.domain_sizes) << '|'
                          << manifest.protocol_compiler_artifact_hash << '|'
                          << manifest.vanishing_g2_hash << '|'
                          << manifest.srs_consistency_hash;
        for (const auto &domain : ceremony_domains)
        {
            const auto hash = manifest.pk_lagrange_hashes.find(domain);
            if (hash == manifest.pk_lagrange_hashes.end())
                throw std::runtime_error(
                    "SRS public transcript is missing a Lagrange material hash: " + domain);
            public_transcript << '|' << domain << '=' << hash->second;
        }
        if (sha256_hex(public_transcript.str()) != manifest.ceremony_transcript_hash)
            throw std::runtime_error(
                "SRS public ceremony transcript binding mismatch");
        if (loaded_vk.domain_zero_g2.size() != manifest.domain_sizes.size())
        {
            throw std::runtime_error("verifier key domain zero G2 registry size mismatch");
        }
        for (const auto &[domain_name, _] : manifest.domain_sizes)
        {
            (void)_;
            if (!loaded_vk.domain_zero_g2.contains(domain_name))
            {
                throw std::runtime_error("missing verifier-key zero-polynomial G2 commitment for domain: " + domain_name);
            }
        }
        if (canonical_vanishing_g2_catalog_hash(
                manifest, loaded_vk.domain_zero_g2) != manifest.vanishing_g2_hash)
        {
            throw std::runtime_error(
                "SRS vanishing G2 catalogue hash mismatch");
        }

        // The manifest hashes authenticate bytes, but they do not prove that the
        // coefficient PK and the G2 verifier catalogue were generated from the
        // same toxic-waste scalar.  Audit the relations needed by ordinary KZG
        // and by proof-local model divisibility before exposing either key.
        if (powers_g1.size() < 2 || !pairing_equal(powers_g1[1], g2_one, powers_g1[0], g2_tau))
        {
            throw std::runtime_error("SRS coefficient/G2 tau consistency audit failed");
        }
        for (const auto &domain_name : selected_lagrange_domains)
        {
            const auto size_it = manifest.domain_sizes.find(domain_name);
            const auto zero_it = loaded_vk.domain_zero_g2.find(domain_name);
            if (size_it == manifest.domain_sizes.end() || zero_it == loaded_vk.domain_zero_g2.end() || size_it->second >= powers_g1.size())
            {
                throw std::runtime_error(
                    "SRS zero-polynomial consistency audit lacks coefficient material for domain: " + domain_name);
            }
            const auto zero_g1 = g1_sub(powers_g1[size_it->second], powers_g1[0]);
            if (!pairing_equal(zero_g1, g2_one, powers_g1[0], zero_it->second))
            {
                throw std::runtime_error(
                    "SRS zero-polynomial G2 consistency audit failed for domain: " + domain_name);
            }
        }
        progress_note("kzg:srs:vk:done");

        G1Point g1_generator = g1_zero();
        mcl::bn::hashAndMapToG1(g1_generator.value, "gatzk:kzg:g1");
        if (powers_g1.empty() || !(powers_g1.front() == g1_generator))
        {
            throw std::runtime_error("SRS coefficient powers do not start with the configured G1 generator");
        }

        const KZGProverKey pk{
            .srs_id = manifest.srs_id,
            .srs_type = manifest.srs_type,
            .degree_bound = manifest.degree_bound,
            .g1_generator = g1_generator,
            .powers_g1 = std::make_shared<const std::vector<G1Point>>(std::move(powers_g1)),
            .lagrange_srs_by_domain = std::move(lagrange_srs),
            .manifest_hash = manifest.manifest_hash,
            .resource_identity = resource_identity,
        };
        const KZGVerifierKey vk{
            .srs_id = manifest.srs_id,
            .srs_type = manifest.srs_type,
            .degree_bound = manifest.degree_bound,
            .g1_generator = g1_generator,
            .g2_one = g2_one,
            .g2_tau = g2_tau,
            .domain_zero_g2 = std::move(loaded_vk.domain_zero_g2),
            .g2_one_prepared = std::make_shared<PreparedG2>(prepare_g2(g2_one)),
            .g2_tau_prepared = std::make_shared<PreparedG2>(prepare_g2(g2_tau)),
            .manifest_hash = manifest.manifest_hash,
            .resource_identity = resource_identity,
        };
        const KZGKeyPair key{
            .manifest = manifest,
            .pk = pk,
            .vk = vk,
            .g1_generator = g1_generator,
            .g2_one = g2_one,
            .g2_tau = g2_tau,
            .g2_one_prepared = vk.g2_one_prepared,
            .g2_tau_prepared = vk.g2_tau_prepared,
            .resource_identity = resource_identity,
        };
        // Close the small TOCTOU window between the pre-cache identity scan
        // and parsing the large point files.  A resource replacement during
        // loading is a hard failure, never a cacheable key.
        const auto post_load_resource_identity = canonical_key_resource_identity(
            srs_root, manifest, selected_lagrange_domains);
        if (post_load_resource_identity != resource_identity)
            throw std::runtime_error(
                "SRS key resource identity changed while loading");
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            cache.emplace(cache_key, key);
        }
        progress_note(
            "kzg:srs:done:ms=" + std::to_string(
                                     elapsed_ms(
                                         srs_load_start,
                                         Clock::now())));
        return key;
    }

    Commitment KZG::commit_constant(
        const std::string &name,
        const FieldElement &value,
        const KZGKeyPair &key)
    {
        return make_commitment(name, constant_polynomial(name, value), key.pk);
    }

    Commitment KZG::commit(
        const std::string &name,
        const algebra::Polynomial &polynomial,
        const KZGKeyPair &key,
        CommitObjectProfile *profile)
    {
        return make_commitment(name, polynomial, key.pk, profile);
    }

    algebra::Polynomial KZG::materialize_taylor_shifted_polynomial(
        const Commitment &source,
        const std::string &shifted_name,
        std::size_t configured_degree,
        const FieldElement &delta,
        const KZGKeyPair &key)
    {
        if (source.opening_polynomial == nullptr)
        {
            throw std::runtime_error(
                "Taylor shift is missing source opening material: " + source.name);
        }
        if (configured_degree > key.pk.degree_bound)
        {
            throw std::runtime_error(
                "Taylor shift exceeds the configured coefficient SRS degree bound");
        }
        auto coefficients = algebra::materialize_blinded_coefficients(
            *source.opening_polynomial, configured_degree);
        auto shifted = algebra::coefficient_taylor_shift(
            coefficients, configured_degree, delta);
        return algebra::Polynomial::from_coefficients(
            shifted_name,
            std::move(shifted));
    }

    Commitment KZG::commit_taylor_shifted(
        const Commitment &source,
        const std::string &shifted_name,
        std::size_t configured_degree,
        const FieldElement &delta,
        const KZGKeyPair &key,
        CommitObjectProfile *profile)
    {
        auto shifted = materialize_taylor_shifted_polynomial(
            source,
            shifted_name,
            configured_degree,
            delta,
            key);
        return make_commitment(
            shifted_name,
            shifted,
            key.pk,
            profile);
    }

    std::vector<Commitment> KZG::commit_constant_batch(
        const std::vector<std::pair<std::string, FieldElement>> &named_values,
        const KZGKeyPair &key,
        CommitBatchProfile *profile)
    {
        std::vector<Commitment> out;
        out.reserve(named_values.size());
        const auto msm_start = Clock::now();
        for (const auto &[name, value] : named_values)
        {
            out.push_back(commit_constant(name, value, key));
        }
        if (profile != nullptr)
        {
            profile->msm_ms += elapsed_ms(msm_start, Clock::now());
        }
        return out;
    }

    std::vector<Commitment> KZG::commit_batch(
        const std::vector<
            std::pair<
                std::string,
                const algebra::Polynomial *>>
            &named_polynomials,
        const KZGKeyPair &key,
        CommitBatchProfile *profile,
        EvaluationCommitmentCache *reuse_cache)
    {
        const auto object_count =
            named_polynomials.size();

        std::vector<Commitment>
            out(object_count);

        std::vector<CommitObjectProfile>
            object_profiles(
                object_count);

        std::vector<std::size_t>
            estimated_work(
                object_count,
                0);

        std::vector<bool>
            preexisting_cache_hit(
                object_count,
                false);

        const auto msm_start =
            Clock::now();

        for (std::size_t index = 0;
             index < object_count;
             ++index)
        {
            const auto &[name, polynomial] =
                named_polynomials[index];

            if (polynomial == nullptr)
            {
                throw std::runtime_error(
                    "null polynomial in "
                    "KZG::commit_batch: " +
                    name);
            }

            estimated_work[index] =
                estimated_commitment_msm_work(
                    *polynomial);

            if (reuse_cache != nullptr &&
                polynomial->basis ==
                    algebra::
                        PolynomialBasis::
                            Evaluation &&
                polynomial->shared_data !=
                    nullptr)
            {
                const auto *lagrange_srs =
                    evaluation_srs_points(
                        *polynomial,
                        key.pk);

                G1Point cached_point;

                preexisting_cache_hit[index] =
                    reuse_cache
                        ->lookup_semantic(
                            polynomial
                                ->domain
                                .get(),
                            lagrange_srs,
                            polynomial
                                ->shared_data,
                            &cached_point);
            }
        }

        // Within one KZG call, several independently blinded objects may share
        // exactly the same immutable semantic evaluation vector. Elect one
        // representative before any worker is launched; aliases are committed
        // only after that representative has populated the semantic cache.
        struct SemanticRepresentativeKey
        {
            const algebra::
                RootOfUnityDomain
                    *domain = nullptr;

            const std::vector<G1Point>
                *lagrange_srs = nullptr;

            const std::vector<FieldElement>
                *values = nullptr;

            bool operator==(
                const SemanticRepresentativeKey &)
                const = default;
        };

        struct SemanticRepresentativeKeyHash
        {
            std::size_t operator()(
                const SemanticRepresentativeKey
                    &key) const noexcept
            {
                const auto mix =
                    [](std::size_t seed,
                       std::size_t value)
                {
                    seed ^=
                        value +
                        std::size_t{
                            0x9e3779b97f4a7c15ULL} +
                        (seed << 6U) +
                        (seed >> 2U);

                    return seed;
                };

                const std::hash<
                    const void *>
                    pointer_hash;

                std::size_t hash =
                    pointer_hash(
                        static_cast<const void *>(
                            key.domain));

                hash =
                    mix(
                        hash,
                        pointer_hash(
                            static_cast<
                                const void *>(
                                key.lagrange_srs)));

                hash =
                    mix(
                        hash,
                        pointer_hash(
                            static_cast<
                                const void *>(
                                key.values)));

                return hash;
            }
        };

        std::unordered_map<
            SemanticRepresentativeKey,
            std::size_t,
            SemanticRepresentativeKeyHash>
            semantic_representative_by_identity;

        semantic_representative_by_identity.reserve(
            object_count);

        std::vector<std::size_t>
            semantic_alias_of(
                object_count,
                object_count);

        if (reuse_cache != nullptr)
        {
            for (std::size_t index = 0;
                 index < object_count;
                 ++index)
            {
                if (preexisting_cache_hit[index])
                {
                    continue;
                }

                const auto *polynomial =
                    named_polynomials[index]
                        .second;

                if (polynomial->basis !=
                        algebra::
                            PolynomialBasis::
                                Evaluation ||
                    polynomial->shared_data ==
                        nullptr)
                {
                    continue;
                }

                const auto *lagrange_srs =
                    evaluation_srs_points(
                        *polynomial,
                        key.pk);

                const SemanticRepresentativeKey
                    representative_key{
                        .domain =
                            polynomial->domain
                                .get(),
                        .lagrange_srs =
                            lagrange_srs,
                        .values =
                            polynomial
                                ->shared_data
                                .get(),
                };

                const auto [found, inserted] =
                    semantic_representative_by_identity
                        .emplace(
                            representative_key,
                            index);

                if (!inserted)
                {
                    semantic_alias_of[index] =
                        found->second;
                }
            }
        }

        const auto cpu_count =
            std::max<std::size_t>(
                1,
                std::thread::
                    hardware_concurrency());

        auto commit_one =
            [&](std::size_t index,
                const G1MSMExecutionOptions
                    *options)
        {
            const auto &[name, polynomial] =
                named_polynomials[index];

            if (polynomial != nullptr &&
                polynomial
                        ->cuda_resident_coefficients !=
                    nullptr &&
                polynomial
                        ->cuda_resident_coefficients
                        ->host_transfer !=
                    nullptr)
            {
                polynomial
                    ->cuda_resident_coefficients
                    ->host_transfer
                    ->wait();
            }

            CommitObjectProfile
                object_profile;

            out[index] =
                make_commitment(
                    name,
                    *polynomial,
                    key.pk,
                    profile != nullptr
                        ? &object_profile
                        : nullptr,
                    options,
                    reuse_cache);

            if (profile != nullptr)
            {
                object_profiles[index] =
                    std::move(
                        object_profile);
            }
        };

        std::vector<bool>
            completed(
                object_count,
                false);

        // Already-cached semantic objects perform no large MSM. Resolve them
        // before forming the CUDA waves.
        for (std::size_t index = 0;
             index < object_count;
             ++index)
        {
            if (!preexisting_cache_hit[index])
            {
                continue;
            }

            commit_one(
                index,
                nullptr);

            completed[index] = true;
        }

        std::vector<bool>
            wave_excluded(
                object_count,
                false);

        for (std::size_t index = 0;
             index < object_count;
             ++index)
        {
            wave_excluded[index] =
                completed[index] ||
                semantic_alias_of[index] !=
                    object_count ||
                (named_polynomials[index]
                         .second != nullptr &&
                 named_polynomials[index]
                         .second
                         ->cuda_resident_coefficients !=
                     nullptr);
        }

        const G1MSMExecutionOptions
            wave_msm_options{
                .max_worker_count = 1,
        };

        SameSrsCommitWaveProfile wave_profile;

        const auto wave_completed =
            execute_cuda_same_srs_commit_waves(
                named_polynomials,
                key.pk,
                estimated_work,
                wave_excluded,
                [&](std::size_t index)
                {
                    commit_one(
                        index,
                        &wave_msm_options);
                },
                &wave_profile);

        for (std::size_t index = 0;
             index < object_count;
             ++index)
        {
            completed[index] =
                completed[index] ||
                wave_completed[index];
        }

        // Handle sparse/tail/singleton work with the fallback scheduler after
        // the explicit CUDA waves remove the large dense same-SRS families.
        std::vector<std::size_t>
            remaining_indices;

        std::size_t
            remaining_total_work = 0;

        std::size_t
            remaining_max_work = 0;

        std::size_t
            remaining_uncached_count = 0;

        for (std::size_t index = 0;
             index < object_count;
             ++index)
        {
            if (completed[index] ||
                semantic_alias_of[index] !=
                    object_count)
            {
                continue;
            }

            remaining_indices
                .push_back(index);

            remaining_total_work +=
                estimated_work[index];

            remaining_max_work =
                std::max(
                    remaining_max_work,
                    estimated_work[index]);

            if (estimated_work[index] != 0)
            {
                ++remaining_uncached_count;
            }
        }

        if (!remaining_indices.empty())
        {
            const auto outer_worker_count =
                commit_batch_outer_worker_count(
                    remaining_indices.size(),
                    remaining_total_work,
                    remaining_max_work,
                    cpu_count);

            const G1MSMExecutionOptions
                fallback_msm_options{
                    .max_worker_count =
                        outer_worker_count > 1
                            ? std::max<std::size_t>(
                                  1,
                                  cpu_count /
                                      std::max<std::size_t>(
                                          1,
                                          std::min(
                                              outer_worker_count,
                                              remaining_uncached_count)))
                            : 0,
            };

            std::vector<
                std::pair<
                    std::string,
                    const algebra::Polynomial *>>
                remaining_view;

            remaining_view.reserve(
                remaining_indices.size());

            for (const auto index :
                 remaining_indices)
            {
                remaining_view.push_back(
                    named_polynomials[index]);
            }

            if (outer_worker_count > 1)
            {
                const auto local_order =
                    commitment_work_order(
                        remaining_view);

                std::atomic<std::size_t>
                    next{0};

                g1_parallel_msm_batch_workers(
                    outer_worker_count,
                    [&](std::size_t)
                    {
                        while (true)
                        {
                            const auto position =
                                next.fetch_add(1);

                            if (position >=
                                local_order.size())
                            {
                                break;
                            }

                            const auto index =
                                remaining_indices[local_order[position]];

                            commit_one(
                                index,
                                &fallback_msm_options);
                        }
                    });

                // std::vector<bool> is bit-packed.  Do not let worker
                // threads update distinct proxy bits concurrently because
                // those bits can share one underlying machine word.
                //
                // g1_parallel_msm_batch_workers() is synchronous and
                // propagates worker failures, so every remaining object is
                // known complete when control reaches this point.
                for (const auto index :
                     remaining_indices)
                {
                    completed[index] = true;
                }
            }
            else
            {
                for (const auto index :
                     remaining_indices)
                {
                    commit_one(
                        index,
                        nullptr);

                    completed[index] = true;
                }
            }
        }

        // Semantic aliases intentionally run after their representative.  The
        // representative has populated EvaluationCommitmentCache, so these
        // objects reuse its evaluation component and pay only their own
        // zero-domain blinder adjustment.
        for (std::size_t index = 0;
             index < object_count;
             ++index)
        {
            if (semantic_alias_of[index] ==
                object_count)
            {
                continue;
            }

            const auto representative =
                semantic_alias_of[index];

            if (!completed[representative])
            {
                throw std::runtime_error(
                    "semantic commitment "
                    "representative was not "
                    "completed before its alias");
            }

            commit_one(
                index,
                nullptr);

            completed[index] = true;
        }

        if (std::any_of(
                completed.begin(),
                completed.end(),
                [](bool value)
                {
                    return !value;
                }))
        {
            throw std::runtime_error(
                "KZG commitment wave left "
                "an object uncommitted");
        }

        if (profile != nullptr)
        {
            profile->used_same_srs_mixed_prefix =
                profile->used_same_srs_mixed_prefix ||
                wave_profile.used_mixed_prefix;
            profile->used_same_srs_exact_count =
                profile->used_same_srs_exact_count ||
                wave_profile.used_exact_count;
            profile->msm_ms +=
                elapsed_ms(
                    msm_start,
                    Clock::now());

            for (auto &object_profile :
                 object_profiles)
            {
                if (!object_profile
                         .name.empty())
                {
                    profile->objects
                        .push_back(
                            std::move(
                                object_profile));
                }
            }
        }

        return out;
    }
    G1Point KZG::evaluation_commitment_component(
        const Commitment &commitment,
        const KZGKeyPair &key)
    {
        if (commitment.opening_polynomial == nullptr)
        {
            throw std::runtime_error(
                "missing opening material for evaluation commitment component: " + commitment.name);
        }
        const auto &polynomial = *commitment.opening_polynomial;
        if (polynomial.basis != algebra::PolynomialBasis::Evaluation || polynomial.domain == nullptr)
        {
            throw std::runtime_error(
                "linear evaluation commitment dependency is not an evaluation polynomial: " + commitment.name);
        }
        const auto *lagrange_srs =
            evaluation_srs_points(polynomial, key.pk);
        if (polynomial.size() > lagrange_srs->size())
        {
            throw std::runtime_error(
                "linear evaluation commitment dependency exceeds its exact Lagrange SRS: " + commitment.name);
        }
        if (!polynomial.is_zero_domain_blinded())
        {
            return commitment.point;
        }
        return g1_sub(
            commitment.point,
            commit_zero_domain_multiple_point(
                polynomial.domain->size,
                polynomial.zero_domain_blinder,
                key.pk,
                commitment.name));
    }

    Commitment KZG::commit_evaluation_from_component(
        const std::string &name,
        const algebra::Polynomial &polynomial,
        const G1Point &evaluation_component,
        const KZGKeyPair &key)
    {
        if (polynomial.basis != algebra::PolynomialBasis::Evaluation || polynomial.domain == nullptr)
        {
            throw std::runtime_error(
                "linear evaluation commitment target is not an evaluation polynomial: " + name);
        }
        const auto *lagrange_srs =
            evaluation_srs_points(polynomial, key.pk);
        if (polynomial.size() > lagrange_srs->size())
        {
            throw std::runtime_error(
                "linear evaluation commitment target exceeds its exact Lagrange SRS: " + name);
        }
        auto point = evaluation_component;
        if (polynomial.is_zero_domain_blinded())
        {
            point = g1_add(
                point,
                commit_zero_domain_multiple_point(
                    polynomial.domain->size,
                    polynomial.zero_domain_blinder,
                    key.pk,
                    name));
        }
        auto material =
            std::make_shared<algebra::Polynomial>(polynomial);
        material->name = name;
        return Commitment{
            .name = name,
            .point = point,
            .opening_polynomial = std::move(material),
        };
    }

    std::vector<Commitment> KZG::commit_owned_batch(
        std::vector<
            std::pair<
                std::string,
                algebra::Polynomial>>
            &&named_polynomials,
        const KZGKeyPair &key,
        CommitBatchProfile *profile,
        const std::string &progress_prefix)
    {
        const auto object_count =
            named_polynomials.size();

        std::vector<Commitment>
            out(object_count);

        std::vector<CommitObjectProfile>
            object_profiles(
                object_count);

        std::vector<std::size_t>
            estimated_work(
                object_count,
                0);

        std::vector<
            std::pair<
                std::string,
                const algebra::Polynomial *>>
            view;

        view.reserve(
            object_count);

        for (std::size_t index = 0;
             index < object_count;
             ++index)
        {
            auto &[name, polynomial] =
                named_polynomials[index];

            estimated_work[index] =
                estimated_commitment_msm_work(
                    polynomial);

            view.push_back(
                {
                    name,
                    &polynomial,
                });
        }

        const auto msm_start =
            Clock::now();

        const auto cpu_count =
            std::max<std::size_t>(
                1,
                std::thread::
                    hardware_concurrency());

        auto commit_one =
            [&](std::size_t index,
                const G1MSMExecutionOptions
                    *options)
        {
            auto &[name, polynomial] =
                named_polynomials[index];

            if (!progress_prefix.empty())
            {
                progress_note(
                    progress_prefix +
                    ":" +
                    name +
                    ":start");
            }

            CommitObjectProfile
                object_profile;

            polynomial.name =
                name;

            auto material =
                std::make_shared<
                    algebra::Polynomial>(
                    std::move(
                        polynomial));

            const auto point =
                commit_polynomial_point(
                    *material,
                    key.pk,
                    profile != nullptr
                        ? &object_profile
                        : nullptr,
                    options);

            finalize_cuda_resident_coefficient_material(
                material.get());

            out[index] =
                Commitment{
                    .name = name,
                    .point = point,
                    .opening_polynomial =
                        material,
            };

            if (profile != nullptr)
            {
                object_profiles[index] =
                    std::move(
                        object_profile);
            }

            if (!progress_prefix.empty())
            {
                progress_note(
                    progress_prefix +
                    ":" +
                    name +
                    ":done");
            }
        };

        std::vector<bool>
            excluded(
                object_count,
                false);

        for (std::size_t index = 0;
             index < object_count;
             ++index)
        {
            excluded[index] =
                view[index].second != nullptr &&
                view[index].second->cuda_resident_coefficients !=
                    nullptr;
        }

        const G1MSMExecutionOptions
            wave_msm_options{
                .max_worker_count = 1,
        };

        SameSrsCommitWaveProfile wave_profile;

        auto completed =
            execute_cuda_same_srs_commit_waves(
                view,
                key.pk,
                estimated_work,
                excluded,
                [&](std::size_t index)
                {
                    commit_one(
                        index,
                        &wave_msm_options);
                },
                &wave_profile);

        std::vector<std::size_t>
            remaining_indices;

        std::size_t
            remaining_total_work = 0;

        std::size_t
            remaining_max_work = 0;

        for (std::size_t index = 0;
             index < object_count;
             ++index)
        {
            if (completed[index])
            {
                continue;
            }

            remaining_indices
                .push_back(index);

            remaining_total_work +=
                estimated_work[index];

            remaining_max_work =
                std::max(
                    remaining_max_work,
                    estimated_work[index]);
        }

        if (!remaining_indices.empty())
        {
            const auto outer_worker_count =
                commit_batch_outer_worker_count(
                    remaining_indices.size(),
                    remaining_total_work,
                    remaining_max_work,
                    cpu_count);

            const G1MSMExecutionOptions
                fallback_msm_options{
                    .max_worker_count =
                        outer_worker_count > 1
                            ? std::max<std::size_t>(
                                  1,
                                  cpu_count /
                                      outer_worker_count)
                            : 0,
            };

            std::vector<
                std::pair<
                    std::string,
                    const algebra::Polynomial *>>
                remaining_view;

            remaining_view.reserve(
                remaining_indices.size());

            for (const auto index :
                 remaining_indices)
            {
                remaining_view.push_back(
                    {
                        named_polynomials[index]
                            .first,
                        &named_polynomials[index]
                             .second,
                    });
            }

            if (outer_worker_count > 1)
            {
                const auto local_order =
                    commitment_work_order(
                        remaining_view);

                std::atomic<std::size_t>
                    next{0};

                g1_parallel_msm_batch_workers(
                    outer_worker_count,
                    [&](std::size_t)
                    {
                        while (true)
                        {
                            const auto position =
                                next.fetch_add(1);

                            if (position >=
                                local_order.size())
                            {
                                break;
                            }

                            const auto index =
                                remaining_indices[local_order[position]];

                            commit_one(
                                index,
                                &fallback_msm_options);
                        }
                    });

                for (const auto index :
                     remaining_indices)
                {
                    completed[index] = true;
                }
            }
            else
            {
                for (const auto index :
                     remaining_indices)
                {
                    commit_one(
                        index,
                        nullptr);

                    completed[index] = true;
                }
            }
        }

        if (std::any_of(
                completed.begin(),
                completed.end(),
                [](bool value)
                {
                    return !value;
                }))
        {
            throw std::runtime_error(
                "owned KZG commitment wave "
                "left an object uncommitted");
        }

        if (profile != nullptr)
        {
            profile->used_same_srs_mixed_prefix =
                profile->used_same_srs_mixed_prefix ||
                wave_profile.used_mixed_prefix;
            profile->used_same_srs_exact_count =
                profile->used_same_srs_exact_count ||
                wave_profile.used_exact_count;
            profile->msm_ms +=
                elapsed_ms(
                    msm_start,
                    Clock::now());

            for (auto &object_profile :
                 object_profiles)
            {
                if (!object_profile
                         .name.empty())
                {
                    profile->objects
                        .push_back(
                            std::move(
                                object_profile));
                }
            }
        }

        return out;
    }
    std::pair<Commitment, Commitment> KZG::reblind_domain_polynomial(
        const Commitment &long_term_anchor,
        const std::string &proof_local_name,
        const std::string &blinder_name,
        const std::string &domain_name,
        std::size_t domain_size,
        std::size_t blinder_coefficient_count,
        const KZGKeyPair &key)
    {
        if (long_term_anchor.opening_polynomial == nullptr || long_term_anchor.opening_polynomial->basis != algebra::PolynomialBasis::Coefficient)
        {
            throw std::runtime_error("model reblinding requires a coefficient-basis anchor polynomial");
        }
        if (domain_size == 0 || blinder_coefficient_count == 0 || domain_size > key.pk.degree_bound || blinder_coefficient_count - 1 > key.pk.degree_bound - domain_size)
        {
            throw std::runtime_error("model reblinding exceeds the configured shifted-KZG degree bound");
        }
        const auto manifest_domain = key.manifest.domain_sizes.find(domain_name);
        if (manifest_domain == key.manifest.domain_sizes.end() || manifest_domain->second != domain_size || !key.vk.domain_zero_g2.contains(domain_name))
        {
            throw std::runtime_error("model reblinding domain is absent or inconsistent in the SRS manifest");
        }

        std::vector<FieldElement> blinder_coefficients(blinder_coefficient_count);
        std::array<std::uint8_t, 64> random_bytes{};
        bool any_nonzero = false;
        for (auto &coefficient : blinder_coefficients)
        {
            if (RAND_bytes(random_bytes.data(), static_cast<int>(random_bytes.size())) != 1)
            {
                throw std::runtime_error("CSPRNG failed while sampling a model reblinder");
            }
            coefficient = FieldElement::from_little_endian_mod(random_bytes.data(), random_bytes.size());
            any_nonzero = any_nonzero || !coefficient.is_zero();
        }
        if (!any_nonzero)
        {
            // A 512-bit reduction is zero only with negligible probability.  A
            // deterministic fallback would violate the fresh-randomness rule,
            // so resample the first coefficient from the CSPRNG instead.
            do
            {
                if (RAND_bytes(random_bytes.data(), static_cast<int>(random_bytes.size())) != 1)
                {
                    throw std::runtime_error("CSPRNG failed while resampling a model reblinder");
                }
                blinder_coefficients.front() =
                    FieldElement::from_little_endian_mod(random_bytes.data(), random_bytes.size());
            } while (blinder_coefficients.front().is_zero());
        }

        auto proof_local_coefficients = long_term_anchor.opening_polynomial->values();
        const auto required_size = domain_size + blinder_coefficients.size();
        proof_local_coefficients.resize(
            std::max(proof_local_coefficients.size(), required_size),
            FieldElement::zero());
        for (std::size_t i = 0; i < blinder_coefficients.size(); ++i)
        {
            proof_local_coefficients[i] -= blinder_coefficients[i];
            proof_local_coefficients[domain_size + i] += blinder_coefficients[i];
        }

        auto commitments = commit_owned_batch(
            {
                {proof_local_name, algebra::Polynomial::from_coefficients(
                                       proof_local_name,
                                       std::move(proof_local_coefficients))},
                {blinder_name, algebra::Polynomial::from_coefficients(
                                   blinder_name,
                                   std::move(blinder_coefficients))},
            },
            key);
        if (commitments.size() != 2 || !verify_domain_divisibility(
                                           commitments[0],
                                           long_term_anchor,
                                           commitments[1],
                                           domain_name,
                                           key.vk))
        {
            throw std::runtime_error("locally generated model reblinding failed its divisibility check");
        }
        return {std::move(commitments[0]), std::move(commitments[1])};
    }
    G1Point KZG::open_point_group(
        const std::vector<Commitment> &commitments,
        const FieldElement &point,
        const std::vector<FieldElement> &claimed_values,
        const FieldElement &folding_challenge,
        const KZGKeyPair &key,
        BatchOpeningProfile *profile,
        const PointGroupOpeningExecutionOptions *execution_options,
        const OpeningPolynomialResolver *opening_resolver,
        const OpeningSemanticSourceResolver *semantic_source_resolver)
    {
        if (commitments.empty() ||
            commitments.size() != claimed_values.size())
        {
            throw std::runtime_error(
                "point-group opening mismatch between commitments and values");
        }

        if (folding_challenge.is_zero())
        {
            throw std::runtime_error(
                "point-group opening folding challenge must be non-zero");
        }

        const auto field_worker_cap =
            execution_options != nullptr
                ? execution_options->max_field_worker_count
                : 0;

        const auto *msm_options =
            execution_options != nullptr
                ? &execution_options->msm
                : nullptr;

        std::vector<std::vector<FieldElement>>
            value_columns;

        value_columns.reserve(
            claimed_values.size());

        for (const auto &value : claimed_values)
        {
            value_columns.push_back({value});
        }

        const auto precompute_start =
            Clock::now();

        const auto powers =
            folding_powers(
                folding_challenge,
                commitments.size());

        const auto folded_plan =
            opening_resolver == nullptr
                ? build_folded_opening_plan(
                      commitments,
                      value_columns,
                      powers,
                      std::vector<FieldElement>{point},
                      field_worker_cap,
                      false)
                : build_streamed_folded_opening_plan(
                      commitments,
                      value_columns,
                      powers,
                      std::vector<FieldElement>{point},
                      field_worker_cap,
                      false,
                      *opening_resolver,
                      semantic_source_resolver,
                      nullptr,
                      profile);

        std::optional<G1Point>
            diagnostic_folded_commitment;

        if (folded_opening_diagnostic_enabled(
                commitments.size()))
        {
            const auto reference_folded_commitment =
                reference_folded_commitment_points(
                    commitments,
                    powers);

            const auto reference_material_commitment =
                reference_folded_opening_plan_commitment(
                    folded_plan,
                    key.pk);

            if (!(reference_folded_commitment ==
                  reference_material_commitment))
            {
                constexpr std::size_t
                    kMaximumReportedMismatches = 8;

                std::vector<std::size_t>
                    mismatch_indices;

                bool isolation_truncated = false;
                bool interaction_only_mismatch = false;

                const auto subset_matches =
                    [&](std::size_t begin,
                        std::size_t end)
                {
                    if (begin >= end ||
                        end > commitments.size())
                    {
                        throw std::runtime_error(
                            "KZG fold diagnostic received invalid "
                            "commitment subset");
                    }

                    std::vector<Commitment>
                        subset_commitments(
                            commitments.begin() +
                                static_cast<std::ptrdiff_t>(begin),
                            commitments.begin() +
                                static_cast<std::ptrdiff_t>(end));

                    std::vector<
                        std::vector<FieldElement>>
                        subset_value_columns;

                    subset_value_columns.reserve(
                        end - begin);

                    std::vector<FieldElement>
                        subset_powers;

                    subset_powers.reserve(
                        end - begin);

                    for (std::size_t index = begin;
                         index < end;
                         ++index)
                    {
                        subset_value_columns.push_back(
                            {claimed_values[index]});

                        subset_powers.push_back(
                            powers[index]);
                    }

                    const auto subset_plan =
                        opening_resolver == nullptr
                            ? build_folded_opening_plan(
                                  subset_commitments,
                                  subset_value_columns,
                                  subset_powers,
                                  std::vector<FieldElement>{
                                      point},
                                  field_worker_cap,
                                  false)
                            : build_streamed_folded_opening_plan(
                                  subset_commitments,
                                  subset_value_columns,
                                  subset_powers,
                                  std::vector<FieldElement>{
                                      point},
                                  field_worker_cap,
                                  false,
                                  *opening_resolver,
                                  semantic_source_resolver,
                                  nullptr,
                                  nullptr);

                    const auto subset_commitment =
                        reference_folded_commitment_points(
                            subset_commitments,
                            subset_powers);

                    const auto subset_material =
                        reference_folded_opening_plan_commitment(
                            subset_plan,
                            key.pk);

                    return subset_commitment ==
                           subset_material;
                };

                std::function<void(
                    std::size_t,
                    std::size_t)>
                    isolate_mismatches;

                isolate_mismatches =
                    [&](std::size_t begin,
                        std::size_t end)
                {
                    if (mismatch_indices.size() >=
                        kMaximumReportedMismatches)
                    {
                        isolation_truncated = true;
                        return;
                    }

                    if (begin >= end)
                    {
                        return;
                    }

                    if (end - begin == 1)
                    {
                        mismatch_indices.push_back(
                            begin);
                        return;
                    }

                    const auto middle =
                        begin +
                        (end - begin) / 2;

                    const bool left_matches =
                        subset_matches(
                            begin,
                            middle);

                    const bool right_matches =
                        subset_matches(
                            middle,
                            end);

                    if (!left_matches)
                    {
                        isolate_mismatches(
                            begin,
                            middle);
                    }

                    if (!right_matches &&
                        mismatch_indices.size() <
                            kMaximumReportedMismatches)
                    {
                        isolate_mismatches(
                            middle,
                            end);
                    }

                    if (left_matches &&
                        right_matches)
                    {
                        interaction_only_mismatch =
                            true;
                    }
                };

                isolate_mismatches(
                    0,
                    commitments.size());

                std::ostringstream diagnostic;

                diagnostic
                    << "KZG fold diagnostic: "
                    << "commitment_material_mismatch"
                    << ";claims="
                    << commitments.size();

                if (!mismatch_indices.empty())
                {
                    diagnostic
                        << ";individual_mismatches=";

                    for (std::size_t position = 0;
                         position <
                         mismatch_indices.size();
                         ++position)
                    {
                        if (position != 0)
                        {
                            diagnostic << ',';
                        }

                        const auto index =
                            mismatch_indices[position];

                        diagnostic
                            << index
                            << ':'
                            << commitments[index].name
                            << ':'
                            << (commitments[index]
                                            .opening_polynomial !=
                                        nullptr
                                    ? "resident"
                                    : "streamed");
                    }
                }
                else
                {
                    diagnostic
                        << ";individual_mismatches=none";
                }

                diagnostic
                    << ";interaction_only="
                    << (interaction_only_mismatch
                            ? "true"
                            : "false")
                    << ";truncated="
                    << (isolation_truncated
                            ? "true"
                            : "false");

                throw std::runtime_error(
                    diagnostic.str());
            }

            diagnostic_folded_commitment =
                reference_folded_commitment;
        }

        if (profile != nullptr)
        {
            profile->precompute_ms +=
                elapsed_ms(
                    precompute_start,
                    Clock::now());

            profile->point_msm_worker_cap =
                msm_options != nullptr
                    ? msm_options->max_worker_count
                    : 0;
        }

        const auto witness_start =
            Clock::now();

        const auto witness = [&]()
        {
            try
            {
                return open_single_point_fold(
                    folded_plan,
                    0,
                    point,
                    key.pk,
                    profile,
                    field_worker_cap,
                    msm_options);
            }
            catch (const std::exception &error)
            {
                if (!std::string_view(error.what()).starts_with("opening mismatch for folded coefficient group:"))
                {
                    throw;
                }
                throw std::runtime_error(
                    std::string(error.what()) +
                    coefficient_opening_failure_diagnostic(
                        commitments,
                        claimed_values,
                        point,
                        folded_plan.coefficient_fold,
                        folded_plan.coefficient_scaled_claim_sums.at(0)));
            }
        }();

        if (diagnostic_folded_commitment.has_value())
        {
            FieldElement folded_value =
                FieldElement::zero();

            for (std::size_t index = 0;
                 index < claimed_values.size();
                 ++index)
            {
                folded_value +=
                    claimed_values[index] *
                    powers[index];
            }

            const auto numerator =
                g1_sub(
                    *diagnostic_folded_commitment,
                    g1_mul(
                        key.vk.g1_generator,
                        folded_value));

            if (!verify_pairing_equation(
                    numerator,
                    witness,
                    point,
                    key.vk))
            {
                throw std::runtime_error(
                    "KZG fold diagnostic: "
                    "witness_mismatch_after_"
                    "commitment_material_match; claims=" +
                    std::to_string(
                        commitments.size()));
            }

            throw std::runtime_error(
                "KZG fold diagnostic: "
                "commitment_material_and_witness_match; "
                "inspect proof transport/registry/challenge replay; "
                "claims=" +
                std::to_string(
                    commitments.size()));
        }

        if (profile != nullptr)
        {
            profile->finalize_ms +=
                elapsed_ms(
                    witness_start,
                    Clock::now());
        }

        return witness;
    }

    std::array<G1Point, 2>
    KZG::open_point_group_pair_shared_stream(
        const std::array<
            std::vector<Commitment>,
            2> &commitments,
        const std::array<FieldElement, 2> &points,
        const std::array<
            std::vector<FieldElement>,
            2> &claimed_values,
        const std::array<FieldElement, 2>
            &folding_challenges,
        const KZGKeyPair &key,
        std::array<BatchOpeningProfile, 2> *profiles,
        const std::array<
            PointGroupOpeningExecutionOptions,
            2> *execution_options,
        const OpeningPolynomialResolver *opening_resolver,
        const OpeningSemanticSourceResolver
            *semantic_source_resolver)
    {
        for (std::size_t index = 0;
             index < 2;
             ++index)
        {
            if (commitments[index].empty() ||
                commitments[index].size() !=
                    claimed_values[index].size())
            {
                throw std::runtime_error(
                    "paired point-group opening input shape mismatch");
            }

            if (folding_challenges[index].is_zero())
            {
                throw std::runtime_error(
                    "paired point-group opening folding challenge must be non-zero");
            }
        }

        if (opening_resolver == nullptr ||
            semantic_source_resolver == nullptr)
        {
            throw std::runtime_error(
                "paired streamed point-group opening requires prevalidated claims and streamed resolvers");
        }

        std::array<
            std::vector<std::vector<FieldElement>>,
            2>
            value_columns;

        std::array<
            std::vector<FieldElement>,
            2>
            powers;

        std::array<std::size_t, 2>
            field_worker_caps{0, 0};

        for (std::size_t group = 0;
             group < 2;
             ++group)
        {
            value_columns[group].reserve(
                claimed_values[group].size());

            for (const auto &value :
                 claimed_values[group])
            {
                value_columns[group].push_back(
                    {value});
            }

            powers[group] =
                folding_powers(
                    folding_challenges[group],
                    commitments[group].size());

            if (execution_options != nullptr)
            {
                field_worker_caps[group] =
                    (*execution_options)[group]
                        .max_field_worker_count;
            }
        }

        const auto precompute_start =
            Clock::now();

        const std::array<BatchOpeningProfile *, 2>
            scan_profiles{
                profiles != nullptr ? &(*profiles)[0] : nullptr,
                profiles != nullptr ? &(*profiles)[1] : nullptr,
        };

        auto folded =
            build_streamed_folded_opening_plan_pair(
                commitments,
                value_columns,
                powers,
                points,
                field_worker_caps,
                *opening_resolver,
                *semantic_source_resolver,
                &scan_profiles);

        const auto precompute_ms =
            elapsed_ms(
                precompute_start,
                Clock::now());

        progress_note(
            "kzg:opening:paired_semantic_scan:"
            "fused_sources=" +
            std::to_string(
                folded
                    .fused_semantic_source_count) +
            ":first_sources=" +
            std::to_string(
                folded
                    .deferred_source_count[0]) +
            ":second_sources=" +
            std::to_string(
                folded
                    .deferred_source_count[1]) +
            ":wall_ms=" +
            std::to_string(
                precompute_ms));

        if (profiles != nullptr)
        {
            for (std::size_t group = 0;
                 group < 2;
                 ++group)
            {
                (*profiles)[group]
                    .precompute_ms +=
                    precompute_ms / 2.0;

                if (execution_options != nullptr)
                {
                    (*profiles)[group]
                        .point_msm_worker_cap =
                        (*execution_options)[group]
                            .msm.max_worker_count;
                }
            }
        }

        std::array<G1Point, 2> witnesses{
            g1_zero(),
            g1_zero(),
        };

        const auto open_one =
            [&](std::size_t group)
        {
            const auto local_start =
                Clock::now();

            const auto *msm_options =
                execution_options != nullptr
                    ? &(*execution_options)[group].msm
                    : nullptr;

            auto *profile =
                profiles != nullptr
                    ? &(*profiles)[group]
                    : nullptr;

            try
            {
                witnesses[group] = open_single_point_fold(
                    folded.plans[group],
                    0,
                    points[group],
                    key.pk,
                    profile,
                    field_worker_caps[group],
                    msm_options);
            }
            catch (const std::exception &error)
            {
                if (!std::string_view(error.what()).starts_with("opening mismatch for folded coefficient group:"))
                {
                    throw;
                }
                throw std::runtime_error(
                    std::string(error.what()) +
                    coefficient_opening_failure_diagnostic(
                        commitments[group],
                        claimed_values[group],
                        points[group],
                        folded.plans[group].coefficient_fold,
                        folded.plans[group].coefficient_scaled_claim_sums.at(0)));
            }

            if (profile != nullptr)
            {
                profile->finalize_ms +=
                    elapsed_ms(
                        local_start,
                        Clock::now());
            }
        };

        auto first_future =
            std::async(
                std::launch::async,
                [&]()
                {
                    open_one(0);
                });

        auto second_future =
            std::async(
                std::launch::async,
                [&]()
                {
                    open_one(1);
                });

        first_future.get();
        second_future.get();

        return witnesses;
    }

    bool KZG::verify_point_group(
        const std::vector<Commitment> &commitments,
        const FieldElement &point,
        const std::vector<FieldElement> &claimed_values,
        const FieldElement &folding_challenge,
        const G1Point &witness,
        const KZGVerifierKey &vk)
    {
        if (commitments.empty() || commitments.size() != claimed_values.size())
        {
            throw std::runtime_error("point-group verification shape mismatch");
        }
        if (folding_challenge.is_zero())
        {
            throw std::runtime_error(
                "point-group verification folding challenge must be non-zero");
        }

        const auto powers = folding_powers(folding_challenge, commitments.size());
        const auto folded_commitments = folded_commitment_points(
            commitments, powers);
        mcl::Fr sum;
        sum.clear();
        for (std::size_t i = 0; i < claimed_values.size(); ++i)
        {
            mcl::Fr product;
            mcl::Fr::mul(
                product,
                claimed_values[i].native(),
                powers[i].native());
            mcl::Fr::add(sum, sum, product);
        }
        const auto folded_value = FieldElement::from_native(sum);
        const auto numerator = g1_sub(
            folded_commitments,
            g1_mul(vk.g1_generator, folded_value));
        return verify_pairing_equation(numerator, witness, point, vk);
    }

    bool KZG::verify_domain_divisibility(
        const Commitment &proof_local,
        const Commitment &long_term_anchor,
        const Commitment &blinder,
        const std::string &domain_name,
        const KZGVerifierKey &vk)
    {
        const auto zero_it = vk.domain_zero_g2.find(domain_name);
        if (zero_it == vk.domain_zero_g2.end() || blinder.point == g1_zero())
        {
            return false;
        }
        const auto tau_to_domain_size = g2_add(zero_it->second, vk.g2_one);
        const auto shifted = g1_add(
            g1_sub(proof_local.point, long_term_anchor.point),
            blinder.point);
        if (util::route2_options().fast_verify_pairing && vk.g2_one_prepared != nullptr)
        {
            return pairing_product_is_one_mixed_prepared(
                blinder.point,
                tau_to_domain_size,
                g1_neg(shifted),
                *vk.g2_one_prepared);
        }
        return pairing_equal(
            blinder.point,
            tau_to_domain_size,
            shifted,
            vk.g2_one);
    }
}
