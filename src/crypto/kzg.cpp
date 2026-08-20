#include "gatzk/crypto/kzg.hpp"

#include <algorithm>
#include <array>
#include <atomic>
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
#include <unordered_map>
#include <utility>
#include <deque>

#include <openssl/sha.h>
#include <openssl/rand.h>

#include "gatzk/algebra/domain_evaluation.hpp"
#include "gatzk/algebra/field_io.hpp"
#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/protocol/schema.hpp"
#include "gatzk/util/logging.hpp"
#include "gatzk/util/route2.hpp"

#if defined(__linux__)
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace gatzk::crypto
{
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

        std::string sha256_bytes_hex(const std::vector<std::uint8_t> &data)
        {
            unsigned char digest[SHA256_DIGEST_LENGTH];
            SHA256(data.data(), data.size(), digest);
            return bytes_to_hex(digest, sizeof(digest));
        }

        std::string trim(std::string value)
        {
            const auto begin = value.find_first_not_of(" \t\r\n\"");
            if (begin == std::string::npos)
            {
                return {};
            }
            const auto end = value.find_last_not_of(" \t\r\n\",");
            return value.substr(begin, end - begin + 1);
        }

        std::string manifest_scalar(const std::string &text, const std::string &key)
        {
            const auto quoted_key = "\"" + key + "\"";
            const auto pos = text.find(quoted_key);
            if (pos == std::string::npos)
            {
                return {};
            }
            const auto colon = text.find(':', pos + quoted_key.size());
            if (colon == std::string::npos)
            {
                return {};
            }
            const auto line_end = text.find_first_of("\r\n", colon + 1);
            return trim(text.substr(colon + 1, line_end == std::string::npos ? std::string::npos : line_end - colon - 1));
        }

        std::string manifest_object(const std::string &text, const std::string &key)
        {
            const auto quoted_key = "\"" + key + "\"";
            const auto pos = text.find(quoted_key);
            if (pos == std::string::npos)
            {
                return {};
            }
            const auto open = text.find('{', pos + quoted_key.size());
            if (open == std::string::npos)
            {
                return {};
            }
            std::size_t depth = 0;
            for (std::size_t i = open; i < text.size(); ++i)
            {
                if (text[i] == '{')
                {
                    ++depth;
                }
                else if (text[i] == '}')
                {
                    --depth;
                    if (depth == 0)
                    {
                        return text.substr(open + 1, i - open - 1);
                    }
                }
            }
            throw std::runtime_error("unterminated SRS manifest object: " + key);
        }

        std::unordered_map<std::string, std::string> parse_string_map(const std::string &object)
        {
            std::unordered_map<std::string, std::string> out;
            std::size_t pos = 0;
            while (true)
            {
                const auto key_begin = object.find('"', pos);
                if (key_begin == std::string::npos)
                {
                    break;
                }
                const auto key_end = object.find('"', key_begin + 1);
                const auto value_begin = object.find('"', object.find(':', key_end) + 1);
                const auto value_end = object.find('"', value_begin + 1);
                if (key_end == std::string::npos || value_begin == std::string::npos || value_end == std::string::npos)
                {
                    throw std::runtime_error("invalid string map in SRS manifest");
                }
                out.emplace(object.substr(key_begin + 1, key_end - key_begin - 1),
                            object.substr(value_begin + 1, value_end - value_begin - 1));
                pos = value_end + 1;
            }
            return out;
        }

        std::unordered_map<std::string, std::size_t> parse_size_map(const std::string &object)
        {
            std::unordered_map<std::string, std::size_t> out;
            std::size_t pos = 0;
            while (true)
            {
                const auto key_begin = object.find('"', pos);
                if (key_begin == std::string::npos)
                {
                    break;
                }
                const auto key_end = object.find('"', key_begin + 1);
                const auto colon = object.find(':', key_end);
                if (key_end == std::string::npos || colon == std::string::npos)
                {
                    throw std::runtime_error("invalid size map in SRS manifest");
                }
                const auto number_begin = object.find_first_of("0123456789", colon + 1);
                const auto number_end = object.find_first_not_of("0123456789", number_begin);
                if (number_begin == std::string::npos)
                {
                    throw std::runtime_error("invalid numeric value in SRS manifest");
                }
                out.emplace(object.substr(key_begin + 1, key_end - key_begin - 1),
                            static_cast<std::size_t>(std::stoull(object.substr(number_begin, number_end - number_begin))));
                pos = number_end;
            }
            return out;
        }

        std::string canonical_domain_catalog_hash(
            const std::unordered_map<std::string, std::size_t> &domains)
        {
            std::vector<std::pair<std::string, std::size_t>> ordered(
                domains.begin(), domains.end());
            std::sort(ordered.begin(), ordered.end());
            std::ostringstream canonical;
            canonical << "zkgat:domain-size-catalog:v1";
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
            canonical << "zkgat:domain-generator-catalog:v1"
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
            canonical << "zkgat:vanishing-g2-catalog:v1";
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
            report << "GATZK-SRS-CONSISTENCY-AUDIT-v1\n";
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

        std::string compiler_artifact_scalar(
            const std::string &text,
            const std::string &key)
        {
            const auto prefix = key + "=";
            const auto begin = text.find(prefix);
            if (begin == std::string::npos || (begin != 0 && text[begin - 1] != '\n'))
                throw std::runtime_error(
                    "protocol compiler artifact is missing " + key);
            const auto value_begin = begin + prefix.size();
            const auto end = text.find('\n', value_begin);
            return text.substr(value_begin, end - value_begin);
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
            SRSManifest manifest;
            manifest.manifest_format = manifest_scalar(text, "manifest_format");
            manifest.srs_id = manifest_scalar(text, "srs_id");
            manifest.curve_id = manifest_scalar(text, "curve_id");
            manifest.field_modulus_id = manifest_scalar(text, "field_modulus_id");
            manifest.srs_type = manifest_scalar(text, "srs_type");
            manifest.ceremony_transcript_hash = manifest_scalar(text, "ceremony_transcript_hash");
            manifest.pk_coef_path = manifest_scalar(text, "pk_coef_path");
            manifest.pk_coef_hash = manifest_scalar(text, "pk_coef_hash");
            manifest.pk_lagrange_paths = parse_string_map(manifest_object(text, "pk_lagrange_paths"));
            manifest.pk_lagrange_hashes = parse_string_map(manifest_object(text, "pk_lagrange_hashes"));
            manifest.domain_sizes = parse_size_map(manifest_object(text, "domain_sizes"));
            manifest.domain_generators = parse_string_map(
                manifest_object(text, "domain_generators"));
            manifest.root_derivation_id = manifest_scalar(text, "root_derivation_id");
            manifest.row_order = manifest_scalar(text, "row_order");
            manifest.protocol_compiler_artifact_path =
                manifest_scalar(text, "protocol_compiler_artifact_path");
            manifest.protocol_compiler_artifact_hash =
                manifest_scalar(text, "protocol_compiler_artifact_hash");
            manifest.protocol_schema_id = manifest_scalar(text, "protocol_schema_id");
            manifest.max_instance_envelope_id =
                manifest_scalar(text, "max_instance_envelope_id");
            manifest.max_instance_envelope_hash =
                manifest_scalar(text, "max_instance_envelope_hash");
            manifest.max_instance_envelope = parse_size_map(
                manifest_object(text, "max_instance_envelope"));
            manifest.proof_schema_template_hash =
                manifest_scalar(text, "proof_schema_template_hash");
            manifest.observation_ledger_hash =
                manifest_scalar(text, "observation_ledger_hash");
            manifest.random_identity_ledger_hash =
                manifest_scalar(text, "random_identity_ledger_hash");
            manifest.security_budget_ledger_hash =
                manifest_scalar(text, "security_budget_ledger_hash");
            manifest.vk_path = manifest_scalar(text, "vk_path");
            manifest.vk_hash = manifest_scalar(text, "vk_hash");
            manifest.vanishing_g2_hash =
                manifest_scalar(text, "vanishing_g2_hash");
            manifest.srs_audit_report_path =
                manifest_scalar(text, "srs_audit_report_path");
            manifest.srs_audit_report_hash =
                manifest_scalar(text, "srs_audit_report_hash");
            manifest.srs_consistency_hash =
                manifest_scalar(text, "srs_consistency_hash");
            manifest.encoding = manifest_scalar(text, "encoding");
            manifest.generator_id = manifest_scalar(text, "generator_id");
            manifest.generator_version = manifest_scalar(text, "generator_version");
            manifest.security_level = manifest_scalar(text, "security_level");
            manifest.kzg_assumption_profile_id = manifest_scalar(text, "kzg_assumption_profile_id");
            manifest.kzg_assumption_domain_catalog_hash =
                manifest_scalar(text, "kzg_assumption_domain_catalog_hash");
            manifest.domain_generator_catalog_hash =
                manifest_scalar(text, "domain_generator_catalog_hash");
            const auto declared_manifest_hash = manifest_scalar(text, "manifest_hash");
            const auto manifest_hash_field = text.rfind("  \"manifest_hash\"");
            if (declared_manifest_hash.empty() || manifest_hash_field == std::string::npos || sha256_hex(text.substr(0, manifest_hash_field)) != declared_manifest_hash)
            {
                throw std::runtime_error("SRS manifest canonical hash mismatch");
            }
            manifest.manifest_hash = sha256_hex(text);
            const auto degree_bound = manifest_scalar(text, "degree_bound");
            if (!degree_bound.empty())
            {
                manifest.degree_bound = static_cast<std::size_t>(std::stoull(degree_bound));
            }
            const auto assumption_degree_bound = manifest_scalar(text, "kzg_assumption_degree_bound");
            if (!assumption_degree_bound.empty())
            {
                manifest.kzg_assumption_degree_bound =
                    static_cast<std::size_t>(std::stoull(assumption_degree_bound));
            }
            const auto assumption_security_bits = manifest_scalar(text, "kzg_assumption_security_bits");
            if (!assumption_security_bits.empty())
            {
                manifest.kzg_assumption_security_bits =
                    static_cast<std::size_t>(std::stoull(assumption_security_bits));
            }
            if (manifest.manifest_format.empty() || manifest.srs_id.empty() || manifest.srs_type.empty() || manifest.degree_bound == 0 || manifest.pk_coef_path.empty() || manifest.pk_lagrange_paths.empty() || manifest.vk_path.empty() || manifest.vanishing_g2_hash.empty() || manifest.srs_audit_report_path.empty() || manifest.srs_audit_report_hash.empty() || manifest.srs_consistency_hash.empty() || manifest.protocol_compiler_artifact_path.empty() || manifest.protocol_compiler_artifact_hash.empty())
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
            constexpr std::string_view magic =
                "GATZK-SRS-COMPILER-ARTIFACT-v3\n";
            if (!artifact.starts_with(magic))
                throw std::runtime_error("protocol compiler artifact magic mismatch");
            const auto hash_field = artifact.rfind("artifact_hash=");
            if (hash_field == std::string::npos || compiler_artifact_scalar(artifact, "artifact_hash") != sha256_hex(artifact.substr(0, hash_field)))
                throw std::runtime_error(
                    "protocol compiler artifact canonical hash mismatch");
            const auto protocol_template_hex =
                compiler_artifact_scalar(artifact, "protocol_template_hex");
            const auto proof_schema_template_hex =
                compiler_artifact_scalar(artifact, "proof_schema_template_hex");
            const auto observation_ledger_template_hex =
                compiler_artifact_scalar(artifact, "observation_ledger_template_hex");
            const auto random_identity_ledger_template_hex =
                compiler_artifact_scalar(artifact, "random_identity_ledger_template_hex");
            const auto security_budget_ledger_template_hex =
                compiler_artifact_scalar(artifact, "security_budget_ledger_template_hex");
            const auto max_instance_envelope_hex =
                compiler_artifact_scalar(artifact, "max_instance_envelope_hex");
            const auto max_instance_envelope_hash =
                compiler_artifact_scalar(artifact, "max_instance_envelope_hash");
            if (protocol_template_hex.empty() || proof_schema_template_hex.empty() || observation_ledger_template_hex.empty() || random_identity_ledger_template_hex.empty() || security_budget_ledger_template_hex.empty() || max_instance_envelope_hex.empty() || max_instance_envelope_hash.empty() || compiler_artifact_scalar(artifact, "protocol_schema_id") != manifest.protocol_schema_id || sha256_hex(decode_canonical_hex(protocol_template_hex)) != manifest.protocol_schema_id || compiler_artifact_scalar(artifact, "max_instance_envelope_id") != manifest.max_instance_envelope_id || max_instance_envelope_hash != manifest.max_instance_envelope_hash || max_instance_envelope_hash != sha256_hex(decode_canonical_hex(max_instance_envelope_hex)) || static_cast<std::size_t>(std::stoull(compiler_artifact_scalar(artifact, "degree_bound"))) != manifest.degree_bound || compiler_artifact_scalar(artifact, "domain_size_catalog_hash") != manifest.kzg_assumption_domain_catalog_hash || compiler_artifact_scalar(artifact, "domain_generator_catalog_hash") != manifest.domain_generator_catalog_hash || compiler_artifact_scalar(artifact, "proof_schema_template_hash") != manifest.proof_schema_template_hash || compiler_artifact_scalar(artifact, "observation_ledger_hash") != manifest.observation_ledger_hash || compiler_artifact_scalar(artifact, "random_identity_ledger_hash") != manifest.random_identity_ledger_hash || compiler_artifact_scalar(artifact, "security_budget_ledger_hash") != manifest.security_budget_ledger_hash)
                throw std::runtime_error(
                    "protocol compiler artifact registries do not match the SRS manifest");
            const auto envelope = decode_canonical_hex(max_instance_envelope_hex);
            const std::vector<std::string> envelope_keys = {
                "format", "max_layers", "max_hidden_heads_per_layer",
                "max_hidden_head_dim", "max_input_dim", "max_output_heads",
                "max_classes", "max_graphs", "max_nodes", "max_edges",
                "max_total_nodes", "max_concat_dim", "max_range_bits",
                "max_coefficient_object_length", "max_object_family_count",
                "max_coefficient_chunks", "security_target_bits"};
            std::string canonical_envelope;
            std::unordered_map<std::string, std::size_t> envelope_sizes;
            for (const auto &key : envelope_keys)
            {
                const auto value = compiler_artifact_scalar(envelope, key);
                canonical_envelope += key + "=" + value + "\n";
                if (key == "format")
                {
                    if (value != "zkgat_max_instance_envelope_v2")
                        throw std::runtime_error(
                            "protocol compiler max envelope format mismatch");
                    continue;
                }
                if (value.empty() || (value.size() > 1 && value.front() == '0') || !std::all_of(value.begin(), value.end(), [](unsigned char ch)
                                                                                                { return ch >= '0' && ch <= '9'; }))
                    throw std::runtime_error(
                        "protocol compiler max envelope integer mismatch: " + key);
                const auto parsed = static_cast<std::size_t>(std::stoull(value));
                if (parsed == 0)
                    throw std::runtime_error(
                        "protocol compiler max envelope has zero bound: " + key);
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

        std::uint64_t read_u64_le_bytes(
            const std::vector<std::uint8_t> &bytes,
            std::size_t &offset,
            const std::string &label)
        {
            if (offset + 8 > bytes.size())
            {
                throw std::runtime_error("truncated SRS file while reading " + label);
            }
            std::uint64_t out = 0;
            for (std::size_t i = 0; i < 8; ++i)
            {
                out |= static_cast<std::uint64_t>(bytes[offset + i]) << (8U * i);
            }
            offset += 8;
            return out;
        }

        std::uint64_t parse_u64_le_bytes(const std::array<std::uint8_t, 8> &bytes)
        {
            std::uint64_t out = 0;
            for (std::size_t i = 0; i < bytes.size(); ++i)
            {
                out |= static_cast<std::uint64_t>(bytes[i]) << (8U * i);
            }
            return out;
        }

        std::vector<G1Point> load_g1_vector_file(
            const std::filesystem::path &path,
            std::string *actual_hash,
            std::size_t max_points);

        void validate_raw_loaded_srs_g1(
            const G1Point &point,
            const std::filesystem::path &path,
            std::size_t index)
        {
            // For raw in-memory material there was no canonical mcl
            // deserialization step. isValid() checks both curve membership
            // and, because BLS12-381 verifyOrderG1 is enabled, subgroup
            // membership. Do not repeat the same subgroup predicate.
            if (point.value.isZero() || !point.value.isValid())
            {
                throw std::runtime_error(
                    "invalid, non-subgroup, or identity SRS G1 point at index " + std::to_string(index) + ": " + path.string());
            }
        }

        void validate_deserialized_srs_g1_identity(
            const G1Point &point,
            const std::filesystem::path &path,
            std::size_t index)
        {
            // mcl BLS12-381 canonical deserialization already performs the
            // curve reconstruction and the configured subgroup/order check.
            // The SRS policy additionally rejects the identity.
            if (point.value.isZero())
            {
                throw std::runtime_error(
                    "identity SRS G1 point at index " + std::to_string(index) + ": " + path.string());
            }
        }

        enum class G1LoadWorkerFailure : unsigned
        {
            None = 0,
            RawValidation,
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
            case G1LoadWorkerFailure::RawValidation:
                return "invalid, non-subgroup, or identity "
                       "SRS G1 point at index " +
                       std::to_string(failure.index) + ": " + path.string();

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

#if defined(__linux__)
        class ScopedReadOnlySrsMapping
        {
        public:
            ScopedReadOnlySrsMapping(
                const std::filesystem::path &path,
                std::size_t expected_size)
                : size_(expected_size)
            {
                fd_ = ::open(
                    path.c_str(),
                    O_RDONLY | O_CLOEXEC);

                if (fd_ < 0)
                {
                    throw std::runtime_error(
                        "failed to open SRS material for mmap: " +
                        path.string() + ": " +
                        std::strerror(errno));
                }

                struct stat st{};

                if (::fstat(fd_, &st) != 0)
                {
                    const auto error =
                        std::string(std::strerror(errno));

                    ::close(fd_);
                    fd_ = -1;

                    throw std::runtime_error(
                        "failed to stat SRS material for mmap: " +
                        path.string() + ": " + error);
                }

                if (st.st_size < 0 ||
                    static_cast<std::uint64_t>(st.st_size) !=
                        static_cast<std::uint64_t>(expected_size))
                {
                    ::close(fd_);
                    fd_ = -1;

                    throw std::runtime_error(
                        "SRS material changed while opening mmap: " +
                        path.string());
                }

                mapping_ =
                    ::mmap(
                        nullptr,
                        size_,
                        PROT_READ,
                        MAP_PRIVATE,
                        fd_,
                        0);

                if (mapping_ == MAP_FAILED)
                {
                    mapping_ = nullptr;

                    const auto error =
                        std::string(std::strerror(errno));

                    ::close(fd_);
                    fd_ = -1;

                    throw std::runtime_error(
                        "failed to mmap SRS material: " +
                        path.string() + ": " + error);
                }
            }

            ~ScopedReadOnlySrsMapping()
            {
                if (mapping_ != nullptr)
                {
                    ::munmap(mapping_, size_);
                }

                if (fd_ >= 0)
                {
                    ::close(fd_);
                }
            }

            ScopedReadOnlySrsMapping(
                const ScopedReadOnlySrsMapping &) = delete;

            ScopedReadOnlySrsMapping &
            operator=(
                const ScopedReadOnlySrsMapping &) = delete;

            const std::uint8_t *data() const
            {
                return static_cast<const std::uint8_t *>(
                    mapping_);
            }

            std::size_t size() const
            {
                return size_;
            }

        private:
            int fd_ = -1;
            void *mapping_ = nullptr;
            std::size_t size_ = 0;
        };

        std::uint64_t load_u64_le_unaligned(
            const std::uint8_t *bytes)
        {
            std::uint64_t value = 0;

            for (std::size_t i = 0; i < 8; ++i)
            {
                value |=
                    static_cast<std::uint64_t>(
                        bytes[i])
                    << (8U * i);
            }

            return value;
        }

        std::vector<G1Point>
        load_mapped_canonical_g1_vector_file_prefix(
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
                    1ULL << 16;

            constexpr std::size_t
                kMappedHashChunkBytes =
                    8ULL << 20;

            ScopedReadOnlySrsMapping mapping(
                path,
                file_size);

            const auto *file_bytes =
                mapping.data();

            const std::array<std::uint8_t, 8>
                expected_magic{{'G', 'A', 'T', 'Z',
                                'K', 'G', '1', 'C'}};

            if (std::memcmp(
                    file_bytes,
                    expected_magic.data(),
                    expected_magic.size()) != 0)
            {
                throw std::runtime_error(
                    "canonical SRS G1 mmap header mismatch: " +
                    path.string());
            }

            const auto mapped_point_count_u64 =
                load_u64_le_unaligned(
                    file_bytes + 8);

            const auto mapped_point_size_u64 =
                load_u64_le_unaligned(
                    file_bytes + 16);

            if (mapped_point_count_u64 !=
                    static_cast<std::uint64_t>(
                        expected_point_count) ||
                mapped_point_size_u64 !=
                    static_cast<std::uint64_t>(
                        point_size))
            {
                throw std::runtime_error(
                    "canonical SRS G1 changed while mapping: " +
                    path.string());
            }

            if (expected_point_count >
                (std::numeric_limits<std::size_t>::max() -
                 kHeaderSize) /
                    point_size)
            {
                throw std::runtime_error(
                    "canonical SRS G1 mmap size overflow: " +
                    path.string());
            }

            const auto expected_size =
                kHeaderSize +
                expected_point_count * point_size;

            if (expected_size != file_size)
            {
                throw std::runtime_error(
                    "canonical SRS G1 mmap size mismatch: " +
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

            std::atomic<std::size_t>
                next_begin{0};

            auto validation_worker =
                [&]()
            {
                std::size_t active_begin =
                    std::numeric_limits<
                        std::size_t>::max();

                try
                {
                    std::vector<std::uint8_t>
                        canonical_bytes(
                            point_size);

                    while (true)
                    {
                        if (worker_failed())
                        {
                            return;
                        }

                        const auto begin =
                            next_begin.fetch_add(
                                kCanonicalWorkerChunkPoints,
                                std::memory_order_relaxed);

                        active_begin = begin;

                        if (begin >= load_count)
                        {
                            return;
                        }

                        const auto end =
                            std::min<std::size_t>(
                                load_count,
                                begin +
                                    kCanonicalWorkerChunkPoints);

                        for (std::size_t index = begin;
                             index < end;
                             ++index)
                        {
                            if (worker_failed())
                            {
                                return;
                            }

                            const auto *point_begin =
                                file_bytes +
                                kHeaderSize +
                                index * point_size;

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
                }
                catch (...)
                {
                    record_worker_failure(
                        G1LoadWorkerFailure::
                            UnexpectedCanonical,
                        active_begin);
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

            // Keep one execution context available for the mandatory
            // full-file SHA-256 scan.  Once hashing completes the calling
            // thread joins validation work, so all CPUs are used for the
            // remaining canonical checks.
            const auto background_worker_count =
                cpu_count > 1
                    ? std::min<std::size_t>(
                          cpu_count - 1,
                          chunk_count)
                    : std::size_t{0};

            std::vector<std::thread> workers;
            workers.reserve(
                background_worker_count);

            for (std::size_t worker = 0;
                 worker < background_worker_count;
                 ++worker)
            {
                workers.emplace_back(
                    validation_worker);
            }

            // Hash exactly the authenticated file bytes.  Validation workers
            // only read the same MAP_PRIVATE/PROT_READ mapping.
            SHA256_CTX mapped_hash;
            SHA256_Init(&mapped_hash);

            std::size_t hash_offset = 0;

            while (hash_offset < mapping.size())
            {
                const auto chunk =
                    std::min<std::size_t>(
                        mapping.size() -
                            hash_offset,
                        kMappedHashChunkBytes);

                SHA256_Update(
                    &mapped_hash,
                    file_bytes + hash_offset,
                    chunk);

                hash_offset += chunk;
            }

            unsigned char
                digest[SHA256_DIGEST_LENGTH];

            SHA256_Final(
                digest,
                &mapped_hash);

            if (actual_hash != nullptr)
            {
                *actual_hash =
                    bytes_to_hex(
                        digest,
                        sizeof(digest));
            }

            // The hash thread now becomes another validation worker instead
            // of leaving one CPU idle for the remainder of a full SRS load.
            validation_worker();

            for (auto &worker : workers)
            {
                worker.join();
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
#endif
        std::vector<G1Point> load_fixed_or_raw_g1_vector_file_prefix(
            const std::filesystem::path &path,
            std::streamoff file_size,
            std::size_t max_points,
            std::string *actual_hash,
            G1LoadWorkerFailureInfo &worker_failure)
        {
            worker_failure = {};
            algebra::ensure_mcl_field_ready();
            std::ifstream input(path, std::ios::binary);
            if (!input)
            {
                throw std::runtime_error("failed to open SRS G1 material: " + path.string());
            }
            SHA256_CTX ctx;
            SHA256_Init(&ctx);
            auto read_hashed = [&](char *out, std::size_t size, const std::string &label)
            {
                input.read(out, static_cast<std::streamsize>(size));
                if (input.gcount() != static_cast<std::streamsize>(size))
                {
                    throw std::runtime_error("truncated SRS G1 material while reading " + label + ": " + path.string());
                }
                SHA256_Update(&ctx, out, size);
            };

            std::array<char, 8> magic{};
            read_hashed(magic.data(), magic.size(), "G1 magic");
            const std::array<char, 8> expected_raw_magic{{'G', 'A', 'T', 'Z', 'K', 'G', '1', 'R'}};
            const std::array<char, 8> expected_canonical_magic{{'G', 'A', 'T', 'Z', 'K', 'G', '1', 'C'}};
            if (magic != expected_raw_magic && magic != expected_canonical_magic)
            {
                return load_g1_vector_file(path, actual_hash, 0);
            }
            std::array<std::uint8_t, 8> count_bytes{};
            read_hashed(reinterpret_cast<char *>(count_bytes.data()), count_bytes.size(), "G1 point count");
            const auto count = parse_u64_le_bytes(count_bytes);
            if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
            {
                throw std::runtime_error("SRS G1 material is too large for this platform");
            }
            std::array<std::uint8_t, 8> point_size_bytes{};
            read_hashed(
                reinterpret_cast<char *>(point_size_bytes.data()),
                point_size_bytes.size(),
                "G1 point size");
            const auto point_size_u64 =
                parse_u64_le_bytes(point_size_bytes);
            if (point_size_u64 == 0 || point_size_u64 > static_cast<std::uint64_t>(
                                                            std::numeric_limits<std::size_t>::max()))
            {
                throw std::runtime_error(
                    "invalid fixed SRS G1 point size");
            }
            const auto point_size =
                static_cast<std::size_t>(point_size_u64);
            if (magic == expected_raw_magic && point_size != sizeof(mcl::G1))
                throw std::runtime_error(
                    "raw SRS G1 material was generated for an incompatible mcl layout");

            const auto point_count = static_cast<std::size_t>(count);
            const auto header_size = static_cast<std::size_t>(24);
            if (point_count > (std::numeric_limits<std::size_t>::max() - header_size) / point_size)
                throw std::runtime_error(
                    "fixed SRS G1 material size overflows this platform");
            const auto expected_size =
                header_size + point_count * point_size;

            if (expected_size != static_cast<std::size_t>(file_size))
            {
                throw std::runtime_error(
                    "fixed SRS G1 material size mismatch: " +
                    path.string());
            }

#if defined(__linux__)
            if (magic == expected_canonical_magic)
            {
                return load_mapped_canonical_g1_vector_file_prefix(
                    path,
                    static_cast<std::size_t>(file_size),
                    point_count,
                    point_size,
                    max_points,
                    actual_hash,
                    worker_failure);
            }
#endif

            const auto load_count =
                max_points == 0
                    ? point_count
                    : std::min(point_count, max_points);

            std::vector<G1Point> out(load_count);

            const auto cpu_count =
                std::max<std::size_t>(
                    1,
                    std::thread::hardware_concurrency());

            const bool has_unloaded_suffix =
                load_count < point_count;

            const auto validation_worker_budget =
                has_unloaded_suffix && cpu_count > 1
                    ? cpu_count - 1
                    : cpu_count;

            // The full-file SHA-256 remains mandatory.  When only a prefix is
            // materialized, overlap validation/deserialization of that prefix with
            // sequential hashing of the authenticated suffix.
            bool suffix_hashed = false;

            auto hash_unloaded_suffix = [&]()
            {
                if (suffix_hashed)
                {
                    return;
                }

                suffix_hashed = true;

                std::size_t remaining_bytes =
                    (point_count - load_count) *
                    point_size;

                if (remaining_bytes == 0)
                {
                    return;
                }

                constexpr std::size_t
                    kSrsHashBufferSize =
                        8ULL << 20;

                auto buffer =
                    std::make_unique<char[]>(
                        kSrsHashBufferSize);

                while (remaining_bytes != 0)
                {
                    const auto chunk =
                        std::min<std::size_t>(
                            remaining_bytes,
                            kSrsHashBufferSize);

                    read_hashed(
                        buffer.get(),
                        chunk,
                        "raw G1 suffix");

                    remaining_bytes -= chunk;
                }
            };

            if (load_count != 0)
            {
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
                                static_cast<unsigned>(
                                    kind),
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
                               std::memory_order_acquire) != static_cast<unsigned>(G1LoadWorkerFailure::
                                                                                       None);
                };

                const auto load_bytes =
                    load_count * point_size;

                if (magic == expected_raw_magic)
                {
                    read_hashed(
                        reinterpret_cast<char *>(
                            &out.front().value),
                        load_bytes,
                        "raw G1 prefix");

                    const auto worker_count =
                        std::min<std::size_t>(
                            validation_worker_budget,
                            load_count);

                    const auto chunk_size =
                        (load_count + worker_count - 1) / worker_count;

                    std::vector<
                        std::future<void>>
                        futures;

                    futures.reserve(worker_count);

                    for (std::size_t worker = 0;
                         worker < worker_count;
                         ++worker)
                    {
                        const auto begin =
                            worker * chunk_size;

                        const auto end =
                            std::min(
                                load_count,
                                begin + chunk_size);

                        if (begin >= end)
                        {
                            break;
                        }

                        futures.push_back(
                            std::async(
                                std::launch::async,
                                [&, begin, end]()
                                {
                                    try
                                    {
                                        for (std::size_t i =
                                                 begin;
                                             i < end;
                                             ++i)
                                        {
                                            if (worker_failed())
                                            {
                                                return;
                                            }

                                            validate_raw_loaded_srs_g1(
                                                out[i],
                                                path,
                                                i);
                                        }
                                    }
                                    catch (...)
                                    {
                                        record_worker_failure(
                                            G1LoadWorkerFailure::
                                                RawValidation,
                                            begin);
                                    }
                                }));
                    }

                    // Hash the authenticated suffix while prefix validation is still
                    // executing.  Only this calling thread touches the input stream and
                    // SHA256_CTX.
                    hash_unloaded_suffix();

                    // Workers never propagate validation
                    // exceptions through std::future.
                    // Their failure state is converted back
                    // into the canonical loader exception on
                    // this calling thread below.
                    for (auto &future : futures)
                    {
                        future.wait();
                    }
                }
                else
                {
                    constexpr std::size_t
                        kCanonicalSrsChunkPoints =
                            1ULL << 16;

                    const auto chunk_count =
                        (load_count +
                         kCanonicalSrsChunkPoints - 1) /
                        kCanonicalSrsChunkPoints;

                    const auto max_inflight =
                        std::max<std::size_t>(
                            1,
                            std::min<std::size_t>(
                                validation_worker_budget,
                                chunk_count));

                    std::deque<
                        std::future<void>>
                        inflight;

                    auto reap_one =
                        [&]()
                    {
                        inflight.front().wait();
                        inflight.pop_front();
                    };

                    for (std::size_t begin = 0;
                         begin < load_count;
                         begin +=
                         kCanonicalSrsChunkPoints)
                    {
                        // Keep exactly max_inflight validation tasks active without
                        // allocating an additional payload while the queue is full.
                        if (inflight.size() >= max_inflight)
                        {
                            reap_one();
                        }

                        const auto current_count =
                            std::min<std::size_t>(
                                kCanonicalSrsChunkPoints,
                                load_count - begin);

                        const auto current_bytes =
                            current_count * point_size;

                        auto payload =
                            std::make_shared<
                                std::vector<
                                    std::uint8_t>>(
                                current_bytes);

                        read_hashed(
                            reinterpret_cast<char *>(
                                payload->data()),
                            payload->size(),
                            "canonical G1 prefix chunk");

                        inflight.emplace_back(
                            std::async(
                                std::launch::async,
                                [&, payload,
                                 begin,
                                 current_count]()
                                {
                                    try
                                    {
                                        std::vector<
                                            std::uint8_t>
                                            canonical_bytes(
                                                point_size);

                                        for (std::size_t local =
                                                 0;
                                             local < current_count;
                                             ++local)
                                        {
                                            if (worker_failed())
                                            {
                                                return;
                                            }

                                            const auto index =
                                                begin + local;

                                            const auto *
                                                point_begin =
                                                    payload->data() +
                                                    local * point_size;

                                            const auto
                                                consumed =
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

                                            const auto
                                                written =
                                                    out[index]
                                                        .value
                                                        .serialize(
                                                            canonical_bytes
                                                                .data(),
                                                            canonical_bytes
                                                                .size());

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
                                }));
                    }

                    // The prefix bytes have now all entered SHA-256 in canonical file
                    // order.  Scan the remaining authenticated bytes while the last
                    // decode/validation tasks are still running.
                    hash_unloaded_suffix();

                    while (!inflight.empty())
                    {
                        reap_one();
                    }
                }

                const auto recorded_failure =
                    static_cast<G1LoadWorkerFailure>(
                        worker_failure_kind.load(
                            std::memory_order_acquire));

                if (recorded_failure != G1LoadWorkerFailure::None)
                {
                    worker_failure.kind =
                        recorded_failure;

                    worker_failure.index =
                        worker_failure_index.load(
                            std::memory_order_acquire);
                }
            }

            // Covers load_count == 0 and is a no-op when either execution branch
            // already scanned the suffix.
            hash_unloaded_suffix();

            unsigned char digest[SHA256_DIGEST_LENGTH];
            SHA256_Final(digest, &ctx);

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
            std::string *actual_hash = nullptr,
            std::size_t max_points = 0)
        {
            std::ifstream input(path, std::ios::binary | std::ios::ate);
            if (!input)
            {
                throw std::runtime_error("failed to open SRS G1 material: " + path.string());
            }
            const auto file_size = input.tellg();
            if (file_size < static_cast<std::streamoff>(16))
            {
                throw std::runtime_error("truncated SRS G1 material: " + path.string());
            }
            std::array<char, 8> magic_probe{};
            input.seekg(0);
            input.read(magic_probe.data(), static_cast<std::streamsize>(magic_probe.size()));
            if (input.gcount() != static_cast<std::streamsize>(magic_probe.size()))
            {
                throw std::runtime_error("truncated SRS G1 material header: " + path.string());
            }
            const std::array<char, 8> expected_raw_magic{{'G', 'A', 'T', 'Z', 'K', 'G', '1', 'R'}};
            const std::array<char, 8> expected_canonical_magic{{'G', 'A', 'T', 'Z', 'K', 'G', '1', 'C'}};
            if (max_points != 0 || magic_probe == expected_raw_magic || magic_probe == expected_canonical_magic)
            {
                G1LoadWorkerFailureInfo
                    worker_failure;

                auto points =
                    load_fixed_or_raw_g1_vector_file_prefix(
                        path,
                        file_size,
                        max_points,
                        actual_hash,
                        worker_failure);

                if (worker_failure.kind != G1LoadWorkerFailure::None)
                {
                    throw std::runtime_error(
                        g1_load_worker_failure_message(
                            worker_failure,
                            path));
                }

                return points;
            }
            input.seekg(0);
            std::vector<std::uint8_t> file_bytes(static_cast<std::size_t>(file_size));
            input.read(reinterpret_cast<char *>(file_bytes.data()), file_size);
            if (input.gcount() != file_size)
            {
                throw std::runtime_error("failed to read SRS G1 material: " + path.string());
            }
            if (actual_hash != nullptr)
            {
                *actual_hash = sha256_bytes_hex(file_bytes);
            }
            const std::array<char, 8> expected_magic{{'G', 'A', 'T', 'Z', 'K', 'G', '1', '\0'}};
            std::array<char, 8> magic{};
            std::copy_n(reinterpret_cast<const char *>(file_bytes.data()), magic.size(), magic.data());
            if (magic != expected_magic && magic != expected_raw_magic)
            {
                throw std::runtime_error("invalid SRS G1 material header: " + path.string());
            }
            std::size_t offset = 8;
            const auto count = read_u64_le_bytes(file_bytes, offset, "G1 point count");
            if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
            {
                throw std::runtime_error("SRS G1 material is too large for this platform");
            }
            if (magic == expected_raw_magic)
            {
                const auto raw_size = read_u64_le_bytes(file_bytes, offset, "raw G1 point size");
                if (raw_size != sizeof(mcl::G1))
                {
                    throw std::runtime_error("raw SRS G1 material was generated for an incompatible mcl layout");
                }
                const auto point_count = static_cast<std::size_t>(count);
                const auto expected_size = offset + point_count * sizeof(mcl::G1);
                if (expected_size != file_bytes.size())
                {
                    throw std::runtime_error("raw SRS G1 material size mismatch: " + path.string());
                }
                std::vector<G1Point> out(point_count);
                if (point_count != 0)
                {
                    std::memcpy(
                        &out.front().value,
                        file_bytes.data() + offset,
                        point_count * sizeof(mcl::G1));
                }
                for (std::size_t i = 0;
                     i < out.size();
                     ++i)
                {
                    validate_raw_loaded_srs_g1(
                        out[i],
                        path,
                        i);
                }
                return out;
            }
            std::vector<std::pair<std::size_t, std::size_t>> payloads;
            payloads.reserve(static_cast<std::size_t>(count));
            for (std::uint64_t i = 0; i < count; ++i)
            {
                const auto size = read_u64_le_bytes(file_bytes, offset, "G1 point size");
                if (size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) || offset + static_cast<std::size_t>(size) > file_bytes.size())
                {
                    throw std::runtime_error("truncated SRS G1 point payload");
                }
                payloads.emplace_back(offset, static_cast<std::size_t>(size));
                offset += static_cast<std::size_t>(size);
            }

            std::vector<G1Point> out;
            out.resize(static_cast<std::size_t>(count));
            const auto worker_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
            const auto chunk_size = (out.size() + worker_count - 1) / worker_count;
            std::vector<std::future<void>> futures;
            futures.reserve(worker_count);
            for (std::size_t worker = 0; worker < worker_count; ++worker)
            {
                const auto begin = worker * chunk_size;
                const auto end = std::min(out.size(), begin + chunk_size);
                if (begin >= end)
                {
                    break;
                }
                futures.push_back(std::async(std::launch::async, [&file_bytes, &payloads, &out, &path, begin, end]()
                                             {
                    for (std::size_t i = begin; i < end; ++i)
                    {
                        const auto [payload_offset, payload_size] = payloads[i];
                        std::vector<std::uint8_t> point_bytes(
                            file_bytes.begin() + static_cast<std::ptrdiff_t>(payload_offset),
                            file_bytes.begin() + static_cast<std::ptrdiff_t>(payload_offset + payload_size));
                        out[i] = deserialize_g1(point_bytes);
                        if (serialize(out[i]) != point_bytes)
                            throw std::runtime_error(
                                "non-canonical SRS G1 encoding at index "
                                + std::to_string(i));
                        validate_deserialized_srs_g1_identity(out[i], path, i);
                    } }));
            }
            for (auto &future : futures)
            {
                future.get();
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
            if (header != "GATZK-VK-v2")
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
            if (manifest.manifest_format != "zkgat_srs_manifest_v8" || manifest.generator_id != "zkgat_srs_setup_tool_v8" || manifest.generator_version != "zkgat_srs_setup_tool_v8")
            {
                throw std::runtime_error("unsupported or legacy SRS manifest format");
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
            if (manifest.kzg_assumption_profile_id != "shifted_kzg_divisibility_v1" || manifest.kzg_assumption_degree_bound != manifest.degree_bound || manifest.kzg_assumption_security_bits != 128 || manifest.kzg_assumption_domain_catalog_hash != canonical_domain_catalog_hash(manifest.domain_sizes))
            {
                throw std::runtime_error("SRS shifted-KZG assumption profile is missing or inconsistent");
            }
            if (manifest.root_derivation_id != "bls12_381_fr_generator_5_v1" || manifest.row_order != "ascending_powers_from_one" || manifest.domain_generators.size() != manifest.domain_sizes.size() || manifest.domain_generator_catalog_hash != canonical_domain_generator_catalog_hash(manifest.domain_sizes, manifest.domain_generators, manifest.root_derivation_id, manifest.row_order))
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
            if (manifest.encoding != "canonical_mcl_bls12_381_fixed_v1" || manifest.srs_audit_report_hash.empty() || manifest.srs_consistency_hash.empty() || manifest.srs_audit_report_hash != manifest.srs_consistency_hash || manifest.vanishing_g2_hash.empty())
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

        std::size_t estimated_commitment_msm_work(const algebra::Polynomial &polynomial)
        {
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
            // 系数多项式采用SRS powers + MSM
            if (polynomial.basis == algebra::PolynomialBasis::Coefficient)
            {
                if (pk.powers_g1 == nullptr)
                {
                    throw std::runtime_error("missing coefficient proving SRS");
                }
                if (polynomial.size() > pk.powers_g1->size() || polynomial.size() > pk.degree_bound + 1)
                {
                    throw std::runtime_error("coefficient polynomial exceeds SRS degree bound: " + polynomial.name);
                }
                stable_msm_options.stable_srs_capacity =
                    pk.powers_g1->size();
                point = g1_multi_scalar_mul_profiled(
                    pk.powers_g1->data(),
                    polynomial.values().data(),
                    polynomial.size(),
                    profile != nullptr ? &msm_profile : nullptr,
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
                profile->vector_length = polynomial.size();
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

        Commitment make_commitment(
            const std::string &name,
            algebra::Polynomial polynomial,
            const KZGProverKey &pk,
            CommitObjectProfile *profile = nullptr,
            const G1MSMExecutionOptions *msm_options = nullptr,
            EvaluationCommitmentCache *reuse_cache = nullptr)
        {
            polynomial.name = name;
            auto material = std::make_shared<algebra::Polynomial>(std::move(polynomial));
            return Commitment{
                .name = name,
                .point = commit_polynomial_point(
                    *material, pk, profile, msm_options, reuse_cache),
                .opening_polynomial = material,
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
        };

        struct FoldedOpeningPlan
        {
            std::vector<FieldElement> coefficient_fold;
            std::vector<FieldElement> coefficient_scaled_claim_sums;
            std::vector<FoldedEvaluationGroup> evaluation_groups;
        };

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
            std::size_t target_offset = 0)
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

            if (group == nullptr ||
                source_values.size() >
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

            auto *folded_native =
                FieldElement::mutable_native_array_data(
                    group->folded_values.data()) +
                target_offset;

            const auto *source_native =
                FieldElement::native_array_data(
                    source_values.data());

            const auto scale_native =
                scale.native();

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
        }

        void accumulate_streamed_evaluation_source_pair(
            FoldedEvaluationGroup *first_group,
            FoldedEvaluationGroup *second_group,
            const std::vector<FieldElement> &source_values,
            const FieldElement &first_scale,
            const FieldElement &second_scale,
            const std::string &label,
            std::size_t worker_cap,
            std::size_t target_offset = 0)
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
                    target_offset);

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
                    target_offset);

                return;
            }

            auto *first_native =
                FieldElement::
                    mutable_native_array_data(
                        first_group->folded_values
                            .data()) +
                target_offset;

            auto *second_native =
                FieldElement::
                    mutable_native_array_data(
                        second_group->folded_values
                            .data()) +
                target_offset;

            const auto *source_native =
                FieldElement::
                    native_array_data(
                        source_values.data());

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

            std::ifstream stream(
                info.file_path,
                std::ios::binary);

            if (!stream)
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
                    info.value_count));

            for (std::size_t offset = 0;
                 offset < info.value_count;)
            {
                const auto count =
                    std::min(
                        kChunkSize,
                        info.value_count - offset);

                chunk.resize(count);

                try
                {
                    algebra::read_field_elements_binary_into(
                        stream,
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
            std::size_t worker_cap)
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
                        offset);
                });
        }

        void accumulate_file_backed_semantic_source_pair(
            FoldedEvaluationGroup *first_group,
            FoldedEvaluationGroup *second_group,
            const OpeningSemanticSourceInfo &info,
            const FieldElement &first_scale,
            const FieldElement &second_scale,
            std::size_t worker_cap)
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
                        offset);
                });
        }

        FieldElement evaluate_with_shared_domain_weights(
            const algebra::Polynomial &polynomial,
            const FieldElement &point,
            const algebra::DomainEvaluationWeights &weights)
        {
            if (polynomial.basis != algebra::PolynomialBasis::Evaluation || polynomial.domain == nullptr)
            {
                throw std::runtime_error(
                    "shared domain-weight evaluation requires an evaluation polynomial");
            }
            const auto &values = polynomial.values();
            FieldElement value = FieldElement::zero();
            if (weights.direct_index.has_value())
            {
                if (*weights.direct_index < values.size())
                {
                    value = values[*weights.direct_index];
                }
            }
            else
            {
                if (values.size() > weights.native_weights.size())
                {
                    throw std::runtime_error(
                        "evaluation polynomial exceeds its domain weight vector");
                }
                mcl::Fr sum;
                sum.clear();
                for (std::size_t i = 0; i < values.size(); ++i)
                {
                    mcl::Fr product;
                    mcl::Fr::mul(product, values[i].native(), weights.native_weights[i]);
                    mcl::Fr::add(sum, sum, product);
                }
                value = FieldElement::from_native(sum);
            }
            if (polynomial.is_zero_domain_blinded())
            {
                value += polynomial.domain->zero_polynomial_eval(point) * algebra::horner(polynomial.zero_domain_blinder, point);
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
                                    const auto &weights = claim_weight_cache.get(
                                        polynomial.domain,
                                        points[point_index]);
                                    expected = evaluate_with_shared_domain_weights(
                                        polynomial,
                                        points[point_index],
                                        weights);
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
                        accumulate_scaled_coefficients(&plan.coefficient_fold, polynomial.values(), term.scale, worker_cap);
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
                *defer_semantic_source)
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
            // VerifyAgainstOpeningPolynomials deliberately retains the former
            // per-polynomial path so focused claim-validation tests remain unchanged.
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

                        if (existing_info.domain == nullptr || existing_info.domain->name != semantic_info->domain->name || existing_info.domain->size != semantic_info->domain->size || !(existing_info.domain->omega == semantic_info->domain->omega) || existing_info.value_count != semantic_info->value_count)
                        {
                            throw std::runtime_error(
                                "conflicting streamed opening semantic source metadata: " + commitments[i].name);
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

                // A singleton source has nothing to deduplicate.  Keep it on the
                // original streamed path so this optimization changes execution only
                // where an actual shared semantic source exists.
                std::size_t shared_source_count = 0;
                std::size_t shared_member_count = 0;

                for (std::size_t fold_index = 0;
                     fold_index < semantic_folds.size();
                     ++fold_index)
                {
                    auto &fold =
                        semantic_folds[fold_index];

                    if (fold.members.size() < 2)
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
                                    worker_cap);
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

                            expected =
                                evaluate_with_shared_domain_weights(
                                    *material,
                                    points[point_index],
                                    claim_weight_cache.get(
                                        material->domain,
                                        points[point_index]));
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
                        accumulate_scaled_coefficients(
                            &plan.coefficient_fold,
                            material->values(),
                            scale,
                            worker_cap);
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
                    worker_cap);

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

            trim_zero_coefficients(
                &plan.coefficient_fold);

            for (auto &group :
                 plan.evaluation_groups)
            {
                trim_zero_coefficients(
                    &group
                         .folded_zero_domain_blinder);
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
                &semantic_source_resolver)
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
                        &defer);
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
                        worker_caps[plan_index]);

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

                const bool same_source =
                    first.representative.name ==
                        second.representative.name &&
                    first.representative.point ==
                        second.representative.point &&
                    first.info.domain != nullptr &&
                    second.info.domain != nullptr &&
                    first.info.domain->name ==
                        second.info.domain->name &&
                    first.info.domain->size ==
                        second.info.domain->size &&
                    first.info.domain->omega ==
                        second.info.domain->omega &&
                    first.info.value_count ==
                        second.info.value_count &&
                    first.info.zero_domain_blinder ==
                        second.info.zero_domain_blinder;

                if (!same_source)
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
                        shared_worker_cap);
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

                ++result
                      .fused_semantic_source_count;
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

            std::vector<mcl::Fr> prefixes(
                domain->size);

            std::vector<mcl::Fr> chunk_products(
                task_count);

            const auto point_native =
                point.native();

            const auto omega_native =
                domain->omega.native();

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

            // One exponentiation per chunk, then walk the domain by
            // multiplication. Build each chunk's prefix products in
            // the same pass.
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
                        mcl::Fr::sub(
                            denominator_native[row],
                            domain_point,
                            point_native);

                        if (denominator_native[row].isZero())
                        {
                            throw std::runtime_error(
                                "opening point overlaps evaluation domain; derivative witness path is not implemented");
                        }

                        prefixes[row] =
                            local_product;

                        mcl::Fr::mul(
                            local_product,
                            local_product,
                            denominator_native[row]);

                        mcl::Fr::mul(
                            domain_point,
                            domain_point,
                            omega_native);
                    }

                    chunk_products[task] =
                        local_product;
                });

            // Reduce the per-chunk products and perform exactly one
            // field inversion for the entire batch.
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

                chunk_products[task] =
                    chunk_inverse;
            }

            // Restore every denominator inverse independently inside
            // each chunk.
            run_chunks(
                [&](std::size_t task,
                    std::size_t begin,
                    std::size_t end)
                {
                    auto local_suffix =
                        chunk_products[task];

                    for (std::size_t row = end;
                         row-- > begin;)
                    {
                        const mcl::Fr original =
                            denominator_native[row];

                        mcl::Fr inverse;

                        mcl::Fr::mul(
                            inverse,
                            local_suffix,
                            prefixes[row]);

                        denominator_native[row] =
                            inverse;

                        mcl::Fr::mul(
                            local_suffix,
                            local_suffix,
                            original);
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
            for (const auto &group : plan.evaluation_groups)
            {
                const auto inverse_start =
                    Clock::now();

                std::vector<FieldElement>
                    inverse_denominators(
                        group.domain->size,
                        FieldElement::zero());

                build_opening_inverse_denominators(
                    &inverse_denominators,
                    group.domain,
                    point,
                    worker_cap);

                if (profile != nullptr)
                {
                    profile->denominator_inverse_ms +=
                        elapsed_ms(
                            inverse_start,
                            Clock::now());

                    profile
                        ->denominator_inverse_cache_misses +=
                        1;
                }
                const auto quotient_assembly_start = Clock::now();
                std::vector<FieldElement> quotient_values(group.domain->size, FieldElement::zero());
                auto *quotient_values_native = FieldElement::mutable_native_array_data(quotient_values.data());
                mcl::Fr zero;
                zero.clear();
                mcl::Fr base_numerator;
                mcl::Fr::sub(base_numerator, zero, group.scaled_claim_sums[point_index].native());
                const auto *folded_values_native =
                    FieldElement::native_array_data(group.folded_values.data());
                parallel_for_chunks_capped(group.domain->size, 1ULL << 15, worker_cap, [&](std::size_t begin, std::size_t end)
                                           {
                    for (std::size_t row = begin; row < end; ++row)
                    {
                        mcl::Fr native_numerator = base_numerator;
                        mcl::Fr::add(
                            native_numerator,
                            native_numerator,
                            folded_values_native[row]);
                        mcl::Fr::mul(
                            quotient_values_native[row],
                            native_numerator,
                            inverse_denominators[row].native());
                    } });
                if (profile != nullptr)
                {
                    profile->quotient_assembly_ms += elapsed_ms(
                        quotient_assembly_start, Clock::now());
                }
                const auto witness_msm_start = Clock::now();
                CommitObjectProfile witness_commit_profile;
                out = g1_add(
                    out,
                    commit_polynomial_point(
                        algebra::Polynomial::from_evaluations(
                            "folded_" + group.domain->name + "_witness",
                            std::move(quotient_values),
                            group.domain),
                        pk,
                        profile != nullptr
                            ? &witness_commit_profile
                            : nullptr,
                        msm_options));
                if (profile != nullptr && progress_enabled())
                {
                    progress_note(
                        "kzg:opening_witness_msm:domain=" + group.domain->name + ":input=" + std::to_string(witness_commit_profile.vector_length) + ":nonzero=" + std::to_string(witness_commit_profile.nonzero_scalar_count) + ":effective=" + std::to_string(witness_commit_profile.effective_scalar_count) + ":sparse=" + std::string(witness_commit_profile.sparse_compaction ? "true" : "false") + ":tail=" + std::string(witness_commit_profile.constant_tail_folding ? "true" : "false") + ":ms=" + std::to_string(witness_commit_profile.msm_ms));
                }
                if (profile != nullptr)
                {
                    profile->witness_msm_ms += elapsed_ms(
                        witness_msm_start, Clock::now());
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
            throw std::runtime_error(
                "SRS manifest lacks Lagrange material for domain " + domain_name + ":" + std::to_string(domain_size));
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
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            if (const auto it = cache.find(cache_key); it != cache.end())
            {
                return it->second;
            }
        }
        const auto srs_load_start = Clock::now();

        const auto manifest = load_manifest_file(srs_path);
        validate_manifest(manifest, expected_srs_type, expected_srs_id);
        const auto srs_root = std::filesystem::path(srs_path);
        validate_protocol_compiler_artifact_material(srs_root, manifest);
        validate_srs_audit_report_material(srs_root, manifest);
        const auto selected_lagrange_domains =
            select_lagrange_srs_domains(manifest, required_lagrange_domains);
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
        public_transcript << "gatzk:trusted-srs-public-transcript:v8|"
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
        };
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

    Commitment KZG::commit_taylor_shifted(
        const Commitment &source,
        const std::string &shifted_name,
        std::size_t configured_degree,
        const FieldElement &delta,
        const KZGKeyPair &key,
        CommitObjectProfile *profile)
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
        return make_commitment(
            shifted_name,
            algebra::Polynomial::from_coefficients(
                shifted_name, std::move(shifted)),
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
        const std::vector<std::pair<std::string, const algebra::Polynomial *>> &named_polynomials,
        const KZGKeyPair &key,
        CommitBatchProfile *profile,
        EvaluationCommitmentCache *reuse_cache)
    {
        std::vector<Commitment> out(named_polynomials.size());
        std::vector<CommitObjectProfile> object_profiles(named_polynomials.size());
        const auto msm_start = Clock::now();
        const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        std::size_t total_scalar_count = 0;
        std::size_t max_scalar_count = 0;
        std::size_t uncached_msm_count = 0;
        for (const auto &[name, polynomial] : named_polynomials)
        {
            if (polynomial == nullptr)
            {
                throw std::runtime_error("null polynomial in KZG::commit_batch: " + name);
            }
            const auto estimated_work = estimated_commitment_msm_work(*polynomial);
            total_scalar_count += estimated_work;
            max_scalar_count = std::max(max_scalar_count, estimated_work);
            bool cache_hit = false;
            if (reuse_cache != nullptr && polynomial->basis == algebra::PolynomialBasis::Evaluation && polynomial->shared_data != nullptr)
            {
                const auto *lagrange_srs =
                    evaluation_srs_points(*polynomial, key.pk);
                G1Point cached_point;
                cache_hit = reuse_cache->lookup_semantic(
                    polynomial->domain.get(),
                    lagrange_srs,
                    polynomial->shared_data,
                    &cached_point);
            }
            if (!cache_hit)
            {
                ++uncached_msm_count;
            }
        }
        const auto outer_worker_count = commit_batch_outer_worker_count(
            named_polynomials.size(),
            total_scalar_count,
            max_scalar_count,
            cpu_count);
        const G1MSMExecutionOptions msm_options{
            .max_worker_count = outer_worker_count > 1
                                    ? std::max<std::size_t>(
                                          1,
                                          cpu_count / std::max<std::size_t>(
                                                          1,
                                                          std::min(
                                                              outer_worker_count,
                                                              uncached_msm_count)))
                                    : 0,
        };
        auto commit_one = [&](std::size_t index)
        {
            const auto &[name, polynomial] = named_polynomials[index];
            CommitObjectProfile object_profile;
            out[index] = make_commitment(
                name,
                *polynomial,
                key.pk,
                profile != nullptr ? &object_profile : nullptr,
                outer_worker_count > 1 ? &msm_options : nullptr,
                reuse_cache);
            if (profile != nullptr)
            {
                object_profiles[index] = std::move(object_profile);
            }
        };

        if (outer_worker_count > 1)
        {
            const auto work_order = commitment_work_order(named_polynomials);
            std::atomic<std::size_t> next{0};
            g1_parallel_msm_batch_workers(
                outer_worker_count,
                [&](std::size_t)
                {
                    while (true)
                    {
                        const auto index = next.fetch_add(1);
                        if (index >= named_polynomials.size())
                        {
                            break;
                        }
                        commit_one(work_order[index]);
                    }
                });
        }
        else
        {
            for (std::size_t index = 0; index < named_polynomials.size(); ++index)
            {
                commit_one(index);
            }
        }
        if (profile != nullptr)
        {
            profile->msm_ms += elapsed_ms(msm_start, Clock::now());
            for (auto &object_profile : object_profiles)
            {
                if (!object_profile.name.empty())
                {
                    profile->objects.push_back(std::move(object_profile));
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
        std::vector<std::pair<std::string, algebra::Polynomial>> &&named_polynomials,
        const KZGKeyPair &key,
        CommitBatchProfile *profile,
        const std::string &progress_prefix)
    {
        std::vector<Commitment> out(named_polynomials.size());
        std::vector<CommitObjectProfile> object_profiles(named_polynomials.size());
        const auto msm_start = Clock::now();
        const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        std::size_t total_scalar_count = 0;
        std::size_t max_scalar_count = 0;
        std::vector<std::pair<std::string, const algebra::Polynomial *>> view;
        view.reserve(named_polynomials.size());
        for (const auto &[name, polynomial] : named_polynomials)
        {
            const auto estimated_work = estimated_commitment_msm_work(polynomial);
            total_scalar_count += estimated_work;
            max_scalar_count = std::max(max_scalar_count, estimated_work);
            view.push_back({name, &polynomial});
        }

        const auto outer_worker_count = commit_batch_outer_worker_count(
            named_polynomials.size(),
            total_scalar_count,
            max_scalar_count,
            cpu_count);
        const G1MSMExecutionOptions msm_options{
            .max_worker_count = outer_worker_count > 1
                                    ? std::max<std::size_t>(1, cpu_count / outer_worker_count)
                                    : 0,
        };
        const auto work_order = outer_worker_count > 1 ? commitment_work_order(view) : std::vector<std::size_t>{};
        auto commit_one = [&](std::size_t index)
        {
            auto &[name, polynomial] = named_polynomials[index];
            if (!progress_prefix.empty())
            {
                progress_note(progress_prefix + ":" + name + ":start");
            }
            CommitObjectProfile object_profile;
            polynomial.name = name;
            auto material = std::make_shared<algebra::Polynomial>(std::move(polynomial));
            out[index] = Commitment{
                .name = name,
                .point = commit_polynomial_point(
                    *material,
                    key.pk,
                    profile != nullptr ? &object_profile : nullptr,
                    outer_worker_count > 1 ? &msm_options : nullptr),
                .opening_polynomial = material,
            };
            if (profile != nullptr)
            {
                object_profiles[index] = std::move(object_profile);
            }
            if (!progress_prefix.empty())
            {
                progress_note(progress_prefix + ":" + name + ":done");
            }
        };

        if (outer_worker_count > 1)
        {
            std::atomic<std::size_t> next{0};
            g1_parallel_msm_batch_workers(
                outer_worker_count,
                [&](std::size_t)
                {
                    while (true)
                    {
                        const auto order_index = next.fetch_add(1);
                        if (order_index >= named_polynomials.size())
                        {
                            break;
                        }
                        commit_one(work_order[order_index]);
                    }
                });
        }
        else
        {
            for (std::size_t index = 0; index < named_polynomials.size(); ++index)
            {
                commit_one(index);
            }
        }

        if (profile != nullptr)
        {
            profile->msm_ms += elapsed_ms(msm_start, Clock::now());
            for (auto &object_profile : object_profiles)
            {
                if (!object_profile.name.empty())
                {
                    profile->objects.push_back(std::move(object_profile));
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
                      nullptr);

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

        const auto witness =
            open_single_point_fold(
                folded_plan,
                0,
                point,
                key.pk,
                profile,
                field_worker_cap,
                msm_options);

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

        auto folded =
            build_streamed_folded_opening_plan_pair(
                commitments,
                value_columns,
                powers,
                points,
                field_worker_caps,
                *opening_resolver,
                *semantic_source_resolver);

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

            witnesses[group] =
                open_single_point_fold(
                    folded.plans[group],
                    0,
                    points[group],
                    key.pk,
                    profile,
                    field_worker_caps[group],
                    msm_options);

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
