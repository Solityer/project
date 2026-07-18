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

#include <openssl/sha.h>
#include <openssl/rand.h>

#include "gatzk/algebra/domain_evaluation.hpp"
#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/protocol/schema.hpp"
#include "gatzk/util/logging.hpp"
#include "gatzk/util/route2.hpp"

namespace gatzk::crypto
{
    namespace
    {
        using algebra::FieldElement;
        using Clock = std::chrono::steady_clock;

        std::string bytes_to_hex(const unsigned char* bytes, std::size_t size)
        {
            std::ostringstream out;
            out << std::hex << std::setfill('0');
            for (std::size_t i = 0; i < size; ++i)
            {
                out << std::setw(2) << static_cast<unsigned int>(bytes[i]);
            }
            return out.str();
        }

        std::string sha256_hex(const std::string& data)
        {
            unsigned char digest[SHA256_DIGEST_LENGTH];
            SHA256(reinterpret_cast<const unsigned char*>(data.data()), data.size(), digest);
            return bytes_to_hex(digest, sizeof(digest));
        }

        std::string sha256_bytes_hex(const std::vector<std::uint8_t>& data)
        {
            unsigned char digest[SHA256_DIGEST_LENGTH];
            SHA256(data.data(), data.size(), digest);
            return bytes_to_hex(digest, sizeof(digest));
        }

        std::string sha256_file_hex_cached(const std::filesystem::path& path)
        {
            static std::mutex cache_mutex;
            static std::unordered_map<std::string, std::string> cache;
            const auto size = std::filesystem::file_size(path);
            const auto mtime = std::filesystem::last_write_time(path).time_since_epoch().count();
            const auto key = path.string() + "|" + std::to_string(size) + "|" + std::to_string(mtime);
            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                if (const auto it = cache.find(key); it != cache.end())
                {
                    return it->second;
                }
            }
            std::ifstream input(path, std::ios::binary);
            if (!input)
            {
                throw std::runtime_error("failed to hash SRS material: " + path.string());
            }
            SHA256_CTX ctx;
            SHA256_Init(&ctx);
            std::vector<char> buffer(1U << 20);
            while (input)
            {
                input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
                const auto count = input.gcount();
                if (count > 0)
                {
                    SHA256_Update(&ctx, buffer.data(), static_cast<std::size_t>(count));
                }
            }
            unsigned char digest[SHA256_DIGEST_LENGTH];
            SHA256_Final(digest, &ctx);
            auto hash = bytes_to_hex(digest, sizeof(digest));
            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                cache.emplace(key, hash);
            }
            return hash;
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

        std::string manifest_scalar(const std::string& text, const std::string& key)
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

        std::string manifest_object(const std::string& text, const std::string& key)
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

        std::unordered_map<std::string, std::string> parse_string_map(const std::string& object)
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

        std::unordered_map<std::string, std::size_t> parse_size_map(const std::string& object)
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
            const std::unordered_map<std::string, std::size_t>& domains)
        {
            std::vector<std::pair<std::string, std::size_t>> ordered(
                domains.begin(), domains.end());
            std::sort(ordered.begin(), ordered.end());
            std::ostringstream canonical;
            canonical << "zkgat:domain-size-catalog:v1";
            for (const auto& [name, size] : ordered)
            {
                canonical << '|' << name << '=' << size;
            }
            return sha256_hex(canonical.str());
        }

        std::string bytes_to_lower_hex(const std::uint8_t* bytes, std::size_t size)
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

        std::string canonical_field_hex(const FieldElement& value)
        {
            std::array<std::uint8_t, 32> bytes{};
            value.write_little_endian(bytes.data(), bytes.size());
            return bytes_to_lower_hex(bytes.data(), bytes.size());
        }

        std::string canonical_domain_generator_catalog_hash(
            const std::unordered_map<std::string, std::size_t>& domains,
            const std::unordered_map<std::string, std::string>& generators,
            const std::string& root_derivation_id,
            const std::string& row_order)
        {
            std::vector<std::pair<std::string, std::size_t>> ordered(
                domains.begin(), domains.end());
            std::sort(ordered.begin(), ordered.end());
            std::ostringstream canonical;
            canonical << "zkgat:domain-generator-catalog:v1"
                      << "|root_derivation_id=" << root_derivation_id
                      << "|row_order=" << row_order;
            for (const auto& [name, size] : ordered)
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

        std::string read_text_file(const std::filesystem::path& path)
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
            const std::string& text,
            const std::string& key)
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
            auto nibble = [](char c) -> unsigned {
                if (c >= '0' && c <= '9') return static_cast<unsigned>(c - '0');
                if (c >= 'a' && c <= 'f')
                    return 10U + static_cast<unsigned>(c - 'a');
                throw std::runtime_error(
                    "protocol compiler artifact hex field is not canonical lowercase hex");
            };
            std::string out(value.size() / 2, '\0');
            for (std::size_t i = 0; i < out.size(); ++i)
                out[i] = static_cast<char>((nibble(value[2 * i]) << 4U)
                    | nibble(value[2 * i + 1]));
            return out;
        }

        SRSManifest load_manifest_file(const std::string& srs_path)
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
            manifest.encoding = manifest_scalar(text, "encoding");
            manifest.generator_version = manifest_scalar(text, "generator_version");
            manifest.security_level = manifest_scalar(text, "security_level");
            manifest.kzg_assumption_profile_id = manifest_scalar(text, "kzg_assumption_profile_id");
            manifest.kzg_assumption_domain_catalog_hash =
                manifest_scalar(text, "kzg_assumption_domain_catalog_hash");
            manifest.domain_generator_catalog_hash =
                manifest_scalar(text, "domain_generator_catalog_hash");
            const auto declared_manifest_hash = manifest_scalar(text, "manifest_hash");
            const auto manifest_hash_field = text.rfind("  \"manifest_hash\"");
            if (declared_manifest_hash.empty() || manifest_hash_field == std::string::npos
                || sha256_hex(text.substr(0, manifest_hash_field)) != declared_manifest_hash)
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
            if (manifest.manifest_format.empty() || manifest.srs_id.empty()
                || manifest.srs_type.empty() || manifest.degree_bound == 0
                || manifest.pk_coef_path.empty() || manifest.pk_lagrange_paths.empty()
                || manifest.vk_path.empty()
                || manifest.protocol_compiler_artifact_path.empty()
                || manifest.protocol_compiler_artifact_hash.empty())
            {
                throw std::runtime_error("SRS manifest is missing required material fields");
            }
            return manifest;
        }

        void validate_protocol_compiler_artifact_material(
            const std::filesystem::path& srs_root,
            const SRSManifest& manifest)
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
            if (hash_field == std::string::npos
                || compiler_artifact_scalar(artifact, "artifact_hash")
                    != sha256_hex(artifact.substr(0, hash_field)))
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
            if (protocol_template_hex.empty()
                || proof_schema_template_hex.empty()
                || observation_ledger_template_hex.empty()
                || random_identity_ledger_template_hex.empty()
                || security_budget_ledger_template_hex.empty()
                || max_instance_envelope_hex.empty()
                || max_instance_envelope_hash.empty()
                || compiler_artifact_scalar(artifact, "protocol_schema_id")
                    != manifest.protocol_schema_id
                || sha256_hex(decode_canonical_hex(protocol_template_hex))
                    != manifest.protocol_schema_id
                || compiler_artifact_scalar(artifact, "max_instance_envelope_id")
                    != manifest.max_instance_envelope_id
                || max_instance_envelope_hash
                    != manifest.max_instance_envelope_hash
                || max_instance_envelope_hash
                    != sha256_hex(decode_canonical_hex(
                        max_instance_envelope_hex))
                || static_cast<std::size_t>(std::stoull(
                       compiler_artifact_scalar(artifact, "degree_bound")))
                    != manifest.degree_bound
                || compiler_artifact_scalar(artifact, "domain_size_catalog_hash")
                    != manifest.kzg_assumption_domain_catalog_hash
                || compiler_artifact_scalar(artifact, "domain_generator_catalog_hash")
                    != manifest.domain_generator_catalog_hash
                || compiler_artifact_scalar(artifact, "proof_schema_template_hash")
                    != manifest.proof_schema_template_hash
                || compiler_artifact_scalar(artifact, "observation_ledger_hash")
                    != manifest.observation_ledger_hash
                || compiler_artifact_scalar(artifact, "random_identity_ledger_hash")
                    != manifest.random_identity_ledger_hash
                || compiler_artifact_scalar(artifact, "security_budget_ledger_hash")
                    != manifest.security_budget_ledger_hash)
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
                if (value.empty()
                    || (value.size() > 1 && value.front() == '0')
                    || !std::all_of(value.begin(), value.end(), [](unsigned char ch)
                       { return ch >= '0' && ch <= '9'; }))
                    throw std::runtime_error(
                        "protocol compiler max envelope integer mismatch: " + key);
                const auto parsed = static_cast<std::size_t>(std::stoull(value));
                if (parsed == 0) throw std::runtime_error(
                    "protocol compiler max envelope has zero bound: " + key);
                envelope_sizes.emplace(key, parsed);
            }
            if (canonical_envelope != envelope
                || envelope_sizes != manifest.max_instance_envelope
                || envelope_sizes.at("security_target_bits") != 120
                || envelope_sizes.at("max_nodes") > manifest.domain_sizes.at("N")
                || envelope_sizes.at("max_edges") > manifest.domain_sizes.at("edge")
                || envelope_sizes.at("max_input_dim") > manifest.domain_sizes.at("in")
                || envelope_sizes.at("max_hidden_head_dim") > manifest.domain_sizes.at("d")
                || envelope_sizes.at("max_classes") > manifest.domain_sizes.at("C")
                || envelope_sizes.at("max_concat_dim") > manifest.domain_sizes.at("cat")
                || envelope_sizes.at("max_graphs") > envelope_sizes.at("max_nodes")
                || envelope_sizes.at("max_total_nodes")
                    < envelope_sizes.at("max_nodes")
                || envelope_sizes.at("max_concat_dim")
                    < envelope_sizes.at("max_hidden_head_dim")
                || envelope_sizes.at("max_range_bits") >= 64
                || envelope_sizes.at("max_coefficient_chunks")
                    > std::numeric_limits<std::size_t>::max()
                        / (manifest.domain_sizes.at("edge") - 1)
                || envelope_sizes.at("max_coefficient_object_length")
                    > envelope_sizes.at("max_coefficient_chunks")
                        * (manifest.domain_sizes.at("edge") - 1)
                || envelope_sizes.at("max_nodes")
                    > std::numeric_limits<std::size_t>::max()
                        / envelope_sizes.at("max_input_dim")
                || envelope_sizes.at("max_nodes")
                        * envelope_sizes.at("max_input_dim")
                    > manifest.domain_sizes.at("FH"))
                throw std::runtime_error(
                    "protocol compiler max envelope exceeds the SRS domain catalogue");
            if (decode_canonical_hex(proof_schema_template_hex)
                    != ::gatzk::protocol::protocol_proof_schema_template_canonical_bytes()
                || decode_canonical_hex(observation_ledger_template_hex)
                    != ::gatzk::protocol::protocol_observation_ledger_template_canonical_bytes()
                || decode_canonical_hex(random_identity_ledger_template_hex)
                    != ::gatzk::protocol::protocol_random_identity_ledger_template_canonical_bytes()
                || decode_canonical_hex(security_budget_ledger_template_hex)
                    != ::gatzk::protocol::protocol_security_budget_ledger_template_canonical_bytes()
                || sha256_hex(decode_canonical_hex(proof_schema_template_hex))
                    != manifest.proof_schema_template_hash
                || sha256_hex(decode_canonical_hex(observation_ledger_template_hex))
                    != manifest.observation_ledger_hash
                || sha256_hex(decode_canonical_hex(random_identity_ledger_template_hex))
                    != manifest.random_identity_ledger_hash
                || sha256_hex(decode_canonical_hex(security_budget_ledger_template_hex))
                    != manifest.security_budget_ledger_hash)
                throw std::runtime_error(
                    "protocol compiler artifact embedded registry bytes mismatch");
        }

        void validate_material_hash_value(
            const std::string& actual_hash,
            const std::string& expected_hash,
            const std::string& label)
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


        std::uint64_t read_u64_le(std::istream& input, const std::string& label)
        {
            std::array<std::uint8_t, 8> bytes{};
            input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
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
            const std::vector<std::uint8_t>& bytes,
            std::size_t& offset,
            const std::string& label)
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

        std::uint64_t parse_u64_le_bytes(const std::array<std::uint8_t, 8>& bytes)
        {
            std::uint64_t out = 0;
            for (std::size_t i = 0; i < bytes.size(); ++i)
            {
                out |= static_cast<std::uint64_t>(bytes[i]) << (8U * i);
            }
            return out;
        }

        std::vector<G1Point> load_g1_vector_file(
            const std::filesystem::path& path,
            std::string* actual_hash,
            std::size_t max_points);

        std::vector<G1Point> load_raw_g1_vector_file_prefix(
            const std::filesystem::path& path,
            std::streamoff file_size,
            std::size_t max_points,
            std::string* actual_hash)
        {
            std::ifstream input(path, std::ios::binary);
            if (!input)
            {
                throw std::runtime_error("failed to open SRS G1 material: " + path.string());
            }
            SHA256_CTX ctx;
            SHA256_Init(&ctx);
            auto read_hashed = [&](char* out, std::size_t size, const std::string& label) {
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
            if (magic != expected_raw_magic)
            {
                return load_g1_vector_file(path, actual_hash, 0);
            }
            std::array<std::uint8_t, 8> count_bytes{};
            read_hashed(reinterpret_cast<char*>(count_bytes.data()), count_bytes.size(), "G1 point count");
            const auto count = parse_u64_le_bytes(count_bytes);
            if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
            {
                throw std::runtime_error("SRS G1 material is too large for this platform");
            }
            std::array<std::uint8_t, 8> raw_size_bytes{};
            read_hashed(reinterpret_cast<char*>(raw_size_bytes.data()), raw_size_bytes.size(), "raw G1 point size");
            const auto raw_size = parse_u64_le_bytes(raw_size_bytes);
            if (raw_size != sizeof(mcl::G1))
            {
                throw std::runtime_error("raw SRS G1 material was generated for an incompatible mcl layout");
            }

            const auto point_count = static_cast<std::size_t>(count);
            const auto header_size = static_cast<std::size_t>(24);
            const auto expected_size = header_size + point_count * sizeof(mcl::G1);
            if (expected_size != static_cast<std::size_t>(file_size))
            {
                throw std::runtime_error("raw SRS G1 material size mismatch: " + path.string());
            }
            const auto load_count = max_points == 0 ? point_count : std::min(point_count, max_points);
            std::vector<G1Point> out(load_count);
            if (load_count != 0)
            {
                const auto load_bytes = load_count * sizeof(mcl::G1);
                read_hashed(reinterpret_cast<char*>(&out.front().value), load_bytes, "raw G1 prefix");
            }
            if (max_points != 0 && load_count < point_count)
            {
                unsigned char digest[SHA256_DIGEST_LENGTH];
                SHA256_Final(digest, &ctx);
                if (actual_hash != nullptr)
                {
                    *actual_hash = "prefix:" + std::to_string(load_count) + ":" + bytes_to_hex(digest, sizeof(digest));
                }
                return out;
            }
            std::array<char, 1 << 20> buffer{};
            std::size_t remaining_bytes = (point_count - load_count) * sizeof(mcl::G1);
            while (remaining_bytes != 0)
            {
                const auto chunk = std::min<std::size_t>(remaining_bytes, buffer.size());
                read_hashed(buffer.data(), chunk, "raw G1 suffix");
                remaining_bytes -= chunk;
            }
            unsigned char digest[SHA256_DIGEST_LENGTH];
            SHA256_Final(digest, &ctx);
            if (actual_hash != nullptr)
            {
                *actual_hash = bytes_to_hex(digest, sizeof(digest));
            }
            return out;
        }

        std::vector<G1Point> load_g1_vector_file(
            const std::filesystem::path& path,
            std::string* actual_hash = nullptr,
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
            if (max_points != 0 || magic_probe == expected_raw_magic)
            {
                return load_raw_g1_vector_file_prefix(path, file_size, max_points, actual_hash);
            }
            input.seekg(0);
            std::vector<std::uint8_t> file_bytes(static_cast<std::size_t>(file_size));
            input.read(reinterpret_cast<char*>(file_bytes.data()), file_size);
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
            std::copy_n(reinterpret_cast<const char*>(file_bytes.data()), magic.size(), magic.data());
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
                return out;
            }
            std::vector<std::pair<std::size_t, std::size_t>> payloads;
            payloads.reserve(static_cast<std::size_t>(count));
            for (std::uint64_t i = 0; i < count; ++i)
            {
                const auto size = read_u64_le_bytes(file_bytes, offset, "G1 point size");
                if (size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())
                    || offset + static_cast<std::size_t>(size) > file_bytes.size())
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
                futures.push_back(std::async(std::launch::async, [&file_bytes, &payloads, &out, begin, end]() {
                    for (std::size_t i = begin; i < end; ++i)
                    {
                        const auto [payload_offset, payload_size] = payloads[i];
                        std::vector<std::uint8_t> point_bytes(
                            file_bytes.begin() + static_cast<std::ptrdiff_t>(payload_offset),
                            file_bytes.begin() + static_cast<std::ptrdiff_t>(payload_offset + payload_size));
                        out[i] = deserialize_g1(point_bytes);
                    }
                }));
            }
            for (auto& future : futures)
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

        LoadedVerifierKey load_vk_file(const std::filesystem::path& path, std::string* actual_hash = nullptr)
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
            auto read_g2 = [&]() {
                const auto size = read_u64_le(input, "G2 point size");
                std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
                input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
                if (input.gcount() != static_cast<std::streamsize>(bytes.size()))
                {
                    throw std::runtime_error("truncated SRS G2 point payload");
                }
                return deserialize_g2(bytes);
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
            const SRSManifest& manifest,
            const std::string& expected_srs_type,
            const std::string& expected_srs_id)
        {
            if (manifest.manifest_format != "zkgat_srs_manifest_v7"
                || manifest.generator_version != "zkgat_srs_setup_tool_v7")
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
            if (manifest.kzg_assumption_profile_id != "shifted_kzg_divisibility_v1"
                || manifest.kzg_assumption_degree_bound != manifest.degree_bound
                || manifest.kzg_assumption_security_bits != 128
                || manifest.kzg_assumption_domain_catalog_hash
                    != canonical_domain_catalog_hash(manifest.domain_sizes))
            {
                throw std::runtime_error("SRS shifted-KZG assumption profile is missing or inconsistent");
            }
            if (manifest.root_derivation_id != "bls12_381_fr_generator_5_v1"
                || manifest.row_order != "ascending_powers_from_one"
                || manifest.domain_generators.size() != manifest.domain_sizes.size()
                || manifest.domain_generator_catalog_hash
                    != canonical_domain_generator_catalog_hash(
                        manifest.domain_sizes,
                        manifest.domain_generators,
                        manifest.root_derivation_id,
                        manifest.row_order))
            {
                throw std::runtime_error("SRS domain generator catalogue is missing or inconsistent");
            }
            if (manifest.protocol_schema_id.empty()
                || manifest.max_instance_envelope_id.empty()
                || manifest.max_instance_envelope_hash.empty()
                || manifest.max_instance_envelope_id
                    != manifest.max_instance_envelope_hash
                || manifest.max_instance_envelope.empty()
                || manifest.proof_schema_template_hash.empty()
                || manifest.observation_ledger_hash.empty()
                || manifest.random_identity_ledger_hash.empty()
                || manifest.security_budget_ledger_hash.empty())
            {
                throw std::runtime_error(
                    "SRS protocol compiler artifact binding is incomplete");
            }
            if (manifest.security_level != "120-bit")
                throw std::runtime_error(
                    "SRS manifest security target does not match compiler target");
            for (const auto& [domain_name, domain_size] : manifest.domain_sizes)
            {
                if (domain_size == 0 || domain_size > manifest.degree_bound)
                {
                    throw std::runtime_error("SRS assumption profile does not cover a registered domain size");
                }
                const auto generator = manifest.domain_generators.find(domain_name);
                if (generator == manifest.domain_generators.end()
                    || generator->second
                        != canonical_field_hex(FieldElement::root_of_unity(domain_size)))
                {
                    throw std::runtime_error(
                        "SRS domain generator does not match canonical root derivation: "
                        + domain_name);
                }
            }
        }

        double elapsed_ms(const Clock::time_point& start, const Clock::time_point& end)
        {
            return std::chrono::duration<double, std::milli>(end - start).count();
        }

        std::vector<FieldElement> folding_powers(const FieldElement& challenge, std::size_t count)
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
            Fn&& fn)
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
                futures.push_back(std::async(std::launch::async, [&, begin, end]() {
                    fn(begin, end);
                }));
            }
            for (auto& future : futures)
            {
                future.get();
            }
        }

        template <class Fn>
        void parallel_for_chunks(std::size_t count, std::size_t min_parallel_count, Fn&& fn)
        {
            parallel_for_chunks_capped(count, min_parallel_count, 0, std::forward<Fn>(fn));
        }

        std::string srs_domain_name(const std::string& domain_name)
        {
            return domain_name == "d" ? "d_h" : domain_name;
        }

        std::string srs_exact_domain_name(const std::string& domain_name, std::size_t domain_size)
        {
            return srs_domain_name(domain_name) + "_" + std::to_string(domain_size);
        }

        std::string polynomial_basis_name(const algebra::Polynomial& polynomial)
        {
            return polynomial.basis == algebra::PolynomialBasis::Coefficient ? "coefficient" : "evaluation";
        }

        std::string polynomial_domain_name(const algebra::Polynomial& polynomial)
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

        void progress_note(const std::string& label)
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
            if (!util::route2_options().fast_msm
                || object_count <= 1
                || cpu_count < 4
                || total_scalar_count < (1ULL << 13))
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
            const std::vector<std::pair<std::string, const algebra::Polynomial*>>& named_polynomials)
        {
            std::vector<std::size_t> work_order(named_polynomials.size());
            std::iota(work_order.begin(), work_order.end(), std::size_t{0});
            auto group_key = [&](std::size_t index) {
                const auto* polynomial = named_polynomials[index].second;
                return std::string(polynomial_basis_name(*polynomial))
                    + "|"
                    + polynomial_domain_name(*polynomial)
                    + "|"
                    + std::to_string(polynomial->domain != nullptr ? polynomial->domain->size : 0);
            };
            std::stable_sort(
                work_order.begin(),
                work_order.end(),
                [&](std::size_t lhs, std::size_t rhs) {
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

        bool should_compact_lagrange_prefix_delta(
            const std::vector<FieldElement>& values,
            std::size_t prefix_size,
            const FieldElement& tail_value)
        {
            if (prefix_size < 4096)
            {
                return true;
            }
            constexpr std::size_t kTargetSamples = 4096;
            const auto stride = std::max<std::size_t>(1, prefix_size / kTargetSamples);
            std::size_t sampled = 0;
            std::size_t sampled_nonzero = 0;
            const auto& tail_native = tail_value.native();
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
            const std::vector<G1Point>& lagrange_srs,
            const std::vector<FieldElement>& values,
            std::size_t prefix_size,
            const FieldElement& tail_value,
            G1MSMProfile* profile,
            const G1MSMExecutionOptions* msm_options = nullptr)
        {
            thread_local std::vector<FieldElement> prefix_delta_scratch;
            prefix_delta_scratch.resize(prefix_size);
            const auto& tail_native = tail_value.native();
            auto* scratch_native = FieldElement::mutable_native_array_data(prefix_delta_scratch.data());
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
            const std::vector<G1Point>& lagrange_srs,
            const std::vector<FieldElement>& values,
            std::size_t prefix_size,
            const FieldElement& tail_value,
            G1MSMProfile* profile,
            const G1MSMExecutionOptions* msm_options = nullptr)
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

            const auto& tail_native = tail_value.native();
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
            auto out = g1_multi_scalar_mul_profiled(
                compact_points.data(),
                compact_scalars.data(),
                compact_scalars.size(),
                profile != nullptr ? &compact_profile : nullptr,
                msm_options);
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
            const std::vector<FieldElement>& multiplier,
            const KZGProverKey& pk,
            std::string_view label)
        {
            if (pk.powers_g1 == nullptr
                || domain_size + multiplier.size() > pk.powers_g1->size()
                || domain_size + multiplier.size() > pk.degree_bound + 1)
            {
                throw std::runtime_error(
                    "zero-domain multiple exceeds coefficient SRS degree bound: "
                    + std::string(label));
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

        // KZG多项式承诺主函数
        G1Point commit_polynomial_point(
            const algebra::Polynomial& polynomial,
            const KZGProverKey& pk,
            CommitObjectProfile* profile = nullptr,
            const G1MSMExecutionOptions* msm_options = nullptr)
        {
            G1MSMProfile msm_profile;
            const auto msm_start = Clock::now();
            G1Point point;
            bool constant_tail_folding = false;
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
                point = g1_multi_scalar_mul_profiled(
                    pk.powers_g1->data(),
                    polynomial.values().data(),
                    polynomial.size(),
                    profile != nullptr ? &msm_profile : nullptr,
                    msm_options);
            }
            else
            {
                if (polynomial.domain == nullptr)
                {
                    throw std::runtime_error("evaluation polynomial is missing its domain: " + polynomial.name);
                }
                // 评估多项式采用 拉格朗日SRS + MSM
                const auto base_key = srs_domain_name(polynomial.domain->name);
                const auto exact_key = srs_exact_domain_name(polynomial.domain->name, polynomial.domain->size);
                auto it = pk.lagrange_srs_by_domain.find(exact_key);
                if (it == pk.lagrange_srs_by_domain.end())
                {
                    it = pk.lagrange_srs_by_domain.find(base_key);
                    if (it != pk.lagrange_srs_by_domain.end()
                        && it->second != nullptr
                        && it->second->size() != polynomial.domain->size)
                    {
                        it = pk.lagrange_srs_by_domain.end();
                    }
                }
                if (it == pk.lagrange_srs_by_domain.end())
                {
                    for (auto candidate = pk.lagrange_srs_by_domain.begin();
                         candidate != pk.lagrange_srs_by_domain.end();
                         ++candidate)
                    {
                        if (candidate->second != nullptr && candidate->second->size() == polynomial.domain->size)
                        {
                            it = candidate;
                            break;
                        }
                    }
                }
                if (it == pk.lagrange_srs_by_domain.end())
                {
                    throw std::runtime_error(
                        "missing exact Lagrange SRS for domain: "
                        + base_key
                        + ":"
                        + std::to_string(polynomial.domain->size));
                }
                if (it->second == nullptr)
                {
                    throw std::runtime_error("missing Lagrange proving SRS material for domain: " + exact_key);
                }
                if (polynomial.size() > it->second->size())
                {
                    throw std::runtime_error("evaluation polynomial exceeds Lagrange SRS domain size: " + polynomial.name);
                }
                const auto& values = polynomial.values();
                const auto can_fold_tail = values.size() == polynomial.domain->size;
                std::size_t tail_start = values.size();
                if (can_fold_tail && !values.empty())
                {
                    const auto& tail_value = values.back();
                    while (tail_start > 0 && values[tail_start - 1] == tail_value)
                    {
                        --tail_start;
                    }
                }
                if (can_fold_tail && should_fold_constant_evaluation_tail(values.size(), tail_start))
                {
                    constant_tail_folding = true;
                    const auto& tail_value = values.back();
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
                        const auto folded_prefix_commitment = tail_value.is_zero()
                            ? g1_multi_scalar_mul_profiled(
                                it->second->data(),
                                values.data(),
                                tail_start,
                                profile != nullptr ? &msm_profile : nullptr,
                                msm_options)
                            : commit_lagrange_prefix_delta(
                                *it->second,
                                values,
                                tail_start,
                                tail_value,
                                profile != nullptr ? &msm_profile : nullptr,
                                msm_options);
                        if (profile != nullptr)
                        {
                            msm_profile.input_count = polynomial.size();
                        }
                        point = g1_add(tail_commitment, folded_prefix_commitment);
                    }
                }
                else
                {
                    point = g1_multi_scalar_mul_profiled(
                        it->second->data(),
                        values.data(),
                        polynomial.size(),
                        profile != nullptr ? &msm_profile : nullptr,
                        msm_options);
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
                profile->msm_ms = elapsed_ms(msm_start, Clock::now());
            }
            return point;
        }

        Commitment make_commitment(
            const std::string& name,
            algebra::Polynomial polynomial,
            const KZGProverKey& pk,
            CommitObjectProfile* profile = nullptr,
            const G1MSMExecutionOptions* msm_options = nullptr)
        {
            polynomial.name = name;
            auto material = std::make_shared<algebra::Polynomial>(std::move(polynomial));
            return Commitment{
                .name = name,
                .point = commit_polynomial_point(*material, pk, profile, msm_options),
                .opening_polynomial = material,
            };
        }

        algebra::Polynomial constant_polynomial(const std::string& name, const FieldElement& value)
        {
            return algebra::Polynomial::from_coefficients(name, {value});
        }


        bool coefficient_single_point_quotient_coefficients(
            std::vector<FieldElement>* quotient,
            const std::vector<FieldElement>& coefficients,
            const FieldElement& point,
            const FieldElement& value);



        bool coefficient_single_point_quotient_coefficients(
            std::vector<FieldElement>* quotient,
            const std::vector<FieldElement>& coefficients,
            const FieldElement& point,
            const FieldElement& value)
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
            const auto* coefficients_native = FieldElement::native_array_data(coefficients.data());
            auto* quotient_native = FieldElement::mutable_native_array_data(quotient->data());
            const auto& point_native = point.native();
            quotient_native[coefficient_count - 2] = coefficients_native[coefficient_count - 1];
            for (std::size_t i = coefficient_count - 1; i-- > 1;)
            {
                mcl::Fr product;
                mcl::Fr::mul(product, point_native, quotient_native[i]);
                mcl::Fr::add(quotient_native[i - 1], coefficients_native[i], product);
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
            const std::vector<FieldElement>& coefficients,
            const FieldElement& point,
            const FieldElement& value,
            const KZGProverKey& pk,
            const G1MSMExecutionOptions* msm_options = nullptr)
        {
            if (pk.powers_g1 == nullptr)
            {
                throw std::runtime_error("missing coefficient proving SRS");
            }
            thread_local std::vector<FieldElement> quotient;
            if (!coefficient_single_point_quotient_coefficients(&quotient, coefficients, point, value))
            {
                return g1_zero();
            }
            if (quotient.size() > pk.powers_g1->size() || quotient.size() > pk.degree_bound + 1)
            {
                throw std::runtime_error("coefficient witness quotient exceeds SRS degree bound");
            }
            return g1_multi_scalar_mul_profiled(
                pk.powers_g1->data(),
                quotient.data(),
                quotient.size(),
                nullptr,
                msm_options);
        }


        void accumulate_scaled_coefficients(
            std::vector<FieldElement>* target,
            const std::vector<FieldElement>& source,
            const FieldElement& scale,
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
            auto* target_native = FieldElement::mutable_native_array_data(target->data());
            const auto* source_native = FieldElement::native_array_data(source.data());
            const auto& scale_native = scale.native();
            parallel_for_chunks_capped(source.size(), 1ULL << 15, worker_cap, [&](std::size_t begin, std::size_t end) {
                for (std::size_t i = begin; i < end; ++i)
                {
                    mcl::Fr product;
                    mcl::Fr::mul(product, source_native[i], scale_native);
                    mcl::Fr::add(target_native[i], target_native[i], product);
                }
            });
        }

        bool all_zero(const std::vector<FieldElement>& values)
        {
            return std::all_of(values.begin(), values.end(), [](const FieldElement& value) {
                return value == FieldElement::zero();
            });
        }

        void trim_zero_coefficients(std::vector<FieldElement>* coefficients)
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
            const std::vector<FieldElement>* values = nullptr;
            const std::vector<FieldElement>* zero_domain_blinder = nullptr;
            std::size_t value_count = 0;
            FieldElement scale = FieldElement::zero();
        };

        struct FoldedEvaluationGroup
        {
            std::shared_ptr<algebra::RootOfUnityDomain> domain;
            std::vector<FoldedEvaluationTerm> terms;
            std::vector<FieldElement> scaled_claim_sums;
        };

        struct FoldedOpeningPlan
        {
            std::vector<FieldElement> coefficient_fold;
            std::vector<FieldElement> coefficient_scaled_claim_sums;
            std::vector<FoldedEvaluationGroup> evaluation_groups;
        };

        FieldElement evaluate_with_shared_domain_weights(
            const algebra::Polynomial& polynomial,
            const FieldElement& point,
            const algebra::DomainEvaluationWeights& weights)
        {
            if (polynomial.basis != algebra::PolynomialBasis::Evaluation
                || polynomial.domain == nullptr)
            {
                throw std::runtime_error(
                    "shared domain-weight evaluation requires an evaluation polynomial");
            }
            const auto& values = polynomial.values();
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
                value += polynomial.domain->zero_polynomial_eval(point)
                    * algebra::horner(polynomial.zero_domain_blinder, point);
            }
            return value;
        }

        FoldedEvaluationGroup& folded_evaluation_group(
            FoldedOpeningPlan* plan,
            const std::shared_ptr<algebra::RootOfUnityDomain>& domain,
            std::size_t point_count)
        {
            auto it = std::find_if(
                plan->evaluation_groups.begin(),
                plan->evaluation_groups.end(),
                [&](const FoldedEvaluationGroup& group)
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
            const std::vector<Commitment>& commitments,
            const std::vector<std::vector<FieldElement>>& claimed_values,
            const std::vector<FieldElement>& powers,
            const std::vector<FieldElement>& points,
            std::size_t worker_cap)
        {
            const auto point_count = points.size();
            algebra::DomainEvaluationWeightCache claim_weight_cache;
            struct FoldedOpeningTerm
            {
                const Commitment* commitment = nullptr;
                FieldElement scale = FieldElement::zero();
                std::vector<FieldElement> scaled_claims;
            };

            std::vector<FoldedOpeningTerm> folded_terms;
            folded_terms.reserve(commitments.size());
            for (std::size_t i = 0; i < commitments.size(); ++i)
            {
                if (commitments[i].opening_polynomial == nullptr)
                {
                    throw std::runtime_error("commitment is missing prover-side opening polynomial: " + commitments[i].name);
                }
                auto existing = std::find_if(
                    folded_terms.begin(),
                    folded_terms.end(),
                    [&](const FoldedOpeningTerm& term)
                    {
                        return term.commitment != nullptr
                            && term.commitment->opening_polynomial.get() == commitments[i].opening_polynomial.get();
                    });
                if (existing == folded_terms.end())
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
                    existing->scale += powers[i];
                    for (std::size_t point_index = 0; point_index < point_count; ++point_index)
                    {
                        existing->scaled_claims[point_index] += claimed_values[i][point_index] * powers[i];
                    }
                }
            }

            parallel_for_chunks_capped(
                folded_terms.size(),
                16,
                worker_cap,
                [&](std::size_t begin, std::size_t end)
                {
                    for (std::size_t term_index = begin; term_index < end; ++term_index)
                    {
                        const auto& term = folded_terms[term_index];
                        if (term.commitment == nullptr
                            || term.commitment->opening_polynomial == nullptr
                            || term.scale.is_zero())
                        {
                            continue;
                        }
                        const auto& polynomial = *term.commitment->opening_polynomial;
                        for (std::size_t point_index = 0;
                             point_index < point_count;
                             ++point_index)
                        {
                            FieldElement expected = FieldElement::zero();
                            if (polynomial.basis == algebra::PolynomialBasis::Coefficient)
                            {
                                expected = polynomial.evaluate(points[point_index]);
                            }
                            else
                            {
                                if (polynomial.domain == nullptr)
                                {
                                    throw std::runtime_error(
                                        "evaluation polynomial is missing its domain: "
                                        + term.commitment->name);
                                }
                                const auto& weights = claim_weight_cache.get(
                                    polynomial.domain, points[point_index]);
                                expected = evaluate_with_shared_domain_weights(
                                    polynomial, points[point_index], weights);
                            }
                            const auto claimed =
                                term.scaled_claims[point_index] / term.scale;
                            if (expected != claimed)
                            {
                                throw std::runtime_error(
                                    (polynomial.basis == algebra::PolynomialBasis::Coefficient
                                         ? "coefficient opening claim mismatch for "
                                         : "evaluation opening claim mismatch for ")
                                    + term.commitment->name
                                    + " at point index "
                                    + std::to_string(point_index));
                            }
                        }
                    }
                });

            FoldedOpeningPlan plan;
            plan.coefficient_scaled_claim_sums.assign(point_count, FieldElement::zero());
            for (const auto& term : folded_terms)
            {
                if (term.commitment == nullptr || term.commitment->opening_polynomial == nullptr)
                {
                    throw std::runtime_error("commitment is missing prover-side opening polynomial");
                }
                if (term.scale.is_zero() && all_zero(term.scaled_claims))
                {
                    continue;
                }
                const auto& polynomial = *term.commitment->opening_polynomial;
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
                auto& group = folded_evaluation_group(&plan, polynomial.domain, point_count);
                const auto& source_values = polynomial.values();
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
            trim_zero_coefficients(&plan.coefficient_fold);
            return plan;
        }

        struct EvaluationInverseDenominatorCacheEntry
        {
            std::shared_ptr<algebra::RootOfUnityDomain> domain;
            FieldElement base_point;
            std::vector<FieldElement> base_inverse_denominators;
        };

        std::optional<std::size_t> small_rotation_shift(
            const algebra::RootOfUnityDomain& domain,
            const FieldElement& from,
            const FieldElement& to)
        {
            if (from == to)
            {
                return 0;
            }
            FieldElement value = from;
            const auto limit = std::min<std::size_t>(domain.size, 8);
            for (std::size_t shift = 1; shift < limit; ++shift)
            {
                value *= domain.omega;
                if (value == to)
                {
                    return shift;
                }
            }
            return std::nullopt;
        }

        G1Point open_single_point_fold(
            const FoldedOpeningPlan& plan,
            std::size_t point_index,
            const FieldElement& point,
            const KZGProverKey& pk,
            std::vector<EvaluationInverseDenominatorCacheEntry>* inverse_denominator_cache = nullptr,
            BatchOpeningProfile* profile = nullptr,
            std::size_t worker_cap = 0,
            const G1MSMExecutionOptions* msm_options = nullptr)
        {
            std::vector<EvaluationInverseDenominatorCacheEntry> local_inverse_denominator_cache;
            auto* inverse_cache = inverse_denominator_cache != nullptr
                ? inverse_denominator_cache
                : &local_inverse_denominator_cache;

            auto out = g1_zero();
            const auto& coefficient_scaled_claim_sum = plan.coefficient_scaled_claim_sums[point_index];
            if (!plan.coefficient_fold.empty() || !(coefficient_scaled_claim_sum == FieldElement::zero()))
            {
                try
                {
                    out = g1_add(
                        out,
                        commit_coefficient_single_point_quotient(
                            plan.coefficient_fold,
                            point,
                            coefficient_scaled_claim_sum,
                            pk,
                            msm_options));
                }
                catch (const std::exception& ex)
                {
                    throw std::runtime_error("opening mismatch for folded coefficient group: " + std::string(ex.what()));
                }
            }
            for (const auto& group : plan.evaluation_groups)
            {
                const std::vector<FieldElement>* inverse_denominators = nullptr;
                std::size_t inverse_rotation_shift = 0;
                FieldElement inverse_rotation_scale = FieldElement::one();
                bool reused_inverse_denominators = false;

                const auto inverse_lookup_start = Clock::now();
                for (auto& entry : *inverse_cache)
                {
                    if (entry.domain.get() != group.domain.get())
                    {
                        continue;
                    }
                    const auto shift = small_rotation_shift(*group.domain, entry.base_point, point);
                    if (!shift.has_value())
                    {
                        continue;
                    }
                    inverse_denominators = &entry.base_inverse_denominators;
                    inverse_rotation_shift = *shift;
                    if (inverse_rotation_shift != 0)
                    {
                        inverse_rotation_scale = group.domain->omega.pow(
                            static_cast<std::uint64_t>(group.domain->size - inverse_rotation_shift));
                        reused_inverse_denominators = true;
                    }
                    break;
                }
                if (reused_inverse_denominators && profile != nullptr)
                {
                    profile->denominator_inverse_reuse_hits += 1;
                    profile->denominator_inverse_reuse_ms += elapsed_ms(inverse_lookup_start, Clock::now());
                }

                if (inverse_denominators == nullptr)
                {
                    const auto inverse_start = Clock::now();
                    std::vector<FieldElement> denominators(group.domain->size, FieldElement::zero());
                    parallel_for_chunks_capped(group.domain->size, 1ULL << 15, worker_cap, [&](std::size_t begin, std::size_t end) {
                        for (std::size_t row = begin; row < end; ++row)
                        {
                            const auto domain_point = group.domain->point_at(row);
                            if (domain_point == point)
                            {
                                throw std::runtime_error("opening point overlaps evaluation domain; derivative witness path is not implemented");
                            }
                            denominators[row] = domain_point - point;
                        }
                    });
                    auto computed_inverse_denominators = algebra::batch_invert(denominators);
                    if (profile != nullptr)
                    {
                        profile->denominator_inverse_ms += elapsed_ms(inverse_start, Clock::now());
                        profile->denominator_inverse_cache_misses += 1;
                    }
                    inverse_cache->push_back(EvaluationInverseDenominatorCacheEntry{
                        .domain = group.domain,
                        .base_point = point,
                        .base_inverse_denominators = std::move(computed_inverse_denominators),
                    });
                    inverse_denominators = &inverse_cache->back().base_inverse_denominators;
                }

                std::vector<FieldElement> quotient_values(group.domain->size, FieldElement::zero());
                auto* quotient_values_native = FieldElement::mutable_native_array_data(quotient_values.data());
                struct NativeTerm
                {
                    const std::vector<FieldElement>* values = nullptr;
                    std::size_t value_count = 0;
                    mcl::Fr scale;
                };
                std::vector<NativeTerm> native_terms;
                native_terms.reserve(group.terms.size());
                for (const auto& term : group.terms)
                {
                    native_terms.push_back(NativeTerm{
                        .values = term.values,
                        .value_count = term.value_count,
                        .scale = term.scale.native(),
                    });
                }
                mcl::Fr zero;
                zero.clear();
                mcl::Fr base_numerator;
                mcl::Fr::sub(base_numerator, zero, group.scaled_claim_sums[point_index].native());
                const bool has_rotation = inverse_rotation_shift != 0;
                const mcl::Fr rotation_scale_native = inverse_rotation_scale.native();
                parallel_for_chunks_capped(group.domain->size, 1ULL << 15, worker_cap, [&](std::size_t begin, std::size_t end) {
                    for (std::size_t row = begin; row < end; ++row)
                    {
                        mcl::Fr native_numerator = base_numerator;
                        for (const auto& term : native_terms)
                        {
                            if (row < term.value_count)
                            {
                                mcl::Fr product;
                                mcl::Fr::mul(product, (*term.values)[row].native(), term.scale);
                                mcl::Fr::add(native_numerator, native_numerator, product);
                            }
                        }
                        const auto inverse_index =
                            !has_rotation
                            ? row
                            : (row + group.domain->size - inverse_rotation_shift) & (group.domain->size - 1U);
                        mcl::Fr::mul(
                            quotient_values_native[row],
                            native_numerator,
                            (*inverse_denominators)[inverse_index].native());
                        if (has_rotation)
                        {
                            mcl::Fr::mul(quotient_values_native[row], quotient_values_native[row], rotation_scale_native);
                        }
                    }
                });
                out = g1_add(
                    out,
                    commit_polynomial_point(
                        algebra::Polynomial::from_evaluations(
                            "folded_" + group.domain->name + "_witness",
                            std::move(quotient_values),
                            group.domain),
                        pk,
                        nullptr,
                        msm_options));

                // The domain-value quotient above is the unique degree < n
                // interpolant of the full quotient on H.  For
                // P^zk=P+Z_H R, the omitted high-degree part is exactly
                // Z_H * (R(X)-R(z))/(X-z); its quotient contribution is the
                // small coefficient polynomial (R(X)-R(z))/(X-z).
                std::vector<FieldElement> blinder_quotient_fold;
                for (const auto &term : group.terms)
                {
                    if (term.zero_domain_blinder == nullptr
                        || term.zero_domain_blinder->empty()
                        || term.scale.is_zero())
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
            }
            return out;
        }

        G1Point folded_single_point_commitment(
            const std::vector<Commitment>& commitments,
            const std::vector<FieldElement>& values,
            const std::vector<FieldElement>& powers,
            const G1Point& g1_generator)
        {
            if (powers.size() != commitments.size())
            {
                throw std::runtime_error("folded commitment power count mismatch");
            }
            auto out = g1_zero();
            for (std::size_t i = 0; i < commitments.size(); ++i)
            {
                const auto value_commitment = g1_mul(g1_generator, values[i]);
                const auto numerator = g1_sub(commitments[i].point, value_commitment);
                out = g1_add(out, g1_mul(numerator, powers[i]));
            }
            return out;
        }

        bool verify_pairing_equation(
            const G1Point& numerator,
            const G1Point& witness,
            const FieldElement& point,
            const KZGVerifierKey& vk)
        {
            const auto rhs = g2_sub(vk.g2_tau, g2_mul(vk.g2_one, point));
            if (rhs == g2_zero())
            {
                // tau must remain outside every proof-derived opening point.  A
                // zero [tau-z]_2 makes the KZG equation degenerate and is an SRS
                // compromise, never a valid proof result.
                return false;
            }
            if (util::route2_options().fast_verify_pairing && vk.g2_one_prepared != nullptr)
            {
                return pairing_equal_with_prepared_rhs_a(numerator, *vk.g2_one_prepared, witness, rhs);
            }
            return pairing_equal(numerator, vk.g2_one, witness, rhs);
        }
    }

    std::size_t serialized_size(const Commitment& commitment)
    {
        return serialized_size(commitment.point);
    }

    std::size_t serialized_size(const OpeningProof& proof)
    {
        std::size_t out = 0;
        for (const auto& witness : proof.witnesses)
        {
            out += serialized_size(witness);
        }
        return out;
    }

        std::string required_lagrange_cache_fragment(
            const std::vector<std::pair<std::string, std::size_t>>& required_lagrange_domains)
        {
            if (required_lagrange_domains.empty())
            {
                return "all";
            }
            std::vector<std::string> parts;
            parts.reserve(required_lagrange_domains.size());
            for (const auto& [domain, size] : required_lagrange_domains)
            {
                parts.push_back(srs_domain_name(domain) + ":" + std::to_string(size));
            }
            std::sort(parts.begin(), parts.end());
            std::ostringstream out;
            for (const auto& part : parts)
            {
                out << '|' << part;
            }
            return out.str();
        }

        void add_unique_domain(std::vector<std::string>* domains, const std::string& domain)
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
            const SRSManifest& manifest,
            const std::vector<std::pair<std::string, std::size_t>>& required_lagrange_domains)
        {
            if (required_lagrange_domains.empty())
            {
                std::vector<std::string> all_domains;
                all_domains.reserve(manifest.pk_lagrange_paths.size());
                for (const auto& [domain, _] : manifest.pk_lagrange_paths)
                {
                    all_domains.push_back(domain);
                }
                std::sort(all_domains.begin(), all_domains.end());
                return all_domains;
            }

            std::vector<std::string> selected;
            for (const auto& [domain_name, domain_size] : required_lagrange_domains)
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
                std::vector<std::string> same_size_candidates;
                for (const auto& [candidate, size] : manifest.domain_sizes)
                {
                    if (size == domain_size && manifest.pk_lagrange_paths.contains(candidate))
                    {
                        same_size_candidates.push_back(candidate);
                    }
                }
                std::sort(same_size_candidates.begin(), same_size_candidates.end());
                if (!same_size_candidates.empty())
                {
                    add_unique_domain(&selected, same_size_candidates.front());
                    continue;
                }
                throw std::runtime_error(
                    "SRS manifest lacks Lagrange material for domain "
                    + domain_name
                    + ":"
                    + std::to_string(domain_size));
            }
            std::sort(selected.begin(), selected.end());
            return selected;
        }

    KZGKeyPair KZG::load_keys(
        const std::string& srs_path,
        const std::string& expected_srs_type,
        const std::string& expected_srs_id)
    {
        return load_keys_for_domains(srs_path, expected_srs_type, expected_srs_id, {});
    }

    KZGKeyPair KZG::load_keys_for_domains(
        const std::string& srs_path,
        const std::string& expected_srs_type,
        const std::string& expected_srs_id,
        const std::vector<std::pair<std::string, std::size_t>>& required_lagrange_domains,
        std::size_t required_coefficient_count)
    {
        static std::mutex cache_mutex;
        static std::unordered_map<std::string, KZGKeyPair> cache;
        const auto cache_key =
            srs_path
            + "|"
            + expected_srs_type
            + "|"
            + expected_srs_id
            + "|coef="
            + std::to_string(required_coefficient_count)
            + "|lagrange="
            + required_lagrange_cache_fragment(required_lagrange_domains);
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            if (const auto it = cache.find(cache_key); it != cache.end())
            {
                return it->second;
            }
        }

        const auto manifest = load_manifest_file(srs_path);
        validate_manifest(manifest, expected_srs_type, expected_srs_id);
        const auto srs_root = std::filesystem::path(srs_path);
        validate_protocol_compiler_artifact_material(srs_root, manifest);
        const auto selected_lagrange_domains =
            select_lagrange_srs_domains(manifest, required_lagrange_domains);

        std::string pk_coef_hash;
        progress_note("kzg:srs:pk_coef:start");
        auto powers_g1 = load_g1_vector_file(
            srs_root / manifest.pk_coef_path,
            &pk_coef_hash,
            required_coefficient_count);
        if (pk_coef_hash.rfind("prefix:", 0) == 0)
        {
            validate_material_hash_value(
                sha256_file_hex_cached(srs_root / manifest.pk_coef_path),
                manifest.pk_coef_hash,
                "pk_coef");
        }
        else
        {
            validate_material_hash_value(pk_coef_hash, manifest.pk_coef_hash, "pk_coef");
        }
        progress_note("kzg:srs:pk_coef:done");
        std::unordered_map<std::string, std::shared_ptr<const std::vector<G1Point>>> lagrange_srs;
        for (const auto& domain : selected_lagrange_domains)
        {
            progress_note("kzg:srs:lagrange:" + domain + ":start");
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
            progress_note("kzg:srs:lagrange:" + domain + ":done");
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
        public_transcript << "gatzk:trusted-srs-public-transcript:v7|"
                          << manifest.srs_id << '|' << manifest.degree_bound << '|'
                          << manifest.pk_coef_hash << '|' << manifest.vk_hash << '|'
                          << canonical_domain_catalog_hash(manifest.domain_sizes) << '|'
                          << manifest.protocol_compiler_artifact_hash;
        for (const auto &domain : ceremony_domains)
        {
            const auto hash = manifest.pk_lagrange_hashes.find(domain);
            if (hash == manifest.pk_lagrange_hashes.end())
                throw std::runtime_error(
                    "SRS public transcript is missing a Lagrange material hash: "
                    + domain);
            public_transcript << '|' << domain << '=' << hash->second;
        }
        if (sha256_hex(public_transcript.str())
            != manifest.ceremony_transcript_hash)
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

        // The manifest hashes authenticate bytes, but they do not prove that the
        // coefficient PK and the G2 verifier catalogue were generated from the
        // same toxic-waste scalar.  Audit the relations needed by ordinary KZG
        // and by proof-local model divisibility before exposing either key.
        if (powers_g1.size() < 2
            || !pairing_equal(powers_g1[1], g2_one, powers_g1[0], g2_tau))
        {
            throw std::runtime_error("SRS coefficient/G2 tau consistency audit failed");
        }
        for (const auto &domain_name : selected_lagrange_domains)
        {
            const auto size_it = manifest.domain_sizes.find(domain_name);
            const auto zero_it = loaded_vk.domain_zero_g2.find(domain_name);
            if (size_it == manifest.domain_sizes.end()
                || zero_it == loaded_vk.domain_zero_g2.end()
                || size_it->second >= powers_g1.size())
            {
                throw std::runtime_error(
                    "SRS zero-polynomial consistency audit lacks coefficient material for domain: "
                    + domain_name);
            }
            const auto zero_g1 = g1_sub(powers_g1[size_it->second], powers_g1[0]);
            if (!pairing_equal(zero_g1, g2_one, powers_g1[0], zero_it->second))
            {
                throw std::runtime_error(
                    "SRS zero-polynomial G2 consistency audit failed for domain: "
                    + domain_name);
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
        progress_note("kzg:srs:done");
        return key;
    }

    Commitment KZG::commit_constant(
        const std::string& name,
        const FieldElement& value,
        const KZGKeyPair& key)
    {
        return make_commitment(name, constant_polynomial(name, value), key.pk);
    }

    Commitment KZG::commit(const std::string& name, const algebra::Polynomial& polynomial, const KZGKeyPair& key)
    {
        return make_commitment(name, polynomial, key.pk);
    }

    Commitment KZG::commit_taylor_shifted(
        const Commitment& source,
        const std::string& shifted_name,
        std::size_t configured_degree,
        const FieldElement& delta,
        const KZGKeyPair& key)
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
            key.pk);
    }

    std::vector<Commitment> KZG::commit_constant_batch(
        const std::vector<std::pair<std::string, FieldElement>>& named_values,
        const KZGKeyPair& key,
        CommitBatchProfile* profile)
    {
        std::vector<Commitment> out;
        out.reserve(named_values.size());
        const auto msm_start = Clock::now();
        for (const auto& [name, value] : named_values)
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
        const std::vector<std::pair<std::string, const algebra::Polynomial*>>& named_polynomials,
        const KZGKeyPair& key,
        CommitBatchProfile* profile)
    {
        std::vector<Commitment> out(named_polynomials.size());
        std::vector<CommitObjectProfile> object_profiles(named_polynomials.size());
        const auto msm_start = Clock::now();
        const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        std::size_t total_scalar_count = 0;
        std::size_t max_scalar_count = 0;
        for (const auto& [name, polynomial] : named_polynomials)
        {
            if (polynomial == nullptr)
            {
                throw std::runtime_error("null polynomial in KZG::commit_batch: " + name);
            }
            total_scalar_count += polynomial->size();
            max_scalar_count = std::max(max_scalar_count, polynomial->size());
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
        auto commit_one = [&](std::size_t index) {
            const auto& [name, polynomial] = named_polynomials[index];
            CommitObjectProfile object_profile;
            out[index] = make_commitment(
                name,
                *polynomial,
                key.pk,
                profile != nullptr ? &object_profile : nullptr,
                outer_worker_count > 1 ? &msm_options : nullptr);
            if (profile != nullptr)
            {
                object_profiles[index] = std::move(object_profile);
            }
        };

        if (outer_worker_count > 1)
        {
            const auto work_order = commitment_work_order(named_polynomials);
            std::atomic<std::size_t> next{0};
            std::vector<std::future<void>> futures;
            futures.reserve(outer_worker_count);
            for (std::size_t worker = 0; worker < outer_worker_count; ++worker)
            {
                futures.push_back(std::async(std::launch::async, [&]() {
                    while (true)
                    {
                        const auto index = next.fetch_add(1);
                        if (index >= named_polynomials.size())
                        {
                            break;
                        }
                        commit_one(work_order[index]);
                    }
                }));
            }
            for (auto& future : futures)
            {
                future.get();
            }
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
            for (auto& object_profile : object_profiles)
            {
                if (!object_profile.name.empty())
                {
                    profile->objects.push_back(std::move(object_profile));
                }
            }
        }
        return out;
    }

    std::vector<Commitment> KZG::commit_owned_batch(
        std::vector<std::pair<std::string, algebra::Polynomial>>&& named_polynomials,
        const KZGKeyPair& key,
        CommitBatchProfile* profile,
        const std::string& progress_prefix)
    {
        std::vector<Commitment> out(named_polynomials.size());
        std::vector<CommitObjectProfile> object_profiles(named_polynomials.size());
        const auto msm_start = Clock::now();
        const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        std::size_t total_scalar_count = 0;
        std::size_t max_scalar_count = 0;
        std::vector<std::pair<std::string, const algebra::Polynomial*>> view;
        view.reserve(named_polynomials.size());
        for (const auto& [name, polynomial] : named_polynomials)
        {
            total_scalar_count += polynomial.size();
            max_scalar_count = std::max(max_scalar_count, polynomial.size());
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
        auto commit_one = [&](std::size_t index) {
            auto& [name, polynomial] = named_polynomials[index];
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
            std::vector<std::future<void>> futures;
            futures.reserve(outer_worker_count);
            for (std::size_t worker = 0; worker < outer_worker_count; ++worker)
            {
                futures.push_back(std::async(std::launch::async, [&]() {
                    while (true)
                    {
                        const auto order_index = next.fetch_add(1);
                        if (order_index >= named_polynomials.size())
                        {
                            break;
                        }
                        commit_one(work_order[order_index]);
                    }
                }));
            }
            for (auto& future : futures)
            {
                future.get();
            }
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
            for (auto& object_profile : object_profiles)
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
        const Commitment& long_term_anchor,
        const std::string& proof_local_name,
        const std::string& blinder_name,
        const std::string& domain_name,
        std::size_t domain_size,
        std::size_t blinder_coefficient_count,
        const KZGKeyPair& key)
    {
        if (long_term_anchor.opening_polynomial == nullptr
            || long_term_anchor.opening_polynomial->basis != algebra::PolynomialBasis::Coefficient)
        {
            throw std::runtime_error("model reblinding requires a coefficient-basis anchor polynomial");
        }
        if (domain_size == 0 || blinder_coefficient_count == 0
            || domain_size > key.pk.degree_bound
            || blinder_coefficient_count - 1 > key.pk.degree_bound - domain_size)
        {
            throw std::runtime_error("model reblinding exceeds the configured shifted-KZG degree bound");
        }
        const auto manifest_domain = key.manifest.domain_sizes.find(domain_name);
        if (manifest_domain == key.manifest.domain_sizes.end()
            || manifest_domain->second != domain_size
            || !key.vk.domain_zero_g2.contains(domain_name))
        {
            throw std::runtime_error("model reblinding domain is absent or inconsistent in the SRS manifest");
        }

        std::vector<FieldElement> blinder_coefficients(blinder_coefficient_count);
        std::array<std::uint8_t, 64> random_bytes{};
        bool any_nonzero = false;
        for (auto& coefficient : blinder_coefficients)
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
        if (commitments.size() != 2
            || !verify_domain_divisibility(
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

    OpeningProof KZG::open_batch(
        const std::vector<Commitment>& commitments,
        const std::vector<FieldElement>& points,
        const std::vector<std::vector<FieldElement>>& claimed_values,
        const FieldElement& folding_challenge,
        const KZGKeyPair& key,
        BatchOpeningProfile* profile)
    {
        if (commitments.size() != claimed_values.size())
        {
            throw std::runtime_error("batch opening mismatch between commitments and values");
        }
        for (const auto& values : claimed_values)
        {
            if (values.size() != points.size())
            {
                throw std::runtime_error("batch opening value count does not match points");
            }
        }
        const auto precompute_start = Clock::now();
        const auto powers = folding_powers(folding_challenge, commitments.size());
        std::size_t max_evaluation_domain_size = 0;
        std::size_t max_coefficient_size = 0;
        for (const auto& commitment : commitments)
        {
            if (commitment.opening_polynomial == nullptr)
            {
                continue;
            }
            const auto& polynomial = *commitment.opening_polynomial;
            if (polynomial.basis == algebra::PolynomialBasis::Evaluation && polynomial.domain != nullptr)
            {
                max_evaluation_domain_size = std::max(max_evaluation_domain_size, polynomial.domain->size);
            }
            else
            {
                max_coefficient_size = std::max(max_coefficient_size, polynomial.size());
            }
        }
        const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        const auto point_worker_count = std::min<std::size_t>(points.size(), std::max<std::size_t>(1, cpu_count / 4));
        const auto point_msm_worker_cap = std::max<std::size_t>(1, cpu_count / point_worker_count);
        const bool run_point_parallel =
            util::route2_options().fast_msm
            && points.size() > 1
            && point_worker_count > 1
            && (max_evaluation_domain_size >= (1ULL << 16) || max_coefficient_size >= (1ULL << 16));
        const auto folded_plan = build_folded_opening_plan(
            commitments,
            claimed_values,
            powers,
            points,
            run_point_parallel ? point_msm_worker_cap : 0);
        if (profile != nullptr)
        {
            profile->precompute_ms += elapsed_ms(precompute_start, Clock::now());
        }

        OpeningProof proof;
        proof.witnesses.reserve(points.size());
        std::vector<EvaluationInverseDenominatorCacheEntry> inverse_denominator_cache;
        inverse_denominator_cache.reserve(points.size());
        const auto witness_start = Clock::now();
        if (run_point_parallel && profile != nullptr)
        {
            profile->point_parallel_batches += 1;
            profile->point_parallel_workers += point_worker_count;
            profile->point_msm_worker_cap += point_msm_worker_cap;
        }
        if (run_point_parallel)
        {
            const auto worker_cap = point_msm_worker_cap;
            const G1MSMExecutionOptions msm_options{.max_worker_count = worker_cap};
            proof.witnesses.resize(points.size());
            std::vector<BatchOpeningProfile> local_profiles(points.size());
            std::atomic<std::size_t> next{0};
            std::vector<std::future<void>> futures;
            futures.reserve(point_worker_count);
            for (std::size_t worker = 0; worker < point_worker_count; ++worker)
            {
                futures.push_back(std::async(std::launch::async, [&]() {
                    while (true)
                    {
                        const auto point_index = next.fetch_add(1);
                        if (point_index >= points.size())
                        {
                            break;
                        }
                        std::vector<EvaluationInverseDenominatorCacheEntry> local_inverse_cache;
                        local_inverse_cache.reserve(1);
                        proof.witnesses[point_index] = open_single_point_fold(
                            folded_plan,
                            point_index,
                            points[point_index],
                            key.pk,
                            &local_inverse_cache,
                            profile != nullptr ? &local_profiles[point_index] : nullptr,
                            worker_cap,
                            &msm_options);
                    }
                }));
            }
            for (auto& future : futures)
            {
                future.get();
            }
            if (profile != nullptr)
            {
                for (const auto& local : local_profiles)
                {
                    profile->denominator_inverse_ms += local.denominator_inverse_ms;
                    profile->denominator_inverse_reuse_ms += local.denominator_inverse_reuse_ms;
                    profile->denominator_inverse_reuse_hits += local.denominator_inverse_reuse_hits;
                    profile->denominator_inverse_cache_misses += local.denominator_inverse_cache_misses;
                }
            }
        }
        else
        {
            for (std::size_t point_index = 0; point_index < points.size(); ++point_index)
            {
                proof.witnesses.push_back(open_single_point_fold(
                    folded_plan,
                    point_index,
                    points[point_index],
                    key.pk,
                    &inverse_denominator_cache,
                    profile));
            }
        }
        if (profile != nullptr)
        {
            profile->finalize_ms += elapsed_ms(witness_start, Clock::now());
        }
        return proof;
    }

    bool KZG::verify_batch(
        const std::vector<Commitment>& commitments,
        const std::vector<FieldElement>& points,
            const std::vector<std::vector<FieldElement>>& claimed_values,
            const FieldElement& folding_challenge,
            const OpeningProof& proof,
            const KZGVerifierKey& vk)
    {
        if (commitments.size() != claimed_values.size() || proof.witnesses.size() != points.size())
        {
            throw std::runtime_error("batch verification shape mismatch");
        }
        for (const auto& values : claimed_values)
        {
            if (values.size() != points.size())
            {
                throw std::runtime_error("batch verification value count does not match points");
            }
        }

        const auto powers = folding_powers(folding_challenge, commitments.size());
        std::vector<std::uint8_t> point_results(points.size(), 0);
        auto verify_point_range = [&](std::size_t begin, std::size_t end)
        {
            for (std::size_t point_index = begin; point_index < end; ++point_index)
            {
                std::vector<FieldElement> values_for_point;
                values_for_point.reserve(commitments.size());
                for (const auto& values : claimed_values)
                {
                    values_for_point.push_back(values[point_index]);
                }
                const auto folded_commitment = folded_single_point_commitment(
                    commitments,
                    values_for_point,
                    powers,
                    vk.g1_generator);
                point_results[point_index] = verify_pairing_equation(
                    folded_commitment,
                    proof.witnesses[point_index],
                    points[point_index],
                    vk)
                    ? 1
                    : 0;
            }
        };
        // Each opening point keeps its own pairing equation and acceptance bit.
        // The two canonical domain points are independent and expensive enough
        // to verify concurrently; cap the fan-out so nested verifier work cannot
        // oversubscribe the machine.
        if (points.size() == 1)
        {
            verify_point_range(0, 1);
        }
        else
        {
            parallel_for_chunks_capped(
                points.size(), 1, 2, verify_point_range);
        }
        return std::all_of(
            point_results.begin(), point_results.end(),
            [](std::uint8_t result) { return result != 0; });
    }

    OpeningProof KZG::open_external_fold(
        const std::vector<std::pair<Commitment, FieldElement>>& commitments_and_values,
        const std::vector<FieldElement>& points,
        const FieldElement& folding_challenge,
        const KZGKeyPair& key)
    {
        if (commitments_and_values.size() != points.size())
        {
            throw std::runtime_error("external opening shape mismatch");
        }
        struct PointGroup
        {
            FieldElement point = FieldElement::zero();
            std::vector<std::size_t> claim_indices;
        };
        std::vector<PointGroup> groups;
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            auto it = std::find_if(groups.begin(), groups.end(), [&](const PointGroup& group) {
                return group.point == points[i];
            });
            if (it == groups.end())
            {
                groups.push_back(PointGroup{.point = points[i], .claim_indices = {i}});
            }
            else
            {
                it->claim_indices.push_back(i);
            }
        }
        OpeningProof proof;
        proof.witnesses.reserve(groups.size());
        const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        const auto worker_count = std::min<std::size_t>(
            groups.size(),
            std::max<std::size_t>(1, cpu_count / 2));
        const bool run_parallel =
            util::route2_options().fast_msm
            && groups.size() > 1
            && worker_count > 1;
        auto open_group = [&](std::size_t group_index, const G1MSMExecutionOptions* options) {
            const auto& group = groups[group_index];
            std::vector<Commitment> commitments;
            std::vector<std::vector<FieldElement>> values;
            commitments.reserve(group.claim_indices.size());
            values.reserve(group.claim_indices.size());
            for (const auto claim_index : group.claim_indices)
            {
                commitments.push_back(commitments_and_values[claim_index].first);
                values.push_back({commitments_and_values[claim_index].second});
            }
            const auto powers = folding_powers(folding_challenge, commitments.size());
            const auto plan = build_folded_opening_plan(
                commitments,
                values,
                powers,
                std::vector<FieldElement>{group.point},
                options == nullptr ? 0 : options->max_worker_count);
            return open_single_point_fold(plan, 0, group.point, key.pk, nullptr, nullptr, 0, options);
        };
        if (run_parallel)
        {
            const auto worker_cap = std::max<std::size_t>(1, cpu_count / worker_count);
            const G1MSMExecutionOptions msm_options{.max_worker_count = worker_cap};
            proof.witnesses.resize(groups.size());
            std::atomic<std::size_t> next{0};
            std::vector<std::future<void>> futures;
            futures.reserve(worker_count);
            for (std::size_t worker = 0; worker < worker_count; ++worker)
            {
                futures.push_back(std::async(std::launch::async, [&]() {
                    while (true)
                    {
                        const auto i = next.fetch_add(1);
                        if (i >= groups.size())
                        {
                            break;
                        }
                        proof.witnesses[i] = open_group(i, &msm_options);
                    }
                }));
            }
            for (auto& future : futures)
            {
                future.get();
            }
        }
        else
        {
            for (std::size_t i = 0; i < groups.size(); ++i)
            {
                proof.witnesses.push_back(open_group(i, nullptr));
            }
        }
        return proof;
    }

    bool KZG::verify_external_fold(
        const std::vector<std::pair<Commitment, FieldElement>>& commitments_and_values,
            const std::vector<FieldElement>& points,
            const FieldElement& folding_challenge,
            const OpeningProof& proof,
            const KZGVerifierKey& vk)
    {
        if (commitments_and_values.size() != points.size())
        {
            throw std::runtime_error("external verification shape mismatch");
        }
        struct PointGroup
        {
            FieldElement point = FieldElement::zero();
            std::vector<std::size_t> claim_indices;
        };
        std::vector<PointGroup> groups;
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            auto it = std::find_if(groups.begin(), groups.end(), [&](const PointGroup& group) {
                return group.point == points[i];
            });
            if (it == groups.end())
            {
                groups.push_back(PointGroup{.point = points[i], .claim_indices = {i}});
            }
            else
            {
                it->claim_indices.push_back(i);
            }
        }
        if (proof.witnesses.size() != groups.size())
        {
            throw std::runtime_error("external point-group witness count mismatch");
        }
        std::vector<std::uint8_t> group_results(groups.size(), 0);
        parallel_for_chunks_capped(
            groups.size(),
            1,
            0,
            [&](std::size_t begin, std::size_t end)
            {
                for (std::size_t group_index = begin; group_index < end; ++group_index)
                {
                    const auto& group = groups[group_index];
                    std::vector<Commitment> commitments;
                    std::vector<std::vector<FieldElement>> values;
                    commitments.reserve(group.claim_indices.size());
                    values.reserve(group.claim_indices.size());
                    for (const auto claim_index : group.claim_indices)
                    {
                        commitments.push_back(commitments_and_values[claim_index].first);
                        values.push_back({commitments_and_values[claim_index].second});
                    }
                    OpeningProof group_proof{
                        .witnesses = {proof.witnesses[group_index]}};
                    group_results[group_index] = verify_batch(
                        commitments,
                        {group.point},
                        values,
                        folding_challenge,
                        group_proof,
                        vk)
                        ? 1
                        : 0;
                }
            });
        const auto failed = std::find(group_results.begin(), group_results.end(), 0);
        if (failed != group_results.end())
        {
            if (std::getenv("GATZK_DEBUG_VERIFY") != nullptr)
            {
                util::info(
                    "external_fold_pairing_fail_group="
                    + std::to_string(
                        static_cast<std::size_t>(
                            std::distance(group_results.begin(), failed))));
            }
            return false;
        }
        return true;
    }

    bool KZG::verify_domain_divisibility(
        const Commitment& proof_local,
        const Commitment& long_term_anchor,
        const Commitment& blinder,
        const std::string& domain_name,
        const KZGVerifierKey& vk)
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
        if (util::route2_options().fast_verify_pairing
            && vk.g2_one_prepared != nullptr)
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
