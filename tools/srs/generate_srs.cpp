#include <array>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <openssl/sha.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include <mcl/bn.hpp>

#include "gatzk/algebra/field.hpp"
#include "gatzk/protocol/schema.hpp"

namespace
{

    void write_u64_le(std::ostream &out, std::uint64_t value)
    {
        std::array<std::uint8_t, 8> bytes{};
        for (std::size_t i = 0; i < bytes.size(); ++i)
        {
            bytes[i] = static_cast<std::uint8_t>((value >> (8U * i)) & 0xffU);
        }
        out.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }

    void append_u64_le(std::string &out, std::uint64_t value)
    {
        std::array<char, 8> bytes{};
        for (std::size_t i = 0; i < bytes.size(); ++i)
        {
            bytes[i] = static_cast<char>((value >> (8U * i)) & 0xffU);
        }
        out.append(bytes.data(), bytes.size());
    }

    std::vector<std::uint8_t> serialize_g1(const mcl::G1 &point)
    {
        std::array<std::uint8_t, 256> buffer{};
        const auto size = point.serialize(buffer.data(), buffer.size());
        if (size == 0)
        {
            throw std::runtime_error("failed to serialize G1 point");
        }
        return {buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(size)};
    }

    void append_serialized_g1(std::string &out, const mcl::G1 &point)
    {
        const auto bytes = serialize_g1(point);
        append_u64_le(out, bytes.size());
        out.append(
            reinterpret_cast<const char *>(bytes.data()),
            reinterpret_cast<const char *>(bytes.data()) + static_cast<std::ptrdiff_t>(bytes.size()));
    }

    void append_raw_g1(std::string &out, const mcl::G1 &point)
    {
        const auto *begin = reinterpret_cast<const char *>(&point);
        out.append(begin, begin + static_cast<std::ptrdiff_t>(sizeof(mcl::G1)));
    }

    std::vector<std::uint8_t> serialize_g2(const mcl::G2 &point)
    {
        std::array<std::uint8_t, 512> buffer{};
        const auto size = point.serialize(buffer.data(), buffer.size());
        if (size == 0)
        {
            throw std::runtime_error("failed to serialize G2 point");
        }
        return {buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(size)};
    }

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

    std::string sha256_file_hex(const std::filesystem::path &path)
    {
        std::ifstream input(path, std::ios::binary);
        if (!input)
        {
            throw std::runtime_error("failed to hash " + path.string());
        }
        SHA256_CTX ctx;
        SHA256_Init(&ctx);
        std::array<char, 1 << 16> buffer{};
        while (input)
        {
            input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
            const auto read = input.gcount();
            if (read > 0)
            {
                SHA256_Update(&ctx, buffer.data(), static_cast<std::size_t>(read));
            }
        }
        unsigned char digest[SHA256_DIGEST_LENGTH];
        SHA256_Final(digest, &ctx);
        return bytes_to_hex(digest, sizeof(digest));
    }

    template <typename ChunkFn>
    void write_g1_chunks(const std::filesystem::path &path, std::uint64_t count, ChunkFn &&chunk_fn)
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out)
        {
            throw std::runtime_error("failed to open " + path.string());
        }
        const std::array<char, 8> magic{{'G', 'A', 'T', 'Z', 'K', 'G', '1', 'R'}};
        out.write(magic.data(), static_cast<std::streamsize>(magic.size()));
        write_u64_le(out, count);
        write_u64_le(out, sizeof(mcl::G1));

        const std::uint64_t chunk_size = 8192;
        const std::size_t max_inflight = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        std::vector<std::future<std::string>> inflight;
        inflight.reserve(max_inflight);
        std::uint64_t next = 0;
        auto launch_next = [&]()
        {
            const auto begin = next;
            const auto end = std::min<std::uint64_t>(count, begin + chunk_size);
            next = end;
            inflight.push_back(std::async(std::launch::async, [&, begin, end]()
                                          { return chunk_fn(begin, end); }));
        };

        while (next < count && inflight.size() < max_inflight)
        {
            launch_next();
        }
        while (!inflight.empty())
        {
            auto bytes = inflight.front().get();
            out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
            inflight.erase(inflight.begin());
            if (next < count)
            {
                launch_next();
            }
        }
    }

    void write_vk(
        const std::filesystem::path &path,
        const mcl::G2 &g2_one,
        const mcl::G2 &g2_tau,
        const std::unordered_map<std::string, mcl::G2> &domain_zero_g2)
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out)
        {
            throw std::runtime_error("failed to open " + path.string());
        }
        out << "GATZK-VK-v2\n";
        for (const auto &point : {g2_one, g2_tau})
        {
            const auto bytes = serialize_g2(point);
            write_u64_le(out, bytes.size());
            out.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        }
        std::vector<std::string> domain_names;
        domain_names.reserve(domain_zero_g2.size());
        for (const auto &[name, _] : domain_zero_g2)
        {
            (void)_;
            domain_names.push_back(name);
        }
        std::sort(domain_names.begin(), domain_names.end());
        write_u64_le(out, domain_names.size());
        for (const auto &name : domain_names)
        {
            write_u64_le(out, name.size());
            out.write(name.data(), static_cast<std::streamsize>(name.size()));
            const auto bytes = serialize_g2(domain_zero_g2.at(name));
            write_u64_le(out, bytes.size());
            out.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        }
    }

    std::unordered_map<std::string, std::size_t> parse_domains(
        int argc, char **argv, int first_argument)
    {
        std::unordered_map<std::string, std::size_t> domains;
        for (int i = first_argument; i < argc; ++i)
        {
            const std::string arg = argv[i];
            const auto eq = arg.find('=');
            if (eq == std::string::npos || eq == 0 || eq + 1 == arg.size())
            {
                throw std::runtime_error("domain arguments must use name=size");
            }
            domains.emplace(arg.substr(0, eq), static_cast<std::size_t>(std::stoull(arg.substr(eq + 1))));
        }
        if (domains.empty()) throw std::runtime_error("domain catalogue must not be empty");
        return domains;
    }

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

    std::string domain_catalog_hash(
        const std::unordered_map<std::string, std::size_t> &domains)
    {
        std::ostringstream canonical;
        canonical << "zkgat:domain-size-catalog:v1";
        for (const auto &name : ordered_domain_names(domains))
        {
            canonical << '|' << name << '=' << domains.at(name);
        }
        return sha256_hex(canonical.str());
    }

    std::string canonical_field_hex(const gatzk::algebra::FieldElement &value)
    {
        std::array<std::uint8_t, 32> bytes{};
        value.write_little_endian(bytes.data(), bytes.size());
        return bytes_to_hex(bytes.data(), bytes.size());
    }

    constexpr const char *kRootDerivationId = "bls12_381_fr_generator_5_v1";
    constexpr const char *kRowOrder = "ascending_powers_from_one";

    std::string domain_generator_catalog_hash(
        const std::unordered_map<std::string, std::size_t> &domains)
    {
        std::ostringstream canonical;
        canonical << "zkgat:domain-generator-catalog:v1"
                  << "|root_derivation_id=" << kRootDerivationId
                  << "|row_order=" << kRowOrder;
        for (const auto &name : ordered_domain_names(domains))
        {
            canonical << '|' << name << ':' << domains.at(name) << ':'
                      << canonical_field_hex(
                             gatzk::algebra::FieldElement::root_of_unity(domains.at(name)));
        }
        return sha256_hex(canonical.str());
    }

    void validate_compiler_inputs(
        std::size_t degree_bound,
        const std::unordered_map<std::string, std::size_t> &domains)
    {
        if (degree_bound == 0) throw std::runtime_error("degree bound must be positive");
        const std::vector<std::string> required_domains{
            "C", "FH", "N", "cat", "d", "edge", "in"};
        if (ordered_domain_names(domains) != required_domains)
            throw std::runtime_error(
                "SRS domain catalogue must contain exactly C,FH,N,cat,d,edge,in");
        for (const auto &[domain_name, domain_size] : domains)
        {
            if (domain_name.empty()
                || domain_name.find_first_not_of(
                       "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")
                    != std::string::npos
                || domain_size == 0
                || (domain_size & (domain_size - 1U)) != 0U)
            {
                throw std::runtime_error(
                    "SRS domain must have a canonical name and power-of-two size: "
                    + domain_name);
            }
            if (domain_size > degree_bound)
            {
                throw std::runtime_error(
                    "SRS degree bound does not cover domain " + domain_name);
            }
        }
    }

    std::string read_file(const std::filesystem::path &path)
    {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("failed to read " + path.string());
        return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }

    std::string string_hex(std::string_view value)
    {
        return bytes_to_hex(
            reinterpret_cast<const unsigned char *>(value.data()), value.size());
    }

    std::string decode_hex(std::string_view value)
    {
        if (value.empty() || (value.size() & 1U) != 0U)
            throw std::runtime_error("compiler artifact hex field is malformed");
        auto nibble = [](char c) -> unsigned {
            if (c >= '0' && c <= '9') return static_cast<unsigned>(c - '0');
            if (c >= 'a' && c <= 'f') return 10U + static_cast<unsigned>(c - 'a');
            throw std::runtime_error("compiler artifact hex field is not canonical lowercase hex");
        };
        std::string out(value.size() / 2, '\0');
        for (std::size_t i = 0; i < out.size(); ++i)
            out[i] = static_cast<char>((nibble(value[2 * i]) << 4U)
                | nibble(value[2 * i + 1]));
        return out;
    }

    struct MaxInstanceEnvelope
    {
        std::string canonical_bytes;
        std::string id;
        std::size_t max_layers = 0;
        std::size_t max_hidden_heads_per_layer = 0;
        std::size_t max_hidden_head_dim = 0;
        std::size_t max_input_dim = 0;
        std::size_t max_output_heads = 0;
        std::size_t max_classes = 0;
        std::size_t max_graphs = 0;
        std::size_t max_nodes = 0;
        std::size_t max_edges = 0;
        std::size_t max_total_nodes = 0;
        std::size_t max_concat_dim = 0;
        std::size_t max_range_bits = 0;
        std::size_t max_coefficient_object_length = 0;
        std::size_t max_object_family_count = 0;
        std::size_t max_coefficient_chunks = 0;
        std::size_t security_target_bits = 0;
    };

    MaxInstanceEnvelope parse_max_instance_envelope(
        std::string contents,
        const std::unordered_map<std::string, std::size_t> &domains)
    {
        if (contents.empty() || contents.back() != '\n'
            || contents.find('\r') != std::string::npos)
            throw std::runtime_error(
                "max instance envelope must use canonical LF lines and a final newline");
        const std::vector<std::string> keys = {
            "format", "max_layers", "max_hidden_heads_per_layer",
            "max_hidden_head_dim", "max_input_dim", "max_output_heads",
            "max_classes", "max_graphs", "max_nodes", "max_edges",
            "max_total_nodes", "max_concat_dim", "max_range_bits",
            "max_coefficient_object_length", "max_object_family_count",
            "max_coefficient_chunks", "security_target_bits"};
        std::unordered_map<std::string, std::string> values;
        std::istringstream input(contents);
        std::string line;
        std::size_t line_index = 0;
        while (std::getline(input, line))
        {
            if (line_index >= keys.size())
                throw std::runtime_error(
                    "max instance envelope has trailing fields");
            const auto equals = line.find('=');
            if (equals == std::string::npos
                || line.substr(0, equals) != keys[line_index]
                || equals + 1 == line.size())
                throw std::runtime_error(
                    "max instance envelope field order/encoding mismatch: "
                    + keys[line_index]);
            values.emplace(keys[line_index], line.substr(equals + 1));
            ++line_index;
        }
        if (line_index != keys.size()
            || values.at("format") != "zkgat_max_instance_envelope_v2")
            throw std::runtime_error(
                "max instance envelope format or field count mismatch");
        auto number = [&](const std::string &key)
        {
            const auto &value = values.at(key);
            if (value.empty()
                || (value.size() > 1 && value.front() == '0')
                || !std::all_of(value.begin(), value.end(), [](unsigned char ch)
                   { return ch >= '0' && ch <= '9'; }))
                throw std::runtime_error(
                    "max instance envelope has non-canonical integer: " + key);
            const auto parsed = static_cast<std::size_t>(std::stoull(value));
            if (parsed == 0)
                throw std::runtime_error(
                    "max instance envelope bound must be positive: " + key);
            return parsed;
        };
        MaxInstanceEnvelope out;
        out.canonical_bytes = std::move(contents);
        out.id = sha256_hex(out.canonical_bytes);
        out.max_layers = number("max_layers");
        out.max_hidden_heads_per_layer = number("max_hidden_heads_per_layer");
        out.max_hidden_head_dim = number("max_hidden_head_dim");
        out.max_input_dim = number("max_input_dim");
        out.max_output_heads = number("max_output_heads");
        out.max_classes = number("max_classes");
        out.max_graphs = number("max_graphs");
        out.max_nodes = number("max_nodes");
        out.max_edges = number("max_edges");
        out.max_total_nodes = number("max_total_nodes");
        out.max_concat_dim = number("max_concat_dim");
        out.max_range_bits = number("max_range_bits");
        out.max_coefficient_object_length = number("max_coefficient_object_length");
        out.max_object_family_count = number("max_object_family_count");
        out.max_coefficient_chunks = number("max_coefficient_chunks");
        out.security_target_bits = number("security_target_bits");
        if (out.security_target_bits != 120)
            throw std::runtime_error(
                "max instance envelope security target must equal compiler target 120");
        if (out.max_total_nodes < out.max_nodes
            || out.max_concat_dim < out.max_hidden_head_dim
            || out.max_range_bits >= 64)
            throw std::runtime_error(
                "max instance envelope has inconsistent semantic bounds");
        const auto edge_chunk_capacity = domains.at("edge") - 1;
        if (out.max_coefficient_chunks
                > std::numeric_limits<std::size_t>::max() / edge_chunk_capacity
            || out.max_coefficient_object_length
                > out.max_coefficient_chunks * edge_chunk_capacity)
            throw std::runtime_error(
                "max instance envelope coefficient object exceeds fixed edge chunks");
        if (out.max_nodes > domains.at("N")
            || out.max_edges > domains.at("edge")
            || out.max_input_dim > domains.at("in")
            || out.max_hidden_head_dim > domains.at("d")
            || out.max_classes > domains.at("C")
            || out.max_concat_dim > domains.at("cat")
            || out.max_graphs > out.max_nodes)
            throw std::runtime_error(
                "max instance envelope exceeds a registered work-domain capacity");
        if (out.max_nodes
                > std::numeric_limits<std::size_t>::max() / out.max_input_dim
            || out.max_nodes * out.max_input_dim > domains.at("FH"))
            throw std::runtime_error(
                "max instance envelope feature matrix exceeds FH capacity");
        return out;
    }

    struct SRSCompilerArtifact
    {
        std::string protocol_template_hex;
        std::string proof_schema_template_hex;
        std::string observation_ledger_template_hex;
        std::string random_identity_ledger_template_hex;
        std::string security_budget_ledger_template_hex;
        std::string protocol_schema_id;
        std::string max_instance_envelope_id;
        std::string max_instance_envelope_hex;
        std::string max_instance_envelope_hash;
        std::unordered_map<std::string, std::size_t> max_instance_envelope;
        std::size_t degree_bound = 0;
        std::unordered_map<std::string, std::size_t> domains;
        std::string domain_size_catalog_hash;
        std::string domain_generator_catalog_hash;
        std::string proof_schema_template_hash;
        std::string observation_ledger_hash;
        std::string random_identity_ledger_hash;
        std::string security_budget_ledger_hash;
        std::string artifact_hash;
    };

    std::string artifact_scalar(const std::string &text, const std::string &key)
    {
        const auto prefix = key + "=";
        const auto begin = text.find(prefix);
        if (begin == std::string::npos || (begin != 0 && text[begin - 1] != '\n'))
            throw std::runtime_error("compiler artifact is missing " + key);
        const auto value_begin = begin + prefix.size();
        const auto end = text.find('\n', value_begin);
        return text.substr(value_begin, end - value_begin);
    }

    SRSCompilerArtifact load_compiler_artifact(const std::filesystem::path &path)
    {
        const auto contents = read_file(path);
        constexpr std::string_view magic = "GATZK-SRS-COMPILER-ARTIFACT-v3\n";
        if (!contents.starts_with(magic))
            throw std::runtime_error("compiler artifact magic mismatch");
        const auto hash_field = contents.rfind("artifact_hash=");
        if (hash_field == std::string::npos)
            throw std::runtime_error("compiler artifact hash field is missing");
        SRSCompilerArtifact out;
        out.protocol_template_hex = artifact_scalar(contents, "protocol_template_hex");
        out.proof_schema_template_hex = artifact_scalar(
            contents, "proof_schema_template_hex");
        out.observation_ledger_template_hex = artifact_scalar(
            contents, "observation_ledger_template_hex");
        out.random_identity_ledger_template_hex = artifact_scalar(
            contents, "random_identity_ledger_template_hex");
        out.security_budget_ledger_template_hex = artifact_scalar(
            contents, "security_budget_ledger_template_hex");
        out.protocol_schema_id = artifact_scalar(contents, "protocol_schema_id");
        out.max_instance_envelope_id = artifact_scalar(contents, "max_instance_envelope_id");
        out.max_instance_envelope_hex = artifact_scalar(
            contents, "max_instance_envelope_hex");
        out.max_instance_envelope_hash = artifact_scalar(
            contents, "max_instance_envelope_hash");
        out.degree_bound = static_cast<std::size_t>(
            std::stoull(artifact_scalar(contents, "degree_bound")));
        out.domain_size_catalog_hash = artifact_scalar(contents, "domain_size_catalog_hash");
        out.domain_generator_catalog_hash = artifact_scalar(contents, "domain_generator_catalog_hash");
        out.proof_schema_template_hash = artifact_scalar(contents, "proof_schema_template_hash");
        out.observation_ledger_hash = artifact_scalar(contents, "observation_ledger_hash");
        out.random_identity_ledger_hash = artifact_scalar(contents, "random_identity_ledger_hash");
        out.security_budget_ledger_hash = artifact_scalar(contents, "security_budget_ledger_hash");
        out.artifact_hash = artifact_scalar(contents, "artifact_hash");
        if (out.artifact_hash != sha256_hex(contents.substr(0, hash_field)))
            throw std::runtime_error("compiler artifact canonical hash mismatch");
        std::size_t position = 0;
        while ((position = contents.find("domain.", position)) != std::string::npos)
        {
            if (position != 0 && contents[position - 1] != '\n') { ++position; continue; }
            const auto equals = contents.find('=', position);
            const auto comma = contents.find(',', equals + 1);
            const auto end = contents.find('\n', comma + 1);
            if (equals == std::string::npos || comma == std::string::npos)
                throw std::runtime_error("malformed compiler artifact domain entry");
            const auto name = contents.substr(position + 7, equals - position - 7);
            const auto size = static_cast<std::size_t>(
                std::stoull(contents.substr(equals + 1, comma - equals - 1)));
            const auto omega = contents.substr(comma + 1, end - comma - 1);
            if (!out.domains.emplace(name, size).second
                || omega != canonical_field_hex(
                    gatzk::algebra::FieldElement::root_of_unity(size)))
                throw std::runtime_error("compiler artifact domain entry mismatch: " + name);
            position = end;
        }
        validate_compiler_inputs(out.degree_bound, out.domains);
        if (out.protocol_template_hex.empty()
            || out.proof_schema_template_hex.empty()
            || out.observation_ledger_template_hex.empty()
            || out.random_identity_ledger_template_hex.empty()
            || out.security_budget_ledger_template_hex.empty()
            || out.protocol_schema_id.empty()
            || out.max_instance_envelope_id.empty()
            || out.max_instance_envelope_hex.empty()
            || out.max_instance_envelope_hash.empty()
            || out.proof_schema_template_hash.empty()
            || out.observation_ledger_hash.empty()
            || out.random_identity_ledger_hash.empty()
            || out.security_budget_ledger_hash.empty()
            || out.domain_size_catalog_hash != domain_catalog_hash(out.domains)
            || out.domain_generator_catalog_hash
                != domain_generator_catalog_hash(out.domains))
            throw std::runtime_error("compiler artifact registries are incomplete or inconsistent");
        const auto envelope = parse_max_instance_envelope(
            decode_hex(out.max_instance_envelope_hex), out.domains);
        if (out.max_instance_envelope_hash != envelope.id
            || out.max_instance_envelope_id != envelope.id)
            throw std::runtime_error(
                "compiler artifact max instance envelope mismatch");
        out.max_instance_envelope = {
            {"max_layers", envelope.max_layers},
            {"max_hidden_heads_per_layer", envelope.max_hidden_heads_per_layer},
            {"max_hidden_head_dim", envelope.max_hidden_head_dim},
            {"max_input_dim", envelope.max_input_dim},
            {"max_output_heads", envelope.max_output_heads},
            {"max_classes", envelope.max_classes},
            {"max_graphs", envelope.max_graphs},
            {"max_nodes", envelope.max_nodes},
            {"max_edges", envelope.max_edges},
            {"max_total_nodes", envelope.max_total_nodes},
            {"max_concat_dim", envelope.max_concat_dim},
            {"max_range_bits", envelope.max_range_bits},
            {"max_coefficient_object_length", envelope.max_coefficient_object_length},
            {"max_object_family_count", envelope.max_object_family_count},
            {"max_coefficient_chunks", envelope.max_coefficient_chunks},
            {"security_target_bits", envelope.security_target_bits},
        };
        if (sha256_hex(decode_hex(out.protocol_template_hex))
            != out.protocol_schema_id)
            throw std::runtime_error(
                "compiler artifact protocol template hash mismatch");
        if (decode_hex(out.proof_schema_template_hex)
                != gatzk::protocol::protocol_proof_schema_template_canonical_bytes()
            || decode_hex(out.observation_ledger_template_hex)
                != gatzk::protocol::protocol_observation_ledger_template_canonical_bytes()
            || decode_hex(out.random_identity_ledger_template_hex)
                != gatzk::protocol::protocol_random_identity_ledger_template_canonical_bytes()
            || decode_hex(out.security_budget_ledger_template_hex)
                != gatzk::protocol::protocol_security_budget_ledger_template_canonical_bytes()
            || sha256_hex(decode_hex(out.proof_schema_template_hex))
                != out.proof_schema_template_hash
            || sha256_hex(decode_hex(out.observation_ledger_template_hex))
                != out.observation_ledger_hash
            || sha256_hex(decode_hex(out.random_identity_ledger_template_hex))
                != out.random_identity_ledger_hash
            || sha256_hex(decode_hex(out.security_budget_ledger_template_hex))
                != out.security_budget_ledger_hash)
            throw std::runtime_error(
                "compiler artifact embedded registry template mismatch");
        return out;
    }

    void compile_protocol_artifact(
        const std::filesystem::path &output_path,
        std::size_t degree_bound,
        const std::filesystem::path &max_instance_envelope_path,
        const std::unordered_map<std::string, std::size_t> &domains)
    {
        validate_compiler_inputs(degree_bound, domains);
        const auto protocol_template =
            gatzk::protocol::protocol_template_canonical_bytes();
        const auto proof_schema_template =
            gatzk::protocol::protocol_proof_schema_template_canonical_bytes();
        const auto observation_ledger_template =
            gatzk::protocol::protocol_observation_ledger_template_canonical_bytes();
        const auto random_identity_ledger_template =
            gatzk::protocol::protocol_random_identity_ledger_template_canonical_bytes();
        const auto security_budget_ledger_template =
            gatzk::protocol::protocol_security_budget_ledger_template_canonical_bytes();
        const auto max_instance_envelope = parse_max_instance_envelope(
            read_file(max_instance_envelope_path), domains);
        if (protocol_template.empty())
            throw std::runtime_error("protocol compiler template must not be empty");
        const auto template_hash = sha256_hex(protocol_template);
        std::ostringstream artifact;
        artifact << "GATZK-SRS-COMPILER-ARTIFACT-v3\n";
        artifact << "protocol_template_hex=" << string_hex(protocol_template) << '\n';
        artifact << "proof_schema_template_hex="
                 << string_hex(proof_schema_template) << '\n';
        artifact << "observation_ledger_template_hex="
                 << string_hex(observation_ledger_template) << '\n';
        artifact << "random_identity_ledger_template_hex="
                 << string_hex(random_identity_ledger_template) << '\n';
        artifact << "security_budget_ledger_template_hex="
                 << string_hex(security_budget_ledger_template) << '\n';
        artifact << "protocol_schema_id=" << template_hash << '\n';
        artifact << "max_instance_envelope_id=" << max_instance_envelope.id << '\n';
        artifact << "max_instance_envelope_hex="
                 << string_hex(max_instance_envelope.canonical_bytes) << '\n';
        artifact << "max_instance_envelope_hash="
                 << max_instance_envelope.id << '\n';
        artifact << "degree_bound=" << degree_bound << '\n';
        artifact << "domain_size_catalog_hash=" << domain_catalog_hash(domains) << '\n';
        artifact << "domain_generator_catalog_hash="
                 << domain_generator_catalog_hash(domains) << '\n';
        artifact << "proof_schema_template_hash="
                 << sha256_hex(proof_schema_template) << '\n';
        artifact << "observation_ledger_hash="
                 << sha256_hex(observation_ledger_template) << '\n';
        artifact << "random_identity_ledger_hash="
                 << sha256_hex(random_identity_ledger_template) << '\n';
        artifact << "security_budget_ledger_hash="
                 << sha256_hex(security_budget_ledger_template) << '\n';
        for (const auto &name : ordered_domain_names(domains))
        {
            artifact << "domain." << name << '=' << domains.at(name) << ','
                     << canonical_field_hex(
                            gatzk::algebra::FieldElement::root_of_unity(domains.at(name)))
                     << '\n';
        }
        const auto prefix = artifact.str();
        artifact << "artifact_hash=" << sha256_hex(prefix) << '\n';
        std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error(
            "failed to write compiler artifact: " + output_path.string());
        output << artifact.str();
    }

    void write_manifest(
        const std::filesystem::path &out_dir,
        const std::string &srs_id,
        std::size_t degree_bound,
        const std::unordered_map<std::string, std::size_t> &domains,
        const std::unordered_map<std::string, std::string> &lagrange_hashes,
        const SRSCompilerArtifact &compiler_artifact)
    {
        const auto pk_coef_hash = sha256_file_hex(out_dir / "pk_coef.bin");
        const auto vk_hash = sha256_file_hex(out_dir / "vk.bin");
        const auto compiler_artifact_hash = sha256_file_hex(
            out_dir / "protocol_compiler_artifact.txt");
        std::ostringstream public_transcript;
        public_transcript << "gatzk:trusted-srs-public-transcript:v7|"
                          << srs_id << '|' << degree_bound << '|'
                          << pk_coef_hash << '|' << vk_hash << '|'
                          << domain_catalog_hash(domains) << '|'
                          << compiler_artifact_hash;
        for (const auto &name : ordered_domain_names(domains))
        {
            public_transcript << '|' << name << '=' << lagrange_hashes.at(name);
        }
        const auto transcript_hash = sha256_hex(public_transcript.str());
        std::ostringstream manifest;
        manifest << "{\n";
        manifest << "  \"manifest_format\": \"zkgat_srs_manifest_v7\",\n";
        manifest << "  \"srs_id\": \"" << srs_id << "\",\n";
        manifest << "  \"curve_id\": \"bls12_381\",\n";
        manifest << "  \"field_modulus_id\": \"bls12_381_fr\",\n";
        manifest << "  \"srs_type\": \"trusted_srs\",\n";
        manifest << "  \"ceremony_transcript_hash\": \"" << transcript_hash << "\",\n";
        manifest << "  \"setup_transcript_hash\": \"" << transcript_hash << "\",\n";
        manifest << "  \"degree_bound\": " << degree_bound << ",\n";
        manifest << "  \"domain_sizes\": {\n";
        std::size_t index = 0;
        const auto domain_names = ordered_domain_names(domains);
        for (const auto &domain : domain_names)
        {
            manifest << "    \"" << domain << "\": " << domains.at(domain) << (++index == domains.size() ? "\n" : ",\n");
        }
        manifest << "  },\n";
        manifest << "  \"domain_generators\": {\n";
        index = 0;
        for (const auto &domain : domain_names)
        {
            manifest << "    \"" << domain << "\": \""
                     << canonical_field_hex(
                            gatzk::algebra::FieldElement::root_of_unity(domains.at(domain)))
                     << "\"" << (++index == domains.size() ? "\n" : ",\n");
        }
        manifest << "  },\n";
        manifest << "  \"root_derivation_id\": \"" << kRootDerivationId << "\",\n";
        manifest << "  \"row_order\": \"" << kRowOrder << "\",\n";
        manifest << "  \"protocol_compiler_artifact_path\": \"protocol_compiler_artifact.txt\",\n";
        manifest << "  \"protocol_compiler_artifact_hash\": \""
                 << compiler_artifact_hash << "\",\n";
        manifest << "  \"protocol_schema_id\": \""
                 << compiler_artifact.protocol_schema_id << "\",\n";
        manifest << "  \"max_instance_envelope_id\": \""
                 << compiler_artifact.max_instance_envelope_id << "\",\n";
        manifest << "  \"max_instance_envelope_hash\": \""
                 << compiler_artifact.max_instance_envelope_hash << "\",\n";
        manifest << "  \"max_instance_envelope\": {\n";
        const std::vector<std::string> envelope_keys = {
            "max_layers", "max_hidden_heads_per_layer", "max_hidden_head_dim",
            "max_input_dim", "max_output_heads", "max_classes", "max_graphs",
            "max_nodes", "max_edges", "max_total_nodes", "max_concat_dim",
            "max_range_bits", "max_coefficient_object_length",
            "max_object_family_count", "max_coefficient_chunks",
            "security_target_bits"};
        for (std::size_t envelope_index = 0;
             envelope_index < envelope_keys.size(); ++envelope_index)
        {
            const auto &key = envelope_keys[envelope_index];
            manifest << "    \"" << key << "\": "
                     << compiler_artifact.max_instance_envelope.at(key)
                     << (envelope_index + 1 == envelope_keys.size() ? "\n" : ",\n");
        }
        manifest << "  },\n";
        manifest << "  \"proof_schema_template_hash\": \""
                 << compiler_artifact.proof_schema_template_hash << "\",\n";
        manifest << "  \"observation_ledger_hash\": \""
                 << compiler_artifact.observation_ledger_hash << "\",\n";
        manifest << "  \"random_identity_ledger_hash\": \""
                 << compiler_artifact.random_identity_ledger_hash << "\",\n";
        manifest << "  \"security_budget_ledger_hash\": \""
                 << compiler_artifact.security_budget_ledger_hash << "\",\n";
        manifest << "  \"pk_coef_path\": \"pk_coef.bin\",\n";
        manifest << "  \"pk_coef_hash\": \"" << pk_coef_hash << "\",\n";
        manifest << "  \"pk_lagrange_paths\": {\n";
        index = 0;
        for (const auto &domain : domain_names)
        {
            manifest << "    \"" << domain << "\": \"pk_lagrange_" << domain << ".bin\""
                     << (++index == domains.size() ? "\n" : ",\n");
        }
        manifest << "  },\n";
        manifest << "  \"pk_lagrange_hashes\": {\n";
        index = 0;
        for (const auto &domain : domain_names)
        {
            manifest << "    \"" << domain << "\": \"" << lagrange_hashes.at(domain) << "\""
                     << (++index == lagrange_hashes.size() ? "\n" : ",\n");
        }
        manifest << "  },\n";
        manifest << "  \"vk_path\": \"vk.bin\",\n";
        manifest << "  \"vk_hash\": \"" << vk_hash << "\",\n";
        manifest << "  \"encoding\": \"raw_mcl_bls12_381_v1\",\n";
        manifest << "  \"generator_version\": \"zkgat_srs_setup_tool_v7\",\n";
        manifest << "  \"security_level\": \"120-bit\",\n";
        manifest << "  \"kzg_assumption_profile_id\": \"shifted_kzg_divisibility_v1\",\n";
        manifest << "  \"kzg_assumption_degree_bound\": " << degree_bound << ",\n";
        manifest << "  \"kzg_assumption_security_bits\": 128,\n";
        manifest << "  \"kzg_assumption_domain_catalog_hash\": \"" << domain_catalog_hash(domains) << "\",\n";
        manifest << "  \"domain_generator_catalog_hash\": \""
                 << domain_generator_catalog_hash(domains) << "\",\n";
        // Materialize the exact prefix before appending the hash field.  Calling
        // manifest.str() inside the chained insertion would hash bytes already
        // inserted earlier in that same expression.
        const auto canonical_manifest_prefix = manifest.str();
        manifest << "  \"manifest_hash\": \"" << sha256_hex(canonical_manifest_prefix) << "\"\n";
        manifest << "}\n";

        std::ofstream out(out_dir / "srs_manifest.json", std::ios::trunc);
        if (!out)
        {
            throw std::runtime_error("failed to open SRS manifest for writing");
        }
        out << manifest.str();
    }

} // namespace

int main(int argc, char **argv)
{
    try
    {
        if (argc >= 2 && std::string_view(argv[1]) == "compile")
        {
            if (argc < 6)
            {
                std::cerr << "usage: generate_srs compile <artifact> <degree-bound> <max-envelope-file> domain=size [...]\n";
                return 2;
            }
            mcl::bn::initPairing(mcl::BLS12_381);
            const auto output_path = std::filesystem::path(argv[2]);
            if (std::filesystem::exists(output_path))
                throw std::runtime_error(
                    "refusing to overwrite compiler artifact: " + output_path.string());
            compile_protocol_artifact(
                output_path,
                static_cast<std::size_t>(std::stoull(argv[3])),
                std::filesystem::path(argv[4]),
                parse_domains(argc, argv, 5));
            (void)load_compiler_artifact(output_path);
            std::cout << "wrote protocol compiler artifact " << output_path << "\n";
            return 0;
        }
        if (argc != 5 || std::string_view(argv[1]) != "ceremony")
        {
            std::cerr << "usage: generate_srs ceremony <output-dir> <compiler-artifact> <srs-id>\n";
            return 2;
        }
        const auto out_dir = std::filesystem::path(argv[2]);
        const auto compiler_artifact_path = std::filesystem::path(argv[3]);
        const std::string srs_id = argv[4];
        if (std::filesystem::exists(out_dir))
            throw std::runtime_error(
                "refusing to overwrite SRS output directory: " + out_dir.string());
        const auto compiler_artifact = load_compiler_artifact(compiler_artifact_path);
        const auto degree_bound = compiler_artifact.degree_bound;
        const auto domains = compiler_artifact.domains;

        mcl::bn::initPairing(mcl::BLS12_381);
        mcl::G1 g1;
        mcl::G2 g2;
        mcl::bn::hashAndMapToG1(g1, "gatzk:kzg:g1");
        mcl::bn::hashAndMapToG2(g2, "gatzk:kzg:g2");

        // Complete the ceremony preflight before creating the output directory
        // or writing any key bytes.  A rejected sample must leave no partial
        // material that could be mistaken for a registered SRS.
        mcl::Fr tau;
        gatzk::algebra::FieldElement tau_field;
        while (true)
        {
            tau.setByCSPRNG();
            if (tau.isZero())
            {
                continue;
            }
            tau_field = gatzk::algebra::FieldElement::from_native(tau);
            bool overlaps_registered_domain = false;
            for (const auto &[_, domain_size] : domains)
            {
                if (tau_field.pow(domain_size) == gatzk::algebra::FieldElement::one())
                {
                    overlaps_registered_domain = true;
                    break;
                }
            }
            if (!overlaps_registered_domain)
            {
                break;
            }
        }
        std::filesystem::create_directories(out_dir);
        std::filesystem::copy_file(
            compiler_artifact_path,
            out_dir / "protocol_compiler_artifact.txt",
            std::filesystem::copy_options::none);

        write_g1_chunks(out_dir / "pk_coef.bin", degree_bound + 1, [&](std::uint64_t begin, std::uint64_t end)
                        {
            std::string bytes;
            bytes.reserve(static_cast<std::size_t>(end - begin) * sizeof(mcl::G1));
            mcl::Fr power;
            mcl::Fr::pow(power, tau, begin);
            for (std::uint64_t i = begin; i < end; ++i) {
                (void)i;
                mcl::G1 point;
                mcl::G1::mul(point, g1, power);
                append_raw_g1(bytes, point);
                mcl::Fr::mul(power, power, tau);
            }
            return bytes; });

        mcl::G2 g2_tau;
        mcl::G2::mul(g2_tau, g2, tau);
        std::unordered_map<std::string, std::string> lagrange_hashes;
        std::unordered_map<std::string, mcl::G2> domain_zero_g2;
        for (const auto &[domain_name, domain_size] : domains)
        {
            const auto omega = gatzk::algebra::FieldElement::root_of_unity(domain_size);
            const auto inv_size = gatzk::algebra::FieldElement(static_cast<std::uint64_t>(domain_size)).inv();
            const auto zero_at_tau = tau_field.pow(domain_size) - gatzk::algebra::FieldElement::one();
            mcl::G2 zero_commitment;
            mcl::G2::mul(zero_commitment, g2, zero_at_tau.native());
            domain_zero_g2.emplace(domain_name, zero_commitment);
            const auto path = out_dir / ("pk_lagrange_" + domain_name + ".bin");
            write_g1_chunks(path, domain_size, [&](std::uint64_t begin, std::uint64_t end)
                            {
                std::string bytes;
                bytes.reserve(static_cast<std::size_t>(end - begin) * sizeof(mcl::G1));
                auto domain_point = omega.pow(begin);
                for (std::uint64_t i = begin; i < end; ++i) {
                    (void)i;
                    const auto scalar = zero_at_tau * domain_point * inv_size / (tau_field - domain_point);
                    mcl::G1 point;
                    mcl::G1::mul(point, g1, scalar.native());
                    append_raw_g1(bytes, point);
                    domain_point *= omega;
                }
                return bytes; });
            lagrange_hashes.emplace(domain_name, sha256_file_hex(path));
        }
        write_vk(out_dir / "vk.bin", g2, g2_tau, domain_zero_g2);
        write_manifest(
            out_dir, srs_id, degree_bound, domains, lagrange_hashes,
            compiler_artifact);

        tau_field.clear_sensitive();
        tau.clear();
        std::cout << "wrote SRS material for " << srs_id << " degree_bound=" << degree_bound << "\n";
        return 0;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "generate_srs: " << ex.what() << "\n";
        return 1;
    }
}
