#include "codec.hpp"
#include "gatzk/mechanism/internal.hpp"
#include "real_input.hpp"

#include <openssl/sha.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>

namespace gatzk::mechanism
{
    namespace
    {
        void append_string(std::string &out, const std::string &value)
        {
            if (value.find_first_of("\\\"\r\n\t") != std::string::npos)
                throw std::runtime_error("mechanism statement contains a non-canonical JSON string");
            out.push_back('"');
            out += value;
            out.push_back('"');
        }

        void append_u64(std::string &out, std::uint64_t value) { out += std::to_string(value); }

        bool is_sha256_hex(const std::string &value)
        {
            return value.size() == 64 && std::all_of(value.begin(), value.end(), [](unsigned char c)
            {
                return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
            });
        }

        class JsonReader
        {
        public:
            explicit JsonReader(const std::string &input) : input_(input) {}
            void token(std::string_view value)
            {
                if (input_.compare(position_, value.size(), value) != 0)
                    throw std::runtime_error("mechanism statement is not canonical JSON");
                position_ += value.size();
            }
            std::string string()
            {
                token("\"");
                const auto end = input_.find('"', position_);
                if (end == std::string::npos)
                    throw std::runtime_error("mechanism statement has an unterminated JSON string");
                auto value = input_.substr(position_, end - position_);
                if (value.find_first_of("\\\r\n\t") != std::string::npos)
                    throw std::runtime_error("mechanism statement contains a non-canonical JSON escape");
                position_ = end + 1;
                return value;
            }
            std::uint64_t u64()
            {
                if (position_ == input_.size() || input_[position_] < '0' || input_[position_] > '9')
                    throw std::runtime_error("mechanism statement integer is not canonical");
                if (input_[position_] == '0' && position_ + 1 < input_.size() && input_[position_ + 1] >= '0' && input_[position_ + 1] <= '9')
                    throw std::runtime_error("mechanism statement has a leading-zero integer");
                std::uint64_t value = 0;
                do
                {
                    const auto digit = static_cast<std::uint64_t>(input_[position_] - '0');
                    if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10U)
                        throw std::runtime_error("mechanism statement integer overflows uint64");
                    value = value * 10U + digit;
                    ++position_;
                } while (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9');
                return value;
            }
            bool done() const { return position_ == input_.size(); }
        private:
            const std::string &input_;
            std::size_t position_ = 0;
        };

        void append_u64_array(std::string &out, const std::vector<std::uint64_t> &values)
        {
            out.push_back('[');
            for (std::size_t i = 0; i < values.size(); ++i)
            {
                if (i != 0)
                    out.push_back(',');
                append_u64(out, values[i]);
            }
            out.push_back(']');
        }

        void write_u32(std::vector<std::uint8_t> &out, std::uint32_t value)
        {
            for (unsigned i = 0; i < 4; ++i)
                out.push_back(static_cast<std::uint8_t>(value >> (8U * i)));
        }
        void write_string(std::vector<std::uint8_t> &out, const std::string &value)
        {
            if (value.size() > std::numeric_limits<std::uint32_t>::max())
                throw std::runtime_error("mechanism proof string exceeds codec limit");
            write_u32(out, static_cast<std::uint32_t>(value.size()));
            out.insert(out.end(), value.begin(), value.end());
        }
        void write_field(std::vector<std::uint8_t> &out, const algebra::FieldElement &value)
        {
            std::array<std::uint8_t, 32> bytes{};
            value.write_little_endian(bytes.data(), bytes.size());
            write_u32(out, static_cast<std::uint32_t>(bytes.size()));
            out.insert(out.end(), bytes.begin(), bytes.end());
        }
        void write_point(std::vector<std::uint8_t> &out, const crypto::G1Point &point)
        {
            const auto bytes = crypto::serialize(point);
            write_u32(out, static_cast<std::uint32_t>(bytes.size()));
            out.insert(out.end(), bytes.begin(), bytes.end());
        }

        class BinaryReader
        {
        public:
            explicit BinaryReader(const std::vector<std::uint8_t> &bytes) : bytes_(bytes) {}
            std::uint8_t u8()
            {
                require(1);
                return bytes_[position_++];
            }
            std::uint32_t u32()
            {
                require(4);
                std::uint32_t value = 0;
                for (unsigned i = 0; i < 4; ++i)
                    value |= static_cast<std::uint32_t>(bytes_[position_++]) << (8U * i);
                return value;
            }
            std::string string(std::size_t max_size = 1U << 20)
            {
                const auto size = u32();
                if (size > max_size)
                    throw std::runtime_error("mechanism proof string exceeds codec limit");
                require(size);
                std::string value(bytes_.begin() + static_cast<std::ptrdiff_t>(position_), bytes_.begin() + static_cast<std::ptrdiff_t>(position_ + size));
                position_ += size;
                return value;
            }
            std::vector<std::uint8_t> blob(std::size_t max_size)
            {
                const auto size = u32();
                if (size > max_size)
                    throw std::runtime_error("mechanism proof blob exceeds codec limit");
                require(size);
                std::vector<std::uint8_t> value(bytes_.begin() + static_cast<std::ptrdiff_t>(position_),
                                                bytes_.begin() + static_cast<std::ptrdiff_t>(position_ + size));
                position_ += size;
                return value;
            }
            void magic(std::string_view value)
            {
                require(value.size());
                for (const auto c : value)
                    if (bytes_[position_++] != static_cast<std::uint8_t>(c))
                        throw std::runtime_error("mechanism proof magic/version mismatch");
            }
            bool done() const { return position_ == bytes_.size(); }
        private:
            void require(std::size_t count) const
            {
                if (count > bytes_.size() - position_)
                    throw std::runtime_error("truncated mechanism proof artifact");
            }
            const std::vector<std::uint8_t> &bytes_;
            std::size_t position_ = 0;
        };

        algebra::FieldElement read_field(BinaryReader &reader)
        {
            const auto bytes = reader.blob(32);
            if (bytes.size() != 32)
                throw std::runtime_error("mechanism proof field encoding must be 32 bytes");
            return algebra::FieldElement::from_canonical_little_endian(bytes.data(), bytes.size());
        }
        crypto::G1Point read_point(BinaryReader &reader)
        {
            const auto bytes = reader.blob(128);
            if (bytes.empty())
                throw std::runtime_error("mechanism proof contains an empty G1 encoding");
            auto point = crypto::deserialize_g1(bytes);
            if (crypto::serialize(point) != bytes)
                throw std::runtime_error("mechanism proof contains a non-canonical G1 encoding");
            return point;
        }
    }

    std::string to_string(Mechanism value)
    {
        if (value == Mechanism::FeatureFolding)
            return "feature_folding";
        if (value == Mechanism::SharedNeighborhood)
            return "shared_neighborhood";
        throw std::runtime_error("invalid mechanism enum");
    }
    std::string to_string(Variant value)
    {
        if (value == Variant::Original)
            return "original";
        if (value == Variant::Control)
            return "control";
        throw std::runtime_error("invalid variant enum");
    }
    std::string to_string(Backend value)
    {
        if (value == Backend::Cpu)
            return "cpu";
        if (value == Backend::Cuda)
            return "cuda";
        throw std::runtime_error("invalid backend enum");
    }
    Mechanism parse_mechanism(const std::string &value)
    {
        if (value == "feature_folding")
            return Mechanism::FeatureFolding;
        if (value == "shared_neighborhood")
            return Mechanism::SharedNeighborhood;
        throw std::runtime_error("unsupported mechanism: " + value);
    }
    Variant parse_variant(const std::string &value)
    {
        if (value == "original")
            return Variant::Original;
        if (value == "control")
            return Variant::Control;
        throw std::runtime_error("unsupported variant: " + value);
    }
    Backend parse_backend(const std::string &value)
    {
        if (value == "cpu")
            return Backend::Cpu;
        if (value == "cuda")
            return Backend::Cuda;
        throw std::runtime_error("unsupported backend: " + value);
    }

    std::string PublicStatement::canonical_json() const
    {
        if (node_ptr.size() != graphs + 1 || edge_ptr.size() != graphs + 1 || topology.size() != edges)
            throw std::runtime_error("mechanism public statement topology shape mismatch");
        std::string out;
        out.reserve(256 + topology.size() * 32);
        const auto ledger_digest = internal::compute_security_budget_ledger_sha256(*this);
        if (!security_budget_ledger_sha256.empty() && security_budget_ledger_sha256 != ledger_digest)
            throw std::runtime_error("mechanism statement security ledger digest is stale");
        out += "{\"format\":\"zkgat_mechanism_statement_v8\",\"dataset\":";
        append_string(out, dataset_id);
        out += ",\"input_preprocess_id\":";
        append_string(out, input_preprocess_id);
        out += ",\"real_input\":{\"bundle_sha256\":";
        append_string(out, real_input_bundle_sha256);
        out += ",\"witness_sha256\":";
        append_string(out, real_input_witness_sha256);
        out += ",\"metadata_sha256\":";
        append_string(out, real_input_metadata_sha256);
        out += ",\"checkpoint_bundle_sha256\":";
        append_string(out, checkpoint_bundle_sha256);
        out += ",\"source_manifest_sha256\":";
        append_string(out, input_source_manifest_sha256);
        out += ",\"layer\":";
        append_u64(out, source_layer);
        out += ",\"head\":";
        append_u64(out, source_head);
        out += ",\"formal_head_width\":";
        append_u64(out, formal_head_width);
        out += "},\"mechanism\":";
        append_string(out, to_string(mechanism));
        out += ",\"variant\":";
        append_string(out, to_string(variant));
        out += ",\"workload\":{\"nodes\":";
        append_u64(out, nodes);
        out += ",\"edges\":";
        append_u64(out, edges);
        out += ",\"width\":";
        append_u64(out, width);
        out += ",\"graphs\":";
        append_u64(out, graphs);
        out += ",\"seed\":";
        append_u64(out, seed);
        out += "},\"node_ptr\":";
        append_u64_array(out, node_ptr);
        out += ",\"edge_ptr\":";
        append_u64_array(out, edge_ptr);
        out += ",\"topology\":[";
        for (std::size_t i = 0; i < topology.size(); ++i)
        {
            if (i != 0)
                out.push_back(',');
            out.push_back('[');
            append_u64(out, topology[i].graph);
            out.push_back(',');
            append_u64(out, topology[i].src);
            out.push_back(',');
            append_u64(out, topology[i].dst);
            out.push_back(']');
        }
        out += "],\"srs\":{\"id\":";
        append_string(out, srs_id);
        out += ",\"manifest_sha256\":";
        append_string(out, srs_manifest_sha256);
        out += ",\"protocol_schema_id\":";
        append_string(out, parent_protocol_schema_id);
        out += ",\"kzg_degree_bound\":";
        append_u64(out, parent_srs_degree_bound);
        out += "},\"domains\":{\"node\":";
        append_string(out, node_domain);
        out += ",\"node_size\":";
        append_u64(out, node_domain_size);
        out += ",\"edge\":";
        append_string(out, edge_domain);
        out += ",\"edge_size\":";
        append_u64(out, edge_domain_size);
        out += "},\"mechanism_schema_id\":";
        append_string(out, mechanism_schema_id);
        out += ",\"security\":";
        out += internal::security_budget_ledger_json(*this);
        out += ",\"pair_input_commitment_sha256\":";
        append_string(out, pair_input_commitment_sha256);
        out.push_back('}');
        return out;
    }

    PublicStatement PublicStatement::parse_canonical_json(const std::string &bytes)
    {
        JsonReader r(bytes);
        PublicStatement s;
        r.token("{\"format\":\"zkgat_mechanism_statement_v8\",\"dataset\":");
        s.dataset_id = r.string();
        r.token(",\"input_preprocess_id\":");
        s.input_preprocess_id = r.string();
        r.token(",\"real_input\":{\"bundle_sha256\":");
        s.real_input_bundle_sha256 = r.string();
        r.token(",\"witness_sha256\":");
        s.real_input_witness_sha256 = r.string();
        r.token(",\"metadata_sha256\":");
        s.real_input_metadata_sha256 = r.string();
        r.token(",\"checkpoint_bundle_sha256\":");
        s.checkpoint_bundle_sha256 = r.string();
        r.token(",\"source_manifest_sha256\":");
        s.input_source_manifest_sha256 = r.string();
        r.token(",\"layer\":");
        s.source_layer = static_cast<std::size_t>(r.u64());
        r.token(",\"head\":");
        s.source_head = static_cast<std::size_t>(r.u64());
        r.token(",\"formal_head_width\":");
        s.formal_head_width = static_cast<std::size_t>(r.u64());
        r.token("},\"mechanism\":");
        s.mechanism = parse_mechanism(r.string());
        r.token(",\"variant\":");
        s.variant = parse_variant(r.string());
        r.token(",\"workload\":{\"nodes\":");
        s.nodes = static_cast<std::size_t>(r.u64());
        r.token(",\"edges\":");
        s.edges = static_cast<std::size_t>(r.u64());
        r.token(",\"width\":");
        s.width = static_cast<std::size_t>(r.u64());
        r.token(",\"graphs\":");
        s.graphs = static_cast<std::size_t>(r.u64());
        if (s.dataset_id != "synthetic" && !internal::valid_real_graph_count(s.dataset_id, s.graphs))
            throw std::runtime_error("real mechanism statement graph count is outside the complete registered component selection");
        r.token(",\"seed\":");
        s.seed = r.u64();
        if (s.nodes == 0 || s.edges < s.nodes || s.width == 0 || s.width > 32 || s.graphs == 0 || s.graphs > s.nodes)
            throw std::runtime_error("mechanism statement workload exceeds the canonical codec bounds");
        r.token("},\"node_ptr\":[");
        if (s.graphs == 0 || s.graphs > 4096)
            throw std::runtime_error("mechanism statement graph count is outside the codec limit");
        for (std::size_t i = 0; i <= s.graphs; ++i)
        {
            if (i != 0)
                r.token(",");
            s.node_ptr.push_back(r.u64());
        }
        r.token("],\"edge_ptr\":[");
        for (std::size_t i = 0; i <= s.graphs; ++i)
        {
            if (i != 0)
                r.token(",");
            s.edge_ptr.push_back(r.u64());
        }
        r.token("],\"topology\":[");
        if (s.edges > (1U << 24))
            throw std::runtime_error("mechanism statement edge count exceeds the codec limit");
        for (std::size_t i = 0; i < s.edges; ++i)
        {
            if (i != 0)
                r.token(",");
            r.token("[");
            Edge e;
            e.graph = r.u64();
            r.token(",");
            e.src = r.u64();
            r.token(",");
            e.dst = r.u64();
            r.token("]");
            s.topology.push_back(e);
        }
        r.token("],\"srs\":{\"id\":");
        s.srs_id = r.string();
        r.token(",\"manifest_sha256\":");
        s.srs_manifest_sha256 = r.string();
        r.token(",\"protocol_schema_id\":");
        s.parent_protocol_schema_id = r.string();
        r.token(",\"kzg_degree_bound\":");
        s.parent_srs_degree_bound = static_cast<std::size_t>(r.u64());
        r.token("},\"domains\":{\"node\":");
        s.node_domain = r.string();
        r.token(",\"node_size\":");
        s.node_domain_size = static_cast<std::size_t>(r.u64());
        r.token(",\"edge\":");
        s.edge_domain = r.string();
        r.token(",\"edge_size\":");
        s.edge_domain_size = static_cast<std::size_t>(r.u64());
        const auto *real_catalog = internal::registered_real_dataset_catalog(s.dataset_id);
        const bool synthetic_catalog = s.dataset_id == "synthetic" && s.node_domain_size == 4096 && s.edge_domain_size == 32768;
        const bool registered_real_domains = real_catalog != nullptr && s.node_domain_size == real_catalog->node_domain_size &&
                                             s.edge_domain_size == real_catalog->edge_domain_size;
        if (s.node_domain != "N" || s.edge_domain != "edge" || (!synthetic_catalog && !registered_real_domains) ||
            s.nodes >= s.node_domain_size || s.edges >= s.edge_domain_size)
            throw std::runtime_error("mechanism statement domain catalog is outside the registered trusted datasets");
        r.token("},\"mechanism_schema_id\":");
        s.mechanism_schema_id = r.string();
        r.token(",\"security\":");
        r.token(internal::security_budget_ledger_json(s));
        s.security_budget_ledger_sha256 = internal::compute_security_budget_ledger_sha256(s);
        r.token(",\"pair_input_commitment_sha256\":");
        s.pair_input_commitment_sha256 = r.string();
        r.token("}");
        if (s.dataset_id == "synthetic")
        {
            if (!s.input_preprocess_id.empty() || !s.real_input_bundle_sha256.empty() || !s.real_input_metadata_sha256.empty() ||
                !s.real_input_witness_sha256.empty() || !s.checkpoint_bundle_sha256.empty() || !s.input_source_manifest_sha256.empty() || s.formal_head_width != 0 ||
                s.node_domain_size != 4096 || s.edge_domain_size != 32768)
                throw std::runtime_error("synthetic mechanism statement has real-input provenance or a nonlegacy domain");
        }
        else
        {
            if (real_catalog == nullptr || s.input_preprocess_id != real_catalog->input_preprocess_id ||
                s.source_layer != 0 || s.source_head != 0 ||
                s.formal_head_width != 8 || s.width > 8 || (s.mechanism == Mechanism::SharedNeighborhood && s.width != 8) ||
                (s.mechanism == Mechanism::FeatureFolding && (s.width < 1 || s.width > 8)) ||
                !is_sha256_hex(s.real_input_bundle_sha256) || !is_sha256_hex(s.real_input_witness_sha256) || !is_sha256_hex(s.real_input_metadata_sha256) ||
                !is_sha256_hex(s.checkpoint_bundle_sha256) || !is_sha256_hex(s.input_source_manifest_sha256) ||
                s.node_domain_size != real_catalog->node_domain_size || s.edge_domain_size != real_catalog->edge_domain_size)
                throw std::runtime_error("real mechanism statement dataset or source provenance is invalid");
        }
        if (!r.done() || s.canonical_json() != bytes)
            throw std::runtime_error("mechanism statement is not in its unique canonical encoding");
        return s;
    }

    namespace internal
    {
        std::string sha256_hex(std::string_view bytes)
        {
            std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
            SHA256(reinterpret_cast<const unsigned char *>(bytes.data()), bytes.size(), digest.data());
            static constexpr char hex[] = "0123456789abcdef";
            std::string out;
            out.reserve(64);
            for (const auto byte : digest)
            {
                out.push_back(hex[byte >> 4U]);
                out.push_back(hex[byte & 15U]);
            }
            return out;
        }

        std::vector<std::uint8_t> encode_proof(const MechanismProof &proof)
        {
            std::vector<std::uint8_t> out{'G','Z','K','M','E','C','H','2'};
            write_u32(out, 2);
            out.push_back(static_cast<std::uint8_t>(proof.mechanism));
            out.push_back(static_cast<std::uint8_t>(proof.variant));
            for (const auto *value : {&proof.statement_sha256, &proof.srs_id, &proof.srs_manifest_sha256, &proof.parent_protocol_schema_id,
                                      &proof.mechanism_schema_id, &proof.security_budget_ledger_sha256})
                write_string(out, *value);
            if (proof.commitments.size() > 4096 || proof.opening_groups.size() > 512)
                throw std::runtime_error("mechanism proof object count exceeds codec limit");
            std::set<std::string> commitment_labels;
            write_u32(out, static_cast<std::uint32_t>(proof.commitments.size()));
            for (const auto &entry : proof.commitments)
            {
                if (entry.label.empty() || !commitment_labels.insert(entry.label).second)
                    throw std::runtime_error("mechanism proof has duplicate or empty commitment labels");
                write_string(out, entry.label);
                write_point(out, entry.point);
            }
            std::set<std::string> group_labels;
            write_u32(out, static_cast<std::uint32_t>(proof.opening_groups.size()));
            for (const auto &group : proof.opening_groups)
            {
                if (group.label.empty() || !group_labels.insert(group.label).second || group.commitment_labels.size() != group.values.size() ||
                    group.commitment_labels.size() > 65536)
                    throw std::runtime_error("mechanism proof has a malformed opening group");
                write_string(out, group.label);
                write_u32(out, static_cast<std::uint32_t>(group.commitment_labels.size()));
                std::set<std::string> members;
                for (const auto &label : group.commitment_labels)
                {
                    if (!commitment_labels.contains(label) || !members.insert(label).second)
                        throw std::runtime_error("mechanism proof group has an unknown or duplicate commitment label");
                    write_string(out, label);
                }
                for (const auto &value : group.values)
                    write_field(out, value);
                write_point(out, group.witness);
            }
            return out;
        }

        MechanismProof decode_proof(const std::vector<std::uint8_t> &bytes)
        {
            if (bytes.size() > (1ULL << 30))
                throw std::runtime_error("mechanism proof exceeds the 1 GiB codec limit");
            BinaryReader r(bytes);
            r.magic("GZKMECH2");
            if (r.u32() != 2)
                throw std::runtime_error("unsupported mechanism proof codec version");
            MechanismProof p;
            const auto mechanism = r.u8();
            const auto variant = r.u8();
            if (mechanism > static_cast<std::uint8_t>(Mechanism::SharedNeighborhood) || variant > static_cast<std::uint8_t>(Variant::Control))
                throw std::runtime_error("mechanism proof has an invalid mechanism or variant");
            p.mechanism = static_cast<Mechanism>(mechanism);
            p.variant = static_cast<Variant>(variant);
            p.statement_sha256 = r.string(128);
            p.srs_id = r.string(256);
            p.srs_manifest_sha256 = r.string(128);
            p.parent_protocol_schema_id = r.string(128);
            p.mechanism_schema_id = r.string(128);
            p.security_budget_ledger_sha256 = r.string(128);
            const auto count = r.u32();
            if (count > 4096)
                throw std::runtime_error("mechanism proof commitment count exceeds codec limit");
            std::set<std::string> commitment_labels;
            for (std::uint32_t i = 0; i < count; ++i)
            {
                CommitmentRecord item{.label = r.string(256), .point = read_point(r)};
                if (item.label.empty() || !commitment_labels.insert(item.label).second)
                    throw std::runtime_error("mechanism proof has duplicate or empty commitment labels");
                p.commitments.push_back(std::move(item));
            }
            const auto group_count = r.u32();
            if (group_count > 512)
                throw std::runtime_error("mechanism proof opening group count exceeds codec limit");
            std::set<std::string> group_labels;
            for (std::uint32_t i = 0; i < group_count; ++i)
            {
                OpeningGroup group;
                group.label = r.string(256);
                if (group.label.empty() || !group_labels.insert(group.label).second)
                    throw std::runtime_error("mechanism proof has duplicate or empty opening group labels");
                const auto member_count = r.u32();
                if (member_count > 65536)
                    throw std::runtime_error("mechanism proof opening group exceeds codec limit");
                std::set<std::string> members;
                for (std::uint32_t j = 0; j < member_count; ++j)
                {
                    auto label = r.string(256);
                    if (!commitment_labels.contains(label) || !members.insert(label).second)
                        throw std::runtime_error("mechanism proof group has unknown or duplicate members");
                    group.commitment_labels.push_back(std::move(label));
                }
                for (std::uint32_t j = 0; j < member_count; ++j)
                    group.values.push_back(read_field(r));
                group.witness = read_point(r);
                p.opening_groups.push_back(std::move(group));
            }
            if (!r.done())
                throw std::runtime_error("mechanism proof contains trailing bytes");
            return p;
        }

        void write_binary_file(const std::filesystem::path &path, const std::vector<std::uint8_t> &bytes)
        {
            std::ofstream f(path, std::ios::binary | std::ios::trunc);
            if (!f)
                throw std::runtime_error("cannot open mechanism proof for writing: " + path.string());
            f.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
            f.flush();
            if (!f)
                throw std::runtime_error("failed writing mechanism proof: " + path.string());
        }
        std::vector<std::uint8_t> read_binary_file(const std::filesystem::path &path, std::size_t max_bytes)
        {
            std::ifstream f(path, std::ios::binary | std::ios::ate);
            if (!f)
                throw std::runtime_error("cannot open mechanism artifact: " + path.string());
            const auto end = f.tellg();
            if (end < 0 || static_cast<std::uintmax_t>(end) > max_bytes)
                throw std::runtime_error("mechanism artifact exceeds read limit");
            std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
            f.seekg(0);
            f.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
            if (!f)
                throw std::runtime_error("failed reading mechanism artifact: " + path.string());
            return bytes;
        }
        std::string read_text_file(const std::filesystem::path &path, std::size_t max_bytes)
        {
            const auto bytes = read_binary_file(path, max_bytes);
            return {bytes.begin(), bytes.end()};
        }
        void write_text_file(const std::filesystem::path &path, const std::string &bytes)
        {
            std::ofstream f(path, std::ios::binary | std::ios::trunc);
            if (!f)
                throw std::runtime_error("cannot open mechanism JSON for writing: " + path.string());
            f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
            f.flush();
            if (!f)
                throw std::runtime_error("failed writing mechanism JSON: " + path.string());
        }
        std::string run_receipt_json(const RunOptions &o, const RunReceipt &r)
        {
            std::string out = "{\"format\":\"zkgat_mechanism_result_v1\",\"mechanism\":\"" + to_string(o.mechanism) + "\",\"variant\":\"" +
                              to_string(o.variant) + "\",\"workload\":{\"nodes\":" + std::to_string(o.nodes) + ",\"edges\":" + std::to_string(o.edges) +
                              ",\"width\":" + std::to_string(o.width) + ",\"graphs\":" + std::to_string(o.graphs) + ",\"seed\":" + std::to_string(o.seed) +
                              "},\"backend\":{\"prover\":\"" + to_string(o.backend) + "\",\"device\":";
            out += o.backend == Backend::Cuda ? std::to_string(r.device) : "null";
            out += ",\"verifier\":\"cpu\",\"actual_cuda\":";
            out += r.actual_cuda ? "true" : "false";
            out += ",\"cuda_ntt_forced\":";
            out += r.cuda_ntt_forced ? "true" : "false";
            out += ",\"device_name\":";
            append_string(out, r.device_name);
            out += ",\"cuda_kernel_invocations_delta\":" + std::to_string(r.cuda_kernel_invocations_delta) + ",\"cuda_stats_delta\":{";
            out += "\"h2d_bytes\":" + std::to_string(r.cuda_h2d_bytes_delta) + ",\"d2h_bytes\":" + std::to_string(r.cuda_d2h_bytes_delta) +
                   ",\"h2d_operations\":" + std::to_string(r.cuda_h2d_operations_delta) + ",\"d2h_operations\":" +
                   std::to_string(r.cuda_d2h_operations_delta) + ",\"stream_synchronizations\":" +
                   std::to_string(r.cuda_stream_synchronizations_delta) + ",\"ntt_calls\":" + std::to_string(r.cuda_ntt_calls_delta) +
                   ",\"intt_calls\":" + std::to_string(r.cuda_intt_calls_delta) + ",\"quotient_calls\":" +
                   std::to_string(r.cuda_quotient_calls_delta) + ",\"t_edge_calls\":" + std::to_string(r.cuda_t_edge_calls_delta) +
                   ",\"t_fh_calls\":" + std::to_string(r.cuda_t_fh_calls_delta) + ",\"opening_fold_calls\":" +
                   std::to_string(r.cuda_opening_fold_calls_delta) + ",\"msm_single_calls\":" +
                   std::to_string(r.cuda_msm_single_calls_delta) + ",\"msm_batch_calls\":" +
                   std::to_string(r.cuda_msm_batch_calls_delta) + ",\"resident_hits\":" +
                   std::to_string(r.cuda_resident_hits_delta) + ",\"resident_misses\":" +
                   std::to_string(r.cuda_resident_misses_delta) + "}},\"security\":" + r.security_budget_ledger_json + ",\"srs\":{\"id\":\"" + r.srs_id + "\",\"manifest_sha256\":\"" +
                   r.srs_manifest_sha256 + "\",\"protocol_schema_id\":\"" +
                   r.parent_protocol_schema_id + "\"},\"pair_input_commitment_sha256\":\"" + r.pair_input_commitment_sha256 + "\",\"statement_sha256\":\"" +
                   r.statement_sha256 + "\",\"proof\":{\"path\":\"" + o.proof_out.string() + "\",\"sha256\":\"" + r.proof_sha256 + "\",\"size_bytes\":" +
                   std::to_string(r.proof_size_bytes) + "},\"metrics\":{\"commitment_time_ms\":" + std::to_string(r.commitment_time_ms) + ",\"prove_time_ms\":" +
                   std::to_string(r.prove_time_ms) + "},\"cuda_warmup_time_ms\":" + std::to_string(r.cuda_warmup_time_ms) +
                   ",\"cuda_warmup_kernel_invocations\":" + std::to_string(r.cuda_warmup_kernel_invocations) + ",\"dataset\":\"" +
                   (r.real_input ? r.real_input_dataset : std::string("synthetic")) + "\",\"real_input\":" +
                   (r.real_input ? ("{\"schema\":\"zkgat_real_mechanism_input_v1\",\"dataset\":\"" + r.real_input_dataset +
                                    "\",\"input_preprocess_id\":\"" + r.real_input_preprocess_id + "\",\"bundle_sha256\":\"" +
                                    r.real_input_bundle_sha256 + "\",\"witness_sha256\":\"" + r.real_input_witness_sha256 +
                                    "\",\"metadata_sha256\":\"" + r.real_input_metadata_sha256 +
                                    "\",\"checkpoint_bundle_sha256\":\"" + r.real_input_checkpoint_bundle_sha256 +
                                    "\",\"source_manifest_sha256\":\"" + r.real_input_source_manifest_sha256 +
                                    "\",\"pair_input_commitment_sha256\":\"" + r.pair_input_commitment_sha256 + "\",\"layer\":" +
                                    std::to_string(r.real_input_layer) + ",\"head\":" + std::to_string(r.real_input_head) +
                                    ",\"formal_head_width\":" + std::to_string(r.real_input_formal_head_width) +
                                    ",\"selected_width\":" + std::to_string(r.real_input_selected_width) + ",\"actual\":{\"N\":" +
                                    std::to_string(r.real_input_nodes) + ",\"E\":" + std::to_string(r.real_input_edges) +
                                    ",\"G\":" + std::to_string(r.real_input_graphs) + "}}")
                            : "null") + ",\"timing_schema\":\"mechanism_commitment_disjoint_verify_parse_accept_v2\"}";
            return out;
        }
        std::string verify_receipt_json(const VerifyReceipt &r)
        {
            return "{\"format\":\"zkgat_mechanism_result_v1\",\"status\":\"VERIFY_OK\",\"proof_sha256\":\"" + r.proof_sha256 + "\",\"statement_sha256\":\"" +
                   r.statement_sha256 + "\",\"srs\":{\"id\":\"" + r.srs_id + "\",\"manifest_sha256\":\"" + r.srs_manifest_sha256 +
                   "\"},\"verifier_backend\":\"cpu\",\"verifier_cuda_kernel_invocations_delta\":" +
                   std::to_string(r.verifier_cuda_kernel_invocations_delta) + ",\"security\":" + r.security_budget_ledger_json +
                   ",\"metrics\":{\"verify_time_ms\":" + std::to_string(r.verify_time_ms) +
                   "},\"timing_schema\":\"mechanism_commitment_disjoint_verify_parse_accept_v2\"}";
        }
    }
}
