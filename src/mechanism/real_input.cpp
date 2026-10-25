#include "real_input.hpp"

#include "gatzk/data/loader.hpp"
#include "gatzk/model/gat.hpp"
#include "gatzk/protocol/schema.hpp"
#include "gatzk/protocol/trace.hpp"
#include "gatzk/util/config.hpp"
#include "codec.hpp"

#include <openssl/sha.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace gatzk::mechanism
{
    namespace
    {
        using protocol::Layer0Head0MechanismWitness;

        void require(bool condition, const std::string &message)
        {
            if (!condition)
                throw std::runtime_error(message);
        }

        void append_u32(std::vector<std::uint8_t> &out, std::uint32_t value)
        {
            for (unsigned i = 0; i < 4; ++i)
                out.push_back(static_cast<std::uint8_t>(value >> (i * 8U)));
        }

        void append_u64(std::vector<std::uint8_t> &out, std::uint64_t value)
        {
            for (unsigned i = 0; i < 8; ++i)
                out.push_back(static_cast<std::uint8_t>(value >> (i * 8U)));
        }

        void append_i64(std::vector<std::uint8_t> &out, std::int64_t value)
        {
            append_u64(out, static_cast<std::uint64_t>(value));
        }

        void append_string(std::vector<std::uint8_t> &out, const std::string &value)
        {
            require(value.size() <= std::numeric_limits<std::uint32_t>::max(), "real input string exceeds the binary format limit");
            append_u32(out, static_cast<std::uint32_t>(value.size()));
            out.insert(out.end(), value.begin(), value.end());
        }

        void append_field(std::vector<std::uint8_t> &out, const algebra::FieldElement &value)
        {
            const auto offset = out.size();
            out.resize(offset + 32);
            value.write_little_endian(out.data() + offset, 32);
        }

        std::uint32_t read_u32(const std::vector<std::uint8_t> &bytes, std::size_t &offset)
        {
            if (offset > bytes.size() || bytes.size() - offset < 4)
                throw std::runtime_error("truncated real mechanism input bundle");
            std::uint32_t out = 0;
            for (unsigned i = 0; i < 4; ++i)
                out |= static_cast<std::uint32_t>(bytes[offset++]) << (i * 8U);
            return out;
        }

        std::uint64_t read_u64(const std::vector<std::uint8_t> &bytes, std::size_t &offset)
        {
            if (offset > bytes.size() || bytes.size() - offset < 8)
                throw std::runtime_error("truncated real mechanism input bundle");
            std::uint64_t out = 0;
            for (unsigned i = 0; i < 8; ++i)
                out |= static_cast<std::uint64_t>(bytes[offset++]) << (i * 8U);
            return out;
        }

        std::int64_t read_i64(const std::vector<std::uint8_t> &bytes, std::size_t &offset)
        {
            return static_cast<std::int64_t>(read_u64(bytes, offset));
        }

        std::string read_string(const std::vector<std::uint8_t> &bytes, std::size_t &offset)
        {
            const auto size = read_u32(bytes, offset);
            if (offset > bytes.size() || bytes.size() - offset < size)
                throw std::runtime_error("truncated real mechanism input string");
            std::string out(reinterpret_cast<const char *>(bytes.data() + offset), size);
            offset += size;
            return out;
        }

        algebra::FieldElement read_field(const std::vector<std::uint8_t> &bytes, std::size_t &offset)
        {
            if (offset > bytes.size() || bytes.size() - offset < 32)
                throw std::runtime_error("truncated real mechanism field element");
            auto value = algebra::FieldElement::from_canonical_little_endian(bytes.data() + offset, 32);
            offset += 32;
            return value;
        }

        std::size_t checked_size(std::uint64_t value, const char *label)
        {
            if (value > std::numeric_limits<std::size_t>::max())
                throw std::runtime_error(std::string("real mechanism input ") + label + " exceeds size_t");
            return static_cast<std::size_t>(value);
        }

        std::string sha256_bytes(const std::string &bytes)
        {
            return internal::sha256_hex(bytes);
        }

        std::string sha256_file(const std::filesystem::path &path)
        {
            std::ifstream in(path, std::ios::binary);
            if (!in)
                throw std::runtime_error("cannot open input for SHA256: " + path.string());
            SHA256_CTX ctx;
            SHA256_Init(&ctx);
            std::array<char, 1U << 16U> buffer{};
            while (in)
            {
                in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
                const auto count = in.gcount();
                if (count > 0)
                    SHA256_Update(&ctx, buffer.data(), static_cast<std::size_t>(count));
            }
            if (!in.eof())
                throw std::runtime_error("failed reading input for SHA256: " + path.string());
            std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
            SHA256_Final(digest.data(), &ctx);
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

        std::filesystem::path resolve_under_project(const std::filesystem::path &project, const std::string &value)
        {
            const std::filesystem::path path(value);
            return std::filesystem::weakly_canonical(path.is_absolute() ? path : project / path);
        }

        struct HashedFile
        {
            std::string path;
            std::string sha256;
        };

        std::vector<HashedFile> hash_directory_files(const std::filesystem::path &root)
        {
            std::error_code error;
            const auto resolved_root = std::filesystem::canonical(root, error);
            if (error || resolved_root.empty() || !std::filesystem::is_directory(resolved_root))
                throw std::runtime_error("required real input source directory does not exist: " + root.string());
            std::set<std::filesystem::path> resolved_paths;
            std::set<std::filesystem::path> visited_directories;
            std::function<void(const std::filesystem::path &)> visit_directory = [&](const std::filesystem::path &directory)
            {
                std::error_code visit_error;
                const auto canonical_directory = std::filesystem::canonical(directory, visit_error);
                if (visit_error)
                    throw std::runtime_error("cannot resolve real input source directory " + directory.string() + ": " + visit_error.message());
                if (!visited_directories.insert(canonical_directory).second)
                    return;
                std::filesystem::directory_iterator it(canonical_directory, std::filesystem::directory_options::none, visit_error), end;
                if (visit_error)
                    throw std::runtime_error("cannot enumerate real input source directory " + canonical_directory.string() + ": " + visit_error.message());
                while (it != end)
                {
                    const auto entry_path = it->path();
                    std::error_code entry_error;
                    const auto resolved = std::filesystem::canonical(entry_path, entry_error);
                    if (entry_error)
                        throw std::runtime_error("cannot resolve real input source entry " + entry_path.string() + ": " + entry_error.message());
                    const auto target_status = std::filesystem::status(resolved, entry_error);
                    if (entry_error)
                        throw std::runtime_error("cannot inspect real input source target " + resolved.string() + ": " + entry_error.message());
                    if (std::filesystem::is_directory(target_status))
                        visit_directory(resolved);
                    else if (std::filesystem::is_regular_file(target_status))
                        resolved_paths.insert(resolved);
                    it.increment(visit_error);
                    if (visit_error)
                        throw std::runtime_error("cannot continue real input source inventory under " + canonical_directory.string() + ": " + visit_error.message());
                }
            };
            visit_directory(resolved_root);
            std::vector<HashedFile> out;
            out.reserve(resolved_paths.size());
            for (const auto &path : resolved_paths)
                out.push_back({path.string(), sha256_file(path)});
            return out;
        }

        std::string hash_inventory(const std::vector<HashedFile> &files)
        {
            std::string canonical;
            for (const auto &file : files)
            {
                canonical += file.path;
                canonical.push_back('\0');
                canonical += file.sha256;
                canonical.push_back('\n');
            }
            return sha256_bytes(canonical);
        }

        std::string json_escape(const std::string &value)
        {
            std::string out;
            for (const unsigned char c : value)
            {
                switch (c)
                {
                    case '"': out += "\\\""; break;
                    case '\\': out += "\\\\"; break;
                    case '\b': out += "\\b"; break;
                    case '\f': out += "\\f"; break;
                    case '\n': out += "\\n"; break;
                    case '\r': out += "\\r"; break;
                    case '\t': out += "\\t"; break;
                    default:
                        if (c < 0x20)
                        {
                            std::ostringstream hex;
                            hex << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned>(c);
                            out += hex.str();
                        }
                        else
                            out.push_back(static_cast<char>(c));
                }
            }
            return out;
        }

        void append_files_json(std::ostringstream &out, const std::vector<HashedFile> &files)
        {
            out << '[';
            for (std::size_t i = 0; i < files.size(); ++i)
            {
                if (i != 0)
                    out << ',';
                out << "{\"path\":\"" << json_escape(files[i].path) << "\",\"sha256\":\"" << files[i].sha256 << "\"}";
            }
            out << ']';
        }

        struct NeighborStats
        {
            std::size_t eligible_nodes = 0;
            std::size_t minimum = 0;
            double median = 0;
            std::size_t maximum = 0;
        };

        NeighborStats neighbor_stats(const data::LocalGraph &local, std::size_t graph)
        {
            const auto node_begin = local.node_ptr.at(graph);
            const auto node_end = local.node_ptr.at(graph + 1);
            std::vector<std::set<std::size_t>> incoming(node_end - node_begin);
            for (const auto &edge : local.edges)
            {
                if (edge.graph_id != graph || edge.src == edge.dst)
                    continue;
                if (edge.dst < node_begin || edge.dst >= node_end || edge.src < node_begin || edge.src >= node_end)
                    throw std::runtime_error("normalized graph edge crosses a graph boundary during neighbor-stat preparation");
                incoming[edge.dst - node_begin].insert(edge.src);
            }
            std::vector<std::size_t> counts;
            counts.reserve(incoming.size());
            NeighborStats out;
            for (const auto &neighbors : incoming)
            {
                counts.push_back(neighbors.size());
                if (neighbors.size() >= 2)
                    ++out.eligible_nodes;
            }
            if (counts.empty())
                return out;
            std::sort(counts.begin(), counts.end());
            out.minimum = counts.front();
            out.maximum = counts.back();
            if (counts.size() % 2 == 0)
                out.median = (static_cast<double>(counts[counts.size() / 2 - 1]) + static_cast<double>(counts[counts.size() / 2])) / 2.0;
            else
                out.median = static_cast<double>(counts[counts.size() / 2]);
            return out;
        }

        std::string scales_json(const Layer0Head0MechanismWitness &w)
        {
            return "{\"S_input\":" + std::to_string(w.s_input) + ",\"S_weight\":" + std::to_string(w.s_w_hidden) +
                   ",\"S_hidden\":" + std::to_string(w.s_h_prime) + ",\"S_exp\":" + std::to_string(w.s_exp) +
                   ",\"S_delta\":" + std::to_string(w.s_delta) + ",\"S_a_src\":" + std::to_string(w.s_a_src_hidden) +
                   ",\"S_a_dst\":" + std::to_string(w.s_a_dst_hidden) + ",\"S_agg_pre\":" + std::to_string(w.s_agg_pre) + '}';
        }
    }

    namespace internal
    {
        const RealDatasetCatalog *registered_real_dataset_catalog(std::string_view dataset) noexcept
        {
            static constexpr std::array<RealDatasetCatalog, 5> catalogs{{
                {"cora", "row_normalize", "cora_formal_srs", 4096, 32768},
                {"citeseer", "row_normalize", "citeseer_formal_srs", 4096, 32768},
                {"pubmed", "row_normalize", "pubmed_formal_srs", 32768, 262144},
                {"ppi", "row_normalize", "ppi_formal_srs", 65536, 8388608},
                {"ogbn_arxiv", "ogbn_arxiv_raw_float32", "ogbn_arxiv_formal_srs", 262144, 16777216},
            }};
            for (const auto &catalog : catalogs)
                if (catalog.dataset == dataset)
                    return &catalog;
            return nullptr;
        }

        bool valid_real_graph_count(std::string_view dataset, std::size_t graphs) noexcept
        {
            if (registered_real_dataset_catalog(dataset) == nullptr)
                return false;
            if (dataset == "ppi")
                return graphs == 1 || graphs == 295;
            return graphs == 1;
        }

        bool valid_real_component_selection(std::string_view dataset, const std::vector<std::size_t> &components) noexcept
        {
            if (components.empty() || !std::is_sorted(components.begin(), components.end()) ||
                std::adjacent_find(components.begin(), components.end()) != components.end())
                return false;
            if (dataset != "ppi")
                return registered_real_dataset_catalog(dataset) != nullptr && components.size() == 1 && components.front() == 0;
            if (components.size() == 1 && components.front() == 0)
                return true;
            constexpr std::size_t kPpiGraphCount = 295;
            if (components.size() != kPpiGraphCount)
                return false;
            for (std::size_t graph = 0; graph < kPpiGraphCount; ++graph)
                if (components[graph] != graph)
                    return false;
            return true;
        }

        std::vector<std::uint8_t> encode_real_mechanism_input(const RealMechanismInput &input)
        {
            const auto &w = input.witness;
            require(w.width == 8 && w.node_count > 0 && w.edge_count >= w.node_count && w.node_ptr.size() >= 2 &&
                        w.edge_ptr.size() == w.node_ptr.size() && w.edge_src_indices.size() == w.edge_count && w.edge_dst_indices.size() == w.edge_count,
                    "real mechanism input has an invalid canonical shape");
            require(w.h_prime_raw.size() == w.node_count && w.h_prime.size() == w.node_count && w.alpha.size() == w.edge_count &&
                        w.u_raw.size() == w.edge_count && w.sum_raw.size() == w.node_count &&
                        w.weighted_edge_features.size() == w.edge_count && w.aggregate_pre_activation.size() == w.node_count,
                    "real mechanism witness vectors have inconsistent lengths");
            const auto graphs = w.node_ptr.size() - 1;
            require(w.node_ptr.front() == 0 && w.node_ptr.back() == w.node_count && w.edge_ptr.front() == 0 && w.edge_ptr.back() == w.edge_count,
                    "real mechanism witness graph pointers do not span its node and edge arrays");
            std::vector<algebra::FieldElement> sum_check(w.node_count, algebra::FieldElement::zero());
            std::vector<std::vector<algebra::FieldElement>> aggregate_check(
                w.node_count, std::vector<algebra::FieldElement>(w.width, algebra::FieldElement::zero()));
            std::vector<__int128> raw_sum_check(w.node_count, 0);
            std::vector<std::size_t> edge_graph(w.edge_count);
            std::vector<std::size_t> incoming_edge_count(w.node_count, 0);
            std::vector<bool> has_self_loop(w.node_count, false);
            std::set<std::tuple<std::size_t, std::size_t, std::size_t>> unique_edges;
            for (std::size_t graph = 0; graph < graphs; ++graph)
            {
                require(w.node_ptr[graph] < w.node_ptr[graph + 1] && w.node_ptr[graph + 1] <= w.node_count &&
                            w.edge_ptr[graph] < w.edge_ptr[graph + 1] && w.edge_ptr[graph + 1] <= w.edge_count,
                        "real mechanism witness has an empty graph boundary");
                for (std::size_t edge = w.edge_ptr[graph]; edge < w.edge_ptr[graph + 1]; ++edge)
                    edge_graph[edge] = graph;
            }
            for (std::size_t node = 0; node < w.node_count; ++node)
            {
                require(w.h_prime_raw[node].size() == w.width && w.h_prime[node].size() == w.width &&
                            w.aggregate_pre_activation[node].size() == w.width,
                        "real mechanism node feature width mismatch");
                for (std::size_t lane = 0; lane < w.width; ++lane)
                    require(algebra::FieldElement::from_signed(w.h_prime_raw[node][lane]) == w.h_prime[node][lane],
                            "real mechanism H-prime field does not match its quantized integer");
                require(algebra::FieldElement::from_signed(w.sum_raw[node]) == w.sum[node],
                        "real mechanism Sum field does not match its quantized integer");
            }
            for (std::size_t edge = 0; edge < w.edge_count; ++edge)
            {
                const auto graph = edge_graph[edge];
                const auto src = w.edge_src_indices[edge];
                const auto dst = w.edge_dst_indices[edge];
                require(src >= w.node_ptr[graph] && src < w.node_ptr[graph + 1] && dst >= w.node_ptr[graph] && dst < w.node_ptr[graph + 1],
                        "real mechanism witness edge crosses a graph boundary");
                if (edge != 0)
                    require(std::tie(edge_graph[edge - 1], w.edge_dst_indices[edge - 1]) <= std::tie(graph, dst),
                            "real mechanism topology is not in canonical graph/destination group order");
                require(unique_edges.emplace(graph, src, dst).second, "real mechanism topology contains a duplicate edge");
                ++incoming_edge_count[dst];
                if (src == dst)
                    has_self_loop[src] = true;
                require(w.weighted_edge_features[edge].size() == w.width &&
                            w.u_raw[edge] > 0 && algebra::FieldElement::from_signed(w.u_raw[edge]) == w.u[edge],
                        "real mechanism edge feature or U integer/field identity is invalid");
                sum_check[dst] += w.u[edge];
                raw_sum_check[dst] += w.u_raw[edge];
                require(w.alpha[edge] == w.u[edge] * w.sum[dst].inv(),
                        "real mechanism normalized attention weight is inconsistent with U/Sum");
                for (std::size_t lane = 0; lane < w.width; ++lane)
                {
                    const auto value = w.alpha[edge] * w.h_prime[src][lane];
                    require(value == w.weighted_edge_features[edge][lane],
                            "real mechanism weighted feature is not alpha times the gathered H-prime coordinate");
                    aggregate_check[dst][lane] += value;
                }
            }
            for (std::size_t node = 0; node < w.node_count; ++node)
            {
                require(incoming_edge_count[node] != 0 && has_self_loop[node],
                        "real mechanism node lacks an incoming edge or its required self-loop");
                require(raw_sum_check[node] > 0 && raw_sum_check[node] <= std::numeric_limits<std::int64_t>::max() &&
                            static_cast<std::int64_t>(raw_sum_check[node]) == w.sum_raw[node],
                        "real mechanism integer Sum is inconsistent with its U edge integers");
            }
            require(sum_check == w.sum && aggregate_check == w.aggregate_pre_activation,
                    "real mechanism denominator or pre-activation aggregate is inconsistent with the edge witness");
            std::vector<std::uint8_t> out{'G','Z','K','R','E','A','L','1'};
            append_u32(out, 1);
            for (const auto *text : {&input.dataset, &input.input_preprocess_id, &input.model_arch_id, &input.model_param_id,
                                     &input.quant_cfg_id, &input.static_table_id, &input.config_sha256, &input.checkpoint_bundle_sha256,
                                     &input.source_manifest_sha256})
                append_string(out, *text);
            append_u64(out, w.node_count);
            append_u64(out, w.edge_count);
            append_u64(out, graphs);
            append_u64(out, w.width);
            for (const auto *values : {&w.node_ptr, &w.edge_ptr})
            {
                for (const auto value : *values)
                    append_u64(out, value);
            }
            append_u64(out, w.edge_count);
            for (std::size_t i = 0; i < w.edge_count; ++i)
            {
                append_u64(out, i < w.edge_src_indices.size() ? w.edge_src_indices[i] : 0);
                append_u64(out, i < w.edge_dst_indices.size() ? w.edge_dst_indices[i] : 0);
                // The per-edge graph ordinal is unambiguously derived from edge_ptr and is written for cross-checking.
                const auto it = std::upper_bound(w.edge_ptr.begin(), w.edge_ptr.end(), i);
                append_u64(out, static_cast<std::uint64_t>(std::distance(w.edge_ptr.begin(), it) - 1));
            }
            append_u64(out, w.h_prime_raw.size());
            for (const auto &row : w.h_prime_raw)
            {
                require(row.size() == w.width, "real mechanism raw H-prime row width mismatch");
                for (const auto value : row)
                    append_i64(out, value);
            }
            append_u64(out, w.u_raw.size());
            for (const auto value : w.u_raw)
                append_i64(out, value);
            append_u64(out, w.sum_raw.size());
            for (const auto value : w.sum_raw)
                append_i64(out, value);
            for (const auto value : {w.s_input, w.s_w_hidden, w.s_h_prime, w.s_a_src_hidden, w.s_a_dst_hidden, w.s_delta, w.s_exp, w.s_agg_pre})
                append_u64(out, value);
            append_u64(out, w.alpha.size());
            for (const auto &value : w.alpha)
                append_field(out, value);
            append_u64(out, w.weighted_edge_features.size());
            for (const auto &row : w.weighted_edge_features)
            {
                require(row.size() == w.width, "real mechanism weighted-edge feature row width mismatch");
                for (const auto &value : row)
                    append_field(out, value);
            }
            append_u64(out, w.aggregate_pre_activation.size());
            for (const auto &row : w.aggregate_pre_activation)
            {
                require(row.size() == w.width, "real mechanism pre-activation row width mismatch");
                for (const auto &value : row)
                    append_field(out, value);
            }
            return out;
        }

        RealMechanismInput read_real_mechanism_input(const std::filesystem::path &path)
        {
            constexpr std::size_t kMaximumBundleBytes = 1U << 30;
            const auto bytes = read_binary_file(path, kMaximumBundleBytes);
            std::size_t offset = 0;
            static constexpr std::array<std::uint8_t, 8> magic{'G','Z','K','R','E','A','L','1'};
            if (bytes.size() < magic.size() || !std::equal(magic.begin(), magic.end(), bytes.begin()))
                throw std::runtime_error("real mechanism input bundle magic is invalid");
            offset += magic.size();
            if (read_u32(bytes, offset) != 1)
                throw std::runtime_error("unsupported real mechanism input bundle version");
            RealMechanismInput out;
            out.dataset = read_string(bytes, offset);
            out.input_preprocess_id = read_string(bytes, offset);
            out.model_arch_id = read_string(bytes, offset);
            out.model_param_id = read_string(bytes, offset);
            out.quant_cfg_id = read_string(bytes, offset);
            out.static_table_id = read_string(bytes, offset);
            out.config_sha256 = read_string(bytes, offset);
            out.checkpoint_bundle_sha256 = read_string(bytes, offset);
            out.source_manifest_sha256 = read_string(bytes, offset);
            auto &w = out.witness;
            w.node_count = checked_size(read_u64(bytes, offset), "node count");
            w.edge_count = checked_size(read_u64(bytes, offset), "edge count");
            const auto graphs = checked_size(read_u64(bytes, offset), "graph count");
            w.width = checked_size(read_u64(bytes, offset), "width");
            if (w.node_count == 0 || w.edge_count < w.node_count || graphs == 0 || graphs > w.node_count || w.width != 8 ||
                w.node_count > (1U << 20) || w.edge_count > (1U << 24))
                throw std::runtime_error("real mechanism input workload is outside supported canonical bounds");
            w.node_ptr.reserve(graphs + 1);
            w.edge_ptr.reserve(graphs + 1);
            for (std::size_t i = 0; i <= graphs; ++i)
                w.node_ptr.push_back(checked_size(read_u64(bytes, offset), "node pointer"));
            for (std::size_t i = 0; i <= graphs; ++i)
                w.edge_ptr.push_back(checked_size(read_u64(bytes, offset), "edge pointer"));
            if (read_u64(bytes, offset) != w.edge_count)
                throw std::runtime_error("real mechanism input edge count header mismatch");
            w.edge_src_indices.reserve(w.edge_count);
            w.edge_dst_indices.reserve(w.edge_count);
            std::vector<std::size_t> edge_graphs;
            edge_graphs.reserve(w.edge_count);
            for (std::size_t i = 0; i < w.edge_count; ++i)
            {
                w.edge_src_indices.push_back(checked_size(read_u64(bytes, offset), "edge source"));
                w.edge_dst_indices.push_back(checked_size(read_u64(bytes, offset), "edge destination"));
                edge_graphs.push_back(checked_size(read_u64(bytes, offset), "edge graph"));
            }
            if (read_u64(bytes, offset) != w.node_count)
                throw std::runtime_error("real mechanism input H-prime node count mismatch");
            w.h_prime_raw.assign(w.node_count, std::vector<std::int64_t>(w.width));
            w.h_prime.assign(w.node_count, std::vector<algebra::FieldElement>(w.width, algebra::FieldElement::zero()));
            for (std::size_t node = 0; node < w.node_count; ++node)
                for (std::size_t lane = 0; lane < w.width; ++lane)
                {
                    const auto raw = read_i64(bytes, offset);
                    w.h_prime_raw[node][lane] = raw;
                    w.h_prime[node][lane] = algebra::FieldElement::from_signed(raw);
                }
            if (read_u64(bytes, offset) != w.edge_count)
                throw std::runtime_error("real mechanism input U edge count mismatch");
            w.u_raw.resize(w.edge_count);
            w.u.resize(w.edge_count, algebra::FieldElement::zero());
            for (std::size_t i = 0; i < w.edge_count; ++i)
            {
                w.u_raw[i] = read_i64(bytes, offset);
                w.u[i] = algebra::FieldElement::from_signed(w.u_raw[i]);
            }
            if (read_u64(bytes, offset) != w.node_count)
                throw std::runtime_error("real mechanism input Sum node count mismatch");
            w.sum_raw.resize(w.node_count);
            w.sum.resize(w.node_count, algebra::FieldElement::zero());
            for (std::size_t i = 0; i < w.node_count; ++i)
            {
                w.sum_raw[i] = read_i64(bytes, offset);
                w.sum[i] = algebra::FieldElement::from_signed(w.sum_raw[i]);
            }
            w.s_input = read_u64(bytes, offset);
            w.s_w_hidden = read_u64(bytes, offset);
            w.s_h_prime = read_u64(bytes, offset);
            w.s_a_src_hidden = read_u64(bytes, offset);
            w.s_a_dst_hidden = read_u64(bytes, offset);
            w.s_delta = read_u64(bytes, offset);
            w.s_exp = read_u64(bytes, offset);
            w.s_agg_pre = read_u64(bytes, offset);
            if (read_u64(bytes, offset) != w.edge_count)
                throw std::runtime_error("real mechanism input alpha edge count mismatch");
            w.alpha.reserve(w.edge_count);
            for (std::size_t i = 0; i < w.edge_count; ++i)
                w.alpha.push_back(read_field(bytes, offset));
            if (read_u64(bytes, offset) != w.edge_count)
                throw std::runtime_error("real mechanism input weighted-feature edge count mismatch");
            w.weighted_edge_features.assign(w.edge_count, std::vector<algebra::FieldElement>(w.width, algebra::FieldElement::zero()));
            for (std::size_t edge = 0; edge < w.edge_count; ++edge)
                for (std::size_t lane = 0; lane < w.width; ++lane)
                    w.weighted_edge_features[edge][lane] = read_field(bytes, offset);
            if (read_u64(bytes, offset) != w.node_count)
                throw std::runtime_error("real mechanism input aggregate node count mismatch");
            w.aggregate_pre_activation.assign(w.node_count, std::vector<algebra::FieldElement>(w.width, algebra::FieldElement::zero()));
            for (std::size_t node = 0; node < w.node_count; ++node)
                for (std::size_t lane = 0; lane < w.width; ++lane)
                    w.aggregate_pre_activation[node][lane] = read_field(bytes, offset);
            if (offset != bytes.size())
                throw std::runtime_error("real mechanism input bundle has trailing bytes");
            out.bundle_sha256 = sha256_hex(std::string_view(reinterpret_cast<const char *>(bytes.data()), bytes.size()));
            out.metadata_sha256.clear();
            out.witness_sha256 = real_mechanism_witness_digest(out);

            if (w.node_ptr.size() != graphs + 1 || w.edge_ptr.size() != graphs + 1 || w.node_ptr.front() != 0 || w.node_ptr.back() != w.node_count ||
                w.edge_ptr.front() != 0 || w.edge_ptr.back() != w.edge_count)
                throw std::runtime_error("real mechanism input graph boundary endpoints are invalid");
            for (std::size_t graph = 0; graph < graphs; ++graph)
                if (w.node_ptr[graph] >= w.node_ptr[graph + 1] || w.edge_ptr[graph] >= w.edge_ptr[graph + 1])
                    throw std::runtime_error("real mechanism input contains an empty graph boundary");
            std::vector<algebra::FieldElement> sum_check(w.node_count, algebra::FieldElement::zero());
            std::vector<std::vector<algebra::FieldElement>> aggregate_check(w.node_count,
                std::vector<algebra::FieldElement>(w.width, algebra::FieldElement::zero()));
            std::vector<__int128> raw_sum_check(w.node_count, 0);
            std::vector<std::size_t> incoming_edge_count(w.node_count, 0);
            std::vector<bool> has_self_loop(w.node_count, false);
            std::set<std::tuple<std::size_t, std::size_t, std::size_t>> unique_edges;
            for (std::size_t edge = 0; edge < w.edge_count; ++edge)
            {
                const auto graph = edge_graphs[edge];
                const auto src = w.edge_src_indices[edge];
                const auto dst = w.edge_dst_indices[edge];
                if (graph >= graphs || edge < w.edge_ptr[graph] || edge >= w.edge_ptr[graph + 1] || src < w.node_ptr[graph] ||
                    src >= w.node_ptr[graph + 1] || dst < w.node_ptr[graph] || dst >= w.node_ptr[graph + 1])
                    throw std::runtime_error("real mechanism input edge crosses a graph boundary");
                if (edge != 0 && std::tie(edge_graphs[edge - 1], w.edge_dst_indices[edge - 1]) >
                                     std::tie(graph, w.edge_dst_indices[edge]))
                    throw std::runtime_error("real mechanism input topology is not in canonical graph/destination group order");
                if (!unique_edges.emplace(graph, src, dst).second)
                    throw std::runtime_error("real mechanism input topology contains a duplicate edge");
                ++incoming_edge_count[dst];
                if (src == dst)
                    has_self_loop[src] = true;
                if (w.u_raw[edge] <= 0 || w.sum_raw[dst] <= 0 || w.alpha[edge] != w.u[edge] * w.sum[dst].inv())
                    throw std::runtime_error("real mechanism normalized attention weight is inconsistent with the positive U/Sum relation");
                sum_check[dst] += w.u[edge];
                raw_sum_check[dst] += w.u_raw[edge];
                for (std::size_t lane = 0; lane < w.width; ++lane)
                {
                    const auto expected = w.alpha[edge] * w.h_prime[src][lane];
                    if (expected != w.weighted_edge_features[edge][lane])
                        throw std::runtime_error("real mechanism input weighted edge feature is inconsistent with alpha times source H-prime");
                    aggregate_check[dst][lane] += expected;
                }
            }
            for (std::size_t node = 0; node < w.node_count; ++node)
            {
                if (incoming_edge_count[node] == 0 || !has_self_loop[node])
                    throw std::runtime_error("real mechanism input node lacks an incoming edge or its required self-loop");
                if (raw_sum_check[node] <= 0 || raw_sum_check[node] > std::numeric_limits<std::int64_t>::max() ||
                    static_cast<std::int64_t>(raw_sum_check[node]) != w.sum_raw[node])
                    throw std::runtime_error("real mechanism input integer Sum is inconsistent with the grouped U denominator");
            }
            if (sum_check != w.sum)
                throw std::runtime_error("real mechanism input finite-field Sum is inconsistent with the grouped U denominator");
            if (aggregate_check != w.aggregate_pre_activation)
                throw std::runtime_error("real mechanism input target differs from the unrounded weighted aggregate");
            return out;
        }

        std::string real_mechanism_witness_digest(const RealMechanismInput &input)
        {
            std::vector<std::uint8_t> bytes;
            append_string(bytes, "zkgat-real-mechanism-witness-v1");
            const auto &w = input.witness;
            for (const auto value : w.node_ptr)
                append_u64(bytes, value);
            for (const auto value : w.edge_ptr)
                append_u64(bytes, value);
            for (std::size_t edge = 0; edge < w.edge_count; ++edge)
            {
                append_u64(bytes, w.edge_src_indices[edge]);
                append_u64(bytes, w.edge_dst_indices[edge]);
                append_field(bytes, w.u[edge]);
                append_field(bytes, w.alpha[edge]);
                for (std::size_t lane = 0; lane < w.width; ++lane)
                    append_field(bytes, w.weighted_edge_features[edge][lane]);
            }
            for (std::size_t node = 0; node < w.node_count; ++node)
            {
                for (std::size_t lane = 0; lane < w.width; ++lane)
                    append_field(bytes, w.h_prime[node][lane]);
                append_field(bytes, w.sum[node]);
                for (std::size_t lane = 0; lane < w.width; ++lane)
                    append_field(bytes, w.aggregate_pre_activation[node][lane]);
            }
            return sha256_hex(std::string_view(reinterpret_cast<const char *>(bytes.data()), bytes.size()));
        }

        std::string real_mechanism_relation_digest(const RealMechanismInput &input, Mechanism mechanism, std::size_t width)
        {
            const auto &w = input.witness;
            if (width == 0 || width > w.width)
                throw std::runtime_error("mechanism width exceeds the real first-head witness width");
            std::vector<std::uint8_t> bytes;
            append_string(bytes, "zkgat-real-mechanism-pair-v1");
            append_string(bytes, to_string(mechanism));
            append_u64(bytes, w.node_count);
            append_u64(bytes, w.edge_count);
            append_u64(bytes, width);
            append_u64(bytes, w.node_ptr.size());
            for (const auto value : w.node_ptr)
                append_u64(bytes, value);
            append_u64(bytes, w.edge_ptr.size());
            for (const auto value : w.edge_ptr)
                append_u64(bytes, value);
            for (std::size_t edge = 0; edge < w.edge_count; ++edge)
            {
                append_u64(bytes, w.edge_src_indices[edge]);
                append_u64(bytes, w.edge_dst_indices[edge]);
            }
            if (mechanism == Mechanism::FeatureFolding)
            {
                for (std::size_t node = 0; node < w.node_count; ++node)
                    for (std::size_t lane = 0; lane < width; ++lane)
                        append_field(bytes, w.h_prime[node][lane]);
                for (std::size_t edge = 0; edge < w.edge_count; ++edge)
                    for (std::size_t lane = 0; lane < width; ++lane)
                        append_field(bytes, w.h_prime[w.edge_src_indices[edge]][lane]);
            }
            else
            {
                for (const auto &value : w.u)
                    append_field(bytes, value);
                for (const auto &value : w.sum)
                    append_field(bytes, value);
                for (const auto &value : w.alpha)
                    append_field(bytes, value);
                for (std::size_t edge = 0; edge < w.edge_count; ++edge)
                    for (std::size_t lane = 0; lane < width; ++lane)
                        append_field(bytes, w.weighted_edge_features[edge][lane]);
                for (std::size_t node = 0; node < w.node_count; ++node)
                    for (std::size_t lane = 0; lane < width; ++lane)
                        append_field(bytes, w.aggregate_pre_activation[node][lane]);
            }
            return sha256_hex(std::string_view(reinterpret_cast<const char *>(bytes.data()), bytes.size()));
        }

    }

    PrepareRealReceipt prepare_real_input(const PrepareRealOptions &options)
    {
        if (options.config_path.empty() || options.bundle_out.empty() || options.metadata_out.empty() || options.selected_components.empty())
            throw std::runtime_error("prepare-real requires config, bundle output, metadata output, and explicit component IDs");
        const auto config_path = std::filesystem::weakly_canonical(options.config_path);
        const auto config_sha = sha256_file(config_path);
        auto config = util::load_config(config_path.string());
        const auto project_root = std::filesystem::path(config.project_root);
        config.data_root = resolve_under_project(project_root, config.data_root).string();
        config.cache_root = resolve_under_project(project_root, config.cache_root).string();
        config.checkpoint_bundle = resolve_under_project(project_root, config.checkpoint_bundle).string();
        const auto *catalog = internal::registered_real_dataset_catalog(config.dataset);
        if (catalog == nullptr)
            throw std::runtime_error("prepare-real dataset is outside the registered trusted dataset catalog: " + config.dataset);
        if (config.input_preprocess_id != catalog->input_preprocess_id || config.srs_id != catalog->srs_id || config.srs_type != "trusted_srs")
            throw std::runtime_error("formal config preprocessing or trusted SRS identity differs from the registered dataset catalog");
        if (!internal::valid_real_component_selection(config.dataset, options.selected_components))
            throw std::runtime_error("real mechanism input requires one complete graph, PPI component 0, or all canonical PPI components 0..294");

        auto dataset = data::load_dataset(config);
        auto local = data::normalize_graph_input(dataset, config, options.selected_components);
        data::validate_input_feature_views(dataset, local, config.quantization.s_input, config.input_preprocess_id);
        auto model = model::load_checkpoint_bundle_parameters(config.checkpoint_bundle);
        if (model.model_arch_id != config.model_arch_id || model.model_param_id != config.model_param_id ||
            model.input_preprocess_id != config.input_preprocess_id || model.quant_cfg_id != config.quant_cfg_id ||
            model.static_table_id != config.static_table_id || model.quantization_scale_chain != config.quantization.canonical_serialization())
            throw std::runtime_error("formal GAT checkpoint identities differ from the selected config");
        if (model.hidden_layers.empty() || model.hidden_layers.front().heads.empty() ||
            model::attention_head_output_width(model.hidden_layers.front().heads.front()) != 8)
            throw std::runtime_error("formal layer-0 head-0 width is not the required eight coordinates");

        protocol::ProtocolContext context;
        context.config = config;
        context.dataset = dataset;
        context.local = local;
        context.model = std::move(model);
        const protocol::CanonicalStaticTableShapeInputs table_inputs{
            .range_bits = config.range_bits,
            .quantization = config.quantization,
            .batching_rule = config.batching_rule,
            .dataset_num_nodes = dataset.num_nodes,
            .dataset_num_features = dataset.num_features,
            .local_num_nodes = local.num_nodes,
            .local_edge_count = local.edges.size(),
        };
        context.tables = protocol::build_canonical_static_tables(table_inputs);
        auto witness = protocol::extract_layer0_head0_mechanism_witness(context);
        if (config.dataset == "ppi" && options.selected_components.size() == 1 && options.selected_components.front() == 0)
            for (std::size_t graph = 0; graph < local.graph_count; ++graph)
                if (neighbor_stats(local, graph).eligible_nodes == 0)
                    throw std::runtime_error("selected PPI component does not satisfy the fixed nontrivial-neighborhood rule");

        auto checkpoint_files = hash_directory_files(config.checkpoint_bundle);
        if (checkpoint_files.empty())
            throw std::runtime_error("formal GAT checkpoint bundle contains no regular files");
        std::string checkpoint_material;
        for (const auto &file : checkpoint_files)
        {
            checkpoint_material += file.path;
            checkpoint_material.push_back('\0');
            checkpoint_material += file.sha256;
            checkpoint_material.push_back('\n');
        }
        const auto checkpoint_sha = sha256_bytes(checkpoint_material);
        const auto cache_files = hash_directory_files(config.cache_root);
        const auto raw_files = hash_directory_files(config.data_root);
        std::map<std::string, HashedFile> unique_sources;
        for (const auto &file : cache_files)
            unique_sources.emplace(file.path, file);
        for (const auto &file : raw_files)
            unique_sources.emplace(file.path, file);
        std::vector<HashedFile> data_sources;
        for (const auto &[path, file] : unique_sources)
        {
            (void)path;
            data_sources.push_back(file);
        }
        if (data_sources.empty())
            throw std::runtime_error("formal dataset/cache source inventory is empty; provenance cannot be established");
        const auto source_manifest_sha = hash_inventory(data_sources);

        internal::RealMechanismInput input;
        input.witness = std::move(witness);
        input.dataset = config.dataset;
        input.input_preprocess_id = config.input_preprocess_id;
        input.model_arch_id = config.model_arch_id;
        input.model_param_id = config.model_param_id;
        input.quant_cfg_id = config.quant_cfg_id;
        input.static_table_id = config.static_table_id;
        input.config_sha256 = config_sha;
        input.checkpoint_bundle_sha256 = checkpoint_sha;
        input.source_manifest_sha256 = source_manifest_sha;
        const auto bundle_bytes = internal::encode_real_mechanism_input(input);
        input.bundle_sha256 = internal::sha256_hex(std::string_view(reinterpret_cast<const char *>(bundle_bytes.data()), bundle_bytes.size()));
        input.witness_sha256 = internal::real_mechanism_witness_digest(input);

        std::filesystem::create_directories(options.bundle_out.parent_path().empty() ? "." : options.bundle_out.parent_path());
        std::filesystem::create_directories(options.metadata_out.parent_path().empty() ? "." : options.metadata_out.parent_path());
        internal::write_binary_file(options.bundle_out, bundle_bytes);

        const auto n_domain = catalog->node_domain_size;
        const auto e_domain = catalog->edge_domain_size;
        if (e_domain > (std::numeric_limits<std::size_t>::max() - 14) / 2)
            throw std::runtime_error("registered edge domain overflows coefficient ledger capacity");
        const auto required_coefficients = 2 * e_domain + 14;
        std::ostringstream metadata;
        metadata << "{\"schema\":\"zkgat_real_mechanism_input_v1\",\"dataset\":\"" << json_escape(config.dataset)
                 << "\",\"config_path\":\"" << json_escape(config_path.string()) << "\",\"config_sha256\":\"" << config_sha
                 << "\",\"input_preprocess_id\":\"" << json_escape(config.input_preprocess_id) << "\",\"model_arch_id\":\""
                 << json_escape(config.model_arch_id) << "\",\"model_param_id\":\"" << json_escape(config.model_param_id)
                 << "\",\"quant_cfg_id\":\"" << json_escape(config.quant_cfg_id) << "\",\"static_table_id\":\""
                 << json_escape(config.static_table_id) << "\",\"checkpoint_bundle\":{\"path\":\"" << json_escape(config.checkpoint_bundle)
                 << "\",\"sha256\":\"" << checkpoint_sha << "\",\"files\":";
        append_files_json(metadata, checkpoint_files);
        metadata << "},\"data_sources\":";
        append_files_json(metadata, data_sources);
        metadata << ",\"bundle\":{\"path\":\"" << json_escape(std::filesystem::absolute(options.bundle_out).string())
                 << "\",\"sha256\":\"" << input.bundle_sha256 << "\"},\"witness_sha256\":\"" << input.witness_sha256
                 << "\",\"source_manifest_sha256\":\"" << source_manifest_sha << "\",\"layer\":0,\"head\":0,\"formal_head_width\":8,\"actual\":{\"N\":"
                 << input.witness.node_count << ",\"E\":" << input.witness.edge_count << ",\"G\":" << input.witness.node_ptr.size() - 1
                 << "},\"selected_components\":[";
        for (std::size_t graph = 0; graph < options.selected_components.size(); ++graph)
        {
            if (graph != 0)
                metadata << ',';
            const auto node_begin = input.witness.node_ptr[graph];
            const auto node_end = input.witness.node_ptr[graph + 1];
            const auto edge_begin = input.witness.edge_ptr[graph];
            const auto edge_end = input.witness.edge_ptr[graph + 1];
            const auto stats = neighbor_stats(local, graph);
            const auto abs_begin = *std::min_element(local.absolute_ids.begin() + node_begin, local.absolute_ids.begin() + node_end);
            const auto abs_end = *std::max_element(local.absolute_ids.begin() + node_begin, local.absolute_ids.begin() + node_end) + 1;
            metadata << "{\"canonical_id\":" << options.selected_components[graph] << ",\"absolute_node_begin\":" << abs_begin
                     << ",\"absolute_node_end\":" << abs_end << ",\"N\":" << node_end - node_begin << ",\"E\":" << edge_end - edge_begin
                     << ",\"nonself_neighbor_stats\":{\"eligible_nodes\":" << stats.eligible_nodes << ",\"min\":" << stats.minimum
                     << ",\"median\":" << std::setprecision(8) << stats.median << ",\"max\":" << stats.maximum << "}}";
        }
        metadata << "],\"node_ptr\":[";
        for (std::size_t i = 0; i < input.witness.node_ptr.size(); ++i)
        {
            if (i != 0)
                metadata << ',';
            metadata << input.witness.node_ptr[i];
        }
        metadata << "],\"edge_ptr\":[";
        for (std::size_t i = 0; i < input.witness.edge_ptr.size(); ++i)
        {
            if (i != 0)
                metadata << ',';
            metadata << input.witness.edge_ptr[i];
        }
        metadata << "],\"scales\":" << scales_json(input.witness) << ",\"srs_expectation\":{\"dataset_id\":\""
                 << config.dataset << "\",\"srs_id\":\"" << config.srs_id << "\",\"n_domain\":" << n_domain
                 << ",\"e_domain\":" << e_domain
                 << ",\"required_coefficient_count\":" << required_coefficients << "},\"timing_scope\":\"input preparation only; excluded from all proof metrics\",\"upstream_relation_proven_by_mechanism_proof\":false}\n";
        const auto metadata_text = metadata.str();
        internal::write_text_file(options.metadata_out, metadata_text);
        PrepareRealReceipt receipt;
        receipt.nodes = input.witness.node_count;
        receipt.edges = input.witness.edge_count;
        receipt.graphs = input.witness.node_ptr.size() - 1;
        receipt.bundle_sha256 = input.bundle_sha256;
        receipt.metadata_sha256 = sha256_bytes(metadata_text);
        return receipt;
    }
}
