#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "../src/mechanism/codec.hpp"
#include "../src/mechanism/real_input.hpp"

namespace
{
    using namespace gatzk;
    using namespace gatzk::mechanism;
    using namespace gatzk::mechanism::internal;
    using Field = algebra::FieldElement;

    void require(bool condition, const std::string &message)
    {
        if (!condition)
            throw std::runtime_error(message);
    }

    std::vector<std::uint8_t> read_bytes(const std::filesystem::path &path)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in)
            throw std::runtime_error("cannot open regression input: " + path.string());
        return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }

    std::string read_text(const std::filesystem::path &path)
    {
        const auto bytes = read_bytes(path);
        return std::string(bytes.begin(), bytes.end());
    }

    void write_bytes(const std::filesystem::path &path, const std::vector<std::uint8_t> &bytes)
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out)
            throw std::runtime_error("cannot write regression input: " + path.string());
        out.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!out)
            throw std::runtime_error("failed writing regression input: " + path.string());
    }

    std::size_t field_after(const std::string &text, const std::string &key)
    {
        const auto key_pos = text.find(key);
        require(key_pos != std::string::npos, "metadata is missing key " + key);
        auto pos = key_pos + key.size();
        while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n' || text[pos] == '\r'))
            ++pos;
        require(pos < text.size() && text[pos] == ':', "metadata key is malformed: " + key);
        ++pos;
        while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n' || text[pos] == '\r'))
            ++pos;
        require(pos < text.size() && text[pos] >= '0' && text[pos] <= '9', "metadata value is not an unsigned integer: " + key);
        std::size_t value = 0;
        while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9')
        {
            const auto digit = static_cast<std::size_t>(text[pos++] - '0');
            require(value <= (std::numeric_limits<std::size_t>::max() - digit) / 10, "metadata integer overflows size_t");
            value = value * 10 + digit;
        }
        return value;
    }

    std::string string_field_after(const std::string &text, const std::string &key)
    {
        const auto key_pos = text.find(key);
        require(key_pos != std::string::npos, "metadata is missing key " + key);
        auto pos = key_pos + key.size();
        while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n' || text[pos] == '\r'))
            ++pos;
        require(pos < text.size() && text[pos] == ':', "metadata key is malformed: " + key);
        ++pos;
        while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n' || text[pos] == '\r'))
            ++pos;
        require(pos < text.size() && text[pos] == '"', "metadata value is not a string: " + key);
        const auto end = text.find('"', ++pos);
        require(end != std::string::npos, "metadata string is unterminated: " + key);
        return text.substr(pos, end - pos);
    }

    std::vector<std::size_t> parse_csv(const std::string &value)
    {
        std::vector<std::size_t> out;
        std::size_t start = 0;
        while (start < value.size())
        {
            const auto comma = value.find(',', start);
            const auto end = comma == std::string::npos ? value.size() : comma;
            require(end > start, "component list contains an empty item");
            std::size_t parsed = 0;
            for (std::size_t i = start; i < end; ++i)
            {
                require(value[i] >= '0' && value[i] <= '9', "component list must contain decimal IDs");
                const auto digit = static_cast<std::size_t>(value[i] - '0');
                require(parsed <= (std::numeric_limits<std::size_t>::max() - digit) / 10, "component ID overflows size_t");
                parsed = parsed * 10 + digit;
            }
            out.push_back(parsed);
            if (comma == std::string::npos)
                break;
            start = comma + 1;
        }
        require(!out.empty(), "component list is empty");
        return out;
    }

    std::vector<std::size_t> metadata_component_ids(const std::string &metadata)
    {
        const auto key = metadata.find("\"selected_components\":[");
        require(key != std::string::npos, "metadata is missing selected_components");
        auto pos = key + std::string("\"selected_components\":[").size();
        std::vector<std::size_t> out;
        while (true)
        {
            const auto id_pos = metadata.find("\"canonical_id\":", pos);
            const auto close = metadata.find(']', pos);
            if (id_pos == std::string::npos || (close != std::string::npos && id_pos > close))
                break;
            out.push_back(field_after(metadata.substr(id_pos), "\"canonical_id\""));
            pos = id_pos + 1;
        }
        return out;
    }

    std::vector<std::string> metadata_object_array(const std::string &metadata, const std::string &key)
    {
        const auto key_pos = metadata.find(key);
        require(key_pos != std::string::npos, "metadata is missing object array " + key);
        auto pos = metadata.find('[', key_pos + key.size());
        require(pos != std::string::npos, "metadata object array has no opening bracket: " + key);
        ++pos;
        std::vector<std::string> out;
        while (pos < metadata.size())
        {
            while (pos < metadata.size() && (metadata[pos] == ',' || metadata[pos] == ' ' || metadata[pos] == '\t' || metadata[pos] == '\n' || metadata[pos] == '\r'))
                ++pos;
            if (pos < metadata.size() && metadata[pos] == ']')
                break;
            require(pos < metadata.size() && metadata[pos] == '{', "metadata object array contains a non-object entry");
            const auto begin = pos;
            std::size_t depth = 0;
            bool in_string = false;
            bool escaped = false;
            for (; pos < metadata.size(); ++pos)
            {
                const auto ch = metadata[pos];
                if (in_string)
                {
                    if (escaped)
                        escaped = false;
                    else if (ch == '\\')
                        escaped = true;
                    else if (ch == '"')
                        in_string = false;
                    continue;
                }
                if (ch == '"')
                    in_string = true;
                else if (ch == '{')
                    ++depth;
                else if (ch == '}' && --depth == 0)
                {
                    ++pos;
                    out.push_back(metadata.substr(begin, pos - begin));
                    break;
                }
            }
            require(pos <= metadata.size() && depth == 0, "metadata object array contains an unterminated object");
        }
        return out;
    }

    double floating_field_after(const std::string &text, const std::string &key)
    {
        const auto key_pos = text.find(key);
        require(key_pos != std::string::npos, "metadata is missing key " + key);
        auto pos = key_pos + key.size();
        while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n' || text[pos] == '\r'))
            ++pos;
        require(pos < text.size() && text[pos] == ':', "metadata key is malformed: " + key);
        ++pos;
        while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n' || text[pos] == '\r'))
            ++pos;
        std::size_t consumed = 0;
        const auto value = std::stod(text.substr(pos), &consumed);
        require(consumed != 0, "metadata value is not numeric: " + key);
        return value;
    }

    std::vector<std::size_t> metadata_pointer_array(const std::string &metadata, const std::string &key)
    {
        const auto key_pos = metadata.find(key);
        require(key_pos != std::string::npos, "metadata is missing pointer array " + key);
        auto pos = key_pos + key.size();
        while (pos < metadata.size() && metadata[pos] != '[')
            ++pos;
        require(pos < metadata.size(), "metadata pointer array has no opening bracket: " + key);
        const auto end = metadata.find(']', ++pos);
        require(end != std::string::npos, "metadata pointer array has no closing bracket: " + key);
        std::vector<std::size_t> values;
        while (pos < end)
        {
            while (pos < end && (metadata[pos] == ',' || metadata[pos] == ' ' || metadata[pos] == '\n' || metadata[pos] == '\r'))
                ++pos;
            if (pos == end)
                break;
            require(metadata[pos] >= '0' && metadata[pos] <= '9', "metadata pointer array contains a non-integer");
            std::size_t value = 0;
            while (pos < end && metadata[pos] >= '0' && metadata[pos] <= '9')
            {
                const auto digit = static_cast<std::size_t>(metadata[pos++] - '0');
                require(value <= (std::numeric_limits<std::size_t>::max() - digit) / 10, "metadata pointer overflows size_t");
                value = value * 10 + digit;
            }
            values.push_back(value);
        }
        return values;
    }

    std::vector<std::size_t> graph_ids_from_edges(const protocol::Layer0Head0MechanismWitness &w)
    {
        std::vector<std::size_t> graph_ids(w.edge_count);
        for (std::size_t graph = 0; graph + 1 < w.edge_ptr.size(); ++graph)
            for (std::size_t edge = w.edge_ptr[graph]; edge < w.edge_ptr[graph + 1]; ++edge)
                graph_ids[edge] = graph;
        return graph_ids;
    }

    void validate_real_relation(const RealMechanismInput &input, const std::string &metadata,
                                const std::string &expected_dataset, const std::vector<std::size_t> &expected_components,
                                const std::filesystem::path &bundle_path)
    {
        const auto &w = input.witness;
        require(input.dataset == expected_dataset, "bundle dataset identity differs from the requested case");
        require(w.width == 8, "formal layer-0 head-0 witness width is not eight");
        require(w.node_count != 0 && w.edge_count >= w.node_count, "real mechanism bundle has an invalid nonempty graph shape");
        require(w.node_ptr.size() == expected_components.size() + 1 && w.edge_ptr.size() == expected_components.size() + 1,
                "bundle graph boundary count differs from the selected components");
        require(w.h_prime_raw.size() == w.node_count && w.h_prime.size() == w.node_count &&
                    w.u_raw.size() == w.edge_count && w.u.size() == w.edge_count &&
                    w.sum_raw.size() == w.node_count && w.sum.size() == w.node_count &&
                    w.alpha.size() == w.edge_count && w.weighted_edge_features.size() == w.edge_count &&
                    w.aggregate_pre_activation.size() == w.node_count,
                "bundle witness arrays have inconsistent lengths");
        require(w.edge_src_indices.size() == w.edge_count && w.edge_dst_indices.size() == w.edge_count,
                "bundle topology arrays have inconsistent lengths");
        require(w.node_ptr.front() == 0 && w.node_ptr.back() == w.node_count &&
                    w.edge_ptr.front() == 0 && w.edge_ptr.back() == w.edge_count,
                "bundle graph boundaries do not cover the whole graph");

        const auto metadata_nodes = field_after(metadata, "\"N\"");
        const auto metadata_edges = field_after(metadata, "\"E\"");
        const auto metadata_graphs = field_after(metadata, "\"G\"");
        require(metadata_nodes == w.node_count && metadata_edges == w.edge_count && metadata_graphs == expected_components.size(),
                "metadata N/E/G differs from the serialized formal graph");
        require(metadata.find("\"layer\":0,\"head\":0,\"formal_head_width\":8") != std::string::npos,
                "metadata does not identify canonical layer 0, head 0 with formal width eight");
        require(metadata_component_ids(metadata) == expected_components,
                "metadata component IDs differ from the explicit selected component list");
        const auto component_records = metadata_object_array(metadata, "\"selected_components\"");
        require(component_records.size() == expected_components.size(), "metadata component record count differs from the selected graph count");
        for (std::size_t graph = 0; graph < component_records.size(); ++graph)
        {
            const auto &record = component_records[graph];
            const auto graph_nodes = w.node_ptr[graph + 1] - w.node_ptr[graph];
            const auto graph_edges = w.edge_ptr[graph + 1] - w.edge_ptr[graph];
            require(field_after(record, "\"canonical_id\"") == expected_components[graph] &&
                        field_after(record, "\"N\"") == graph_nodes && field_after(record, "\"E\"") == graph_edges,
                    "metadata component ID or per-component N/E differs from serialized boundaries");
            require(field_after(record, "\"absolute_node_end\"") > field_after(record, "\"absolute_node_begin\""),
                    "metadata component absolute node range is empty");
            std::vector<std::set<std::size_t>> incoming(graph_nodes);
            for (std::size_t edge = w.edge_ptr[graph]; edge < w.edge_ptr[graph + 1]; ++edge)
            {
                const auto src = w.edge_src_indices[edge];
                const auto dst = w.edge_dst_indices[edge];
                if (src != dst)
                    incoming[dst - w.node_ptr[graph]].insert(src);
            }
            std::vector<std::size_t> degrees;
            degrees.reserve(graph_nodes);
            std::size_t eligible = 0;
            for (const auto &neighbors : incoming)
            {
                degrees.push_back(neighbors.size());
                if (neighbors.size() >= 2)
                    ++eligible;
            }
            std::sort(degrees.begin(), degrees.end());
            const auto &stats_key = "\"nonself_neighbor_stats\"";
            const auto stats_start = record.find(stats_key);
            require(stats_start != std::string::npos, "metadata component omits nonself neighbor statistics");
            const auto stats = record.substr(stats_start);
            const auto median = degrees.size() % 2 == 0
                                    ? (static_cast<double>(degrees[degrees.size() / 2 - 1]) + static_cast<double>(degrees[degrees.size() / 2])) / 2.0
                                    : static_cast<double>(degrees[degrees.size() / 2]);
            require(field_after(stats, "\"eligible_nodes\"") == eligible && field_after(stats, "\"min\"") == degrees.front() &&
                        field_after(stats, "\"max\"") == degrees.back() &&
                        std::abs(floating_field_after(stats, "\"median\"") - median) <= 1e-5,
                    "metadata nonself neighbor statistics differ from the serialized graph topology");
        }
        require(metadata_pointer_array(metadata, "\"node_ptr\"") == w.node_ptr &&
                    metadata_pointer_array(metadata, "\"edge_ptr\"") == w.edge_ptr,
                "metadata graph boundaries differ from the serialized topology");
        const auto bundle_bytes = read_bytes(bundle_path);
        const auto bundle_sha = sha256_hex(std::string_view(reinterpret_cast<const char *>(bundle_bytes.data()), bundle_bytes.size()));
        const auto bundle_metadata_pos = metadata.find("\"bundle\":{");
        require(bundle_metadata_pos != std::string::npos, "metadata is missing the real-input bundle record");
        require(bundle_sha == input.bundle_sha256 &&
                    string_field_after(metadata.substr(bundle_metadata_pos), "\"sha256\"") == bundle_sha,
                "metadata or bundle SHA-256 does not match the serialized real input");
        require(string_field_after(metadata, "\"witness_sha256\"") == input.witness_sha256,
                "metadata witness digest differs from the serialized real input");
        require(string_field_after(metadata, "\"dataset\"") == expected_dataset,
                "metadata dataset identity differs from the requested case");
        const auto srs_expectation_pos = metadata.find("\"srs_expectation\":{");
        require(srs_expectation_pos != std::string::npos, "metadata is missing its trusted SRS expectation");
        const auto srs_expectation = metadata.substr(srs_expectation_pos);
        const auto *catalog = registered_real_dataset_catalog(expected_dataset);
        require(catalog != nullptr, "requested real dataset has no registered catalog entry");
        require(input.input_preprocess_id == catalog->input_preprocess_id &&
                    string_field_after(metadata, "\"input_preprocess_id\"") == catalog->input_preprocess_id,
                "metadata or bundle preprocessing identity differs from the dataset catalog");
        const auto expected_srs_id = catalog->srs_id;
        require(string_field_after(srs_expectation, "\"dataset_id\"") == expected_dataset &&
                    string_field_after(srs_expectation, "\"srs_id\"") == expected_srs_id,
                "metadata dataset or registered SRS identity differs from the canonical input");
        require(field_after(srs_expectation, "\"n_domain\"") == catalog->node_domain_size &&
                    field_after(srs_expectation, "\"e_domain\"") == catalog->edge_domain_size &&
                    field_after(srs_expectation, "\"required_coefficient_count\"") == 2 * catalog->edge_domain_size + 14,
                "metadata trusted SRS domains or mechanism coefficient prefix differ from the dataset catalog");
        const auto checkpoint_metadata_pos = metadata.find("\"checkpoint_bundle\":{");
        require(checkpoint_metadata_pos != std::string::npos &&
                    string_field_after(metadata.substr(checkpoint_metadata_pos), "\"sha256\"") == input.checkpoint_bundle_sha256 &&
                    string_field_after(metadata, "\"source_manifest_sha256\"") == input.source_manifest_sha256,
                "metadata checkpoint or source provenance differs from the serialized input");

        const auto edge_graph = graph_ids_from_edges(w);
        std::vector<Field> denominator(w.node_count, Field::zero());
        std::vector<std::vector<Field>> aggregate(w.node_count, std::vector<Field>(w.width, Field::zero()));
        std::size_t nonself_edges = 0;
        for (std::size_t node = 0; node < w.node_count; ++node)
        {
            require(w.h_prime[node].size() == w.width && w.h_prime_raw[node].size() == w.width,
                    "H-prime row width differs from the formal head width");
            for (std::size_t lane = 0; lane < w.width; ++lane)
                require(w.h_prime[node][lane] == Field::from_signed(w.h_prime_raw[node][lane]),
                        "serialized H-prime field value differs from its formal quantized integer");
        }
        for (std::size_t edge = 0; edge < w.edge_count; ++edge)
        {
            const auto graph = edge_graph[edge];
            const auto src = w.edge_src_indices[edge];
            const auto dst = w.edge_dst_indices[edge];
            require(graph + 1 < w.node_ptr.size() && edge >= w.edge_ptr[graph] && edge < w.edge_ptr[graph + 1],
                    "edge graph identity disagrees with its graph boundary");
            require(src >= w.node_ptr[graph] && src < w.node_ptr[graph + 1] &&
                        dst >= w.node_ptr[graph] && dst < w.node_ptr[graph + 1],
                    "edge crosses a selected graph-component boundary");
            require(w.u[edge] == Field::from_signed(w.u_raw[edge]), "serialized U differs from its formal quantized integer");
            require(w.weighted_edge_features[edge].size() == w.width, "weighted edge feature row has an invalid width");
            denominator[dst] += w.u[edge];
            if (src != dst)
                ++nonself_edges;
            require(w.alpha[edge] * w.sum[dst] == w.u[edge], "normalized alpha does not equal U divided by its destination Sum");
            for (std::size_t lane = 0; lane < w.width; ++lane)
            {
                const auto expected = w.alpha[edge] * w.h_prime[src][lane];
                require(w.weighted_edge_features[edge][lane] == expected,
                        "weighted edge feature differs from normalized alpha times source H-prime");
                aggregate[dst][lane] += expected;
            }
        }
        for (std::size_t node = 0; node < w.node_count; ++node)
        {
            require(w.sum[node] == Field::from_signed(w.sum_raw[node]), "serialized Sum differs from its formal quantized integer");
            require(denominator[node] == w.sum[node] && !w.sum[node].is_zero(),
                    "Sum is not the positive grouped denominator of incoming U values");
            require(aggregate[node] == w.aggregate_pre_activation[node],
                    "aggregate target differs from the exact pre-activation weighted sum");
        }
        if (expected_dataset == "ppi")
        {
            require(nonself_edges != 0, "selected PPI graph components contain no nontrivial neighbor edges");
            for (std::size_t graph = 0; graph < expected_components.size(); ++graph)
                require(w.edge_ptr[graph + 1] - w.edge_ptr[graph] > w.node_ptr[graph + 1] - w.node_ptr[graph],
                        "selected PPI component has no nonself edges after graph-boundary normalization");
            require(metadata.find("\"eligible_nodes\":") != std::string::npos &&
                        metadata.find("\"nonself_neighbor_stats\":") != std::string::npos,
                    "PPI metadata omits nonself neighborhood statistics");
        }

        for (std::size_t width = 1; width <= 8; ++width)
        {
            const auto ff = real_mechanism_relation_digest(input, Mechanism::FeatureFolding, width);
            const auto ff_again = real_mechanism_relation_digest(input, Mechanism::FeatureFolding, width);
            require(ff == ff_again && ff.size() == 64, "feature-folding D-prefix relation digest is not deterministic");
            for (std::size_t edge = 0; edge < w.edge_count; ++edge)
                for (std::size_t lane = 0; lane < width; ++lane)
                    require(w.h_prime[w.edge_src_indices[edge]][lane] == Field::from_signed(w.h_prime_raw[w.edge_src_indices[edge]][lane]),
                            "feature-folding edge source does not select the corresponding H-prime prefix coordinate");
        }
        const auto shared = real_mechanism_relation_digest(input, Mechanism::SharedNeighborhood, 8);
        require(shared.size() == 64 && shared == real_mechanism_relation_digest(input, Mechanism::SharedNeighborhood, 8),
                "shared-neighborhood relation digest is not deterministic");
        bool rejected_too_wide = false;
        try
        {
            (void)real_mechanism_relation_digest(input, Mechanism::FeatureFolding, 9);
        }
        catch (const std::runtime_error &)
        {
            rejected_too_wide = true;
        }
        require(rejected_too_wide, "feature-folding relation accepted a width larger than the formal head");
    }

    RealMechanismInput make_codec_fixture()
    {
        RealMechanismInput input;
        input.dataset = "fixture";
        input.input_preprocess_id = "fixture_preprocess";
        input.model_arch_id = "fixture_arch";
        input.model_param_id = "fixture_params";
        input.quant_cfg_id = "fixture_quant";
        input.static_table_id = "fixture_tables";
        input.config_sha256 = std::string(64, '1');
        input.checkpoint_bundle_sha256 = std::string(64, '2');
        input.source_manifest_sha256 = std::string(64, '3');
        auto &w = input.witness;
        w.node_count = 4;
        w.edge_count = 6;
        w.width = 8;
        w.node_ptr = {0, 2, 4};
        w.edge_ptr = {0, 3, 6};
        w.edge_src_indices = {0, 1, 1, 2, 3, 3};
        w.edge_dst_indices = {0, 0, 1, 2, 2, 3};
        w.h_prime_raw.assign(w.node_count, std::vector<std::int64_t>(w.width));
        w.h_prime.assign(w.node_count, std::vector<Field>(w.width, Field::zero()));
        for (std::size_t node = 0; node < w.node_count; ++node)
            for (std::size_t lane = 0; lane < w.width; ++lane)
            {
                w.h_prime_raw[node][lane] = static_cast<std::int64_t>(node * 17 + lane + 1);
                w.h_prime[node][lane] = Field::from_signed(w.h_prime_raw[node][lane]);
            }
        w.u_raw = {2, 3, 5, 7, 11, 13};
        w.u.reserve(w.edge_count);
        w.alpha.resize(w.edge_count, Field::zero());
        w.weighted_edge_features.assign(w.edge_count, std::vector<Field>(w.width, Field::zero()));
        w.sum_raw.assign(w.node_count, 0);
        w.sum.assign(w.node_count, Field::zero());
        w.aggregate_pre_activation.assign(w.node_count, std::vector<Field>(w.width, Field::zero()));
        for (std::size_t edge = 0; edge < w.edge_count; ++edge)
            w.sum_raw[w.edge_dst_indices[edge]] += w.u_raw[edge];
        for (std::size_t node = 0; node < w.node_count; ++node)
            w.sum[node] = Field::from_signed(w.sum_raw[node]);
        for (std::size_t edge = 0; edge < w.edge_count; ++edge)
        {
            w.u.push_back(Field::from_signed(w.u_raw[edge]));
            w.alpha[edge] = w.u[edge] / w.sum[w.edge_dst_indices[edge]];
            for (std::size_t lane = 0; lane < w.width; ++lane)
            {
                const auto value = w.alpha[edge] * w.h_prime[w.edge_src_indices[edge]][lane];
                w.weighted_edge_features[edge][lane] = value;
                w.aggregate_pre_activation[w.edge_dst_indices[edge]][lane] += value;
            }
        }
        w.s_input = 256;
        w.s_w_hidden = 128;
        w.s_h_prime = 32768;
        w.s_a_src_hidden = 128;
        w.s_a_dst_hidden = 128;
        w.s_delta = 4096;
        w.s_exp = 65536;
        w.s_agg_pre = 1U << 20U;
        return input;
    }

    void run_ff_width_prefix_relation_tests()
    {
        const auto input = make_codec_fixture();
        std::string previous_digest;
        for (std::size_t width = 1; width <= 8; ++width)
        {
            const auto digest = real_mechanism_relation_digest(input, Mechanism::FeatureFolding, width);
            require(digest.size() == 64 && digest != previous_digest,
                    "feature-folding relation digest failed to cover a distinct D-prefix from 1 through 8");
            require(digest == real_mechanism_relation_digest(input, Mechanism::FeatureFolding, width),
                    "feature-folding D-prefix relation digest is not deterministic");
            previous_digest = digest;
        }
        require(!real_mechanism_relation_digest(input, Mechanism::SharedNeighborhood, 8).empty(),
                "shared-neighborhood width-eight relation digest is empty");
    }

    void expect_bundle_rejected(const std::vector<std::uint8_t> &bytes, const std::filesystem::path &path, const std::string &case_name)
    {
        write_bytes(path, bytes);
        bool rejected = false;
        try
        {
            (void)read_real_mechanism_input(path);
        }
        catch (const std::runtime_error &)
        {
            rejected = true;
        }
        require(rejected, "real-input bundle reader accepted the " + case_name + " negative fixture");
    }

    void expect_input_rejected(const RealMechanismInput &input, const std::filesystem::path &path, const std::string &case_name)
    {
        std::vector<std::uint8_t> bytes;
        try
        {
            bytes = encode_real_mechanism_input(input);
        }
        catch (const std::runtime_error &)
        {
            return;
        }
        expect_bundle_rejected(bytes, path, case_name);
    }

    std::size_t first_alpha_offset(const std::vector<std::uint8_t> &bytes, std::size_t graphs)
    {
        std::size_t offset = 8 + 4;
        for (unsigned i = 0; i < 9; ++i)
        {
            require(offset + 4 <= bytes.size(), "fixture binary string header is truncated");
            const auto size = static_cast<std::uint32_t>(bytes[offset]) |
                              (static_cast<std::uint32_t>(bytes[offset + 1]) << 8U) |
                              (static_cast<std::uint32_t>(bytes[offset + 2]) << 16U) |
                              (static_cast<std::uint32_t>(bytes[offset + 3]) << 24U);
            offset += 4;
            require(size <= bytes.size() - offset, "fixture binary string is truncated");
            offset += size;
        }
        offset += 4 * 8;
        offset += 2 * (graphs + 1) * 8;
        offset += 8; // edge count
        offset += 6 * 3 * 8; // edge records: source, destination, graph
        offset += 8 + 4 * 8 * 8; // H-prime dimensions and raw rows
        offset += 8 + 6 * 8; // raw U
        offset += 8 + 4 * 8; // raw Sum
        offset += 8 * 8; // scales
        offset += 8; // alpha count
        require(offset + 32 <= bytes.size(), "fixture alpha payload is truncated");
        return offset;
    }

    void run_codec_negative_tests(const std::filesystem::path &root)
    {
        std::filesystem::create_directories(root);
        const auto base = make_codec_fixture();
        const auto valid_bytes = encode_real_mechanism_input(base);
        const auto valid_path = root / "fixture.realinput";
        write_bytes(valid_path, valid_bytes);
        const auto parsed = read_real_mechanism_input(valid_path);
        require(parsed.witness.node_ptr == base.witness.node_ptr && parsed.witness.edge_ptr == base.witness.edge_ptr &&
                    parsed.witness.edge_src_indices == base.witness.edge_src_indices &&
                    parsed.witness.edge_dst_indices == base.witness.edge_dst_indices,
                "real-input codec roundtrip changed fixture graph boundaries or edge ordering");
        require(real_mechanism_witness_digest(parsed) == real_mechanism_witness_digest(base),
                "real-input codec roundtrip changed the witness digest");

        auto stable_source_order = base;
        std::swap(stable_source_order.witness.edge_src_indices[0], stable_source_order.witness.edge_src_indices[1]);
        for (auto &row : stable_source_order.witness.aggregate_pre_activation)
            std::fill(row.begin(), row.end(), Field::zero());
        for (std::size_t edge = 0; edge < stable_source_order.witness.edge_count; ++edge)
            for (std::size_t lane = 0; lane < stable_source_order.witness.width; ++lane)
            {
                const auto value = stable_source_order.witness.alpha[edge] *
                                   stable_source_order.witness.h_prime[stable_source_order.witness.edge_src_indices[edge]][lane];
                stable_source_order.witness.weighted_edge_features[edge][lane] = value;
                stable_source_order.witness.aggregate_pre_activation[stable_source_order.witness.edge_dst_indices[edge]][lane] += value;
            }
        const auto stable_order_bytes = encode_real_mechanism_input(stable_source_order);
        write_bytes(root / "stable-loader-order.realinput", stable_order_bytes);
        const auto stable_order_read = read_real_mechanism_input(root / "stable-loader-order.realinput");
        require(stable_order_read.witness.edge_src_indices == stable_source_order.witness.edge_src_indices &&
                    stable_order_read.witness.edge_dst_indices == stable_source_order.witness.edge_dst_indices,
                "real-input codec rejected or reordered loader-stable edges within one destination group");

        auto truncated = valid_bytes;
        truncated.pop_back();
        expect_bundle_rejected(truncated, root / "truncated.realinput", "truncated");

        auto trailing = valid_bytes;
        trailing.push_back(0);
        expect_bundle_rejected(trailing, root / "trailing.realinput", "trailing-byte");

        auto noncanonical_field = valid_bytes;
        const auto alpha_offset = first_alpha_offset(valid_bytes, base.witness.node_ptr.size() - 1);
        std::fill(noncanonical_field.begin() + static_cast<std::ptrdiff_t>(alpha_offset),
                  noncanonical_field.begin() + static_cast<std::ptrdiff_t>(alpha_offset + 32), 0xff);
        expect_bundle_rejected(noncanonical_field, root / "noncanonical-field.realinput", "noncanonical-field");

        auto cross_component = base;
        cross_component.witness.edge_src_indices[0] = 2;
        expect_input_rejected(cross_component, root / "cross-component.realinput", "cross-component-edge");

        auto destination_disorder = base;
        std::swap(destination_disorder.witness.edge_dst_indices[1], destination_disorder.witness.edge_dst_indices[2]);
        expect_input_rejected(destination_disorder, root / "destination-disorder.realinput", "destination-group-disorder");

        auto corrupt_sum = base;
        ++corrupt_sum.witness.sum_raw[0];
        expect_input_rejected(corrupt_sum, root / "corrupt-sum.realinput", "corrupt-Sum");

        auto corrupt_aggregate = base;
        corrupt_aggregate.witness.aggregate_pre_activation[0][0] += Field::one();
        expect_input_rejected(corrupt_aggregate, root / "corrupt-aggregate.realinput", "corrupt-pre-activation-target");
    }

    void run_registered_dataset_catalog_tests()
    {
        struct Expected
        {
            std::string_view dataset;
            std::string_view preprocess;
            std::string_view srs_id;
            std::size_t node_domain;
            std::size_t edge_domain;
        };
        constexpr std::array<Expected, 5> expected{{
            {"cora", "row_normalize", "cora_formal_srs", 4096, 32768},
            {"citeseer", "row_normalize", "citeseer_formal_srs", 4096, 32768},
            {"pubmed", "row_normalize", "pubmed_formal_srs", 32768, 262144},
            {"ppi", "row_normalize", "ppi_formal_srs", 65536, 8388608},
            {"ogbn_arxiv", "ogbn_arxiv_raw_float32", "ogbn_arxiv_formal_srs", 262144, 16777216},
        }};
        for (const auto &entry : expected)
        {
            const auto *catalog = registered_real_dataset_catalog(entry.dataset);
            require(catalog != nullptr && catalog->input_preprocess_id == entry.preprocess && catalog->srs_id == entry.srs_id &&
                        catalog->node_domain_size == entry.node_domain && catalog->edge_domain_size == entry.edge_domain,
                    "registered real dataset catalog entry differs from the canonical config/SRS set");
            require(valid_real_component_selection(entry.dataset, {0}), "canonical graph component 0 was rejected");
        }
        require(registered_real_dataset_catalog("unknown") == nullptr, "unregistered dataset received a trusted catalog entry");
        const auto *ogbn = registered_real_dataset_catalog("ogbn_arxiv");
        require(ogbn != nullptr && ogbn->input_preprocess_id != "row_normalize",
                "OGBN-Arxiv was assigned the wrong formal preprocessing identity");
        std::vector<std::size_t> all_ppi_components(295);
        for (std::size_t graph = 0; graph < all_ppi_components.size(); ++graph)
            all_ppi_components[graph] = graph;
        require(valid_real_component_selection("ppi", all_ppi_components), "full canonical PPI component set 0..294 was rejected");
        require(valid_real_graph_count("ppi", 1) && valid_real_graph_count("ppi", 295) && !valid_real_graph_count("ppi", 2) &&
                    !valid_real_graph_count("ppi", 294) && valid_real_graph_count("pubmed", 1) &&
                    !valid_real_graph_count("pubmed", 2),
                "real dataset graph count admission differs from the complete registered component selection");
        require(!valid_real_component_selection("ppi", {0, 9}) && !valid_real_component_selection("ppi", {0, 1, 3}) &&
                    !valid_real_component_selection("cora", {0, 1}),
                "partial or noncanonical graph selections were accepted");
    }

    PublicStatement make_ppi_statement_fixture(std::size_t graphs)
    {
        PublicStatement statement;
        statement.dataset_id = "ppi";
        statement.input_preprocess_id = "row_normalize";
        statement.real_input_bundle_sha256 = std::string(64, '1');
        statement.real_input_witness_sha256 = std::string(64, '2');
        statement.real_input_metadata_sha256 = std::string(64, '3');
        statement.checkpoint_bundle_sha256 = std::string(64, '4');
        statement.input_source_manifest_sha256 = std::string(64, '5');
        statement.formal_head_width = 8;
        statement.mechanism = Mechanism::FeatureFolding;
        statement.variant = Variant::Original;
        statement.nodes = graphs;
        statement.edges = graphs;
        statement.width = 3;
        statement.graphs = graphs;
        statement.seed = 9;
        statement.node_ptr.resize(graphs + 1);
        statement.edge_ptr.resize(graphs + 1);
        statement.topology.reserve(graphs);
        for (std::size_t graph = 0; graph <= graphs; ++graph)
        {
            statement.node_ptr[graph] = graph;
            statement.edge_ptr[graph] = graph;
            if (graph != graphs)
                statement.topology.push_back(Edge{graph, graph, graph});
        }
        statement.srs_id = "mechanism_ablation_test_only_parser_fixture";
        statement.srs_manifest_sha256 = std::string(64, '6');
        statement.parent_protocol_schema_id = "mechanism_ablation_test_only_schema_fixture";
        statement.parent_srs_degree_bound = 33554431;
        statement.node_domain_size = 65536;
        statement.edge_domain_size = 8388608;
        statement.pair_input_commitment_sha256 = std::string(64, '7');
        statement.security_budget_ledger_sha256 = internal::compute_security_budget_ledger_sha256(statement);
        statement.mechanism_schema_id = internal::compute_mechanism_schema_id(statement);
        return statement;
    }

    void run_ppi_statement_graph_count_tests()
    {
        for (const auto graphs : {1U, 295U})
        {
            const auto encoded = make_ppi_statement_fixture(graphs).canonical_json();
            const auto parsed = PublicStatement::parse_canonical_json(encoded);
            require(parsed.dataset_id == "ppi" && parsed.graphs == graphs,
                    "canonical PPI statement parser rejected an allowed complete component count");
        }
        const auto invalid = make_ppi_statement_fixture(2).canonical_json();
        bool rejected = false;
        try
        {
            (void)PublicStatement::parse_canonical_json(invalid);
        }
        catch (const std::runtime_error &)
        {
            rejected = true;
        }
        require(rejected, "independent statement parser accepted PPI graph count 2 instead of component 0 or full 0..294");
    }

    void run_real_bundle_case(const std::string &dataset, const std::filesystem::path &bundle_path,
                              const std::filesystem::path &metadata_path, const std::string &component_csv)
    {
        const auto input = read_real_mechanism_input(bundle_path);
        const auto metadata = read_text(metadata_path);
        validate_real_relation(input, metadata, dataset, parse_csv(component_csv), bundle_path);
        std::cout << "REAL_MECHANISM_INPUT_OK dataset=" << dataset << " N=" << input.witness.node_count
                  << " E=" << input.witness.edge_count << " G=" << input.witness.node_ptr.size() - 1
                  << " FF_D_prefixes=1..8 SN_width=8 components=" << component_csv << '\n';
    }
}

int main(int argc, char **argv)
{
    try
    {
        const bool catalog_only = argc == 2 && std::string_view(argv[1]) == "--catalog-only";
        if (!catalog_only && (argc < 5 || (argc - 1) % 4 != 0))
            throw std::runtime_error("usage: gatzk_real_mechanism_input_regression DATASET BUNDLE METADATA COMPONENT_IDS [DATASET BUNDLE METADATA COMPONENT_IDS ...]");
        const auto test_root = std::filesystem::temp_directory_path() /
                               ("gatzk_real_mechanism_input_regression_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        run_registered_dataset_catalog_tests();
        run_ppi_statement_graph_count_tests();
        run_codec_negative_tests(test_root);
        run_ff_width_prefix_relation_tests();
        if (catalog_only)
        {
            std::filesystem::remove_all(test_root);
            std::cout << "REAL_MECHANISM_INPUT_REGRESSION_OK catalog=5 ppi_components=0..294 ppi_statement_G2=REJECTED "
                         "FF_D_prefixes=1..8 SN_width=8 fixture_codec_negatives=7 real_cases=0\n";
            return 0;
        }
        for (int arg = 1; arg < argc; arg += 4)
            run_real_bundle_case(argv[arg], argv[arg + 1], argv[arg + 2], argv[arg + 3]);
        std::filesystem::remove_all(test_root);
        std::cout << "REAL_MECHANISM_INPUT_REGRESSION_OK fixture_codec_negatives=7 real_cases=" << (argc - 1) / 4 << '\n';
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "real mechanism input regression: " << error.what() << '\n';
        return 1;
    }
}
