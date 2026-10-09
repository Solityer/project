#include <algorithm>
#include <array>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include <mcl/bn.hpp>
#include <openssl/sha.h>

#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/crypto/curve.hpp"
#include "gatzk/crypto/kzg.hpp"
#include "gatzk/protocol/proof_codec.hpp"
#include "gatzk/protocol/prover.hpp"
#include "gatzk/protocol/schema.hpp"
#include "gatzk/protocol/trace.hpp"
#include "gatzk/protocol/verifier.hpp"
#include "gatzk/protocol/verifier_context.hpp"
#include "gatzk/util/config.hpp"
#include "gatzk/util/route2.hpp"

namespace
{
    namespace algebra = gatzk::algebra;
    namespace protocol = gatzk::protocol;
    namespace util = gatzk::util;
    using Field = algebra::FieldElement;

    constexpr unsigned int kStageTimeoutSeconds = 600;
    constexpr std::string_view kCoraDefaultRegistryDigest = "87eb1a4d793dfa677bbd8ae387292110a709ea54e7988cbdb5bfd551393bcc2e";
    constexpr std::string_view kCoraDefaultProtocolSchema = "2b6322e850c7fa7cee3f8b6891615a86bedc78d2e35c601f1e7fa1111138af30";
    constexpr std::string_view kCoraDefaultInstanceSchema = "91b082fe6ad3aa2deb559d7ec11a50267f6106101ceb9953f7dd0eca152c0b76";

    void timeout_handler(int)
    {
        constexpr char message[] = "protocol variant regression: stage exceeded hard 600 second limit\n";
        (void)::write(STDERR_FILENO, message, sizeof(message) - 1);
        _exit(124);
    }

    template <typename Fn> auto bounded_stage(std::string_view name, Fn &&fn)
    {
        const auto started = std::chrono::steady_clock::now();
        std::cerr << "[protocol-variant-test] stage:start " << name << " timeout_s=" << kStageTimeoutSeconds << "\n";
        std::signal(SIGALRM, timeout_handler);
        alarm(kStageTimeoutSeconds);
        try
        {
            auto result = fn();
            alarm(0);
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count();
            std::cerr << "[protocol-variant-test] stage:done " << name << " elapsed_ms=" << elapsed << "\n";
            return result;
        }
        catch (...)
        {
            alarm(0);
            throw;
        }
    }

    void require(bool condition, const std::string &message)
    {
        if (!condition)
            throw std::runtime_error(message);
    }

    std::string shell_quote(std::string_view value)
    {
        std::string out = "'";
        for (const char ch : value)
        {
            if (ch == '\'')
                out += "'\\''";
            else
                out.push_back(ch);
        }
        out.push_back('\'');
        return out;
    }

    void write_text(const std::filesystem::path &path, const std::string &contents)
    {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out)
            throw std::runtime_error("cannot write test fixture file: " + path.string());
        out.write(contents.data(), static_cast<std::streamsize>(contents.size()));
        if (!out)
            throw std::runtime_error("failed while writing test fixture file: " + path.string());
    }

    std::vector<std::uint8_t> read_bytes(const std::filesystem::path &path)
    {
        std::ifstream input(path, std::ios::binary);
        if (!input)
            throw std::runtime_error("cannot read test fixture file: " + path.string());
        return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }

    std::string sha256_hex(const std::vector<std::uint8_t> &bytes)
    {
        std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
        SHA256(bytes.data(), bytes.size(), digest.data());
        static constexpr char digits[] = "0123456789abcdef";
        std::string out(digest.size() * 2, '0');
        for (std::size_t i = 0; i < digest.size(); ++i)
        {
            out[2 * i] = digits[digest[i] >> 4U];
            out[2 * i + 1] = digits[digest[i] & 0x0fU];
        }
        return out;
    }

    std::string canonical_field_values_sha256(const std::vector<Field> &values)
    {
        constexpr std::size_t kFieldBytes = 32;
        if (values.size() > std::numeric_limits<std::size_t>::max() / kFieldBytes)
            throw std::runtime_error("canonical field sequence byte length overflows size_t");
        std::vector<std::uint8_t> bytes(values.size() * kFieldBytes);
        for (std::size_t index = 0; index < values.size(); ++index)
        {
            const auto written = values[index].write_little_endian(bytes.data() + index * kFieldBytes, kFieldBytes);
            require(written <= kFieldBytes, "canonical field value exceeds its fixed-width encoding at index " + std::to_string(index));
        }
        return sha256_hex(bytes);
    }

    std::string fixed_width_field_hex(const Field &value, const std::string &domain_name)
    {
        std::array<std::uint8_t, 32> bytes{};
        const auto written = value.write_little_endian(bytes.data(), bytes.size());
        require(written == bytes.size(), "test fixture domain root is not canonically full width: " + domain_name);
        static constexpr char digits[] = "0123456789abcdef";
        std::string out;
        out.reserve(bytes.size() * 2);
        for (const auto byte : bytes)
        {
            out.push_back(digits[(byte >> 4U) & 0x0fU]);
            out.push_back(digits[byte & 0x0fU]);
        }
        return out;
    }

    std::size_t next_power_of_two(std::size_t value)
    {
        if (value == 0)
            throw std::runtime_error("test domain size cannot be zero");
        --value;
        for (std::size_t shift = 1; shift < sizeof(value) * 8; shift <<= 1U)
            value |= value >> shift;
        if (value == std::numeric_limits<std::size_t>::max())
            throw std::runtime_error("test domain size overflow");
        return value + 1;
    }

    struct FixtureDimensions
    {
        std::array<std::pair<std::string, std::size_t>, 7> domains;
        std::size_t degree_bound = 0;
        std::size_t max_coefficient_length = 0;
    };

    FixtureDimensions fixture_dimensions(const util::AppConfig &config)
    {
        constexpr std::size_t nodes = 4;
        constexpr std::size_t features = 500;
        constexpr std::size_t classes = 3;
        constexpr std::size_t hidden_width = 8;
        constexpr std::size_t concat_width = 64;
        constexpr std::size_t feature_rows = nodes * features;

        const auto table = protocol::plan_canonical_static_table_shape(protocol::CanonicalStaticTableShapeInputs{
            .range_bits = config.range_bits,
            .quantization = config.quantization,
            .batching_rule = config.batching_rule,
            .dataset_num_nodes = nodes,
            .dataset_num_features = features,
            .local_num_nodes = nodes,
            .local_edge_count = 12,
        });
        const auto edge_rows = std::max({protocol::minimum_edge_trace_length(nodes, 12, hidden_width, classes), table.range_table_size,
                                         table.lrelu_table_size, table.elu_table_size, table.exp_table_size});
        const auto fh = next_power_of_two(feature_rows + 2);
        const auto edge = next_power_of_two(edge_rows + 2);
        const auto in = next_power_of_two(std::max(features, config.d_in_profile.at(1)) + 2);
        const auto d = next_power_of_two(hidden_width + 2);
        const auto cat = next_power_of_two(concat_width + 2);
        const auto c = next_power_of_two(classes + 2);
        const auto n = next_power_of_two(nodes + 2);
        const auto max_domain = std::max({fh, edge, in, d, cat, c, n});
        if (max_domain > std::numeric_limits<std::size_t>::max() / 4)
            throw std::runtime_error("test quotient coefficient count overflows size_t");

        return FixtureDimensions{
            .domains = {{{"C", c}, {"FH", fh}, {"N", n}, {"cat", cat}, {"d", d}, {"edge", edge}, {"in", in}}},
            .degree_bound = max_domain * 4,
            .max_coefficient_length = std::max({feature_rows, features * hidden_width, concat_width * classes, nodes * classes})};
    }

    void create_mini_pubmed_cache(const std::filesystem::path &cache_root)
    {
        const auto dataset = cache_root / "pubmed";
        std::filesystem::create_directories(dataset);
        write_text(dataset / "meta.cfg",
                   "name=pubmed\nnum_nodes=4\nnum_features=500\nnum_classes=3\ngraph_count=1\n"
                   "task_type=transductive_node_classification\nreport_unit=node\ninput_preprocess_id=row_normalize\n");
        std::ostringstream features;
        features << "4 500\n";
        features << "0:1.0 1:0.5 17:0.25\n";
        features << "0:0.25 2:1.0 499:0.5\n";
        features << "7:0.75 33:0.5 499:0.125\n";
        features << "1:0.25 256:0.875 498:0.375\n";
        write_text(dataset / "features.txt", features.str());
        write_text(dataset / "edges.txt", "0 1\n1 0\n1 2\n2 1\n2 3\n3 2\n3 0\n0 3\n");
        write_text(dataset / "labels.txt", "0\n1\n2\n0\n");
    }

    void validate_test_only_srs(const std::filesystem::path &srs_dir, const std::string &srs_id, const FixtureDimensions &dimensions)
    {
        const auto audited_keys = gatzk::crypto::KZG::load_keys(srs_dir.string(), "trusted_srs", srs_id);
        require(audited_keys.manifest.root_derivation_id == "bls12_381_fr_generator_5", "test-only SRS uses unexpected domain-root derivation");
        require(audited_keys.manifest.degree_bound == dimensions.degree_bound, "test-only SRS degree bound differs from the fixture plan");
        require(audited_keys.manifest.domain_sizes.size() == dimensions.domains.size(), "test-only SRS domain count differs from the fixture plan");
        for (const auto &[name, size] : dimensions.domains)
        {
            const auto manifest_size = audited_keys.manifest.domain_sizes.find(name);
            require(manifest_size != audited_keys.manifest.domain_sizes.end() && manifest_size->second == size,
                    "test-only SRS domain size differs from the fixture plan: " + name);
            const auto expected = fixed_width_field_hex(Field::root_of_unity(size), name);
            const auto manifest_root = audited_keys.manifest.domain_generators.find(name);
            require(manifest_root != audited_keys.manifest.domain_generators.end() && manifest_root->second == expected,
                    "test-only SRS manifest domain root does not match full-width native generator: " + name);
        }
    }

    void create_test_only_trusted_srs(const std::filesystem::path &root, const util::AppConfig &config)
    {
        const auto dimensions = fixture_dimensions(config);
        const auto srs_parent = root / "srs";
        const auto compiler_artifact = root / "protocol_compiler_artifact.txt";
        const auto envelope = root / "max_instance_envelope.txt";
        const std::string srs_id = "protocol_variant_native_test_only";
        const auto srs_dir = srs_parent / srs_id;

        std::ostringstream envelope_bytes;
        envelope_bytes << "format=zkgat_max_instance_envelope\n"
                       << "max_layers=3\n"
                       << "max_hidden_heads_per_layer=8\n"
                       << "max_hidden_head_dim=8\n"
                       << "max_input_dim=500\n"
                       << "max_output_heads=2\n"
                       << "max_classes=3\n"
                       << "max_graphs=1\n"
                       << "max_nodes=4\n"
                       << "max_edges=16\n"
                       << "max_total_nodes=4\n"
                       << "max_concat_dim=64\n"
                       << "max_range_bits=12\n"
                       << "max_coefficient_object_length=" << std::max<std::size_t>(8192, dimensions.max_coefficient_length) << "\n"
                       << "max_object_family_count=100000\n"
                       << "max_coefficient_chunks=2\n"
                       << "security_target_bits=120\n";
        write_text(envelope, envelope_bytes.str());
        std::filesystem::create_directories(srs_parent);

        std::ostringstream compile_command;
        compile_command << "timeout --signal=TERM --kill-after=5s " << kStageTimeoutSeconds << "s env GATZK_SRS_WORKERS=8 "
                        << shell_quote(GATZK_PROTOCOL_VARIANT_TEST_SRS_SETUP) << " compile " << shell_quote(compiler_artifact.string()) << ' '
                        << dimensions.degree_bound << ' ' << shell_quote(envelope.string());
        for (const auto &[name, size] : dimensions.domains)
            compile_command << ' ' << name << '=' << size;
        const auto compile_status = std::system(compile_command.str().c_str());
        if (compile_status == -1 || !WIFEXITED(compile_status) || WEXITSTATUS(compile_status) != 0)
            throw std::runtime_error("test-only SRS compiler artifact generation failed");

        std::ostringstream setup_command;
        setup_command << "timeout --signal=TERM --kill-after=5s " << kStageTimeoutSeconds << "s env GATZK_SRS_WORKERS=8 "
                      << shell_quote(GATZK_PROTOCOL_VARIANT_TEST_SRS_SETUP) << " ceremony " << shell_quote(srs_dir.string()) << ' '
                      << shell_quote(compiler_artifact.string()) << ' ' << srs_id;
        const auto setup_status = std::system(setup_command.str().c_str());
        if (setup_status == -1 || !WIFEXITED(setup_status) || WEXITSTATUS(setup_status) != 0)
            throw std::runtime_error("test-only native trusted KZG SRS setup/audit failed");

        validate_test_only_srs(srs_dir, srs_id, dimensions);

        std::cerr << "[protocol-variant-test] test-only-SRS domains FH=" << dimensions.domains[1].second << " edge=" << dimensions.domains[5].second
                  << " in=" << dimensions.domains[6].second << " d=" << dimensions.domains[4].second << " cat=" << dimensions.domains[3].second
                  << " C=" << dimensions.domains[0].second << " N=" << dimensions.domains[2].second << " degree_bound=" << dimensions.degree_bound
                  << " estimated_g1_points=" << dimensions.degree_bound + 1 << "+"
                  << (dimensions.domains[0].second + dimensions.domains[1].second + dimensions.domains[2].second + dimensions.domains[3].second +
                      dimensions.domains[4].second + dimensions.domains[5].second + dimensions.domains[6].second)
                  << "\n";
    }

    void check_default_cora_identity_snapshot(const std::filesystem::path &project_root)
    {
        const auto registry_path = project_root / "artifacts/verifier_registry/cora/verifier_registry.txt";
        const auto bytes = bounded_stage("default_cora_identity_snapshot_parse", [&] { return read_bytes(registry_path); });
        require(sha256_hex(bytes) == kCoraDefaultRegistryDigest, "default Cora verifier registry digest changed");
        const auto artifact = bounded_stage("default_cora_identity_snapshot_decode", [&]
                                            { return protocol::parse_verifier_registration_artifact(bytes, std::string(kCoraDefaultRegistryDigest)); });
        require(artifact.config.protocol_variant == "zkgat", "default Cora registry is not bound to zkgat");
        require(artifact.registry.protocol_schema_id == kCoraDefaultProtocolSchema, "default Cora protocol schema identity changed");
        require(artifact.registry.instance_schema_hash == kCoraDefaultInstanceSchema, "default Cora instance schema identity changed");
    }

    std::vector<std::string> decode_canonical_string_sequence(std::string_view bytes)
    {
        std::vector<std::string> fields;
        std::size_t position = 0;
        while (position < bytes.size())
        {
            if (bytes.size() - position < sizeof(std::uint64_t))
                throw std::runtime_error("truncated canonical string sequence");
            std::uint64_t length = 0;
            for (std::size_t i = 0; i < sizeof(std::uint64_t); ++i)
                length |= static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[position + i])) << (8U * i);
            position += sizeof(std::uint64_t);
            if (length > bytes.size() - position)
                throw std::runtime_error("canonical string sequence length exceeds remaining data");
            fields.emplace_back(bytes.substr(position, static_cast<std::size_t>(length)));
            position += static_cast<std::size_t>(length);
        }
        return fields;
    }

    std::string encode_canonical_string_sequence(const std::vector<std::string> &fields)
    {
        std::string bytes;
        for (const auto &field : fields)
        {
            const auto length = static_cast<std::uint64_t>(field.size());
            for (std::size_t i = 0; i < sizeof(length); ++i)
                bytes.push_back(static_cast<char>((length >> (8U * i)) & 0xffU));
            bytes.append(field);
        }
        return bytes;
    }

    std::string remove_one_route_terminal(std::string_view encoded, std::string_view terminal_id)
    {
        auto fields = decode_canonical_string_sequence(encoded);
        const auto section = std::find(fields.begin(), fields.end(), "route_terminal_relations");
        if (section == fields.end() || std::distance(section, fields.end()) < 3)
            throw std::runtime_error("binding relation has no route terminal catalog");
        const auto section_index = static_cast<std::size_t>(std::distance(fields.begin(), section));
        const auto count_index = section_index + 1;
        const auto count = static_cast<std::size_t>(std::stoull(fields.at(count_index)));
        constexpr std::size_t kTerminalFieldCount = 13;
        const auto first_terminal = section_index + 2;
        for (std::size_t index = 0; index < count; ++index)
        {
            const auto terminal_begin = first_terminal + index * kTerminalFieldCount;
            if (terminal_begin + kTerminalFieldCount > fields.size())
                throw std::runtime_error("route terminal catalog is truncated");
            if (fields[terminal_begin] == terminal_id)
            {
                fields.erase(fields.begin() + static_cast<std::ptrdiff_t>(terminal_begin),
                             fields.begin() + static_cast<std::ptrdiff_t>(terminal_begin + kTerminalFieldCount));
                fields[count_index] = std::to_string(count - 1);
                return encode_canonical_string_sequence(fields);
            }
        }
        throw std::runtime_error("requested route terminal was not found: " + std::string(terminal_id));
    }

    void validate_shared_terminal_catalog(const protocol::ProtocolContext &context)
    {
        const auto metadata = protocol::build_binding_relation_metadata(context, *context.compiler_artifact);
        for (std::size_t layer = 0; layer < context.model.hidden_layers.size(); ++layer)
        {
            for (std::size_t head = 0; head < context.model.hidden_layers[layer].heads.size(); ++head)
            {
                const auto target = "route-terminal:hidden:l" + std::to_string(layer) + ":h" + std::to_string(head) + ":t_den";
                const auto parent = std::find_if(metadata.begin(), metadata.end(), [&](const auto &entry)
                                                 { return std::any_of(entry.route_terminal_relations.begin(), entry.route_terminal_relations.end(),
                                                                      [&](const auto &route) { return route.relation_id == target; }); });
                require(parent != metadata.end(), "shared hidden denominator terminal is missing: " + target);
                const auto route_prefix = "route-terminal:hidden:l" + std::to_string(layer) + ":h" + std::to_string(head) + ":";
                const auto target_route_count = std::count_if(parent->route_terminal_relations.begin(), parent->route_terminal_relations.end(),
                                                              [&](const auto &route)
                                                              { return route.relation_id.compare(0, route_prefix.size(), route_prefix) == 0; });
                require(target_route_count == 4, "shared hidden route must have exactly four destination-group terminals per head");
                const std::array<std::string, 4> roles = {"src", "dst", "t_den", "t_agg"};
                for (const auto &role : roles)
                {
                    const auto route_id = "route-terminal:hidden:l" + std::to_string(layer) + ":h" + std::to_string(head) + ":" + role;
                    require(std::any_of(parent->route_terminal_relations.begin(), parent->route_terminal_relations.end(),
                                        [&](const auto &route) { return route.relation_id == route_id; }),
                            "shared hidden route terminal role is missing: " + route_id);
                }
            }
        }
        for (std::size_t head = 0; head < context.model.output_layer.heads.size(); ++head)
        {
            const auto target = "route-terminal:output:" + std::to_string(head) + ":t_den";
            const auto parent = std::find_if(metadata.begin(), metadata.end(), [&](const auto &entry)
                                             { return std::any_of(entry.route_terminal_relations.begin(), entry.route_terminal_relations.end(),
                                                                  [&](const auto &route) { return route.relation_id == target; }); });
            require(parent != metadata.end(), "shared output denominator terminal is missing: " + target);
            const auto route_prefix = "route-terminal:output:" + std::to_string(head) + ":";
            const auto target_route_count = std::count_if(parent->route_terminal_relations.begin(), parent->route_terminal_relations.end(),
                                                          [&](const auto &route)
                                                          { return route.relation_id.compare(0, route_prefix.size(), route_prefix) == 0; });
            require(target_route_count == 4, "shared output route must have exactly four destination-group terminals per head");
            for (const auto &role : {"src", "dst", "t_den", "t_agg"})
            {
                const auto route_id = "route-terminal:output:" + std::to_string(head) + ":" + role;
                require(std::any_of(parent->route_terminal_relations.begin(), parent->route_terminal_relations.end(),
                                    [&](const auto &route) { return route.relation_id == route_id; }),
                        "shared output route terminal role is missing: " + route_id);
            }
        }
    }

    void validate_feature_copy_catalog(const protocol::ProtocolContext &context)
    {
        const auto metadata = protocol::build_binding_relation_metadata(context, *context.compiler_artifact);
        const auto feature_routes = protocol::build_feature_copy_route_metadata(context);
        for (std::size_t layer = 0; layer < context.model.hidden_layers.size(); ++layer)
        {
            const auto &layer_shape = context.model.hidden_layers[layer];
            for (std::size_t head = 0; head < layer_shape.heads.size(); ++head)
            {
                const auto relation_id = "bind:hidden:" + std::to_string(layer + 1) + ":" + std::to_string(head);
                const auto relation_it = std::find_if(metadata.begin(), metadata.end(), [&](const auto &entry) { return entry.relation_id == relation_id; });
                require(relation_it != metadata.end(), "feature-copy owner relation is missing: " + relation_id);
                const auto head_dim = layer_shape.shape.head_dim;
                const auto head_prefix = protocol::hidden_head_prefix(layer, head);
                std::vector<const protocol::FeatureCopyRouteRelationMeta *> head_routes;
                for (const auto &route : feature_routes)
                {
                    if (route.source_matrix_object == head_prefix + "_H_prime")
                        head_routes.push_back(&route);
                }
                require(head_routes.size() == head_dim, "feature-copy route count differs from head dimension: " + relation_id);
                std::vector<bool> coordinates_seen(head_dim, false);
                for (const auto *route_pointer : head_routes)
                {
                    const auto &route = *route_pointer;
                    require(route.coordinate < head_dim, "feature-copy coordinate exceeds head dimension: " + route.relation_id);
                    require(!coordinates_seen[route.coordinate], "duplicate feature-copy coordinate route: " + route.relation_id);
                    coordinates_seen[route.coordinate] = true;
                    const auto r = route.coordinate;
                    const auto route_prefix = protocol::hidden_feature_route_prefix(layer, head, r);
                    const auto source_column = head_prefix + "_H_prime_CanCol" + std::to_string(r);
                    const auto expected_relation_id = "feature-copy:hidden:l" + std::to_string(layer) + ":h" + std::to_string(head) + ":r" +
                                                      std::to_string(r);
                    const auto terminal_id = "route-terminal:hidden:l" + std::to_string(layer) + ":h" + std::to_string(head) + ":feature:r" +
                                             std::to_string(r);
                    require(route.relation_id == expected_relation_id && route.source_matrix_object == head_prefix + "_H_prime" &&
                                route.source_column_object == source_column && route.edge_value_object == route_prefix + "_H_edge" &&
                                route.node_index_object == "P_I" && route.edge_index_object == "P_src",
                            "feature-copy route is not bound to the canonical source column and edge coordinate: " + route_prefix);
                    require(std::count_if(feature_routes.begin(), feature_routes.end(), [&](const auto &candidate)
                                          { return candidate.relation_id == route.relation_id; }) == 1,
                            "feature-copy relation identifier is not unique: " + route.relation_id);
                    require(std::count_if(feature_routes.begin(), feature_routes.end(), [&](const auto &candidate)
                                          { return candidate.source_column_object == route.source_column_object; }) == 1 &&
                                std::count_if(feature_routes.begin(), feature_routes.end(), [&](const auto &candidate)
                                              { return candidate.edge_value_object == route.edge_value_object; }) == 1 &&
                                std::count_if(feature_routes.begin(), feature_routes.end(), [&](const auto &candidate)
                                              { return candidate.table_object == route.table_object; }) == 1 &&
                                std::count_if(feature_routes.begin(), feature_routes.end(), [&](const auto &candidate)
                                              { return candidate.query_object == route.query_object; }) == 1 &&
                                std::count_if(feature_routes.begin(), feature_routes.end(), [&](const auto &candidate)
                                              { return candidate.eta_challenge_label == route.eta_challenge_label; }) == 1 &&
                                std::count_if(feature_routes.begin(), feature_routes.end(), [&](const auto &candidate)
                                              { return candidate.beta_challenge_label == route.beta_challenge_label; }) == 1 &&
                                std::count_if(feature_routes.begin(), feature_routes.end(), [&](const auto &candidate)
                                              { return candidate.terminal_relation_id == route.terminal_relation_id; }) == 1,
                            "feature-copy coordinate does not map uniquely to its source/edge/lookup/challenge/terminal labels: " + route.relation_id);
                    require(route.table_object == route_prefix + "_Table" && route.query_object == route_prefix + "_Query" &&
                                route.multiplicity_object == head_prefix + "_m_src" && route.node_inverse_object == route_prefix + "_J_node" &&
                                route.edge_inverse_object == route_prefix + "_J_edge" && route.node_accumulator_object == route_prefix + "_R_node" &&
                                route.edge_accumulator_object == route_prefix + "_R_edge",
                            "feature-copy route is missing a required independent source lookup proof object: " + route_prefix);
                    require(route.eta_challenge_label == protocol::hidden_feature_route_challenge_name("eta_src_feat_h", layer, head, r) &&
                                route.beta_challenge_label == protocol::hidden_feature_route_challenge_name("beta_src_feat_h", layer, head, r) &&
                                route.terminal_relation_id == terminal_id,
                            "feature-copy route challenge or terminal identity mismatch: " + route_prefix);
                    require(route.node_domain == context.domains.n->name && route.edge_domain == context.domains.edge->name &&
                                route.row_count == context.local.num_nodes && route.edge_count == context.local.edges.size(),
                            "feature-copy route dimensions/domain differ from the shared fixture: " + route_prefix);

                    const auto serialized_route = protocol::feature_copy_route_registry_entry(route);
                    require(std::count(context.compiler_artifact->binding_relation_registry.begin(),
                                       context.compiler_artifact->binding_relation_registry.end(), serialized_route) == 1,
                            "feature-copy coordinate must have exactly one complete authenticated compiler-instance registry entry: " +
                                route.relation_id);

                    require(std::count_if(metadata.begin(), metadata.end(), [&](const auto &entry)
                                          { return std::count_if(entry.route_terminal_relations.begin(), entry.route_terminal_relations.end(),
                                                                 [&](const auto &terminal) { return terminal.relation_id == terminal_id; }) != 0; }) == 1,
                            "feature-copy route terminal does not have exactly one owning relation: " + terminal_id);
                    require(std::count_if(relation_it->route_terminal_relations.begin(), relation_it->route_terminal_relations.end(),
                                          [&](const auto &terminal) { return terminal.relation_id == terminal_id; }) == 1,
                            "feature-copy route terminal is not uniquely owned by its hidden binding relation: " + terminal_id);
                    const auto terminal = std::find_if(relation_it->route_terminal_relations.begin(), relation_it->route_terminal_relations.end(),
                                                       [&](const auto &entry) { return entry.relation_id == terminal_id; });
                    require(terminal->node_accumulator_object == route.node_accumulator_object &&
                                terminal->edge_accumulator_object == route.edge_accumulator_object &&
                                terminal->shifted_accumulator_object == route_prefix + "_R_edge_RouteShifted" &&
                                terminal->relation_virtual_object == route_prefix + "_R_node_RouteTerminalVirtual",
                            "feature-copy terminal does not close the typed coordinate route: " + terminal_id);
                }
                require(std::all_of(coordinates_seen.begin(), coordinates_seen.end(), [](bool seen) { return seen; }),
                        "feature-copy catalog omits a hidden coordinate: " + relation_id);
            }
        }
    }

    void reject_missing_shared_terminal(protocol::ProtocolContext &context, const protocol::VerifierContext &verifier_context)
    {
        const auto metadata = protocol::build_binding_relation_metadata(context, *context.compiler_artifact);
        const std::string target = "route-terminal:hidden:l0:h0:t_den";
        const auto metadata_it = std::find_if(metadata.begin(), metadata.end(), [&](const auto &entry)
                                              { return std::any_of(entry.route_terminal_relations.begin(), entry.route_terminal_relations.end(),
                                                                   [&](const auto &route) { return route.relation_id == target; }); });
        require(metadata_it != metadata.end(), "target hidden denominator terminal missing before negative test");
        const auto relation_index = static_cast<std::size_t>(std::distance(metadata.begin(), metadata_it));

        auto malformed_compiler = *context.compiler_artifact;
        malformed_compiler.binding_relation_registry.at(relation_index) =
            remove_one_route_terminal(malformed_compiler.binding_relation_registry.at(relation_index), target);
        std::string reason;
        require(!protocol::validate_protocol_compiler_artifact(context, malformed_compiler, &reason),
                "compiler accepted a binding catalog with one required route terminal removed");

        auto malformed_verifier = verifier_context;
        malformed_verifier.compiler_artifact = std::make_shared<const protocol::ProtocolCompilerArtifact>(std::move(malformed_compiler));
        require(!protocol::validate_verifier_context(malformed_verifier), "independent registry context accepted a missing route terminal");
    }

    void reject_missing_feature_coordinate(protocol::ProtocolContext &context, const protocol::VerifierContext &verifier_context,
                                           const std::filesystem::path &temp_root)
    {
        require(!context.model.hidden_layers.empty() && !context.model.hidden_layers.front().heads.empty(),
                "feature-coordinate negative test requires one hidden head");
        const std::size_t layer = 0;
        const std::size_t head = 0;
        const auto feature_routes = protocol::build_feature_copy_route_metadata(context);
        const auto target_coordinate = context.model.hidden_layers[layer].shape.head_dim / 2;
        require(target_coordinate < context.model.hidden_layers[layer].shape.head_dim, "feature-coordinate target is outside head dimension");
        const auto head_prefix = protocol::hidden_head_prefix(layer, head);
        const auto target_route = std::find_if(feature_routes.begin(), feature_routes.end(), [&](const auto &route)
                                               { return route.source_matrix_object == head_prefix + "_H_prime" && route.coordinate == target_coordinate; });
        require(target_route != feature_routes.end(), "feature-copy target route is missing before coordinate negative test");

        auto malformed_compiler = *context.compiler_artifact;
        const auto target_entry = protocol::feature_copy_route_registry_entry(*target_route);
        const auto target_entry_it = std::find(malformed_compiler.binding_relation_registry.begin(),
                                               malformed_compiler.binding_relation_registry.end(), target_entry);
        require(target_entry_it != malformed_compiler.binding_relation_registry.end() &&
                    std::count(malformed_compiler.binding_relation_registry.begin(), malformed_compiler.binding_relation_registry.end(), target_entry) == 1,
                "feature-copy coordinate is not one independently registered typed compiler entry");
        malformed_compiler.binding_relation_registry.erase(target_entry_it);
        std::string reason;
        require(!protocol::validate_protocol_compiler_artifact(context, malformed_compiler, &reason),
                "compiler accepted a typed binding catalog with one feature coordinate omitted");

        auto malformed_verifier = verifier_context;
        malformed_verifier.compiler_artifact = std::make_shared<const protocol::ProtocolCompilerArtifact>(malformed_compiler);
        require(!protocol::validate_verifier_context(malformed_verifier), "independent verifier context accepted a missing feature coordinate");

        auto malformed_registration = protocol::make_verifier_registration_artifact(context);
        malformed_registration.compiler_artifact = std::move(malformed_compiler);
        auto malformed_bytes = protocol::serialize_verifier_registration_artifact(malformed_registration);
        const auto malformed_digest = sha256_hex(malformed_bytes);
        const auto malformed_path = temp_root / "registry-missing-feature-coordinate.bin";
        write_text(malformed_path, std::string(reinterpret_cast<const char *>(malformed_bytes.data()), malformed_bytes.size()));
        bool registry_rejected = false;
        try
        {
            auto loaded = protocol::load_verifier_context_from_registry(malformed_path.string(), malformed_digest);
            registry_rejected = !protocol::validate_verifier_context(loaded);
        }
        catch (const std::exception &)
        {
            registry_rejected = true;
        }
        require(registry_rejected, "independent registry loader accepted a missing feature coordinate");
    }

    struct VariantFixture
    {
        protocol::RunMetrics metrics;
        protocol::ProtocolContext prover;
        protocol::VerifierContext verifier;
        std::filesystem::path registry_path;
        std::string registry_digest;

        explicit VariantFixture(util::AppConfig config) : metrics(), prover(protocol::build_context(config, &metrics, false))
        {
            registry_path = config.verifier_registry_path;
            const auto registration_before = protocol::make_verifier_registration_artifact(prover);
            const auto bytes_before = protocol::serialize_verifier_registration_artifact(registration_before);

            const std::array<std::string_view, 5> canonical_query_aliases = {
                "P_Q_qry_src", "P_Q_qry_dst", "P_Q_qry_L", "P_Q_qry_R", "P_Q_qry_exp"};
            for (const auto alias_name_view : canonical_query_aliases)
            {
                const std::string alias_name(alias_name_view);
                auto polynomial_it = prover.public_polynomials.find(alias_name);
                require(polynomial_it != prover.public_polynomials.end(), "canonical query alias is missing: " + alias_name);
                auto &polynomial = polynomial_it->second;
                require(polynomial.basis == algebra::PolynomialBasis::Evaluation && polynomial.domain != nullptr && polynomial.shared_data != nullptr,
                        "canonical query alias is not shared evaluation storage: " + alias_name);

                const auto original_values = polynomial.values();
                const auto original_value_hash = canonical_field_values_sha256(original_values);
                const auto original_domain = polynomial.domain;
                const auto original_basis = polynomial.basis;
                const auto original_blinder = polynomial.zero_domain_blinder;
                const auto original_cuda_storage = polynomial.cuda_resident_coefficients;
                const auto owner_it = prover.public_polynomials.find("P_Q_edge_valid");
                require(owner_it != prover.public_polynomials.end() && owner_it->second.shared_data != nullptr &&
                            polynomial.shared_data.get() == owner_it->second.shared_data.get(),
                        "test fixture unexpectedly changed canonical query alias storage ownership: " + alias_name);
                const auto commitment_it = prover.public_commitments.find(alias_name);
                require(commitment_it != prover.public_commitments.end() && commitment_it->second.opening_polynomial != nullptr,
                        "canonical query alias commitment material is missing: " + alias_name);
                const auto commitment_values = commitment_it->second.opening_polynomial->values();
                const auto commitment_value_hash = canonical_field_values_sha256(commitment_values);
                require(commitment_values == original_values && commitment_value_hash == original_value_hash,
                        "canonical query alias differs from its committed public evaluation: " + alias_name);

                require(polynomial.values() == original_values && canonical_field_values_sha256(polynomial.values()) == original_value_hash,
                        "canonical query alias values/hash changed during fixture verification: " + alias_name);
                require(polynomial.domain == original_domain && polynomial.basis == original_basis &&
                            polynomial.zero_domain_blinder == original_blinder && polynomial.cuda_resident_coefficients == original_cuda_storage,
                        "canonical query alias metadata changed during fixture verification: " + alias_name);
                const auto &committed_polynomial = *commitment_it->second.opening_polynomial;
                require(committed_polynomial.values() == commitment_values && canonical_field_values_sha256(committed_polynomial.values()) == commitment_value_hash,
                        "canonical query alias public commitment material changed during fixture verification: " + alias_name);
                std::cerr << "[protocol-variant-test] canonical_alias_verified variant=" << config.protocol_variant << " polynomial=" << alias_name
                          << " shared_storage_owner=P_Q_edge_valid value_sha256=" << original_value_hash << "\n";
            }

            auto registration = protocol::make_verifier_registration_artifact(prover);
            auto bytes = protocol::serialize_verifier_registration_artifact(registration);
            require(bytes == bytes_before, "canonical query alias fixture checks changed verifier registration for " + config.protocol_variant);
            require(registration.registry.protocol_schema_id == registration_before.registry.protocol_schema_id &&
                        registration.registry.instance_schema_hash == registration_before.registry.instance_schema_hash,
                    "canonical query alias fixture checks changed protocol hashes for " + config.protocol_variant);
            std::cerr << "[protocol-variant-test] canonical_alias_identity variant=" << config.protocol_variant
                      << " registry_sha256=" << sha256_hex(bytes) << "\n";
            registry_digest = sha256_hex(bytes);
            write_text(registry_path, std::string(reinterpret_cast<const char *>(bytes.data()), bytes.size()));
            verifier = protocol::load_verifier_context_from_registry(registry_path.string(), registry_digest);
            require(protocol::validate_verifier_context(verifier), "independent verifier context validation failed for " + config.protocol_variant);
            require(verifier.config.protocol_variant == config.protocol_variant, "registry protocol identity mismatch for " + config.protocol_variant);
        }
    };

    VariantFixture make_variant_fixture(const util::AppConfig &base, const std::filesystem::path &root, const std::string &variant)
    {
        auto config = base;
        config.protocol_variant = variant;
        config.verifier_registry_path = (root / ("registry-" + variant + ".bin")).string();
        config.verifier_registry_digest.clear();
        return VariantFixture(std::move(config));
    }

    std::vector<std::uint8_t> prove_roundtrip(VariantFixture &fixture, const std::filesystem::path &proof_path, const std::string &label,
                                              const std::function<void(const protocol::TraceArtifacts &)> &trace_validator = {})
    {
        auto trace = bounded_stage(label + ":trace", [&] { return protocol::build_trace(fixture.prover, &fixture.metrics); });
        if (trace_validator)
            trace_validator(trace);
        auto proof = bounded_stage(label + ":prove", [&] { return protocol::prove(fixture.prover, trace, &fixture.metrics); });
        require(proof.public_metadata.protocol_name == fixture.prover.config.protocol_variant, "proof metadata protocol identity mismatch");
        auto bytes = bounded_stage(label + ":proof_serialize", [&] { return protocol::serialize_proof(proof); });
        require(!bytes.empty(), "native proof serialization is empty");
        write_text(proof_path, std::string(reinterpret_cast<const char *>(bytes.data()), bytes.size()));
        require(std::filesystem::file_size(proof_path) == bytes.size(), "native proof artifact size mismatch");
        const auto parsed = bounded_stage(label + ":independent_parse", [&] { return protocol::parse_proof(fixture.verifier, bytes); });
        require(bounded_stage(label + ":independent_verify", [&] { return protocol::verify(fixture.verifier, parsed); }),
                "independent native KZG verifier rejected valid proof for " + fixture.prover.config.protocol_variant);
        return bytes;
    }

    void require_cross_variant_rejection(const protocol::VerifierContext &verifier, const std::vector<std::uint8_t> &proof_bytes,
                                         const std::string &target_variant)
    {
        bool accepted = false;
        try
        {
            const auto parsed = protocol::parse_proof(verifier, proof_bytes);
            accepted = protocol::verify(verifier, parsed);
        }
        catch (const std::exception &)
        {
            accepted = false;
        }
        require(!accepted, "proof from another protocol variant was accepted by " + target_variant + " verifier");
    }

    void mutate_committed_evaluation(protocol::TraceArtifacts *trace, const std::string &label)
    {
        auto polynomial_it = trace->polynomials.find(label);
        auto commitment_it = trace->commitments.find(label);
        if (polynomial_it == trace->polynomials.end() || commitment_it == trace->commitments.end())
            throw std::runtime_error("negative-test route witness is not retained in memory: " + label);
        const auto &source = polynomial_it->second;
        if (source.basis != algebra::PolynomialBasis::Evaluation || source.domain == nullptr || source.values().empty())
            throw std::runtime_error("negative-test target is not a retained evaluation polynomial: " + label);
        auto values = source.values();
        values.front() += Field::one();
        auto corrupted = std::make_shared<algebra::Polynomial>(algebra::Polynomial::from_evaluations(label, std::move(values), source.domain));
        corrupted->set_zero_domain_blinder(source.zero_domain_blinder);
        polynomial_it->second = *corrupted;
        commitment_it->second.opening_polynomial = std::move(corrupted);
    }

    void prove_and_reject_mutated_witness(VariantFixture &fixture, const std::string &target, const std::filesystem::path &proof_path)
    {
        auto trace = bounded_stage(target + ":trace", [&] { return protocol::build_trace(fixture.prover, &fixture.metrics); });
        mutate_committed_evaluation(&trace, target);
        auto proof = bounded_stage(target + ":prove", [&] { return protocol::prove(fixture.prover, trace, &fixture.metrics); });
        const auto bytes = bounded_stage(target + ":serialize", [&] { return protocol::serialize_proof(proof); });
        write_text(proof_path, std::string(reinterpret_cast<const char *>(bytes.data()), bytes.size()));
        const auto parsed = bounded_stage(target + ":independent_parse", [&] { return protocol::parse_proof(fixture.verifier, bytes); });
        require(!bounded_stage(target + ":independent_verify", [&] { return protocol::verify(fixture.verifier, parsed); }),
                "independent verifier accepted mutated destination-group witness: " + target);
    }

    std::optional<std::size_t> validate_feature_coordinate_source_tuple_mismatch(const protocol::ProtocolContext &context,
                                                                                 const protocol::TraceArtifacts &trace)
    {
        require(!context.model.hidden_layers.empty() && !context.model.hidden_layers.front().heads.empty(),
                "feature-coordinate tuple test requires a hidden head");
        const auto routes = protocol::build_feature_copy_route_metadata(context);
        const auto head_dim = context.model.hidden_layers.front().shape.head_dim;
        const auto target_coordinate = head_dim / 2;
        const auto source_matrix = protocol::hidden_head_prefix(0, 0) + "_H_prime";
        const auto target_route = std::find_if(routes.begin(), routes.end(), [&](const auto &route)
                                               { return route.source_matrix_object == source_matrix && route.coordinate == target_coordinate; });
        require(target_route != routes.end(), "feature-coordinate tuple test target route is missing");
        const auto target_source = protocol::resolve_trace_opening_polynomial(trace, target_route->source_column_object, context.domains.n);
        const auto target_edge = protocol::resolve_trace_opening_polynomial(trace, target_route->edge_value_object, context.domains.edge);
        const auto node_index = context.public_polynomials.find(target_route->node_index_object);
        const auto edge_index_polynomial = context.public_polynomials.find(target_route->edge_index_object);
        require(target_source != nullptr && target_edge != nullptr && target_source->basis == algebra::PolynomialBasis::Evaluation &&
                    target_edge->basis == algebra::PolynomialBasis::Evaluation && node_index != context.public_polynomials.end() &&
                    edge_index_polynomial != context.public_polynomials.end() && node_index->second.basis == algebra::PolynomialBasis::Evaluation &&
                    edge_index_polynomial->second.basis == algebra::PolynomialBasis::Evaluation,
                "feature-coordinate tuple test could not resolve spilled canonical/edge/index evaluations");
        const auto &target_source_values = target_source->values();
        const auto &target_edge_values = target_edge->values();
        const auto &node_index_values = node_index->second.values();
        const auto &edge_index_values = edge_index_polynomial->second.values();
        for (std::size_t edge = 0; edge < context.local.edges.size(); ++edge)
        {
            const auto source = context.local.edges[edge].src;
            require(source < target_source_values.size() && source < node_index_values.size() && edge < target_edge_values.size() &&
                        edge < edge_index_values.size() && node_index_values[source] == edge_index_values[edge] &&
                        target_edge_values[edge] == target_source_values[source],
                    "feature-coordinate trace does not route the canonical source index/value tuple to its edge witness");
        }

        for (const auto &route : routes)
        {
            if (route.source_matrix_object != source_matrix || route.coordinate == target_coordinate)
                continue;
            const auto alternate_source = protocol::resolve_trace_opening_polynomial(trace, route.source_column_object, context.domains.n);
            require(alternate_source != nullptr && alternate_source->basis == algebra::PolynomialBasis::Evaluation,
                    "feature-coordinate tuple test could not resolve alternate canonical source coordinate");
            const auto &alternate_values = alternate_source->values();
            for (std::size_t edge = 0; edge < context.local.edges.size(); ++edge)
            {
                const auto source = context.local.edges[edge].src;
                if (source >= alternate_values.size() || target_source_values[source].is_zero() ||
                    target_source_values[source] == alternate_values[source])
                    continue;
                const std::pair<Field, Field> correct_tuple{node_index_values[source], target_source_values[source]};
                const std::pair<Field, Field> wrong_coordinate_tuple{edge_index_values[edge], alternate_values[source]};
                require(correct_tuple != wrong_coordinate_tuple,
                        "feature-coordinate tuple test found identical tuples after coordinate substitution");
                std::cerr << "[protocol-variant-test] feature_coordinate_tuple_distinguishes coordinate=" << target_coordinate
                          << " alternate_coordinate=" << route.coordinate << " edge=" << edge << " source=" << source << "\n";
                return route.coordinate;
            }
        }
        std::cerr << "[protocol-variant-test] feature_coordinate_tuple_fallback=correct_H_edge_plus_one target_coordinate="
                  << target_coordinate << " reason=no_active_source_coordinate_differs\n";
        return std::nullopt;
    }

    void reject_feature_coordinate_commitment_substitution(VariantFixture &fixture, const std::vector<std::uint8_t> &valid_proof_bytes,
                                                           const std::filesystem::path &proof_path)
    {
        require(!fixture.prover.model.hidden_layers.empty() && !fixture.prover.model.hidden_layers.front().heads.empty(),
                "feature-coordinate negative test requires a hidden head");
        const auto routes = protocol::build_feature_copy_route_metadata(fixture.prover);
        const auto head_dim = fixture.prover.model.hidden_layers.front().shape.head_dim;
        const auto selected_coordinate = head_dim / 2;
        require(head_dim > 0 && selected_coordinate < head_dim, "feature-coordinate negative test requires a source coordinate");
        const auto source_matrix = protocol::hidden_head_prefix(0, 0) + "_H_prime";
        const auto target_route = std::find_if(routes.begin(), routes.end(), [&](const auto &route)
                                               { return route.source_matrix_object == source_matrix && route.coordinate == selected_coordinate; });
        require(target_route != routes.end(), "selected feature-copy coordinate route is missing");

        auto proof = bounded_stage("feature_coordinate_commitment_substitution:parse_valid",
                                   [&] { return protocol::parse_proof(fixture.verifier, valid_proof_bytes); });
        auto target_commitment = std::find_if(proof.dynamic_commitments.begin(), proof.dynamic_commitments.end(),
                                              [&](const auto &entry) { return entry.first == target_route->edge_value_object; });
        require(target_commitment != proof.dynamic_commitments.end(),
                "valid proof omits the selected feature-coordinate dynamic commitment: " + target_route->edge_value_object);
        const auto original_point = target_commitment->second.point;
        target_commitment->second.point = gatzk::crypto::g1_add(original_point, fixture.prover.kzg.g1_generator);
        require(!(target_commitment->second.point == original_point), "feature-coordinate commitment mutation did not change the KZG point");

        const auto mutated_bytes = bounded_stage("feature_coordinate_commitment_substitution:serialize", [&] { return protocol::serialize_proof(proof); });
        require(!mutated_bytes.empty(), "mutated feature-coordinate proof serialization is empty");
        write_text(proof_path, std::string(reinterpret_cast<const char *>(mutated_bytes.data()), mutated_bytes.size()));
        std::unique_ptr<protocol::Proof> reparsed;
        try
        {
            reparsed = std::make_unique<protocol::Proof>(bounded_stage(
                "feature_coordinate_commitment_substitution:independent_parse",
                [&] { return protocol::parse_proof(fixture.verifier, mutated_bytes); }));
        }
        catch (const std::exception &error)
        {
            std::cerr << "[protocol-variant-test] feature_coordinate_commitment_substitution_rejected stage=independent_parse relation="
                      << target_route->relation_id << " coordinate=" << target_route->coordinate
                      << " point_perturbation=plus_g1_generator error=" << error.what() << "\n";
            return;
        }
        require(!bounded_stage("feature_coordinate_commitment_substitution:independent_verify",
                               [&] { return protocol::verify(fixture.verifier, *reparsed); }),
                "independent verifier accepted a proof after a Feature coordinate KZG commitment point perturbation");
        std::cerr << "[protocol-variant-test] feature_coordinate_commitment_substitution_rejected stage=independent_verify relation="
                  << target_route->relation_id << " coordinate=" << target_route->coordinate
                  << " point_perturbation=plus_g1_generator\n";
    }

    void prove_and_reject_mutated_feature_route_trace_witness(VariantFixture &fixture, std::optional<std::size_t> alternate_coordinate,
                                                               const std::filesystem::path &proof_path)
    {
        require(!fixture.prover.model.hidden_layers.empty() && !fixture.prover.model.hidden_layers.front().heads.empty(),
                "feature trace-witness mutation requires a hidden head");
        const auto target_coordinate = fixture.prover.model.hidden_layers.front().shape.head_dim / 2;
        const auto source_matrix = protocol::hidden_head_prefix(0, 0) + "_H_prime";
        const auto routes = protocol::build_feature_copy_route_metadata(fixture.prover);
        const auto target_route = std::find_if(routes.begin(), routes.end(), [&](const auto &route)
                                               { return route.source_matrix_object == source_matrix && route.coordinate == target_coordinate; });
        require(target_route != routes.end(), "feature trace-witness mutation target route is missing");
        const auto alternate_route = alternate_coordinate.has_value()
                                         ? std::find_if(routes.begin(), routes.end(), [&](const auto &route)
                                                        { return route.source_matrix_object == source_matrix && route.coordinate == *alternate_coordinate; })
                                         : routes.end();
        require(!alternate_coordinate.has_value() ||
                    (alternate_route != routes.end() && target_coordinate != alternate_route->coordinate),
                "feature trace-witness mutation alternate coordinate route is missing or aliases the target");

        auto trace = bounded_stage("feature_coordinate_trace_witness_mutation:trace",
                                   [&] { return protocol::build_trace(fixture.prover, &fixture.metrics); });
        const auto detected_alternate = validate_feature_coordinate_source_tuple_mismatch(fixture.prover, trace);
        require(detected_alternate == alternate_coordinate,
                "feature trace-witness mutation selected a coordinate that does not differ on the valid trace");

        const auto target = protocol::resolve_trace_opening_polynomial(trace, target_route->edge_value_object, fixture.prover.domains.edge);
        const auto target_source = protocol::resolve_trace_opening_polynomial(trace, target_route->source_column_object, fixture.prover.domains.n);
        const auto node_index = fixture.prover.public_polynomials.find(target_route->node_index_object);
        const auto edge_index = fixture.prover.public_polynomials.find(target_route->edge_index_object);
        require(target != nullptr && target_source != nullptr &&
                    node_index != fixture.prover.public_polynomials.end() && edge_index != fixture.prover.public_polynomials.end() &&
                    target->basis == algebra::PolynomialBasis::Evaluation &&
                    target->domain == fixture.prover.domains.edge &&
                    node_index->second.basis == algebra::PolynomialBasis::Evaluation &&
                    edge_index->second.basis == algebra::PolynomialBasis::Evaluation,
                "feature trace-witness mutation could not resolve same-domain coordinate evaluations");

        const auto &target_values = target->values();
        const auto &target_source_values = target_source->values();
        const auto &node_index_values = node_index->second.values();
        const auto &edge_index_values = edge_index->second.values();
        std::vector<Field> replacement_values = target_values;
        std::shared_ptr<const algebra::Polynomial> alternate;
        std::size_t changed_edge = fixture.prover.local.edges.size();
        if (alternate_coordinate.has_value())
        {
            alternate = protocol::resolve_trace_opening_polynomial(trace, alternate_route->edge_value_object, fixture.prover.domains.edge);
            const auto alternate_source = protocol::resolve_trace_opening_polynomial(trace, alternate_route->source_column_object, fixture.prover.domains.n);
            require(alternate != nullptr && alternate_source != nullptr && alternate->basis == algebra::PolynomialBasis::Evaluation &&
                        alternate_source->basis == algebra::PolynomialBasis::Evaluation && alternate->domain == fixture.prover.domains.edge &&
                        alternate->size() == target->size(),
                    "feature trace-witness mutation could not resolve the alternate coordinate evaluation");
            replacement_values = alternate->values();
            const auto &alternate_source_values = alternate_source->values();
            const auto &alternate_values = alternate->values();
            for (std::size_t edge = 0; edge < fixture.prover.local.edges.size(); ++edge)
            {
                const auto source = fixture.prover.local.edges[edge].src;
                require(source < alternate_source_values.size() && edge < alternate_values.size() &&
                            alternate_values[edge] == alternate_source_values[source],
                        "alternate feature coordinate is not copied into its real edge route");
                if (changed_edge == fixture.prover.local.edges.size() && target_values[edge] != alternate_values[edge])
                    changed_edge = edge;
            }
            require(changed_edge < fixture.prover.local.edges.size(),
                    "selected alternate coordinate no longer differs at any active feature edge");
        }
        else
        {
            require(!fixture.prover.local.edges.empty(), "feature trace-witness mutation needs at least one active edge");
            changed_edge = 0;
            replacement_values[changed_edge] += Field::one();
        }
        for (std::size_t edge = 0; edge < fixture.prover.local.edges.size(); ++edge)
        {
            const auto source = fixture.prover.local.edges[edge].src;
            require(source < target_source_values.size() && source < node_index_values.size() && edge < target_values.size() &&
                        edge < edge_index_values.size() && node_index_values[source] == edge_index_values[edge] &&
                        target_values[edge] == target_source_values[source],
                    "feature trace-witness mutation source coordinate is not copied into its real edge route");
        }

        const auto query = protocol::resolve_trace_opening_polynomial(trace, target_route->query_object, fixture.prover.domains.edge);
        const auto eta_it = trace.challenges.find(target_route->eta_challenge_label);
        require(query != nullptr && query->basis == algebra::PolynomialBasis::Evaluation &&
                    changed_edge < query->values().size() && eta_it != trace.challenges.end() && !eta_it->second.is_zero(),
                "feature trace-witness mutation cannot evaluate its registered coordinate quotient relation");
        const auto target_relation_residual = query->values()[changed_edge] -
                                              (edge_index_values[changed_edge] + eta_it->second * target_values[changed_edge]);
        const auto substituted_relation_residual = query->values()[changed_edge] -
                                                   (edge_index_values[changed_edge] + eta_it->second * replacement_values[changed_edge]);
        require(target_relation_residual.is_zero() && !substituted_relation_residual.is_zero(),
                "feature quotient relation Query - (P_src + eta_r*H_edge_r) does not distinguish the selected coordinate substitution");
        std::cerr << "[protocol-variant-test] feature_quotient_relation_residual_verified relation=" << target_route->relation_id
                  << " valid_zero=1 substituted_nonzero=1 edge=" << changed_edge << "\n";

        const auto target_commitment_it = trace.commitments.find(target_route->edge_value_object);
        require(target_commitment_it != trace.commitments.end(),
                "feature trace-witness mutation target commitment is missing: " + target_route->edge_value_object);
        auto replacement = algebra::Polynomial::from_evaluations(target_route->edge_value_object, replacement_values, target->domain);
        replacement.set_zero_domain_blinder(target->zero_domain_blinder);
        replacement.cuda_resident_coefficients = target->cuda_resident_coefficients;
        require(replacement.domain == target->domain && replacement.zero_domain_blinder == target->zero_domain_blinder &&
                    replacement.values() == replacement_values,
                "feature trace-witness replacement did not preserve the target domain/blinder or selected wrong values");
        auto replacement_ptr = std::make_shared<const algebra::Polynomial>(std::move(replacement));
        trace.polynomials[target_route->edge_value_object] = *replacement_ptr;
        trace.spilled_polynomials.erase(target_route->edge_value_object);
        target_commitment_it->second.opening_polynomial = replacement_ptr;
        const auto observed_replacement = protocol::resolve_trace_opening_polynomial(
            trace, target_route->edge_value_object, fixture.prover.domains.edge);
        require(observed_replacement != nullptr && observed_replacement->values() == replacement_values,
                "feature trace-witness mutation was not installed in the trace's actual opening material");
        const auto replacement_description = alternate_coordinate.has_value()
                                                 ? "coordinate_" + std::to_string(*alternate_coordinate)
                                                 : std::string("correct_plus_one");
        std::cerr << "[protocol-variant-test] feature_trace_witness_mutated relation=" << target_route->relation_id
                  << " target_coordinate=" << target_coordinate << " replacement=" << replacement_description
                  << " edge=" << changed_edge << " target_value_sha256=" << canonical_field_values_sha256(target_values)
                  << " replacement_value_sha256=" << canonical_field_values_sha256(replacement_values) << "\n";

        std::unique_ptr<protocol::Proof> proof;
        try
        {
            proof = std::make_unique<protocol::Proof>(bounded_stage(
                "feature_coordinate_trace_witness_mutation:prove",
                [&] { return protocol::prove(fixture.prover, trace, &fixture.metrics); }));
        }
        catch (const std::exception &error)
        {
            std::cerr << "[protocol-variant-test] feature_trace_witness_rejected stage=prove fail_closed=1 error=" << error.what() << "\n";
            return;
        }

        const auto bytes = bounded_stage("feature_coordinate_trace_witness_mutation:serialize",
                                         [&] { return protocol::serialize_proof(*proof); });
        require(!bytes.empty(), "mutated feature trace-witness proof serialization is empty");
        write_text(proof_path, std::string(reinterpret_cast<const char *>(bytes.data()), bytes.size()));
        protocol::Proof parsed;
        try
        {
            parsed = bounded_stage("feature_coordinate_trace_witness_mutation:independent_parse",
                                   [&] { return protocol::parse_proof(fixture.verifier, bytes); });
        }
        catch (const std::exception &error)
        {
            std::cerr << "[protocol-variant-test] feature_trace_witness_rejected stage=independent_parse error=" << error.what() << "\n";
            return;
        }

        bool accepted = false;
        try
        {
            accepted = bounded_stage("feature_coordinate_trace_witness_mutation:independent_verify",
                                     [&] { return protocol::verify(fixture.verifier, parsed); });
        }
        catch (const std::exception &error)
        {
            std::cerr << "[protocol-variant-test] feature_trace_witness_rejected stage=independent_verify fail_closed=1 error="
                      << error.what() << "\n";
            return;
        }
        require(!accepted, "independent verifier accepted a proof after replacing one actual Feature H_edge coordinate trace witness");
        std::cerr << "[protocol-variant-test] feature_trace_witness_rejected stage=independent_verify relation=" << target_route->relation_id
                  << " replacement=" << replacement_description << " proof=" << proof_path.string() << "\n";
    }

    void require_same_fixture_inputs(const VariantFixture &lhs, const VariantFixture &rhs)
    {
        const auto same_edges = [](const auto &left, const auto &right)
        {
            if (left.size() != right.size())
                return false;
            for (std::size_t index = 0; index < left.size(); ++index)
            {
                if (left[index].src != right[index].src || left[index].dst != right[index].dst || left[index].graph_id != right[index].graph_id ||
                    left[index].stable_index != right[index].stable_index)
                    return false;
            }
            return true;
        };
        const auto same_public_input = [](const auto &left, const auto &right)
        {
            return left.N_total == right.N_total && left.N == right.N && left.E == right.E && left.G_batch == right.G_batch && left.I == right.I &&
                   left.src == right.src && left.dst == right.dst && left.node_ptr == right.node_ptr && left.edge_ptr == right.edge_ptr;
        };
        const auto same_hidden_profile = [](const auto &left, const auto &right)
        {
            return left.size() == right.size() && std::equal(left.begin(), left.end(), right.begin(),
                                                             [](const auto &a, const auto &b)
                                                             { return a.head_count == b.head_count && a.head_dim == b.head_dim; });
        };
        const auto &left_config = lhs.prover.config;
        const auto &right_config = rhs.prover.config;
        const auto &left_manifest = lhs.prover.kzg.manifest;
        const auto &right_manifest = rhs.prover.kzg.manifest;
        require(lhs.prover.dataset.name == rhs.prover.dataset.name && lhs.prover.dataset.num_nodes == rhs.prover.dataset.num_nodes &&
                    lhs.prover.dataset.num_features == rhs.prover.dataset.num_features && lhs.prover.dataset.num_classes == rhs.prover.dataset.num_classes &&
                    lhs.prover.dataset.input_preprocess_id == rhs.prover.dataset.input_preprocess_id &&
                    lhs.prover.dataset.features_fp == rhs.prover.dataset.features_fp && lhs.prover.dataset.features == rhs.prover.dataset.features &&
                    lhs.prover.dataset.labels == rhs.prover.dataset.labels && lhs.prover.dataset.labels_multi == rhs.prover.dataset.labels_multi &&
                    same_edges(lhs.prover.dataset.edges, rhs.prover.dataset.edges) && lhs.prover.dataset.node_ptr == rhs.prover.dataset.node_ptr &&
                    lhs.prover.dataset.edge_ptr == rhs.prover.dataset.edge_ptr &&
                    lhs.prover.local.name == rhs.prover.local.name && lhs.prover.local.num_nodes == rhs.prover.local.num_nodes &&
                    lhs.prover.local.num_features == rhs.prover.local.num_features && lhs.prover.local.num_classes == rhs.prover.local.num_classes &&
                    lhs.prover.local.input_preprocess_id == rhs.prover.local.input_preprocess_id &&
                    lhs.prover.local.absolute_ids == rhs.prover.local.absolute_ids && lhs.prover.local.features_fp == rhs.prover.local.features_fp &&
                    lhs.prover.local.features == rhs.prover.local.features && lhs.prover.local.labels == rhs.prover.local.labels &&
                    lhs.prover.local.labels_multi == rhs.prover.local.labels_multi && same_edges(lhs.prover.local.edges, rhs.prover.local.edges) &&
                    same_public_input(lhs.prover.local.public_input, rhs.prover.local.public_input) &&
                    lhs.prover.local.node_ptr == rhs.prover.local.node_ptr && lhs.prover.local.edge_ptr == rhs.prover.local.edge_ptr &&
                    lhs.prover.model.model_arch_id == rhs.prover.model.model_arch_id &&
                    lhs.prover.model.model_param_id == rhs.prover.model.model_param_id &&
                    left_config.checkpoint_bundle == right_config.checkpoint_bundle && left_config.dataset == right_config.dataset &&
                    left_config.data_root == right_config.data_root && left_config.cache_root == right_config.cache_root &&
                    left_config.task_type == right_config.task_type && left_config.report_unit == right_config.report_unit &&
                    left_config.batching_rule == right_config.batching_rule && left_config.subgraph_rule == right_config.subgraph_rule &&
                    left_config.self_loop_rule == right_config.self_loop_rule && left_config.edge_sort_rule == right_config.edge_sort_rule &&
                    left_config.chunking_rule == right_config.chunking_rule && left_config.input_preprocess_id == right_config.input_preprocess_id &&
                    left_config.quantization.canonical_serialization() == right_config.quantization.canonical_serialization() &&
                    left_config.quant_cfg_id == right_config.quant_cfg_id && left_config.model_arch_id == right_config.model_arch_id &&
                    left_config.model_param_id == right_config.model_param_id && left_config.static_table_id == right_config.static_table_id &&
                    left_config.degree_bound_id == right_config.degree_bound_id && left_config.security_profile_id == right_config.security_profile_id &&
                    left_config.range_bits == right_config.range_bits && left_config.privacy_policy == right_config.privacy_policy &&
                    left_config.input_privacy == right_config.input_privacy && left_config.output_mode == right_config.output_mode &&
                    left_config.opening_mode == right_config.opening_mode && left_config.binding_mode == right_config.binding_mode &&
                    left_config.proof_task == right_config.proof_task && left_config.srs_type == right_config.srs_type &&
                    left_config.srs_id == right_config.srs_id && left_config.srs_path == right_config.srs_path &&
                    left_config.layer_count == right_config.layer_count && left_config.K_out == right_config.K_out &&
                    left_config.batch_graphs == right_config.batch_graphs && left_config.dump_trace == right_config.dump_trace &&
                    left_config.auto_prepare_dataset == right_config.auto_prepare_dataset &&
                    left_config.symmetrize_edges == right_config.symmetrize_edges && left_config.deduplicate_edges == right_config.deduplicate_edges &&
                    left_config.num_classes == right_config.num_classes && left_config.seed == right_config.seed &&
                    same_hidden_profile(left_config.hidden_profile, right_config.hidden_profile) && left_config.d_in_profile == right_config.d_in_profile &&
                    lhs.metrics.backend_name == rhs.metrics.backend_name && lhs.metrics.crypto_backend_name == rhs.metrics.crypto_backend_name &&
                    lhs.metrics.algebra_backend_name == rhs.metrics.algebra_backend_name && lhs.metrics.compute_backend_name == rhs.metrics.compute_backend_name &&
                    left_manifest.srs_id == right_manifest.srs_id && left_manifest.manifest_hash == right_manifest.manifest_hash &&
                    left_manifest.degree_bound == right_manifest.degree_bound && left_manifest.security_level == right_manifest.security_level &&
                    left_manifest.kzg_assumption_profile_id == right_manifest.kzg_assumption_profile_id &&
                    left_manifest.kzg_assumption_degree_bound == right_manifest.kzg_assumption_degree_bound &&
                    left_manifest.kzg_assumption_security_bits == right_manifest.kzg_assumption_security_bits &&
                    left_manifest.domain_sizes == right_manifest.domain_sizes && left_manifest.domain_generators == right_manifest.domain_generators,
                "protocol variants do not share identical checkpoint/model, public graph/features/input, quantization, security, backend, and native SRS");
    }

    int run()
    {
        const std::filesystem::path project_root(GATZK_PROTOCOL_VARIANT_TEST_ROOT);
        std::filesystem::current_path(project_root);
        mcl::bn::initPairing(mcl::BLS12_381);

        bounded_stage("default_cora_identity_snapshot", [&]
                      {
                          check_default_cora_identity_snapshot(project_root);
                          return 0;
                      });

        auto base_config = util::load_config((project_root / "configs/pubmed_secure.cfg").string());
        require(base_config.protocol_variant == "zkgat", "base formal config no longer defaults to zkgat");
        base_config.project_root = project_root.string();
        base_config.cache_root.clear();
        base_config.srs_path.clear();
        base_config.verifier_registry_path.clear();
        base_config.verifier_registry_digest.clear();
        base_config.export_dir.clear();
        base_config.auto_prepare_dataset = false;

        const auto temp_root = std::filesystem::temp_directory_path() / ("gatzk_protocol_variant_regression_" + std::to_string(::getpid()));
        if (std::filesystem::exists(temp_root))
            throw std::runtime_error("test-only fixture directory already exists; refusing to reuse stale resources: " + temp_root.string());
        std::filesystem::create_directories(temp_root);
        bool keep_artifacts = std::getenv("GATZK_PROTOCOL_VARIANT_TEST_KEEP_ARTIFACTS") != nullptr;

        try
        {
            const auto cache_root = temp_root / "cache";
            bounded_stage("mini_pubmed_graph_setup", [&]
                          {
                              create_mini_pubmed_cache(cache_root);
                              return 0;
                          });
            base_config.cache_root = cache_root.string();
            base_config.srs_id = "protocol_variant_native_test_only";
            const char *reuse_srs = std::getenv("GATZK_PROTOCOL_VARIANT_TEST_REUSE_SRS");
            base_config.srs_path = reuse_srs == nullptr || *reuse_srs == '\0' ? (temp_root / "srs" / base_config.srs_id).string() : reuse_srs;
            bounded_stage("native_test_only_srs_setup_and_audit", [&]
                          {
                              if (reuse_srs == nullptr || *reuse_srs == '\0')
                                  create_test_only_trusted_srs(temp_root, base_config);
                              else
                                  validate_test_only_srs(base_config.srs_path, base_config.srs_id, fixture_dimensions(base_config));
                              return 0;
                          });

            auto zkgat = bounded_stage("zkgat_context_and_independent_registry", [&]
                                       { return make_variant_fixture(base_config, temp_root, "zkgat"); });
            auto shared = bounded_stage("wo_shared_neighborhood_context_and_independent_registry", [&]
                                        { return make_variant_fixture(base_config, temp_root, "wo_shared_neighborhood"); });
            auto feature = bounded_stage("wo_feature_folding_context_and_independent_registry", [&]
                                         { return make_variant_fixture(base_config, temp_root, "wo_feature_folding"); });
            require_same_fixture_inputs(zkgat, shared);
            require_same_fixture_inputs(zkgat, feature);
            require(zkgat.registry_digest != shared.registry_digest &&
                        zkgat.prover.compiler_artifact->instance_canonical_bytes != shared.prover.compiler_artifact->instance_canonical_bytes,
                    "protocol variants do not have distinct authenticated protocol identities");
            require(zkgat.registry_digest != feature.registry_digest && shared.registry_digest != feature.registry_digest &&
                        zkgat.prover.compiler_artifact->instance_canonical_bytes != feature.prover.compiler_artifact->instance_canonical_bytes &&
                        shared.prover.compiler_artifact->instance_canonical_bytes != feature.prover.compiler_artifact->instance_canonical_bytes,
                    "three protocol variants do not have distinct authenticated protocol identities");

            bounded_stage("shared_terminal_catalog_and_validation_negatives", [&]
                          {
                              validate_shared_terminal_catalog(shared.prover);
                              reject_missing_shared_terminal(shared.prover, shared.verifier);
                              return 0;
                          });
            bounded_stage("feature_coordinate_catalog_and_validation_negatives", [&]
                          {
                              validate_feature_copy_catalog(feature.prover);
                              reject_missing_feature_coordinate(feature.prover, feature.verifier, temp_root);
                              return 0;
                          });

            const bool all_cuda_only = std::getenv("GATZK_PROTOCOL_VARIANT_TEST_ALL_CUDA_ONLY") != nullptr;
            const bool feature_cuda_only = std::getenv("GATZK_PROTOCOL_VARIANT_TEST_FEATURE_CUDA_ONLY") != nullptr;
            const bool feature_cpu_only = std::getenv("GATZK_PROTOCOL_VARIANT_TEST_FEATURE_CPU_ONLY") != nullptr;
            if (all_cuda_only || feature_cuda_only || feature_cpu_only)
            {
                const auto previous_route = util::route2_options();
                if (all_cuda_only || feature_cuda_only)
                {
                    require(std::getenv("GATZK_RUN_PROGRESS") != nullptr,
                            "CUDA-only protocol variant test requires progress logging so CUDA route evidence is captured");
                    require(algebra::cuda_backend_build_enabled() && algebra::cuda_backend_available() && algebra::cuda_field_hotspots_available(),
                            "CUDA-only protocol variant test requires a built and available CUDA runtime/device");
                    const auto device_summary = algebra::cuda_device_summary();
                    require(device_summary.find("cuda_runtime=true") != std::string::npos &&
                                device_summary.find("cuda_device=unavailable") == std::string::npos &&
                                device_summary.find("cuda_device=unknown") == std::string::npos,
                            "CUDA-only protocol variant test could not identify the active CUDA device: " + device_summary);
                    std::cerr << "[protocol-variant-test] CUDA_DEVICE_IDENTITY " << device_summary << "\n";
                    auto cuda_route = previous_route;
                    cuda_route.cuda_field_hotspots = true;
                    util::set_route2_options(cuda_route);
                }
                try
                {
                    if (all_cuda_only)
                    {
                        const auto route_activity = [](const algebra::CudaExecutionStats &stats)
                        {
                            return stats.ntt_calls + stats.intt_calls + stats.quotient_calls + stats.t_edge_calls + stats.t_fh_calls +
                                   stats.opening_fold_calls + stats.msm_single_calls + stats.msm_batch_calls;
                        };
                        const std::array<std::pair<VariantFixture *, std::string_view>, 3> fixtures = {
                            std::pair<VariantFixture *, std::string_view>{&zkgat, "zkgat"},
                            std::pair<VariantFixture *, std::string_view>{&shared, "wo_shared_neighborhood"},
                            std::pair<VariantFixture *, std::string_view>{&feature, "wo_feature_folding"},
                        };
                        for (const auto &[fixture, label_view] : fixtures)
                        {
                            const std::string label(label_view);
                            const auto kernels_before = algebra::cuda_kernel_invocation_count();
                            const auto route_before = route_activity(algebra::cuda_execution_stats_snapshot());
                            std::optional<std::size_t> alternate_feature_coordinate;
                            const auto proof = prove_roundtrip(
                                *fixture, temp_root / (label + "_cuda.proof"), label + "_cuda",
                                fixture == &feature
                                    ? std::function<void(const protocol::TraceArtifacts &)>([&](const auto &trace)
                                      { alternate_feature_coordinate = validate_feature_coordinate_source_tuple_mismatch(feature.prover, trace); })
                                    : std::function<void(const protocol::TraceArtifacts &)>{});
                            const auto kernels_after = algebra::cuda_kernel_invocation_count();
                            const auto route_after = route_activity(algebra::cuda_execution_stats_snapshot());
                            require(kernels_after > kernels_before,
                                    "CUDA-only proof did not launch a counted CUDA kernel for " + label);
                            require(route_after > route_before,
                                    "CUDA-only proof did not record a CUDA protocol route for " + label);
                            std::cerr << "[protocol-variant-test] CUDA_SHORT_PROOF_VERIFY_OK variant=" << label
                                      << " requested_backend=cuda_hotspots independent_registry_verify=1 kernels_delta="
                                      << (kernels_after - kernels_before) << " route_calls_delta=" << (route_after - route_before)
                                      << " registry_sha256=" << fixture->registry_digest << " proof_bytes=" << proof.size() << "\n";
                        }
                        util::set_route2_options(previous_route);
                    }
                    else
                    {
                    std::optional<std::size_t> alternate_feature_coordinate;
                    const auto feature_proof = prove_roundtrip(feature,
                                                               temp_root / (feature_cuda_only ? "wo_feature_folding_cuda.proof" : "wo_feature_folding_cpu.proof"),
                                                               feature_cuda_only ? "wo_feature_folding_cuda" : "wo_feature_folding_cpu",
                                                               [&](const auto &trace)
                                                               { alternate_feature_coordinate = validate_feature_coordinate_source_tuple_mismatch(feature.prover, trace); });
                    if (feature_cuda_only)
                    {
                        util::set_route2_options(previous_route);
                        require(feature.metrics.quotient_t_edge_ms > 0.0,
                                "CUDA-only Feature proof did not record the t_edge quotient execution");
                        std::cerr << "[protocol-variant-test] NATIVE_FEATURE_CUDA_PROOF_VERIFIED quotient_t_edge_ms="
                                  << feature.metrics.quotient_t_edge_ms << " alternate_coordinate="
                                  << (alternate_feature_coordinate.has_value() ? std::to_string(*alternate_feature_coordinate) : "none") << "\n";
                    }
                    else
                    {
                        reject_feature_coordinate_commitment_substitution(
                            feature, feature_proof, temp_root / "negative-feature-coordinate-commitment-substitution.proof");
                        prove_and_reject_mutated_feature_route_trace_witness(
                            feature, alternate_feature_coordinate, temp_root / "negative-feature-coordinate-trace-witness.proof");
                        std::cerr << "[protocol-variant-test] NATIVE_FEATURE_CPU_REGRESSION_OK alternate_coordinate="
                                  << (alternate_feature_coordinate.has_value() ? std::to_string(*alternate_feature_coordinate) : "none") << "\n";
                    }
                    }
                }
                catch (...)
                {
                    util::set_route2_options(previous_route);
                    throw;
                }
                bounded_stage("default_cora_identity_snapshot_final", [&]
                              {
                                  check_default_cora_identity_snapshot(project_root);
                                  return 0;
                              });
                if (all_cuda_only)
                    std::cerr << "[protocol-variant-test] NATIVE_ALL_VARIANT_CUDA_SHORT_REGRESSION_OK artifacts=" << temp_root.string() << "\n";
                if (!keep_artifacts)
                    std::filesystem::remove_all(temp_root);
                return 0;
            }

            const auto zkgat_proof = prove_roundtrip(zkgat, temp_root / "zkgat.proof", "zkgat");
            if (std::getenv("GATZK_PROTOCOL_VARIANT_TEST_ZKGAT_POSITIVE_ONLY") != nullptr)
            {
                std::cerr << "[protocol-variant-test] NATIVE_ZKGAT_POSITIVE_REGRESSION_OK proof=" << (temp_root / "zkgat.proof").string() << "\n";
                if (!keep_artifacts)
                    std::filesystem::remove_all(temp_root);
                return 0;
            }
            const auto shared_proof = prove_roundtrip(shared, temp_root / "wo_shared_neighborhood.proof", "wo_shared_neighborhood");
            std::optional<std::size_t> alternate_feature_coordinate;
            const auto feature_proof = prove_roundtrip(feature, temp_root / "wo_feature_folding.proof", "wo_feature_folding",
                                                       [&](const auto &trace)
                                                       { alternate_feature_coordinate = validate_feature_coordinate_source_tuple_mismatch(feature.prover, trace); });
            bounded_stage("cross_variant_proof_rejection", [&]
                          {
                              const std::array<VariantFixture *, 3> fixtures = {&zkgat, &shared, &feature};
                              const std::array<const std::vector<std::uint8_t> *, 3> proofs = {&zkgat_proof, &shared_proof, &feature_proof};
                              for (std::size_t verifier_index = 0; verifier_index < fixtures.size(); ++verifier_index)
                              {
                                  for (std::size_t proof_index = 0; proof_index < proofs.size(); ++proof_index)
                                  {
                                      if (verifier_index == proof_index)
                                          continue;
                                      require_cross_variant_rejection(fixtures[verifier_index]->verifier, *proofs[proof_index],
                                                                      fixtures[verifier_index]->prover.config.protocol_variant);
                                  }
                              }
                              return 0;
                          });

            reject_feature_coordinate_commitment_substitution(feature, feature_proof,
                                                              temp_root / "negative-feature-coordinate-commitment-substitution.proof");
            prove_and_reject_mutated_feature_route_trace_witness(feature, alternate_feature_coordinate,
                                                                 temp_root / "negative-feature-coordinate-trace-witness.proof");

            if (std::getenv("GATZK_PROTOCOL_VARIANT_TEST_SKIP_SHARED_RELATION_TAMPER_NEGATIVES") == nullptr)
            {
                const auto den_target = protocol::hidden_head_prefix(0, 0) + "_R_t_den";
                prove_and_reject_mutated_witness(shared, den_target, temp_root / "negative-denominator.proof");
                const auto agg_target = protocol::hidden_head_prefix(0, 0) + "_R_t_agg";
                prove_and_reject_mutated_witness(shared, agg_target, temp_root / "negative-aggregate.proof");
            }
            else
            {
                std::cerr << "[protocol-variant-test] shared denominator/aggregate witness tamper negatives skipped by explicit env; previously passed on stage1 checkpoint\n";
            }

            bounded_stage("default_cora_identity_snapshot_final", [&]
                          {
                              check_default_cora_identity_snapshot(project_root);
                              return 0;
                          });

            std::cerr << "[protocol-variant-test] NATIVE_PROTOCOL_VARIANT_REGRESSION_OK artifacts=" << temp_root.string() << "\n";
            if (!keep_artifacts)
                std::filesystem::remove_all(temp_root);
            return 0;
        }
        catch (...)
        {
            std::cerr << "[protocol-variant-test] failure artifacts retained at " << temp_root.string() << "\n";
            throw;
        }
    }
}

int main()
{
    try
    {
        return run();
    }
    catch (const std::exception &error)
    {
        std::cerr << "protocol variant regression failed: " << error.what() << "\n";
        return 1;
    }
}
