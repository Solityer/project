#include <algorithm>
#include <array>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
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
        constexpr char message[] = "default proof regression: stage exceeded hard 600 second limit\n";
        (void)::write(STDERR_FILENO, message, sizeof(message) - 1);
        _exit(124);
    }

    template <typename Fn> auto bounded_stage(std::string_view name, Fn &&fn)
    {
        const auto started = std::chrono::steady_clock::now();
        std::cerr << "[default-proof-test] stage:start " << name << " timeout_s=" << kStageTimeoutSeconds << "\n";
        std::signal(SIGALRM, timeout_handler);
        alarm(kStageTimeoutSeconds);
        try
        {
            auto result = fn();
            alarm(0);
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count();
            std::cerr << "[default-proof-test] stage:done " << name << " elapsed_ms=" << elapsed << "\n";
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
        const std::string srs_id = "default_gat_native_test_only";
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
                        << shell_quote(GATZK_DEFAULT_PROOF_TEST_SRS_SETUP) << " compile " << shell_quote(compiler_artifact.string()) << ' '
                        << dimensions.degree_bound << ' ' << shell_quote(envelope.string());
        for (const auto &[name, size] : dimensions.domains)
            compile_command << ' ' << name << '=' << size;
        const auto compile_status = std::system(compile_command.str().c_str());
        if (compile_status == -1 || !WIFEXITED(compile_status) || WEXITSTATUS(compile_status) != 0)
            throw std::runtime_error("test-only SRS compiler artifact generation failed");

        std::ostringstream setup_command;
        setup_command << "timeout --signal=TERM --kill-after=5s " << kStageTimeoutSeconds << "s env GATZK_SRS_WORKERS=8 "
                      << shell_quote(GATZK_DEFAULT_PROOF_TEST_SRS_SETUP) << " ceremony " << shell_quote(srs_dir.string()) << ' '
                      << shell_quote(compiler_artifact.string()) << ' ' << srs_id;
        const auto setup_status = std::system(setup_command.str().c_str());
        if (setup_status == -1 || !WIFEXITED(setup_status) || WEXITSTATUS(setup_status) != 0)
            throw std::runtime_error("test-only native trusted KZG SRS setup/audit failed");

        validate_test_only_srs(srs_dir, srs_id, dimensions);

        std::cerr << "[default-proof-test] test-only-SRS domains FH=" << dimensions.domains[1].second << " edge=" << dimensions.domains[5].second
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
        require(artifact.registry.protocol_schema_id == kCoraDefaultProtocolSchema, "default Cora protocol schema identity changed");
        require(artifact.registry.instance_schema_hash == kCoraDefaultInstanceSchema, "default Cora instance schema identity changed");
    }
    std::string srs_id() { return "default_gat_native_test_only"; }

    struct DefaultFixture
    {
        protocol::RunMetrics metrics;
        protocol::ProtocolContext prover;
        protocol::VerifierContext verifier;
        std::filesystem::path registry_path;
        std::string registry_digest;

        static protocol::RunMetrics make_metrics(std::string_view backend)
        {
            protocol::RunMetrics result;
            result.compute_backend_name = backend == "gpu" ? "cuda_hotspots" : "cpu";
            return result;
        }

        explicit DefaultFixture(util::AppConfig config, std::string_view backend)
            : metrics(make_metrics(backend)), prover(protocol::build_context(config, &metrics, false))
        {
            metrics.backend_name = metrics.crypto_backend_name + "/" + metrics.compute_backend_name;
            registry_path = config.verifier_registry_path;
            const auto artifact = protocol::make_verifier_registration_artifact(prover);
            const auto bytes = protocol::serialize_verifier_registration_artifact(artifact);
            registry_digest = sha256_hex(bytes);
            write_text(registry_path, std::string(reinterpret_cast<const char *>(bytes.data()), bytes.size()));
            verifier = protocol::load_verifier_context_from_registry(registry_path.string(), registry_digest);
            require(protocol::validate_verifier_context(verifier), "independent default verifier context validation failed");
            std::cerr << "[default-proof-test] registry_sha256=" << registry_digest
                      << " protocol_schema_id=" << artifact.registry.protocol_schema_id
                      << " instance_schema_hash=" << artifact.registry.instance_schema_hash << "\n";
        }
    };

    void require_rejected(const protocol::VerifierContext &verifier, const std::vector<std::uint8_t> &bytes, std::string_view case_name)
    {
        bool accepted = false;
        try
        {
            const auto parsed = protocol::parse_proof(verifier, bytes);
            accepted = protocol::verify(verifier, parsed);
        }
        catch (const std::exception &)
        {
            accepted = false;
        }
        require(!accepted, "independent verifier accepted negative proof case: " + std::string(case_name));
        std::cerr << "[default-proof-test] negative_rejected case=" << case_name << "\n";
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

    std::vector<std::uint8_t> prove_roundtrip(DefaultFixture &fixture, const std::filesystem::path &proof_path)
    {
        auto trace = bounded_stage("default_trace", [&] { return protocol::build_trace(fixture.prover, &fixture.metrics); });
        auto proof = bounded_stage("default_prove", [&] { return protocol::prove(fixture.prover, trace, &fixture.metrics); });
        require(proof.public_metadata.protocol_name == "zkgat", "default proof metadata protocol_name is not zkgat");
        const auto bytes = bounded_stage("canonical_proof_serialize", [&] { return protocol::serialize_proof(proof); });
        require(!bytes.empty(), "canonical default proof serialization is empty");
        write_text(proof_path, std::string(reinterpret_cast<const char *>(bytes.data()), bytes.size()));
        require(std::filesystem::file_size(proof_path) == bytes.size(), "default proof artifact size mismatch");
        const auto parsed = bounded_stage("independent_proof_parse", [&] { return protocol::parse_proof(fixture.verifier, bytes); });
        require(parsed.public_metadata.protocol_name == "zkgat", "independent parse changed default protocol identity");
        require(bounded_stage("independent_proof_verify", [&] { return protocol::verify(fixture.verifier, parsed); }),
                "independent native KZG verifier rejected valid zkgat proof");
        std::cerr << "[default-proof-test] positive_independent_verify=1 proof_bytes=" << bytes.size()
                  << " protocol_name=" << parsed.public_metadata.protocol_name << "\n";
        return bytes;
    }

    void run_negative_checks(DefaultFixture &fixture, const protocol::Proof &valid_proof,
                             const std::vector<std::uint8_t> &valid_bytes, const std::filesystem::path &temp_root)
    {
        auto wrong_name = valid_proof;
        wrong_name.public_metadata.protocol_name = "not_zkgat";
        require_rejected(fixture.verifier, protocol::serialize_proof(wrong_name), "protocol_name_identity");

        auto wrong_schema = valid_proof;
        wrong_schema.public_metadata.protocol_schema_id.assign(64, '0');
        require_rejected(fixture.verifier, protocol::serialize_proof(wrong_schema), "protocol_schema_identity");

        auto changed_commitment = valid_proof;
        require(!changed_commitment.dynamic_commitments.empty(), "valid proof has no dynamic commitment to tamper");
        changed_commitment.dynamic_commitments.front().second.point =
            gatzk::crypto::g1_add(changed_commitment.dynamic_commitments.front().second.point, fixture.prover.kzg.g1_generator);
        const auto changed_commitment_bytes = protocol::serialize_proof(changed_commitment);
        write_text(temp_root / "negative-commitment.proof",
                   std::string(reinterpret_cast<const char *>(changed_commitment_bytes.data()), changed_commitment_bytes.size()));
        require_rejected(fixture.verifier, changed_commitment_bytes, "commitment_tamper");

        auto changed_bytes = valid_bytes;
        require(changed_bytes.size() > 32, "valid proof is unexpectedly short for serialized tamper test");
        changed_bytes[changed_bytes.size() / 2] ^= 0x01U;
        write_text(temp_root / "negative-serialized-proof.proof", std::string(reinterpret_cast<const char *>(changed_bytes.data()), changed_bytes.size()));
        require_rejected(fixture.verifier, changed_bytes, "serialized_bytes_tamper");

        const auto target = protocol::hidden_head_prefix(0, 0) + "_R_t";
        auto trace = bounded_stage("mutated_witness_trace", [&] { return protocol::build_trace(fixture.prover, &fixture.metrics); });
        mutate_committed_evaluation(&trace, target);
        auto proof = bounded_stage("mutated_witness_prove", [&] { return protocol::prove(fixture.prover, trace, &fixture.metrics); });
        const auto bytes = protocol::serialize_proof(proof);
        write_text(temp_root / "negative-witness.proof", std::string(reinterpret_cast<const char *>(bytes.data()), bytes.size()));
        require_rejected(fixture.verifier, bytes, "witness_tamper");
    }

    int run(int argc, char **argv)
    {
        std::string backend = std::getenv("GATZK_DEFAULT_PROOF_TEST_BACKEND") == nullptr ? "cpu" : std::getenv("GATZK_DEFAULT_PROOF_TEST_BACKEND");
        if (argc == 2)
            backend = argv[1];
        else if (argc != 1)
            throw std::runtime_error("usage: gatzk_default_proof_regression [cpu|gpu]");
        if (backend != "cpu" && backend != "gpu")
            throw std::runtime_error("default proof regression backend must be cpu or gpu");

        const std::filesystem::path project_root(GATZK_DEFAULT_PROOF_TEST_ROOT);
        std::filesystem::current_path(project_root);
        mcl::bn::initPairing(mcl::BLS12_381);
        bounded_stage("default_cora_identity_snapshot", [&]
                      {
                          check_default_cora_identity_snapshot(project_root);
                          return 0;
                      });

        const auto previous_route = util::route2_options();
        auto route = previous_route;
        route.cuda_trace_hotspots = backend == "gpu";
        route.cuda_field_hotspots = backend == "gpu";
        if (backend == "gpu")
        {
            require(algebra::cuda_backend_build_enabled() && algebra::cuda_backend_available() && algebra::cuda_field_hotspots_available(),
                    "GPU backend requested but CUDA build/runtime/device is unavailable");
            const auto device = algebra::cuda_device_summary();
            require(device.find("cuda_runtime=true") != std::string::npos && device.find("cuda_device=unavailable") == std::string::npos &&
                        device.find("cuda_device=unknown") == std::string::npos,
                    "GPU backend requested but the CUDA device identity could not be verified: " + device);
            std::cerr << "[default-proof-test] CUDA_DEVICE_IDENTITY " << device << "\n";
        }
        util::set_route2_options(route);

        const auto temp_root = std::filesystem::temp_directory_path() / ("gatzk_default_proof_regression_" + std::to_string(::getpid()));
        if (std::filesystem::exists(temp_root))
            throw std::runtime_error("test-only fixture directory already exists; refusing stale resources: " + temp_root.string());
        std::filesystem::create_directories(temp_root);
        const bool keep_artifacts = std::getenv("GATZK_DEFAULT_PROOF_TEST_KEEP_ARTIFACTS") != nullptr;
        try
        {
            const auto cache_root = temp_root / "cache";
            bounded_stage("mini_pubmed_graph_setup", [&]
                          {
                              create_mini_pubmed_cache(cache_root);
                              return 0;
                          });
            auto config = util::load_config((project_root / "configs/pubmed_secure.cfg").string());
            config.project_root = project_root.string();
            config.cache_root = cache_root.string();
            config.srs_id = srs_id();
            config.srs_path = (temp_root / "srs" / config.srs_id).string();
            config.verifier_registry_path = (temp_root / "verifier_registry.bin").string();
            config.verifier_registry_digest.clear();
            config.export_dir.clear();
            config.auto_prepare_dataset = false;
            config.dump_trace = false;

            const char *reuse_srs = std::getenv("GATZK_DEFAULT_PROOF_TEST_REUSE_SRS");
            bounded_stage("native_test_only_srs_setup_and_audit", [&]
                          {
                              if (reuse_srs == nullptr || *reuse_srs == '\0')
                                  create_test_only_trusted_srs(temp_root, config);
                              else
                              {
                                  config.srs_path = reuse_srs;
                                  validate_test_only_srs(config.srs_path, config.srs_id, fixture_dimensions(config));
                              }
                              return 0;
                          });

            DefaultFixture fixture(config, backend);
            const auto kernels_before = algebra::cuda_kernel_invocation_count();
            const auto cuda_before = algebra::cuda_execution_stats_snapshot();
            const auto proof_bytes = prove_roundtrip(fixture, temp_root / "zkgat.proof");
            const auto parsed = protocol::parse_proof(fixture.verifier, proof_bytes);
            run_negative_checks(fixture, parsed, proof_bytes, temp_root);
            const auto kernels_after = algebra::cuda_kernel_invocation_count();
            const auto cuda_stats = algebra::cuda_execution_stats_snapshot();
            const auto route_calls = [](const algebra::CudaExecutionStats &stats)
            {
                return stats.ntt_calls + stats.intt_calls + stats.quotient_calls + stats.t_edge_calls + stats.t_fh_calls +
                       stats.opening_fold_calls + stats.msm_single_calls + stats.msm_batch_calls;
            };
            const auto cuda_route_delta = route_calls(cuda_stats) - route_calls(cuda_before);
            if (backend == "gpu")
                require(kernels_after > kernels_before || cuda_route_delta > 0,
                        "GPU proof path requested but the tiny default fixture executed no native CUDA kernel or CUDA route");
            std::cerr << "[default-proof-test] backend_requested=" << backend
                      << " backend_reported=" << fixture.metrics.compute_backend_name
                      << " cuda_kernel_invocations_delta=" << (kernels_after - kernels_before)
                      << " cuda_route_calls_delta=" << cuda_route_delta
                      << "\n";
            check_default_cora_identity_snapshot(project_root);
            std::cerr << "[default-proof-test] DEFAULT_PROOF_REGRESSION_OK artifact=" << (temp_root / "zkgat.proof").string() << "\n";
            util::set_route2_options(previous_route);
            if (!keep_artifacts)
                std::filesystem::remove_all(temp_root);
            return 0;
        }
        catch (...)
        {
            util::set_route2_options(previous_route);
            std::cerr << "[default-proof-test] failure_artifacts=" << temp_root.string() << "\n";
            throw;
        }
    }
}

int main(int argc, char **argv)
{
    try
    {
        return run(argc, argv);
    }
    catch (const std::exception &error)
    {
        std::cerr << "default proof regression failed: " << error.what() << "\n";
        return 1;
    }
}
