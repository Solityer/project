#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gatzk/protocol/proof_codec.hpp"
#include "gatzk/protocol/prover.hpp"
#include "gatzk/protocol/quotients.hpp"
#include "gatzk/protocol/schema.hpp"
#include "gatzk/protocol/trace.hpp"
#include "gatzk/protocol/verifier.hpp"
#include "gatzk/util/config.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

std::vector<std::uint8_t> read_bytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("failed to open proof bytes: " + path.string());
    }
    const auto end = input.tellg();
    if (end < 0) {
        throw std::runtime_error("failed to size proof bytes: " + path.string());
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
    input.seekg(0);
    if (!bytes.empty()
        && !input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()))) {
        throw std::runtime_error("failed to read proof bytes: " + path.string());
    }
    return bytes;
}

void write_bytes(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output
        || (!bytes.empty()
            && !output.write(
                reinterpret_cast<const char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size())))) {
        throw std::runtime_error("failed to write proof bytes: " + path.string());
    }
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    return values.size() % 2 == 0
        ? (values[middle - 1] + values[middle]) / 2.0
        : values[middle];
}

void require_small_benchmark_config(const gatzk::protocol::ProtocolContext& context) {
    constexpr std::size_t kMaxNodes = 1024;
    constexpr std::size_t kMaxEdges = 16384;
    if (context.local.num_nodes > kMaxNodes || context.local.edges.size() > kMaxEdges) {
        throw std::runtime_error(
            "proof codec stage benchmark refuses non-small inputs (max nodes=1024, edges=16384)");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 4 || argc > 5) {
            throw std::runtime_error(
                "usage: gatzk_proof_codec_stage_bench "
                "<generate|parse|verify|artifact|artifact-build|proof-schema|binding-metadata> "
                "<config> <proof.bin> [repeats]");
        }
        const std::string mode = argv[1];
        const auto config = gatzk::util::load_config(argv[2]);
        const std::filesystem::path proof_path = argv[3];
        const auto repeats = argc == 5 ? static_cast<std::size_t>(std::stoull(argv[4])) : 5;
        if (repeats == 0) {
            throw std::runtime_error("repeats must be positive");
        }

        const auto context = gatzk::protocol::build_context(config, nullptr);
        require_small_benchmark_config(context);

        if (mode == "generate") {
            const auto trace = gatzk::protocol::build_trace(context, nullptr);
            const auto proof = gatzk::protocol::prove(context, trace, nullptr);
            const auto bytes = gatzk::protocol::serialize_proof(proof);
            const auto parsed = gatzk::protocol::parse_proof(context, bytes);
            if (!gatzk::protocol::verify(context, parsed, nullptr)) {
                throw std::runtime_error("generated proof did not verify");
            }
            write_bytes(proof_path, bytes);
            std::cout << "mode=generate proof_size_bytes=" << bytes.size()
                      << " verify_ok=true path=" << proof_path.string() << '\n';
            return 0;
        }

        std::vector<double> samples;
        samples.reserve(repeats);
        std::vector<double> verify_metadata_samples;
        std::vector<double> verify_transcript_samples;
        std::vector<double> verify_domain_opening_samples;
        std::vector<double> verify_quotient_samples;
        std::vector<double> verify_external_samples;
        bool verify_ok = true;
        std::size_t metadata_sink = 0;
        if (mode == "artifact") {
            for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
                std::string reason;
                const auto start = Clock::now();
                const auto valid = context.compiler_artifact != nullptr
                    && gatzk::protocol::validate_protocol_compiler_artifact(
                        context, *context.compiler_artifact, &reason);
                samples.push_back(elapsed_ms(start));
                if (!valid) {
                    throw std::runtime_error(
                        "compiler artifact validation failed: " + reason);
                }
                metadata_sink += reason.size();
            }
        } else if (mode == "artifact-build") {
            for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
                const auto start = Clock::now();
                const auto artifact =
                    gatzk::protocol::build_protocol_compiler_artifact(context);
                samples.push_back(elapsed_ms(start));
                metadata_sink += artifact.proof_schema.entries.size();
            }
        } else if (mode == "proof-schema") {
            for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
                const auto start = Clock::now();
                const auto schema = gatzk::protocol::build_proof_schema(context);
                samples.push_back(elapsed_ms(start));
                metadata_sink += schema.entries.size();
            }
        } else if (mode == "binding-metadata") {
            for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
                const auto start = Clock::now();
                const auto metadata =
                    gatzk::protocol::build_binding_relation_metadata(context);
                samples.push_back(elapsed_ms(start));
                metadata_sink += metadata.size();
            }
        }

        const auto bytes = mode == "artifact" || mode == "artifact-build"
                || mode == "proof-schema"
                || mode == "binding-metadata"
            ? std::vector<std::uint8_t>{}
            : read_bytes(proof_path);
        if (mode == "parse") {
            for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
                const auto start = Clock::now();
                const auto proof = gatzk::protocol::parse_proof(context, bytes);
                samples.push_back(elapsed_ms(start));
                if (proof.block_order.empty()) {
                    throw std::runtime_error("parsed proof is unexpectedly empty");
                }
            }
        } else if (mode == "verify") {
            const auto proof = gatzk::protocol::parse_proof(context, bytes);
            for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
                gatzk::protocol::RunMetrics metrics;
                const auto start = Clock::now();
                verify_ok = gatzk::protocol::verify(
                    context, proof, &metrics) && verify_ok;
                samples.push_back(elapsed_ms(start));
                verify_metadata_samples.push_back(metrics.verify_metadata_ms);
                verify_transcript_samples.push_back(metrics.verify_transcript_ms);
                verify_domain_opening_samples.push_back(
                    metrics.verify_domain_opening_ms);
                verify_quotient_samples.push_back(metrics.verify_quotient_ms);
                verify_external_samples.push_back(metrics.verify_external_fold_ms);
            }
        } else if (mode == "artifact" || mode == "artifact-build"
                   || mode == "proof-schema"
                   || mode == "binding-metadata") {
            // The corresponding samples were collected before proof bytes are
            // loaded so these modes isolate parser setup work.
        } else {
            throw std::runtime_error("unknown mode: " + mode);
        }

        std::cout << std::fixed << std::setprecision(3)
                  << "mode=" << mode
                  << " repeats=" << repeats
                  << " proof_size_bytes=" << bytes.size()
                  << " median_ms=" << median(samples)
                  << " min_ms=" << *std::min_element(samples.begin(), samples.end())
                  << " max_ms=" << *std::max_element(samples.begin(), samples.end())
                  << " verify_metadata_median_ms="
                  << (verify_metadata_samples.empty()
                        ? 0.0 : median(verify_metadata_samples))
                  << " verify_transcript_median_ms="
                  << (verify_transcript_samples.empty()
                        ? 0.0 : median(verify_transcript_samples))
                  << " verify_domain_opening_median_ms="
                  << (verify_domain_opening_samples.empty()
                        ? 0.0 : median(verify_domain_opening_samples))
                  << " verify_quotient_median_ms="
                  << (verify_quotient_samples.empty()
                        ? 0.0 : median(verify_quotient_samples))
                  << " verify_external_median_ms="
                  << (verify_external_samples.empty()
                        ? 0.0 : median(verify_external_samples))
                  << " verify_ok=" << (verify_ok ? "true" : "false")
                  << " metadata_sink=" << metadata_sink
                  << '\n';
        return verify_ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
