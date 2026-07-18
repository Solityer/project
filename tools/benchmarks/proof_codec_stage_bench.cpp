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
                "usage: gatzk_proof_codec_stage_bench <generate|parse|verify> <config> <proof.bin> [repeats]");
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

        const auto bytes = read_bytes(proof_path);
        std::vector<double> samples;
        samples.reserve(repeats);
        bool verify_ok = true;
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
                const auto start = Clock::now();
                verify_ok = gatzk::protocol::verify(context, proof, nullptr) && verify_ok;
                samples.push_back(elapsed_ms(start));
            }
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
                  << " verify_ok=" << (verify_ok ? "true" : "false")
                  << '\n';
        return verify_ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
