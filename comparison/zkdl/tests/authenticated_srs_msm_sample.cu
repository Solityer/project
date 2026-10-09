#include "../gat_adapter/gat_native_resources.hpp"
#include "../cpu-backend.cuh"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace zkdl_gat;

static void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

static std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1U);
}

static std::map<std::string, std::string> read_config_values(const std::filesystem::path& path) {
    std::ifstream input(path);
    require(static_cast<bool>(input), "cannot open SRS config");
    std::map<std::string, std::string> values;
    std::string line;
    while (std::getline(input, line)) {
        const auto comment = line.find('#');
        if (comment != std::string::npos) line.resize(comment);
        line = trim(line);
        if (line.empty()) continue;
        const auto equal = line.find('=');
        require(equal != std::string::npos, "malformed config key=value line");
        const auto key = trim(line.substr(0U, equal));
        const auto value = trim(line.substr(equal + 1U));
        require(!key.empty() && !value.empty() && values.emplace(key, value).second,
                "empty or duplicate config key/value");
    }
    return values;
}

static std::string quote_config(const std::string& value) {
    std::string quoted;
    quoted.push_back('"');
    for (const unsigned char byte : value) {
        if (byte == '"' || byte == '\\') {
            quoted.push_back('\\');
            quoted.push_back(static_cast<char>(byte));
        } else if (byte == '\n') quoted += "\\n";
        else if (byte == '\r') quoted += "\\r";
        else if (byte == '\t') quoted += "\\t";
        else quoted.push_back(static_cast<char>(byte));
    }
    quoted.push_back('"');
    return quoted;
}

static std::string binding_digest(const std::map<std::string, std::string>& values) {
    std::string canonical = "{";
    bool first = true;
    for (const auto& entry : values) {
        if (entry.first == "srs_manifest_digest" || entry.first == "verifier_registry_digest" ||
            entry.first == "model_blinder_sidecar_digest") continue;
        if (!first) canonical.push_back(',');
        first = false;
        canonical += quote_config(entry.first) + ":" + quote_config(entry.second);
    }
    canonical.push_back('}');
    const std::vector<zkdl::byte> bytes(canonical.begin(), canonical.end());
    const auto digest = zkdl::sha256(bytes);
    static const char* hex = "0123456789abcdef";
    std::string result;
    result.reserve(digest.size() * 2U);
    for (const auto byte : digest) {
        result.push_back(hex[(byte >> 4U) & 0x0fU]);
        result.push_back(hex[byte & 0x0fU]);
    }
    return result;
}

static std::uint32_t next_word(std::uint32_t& state) {
    state ^= state << 13U;
    state ^= state >> 17U;
    state ^= state << 5U;
    return state;
}

static void sample_authenticated_srs(const std::filesystem::path& config_path) {
    const auto config_values = read_config_values(config_path);
    Config config;
    config.baseline_root = std::filesystem::absolute(config_path).parent_path().parent_path();
    config.config_path = std::filesystem::absolute(config_path);
    config.srs_path = config_values.at("srs_path");
    config.srs_id = config_values.at("srs_id");
    config.srs_type = config_values.at("srs_type");
    config.srs_manifest_digest = config_values.at("srs_manifest_digest");
    config.chunking_rule = config_values.at("chunking_rule");
    config.config_binding_digest = binding_digest(config_values);

    AuthenticatedSrsResource resource;
    std::string error;
    const auto load_start = std::chrono::steady_clock::now();
    require(load_authenticated_srs(config, &resource, &error),
            "authenticated SRS load failed: " + error);
    const double load_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - load_start).count();
    constexpr std::size_t kMaximumPoints = 1U << 18U;
    require(resource.authenticated && resource.selected_chunk_index < resource.chunks.size() &&
                resource.chunks[resource.selected_chunk_index].authenticated &&
                resource.chunks[resource.selected_chunk_index].materialized &&
                resource.selected_family_id == "*" && resource.count == kMaximumPoints &&
                resource.generators.size() == kMaximumPoints && resource.point_size == 48U,
            "authenticated selected SRS chunk differs from bounded 2^18 sample contract");

    std::vector<zkdl::CpuG1Point> points;
    points.reserve(resource.generators.size());
    const auto decode_start = std::chrono::steady_clock::now();
    for (const auto& encoded : resource.generators) {
        std::size_t offset = 0U;
        G1Jacobian_t native{};
        require(zkdl::read_g1(encoded, offset, native) && offset == encoded.size(),
                "authenticated SRS point failed native G1 decode");
        points.push_back(zkdl::cpu_g1_from_native(native));
    }
    const double point_decode_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - decode_start).count();

    std::vector<zkdl::FrWire> scalars(kMaximumPoints);
    std::uint32_t seed = 0x7f4a7c15U;
    const auto scalar_start = std::chrono::steady_clock::now();
    for (auto& scalar : scalars) {
        do {
            for (auto& limb : scalar) limb = next_word(seed);
        } while (!zkdl::fr_canonical(scalar));
    }
    const double scalar_setup_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - scalar_start).count();

    const auto first_start = std::chrono::steady_clock::now();
    const auto first = zkdl::cpu_g1_sum(points, scalars);
    const double first_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - first_start).count();
    const auto repeat_start = std::chrono::steady_clock::now();
    const auto repeated = zkdl::cpu_g1_sum(points, scalars);
    const double repeat_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - repeat_start).count();
    require(zkdl::cpu_g1_equal(first, repeated),
            "authenticated distinct-SRS first/repeated MSM outputs differ");

    constexpr std::size_t kOraclePoints = 64U;
    const std::vector<zkdl::CpuG1Point> oracle_points(points.begin(), points.begin() + kOraclePoints);
    const std::vector<zkdl::FrWire> oracle_scalars(scalars.begin(), scalars.begin() + kOraclePoints);
    const auto oracle_msm = zkdl::cpu_g1_sum(oracle_points, oracle_scalars);
    auto accumulated = zkdl::CpuG1Point::zero();
    for (std::size_t index = 0U; index < kOraclePoints; ++index)
        accumulated = zkdl::cpu_g1_add(
            accumulated, zkdl::cpu_g1_mul(oracle_points[index], oracle_scalars[index]));
    require(zkdl::cpu_g1_equal(oracle_msm, accumulated),
            "actual-SRS bounded prefix MSM differs from independent scalar-multiply accumulation");

    constexpr std::size_t kChunkPoints = 1U << 14U;
    const std::size_t scratch_bytes =
        zkdl_blst::blst_p1s_mult_pippenger_scratch_sizeof(kChunkPoints);
    require(scratch_bytes != 0U && scratch_bytes % sizeof(zkdl_blst::limb_t) == 0U,
            "BLST Pippenger returned invalid bounded scratch shape");
    std::vector<zkdl_blst::blst_p1_affine> affine(kChunkPoints);
    std::vector<const zkdl_blst::byte*> scalar_ptrs(kChunkPoints);
    std::vector<zkdl_blst::limb_t> scratch(scratch_bytes / sizeof(zkdl_blst::limb_t));
    const zkdl_blst::blst_p1_affine* affine_sources[2] = {affine.data(), nullptr};
    zkdl::CpuG1Point decomposed_result = zkdl::CpuG1Point::zero();
    double conversion_validation_ms = 0.0;
    double pippenger_ms = 0.0;
    double partial_merge_ms = 0.0;
    for (std::size_t offset = 0U; offset < points.size();) {
        const std::size_t count = std::min(kChunkPoints, points.size() - offset);
        const auto conversion_start = std::chrono::steady_clock::now();
        for (std::size_t index = 0U; index < count; ++index) {
            require(zkdl::fr_canonical(scalars[offset + index]),
                    "non-canonical scalar in authenticated-SRS MSM sample");
            const zkdl_blst::blst_p1 projective = zkdl::cpu_g1_to_blst(points[offset + index]);
            zkdl_blst::blst_p1_to_affine(&affine[index], &projective);
            scalar_ptrs[index] = reinterpret_cast<const zkdl_blst::byte*>(
                scalars[offset + index].data());
        }
        conversion_validation_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - conversion_start).count();

        zkdl_blst::blst_p1 partial{};
        const auto pippenger_start = std::chrono::steady_clock::now();
        zkdl_blst::blst_p1s_mult_pippenger(
            &partial, affine_sources, count, scalar_ptrs.data(), sizeof(zkdl::FrWire) * 8U,
            scratch.data());
        pippenger_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - pippenger_start).count();

        const auto merge_start = std::chrono::steady_clock::now();
        decomposed_result = zkdl::cpu_g1_add(
            decomposed_result, zkdl::cpu_g1_from_blst(partial));
        partial_merge_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - merge_start).count();
        offset += count;
    }
    require(zkdl::cpu_g1_equal(first, decomposed_result),
            "decomposed actual-SRS conversion/Pippenger result differs from cpu_g1_sum");

    std::cout << "authenticated_srs_msm status=PASS family=" << resource.selected_family_id
              << " chunk=" << resource.selected_chunk_id << " points=" << points.size()
              << " point_size=" << resource.point_size
              << " srs_manifest_digest=" << config.srs_manifest_digest
              << " authenticated_load_and_decode_ms=" << load_ms
              << " native_point_decode_ms=" << point_decode_ms
              << " scalar_setup_ms=" << scalar_setup_ms
              << " full_first_conversion_validation_and_pippenger_ms=" << first_ms
              << " full_same_input_repeat_conversion_validation_and_pippenger_ms=" << repeat_ms
              << " isolated_conversion_validation_ms=" << conversion_validation_ms
              << " isolated_pippenger_ms=" << pippenger_ms
              << " isolated_partial_merge_ms=" << partial_merge_ms
              << " independent_oracle_points=" << kOraclePoints
              << " cpu_g1_sum_workspace_bound_bytes="
              << zkdl::cpu_g1_msm_workspace_bound_bytes() << '\n';
}

int main(int argc, char** argv) {
    try {
        require(argc == 2, "usage: authenticated_srs_msm_sample CONFIG_PATH");
        sample_authenticated_srs(argv[1]);
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "authenticated_srs_msm_sample=FAIL reason=" << exception.what() << '\n';
        return 1;
    }
}
