#include <cuda_runtime.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "fr-tensor.cuh"
#include "g1-tensor.cuh"
#include "commitment.cuh"
#include "proof.cuh"
#include "timer.hpp"

namespace {

std::string json_escape(const std::string& value) {
    std::string out;
    for (const char ch : value) {
        if (ch == '\\') {
            out += "\\\\";
        } else if (ch == '"') {
            out += "\\\"";
        } else {
            out.push_back(ch);
        }
    }
    return out;
}

Fr_t scalar_from_u32(unsigned int value) {
    return {value, 0, 0, 0, 0, 0, 0, 0};
}

void write_probe(
    const std::string& path,
    bool active,
    const std::string& device_name,
    int device_count,
    const std::string& status,
    double commitment_ms,
    double prove_ms,
    double verify_ms,
    unsigned int proof_size_bytes,
    const std::string& error) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to write GPU probe: " + path);
    }
    out << "{\n";
    out << "  \"gpu_backend_active\": " << (active ? "true" : "false") << ",\n";
    out << "  \"gpu_device_name\": \"" << json_escape(device_name) << "\",\n";
    out << "  \"gpu_device_count\": " << device_count << ",\n";
    out << "  \"gpu_backend_name\": \"zkDL CUDA tensor commitment and sumcheck primitive backend\",\n";
    out << "  \"cpu_fallback_used\": false,\n";
    out << "  \"primitive_probe_status\": \"" << json_escape(status) << "\",\n";
    if (active) {
        out << "  \"primitive_commitment_time_ms\": " << std::fixed << std::setprecision(3) << commitment_ms << ",\n";
        out << "  \"primitive_prove_time_ms\": " << std::fixed << std::setprecision(3) << prove_ms << ",\n";
        out << "  \"primitive_verify_time_ms\": " << std::fixed << std::setprecision(3) << verify_ms << ",\n";
        out << "  \"primitive_proof_size_bytes\": " << proof_size_bytes << ",\n";
    } else {
        out << "  \"primitive_commitment_time_ms\": null,\n";
        out << "  \"primitive_prove_time_ms\": null,\n";
        out << "  \"primitive_verify_time_ms\": null,\n";
        out << "  \"primitive_proof_size_bytes\": null,\n";
    }
    out << "  \"error\": \"" << json_escape(error) << "\"\n";
    out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string out_path = "/tmp/zkdl_gat_gpu_probe.json";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--out" && i + 1 < argc) {
            out_path = argv[++i];
        }
    }

    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count <= 0) {
        write_probe(out_path, false, "", 0, "cuda_device_not_found", 0.0, 0.0, 0.0, 0U, cudaGetErrorString(status));
        return 0;
    }

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    cudaSetDevice(0);

    try {
        constexpr unsigned int n = 32;
        constexpr unsigned int generator_count = 16;
        std::vector<Fr_t> left(n);
        std::vector<Fr_t> right(n);
        for (unsigned int i = 0; i < n; ++i) {
            left[i] = scalar_from_u32(i + 1U);
            right[i] = scalar_from_u32((i + 1U) * 3U);
        }

        FrTensor a(n, left.data());
        FrTensor b(n, right.data());
        a.mont();
        b.mont();

        unsigned int proof_size = 0;
        const auto commitment_start = std::chrono::steady_clock::now();
        Commitment generators(generator_count, G1Jacobian_generator);
        FrTensor generator_scalars = FrTensor::random(generator_count);
        generators *= generator_scalars;
        const auto commitment = generators.commit(a);
        proof_size += commitment.size * 36U;
        const auto commitment_end = std::chrono::steady_clock::now();

        const auto prove_start = std::chrono::steady_clock::now();
        auto proof = inner_product_sumcheck(a, b, random_vec(ceilLog2(n)));
        proof_size += static_cast<unsigned int>(proof.size() * sizeof(Fr_t));
        const auto prove_end = std::chrono::steady_clock::now();

        const auto verify_start = std::chrono::steady_clock::now();
        generators.open(a, commitment, random_vec(ceilLog2(n)), proof_size);
        const auto verify_end = std::chrono::steady_clock::now();

        const auto commitment_ms = std::chrono::duration<double, std::milli>(commitment_end - commitment_start).count();
        const auto prove_ms = std::chrono::duration<double, std::milli>(prove_end - prove_start).count();
        const auto verify_ms = std::chrono::duration<double, std::milli>(verify_end - verify_start).count();
        write_probe(out_path, true, prop.name, device_count, "ok", commitment_ms, prove_ms, verify_ms, proof_size, "");
        return 0;
    } catch (const std::exception& error) {
        write_probe(out_path, false, prop.name, device_count, "zkdl_primitive_probe_failed", 0.0, 0.0, 0.0, 0U, error.what());
        return 0;
    }
}
