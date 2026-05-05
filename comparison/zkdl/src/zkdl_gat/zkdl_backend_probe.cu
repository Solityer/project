#include <cuda_runtime.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "commitment.cuh"
#include "fr-tensor.cuh"
#include "proof.cuh"

namespace {

std::string json_escape(const std::string& value) {
    std::string out;
    for (const char ch : value) {
        if (ch == '\\') {
            out += "\\\\";
        } else if (ch == '"') {
            out += "\\\"";
        } else {
            out += ch;
        }
    }
    return out;
}

void write_probe_json(
    const std::string& output_path,
    bool active,
    int device_count,
    const std::string& device_name,
    const std::string& status,
    double commitment_time_ms,
    double prove_time_ms,
    double verify_time_ms,
    unsigned int proof_size_bytes) {
    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("failed to write gpu probe json");
    }
    out << "{\n";
    out << "  \"gpu_backend_active\": " << (active ? "true" : "false") << ",\n";
    out << "  \"gpu_device_name\": \"" << json_escape(device_name) << "\",\n";
    out << "  \"gpu_device_count\": " << device_count << ",\n";
    out << "  \"gpu_backend_name\": \"zkDL CUDA tensor commitment and sumcheck primitive backend\",\n";
    out << "  \"cpu_fallback_used\": false,\n";
    out << "  \"primitive_probe_status\": \"" << status << "\",\n";
    out << "  \"primitive_commitment_time_ms\": " << commitment_time_ms << ",\n";
    out << "  \"primitive_prove_time_ms\": " << prove_time_ms << ",\n";
    out << "  \"primitive_verify_time_ms\": " << verify_time_ms << ",\n";
    out << "  \"primitive_proof_size_bytes\": " << proof_size_bytes << "\n";
    out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string output_path = "runs/gpu_probe.json";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else {
            std::cerr << "usage: zkdl_gat_cuda_probe [--output path]\n";
            return 2;
        }
    }

    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
        write_probe_json(output_path, false, 0, "", "cuda_device_not_found", 0.0, 0.0, 0.0, 0);
        return 0;
    }

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    cudaSetDevice(0);

    try {
        constexpr uint tensor_size = 8;
        constexpr uint commitment_window = 4;
        const auto u = random_vec(ceilLog2(tensor_size));
        FrTensor a = FrTensor::random(tensor_size);
        FrTensor b = FrTensor::random(tensor_size);
        Commitment generators = Commitment::random_generators(commitment_window);

        const auto commitment_start = std::chrono::steady_clock::now();
        G1TensorJacobian commitment = generators.commit(a);
        cudaDeviceSynchronize();
        const auto commitment_end = std::chrono::steady_clock::now();

        const auto prove_start = std::chrono::steady_clock::now();
        std::vector<Fr_t> proof = inner_product_sumcheck(a, b, u);
        cudaDeviceSynchronize();
        const auto prove_end = std::chrono::steady_clock::now();

        unsigned int proof_size_bytes = static_cast<unsigned int>(proof.size() * sizeof(Fr_t));
        const auto verify_start = std::chrono::steady_clock::now();
        generators.open(a, commitment, u, proof_size_bytes);
        cudaDeviceSynchronize();
        const auto verify_end = std::chrono::steady_clock::now();

        const auto ms = [](auto begin, auto end) {
            return std::chrono::duration<double, std::milli>(end - begin).count();
        };
        write_probe_json(
            output_path,
            true,
            device_count,
            prop.name,
            "ok",
            ms(commitment_start, commitment_end),
            ms(prove_start, prove_end),
            ms(verify_start, verify_end),
            proof_size_bytes);
        return 0;
    } catch (const std::exception& error) {
        write_probe_json(output_path, true, device_count, prop.name, error.what(), 0.0, 0.0, 0.0, 0);
        return 0;
    }
}
