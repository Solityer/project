#include "gat_cuda.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace zkgpt {
namespace gat {
namespace {

__global__ void projection_kernel(const std::int64_t *input,
                                  const std::int64_t *weights,
                                  std::int64_t *projection,
                                  std::size_t rows, std::size_t inner,
                                  std::size_t cols) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = rows * cols;
    if (index >= total) return;
    const std::size_t row = index / cols;
    const std::size_t col = index % cols;
    std::int64_t value = 0;
    for (std::size_t k = 0; k < inner; ++k) {
        // The canonical bound ledger keeps these products inside the signed
        // field-safe range; the CUDA bridge uses the same int64 representation
        // as the C++ adapter and does not reinterpret them as floats.
        value += input[row * inner + k] * weights[k * cols + col];
    }
    projection[index] = value;
}

__global__ void sparse_aggregate_kernel(const std::int64_t *projection,
                                        const std::int64_t *src,
                                        const std::int64_t *dst,
                                        std::int64_t *aggregate,
                                        std::size_t edges, std::size_t cols) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = edges * cols;
    if (index >= total) return;
    const std::size_t edge = index / cols;
    const std::size_t col = index % cols;
    const std::size_t source = static_cast<std::size_t>(src[edge]);
    const std::size_t destination = static_cast<std::size_t>(dst[edge]);
    atomicAdd(reinterpret_cast<unsigned long long *>(aggregate + destination * cols + col),
              static_cast<unsigned long long>(projection[source * cols + col]));
}

bool check_cuda(cudaError_t code, const char *operation, std::string &diagnostic) {
    if (code == cudaSuccess) return true;
    std::ostringstream out;
    out << operation << ": " << cudaGetErrorString(code);
    diagnostic = out.str();
    return false;
}

std::string checksum(const std::vector<std::int64_t> &values) {
    // This checksum is telemetry only; proof identity remains the native
    // transcript/artifact digest.  Keep it deterministic and inexpensive.
    std::uint64_t state = 1469598103934665603ULL;
    for (std::int64_t value : values) {
        const std::uint64_t bits = static_cast<std::uint64_t>(value);
        for (unsigned shift = 0; shift < 64; shift += 8) {
            state ^= (bits >> shift) & 0xffU;
            state *= 1099511628211ULL;
        }
    }
    std::ostringstream out;
    out << std::hex << state;
    return out.str();
}

} // namespace

GpuExecutionEvidence execute_gpu_matmul_aggregation(
    const Tensor2D &input, const std::vector<Int> &weights,
    std::size_t weight_rows, std::size_t weight_cols, const Graph &graph) {
    GpuExecutionEvidence evidence;
    evidence.compiled = true;
    evidence.matmul_rows = input.rows;
    evidence.matmul_inner = input.cols;
    evidence.matmul_cols = weight_cols;
    evidence.aggregation_edges = graph.edges();
    try {
        graph.validate();
        if (input.rows != graph.num_nodes || input.cols != weight_rows ||
            input.values.size() != input.rows * input.cols ||
            weights.size() != weight_rows * weight_cols || weight_cols == 0) {
            evidence.diagnostic = "CUDA MatMul relation shape mismatch";
            return evidence;
        }
        int device_count = 0;
        if (!check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount", evidence.diagnostic)) return evidence;
        if (device_count <= 0) {
            evidence.diagnostic = "CUDA device count is zero; refusing CPU fallback";
            return evidence;
        }
        evidence.device_available = true;

        std::int64_t *device_input = nullptr;
        std::int64_t *device_weights = nullptr;
        std::int64_t *device_projection = nullptr;
        std::int64_t *device_src = nullptr;
        std::int64_t *device_dst = nullptr;
        std::int64_t *device_aggregate = nullptr;
        const std::size_t input_bytes = input.values.size() * sizeof(std::int64_t);
        const std::size_t weight_bytes = weights.size() * sizeof(std::int64_t);
        const std::size_t projection_count = input.rows * weight_cols;
        const std::size_t projection_bytes = projection_count * sizeof(std::int64_t);
        const std::size_t edge_bytes = graph.edges() * sizeof(std::int64_t);
        const auto release = [&]() {
            cudaFree(device_input); cudaFree(device_weights); cudaFree(device_projection);
            cudaFree(device_src); cudaFree(device_dst); cudaFree(device_aggregate);
        };
        if (!check_cuda(cudaMalloc(reinterpret_cast<void **>(&device_input), input_bytes), "cudaMalloc(input)", evidence.diagnostic) ||
            !check_cuda(cudaMalloc(reinterpret_cast<void **>(&device_weights), weight_bytes), "cudaMalloc(weights)", evidence.diagnostic) ||
            !check_cuda(cudaMalloc(reinterpret_cast<void **>(&device_projection), projection_bytes), "cudaMalloc(projection)", evidence.diagnostic) ||
            !check_cuda(cudaMalloc(reinterpret_cast<void **>(&device_src), edge_bytes), "cudaMalloc(src)", evidence.diagnostic) ||
            !check_cuda(cudaMalloc(reinterpret_cast<void **>(&device_dst), edge_bytes), "cudaMalloc(dst)", evidence.diagnostic) ||
            !check_cuda(cudaMalloc(reinterpret_cast<void **>(&device_aggregate), projection_bytes), "cudaMalloc(aggregate)", evidence.diagnostic)) {
            release();
            return evidence;
        }
        if (!check_cuda(cudaMemcpy(device_input, input.values.data(), input_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(input)", evidence.diagnostic) ||
            !check_cuda(cudaMemcpy(device_weights, weights.data(), weight_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(weights)", evidence.diagnostic) ||
            !check_cuda(cudaMemcpy(device_src, graph.src.data(), edge_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(src)", evidence.diagnostic) ||
            !check_cuda(cudaMemcpy(device_dst, graph.dst.data(), edge_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(dst)", evidence.diagnostic) ||
            !check_cuda(cudaMemset(device_aggregate, 0, projection_bytes), "cudaMemset(aggregate)", evidence.diagnostic)) {
            release();
            return evidence;
        }
        const unsigned block = 256;
        const unsigned projection_grid = static_cast<unsigned>((projection_count + block - 1) / block);
        projection_kernel<<<projection_grid, block>>>(device_input, device_weights,
                                                       device_projection, input.rows,
                                                       weight_rows, weight_cols);
        ++evidence.kernel_invocations;
        if (!check_cuda(cudaGetLastError(), "projection_kernel launch", evidence.diagnostic) ||
            !check_cuda(cudaDeviceSynchronize(), "projection_kernel synchronize", evidence.diagnostic)) {
            release();
            return evidence;
        }
        const std::size_t aggregate_count = graph.edges() * weight_cols;
        const unsigned aggregate_grid = static_cast<unsigned>((aggregate_count + block - 1) / block);
        sparse_aggregate_kernel<<<aggregate_grid, block>>>(device_projection, device_src,
                                                            device_dst, device_aggregate,
                                                            graph.edges(), weight_cols);
        ++evidence.kernel_invocations;
        if (!check_cuda(cudaGetLastError(), "sparse_aggregate_kernel launch", evidence.diagnostic) ||
            !check_cuda(cudaDeviceSynchronize(), "sparse_aggregate_kernel synchronize", evidence.diagnostic)) {
            release();
            return evidence;
        }
        std::vector<std::int64_t> host_aggregate(projection_count, 0);
        if (!check_cuda(cudaMemcpy(host_aggregate.data(), device_aggregate, projection_bytes,
                                   cudaMemcpyDeviceToHost), "cudaMemcpy(aggregate)", evidence.diagnostic)) {
            release();
            return evidence;
        }
        release();
        evidence.kernel_executed = evidence.kernel_invocations == 2;
        evidence.output_checksum = checksum(host_aggregate);
        if (!evidence.kernel_executed) evidence.diagnostic = "CUDA relation did not execute both kernels";
        return evidence;
    } catch (const std::exception &error) {
        evidence.diagnostic = error.what();
        return evidence;
    }
}

} // namespace gat
} // namespace zkgpt
