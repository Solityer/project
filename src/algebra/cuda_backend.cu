#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

#include "gatzk/algebra/vector_ops.hpp"

namespace gatzk::algebra {

bool cuda_backend_runtime_available() {
    int device_count = 0;
    const auto status = cudaGetDeviceCount(&device_count);
    return status == cudaSuccess && device_count > 0;
}

}  // namespace gatzk::algebra

namespace gatzk::protocol {
namespace {

__global__ void histogram_indices_kernel(
    const unsigned long long* indices,
    std::size_t count,
    unsigned long long* buckets,
    std::size_t bucket_count) {
    const auto tid =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (tid >= count) {
        return;
    }
    const auto bucket = static_cast<std::size_t>(indices[tid]);
    if (bucket < bucket_count) {
        atomicAdd(&buckets[bucket], 1ULL);
    }
}

}  // namespace

std::vector<std::size_t> lookup_histogram_indices_cuda(
    const std::vector<std::size_t>& indices,
    std::size_t domain_size) {
    std::vector<std::size_t> out(domain_size, 0);
    if (indices.empty() || domain_size == 0) {
        return out;
    }

    std::vector<unsigned long long> host_indices(indices.size(), 0ULL);
    for (std::size_t i = 0; i < indices.size(); ++i) {
        host_indices[i] = static_cast<unsigned long long>(indices[i]);
    }
    std::vector<unsigned long long> host_counts(domain_size, 0ULL);

    unsigned long long* device_indices = nullptr;
    unsigned long long* device_counts = nullptr;
    const auto indices_bytes = sizeof(unsigned long long) * host_indices.size();
    const auto counts_bytes = sizeof(unsigned long long) * host_counts.size();
    if (cudaMalloc(&device_indices, indices_bytes) != cudaSuccess) {
        throw std::runtime_error("cuda histogram failed to allocate index buffer");
    }
    if (cudaMalloc(&device_counts, counts_bytes) != cudaSuccess) {
        cudaFree(device_indices);
        throw std::runtime_error("cuda histogram failed to allocate count buffer");
    }

    const auto cleanup = [&]() {
        cudaFree(device_indices);
        cudaFree(device_counts);
    };

    if (cudaMemcpy(device_indices, host_indices.data(), indices_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cleanup();
        throw std::runtime_error("cuda histogram failed to upload indices");
    }
    if (cudaMemset(device_counts, 0, counts_bytes) != cudaSuccess) {
        cleanup();
        throw std::runtime_error("cuda histogram failed to reset bucket buffer");
    }

    constexpr unsigned int block_size = 256;
    const auto grid_size = static_cast<unsigned int>((host_indices.size() + block_size - 1U) / block_size);
    histogram_indices_kernel<<<grid_size, block_size>>>(
        device_indices,
        host_indices.size(),
        device_counts,
        host_counts.size());
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        cleanup();
        throw std::runtime_error("cuda histogram kernel launch failed");
    }

    if (cudaMemcpy(host_counts.data(), device_counts, counts_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cleanup();
        throw std::runtime_error("cuda histogram failed to download counts");
    }
    cleanup();

    for (std::size_t i = 0; i < domain_size; ++i) {
        out[i] = static_cast<std::size_t>(host_counts[i]);
    }
    return out;
}

}  // namespace gatzk::protocol
