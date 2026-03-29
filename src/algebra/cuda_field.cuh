#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "gatzk/algebra/packed_field.hpp"

namespace gatzk::algebra::cuda_detail {

constexpr int kFieldKernelBlockSize = 256;

inline void cuda_check(cudaError_t status, const char* what) {
    if (status == cudaSuccess) {
        return;
    }
    std::ostringstream stream;
    stream << what << " failed: " << cudaGetErrorString(status);
    throw std::runtime_error(stream.str());
}

inline bool cuda_trace_enabled() {
    static const bool enabled = std::getenv("GATZK_CUDA_TRACE") != nullptr;
    return enabled;
}

inline void cuda_trace(const std::string& label, float milliseconds) {
    if (!cuda_trace_enabled()) {
        return;
    }
    std::cerr << "[cuda] " << label << "_ms=" << milliseconds << '\n';
}

struct BufferKey {
    std::uintptr_t host_ptr = 0;
    std::size_t count = 0;

    bool operator==(const BufferKey& other) const {
        return host_ptr == other.host_ptr && count == other.count;
    }
};

struct BufferKeyHash {
    std::size_t operator()(const BufferKey& key) const {
        return std::hash<std::uintptr_t>{}(key.host_ptr) ^ (std::hash<std::size_t>{}(key.count) << 1U);
    }
};

struct DeviceBuffer {
    PackedFieldElement* data = nullptr;
    std::size_t count = 0;
};

__device__ __forceinline__ PackedFieldElement field_zero() {
    return PackedFieldElement{{0, 0, 0, 0}};
}

__device__ __forceinline__ PackedFieldElement field_modulus() {
    return PackedFieldElement{{
        0xffffffff00000001ULL,
        0x53bda402fffe5bfeULL,
        0x3339d80809a1d805ULL,
        0x73eda753299d7d48ULL,
    }};
}

__device__ __forceinline__ PackedFieldElement field_r2() {
    return PackedFieldElement{{
        0xc999e990f3f29c6dULL,
        0x2b6cedcb87925c23ULL,
        0x05d314967254398fULL,
        0x0748d9d99f59ff11ULL,
    }};
}

__device__ __forceinline__ PackedFieldElement field_one() {
    return PackedFieldElement{{1ULL, 0ULL, 0ULL, 0ULL}};
}

__device__ __forceinline__ bool field_ge(
    const PackedFieldElement& lhs,
    const PackedFieldElement& rhs) {
    for (int limb = 3; limb >= 0; --limb) {
        if (lhs.limbs[limb] != rhs.limbs[limb]) {
            return lhs.limbs[limb] > rhs.limbs[limb];
        }
    }
    return true;
}

__device__ __forceinline__ PackedFieldElement field_sub(
    const PackedFieldElement& lhs,
    const PackedFieldElement& rhs) {
    PackedFieldElement out{};
    std::uint64_t borrow = 0;
    for (int limb = 0; limb < 4; ++limb) {
        const unsigned __int128 rhs_term = static_cast<unsigned __int128>(rhs.limbs[limb]) + borrow;
        const auto rhs_limb = static_cast<std::uint64_t>(rhs_term);
        out.limbs[limb] = lhs.limbs[limb] - rhs_limb;
        borrow = static_cast<std::uint64_t>(rhs_term >> 64U);
        if (lhs.limbs[limb] < rhs_limb) {
            ++borrow;
        }
    }
    return out;
}

__device__ __forceinline__ PackedFieldElement field_add_mod(
    const PackedFieldElement& lhs,
    const PackedFieldElement& rhs) {
    PackedFieldElement out{};
    unsigned __int128 carry = 0;
    for (int limb = 0; limb < 4; ++limb) {
        const unsigned __int128 acc =
            static_cast<unsigned __int128>(lhs.limbs[limb]) + rhs.limbs[limb] + carry;
        out.limbs[limb] = static_cast<std::uint64_t>(acc);
        carry = acc >> 64U;
    }
    const auto modulus = field_modulus();
    if (carry != 0 || field_ge(out, modulus)) {
        out = field_sub(out, modulus);
    }
    return out;
}

__device__ __forceinline__ PackedFieldElement field_add_raw(
    const PackedFieldElement& lhs,
    const PackedFieldElement& rhs) {
    PackedFieldElement out{};
    unsigned __int128 carry = 0;
    for (int limb = 0; limb < 4; ++limb) {
        const unsigned __int128 acc =
            static_cast<unsigned __int128>(lhs.limbs[limb]) + rhs.limbs[limb] + carry;
        out.limbs[limb] = static_cast<std::uint64_t>(acc);
        carry = acc >> 64U;
    }
    return out;
}

__device__ __forceinline__ PackedFieldElement field_sub_mod(
    const PackedFieldElement& lhs,
    const PackedFieldElement& rhs) {
    PackedFieldElement out{};
    std::uint64_t borrow = 0;
    for (int limb = 0; limb < 4; ++limb) {
        const unsigned __int128 rhs_term = static_cast<unsigned __int128>(rhs.limbs[limb]) + borrow;
        const auto rhs_limb = static_cast<std::uint64_t>(rhs_term);
        out.limbs[limb] = lhs.limbs[limb] - rhs_limb;
        borrow = static_cast<std::uint64_t>(rhs_term >> 64U);
        if (lhs.limbs[limb] < rhs_limb) {
            ++borrow;
        }
    }
    if (borrow != 0) {
        out = field_add_raw(out, field_modulus());
    }
    return out;
}

__device__ __forceinline__ PackedFieldElement montgomery_mul(
    const PackedFieldElement& lhs,
    const PackedFieldElement& rhs) {
    constexpr std::uint64_t kN0 = 0xfffffffeffffffffULL;
    const auto modulus = field_modulus();
    std::uint64_t t[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int i = 0; i < 4; ++i) {
        unsigned __int128 carry = 0;
        for (int j = 0; j < 4; ++j) {
            const unsigned __int128 acc =
                static_cast<unsigned __int128>(lhs.limbs[i]) * rhs.limbs[j]
                + t[i + j]
                + carry;
            t[i + j] = static_cast<std::uint64_t>(acc);
            carry = acc >> 64U;
        }
        for (int k = i + 4; carry != 0 && k < 8; ++k) {
            const unsigned __int128 acc = static_cast<unsigned __int128>(t[k]) + carry;
            t[k] = static_cast<std::uint64_t>(acc);
            carry = acc >> 64U;
        }
    }

    for (int i = 0; i < 4; ++i) {
        const std::uint64_t m = t[i] * kN0;
        unsigned __int128 carry = 0;
        for (int j = 0; j < 4; ++j) {
            const unsigned __int128 acc =
                static_cast<unsigned __int128>(m) * modulus.limbs[j]
                + t[i + j]
                + carry;
            t[i + j] = static_cast<std::uint64_t>(acc);
            carry = acc >> 64U;
        }
        for (int k = i + 4; carry != 0 && k < 8; ++k) {
            const unsigned __int128 acc = static_cast<unsigned __int128>(t[k]) + carry;
            t[k] = static_cast<std::uint64_t>(acc);
            carry = acc >> 64U;
        }
    }

    PackedFieldElement out{};
    for (int limb = 0; limb < 4; ++limb) {
        out.limbs[limb] = t[limb + 4];
    }
    if (field_ge(out, modulus)) {
        out = field_sub(out, modulus);
    }
    return out;
}

__global__ inline void encode_montgomery_kernel(PackedFieldElement* values, std::size_t count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    values[index] = montgomery_mul(values[index], field_r2());
}

__global__ inline void decode_montgomery_kernel(PackedFieldElement* values, std::size_t count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    values[index] = montgomery_mul(values[index], field_one());
}

class DeviceMontgomeryBufferCache {
  public:
    ~DeviceMontgomeryBufferCache() {
        for (auto& [key, entry] : entries_) {
            (void)key;
            if (entry.data != nullptr) {
                cudaFree(entry.data);
            }
        }
    }

    const DeviceBuffer& acquire(const PackedFieldBuffer& host, const std::string& trace_label) {
        const BufferKey key{
            reinterpret_cast<std::uintptr_t>(host.data()),
            host.size(),
        };
        std::lock_guard<std::mutex> lock(mutex_);
        if (const auto it = entries_.find(key); it != entries_.end()) {
            return it->second;
        }

        DeviceBuffer entry;
        entry.count = host.size();
        if (entry.count == 0) {
            auto [it, inserted] = entries_.emplace(key, entry);
            (void)inserted;
            return it->second;
        }

        const auto bytes = entry.count * sizeof(PackedFieldElement);
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&entry.data), bytes), "cudaMalloc(field buffer)");
        cuda_check(cudaMemcpy(entry.data, host.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy(field buffer)");

        cudaEvent_t start;
        cudaEvent_t stop;
        const bool trace = cuda_trace_enabled();
        if (trace) {
            cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
            cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
            cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
        }
        const auto block_count =
            static_cast<int>((entry.count + kFieldKernelBlockSize - 1U) / kFieldKernelBlockSize);
        encode_montgomery_kernel<<<block_count, kFieldKernelBlockSize>>>(entry.data, entry.count);
        cuda_check(cudaGetLastError(), "encode_montgomery_kernel launch");
        cuda_check(cudaDeviceSynchronize(), "encode_montgomery_kernel sync");
        if (trace) {
            cuda_check(cudaEventRecord(stop), "cudaEventRecord(stop)");
            cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
            float milliseconds = 0.0f;
            cuda_check(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");
            cuda_trace(trace_label + "_encode", milliseconds);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        auto [it, inserted] = entries_.emplace(key, entry);
        (void)inserted;
        return it->second;
    }

  private:
    std::mutex mutex_;
    std::unordered_map<BufferKey, DeviceBuffer, BufferKeyHash> entries_;
};

class DevicePackedFieldArena {
  public:
    ~DevicePackedFieldArena() {
        for (auto& [count, buffers] : free_lists_) {
            (void)count;
            for (auto* buffer : buffers) {
                if (buffer != nullptr) {
                    cudaFree(buffer);
                }
            }
        }
    }

    std::shared_ptr<void> acquire(std::size_t count) {
        if (count == 0) {
            return {};
        }

        PackedFieldElement* buffer = nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto& free_list = free_lists_[count];
            if (!free_list.empty()) {
                buffer = free_list.back();
                free_list.pop_back();
            }
        }
        if (buffer == nullptr) {
            cuda_check(
                cudaMalloc(reinterpret_cast<void**>(&buffer), count * sizeof(PackedFieldElement)),
                "cudaMalloc(device arena buffer)");
        }

        return std::shared_ptr<void>(
            buffer,
            [this, count](void* raw) {
                if (raw == nullptr) {
                    return;
                }
                std::lock_guard<std::mutex> lock(mutex_);
                free_lists_[count].push_back(static_cast<PackedFieldElement*>(raw));
            });
    }

  private:
    std::mutex mutex_;
    std::unordered_map<std::size_t, std::vector<PackedFieldElement*>> free_lists_;
};

inline DeviceBuffer upload_transient_montgomery_buffer(
    const PackedFieldBuffer& host,
    const std::string& trace_label) {
    DeviceBuffer out;
    out.count = host.size();
    if (out.count == 0) {
        return out;
    }

    const auto bytes = out.count * sizeof(PackedFieldElement);
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&out.data), bytes), "cudaMalloc(transient field buffer)");
    cuda_check(cudaMemcpy(out.data, host.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy(transient field buffer)");

    cudaEvent_t start;
    cudaEvent_t stop;
    const bool trace = cuda_trace_enabled();
    if (trace) {
        cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
        cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
        cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
    }
    const auto block_count =
        static_cast<int>((out.count + kFieldKernelBlockSize - 1U) / kFieldKernelBlockSize);
    encode_montgomery_kernel<<<block_count, kFieldKernelBlockSize>>>(out.data, out.count);
    cuda_check(cudaGetLastError(), "encode_montgomery_kernel launch");
    cuda_check(cudaDeviceSynchronize(), "encode_montgomery_kernel sync");
    if (trace) {
        cuda_check(cudaEventRecord(stop), "cudaEventRecord(stop)");
        cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
        float milliseconds = 0.0f;
        cuda_check(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");
        cuda_trace(trace_label + "_encode", milliseconds);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return out;
}

inline void free_device_buffer(DeviceBuffer* buffer) {
    if (buffer == nullptr || buffer->data == nullptr) {
        return;
    }
    cudaFree(buffer->data);
    buffer->data = nullptr;
    buffer->count = 0;
}

inline PackedFieldBuffer copy_back_decoded(
    PackedFieldElement* device_values,
    std::size_t count,
    const std::string& trace_label) {
    PackedFieldBuffer host(count);
    if (count == 0) {
        return host;
    }

    cudaEvent_t start;
    cudaEvent_t stop;
    const bool trace = cuda_trace_enabled();
    if (trace) {
        cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
        cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
        cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
    }

    const auto block_count =
        static_cast<int>((count + kFieldKernelBlockSize - 1U) / kFieldKernelBlockSize);
    decode_montgomery_kernel<<<block_count, kFieldKernelBlockSize>>>(device_values, count);
    cuda_check(cudaGetLastError(), "decode_montgomery_kernel launch");
    cuda_check(cudaDeviceSynchronize(), "decode_montgomery_kernel sync");
    cuda_check(
        cudaMemcpy(host.data(), device_values, count * sizeof(PackedFieldElement), cudaMemcpyDeviceToHost),
        "cudaMemcpy(decoded field buffer)");

    if (trace) {
        cuda_check(cudaEventRecord(stop), "cudaEventRecord(stop)");
        cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
        float milliseconds = 0.0f;
        cuda_check(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");
        cuda_trace(trace_label + "_decode", milliseconds);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return host;
}

inline PackedFieldBuffer copy_back_decoded(
    const std::shared_ptr<void>& storage,
    std::size_t count,
    const std::string& trace_label,
    DevicePackedFieldArena* arena = nullptr) {
    if (storage == nullptr) {
        return {};
    }
    const auto* source = static_cast<const PackedFieldElement*>(storage.get());
    std::shared_ptr<void> temp_storage;
    if (arena != nullptr) {
        temp_storage = arena->acquire(count);
    } else {
        DevicePackedFieldArena local_arena;
        temp_storage = local_arena.acquire(count);
    }
    auto* temp = static_cast<PackedFieldElement*>(temp_storage.get());
    cuda_check(
        cudaMemcpy(temp, source, count * sizeof(PackedFieldElement), cudaMemcpyDeviceToDevice),
        "cudaMemcpy(device-to-device decode staging)");
    return copy_back_decoded(temp, count, trace_label);
}

}  // namespace gatzk::algebra::cuda_detail
