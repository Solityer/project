#include "gatzk/algebra/eval_backend.hpp"

#include <cstring>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_field.cuh"

namespace gatzk::algebra {
namespace {

cuda_detail::DeviceMontgomeryBufferCache& field_buffer_cache() {
    static cuda_detail::DeviceMontgomeryBufferCache cache;
    return cache;
}

cuda_detail::DevicePackedFieldArena& output_arena() {
    static cuda_detail::DevicePackedFieldArena arena;
    return arena;
}

struct DeviceU32Buffer {
    std::uint32_t* data = nullptr;
    std::size_t count = 0;
};

class PersistentU32BufferCache {
  public:
    ~PersistentU32BufferCache() {
        for (auto& [key, entry] : pointer_entries_) {
            (void)key;
            if (entry.data != nullptr) {
                cudaFree(entry.data);
            }
        }
        for (auto& [key, entry] : content_entries_) {
            (void)key;
            if (entry.data != nullptr) {
                cudaFree(entry.data);
            }
        }
    }

    const DeviceU32Buffer& acquire_by_pointer(
        const std::vector<std::uint32_t>& host,
        const char* label) {
        const cuda_detail::BufferKey key{
            reinterpret_cast<std::uintptr_t>(host.data()),
            host.size(),
        };
        std::lock_guard<std::mutex> lock(mutex_);
        if (const auto it = pointer_entries_.find(key); it != pointer_entries_.end()) {
            return it->second;
        }
        auto [it, inserted] = pointer_entries_.emplace(key, upload(host, label));
        (void)inserted;
        return it->second;
    }

    const DeviceU32Buffer& acquire_by_content(
        const std::vector<std::uint32_t>& host,
        const char* label) {
        const auto key = content_key(host);
        std::lock_guard<std::mutex> lock(mutex_);
        if (const auto it = content_entries_.find(key); it != content_entries_.end()) {
            return it->second;
        }
        auto [it, inserted] = content_entries_.emplace(key, upload(host, label));
        (void)inserted;
        return it->second;
    }

  private:
    static DeviceU32Buffer upload(const std::vector<std::uint32_t>& host, const char* label) {
        DeviceU32Buffer out;
        out.count = host.size();
        if (out.count == 0) {
            return out;
        }
        cuda_detail::cuda_check(
            cudaMalloc(reinterpret_cast<void**>(&out.data), out.count * sizeof(std::uint32_t)),
            label);
        cuda_detail::cuda_check(
            cudaMemcpy(out.data, host.data(), out.count * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
            label);
        return out;
    }

    static std::string content_key(const std::vector<std::uint32_t>& host) {
        std::string key;
        key.resize(host.size() * sizeof(std::uint32_t));
        if (!host.empty()) {
            std::memcpy(key.data(), host.data(), key.size());
        }
        return key;
    }

    std::mutex mutex_;
    std::unordered_map<cuda_detail::BufferKey, DeviceU32Buffer, cuda_detail::BufferKeyHash> pointer_entries_;
    std::unordered_map<std::string, DeviceU32Buffer> content_entries_;
};

PersistentU32BufferCache& u32_buffer_cache() {
    static PersistentU32BufferCache cache;
    return cache;
}

std::vector<std::uint32_t> to_u32_indices(const std::vector<std::size_t>& values, const char* label) {
    std::vector<std::uint32_t> out;
    out.reserve(values.size());
    for (const auto value : values) {
        if (value > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
            throw std::runtime_error(std::string(label) + " exceeds uint32_t");
        }
        out.push_back(static_cast<std::uint32_t>(value));
    }
    return out;
}

__global__ void packed_eval_subset_kernel(
    const PackedFieldElement* values,
    const PackedFieldElement* weights,
    std::size_t domain_size,
    std::size_t row_count,
    PackedFieldElement* out) {
    __shared__ PackedFieldElement partial[cuda_detail::kFieldKernelBlockSize];
    const auto row = static_cast<std::size_t>(blockIdx.x);
    auto acc = cuda_detail::field_zero();
    for (std::size_t column = threadIdx.x; column < domain_size; column += blockDim.x) {
        const auto value = values[column * row_count + row];
        acc = cuda_detail::field_add_mod(acc, cuda_detail::montgomery_mul(value, weights[column]));
    }
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2U; stride > 0; stride >>= 1U) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] = cuda_detail::field_add_mod(
                partial[threadIdx.x],
                partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[row] = partial[0];
    }
}

__global__ void packed_eval_full_kernel(
    const PackedFieldElement* values,
    const PackedFieldElement* weights,
    const std::uint32_t* row_indices,
    std::size_t domain_size,
    std::size_t total_rows,
    PackedFieldElement* out) {
    __shared__ PackedFieldElement partial[cuda_detail::kFieldKernelBlockSize];
    const auto row = static_cast<std::size_t>(blockIdx.x);
    const auto row_index = row_indices[row];
    auto acc = cuda_detail::field_zero();
    for (std::size_t column = threadIdx.x; column < domain_size; column += blockDim.x) {
        const auto value = values[column * total_rows + row_index];
        acc = cuda_detail::field_add_mod(acc, cuda_detail::montgomery_mul(value, weights[column]));
    }
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2U; stride > 0; stride >>= 1U) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] = cuda_detail::field_add_mod(
                partial[threadIdx.x],
                partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[row] = partial[0];
    }
}

__global__ void packed_eval_rot_subset_kernel(
    const PackedFieldElement* values,
    const PackedFieldElement* weights,
    const std::uint32_t* rotations,
    std::size_t domain_size,
    std::size_t row_count,
    std::uint32_t domain_mask,
    PackedFieldElement* out) {
    __shared__ PackedFieldElement partial[cuda_detail::kFieldKernelBlockSize];
    const auto output_index = static_cast<std::size_t>(blockIdx.x);
    const auto point_index = output_index / row_count;
    const auto row = output_index % row_count;
    const auto rotation = rotations[point_index];
    auto acc = cuda_detail::field_zero();
    for (std::size_t column = threadIdx.x; column < domain_size; column += blockDim.x) {
        const auto rotated_column = (static_cast<std::uint32_t>(column) + rotation) & domain_mask;
        const auto value = values[static_cast<std::size_t>(rotated_column) * row_count + row];
        acc = cuda_detail::field_add_mod(acc, cuda_detail::montgomery_mul(value, weights[column]));
    }
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2U; stride > 0; stride >>= 1U) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] = cuda_detail::field_add_mod(
                partial[threadIdx.x],
                partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[output_index] = partial[0];
    }
}

__global__ void packed_eval_rot_full_kernel(
    const PackedFieldElement* values,
    const PackedFieldElement* weights,
    const std::uint32_t* row_indices,
    const std::uint32_t* rotations,
    std::size_t domain_size,
    std::size_t row_count,
    std::size_t total_rows,
    std::uint32_t domain_mask,
    PackedFieldElement* out) {
    __shared__ PackedFieldElement partial[cuda_detail::kFieldKernelBlockSize];
    const auto output_index = static_cast<std::size_t>(blockIdx.x);
    const auto point_index = output_index / row_count;
    const auto row = output_index % row_count;
    const auto row_index = row_indices[row];
    const auto rotation = rotations[point_index];
    auto acc = cuda_detail::field_zero();
    for (std::size_t column = threadIdx.x; column < domain_size; column += blockDim.x) {
        const auto rotated_column = (static_cast<std::uint32_t>(column) + rotation) & domain_mask;
        const auto value = values[static_cast<std::size_t>(rotated_column) * total_rows + row_index];
        acc = cuda_detail::field_add_mod(acc, cuda_detail::montgomery_mul(value, weights[column]));
    }
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2U; stride > 0; stride >>= 1U) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] = cuda_detail::field_add_mod(
                partial[threadIdx.x],
                partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[output_index] = partial[0];
    }
}

template <typename LaunchFn>
PackedEvaluationDeviceResult run_eval_kernel_device(
    std::size_t output_count,
    std::size_t row_count,
    std::size_t point_count,
    const std::string& trace_label,
    LaunchFn&& launch) {
    PackedEvaluationDeviceResult result;
    result.row_count = row_count;
    result.point_count = point_count;
    if (output_count == 0) {
        return result;
    }

    result.buffer.storage = output_arena().acquire(output_count);
    result.buffer.count = output_count;
    auto* device_out = static_cast<PackedFieldElement*>(result.buffer.storage.get());
    try {
        cudaEvent_t start;
        cudaEvent_t stop;
        const bool trace = cuda_detail::cuda_trace_enabled();
        if (trace) {
            cuda_detail::cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
            cuda_detail::cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
            cuda_detail::cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
        }

        launch(device_out);
        cuda_detail::cuda_check(cudaGetLastError(), "packed eval kernel launch");
        cuda_detail::cuda_check(cudaDeviceSynchronize(), "packed eval sync");

        if (trace) {
            cuda_detail::cuda_check(cudaEventRecord(stop), "cudaEventRecord(stop)");
            cuda_detail::cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
            float milliseconds = 0.0f;
            cuda_detail::cuda_check(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");
            cuda_detail::cuda_trace(trace_label + "_kernel", milliseconds);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    } catch (...) {
        result.buffer = {};
        result.row_count = 0;
        result.point_count = 0;
        throw;
    }
    return result;
}

PackedEvaluationDeviceResult evaluate_with_packed_native_weights_device_impl(
    const PackedEvaluationBackend& backend,
    const std::vector<std::string>& labels,
    const PackedFieldBuffer& weights) {
    if (weights.size() != backend.domain()->size) {
        throw std::runtime_error("packed evaluation backend weight size mismatch");
    }

    const auto& row_indices = backend.subset_row_indices(labels);
    if (row_indices.empty()) {
        return {};
    }

    const auto& weights_device = field_buffer_cache().acquire(weights, "packed_eval_weights");
    const auto* subset_packed = backend.subset_packed_values_device_ready(labels);
    if (subset_packed != nullptr) {
        const auto& values_device = field_buffer_cache().acquire(*subset_packed, "packed_eval_subset_values");
        return run_eval_kernel_device(
            row_indices.size(),
            row_indices.size(),
            1,
            "packed_eval_subset",
            [&](PackedFieldElement* device_out) {
                packed_eval_subset_kernel<<<static_cast<int>(row_indices.size()), cuda_detail::kFieldKernelBlockSize>>>(
                    values_device.data,
                    weights_device.data,
                    backend.domain()->size,
                    row_indices.size(),
                    device_out);
            });
    }

    const auto& values_device = field_buffer_cache().acquire(
        backend.packed_values_device_ready(),
        "packed_eval_full_values");
    const auto& device_rows = u32_buffer_cache().acquire_by_pointer(
        backend.subset_row_indices_u32(labels),
        "cudaMalloc(row indices)");
    return run_eval_kernel_device(
        row_indices.size(),
        row_indices.size(),
        1,
        "packed_eval_full",
        [&](PackedFieldElement* device_out) {
            packed_eval_full_kernel<<<static_cast<int>(row_indices.size()), cuda_detail::kFieldKernelBlockSize>>>(
                values_device.data,
                weights_device.data,
                device_rows.data,
                backend.domain()->size,
                backend.polynomials().size(),
                device_out);
        });
}

PackedEvaluationDeviceResult evaluate_with_packed_native_weight_rotations_device_impl(
    const PackedEvaluationBackend& backend,
    const std::vector<std::string>& labels,
    const PackedFieldBuffer& representative_weights,
    const std::vector<std::size_t>& rotations) {
    if (representative_weights.size() != backend.domain()->size) {
        throw std::runtime_error("packed rotated evaluation backend weight size mismatch");
    }
    if (rotations.empty()) {
        return {};
    }

    const auto& row_indices = backend.subset_row_indices(labels);
    if (row_indices.empty()) {
        return {};
    }

    const auto point_count = rotations.size();
    const auto row_count = row_indices.size();
    const auto output_count = point_count * row_count;
    const auto domain_mask = backend.domain()->size - 1U;
    if (domain_mask > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::runtime_error("packed rotated evaluation backend domain mask exceeds uint32_t");
    }

    const auto& weights_device = field_buffer_cache().acquire(representative_weights, "packed_eval_rot_weights");
    const auto* subset_packed = backend.subset_packed_values_device_ready(labels);
    const auto rotations_u32 = to_u32_indices(rotations, "packed eval rotation");
    const auto& device_rotations = u32_buffer_cache().acquire_by_content(
        rotations_u32,
        "cudaMalloc(rotations)");
    if (subset_packed != nullptr) {
        const auto& values_device = field_buffer_cache().acquire(
            *subset_packed,
            "packed_eval_rot_subset_values");
        return run_eval_kernel_device(
            output_count,
            row_count,
            point_count,
            "packed_eval_rot_subset",
            [&](PackedFieldElement* device_out) {
                packed_eval_rot_subset_kernel<<<static_cast<int>(output_count), cuda_detail::kFieldKernelBlockSize>>>(
                    values_device.data,
                    weights_device.data,
                    device_rotations.data,
                    backend.domain()->size,
                    row_count,
                    static_cast<std::uint32_t>(domain_mask),
                    device_out);
            });
    }

    const auto& values_device = field_buffer_cache().acquire(
        backend.packed_values_device_ready(),
        "packed_eval_rot_full_values");
    const auto& device_rows = u32_buffer_cache().acquire_by_pointer(
        backend.subset_row_indices_u32(labels),
        "cudaMalloc(row indices)");
    return run_eval_kernel_device(
        output_count,
        row_count,
        point_count,
        "packed_eval_rot_full",
        [&](PackedFieldElement* device_out) {
            packed_eval_rot_full_kernel<<<static_cast<int>(output_count), cuda_detail::kFieldKernelBlockSize>>>(
                values_device.data,
                weights_device.data,
                device_rows.data,
                device_rotations.data,
                backend.domain()->size,
                row_count,
                backend.polynomials().size(),
                static_cast<std::uint32_t>(domain_mask),
                device_out);
        });
}

std::vector<FieldElement> materialize_device_result_impl(const PackedEvaluationDeviceResult& result) {
    return unpack_field_elements(cuda_detail::copy_back_decoded(
        result.buffer.storage,
        result.buffer.count,
        "packed_eval_materialize",
        &output_arena()));
}

std::vector<std::vector<FieldElement>> materialize_device_rotation_result_impl(
    const PackedEvaluationDeviceResult& result) {
    auto flat = unpack_field_elements(cuda_detail::copy_back_decoded(
        result.buffer.storage,
        result.buffer.count,
        "packed_eval_rot_materialize",
        &output_arena()));
    std::vector<std::vector<FieldElement>> out(
        result.point_count,
        std::vector<FieldElement>(result.row_count, FieldElement::zero()));
    for (std::size_t point_index = 0; point_index < result.point_count; ++point_index) {
        for (std::size_t row = 0; row < result.row_count; ++row) {
            out[point_index][row] = flat[point_index * result.row_count + row];
        }
    }
    return out;
}

}  // namespace

PackedEvaluationDeviceResult evaluate_device_with_packed_native_weights_cuda(
    const PackedEvaluationBackend& backend,
    const std::vector<std::string>& labels,
    const PackedFieldBuffer& weights) {
    return evaluate_with_packed_native_weights_device_impl(backend, labels, weights);
}

std::vector<FieldElement> evaluate_with_packed_native_weights_cuda(
    const PackedEvaluationBackend& backend,
    const std::vector<std::string>& labels,
    const PackedFieldBuffer& weights) {
    return materialize_device_result_impl(evaluate_with_packed_native_weights_device_impl(backend, labels, weights));
}

PackedEvaluationDeviceResult evaluate_device_with_packed_native_weight_rotations_cuda(
    const PackedEvaluationBackend& backend,
    const std::vector<std::string>& labels,
    const PackedFieldBuffer& representative_weights,
    const std::vector<std::size_t>& rotations) {
    return evaluate_with_packed_native_weight_rotations_device_impl(backend, labels, representative_weights, rotations);
}

std::vector<FieldElement> materialize_device_result_cuda(const PackedEvaluationDeviceResult& result) {
    return materialize_device_result_impl(result);
}

std::vector<std::vector<FieldElement>> materialize_device_rotation_result_cuda(
    const PackedEvaluationDeviceResult& result) {
    return materialize_device_rotation_result_impl(result);
}

std::vector<std::vector<FieldElement>> evaluate_with_packed_native_weight_rotations_cuda(
    const PackedEvaluationBackend& backend,
    const std::vector<std::string>& labels,
    const PackedFieldBuffer& representative_weights,
    const std::vector<std::size_t>& rotations) {
    return materialize_device_rotation_result_impl(
        evaluate_with_packed_native_weight_rotations_device_impl(backend, labels, representative_weights, rotations));
}

}  // namespace gatzk::algebra
