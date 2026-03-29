#include "gatzk/algebra/vector_ops.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "cuda_field.cuh"

namespace gatzk::algebra {
namespace {

__global__ void dot_product_kernel(
    const PackedFieldElement* lhs,
    const PackedFieldElement* rhs,
    std::size_t count,
    PackedFieldElement* partial_out) {
    __shared__ PackedFieldElement partial[cuda_detail::kFieldKernelBlockSize];

    auto acc = cuda_detail::field_zero();
    const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        acc = cuda_detail::field_add_mod(acc, cuda_detail::montgomery_mul(lhs[index], rhs[index]));
    }

    partial[threadIdx.x] = acc;
    __syncthreads();

    for (unsigned stride_width = blockDim.x / 2U; stride_width > 0; stride_width >>= 1U) {
        if (threadIdx.x < stride_width) {
            partial[threadIdx.x] = cuda_detail::field_add_mod(
                partial[threadIdx.x],
                partial[threadIdx.x + stride_width]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial_out[blockIdx.x] = partial[0];
    }
}

__global__ void reduce_sum_kernel(
    const PackedFieldElement* input,
    std::size_t count,
    PackedFieldElement* partial_out) {
    __shared__ PackedFieldElement partial[cuda_detail::kFieldKernelBlockSize];

    auto acc = cuda_detail::field_zero();
    const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        acc = cuda_detail::field_add_mod(acc, input[index]);
    }

    partial[threadIdx.x] = acc;
    __syncthreads();

    for (unsigned stride_width = blockDim.x / 2U; stride_width > 0; stride_width >>= 1U) {
        if (threadIdx.x < stride_width) {
            partial[threadIdx.x] = cuda_detail::field_add_mod(
                partial[threadIdx.x],
                partial[threadIdx.x + stride_width]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial_out[blockIdx.x] = partial[0];
    }
}

FieldElement run_dot_product_kernel(
    const PackedFieldBuffer& packed_lhs,
    const PackedFieldBuffer& packed_rhs,
    const std::string& trace_label) {
    if (packed_lhs.size() != packed_rhs.size()) {
        throw std::runtime_error("dot product packed size mismatch");
    }
    if (packed_lhs.empty()) {
        return FieldElement::zero();
    }

    auto lhs_device = cuda_detail::upload_transient_montgomery_buffer(packed_lhs, trace_label + "_lhs");
    auto rhs_device = cuda_detail::upload_transient_montgomery_buffer(packed_rhs, trace_label + "_rhs");

    FieldElement out = FieldElement::zero();
    PackedFieldElement* partial = nullptr;
    PackedFieldElement* next = nullptr;
    try {
        const auto block_count = std::min<std::size_t>(
            1024U,
            (packed_lhs.size() + cuda_detail::kFieldKernelBlockSize - 1U) / cuda_detail::kFieldKernelBlockSize);
        cuda_detail::cuda_check(
            cudaMalloc(reinterpret_cast<void**>(&partial), block_count * sizeof(PackedFieldElement)),
            "cudaMalloc(dot product partial)");

        cudaEvent_t start;
        cudaEvent_t stop;
        const bool trace = cuda_detail::cuda_trace_enabled();
        if (trace) {
            cuda_detail::cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
            cuda_detail::cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
            cuda_detail::cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
        }

        dot_product_kernel<<<static_cast<int>(block_count), cuda_detail::kFieldKernelBlockSize>>>(
            lhs_device.data,
            rhs_device.data,
            packed_lhs.size(),
            partial);
        cuda_detail::cuda_check(cudaGetLastError(), "dot_product_kernel launch");

        auto partial_count = block_count;
        while (partial_count > 1) {
            const auto next_count =
                (partial_count + cuda_detail::kFieldKernelBlockSize - 1U) / cuda_detail::kFieldKernelBlockSize;
            cuda_detail::cuda_check(
                cudaMalloc(reinterpret_cast<void**>(&next), next_count * sizeof(PackedFieldElement)),
                "cudaMalloc(dot product reduction)");
            reduce_sum_kernel<<<static_cast<int>(next_count), cuda_detail::kFieldKernelBlockSize>>>(
                partial,
                partial_count,
                next);
            cuda_detail::cuda_check(cudaGetLastError(), "reduce_sum_kernel launch");
            cuda_detail::cuda_check(cudaFree(partial), "cudaFree(dot product partial)");
            partial = next;
            next = nullptr;
            partial_count = next_count;
        }

        cuda_detail::cuda_check(cudaDeviceSynchronize(), "dot product sync");
        if (trace) {
            cuda_detail::cuda_check(cudaEventRecord(stop), "cudaEventRecord(stop)");
            cuda_detail::cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
            float milliseconds = 0.0f;
            cuda_detail::cuda_check(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");
            cuda_detail::cuda_trace(trace_label + "_kernel", milliseconds);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        const auto decoded = cuda_detail::copy_back_decoded(partial, 1, trace_label);
        out = unpack_field_element(decoded[0]);
    } catch (...) {
        if (next != nullptr) {
            cudaFree(next);
        }
        if (partial != nullptr) {
            cudaFree(partial);
        }
        cuda_detail::free_device_buffer(&lhs_device);
        cuda_detail::free_device_buffer(&rhs_device);
        throw;
    }

    if (partial != nullptr) {
        cudaFree(partial);
    }
    cuda_detail::free_device_buffer(&lhs_device);
    cuda_detail::free_device_buffer(&rhs_device);
    return out;
}

}  // namespace

bool cuda_backend_runtime_available() {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

FieldElement dot_product_cuda(
    const std::vector<FieldElement>& lhs,
    const std::vector<FieldElement>& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::runtime_error("dot product size mismatch");
    }
    return run_dot_product_kernel(
        pack_field_elements(lhs),
        pack_field_elements(rhs),
        "dot_product");
}

FieldElement dot_product_native_weights_cuda(
    const std::vector<FieldElement>& lhs,
    const PackedFieldBuffer& packed_rhs) {
    if (lhs.size() != packed_rhs.size()) {
        throw std::runtime_error("dot product size mismatch");
    }
    return run_dot_product_kernel(
        pack_field_elements(lhs),
        packed_rhs,
        "dot_product_native");
}

}  // namespace gatzk::algebra
