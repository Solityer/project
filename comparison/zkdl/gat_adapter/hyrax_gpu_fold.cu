#include "../cpu-backend.cuh"
#include "../fr-tensor.cuh"
#include "../g1-tensor.cuh"
#include "../commitment.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace zkdl {
namespace {

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
}

std::string device_uuid_text(const cudaUUID_t& uuid) {
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const unsigned char value : uuid.bytes)
        output << std::setw(2) << static_cast<unsigned int>(value);
    return output.str();
}

struct AllocationTracker {
    std::uint64_t active_bytes = 0U;
    std::uint64_t peak_bytes = 0U;
};

template <class T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(AllocationTracker& tracker) : tracker_(tracker) {}
    ~DeviceBuffer() { reset(); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void ensure(std::size_t count, const char* label) {
        if (count <= count_) return;
        reset();
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::overflow_error(std::string(label) + " allocation size overflows");
        const std::size_t bytes = count * sizeof(T);
        if (bytes > std::numeric_limits<std::uint64_t>::max() - tracker_.active_bytes)
            throw std::overflow_error(std::string(label) + " tracked allocation overflows");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&data_), bytes), label);
        count_ = count;
        tracker_.active_bytes += static_cast<std::uint64_t>(bytes);
        tracker_.peak_bytes = std::max(tracker_.peak_bytes, tracker_.active_bytes);
    }

    void reset() noexcept {
        if (data_ == nullptr) return;
        const std::uint64_t bytes = static_cast<std::uint64_t>(count_) * sizeof(T);
        (void)cudaFree(data_);
        data_ = nullptr;
        count_ = 0U;
        tracker_.active_bytes = bytes <= tracker_.active_bytes
            ? tracker_.active_bytes - bytes : 0U;
    }

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return count_; }

private:
    AllocationTracker& tracker_;
    T* data_ = nullptr;
    std::size_t count_ = 0U;
};

extern "C" __global__ void hyrax_lr_products_kernel(
    const G1Jacobian_t* current_g, const Fr_t* current_a,
    G1Jacobian_t* left_products, G1Jacobian_t* right_products,
    uint half) {
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= half) return;
    left_products[index] =
        G1Jacobian_mul(current_g[half + index], current_a[index]);
    right_products[index] =
        G1Jacobian_mul(current_g[index], current_a[half + index]);
}

extern "C" __global__ void hyrax_g_fold_kernel(
    const G1Jacobian_t* current_g, G1Jacobian_t* next_g,
    Fr_t challenge, Fr_t inverse, uint half) {
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= half) return;
    const G1Jacobian_t low = G1Jacobian_mul(current_g[index], inverse);
    const G1Jacobian_t high =
        G1Jacobian_mul(current_g[half + index], challenge);
    next_g[index] = G1Jacobian_add_safe(low, high);
}

extern "C" __global__ void hyrax_safe_reduce_kernel(
    const G1Jacobian_t* input, G1Jacobian_t* output,
    uint input_count, uint output_count) {
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= output_count) return;
    const uint left_index = 2U * index;
    const uint right_index = left_index + 1U;
    output[index] = right_index < input_count
        ? G1Jacobian_add_safe(input[left_index], input[right_index])
        : input[left_index];
}

class NativeHyraxGpuFoldSession final : public CpuHyraxGpuFoldSession {
public:
    NativeHyraxGpuFoldSession(const CpuIpaBasisContext& basis,
                              const CpuG1Point& h, const CpuG1Point& u)
        : basis_(basis), basis_device_(allocations_),
          current_g_(allocations_), next_g_(allocations_),
          current_a_(allocations_), left_products_(allocations_),
          right_products_(allocations_), left_scratch_(allocations_),
          right_scratch_(allocations_) {
        if (!basis_.valid() || basis_.empty() ||
            !basis_.owns_exact_generators(basis_.generators(), h, u))
            throw std::invalid_argument(
                "CUDA Hyrax fold requires the exact authenticated G/H/U basis");
        int device_count = 0;
        cuda_check(cudaGetDeviceCount(&device_count), "query CUDA device count");
        if (device_count <= 0)
            throw std::runtime_error("CUDA Hyrax fold requested but no device is available");
        int ordinal = -1;
        cuda_check(cudaGetDevice(&ordinal), "query current CUDA device");
        cudaDeviceProp properties{};
        cuda_check(cudaGetDeviceProperties(&properties, ordinal),
                   "query CUDA device properties");
        stats_.device_ordinal = ordinal;
        stats_.device_name = properties.name;
        stats_.device_uuid = device_uuid_text(properties.uuid);

        const auto& generators = basis_.generators();
        basis_device_.ensure(generators.size(), "allocate authenticated G basis");
        std::vector<G1Jacobian_t> native_generators;
        native_generators.reserve(generators.size());
        for (const CpuG1Point& generator : generators)
            native_generators.push_back(cpu_g1_to_native(generator));
        cuda_check(cudaMemcpy(basis_device_.data(), native_generators.data(),
                              native_generators.size() * sizeof(G1Jacobian_t),
                              cudaMemcpyHostToDevice),
                   "upload authenticated G basis");
        cuda_check(cudaDeviceSynchronize(), "synchronize authenticated G basis upload");
    }

    bool owns_exact_generators(
        const CpuIpaBasisContext& basis,
        const std::vector<CpuG1Point>& generators,
        const CpuG1Point& h, const CpuG1Point& u) const noexcept override {
        return basis_.owns_exact_generators(generators, h, u) &&
               basis.owns_exact_generators(generators, h, u);
    }

    void reset(std::size_t generator_count) override {
        if (generator_count == 0U || generator_count > basis_device_.size())
            throw std::invalid_argument("CUDA Hyrax fold generator prefix is invalid");
        generator_count_ = generator_count;
        current_size_ = cpu_ipa_padded_size(
            static_cast<std::uint32_t>(generator_count));
        if (current_size_ > std::numeric_limits<uint32_t>::max())
            throw std::overflow_error("CUDA Hyrax fold domain exceeds kernel index range");
        current_g_.ensure(current_size_, "allocate active Hyrax G fold");
        next_g_.ensure(current_size_, "allocate next Hyrax G fold");
        current_a_.ensure(current_size_, "allocate Hyrax scalar fold input");
        left_products_.ensure(current_size_ / 2U,
                              "allocate Hyrax left products");
        right_products_.ensure(current_size_ / 2U,
                               "allocate Hyrax right products");
        left_scratch_.ensure(current_size_ / 2U,
                             "allocate Hyrax left reduction scratch");
        right_scratch_.ensure(current_size_ / 2U,
                              "allocate Hyrax right reduction scratch");
        cuda_check(cudaMemcpy(current_g_.data(), basis_device_.data(),
                              generator_count_ * sizeof(G1Jacobian_t),
                              cudaMemcpyDeviceToDevice),
                   "reset active G from authenticated basis");
        if (current_size_ > generator_count_) {
            const G1Jacobian_t native_identity =
                cpu_g1_to_native(CpuG1Point::zero());
            std::vector<G1Jacobian_t> identity_tail(
                current_size_ - generator_count_, native_identity);
            cuda_check(cudaMemcpy(current_g_.data() + generator_count_,
                                  identity_tail.data(),
                                  identity_tail.size() * sizeof(G1Jacobian_t),
                                  cudaMemcpyHostToDevice),
                       "pad active G with identity points");
        }
        current_g_ptr_ = current_g_.data();
        next_g_ptr_ = next_g_.data();
        pending_kernel_count_ = 0U;
        cuda_check(cudaDeviceSynchronize(), "synchronize Hyrax G reset");
    }

    std::pair<CpuG1Point, CpuG1Point> compute_lr(
        const std::vector<FrWire>& current_a,
        std::size_t current_size) override {
        if (current_size < 2U || current_size != current_size_ ||
            (current_size & (current_size - 1U)) != 0U ||
            current_a.size() < current_size)
            throw std::invalid_argument("CUDA Hyrax L/R fold shape mismatch");
        std::vector<Fr_t> native_a(current_size);
        for (std::size_t index = 0U; index < current_size; ++index) {
            if (!fr_canonical(current_a[index]))
                throw std::invalid_argument("non-canonical CUDA Hyrax scalar");
            native_a[index] = fr_value(current_a[index]);
        }
        cuda_check(cudaMemcpy(current_a_.data(), native_a.data(),
                              current_size * sizeof(Fr_t),
                              cudaMemcpyHostToDevice),
                   "upload current Hyrax scalar fold");
        const uint half = static_cast<uint>(current_size / 2U);
        const uint blocks = (half + 127U) / 128U;
        hyrax_lr_products_kernel<<<blocks, 128U>>>(
            current_g_ptr_, current_a_.data(), left_products_.data(),
            right_products_.data(), half);
        launch_checked("launch Hyrax G L/R products");
        std::uint32_t active = half;
        G1Jacobian_t* left_input = left_products_.data();
        G1Jacobian_t* left_output = left_scratch_.data();
        G1Jacobian_t* right_input = right_products_.data();
        G1Jacobian_t* right_output = right_scratch_.data();
        while (active > 1U) {
            const uint output_count = (active + 1U) / 2U;
            const uint reduction_blocks = (output_count + 127U) / 128U;
            hyrax_safe_reduce_kernel<<<reduction_blocks, 128U>>>(
                left_input, left_output, active, output_count);
            launch_checked("launch safe Hyrax left reduction");
            hyrax_safe_reduce_kernel<<<reduction_blocks, 128U>>>(
                right_input, right_output, active, output_count);
            launch_checked("launch safe Hyrax right reduction");
            std::swap(left_input, left_output);
            std::swap(right_input, right_output);
            active = output_count;
        }
        G1Jacobian_t left_native{}, right_native{};
        cuda_check(cudaMemcpy(&left_native, left_input, sizeof(left_native),
                              cudaMemcpyDeviceToHost),
                   "download Hyrax left commitment");
        cuda_check(cudaMemcpy(&right_native, right_input, sizeof(right_native),
                              cudaMemcpyDeviceToHost),
                   "download Hyrax right commitment");
        synchronize_and_record();
        return {cpu_g1_from_native(left_native),
                cpu_g1_from_native(right_native)};
    }

    void fold(const FrWire& challenge, const FrWire& inverse,
              std::size_t half_size) override {
        if (current_size_ < 2U || half_size != current_size_ / 2U ||
            half_size > std::numeric_limits<uint32_t>::max() ||
            !fr_canonical(challenge) || !fr_canonical(inverse))
            throw std::invalid_argument("CUDA Hyrax G fold shape/scalar mismatch");
        const uint half = static_cast<uint>(half_size);
        const uint blocks = (half + 127U) / 128U;
        hyrax_g_fold_kernel<<<blocks, 128U>>>(
            current_g_ptr_, next_g_ptr_, fr_value(challenge),
            fr_value(inverse), half);
        launch_checked("launch Hyrax G fold");
        std::swap(current_g_ptr_, next_g_ptr_);
        current_size_ = half_size;
    }

    CpuG1Point final_generator() override {
        if (current_size_ != 1U)
            throw std::invalid_argument("CUDA Hyrax G fold did not reach one point");
        G1Jacobian_t native{};
        cuda_check(cudaMemcpy(&native, current_g_ptr_, sizeof(native),
                              cudaMemcpyDeviceToHost),
                   "download final Hyrax Ghat");
        synchronize_and_record();
        return cpu_g1_from_native(native);
    }

    CpuHyraxGpuFoldStats stats() const override {
        CpuHyraxGpuFoldStats result = stats_;
        result.peak_allocated_bytes = allocations_.peak_bytes;
        return result;
    }

private:
    void launch_checked(const char* label) {
        cuda_check(cudaGetLastError(), label);
        ++pending_kernel_count_;
    }

    void synchronize_and_record() {
        cuda_check(cudaDeviceSynchronize(), "synchronize Hyrax GPU fold kernels");
        if (pending_kernel_count_ != 0U) {
            __atomic_add_fetch(&gpu_crypto_kernel_counter(),
                               static_cast<std::uint64_t>(pending_kernel_count_),
                               __ATOMIC_RELAXED);
            stats_.kernel_count += pending_kernel_count_;
            pending_kernel_count_ = 0U;
        }
    }

    AllocationTracker allocations_;
    CpuHyraxGpuFoldStats stats_;
    CpuIpaBasisContext basis_;
    DeviceBuffer<G1Jacobian_t> basis_device_;
    DeviceBuffer<G1Jacobian_t> current_g_;
    DeviceBuffer<G1Jacobian_t> next_g_;
    DeviceBuffer<Fr_t> current_a_;
    DeviceBuffer<G1Jacobian_t> left_products_;
    DeviceBuffer<G1Jacobian_t> right_products_;
    DeviceBuffer<G1Jacobian_t> left_scratch_;
    DeviceBuffer<G1Jacobian_t> right_scratch_;
    G1Jacobian_t* current_g_ptr_ = nullptr;
    G1Jacobian_t* next_g_ptr_ = nullptr;
    std::size_t generator_count_ = 0U;
    std::size_t current_size_ = 0U;
    std::uint64_t pending_kernel_count_ = 0U;
};

}  // namespace

std::unique_ptr<CpuHyraxGpuFoldSession> make_cpu_hyrax_gpu_fold_session(
    const CpuIpaBasisContext& basis, const CpuG1Point& h,
    const CpuG1Point& u) {
    return std::unique_ptr<CpuHyraxGpuFoldSession>(
        new NativeHyraxGpuFoldSession(basis, h, u));
}

bool query_cpu_hyrax_gpu_device_info(CpuHyraxGpuCapacityInfo* info,
                                     std::string* error) {
    if (info == nullptr) {
        if (error) *error = "CUDA Hyrax device info output is null";
        return false;
    }
    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count <= 0) {
        if (error) *error = status == cudaSuccess
            ? "no CUDA device is available"
            : std::string("query CUDA device count: ") + cudaGetErrorString(status);
        return false;
    }
    int ordinal = -1;
    status = cudaGetDevice(&ordinal);
    if (status != cudaSuccess) {
        if (error) *error = std::string("query current CUDA device: ") +
                             cudaGetErrorString(status);
        return false;
    }
    cudaDeviceProp properties{};
    status = cudaGetDeviceProperties(&properties, ordinal);
    if (status != cudaSuccess) {
        if (error) *error = std::string("query CUDA device properties: ") +
                             cudaGetErrorString(status);
        return false;
    }
    std::size_t free_bytes = 0U, total_bytes = 0U;
    status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess) {
        if (error) *error = std::string("query CUDA memory availability: ") +
                             cudaGetErrorString(status);
        return false;
    }
    *info = CpuHyraxGpuCapacityInfo{};
    info->cuda_free_bytes = free_bytes;
    info->cuda_total_bytes = total_bytes;
    info->device_ordinal = ordinal;
    info->device_name = properties.name;
    info->device_uuid = device_uuid_text(properties.uuid);
    return true;
}

bool query_cpu_hyrax_gpu_capacity(
    std::uint64_t authenticated_basis_size,
    std::uint64_t max_opening_padded_generator_size,
    CpuHyraxGpuCapacityInfo* info, std::string* error) {
    if (authenticated_basis_size == 0U ||
        max_opening_padded_generator_size == 0U || info == nullptr) {
        if (error) *error = "CUDA Hyrax capacity input is empty";
        return false;
    }
    constexpr std::uint64_t kBasisPointBytes = sizeof(G1Jacobian_t);
    constexpr std::uint64_t kFoldDomainBytes =
        4U * sizeof(G1Jacobian_t) + sizeof(Fr_t);
    static_assert(sizeof(G1Jacobian_t) == 144U,
                  "update the audited CUDA G1 workspace layout");
    static_assert(sizeof(Fr_t) == 32U,
                  "update the audited CUDA scalar workspace layout");
    if (authenticated_basis_size >
            (std::numeric_limits<std::uint64_t>::max() / kBasisPointBytes) ||
        max_opening_padded_generator_size >
            (std::numeric_limits<std::uint64_t>::max() / kFoldDomainBytes)) {
        if (error) *error = "CUDA Hyrax capacity byte calculation overflows";
        return false;
    }
    const std::uint64_t basis_bytes =
        authenticated_basis_size * kBasisPointBytes;
    const std::uint64_t fold_bytes =
        max_opening_padded_generator_size * kFoldDomainBytes;
    if (fold_bytes > std::numeric_limits<std::uint64_t>::max() - basis_bytes) {
        if (error) *error = "CUDA Hyrax workspace byte calculation overflows";
        return false;
    }
    if (!query_cpu_hyrax_gpu_device_info(info, error)) return false;
    info->authenticated_basis_size = authenticated_basis_size;
    info->max_opening_padded_generator_size =
        max_opening_padded_generator_size;
    info->workspace_required_bytes = basis_bytes + fold_bytes;
    return true;
}

}  // namespace zkdl
