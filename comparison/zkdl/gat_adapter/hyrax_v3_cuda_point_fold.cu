#include "hyrax_v3_cuda_point_fold.cuh"

#include "../g1-tensor.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace zkdl_cuda_fold {
namespace {

constexpr unsigned int kFoldThreads = 128U;

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

bool is_power_of_two(std::size_t value) {
    return value != 0U && (value & (value - 1U)) == 0U;
}

bool digest_is_zero(const std::array<std::uint8_t, 32U>& digest) {
    return std::all_of(digest.begin(), digest.end(),
                       [](std::uint8_t byte) { return byte == 0U; });
}

bool uuid_is_zero(const std::array<std::uint8_t, 16U>& uuid) {
    return std::all_of(uuid.begin(), uuid.end(),
                       [](std::uint8_t byte) { return byte == 0U; });
}

std::uint64_t checked_mul_bytes(std::size_t count, std::size_t width,
                                const char* what) {
    if (width != 0U && count > std::numeric_limits<std::uint64_t>::max() / width)
        throw std::overflow_error(std::string(what) + " byte size overflow");
    return static_cast<std::uint64_t>(count) *
           static_cast<std::uint64_t>(width);
}

std::uint64_t checked_add_bytes(std::uint64_t left, std::uint64_t right,
                                const char* what) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left)
        throw std::overflow_error(std::string(what) + " byte size overflow");
    return left + right;
}

Fr_t native_scalar_copy(const zkdl::FrWire& wire) {
    static_assert(sizeof(Fr_t) == sizeof(zkdl::FrWire),
                  "FrWire and native Fr_t must have equal limb storage size");
    Fr_t out{};
    for (std::size_t i = 0U; i < wire.size(); ++i) out.val[i] = wire[i];
    return out;
}

__global__ void hyrax_v3_point_fold_kernel(
    const G1Jacobian_t* current, G1Jacobian_t* next,
    G1Jacobian_t* window_tables, std::size_t half,
    Fr_t challenge, Fr_t inverse) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) *
                                  blockDim.x + threadIdx.x;
    if (index >= half) return;

    // The local g1-tensor helpers are baseline-local modified CUDA code, not
    // pristine upstream helpers.  add_safe covers infinity, equal points and
    // inverse points; fixed_window4 consumes canonical little-endian scalars.
    G1Jacobian_t* const scratch =
        window_tables + index * G1ScalarWindowSize;
    const G1Jacobian_t low = current[index];
    const G1Jacobian_t high = current[half + index];
    const G1Jacobian_t low_scaled =
        G1Jacobian_mul_fixed_window4(low, inverse, scratch);
    const G1Jacobian_t high_scaled =
        G1Jacobian_mul_fixed_window4(high, challenge, scratch);
    next[index] = G1Jacobian_add_safe(low_scaled, high_scaled);
}

void copy_scalar_limbs(Fr_t& destination, const zkdl::FrWire& source) {
    for (std::size_t i = 0U; i < source.size(); ++i)
        destination.val[i] = source[i];
}

}  // namespace

CudaDeviceIdentity query_cuda_device_identity(int ordinal) {
    if (ordinal < 0) throw std::invalid_argument("negative CUDA device ordinal");
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, ordinal),
               "cudaGetDeviceProperties");
    cudaUUID_t raw_uuid{};
    cuda_check(cudaDeviceGetUuid(&raw_uuid, ordinal), "cudaDeviceGetUuid");

    CudaDeviceIdentity identity{};
    identity.ordinal = ordinal;
    for (std::size_t i = 0U; i < identity.uuid.size(); ++i)
        identity.uuid[i] = static_cast<std::uint8_t>(raw_uuid.bytes[i]);
    identity.compute_capability_major = properties.major;
    identity.compute_capability_minor = properties.minor;
    return identity;
}

MontJacobianPoint pack_blst_p1(const zkdl_blst::blst_p1& point) {
    static_assert(sizeof(zkdl_blst::blst_fp) == 48U,
                  "BLST Fp must contain six u64 limbs");
    static_assert(sizeof(G1Jacobian_t) == 144U,
                  "local native Jacobian must contain three 12-limb Fp values");
    MontJacobianPoint packed{};
    const zkdl_blst::blst_fp* const coordinates[3] = {
        &point.x, &point.y, &point.z};
    for (std::size_t coordinate = 0U; coordinate < 3U; ++coordinate) {
        for (std::size_t limb = 0U; limb < 6U; ++limb) {
            packed.limb[coordinate * 12U + 2U * limb] =
                static_cast<std::uint32_t>(coordinates[coordinate]->l[limb]);
            packed.limb[coordinate * 12U + 2U * limb + 1U] =
                static_cast<std::uint32_t>(coordinates[coordinate]->l[limb] >> 32U);
        }
    }
    return packed;
}

zkdl_blst::blst_p1 unpack_blst_p1(const MontJacobianPoint& point) {
    zkdl_blst::blst_p1 unpacked{};
    zkdl_blst::blst_fp* const coordinates[3] = {
        &unpacked.x, &unpacked.y, &unpacked.z};
    for (std::size_t coordinate = 0U; coordinate < 3U; ++coordinate) {
        for (std::size_t limb = 0U; limb < 6U; ++limb) {
            coordinates[coordinate]->l[limb] =
                static_cast<std::uint64_t>(point.limb[coordinate * 12U + 2U * limb]) |
                (static_cast<std::uint64_t>(
                     point.limb[coordinate * 12U + 2U * limb + 1U]) << 32U);
        }
    }
    return unpacked;
}

struct HyraxV3CudaPointFoldSession::Impl {
    CudaDeviceIdentity device{};
    PointFoldStats counters{};
    std::size_t max_current_count = 0U;
    std::size_t max_output_count = 0U;
    G1Jacobian_t* device_current = nullptr;
    G1Jacobian_t* device_next = nullptr;
    G1Jacobian_t* device_windows = nullptr;
    MontJacobianPoint* pinned_current = nullptr;
    MontJacobianPoint* pinned_next = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t h2d_begin = nullptr;
    cudaEvent_t h2d_end = nullptr;
    cudaEvent_t kernel_begin = nullptr;
    cudaEvent_t kernel_end = nullptr;
    cudaEvent_t d2h_begin = nullptr;
    cudaEvent_t d2h_end = nullptr;

    void release_noexcept() noexcept {
        int previous_device = -1;
        const bool had_previous = cudaGetDevice(&previous_device) == cudaSuccess;
        const bool switched = had_previous && previous_device != device.ordinal &&
                              cudaSetDevice(device.ordinal) == cudaSuccess;
        if (stream != nullptr) (void)cudaStreamSynchronize(stream);
        if (device_windows != nullptr) (void)cudaFree(device_windows);
        if (device_next != nullptr) (void)cudaFree(device_next);
        if (device_current != nullptr) (void)cudaFree(device_current);
        if (pinned_next != nullptr) (void)cudaFreeHost(pinned_next);
        if (pinned_current != nullptr) (void)cudaFreeHost(pinned_current);
        if (d2h_end != nullptr) (void)cudaEventDestroy(d2h_end);
        if (d2h_begin != nullptr) (void)cudaEventDestroy(d2h_begin);
        if (kernel_end != nullptr) (void)cudaEventDestroy(kernel_end);
        if (kernel_begin != nullptr) (void)cudaEventDestroy(kernel_begin);
        if (h2d_end != nullptr) (void)cudaEventDestroy(h2d_end);
        if (h2d_begin != nullptr) (void)cudaEventDestroy(h2d_begin);
        if (stream != nullptr) (void)cudaStreamDestroy(stream);
        device_windows = nullptr;
        device_next = nullptr;
        device_current = nullptr;
        pinned_next = nullptr;
        pinned_current = nullptr;
        d2h_end = nullptr;
        d2h_begin = nullptr;
        kernel_end = nullptr;
        kernel_begin = nullptr;
        h2d_end = nullptr;
        h2d_begin = nullptr;
        stream = nullptr;
        if (switched && had_previous) (void)cudaSetDevice(previous_device);
    }

    ~Impl() { release_noexcept(); }

    void check_identity() const {
        int current_ordinal = -1;
        cuda_check(cudaGetDevice(&current_ordinal), "cudaGetDevice");
        if (current_ordinal != device.ordinal)
            throw std::runtime_error("CUDA point-fold session device ordinal changed");
        const CudaDeviceIdentity observed = query_cuda_device_identity(current_ordinal);
        if (observed.uuid != device.uuid ||
            observed.compute_capability_major != 12 ||
            observed.compute_capability_minor != 1)
            throw std::runtime_error("CUDA point-fold session device identity changed");
    }
};

HyraxV3CudaPointFoldSession::HyraxV3CudaPointFoldSession(
    std::size_t max_current_count,
    const CudaDeviceIdentity& fixed_device,
    const PointFoldSourceBinding& source_binding)
    : impl_(std::make_unique<Impl>()) {
    Impl& state = *impl_;
    if (!is_power_of_two(max_current_count) || max_current_count < 2U)
        throw std::invalid_argument("point-fold reservation must be a power of two >= 2");
    if (fixed_device.ordinal < 0 || uuid_is_zero(fixed_device.uuid) ||
        fixed_device.compute_capability_major != 12 ||
        fixed_device.compute_capability_minor != 1)
        throw std::invalid_argument("point-fold session requires a fixed sm_121 device identity");
    if (digest_is_zero(source_binding.authenticated_basis_manifest_sha256) ||
        digest_is_zero(source_binding.proof_context_sha256))
        throw std::invalid_argument("point-fold source binding digests must be nonzero");

    state.device = fixed_device;
    state.max_current_count = max_current_count;
    state.max_output_count = max_current_count / 2U;
    state.counters.device = fixed_device;
    state.counters.source_binding = source_binding;
    state.counters.reserved_current_count = max_current_count;

    state.check_identity();

    const std::uint64_t input_bytes =
        checked_mul_bytes(max_current_count, sizeof(G1Jacobian_t), "input buffer");
    const std::uint64_t output_bytes =
        checked_mul_bytes(state.max_output_count, sizeof(G1Jacobian_t), "output buffer");
    const std::uint64_t window_count = checked_mul_bytes(
        state.max_output_count, G1ScalarWindowSize, "fixed-window scratch");
    const std::uint64_t window_bytes = checked_mul_bytes(
        static_cast<std::size_t>(window_count), sizeof(G1Jacobian_t),
        "fixed-window scratch");
    state.counters.device_buffer_bytes = checked_add_bytes(
        checked_add_bytes(input_bytes, output_bytes, "device buffers"),
        window_bytes, "device buffers");
    state.counters.host_pinned_buffer_bytes = checked_add_bytes(
        checked_mul_bytes(max_current_count, sizeof(MontJacobianPoint),
                          "pinned input"),
        checked_mul_bytes(state.max_output_count, sizeof(MontJacobianPoint),
                          "pinned output"),
        "pinned host buffers");
    state.counters.peak_buffer_allocation_bytes = checked_add_bytes(
        state.counters.device_buffer_bytes,
        state.counters.host_pinned_buffer_bytes, "point-fold buffers");

    std::size_t free_bytes = 0U;
    std::size_t total_bytes = 0U;
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    state.counters.free_device_bytes_at_construction = free_bytes;
    state.counters.total_device_bytes_at_construction = total_bytes;
    if (state.counters.device_buffer_bytes > free_bytes)
        throw std::runtime_error("CUDA point-fold buffers exceed currently free device memory");

    try {
        cuda_check(cudaStreamCreateWithFlags(&state.stream, cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags");
        cuda_check(cudaEventCreate(&state.h2d_begin), "cudaEventCreate(h2d begin)");
        cuda_check(cudaEventCreate(&state.h2d_end), "cudaEventCreate(h2d end)");
        cuda_check(cudaEventCreate(&state.kernel_begin), "cudaEventCreate(kernel begin)");
        cuda_check(cudaEventCreate(&state.kernel_end), "cudaEventCreate(kernel end)");
        cuda_check(cudaEventCreate(&state.d2h_begin), "cudaEventCreate(d2h begin)");
        cuda_check(cudaEventCreate(&state.d2h_end), "cudaEventCreate(d2h end)");
        cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&state.pinned_current),
                                 static_cast<std::size_t>(input_bytes),
                                 cudaHostAllocDefault),
                   "cudaHostAlloc(input)");
        cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&state.pinned_next),
                                 static_cast<std::size_t>(output_bytes),
                                 cudaHostAllocDefault),
                   "cudaHostAlloc(output)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&state.device_current),
                              static_cast<std::size_t>(input_bytes)),
                   "cudaMalloc(input)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&state.device_next),
                              static_cast<std::size_t>(output_bytes)),
                   "cudaMalloc(output)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&state.device_windows),
                              static_cast<std::size_t>(window_bytes)),
                   "cudaMalloc(fixed-window scratch)");
    } catch (...) {
        state.release_noexcept();
        throw;
    }
}

HyraxV3CudaPointFoldSession::~HyraxV3CudaPointFoldSession() = default;

std::vector<MontJacobianPoint> HyraxV3CudaPointFoldSession::fold(
    const MontJacobianPoint* current, std::size_t current_count,
    const zkdl::FrWire& challenge, const zkdl::FrWire& inverse) {
    Impl& state = *impl_;
    state.check_identity();
    if (!is_power_of_two(current_count) || current_count < 2U ||
        current_count > state.max_current_count)
        throw std::invalid_argument("point-fold input count must be a reserved power of two >= 2");
    if (current == nullptr)
        throw std::invalid_argument("point-fold input pointer is null");
    if (!zkdl::fr_canonical(challenge) || !zkdl::fr_canonical(inverse) ||
        challenge == zkdl::FrWire{} || inverse == zkdl::FrWire{})
        throw std::invalid_argument("point-fold scalars must be canonical and nonzero");
    const zkdl::FrWire product = zkdl::fr_mul(challenge, inverse);
    zkdl::FrWire one{};
    one[0] = 1U;
    if (product != one)
        throw std::invalid_argument("point-fold challenge and inverse do not multiply to one");

    const std::size_t half = current_count / 2U;
    const std::size_t input_bytes = current_count * sizeof(MontJacobianPoint);
    const std::size_t output_bytes = half * sizeof(MontJacobianPoint);
    std::memcpy(state.pinned_current, current, input_bytes);

    Fr_t native_challenge{};
    Fr_t native_inverse{};
    copy_scalar_limbs(native_challenge, challenge);
    copy_scalar_limbs(native_inverse, inverse);

    cuda_check(cudaEventRecord(state.h2d_begin, state.stream),
               "cudaEventRecord(h2d begin)");
    cuda_check(cudaMemcpyAsync(state.device_current, state.pinned_current,
                               input_bytes, cudaMemcpyHostToDevice, state.stream),
               "cudaMemcpyAsync(H2D points)");
    ++state.counters.h2d_copy_count;
    state.counters.bytes_h2d += input_bytes;
    cuda_check(cudaEventRecord(state.h2d_end, state.stream),
               "cudaEventRecord(h2d end)");

    cuda_check(cudaEventRecord(state.kernel_begin, state.stream),
               "cudaEventRecord(kernel begin)");
    const unsigned int blocks = static_cast<unsigned int>(
        (half + kFoldThreads - 1U) / kFoldThreads);
    hyrax_v3_point_fold_kernel<<<blocks, kFoldThreads, 0U, state.stream>>>(
        state.device_current, state.device_next, state.device_windows, half,
        native_challenge, native_inverse);
    cuda_check(cudaGetLastError(), "Hyrax v3 point-fold kernel launch");
    ++state.counters.kernel_launch_count;
    cuda_check(cudaEventRecord(state.kernel_end, state.stream),
               "cudaEventRecord(kernel end)");

    cuda_check(cudaEventRecord(state.d2h_begin, state.stream),
               "cudaEventRecord(d2h begin)");
    cuda_check(cudaMemcpyAsync(state.pinned_next, state.device_next,
                               output_bytes, cudaMemcpyDeviceToHost, state.stream),
               "cudaMemcpyAsync(D2H points)");
    ++state.counters.d2h_copy_count;
    state.counters.bytes_d2h += output_bytes;
    cuda_check(cudaEventRecord(state.d2h_end, state.stream),
               "cudaEventRecord(d2h end)");
    cuda_check(cudaEventSynchronize(state.d2h_end),
               "cudaEventSynchronize(point-fold completion)");

    cuda_check(cudaEventElapsedTime(&state.counters.last_h2d_event_ms,
                                    state.h2d_begin, state.h2d_end),
               "cudaEventElapsedTime(H2D)");
    cuda_check(cudaEventElapsedTime(&state.counters.last_kernel_event_ms,
                                    state.kernel_begin, state.kernel_end),
               "cudaEventElapsedTime(kernel)");
    cuda_check(cudaEventElapsedTime(&state.counters.last_d2h_event_ms,
                                    state.d2h_begin, state.d2h_end),
               "cudaEventElapsedTime(D2H)");

    state.counters.last_current_count = current_count;
    ++state.counters.successful_fold_calls;
    state.counters.cumulative_h2d_event_ms += state.counters.last_h2d_event_ms;
    state.counters.cumulative_kernel_event_ms += state.counters.last_kernel_event_ms;
    state.counters.cumulative_d2h_event_ms += state.counters.last_d2h_event_ms;

    std::vector<MontJacobianPoint> output(half);
    std::memcpy(output.data(), state.pinned_next, output_bytes);
    return output;
}

const PointFoldStats& HyraxV3CudaPointFoldSession::stats() const noexcept {
    return impl_->counters;
}

}  // namespace zkdl_cuda_fold
