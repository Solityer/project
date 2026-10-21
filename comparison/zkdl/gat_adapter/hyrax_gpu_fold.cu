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

struct HyraxFoldTransfer {
    G1Jacobian_t left{};
    G1Jacobian_t right{};
    Fr_t cross_left{};
    Fr_t cross_right{};
};

static_assert(sizeof(HyraxFoldTransfer) ==
              2U * sizeof(G1Jacobian_t) + 2U * sizeof(Fr_t),
              "update the audited CUDA Hyrax transfer layout");

constexpr uint kHyraxMsmThreads = 128U;
constexpr uint kHyraxMsmWarps = 4U;
constexpr uint kHyraxMsmWarpWidth = 32U;
constexpr uint kHyraxMsmBucketCount = 15U;
constexpr uint kHyraxMsmWindowBits = 4U;
constexpr uint kHyraxMsmWindowCount = 64U;
constexpr unsigned int kHyraxFullWarpMask = 0xffffffffU;

static_assert(sizeof(G1Jacobian_t) == 144U,
              "update the audited CUDA G1 point shuffle layout");
static_assert(sizeof(Fr_t) == 32U,
              "update the audited CUDA scalar digit layout");

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

__device__ inline uint hyrax_scalar_nibble(const Fr_t& scalar, uint window) {
    const uint limb = window >> 3U;
    const uint shift = (window & 7U) * kHyraxMsmWindowBits;
    return (scalar.val[limb] >> shift) & 0x0fU;
}

/* Every lane in the warp calls this with the full active mask.  The returned
 * point is the projective point held by source_lane, including Montgomery
 * limbs, without passing through an affine conversion. */
__device__ inline G1Jacobian_t hyrax_shuffle_point(unsigned int mask,
                                                    G1Jacobian_t point,
                                                    uint source_lane) {
    G1Jacobian_t shuffled{};
#pragma unroll
    for (uint limb = 0U; limb < 12U; ++limb) {
        shuffled.x.val[limb] = __shfl_sync(mask, point.x.val[limb], source_lane);
        shuffled.y.val[limb] = __shfl_sync(mask, point.y.val[limb], source_lane);
        shuffled.z.val[limb] = __shfl_sync(mask, point.z.val[limb], source_lane);
    }
    return shuffled;
}

/* Buckets are stored as digit-1..digit-15 at shared indices 0..14.  Warp
 * lanes 1..15 hold those buckets so a four-stage suffix scan creates
 * S_d = sum_{j=d..15} B_j.  Reducing S_1..S_15 yields the exact weighted
 * bucket sum.  Lane zero has no bucket and starts the reduction at identity;
 * lanes 16..31 stay identity.  Every lane executes every full-mask shuffle. */
__device__ inline G1Jacobian_t hyrax_warp_weighted_bucket_sum(
    const G1Jacobian_t* buckets, uint lane, G1Jacobian_t identity) {
    G1Jacobian_t suffix = identity;
    if (lane >= 1U && lane <= kHyraxMsmBucketCount)
        suffix = buckets[lane - 1U];

#pragma unroll
    for (uint offset = 1U; offset <= 8U; offset <<= 1U) {
        const uint requested_source = lane + offset;
        const uint source_lane = requested_source < kHyraxMsmWarpWidth
            ? requested_source : kHyraxMsmWarpWidth - 1U;
        const G1Jacobian_t neighbor = hyrax_shuffle_point(
            kHyraxFullWarpMask, suffix, source_lane);
        if (lane >= 1U && lane <= kHyraxMsmBucketCount &&
            requested_source <= kHyraxMsmBucketCount)
            suffix = G1Jacobian_add_safe(suffix, neighbor);
    }

    /* Lane zero is not a bucket and must contribute identity to the final
     * reduction.  In particular, it must not duplicate S_1. */
    G1Jacobian_t reduction = suffix;
    if (lane == 0U) reduction = identity;
#pragma unroll
    for (uint offset = 8U; offset > 0U; offset >>= 1U) {
        const uint requested_source = lane + offset;
        const uint source_lane = requested_source < kHyraxMsmWarpWidth
            ? requested_source : kHyraxMsmWarpWidth - 1U;
        const G1Jacobian_t neighbor = hyrax_shuffle_point(
            kHyraxFullWarpMask, reduction, source_lane);
        if (lane < offset)
            reduction = G1Jacobian_add_safe(reduction, neighbor);
    }
    return reduction;
}

__device__ inline G1Jacobian_t hyrax_warp_digit_group_sum(
    G1Jacobian_t point, uint digit, uint lane, unsigned int* group_mask_out,
    uint* leader_out) {
    const unsigned int group_mask =
        __match_any_sync(kHyraxFullWarpMask, digit);
    const uint leader = static_cast<uint>(__ffs(group_mask) - 1);
    const uint group_size = static_cast<uint>(__popc(group_mask));
    const uint maximum_group_size =
        __reduce_max_sync(kHyraxFullWarpMask, group_size);
    unsigned int remaining = group_mask;
    G1Jacobian_t sum = blstrs__g1__G1Affine_ZERO;
    bool have_point = false;
    for (uint rank = 0U; rank < maximum_group_size; ++rank) {
        const uint source_lane = remaining != 0U
            ? static_cast<uint>(__ffs(remaining) - 1)
            : leader;
        const G1Jacobian_t source =
            hyrax_shuffle_point(kHyraxFullWarpMask, point, source_lane);
        if (lane == leader && remaining != 0U) {
            if (have_point) sum = G1Jacobian_add_safe(sum, source);
            else {
                sum = source;
                have_point = true;
            }
        }
        remaining &= remaining - 1U;
    }
    *group_mask_out = group_mask;
    *leader_out = leader;
    return sum;
}

/* One CTA owns 128 MSM terms and one 4-bit window.  A warp groups terms with
 * equal nonzero digits, sums each group at its leader, and writes one point
 * to that digit's private warp bucket.  The four warp buckets are combined
 * before the first warp computes the weighted sum with a parallel suffix
 * scan and reduction, equivalent to Pippenger's running-bucket identity. */
extern "C" __global__ void hyrax_lr_msm_window_partials_kernel(
    const G1Jacobian_t* current_g, const Fr_t* current_a,
    G1Jacobian_t* left_partials, G1Jacobian_t* right_partials,
    uint half) {
    __shared__ G1Jacobian_t left_buckets[kHyraxMsmWarps][kHyraxMsmBucketCount];
    __shared__ G1Jacobian_t right_buckets[kHyraxMsmWarps][kHyraxMsmBucketCount];
    const uint lane = threadIdx.x & (kHyraxMsmWarpWidth - 1U);
    const uint warp = threadIdx.x / kHyraxMsmWarpWidth;
    const uint index = blockIdx.x * kHyraxMsmThreads + threadIdx.x;
    const uint window = blockIdx.y;
    const G1Jacobian_t identity = blstrs__g1__G1Affine_ZERO;

    if (lane < kHyraxMsmBucketCount) {
        left_buckets[warp][lane] = identity;
        right_buckets[warp][lane] = identity;
    }
    __syncthreads();

    const bool valid = index < half;
    const G1Jacobian_t left_point = valid ? current_g[half + index] : identity;
    const G1Jacobian_t right_point = valid ? current_g[index] : identity;
    const uint left_digit = valid ? hyrax_scalar_nibble(current_a[index], window) : 0U;
    const uint right_digit = valid
        ? hyrax_scalar_nibble(current_a[half + index], window) : 0U;
    unsigned int left_group = 0U;
    unsigned int right_group = 0U;
    uint left_leader = 0U;
    uint right_leader = 0U;
    /* Every active warp lane participates in both group reductions.  The
     * loop bound is the largest digit group in the warp, usually far below
     * 32, and does not assume that equal digits occupy adjacent lanes. */
    const G1Jacobian_t left_sum = hyrax_warp_digit_group_sum(
        left_point, left_digit, lane, &left_group, &left_leader);
    const G1Jacobian_t right_sum = hyrax_warp_digit_group_sum(
        right_point, right_digit, lane, &right_group, &right_leader);
    if (lane == left_leader && left_digit != 0U)
        left_buckets[warp][left_digit - 1U] = left_sum;
    if (lane == right_leader && right_digit != 0U)
        right_buckets[warp][right_digit - 1U] = right_sum;
    __syncthreads();

    if (warp == 0U && lane < kHyraxMsmBucketCount) {
        G1Jacobian_t left_bucket = G1Jacobian_add_safe(
            left_buckets[0U][lane], left_buckets[1U][lane]);
        left_bucket = G1Jacobian_add_safe(left_bucket, left_buckets[2U][lane]);
        left_bucket = G1Jacobian_add_safe(left_bucket, left_buckets[3U][lane]);
        left_buckets[0U][lane] = left_bucket;

        G1Jacobian_t right_bucket = G1Jacobian_add_safe(
            right_buckets[0U][lane], right_buckets[1U][lane]);
        right_bucket = G1Jacobian_add_safe(right_bucket, right_buckets[2U][lane]);
        right_bucket = G1Jacobian_add_safe(right_bucket, right_buckets[3U][lane]);
        right_buckets[0U][lane] = right_bucket;
    }
    __syncthreads();

    if (warp == 0U) {
        const G1Jacobian_t left_window_sum = hyrax_warp_weighted_bucket_sum(
            left_buckets[0U], lane, identity);
        const G1Jacobian_t right_window_sum = hyrax_warp_weighted_bucket_sum(
            right_buckets[0U], lane, identity);
        const std::size_t partial_index =
            static_cast<std::size_t>(window) * gridDim.x + blockIdx.x;
        if (lane == 0U) {
            left_partials[partial_index] = left_window_sum;
            right_partials[partial_index] = right_window_sum;
        }
    }
}

extern "C" __global__ void hyrax_msm_safe_reduce_windows_kernel(
    const G1Jacobian_t* input, G1Jacobian_t* output,
    uint input_count, uint output_count) {
    const uint window = blockIdx.y;
    const uint output_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_index >= output_count) return;
    const uint left_index = 2U * output_index;
    const uint right_index = left_index + 1U;
    const std::size_t input_offset = static_cast<std::size_t>(window) * input_count;
    const std::size_t output_offset = static_cast<std::size_t>(window) * output_count;
    output[output_offset + output_index] = right_index < input_count
        ? G1Jacobian_add_safe(input[input_offset + left_index],
                              input[input_offset + right_index])
        : input[input_offset + left_index];
}

extern "C" __global__ void hyrax_msm_horner_windows_kernel(
    const G1Jacobian_t* left_window_sums,
    const G1Jacobian_t* right_window_sums,
    G1Jacobian_t* left_result, G1Jacobian_t* right_result) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) return;
    G1Jacobian_t left = blstrs__g1__G1Affine_ZERO;
    G1Jacobian_t right = blstrs__g1__G1Affine_ZERO;
    for (int window = static_cast<int>(kHyraxMsmWindowCount) - 1;
         window >= 0; --window) {
#pragma unroll
        for (uint bit = 0U; bit < kHyraxMsmWindowBits; ++bit) {
            left = blstrs__g1__G1Affine_double(left);
            right = blstrs__g1__G1Affine_double(right);
        }
        left = G1Jacobian_add_safe(
            left, left_window_sums[static_cast<uint>(window)]);
        right = G1Jacobian_add_safe(
            right, right_window_sums[static_cast<uint>(window)]);
    }
    left_result[0] = left;
    right_result[0] = right;
}

extern "C" __global__ void hyrax_scalar_cross_partials_kernel(
    const Fr_t* current_a, const Fr_t* current_b,
    Fr_t* next_a_partials, Fr_t* next_b_partials, uint half) {
    __shared__ Fr_t left_partials[128];
    __shared__ Fr_t right_partials[128];
    const uint lane = threadIdx.x;
    const uint index = blockIdx.x * blockDim.x + lane;
    Fr_t left{}, right{};
    if (index < half) {
        left = Fr_field_mul(current_a[index], current_b[half + index]);
        right = Fr_field_mul(current_a[half + index], current_b[index]);
    }
    left_partials[lane] = left;
    right_partials[lane] = right;
    __syncthreads();
    for (uint stride = 64U; stride != 0U; stride >>= 1U) {
        if (lane < stride) {
            left_partials[lane] = blstrs__scalar__Scalar_add(
                left_partials[lane], left_partials[lane + stride]);
            right_partials[lane] = blstrs__scalar__Scalar_add(
                right_partials[lane], right_partials[lane + stride]);
        }
        __syncthreads();
    }
    if (lane == 0U) {
        next_a_partials[blockIdx.x] = left_partials[0];
        next_b_partials[blockIdx.x] = right_partials[0];
    }
}

extern "C" __global__ void hyrax_scalar_cross_reduce_kernel(
    Fr_t* left_partials, Fr_t* right_partials, uint partial_count) {
    __shared__ Fr_t left_sums[128];
    __shared__ Fr_t right_sums[128];
    const uint lane = threadIdx.x;
    Fr_t left{}, right{};
    for (uint index = lane; index < partial_count; index += blockDim.x) {
        left = blstrs__scalar__Scalar_add(left, left_partials[index]);
        right = blstrs__scalar__Scalar_add(right, right_partials[index]);
    }
    left_sums[lane] = left;
    right_sums[lane] = right;
    __syncthreads();
    for (uint stride = 64U; stride != 0U; stride >>= 1U) {
        if (lane < stride) {
            left_sums[lane] = blstrs__scalar__Scalar_add(
                left_sums[lane], left_sums[lane + stride]);
            right_sums[lane] = blstrs__scalar__Scalar_add(
                right_sums[lane], right_sums[lane + stride]);
        }
        __syncthreads();
    }
    if (lane == 0U) {
        left_partials[0] = left_sums[0];
        right_partials[0] = right_sums[0];
    }
}

extern "C" __global__ void hyrax_scalar_fold_kernel(
    const Fr_t* current_a, const Fr_t* current_b,
    Fr_t* next_a, Fr_t* next_b, Fr_t challenge, Fr_t inverse,
    uint half) {
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= half) return;
    next_a[index] = blstrs__scalar__Scalar_add(
        Fr_field_mul(current_a[index], challenge),
        Fr_field_mul(current_a[half + index], inverse));
    next_b[index] = blstrs__scalar__Scalar_add(
        Fr_field_mul(current_b[index], inverse),
        Fr_field_mul(current_b[half + index], challenge));
}

extern "C" __global__ void hyrax_g_fold_kernel(
    const G1Jacobian_t* current_g, G1Jacobian_t* next_g,
    Fr_t challenge, Fr_t inverse, uint half) {
    __shared__ uint8_t joint_digits[256U];
    __shared__ int highest_nonzero_bit;
    for (uint bit = threadIdx.x; bit < 256U; bit += blockDim.x) {
        const uint limb = bit >> 5U;
        const uint shift = bit & 31U;
        const uint inverse_bit = (inverse.val[limb] >> shift) & 1U;
        const uint challenge_bit = (challenge.val[limb] >> shift) & 1U;
        joint_digits[bit] = static_cast<uint8_t>(inverse_bit | (challenge_bit << 1U));
    }
    __syncthreads();
    if (threadIdx.x == 0U) {
        highest_nonzero_bit = 255;
        while (highest_nonzero_bit >= 0 &&
               joint_digits[highest_nonzero_bit] == 0U)
            --highest_nonzero_bit;
    }
    __syncthreads();
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= half) return;
    const G1Jacobian_t low = current_g[index];
    const G1Jacobian_t high = current_g[half + index];
    const G1Jacobian_t both = G1Jacobian_add_safe(low, high);
    G1Jacobian_t result = blstrs__g1__G1Affine_ZERO;
    for (int bit = highest_nonzero_bit; bit >= 0; --bit) {
        result = blstrs__g1__G1Affine_double(result);
        switch (joint_digits[bit]) {
            case 1U: result = G1Jacobian_add_safe(result, low); break;
            case 2U: result = G1Jacobian_add_safe(result, high); break;
            case 3U: result = G1Jacobian_add_safe(result, both); break;
            default: break;
        }
    }
    next_g[index] = result;
}

extern "C" __global__ void hyrax_fill_identity_tail_kernel(
    G1Jacobian_t* points, uint count, G1Jacobian_t identity) {
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) points[index] = identity;
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

extern "C" __global__ void hyrax_pack_round_transfer_kernel(
    const G1Jacobian_t* left, const G1Jacobian_t* right,
    const Fr_t* cross_left, const Fr_t* cross_right,
    HyraxFoldTransfer* transfer) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) return;
    transfer->left = left[0];
    transfer->right = right[0];
    transfer->cross_left = cross_left[0];
    transfer->cross_right = cross_right[0];
}

extern "C" __global__ void hyrax_pack_final_transfer_kernel(
    const G1Jacobian_t* generator, const Fr_t* final_a,
    const Fr_t* final_b, HyraxFoldTransfer* transfer) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) return;
    transfer->left = generator[0];
    transfer->right = G1Jacobian_t{};
    transfer->cross_left = final_a[0];
    transfer->cross_right = final_b[0];
}

class NativeHyraxGpuFoldSession final : public CpuHyraxGpuFoldSession {
public:
    NativeHyraxGpuFoldSession(const CpuIpaBasisContext& basis,
                              const CpuG1Point& h, const CpuG1Point& u)
        : basis_(basis), basis_device_(allocations_),
          current_g_(allocations_), next_g_(allocations_),
          current_a_(allocations_), next_a_(allocations_),
          current_b_(allocations_), next_b_(allocations_),
          transfer_device_(allocations_), left_products_(allocations_),
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
        next_a_.ensure(current_size_, "allocate next Hyrax a fold");
        current_b_.ensure(current_size_, "allocate Hyrax b fold input");
        next_b_.ensure(current_size_, "allocate next Hyrax b fold");
        transfer_device_.ensure(1U, "allocate Hyrax fold transfer");
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
        pending_kernel_count_ = 0U;
        if (current_size_ > generator_count_) {
            const G1Jacobian_t native_identity =
                cpu_g1_to_native(CpuG1Point::zero());
            const uint identity_tail_count = static_cast<uint>(
                current_size_ - generator_count_);
            const uint blocks =
                1U + (identity_tail_count - 1U) / 128U;
            hyrax_fill_identity_tail_kernel<<<blocks, 128U>>>(
                current_g_.data() + generator_count_, identity_tail_count,
                native_identity);
            launch_checked("launch Hyrax G identity tail fill");
        }
        current_g_ptr_ = current_g_.data();
        next_g_ptr_ = next_g_.data();
        current_a_ptr_ = current_a_.data();
        next_a_ptr_ = next_a_.data();
        current_b_ptr_ = current_b_.data();
        next_b_ptr_ = next_b_.data();
        scalars_initialized_ = false;
    }

    void initialize_scalars(const std::vector<FrWire>& current_a,
                            const std::vector<FrWire>& current_b) override {
        if (current_size_ == 0U || current_a.size() != current_size_ ||
            current_b.size() != current_size_)
            throw std::invalid_argument("CUDA Hyrax scalar initialization shape mismatch");
        std::vector<Fr_t> native_scalars(current_size_);
        for (std::size_t index = 0U; index < current_size_; ++index) {
            if (!fr_canonical(current_a[index]))
                throw std::invalid_argument("non-canonical CUDA Hyrax a scalar");
            native_scalars[index] = fr_value(current_a[index]);
        }
        cuda_check(cudaMemcpy(current_a_.data(), native_scalars.data(),
                              current_size_ * sizeof(Fr_t),
                              cudaMemcpyHostToDevice),
                   "upload initial Hyrax a scalars");
        for (std::size_t index = 0U; index < current_size_; ++index) {
            if (!fr_canonical(current_b[index]))
                throw std::invalid_argument("non-canonical CUDA Hyrax b scalar");
            native_scalars[index] = fr_value(current_b[index]);
        }
        cuda_check(cudaMemcpy(current_b_.data(), native_scalars.data(),
                              current_size_ * sizeof(Fr_t),
                              cudaMemcpyHostToDevice),
                   "upload initial Hyrax b scalars");
        current_a_ptr_ = current_a_.data();
        next_a_ptr_ = next_a_.data();
        current_b_ptr_ = current_b_.data();
        next_b_ptr_ = next_b_.data();
        scalars_initialized_ = true;
    }

    CpuHyraxGpuFoldRound compute_lr(std::size_t current_size) override {
        if (current_size < 2U || current_size != current_size_ ||
            (current_size & (current_size - 1U)) != 0U || !scalars_initialized_)
            throw std::invalid_argument("CUDA Hyrax L/R fold shape mismatch");
        const uint half = static_cast<uint>(current_size / 2U);
        const uint blocks = 1U + (half - 1U) / 128U;

        G1Jacobian_t* left_final = nullptr;
        G1Jacobian_t* right_final = nullptr;
        if (half >= kHyraxMsmThreads) {
            const std::uint64_t partial_count =
                static_cast<std::uint64_t>(kHyraxMsmWindowCount) * blocks;
            if (partial_count > half)
                throw std::logic_error(
                    "CUDA Hyrax window MSM exceeds the resident product workspace");
            hyrax_lr_msm_window_partials_kernel<<<
                dim3(blocks, kHyraxMsmWindowCount), kHyraxMsmThreads>>>(
                    current_g_ptr_, current_a_ptr_, left_products_.data(),
                    right_products_.data(), half);
            launch_checked("launch Hyrax windowed L/R MSM buckets");

            std::uint32_t active = blocks;
            G1Jacobian_t* left_input = left_products_.data();
            G1Jacobian_t* left_output = left_scratch_.data();
            G1Jacobian_t* right_input = right_products_.data();
            G1Jacobian_t* right_output = right_scratch_.data();
            while (active > 1U) {
                const uint output_count = (active + 1U) / 2U;
                const uint reduction_blocks =
                    (output_count + 127U) / 128U;
                hyrax_msm_safe_reduce_windows_kernel<<<
                    dim3(reduction_blocks, kHyraxMsmWindowCount), 128U>>>(
                    left_input, left_output, active, output_count);
                launch_checked("reduce Hyrax left MSM windows");
                hyrax_msm_safe_reduce_windows_kernel<<<
                    dim3(reduction_blocks, kHyraxMsmWindowCount), 128U>>>(
                    right_input, right_output, active, output_count);
                launch_checked("reduce Hyrax right MSM windows");
                std::swap(left_input, left_output);
                std::swap(right_input, right_output);
                active = output_count;
            }
            hyrax_msm_horner_windows_kernel<<<1U, 1U>>>(
                left_input, right_input, left_output, right_output);
            launch_checked("combine Hyrax L/R MSM windows");
            left_final = left_output;
            right_final = right_output;
        } else {
            hyrax_lr_products_kernel<<<blocks, 128U>>>(
                current_g_ptr_, current_a_ptr_, left_products_.data(),
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
            left_final = left_input;
            right_final = right_input;
        }

        hyrax_scalar_cross_partials_kernel<<<blocks, 128U>>>(
            current_a_ptr_, current_b_ptr_, next_a_ptr_, next_b_ptr_, half);
        launch_checked("launch Hyrax scalar cross-product partials");
        hyrax_scalar_cross_reduce_kernel<<<1U, 128U>>>(
            next_a_ptr_, next_b_ptr_, blocks);
        launch_checked("launch Hyrax scalar cross-product reduction");
        hyrax_pack_round_transfer_kernel<<<1U, 1U>>>(
            left_final, right_final, next_a_ptr_, next_b_ptr_,
            transfer_device_.data());
        launch_checked("pack Hyrax L/R and cross terms");
        HyraxFoldTransfer transfer{};
        cuda_check(cudaMemcpy(&transfer, transfer_device_.data(), sizeof(transfer),
                              cudaMemcpyDeviceToHost),
                   "download Hyrax round commitments and cross terms");
        record_completed_kernels();
        return {cpu_g1_from_native(transfer.left),
                cpu_g1_from_native(transfer.right),
                fr_wire(transfer.cross_left), fr_wire(transfer.cross_right)};
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
        hyrax_scalar_fold_kernel<<<blocks, 128U>>>(
            current_a_ptr_, current_b_ptr_, next_a_ptr_, next_b_ptr_,
            fr_value(challenge), fr_value(inverse), half);
        launch_checked("launch Hyrax a/b scalar fold");
        std::swap(current_g_ptr_, next_g_ptr_);
        std::swap(current_a_ptr_, next_a_ptr_);
        std::swap(current_b_ptr_, next_b_ptr_);
        current_size_ = half_size;
    }

    CpuHyraxGpuFoldFinalState final_state() override {
        if (current_size_ != 1U || !scalars_initialized_)
            throw std::invalid_argument("CUDA Hyrax G fold did not reach one point");
        hyrax_pack_final_transfer_kernel<<<1U, 1U>>>(
            current_g_ptr_, current_a_ptr_, current_b_ptr_, transfer_device_.data());
        launch_checked("pack Hyrax final generator and scalars");
        HyraxFoldTransfer transfer{};
        cuda_check(cudaMemcpy(&transfer, transfer_device_.data(), sizeof(transfer),
                              cudaMemcpyDeviceToHost),
                   "download final Hyrax generator and scalars");
        record_completed_kernels();
        return {cpu_g1_from_native(transfer.left), fr_wire(transfer.cross_left),
                fr_wire(transfer.cross_right)};
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

    void record_completed_kernels() {
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
    DeviceBuffer<Fr_t> next_a_;
    DeviceBuffer<Fr_t> current_b_;
    DeviceBuffer<Fr_t> next_b_;
    DeviceBuffer<HyraxFoldTransfer> transfer_device_;
    DeviceBuffer<G1Jacobian_t> left_products_;
    DeviceBuffer<G1Jacobian_t> right_products_;
    DeviceBuffer<G1Jacobian_t> left_scratch_;
    DeviceBuffer<G1Jacobian_t> right_scratch_;
    G1Jacobian_t* current_g_ptr_ = nullptr;
    G1Jacobian_t* next_g_ptr_ = nullptr;
    Fr_t* current_a_ptr_ = nullptr;
    Fr_t* next_a_ptr_ = nullptr;
    Fr_t* current_b_ptr_ = nullptr;
    Fr_t* next_b_ptr_ = nullptr;
    std::size_t generator_count_ = 0U;
    std::size_t current_size_ = 0U;
    bool scalars_initialized_ = false;
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
        4U * sizeof(G1Jacobian_t) + 4U * sizeof(Fr_t);
    constexpr std::uint64_t kRoundTransferBytes = sizeof(HyraxFoldTransfer);
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
    if (fold_bytes > std::numeric_limits<std::uint64_t>::max() - basis_bytes ||
        kRoundTransferBytes > std::numeric_limits<std::uint64_t>::max() -
                                  basis_bytes - fold_bytes) {
        if (error) *error = "CUDA Hyrax workspace byte calculation overflows";
        return false;
    }
    if (!query_cpu_hyrax_gpu_device_info(info, error)) return false;
    info->authenticated_basis_size = authenticated_basis_size;
    info->max_opening_padded_generator_size =
        max_opening_padded_generator_size;
    info->workspace_required_bytes = basis_bytes + fold_bytes + kRoundTransferBytes;
    return true;
}

}  // namespace zkdl
