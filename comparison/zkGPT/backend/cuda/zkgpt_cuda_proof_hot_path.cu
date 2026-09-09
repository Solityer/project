#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Native CUDA field-operation provider for the formal zkGPT path.  The input
// wire is explicit rather than a post-hoc digest stream:
//
//   ZKGPT_GPU_BATCH_V3
//   LASSO <count> <beta limb0..3> <count field words>
//   SUMCHECK <count> <rounds> <round challenges> <count field words>
//   SUMCHECK_PRODUCT2 <count> <rounds> <round challenges> <left> <right>
//   SUMCHECK_PRODUCT3 <count> <rounds> <round challenges> <selector> <left> <right>
//   HYRAX <count> <rounds> <round challenges> <count field words>
//   END
//
// Every field word is four little-endian u64 limbs.  The host only parses and
// copies canonical inputs; operation outputs are computed by CUDA kernels and
// copied back as canonical BN254-Fr proof-message scalars.

struct FieldWord {
    unsigned long long limb[4];
};

struct RoundResult {
    FieldWord challenge;
    std::vector<FieldWord> coefficients;
    FieldWord folded_claim;
};

enum class OperationKind {
    Lasso,
    Sumcheck,
    SumcheckProduct2,
    SumcheckProduct3,
    Hyrax,
};

struct Operation {
    OperationKind kind;
    bool allow_partial = false;
    std::vector<FieldWord> values;
    std::vector<FieldWord> values_right;
    std::vector<FieldWord> values_third;
    std::vector<FieldWord> challenges;
    FieldWord beta;
};

struct OperationResult {
    OperationKind kind;
    std::size_t input_count = 0;
    std::size_t padded_count = 0;
    std::size_t operand_count = 1;
    std::vector<RoundResult> rounds;
    bool complete = true;
    FieldWord terminal = {{0ULL, 0ULL, 0ULL, 0ULL}};
};

struct KernelCounters {
    std::size_t total = 0;
    std::size_t lasso = 0;
    std::size_t matmul_sumcheck = 0;
    std::size_t gkr_relation = 0;
    std::size_t hyrax = 0;
    std::size_t transcript = 0;
};

struct TimingCounters {
    double memory_copy_h2d_ms = 0.0;
    double memory_copy_d2h_ms = 0.0;
    double synchronization_time_ms = 0.0;
};

static void check(cudaError_t error, const char* what);

static KernelCounters g_kernel_counters;
static TimingCounters g_timing_counters;

enum class KernelClass {
    Lasso,
    MatMulSumcheck,
    GkrRelation,
    Hyrax,
    Transcript,
};

static KernelClass g_product_kernel_class = KernelClass::GkrRelation;

static void classify_product_kernel(const std::string& label)
{
    if (label.find("/matmul") != std::string::npos || label.find("matmul") != std::string::npos) {
        g_product_kernel_class = KernelClass::MatMulSumcheck;
    } else if (label.find("/gkr") != std::string::npos || label.find("gkr") != std::string::npos) {
        g_product_kernel_class = KernelClass::GkrRelation;
    } else {
        // Product operations without an explicit native relation label are
        // not accepted as a formal GPU claim; the counter remains useful for
        // probe diagnostics while Python-side evidence rejects the unknown
        // relation identity.
        g_product_kernel_class = KernelClass::GkrRelation;
    }
}

static void note_kernel(KernelClass kind)
{
    ++g_kernel_counters.total;
    switch (kind) {
    case KernelClass::Lasso: ++g_kernel_counters.lasso; break;
    case KernelClass::MatMulSumcheck: ++g_kernel_counters.matmul_sumcheck; break;
    case KernelClass::GkrRelation: ++g_kernel_counters.gkr_relation; break;
    case KernelClass::Hyrax: ++g_kernel_counters.hyrax; break;
    case KernelClass::Transcript: ++g_kernel_counters.transcript; break;
    }
}

static double elapsed_ms(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
}

static void timed_device_synchronize(const char* what)
{
    const auto start = std::chrono::steady_clock::now();
    check(cudaDeviceSynchronize(), what);
    g_timing_counters.synchronization_time_ms += elapsed_ms(start);
}

static void timed_event_synchronize(cudaEvent_t event, const char* what)
{
    const auto start = std::chrono::steady_clock::now();
    check(cudaEventSynchronize(event), what);
    g_timing_counters.synchronization_time_ms += elapsed_ms(start);
}

static void timed_memcpy(
    void* destination,
    const void* source,
    std::size_t bytes,
    cudaMemcpyKind kind,
    const char* what)
{
    const auto start = std::chrono::steady_clock::now();
    check(cudaMemcpy(destination, source, bytes, kind), what);
    const double duration = elapsed_ms(start);
    if (kind == cudaMemcpyHostToDevice) {
        g_timing_counters.memory_copy_h2d_ms += duration;
    } else if (kind == cudaMemcpyDeviceToHost) {
        g_timing_counters.memory_copy_d2h_ms += duration;
    }
}

__host__ __device__ __forceinline__ FieldWord word_zero()
{
    return {{0ULL, 0ULL, 0ULL, 0ULL}};
}

__host__ __device__ __forceinline__ FieldWord word_one()
{
    return {{1ULL, 0ULL, 0ULL, 0ULL}};
}

__host__ __device__ __forceinline__ FieldWord fr_modulus()
{
    return {{
        0x43e1f593f0000001ULL,
        0x2833e84879b97091ULL,
        0xb85045b68181585dULL,
        0x30644e72e131a029ULL,
    }};
}

__host__ __device__ __forceinline__ bool word_equal(FieldWord left, FieldWord right)
{
    return left.limb[0] == right.limb[0]
        && left.limb[1] == right.limb[1]
        && left.limb[2] == right.limb[2]
        && left.limb[3] == right.limb[3];
}

__host__ __device__ __forceinline__ bool word_ge(FieldWord left, FieldWord right)
{
    for (int index = 3; index >= 0; --index) {
        if (left.limb[index] != right.limb[index]) {
            return left.limb[index] > right.limb[index];
        }
    }
    return true;
}

__host__ __device__ __forceinline__ unsigned long long add_with_carry(
    unsigned long long left,
    unsigned long long right,
    unsigned long long carry,
    unsigned long long* next_carry)
{
    const unsigned long long first = left + right;
    const unsigned long long carry_first = first < left ? 1ULL : 0ULL;
    const unsigned long long result = first + carry;
    const unsigned long long carry_second = result < first ? 1ULL : 0ULL;
    *next_carry = carry_first | carry_second;
    return result;
}

__host__ __device__ __forceinline__ FieldWord word_add_raw(FieldWord left, FieldWord right)
{
    FieldWord out = word_zero();
    unsigned long long carry = 0ULL;
    for (int index = 0; index < 4; ++index) {
        out.limb[index] = add_with_carry(left.limb[index], right.limb[index], carry, &carry);
    }
    return out;
}

__host__ __device__ __forceinline__ FieldWord word_sub_raw(FieldWord left, FieldWord right)
{
    FieldWord out = word_zero();
    unsigned long long borrow = 0ULL;
    for (int index = 0; index < 4; ++index) {
        const unsigned long long right_with_borrow = right.limb[index] + borrow;
        const unsigned long long carry = right_with_borrow < right.limb[index] ? 1ULL : 0ULL;
        out.limb[index] = left.limb[index] - right_with_borrow;
        borrow = (left.limb[index] < right_with_borrow) | carry;
    }
    return out;
}

__host__ __device__ __forceinline__ FieldWord fr_add(FieldWord left, FieldWord right)
{
    // Canonical Fr values are below 2^254, so their sum cannot overflow the
    // four-limb container.  Reduction is performed on every addition.
    FieldWord out = word_add_raw(left, right);
    const FieldWord modulus = fr_modulus();
    if (word_ge(out, modulus)) {
        out = word_sub_raw(out, modulus);
    }
    return out;
}

__host__ __device__ __forceinline__ FieldWord fr_sub(FieldWord left, FieldWord right)
{
    if (word_ge(left, right)) {
        return word_sub_raw(left, right);
    }
    return word_sub_raw(fr_modulus(), word_sub_raw(right, left));
}

__host__ __device__ __forceinline__ FieldWord fr_mul(FieldWord left, FieldWord right)
{
    // Constant-time binary multiplication over all four limbs.  This is a
    // real Fr operation, not a low-limb checksum: every intermediate is
    // reduced modulo the protocol scalar modulus.
    FieldWord result = word_zero();
    FieldWord addend = left;
    for (int limb = 0; limb < 4; ++limb) {
        unsigned long long bits = right.limb[limb];
        for (int bit = 0; bit < 64; ++bit) {
            if ((bits >> bit) & 1ULL) {
                result = fr_add(result, addend);
            }
            addend = fr_add(addend, addend);
        }
    }
    return result;
}

static void check(cudaError_t error, const char* what)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
    }
}

static bool is_power_of_two(std::size_t value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

static std::size_t next_power_of_two(std::size_t value)
{
    std::size_t result = 1;
    while (result < value) {
        if (result > (std::numeric_limits<std::size_t>::max() >> 1)) {
            throw std::runtime_error("field vector is too large");
        }
        result <<= 1;
    }
    return result;
}

static std::size_t log2_power_of_two(std::size_t value)
{
    if (!is_power_of_two(value)) {
        throw std::runtime_error("field vector length is not a power of two");
    }
    std::size_t result = 0;
    while (value > 1) {
        value >>= 1;
        ++result;
    }
    return result;
}

// Lasso leaves are beta + query.  Padding leaves are one, so the root is the
// exact padded grand product used as a proof message.
__global__ void lasso_product_tree_init_kernel(
    const FieldWord* values,
    std::size_t count,
    FieldWord beta,
    FieldWord* tree,
    std::size_t padded_count)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = blockDim.x * gridDim.x;
    for (std::size_t position = index; position < padded_count; position += stride) {
        tree[position] = position < count ? fr_add(beta, values[position]) : word_one();
    }
}

__global__ void lasso_product_tree_level_kernel(
    const FieldWord* input,
    FieldWord* output,
    std::size_t count)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t pair_count = count >> 1;
    if (index < pair_count) {
        output[index] = fr_mul(input[index << 1], input[(index << 1) + 1]);
    }
}

// For an MLE table, one Sumcheck round is
// g(t)=sum_i(left_i+t*(right_i-left_i)); the two canonical coefficients and
// the subsequent challenge fold are computed by CUDA kernels.
__global__ void sumcheck_round_coefficients_kernel(
    const FieldWord* values,
    std::size_t count,
    FieldWord* coefficients)
{
    __shared__ FieldWord left_sums[256];
    __shared__ FieldWord right_sums[256];
    const unsigned int thread = threadIdx.x;
    FieldWord left = word_zero();
    FieldWord right = word_zero();
    const std::size_t pair_count = count >> 1;
    for (std::size_t pair = thread; pair < pair_count; pair += blockDim.x) {
        left = fr_add(left, values[pair << 1]);
        right = fr_add(right, values[(pair << 1) + 1]);
    }
    left_sums[thread] = left;
    right_sums[thread] = right;
    __syncthreads();
    for (unsigned int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if (thread < step) {
            left_sums[thread] = fr_add(left_sums[thread], left_sums[thread + step]);
            right_sums[thread] = fr_add(right_sums[thread], right_sums[thread + step]);
        }
        __syncthreads();
    }
    if (thread == 0) {
        coefficients[0] = left_sums[0];
        coefficients[1] = fr_sub(right_sums[0], left_sums[0]);
    }
}

__global__ void product2_round_coefficients_kernel(
    const FieldWord* left_values,
    const FieldWord* right_values,
    std::size_t count,
    FieldWord* coefficients)
{
    __shared__ FieldWord c0_sums[256];
    __shared__ FieldWord c1_sums[256];
    __shared__ FieldWord c2_sums[256];
    const unsigned int thread = threadIdx.x;
    FieldWord c0 = word_zero();
    FieldWord c1 = word_zero();
    FieldWord c2 = word_zero();
    const std::size_t pair_count = count >> 1;
    for (std::size_t pair = thread; pair < pair_count; pair += blockDim.x) {
        const FieldWord a0 = left_values[pair << 1];
        const FieldWord a1 = left_values[(pair << 1) + 1];
        const FieldWord b0 = right_values[pair << 1];
        const FieldWord b1 = right_values[(pair << 1) + 1];
        const FieldWord da = fr_sub(a1, a0);
        const FieldWord db = fr_sub(b1, b0);
        c0 = fr_add(c0, fr_mul(a0, b0));
        c1 = fr_add(c1, fr_add(fr_mul(da, b0), fr_mul(a0, db)));
        c2 = fr_add(c2, fr_mul(da, db));
    }
    c0_sums[thread] = c0;
    c1_sums[thread] = c1;
    c2_sums[thread] = c2;
    __syncthreads();
    for (unsigned int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if (thread < step) {
            c0_sums[thread] = fr_add(c0_sums[thread], c0_sums[thread + step]);
            c1_sums[thread] = fr_add(c1_sums[thread], c1_sums[thread + step]);
            c2_sums[thread] = fr_add(c2_sums[thread], c2_sums[thread + step]);
        }
        __syncthreads();
    }
    if (thread == 0) {
        coefficients[0] = c0_sums[0];
        coefficients[1] = c1_sums[0];
        coefficients[2] = c2_sums[0];
    }
}

__global__ void product3_round_coefficients_kernel(
    const FieldWord* selector_values,
    const FieldWord* left_values,
    const FieldWord* right_values,
    std::size_t count,
    FieldWord* coefficients)
{
    __shared__ FieldWord c0_sums[256];
    __shared__ FieldWord c1_sums[256];
    __shared__ FieldWord c2_sums[256];
    __shared__ FieldWord c3_sums[256];
    const unsigned int thread = threadIdx.x;
    FieldWord c0 = word_zero();
    FieldWord c1 = word_zero();
    FieldWord c2 = word_zero();
    FieldWord c3 = word_zero();
    const std::size_t pair_count = count >> 1;
    for (std::size_t pair = thread; pair < pair_count; pair += blockDim.x) {
        const FieldWord s0 = selector_values[pair << 1];
        const FieldWord s1 = selector_values[(pair << 1) + 1];
        const FieldWord a0 = left_values[pair << 1];
        const FieldWord a1 = left_values[(pair << 1) + 1];
        const FieldWord b0 = right_values[pair << 1];
        const FieldWord b1 = right_values[(pair << 1) + 1];
        const FieldWord ds = fr_sub(s1, s0);
        const FieldWord da = fr_sub(a1, a0);
        const FieldWord db = fr_sub(b1, b0);
        c0 = fr_add(c0, fr_mul(fr_mul(s0, a0), b0));
        c1 = fr_add(c1, fr_add(fr_add(fr_mul(fr_mul(ds, a0), b0), fr_mul(fr_mul(s0, da), b0)), fr_mul(fr_mul(s0, a0), db)));
        c2 = fr_add(c2, fr_add(fr_add(fr_mul(fr_mul(ds, da), b0), fr_mul(fr_mul(ds, a0), db)), fr_mul(fr_mul(s0, da), db)));
        c3 = fr_add(c3, fr_mul(fr_mul(ds, da), db));
    }
    c0_sums[thread] = c0;
    c1_sums[thread] = c1;
    c2_sums[thread] = c2;
    c3_sums[thread] = c3;
    __syncthreads();
    for (unsigned int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if (thread < step) {
            c0_sums[thread] = fr_add(c0_sums[thread], c0_sums[thread + step]);
            c1_sums[thread] = fr_add(c1_sums[thread], c1_sums[thread + step]);
            c2_sums[thread] = fr_add(c2_sums[thread], c2_sums[thread + step]);
            c3_sums[thread] = fr_add(c3_sums[thread], c3_sums[thread + step]);
        }
        __syncthreads();
    }
    if (thread == 0) {
        coefficients[0] = c0_sums[0];
        coefficients[1] = c1_sums[0];
        coefficients[2] = c2_sums[0];
        coefficients[3] = c3_sums[0];
    }
}

__global__ void multilinear_fold_kernel(
    const FieldWord* values,
    std::size_t count,
    FieldWord challenge,
    FieldWord* folded)
{
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t pair_count = count >> 1;
    if (index < pair_count) {
        const FieldWord left = values[index << 1];
        const FieldWord right = values[(index << 1) + 1];
        folded[index] = fr_add(left, fr_mul(challenge, fr_sub(right, left)));
    }
}

__global__ void product_terminal_claim_kernel(
    const FieldWord* first,
    const FieldWord* second,
    const FieldWord* third,
    bool cubic,
    FieldWord* output)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const FieldWord pair = fr_mul(first[0], second[0]);
        *output = cubic ? fr_mul(pair, third[0]) : pair;
    }
}

// The transcript accumulator is also a field operation on the device.  It
// binds the exact operation inputs and outputs without using a host-only
// post-hoc hash as the proof result.
__global__ void transcript_field_accumulator_kernel(
    const FieldWord* stream,
    std::size_t count,
    FieldWord base,
    FieldWord* output)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        FieldWord state = word_zero();
        for (std::size_t index = 0; index < count; ++index) {
            state = fr_add(fr_mul(state, base), stream[index]);
        }
        *output = state;
    }
}

static int grid_for(std::size_t count)
{
    constexpr int threads = 256;
    const std::size_t needed = (count + threads - 1) / threads;
    return static_cast<int>(std::max<std::size_t>(1, std::min<std::size_t>(needed, 1024)));
}

static FieldWord run_lasso_gpu(const Operation& operation, OperationResult* result)
{
    const std::size_t count = operation.values.size();
    const std::size_t padded = next_power_of_two(count);
    result->input_count = count;
    result->padded_count = padded;
    result->operand_count = 1;
    FieldWord* device_values = nullptr;
    FieldWord* device_tree_a = nullptr;
    FieldWord* device_tree_b = nullptr;
    try {
        check(cudaMalloc(&device_values, sizeof(FieldWord) * count), "cudaMalloc(lasso values)");
        check(cudaMalloc(&device_tree_a, sizeof(FieldWord) * padded), "cudaMalloc(lasso tree a)");
        check(cudaMalloc(&device_tree_b, sizeof(FieldWord) * padded), "cudaMalloc(lasso tree b)");
        timed_memcpy(device_values, operation.values.data(), sizeof(FieldWord) * count, cudaMemcpyHostToDevice, "cudaMemcpy(lasso values)");
        note_kernel(KernelClass::Lasso);
        lasso_product_tree_init_kernel<<<grid_for(padded), 256>>>(
            device_values, count, operation.beta, device_tree_a, padded);
        check(cudaGetLastError(), "lasso_product_tree_init_kernel launch");
        timed_device_synchronize("lasso_product_tree_init_kernel synchronize");
        std::size_t current_count = padded;
        FieldWord* current = device_tree_a;
        FieldWord* next = device_tree_b;
        while (current_count > 1) {
            note_kernel(KernelClass::Lasso);
            lasso_product_tree_level_kernel<<<grid_for(current_count >> 1), 256>>>(current, next, current_count);
            check(cudaGetLastError(), "lasso_product_tree_level_kernel launch");
            timed_device_synchronize("lasso_product_tree_level_kernel synchronize");
            std::swap(current, next);
            current_count >>= 1;
        }
        FieldWord terminal = word_zero();
        timed_memcpy(&terminal, current, sizeof(FieldWord), cudaMemcpyDeviceToHost, "cudaMemcpy(lasso root)");
        check(cudaFree(device_values), "cudaFree(lasso values)");
        check(cudaFree(device_tree_a), "cudaFree(lasso tree a)");
        check(cudaFree(device_tree_b), "cudaFree(lasso tree b)");
        result->terminal = terminal;
        return terminal;
    } catch (...) {
        if (device_values != nullptr) cudaFree(device_values);
        if (device_tree_a != nullptr) cudaFree(device_tree_a);
        if (device_tree_b != nullptr) cudaFree(device_tree_b);
        throw;
    }
}

static OperationResult run_sumcheck_gpu(const Operation& operation)
{
    OperationResult result;
    result.kind = OperationKind::Sumcheck;
    result.input_count = operation.values.size();
    result.padded_count = operation.values.size();
    result.operand_count = 1;
    const std::size_t expected_rounds = log2_power_of_two(operation.values.size());
    if (operation.challenges.size() > expected_rounds
        || (!operation.allow_partial && operation.challenges.size() != expected_rounds)) {
        throw std::runtime_error("sumcheck challenge count does not match log2(input count)");
    }
    result.complete = operation.challenges.size() == expected_rounds;
    FieldWord* device_current = nullptr;
    FieldWord* device_next = nullptr;
    FieldWord* device_coefficients = nullptr;
    try {
        const std::size_t count = operation.values.size();
        check(cudaMalloc(&device_current, sizeof(FieldWord) * count), "cudaMalloc(sumcheck current)");
        check(cudaMalloc(&device_next, sizeof(FieldWord) * count), "cudaMalloc(sumcheck next)");
        check(cudaMalloc(&device_coefficients, sizeof(FieldWord) * 2), "cudaMalloc(sumcheck coefficients)");
        timed_memcpy(device_current, operation.values.data(), sizeof(FieldWord) * count, cudaMemcpyHostToDevice, "cudaMemcpy(sumcheck values)");
        std::size_t current_count = count;
        for (const FieldWord challenge : operation.challenges) {
            note_kernel(KernelClass::GkrRelation);
            sumcheck_round_coefficients_kernel<<<1, 256>>>(device_current, current_count, device_coefficients);
            check(cudaGetLastError(), "sumcheck_round_coefficients_kernel launch");
            timed_device_synchronize("sumcheck_round_coefficients_kernel synchronize");
            FieldWord coefficients[2] = {word_zero(), word_zero()};
            timed_memcpy(coefficients, device_coefficients, sizeof(coefficients), cudaMemcpyDeviceToHost, "cudaMemcpy(sumcheck coefficients)");
            note_kernel(KernelClass::GkrRelation);
            multilinear_fold_kernel<<<grid_for(current_count >> 1), 256>>>(
                device_current, current_count, challenge, device_next);
            check(cudaGetLastError(), "sumcheck multilinear_fold_kernel launch");
            timed_device_synchronize("sumcheck multilinear_fold_kernel synchronize");
            RoundResult round{};
            round.challenge = challenge;
            round.coefficients.assign(coefficients, coefficients + 2);
            std::swap(device_current, device_next);
            current_count >>= 1;
            timed_memcpy(&round.folded_claim, device_current, sizeof(FieldWord), cudaMemcpyDeviceToHost, "cudaMemcpy(sumcheck folded claim)");
            result.rounds.push_back(round);
        }
        timed_memcpy(&result.terminal, device_current, sizeof(FieldWord), cudaMemcpyDeviceToHost, "cudaMemcpy(sumcheck terminal)");
        check(cudaFree(device_current), "cudaFree(sumcheck current)");
        check(cudaFree(device_next), "cudaFree(sumcheck next)");
        check(cudaFree(device_coefficients), "cudaFree(sumcheck coefficients)");
        return result;
    } catch (...) {
        if (device_current != nullptr) cudaFree(device_current);
        if (device_next != nullptr) cudaFree(device_next);
        if (device_coefficients != nullptr) cudaFree(device_coefficients);
        throw;
    }
}

static OperationResult run_product_sumcheck_gpu(const Operation& operation, bool cubic)
{
    OperationResult result;
    result.kind = operation.kind;
    result.input_count = operation.values.size();
    result.padded_count = operation.values.size();
    result.operand_count = cubic ? 3 : 2;
    const std::size_t expected_rounds = log2_power_of_two(operation.values.size());
    if (operation.challenges.size() > expected_rounds
        || (!operation.allow_partial && operation.challenges.size() != expected_rounds)) {
        throw std::runtime_error("product Sumcheck challenge count does not match log2(input count)");
    }
    result.complete = operation.challenges.size() == expected_rounds;
    if (operation.values_right.size() != operation.values.size()
        || (cubic && operation.values_third.size() != operation.values.size())) {
        throw std::runtime_error("product Sumcheck operand lengths do not match");
    }
    FieldWord* device_a = nullptr;
    FieldWord* device_a_next = nullptr;
    FieldWord* device_b = nullptr;
    FieldWord* device_b_next = nullptr;
    FieldWord* device_c = nullptr;
    FieldWord* device_c_next = nullptr;
    FieldWord* device_coefficients = nullptr;
    FieldWord* device_claim = nullptr;
    try {
        const std::size_t count = operation.values.size();
        const std::size_t coefficient_count = cubic ? 4 : 3;
        check(cudaMalloc(&device_a, sizeof(FieldWord) * count), "cudaMalloc(product Sumcheck a)");
        check(cudaMalloc(&device_a_next, sizeof(FieldWord) * count), "cudaMalloc(product Sumcheck a next)");
        check(cudaMalloc(&device_b, sizeof(FieldWord) * count), "cudaMalloc(product Sumcheck b)");
        check(cudaMalloc(&device_b_next, sizeof(FieldWord) * count), "cudaMalloc(product Sumcheck b next)");
        if (cubic) {
            check(cudaMalloc(&device_c, sizeof(FieldWord) * count), "cudaMalloc(product Sumcheck c)");
            check(cudaMalloc(&device_c_next, sizeof(FieldWord) * count), "cudaMalloc(product Sumcheck c next)");
        }
        check(cudaMalloc(&device_coefficients, sizeof(FieldWord) * coefficient_count), "cudaMalloc(product Sumcheck coefficients)");
        check(cudaMalloc(&device_claim, sizeof(FieldWord)), "cudaMalloc(product Sumcheck terminal claim)");
        if (cubic) {
            timed_memcpy(device_a, operation.values.data(), sizeof(FieldWord) * count, cudaMemcpyHostToDevice, "cudaMemcpy(product Sumcheck selector)");
            timed_memcpy(device_b, operation.values_right.data(), sizeof(FieldWord) * count, cudaMemcpyHostToDevice, "cudaMemcpy(product Sumcheck left)");
            timed_memcpy(device_c, operation.values_third.data(), sizeof(FieldWord) * count, cudaMemcpyHostToDevice, "cudaMemcpy(product Sumcheck right)");
        } else {
            timed_memcpy(device_a, operation.values.data(), sizeof(FieldWord) * count, cudaMemcpyHostToDevice, "cudaMemcpy(product Sumcheck left)");
            timed_memcpy(device_b, operation.values_right.data(), sizeof(FieldWord) * count, cudaMemcpyHostToDevice, "cudaMemcpy(product Sumcheck right)");
        }
        std::size_t current_count = count;
        for (const FieldWord challenge : operation.challenges) {
            if (cubic) {
                note_kernel(g_product_kernel_class);
                product3_round_coefficients_kernel<<<1, 256>>>(device_a, device_b, device_c, current_count, device_coefficients);
            } else {
                note_kernel(g_product_kernel_class);
                product2_round_coefficients_kernel<<<1, 256>>>(device_a, device_b, current_count, device_coefficients);
            }
            check(cudaGetLastError(), "product Sumcheck coefficient kernel launch");
            timed_device_synchronize("product Sumcheck coefficient kernel synchronize");
            std::vector<FieldWord> coefficients(coefficient_count, word_zero());
            timed_memcpy(coefficients.data(), device_coefficients, sizeof(FieldWord) * coefficient_count, cudaMemcpyDeviceToHost, "cudaMemcpy(product Sumcheck coefficients)");
            note_kernel(g_product_kernel_class);
            multilinear_fold_kernel<<<grid_for(current_count >> 1), 256>>>(device_a, current_count, challenge, device_a_next);
            check(cudaGetLastError(), "product Sumcheck fold a launch");
            note_kernel(g_product_kernel_class);
            multilinear_fold_kernel<<<grid_for(current_count >> 1), 256>>>(device_b, current_count, challenge, device_b_next);
            check(cudaGetLastError(), "product Sumcheck fold b launch");
            if (cubic) {
                note_kernel(g_product_kernel_class);
                multilinear_fold_kernel<<<grid_for(current_count >> 1), 256>>>(device_c, current_count, challenge, device_c_next);
                check(cudaGetLastError(), "product Sumcheck fold c launch");
            }
            timed_device_synchronize("product Sumcheck fold synchronize");
            std::swap(device_a, device_a_next);
            std::swap(device_b, device_b_next);
            if (cubic) std::swap(device_c, device_c_next);
            current_count >>= 1;
            note_kernel(g_product_kernel_class);
            product_terminal_claim_kernel<<<1, 1>>>(device_a, device_b, device_c, cubic, device_claim);
            check(cudaGetLastError(), "product Sumcheck terminal claim launch");
            timed_device_synchronize("product Sumcheck terminal claim synchronize");
            RoundResult round{};
            round.challenge = challenge;
            round.coefficients = std::move(coefficients);
            timed_memcpy(&round.folded_claim, device_claim, sizeof(FieldWord), cudaMemcpyDeviceToHost, "cudaMemcpy(product Sumcheck folded claim)");
            result.rounds.push_back(std::move(round));
        }
        note_kernel(g_product_kernel_class);
        product_terminal_claim_kernel<<<1, 1>>>(device_a, device_b, device_c, cubic, device_claim);
        check(cudaGetLastError(), "product Sumcheck final claim launch");
        timed_device_synchronize("product Sumcheck final claim synchronize");
        timed_memcpy(&result.terminal, device_claim, sizeof(FieldWord), cudaMemcpyDeviceToHost, "cudaMemcpy(product Sumcheck terminal)");
        check(cudaFree(device_a), "cudaFree(product Sumcheck a)");
        check(cudaFree(device_a_next), "cudaFree(product Sumcheck a next)");
        check(cudaFree(device_b), "cudaFree(product Sumcheck b)");
        check(cudaFree(device_b_next), "cudaFree(product Sumcheck b next)");
        if (cubic) {
            check(cudaFree(device_c), "cudaFree(product Sumcheck c)");
            check(cudaFree(device_c_next), "cudaFree(product Sumcheck c next)");
        }
        check(cudaFree(device_coefficients), "cudaFree(product Sumcheck coefficients)");
        check(cudaFree(device_claim), "cudaFree(product Sumcheck terminal claim)");
        return result;
    } catch (...) {
        if (device_a != nullptr) cudaFree(device_a);
        if (device_a_next != nullptr) cudaFree(device_a_next);
        if (device_b != nullptr) cudaFree(device_b);
        if (device_b_next != nullptr) cudaFree(device_b_next);
        if (device_c != nullptr) cudaFree(device_c);
        if (device_c_next != nullptr) cudaFree(device_c_next);
        if (device_coefficients != nullptr) cudaFree(device_coefficients);
        if (device_claim != nullptr) cudaFree(device_claim);
        throw;
    }
}

static OperationResult run_hyrax_gpu(const Operation& operation)
{
    OperationResult result;
    result.kind = OperationKind::Hyrax;
    result.input_count = operation.values.size();
    result.padded_count = operation.values.size();
    result.operand_count = 1;
    const std::size_t expected_rounds = log2_power_of_two(operation.values.size());
    if (operation.challenges.size() > expected_rounds
        || (!operation.allow_partial && operation.challenges.size() != expected_rounds)) {
        throw std::runtime_error("Hyrax fold challenge count does not match log2(input count)");
    }
    result.complete = operation.challenges.size() == expected_rounds;
    FieldWord* device_current = nullptr;
    FieldWord* device_next = nullptr;
    try {
        const std::size_t count = operation.values.size();
        check(cudaMalloc(&device_current, sizeof(FieldWord) * count), "cudaMalloc(Hyrax current)");
        check(cudaMalloc(&device_next, sizeof(FieldWord) * count), "cudaMalloc(Hyrax next)");
        timed_memcpy(device_current, operation.values.data(), sizeof(FieldWord) * count, cudaMemcpyHostToDevice, "cudaMemcpy(Hyrax values)");
        std::size_t current_count = count;
        for (const FieldWord challenge : operation.challenges) {
            note_kernel(KernelClass::Hyrax);
            multilinear_fold_kernel<<<grid_for(current_count >> 1), 256>>>(
                device_current, current_count, challenge, device_next);
            check(cudaGetLastError(), "Hyrax multilinear_fold_kernel launch");
            timed_device_synchronize("Hyrax multilinear_fold_kernel synchronize");
            std::swap(device_current, device_next);
            current_count >>= 1;
            RoundResult round{};
            round.challenge = challenge;
            timed_memcpy(&round.folded_claim, device_current, sizeof(FieldWord), cudaMemcpyDeviceToHost, "cudaMemcpy(Hyrax folded claim)");
            result.rounds.push_back(round);
        }
        timed_memcpy(&result.terminal, device_current, sizeof(FieldWord), cudaMemcpyDeviceToHost, "cudaMemcpy(Hyrax terminal)");
        check(cudaFree(device_current), "cudaFree(Hyrax current)");
        check(cudaFree(device_next), "cudaFree(Hyrax next)");
        return result;
    } catch (...) {
        if (device_current != nullptr) cudaFree(device_current);
        if (device_next != nullptr) cudaFree(device_next);
        throw;
    }
}

static FieldWord run_transcript_gpu(const std::vector<FieldWord>& stream)
{
    FieldWord* device_stream = nullptr;
    FieldWord* device_output = nullptr;
    try {
        check(cudaMalloc(&device_stream, sizeof(FieldWord) * stream.size()), "cudaMalloc(transcript stream)");
        check(cudaMalloc(&device_output, sizeof(FieldWord)), "cudaMalloc(transcript output)");
        timed_memcpy(device_stream, stream.data(), sizeof(FieldWord) * stream.size(), cudaMemcpyHostToDevice, "cudaMemcpy(transcript stream)");
        const FieldWord base = {{0x9e3779b97f4a7c15ULL, 0ULL, 0ULL, 0ULL}};
        note_kernel(KernelClass::Transcript);
        transcript_field_accumulator_kernel<<<1, 1>>>(device_stream, stream.size(), base, device_output);
        check(cudaGetLastError(), "transcript_field_accumulator_kernel launch");
        timed_device_synchronize("transcript_field_accumulator_kernel synchronize");
        FieldWord output = word_zero();
        timed_memcpy(&output, device_output, sizeof(FieldWord), cudaMemcpyDeviceToHost, "cudaMemcpy(transcript output)");
        check(cudaFree(device_stream), "cudaFree(transcript stream)");
        check(cudaFree(device_output), "cudaFree(transcript output)");
        return output;
    } catch (...) {
        if (device_stream != nullptr) cudaFree(device_stream);
        if (device_output != nullptr) cudaFree(device_output);
        throw;
    }
}

static FieldWord parse_word_token(std::istream& input, const char* what)
{
    FieldWord word = word_zero();
    for (int limb = 0; limb < 4; ++limb) {
        std::string token;
        if (!(input >> token) || token.empty() || token[0] == '-') {
            throw std::runtime_error(std::string("missing ") + what + " limb");
        }
        std::size_t consumed = 0;
        unsigned long long value = 0ULL;
        try {
            value = std::stoull(token, &consumed, 10);
        } catch (const std::exception&) {
            throw std::runtime_error(std::string("invalid ") + what + " limb");
        }
        if (consumed != token.size()) {
            throw std::runtime_error(std::string("invalid ") + what + " limb");
        }
        word.limb[limb] = value;
    }
    if (word_ge(word, fr_modulus())) {
        throw std::runtime_error(std::string("non-canonical ") + what + " field element");
    }
    return word;
}

static std::size_t parse_count_token(std::istream& input, const char* what)
{
    std::string token;
    if (!(input >> token) || token.empty() || token[0] == '-') {
        throw std::runtime_error(std::string("missing ") + what);
    }
    std::size_t consumed = 0;
    unsigned long long value = 0ULL;
    try {
        value = std::stoull(token, &consumed, 10);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("invalid ") + what);
    }
    if (consumed != token.size() || value == 0ULL || value > (1ULL << 30)) {
        throw std::runtime_error(std::string("invalid ") + what);
    }
    return static_cast<std::size_t>(value);
}

static std::size_t parse_round_count_token(std::istream& input)
{
    std::string token;
    if (!(input >> token) || token.empty() || token[0] == '-') {
        throw std::runtime_error("missing round count");
    }
    std::size_t consumed = 0;
    unsigned long long value = 0ULL;
    try {
        value = std::stoull(token, &consumed, 10);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid round count");
    }
    if (consumed != token.size() || value > (1ULL << 30)) {
        throw std::runtime_error("invalid round count");
    }
    return static_cast<std::size_t>(value);
}

static OperationKind parse_kind(const std::string& token)
{
    if (token == "LASSO") return OperationKind::Lasso;
    if (token == "SUMCHECK") return OperationKind::Sumcheck;
    if (token == "SUMCHECK_PRODUCT2") return OperationKind::SumcheckProduct2;
    if (token == "SUMCHECK_PRODUCT3") return OperationKind::SumcheckProduct3;
    if (token == "HYRAX") return OperationKind::Hyrax;
    throw std::runtime_error("unknown GPU operation: " + token);
}

static std::vector<Operation> read_operations(const std::string& path)
{
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open input file: " + path);
    }
    std::string magic;
    if (!(input >> magic) || magic != "ZKGPT_GPU_BATCH_V3") {
        throw std::runtime_error("GPU input must use canonical ZKGPT_GPU_BATCH_V3 wire");
    }
    std::vector<Operation> operations;
    bool ended = false;
    std::string token;
    while (input >> token) {
        if (token == "END") {
            ended = true;
            break;
        }
        Operation operation{};
        const std::string partial_suffix = "_PREFIX";
        if (token.size() > partial_suffix.size()
            && token.compare(token.size() - partial_suffix.size(), partial_suffix.size(), partial_suffix) == 0) {
            operation.allow_partial = true;
            token.erase(token.size() - partial_suffix.size());
        }
        operation.kind = parse_kind(token);
        const std::size_t count = parse_count_token(input, "operation count");
        std::size_t rounds = 0;
        if (operation.kind == OperationKind::Lasso) {
            operation.beta = parse_word_token(input, "Lasso beta");
        } else {
            rounds = parse_round_count_token(input);
        }
        if (operation.kind != OperationKind::Lasso) {
            const std::size_t expected = log2_power_of_two(count);
            if (rounds > expected || (!operation.allow_partial && rounds != expected)) {
                throw std::runtime_error("round count does not match log2(operation count)");
            }
            operation.challenges.reserve(rounds);
            for (std::size_t index = 0; index < rounds; ++index) {
                operation.challenges.push_back(parse_word_token(input, "round challenge"));
            }
        }
        operation.values.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            operation.values.push_back(parse_word_token(input, "operation value"));
        }
        if (operation.kind == OperationKind::SumcheckProduct2
            || operation.kind == OperationKind::SumcheckProduct3) {
            operation.values_right.reserve(count);
            for (std::size_t index = 0; index < count; ++index) {
                operation.values_right.push_back(parse_word_token(input, "product Sumcheck right operand"));
            }
            if (operation.kind == OperationKind::SumcheckProduct3) {
                operation.values_third.reserve(count);
                for (std::size_t index = 0; index < count; ++index) {
                    operation.values_third.push_back(parse_word_token(input, "product Sumcheck third operand"));
                }
            }
        }
        operations.push_back(std::move(operation));
    }
    if (!ended || operations.empty()) {
        throw std::runtime_error("GPU input has no complete operation batch");
    }
    if (input >> token) {
        throw std::runtime_error("trailing tokens after GPU operation batch");
    }
    return operations;
}

static std::string hex256(const FieldWord& value)
{
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (int limb = 3; limb >= 0; --limb) {
        output << std::setw(16) << value.limb[limb];
    }
    return output.str();
}

static std::string json_escape(const std::string& value)
{
    std::ostringstream output;
    for (const char character : value) {
        switch (character) {
        case '\\': output << "\\\\"; break;
        case '"': output << "\\\""; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default: output << character; break;
        }
    }
    return output.str();
}

static std::string device_uuid_hex(const cudaDeviceProp& device)
{
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned char byte : device.uuid.bytes) {
        output << std::setw(2) << static_cast<unsigned int>(byte);
    }
    return output.str();
}

static void write_round_json(std::ostream& output, const RoundResult& round, bool sumcheck)
{
    output << "{\"challenge_fr_hex\":\"" << hex256(round.challenge) << "\"";
    if (sumcheck) {
        output << ",\"coefficients_fr_hex\":[";
        for (std::size_t index = 0; index < round.coefficients.size(); ++index) {
            if (index != 0) output << ",";
            output << "\"" << hex256(round.coefficients[index]) << "\"";
        }
        output << "]";
    }
    output << ",\"folded_claim_fr_hex\":\"" << hex256(round.folded_claim) << "\"}";
}

int main(int argc, char** argv)
{
    cudaEvent_t event_start = nullptr;
    cudaEvent_t event_stop = nullptr;
    try {
        if (argc != 4) {
            std::cerr << "usage: zkgpt_cuda_native_prover <gpu-batch-input> <output-json> <label>\n";
            return 2;
        }
        g_kernel_counters = KernelCounters{};
        g_timing_counters = TimingCounters{};
        classify_product_kernel(argv[3]);
        int device_count = 0;
        check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_count <= 0) {
            throw std::runtime_error("no CUDA devices available");
        }
        cudaDeviceProp device{};
        check(cudaGetDeviceProperties(&device, 0), "cudaGetDeviceProperties");
        check(cudaSetDevice(0), "cudaSetDevice");
        check(cudaEventCreate(&event_start), "cudaEventCreate(start)");
        check(cudaEventCreate(&event_stop), "cudaEventCreate(stop)");
        check(cudaEventRecord(event_start, 0), "cudaEventRecord(start)");

        const std::vector<Operation> operations = read_operations(argv[1]);
        std::vector<OperationResult> results;
        std::vector<FieldWord> transcript_stream;
        std::size_t input_field_count = 0;
        std::size_t lasso_count = 0;
        std::size_t sumcheck_round_count = 0;
        FieldWord lasso_aggregate = word_one();
        FieldWord sumcheck_aggregate = word_zero();
        FieldWord gkr_aggregate = word_zero();
        for (const Operation& operation : operations) {
            OperationResult result{};
            result.kind = operation.kind;
            input_field_count += operation.values.size() + operation.values_right.size() + operation.values_third.size();
            transcript_stream.insert(transcript_stream.end(), operation.values.begin(), operation.values.end());
            transcript_stream.insert(transcript_stream.end(), operation.values_right.begin(), operation.values_right.end());
            transcript_stream.insert(transcript_stream.end(), operation.values_third.begin(), operation.values_third.end());
            transcript_stream.insert(transcript_stream.end(), operation.challenges.begin(), operation.challenges.end());
            if (operation.kind == OperationKind::Lasso) {
                transcript_stream.push_back(operation.beta);
                run_lasso_gpu(operation, &result);
                lasso_aggregate = fr_mul(lasso_aggregate, result.terminal);
                lasso_count += operation.values.size();
            } else if (operation.kind == OperationKind::Sumcheck) {
                result = run_sumcheck_gpu(operation);
                sumcheck_aggregate = fr_add(sumcheck_aggregate, result.terminal);
                gkr_aggregate = fr_add(gkr_aggregate, result.terminal);
                sumcheck_round_count += result.rounds.size();
                for (const RoundResult& round : result.rounds) {
                    transcript_stream.insert(transcript_stream.end(), round.coefficients.begin(), round.coefficients.end());
                    transcript_stream.push_back(round.folded_claim);
                }
            } else if (operation.kind == OperationKind::SumcheckProduct2
                       || operation.kind == OperationKind::SumcheckProduct3) {
                result = run_product_sumcheck_gpu(operation, operation.kind == OperationKind::SumcheckProduct3);
                sumcheck_aggregate = fr_add(sumcheck_aggregate, result.terminal);
                gkr_aggregate = fr_add(gkr_aggregate, result.terminal);
                sumcheck_round_count += result.rounds.size();
                for (const RoundResult& round : result.rounds) {
                    transcript_stream.insert(transcript_stream.end(), round.coefficients.begin(), round.coefficients.end());
                    transcript_stream.push_back(round.folded_claim);
                }
            } else {
                result = run_hyrax_gpu(operation);
                for (const RoundResult& round : result.rounds) {
                    transcript_stream.push_back(round.folded_claim);
                }
            }
            transcript_stream.push_back(result.terminal);
            results.push_back(std::move(result));
        }
        if (input_field_count == 0 || transcript_stream.empty()) {
            throw std::runtime_error("empty GPU operation batch");
        }
        const FieldWord transcript = run_transcript_gpu(transcript_stream);
        check(cudaEventRecord(event_stop, 0), "cudaEventRecord(stop)");
        timed_event_synchronize(event_stop, "cudaEventSynchronize(stop)");
        float cuda_event_elapsed_ms = 0.0f;
        check(cudaEventElapsedTime(&cuda_event_elapsed_ms, event_start, event_stop), "cudaEventElapsedTime");

        std::ofstream output(argv[2]);
        if (!output) {
            throw std::runtime_error("failed to open output file");
        }
        output << "{\n"
               << "  \"protocol\": \"zkgpt/native-bn254-fr-batch-v1\",\n"
               << "  \"schema\": \"zkGPT_gpu_proof_messages_v3\",\n"
               << "  \"backend\": \"gpu\",\n"
               << "  \"cuda_kernel\": \"zkgpt_native_bn254_fr_batch_v3\",\n"
               << "  \"cuda_kernel_version\": \"zkgpt_native_bn254_fr_batch_v3\",\n"
               << "  \"cuda_kernel_executed\": true,\n"
               << "  \"gpu_runtime_present\": true,\n"
               << "  \"cpu_fallback\": false,\n"
               << "  \"device_count\": " << device_count << ",\n"
               << "  \"device_name\": \"" << json_escape(device.name) << "\",\n"
               << "  \"gpu_uuid\": \"" << device_uuid_hex(device) << "\",\n"
               << "  \"gpu_compute_capability\": \"" << device.major << "." << device.minor << "\",\n"
               << "  \"gpu_arch\": \"sm_" << device.major << device.minor << "\",\n"
               << "  \"cuda_event_timing\": \"cudaEventElapsedTime\",\n"
               << "  \"cuda_event_elapsed_ms\": " << cuda_event_elapsed_ms << ",\n"
               << "  \"transfer_timing\": \"host_steady_clock\",\n"
               << "  \"memory_copy_h2d_ms\": " << g_timing_counters.memory_copy_h2d_ms << ",\n"
               << "  \"memory_copy_d2h_ms\": " << g_timing_counters.memory_copy_d2h_ms << ",\n"
               << "  \"synchronization_time_ms\": " << g_timing_counters.synchronization_time_ms << ",\n"
               << "  \"label\": \"" << json_escape(argv[3]) << "\",\n"
               << "  \"wire_version\": \"ZKGPT_GPU_BATCH_V3\",\n"
               << "  \"field_words_processed\": " << input_field_count << ",\n"
               << "  \"kernel_invocation_count\": " << g_kernel_counters.total << ",\n"
               << "  \"kernel_invocation_counts\": {\"lasso\": " << g_kernel_counters.lasso
               << ", \"matmul_sumcheck\": " << g_kernel_counters.matmul_sumcheck
               << ", \"gkr_relation\": " << g_kernel_counters.gkr_relation
               << ", \"hyrax\": " << g_kernel_counters.hyrax
               << ", \"transcript\": " << g_kernel_counters.transcript << "},\n"
               << "  \"primitive_counts\": {\"lasso\": " << lasso_count
               << ", \"matmul_sumcheck\": " << sumcheck_round_count
               << ", \"gkr_relation\": " << sumcheck_round_count
               << ", \"transcript\": " << transcript_stream.size() << "},\n"
               << "  \"lasso_grand_product_fr_hex\": \"" << hex256(lasso_aggregate) << "\",\n"
               << "  \"matmul_sumcheck_fr_hex\": \"" << hex256(sumcheck_aggregate) << "\",\n"
               << "  \"gkr_relation_fr_hex\": \"" << hex256(gkr_aggregate) << "\",\n"
               << "  \"transcript_state_low256_hex\": \"" << hex256(transcript) << "\",\n"
               << "  \"transcript_accumulator_fr_hex\": \"" << hex256(transcript) << "\",\n"
               << "  \"operations\": [";
        for (std::size_t index = 0; index < results.size(); ++index) {
            if (index != 0) output << ",";
            const OperationResult& result = results[index];
            const char* kind = result.kind == OperationKind::Lasso
                ? "lasso_product_tree"
                : (result.kind == OperationKind::Sumcheck
                    ? "sumcheck_mle_fold"
                    : (result.kind == OperationKind::SumcheckProduct2
                        ? "sumcheck_product2_fold"
                        : (result.kind == OperationKind::SumcheckProduct3
                            ? "sumcheck_product3_fold"
                            : "hyrax_linear_eval_fold")));
            output << "{\"kind\":\"" << kind << "\",\"input_count\":" << result.input_count
                   << ",\"padded_count\":" << result.padded_count
                   << ",\"operand_count\":" << result.operand_count
                   << ",\"complete\":" << (result.complete ? "true" : "false");
            if (result.kind == OperationKind::Lasso) {
                output << ",\"grand_product_fr_hex\":\"" << hex256(result.terminal)
                       << "\",\"terminal_value_fr_hex\":\"" << hex256(result.terminal) << "\"";
            } else {
                output << ",\"rounds\":[";
                for (std::size_t round_index = 0; round_index < result.rounds.size(); ++round_index) {
                    if (round_index != 0) output << ",";
                    write_round_json(
                        output,
                        result.rounds[round_index],
                        result.kind == OperationKind::Sumcheck
                            || result.kind == OperationKind::SumcheckProduct2
                            || result.kind == OperationKind::SumcheckProduct3);
                }
                output << "],\"terminal_value_fr_hex\":\"" << hex256(result.terminal) << "\"";
            }
            output << "}";
        }
        output << "]\n}\n";
        check(cudaEventDestroy(event_start), "cudaEventDestroy(start)");
        event_start = nullptr;
        check(cudaEventDestroy(event_stop), "cudaEventDestroy(stop)");
        event_stop = nullptr;
        std::cout << "cuda_native_bn254_fr_batch_ok device=\"" << device.name
                  << "\" operations=" << results.size()
                  << " field_elements=" << input_field_count << "\n";
        return 0;
    } catch (const std::exception& exception) {
        if (event_start != nullptr) cudaEventDestroy(event_start);
        if (event_stop != nullptr) cudaEventDestroy(event_stop);
        std::cerr << "fatal: " << exception.what() << "\n";
        return 1;
    }
}
