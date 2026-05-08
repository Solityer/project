#ifndef FR_TENSOR_CUH
#define FR_TENSOR_CUH

#include <iostream>
#include <iomanip>
#include <utility>      // std::pair, std::make_pair
#include <vector>
#include <curand_kernel.h>
#include <random>
#include <cstring>
#include <cstdint>
#include "bls12-381.cuh"
using namespace std;

typedef blstrs__scalar__Scalar Fr_t;
typedef blstrs__g1__G1Affine_affine G1Affine_t;
typedef blstrs__g1__G1Affine_jacobian G1Jacobian_t;

const uint FrNumThread = 4;
const uint FrSharedMemorySize = 2 * sizeof(Fr_t) * FrNumThread; 

inline Fr_t Fr_host_zero()
{
    return {{0, 0, 0, 0, 0, 0, 0, 0}};
}

inline Fr_t Fr_host_one_mont()
{
    return {{4294967294U, 1U, 215042U, 1485092858U, 3971764213U, 2576109551U, 2898593135U, 405057881U}};
}

inline Fr_t Fr_host_one_raw()
{
    return {{1U, 0, 0, 0, 0, 0, 0, 0}};
}

inline Fr_t Fr_host_modulus()
{
    return {{1U, 4294967295U, 4294859774U, 1404937218U, 161601541U, 859428872U, 698187080U, 1944954707U}};
}

inline Fr_t Fr_host_r2()
{
    return {{4092763245U, 3382307216U, 2274516003U, 728559051U, 1918122383U, 97719446U, 2673475345U, 122214873U}};
}

inline bool Fr_host_eq(const Fr_t& a, const Fr_t& b)
{
    return std::memcmp(&a, &b, sizeof(Fr_t)) == 0;
}

inline bool Fr_host_gte(const Fr_t& a, const Fr_t& b)
{
    for (int i = 7; i >= 0; --i)
    {
        if (a.val[i] > b.val[i]) return true;
        if (a.val[i] < b.val[i]) return false;
    }
    return true;
}

inline Fr_t Fr_host_add_raw(Fr_t a, const Fr_t& b)
{
    std::uint64_t carry = 0;
    for (uint i = 0; i < 8; ++i)
    {
        const std::uint64_t sum = static_cast<std::uint64_t>(a.val[i]) + b.val[i] + carry;
        a.val[i] = static_cast<uint>(sum & 0xffffffffULL);
        carry = sum >> 32U;
    }
    return a;
}

inline Fr_t Fr_host_sub_raw(Fr_t a, const Fr_t& b)
{
    std::uint64_t borrow = 0;
    for (uint i = 0; i < 8; ++i)
    {
        const std::uint64_t lhs = a.val[i];
        const std::uint64_t rhs = static_cast<std::uint64_t>(b.val[i]) + borrow;
        if (lhs >= rhs)
        {
            a.val[i] = static_cast<uint>(lhs - rhs);
            borrow = 0;
        }
        else
        {
            a.val[i] = static_cast<uint>((1ULL << 32U) + lhs - rhs);
            borrow = 1;
        }
    }
    return a;
}

inline Fr_t Fr_host_add(const Fr_t& a, const Fr_t& b)
{
    Fr_t out = Fr_host_add_raw(a, b);
    const Fr_t p = Fr_host_modulus();
    if (Fr_host_gte(out, p)) out = Fr_host_sub_raw(out, p);
    return out;
}

inline Fr_t Fr_host_sub(const Fr_t& a, const Fr_t& b)
{
    Fr_t out = Fr_host_sub_raw(a, b);
    if (!Fr_host_gte(a, b)) out = Fr_host_add_raw(out, Fr_host_modulus());
    return out;
}

inline uint Fr_host_add_with_carry(uint a, uint* carry)
{
    const std::uint64_t sum = static_cast<std::uint64_t>(a) + *carry;
    *carry = static_cast<uint>(sum >> 32U);
    return static_cast<uint>(sum & 0xffffffffULL);
}

inline uint Fr_host_mac_with_carry(uint a, uint b, uint c, uint* carry)
{
    const std::uint64_t result = static_cast<std::uint64_t>(a) * b + c + *carry;
    *carry = static_cast<uint>(result >> 32U);
    return static_cast<uint>(result & 0xffffffffULL);
}

inline Fr_t Fr_host_mul(const Fr_t& a, const Fr_t& b)
{
    const Fr_t p = Fr_host_modulus();
    uint t[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (uint i = 0; i < 8; ++i)
    {
        uint carry = 0;
        for (uint j = 0; j < 8; ++j)
        {
            t[j] = Fr_host_mac_with_carry(a.val[j], b.val[i], t[j], &carry);
        }
        t[8] = Fr_host_add_with_carry(t[8], &carry);
        t[9] = carry;

        carry = 0;
        const uint m = 4294967295U * t[0];
        Fr_host_mac_with_carry(m, p.val[0], t[0], &carry);
        for (uint j = 1; j < 8; ++j)
        {
            t[j - 1] = Fr_host_mac_with_carry(m, p.val[j], t[j], &carry);
        }

        t[7] = Fr_host_add_with_carry(t[8], &carry);
        t[8] = t[9] + carry;
    }

    Fr_t out = Fr_host_zero();
    for (uint i = 0; i < 8; ++i) out.val[i] = t[i];
    if (Fr_host_gte(out, p)) out = Fr_host_sub_raw(out, p);
    return out;
}

inline Fr_t Fr_host_mont(const Fr_t& a)
{
    return Fr_host_mul(a, Fr_host_r2());
}

inline Fr_t Fr_host_unmont(const Fr_t& a)
{
    return Fr_host_mul(a, Fr_host_one_raw());
}

inline Fr_t Fr_host_mle_step(const Fr_t& low, const Fr_t& high, const Fr_t& challenge)
{
    return Fr_host_add(low, Fr_host_mul(challenge, Fr_host_sub(high, low)));
}

inline uint Fr_partial_me_out_size(uint size, uint window_size)
{
    if (window_size == 0) throw std::runtime_error("Invalid window size");
    const uint windows = (size + window_size - 1U) / window_size;
    return ((windows + 1U) / 2U) * window_size;
}

ostream& operator<<(ostream& os, const Fr_t& x)
{
  os << "0x" << std::hex;
  for (uint i = 8; i > 0; -- i)
  {
    os << std::setfill('0') << std::setw(8) << x.val[i - 1];
  }
  return os << std::dec << std::setw(0) << std::setfill(' ');
}



// define the kernels

// Elementwise addition
KERNEL void Fr_elementwise_add(GLOBAL Fr_t* arr1, GLOBAL Fr_t* arr2, GLOBAL Fr_t* arr_out, uint n) 
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__scalar__Scalar_add(arr1[gid], arr2[gid]);
}

// Broadcast addition
KERNEL void Fr_broadcast_add(GLOBAL Fr_t* arr, Fr_t x, GLOBAL Fr_t* arr_out, uint n)
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__scalar__Scalar_add(arr[gid], x);
}

// Elementwise negation
KERNEL void Fr_elementwise_neg(GLOBAL Fr_t* arr, GLOBAL Fr_t* arr_out, uint n) 
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__scalar__Scalar_sub(blstrs__scalar__Scalar_ZERO, arr[gid]);
}

// Elementwise subtraction
KERNEL void Fr_elementwise_sub(GLOBAL Fr_t* arr1, GLOBAL Fr_t* arr2, GLOBAL Fr_t* arr_out, uint n) {
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__scalar__Scalar_sub(arr1[gid], arr2[gid]);
}

// Broadcast subtraction
KERNEL void Fr_broadcast_sub(GLOBAL Fr_t* arr, Fr_t x, GLOBAL Fr_t* arr_out, uint n)
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__scalar__Scalar_sub(arr[gid], x);
}

// To montegomery form
KERNEL void Fr_elementwise_mont(GLOBAL Fr_t* arr, GLOBAL Fr_t* arr_out, uint n) 
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__scalar__Scalar_mont(arr[gid]);
}

// From montgomery form
KERNEL void Fr_elementwise_unmont(GLOBAL Fr_t* arr, GLOBAL Fr_t* arr_out, uint n) 
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__scalar__Scalar_unmont(arr[gid]);
}

// Elementwise montegomery multiplication
KERNEL void Fr_elementwise_mont_mul(GLOBAL Fr_t* arr1, GLOBAL Fr_t* arr2, GLOBAL Fr_t* arr_out, uint n) {
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__scalar__Scalar_mul(arr1[gid], arr2[gid]);
}

// Broadcast montegomery multiplication
KERNEL void Fr_broadcast_mont_mul(GLOBAL Fr_t* arr, Fr_t x, GLOBAL Fr_t* arr_out, uint n)
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = blstrs__scalar__Scalar_mul(arr[gid], x);
}


class G1TensorAffine;
class G1TensorJacobian;
class Commitment;
class zkFC;
class zkReLU;
class zkMatMul;

// define the class FrTensor

class FrTensor
{   
    private:
    Fr_t* gpu_data;

    public:
    const uint size;
    FrTensor(uint size): size(size), gpu_data(nullptr)
    {
        cudaMalloc((void **)&gpu_data, sizeof(Fr_t) * size);
    }

    FrTensor(uint size, const Fr_t* cpu_data): size(size), gpu_data(nullptr)
    {
        cudaMalloc((void **)&gpu_data, sizeof(Fr_t) * size);
        cudaMemcpy(gpu_data, cpu_data, sizeof(Fr_t) * size, cudaMemcpyHostToDevice);
    }

    FrTensor(const FrTensor& t): size(t.size), gpu_data(nullptr)
    {
        cudaMalloc((void **)&gpu_data, sizeof(Fr_t) * size);
        cudaMemcpy(gpu_data, t.gpu_data, sizeof(Fr_t) * size, cudaMemcpyDeviceToDevice);
    }

    ~FrTensor()
    {
        cudaFree(gpu_data);
        gpu_data = nullptr;
    }

    Fr_t operator()(uint idx) const
    {
        Fr_t out;
        cudaMemcpy(&out, gpu_data + idx, sizeof(Fr_t), cudaMemcpyDeviceToHost);
        return out;
    }

    FrTensor operator+(const FrTensor& t) const
    {
        if (size != t.size) throw std::runtime_error("Incompatible dimensions");
        FrTensor out(size);
        Fr_elementwise_add<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, t.gpu_data, out.gpu_data, size);
        cudaDeviceSynchronize();
        return out;
    }

    FrTensor operator+(const Fr_t& x) const
    {
        FrTensor out(size);
        Fr_broadcast_add<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, x, out.gpu_data, size);
        cudaDeviceSynchronize();
        return out;
    }

    FrTensor& operator+=(const FrTensor& t)
    {
        if (size != t.size) throw std::runtime_error("Incompatible dimensions");
        Fr_elementwise_add<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, t.gpu_data, gpu_data, size);
        cudaDeviceSynchronize();
        return *this;
    }
    
    FrTensor& operator+=(const Fr_t& x)
    {
        Fr_broadcast_add<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, x, gpu_data, size);
        cudaDeviceSynchronize();
        return *this;
    }

    FrTensor operator-() const
    {
        FrTensor out(size);
        Fr_elementwise_neg<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, out.gpu_data, size);
        cudaDeviceSynchronize();
        return out;
    }

    FrTensor operator-(const FrTensor& t) const
    {
        if (size != t.size) throw std::runtime_error("Incompatible dimensions");
        FrTensor out(size);
        Fr_elementwise_sub<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, t.gpu_data, out.gpu_data, size);
        cudaDeviceSynchronize();
        return out;
    }

    FrTensor operator-(const Fr_t& x) const
    {
        FrTensor out(size);
        Fr_broadcast_sub<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, x, out.gpu_data, size);
        cudaDeviceSynchronize();
        return out;
    }

    FrTensor& operator-=(const FrTensor& t)
    {
        if (size != t.size) throw std::runtime_error("Incompatible dimensions");
        Fr_elementwise_sub<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, t.gpu_data, gpu_data, size);
        cudaDeviceSynchronize();
        return *this;
    }
    
    FrTensor& operator-=(const Fr_t& x)
    {
        Fr_broadcast_sub<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, x, gpu_data, size);
        cudaDeviceSynchronize();
        return *this;
    }

    FrTensor& mont()
    {
        Fr_elementwise_mont<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, gpu_data, size);
        cudaDeviceSynchronize();
        return *this;
    } 

    FrTensor& unmont()
    {
        Fr_elementwise_unmont<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, gpu_data, size);
        cudaDeviceSynchronize();
        return *this;
    } 

    FrTensor operator*(const FrTensor& t) const
    {
        if (size != t.size) throw std::runtime_error("Incompatible dimensions");
        FrTensor out(size);
        Fr_elementwise_mont_mul<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, t.gpu_data, out.gpu_data, size);
        cudaDeviceSynchronize();
        return out;
    }

    FrTensor operator*(const Fr_t& x) const
    {
        FrTensor out(size);
        Fr_broadcast_mont_mul<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, x, out.gpu_data, size);
        cudaDeviceSynchronize();
        return out;
    }

    FrTensor& operator*=(const FrTensor& t)
    {
        if (size != t.size) throw std::runtime_error("Incompatible dimensions");
        Fr_elementwise_mont_mul<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, t.gpu_data, gpu_data, size);
        cudaDeviceSynchronize();
        return *this;
    }
    
    FrTensor& operator*=(const Fr_t& x)
    {
        Fr_broadcast_mont_mul<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, x, gpu_data, size);
        cudaDeviceSynchronize();
        return *this;
    }

    Fr_t sum() const;

    Fr_t operator()(const vector<Fr_t>& u) const;

    std::pair<FrTensor, FrTensor> split(uint window_size) const;

    FrTensor partial_me(vector<Fr_t> u, uint window_size) const;

    std::vector<Fr_t> to_host() const;

    static FrTensor random_int(uint size, uint num_bits);
    static FrTensor random(uint size);

    friend Fr_t Fr_me(const FrTensor& t, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end);

    friend FrTensor Fr_partial_me(const FrTensor& t, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, uint window_size);

    friend void Fr_ip_sc(const FrTensor& a, const FrTensor& b, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, vector<Fr_t>& proof);
    friend void Fr_hp_sc(const FrTensor& a, const FrTensor& b, vector<Fr_t>::const_iterator u_begin, vector<Fr_t>::const_iterator u_end, vector<Fr_t>::const_iterator v_begin, vector<Fr_t>::const_iterator v_end, vector<Fr_t>& proof);
    friend void Fr_bin_sc(const FrTensor& a, vector<Fr_t>::const_iterator u_begin, vector<Fr_t>::const_iterator u_end, vector<Fr_t>::const_iterator v_begin, vector<Fr_t>::const_iterator v_end, vector<Fr_t>& proof);

    friend class G1TensorAffine;
    friend class G1TensorJacobian;
    friend class Commitment;
    friend class zkFC;
    friend class zkReLU;
    friend class zkMatMul;
};

KERNEL void Fr_sum_reduction(GLOBAL Fr_t *arr, GLOBAL Fr_t *output, uint n) {
    extern __shared__ Fr_t frsum_sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (2 * blockDim.x) + threadIdx.x;

    // Load input into shared memory
    frsum_sdata[tid] = (i < n) ? arr[i] : blstrs__scalar__Scalar_ZERO;
    if (i + blockDim.x < n) frsum_sdata[tid] = blstrs__scalar__Scalar_add(frsum_sdata[tid], arr[i + blockDim.x]);

    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            frsum_sdata[tid] = blstrs__scalar__Scalar_add(frsum_sdata[tid], frsum_sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write the result for this block to output
    if (tid == 0) output[blockIdx.x] = frsum_sdata[0];
}

Fr_t FrTensor::sum() const
{
    Fr_t *ptr_input, *ptr_output;
    uint curSize = size;
    cudaMalloc((void**)&ptr_input, size * sizeof(Fr_t));
    cudaMalloc((void**)&ptr_output, ((size + 1)/ 2) * sizeof(Fr_t));
    cudaMemcpy(ptr_input, gpu_data, size * sizeof(Fr_t), cudaMemcpyDeviceToDevice);

    while(curSize > 1) {
        uint gridSize = (curSize + FrNumThread - 1) / FrNumThread;
        Fr_sum_reduction<<<gridSize, FrNumThread, FrSharedMemorySize>>>(ptr_input, ptr_output, curSize);
        cudaDeviceSynchronize(); // Ensure kernel completion before proceeding
        
        // Swap pointers. Use the output from this step as the input for the next step.
        Fr_t *temp = ptr_input;
        ptr_input = ptr_output;
        ptr_output = temp;
        
        curSize = gridSize;  // The output size is equivalent to the grid size used in the kernel launch
    }

    Fr_t finalSum;
    cudaMemcpy(&finalSum, ptr_input, sizeof(Fr_t), cudaMemcpyDeviceToHost);

    cudaFree(ptr_input);
    cudaFree(ptr_output);

    return finalSum;
}


Fr_t FrTensor::operator()(const vector<Fr_t>& u) const
{
    uint log_dim = u.size();
    if (size <= ((1 << log_dim) / 2) || size > (1 << log_dim)) throw std::runtime_error("Incompatible dimensions");
    return Fr_me(*this, u.begin(), u.end());
}

KERNEL void random_int_kernel(Fr_t* gpu_data, uint num_bits, uint n, unsigned long seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    
    // Initialize the RNG state for this thread.
    curand_init(seed, tid, 0, &state);  
    
    if (tid < n) {
        gpu_data[tid] = {curand(&state) & ((1U << num_bits) - 1), 0, 0, 0, 0, 0, 0, 0};
        gpu_data[tid] = blstrs__scalar__Scalar_sub(gpu_data[tid], {1U << (num_bits - 1), 0, 0, 0, 0, 0, 0, 0});
    }
}

FrTensor FrTensor::random_int(uint size, uint num_bits)
{
    // Create a random device
    std::random_device rd;

    // Initialize a 64-bit Mersenne Twister random number generator
    // with a seed from the random device
    std::mt19937_64 rng(rd());

    // Define the range for your unsigned long numbers
    std::uniform_int_distribution<unsigned long> distribution(0, ULONG_MAX);

    // Generate a random unsigned long number
    unsigned long seed = distribution(rng);

    FrTensor out(size);
    random_int_kernel<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(out.gpu_data, num_bits, size, seed);
    cudaDeviceSynchronize();
    return out;
}

KERNEL void random_kernel(Fr_t* gpu_data, uint n, unsigned long seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;

    if (tid >= static_cast<int>(n)) return;
    
    // Initialize the RNG state for this thread.
    curand_init(seed, tid, 0, &state);  
    gpu_data[tid] = {curand(&state), curand(&state), curand(&state), curand(&state), curand(&state), curand(&state), curand(&state), curand(&state) % 1944954707};
}

FrTensor FrTensor::random(uint size)
{
    // Create a random device
    std::random_device rd;

    // Initialize a 64-bit Mersenne Twister random number generator
    // with a seed from the random device
    std::mt19937_64 rng(rd());

    // Define the range for your unsigned long numbers
    std::uniform_int_distribution<unsigned long> distribution(0, ULONG_MAX);

    // Generate a random unsigned long number
    unsigned long seed = distribution(rng);

    FrTensor out(size);
    random_kernel<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(out.gpu_data, size, seed);
    cudaDeviceSynchronize();
    return out;
}

FrTensor FrTensor::partial_me(vector<Fr_t> u, uint window_size) const
{
    if (window_size == 0) throw std::runtime_error("Invalid window size");
    if (u.empty()) return *this;
    const std::size_t max_size = static_cast<std::size_t>(window_size) << u.size();
    const std::size_t min_size = static_cast<std::size_t>(window_size) << (u.size() - 1);
    if (size <= min_size || static_cast<std::size_t>(size) > max_size) throw std::runtime_error("Incompatible dimensions");
    return Fr_partial_me(*this, u.begin(), u.end(), window_size);
}

KERNEL void Fr_split_by_window(GLOBAL Fr_t *arr_in, GLOBAL Fr_t *arr0, GLOBAL Fr_t *arr1, uint in_size, uint out_size, uint window_size)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= out_size) return;
    
    uint window_id = gid / window_size;
    uint idx_in_window = gid % window_size;
    uint gid0 = 2 * window_id * window_size + idx_in_window;
    uint gid1 = (2 * window_id + 1) * window_size + idx_in_window;
    arr0[gid] = (gid0 < in_size) ? arr_in[gid0] : blstrs__scalar__Scalar_ZERO;
    arr1[gid] = (gid1 < in_size) ? arr_in[gid1] : blstrs__scalar__Scalar_ZERO;
}

std::pair<FrTensor, FrTensor> FrTensor::split(uint window_size) const
{
    if (window_size < 1 || window_size >= size) throw std::runtime_error("Invalid window size.");
    uint out_size = Fr_partial_me_out_size(size, window_size);
    std::vector<Fr_t> host_in(size);
    cudaMemcpy(host_in.data(), gpu_data, sizeof(Fr_t) * size, cudaMemcpyDeviceToHost);
    std::vector<Fr_t> host0(out_size, Fr_host_zero());
    std::vector<Fr_t> host1(out_size, Fr_host_zero());
    for (uint gid = 0; gid < out_size; ++gid)
    {
        const uint window_id = gid / window_size;
        const uint idx_in_window = gid % window_size;
        const uint gid0 = 2 * window_id * window_size + idx_in_window;
        const uint gid1 = (2 * window_id + 1) * window_size + idx_in_window;
        if (gid0 < size) host0[gid] = host_in[gid0];
        if (gid1 < size) host1[gid] = host_in[gid1];
    }
    return {
        FrTensor(out_size, host0.data()),
        FrTensor(out_size, host1.data())
    };
}

std::vector<Fr_t> FrTensor::to_host() const
{
    std::vector<Fr_t> out(size);
    if (!out.empty())
    {
        cudaMemcpy(out.data(), gpu_data, sizeof(Fr_t) * size, cudaMemcpyDeviceToHost);
    }
    return out;
}

KERNEL void Fr_me_step(GLOBAL Fr_t *arr_in, GLOBAL Fr_t *arr_out, Fr_t x, uint in_size, uint out_size)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= out_size) return;
    
    uint gid0 = 2 * gid;
    uint gid1 = 2 * gid + 1;
    if (gid1 < in_size) arr_out[gid] = blstrs__scalar__Scalar_add(arr_in[gid0], blstrs__scalar__Scalar_mul(x, blstrs__scalar__Scalar_sub(arr_in[gid1], arr_in[gid0])));
    else if (gid0 < in_size) arr_out[gid] = blstrs__scalar__Scalar_sub(arr_in[gid0], blstrs__scalar__Scalar_mul(x, arr_in[gid0]));
    else arr_out[gid] = blstrs__scalar__Scalar_ZERO;
}

Fr_t Fr_me(const FrTensor& t, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end)
{
    std::vector<Fr_t> current(t.size);
    if (!current.empty())
    {
        cudaMemcpy(current.data(), t.gpu_data, sizeof(Fr_t) * current.size(), cudaMemcpyDeviceToHost);
    }
    for (auto it = begin; it != end; ++it)
    {
        const uint out_size = static_cast<uint>((current.size() + 1U) / 2U);
        std::vector<Fr_t> next(out_size, Fr_host_zero());
        for (uint gid = 0; gid < out_size; ++gid)
        {
            const uint gid0 = 2U * gid;
            const uint gid1 = gid0 + 1U;
            const Fr_t low = (gid0 < current.size()) ? current[gid0] : Fr_host_zero();
            const Fr_t high = (gid1 < current.size()) ? current[gid1] : Fr_host_zero();
            next[gid] = Fr_host_mle_step(low, high, *it);
        }
        current.swap(next);
    }
    return current.empty() ? Fr_host_zero() : current.front();
}

KERNEL void Fr_partial_me_step(GLOBAL Fr_t *arr_in, GLOBAL Fr_t *arr_out, Fr_t x, uint in_size, uint out_size, uint window_size)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= out_size) return;
    
    uint window_id = gid / window_size;
    uint idx_in_window = gid % window_size;
    uint gid0 = 2 * window_id * window_size + idx_in_window;
    uint gid1 = (2 * window_id + 1) * window_size + idx_in_window;
    if (gid1 < in_size) arr_out[gid] = blstrs__scalar__Scalar_add(arr_in[gid0], blstrs__scalar__Scalar_mul(x, blstrs__scalar__Scalar_sub(arr_in[gid1], arr_in[gid0])));
    else if (gid0 < in_size) arr_out[gid] = blstrs__scalar__Scalar_sub(arr_in[gid0], blstrs__scalar__Scalar_mul(x, arr_in[gid0]));
    else arr_out[gid] = blstrs__scalar__Scalar_ZERO;
}

FrTensor Fr_partial_me(const FrTensor& t, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, uint window_size)
{
    if (begin >= end) return t;
    if (window_size == 0) throw std::runtime_error("Invalid window size");

    std::vector<Fr_t> current(t.size);
    if (!current.empty())
    {
        cudaMemcpy(current.data(), t.gpu_data, sizeof(Fr_t) * current.size(), cudaMemcpyDeviceToHost);
    }
    for (auto it = begin; it != end; ++it)
    {
        const uint in_size = static_cast<uint>(current.size());
        const uint out_size = Fr_partial_me_out_size(in_size, window_size);
        std::vector<Fr_t> next(out_size, Fr_host_zero());
        for (uint gid = 0; gid < out_size; ++gid)
        {
            const uint window_id = gid / window_size;
            const uint idx_in_window = gid % window_size;
            const uint gid0 = 2U * window_id * window_size + idx_in_window;
            const uint gid1 = (2U * window_id + 1U) * window_size + idx_in_window;
            const Fr_t low = (gid0 < in_size) ? current[gid0] : Fr_host_zero();
            const Fr_t high = (gid1 < in_size) ? current[gid1] : Fr_host_zero();
            next[gid] = Fr_host_mle_step(low, high, *it);
        }
        current.swap(next);
    }
    return FrTensor(static_cast<uint>(current.size()), current.data());
}

ostream& operator<<(ostream& os, const FrTensor& A)
{
    os << '['; 
    for (uint i = 0; i < A.size - 1; ++ i) os << A(i) << '\n';
    os << A(A.size-1) << ']';
    return os;
}

#endif
