#ifndef FR_TENSOR_CUH
#define FR_TENSOR_CUH

#include <iostream>
#include <iomanip>
#include <utility>      // std::pair, std::make_pair
#include <vector>
#include <curand_kernel.h>
#include <random>
#include "bls12-381.cuh"
using namespace std;

typedef blstrs__scalar__Scalar Fr_t;
typedef blstrs__g1__G1Affine_affine G1Affine_t;
typedef blstrs__g1__G1Affine_jacobian G1Jacobian_t;

const uint FrNumThread = 256;
const uint FrSharedMemorySize = 2 * sizeof(Fr_t) * FrNumThread; 

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

DEVICE Fr_t Fr_field_mul(Fr_t a, Fr_t b);

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
  arr_out[gid] = Fr_field_mul(arr[gid], x);
}

// Canonical field multiplication for wire values.  The upstream kernels
// historically mixed Montgomery and canonical encodings.  Opening points,
// transcript challenges and serialized claims use canonical residues, so this
// helper makes the conversion explicit at the native CUDA boundary.
DEVICE Fr_t Fr_field_mul(Fr_t a, Fr_t b)
{
    return blstrs__scalar__Scalar_unmont(
        blstrs__scalar__Scalar_mul(blstrs__scalar__Scalar_mont(a),
                                   blstrs__scalar__Scalar_mont(b)));
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

    // Explicit host/device transfer hooks used by the verifier-facing
    // envelope and by the GAT adapter.  They expose the native tensor storage
    // without introducing a second tensor implementation.
    const Fr_t* device_data() const { return gpu_data; }
    Fr_t* device_data() { return gpu_data; }

    std::vector<Fr_t> to_host() const
    {
        std::vector<Fr_t> out(size);
        if (size != 0) cudaMemcpy(out.data(), gpu_data, sizeof(Fr_t) * size, cudaMemcpyDeviceToHost);
        return out;
    }

    void from_host(const Fr_t* cpu_data, uint count)
    {
        if (count != size) throw std::runtime_error("Incompatible tensor transfer size");
        if (size != 0) cudaMemcpy(gpu_data, cpu_data, sizeof(Fr_t) * size, cudaMemcpyHostToDevice);
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
        // Each reduction block consumes two input tiles of blockDim.x.
        uint gridSize = (curSize + 2 * FrNumThread - 1) / (2 * FrNumThread);
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
    if (log_dim >= 32 || size == 0) throw std::runtime_error("Incompatible dimensions");
    const uint64_t domain = uint64_t(1) << log_dim;
    if (size <= domain / 2 || uint64_t(size) > domain) throw std::runtime_error("Incompatible dimensions");
    return Fr_me(*this, u.begin(), u.end());
}

KERNEL void random_int_kernel(Fr_t* gpu_data, uint num_bits, uint n, unsigned long seed)
{
    if (num_bits == 0 || num_bits > 32) return;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    
    // Initialize the RNG state for this thread.
    curand_init(seed, tid, 0, &state);  
    
    if (tid < n) {
        const uint mask = num_bits == 32 ? 0xffffffffU : ((1U << num_bits) - 1U);
        const uint sign = num_bits == 32 ? 0x80000000U : (1U << (num_bits - 1));
        gpu_data[tid] = {curand(&state) & mask, 0, 0, 0, 0, 0, 0, 0};
        gpu_data[tid] = blstrs__scalar__Scalar_sub(gpu_data[tid], {sign, 0, 0, 0, 0, 0, 0, 0});
    }
}

FrTensor FrTensor::random_int(uint size, uint num_bits)
{
    if (num_bits == 0 || num_bits > 32) throw std::invalid_argument("random_int bit width must be in [1,32]");
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

    if (tid >= n) return;
    
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
    // Prevent overflow in the power-of-two shape calculation.  An empty
    // point is a valid no-op partial MLE evaluation.
    if (window_size == 0 || u.size() >= 32) throw std::runtime_error("Incompatible dimensions");
    const uint64_t required = uint64_t(window_size) << u.size();
    if (required == 0 || uint64_t(size) <= required / 2U) throw std::runtime_error("Incompatible dimensions");
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
    uint out_size = (size + 1) / 2;
    std::pair<FrTensor, FrTensor> out {out_size, out_size};
    Fr_split_by_window<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(gpu_data, out.first.gpu_data, out.second.gpu_data, size, out_size, window_size);
    cudaDeviceSynchronize();
    return out;
}

KERNEL void Fr_me_step(GLOBAL Fr_t *arr_in, GLOBAL Fr_t *arr_out, Fr_t x, uint in_size, uint out_size)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= out_size) return;
    
    uint gid0 = 2 * gid;
    uint gid1 = 2 * gid + 1;
    if (gid1 < in_size) arr_out[gid] = blstrs__scalar__Scalar_add(arr_in[gid0], Fr_field_mul(x, blstrs__scalar__Scalar_sub(arr_in[gid1], arr_in[gid0])));
    else if (gid0 < in_size) arr_out[gid] = blstrs__scalar__Scalar_sub(arr_in[gid0], Fr_field_mul(x, arr_in[gid0]));
    else arr_out[gid] = blstrs__scalar__Scalar_ZERO;
}

Fr_t Fr_me(const FrTensor& t, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end)
{
    FrTensor t_new((t.size + 1) / 2);
    if (begin >= end) return t(0);
    Fr_me_step<<<(t_new.size+FrNumThread-1)/FrNumThread,FrNumThread>>>(t.gpu_data, t_new.gpu_data, *begin, t.size, t_new.size);
    cudaDeviceSynchronize();
    return Fr_me(t_new, begin + 1, end);
}

KERNEL void Fr_partial_me_step(GLOBAL Fr_t *arr_in, GLOBAL Fr_t *arr_out, Fr_t x, uint in_size, uint out_size, uint window_size)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= out_size) return;
    
    uint window_id = gid / window_size;
    uint idx_in_window = gid % window_size;
    uint gid0 = 2 * window_id * window_size + idx_in_window;
    uint gid1 = (2 * window_id + 1) * window_size + idx_in_window;
    if (gid1 < in_size) arr_out[gid] = blstrs__scalar__Scalar_add(arr_in[gid0], Fr_field_mul(x, blstrs__scalar__Scalar_sub(arr_in[gid1], arr_in[gid0])));
    else if (gid0 < in_size) arr_out[gid] = blstrs__scalar__Scalar_sub(arr_in[gid0], Fr_field_mul(x, arr_in[gid0]));
    else arr_out[gid] = blstrs__scalar__Scalar_ZERO;
}

FrTensor Fr_partial_me(const FrTensor& t, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, uint window_size)
{
    if (begin >= end) return t;
    uint num_windows = (t.size + 2 * window_size - 1) / (2 * window_size);
    uint out_size = window_size * num_windows;
    FrTensor t_new(out_size);
    Fr_partial_me_step<<<(t_new.size+FrNumThread-1)/FrNumThread,FrNumThread>>>(t.gpu_data, t_new.gpu_data, *begin, t.size, t_new.size, window_size);
    cudaDeviceSynchronize();
    return Fr_partial_me(t_new, begin + 1, end, window_size);
}

ostream& operator<<(ostream& os, const FrTensor& A)
{
    os << '['; 
    for (uint i = 0; i < A.size - 1; ++ i) os << A(i) << '\n';
    os << A(A.size-1) << ']';
    return os;
}

#endif
