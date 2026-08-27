#ifndef FR_TENSOR_CUH
#define FR_TENSOR_CUH

#include <iostream>
#include <iomanip>
#include <utility>      // std::pair, std::make_pair
#include <vector>
#include <cstring>
#include <cstdint>
#include "../../../src/core/bls12-381.cuh"
using namespace std;

typedef blstrs__scalar__Scalar Fr_t;
typedef blstrs__g1__G1Affine_affine G1Affine_t;
typedef blstrs__g1__G1Affine_jacobian G1Jacobian_t;

const uint FrNumThread = 4;
const uint FrSharedMemorySize = 2 * sizeof(Fr_t) * FrNumThread; 

// 返回 Fr 零元；用于 padding 区和默认 witness。
inline Fr_t Fr_host_zero()
{
    return {{0, 0, 0, 0, 0, 0, 0, 0}};
}

// 返回 Montgomery 形态的一；sumcheck/opening 中的乘法使用。
inline Fr_t Fr_host_one_mont()
{
    return {{4294967294U, 1U, 215042U, 1485092858U, 3971764213U, 2576109551U, 2898593135U, 405057881U}};
}

// 返回 raw 形态的一；Montgomery 转换时使用。
inline Fr_t Fr_host_one_raw()
{
    return {{1U, 0, 0, 0, 0, 0, 0, 0}};
}

// 返回 BLS12-381 scalar field 模数；负数编码和范围检查使用。
inline Fr_t Fr_host_modulus()
{
    return {{1U, 4294967295U, 4294859774U, 1404937218U, 161601541U, 859428872U, 698187080U, 1944954707U}};
}

// 返回 Montgomery R^2；raw 到 mont 转换使用。
inline Fr_t Fr_host_r2()
{
    return {{4092763245U, 3382307216U, 2274516003U, 728559051U, 1918122383U, 97719446U, 2673475345U, 122214873U}};
}

// 比较两个 Fr 编码是否相同；verifier 和 selftest 精确检查使用。
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

// host Fr 加法；输入 raw/mont 同形元素，输出同形元素。
inline Fr_t Fr_host_add(const Fr_t& a, const Fr_t& b)
{
    Fr_t out = Fr_host_add_raw(a, b);
    const Fr_t p = Fr_host_modulus();
    if (Fr_host_gte(out, p)) out = Fr_host_sub_raw(out, p);
    return out;
}

// host Fr 减法；用于线性关系和 MLE 折叠。
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

// host Fr Montgomery 乘法；用于 verifier 重放 sumcheck。
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

// raw Fr 转 Montgomery 形态；commitment/sumcheck 输入使用 mont 值。
inline Fr_t Fr_host_mont(const Fr_t& a)
{
    return Fr_host_mul(a, Fr_host_r2());
}

// Montgomery Fr 转 raw 形态；commitment 标量乘法前使用。
inline Fr_t Fr_host_unmont(const Fr_t& a)
{
    return Fr_host_mul(a, Fr_host_one_raw());
}

// 单步 MLE 折叠 low + u*(high-low)；opening 与 sumcheck 共用。
inline Fr_t Fr_host_mle_step(const Fr_t& low, const Fr_t& high, const Fr_t& challenge)
{
    return Fr_host_add(low, Fr_host_mul(challenge, Fr_host_sub(high, low)));
}

// 计算 partial MLE 一轮输出长度；影响 opening transcript 维度。
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

// GPU Fr 张量容器；official GPU relation proof 的 witness/commitment 输入。
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

    // 对张量全部元素求和；输出 sumcheck claim。
    Fr_t sum() const;

    // 在挑战点 u 上计算 MLE；输出 opening 声明值。
    Fr_t operator()(const vector<Fr_t>& u) const;

    // 按 window 拆分 MLE 左右子张量；opening 递归使用。
    std::pair<FrTensor, FrTensor> split(uint window_size) const;

    // 计算 partial MLE；输入外层挑战，输出缩减 FrTensor。
    FrTensor partial_me(vector<Fr_t> u, uint window_size) const;

    // 拷贝 GPU FrTensor 到 host；用于 verifier 和 proof object 摘要。
    std::vector<Fr_t> to_host() const;

    friend Fr_t Fr_me(const FrTensor& t, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end);

    friend FrTensor Fr_partial_me(const FrTensor& t, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, uint window_size);

    friend class G1TensorAffine;
    friend class G1TensorJacobian;
    friend class Commitment;
    friend class zkFC;
    friend class zkReLU;
    friend class zkMatMul;
};

// GPU 归约求和 kernel；用于 FrTensor::sum。
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

// 对 GPU FrTensor 求和；输出 relation 的 claimed_sum。
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


// 在 MLE 点 u 上评值 FrTensor；输出 commitment opening 的声明值。
Fr_t FrTensor::operator()(const vector<Fr_t>& u) const
{
    uint log_dim = u.size();
    const uint padded = 1U << log_dim;
    if (size <= (padded / 2U) || size > padded) throw std::runtime_error("Incompatible dimensions");
    return Fr_me(*this, u.begin(), u.end());
}

// FrTensor partial MLE 入口；输出缩减张量给 commitment opening。
FrTensor FrTensor::partial_me(vector<Fr_t> u, uint window_size) const
{
    if (window_size == 0) throw std::runtime_error("Invalid window size");
    if (u.empty()) return *this;
    const std::size_t max_size = static_cast<std::size_t>(window_size) << u.size();
    const std::size_t min_size = static_cast<std::size_t>(window_size) << (u.size() - 1);
    if (size <= min_size || static_cast<std::size_t>(size) > max_size) throw std::runtime_error("Incompatible dimensions");
    return Fr_partial_me(*this, u.begin(), u.end(), window_size);
}

// GPU 按窗口拆分 FrTensor；Gate 自测用它检查 partial MLE 维度。
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

// 将 FrTensor 拆成低/高两半；opening 递归使用。
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

// 拷贝 FrTensor 到 host vector；用于 proof generation/verifier。
std::vector<Fr_t> FrTensor::to_host() const
{
    std::vector<Fr_t> out(size);
    if (!out.empty())
    {
        cudaMemcpy(out.data(), gpu_data, sizeof(Fr_t) * size, cudaMemcpyDeviceToHost);
    }
    return out;
}

// GPU MLE 单轮折叠 kernel；输入当前数组和挑战，输出下一轮数组。
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

// host 驱动的 FrTensor MLE 评值；输出 commitment opening 值。
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

// GPU partial MLE 单轮折叠 kernel；按 window 保留内层维。
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

// 计算 FrTensor partial MLE；输出缩减后的 GPU 张量。
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
