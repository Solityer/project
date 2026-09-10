#ifndef G1_TENSOR_CUH
#define G1_TENSOR_CUH

#include <iostream>
#include <iomanip>
#include "bls12-381.cuh"
#include "fr-tensor.cuh"
using namespace std;

typedef blstrs__fp__Fp Fp_t;
const uint G1NumThread = 64;
const uint G1AffineSharedMemorySize = 2 * sizeof(G1Affine_t) * G1NumThread; 
const uint G1JacobianSharedMemorySize = 2 * sizeof(G1Jacobian_t) * G1NumThread;
// A fixed-window table is supplied by the caller in global scratch memory.
// Keeping the table out of the device stack avoids the register/local-memory
// blow-up caused by a 256-round unrolled scalar multiplication.
const uint G1ScalarWindowBits = 4;
const uint G1ScalarWindowSize = 1U << G1ScalarWindowBits;
const uint G1ScalarWindowCount =
    (blstrs__scalar__Scalar_LIMBS * blstrs__scalar__Scalar_LIMB_BITS) /
    G1ScalarWindowBits;
const uint G1ScalarWindowTile = 64;

DEVICE Fp_t Fp_minus(Fp_t a) {
	return blstrs__fp__Fp_sub(blstrs__fp__Fp_ZERO, a);
}

DEVICE G1Affine_t G1Affine_minus(G1Affine_t a) {
	return {a.x, Fp_minus(a.y)};
}

DEVICE G1Jacobian_t G1Jacobian_minus(G1Jacobian_t a) {
	return {a.x, Fp_minus(a.y), a.z};
}

ostream& operator<<(ostream& os, const Fp_t& x)
{
  os << "0x" << std::hex;
  for (uint i = 12; i > 0; -- i)
  {
    os << std::setfill('0') << std::setw(8) << x.val[i - 1];
  }
  return os << std::dec << std::setw(0) << std::setfill(' ');
}

ostream& operator<<(ostream& os, const G1Affine_t& g)
{
	os << "(" << g.x << ", " << g.y << ")";
	return os;
}

ostream& operator<<(ostream& os, const G1Jacobian_t& g)
{
	os << "(" << g.x << ", " << g.y <<  ", " << g.z << ")";
	return os;
}


// x_mont = 0x120177419e0bfb75edce6ecc21dbf440f0ae6acdf3d0e747154f95c7143ba1c17817fc679976fff55cb38790fd530c16
const Fp_t G1_generator_x_mont = {
    4250078230,
    1555269520,
    2574712821,
    2014837863,
    339452353,
    357537223,
    4090554183,
    4037962445,
    568063040,
    3989728972,
    2651585397,
    302085953
};

// y_mont = 0xbbc3efc5008a26a0e1c8c3fad0059c051ac582950405194dd595f13570725ce8c22631a7918fd8ebaac93d50ce72271
const Fp_t G1_generator_y_mont = {
    216474225,
    3131872213,
    2031680910,
    2351063834,
    1460086222,
    3713621779,
    1346392468,
    1370249257,
    2902481344,
    236751935,
    1342743146,
    196886268
};

const Fp_t G1_ONE = {196605, 1980301312, 3289120770, 3958636555, 1405573306, 1598593111, 1884444485, 2010011731, 2723605613, 1543969431, 4202751123, 368467651};

const G1Affine_t G1Affine_generator {G1_generator_x_mont, G1_generator_y_mont};
const G1Jacobian_t G1Jacobian_generator {G1_generator_x_mont, G1_generator_y_mont, G1_ONE};

class G1Tensor
{
    public:
    const uint size;

    G1Tensor(uint size): size(size) {}
};

class G1TensorAffine;
class G1TensorJacobian;

class G1TensorAffine: public G1Tensor
{
    protected:
    G1Affine_t* gpu_data;

    public: 
    G1TensorAffine(const G1TensorAffine&);

    G1TensorAffine(uint size);

    G1TensorAffine(uint size, const G1Affine_t&);

    G1TensorAffine(uint size, const G1Affine_t* cpu_data);

    ~G1TensorAffine();

	G1Affine_t operator()(uint idx) const;
	// {
	// 	G1Affine_t out;
	// 	cudaMemcpy(&out, gpu_data + idx, sizeof(G1Affine_t), cudaMemcpyDeviceToHost);
	// 	return out;
	// }

    const G1Affine_t* device_data() const { return gpu_data; }
    G1Affine_t* device_data() { return gpu_data; }
    std::vector<G1Affine_t> to_host() const
    {
        std::vector<G1Affine_t> out(size);
        if (size != 0) cudaMemcpy(out.data(), gpu_data, sizeof(G1Affine_t) * size, cudaMemcpyDeviceToHost);
        return out;
    }

    G1TensorAffine operator-() const;

    G1TensorJacobian& operator*(const FrTensor&);

    friend class G1TensorJacobian;
};

class Commitment;

class G1TensorJacobian: public G1Tensor
{
    protected:
    G1Jacobian_t* gpu_data;

    public: 
    G1TensorJacobian(const G1TensorJacobian&);

    G1TensorJacobian(uint size);

    G1TensorJacobian(uint size, const G1Jacobian_t&);

    G1TensorJacobian(uint size, const G1Jacobian_t* cpu_data);

    G1TensorJacobian(const G1TensorAffine& affine_tensor);

    ~G1TensorJacobian();

	G1Jacobian_t operator()(uint) const;

    const G1Jacobian_t* device_data() const { return gpu_data; }
    G1Jacobian_t* device_data() { return gpu_data; }
    std::vector<G1Jacobian_t> to_host() const
    {
        std::vector<G1Jacobian_t> out(size);
        if (size != 0) cudaMemcpy(out.data(), gpu_data, sizeof(G1Jacobian_t) * size, cudaMemcpyDeviceToHost);
        return out;
    }

    G1TensorJacobian operator-() const;

    G1TensorJacobian operator+(const G1TensorJacobian&) const;
    
    G1TensorJacobian operator+(const G1TensorAffine&) const;

    G1TensorJacobian operator+(const G1Jacobian_t&) const;

    G1TensorJacobian operator+(const G1Affine_t&) const;

    G1TensorJacobian& operator+=(const G1TensorJacobian&);
    
    G1TensorJacobian& operator+=(const G1TensorAffine&);

    G1TensorJacobian& operator+=(const G1Jacobian_t&);

    G1TensorJacobian& operator+=(const G1Affine_t&);

    G1TensorJacobian operator-(const G1TensorJacobian&) const;
    
    G1TensorJacobian operator-(const G1TensorAffine&) const;

    G1TensorJacobian operator-(const G1Jacobian_t&) const;

    G1TensorJacobian operator-(const G1Affine_t&) const;

    G1TensorJacobian& operator-=(const G1TensorJacobian&);
    
    G1TensorJacobian& operator-=(const G1TensorAffine&);

    G1TensorJacobian& operator-=(const G1Jacobian_t&);

    G1TensorJacobian& operator-=(const G1Affine_t&);

    G1Jacobian_t sum() const;

    G1TensorJacobian operator*(const FrTensor&) const;

    G1TensorJacobian& operator*=(const FrTensor&);

    G1Jacobian_t operator()(const vector<Fr_t>& u) const;

    friend G1Jacobian_t G1_me(const G1TensorJacobian& t, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end);

    friend class G1TensorAffine;
    friend class Commitment;
};

// Implement G1Affine

G1TensorAffine::G1TensorAffine(const G1TensorAffine& t): G1Tensor(t.size), gpu_data(nullptr)
{
    cudaMalloc((void **)&gpu_data, sizeof(G1Affine_t) * size);
    cudaMemcpy(gpu_data, t.gpu_data, sizeof(G1Affine_t) * size, cudaMemcpyDeviceToDevice);
}

G1TensorAffine::G1TensorAffine(uint size): G1Tensor(size), gpu_data(nullptr)
{
    cudaMalloc((void **)&gpu_data, sizeof(G1Affine_t) * size);
}

KERNEL void G1Affine_assign_broadcast(GLOBAL G1Affine_t* arr, GLOBAL G1Affine_t g, uint n)
{
	const uint gid = GET_GLOBAL_ID();
	if (gid >= n) return;
	arr[gid] = g;
}

G1TensorAffine::G1TensorAffine(uint size, const G1Affine_t& g): G1Tensor(size), gpu_data(nullptr)
{
    cudaMalloc((void **)&gpu_data, sizeof(G1Affine_t) * size);
    G1Affine_assign_broadcast<<<(size+G1NumThread-1)/G1NumThread,G1NumThread>>>(gpu_data, g, size);
    cudaDeviceSynchronize();
}

G1TensorAffine::G1TensorAffine(uint size, const G1Affine_t* cpu_data): G1Tensor(size), gpu_data(nullptr)
{
    cudaMalloc((void **)&gpu_data, sizeof(G1Affine_t) * size);
    cudaMemcpy(gpu_data, cpu_data, sizeof(G1Affine_t) * size, cudaMemcpyHostToDevice);
}

G1TensorAffine::~G1TensorAffine()
{
    cudaFree(gpu_data);
    gpu_data = nullptr;
}

G1Affine_t G1TensorAffine::operator()(uint idx) const
{
    G1Affine_t out;
    cudaMemcpy(&out, gpu_data + idx, sizeof(G1Affine_t), cudaMemcpyDeviceToHost);
    return out;
}

KERNEL void G1_affine_elementwise_minus(GLOBAL G1Affine_t* arr_in, GLOBAL G1Affine_t* arr_out, uint n)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;
    arr_out[gid] = {arr_in[gid].x, blstrs__fp__Fp_sub(blstrs__fp__Fp_ZERO, arr_in[gid].y)};
}

G1TensorAffine G1TensorAffine::operator-() const
{
    G1TensorAffine out(size);
    G1_affine_elementwise_minus<<<(size+G1NumThread-1)/G1NumThread,G1NumThread>>>(gpu_data, out.gpu_data, size);
    cudaDeviceSynchronize();
    return out;
}


// Implement G1TensorJacobian

G1TensorJacobian::G1TensorJacobian(const G1TensorJacobian& t): G1Tensor(t.size), gpu_data(nullptr)
{
    cudaMalloc((void **)&gpu_data, sizeof(G1Jacobian_t) * size);
    cudaMemcpy(gpu_data, t.gpu_data, sizeof(G1Jacobian_t) * size, cudaMemcpyDeviceToDevice);
}

G1TensorJacobian::G1TensorJacobian(uint size): G1Tensor(size), gpu_data(nullptr)
{
    cudaMalloc((void **)&gpu_data, sizeof(G1Jacobian_t) * size);
}

G1TensorJacobian::G1TensorJacobian(uint size, const G1Jacobian_t* cpu_data): G1Tensor(size), gpu_data(nullptr)
{
    cudaMalloc((void **)&gpu_data, sizeof(G1Jacobian_t) * size);
    cudaMemcpy(gpu_data, cpu_data, sizeof(G1Jacobian_t) * size, cudaMemcpyHostToDevice);
}

KERNEL void G1Jacobian_assign_broadcast(GLOBAL G1Jacobian_t* arr, G1Jacobian_t g, uint n)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;
    arr[gid] = g;
}

G1TensorJacobian::G1TensorJacobian(uint size, const G1Jacobian_t& g): G1Tensor(size), gpu_data(nullptr)
{
    cudaMalloc((void **)&gpu_data, sizeof(G1Jacobian_t) * size);
    G1Jacobian_assign_broadcast<<<(size+G1NumThread-1)/G1NumThread,G1NumThread>>>(gpu_data, g, size);
    cudaDeviceSynchronize();
}

KERNEL void G1_affine_to_jacobian(GLOBAL G1Affine_t* arr_affine, GLOBAL G1Jacobian_t* arr_jacobian, uint n)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;
    arr_jacobian[gid] = {arr_affine[gid].x, arr_affine[gid].y, blstrs__fp__Fp_ONE};
}

G1TensorJacobian::G1TensorJacobian(const G1TensorAffine& affine_tensor): G1Tensor(affine_tensor.size), gpu_data(nullptr)
{
    cudaMalloc((void **)&gpu_data, sizeof(G1Jacobian_t) * size);
    G1_affine_to_jacobian<<<(size+G1NumThread-1)/G1NumThread,G1NumThread>>>(affine_tensor.gpu_data, gpu_data, size);
    cudaDeviceSynchronize();
}

G1TensorJacobian::~G1TensorJacobian()
{
    cudaFree(gpu_data);
    gpu_data = nullptr;
}

G1Jacobian_t G1TensorJacobian::operator()(uint idx) const
{
	G1Jacobian_t out;
	cudaMemcpy(&out, gpu_data + idx, sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
	return out;
}

KERNEL void G1_jacobian_elementwise_minus(GLOBAL G1Jacobian_t* arr_in, GLOBAL G1Jacobian_t* arr_out, uint n)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;
    arr_out[gid] = {arr_in[gid].x, blstrs__fp__Fp_sub(blstrs__fp__Fp_ZERO, arr_in[gid].y), arr_in[gid].z};
}

G1TensorJacobian G1TensorJacobian::operator-() const
{
    G1TensorJacobian out(size);
    G1_jacobian_elementwise_minus<<<(size+G1NumThread-1)/G1NumThread,G1NumThread>>>(gpu_data, out.gpu_data, size);
    cudaDeviceSynchronize();
    return out;
}

KERNEL void G1_jacobian_elementwise_add(GLOBAL G1Jacobian_t* arr1, GLOBAL G1Jacobian_t* arr2, GLOBAL G1Jacobian_t* arr_out, uint n)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;
    arr_out[gid] = blstrs__g1__G1Affine_add(arr1[gid], arr2[gid]);
}

KERNEL void G1_jacobian_broadcast_add(GLOBAL G1Jacobian_t* arr, G1Jacobian_t x, GLOBAL G1Jacobian_t* arr_out, uint n)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;
    arr_out[gid] = blstrs__g1__G1Affine_add(arr[gid], x);
}

KERNEL void G1_jacobian_elementwise_madd(GLOBAL G1Jacobian_t* arr1, GLOBAL G1Affine_t* arr2, GLOBAL G1Jacobian_t* arr_out, uint n)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;
    arr_out[gid] = blstrs__g1__G1Affine_add_mixed(arr1[gid], arr2[gid]);
}

KERNEL void G1_jacobian_broadcast_madd(GLOBAL G1Jacobian_t* arr, G1Affine_t x, GLOBAL G1Jacobian_t* arr_out, uint n)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;
    arr_out[gid] = blstrs__g1__G1Affine_add_mixed(arr[gid], x);
}

G1TensorJacobian G1TensorJacobian::operator+(const G1TensorJacobian& t) const
{
	if (size != t.size) throw std::runtime_error("Incompatible dimensions");
	G1TensorJacobian out(size);
	G1_jacobian_elementwise_add<<<(size+G1NumThread-1)/G1NumThread,G1NumThread>>>(gpu_data, t.gpu_data, out.gpu_data, size);
	cudaDeviceSynchronize();
	return out;
}
    
G1TensorJacobian G1TensorJacobian::operator+(const G1TensorAffine& t) const
{
	if (size != t.size) throw std::runtime_error("Incompatible dimensions");
	G1TensorJacobian out(size);
	G1_jacobian_elementwise_madd<<<(size+G1NumThread-1)/G1NumThread,G1NumThread>>>(gpu_data, t.gpu_data, out.gpu_data, size);
	cudaDeviceSynchronize();
	return out;
}

G1TensorJacobian G1TensorJacobian::operator+(const G1Jacobian_t& x) const
{
	G1TensorJacobian out(size);
	G1_jacobian_broadcast_add<<<(size+G1NumThread-1)/G1NumThread,G1NumThread>>>(gpu_data, x, out.gpu_data, size);
	cudaDeviceSynchronize();
	return out;
}

G1TensorJacobian G1TensorJacobian::operator+(const G1Affine_t& x) const
{
	G1TensorJacobian out(size);
	G1_jacobian_broadcast_madd<<<(size+G1NumThread-1)/G1NumThread,G1NumThread>>>(gpu_data, x, out.gpu_data, size);
	cudaDeviceSynchronize();
	return out;
}

G1TensorJacobian& G1TensorJacobian::operator+=(const G1TensorJacobian& t)
{
	if (size != t.size) throw std::runtime_error("Incompatible dimensions");
	G1_jacobian_elementwise_add<<<(size+G1NumThread-1)/G1NumThread,G1NumThread>>>(gpu_data, t.gpu_data, gpu_data, size);
	cudaDeviceSynchronize();
	return *this;
}
    
G1TensorJacobian& G1TensorJacobian::operator+=(const G1TensorAffine& t)
{
	if (size != t.size) throw std::runtime_error("Incompatible dimensions");
	G1_jacobian_elementwise_madd<<<(size+G1NumThread-1)/G1NumThread,G1NumThread>>>(gpu_data, t.gpu_data, gpu_data, size);
	cudaDeviceSynchronize();
	return *this;
}

G1TensorJacobian& G1TensorJacobian::operator+=(const G1Jacobian_t& x)
{
	G1_jacobian_broadcast_add<<<(size+G1NumThread-1)/G1NumThread,G1NumThread>>>(gpu_data, x, gpu_data, size);
	cudaDeviceSynchronize();
	return *this;
}

G1TensorJacobian& G1TensorJacobian::operator+=(const G1Affine_t& x)
{
	G1_jacobian_broadcast_madd<<<(size+G1NumThread-1)/G1NumThread,G1NumThread>>>(gpu_data, x, gpu_data, size);
	cudaDeviceSynchronize();
	return *this;
}

KERNEL void G1_jacobian_elementwise_sub(GLOBAL G1Jacobian_t* arr1, GLOBAL G1Jacobian_t* arr2, GLOBAL G1Jacobian_t* arr_out, uint n)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;
    arr_out[gid] = blstrs__g1__G1Affine_add(arr1[gid], G1Jacobian_minus(arr2[gid]));
}

KERNEL void G1_jacobian_broadcast_sub(GLOBAL G1Jacobian_t* arr, G1Jacobian_t x, GLOBAL G1Jacobian_t* arr_out, uint n)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;
    arr_out[gid] = blstrs__g1__G1Affine_add(arr[gid], G1Jacobian_minus(x));
}

KERNEL void G1_jacobian_elementwise_msub(GLOBAL G1Jacobian_t* arr1, GLOBAL G1Affine_t* arr2, GLOBAL G1Jacobian_t* arr_out, uint n)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;
    arr_out[gid] = blstrs__g1__G1Affine_add_mixed(arr1[gid], G1Affine_minus(arr2[gid]));
}

KERNEL void G1_jacobian_broadcast_msub(GLOBAL G1Jacobian_t* arr, G1Affine_t x, GLOBAL G1Jacobian_t* arr_out, uint n)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;
    arr_out[gid] = blstrs__g1__G1Affine_add_mixed(arr[gid], G1Affine_minus(x));
}

G1TensorJacobian G1TensorJacobian::operator-(const G1TensorJacobian& t) const
{
	if (size != t.size) throw std::runtime_error("Incompatible dimensions");
	G1TensorJacobian out(size);
	G1_jacobian_elementwise_sub<<<(size+G1NumThread-1)/G1NumThread,G1NumThread>>>(gpu_data, t.gpu_data, out.gpu_data, size);
	cudaDeviceSynchronize();
	return out;
}
    
G1TensorJacobian G1TensorJacobian::operator-(const G1TensorAffine& t) const
{
	if (size != t.size) throw std::runtime_error("Incompatible dimensions");
	G1TensorJacobian out(size);
	G1_jacobian_elementwise_msub<<<(size+G1NumThread-1)/G1NumThread,G1NumThread>>>(gpu_data, t.gpu_data, out.gpu_data, size);
	cudaDeviceSynchronize();
	return out;
}

G1TensorJacobian G1TensorJacobian::operator-(const G1Jacobian_t& x) const
{
	G1TensorJacobian out(size);
	G1_jacobian_broadcast_sub<<<(size+G1NumThread-1)/G1NumThread,G1NumThread>>>(gpu_data, x, out.gpu_data, size);
	cudaDeviceSynchronize();
	return out;
}

G1TensorJacobian G1TensorJacobian::operator-(const G1Affine_t& x) const
{
	G1TensorJacobian out(size);
	G1_jacobian_broadcast_msub<<<(size+G1NumThread-1)/G1NumThread,G1NumThread>>>(gpu_data, x, out.gpu_data, size);
	cudaDeviceSynchronize();
	return out;
}

G1TensorJacobian& G1TensorJacobian::operator-=(const G1TensorJacobian& t)
{
	if (size != t.size) throw std::runtime_error("Incompatible dimensions");
	G1_jacobian_elementwise_sub<<<(size+G1NumThread-1)/G1NumThread,G1NumThread>>>(gpu_data, t.gpu_data, gpu_data, size);
	cudaDeviceSynchronize();
	return *this;
}
    
G1TensorJacobian& G1TensorJacobian::operator-=(const G1TensorAffine& t)
{
	if (size != t.size) throw std::runtime_error("Incompatible dimensions");
	G1_jacobian_elementwise_msub<<<(size+G1NumThread-1)/G1NumThread,G1NumThread>>>(gpu_data, t.gpu_data, gpu_data, size);
	cudaDeviceSynchronize();
	return *this;
}

G1TensorJacobian& G1TensorJacobian::operator-=(const G1Jacobian_t& x)
{
	G1_jacobian_broadcast_sub<<<(size+G1NumThread-1)/G1NumThread,G1NumThread>>>(gpu_data, x, gpu_data, size);
	cudaDeviceSynchronize();
	return *this;
}

G1TensorJacobian& G1TensorJacobian::operator-=(const G1Affine_t& x)
{
	G1_jacobian_broadcast_msub<<<(size+G1NumThread-1)/G1NumThread,G1NumThread>>>(gpu_data, x, gpu_data, size);
	cudaDeviceSynchronize();
	return *this;
}

KERNEL void G1Jacobian_sum_reduction(GLOBAL G1Jacobian_t *arr, GLOBAL G1Jacobian_t *output, uint n) {
    extern __shared__ G1Jacobian_t g1sum_sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (2 * blockDim.x) + threadIdx.x;

    // Load input into shared memory
    g1sum_sdata[tid] = (i < n) ? arr[i] : blstrs__g1__G1Affine_ZERO;
    if (i + blockDim.x < n) g1sum_sdata[tid] = blstrs__g1__G1Affine_add(g1sum_sdata[tid], arr[i + blockDim.x]);

    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            g1sum_sdata[tid] = blstrs__g1__G1Affine_add(g1sum_sdata[tid], g1sum_sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write the result for this block to output
    if (tid == 0) output[blockIdx.x] = g1sum_sdata[0];
}

G1Jacobian_t G1TensorJacobian::sum() const
{
    G1Jacobian_t *ptr_input, *ptr_output;
    uint curSize = size;
    cudaMalloc((void**)&ptr_input, size * sizeof(G1Jacobian_t));
    cudaMalloc((void**)&ptr_output, ((size + 1)/ 2) * sizeof(G1Jacobian_t));
    cudaMemcpy(ptr_input, gpu_data, size * sizeof(G1Jacobian_t), cudaMemcpyDeviceToDevice);

    while(curSize > 1) {
        // Each reduction block consumes two input tiles of blockDim.x.
        uint gridSize = (curSize + 2 * G1NumThread - 1) / (2 * G1NumThread);
        G1Jacobian_sum_reduction<<<gridSize, G1NumThread, G1JacobianSharedMemorySize>>>(ptr_input, ptr_output, curSize);
        cudaDeviceSynchronize(); // Ensure kernel completion before proceeding
        
        // Swap pointers. Use the output from this step as the input for the next step.
        G1Jacobian_t *temp = ptr_input;
        ptr_input = ptr_output;
        ptr_output = temp;
        
        curSize = gridSize;  // The output size is equivalent to the grid size used in the kernel launch
    }

    G1Jacobian_t finalSum;
    cudaMemcpy(&finalSum, ptr_input, sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);

    cudaFree(ptr_input);
    cudaFree(ptr_output);

    return finalSum;
}

/* Native Jacobian addition with explicit infinity/inverse handling.  The
 * upstream affine-named helper evaluates the generic formula even when the
 * two inputs are inverses (H=0); opening folds hit that case for repeated
 * generators.  Keep the same BLS12-381 Jacobian relation but return the
 * canonical infinity before entering the generic H=0 arithmetic. */
DEVICE G1Jacobian_t G1Jacobian_add_safe(G1Jacobian_t a, G1Jacobian_t b) {
    const Fp_t zero = blstrs__fp__Fp_ZERO;
    if (blstrs__fp__Fp_eq(a.z, zero)) return b;
    if (blstrs__fp__Fp_eq(b.z, zero)) return a;

    const Fp_t z1z1 = blstrs__fp__Fp_sqr(a.z);
    const Fp_t z2z2 = blstrs__fp__Fp_sqr(b.z);
    const Fp_t u1 = blstrs__fp__Fp_mul(a.x, z2z2);
    const Fp_t u2 = blstrs__fp__Fp_mul(b.x, z1z1);
    const Fp_t s1 = blstrs__fp__Fp_mul(
        blstrs__fp__Fp_mul(a.y, b.z), z2z2);
    const Fp_t s2 = blstrs__fp__Fp_mul(
        blstrs__fp__Fp_mul(b.y, a.z), z1z1);

    if (blstrs__fp__Fp_eq(u1, u2)) {
        if (blstrs__fp__Fp_eq(s1, s2))
            return blstrs__g1__G1Affine_double(a);
        return blstrs__g1__G1Affine_ZERO;
    }

    const Fp_t h = blstrs__fp__Fp_sub(u2, u1);
    Fp_t i = blstrs__fp__Fp_double(h);
    i = blstrs__fp__Fp_sqr(i);
    const Fp_t j = blstrs__fp__Fp_mul(h, i);
    Fp_t r = blstrs__fp__Fp_sub(s2, s1);
    r = blstrs__fp__Fp_double(r);
    const Fp_t v = blstrs__fp__Fp_mul(u1, i);
    G1Jacobian_t out;
    out.x = blstrs__fp__Fp_sub(
        blstrs__fp__Fp_sub(blstrs__fp__Fp_sqr(r), j),
        blstrs__fp__Fp_double(v));
    out.y = blstrs__fp__Fp_mul(
        blstrs__fp__Fp_sub(v, out.x), r);
    Fp_t sj = blstrs__fp__Fp_mul(s1, j);
    sj = blstrs__fp__Fp_double(sj);
    out.y = blstrs__fp__Fp_sub(out.y, sj);
    out.z = blstrs__fp__Fp_add(a.z, b.z);
    out.z = blstrs__fp__Fp_sqr(out.z);
    out.z = blstrs__fp__Fp_sub(blstrs__fp__Fp_sub(out.z, z1z1), z2z2);
    out.z = blstrs__fp__Fp_mul(out.z, h);
    return out;
}

/*
 * Fixed-window BLS12-381 scalar multiplication for canonical scalar wires.
 *
 * The 4-bit table is caller-owned global scratch rather than a local array.
 * This keeps the per-thread frame bounded while retaining the native
 * Jacobian formulas and the same little-endian scalar encoding used by the
 * CPU reference.  The table is [0..15]P and the accumulator scans all 64
 * nibbles from most significant to least significant.  Canonical scalar
 * validation remains an input/statement responsibility; this routine does
 * not reduce or reinterpret a non-canonical wire.
 */
DEVICE G1Jacobian_t G1Jacobian_mul_fixed_window4(
    G1Jacobian_t point, Fr_t scalar, GLOBAL G1Jacobian_t* window_table)
{
    window_table[0] = blstrs__g1__G1Affine_ZERO;
    window_table[1] = point;
    for (uint i = 2; i < G1ScalarWindowSize; ++i)
        window_table[i] = G1Jacobian_add_safe(window_table[i - 1], point);

    G1Jacobian_t result = blstrs__g1__G1Affine_ZERO;
    for (int window = static_cast<int>(G1ScalarWindowCount) - 1;
         window >= 0; --window) {
        result = blstrs__g1__G1Affine_double(result);
        result = blstrs__g1__G1Affine_double(result);
        result = blstrs__g1__G1Affine_double(result);
        result = blstrs__g1__G1Affine_double(result);

        const uint limb = static_cast<uint>(window) >> 3;
        const uint shift = (static_cast<uint>(window) & 7U) * G1ScalarWindowBits;
        const uint digit = (scalar.val[limb] >> shift) & (G1ScalarWindowSize - 1U);
        if (digit != 0U)
            result = G1Jacobian_add_safe(result, window_table[digit]);
    }
    return result;
}

DEVICE G1Jacobian_t G1Jacobian_mul(G1Jacobian_t a, Fr_t x) {
    G1Jacobian_t out = blstrs__g1__G1Affine_ZERO;
    // Scalar wires are canonical residues, not Montgomery values.  Walk only
    // through the highest set bit to preserve the native double-and-add
    // relation without forcing a 256-iteration unrolled local frame.
    int highest = blstrs__scalar__Scalar_LIMBS * blstrs__scalar__Scalar_LIMB_BITS - 1;
    while (highest >= 0 && ((x.val[highest / blstrs__scalar__Scalar_LIMB_BITS] >>
                             (highest % blstrs__scalar__Scalar_LIMB_BITS)) & 1U) == 0U) {
        --highest;
    }
    if (highest < 0) return out;
    for (int i = 0; i <= highest; ++i) {
        if ((x.val[i / blstrs__scalar__Scalar_LIMB_BITS] >>
             (i % blstrs__scalar__Scalar_LIMB_BITS)) & 1U) {
            out = G1Jacobian_add_safe(out, a);
        }
        a = blstrs__g1__G1Affine_double(a); // (1 << i) * original_a
    }
    return out;
}


KERNEL void G1_jacobian_elementwise_mul(GLOBAL G1Jacobian_t* arr_g1, GLOBAL Fr_t* arr_fr, GLOBAL G1Jacobian_t* arr_out, uint n)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;
    arr_out[gid] = G1Jacobian_mul(arr_g1[gid], arr_fr[gid]);
}

KERNEL void G1_jacobian_elementwise_mul_broadcast(GLOBAL G1Jacobian_t* arr_g1, GLOBAL Fr_t* arr_fr, GLOBAL G1Jacobian_t* arr_out, uint n, uint m)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= m * n) return;
    arr_out[gid] = G1Jacobian_mul(arr_g1[gid % n], arr_fr[gid]);
}

G1TensorJacobian G1TensorJacobian::operator*(const FrTensor& scalar_tensor) const {
    if (scalar_tensor.size % size != 0) throw std::runtime_error("Incompatible dimensions");
    uint m = scalar_tensor.size / size;
    G1TensorJacobian out(scalar_tensor.size);  // output size will be same as scalar_tensor
    G1_jacobian_elementwise_mul_broadcast<<<(scalar_tensor.size+G1NumThread-1)/G1NumThread,G1NumThread>>>(gpu_data, scalar_tensor.gpu_data, out.gpu_data, size, m);
    cudaDeviceSynchronize();
    return out;
}

G1TensorJacobian& G1TensorJacobian::operator*=(const FrTensor& scalar_tensor) {
    if (size != scalar_tensor.size) throw std::runtime_error("Incompatible dimensions 01");
    G1_jacobian_elementwise_mul<<<(size+G1NumThread-1)/G1NumThread,G1NumThread>>>(gpu_data, scalar_tensor.gpu_data, gpu_data, size);
    cudaDeviceSynchronize();
    return *this;
}

KERNEL void G1_me_step(GLOBAL G1Jacobian_t *arr_in, GLOBAL G1Jacobian_t *arr_out, Fr_t x, uint in_size, uint out_size)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= out_size) return;

    uint gid0 = 2 * gid;
    uint gid1 = 2 * gid + 1;
    // x is a canonical wire scalar; G1Jacobian_mul consumes the same wire.
    if (gid1 < in_size) arr_out[gid] = blstrs__g1__G1Affine_add(arr_in[gid0], G1Jacobian_mul(blstrs__g1__G1Affine_add(arr_in[gid1], G1Jacobian_minus(arr_in[gid0])), x));
    else if (gid0 < in_size) arr_out[gid] = blstrs__g1__G1Affine_add(arr_in[gid0], G1Jacobian_minus(G1Jacobian_mul(arr_in[gid0], x)));
    else arr_out[gid] = blstrs__g1__G1Affine_ZERO;
}

G1Jacobian_t G1_me(const G1TensorJacobian& t, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end)
{
    G1TensorJacobian t_new((t.size + 1) / 2);
    if (begin >= end) return t(0);
    G1_me_step<<<(t_new.size+G1NumThread-1)/G1NumThread,G1NumThread>>>(t.gpu_data, t_new.gpu_data, *begin, t.size, t_new.size);
    cudaDeviceSynchronize();
    return G1_me(t_new, begin + 1, end);
}

G1Jacobian_t G1TensorJacobian::operator()(const vector<Fr_t>& u) const
{
    uint log_dim = u.size();
    if (log_dim >= 32 || size == 0) throw std::runtime_error("Incompatible dimensions");
    const uint64_t domain = uint64_t(1) << log_dim;
    if (size <= domain / 2U || uint64_t(size) > domain) throw std::runtime_error("Incompatible dimensions");
    return G1_me(*this, u.begin(), u.end());
}

#endif
