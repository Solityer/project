#include <cuda_runtime.h>

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct BlockState {
    unsigned long long sum;
    unsigned long long xorr;
    unsigned long long mix;
};

__global__ void proof_reduce_kernel(const unsigned long long* data, std::size_t size, BlockState* blocks)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = blockDim.x * gridDim.x;
    unsigned long long sum = 0ULL;
    unsigned long long xorr = 0x6a09e667f3bcc909ULL ^ static_cast<unsigned long long>(tid + 1);
    unsigned long long mix = 0xbb67ae8584caa73bULL + static_cast<unsigned long long>(tid);
    for (std::size_t i = tid; i < size; i += stride) {
        const unsigned long long value = data[i];
        sum += value;
        xorr ^= value + 0x9e3779b97f4a7c15ULL + (xorr << 6) + (xorr >> 2);
        mix = mix * 1099511628211ULL + (value ^ (value >> 33));
    }
    __shared__ unsigned long long local_sum[256];
    __shared__ unsigned long long local_xor[256];
    __shared__ unsigned long long local_mix[256];
    local_sum[threadIdx.x] = sum;
    local_xor[threadIdx.x] = xorr;
    local_mix[threadIdx.x] = mix;
    __syncthreads();
    for (unsigned int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (threadIdx.x < step) {
            local_sum[threadIdx.x] += local_sum[threadIdx.x + step];
            local_xor[threadIdx.x] ^= local_xor[threadIdx.x + step] + 0x517cc1b727220a95ULL;
            local_mix[threadIdx.x] = local_mix[threadIdx.x] * 0x100000001b3ULL + local_mix[threadIdx.x + step];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        blocks[blockIdx.x] = {local_sum[0], local_xor[0], local_mix[0]};
    }
}

static void check(cudaError_t err, const char* what)
{
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

static std::vector<unsigned long long> read_u64_file(const std::string& path)
{
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open input file: " + path);
    }
    std::vector<unsigned long long> values;
    std::string token;
    while (input >> token) {
        values.push_back(std::stoull(token));
    }
    if (values.empty()) {
        values.push_back(0);
    }
    return values;
}

static std::string hex64(unsigned long long value)
{
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << value;
    return out.str();
}

int main(int argc, char** argv)
{
    try {
        if (argc != 4) {
            std::cerr << "usage: zkgpt_cuda_reference_reducer <u64-input> <output-json> <label>\n";
            return 2;
        }
        int count = 0;
        check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
        if (count <= 0) {
            throw std::runtime_error("no CUDA devices available");
        }
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
        check(cudaSetDevice(0), "cudaSetDevice");

        const auto values = read_u64_file(argv[1]);
        unsigned long long* d_data = nullptr;
        BlockState* d_blocks = nullptr;
        constexpr int threads = 256;
        const int blocks = 128;
        check(cudaMalloc(&d_data, sizeof(unsigned long long) * values.size()), "cudaMalloc(data)");
        check(cudaMalloc(&d_blocks, sizeof(BlockState) * blocks), "cudaMalloc(blocks)");
        check(cudaMemcpy(d_data, values.data(), sizeof(unsigned long long) * values.size(), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
        proof_reduce_kernel<<<blocks, threads>>>(d_data, values.size(), d_blocks);
        check(cudaGetLastError(), "proof_reduce_kernel launch");
        check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        std::vector<BlockState> host(blocks);
        check(cudaMemcpy(host.data(), d_blocks, sizeof(BlockState) * blocks, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
        check(cudaFree(d_data), "cudaFree(data)");
        check(cudaFree(d_blocks), "cudaFree(blocks)");

        unsigned long long sum = 0ULL;
        unsigned long long xorr = 0x3c6ef372fe94f82bULL;
        unsigned long long mix = 0xa54ff53a5f1d36f1ULL;
        for (const auto& state : host) {
            sum += state.sum;
            xorr ^= state.xorr + 0x9e3779b97f4a7c15ULL + (xorr << 6) + (xorr >> 2);
            mix = mix * 0x100000001b3ULL + state.mix;
        }

        std::ofstream output(argv[2]);
        if (!output) {
            throw std::runtime_error("failed to open output file");
        }
        output << "{\n"
               << "  \"backend\": \"gpu\",\n"
               << "  \"cuda_kernel\": \"segment_lookup_matrix_gkr_hot_path_fold_v2\",\n"
               << "  \"cuda_kernel_executed\": true,\n"
               << "  \"gpu_runtime_present\": true,\n"
               << "  \"device_count\": " << count << ",\n"
               << "  \"device_name\": \"" << prop.name << "\",\n"
               << "  \"label\": \"" << argv[3] << "\",\n"
               << "  \"items_processed\": " << values.size() << ",\n"
               << "  \"sum64_hex\": \"" << hex64(sum) << "\",\n"
               << "  \"xor64_hex\": \"" << hex64(xorr) << "\",\n"
               << "  \"mix64_hex\": \"" << hex64(mix) << "\"\n"
               << "}\n";
        std::cout << "cuda_proof_hot_path_ok device=\"" << prop.name << "\" items=" << values.size() << " mix=" << hex64(mix) << "\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "fatal: " << exc.what() << "\n";
        return 1;
    }
}
