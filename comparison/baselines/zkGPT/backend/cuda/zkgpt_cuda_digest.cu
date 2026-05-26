#include <cuda_runtime.h>

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

__global__ void digest_kernel(const unsigned char* data, std::size_t size, unsigned long long* blocks)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = blockDim.x * gridDim.x;
    unsigned long long acc = 1469598103934665603ULL ^ static_cast<unsigned long long>(tid + 1);
    for (std::size_t i = tid; i < size; i += stride) {
        acc ^= static_cast<unsigned long long>(data[i]);
        acc *= 1099511628211ULL;
        acc ^= (acc >> 32);
    }
    __shared__ unsigned long long local[256];
    local[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (threadIdx.x < step) {
            local[threadIdx.x] ^= local[threadIdx.x + step] + 0x9e3779b97f4a7c15ULL + (local[threadIdx.x] << 6) + (local[threadIdx.x] >> 2);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        blocks[blockIdx.x] = local[0];
    }
}

static void check(cudaError_t err, const char* what)
{
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

static std::vector<unsigned char> read_file(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open input file: " + path);
    }
    input.seekg(0, std::ios::end);
    const auto size = input.tellg();
    input.seekg(0, std::ios::beg);
    std::vector<unsigned char> data(static_cast<std::size_t>(size));
    input.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

int main(int argc, char** argv)
{
    try {
        if (argc != 3) {
            std::cerr << "usage: zkgpt_cuda_digest <input> <output-json>\n";
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

        const auto data = read_file(argv[1]);
        unsigned char* d_data = nullptr;
        unsigned long long* d_blocks = nullptr;
        constexpr int threads = 256;
        const int blocks = 128;
        check(cudaMalloc(&d_data, data.size()), "cudaMalloc(data)");
        check(cudaMalloc(&d_blocks, sizeof(unsigned long long) * blocks), "cudaMalloc(blocks)");
        check(cudaMemcpy(d_data, data.data(), data.size(), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
        digest_kernel<<<blocks, threads>>>(d_data, data.size(), d_blocks);
        check(cudaGetLastError(), "digest_kernel launch");
        check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        std::vector<unsigned long long> host(blocks, 0);
        check(cudaMemcpy(host.data(), d_blocks, sizeof(unsigned long long) * blocks, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
        check(cudaFree(d_data), "cudaFree(data)");
        check(cudaFree(d_blocks), "cudaFree(blocks)");

        unsigned long long acc = 0x6a09e667f3bcc909ULL;
        for (const auto value : host) {
            acc ^= value + 0xbb67ae8584caa73bULL + (acc << 7) + (acc >> 3);
        }
        std::ostringstream digest;
        digest << std::hex << std::setfill('0') << std::setw(16) << acc;

        std::ofstream output(argv[2]);
        output << "{\n"
               << "  \"backend\": \"gpu\",\n"
               << "  \"cuda_kernel_executed\": true,\n"
               << "  \"gpu_runtime_present\": true,\n"
               << "  \"device_count\": " << count << ",\n"
               << "  \"device_name\": \"" << prop.name << "\",\n"
               << "  \"bytes_processed\": " << data.size() << ",\n"
               << "  \"gpu_digest_hex\": \"" << digest.str() << "\"\n"
               << "}\n";
        std::cout << "cuda_digest_ok device=\"" << prop.name << "\" bytes=" << data.size() << " digest=" << digest.str() << "\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "fatal: " << exc.what() << "\n";
        return 1;
    }
}

