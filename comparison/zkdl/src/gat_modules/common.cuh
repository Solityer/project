#pragma once

// 通用计时、路径、CUDA buffer 与格式化工具。

using Clock = std::chrono::steady_clock;

struct ScopedTimer {
    Clock::time_point start = Clock::now();
    // 返回从构造到当前的毫秒数；只用于 timing 字段，不改变 proof 语义。
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }
};

// 检查 CUDA API 返回值；失败时抛错阻止写入不完整 official artifact。
void cuda_check(cudaError_t status, const std::string& where) {
    if (status != cudaSuccess) {
        throw std::runtime_error(where + ": " + cudaGetErrorString(status));
    }
}

// 同步 CUDA kernel 并检查错误；用于 GPU path 与 CPU/GPU 隔离后的 GPU trace。
void cuda_sync(const std::string& where) {
    cuda_check(cudaDeviceSynchronize(), where);
    cuda_check(cudaGetLastError(), where);
}

template <typename T>
// RAII 设备缓冲区；输入为元素数量，析构释放 GPU 内存。
struct DeviceBuffer {
    T* ptr = nullptr;
    std::size_t n = 0;

    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) : n(count) {
        if (n != 0) {
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * n), "cudaMalloc");
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr(other.ptr), n(other.n) {
        other.ptr = nullptr;
        other.n = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr != nullptr) {
                cudaFree(ptr);
            }
            ptr = other.ptr;
            n = other.n;
            other.ptr = nullptr;
            other.n = 0;
        }
        return *this;
    }
    ~DeviceBuffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }
};

template <typename T>
// 将 host vector 拷贝到 device；输出 DeviceBuffer，副作用仅为 CUDA 内存分配。
DeviceBuffer<T> to_device(const std::vector<T>& host) {
    DeviceBuffer<T> out(host.size());
    if (!host.empty()) {
        cuda_check(cudaMemcpy(out.ptr, host.data(), sizeof(T) * host.size(), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    }
    return out;
}

template <typename T>
// 将 device buffer 拷回 host vector；用于 GPU forward trace 和 proof witness 编码。
std::vector<T> to_host(const DeviceBuffer<T>& device) {
    std::vector<T> out(device.n);
    if (device.n != 0) {
        cuda_check(cudaMemcpy(out.data(), device.ptr, sizeof(T) * device.n, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    }
    return out;
}

// 比较两个 Fr 原始编码是否完全一致；用于 verifier/selftest 精确检查。
bool fr_equal_host(const Fr_t& lhs, const Fr_t& rhs) {
    return std::memcmp(&lhs, &rhs, sizeof(Fr_t)) == 0;
}

// CUDA kernel 计算 1-u；输入/输出均为 Montgomery Fr。
__global__ void gat_fr_mont_one_minus_kernel(Fr_t u, Fr_t* out) {
    *out = blstrs__scalar__Scalar_sub(blstrs__scalar__Scalar_ONE, u);
}

// host 包装 1-u kernel；输出参与 MLE 折叠，不写 official artifact。
Fr_t fr_mont_one_minus_host(Fr_t u) {
    Fr_t host;
    Fr_t* device = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&device), sizeof(Fr_t)), "cudaMalloc fr_mont_one_minus");
    gat_fr_mont_one_minus_kernel<<<1,1>>>(u, device);
    cuda_sync("gat_fr_mont_one_minus_kernel");
    cuda_check(cudaMemcpy(&host, device, sizeof(Fr_t), cudaMemcpyDeviceToHost), "cudaMemcpy fr_mont_one_minus");
    cudaFree(device);
    return host;
}

// 将 Montgomery Fr 转回 raw Fr；用于 JSON/摘要前的规范值处理。
Fr_t fr_unmont_host(Fr_t value) {
    FrTensor tensor(1, &value);
    tensor.unmont();
    return tensor(0);
}

// 在 host 上相加两个 G1 Jacobian 点；用于承诺聚合检查。
G1Jacobian_t g1_add_host(G1Jacobian_t lhs, G1Jacobian_t rhs) {
    G1TensorJacobian a(1, &lhs);
    G1TensorJacobian b(1, &rhs);
    auto sum = a + b;
    return sum(0);
}

// 去除字符串两端空白；用于读取配置和 cache 文本。
std::string trim(std::string value) {
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch) { return !std::isspace(ch); }));
    value.erase(std::find_if(value.rbegin(), value.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), value.end());
    return value;
}

// 读取 UTF-8/ASCII 文本文件；输入路径必须来自 zkDL/主项目只读数据。
std::string read_text(const fs::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open " + path.string());
    }
    std::ostringstream stream;
    stream << input.rdbuf();
    return stream.str();
}

// 创建输出目录；full-chain runner 只对 zkDL output_root 内调用。
void ensure_dir(const fs::path& path) {
    fs::create_directories(path);
}

fs::path device_run_dir(const fs::path& output_root, const std::string& dataset, const std::string& device = "gpu") {
    return output_root / "runs" / dataset / device / "warm";
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (const char ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << ch; break;
        }
    }
    return out.str();
}

std::string format_ms(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << value;
    return out.str();
}

std::size_t next_power_of_two(std::size_t value) {
    std::size_t out = 1;
    while (out < value) {
        out <<= 1U;
    }
    return out;
}

std::uint64_t edge_key(std::size_t src, std::size_t dst, std::size_t graph_id) {
    return (static_cast<std::uint64_t>(graph_id) << 48U)
        ^ (static_cast<std::uint64_t>(src) << 24U)
        ^ static_cast<std::uint64_t>(dst);
}
