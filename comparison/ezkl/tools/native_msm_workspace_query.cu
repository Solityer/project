#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/version.cuh>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
constexpr unsigned kScalarBits = 254;
constexpr unsigned kC = 16;
constexpr unsigned kChunkScalars = 1U << 20;
constexpr unsigned kWindowCount = (kScalarBits + kC - 1) / kC;
constexpr unsigned kM = kChunkScalars * kWindowCount;
constexpr unsigned kB = (kWindowCount << kC) - kWindowCount;
constexpr unsigned kHMax = kB;
constexpr unsigned kLMax = 104206;
static_assert(kWindowCount == 16 && kM == 16777216 && kB == 1048560, "ICICLE MSM profile drift");
static_assert(kLMax <= kHMax && kHMax <= kB, "invalid conservative ICICLE MSM query bounds");
struct Query { const char* name; std::uint64_t num_items; std::uint64_t scratch_bytes; };
void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
void check_cub(cudaError_t status, const char* operation) { check_cuda(status, operation); }
std::uint64_t checked_size(size_t bytes, const char* name) {
  if (bytes > std::numeric_limits<std::uint64_t>::max()) throw std::runtime_error(std::string(name) + " scratch size overflows uint64");
  return static_cast<std::uint64_t>(bytes);
}
std::uint64_t checked_add(std::uint64_t a, std::uint64_t b) {
  if (b > std::numeric_limits<std::uint64_t>::max() - a) throw std::runtime_error("scratch byte sum overflows uint64");
  return a + b;
}
bool valid_sha256(const std::string& value) {
  if (value.size() != 64) return false;
  for (char c : value) if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
  return true;
}
std::string uuid_string(const cudaUUID_t& uuid) {
  const auto* bytes = reinterpret_cast<const unsigned char*>(uuid.bytes);
  std::ostringstream result;
  result << "GPU-" << std::hex << std::setfill('0');
  for (unsigned i = 0; i < 16; ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10) result << '-';
    result << std::setw(2) << static_cast<unsigned>(bytes[i]);
  }
  return result.str();
}
Query query_sort_pairs(const char* name, unsigned count, cudaStream_t stream) {
  size_t bytes = 0;
  check_cub(cub::DeviceRadixSort::SortPairs(static_cast<void*>(nullptr), bytes,
      static_cast<const unsigned*>(nullptr), static_cast<unsigned*>(nullptr),
      static_cast<const unsigned*>(nullptr), static_cast<unsigned*>(nullptr), count, 0,
      sizeof(unsigned) * 8, stream), name);
  return {name, count, checked_size(bytes, name)};
}
Query query_sort_pairs_descending(const char* name, unsigned count, cudaStream_t stream) {
  size_t bytes = 0;
  check_cub(cub::DeviceRadixSort::SortPairsDescending(static_cast<void*>(nullptr), bytes,
      static_cast<const unsigned*>(nullptr), static_cast<unsigned*>(nullptr),
      static_cast<const unsigned*>(nullptr), static_cast<unsigned*>(nullptr), count, 0,
      sizeof(unsigned) * 8, stream), name);
  return {name, count, checked_size(bytes, name)};
}
Query query_rle(const char* name, unsigned count, cudaStream_t stream) {
  size_t bytes = 0;
  check_cub(cub::DeviceRunLengthEncode::Encode(static_cast<void*>(nullptr), bytes,
      static_cast<const unsigned*>(nullptr), static_cast<unsigned*>(nullptr),
      static_cast<unsigned*>(nullptr), static_cast<unsigned*>(nullptr), count, stream), name);
  return {name, count, checked_size(bytes, name)};
}
Query query_exclusive_scan(const char* name, unsigned count, cudaStream_t stream) {
  size_t bytes = 0;
  check_cub(cub::DeviceScan::ExclusiveSum(static_cast<void*>(nullptr), bytes,
      static_cast<const unsigned*>(nullptr), static_cast<unsigned*>(nullptr), count, stream), name);
  return {name, count, checked_size(bytes, name)};
}
Query query_inclusive_scan(const char* name, unsigned count, cudaStream_t stream) {
  size_t bytes = 0;
  check_cub(cub::DeviceScan::InclusiveSum(static_cast<void*>(nullptr), bytes,
      static_cast<const unsigned*>(nullptr), static_cast<unsigned*>(nullptr), count, stream), name);
  return {name, count, checked_size(bytes, name)};
}
}

int main(int argc, char** argv) {
  try {
    if (argc != 5 || std::string(argv[1]) != "--icicle-source-tree-sha256" ||
        std::string(argv[3]) != "--native-backend-manifest-sha256") {
      throw std::runtime_error("usage: native_msm_workspace_query --icicle-source-tree-sha256 HEX --native-backend-manifest-sha256 HEX");
    }
    const std::string source_sha256 = argv[2];
    const std::string manifest_sha256 = argv[4];
    if (!valid_sha256(source_sha256) || !valid_sha256(manifest_sha256)) throw std::runtime_error("source and manifest identities must be lowercase SHA-256 hex");
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < 1) throw std::runtime_error("no CUDA device is visible");
    check_cuda(cudaSetDevice(0), "cudaSetDevice(0)");
    int selected_device = -1;
    check_cuda(cudaGetDevice(&selected_device), "cudaGetDevice");
    if (selected_device != 0) throw std::runtime_error("selected CUDA logical device is not zero");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, selected_device), "cudaGetDeviceProperties");
    if (properties.totalGlobalMem == 0) throw std::runtime_error("CUDA device reported zero total memory");
    int runtime_version = 0;
    int driver_version = 0;
    check_cuda(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
    check_cuda(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
    const cudaStream_t stream = nullptr;
    const Query queries[] = {
      query_sort_pairs("scalar_index_sort_pairs", kM, stream),
      query_rle("sorted_bucket_run_length_encode", kM, stream),
      query_exclusive_scan("bucket_offsets_exclusive_scan", kB + 1, stream),
      query_sort_pairs_descending("bucket_size_offset_sort_pairs_descending", kHMax, stream),
      query_sort_pairs_descending("bucket_size_index_sort_pairs_descending", kHMax, stream),
      query_inclusive_scan("large_bucket_prefix_inclusive_scan", kLMax, stream),
    };
    std::uint64_t total_scratch_bytes = 0;
    for (const Query& query : queries) total_scratch_bytes = checked_add(total_scratch_bytes, query.scratch_bytes);
#if defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__) && defined(__CUDACC_VER_BUILD__)
    const std::string toolkit_version = std::to_string(__CUDACC_VER_MAJOR__) + "." +
        std::to_string(__CUDACC_VER_MINOR__) + "." + std::to_string(__CUDACC_VER_BUILD__);
#else
#error CUDA compiler version macros are required for workspace identity
#endif
    std::cout << "{\"schema\":\"ezkl.icicle.cub_msm_workspace_query.v1\""
              << ",\"icicle_source_tree_sha256\":\"" << source_sha256 << "\""
              << ",\"native_backend_manifest_sha256\":\"" << manifest_sha256 << "\""
              << ",\"device_uuid\":\"" << uuid_string(properties.uuid) << "\""
              << ",\"total_memory_bytes\":" << static_cast<std::uint64_t>(properties.totalGlobalMem)
              << ",\"scalar_bits\":" << kScalarBits << ",\"c\":" << kC
              << ",\"chunk_scalars\":" << kChunkScalars << ",\"window_count\":" << kWindowCount
              << ",\"M\":" << kM << ",\"B\":" << kB << ",\"H_max\":" << kHMax << ",\"L_max\":" << kLMax
              << ",\"cuda_runtime_version\":" << runtime_version << ",\"cuda_driver_version\":" << driver_version
              << ",\"cuda_toolkit_version\":\"" << toolkit_version << "\""
              << ",\"cub_version\":" << CUB_VERSION
              << ",\"device_compute_capability\":\"" << properties.major << "." << properties.minor << "\""
              << ",\"queries\":[";
    for (size_t i = 0; i < sizeof(queries) / sizeof(queries[0]); ++i) {
      if (i) std::cout << ',';
      std::cout << "{\"name\":\"" << queries[i].name << "\",\"num_items\":" << queries[i].num_items
                << ",\"scratch_bytes\":" << queries[i].scratch_bytes << '}';
    }
    std::cout << "],\"scratch_bytes_sum\":" << total_scratch_bytes << "}\n";
    return std::cout.good() ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "native MSM CUB workspace query failed: " << error.what() << '\n';
    return 1;
  }
}
