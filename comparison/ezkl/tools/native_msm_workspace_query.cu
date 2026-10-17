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
#include <vector>

#include <curves/params/bn254.cuh>

namespace {
constexpr unsigned kScalarBits = 254;
constexpr unsigned kC = 16;
constexpr unsigned kChunkScalars = 1U << 20;
constexpr unsigned kWindowCount = (kScalarBits + kC - 1) / kC;
constexpr unsigned kM = kChunkScalars * kWindowCount;
constexpr unsigned kB = (kWindowCount << kC) - kWindowCount;
constexpr unsigned kHMax = kB;
constexpr unsigned kAverageBucketSize = kChunkScalars / (1U << kC);
constexpr unsigned kBucketThreshold = 10U * kAverageBucketSize;
constexpr unsigned kLMax = kM / (kBucketThreshold + 1U);
constexpr unsigned kLargeBucketThreadsMax =
    (kM + kAverageBucketSize - 1U) / kAverageBucketSize + kLMax;
static_assert(kWindowCount == 16 && kM == 16777216 && kB == 1048560, "ICICLE MSM profile drift");
static_assert(kAverageBucketSize == 16 && kBucketThreshold == 160 && kLMax == 104206,
              "large-bucket bound no longer matches the pinned ICICLE MSM implementation");
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
std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b) {
  if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a)
    throw std::runtime_error("workspace byte product overflows uint64");
  return a * b;
}
std::uint64_t array_bytes(std::uint64_t count, std::uint64_t element_size) {
  return checked_mul(count, element_size);
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
void print_memory_only(const std::string& session_id, const cudaDeviceProp& properties) {
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
  std::cout << "{\"schema\":\"ezkl.icicle.device_memory_sample.v1\""
            << ",\"session_id\":\"" << session_id << "\""
            << ",\"device_uuid\":\"" << uuid_string(properties.uuid) << "\""
            << ",\"free_memory_bytes\":" << static_cast<std::uint64_t>(free_bytes)
            << ",\"total_memory_bytes\":" << static_cast<std::uint64_t>(total_bytes)
            << ",\"integrated_device\":" << (properties.integrated ? "true" : "false") << "}\n";
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

std::uint64_t scratch_sum(const std::vector<Query>& queries) {
  std::uint64_t total = 0;
  for (const Query& query : queries) total = checked_add(total, query.scratch_bytes);
  return total;
}

void print_queries(const std::vector<Query>& queries) {
  std::cout << '[';
  for (size_t i = 0; i < queries.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << "{\"name\":\"" << queries[i].name << "\",\"num_items\":" << queries[i].num_items
              << ",\"scratch_bytes\":" << queries[i].scratch_bytes << '}';
  }
  std::cout << ']';
}

std::uint64_t device_workspace_upper_bound(std::uint64_t scalar_bytes,
                                           std::uint64_t affine_bytes,
                                           std::uint64_t projective_bytes,
                                           std::uint64_t uint_bytes,
                                           std::uint64_t scratch_bytes_sum) {
  std::uint64_t total = 0;
  const auto add = [&total](std::uint64_t bytes) { total = checked_add(total, bytes); };
  // This sums source-sized array maxima across MSM phases. Several entries
  // are not simultaneously live; summing them is intentionally conservative.
  add(array_bytes(kChunkScalars, scalar_bytes));                         // uploaded scalars
  add(array_bytes(kChunkScalars, affine_bytes));                         // uploaded points
  add(array_bytes(4ULL * kM, uint_bytes));                               // input and sorted indices
  add(array_bytes(kB + 1ULL, uint_bytes));                               // single bucket indices
  add(array_bytes(kB + 1ULL, uint_bytes));                               // bucket sizes
  add(array_bytes(kB + 1ULL, uint_bytes));                               // bucket offsets
  add(array_bytes(kB, uint_bytes));                                      // sorted bucket sizes
  add(array_bytes(kB, uint_bytes));                                      // sorted bucket offsets
  add(array_bytes(kB, uint_bytes));                                      // sorted single bucket indices
  add(4ULL * uint_bytes);                                                // scalar counters
  add(array_bytes(kB + kWindowCount, projective_bytes));                // main buckets
  add(array_bytes(kLMax + 1, uint_bytes));                              // large bucket prefix sums
  add(array_bytes(kLargeBucketThreadsMax, uint_bytes));                // large bucket indices
  add(array_bytes(kLargeBucketThreadsMax, projective_bytes));           // large bucket points

  // The iterative reduction allocates one source array, two source-sized
  // temporaries, and one target array per iteration. Compute the largest
  // live iteration from the implementation's same bit/window recurrence.
  std::uint64_t reduction_peak_elements = 0;
  unsigned source_bits = kC;
  unsigned source_windows = kWindowCount;
  std::uint64_t source_count = kB + kWindowCount;
  for (;;) {
    const unsigned target_bits = (source_bits + 1U) >> 1U;
    const unsigned target_windows = source_windows << 1U;
    const std::uint64_t target_count = static_cast<std::uint64_t>(target_windows) << target_bits;
    std::uint64_t live = checked_add(checked_mul(3, source_count), target_count);
    if (target_bits == 1U) {
      const std::uint64_t final_results =
          (static_cast<std::uint64_t>(kC) * source_windows < kScalarBits)
              ? static_cast<std::uint64_t>(kC) * source_windows
              : kScalarBits;
      live = checked_add(live, final_results);
    }
    if (live > reduction_peak_elements) reduction_peak_elements = live;
    if (target_bits == 1U) break;
    source_bits = target_bits;
    source_windows = target_windows;
    source_count = target_count;
  }
  add(array_bytes(reduction_peak_elements, projective_bytes));
  add(array_bytes(kWindowCount + (kScalarBits + 1ULL), projective_bytes));  // reduction/final result buffers
  add(scratch_bytes_sum);                                                // sum of all queried CUB temporaries
  return total;
}
}

int main(int argc, char** argv) {
  try {
    bool memory_only = false;
    std::string session_id;
    std::string source_sha256;
    std::string manifest_sha256;
    for (int i = 1; i < argc; ++i) {
      const std::string arg(argv[i]);
      if (arg == "--memory-only") {
        memory_only = true;
      } else if (arg == "--session-id" && i + 1 < argc) {
        session_id = argv[++i];
      } else if (arg == "--icicle-source-tree-sha256" && i + 1 < argc) {
        source_sha256 = argv[++i];
      } else if (arg == "--native-backend-manifest-sha256" && i + 1 < argc) {
        manifest_sha256 = argv[++i];
      } else {
        throw std::runtime_error("unknown or incomplete argument: " + arg);
      }
    }
    if (session_id.size() != 32) throw std::runtime_error("session id must be 32 lowercase hex characters");
    for (char c : session_id)
      if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) throw std::runtime_error("invalid session id");
    if (!memory_only && (argc != 7 || !valid_sha256(source_sha256) || !valid_sha256(manifest_sha256))) {
      throw std::runtime_error("usage: native_msm_workspace_query --session-id HEX --icicle-source-tree-sha256 HEX --native-backend-manifest-sha256 HEX");
    }
    if (memory_only && (argc != 4 || !source_sha256.empty() || !manifest_sha256.empty())) {
      throw std::runtime_error("usage: native_msm_workspace_query --memory-only --session-id HEX");
    }
    if (!memory_only && (!valid_sha256(source_sha256) || !valid_sha256(manifest_sha256)))
      throw std::runtime_error("source and manifest identities must be lowercase SHA-256 hex");
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
    if (memory_only) {
      print_memory_only(session_id, properties);
      return std::cout.good() ? 0 : 1;
    }
    int runtime_version = 0;
    int driver_version = 0;
    check_cuda(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
    check_cuda(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
    const cudaStream_t stream = nullptr;
    const std::vector<Query> queries = {
        query_sort_pairs("scalar_index_sort_pairs", kM, stream),
        query_rle("sorted_bucket_run_length_encode", kM, stream),
        query_exclusive_scan("bucket_offsets_exclusive_scan", kB + 1, stream),
        query_sort_pairs_descending("bucket_size_offset_sort_pairs_descending", kHMax, stream),
        query_sort_pairs_descending("bucket_size_index_sort_pairs_descending", kHMax, stream),
        query_inclusive_scan("large_bucket_prefix_inclusive_scan", kLMax, stream),
    };
    const std::uint64_t total_scratch_bytes = scratch_sum(queries);
    const auto scalar_size = static_cast<std::uint64_t>(sizeof(bn254::scalar_t));
    const auto affine_size = static_cast<std::uint64_t>(sizeof(bn254::affine_t));
    const auto projective_size = static_cast<std::uint64_t>(sizeof(bn254::projective_t));
    const auto uint_size = static_cast<std::uint64_t>(sizeof(unsigned));
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    const auto device_peak = device_workspace_upper_bound(
        scalar_size, affine_size, projective_size, uint_size, total_scratch_bytes);
#if defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__) && defined(__CUDACC_VER_BUILD__)
    const std::string toolkit_version = std::to_string(__CUDACC_VER_MAJOR__) + "." +
        std::to_string(__CUDACC_VER_MINOR__) + "." + std::to_string(__CUDACC_VER_BUILD__);
#else
#error CUDA compiler version macros are required for workspace identity
#endif
    std::cout << "{\"schema\":\"ezkl.icicle.cub_msm_workspace_query.v2\""
              << ",\"session_id\":\"" << session_id << "\""
              << ",\"icicle_source_tree_sha256\":\"" << source_sha256 << "\""
              << ",\"native_backend_manifest_sha256\":\"" << manifest_sha256 << "\""
              << ",\"device_uuid\":\"" << uuid_string(properties.uuid) << "\""
              << ",\"free_memory_bytes\":" << static_cast<std::uint64_t>(free_bytes)
              << ",\"total_memory_bytes\":" << static_cast<std::uint64_t>(total_bytes)
              << ",\"integrated_device\":" << (properties.integrated ? "true" : "false")
              << ",\"scalar_bits\":" << kScalarBits << ",\"c\":" << kC
              << ",\"chunk_scalars\":" << kChunkScalars << ",\"window_count\":" << kWindowCount
              << ",\"M\":" << kM << ",\"B\":" << kB << ",\"H_max\":" << kHMax << ",\"L_max\":" << kLMax
              << ",\"cuda_runtime_version\":" << runtime_version << ",\"cuda_driver_version\":" << driver_version
              << ",\"cuda_toolkit_version\":\"" << toolkit_version << "\""
              << ",\"cub_version\":" << CUB_VERSION
              << ",\"device_compute_capability\":\"" << properties.major << "." << properties.minor << "\""
              << ",\"native_type_sizes\":{\"scalar_bytes\":" << scalar_size
              << ",\"affine_bytes\":" << affine_size << ",\"projective_bytes\":" << projective_size
              << ",\"unsigned_bytes\":" << uint_size << "}"
              << ",\"device_workspace_peak_bytes\":" << device_peak
              << ",\"queries\":";
    print_queries(queries);
    std::cout << ",\"scratch_bytes_sum\":" << total_scratch_bytes << "}\n";
    return std::cout.good() ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "native MSM CUB workspace query failed: " << error.what() << '\n';
    return 1;
  }
}
