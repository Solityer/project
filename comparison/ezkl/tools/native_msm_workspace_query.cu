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
#include <utility>
#include <vector>

#include <curves/params/bn254.cuh>

namespace {
constexpr unsigned kScalarBits = 254, kC = 16, kChunkScalars = 1U << 20;
constexpr unsigned kTotalWindowCount = (kScalarBits + kC - 1U) / kC, kM = kChunkScalars * kTotalWindowCount;
constexpr unsigned kLargeBucketFactor = 10, kOrdinaryPrecomputeFactor = 1, kPreparedPrecomputeFactor = 4;
struct MsmPathProfile { unsigned precompute_factor, window_count, bucket_count, average_bucket_size, large_bucket_threshold, large_bucket_count_max, large_bucket_threads_max; };
constexpr MsmPathProfile make_path_profile(unsigned precompute_factor) {
  const unsigned window_count = (kTotalWindowCount - 1U) / precompute_factor + 1U;
  const unsigned bucket_count = (window_count << kC) - window_count;
  const unsigned average_bucket_size = (kChunkScalars / (1U << kC)) * precompute_factor;
  const unsigned large_bucket_threshold = kLargeBucketFactor * average_bucket_size;
  const unsigned large_bucket_count_max = kM / (large_bucket_threshold + 1U);
  const unsigned large_bucket_threads_max = (kM + average_bucket_size - 1U) / average_bucket_size + large_bucket_count_max;
  return {precompute_factor, window_count, bucket_count, average_bucket_size, large_bucket_threshold, large_bucket_count_max, large_bucket_threads_max};
}
constexpr MsmPathProfile kOrdinaryPath = make_path_profile(kOrdinaryPrecomputeFactor);
constexpr MsmPathProfile kPreparedPath = make_path_profile(kPreparedPrecomputeFactor);
static_assert(kTotalWindowCount == 16 && kM == 16777216, "ICICLE MSM profile drift");
static_assert(kOrdinaryPath.window_count == 16 && kOrdinaryPath.bucket_count == 1048560 && kOrdinaryPath.average_bucket_size == 16 && kOrdinaryPath.large_bucket_threshold == 160 && kOrdinaryPath.large_bucket_count_max == 104206 && kOrdinaryPath.large_bucket_threads_max == 1152782, "factor-1 ICICLE MSM bounds drift");
static_assert(kPreparedPath.window_count == 4 && kPreparedPath.bucket_count == 262140 && kPreparedPath.average_bucket_size == 64 && kPreparedPath.large_bucket_threshold == 640 && kPreparedPath.large_bucket_count_max == 26173 && kPreparedPath.large_bucket_threads_max == 288317, "factor-4 ICICLE MSM bounds drift");
static_assert(kOrdinaryPath.large_bucket_count_max <= kOrdinaryPath.bucket_count && kPreparedPath.large_bucket_count_max <= kPreparedPath.bucket_count, "invalid conservative ICICLE MSM query bounds");
struct Query { const char* name; std::uint64_t num_items; std::uint64_t scratch_bytes; };
struct PathQuery { MsmPathProfile profile; std::vector<Query> queries; std::uint64_t scratch_bytes; std::uint64_t device_bytes; std::uint64_t point_upload_bytes; std::uint64_t precompute_device_temporary_bytes; };
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
std::vector<Query> query_path_operations(const MsmPathProfile& profile, cudaStream_t stream) {
  return {query_sort_pairs("scalar_index_sort_pairs", kM, stream), query_rle("sorted_bucket_run_length_encode", kM, stream), query_exclusive_scan("bucket_offsets_exclusive_scan", profile.bucket_count + 1U, stream), query_sort_pairs_descending("bucket_size_offset_sort_pairs_descending", profile.bucket_count, stream), query_sort_pairs_descending("bucket_size_index_sort_pairs_descending", profile.bucket_count, stream), query_inclusive_scan("large_bucket_prefix_inclusive_scan", profile.large_bucket_count_max, stream)};
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

std::uint64_t device_workspace_upper_bound(const MsmPathProfile& profile, bool points_are_host_resident, std::uint64_t scalar_bytes, std::uint64_t affine_bytes, std::uint64_t projective_bytes, std::uint64_t uint_bytes, std::uint64_t scratch_bytes_sum) {
  std::uint64_t total = 0;
  const auto add = [&total](std::uint64_t bytes) { total = checked_add(total, bytes); };
  // This sums source-sized array maxima across MSM phases. Several entries
  // are not simultaneously live; summing them is intentionally conservative.
  add(array_bytes(kChunkScalars, scalar_bytes));
  if (points_are_host_resident) add(array_bytes(kChunkScalars, affine_bytes));
  add(array_bytes(4ULL * kM, uint_bytes));
  add(array_bytes(profile.bucket_count + 1ULL, uint_bytes));
  add(array_bytes(profile.bucket_count + 1ULL, uint_bytes));
  add(array_bytes(profile.bucket_count + 1ULL, uint_bytes));
  add(array_bytes(profile.bucket_count, uint_bytes));
  add(array_bytes(profile.bucket_count, uint_bytes));
  add(array_bytes(profile.bucket_count, uint_bytes));
  add(4ULL * uint_bytes);
  add(array_bytes(profile.bucket_count + profile.window_count, projective_bytes));
  add(array_bytes(profile.large_bucket_count_max + 1ULL, uint_bytes));
  add(array_bytes(profile.large_bucket_threads_max, uint_bytes));
  add(array_bytes(profile.large_bucket_threads_max, projective_bytes));

  // The iterative reduction allocates one source array, two source-sized
  // temporaries, and one target array per iteration. Compute the largest
  // live iteration from the implementation's same bit/window recurrence.
  std::uint64_t reduction_peak_elements = 0;
  unsigned source_bits = kC;
  unsigned source_windows = profile.window_count;
  std::uint64_t source_count = profile.bucket_count + profile.window_count;
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
  add(array_bytes(profile.window_count + (kScalarBits + 1ULL), projective_bytes));
  add(scratch_bytes_sum);
  return total;
}
PathQuery build_path_query(const MsmPathProfile& profile, bool points_are_host_resident, std::uint64_t scalar_bytes, std::uint64_t affine_bytes, std::uint64_t projective_bytes, std::uint64_t uint_bytes, cudaStream_t stream) {
  std::vector<Query> queries = query_path_operations(profile, stream);
  const std::uint64_t scratch_bytes = scratch_sum(queries);
  const std::uint64_t point_upload_bytes = points_are_host_resident ? array_bytes(kChunkScalars, affine_bytes) : 0;
  const std::uint64_t device_bytes = device_workspace_upper_bound(profile, points_are_host_resident, scalar_bytes, affine_bytes, projective_bytes, uint_bytes, scratch_bytes);
  return {profile, std::move(queries), scratch_bytes, device_bytes, point_upload_bytes, 0};
}
void print_path_query(const char* name, const PathQuery& path) {
  std::cout << "\"" << name << "\":{\"precompute_factor\":" << path.profile.precompute_factor
            << ",\"window_count_per_msm\":" << path.profile.window_count << ",\"B\":" << path.profile.bucket_count
            << ",\"H_max\":" << path.profile.bucket_count << ",\"average_bucket_size\":" << path.profile.average_bucket_size
            << ",\"large_bucket_threshold\":" << path.profile.large_bucket_threshold
            << ",\"L_max\":" << path.profile.large_bucket_count_max
            << ",\"large_bucket_threads_max\":" << path.profile.large_bucket_threads_max
            << ",\"point_upload_bytes\":" << path.point_upload_bytes
            << ",\"precompute_device_temporary_bytes\":" << path.precompute_device_temporary_bytes
            << ",\"device_workspace_peak_bytes\":" << path.device_bytes << ",\"queries\":";
  print_queries(path.queries);
  std::cout << ",\"scratch_bytes_sum\":" << path.scratch_bytes << '}';
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
    const auto scalar_size = static_cast<std::uint64_t>(sizeof(bn254::scalar_t));
    const auto affine_size = static_cast<std::uint64_t>(sizeof(bn254::affine_t));
    const auto projective_size = static_cast<std::uint64_t>(sizeof(bn254::projective_t));
    const auto uint_size = static_cast<std::uint64_t>(sizeof(unsigned));
    const PathQuery ordinary_path = build_path_query(kOrdinaryPath, true, scalar_size, affine_size, projective_size, uint_size, stream);
    const PathQuery prepared_path = build_path_query(kPreparedPath, false, scalar_size, affine_size, projective_size, uint_size, stream);
    const std::uint64_t device_peak = ordinary_path.device_bytes > prepared_path.device_bytes ? ordinary_path.device_bytes : prepared_path.device_bytes;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
#if defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__) && defined(__CUDACC_VER_BUILD__)
    const std::string toolkit_version = std::to_string(__CUDACC_VER_MAJOR__) + "." +
        std::to_string(__CUDACC_VER_MINOR__) + "." + std::to_string(__CUDACC_VER_BUILD__);
#else
#error CUDA compiler version macros are required for workspace identity
#endif
    std::cout << "{\"schema\":\"ezkl.icicle.cub_msm_workspace_query.v3\""
              << ",\"session_id\":\"" << session_id << "\""
              << ",\"icicle_source_tree_sha256\":\"" << source_sha256 << "\""
              << ",\"native_backend_manifest_sha256\":\"" << manifest_sha256 << "\""
              << ",\"device_uuid\":\"" << uuid_string(properties.uuid) << "\""
              << ",\"free_memory_bytes\":" << static_cast<std::uint64_t>(free_bytes)
              << ",\"total_memory_bytes\":" << static_cast<std::uint64_t>(total_bytes)
              << ",\"integrated_device\":" << (properties.integrated ? "true" : "false")
              << ",\"scalar_bits\":" << kScalarBits << ",\"c\":" << kC
              << ",\"chunk_scalars\":" << kChunkScalars << ",\"total_window_count\":" << kTotalWindowCount
              << ",\"M\":" << kM << ",\"supported_paths\":[\"ordinary_factor1\",\"prepared_factor4\"]"
              << ",\"cuda_runtime_version\":" << runtime_version << ",\"cuda_driver_version\":" << driver_version
              << ",\"cuda_toolkit_version\":\"" << toolkit_version << "\""
              << ",\"cub_version\":" << CUB_VERSION
              << ",\"device_compute_capability\":\"" << properties.major << "." << properties.minor << "\""
              << ",\"native_type_sizes\":{\"scalar_bytes\":" << scalar_size
              << ",\"affine_bytes\":" << affine_size << ",\"projective_bytes\":" << projective_size
              << ",\"unsigned_bytes\":" << uint_size << "}"
              << ',';
    print_path_query("ordinary_factor1", ordinary_path);
    std::cout << ',';
    print_path_query("prepared_factor4", prepared_path);
    std::cout << ",\"device_workspace_peak_bytes\":" << device_peak << "}\n";
    return std::cout.good() ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "native MSM CUB workspace query failed: " << error.what() << '\n';
    return 1;
  }
}
