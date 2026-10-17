#pragma once

#include "global_var.hpp"
#include <cybozu/sha2.hpp>
#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#ifdef __linux__
#include <linux/falloc.h>
#include <linux/fiemap.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
#endif

namespace zkgpt {
namespace native_storage {
struct Mapping { std::size_t bytes; std::size_t file_bytes; std::size_t live_bytes; std::uint64_t allocated_bytes; std::size_t allocation_unit_bytes; int fd; };
inline std::mutex mapping_mutex;
inline std::unordered_map<void*, Mapping> mappings;
inline constexpr std::size_t heap_limit = std::size_t(8) << 20;

inline std::uint64_t fd_allocated_bytes(int fd) {
    struct stat info{};
    if (::fstat(fd, &info) != 0 || info.st_blocks < 0) throw std::runtime_error("cannot inspect native scratch allocation blocks");
    const unsigned __int128 bytes = static_cast<unsigned __int128>(info.st_blocks) * 512U;
    if (bytes > std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("native scratch allocated block count exceeds u64");
    return static_cast<std::uint64_t>(bytes);
}

inline std::size_t page_size() {
    const long size = ::sysconf(_SC_PAGESIZE);
    if (size <= 0) throw std::runtime_error("cannot inspect native scratch page size");
    return static_cast<std::size_t>(size);
}

namespace detail {
inline bool punched_range_has_no_allocated_extent(int fd, std::uint64_t begin, std::uint64_t end) {
    // An empty range cannot establish that the punch removed its data.
    if (begin >= end) return false;
#ifdef __linux__
    static_assert(offsetof(struct fiemap, fm_extents) == sizeof(struct fiemap));
    static_assert(alignof(struct fiemap) >= alignof(struct fiemap_extent));
    constexpr std::size_t extent_offset = offsetof(struct fiemap, fm_extents);
    alignas(struct fiemap) unsigned char buffer[extent_offset + sizeof(struct fiemap_extent)]{};
    auto* request = reinterpret_cast<struct fiemap*>(buffer);
    request->fm_start = begin;
    request->fm_length = end - begin;
    request->fm_flags = 0;
    request->fm_extent_count = 1;
    request->fm_mapped_extents = 0;

    int ioctl_result;
    do { ioctl_result = ::ioctl(fd, FS_IOC_FIEMAP, request); } while (ioctl_result != 0 && errno == EINTR);
    if (ioctl_result != 0) {
        const int error = errno;
        throw std::runtime_error("cannot inspect native scratch reclaimed extents: " + std::to_string(error));
    }
    if (request->fm_start != begin || request->fm_length != end - begin ||
        request->fm_flags != 0U || request->fm_extent_count != 1U || request->fm_mapped_extents > 1U)
        throw std::runtime_error("native scratch FIEMAP returned unsupported extent metadata");
    if (request->fm_mapped_extents == 0U) return true;

    const auto* extent = reinterpret_cast<const struct fiemap_extent*>(buffer + extent_offset);
    if (extent->fe_length == 0U ||
        extent->fe_logical > std::numeric_limits<std::uint64_t>::max() - extent->fe_length)
        throw std::runtime_error("native scratch FIEMAP returned an invalid extent range");
    const std::uint64_t extent_end = extent->fe_logical + extent->fe_length;

    // Every FIEMAP extent is treated as allocated, including unwritten,
    // delayed-allocation, and unknown extents. Only a proven logical hole passes.
    if (extent_end <= begin || extent->fe_logical >= end)
        throw std::runtime_error("native scratch FIEMAP returned an extent outside the queried range");
    return false;
#else
    (void)fd;
    throw std::runtime_error("native scratch FIEMAP extent inspection is unsupported on this platform");
#endif
}
} // namespace detail

inline std::size_t filesystem_allocation_unit_bytes(const std::filesystem::path& directory) {
    struct statvfs capacity{};
    if (::statvfs(directory.c_str(), &capacity) != 0)
        throw std::runtime_error("cannot inspect native scratch allocation unit: " + directory.string());
    const std::uint64_t unit = capacity.f_frsize != 0U ? capacity.f_frsize : capacity.f_bsize;
    if (unit == 0U || unit > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("native scratch filesystem allocation unit is unsupported");
    return static_cast<std::size_t>(unit);
}

inline std::filesystem::path baseline_root() {
    std::vector<std::filesystem::path> candidates{std::filesystem::current_path()};
    char executable[4096];
    const ssize_t length = ::readlink("/proc/self/exe", executable, sizeof(executable) - 1);
    if (length > 0) { executable[length] = 0; candidates.emplace_back(std::filesystem::path(executable).parent_path()); }
    for (auto candidate : candidates) {
        for (; !candidate.empty(); candidate = candidate.parent_path()) {
            if (std::filesystem::is_regular_file(candidate / "GAT-ZKML-zkGPT.md") && std::filesystem::is_directory(candidate / "gat_adapter")) return std::filesystem::canonical(candidate);
            if (candidate == candidate.parent_path()) break;
        }
    }
    throw std::runtime_error("native scratch storage requires the zkGPT baseline directory");
}

inline std::filesystem::path scratch_directory() {
    const auto root = baseline_root();
    const char* configured = std::getenv("ZKGPT_NATIVE_SCRATCH_DIR");
    auto path = configured && *configured ? std::filesystem::path(configured) : root / "artifacts/private/native_scratch";
    if (path.is_relative()) path = root / path;
    path = path.lexically_normal();
    auto relative = path.lexically_relative(root);
    if (relative.empty() || *relative.begin() == "..") throw std::runtime_error("native scratch path must be inside zkGPT");
    auto checked = root;
    for (const auto& component : relative) {
        checked /= component;
        if (std::filesystem::is_symlink(std::filesystem::symlink_status(checked))) throw std::runtime_error("native scratch path contains a symlink");
    }
    std::filesystem::create_directories(path);
    if (std::filesystem::canonical(path).lexically_relative(root).begin()->string() == "..") throw std::runtime_error("native scratch path escapes zkGPT");
    if (::chmod(path.c_str(), 0700) != 0) throw std::runtime_error("cannot protect native scratch directory");
    return path;
}

inline std::uint64_t available_bytes(const std::filesystem::path& directory) {
    struct statvfs capacity{};
    if (::statvfs(directory.c_str(), &capacity) != 0)
        throw std::runtime_error("cannot inspect native storage capacity: " + directory.string());
    const unsigned __int128 available = static_cast<unsigned __int128>(capacity.f_bavail) * capacity.f_frsize;
    if (available > std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("native storage capacity exceeds u64");
    return static_cast<std::uint64_t>(available);
}

inline std::uint64_t scratch_available_bytes() {
    return available_bytes(scratch_directory());
}

inline void* allocate(std::size_t bytes, std::size_t alignment) {
    if (bytes <= heap_limit) return ::operator new(bytes, std::align_val_t(alignment));
    if (bytes > static_cast<std::size_t>(std::numeric_limits<off_t>::max())) throw std::length_error("native scratch allocation exceeds file offset capacity");
    const auto directory = scratch_directory();
    const std::uint64_t available = available_bytes(directory);
    const std::size_t allocation_unit = filesystem_allocation_unit_bytes(directory);
    if (bytes > std::numeric_limits<std::size_t>::max() - (allocation_unit - 1U))
        throw std::length_error("native scratch allocation block rounding overflows");
    const std::size_t physical_reservation = ((bytes + allocation_unit - 1U) / allocation_unit) * allocation_unit;
    if (available < physical_reservation) throw std::runtime_error("insufficient zkGPT native scratch disk capacity: required " + std::to_string(physical_reservation) + " allocated bytes");
    std::string pattern = (directory / "oracle-XXXXXX").string();
    std::vector<char> filename(pattern.begin(), pattern.end());
    filename.push_back(0);
    const int fd = ::mkstemp(filename.data());
    if (fd < 0) throw std::runtime_error("cannot create native scratch file");
    auto close_failure = [&]() { ::unlink(filename.data()); ::close(fd); };
    if (::fchmod(fd, 0600) != 0 || ::fcntl(fd, F_SETFD, FD_CLOEXEC) != 0) { close_failure(); throw std::runtime_error("cannot protect native scratch file"); }
    const int reserve_error = ::posix_fallocate(fd, 0, static_cast<off_t>(bytes));
    if (reserve_error != 0) { close_failure(); throw std::runtime_error("native scratch disk reservation failed: " + std::to_string(reserve_error)); }
    if (::unlink(filename.data()) != 0) { close_failure(); throw std::runtime_error("cannot unlink private native scratch file"); }
    void* pointer = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (pointer == MAP_FAILED) { ::close(fd); throw std::bad_alloc(); }
    if (reinterpret_cast<std::uintptr_t>(pointer) % alignment != 0) { ::munmap(pointer, bytes); ::close(fd); throw std::runtime_error("native scratch object alignment is unsupported"); }
    ::madvise(pointer, bytes, MADV_SEQUENTIAL);
    try { std::lock_guard<std::mutex> lock(mapping_mutex); mappings.emplace(pointer, Mapping{bytes, bytes, bytes, fd_allocated_bytes(fd), allocation_unit, fd}); }
    catch (...) { ::munmap(pointer, bytes); ::close(fd); throw; }
    return pointer;
}

struct ReclaimReport { std::uint64_t before_bytes; std::uint64_t after_bytes; std::size_t mapping_bytes; std::size_t live_bytes; bool mapped; };

inline std::uint64_t physical_bytes(const void* pointer, std::size_t virtual_bytes) {
    if (pointer == nullptr || virtual_bytes == 0U) return 0U;
    std::lock_guard<std::mutex> lock(mapping_mutex);
    const auto found = mappings.find(const_cast<void*>(pointer));
    if (found == mappings.end()) return virtual_bytes;
    found->second.allocated_bytes = fd_allocated_bytes(found->second.fd);
    return found->second.allocated_bytes;
}

inline void record_live_bound(void* pointer, std::size_t live_bytes, std::size_t virtual_bytes) {
    if (live_bytes > virtual_bytes) throw std::out_of_range("native scratch live bound exceeds its mapping capacity");
    if (pointer == nullptr || virtual_bytes == 0U) {
        if (live_bytes != 0U) throw std::invalid_argument("native scratch live bound has no allocation");
        return;
    }
    std::lock_guard<std::mutex> lock(mapping_mutex);
    const auto found = mappings.find(pointer);
    if (found == mappings.end()) return;
    if (found->second.bytes != virtual_bytes || found->second.file_bytes != virtual_bytes)
        throw std::logic_error("native scratch live bound changed its original mapping length");
    found->second.live_bytes = live_bytes;
}

inline ReclaimReport reclaim_tail(void* pointer, std::size_t active_bytes, std::size_t virtual_bytes) {
    if (active_bytes > virtual_bytes) throw std::out_of_range("native scratch live prefix exceeds its allocation capacity");
    if (pointer == nullptr || virtual_bytes == 0U) {
        if (active_bytes != 0U) throw std::invalid_argument("native scratch live prefix has no allocation");
        return {0U, 0U, 0U, 0U, false};
    }
    std::lock_guard<std::mutex> lock(mapping_mutex);
    const auto found = mappings.find(pointer);
    if (found == mappings.end()) return {virtual_bytes, virtual_bytes, virtual_bytes, active_bytes, false};
    Mapping& mapping = found->second;
    if (mapping.bytes != virtual_bytes || mapping.file_bytes != virtual_bytes) throw std::logic_error("native scratch mapping capacity changed while live");
    const std::uint64_t before = fd_allocated_bytes(mapping.fd);
    const std::size_t page = page_size();
    if (active_bytes > std::numeric_limits<std::size_t>::max() - (page - 1U)) throw std::overflow_error("native scratch live prefix alignment overflows");
    const std::size_t aligned_live = std::min(mapping.file_bytes, ((active_bytes + page - 1U) / page) * page);
    const std::size_t aligned_end = mapping.file_bytes - mapping.file_bytes % page;
    if (aligned_live < aligned_end) {
#ifdef FALLOC_FL_PUNCH_HOLE
        // Reclaiming an already sparse tail is valid; physical allocation is checked below.
        if (::fallocate(mapping.fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, static_cast<off_t>(aligned_live), static_cast<off_t>(aligned_end - aligned_live)) != 0)
            throw std::runtime_error("native scratch tail reclamation failed: " + std::to_string(errno));
#else
        throw std::runtime_error("native scratch tail reclamation is unsupported by this platform");
#endif
    }
    const std::uint64_t after = fd_allocated_bytes(mapping.fd);
    const std::size_t trailing_fragment = mapping.file_bytes - aligned_end;
    const auto round_up = [](std::size_t bytes, std::size_t unit) {
        if (bytes == 0U) return std::size_t(0);
        if (bytes > std::numeric_limits<std::size_t>::max() - (unit - 1U))
            return std::numeric_limits<std::size_t>::max();
        return ((bytes + unit - 1U) / unit) * unit;
    };
    const std::size_t allocated_live_bound = round_up(aligned_live, mapping.allocation_unit_bytes);
    const std::size_t allocated_tail_bound = round_up(trailing_fragment, mapping.allocation_unit_bytes);
    const std::size_t expected_bound = allocated_live_bound > std::numeric_limits<std::size_t>::max() - allocated_tail_bound
        ? std::numeric_limits<std::size_t>::max() : allocated_live_bound + allocated_tail_bound;
    const std::size_t allocation_margin = mapping.allocation_unit_bytes;
    if (allocation_margin == 0U || expected_bound > std::numeric_limits<std::size_t>::max() - allocation_margin)
        throw std::overflow_error("native scratch physical block bound overflows");
    const std::size_t physical_bound = expected_bound + allocation_margin;
    if (after > physical_bound &&
        !detail::punched_range_has_no_allocated_extent(mapping.fd, aligned_live, aligned_end))
        throw std::runtime_error(
            "native scratch allocated blocks exceed the live-prefix bound after reclamation (allocated=" +
            std::to_string(after) + ", bound=" + std::to_string(physical_bound) +
            ", active=" + std::to_string(active_bytes) + ", page_live=" +
            std::to_string(aligned_live) + ", file=" + std::to_string(mapping.file_bytes) +
            ", tail=" + std::to_string(trailing_fragment) + ")");
    mapping.live_bytes = active_bytes;
    mapping.allocated_bytes = after;
    return {before, after, mapping.bytes, mapping.live_bytes, true};
}

inline void deallocate(void* pointer, std::size_t alignment) noexcept {
    if (!pointer) return;
    Mapping mapping{};
    { std::lock_guard<std::mutex> lock(mapping_mutex); const auto found = mappings.find(pointer); if (found != mappings.end()) { mapping = found->second; mappings.erase(found); } }
    if (mapping.bytes) { ::munmap(pointer, mapping.bytes); ::close(mapping.fd); }
    else ::operator delete(pointer, std::align_val_t(alignment));
}
}

template<class T> class NativeScratchAllocator {
public:
    using value_type = T;
    using is_always_equal = std::true_type;
    NativeScratchAllocator() noexcept = default;
    template<class U> NativeScratchAllocator(const NativeScratchAllocator<U>&) noexcept {}
    T* allocate(std::size_t count) {
        static_assert(std::is_standard_layout_v<T> && std::is_trivially_destructible_v<T>, "native scratch objects require a value layout with trivial destruction");
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) throw std::length_error("native scratch byte size overflow");
        return static_cast<T*>(native_storage::allocate(count * sizeof(T), alignof(T)));
    }
    void deallocate(T* pointer, std::size_t) noexcept { native_storage::deallocate(pointer, alignof(T)); }
    template<class U> bool operator==(const NativeScratchAllocator<U>&) const noexcept { return true; }
    template<class U> bool operator!=(const NativeScratchAllocator<U>&) const noexcept { return false; }
};

template<class T> using NativeVector = std::vector<T, NativeScratchAllocator<T>>;
using NativeFieldVector = NativeVector<F>;

class NativeFieldView {
public:
    using Getter = std::function<F(std::size_t)>;

    NativeFieldView() = default;

    // The caller owns the referenced storage. Keep lifetime_token alive until
    // every synchronous consumer has returned; getters must be thread-safe and read-only.
    static NativeFieldView borrowed(std::size_t logical_size,
                                    const void* lifetime_token,
                                    Getter getter) {
        if (lifetime_token == nullptr || !getter)
            throw std::invalid_argument("native field view requires a live lifetime token and getter");
        NativeFieldView view;
        view.logical_size_ = logical_size;
        view.lifetime_token_ = lifetime_token;
        view.getter_ = std::move(getter);
        return view;
    }

    template<class Vector> static NativeFieldView from_vector(const Vector& values) {
        const Vector* borrowed_values = &values;
        return borrowed(values.size(), borrowed_values,
                        [borrowed_values](std::size_t index) { return borrowed_values->at(index); });
    }

    bool configured() const noexcept {
        return logical_size_ != 0 || lifetime_token_ != nullptr || static_cast<bool>(getter_);
    }
    bool valid() const noexcept { return lifetime_token_ != nullptr && static_cast<bool>(getter_); }
    std::size_t size() const noexcept { return logical_size_; }

    F at(std::size_t index) const {
        if (!valid()) throw std::logic_error("native field view is not live");
        if (index >= logical_size_) throw std::out_of_range("native field view index exceeds logical size");
        return getter_(index);
    }
    F operator[](std::size_t index) const { return at(index); }

private:
    std::size_t logical_size_ = 0;
    const void* lifetime_token_ = nullptr;
    Getter getter_;
};

class NativeStreamingDigest {
    cybozu::Sha256 hash_;
public:
    explicit NativeStreamingDigest(const std::string& domain) { hash_.update(domain); }
    void bytes(const void* data, std::size_t size) { hash_.update(data, size); }
    void u64(std::uint64_t value) { char encoded[8]; for (unsigned byte = 0; byte < 8; ++byte) encoded[byte] = static_cast<char>((value >> (8 * byte)) & 255); bytes(encoded, sizeof(encoded)); }
    void string(const std::string& value) { u64(value.size()); bytes(value.data(), value.size()); }
    std::string finish() { char digest[32]; hash_.digest(digest, sizeof(digest), nullptr, 0); static const char hex[] = "0123456789abcdef"; std::string result; result.reserve(64); for (unsigned char byte : digest) { result.push_back(hex[byte >> 4]); result.push_back(hex[byte & 15]); } return result; }
};

template<class Vector> std::string native_field_vector_digest(const Vector& values, const std::string& domain) {
    NativeStreamingDigest digest(domain);
    digest.u64(2);
    digest.u64(values.size());
    for (std::size_t index = 0; index < values.size(); ++index)
        digest.string(values[index].getStr(16));
    return digest.finish();
}

template<class Callback> void native_for_each_equality_weight(const std::vector<F>& point, std::size_t count, Callback callback) {
    if (point.size() >= std::numeric_limits<std::size_t>::digits || count > (std::size_t(1) << point.size())) throw std::length_error("native equality domain is invalid");
    const std::size_t low_bits = std::min<std::size_t>(12, point.size());
    const std::size_t width = std::size_t(1) << low_bits;
    std::vector<F> low(width, F(0));
    low[0] = F(1);
    for (std::size_t bit = 0; bit < low_bits; ++bit) {
        const std::size_t active = std::size_t(1) << bit;
        for (std::size_t index = 0; index < active; ++index) { const F value = low[index]; low[index] = value * (F(1) - point[bit]); low[index + active] = value * point[bit]; }
    }
    for (std::size_t offset = 0; offset < count; offset += width) {
        F high(1);
        for (std::size_t bit = low_bits; bit < point.size(); ++bit) high *= ((offset >> bit) & 1) ? point[bit] : F(1) - point[bit];
        for (std::size_t index = 0; index < width && offset + index < count; ++index) callback(offset + index, high * low[index]);
    }
}

template<class Vector> F native_mle_evaluate(const Vector& values, const std::vector<F>& point) {
    if (point.size() >= std::numeric_limits<std::size_t>::digits || values.size() > (std::size_t(1) << point.size())) throw std::length_error("native MLE domain is invalid");
    const std::size_t low_bits = std::min<std::size_t>(12, point.size());
    const std::size_t width = std::size_t(1) << low_bits;
    std::vector<F> tile(width, F(0));
    F evaluation(0);
    for (std::size_t offset = 0; offset < values.size(); offset += width) {
        for (std::size_t index = 0; index < width; ++index) tile[index] = offset + index < values.size() ? values[offset + index] : F(0);
        std::size_t active = width;
        for (std::size_t bit = 0; bit < low_bits; ++bit) {
            active >>= 1;
            for (std::size_t index = 0; index < active; ++index) tile[index] = tile[2 * index] * (F(1) - point[bit]) + tile[2 * index + 1] * point[bit];
        }
        F high(1);
        for (std::size_t bit = low_bits; bit < point.size(); ++bit) high *= ((offset >> bit) & 1) ? point[bit] : F(1) - point[bit];
        evaluation += high * tile[0];
    }
    return evaluation;
}

}
