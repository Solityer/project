#pragma once
#include <openssl/evp.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

namespace zkdl_gat {
using StoredFieldWire = std::array<std::uint32_t, 8U>;
inline constexpr StoredFieldWire kStoredScalarModulus{{1U,4294967295U,4294859774U,1404937218U,161601541U,859428872U,698187080U,1944954707U}};

inline StoredFieldWire stored_subtract(const StoredFieldWire& left, const StoredFieldWire& right) {
    StoredFieldWire result{};
    std::uint64_t borrow = 0U;
    for (std::size_t index = 0U; index < result.size(); ++index) { const std::uint64_t sub = static_cast<std::uint64_t>(right[index]) + borrow; result[index] = static_cast<std::uint32_t>(static_cast<std::uint64_t>(left[index]) - sub); borrow = left[index] < sub ? 1U : 0U; }
    return result;
}

inline bool stored_signed_value(const StoredFieldWire& value, std::int64_t& result) {
    const auto small = [](const StoredFieldWire& input) { return std::all_of(input.begin() + 2U, input.end(), [](std::uint32_t limb) { return limb == 0U; }); };
    if (small(value) && value[1U] <= 0x7fffffffU) { result = static_cast<std::int64_t>((static_cast<std::uint64_t>(value[1U]) << 32U) | value[0U]); return true; }
    const auto magnitude = stored_subtract(kStoredScalarModulus, value);
    if (!small(magnitude) || magnitude[1U] > 0x80000000U || (magnitude[1U] == 0x80000000U && magnitude[0U] != 0U)) return false;
    const std::uint64_t absolute = (static_cast<std::uint64_t>(magnitude[1U]) << 32U) | magnitude[0U];
    result = absolute == (1ULL << 63U) ? std::numeric_limits<std::int64_t>::min() : -static_cast<std::int64_t>(absolute);
    return absolute != 0U;
}

inline StoredFieldWire stored_from_signed(std::int64_t value) {
    const std::uint64_t magnitude = value < 0 ? static_cast<std::uint64_t>(-(value + 1)) + 1U : static_cast<std::uint64_t>(value);
    StoredFieldWire result{};
    result[0U] = static_cast<std::uint32_t>(magnitude); result[1U] = static_cast<std::uint32_t>(magnitude >> 32U);
    return value < 0 ? stored_subtract(kStoredScalarModulus, result) : result;
}

struct StoredFileDescriptor {
    int fd = -1;
    explicit StoredFileDescriptor(int value) : fd(value) {}
    StoredFileDescriptor(const StoredFileDescriptor&) = delete;
    ~StoredFileDescriptor() { if (fd >= 0) ::close(fd); }
};

class NativeFamilyStorage {
    std::filesystem::path path_;
    std::uint32_t encoding_ = 0U;
    std::uint64_t count_ = 0U;
    std::uint64_t bytes_ = 0U;
    std::array<std::uint8_t, 32U> digest_{};
    mutable std::mutex mutex_;
    mutable bool authenticated_ = false;
    mutable struct stat identity_{};
    static constexpr std::size_t kHeaderBytes = 24U;
    static constexpr std::size_t kBufferBytes = 1U << 20U;

    static void checked_write(int fd, const std::uint8_t* bytes, std::size_t count) {
        while (count != 0U) { const ssize_t written = ::write(fd, bytes, count); if (written < 0 && errno == EINTR) continue; if (written <= 0) throw std::runtime_error("private family storage write failed"); bytes += written; count -= static_cast<std::size_t>(written); }
    }

    static std::array<std::uint8_t, 32U> hash_file(int fd) {
        std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> hash(EVP_MD_CTX_new(), EVP_MD_CTX_free);
        if (!hash || EVP_DigestInit_ex(hash.get(), EVP_sha256(), nullptr) != 1) throw std::runtime_error("private family digest initialization failed");
        std::vector<std::uint8_t> buffer(kBufferBytes);
        std::uint64_t offset = 0U;
        for (;;) { const ssize_t count = ::pread(fd, buffer.data(), buffer.size(), static_cast<off_t>(offset)); if (count < 0 && errno == EINTR) continue; if (count < 0) throw std::runtime_error("private family digest read failed"); if (count == 0) break; if (EVP_DigestUpdate(hash.get(), buffer.data(), static_cast<std::size_t>(count)) != 1) throw std::runtime_error("private family digest update failed"); offset += static_cast<std::uint64_t>(count); }
        std::array<std::uint8_t, 32U> digest{};
        unsigned length = 0U;
        if (EVP_DigestFinal_ex(hash.get(), digest.data(), &length) != 1 || length != digest.size()) throw std::runtime_error("private family digest finalization failed");
        return digest;
    }

    void authenticate(int fd) const {
        std::lock_guard<std::mutex> lock(mutex_);
        struct stat identity{};
        if (::fstat(fd, &identity) != 0 || !S_ISREG(identity.st_mode) || identity.st_uid != ::geteuid() || (identity.st_mode & 077U) != 0U || identity.st_size < 0 || static_cast<std::uint64_t>(identity.st_size) != bytes_) throw std::runtime_error("private family storage identity/permissions are invalid");
        if (!authenticated_) { if (hash_file(fd) != digest_) throw std::runtime_error("private family digest authentication failed"); identity_ = identity; authenticated_ = true; }
        else if (identity.st_dev != identity_.st_dev || identity.st_ino != identity_.st_ino || identity.st_size != identity_.st_size || identity.st_mtim.tv_sec != identity_.st_mtim.tv_sec || identity.st_mtim.tv_nsec != identity_.st_mtim.tv_nsec) throw std::runtime_error("private family storage changed after authentication");
    }

    NativeFamilyStorage(const std::filesystem::path& path, std::uint32_t encoding, std::uint64_t count, std::uint64_t bytes, const std::array<std::uint8_t, 32U>& digest) : path_(path), encoding_(encoding), count_(count), bytes_(bytes), digest_(digest) {}
public:
    std::uint64_t count() const noexcept { return count_; }

    template<class Reader>
    static std::shared_ptr<const NativeFamilyStorage> persist_generated(const std::filesystem::path& directory, std::uint64_t count, std::uint32_t encoding, Reader reader) {
        if (count == 0U || count > (1ULL << 30U) || encoding > 3U) throw std::runtime_error("private family logical values/encoding are invalid");
        if (encoding == 0U) {
            bool boolean = true, signed64 = true;
            for (std::uint64_t index = 0U; index < count; ++index) { const auto value = reader(index); boolean = boolean && value[0U] <= 1U && std::all_of(value.begin() + 1U, value.end(), [](std::uint32_t limb) { return limb == 0U; }); std::int64_t decoded = 0; signed64 = signed64 && stored_signed_value(value, decoded); }
            encoding = boolean ? 1U : (signed64 ? 2U : 3U);
        }
        const std::uint64_t payload = encoding == 1U ? (count + 7U) / 8U : count * (encoding == 2U ? 8U : 32U);
        if (std::filesystem::is_symlink(directory) || !std::filesystem::is_directory(directory)) throw std::runtime_error("private family directory is invalid");
        if (std::filesystem::space(directory).available < payload + kHeaderBytes) throw std::runtime_error("insufficient private family scratch space");
        std::string filename = (directory / ".family-XXXXXX").string();
        std::vector<char> writable(filename.begin(), filename.end()); writable.push_back('\0');
        StoredFileDescriptor file(::mkstemp(writable.data()));
        if (file.fd < 0) throw std::runtime_error("cannot create private family storage");
        const std::filesystem::path temporary(writable.data());
        try {
            if (::fchmod(file.fd, 0600) != 0) throw std::runtime_error("private family chmod failed");
            std::array<std::uint8_t, kHeaderBytes> header{{'G','A','T','F','A','M','1',0U}};
            for (unsigned byte = 0U; byte < 4U; ++byte) header[8U + byte] = static_cast<std::uint8_t>(encoding >> (8U * byte));
            for (unsigned byte = 0U; byte < 8U; ++byte) header[16U + byte] = static_cast<std::uint8_t>(count >> (8U * byte));
            checked_write(file.fd, header.data(), header.size());
            std::vector<std::uint8_t> buffer;
            buffer.reserve(kBufferBytes + 32U);
            std::uint8_t packed = 0U;
            for (std::uint64_t index = 0U; index < count; ++index) {
                const auto value = reader(index);
                if (!std::lexicographical_compare(value.rbegin(), value.rend(), kStoredScalarModulus.rbegin(), kStoredScalarModulus.rend())) throw std::runtime_error("private family contains a non-canonical field residue");
                if (encoding == 1U && (value[0U] > 1U || std::any_of(value.begin() + 1U, value.end(), [](std::uint32_t limb) { return limb != 0U; }))) throw std::runtime_error("private family boolean conversion failed");
                if (encoding == 1U) { packed |= static_cast<std::uint8_t>(value[0U] << (index & 7U)); if ((index & 7U) == 7U || index + 1U == count) { buffer.push_back(packed); packed = 0U; } }
                else if (encoding == 2U) { std::int64_t decoded = 0; if (!stored_signed_value(value, decoded)) throw std::runtime_error("private family signed conversion failed"); const auto raw = static_cast<std::uint64_t>(decoded); for (unsigned byte = 0U; byte < 8U; ++byte) buffer.push_back(static_cast<std::uint8_t>(raw >> (8U * byte))); }
                else { for (const auto limb : value) for (unsigned byte = 0U; byte < 4U; ++byte) buffer.push_back(static_cast<std::uint8_t>(limb >> (8U * byte))); }
                if (buffer.size() >= kBufferBytes) { checked_write(file.fd, buffer.data(), buffer.size()); buffer.clear(); }
            }
            if (!buffer.empty()) checked_write(file.fd, buffer.data(), buffer.size());
            if (::fsync(file.fd) != 0) throw std::runtime_error("private family fsync failed");
            const auto digest = hash_file(file.fd);
            static constexpr char hex[] = "0123456789abcdef";
            std::string name;
            for (const auto byte : digest) { name.push_back(hex[byte >> 4U]); name.push_back(hex[byte & 15U]); }
            const auto destination = directory / (name + ".family");
            if (::link(temporary.c_str(), destination.c_str()) != 0 && errno != EEXIST) throw std::runtime_error("cannot publish private family storage");
            ::unlink(temporary.c_str());
            static std::mutex pool_mutex;
            static std::map<std::string, std::weak_ptr<const NativeFamilyStorage>> pool;
            std::lock_guard<std::mutex> lock(pool_mutex);
            const auto found = pool.find(destination.string());
            if (found != pool.end()) { const auto previous = found->second.lock(); if (previous) return previous; }
            const auto storage = std::shared_ptr<const NativeFamilyStorage>(new NativeFamilyStorage(destination, encoding, count, payload + kHeaderBytes, digest));
            pool[destination.string()] = storage;
            return storage;
        } catch (...) { ::unlink(temporary.c_str()); throw; }
    }

    static std::shared_ptr<const NativeFamilyStorage> persist(const std::filesystem::path& directory, const std::vector<StoredFieldWire>& values) { return persist_generated(directory, values.size(), 0U, [&](std::uint64_t index) { return values.at(static_cast<std::size_t>(index)); }); }

    bool read_padded(const std::vector<std::uint64_t>& logical_shape, const std::vector<std::uint64_t>& padded_shape, std::uint64_t offset, std::uint32_t count, std::vector<StoredFieldWire>& values, std::string* error) const {
        try {
            if (count > (1U << 18U) || logical_shape.empty() || logical_shape.size() != padded_shape.size()) throw std::runtime_error("private family reader shape/range is invalid");
            std::uint64_t logical_count = 1U, padded_count = 1U;
            for (std::size_t axis = 0U; axis < logical_shape.size(); ++axis) { if (logical_shape[axis] == 0U || padded_shape[axis] < logical_shape[axis] || logical_count > std::numeric_limits<std::uint64_t>::max() / logical_shape[axis] || padded_count > (1ULL << 30U) / padded_shape[axis]) throw std::runtime_error("private family Cartesian shape overflows"); logical_count *= logical_shape[axis]; padded_count *= padded_shape[axis]; }
            if (logical_count != count_ || offset > padded_count || count > padded_count - offset) throw std::runtime_error("private family reader exceeds its authenticated Cartesian domain");
            StoredFileDescriptor file(::open(path_.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC));
            if (file.fd < 0) throw std::runtime_error("private family storage open failed");
            authenticate(file.fd);
            void* mapping = ::mmap(nullptr, bytes_, PROT_READ, MAP_PRIVATE, file.fd, 0);
            if (mapping == MAP_FAILED) throw std::runtime_error("private family mapping failed");
            struct MappingLease { void* address; std::size_t bytes; ~MappingLease() { ::munmap(address, bytes); } } lease{mapping, static_cast<std::size_t>(bytes_)};
            const auto* payload = static_cast<const std::uint8_t*>(mapping) + kHeaderBytes;
            values.assign(count, StoredFieldWire{});
            for (std::uint32_t index = 0U; index < count; ++index) {
                std::uint64_t padded = offset + index, logical = 0U, stride = 1U;
                bool active = true;
                for (std::size_t axis = logical_shape.size(); axis-- > 0U;) { const std::uint64_t coordinate = padded % padded_shape[axis]; padded /= padded_shape[axis]; if (coordinate >= logical_shape[axis]) { active = false; break; } logical += coordinate * stride; stride *= logical_shape[axis]; }
                if (!active) continue;
                if (encoding_ == 1U) values[index][0U] = (payload[logical / 8U] >> (logical & 7U)) & 1U;
                else if (encoding_ == 2U) { std::uint64_t raw = 0U; for (unsigned byte = 0U; byte < 8U; ++byte) raw |= static_cast<std::uint64_t>(payload[logical * 8U + byte]) << (8U * byte); std::int64_t decoded; std::memcpy(&decoded, &raw, sizeof(decoded)); values[index] = stored_from_signed(decoded); }
                else { for (unsigned limb = 0U; limb < 8U; ++limb) for (unsigned byte = 0U; byte < 4U; ++byte) values[index][limb] |= static_cast<std::uint32_t>(payload[logical * 32U + limb * 4U + byte]) << (8U * byte); }
            }
            return true;
        } catch (const std::exception& exception) { if (error) *error = exception.what(); return false; }
    }
};
}
