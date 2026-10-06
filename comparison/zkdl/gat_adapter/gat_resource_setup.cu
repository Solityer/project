#include "third_party/blst/bindings/blst.h"
#include "gat_adapter/integer_bound_ledger.hpp"
#include <openssl/bn.h>
#include <openssl/crypto.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/random.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

namespace {
using Wire = std::array<std::uint8_t, 48U>;
constexpr std::uint64_t kDomain = 1ULL << 18U;
constexpr std::array<std::uint8_t, 32U> kScalarModulus = {{0x01,0x00,0x00,0x00,0xff,0xff,0xff,0xff,0xfe,0x5b,0xfe,0xff,0x02,0xa4,0xbd,0x53,0x05,0xd8,0xa1,0x09,0x08,0xd8,0x39,0x33,0x48,0x7d,0x9d,0x29,0x53,0xa7,0xed,0x73}};

struct SecretScalar {
    std::array<std::uint8_t, 32U> bytes{};
    SecretScalar() = default;
    SecretScalar(const SecretScalar&) = delete;
    SecretScalar& operator=(const SecretScalar&) = delete;
    ~SecretScalar() { OPENSSL_cleanse(bytes.data(), bytes.size()); }
};

void random_scalar(SecretScalar& scalar) {
    for (;;) {
        std::size_t offset = 0U;
        while (offset < scalar.bytes.size()) {
            const ssize_t count = ::getrandom(scalar.bytes.data() + offset, scalar.bytes.size() - offset, 0);
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) throw std::runtime_error("OS entropy unavailable for trusted native SRS setup");
            offset += static_cast<std::size_t>(count);
        }
        scalar.bytes.back() &= 0x7fU;
        const bool nonzero = std::any_of(scalar.bytes.begin(), scalar.bytes.end(), [](std::uint8_t value) { return value != 0U; });
        if (nonzero && std::lexicographical_compare(scalar.bytes.rbegin(), scalar.bytes.rend(), kScalarModulus.rbegin(), kScalarModulus.rend())) return;
        OPENSSL_cleanse(scalar.bytes.data(), scalar.bytes.size());
    }
}

std::filesystem::path setup_root;

std::filesystem::path baseline_root() { if (setup_root.empty()) throw std::runtime_error("explicit baseline root is required"); return setup_root; }

void configure_root(const std::filesystem::path& input) {
    if (std::filesystem::is_symlink(input) || !std::filesystem::is_directory(input)) throw std::runtime_error("baseline root must be a regular directory");
    setup_root = std::filesystem::canonical(input);
    const auto executable = std::filesystem::canonical("/proc/self/exe");
    const auto relative = executable.lexically_relative(setup_root);
    std::vector<std::string> components;
    for (const auto& item : relative) components.push_back(item.string());
    const bool direct = components == std::vector<std::string>{"gat_resource_setup"};
    const bool snapshot = components.size() == 4U && components[0U] == "runs" && components[1U] != ".." && components[2U] == "build" && components[3U] == "gat_resource_setup";
    if ((!direct && !snapshot) || !std::filesystem::is_regular_file(setup_root / "Makefile") || !std::filesystem::is_directory(setup_root / "gat_adapter")) throw std::runtime_error("native resource binary does not belong to the supplied baseline root");
}

std::filesystem::path local_path(const std::filesystem::path& path) {
    const auto absolute = std::filesystem::absolute(path).lexically_normal();
    const auto relative = absolute.lexically_relative(baseline_root());
    if (relative.empty() || relative.is_absolute() || *relative.begin() == "..") throw std::runtime_error("resource path escapes baseline root");
    auto current = absolute;
    while (!current.empty()) {
        if (std::filesystem::is_symlink(current)) throw std::runtime_error("resource path contains a symlink");
        const auto parent = current.parent_path();
        if (parent == current) break;
        current = parent;
    }
    return absolute;
}

struct OutputFile {
    int fd = -1;
    explicit OutputFile(const std::filesystem::path& path) { fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600); if (fd < 0) throw std::runtime_error("cannot create a fresh SRS frame: " + path.string()); }
    ~OutputFile() { if (fd >= 0) ::close(fd); }
    void write(const std::uint8_t* bytes, std::size_t count) {
        while (count != 0U) {
            const ssize_t done = ::write(fd, bytes, count);
            if (done < 0 && errno == EINTR) continue;
            if (done <= 0) throw std::runtime_error("SRS frame write failed");
            bytes += done; count -= static_cast<std::size_t>(done);
        }
    }
    void finish() { if (::fsync(fd) != 0) throw std::runtime_error("SRS frame fsync failed"); if (::close(fd) != 0) { fd = -1; throw std::runtime_error("SRS frame close failed"); } fd = -1; }
};

void write_header(OutputFile& file, std::uint64_t count) {
    std::array<std::uint8_t, 40U> bytes{{'G','A','T','Z','K','G','1','C'}};
    const auto put = [&](std::size_t offset, std::uint64_t value, std::size_t width) { for (std::size_t i = 0U; i < width; ++i) bytes[offset + i] = static_cast<std::uint8_t>(value >> (8U * i)); };
    put(8U, 1U, 4U); put(12U, 48U, 4U); put(16U, count, 8U); put(24U, kDomain, 8U); put(32U, kDomain - 1U, 8U);
    file.write(bytes.data(), bytes.size());
}

void generate_frame(const std::filesystem::path& path, std::uint64_t count, std::set<Wire>& unique) {
    OutputFile file(path);
    write_header(file, count);
    constexpr std::size_t batch_size = 1024U;
    for (std::uint64_t offset = 0U; offset < count; offset += batch_size) {
        const std::size_t batch = static_cast<std::size_t>(std::min<std::uint64_t>(batch_size, count - offset));
        std::vector<blst_p1> points(batch);
        std::vector<blst_p1_affine> affine(batch);
        for (auto& point : points) { SecretScalar scalar; random_scalar(scalar); blst_p1_mult(&point, blst_p1_generator(), scalar.bytes.data(), 255U); }
        const blst_p1* sources[] = {points.data(), nullptr};
        blst_p1s_to_affine(affine.data(), sources, batch);
        for (auto& point : affine) {
            Wire encoded{};
            blst_p1_affine_compress(encoded.data(), &point);
            if (blst_p1_affine_is_inf(&point) || !blst_p1_affine_in_g1(&point) || !unique.insert(encoded).second) throw std::runtime_error("nonzero independent SRS basis validation failed");
            file.write(encoded.data(), encoded.size());
        }
    }
    file.finish();
}

void generate_srs(const std::filesystem::path& output) {
    const auto directory = local_path(output);
    if (!std::filesystem::is_directory(directory) || !std::filesystem::is_empty(directory)) throw std::runtime_error("SRS output must be an existing empty baseline-local directory");
    const auto wildcard = directory / "wildcard";
    std::filesystem::create_directory(wildcard);
    std::set<Wire> unique;
    generate_frame(wildcard / "G.chunk0.bin", kDomain, unique);
    generate_frame(wildcard / "H.chunk0.bin", 1U, unique);
    generate_frame(wildcard / "U.chunk0.bin", 1U, unique);
    std::cerr << "SRS_GENERATED trusted_setup=native_local_ephemeral_random points=" << unique.size() << '\n';
}

std::string decimal_coordinate(const blst_fp& coordinate) {
    Wire encoded{};
    blst_bendian_from_fp(encoded.data(), &coordinate);
    std::unique_ptr<BIGNUM, decltype(&BN_free)> value(BN_bin2bn(encoded.data(), encoded.size(), nullptr), BN_free);
    if (!value) throw std::runtime_error("public coordinate conversion failed");
    char* decimal = BN_bn2dec(value.get());
    if (decimal == nullptr) throw std::runtime_error("public coordinate decimal encoding failed");
    const std::string result(decimal);
    OPENSSL_free(decimal);
    return result;
}

void export_prefix(const std::filesystem::path& frame, std::uint64_t count) {
    const auto path = local_path(frame);
    if (!std::filesystem::is_regular_file(path)) throw std::runtime_error("SRS frame is not a regular file");
    std::ifstream stream(path, std::ios::binary);
    std::array<std::uint8_t, 40U> header{};
    stream.read(reinterpret_cast<char*>(header.data()), header.size());
    const auto get = [&](std::size_t offset, std::size_t width) { std::uint64_t value = 0U; for (std::size_t i = 0U; i < width; ++i) value |= static_cast<std::uint64_t>(header[offset + i]) << (8U * i); return value; };
    constexpr std::array<std::uint8_t, 8U> magic{{'G','A','T','Z','K','G','1','C'}};
    const std::uint64_t available = get(16U, 8U);
    if (!stream || !std::equal(magic.begin(), magic.end(), header.begin()) || get(8U, 4U) != 1U || get(12U, 4U) != 48U || available == 0U || available > kDomain || count == 0U || count > available || get(24U, 8U) != kDomain || get(32U, 8U) != kDomain - 1U || std::filesystem::file_size(path) != header.size() + available * 48U) throw std::runtime_error("SRS frame header/count/size is invalid");
    std::cout << "{\"schema\":\"zkDL_gat_srs_prefix_v1\",\"points\":[";
    for (std::uint64_t index = 0U; index < count; ++index) {
        Wire encoded{}, roundtrip{};
        stream.read(reinterpret_cast<char*>(encoded.data()), encoded.size());
        blst_p1_affine point{};
        if (!stream || blst_p1_uncompress(&point, encoded.data()) != BLST_SUCCESS || !blst_p1_affine_on_curve(&point) || !blst_p1_affine_in_g1(&point) || blst_p1_affine_is_inf(&point)) throw std::runtime_error("SRS prefix contains an invalid subgroup point");
        blst_p1_affine_compress(roundtrip.data(), &point);
        if (roundtrip != encoded) throw std::runtime_error("SRS prefix point is not canonical");
        if (index != 0U) std::cout << ',';
        std::cout << "[\"" << decimal_coordinate(point.x) << "\",\"" << decimal_coordinate(point.y) << "\"]";
    }
    std::cout << "]}\n";
    if (!std::cout) throw std::runtime_error("SRS prefix output failed");
}

void verify_ledger(const std::filesystem::path& input) {
    const auto path = local_path(input);
    if (!std::filesystem::is_regular_file(path)) throw std::runtime_error("integer-bound ledger is not a regular file");
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot read integer-bound ledger: " + path.string());
    const std::string text{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    const auto ledger = zkdl_gat::integer_bound_ledger::parse(text);
    const auto actual = zkdl_gat::integer_bound_ledger::digest(ledger);
    if (actual != ledger.digest) throw std::runtime_error("authenticated integer-bound ledger digest mismatch");
    std::cout << "native_integer_bound_ledger=PASS digest=" << actual << " path=" << path << '\n';
}
}

int main(int argc, char** argv) {
    try {
        if (argc == 6 && std::string(argv[1]) == "generate-srs" && std::string(argv[2]) == "--output" && std::string(argv[4]) == "--baseline-root") { configure_root(argv[5]); generate_srs(argv[3]); }
        else if (argc == 6 && std::string(argv[1]) == "verify-ledger" && std::string(argv[2]) == "--path" && std::string(argv[4]) == "--baseline-root") { configure_root(argv[5]); verify_ledger(argv[3]); }
        else if (argc == 8 && std::string(argv[1]) == "export-srs-prefix" && std::string(argv[2]) == "--frame" && std::string(argv[4]) == "--count" && std::string(argv[6]) == "--baseline-root") {
            configure_root(argv[7]);
            const std::string count_text(argv[5]);
            if (count_text.empty() || !std::all_of(count_text.begin(), count_text.end(), [](char value) { return value >= '0' && value <= '9'; })) throw std::runtime_error("SRS prefix count is not an unsigned integer");
            export_prefix(argv[3], std::stoull(count_text));
        } else throw std::runtime_error("usage: gat_resource_setup generate-srs --output DIR --baseline-root ROOT | verify-ledger --path FILE --baseline-root ROOT | export-srs-prefix --frame PATH --count N --baseline-root ROOT");
        return 0;
    } catch (const std::exception& error) { std::cerr << "BLOCKED: " << error.what() << '\n'; return 2; }
}
