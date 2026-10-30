#include "../zkrelation-proof.cuh"

#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace zkdl_gat::relation;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<zkdl::byte> count_only(std::uint64_t count) {
    std::vector<zkdl::byte> encoded;
    zkdl::append_u64(encoded, count);
    return encoded;
}

void expect_rejected(const std::vector<zkdl::byte>& encoded, std::uint64_t expected,
                     const std::string& label) {
    std::vector<FrWire> output{zkdl::fr_from_u32(77U)};
    const std::vector<FrWire> sentinel = output;
    require(!decode_public_field_vector(encoded, expected, output),
            label + " was accepted");
    require(output == sentinel, label + " changed the output on failure");
}

void check_malformed_vectors() {
    auto mismatch = count_only(2U);
    expect_rejected(mismatch, 3U, "domain count mismatch");

    auto short_wire = count_only(2U);
    append_fr_wire(short_wire, zkdl::fr_from_u32(1U));
    expect_rejected(short_wire, 2U, "short field payload");

    auto trailing_wire = count_only(1U);
    append_fr_wire(trailing_wire, zkdl::fr_from_u32(1U));
    trailing_wire.push_back(0U);
    expect_rejected(trailing_wire, 1U, "trailing payload byte");

    auto noncanonical_wire = count_only(1U);
    FrWire noncanonical{};
    noncanonical.fill(std::numeric_limits<std::uint32_t>::max());
    append_fr_wire(noncanonical_wire, noncanonical);
    expect_rejected(noncanonical_wire, 1U, "noncanonical field element");

    expect_rejected(count_only(std::numeric_limits<std::uint64_t>::max()),
                    (std::uint64_t{1U} << 24U), "UINT64_MAX wire count");
    expect_rejected(count_only(0U), 0U, "empty public vector");

    std::vector<zkdl::byte> over_limit_descriptor = count_only((std::uint64_t{1U} << 20U) + 1U);
    size_t offset = 0U;
    std::vector<FrWire> descriptor_values;
    require(!read_fr_vector(over_limit_descriptor, offset, descriptor_values),
            "generic descriptor vector cap was raised");
    require(descriptor_values.empty(), "generic descriptor vector allocated on rejected count");
}

void check_formal_domain(std::uint64_t count, const char* label) {
    require(count <= (std::numeric_limits<size_t>::max() - sizeof(std::uint64_t)) /
                        sizeof(FrWire),
            "fixture domain does not fit size_t");
    std::vector<zkdl::byte> encoded;
    encoded.reserve(sizeof(std::uint64_t) + static_cast<size_t>(count) * sizeof(FrWire));
    zkdl::append_u64(encoded, count);
    for (std::uint64_t index = 0U; index < count; ++index) {
        FrWire value{};
        value[0U] = static_cast<std::uint32_t>(index);
        append_fr_wire(encoded, value);
    }

    std::vector<FrWire> decoded;
    const auto started = std::chrono::steady_clock::now();
    require(decode_public_field_vector(encoded, count, decoded),
            std::string(label) + " public Y decode failed");
    require(decoded.size() == count, std::string(label) + " decoded wrong element count");
    const std::uint64_t probes[] = {0U, count / 2U, count - 1U};
    for (const auto index : probes) {
        FrWire expected{};
        expected[0U] = static_cast<std::uint32_t>(index);
        require(decoded[static_cast<size_t>(index)] == expected,
                std::string(label) + " changed public Y element order/value");
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started).count();
    std::cout << "public_field_vector_decode=" << label
              << " status=PASS elements=" << count
              << " seconds=" << static_cast<double>(elapsed) / 1000.0 << '\n';
}

}  // namespace

int main() {
    try {
        check_malformed_vectors();
        check_formal_domain(std::uint64_t{1U} << 23U, "ppi");
        check_formal_domain(std::uint64_t{1U} << 24U, "ogbn_arxiv");
        std::cout << "public_field_vector_decode_fixture=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "public_field_vector_decode_fixture=FAIL reason=" << error.what() << '\n';
        return 1;
    }
}
