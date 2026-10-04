#include "zkrelation-proof.cuh"
#include "zkmatmul-proof.cuh"
#include "gat_adapter/gat_native_resources.cu"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

using namespace zkdl_gat::relation;

static void put_u32(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) bytes.push_back(static_cast<std::uint8_t>(value >> (8U * i)));
}
static void put_u64(std::vector<std::uint8_t>& bytes, std::uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) bytes.push_back(static_cast<std::uint8_t>(value >> (8U * i)));
}
static void put_frame_u32(std::vector<std::uint8_t>& bytes, std::size_t offset, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) bytes[offset + i] = static_cast<std::uint8_t>(value >> (8U * i));
}
static void put_frame_u64(std::vector<std::uint8_t>& bytes, std::size_t offset, std::uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) bytes[offset + i] = static_cast<std::uint8_t>(value >> (8U * i));
}

int main() {
    std::uint32_t narrow = 0;
    if (!checked_relation_u32(std::uint64_t(UINT32_MAX), narrow, "u32 max") || narrow != UINT32_MAX) return 1;
    if (checked_relation_u32(std::uint64_t(UINT32_MAX) + 1U, narrow, "u32 overflow")) return 2;

    if (!parse_relation_u32_suffix("4294967295", narrow) || narrow != UINT32_MAX) return 3;
    if (parse_relation_u32_suffix("4294967296", narrow)) return 4;
    if (parse_relation_u32_suffix("-1", narrow)) return 5;
    if (parse_relation_u32_suffix("", narrow)) return 6;

    std::uint32_t segment_size = 0, segment_count = 0;
    if (!SegmentedTensorProvider::shape(1ULL << 30, 1U << 18, segment_size, segment_count) ||
        segment_size != (1U << 18) || segment_count != (1U << 12)) return 7;
    if (SegmentedTensorProvider::shape(1ULL << 31, 1U << 18, segment_size, segment_count)) return 8;

    std::uint64_t product = 0;
    if (!zkdl::MatMulShape::checked_product(65536U, 65536U, 65536U, product) || product != (1ULL << 48)) return 9;
    if (zkdl::MatMulShape::checked_product(UINT32_MAX, UINT32_MAX, UINT32_MAX, product)) return 10;

    // Invoke the production nested proof parser with a declared length larger
    // than remaining bytes; it must reject before slicing/converting offsets.
    std::vector<std::uint8_t> padding{'P','A','D','1'};
    put_u32(padding, 1U);
    put_u64(padding, 1U); padding.push_back('x');
    put_u32(padding, 0U); put_u32(padding, 1U);
    padding.insert(padding.end(), 32U, 0U);
    put_u64(padding, 1U);
    zkdl_gat::relation::PaddingProofRecord padding_record;
    std::string parse_error;
    if (zkdl_gat::relation::PaddingProofRecord::deserialize(padding, padding_record, &parse_error)) return 11;

    // Exercise current SRS-frame count/48-byte extent caps and malformed payload.
    constexpr std::uint64_t header_bytes = 40U;
    constexpr std::uint64_t point_bytes = 48U;
    constexpr std::uint64_t frame_cap = (1ULL << 32U);
    const std::uint64_t max_count = (frame_cap - header_bytes) / point_bytes;
    if (!zkdl_gat::valid_srs_frame_bytes(1U, header_bytes + point_bytes) ||
        !zkdl_gat::valid_srs_frame_bytes(max_count, header_bytes + max_count * point_bytes) ||
        zkdl_gat::valid_srs_frame_bytes(max_count + 1U, header_bytes) ||
        zkdl_gat::valid_srs_frame_bytes(UINT64_MAX, header_bytes)) return 12;

    auto frame = std::vector<std::uint8_t>(header_bytes, 0U);
    const std::uint8_t magic[8] = {'G','A','T','Z','K','G','1','C'};
    std::copy(magic, magic + 8, frame.begin());
    put_frame_u32(frame, 8U, 1U); put_frame_u32(frame, 12U, 48U);
    put_frame_u64(frame, 16U, max_count + 1U); put_frame_u64(frame, 24U, 1U); put_frame_u64(frame, 32U, 0U);
    zkdl_gat::PointFrame parsed_frame;
    parse_error.clear();
    if (zkdl_gat::decode_frame(frame, parsed_frame, &parse_error)) return 13;

    put_frame_u64(frame, 16U, 1U);
    parse_error.clear();
    if (zkdl_gat::decode_frame(frame, parsed_frame, &parse_error) || parse_error.find("truncated/trailing payload") == std::string::npos) return 14;

    // Test the production bounded resource reader without creating a large file.
    const auto temp = std::filesystem::temp_directory_path() / "zkdl_resource_parser_negative.bin";
    { std::ofstream out(temp, std::ios::binary); out.put('a'); out.put('b'); }
    std::vector<std::uint8_t> read_output;
    parse_error.clear();
    const bool read_ok = zkdl_gat::read_bytes(temp, read_output, &parse_error, 1U);
    std::filesystem::remove(temp);
    if (read_ok) return 15;

    std::cout << "WIDTH_RESOURCE_NEGATIVES_PASS checked_u32 suffix_overflow/negative/empty domain_cap product_overflow nested_length_gt_remaining srs_count_48B_cap frame_payload_truncated bounded_read_bytes\n";
    return 0;
}
