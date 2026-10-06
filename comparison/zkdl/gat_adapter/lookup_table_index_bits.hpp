#ifndef ZKDL_GAT_LOOKUP_TABLE_INDEX_BITS_HPP
#define ZKDL_GAT_LOOKUP_TABLE_INDEX_BITS_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace zkdl_gat {

// The canonical unsigned bit width for an index into a table with this many
// logical rows.  The largest valid offset is rows - 1, so non-power-of-two
// tables use ceil(log2(rows)) bits as well.
inline std::uint32_t table_index_bit_width(std::size_t logical_rows) {
    constexpr std::uint64_t kMaxNativeRows = std::uint64_t(1U) << 30U;
    if (logical_rows < 2U || logical_rows > kMaxNativeRows)
        throw std::invalid_argument("lookup table has an invalid logical row count");
    std::uint64_t domain = 1U;
    std::uint32_t bits = 0U;
    while (domain < logical_rows) {
        if (domain > kMaxNativeRows / 2U)
            throw std::invalid_argument("lookup table index bit width exceeds native domain");
        domain <<= 1U;
        ++bits;
    }
    return bits;
}

}  // namespace zkdl_gat

#endif
