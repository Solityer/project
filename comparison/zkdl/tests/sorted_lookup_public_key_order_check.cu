#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "../gat_adapter/gat_native_core.cuh"

namespace {

using zkdl::FrWire;
using Wide = unsigned __int128;

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

FrWire table_key(std::uint64_t row, std::int64_t value, std::int64_t minimum,
                 std::uint32_t y_bits) {
    const __int128 difference = static_cast<__int128>(value) - minimum;
    require(difference >= 0, "test table value is below its declared minimum");
    const Wide packed = (static_cast<Wide>(row) << y_bits) |
                       static_cast<Wide>(difference);
    return zkdl_gat::relation::sorted_lookup_field_from_u128(packed);
}

}  // namespace

int main() {
    try {
        constexpr std::int64_t kLreluMinimum = -419430;
        constexpr std::uint32_t kOgbnYBits = 22U;

        // Samples are from the existing authenticated OGBN-Arxiv static table.
        const FrWire before_limb_wrap = table_key(1023U, -419226, kLreluMinimum,
                                                   kOgbnYBits);
        const FrWire after_limb_wrap = table_key(1024U, -419226, kLreluMinimum,
                                                  kOgbnYBits);
        require(before_limb_wrap[0U] > after_limb_wrap[0U] &&
                    after_limb_wrap[1U] == before_limb_wrap[1U] + 1U,
                "row 1023/1024 sample does not cross the low 32-bit limb");
        require(!zkdl_gat::sorted_lookup_detail::key_order_violation(
                    after_limb_wrap, before_limb_wrap),
                "production sorted-table guard rejected the OGBN row 1023/1024 sequence");

        // The same authenticated table has a second carry at the 262,144-row segment edge.
        const FrWire before_segment = table_key(262143U, -367002, kLreluMinimum,
                                                 kOgbnYBits);
        const FrWire after_segment = table_key(262144U, -367002, kLreluMinimum,
                                                kOgbnYBits);
        require(before_segment[0U] > after_segment[0U] &&
                    after_segment[1U] == before_segment[1U] + 1U,
                "OGBN segment-edge sample does not cross the low 32-bit limb");
        require(!zkdl_gat::sorted_lookup_detail::key_order_violation(
                    after_segment, before_segment),
                "production sorted-table guard rejected the OGBN segment-edge sequence");

        require(zkdl_gat::sorted_lookup_detail::key_order_violation(
                    after_limb_wrap, after_limb_wrap),
                "production sorted-table guard accepted a duplicate key");
        require(zkdl_gat::sorted_lookup_detail::key_order_violation(
                    before_limb_wrap, after_limb_wrap),
                "production sorted-table guard accepted a reversed key sequence");

        const Wide below_64 = (static_cast<Wide>(1U) << 64U) - 1U;
        const Wide at_64 = static_cast<Wide>(1U) << 64U;
        const FrWire below_64_key = zkdl_gat::relation::sorted_lookup_field_from_u128(below_64);
        const FrWire at_64_key = zkdl_gat::relation::sorted_lookup_field_from_u128(at_64);
        require(!zkdl_gat::sorted_lookup_detail::key_order_violation(at_64_key, below_64_key) &&
                    zkdl_gat::sorted_lookup_detail::key_order_violation(below_64_key, at_64_key),
                "production sorted-table guard mishandled a greater-than-64-bit key boundary");

        std::cout << "sorted_lookup_public_key_order=PASS ogbn_pairs=2 negative=duplicate,reversed "
                     "wide_boundary=64\n";
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << "sorted_lookup_public_key_order=FAIL error=" << exception.what() << '\n';
        return 1;
    }
}
