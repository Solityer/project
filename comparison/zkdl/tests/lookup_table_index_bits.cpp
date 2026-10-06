#include "gat_adapter/lookup_table_index_bits.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>

int main() {
    try {
        if (zkdl_gat::table_index_bit_width(4097U) != 13U ||
            zkdl_gat::table_index_bit_width(8193U) != 14U) {
            std::cerr << "lookup table index width does not cover the authenticated table rows\n";
            return 1;
        }
        for (const auto test : {std::pair<std::uint64_t, std::uint32_t>{4096U, 13U},
                                {8192U, 14U}}) {
            if ((test.first >> test.second) != 0U ||
                (test.first >> (test.second - 1U)) == 0U) {
                std::cerr << "lookup maximum-offset boundary check failed\n";
                return 1;
            }
        }
        for (const std::size_t invalid_rows : std::array<std::size_t, 3>{
                 0U, 1U, (std::size_t(1U) << 30U) | 1U}) {
            bool rejected = false;
            try {
                (void)zkdl_gat::table_index_bit_width(invalid_rows);
            } catch (const std::invalid_argument &) {
                rejected = true;
            }
            if (!rejected) {
                std::cerr << "invalid lookup table shape unexpectedly accepted\n";
                return 1;
            }
        }
        std::cout << "lookup_table_index_bits=PASS rows4097=13 rows8193=14 max_offsets=covered invalid_shape=rejected\n";
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
