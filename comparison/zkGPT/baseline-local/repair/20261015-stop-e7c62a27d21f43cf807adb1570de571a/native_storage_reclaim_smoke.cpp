#include "gat_adapter/native_storage.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

int main() {
    constexpr std::size_t bytes = std::size_t(16) << 20;
    constexpr std::size_t active = std::size_t(8) << 20;
    const std::size_t alignment = alignof(std::max_align_t);
    auto* data = static_cast<unsigned char*>(zkgpt::native_storage::allocate(bytes, alignment));
    std::memset(data, 0x5a, active);
    data[active - 1] = 0x27;
    const auto first = zkgpt::native_storage::reclaim_tail(data, active, bytes);
    assert(first.mapped && first.mapping_bytes == bytes && first.live_bytes == active);
    assert(data[0] == 0x5a && data[active / 2] == 0x5a && data[active - 1] == 0x27);

    const auto repeat = zkgpt::native_storage::reclaim_tail(data, active, bytes);
    assert(repeat.mapped && repeat.mapping_bytes == bytes && repeat.live_bytes == active);
    const auto same_page_shrink = zkgpt::native_storage::reclaim_tail(data, active - 1, bytes);
    assert(same_page_shrink.live_bytes == active - 1);
    assert(data[0] == 0x5a && data[active / 2] == 0x5a && data[active - 2] == 0x5a);

    bool oversized_prefix_rejected = false;
    try { (void)zkgpt::native_storage::reclaim_tail(data, bytes + 1, bytes); }
    catch (const std::out_of_range&) { oversized_prefix_rejected = true; }
    assert(oversized_prefix_rejected);
    const auto measured = zkgpt::native_storage::physical_bytes(data, bytes);
    assert(measured == same_page_shrink.after_bytes);
    std::cout << "NATIVE_STORAGE_RECLAIM_REPEAT_OK"
              << " first_before=" << first.before_bytes
              << " first_after=" << first.after_bytes
              << " repeat_before=" << repeat.before_bytes
              << " repeat_after=" << repeat.after_bytes
              << " within_page_after=" << same_page_shrink.after_bytes
              << " oversized_prefix_rejected=1 live_prefix_preserved=1\n";
    zkgpt::native_storage::deallocate(data, alignment);
}
