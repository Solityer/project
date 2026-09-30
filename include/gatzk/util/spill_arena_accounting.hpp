#pragma once

#include <cstddef>
#include <string_view>

namespace gatzk::util
{

    std::size_t checked_spill_bytes(std::size_t element_count, std::size_t element_size, std::string_view label);

    std::size_t spill_align_up(std::size_t value, std::size_t alignment, std::string_view label);

    bool spill_arena_can_grow(std::size_t current_bytes, std::size_t growth_bytes, std::size_t total_cap_bytes);

}
