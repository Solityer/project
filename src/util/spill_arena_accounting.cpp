#include "gatzk/util/spill_arena_accounting.hpp"

#include <limits>
#include <stdexcept>
#include <string>

namespace gatzk::util
{

    std::size_t checked_spill_bytes(
        std::size_t element_count,
        std::size_t element_size,
        std::string_view label)
    {
        if (element_size == 0 ||
            element_count >
                std::numeric_limits<std::size_t>::max() / element_size)
        {
            throw std::runtime_error(
                "spill byte-size overflow: " + std::string(label));
        }
        return element_count * element_size;
    }

    std::size_t spill_align_up(
        std::size_t value,
        std::size_t alignment,
        std::string_view label)
    {
        if (alignment == 0 ||
            value > std::numeric_limits<std::size_t>::max() - (alignment - 1))
        {
            throw std::runtime_error(
                "spill alignment overflow: " + std::string(label));
        }
        return (value + alignment - 1) / alignment * alignment;
    }

    bool spill_arena_can_grow(
        std::size_t current_bytes,
        std::size_t growth_bytes,
        std::size_t total_cap_bytes)
    {
        return current_bytes <= total_cap_bytes &&
               growth_bytes <= total_cap_bytes - current_bytes;
    }

} // namespace gatzk::util
