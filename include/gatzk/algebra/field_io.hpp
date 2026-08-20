#pragma once

#include <cstddef>
#include <iosfwd>
#include <span>
#include <vector>

#include "gatzk/algebra/field.hpp"

namespace gatzk::algebra
{

    // Execution-only spill I/O. Each element uses the canonical 32-byte little-
    // endian field encoding; this does not participate in proof serialization.
    void write_field_elements_binary(
        std::ostream &stream,
        std::span<const FieldElement> values);

    // Read exactly values.size() canonical field elements into caller-owned
    // storage. This is the streaming primitive used by large trace/opening paths
    // so they can reuse bounded buffers instead of allocating one vector per pass.
    void read_field_elements_binary_into(
        std::istream &stream,
        std::span<FieldElement> values);

    std::vector<FieldElement> read_field_elements_binary(
        std::istream &stream,
        std::size_t value_count);

} // namespace gatzk::algebra