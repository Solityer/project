#pragma once

#include <cstddef>
#include <filesystem>
#include <iosfwd>
#include <span>
#include <string_view>
#include <vector>

#include "gatzk/algebra/field.hpp"

namespace gatzk::algebra
{

    // Return the exact number of bytes occupied by value_count canonical
    // field encodings in a spill file.  Spill files use the fixed wire
    // encoding below rather than sizeof(FieldElement), whose in-memory size
    // is implementation-specific.
    std::size_t field_elements_binary_size(std::size_t value_count);

    // Fail closed unless path contains exactly value_count canonical field
    // encodings.  This check is intentionally separate from the streaming
    // reader so every file-backed production consumer can reject both
    // truncation and trailing bytes before consuming any values.
    void require_field_elements_binary_size(const std::filesystem::path &path, std::size_t value_count, std::string_view operation);

    // Execution-only spill I/O. Each element uses the canonical 32-byte little-
    // endian field encoding; this does not participate in proof serialization.
    void write_field_elements_binary(std::ostream &stream, std::span<const FieldElement> values);

    // Read exactly values.size() canonical field elements into caller-owned
    // storage. This is the streaming primitive used by large trace/opening paths
    // so they can reuse bounded buffers instead of allocating one vector per pass.
    void read_field_elements_binary_into(std::istream &stream, std::span<FieldElement> values);

    std::vector<FieldElement> read_field_elements_binary(std::istream &stream, std::size_t value_count);

} // namespace gatzk::algebra
