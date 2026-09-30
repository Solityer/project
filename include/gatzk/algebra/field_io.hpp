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

    std::size_t field_elements_binary_size(std::size_t value_count);

    void require_field_elements_binary_size(const std::filesystem::path &path, std::size_t value_count, std::string_view operation);

    void write_field_elements_binary(std::ostream &stream, std::span<const FieldElement> values);

    void read_field_elements_binary_into(std::istream &stream, std::span<FieldElement> values);

    std::vector<FieldElement> read_field_elements_binary(std::istream &stream, std::size_t value_count);

}
