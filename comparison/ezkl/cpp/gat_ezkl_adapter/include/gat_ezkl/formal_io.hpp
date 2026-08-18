#pragma once

#include <filesystem>
#include <string>

namespace gat_ezkl {

std::string json_escape(const std::string& value);
void write_text_file(const std::filesystem::path& path, const std::string& contents);

}  // namespace gat_ezkl
