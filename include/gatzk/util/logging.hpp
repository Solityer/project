#pragma once

#include <map>
#include <string>
#include <vector>

namespace gatzk::util {

void ensure_directory(const std::string& path);
void write_lines(const std::string& path, const std::vector<std::string>& lines);
void write_key_values(const std::string& path, const std::map<std::string, std::string>& entries);
void info(const std::string& message);

}  // namespace gatzk::util
