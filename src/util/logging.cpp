#include "gatzk/util/logging.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

namespace gatzk::util {

void ensure_directory(const std::string& path) {
    std::filesystem::create_directories(path);
}

void write_lines(const std::string& path, const std::vector<std::string>& lines) {
    std::ofstream output(path);
    for (const auto& line : lines) {
        output << line << '\n';
    }
}

void write_key_values(const std::string& path, const std::map<std::string, std::string>& entries) {
    std::ofstream output(path);
    for (const auto& [key, value] : entries) {
        output << key << " = " << value << '\n';
    }
}

void info(const std::string& message) {
    std::cout << "[gatzk] " << message << '\n';
}

}  // namespace gatzk::util
