#include "gatzk/util/logging.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

namespace gatzk::util {
namespace {

std::string escape_json(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (const auto ch : value) {
        switch (ch) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            default: out.push_back(ch); break;
        }
    }
    return out;
}

}  // namespace

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

void write_json_object(const std::string& path, const std::map<std::string, std::string>& entries) {
    std::ofstream output(path);
    output << "{\n";
    std::size_t index = 0;
    for (const auto& [key, value] : entries) {
        output << "  \"" << escape_json(key) << "\": "
               << "\"" << escape_json(value) << "\"";
        if (++index != entries.size()) {
            output << ',';
        }
        output << '\n';
    }
    output << "}\n";
}

void info(const std::string& message) {
    std::cout << "[gatzk] " << message << '\n';
}

}  // namespace gatzk::util
