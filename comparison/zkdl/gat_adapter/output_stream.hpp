#pragma once

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace zkdl_gat {

inline void finish_output_stream(std::ofstream &output, const std::filesystem::path &path, const std::string &label) {
    output.flush();
    const bool flushed = static_cast<bool>(output);
    output.close();
    const bool closed = static_cast<bool>(output);
    if (!flushed || !closed) throw std::runtime_error(label + " write/close failed: " + path.string());
}

}  // namespace zkdl_gat
