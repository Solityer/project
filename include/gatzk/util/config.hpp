#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace gatzk::util {

struct AppConfig {
    std::string dataset = "toy";
    std::string data_root = "data";
    std::string cache_root = "data/cache";
    std::string export_dir = "runs/default";
    std::string config_dir = ".";
    std::string project_root = ".";
    std::size_t hidden_dim = 4;
    std::size_t num_classes = 2;
    std::size_t range_bits = 10;
    std::uint64_t seed = 7;
    std::size_t local_nodes = 5;
    std::size_t center_node = 0;
    bool dump_trace = true;
    bool auto_prepare_dataset = false;
    bool prove_enabled = true;
};

AppConfig load_config(const std::string& path);

}  // namespace gatzk::util
