#include "zkdl_gat/baseline.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

std::string require_arg(int& index, int argc, char** argv, const std::string& name) {
    if (index + 1 >= argc) {
        throw std::runtime_error("missing value for " + name);
    }
    return argv[++index];
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string config_path;
        std::string route;
        std::string output_path;
        std::string gpu_probe_path = "runs/gpu_probe.json";

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--config") {
                config_path = require_arg(i, argc, argv, arg);
            } else if (arg == "--route") {
                route = require_arg(i, argc, argv, arg);
            } else if (arg == "--out") {
                output_path = require_arg(i, argc, argv, arg);
            } else if (arg == "--gpu-probe") {
                gpu_probe_path = require_arg(i, argc, argv, arg);
            } else {
                throw std::runtime_error("unknown argument: " + arg);
            }
        }

        if (config_path.empty() || route.empty() || output_path.empty()) {
            throw std::runtime_error("usage: zkdl_gat_runner --config <cfg> --route <cpu|gpu> --out <manifest>");
        }
        if (route != "cpu" && route != "gpu") {
            throw std::runtime_error("--route must be cpu or gpu");
        }

        const auto config = zkdl_gat::load_dataset_config(config_path);
        const auto gpu_probe = route == "gpu" ? zkdl_gat::load_gpu_probe(gpu_probe_path) : zkdl_gat::GpuProbe{};
        auto manifest = zkdl_gat::build_manifest(config, route, gpu_probe);
        zkdl_gat::write_manifest_json(manifest, output_path);

        std::cout
            << "dataset=" << config.dataset_id
            << " route=" << route
            << " status=" << manifest.string_values.at("status")
            << " failure_step=" << manifest.string_values.at("failure_step")
            << " block_reason=" << manifest.string_values.at("block_reason")
            << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
