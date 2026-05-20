#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "formal_relations.cuh"
#include "timer.hpp"

namespace fs = std::filesystem;

namespace {

#include "gat_modules/common.cuh"
#include "gat_modules/data_cache.cuh"
#include "gat_modules/experiment_specs.cuh"
#include "gat_modules/model_bundle.cuh"
#include "gat_modules/cpu_forward.cuh"
#include "gat_modules/gpu_forward.cuh"
#include "gat_modules/formal_common.cuh"
#include "gat_modules/formal_sparse_utils.cuh"
#include "gat_modules/cpu_formal_backend.cuh"
#include "gat_modules/cpu_formal_runner.cuh"
#include "gat_modules/formal_softmax_runners.cuh"
#include "gat_modules/gpu_formal_runner.cuh"
#include "gat_modules/run_dispatch.cuh"

} // namespace

int main(int argc, char** argv) {
    try {
        fs::path project_root = "/home/pzh/project";
        fs::path output_root = fs::current_path();
        bool formal_cpu_chain = false;
        bool formal_sparse_grouped = false;
        bool formal_softmax_aggregation = false;
        std::string only_dataset;
        std::size_t softmax_max_feature_columns = 0;
        std::size_t formal_commitment_block_size = 64;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help") {
                std::cout
                    << "usage: ./gat_zkdl [--run-all|--dataset DATASET] [--output-root DIR]\n"
                    << "       ./gat_zkdl --formal-cpu-chain --dataset DATASET --output-root DIR\n"
                    << "       ./gat_zkdl --formal-sparse-grouped --dataset DATASET [--formal-commitment-block-size N] --output-root DIR\n"
                    << "       ./gat_zkdl --formal-softmax-aggregation --dataset DATASET [--max-feature-columns N] [--formal-commitment-block-size N] --output-root DIR\n";
                return 0;
            } else if (arg == "--project-root" && i + 1 < argc) {
                project_root = argv[++i];
            } else if (arg == "--output-root" && i + 1 < argc) {
                output_root = argv[++i];
            } else if (arg == "--dataset" && i + 1 < argc) {
                only_dataset = argv[++i];
            } else if (arg == "--run-all") {
                continue;
            } else if (arg == "--formal-cpu-chain") {
                formal_cpu_chain = true;
            } else if (arg == "--formal-sparse-grouped") {
                formal_sparse_grouped = true;
            } else if (arg == "--formal-softmax-aggregation") {
                formal_softmax_aggregation = true;
            } else if (arg == "--max-feature-columns" && i + 1 < argc) {
                softmax_max_feature_columns = static_cast<std::size_t>(std::stoull(argv[++i]));
            } else if (arg == "--formal-commitment-block-size" && i + 1 < argc) {
                formal_commitment_block_size = static_cast<std::size_t>(std::stoull(argv[++i]));
            }
        }
        output_root = fs::absolute(output_root);
        const auto allowed_root = fs::absolute("/home/pzh/project/comparison/baselines/zkdl");
        const auto rel = fs::relative(output_root, allowed_root);
        if (rel.empty() || *rel.begin() == "..") {
            throw std::runtime_error("output-root must stay inside /home/pzh/project/comparison/baselines/zkdl");
        }
        ensure_dir(output_root);

        const auto specs = default_dataset_run_specs();

        if (formal_cpu_chain) {
            run_selected_cpu_formal_chain(
                project_root, output_root, specs, only_dataset, formal_commitment_block_size);
            return 0;
        }

        if (formal_sparse_grouped) {
            run_selected_sparse_grouped(
                project_root, output_root, specs, only_dataset, formal_commitment_block_size);
            return 0;
        }

        if (formal_softmax_aggregation) {
            run_selected_softmax_aggregation(
                project_root, output_root, specs, only_dataset,
                softmax_max_feature_columns, formal_commitment_block_size);
            return 0;
        }

        throw std::runtime_error(
            "默认 --run-all 旧调度会混写 GPU forward/debug 与正式结果，已禁用；"
            "请使用 --formal-cpu-chain、--formal-sparse-grouped 或 "
            "--formal-softmax-aggregation，并通过 update_zkdl_gat_results.py 导出 CPU/GPU 分目录结果");
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
