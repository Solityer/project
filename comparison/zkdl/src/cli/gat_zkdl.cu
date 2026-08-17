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
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "formal_relations.cuh"
#include "timer.hpp"

namespace fs = std::filesystem;

namespace {

#include "gat_modules/common.cuh"
#include "gat_modules/data_cache.cuh"
#include "gat_modules/experiment_specs.cuh"
#include "gat_modules/model_bundle.cuh"
#include "gat_modules/fixed_point_projection.cuh"
#include "gat_modules/fixed_point_attention.cuh"
#include "gat_modules/cpu_forward.cuh"
#include "gat_modules/gpu_forward.cuh"
#include "gat_modules/formal_common.cuh"
#include "gat_modules/protocol_context.cuh"
#include "gat_modules/formal_sparse_utils.cuh"
#include "gat_modules/full_proof_object.cuh"
#include "gat_modules/full_verifier.cuh"
#include "gat_modules/softmax_lookup_runner.cuh"
#include "gat_modules/cpu_formal_backend.cuh"
#include "gat_modules/cpu_linear_relations.cuh"
#include "gat_modules/cpu_multi_column_relations.cuh"
#include "gat_modules/full_output_binding_runner.cuh"
#include "gat_modules/projection_runner.cuh"
#include "gat_modules/rescale_rounding_runner.cuh"
#include "gat_modules/attention_dot_runner.cuh"
#include "gat_modules/range_comparison_runner.cuh"
#include "gat_modules/expmap_lookup_runner.cuh"
#include "gat_modules/nonlinear_lookup_runner.cuh"
#include "gat_modules/output_attention_runner.cuh"
#include "gat_modules/hidden_concat_runner.cuh"
#include "gat_modules/full_chain_runner.cuh"
#include "gat_modules/cpu_full_chain_runner.cuh"
#include "gat_modules/run_dispatch.cuh"

} // namespace

// CLI 入口只解析路径、dataset、device 与 commitment block size。
// 真正的 CPU/GPU full-chain proof 生成、manifest 写入和 verifier gate 都交给 run_dispatch。
int main(int argc, char** argv) {
    try {
        fs::path project_root = "/home/pzh/project";
        fs::path output_root = fs::current_path();
        bool formal_full_chain = false;
        std::string formal_device = "gpu";
        std::string only_dataset;
        std::size_t formal_commitment_block_size = 64;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help") {
                std::cout
                    << "       ./gat_zkdl --formal-full-chain --device cpu|gpu --dataset DATASET --output-root DIR\n";
                return 0;
            } else if (arg == "--project-root" && i + 1 < argc) {
                project_root = argv[++i];
            } else if (arg == "--output-root" && i + 1 < argc) {
                output_root = argv[++i];
            } else if (arg == "--dataset" && i + 1 < argc) {
                only_dataset = argv[++i];
            } else if (arg == "--device" && i + 1 < argc) {
                formal_device = argv[++i];
            } else if (arg == "--run-all") {
                throw std::runtime_error("--run-all 非 official 调度已移除；请显式使用 --formal-full-chain");
            } else if (arg == "--formal-full-chain") {
                formal_full_chain = true;
            } else if (arg == "--formal-commitment-block-size" && i + 1 < argc) {
                formal_commitment_block_size = static_cast<std::size_t>(std::stoull(argv[++i]));
            } else {
                throw std::runtime_error("unknown argument: " + arg);
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

        if (formal_full_chain) {
            run_selected_full_chain(project_root, output_root, specs, only_dataset, formal_device, formal_commitment_block_size);
            return 0;
        }

        throw std::runtime_error(
            "未指定 official full-chain 入口；请使用 --formal-full-chain，并通过 update_zkdl_gat_results.py 导出 CPU/GPU 分目录结果");
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
