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
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "formal_relations.cuh"
#include "timer.hpp"

namespace fs = std::filesystem;

namespace {

#include "gat_modules/common.cuh"
#include "gat_modules/data_cache.cuh"
#include "gat_modules/model_bundle.cuh"
#include "gat_modules/cpu_forward.cuh"
#include "gat_modules/gpu_forward.cuh"
#include "gat_modules/formal_common.cuh"
#include "gat_modules/formal_sparse_utils.cuh"
#include "gat_modules/cpu_formal_backend.cuh"
#include "gat_modules/formal_softmax_runners.cuh"
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

        const std::vector<DatasetRunSpec> specs = {
            {"cora", "cora", "cora_gat", "cora_full.cfg", "transductive_node_classification", "node", "whole_graph_single", 1, 12},
            {"citeseer", "citeseer", "citeseer_gat", "citeseer_full.cfg", "transductive_node_classification", "node", "whole_graph_single", 1, 12},
            {"pubmed", "pubmed", "pubmed_gat", "pubmed_full.cfg", "transductive_node_classification", "node", "whole_graph_single", 1, 12},
            {"ppi", "ppi", "ppi_gat", "ppi_batch_formal.cfg", "inductive_multi_graph_node_classification", "graph", "multi_graph_batch", 295, 12},
            {"ogbn-arxiv", "ogbn_arxiv", "ogbn_arxiv_gat", "ogbn_arxiv_full.cfg", "transductive_node_classification", "node", "whole_graph_single", 1, 12},
        };

        if (formal_cpu_chain) {
            if (only_dataset.empty()) {
                throw std::runtime_error("--formal-cpu-chain requires --dataset");
            }
            bool found = false;
            for (const auto& spec : specs) {
                if (spec.dataset != only_dataset) {
                    continue;
                }
                found = true;
                run_cpu_formal_chain(project_root, output_root, spec);
            }
            if (!found) {
                throw std::runtime_error("unknown dataset for CPU formal chain: " + only_dataset);
            }
            return 0;
        }

        if (formal_sparse_grouped) {
            if (only_dataset.empty()) {
                throw std::runtime_error("--formal-sparse-grouped requires --dataset");
            }
            bool found = false;
            for (const auto& spec : specs) {
                if (spec.dataset != only_dataset) {
                    continue;
                }
                found = true;
                run_formal_sparse_grouped_gadget(project_root, output_root, spec, formal_commitment_block_size);
            }
            if (!found) {
                throw std::runtime_error("unknown dataset for formal sparse grouped gadget: " + only_dataset);
            }
            return 0;
        }

        if (formal_softmax_aggregation) {
            if (only_dataset.empty()) {
                throw std::runtime_error("--formal-softmax-aggregation requires --dataset");
            }
            bool found = false;
            for (const auto& spec : specs) {
                if (spec.dataset != only_dataset) {
                    continue;
                }
                found = true;
                run_formal_softmax_aggregation_gadget(
                    project_root, output_root, spec, softmax_max_feature_columns, formal_commitment_block_size);
            }
            if (!found) {
                throw std::runtime_error("unknown dataset for formal softmax aggregation gadget: " + only_dataset);
            }
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
