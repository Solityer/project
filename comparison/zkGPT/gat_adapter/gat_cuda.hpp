#pragma once

#include "gat_types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace zkgpt {
namespace gat {

// Evidence returned by the baseline-native CUDA relation bridge.  The bridge
// computes one real projection (MatMul) and one sparse destination aggregate
// on device.  It is deliberately separate from the proof artifact: a GPU
// request is accepted only when both kernels launch and the device is
// synchronized successfully.
struct GpuExecutionEvidence {
    bool compiled = false;
    bool device_available = false;
    bool kernel_executed = false;
    std::uint64_t kernel_invocations = 0;
    std::size_t matmul_rows = 0;
    std::size_t matmul_inner = 0;
    std::size_t matmul_cols = 0;
    std::size_t aggregation_edges = 0;
    std::string output_checksum;
    std::string diagnostic;
};

GpuExecutionEvidence execute_gpu_matmul_aggregation(
    const Tensor2D &input, const std::vector<Int> &weights,
    std::size_t weight_rows, std::size_t weight_cols, const Graph &graph);

} // namespace gat
} // namespace zkgpt
