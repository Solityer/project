#include "gat_cuda.hpp"

namespace zkgpt {
namespace gat {

GpuExecutionEvidence execute_gpu_matmul_aggregation(const Tensor2D&, const std::vector<Int>&, std::size_t, std::size_t, const Graph&) {
    GpuExecutionEvidence evidence;
    evidence.compiled = false;
    evidence.diagnostic = "CUDA backend was not compiled; refusing CPU fallback";
    return evidence;
}

}
}
