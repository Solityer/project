// Narrow bridge from the adapter runner to the native zkDL relation core.
//
// All proof construction/replay lives in gat_native_core.cu and the original
// proof stack. This translation unit intentionally contains no capability
// preflight, function-address probe, metadata-only acceptance, or CPU/GPU
// fallback. A missing witness/resource is reported by the core as BLOCKED.

#include "gat_native_bridge.hpp"
#include "gat_native_core.cuh"

namespace zkdl_gat {

bool zkdl_gat_native_prove(const NativeProveRequest& request,
                           NativeProofResult* result,
                           std::string* error) {
    return zkdl_gat_native_core_prove(request, result, error);
}

bool zkdl_gat_native_verify(const NativeVerifyRequest& request,
                            std::string* error) {
    return zkdl_gat_native_core_verify(request, error);
}

}  // namespace zkdl_gat
