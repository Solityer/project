#pragma once

// Narrow integration seam between the GAT adapter and the official zkDL
// commitment/Sumcheck implementation.  The adapter owns parsing, canonical
// graph/model witnesses, and the relation plan; the official core owns all
// commitments, transcript challenges, Sumcheck messages, and terminal
// evaluation/opening verification.  Keeping this seam in a separate header
// makes it impossible for the adapter to grow a second proof system.

#include "gat_adapter.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace zkdl_gat {

struct NativeProveRequest {
    const Config* config = nullptr;
    const Graph* graph = nullptr;
    const ModelBundle* model = nullptr;
    const StaticTables* tables = nullptr;
    const RelationPlan* relation = nullptr;
    const GatTrace* trace = nullptr;
    const PublicStatement* statement = nullptr;
    Backend backend = Backend::Cpu;
};

struct NativeProofResult {
    ProofEnvelope envelope;
    std::vector<std::uint8_t> proof_bytes;
};

struct NativeVerifyRequest {
    const Config* config = nullptr;
    const PublicStatement* statement = nullptr;
    const ProofEnvelope* envelope = nullptr;
    const std::vector<std::uint8_t>* proof_bytes = nullptr;
    Backend backend = Backend::Cpu;
};

// The core worker supplies these functions when the native bridge is linked.
// They are weak on purpose: a baseline-local build without the bridge must
// remain linkable, but every prove/verify command then returns a hard failure
// rather than producing a pseudo-proof or silently switching to CPU.
#if defined(__GNUC__) || defined(__clang__)
#define ZKDL_GAT_WEAK __attribute__((weak))
#else
#define ZKDL_GAT_WEAK
#endif

bool zkdl_gat_native_prove(const NativeProveRequest& request,
                           NativeProofResult* result,
                           std::string* error) ZKDL_GAT_WEAK;
bool zkdl_gat_native_verify(const NativeVerifyRequest& request,
                            std::string* error) ZKDL_GAT_WEAK;

inline bool native_bridge_available() noexcept {
    return zkdl_gat_native_prove != nullptr && zkdl_gat_native_verify != nullptr;
}

#undef ZKDL_GAT_WEAK

}  // namespace zkdl_gat
