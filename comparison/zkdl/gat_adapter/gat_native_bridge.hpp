#pragma once

#include "gat_adapter.hpp"
#include "gat_native_abi.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace zkdl_gat {

struct NativeProveRequest {
    const Config *config = nullptr;
    const Graph *graph = nullptr;
    const ModelBundle *model = nullptr;
    const StaticTables *tables = nullptr;
    const RelationPlan *relation = nullptr;
    const GatTrace *trace = nullptr;
    const PublicStatement *statement = nullptr;
    Backend backend = Backend::Cpu;

    const NativeRelationWitnessInput *witness = nullptr;
};

struct NativeProofResult {
    ProofEnvelope envelope;
    std::vector<std::uint8_t> proof_bytes;
};

struct NativeVerifyRequest {
    const Config *config = nullptr;
    const PublicStatement *statement = nullptr;
    ProofEnvelope *envelope = nullptr;
    const std::vector<std::uint8_t> *proof_bytes = nullptr;
    Backend backend = Backend::Cpu;

    const Graph *graph = nullptr;

    const NativeRelationWitnessInput *public_context = nullptr;

    const RelationPlan *relation = nullptr;
    const StaticTables *tables = nullptr;
};

#if defined(__GNUC__) || defined(__clang__)
#define ZKDL_GAT_WEAK __attribute__((weak))
#else
#define ZKDL_GAT_WEAK
#endif

bool zkdl_gat_native_prove(const NativeProveRequest &request, NativeProofResult *result, std::string *error) ZKDL_GAT_WEAK;
bool zkdl_gat_native_verify(const NativeVerifyRequest &request, std::string *error) ZKDL_GAT_WEAK;

inline bool native_bridge_available() noexcept { return zkdl_gat_native_prove != nullptr && zkdl_gat_native_verify != nullptr; }

#undef ZKDL_GAT_WEAK

}
