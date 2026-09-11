#pragma once

// Baseline-local authenticated resource loaders for the native GAT bridge.
// These declarations intentionally expose only canonical wire material.  The
// proof stack remains responsible for decoding field/curve objects after the
// resource identity has been authenticated.

#include "gat_adapter.hpp"
#include "gat_native_abi.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace zkdl_gat {

struct AuthenticatedSrsResource {
    std::string manifest_path;
    std::array<std::uint8_t, 32U> manifest_digest{};
    std::array<std::uint8_t, 32U> generator_digest{};
    std::uint64_t count = 0U;
    std::uint64_t domain = 0U;
    std::uint64_t degree = 0U;
    std::uint32_t point_size = 0U;
    std::vector<std::vector<std::uint8_t>> generators;
    std::vector<std::uint8_t> blinding_generator;
    std::vector<std::uint8_t> u_generator;
    bool authenticated = false;
};

struct AuthenticatedModelRegistry {
    std::string path;
    std::array<std::uint8_t, 32U> digest{};
    std::string schema;
    std::string dataset;
    std::string model_arch_id;
    std::string model_param_id;
    std::string quant_cfg_id;
    std::string static_table_id;
    std::string srs_id;
    std::string srs_manifest_digest;
    std::string generator_digest;
    std::string config_digest;
    std::string verifier_registry_digest;
    std::string model_manifest_digest;
    std::string static_table_digest;
    std::string scale_chain_digest;
    std::string integer_bound_ledger_digest;
    std::vector<NativeCommitmentInput> model_commitments;
    bool authenticated = false;
};

struct AuthenticatedModelBlinders {
    std::string path;
    std::array<std::uint8_t, 32U> digest{};
    std::array<std::uint8_t, 32U> registry_digest{};
    std::vector<NativeModelBlinderInput> entries;
    bool authenticated = false;
};

// All three loaders are fail-closed.  They never create missing files,
// derive generators, or accept a caller-provided digest in lieu of parsing
// the authenticated resource itself.
bool load_authenticated_srs(const Config& config,
                             AuthenticatedSrsResource* resource,
                             std::string* error = nullptr);
bool load_authenticated_model_registry(const Config& config,
                                       const AuthenticatedSrsResource& srs,
                                       AuthenticatedModelRegistry* resource,
                                       std::string* error = nullptr);
bool load_authenticated_model_blinders(const Config& config,
                                       const AuthenticatedModelRegistry& registry,
                                       AuthenticatedModelBlinders* resource,
                                       std::string* error = nullptr);

}  // namespace zkdl_gat
