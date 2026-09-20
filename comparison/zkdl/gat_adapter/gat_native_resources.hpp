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
    // A manifest may contain one authenticated prefix basis per relation
    // family/chunk.  `generators` below is only the deterministic common
    // execution chunk selected for the current request; chunks are never
    // concatenated into an unbounded global SRS domain.
    struct Chunk {
        std::string family_id;
        std::string chunk_id;
        std::uint64_t chunk_index = 0U;
        std::uint64_t point_offset = 0U;
        std::uint64_t point_count = 0U;
        std::uint64_t domain = 0U;
        std::uint64_t degree = 0U;
        std::uint32_t point_size = 0U;
        std::string generator_id;
        std::string basis_digest;
        std::string g_path;
        std::string h_path;
        std::string u_path;
        std::string g_digest;
        std::string h_digest;
        std::string u_digest;
        // Canonical digest of the G frame; retained separately so the
        // selected chunk identity is explicit in resource metadata.
        std::string generator_digest;
        std::vector<std::vector<std::uint8_t>> generators;
        std::vector<std::uint8_t> blinding_generator;
        std::vector<std::uint8_t> u_generator;
        bool materialized = false;
        bool authenticated = false;
    };
    std::string chunking_rule;
    std::string generator_id;
    std::string basis_digest;
    std::vector<Chunk> chunks;
    std::string selected_family_id;
    std::string selected_chunk_id;
    std::uint64_t selected_chunk_index = 0U;
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

// Select the smallest authenticated prefix chunk that covers one canonical
// relation-family domain.  The returned pointer remains owned by `resource`.
// Callers must use the common chunk material exposed by `resource` for the
// native proof context; this helper authenticates family/chunk choice and
// materializes its independent authenticated basis.  The caller that binds
// it to the common proof basis must perform the exact-prefix comparison.
bool select_authenticated_srs_chunk(AuthenticatedSrsResource* resource,
                                     const std::string& family_id,
                                     std::uint64_t required_domain,
                                     const AuthenticatedSrsResource::Chunk** chunk,
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
