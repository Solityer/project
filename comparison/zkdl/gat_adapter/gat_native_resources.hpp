#pragma once

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

    struct Chunk {
        struct Segment {
            std::uint64_t segment_index = 0U;
            std::uint64_t logical_offset = 0U;
            std::uint64_t actual_length = 0U;
            std::uint64_t padded_length = 0U;
            std::uint64_t basis_offset = 0U;
            std::uint64_t basis_count = 0U;
        };
        std::string family_id;
        std::string chunk_id;
        std::uint64_t chunk_index = 0U;
        std::uint64_t point_offset = 0U;
        std::uint64_t point_count = 0U;
        std::uint64_t domain = 0U;

        std::uint64_t logical_domain = 0U;
        std::uint64_t basis_domain = 0U;
        std::uint64_t segment_size = 0U;
        std::uint64_t segment_count = 0U;
        std::vector<Segment> segments;
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
        std::uint64_t g_bytes = 0U;
        std::uint64_t h_bytes = 0U;
        std::uint64_t u_bytes = 0U;

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
    std::uint64_t logical_domain = 0U;
    std::uint64_t basis_domain = 0U;
    std::uint64_t segment_size = 0U;
    std::uint64_t segment_count = 0U;
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

bool load_authenticated_srs(const Config &config, AuthenticatedSrsResource *resource, std::string *error = nullptr);

bool select_authenticated_srs_chunk(AuthenticatedSrsResource *resource, const std::string &family_id, std::uint64_t required_domain,
                                    const AuthenticatedSrsResource::Chunk **chunk, std::string *error = nullptr);
bool load_authenticated_model_registry(const Config &config, const AuthenticatedSrsResource &srs, AuthenticatedModelRegistry *resource,
                                       std::string *error = nullptr);
bool load_authenticated_model_blinders(const Config &config, const AuthenticatedModelRegistry &registry, AuthenticatedModelBlinders *resource,
                                       std::string *error = nullptr);

}
