#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "gat_ezkl/checkpoint.hpp"
#include "gat_ezkl/dataset.hpp"

namespace gat_ezkl {

// All formal identities use SHA-256 over a domain-separated sequence of
// length-prefixed fields.  Integer fields are fixed-width little endian.
std::string formal_sha256_bytes(
    const std::string& domain,
    const std::vector<std::pair<std::string, std::vector<std::uint8_t>>>& fields);
std::string formal_sha256_text(const std::string& domain, const std::string& role,
                               const std::string& text);
std::string formal_sha256_file(const std::string& domain, const std::string& role,
                               const std::filesystem::path& path);
std::string formal_hash_checkpoint(
    const std::filesystem::path& manifest,
    const std::filesystem::path& tensors);
std::string formal_hash_graph(const NormalizedGraph& graph);
std::string formal_hash_shape(const NormalizedGraph& graph, const CheckpointBundle& bundle);
std::string formal_hash_metadata(const std::string& metadata_json);

}  // namespace gat_ezkl
