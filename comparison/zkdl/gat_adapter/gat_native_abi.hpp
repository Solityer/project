#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "gat_family_storage.hpp"

namespace zkdl_gat {

using NativeFieldWire = std::array<std::uint32_t, 8U>;

constexpr std::size_t kNativeFieldWireBytes = 32U;
constexpr std::size_t kNativeG1WireBytes = 48U;

struct NativeTensorWitnessInput {
    std::string id;
    std::uint32_t visibility = 0U;
    std::uint32_t domain_size = 0U;
    std::vector<NativeFieldWire> values;
    std::shared_ptr<const NativeFamilyStorage> storage;
    std::vector<std::uint64_t> logical_shape;
    std::vector<std::uint64_t> padded_shape;
};

struct NativeTensorShapeInput {
    std::string id;
    std::uint32_t domain_size = 0U;
    std::uint32_t visibility = 0U;
    std::vector<std::uint64_t> logical_shape;
    std::vector<std::uint64_t> padded_shape;
    std::string mle_order;
    std::string stack_axis;
    std::string mapping;
    std::string padding_rule;
    std::string registry_id;
};

struct NativeSegmentCommitmentInput {
    std::uint32_t segment_index = 0U;
    std::uint64_t logical_offset = 0U;
    std::uint32_t actual_length = 0U;
    std::uint32_t padded_length = 0U;
    std::uint32_t padding_length = 0U;
    std::array<std::uint8_t, 32U> frame_digest{};
    std::vector<std::uint8_t> commitment;
};

struct NativeCommitmentInput {
    std::string tensor_id;
    std::uint32_t visibility = 0U;
    std::uint32_t domain_size = 0U;

    std::vector<std::uint8_t> commitment;
    std::uint32_t segment_binding_version = 1U;
    std::uint32_t segment_size = 0U;
    std::uint32_t segment_count = 0U;
    std::array<std::uint8_t, 32U> srs_manifest_digest{};
    std::array<std::uint8_t, 32U> basis_digest{};
    std::array<std::uint8_t, 32U> segment_commitment_digest{};
    std::vector<NativeSegmentCommitmentInput> segment_commitments;
};

struct NativeHidingRecordInput {

    std::vector<std::uint8_t> encoded;
};

struct NativeModelBlinderInput {
    std::string tensor_id;
    NativeFieldWire blinder{};
    std::array<std::uint8_t, 32U> registry_digest{};
};

struct NativeSortedLookupProviderInput {
    std::string operation_id;
    std::string table_ref;
    std::string query_ref;
    std::string merge_ref;
    std::string scratch_path;
    std::string scratch_digest;
    std::string layout_digest;
    std::vector<std::string> commitment_roles;
    std::uint64_t table_logical_rows = 0U;
    std::uint64_t table_domain = 0U;
    std::uint64_t query_logical_rows = 0U;
    std::uint64_t query_domain = 0U;
    std::uint64_t merge_logical_rows = 0U;
    std::uint64_t merge_domain = 0U;
    std::uint64_t segment_size = 0U;
    std::uint32_t x_bits = 0U;
    std::uint32_t y_bits = 0U;
    std::string key_max_u128;
    bool owner_only = true;
    bool active_prefix = true;
    bool canonical_padding = true;
};

struct NativeRelationWitnessInput {

    // Measured by the canonical native witness builder for sorted-lookup
    // protocol auxiliary witness generation/sorting. Explicit file I/O is
    // removed before this value is handed to the prover core.
    double sorted_lookup_protocol_construction_seconds = 0.0;

    // Net proof-specific SignedRoundDiv auxiliary construction and relation-view
    // reader computation. Family storage and digest I/O are outside this value.
    double proof_auxiliary_construction_seconds = 0.0;

    std::vector<NativeTensorWitnessInput> tensors;
    std::vector<NativeTensorShapeInput> tensor_shapes;

    std::vector<std::vector<std::uint8_t>> operations;
    std::vector<NativeCommitmentInput> registered_model_commitments;
    std::vector<NativeHidingRecordInput> hiding_commitment_records;
    std::vector<NativeModelBlinderInput> model_blinders;

    std::vector<NativeSortedLookupProviderInput> sorted_lookup_providers;

    std::vector<std::vector<std::uint8_t>> generators;

    std::vector<std::vector<std::uint8_t>> ipa_generators;
    std::vector<std::uint8_t> ipa_blinding_generator;
    std::vector<std::uint8_t> ipa_u_generator;

    std::string registry_path;
    std::array<std::uint8_t, 32U> srs_digest{};
    std::array<std::uint8_t, 32U> registry_digest{};

    std::array<std::uint8_t, 32U> model_blinder_digest{};
    bool srs_authenticated = false;
    bool registry_authenticated = false;
    bool model_blinders_authenticated = false;
};

}
