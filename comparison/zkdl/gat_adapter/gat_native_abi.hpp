#pragma once

// Plain, proof-stack-independent input ABI for the zkDL GAT bridge.
//
// This header is shared by the adapter runner and the bridge request types. It
// deliberately contains no native field, curve, or header-only proof types.
// Field values and curve points cross the seam in their canonical serialized
// forms; gat_native_core.cu is the only translation unit that decodes them
// into the official zkDL relation types.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace zkdl_gat {

using NativeFieldWire = std::array<std::uint32_t, 8U>;

// The canonical zkDL Fr wire is eight little-endian uint32 limbs.  A G1 point
// is serialized by zkdl::append_g1 as three coordinates of twelve limbs each
// (144 bytes); keeping the point opaque here prevents a second curve ABI.
constexpr std::size_t kNativeFieldWireBytes = 32U;
constexpr std::size_t kNativeG1WireBytes = 144U;

struct NativeTensorWitnessInput {
    std::string id;
    std::uint32_t visibility = 0U;
    std::uint32_t domain_size = 0U;
    std::vector<NativeFieldWire> values;
};

// Shape-only descriptors are safe for independent verification.  They carry
// no tensor values and are checked against the authenticated RelationPlan by
// the native core.
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

struct NativeCommitmentInput {
    std::string tensor_id;
    std::uint32_t visibility = 0U;
    std::uint32_t domain_size = 0U;
    // Exactly one canonical zkdl::append_g1 payload.
    std::vector<std::uint8_t> commitment;
};

struct NativeHidingRecordInput {
    // Exactly one canonical relation::append_hiding_record payload.  The
    // payload contains only public commitment/blinding/Schnorr data, never a
    // private tensor value.
    std::vector<std::uint8_t> encoded;
};

struct NativeRelationWitnessInput {
    // Prove requests carry all canonical field-MLE tensors.  Verify requests
    // may carry only public tensors (currently public_Y); the core rejects
    // private values in an independent verifier context.
    std::vector<NativeTensorWitnessInput> tensors;
    std::vector<NativeTensorShapeInput> tensor_shapes;

    // Optional canonical relation::append_operation payloads.  The native
    // core always reconstructs equations from the authenticated plan and
    // treats these bytes as a consistency witness only.
    std::vector<std::vector<std::uint8_t>> operations;
    std::vector<NativeCommitmentInput> registered_model_commitments;
    std::vector<NativeHidingRecordInput> hiding_commitment_records;
    // Each entry is exactly one canonical zkdl::append_g1 payload.
    std::vector<std::vector<std::uint8_t>> generators;

    std::string registry_path;
    std::array<std::uint8_t, 32U> srs_digest{};
    std::array<std::uint8_t, 32U> registry_digest{};
    bool srs_authenticated = false;
    bool registry_authenticated = false;
};

}  // namespace zkdl_gat
