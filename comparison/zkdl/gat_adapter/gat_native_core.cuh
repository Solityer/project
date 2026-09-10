#pragma once

#include "gat_native_bridge.hpp"
#include "../zkrelation-proof.cuh"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace zkdl_gat {

/*
 * The runner currently owns the canonical graph/model trace objects, but it
 * does not yet expose their field-MLE material to the native bridge.  This
 * explicit bundle is the only witness seam accepted by the native core.  It
 * is deliberately not inferred from int64/double values inside the core:
 * callers must supply canonical FrWire tensors and authenticated generators.
 */
struct NativeRelationWitness {
    std::vector<relation::TensorWitness> tensors;
    struct TensorShape {
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
    // Shape-only public registry used by the independent verifier; it never
    // carries private tensor values.
    std::vector<TensorShape> tensor_shapes;
    // Exact stage-qualified descriptors are supplied by the canonical
    // adapter.  The core never invents a selector/reindex mapping when this
    // vector is present.
    std::vector<relation::OperationDescriptor> operations;
    // Commitments for visibility=2 model tensors, parsed from the
    // authenticated registry/VK map by the adapter.  The core compares these
    // points with the generated canonical commitments; it never invents a
    // model commitment from metadata.
    std::vector<relation::CommitmentWire> registered_model_commitments;
    // One native CpuHidingCommitment::record() for every model/private tensor.
    // The record carries only the value commitment, public blinding data, and
    // its Schnorr response; the core verifies the blinder knowledge before
    // opening the underlying value commitment.
    std::vector<zkdl::CommitmentRecord> hiding_commitment_records;
    std::vector<zkdl::CpuG1Point> generators;
    std::string registry_path;
    std::array<zkdl::byte, 32> srs_digest{};
    std::array<zkdl::byte, 32> registry_digest{};
    bool srs_authenticated = false;
    bool registry_authenticated = false;
};

bool zkdl_gat_native_core_prove(const NativeProveRequest& request,
                                NativeProofResult* result,
                                std::string* error);

bool zkdl_gat_native_core_verify(const NativeVerifyRequest& request,
                                 std::string* error);

/* Pure relation entry points used by short no-resource tests and by the
 * bridge once the runner has populated NativeRelationWitness. */
bool zkdl_gat_relation_prove(const relation::ProverContext& context,
                             relation::Proof* proof, std::string* error = nullptr);
bool zkdl_gat_relation_verify(const relation::Proof& proof,
                              const relation::VerifierContext& context,
                              std::string* error = nullptr);

}  // namespace zkdl_gat
