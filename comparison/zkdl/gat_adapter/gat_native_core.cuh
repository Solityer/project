#pragma once

#include "gat_native_bridge.hpp"
#include "../zkrelation-proof.cuh"

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace zkdl_gat {

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

    std::vector<TensorShape> tensor_shapes;

    std::vector<relation::OperationDescriptor> operations;
    std::vector<NativeSortedLookupProviderInput> sorted_lookup_providers;

    std::vector<relation::CommitmentWire> registered_model_commitments;

    std::vector<zkdl::CommitmentRecord> hiding_commitment_records;

    std::map<std::string, zkdl::FrWire> blinding_scalars;
    std::vector<zkdl::CpuG1Point> generators;

    std::vector<zkdl::CpuG1Point> ipa_generators;
    zkdl::CpuIpaBasisContext authenticated_ipa_basis;
    zkdl::CpuG1Point ipa_blinding_generator;
    zkdl::CpuG1Point ipa_u_generator;
    std::string registry_path;
    std::array<zkdl::byte, 32> srs_digest{};
    std::array<zkdl::byte, 32> registry_digest{};
    bool srs_authenticated = false;
    bool registry_authenticated = false;
};

bool zkdl_gat_native_core_prove(const NativeProveRequest &request, NativeProofResult *result, std::string *error);

bool zkdl_gat_native_core_verify(const NativeVerifyRequest &request, std::string *error);

bool zkdl_gat_relation_prove(const relation::ProverContext &context, relation::Proof *proof, std::string *error = nullptr);
bool zkdl_gat_relation_verify(const relation::Proof &proof, const relation::VerifierContext &context, std::string *error = nullptr);

}
