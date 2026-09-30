#pragma once

#include <string>

#include "gatzk/protocol/proof.hpp"

namespace gatzk::protocol
{

    VerifierRegistrationArtifact make_verifier_registration_artifact(const ProtocolContext &prover_context);

    std::vector<std::uint8_t> serialize_verifier_registration_artifact(const VerifierRegistrationArtifact &artifact);
    VerifierRegistrationArtifact parse_verifier_registration_artifact(const std::vector<std::uint8_t> &bytes);
    VerifierRegistrationArtifact parse_verifier_registration_artifact(const std::vector<std::uint8_t> &bytes, const std::string &trusted_registry_digest);

    VerifierContext load_verifier_context_from_registry(const std::string &registry_path, const std::string &trusted_registry_digest);

    void write_verifier_registry(const ProtocolContext &prover_context, const std::string &registry_path);

    std::string canonical_verifier_registry_path(const ProtocolContext &prover_context);
    std::string canonical_verifier_registry_path(const VerifierContext &verifier_context);

    bool validate_verifier_context(const VerifierContext &verifier_context);

    namespace detail
    {
        std::size_t canonical_static_table_serialization_size(const StaticTables &tables);
        std::string canonical_static_table_digest(const StaticTables &tables);
        std::string canonical_static_table_spec_hash(const ProtocolContext &context);
        std::string canonical_static_table_spec_hash(const VerifierContext &context);
        std::string canonical_range_and_scale_certificate(const ProtocolContext &context);
        std::string canonical_range_and_scale_certificate(const VerifierContext &context);
    }

    namespace detail
    {
        const ProtocolContext *validated_verifier_execution_context(const VerifierContext &verifier_context);

        ProtocolContext make_verifier_execution_context(const VerifierContext &verifier_context);
    }

}
