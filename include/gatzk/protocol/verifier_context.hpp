#pragma once

#include <string>

#include "gatzk/protocol/proof.hpp"

namespace gatzk::protocol
{

    // Candidate registration is a setup-time operation.  It is allowed to
    // inspect a proving context, but its result is a self-contained artifact
    // and the formal loader below consumes only the authenticated artifact.
    VerifierRegistrationArtifact make_verifier_registration_artifact(
        const ProtocolContext &prover_context);

    std::vector<std::uint8_t> serialize_verifier_registration_artifact(
        const VerifierRegistrationArtifact &artifact);
    VerifierRegistrationArtifact parse_verifier_registration_artifact(
        const std::vector<std::uint8_t> &bytes);

    // Formal path: load a candidate VK_model/VK_static registry artifact,
    // validate its canonical digest and exact entry set against an already
    // sanitized public-input context, and require the separately supplied
    // trusted digest before constructing an independent verifier context.
    // This API never accepts a ProtocolContext.  Missing, malformed, or
    // unauthenticated registry material is a hard failure.
    VerifierContext load_verifier_context_from_registry(
        const std::string &registry_path,
        const std::string &trusted_registry_digest);

    // Registration is an explicit setup action.  It is provided so the owner
    // can create the registry after the canonical checkpoint/SRS exist; this
    // round does not invoke it.
    void write_verifier_registry(
        const ProtocolContext &prover_context,
        const std::string &registry_path);

    std::string canonical_verifier_registry_path(
        const ProtocolContext &prover_context);
    std::string canonical_verifier_registry_path(
        const VerifierContext &verifier_context);

    // Validate the complete point-only registry and all derived commitment
    // maps before parser/verifier code is allowed to consume the context.
    // This is intentionally independent of the proving context.
    bool validate_verifier_context(const VerifierContext &verifier_context);

    // Shape-only projection used by verifier execution.  It contains public
    // data and a VK copy, and never points back to a proving context.
    namespace detail
    {
        ProtocolContext make_verifier_execution_context(
            const VerifierContext &verifier_context);
    }

} // namespace gatzk::protocol
