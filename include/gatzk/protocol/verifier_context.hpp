#pragma once

#include <string>

#include "gatzk/protocol/proof.hpp"

namespace gatzk::protocol
{

    // Construct a point-only verifier view for short tests and migration
    // checks.  This path marks the result as an in-process transition and is
    // therefore rejected by the formal main verifier; it still performs the
    // complete canonical registry validation.
    VerifierContext make_verifier_context(const ProtocolContext &prover_context);

    // Copy only public verifier inputs (dataset/features, shape, domains,
    // public polynomials and VK).  No model/static commitment points are
    // copied here; the formal loader hydrates those exclusively from the
    // authenticated registry artifact.
    VerifierContext make_verifier_public_inputs(
        const ProtocolContext &prover_context);

    // Candidate registration is a setup-time operation.  It is allowed to
    // inspect a proving context, but its result is a self-contained artifact
    // and the formal loader below never accepts that context (or a transition
    // verifier projection).
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

    // Internal compatibility projection used by the existing proving/schema
    // implementation.  It contains only shape/public data and a VK copy; it
    // never points back to the original proving context.
    namespace detail
    {
        ProtocolContext make_verifier_execution_context(
            const VerifierContext &verifier_context);
    }

} // namespace gatzk::protocol
