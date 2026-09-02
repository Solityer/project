#!/usr/bin/env python3
"""Short source-level guards for the point-only verifier boundary."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


proof = read("include/gatzk/protocol/proof.hpp")
verifier = read("include/gatzk/protocol/verifier.hpp")
codec = read("include/gatzk/protocol/proof_codec.hpp")
context_header = read("include/gatzk/protocol/verifier_context.hpp")
context_source = read("src/protocol/verifier_context.cpp")
main = read("src/main.cpp")
artifact = read("src/protocol/artifact_export.cpp")
codec_source = read("src/protocol/proof_codec.cpp")


verifier_context = proof.split("struct VerifierContext", 1)[1].split(
    "struct PiBindProof", 1
)[0]
for forbidden in (
    "model::ModelParameters",
    "crypto::KZGKeyPair",
    "public_fh_coefficient_materials",
    "checkpoint_bundle",
    "opening_polynomial",
):
    assert forbidden not in verifier_context, forbidden

assert "const VerifierContext& context" in verifier
assert "const VerifierContext& context" in codec
assert "load_verifier_context_from_registry(\n        const std::string &registry_path" in context_header
assert "load_verifier_context_from_registry(\n        const VerifierContext" not in context_header
assert "load_verifier_context_from_registry(\n        const ProtocolContext" not in context_header
assert "VerifierRegistrationArtifact" in context_header
assert "parse_verifier_registration_artifact" in context_source
assert "trusted_registry_digest" in context_source
assert "const auto artifact = parse_verifier_registration_artifact(\n            bytes, trusted_registry_digest);" in context_source
assert "parse_verifier_artifact" not in main
assert "make_verifier_public_inputs(context)" not in main
assert "parse_proof(context," not in main
assert "verify(context," not in main
assert "private_plain_value_count(context," not in main
assert "proof_contains_prover_material" in codec_source
assert "checkpoint_bundle_manifest_hash" not in artifact
assert "checkpoint_source_hash" not in artifact
assert '"source_tree_hash"' in artifact
assert '"binary_hash"' in artifact
assert '"build_identity"' in artifact
assert '"/columns/"' not in artifact
assert "integer_bound_ledger_digest" in context_source
assert "public_input_digest" in context_source
assert "security_budget_ledger_digest" in context_source
assert "security_budget_exact" in context_source
assert "tamper_test_case_map" in main
assert "registry_file_body" not in context_source
assert "read_registry_file" not in context_source
assert "parse_verifier_registration_artifact" in context_header
assert "serialize_verifier_registration_artifact" in context_header
runtime = read("tests/verifier_privacy_runtime.cpp")
for marker in (
    "canonical roundtrip",
    "single-byte artifact tamper",
    "artifact trailing byte",
    "non-canonical section order",
    "compiler section tamper",
    "untrusted digest",
    "aliases mutable source ownership",
):
    assert marker in runtime, marker
print("verifier privacy contract: PASS")
