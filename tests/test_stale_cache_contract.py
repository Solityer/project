#!/usr/bin/env python3
"""Lightweight source contracts for formal execution-cache identities.

This test intentionally does not construct a proof or a large dataset.  It
guards the cache boundary against incomplete shape/endpoint-only keys and against
proof-local trace values being reused after a new Fiat--Shamir nonce.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACE = (ROOT / "src/protocol/trace.cpp").read_text(encoding="utf-8")
PROVER = (ROOT / "src/protocol/prover.cpp").read_text(encoding="utf-8")
ARTIFACT_EXPORT = (ROOT / "src/protocol/artifact_export.cpp").read_text(encoding="utf-8")


def section(source: str, start: str, end: str) -> str:
    begin = source.index(start)
    finish = source.index(end, begin)
    return source[begin:finish]


model_key = section(
    TRACE,
    "std::string quantized_model_cache_key(",
    "struct HiddenHeadStaticQuantizedParameters",
)
assert '"zkgat:quantized-model-cache"' in TRACE
assert "append_cache_model_identity(&bytes, context.model)" in model_key
assert "config.quantization.canonical_serialization()" in model_key or "append_cache_config_identity" in model_key
assert "output_bias_fp" in TRACE
assert "config.verifier_registry_digest" in TRACE
assert "identity == cache_key" in section(
    TRACE,
    "const HiddenHeadStaticQuantizedParameters &cached_quantized_hidden_head_parameters(",
    "struct HiddenQuantizedWitness",
)
assert "payload_digest" in model_key or "payload_digest" in TRACE


registry_digest_assignment = section(
    ARTIFACT_EXPORT,
    "const auto verifier_registry_digest =",
    "const auto source_manifest_path =",
)
assert "verifier_context.authenticated_registry_digest" in registry_digest_assignment
assert "registry.canonical_digest()" not in registry_digest_assignment


lookup = section(
    TRACE,
    "std::string value_index_cache_key(",
    "std::vector<FieldElement> multiplicity_from_indices(",
)
assert '"zkgat:lookup-index:value"' in lookup
assert '"zkgat:lookup-index:pair"' in lookup
assert '"zkgat:lookup-table:single"' in lookup
assert '"zkgat:lookup-table:pair"' in lookup
assert "append_cache_field_vector(&bytes, values)" in lookup
assert "append_cache_field(&bytes, left)" in lookup
assert "append_cache_field(&bytes, right)" in lookup
assert "first=" not in lookup and "last=" not in lookup

table_cache = section(
    TRACE,
    "cached_single_lookup_table_shared(",
    "struct SingleLookupArtifacts",
)
assert "cache_namespace" in table_cache
assert "payload_digest" in table_cache
assert "field_vector_payload_digest" in table_cache
assert "values.size() > domain_size" in table_cache


memo = section(
    PROVER,
    "class EvaluationMemoization",
    "std::shared_ptr<const mcl::Ntt<mcl::Fr>> ntt_plan",
)
assert "proof_cache_key_" in memo
assert '"|proof_nonce="' in memo
assert "trace_cache_reuse_enabled_" in memo
assert "proof_cache_key_," in memo
assert "trace.public_metadata.proof_nonce.empty()" in memo

print("stale cache contract: PASS")
