#!/usr/bin/env python3
"""Short source-level guards for canonical trace/static cache identities."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
trace = (ROOT / "src/protocol/trace.cpp").read_text(encoding="utf-8")

assert '"zkgat:trace-cache-identity"' in trace
assert 'return "sha256:" + sha256_hex(bytes);' in trace
assert "context.compiler_artifact->instance_canonical_bytes" in trace
assert "context.local.public_input" in trace
assert "context.local.features" in trace
assert "context.public_polynomials" in trace
assert "context.static_commitments" in trace
assert "artifacts->identity = cache_key" in trace
assert "it->second->identity == cache_key" in trace
assert "std::hash<std::string>" not in trace
assert "fnv1a64" not in trace

print("trace cache contract: PASS")
