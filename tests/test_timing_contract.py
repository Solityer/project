#!/usr/bin/env python3
"""Source-level guards for the formal verify timing boundary."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = (ROOT / "src/main.cpp").read_text(encoding="utf-8")
TRACE = (ROOT / "src/protocol/trace.cpp").read_text(encoding="utf-8")
PROOF = (ROOT / "include/gatzk/protocol/proof.hpp").read_text(encoding="utf-8")


# The exclusive prove timer starts only after the canonical forward, reference
# hashes, public metadata, and public selector preparation, immediately before
# the first P_H dynamic commitment branch.
trace_build_start = TRACE.index(
    "TraceArtifacts build_multihead_trace(const ProtocolContext &context, RunMetrics *metrics)"
)
trace_p_h_branch = TRACE.index(
    "if (static_artifacts.full_feature_identity)",
    trace_build_start,
)
trace_prove_start = TRACE.index(
    "timing::start_prove_timer(metrics);",
    trace_build_start,
)
assert TRACE.count("timing::start_prove_timer(metrics);") == 1
assert TRACE.index("const auto forward_artifacts = cached_forward_artifacts(", trace_build_start) < trace_prove_start
assert TRACE.index("record_public_output_hashes(", trace_build_start) < trace_prove_start
assert TRACE.index("trace.public_metadata.public_output = canonical_public_output_descriptor(", trace_build_start) < trace_prove_start
assert trace_prove_start < trace_p_h_branch


# The prove clock is closed by the canonical serializer boundary.  The formal
# totals must come from the timing state machine, not residual profile math.
prove_start = MAIN.index("progress(mode_label + \":prove:start\");")
serialize_call = MAIN.index(
    "const auto proof_bytes = gatzk::protocol::serialize_proof(proof);",
    prove_start,
)
serialization_end = MAIN.index(
    "const auto serialization_end = std::chrono::steady_clock::now();",
    serialize_call,
)
serialization_metric = MAIN.index(
    "metrics.proof_serialization_ms =",
    serialization_end,
)
finish_prove = MAIN.index(
    "gatzk::protocol::timing::finish_prove_timer(&metrics);",
    serialization_metric,
)
assert serialization_end < serialization_metric < finish_prove
assert MAIN.index(
    "gatzk::protocol::timing::assert_prove_finished(metrics);",
    finish_prove,
) > finish_prove
assert MAIN.count("gatzk::protocol::timing::finish_prove_timer(&metrics);") == 1
assert MAIN.count("gatzk::protocol::timing::assert_prove_finished(metrics);") == 1
for forbidden in (
    "prover_pipeline_end",
    "prover_pipeline_ms",
    "prove_without_finalize_ms",
    "prove_overlap_tolerance_ms",
    "metrics.commitment_time_ms =",
    "metrics.prove_time_ms =",
    "metrics.prove_finalize_ms =",
    "metrics.prove_accounted_ms =",
    "metrics.prove_accounting_gap_ms =",
):
    assert forbidden not in MAIN, forbidden


# The reference timer ends after all data used by final acceptance is ready;
# verify_end follows final acceptance directly and excludes bookkeeping.
verify_start = MAIN.index("const auto verify_start = std::chrono::steady_clock::now();")
tamper_start = MAIN.index(
    "const auto tamper_start = std::chrono::steady_clock::now();",
    verify_start,
)
verify_block = MAIN[verify_start:tamper_start]


parse_call = verify_block.index(
    "const auto parsed_proof = gatzk::protocol::parse_proof(verifier_context, proof_bytes);"
)
parse_end = verify_block.index(
    "const auto proof_parse_end = std::chrono::steady_clock::now();",
    parse_call,
)
verify_start_statement = "const auto verify_start = std::chrono::steady_clock::now();"
assert verify_block[len(verify_start_statement):parse_call].strip() == ""
parse_statement = (
    "const auto parsed_proof = "
    "gatzk::protocol::parse_proof(verifier_context, proof_bytes);"
)
assert verify_block[parse_call + len(parse_statement):parse_end].strip() == ""
parse_metric = verify_block.index("metrics.proof_parse_ms =", parse_end)
parse_metric_end = verify_block.index(
    "trace.proof_public_output_hash =",
    parse_metric,
)
digest = verify_block.index("gatzk::protocol::canonical_output_digest(", parse_metric_end)
assert parse_call < parse_end < parse_metric < parse_metric_end < digest

reference_end = verify_block.index(
    "const auto reference_end = std::chrono::steady_clock::now();"
)
final_acceptance = verify_block.index(
    "gatzk::protocol::final_formal_run_accepted("
)
verify_end = verify_block.index(
    "const auto verify_end = std::chrono::steady_clock::now();"
)
assert reference_end < final_acceptance < verify_end
final_acceptance_end = verify_block.index(");", final_acceptance) + 2
assert verify_block[final_acceptance_end:verify_end].strip() == ""
reference_metric = verify_block.index(
    "metrics.reference_consistency_ms =",
    reference_end,
)
assert verify_end < reference_metric

assert "private_plain_value_count(verifier_context, parsed_proof)" in verify_block
assert "metrics.verify_point_group_registry_ms" in verify_block
assert "metrics.payload_audit_ms + metrics.reference_consistency_ms" in MAIN
assert "metrics.verify_point_group_registry_ms + metrics.verify_quotient_ms" in MAIN

# Registration/VK loading and the independent reference forward happen before
# this source-level verify block; tamper checks and export happen after it.
assert MAIN.index("load_verifier_context_from_registry(") < verify_start
assert MAIN.index("const auto tamper_start", verify_start) > verify_end
assert "independent_quantized_reference_forward(" not in verify_block

print("verify timing contract: PASS")
