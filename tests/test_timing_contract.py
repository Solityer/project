#!/usr/bin/env python3
"""Source-level guards for the formal verify timing boundary."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = (ROOT / "src/main.cpp").read_text(encoding="utf-8")
TRACE = (ROOT / "src/protocol/trace.cpp").read_text(encoding="utf-8")
PROOF = (ROOT / "include/gatzk/protocol/proof.hpp").read_text(encoding="utf-8")
AGGREGATE = (ROOT / "scripts/aggregate_secure_results.py").read_text(encoding="utf-8")
EXPORT = (ROOT / "scripts/export_benchmark_table.py").read_text(encoding="utf-8")


CANONICAL_TIMING_SCHEMA = "exclusive_commit_prove_protocol_verify"
assert f'"{CANONICAL_TIMING_SCHEMA}"' in PROOF
assert f'"{CANONICAL_TIMING_SCHEMA}"' in AGGREGATE
assert f'"{CANONICAL_TIMING_SCHEMA}"' in EXPORT


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


# The protocol verify clock starts at parse_proof and ends at the protocol
# verifier return.  Output digest, external payload audit, reference
# consistency, final acceptance, tamper checks, and export are outside it.
verify_start = MAIN.index("const auto verify_start = std::chrono::steady_clock::now();")
verify_end = MAIN.index(
    "const auto verify_end = std::chrono::steady_clock::now();",
    verify_start,
)
tamper_start = MAIN.index(
    "const auto tamper_start = std::chrono::steady_clock::now();",
    verify_start,
)
protocol_verify_block = MAIN[verify_start:verify_end]
post_protocol_block = MAIN[verify_end:tamper_start]


parse_call = protocol_verify_block.index(
    "const auto parsed_proof = gatzk::protocol::parse_proof(verifier_context, proof_bytes);"
)
parse_end = protocol_verify_block.index(
    "const auto proof_parse_end = std::chrono::steady_clock::now();",
    parse_call,
)
verify_start_statement = "const auto verify_start = std::chrono::steady_clock::now();"
assert protocol_verify_block[len(verify_start_statement):parse_call].strip() == ""
parse_statement = (
    "const auto parsed_proof = "
    "gatzk::protocol::parse_proof(verifier_context, proof_bytes);"
)
assert protocol_verify_block[parse_call + len(parse_statement):parse_end].strip() == ""
parse_metric = protocol_verify_block.index("metrics.proof_parse_ms =", parse_end)
assert parse_call < parse_end < parse_metric
verifier_call = protocol_verify_block.index(
    "const auto verifier_accepted =",
    parse_metric,
)
assert parse_metric < verifier_call
assert "gatzk::protocol::canonical_output_digest(" not in protocol_verify_block
assert "private_plain_value_count(verifier_context, parsed_proof)" not in protocol_verify_block
assert "payload_audit_ms" not in protocol_verify_block
assert "reference_consistency" not in protocol_verify_block
assert "final_formal_run_accepted(" not in protocol_verify_block
assert "metrics.verify_point_group_registry_ms" in MAIN

verify_metric = post_protocol_block.index("metrics.verify_time_ms =")
assert verify_metric >= 0
assert "gatzk::protocol::canonical_output_digest(" in post_protocol_block
assert "private_plain_value_count(verifier_context, parsed_proof)" in post_protocol_block
assert "metrics.payload_audit_ms" in post_protocol_block
assert "metrics.reference_consistency_ms" in post_protocol_block
assert "gatzk::protocol::final_formal_run_accepted(" in post_protocol_block

accounted_start = MAIN.index("auto verify_accounted_ms =")
accounted_end = MAIN.index("};", accounted_start)
accounted_block = MAIN[accounted_start:accounted_end]
assert "payload_audit_ms" not in accounted_block
assert "reference_consistency_ms" not in accounted_block
misc_start = MAIN.index("metrics.verify_misc_ms =", verify_end)
misc_end = MAIN.index(";", misc_start)
misc_block = MAIN[misc_start:misc_end]
assert "payload_audit_ms" not in misc_block
assert "reference_consistency_ms" not in misc_block

# Registration/VK loading and the independent reference forward happen before
# this source-level verify block; tamper checks and export happen after it.
assert MAIN.index("load_verifier_context_from_registry(") < verify_start
assert MAIN.index("const auto tamper_start", verify_start) > verify_end
assert "independent_quantized_reference_forward(" not in protocol_verify_block

print("verify timing contract: PASS")
