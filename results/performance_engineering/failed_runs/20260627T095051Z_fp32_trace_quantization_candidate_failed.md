# fp32 Trace Quantization Candidate Failed

- generated_at: 2026-06-27T09:50:51Z
- scope: main experiment only
- comparison_touched: no
- candidate: restore `fp32_bundle_export` trace quantization for `S/Z` lookup queries instead of integer raw-matmul overrides
- intended_fix: remove OGBN LReLU table escape caused by formal raw span expansion
- files_temporarily_changed: `src/protocol/trace.cpp`
- verification_result: failed
- failing_command: `GATZK_RUN_FULL_SECURITY_TESTS=1 ./build/gatzk_tests`
- failure: `opening mismatch for t_in: scaled claimed value does not match committed polynomial`
- diagnosis: current projection constraints are coupled to raw-field projection witnesses; changing only `S/Z` trace quantization breaks proof-equivalence.
- retained: no
- rollback_status: reverted before further work
- official_result_pollution: no official timing or final export was run from this candidate
