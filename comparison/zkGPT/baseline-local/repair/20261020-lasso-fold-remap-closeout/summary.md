# Targeted closeout

## Final source changes

Seven net source/test files changed across the paired-rank and Lasso work:

- `src/circuit.cpp`: fuses binary U/V remapping into one gate pass and stores both subset-rank lanes in paired word-group prefixes, without allocating a second full-size rank array.
- `gat_adapter/gat_proof.cpp`: updates capacity accounting for paired rank slots and applies the missing sumcheck challenge in the smoke-only `exercise_official_primitives` fixture.
- `scripts/formal_preflight.py`: uses the paired-rank slot formula when checking expected capacity.
- `tests/native_masked_layered_gkr_smoke.cpp`: expands the bitmap subset fixture to exercise five packed words, an odd-word count, a partial final word, distinct U/V sets, duplicates, and layer-0/prior-layer references.
- `tests/gat_proof_plan_smoke.cpp`: checks the paired-rank capacity count.
- `src/native_lasso.cpp`: reuses `MatrixRowFold` values to compute raw scalar openings, removing the duplicate sparse `evaluate_mle` scan while preserving masked Hyrax openings.
- `tests/native_lasso_range_smoke.cpp`: checks sparse-versus-row-fold scalar equivalence for non-T2 and T2 padded cases at non-Boolean points.

The `gat_proof_plan_smoke` crash was caused by the smoke helper calling `sumcheckFinalize1` before applying the challenge after `sumcheckUpdate1`. The helper now calls `sumcheckApplyChallenge(challenge)`. This helper is called by `tests/gat_proof_plan_smoke.cpp` only; the formal prepare path does not call it, and the production verifier already applies each challenge. A speculative `src/prover.cpp` guard change was reverted and is not part of the net changes.

## Bounded checks

- The initial build reported all four targets built (`native_lasso_range_smoke`, `native_masked_layered_gkr_smoke`, `gat_proof_plan_smoke`, and `zkgpt_gat_prover`), but its shell pipeline returned 1 because `tee` could not open the then-missing repair directory. The complete build output was not captured; see `build-summary.txt`.
- The initial selected smoke group passed `native_lasso_range_smoke` (2.72 s) and `native_masked_layered_gkr_smoke` (4.09 s); `gat_proof_plan_smoke` failed before the fixture repair. The original logs remain in `ctest.log` and `gat-proof-plan-debug.log`.
- `scripts/formal_preflight.py` syntax compilation passed in 0.02 s; see `formal-preflight-syntax.log`.
- After the fixture repair, rebuilding `gat_proof_plan_smoke` and `zkgpt_gat_prover` passed; see `challenge-transition-final-build.log`.
- The single permitted retest of `gat_proof_plan_smoke` passed (1/1, 2.03 s); see `challenge-transition-final-plan-retest.log`. The already-passing Lasso and GKR tests were not rerun.

No complete dataset was run, and the Python capacity consumer was not dynamically executed. These bounded checks do not establish a five-dataset runtime guarantee or a 24-hour completion guarantee.
