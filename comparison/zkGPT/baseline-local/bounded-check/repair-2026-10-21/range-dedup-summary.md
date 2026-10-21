# zkGPT range relation and duplicate-forward repair

## Changes

- `gat_adapter/gat_proof.cpp` removes the redundant GKR `b²-b` residual for each range bit while retaining each range's linear reconstruction residual. The native Lasso `{0,1}` proof, `L_qry` table/provider binding, GKR input binding, openings, and independent verifier remain in the proof. The range emitter, symbolic shape collector, capacity counters, and materialized-plan checks now use the same reduced counts. Official layered relations and their MatMul/rounding descriptors use `zkgpt-gat-relation-v2`; sparse GKR keeps its own v1 schema. Compact readers and verifier statement reconstruction enforce the new relation schema. The outer proof artifact wire format remains v8.
- Formal proving now creates one fresh canonical forward trace and performs the existing tuple, range, division-profile, table, and ledger admission checks on that trace before building the plan. The public/untrusted `consume` path still performs full canonical relation recomputation and the same admission checks. Capacity preparation no longer runs another full GAT forward; its v11 report says `forward_value_admission_ready=false` and `forward_value_admission_mode=deferred_to_owned_forward_v1`. Formal proving must complete the value checks before it can create the private owner. Forward generation and admission have separate measured log fields, both inside the Prove stage.
- The smoke fixtures cover the new owner route through private context writing, proof generation, artifact readback and independent verification; rejection of a missing range-family commitment; rejection of an `L_qry` commitment-root mutation; and rejection of a non-binary `range_all` query by the native T2 table/provider binding.

Files changed for this repair:

- `gat_adapter/gat_proof.cpp`, `gat_adapter/gat_proof.hpp`, `gat_adapter/gat_cli.cpp`
- `gat_adapter/gat_forward.cpp`, `gat_adapter/gat_forward_internal.hpp`, `gat_adapter/gat_formal_internal.hpp`
- `src/proof_session.cpp`, `src/proof_session.hpp`
- `scripts/formal_preflight.py`
- `tests/gat_adapter_smoke.cpp`, `tests/native_lasso_range_smoke.cpp`, `tests/test_t2_capacity_consumer.py`

## Bounded checks

- The four requested targets built successfully in 66.25 seconds: `gat_adapter_smoke`, `native_lasso_range_smoke`, `zkgpt_gat_prover`, and `zkgpt_gat_verifier`.
- `gat_adapter_smoke --formal-private` passed in 5.37 seconds. It performed the fresh-forward owner path, private context write, complete tiny native proof, serialized artifact readback, independent verification, and the range-family and `com_L` tamper rejection checks.
- `gat_adapter_smoke --formal-owner-check` passed in 0.10 seconds. It kept the untrusted full-validation path and rejected a tampered trace at `consume`.
- `native_lasso_range_smoke` passed in 3.03 seconds. The new `range_all` T2 fixture rejected an `L_qry` vector containing field value 2 because it differed from the binary table/provider value.
- `python3 tests/test_t2_capacity_consumer.py` passed in 0.37 seconds using the built native CLI. It accepted capacity schema v11 only with the explicit deferred mode and false readiness flag, and checked the compact T2 and nonzero eligible L_qry LUT workspace fields.
- The formal fixture's plan construction exercised the exact emitted-versus-capacity gate-count assertion. No separate plan smoke, benchmark, or dataset run was performed.

The first run exposed fixture/command issues, all corrected within this bounded group: the initial smoke binary path omitted the `tests/` subdirectory; the first missing-family assertion used a slash-form label although the canonical family ID is `range_all`; and the first invalid Lasso fixture used a noncanonical slash-form family ID, followed by an assertion that required a later rejection reason even though the provider binding rejected it earlier. The original logs remain in `range-dedup-2026-10-21/`; the final corrected smoke runs pass. These were harness issues, not proof or verifier failures.

## Gate-count estimate and limits

The prior v9 report's conservative shape counts imply the removed range-bit gates account for about 60.424%–60.728% of PPI's old unary-plus-binary gate count and 54.059%–54.510% of OGBN-Arxiv's. The inputs were PPI range count 8,342,084,320–8,384,027,360 versus old U+B 27,611,830,352, and Arxiv 12,055,429,976–12,156,093,272 versus old U+B 44,601,192,180. These are estimates from the prior report, not new large-dataset measurements or runtime speedups.

No full dataset was run. The short checks establish local proof composition, fail-closed behavior, and the producer/consumer capacity contract; they do not establish PPI/OGBN completion time or the five-dataset 24-hour target.

## Detached run commands

From `/home/pzh/project/comparison/zkGPT`, launch only PPI and OGBN-Arxiv with persistent detached logs:

```bash
bash scripts/run_formal_detached.sh --datasets ppi ogbn_arxiv
bash scripts/formal_progress.sh --once
python3 scripts/stop_formal.py --launch-id <32-hex-launch-id>
```

Use the exact `launch_id` printed by the launcher for the stop command. No formal process was started by this repair.

## Logs

All build and check logs, including the initial failed fixture attempts, are under `baseline-local/bounded-check/repair-2026-10-21/range-dedup-2026-10-21/`.
