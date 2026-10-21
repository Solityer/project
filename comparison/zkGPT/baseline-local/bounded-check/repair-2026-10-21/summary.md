# zkGPT bounded repair handoff — 2026-10-21

## Scope and result

This repair targets the PPI/OGBN-Arxiv formal path after the prior subset launch stopped before producing a PPI proof. It removes repeated work and the compact T2 Booleanity working-set bottleneck while preserving the native proof protocol, complete dataset computation, public validation, and independent verification. No formal dataset, benchmark, or 24-hour end-to-end run was executed, so completion time and full-dataset success remain unverified.

## Final source and test files changed

- `src/gat_formal_internal.hpp`, `src/gat_proof.cpp`, `src/gat_cli.cpp`, `tests/gat_adapter_smoke.cpp`: the private formal owner binds the exact validated plan and context. Initial full plan validation and the distinct context-layer metadata checks remain, each once (five large gate-array passes reduced to two). The formal-owned writer/prover reuses those checks only for the exact owner-bound plan. Public writer/prover and independent verifier paths retain full validation. Added stage boundary timing logs.
- `src/native_lasso.cpp`, `tests/native_lasso_range_smoke.cpp`: compact H16 T2 Booleanity computes the same first four query-round messages without materializing the full Q-element field column. Logical witness-domain accounting remains Q fields; the physical column is Q/16 fields for q>=4. At the prior PPI/OGBN shapes, this changes the column from 256/512 GiB to 16/32 GiB, plus the separately accounted histogram/equality-table transient. The native prover now emits per-family Booleanity begin/complete timing markers. The range smoke accepts fail-closed rejection of a tampered transcript during Fiat–Shamir reconstruction and otherwise requires the verifier to reject it.
- `scripts/formal_preflight.py`, `tests/test_t2_capacity_consumer.py`: producer and Python consumer capacity semantics are synchronized, including logical witness bytes, compact physical T2 column, peak Booleanity workspace, mapped bytes, and standard-heap transient. An initial capacity-fixture run exposed a `KeyError` because the newly checked Boolean field was read from the integer-only `values` map. All added Boolean accesses now read the strict-validated `workspace` object.
- `scripts/formal_progress.py`, `tests/test_formal_progress.py`: read-only progress reports a stale active launch as `NO_LIVE_OWNER`/interrupted instead of displaying a stale `RUNNING` stage indefinitely; it re-reads status for terminal-state races and does not mutate status or send signals.

## Bounded validation actually performed

Logs are preserved in this directory. The first failures are retained alongside their corrected runs.

- Build passed once for `gat_adapter_smoke`, `native_lasso_range_smoke`, `zkgpt_gat_prover`, and `zkgpt_gat_verifier` with `--parallel 2` (`build.log`).
- `gat_adapter_smoke` passed, 2.63 s (`ctest.log`).
- The first `native_lasso_range_smoke` attempt rejected a tampered Fiat–Shamir transcript during reconstruction. The fixture was corrected to accept this fail-closed rejection, while still requiring verifier rejection if reconstruction succeeds. Only that target was rebuilt and rerun; it passed, 2.82 s (`native-lasso-rebuild.log`, `native-lasso-ctest-fix.log`).
- The formal-progress unit test passed, 2 cases (`formal-progress-test.log`).
- The first real tiny producer-to-consumer capacity fixture failed on the `t2_booleanity` lookup described above (`t2-capacity-consumer-test.log`). After correcting the workspace lookup, its only rerun passed, 1 case in 0.294 s (`t2-capacity-consumer-fix.log`). It exercises the native CLI on a tiny temporary fixture; it is not a PPI/OGBN run.

No full dataset or benchmark was run. These bounded checks do not establish that PPI and OGBN-Arxiv finish within 24 hours or that their complete proofs independently verify.

## Launch, progress, and safe stop commands

The detached launcher is self-backgrounding (`nohup setsid`) and stores the console and status under `baseline-local/formal-detached/<launch_id>/`; per-session stage logs are under `baseline-local/formal-owner-logs/<session_id>/`.

```bash
cd /home/pzh/project/comparison/zkGPT
./scripts/run_formal_detached.sh --datasets ppi ogbn_arxiv
```

Use the printed launch ID to inspect progress once or continuously:

```bash
./scripts/formal_progress.sh --once
./scripts/formal_progress.sh
```

To stop only that exact launch, pass the printed 32-character launch ID. The stop helper checks recorded supervisor/child process identity and uses PID-bound `pidfd` SIGTERM; it refuses when the exact live owner cannot be validated.

```bash
python3 scripts/stop_formal.py --launch-id <launch_id>
```

## Existing stale launch and remaining operational boundary

The current `latest_launch.json` points to launch `8ac696be4d4f471d9f6dc8be72501d04`, whose detached status is still recorded as `RUNNING`; `baseline-local/formal-owner.json` also retains session `369464a5dc5de9442f551459b92f9f12` as `RUNNING` at PPI. The preserved incident inspection recorded that the exact supervisor and child PIDs were absent. No signal or manual status edit was made.

Source inspection shows that starting a new detached launch replaces `latest_launch.json`; the old detached `RUNNING` status itself is not a launch gate. The formal owner acquires its lock and then calls stale-owner recovery. If the lock is available and the recorded owner PID/start-time identity is no longer live, it marks the old formal session `INTERRUPTED` and proceeds. If the exact old owner is still alive, startup refuses. Thus the stale record is recoverable under the recorded no-live-process condition; do not delete or hand-edit it. The current progress command may report the old launch as interrupted until a new detached launch writes a new pointer.

For an explicitly selected PPI/OGBN scope, preparation creates a session-scoped resource manifest for that immutable dataset scope; it does not require the all-five global resource manifest. Actual resource authentication, capacity admission, proof generation, and independent verification still occur during the formal run and are not established by these checks.
