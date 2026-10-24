# zkGPT formal runner operational cleanup

This bounded repair removes repeated outer execution authentication while keeping the native proof statement, transcript bindings, proof artifact, and independent verifier. It does not establish a five-dataset runtime or completion result.

The official runner now uses one owner process for prepare, serial dataset execution, and aggregation. It no longer forks prepare/cell/finalize owners or creates approval-FD and receipt-hash chains. Each selected dataset gets a fresh isolated run directory. Dataset-local resource or capacity failures are recorded and later selected datasets are still attempted; shared build/backend failures remain run-wide blockers. Static capacity admission and live host-memory/disk checks remain active. A successful cell requires the current prover invocation to exit successfully, a fresh nonempty proof artifact with matching actual file size, and one successful independent native verifier invocation with a fresh `VERIFY_OK` report. Full completion still requires exactly five verified cells; subset output remains explicitly incomplete.

The per-cell preflight gate now checks shared preconditions, session label, backend readiness, and that cell's own `READY` full-relation capacity entry. It no longer rejects an eligible cell because another selected dataset made the overall preflight `PARTIAL`. The focused test covers PPI blocked with OGBN-Arxiv still admitted and attempted.

The formal path no longer consumes `run_nonce`, session-manifest SHA, source/build provenance hashes, result/report artifact SHA, or artifact-readback timing. The build script no longer hashes the source tree, binaries, and runtime libraries after formal builds. Capacity keeps native shape and resource bounds while omitting the extra canonical-graph provenance digest. Native identity strings used by the proof context remain aligned across Python producer/consumer code; the `integrity.sha256` suffix is retained only as the existing native statement identity label and is not read or authenticated by this Python check.

The Python/native contract is synchronized to capacity schema `zkgpt-gat-capacity-v12`, prover result schema `zkgpt-gat-result-v3`, independent verifier schema `zkgpt-gat-independent-verifier-v5`, and timing semantics `zkgpt-gat-paper-timing-v5-native-commitment-wire-io-excluded`. The proof artifact wire protocol was not changed in this cleanup. `RUNNING_GAT.md` now documents the existing detached launcher, dataset selection, progress watcher, and safe stop command.

Files changed in this cleanup:

- `build.sh`
- `scripts/common.py`
- `scripts/run_formal.py`
- `scripts/formal_preflight.py`
- `scripts/gat_zkgpt_experiment.py`
- `scripts/register_model.py`
- `scripts/formal_progress.py`
- `tests/test_formal_dataset_scope.py`
- `tests/test_t2_capacity_consumer.py`
- `RUNNING_GAT.md`
- removed the obsolete `scripts/prepare_resources.py` entrypoint; no official shell wrapper or `RUNNING_GAT.md` command references it.

The only build command was:

```bash
cmake --build build/cpu --target zkgpt_gat_prover zkgpt_gat_verifier --parallel 2
```

Both targets reached 100% and linked. The build log is `baseline-local/bounded-check/repair-2026-10-22/operational-cleanup/build.log` and records `exit_code=0`, `elapsed_seconds=60`. The enclosing tool session reported exit code 1 despite the shell recording the successful CMake return code and explicitly exiting with that saved value. This command had no trailing conditional/test command that could explain the discrepancy; the wrapper/session discrepancy is unresolved, while the captured CMake result and target-link output show a successful build.

The focused runner command was:

```bash
python3 tests/test_formal_dataset_scope.py
```

All four tests passed in 0.128 seconds; `runner-focused.log` records `exit_code=0`. The enclosing tool session again reported 1 despite the test process's captured zero exit code and `OK` output, with no trailing conditional in the command. This is reported as a tool/session status discrepancy, not hidden as a test failure.

The first tiny CLI attempt used a stale binary from `build/cpu-tests` and failed capacity schema parsing; `tiny-cli.log` preserves that failure. The fixture was corrected to use the already-built formal binaries in `build/cpu`. Its next attempt reached Python report validation and exposed a mismatch between Python resource identity labels and native statement labels; `tiny-cli-recheck.log` preserves that failure. Python labels were aligned to the native statement strings without reading or hashing sidecars. The final bounded command was:

```bash
python3 tests/test_t2_capacity_consumer.py
```

It passed in 21.589 seconds (22-second shell timer), including capacity v12 parsing, registration, deferred proof, independent native verification, and Python result/report/metrics consumption. The final log is `tiny-cli-final.log` and records `exit_code=0`.


No remaining outer nonce/manifest/source/build/receipt authentication gate is known on the official runner path. No full dataset or formal five-dataset run was started, so PPI/OGBN-Arxiv completion time and a 24-hour end-to-end bound remain unverified. No formal launcher was started by this repair. The ready-to-use commands are:

```bash
cd /home/pzh/project/comparison/zkGPT
./scripts/run_formal_detached.sh
./scripts/run_formal_detached.sh --datasets ppi ogbn_arxiv
./scripts/formal_progress.sh --once
./scripts/formal_progress.sh
python3 scripts/stop_formal.py
python3 scripts/stop_formal.py --launch-id <launch_id>
```
