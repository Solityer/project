# zkDL GAT command reference

All commands run from `/home/pzh/project/comparison/zkdl` with baseline-local paths.

| Purpose | Command |
| --- | --- |
| Native build | `make baseline-local-build` |
| Static five-input preflight | `python3 scripts/formal_preflight.py` |
| User five-dataset formal experiment | `ZKDL_ALLOW_FULL_DATASET_RUN=1 bash scripts/run_five.sh cpu` |
| User single CPU experiment | `ZKDL_ALLOW_FULL_DATASET_RUN=1 bash scripts/run_single.sh configs/cora_secure.cfg cpu` |
| Independent verification | `bash scripts/verify_single.sh configs/<session>_cora.cfg runs/<session>/cora_cpu` |
| Collect one session | `python3 scripts/collect_results.py --runs runs/<session> --output results/<session>/collected.json --require-selected-backend cpu` |
| Validate one session | `python3 scripts/final_validate.py --runs runs/<session> --backend cpu` |

The formal five-dataset entry performs static input admission, session input/metadata snapshots, a locked fresh native build with frozen source/binary provenance, automatic authenticated local trusted setup and fresh model registration, all five native preparations, then full proofs and independent verification. It creates a fresh run session, keeps every stage log and collects only that session. The local trusted setup relies on honest CSPRNG sampling and destruction of all secret setup scalars; it is not transparent.

Formal success is `final_validation=PASS records=5 selected_backend=cpu independent_verify=5` and `formal_run=PASS`. The final line prints the session's run, result and log paths. Every session keeps its configs, resources, registries and binary snapshots independently. Verifiers and collectors read public metadata and proof resources, while weights and model blinding material remain prover-only. The single-session privacy statement is conditional on native masking/composition, Fiat–Shamir ROM and the semi-honest verifier model. The full-dataset gate belongs to the user; development agents must not invoke it.
