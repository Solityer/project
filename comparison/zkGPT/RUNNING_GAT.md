# Running the zkGPT GAT comparison

The unique five-dataset formal command is:

```bash
cd /home/pzh/project/comparison/zkGPT && ./run_zkgpt_cpu_formal.sh
```

Current status: all five formal capacities are **BLOCKED** by `unsupported_native_masked_lookup_capacity`. The native masked Hyrax opening sends a full-domain Schnorr response, and the standard masked Lasso opens its dense one-hot matrix. Even Cora requires at least 647 GB of response encoding, exceeding the authenticated native vector and 1 GiB artifact limits. The existing source has no compatible compressed masked opening. This cannot be repaired by more RAM, lifting allocation limits, or using the incomplete virtual sparse path. The command stops at preflight; it currently cannot produce the five formal results. The current capacity evidence is in `results/formal_preflight/cpu/`.

The command authenticates baseline-local resources, builds native CPU prover and verifier, checks execution preconditions, registers each model when needed and upgrades recognized older registration schemas through validated staged native writes, generates each real full-GAT proof, invokes the independent verifier, collects five results and validates the matrix by independently verifying every proof again. GPU is unsupported for the formal `masked_full` relation and is rejected.

Execution preflight checks resources, build provenance, native capabilities and all five capacity plans. It reports `readiness_scope=formal_execution_preconditions` and `formal_results_complete=false`; it is separate from development proof checks and does not establish protocol success. Formal build and execution do not depend on `tests/`.

Results and proofs are in `baseline-local/runs/{cora,citeseer,pubmed,ppi,ogbn_arxiv}_secure/cpu/`. Each successful cell contains `result.json`, `proof.artifact`, `verifier_report.json` and `execution.json`. The matrix is `results/collected_results.json`; final acceptance is `results/final_validation.json` with `status=VALIDATED`, `cells_validated=5`, and terminal `FINAL_VALIDATION_OK cells=5` followed by `ALL_FIVE_DATASETS_COMPLETED`. The main log is `logs/zkgpt_cpu_formal_*.log`.

For a single dataset, after resource preparation, CPU build and successful execution preflight:

```bash
python3 scripts/gat_zkgpt_experiment.py --datasets cora --backends cpu
```

The runner preserves current valid long-lived `model_registration.json` and owner-only `com_W.prover.secret`; it removes only allowlisted stale inference outputs. An incomplete registration pair or any rejected native proof fails closed. Collection and final validation recheck current resources, native build provenance and the exact independently verified artifacts.
