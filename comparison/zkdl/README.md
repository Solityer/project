# zkDL GAT baseline-local adapter

The formal path uses the baseline-local native CUDA/C++ project and its CPU backend for commitment, full GAT relation proving and independent verification. Inputs, models, resources, sources, binaries and results stay under this directory.

Run the five complete datasets with this single user command:

```bash
cd /home/pzh/project/comparison/zkdl
ZKDL_ALLOW_FULL_DATASET_RUN=1 bash scripts/run_five.sh cpu
```

The entry first checks all five input configurations, snapshots the inputs/checkpoint metadata and creates session-specific configs and resources. It force-builds the native binaries under a build lock, freezes both binaries inside the session and records their source/binary hashes. Each dataset receives fresh model registration randomness and its own registry. It then prepares and authenticates all five resource sets and native relation plans before proving any dataset. Each proof is independently verified; collection and final validation only read the new session directory. Setup, preparation, proving, verification and their logs are recorded separately.

The native Pedersen/Hyrax basis is sampled from the operating-system CSPRNG using independent nonzero BLS12-381 scalars. This is an honest local trusted setup and requires destruction of its secret scalars; it is not a transparent setup. The baseline-local resource helper uses the pinned blst dependency for equivalent group arithmetic and strict point decoding. It does not change the native proof relation or commitment scheme. `setup_register.sh` automatically creates missing authenticated resources inside this baseline.

Artifacts are under `runs/formal_<UTC timestamp>_<unique suffix>/<dataset>_cpu/`; stage logs and elapsed seconds are under the same session's `logs/`. The matching collection is `results/<session>/collected.json` and `.csv`. Success requires `final_validation=PASS records=5 selected_backend=cpu independent_verify=5` followed by `formal_run=PASS` with the exact output paths. A failed stage exits nonzero and prints its name and log directory.

For independent verification of an existing session cell:

```bash
bash scripts/verify_single.sh configs/<session>_cora.cfg runs/<session>/cora_cpu
```

The privacy claim is conditional on the native zero-knowledge masking/composition and Fiat–Shamir random-oracle assumptions for a semi-honest verifier. Registries are used within one inference session; no repeated registered-model, multi-session privacy guarantee is claimed. Private weights and model blinding material are authenticated on the prover side and omitted from verifier/collector file access.

`run_matrix.sh` is a compatibility wrapper for the same five-dataset CPU entry. Development checks remain under `tests/`; formal execution does not use test fixtures.
