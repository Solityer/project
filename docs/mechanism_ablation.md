# Mechanism-level zkGAT ablations

`scripts/run_mechanism_ablation.py` runs isolated Feature Folding and Shared Neighborhood proof workloads. Each workload is paired across `original` and `control`; it is not a full GAT inference proof. The default run covers three workload sizes, both mechanisms, both variants, and one repetition.

Use the registered Cora trusted SRS for these examples. The runner checks the selected SRS manifest, the source identity embedded in the executable, build configuration, backend/device receipt, and the exact serialized proof bytes before it accepts a cell.

Run the CPU pairs with the pinned CPU build tree. The build step refreshes its source manifest before the runner checks provenance:

```bash
CPU_BUILD_DIR=/home/pzh/builds/gatzk-mechanism-cpu-test
cmake -S . -B "$CPU_BUILD_DIR" -DGATZK_ENABLE_CUDA_BACKEND=OFF
cmake --build "$CPU_BUILD_DIR" --target gatzk_mechanism -j2
python3 -B scripts/run_mechanism_ablation.py \
  --backend cpu --binary "$CPU_BUILD_DIR/gatzk_mechanism" \
  --srs artifacts/srs/cora_formal_srs
```

Run the same pairs on CUDA device 0. This self-contained example builds into the repository-local `build_cuda/` directory and passes the resulting absolute binary path:

```bash
CUDA_BUILD_DIR="$PWD/build_cuda"
cmake -S . -B "$CUDA_BUILD_DIR" -DGATZK_ENABLE_CUDA_BACKEND=ON
cmake --build "$CUDA_BUILD_DIR" --target gatzk_mechanism -j2
python3 -B scripts/run_mechanism_ablation.py \
  --backend cuda --device 0 --binary "$CUDA_BUILD_DIR/gatzk_mechanism" \
  --srs artifacts/srs/cora_formal_srs
```

Each invocation writes a fresh session under `runs/mechanism_ablation_secure/`. `summary.md`, `summary.json`, and `cells.csv` contain the paired results; `session.log` records orchestration, and each cell directory contains the serialized proof, public statement, prover and verifier logs, machine-readable receipts, and exit codes. The witness seed is shared by the two variants in a pair, stored in a mode-0600 temporary file, redacted from logs, and removed after the pair. The verifier runs as a second process and receives only the serialized proof, public statement, and SRS. It uses the strict CPU verification path for either prover backend.

The reported timing partitions use `mechanism_commitment_disjoint_verify_parse_accept_v2`. `commitment_time_ms` measures the cold native KZG commitment phase, including first-use static-base preparation invoked inside KZG commitment; Taylor-shift coefficient materialization is excluded and is counted in `prove_time_ms`. The phases are disjoint. `prove_time_ms` includes native online proving, Taylor-shift coefficient materialization, and canonical proof encoding, while proof artifact file writes and JSON receipt I/O remain outside the timer. `verify_time_ms` covers the independent CPU verifier's parsing and proof decoding through acceptance; SRS and artifact file reads happen before that timer. `proof_size_bytes` is checked against the actual canonical proof file length. CUDA H2D counters are the existing tracked transfer counters and do not report the full static-base upload byte volume. Only complete pairs with two independent `VERIFY_OK` results contribute to aggregate metrics; failed, rejected, timed-out, or provenance-mismatched cells remain visible as failures and are excluded from aggregates.

Supply one or more explicit scales with repeatable `--workload N,E,D,G` options, where `N` is nodes, `E` is edges, `D` is mechanism feature width, and `G` is graph count. After building as above, these are the three-scale, three-repetition paired runs; execute the CUDA command after the CPU run has finished:

```bash
python3 -B scripts/run_mechanism_ablation.py \
  --backend cpu --binary "$CPU_BUILD_DIR/gatzk_mechanism" \
  --srs artifacts/srs/cora_formal_srs \
  --workload 16,48,2,1 \
  --workload 64,256,4,2 \
  --workload 256,1024,8,4 \
  --repetitions 3
python3 -B scripts/run_mechanism_ablation.py \
  --backend cuda --device 0 --binary "$CUDA_BUILD_DIR/gatzk_mechanism" \
  --srs artifacts/srs/cora_formal_srs \
  --workload 16,48,2,1 \
  --workload 64,256,4,2 \
  --workload 256,1024,8,4 \
  --repetitions 3
```

For a separate maximum-scale capacity check, the current native workload limits are `0 < N < 4096`, `N <= E < 32768`, `1 <= D <= 32`, and `1 <= G <= N`, making `4095,32767,8,16` the largest legal example. The native mechanism protocol also checks each graph's topology, domain, degree, security ledger, and trusted-SRS capacity. A rejected scale is a failed cell; the runner does not silently lower it and report it as the requested run. Use a new session for each backend or parameter set.
