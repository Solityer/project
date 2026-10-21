# zkGPT Hyrax binary-row commitment optimization

## Changes

- `src/hyrax.cpp`: tensor commitments with at least 128 rows and at least 8 columns divisible by 8 use a per-call subset-sum table over that call's canonical column generators. It is allocated only when the first nonzero binary block is encountered. Each scalar is read once in the existing order; all-zero blocks skip the table; nonbinary or unaligned blocks use the existing MSM fallback. Row blindings and proof/opening/verification code are unchanged.
- `gat_adapter/gat_proof.hpp`, `gat_adapter/gat_proof.cpp`, and `gat_adapter/gat_cli.cpp`: capacity schema v10 reports the strict `hyrax_l_qry_binary_lut_workspace_bytes` field. The producer includes the LUT, canonical generators, and fallback chunk in commit transient space, and combines this with retained tensor state and the larger opening transient before estimating host peak.
- `scripts/formal_preflight.py`: strictly consumes the new schema field, recomputes its value from the canonical L_qry shape, and checks combined/host capacity against the same workspace formula.
- `tests/native_masked_hyrax_smoke.cpp`: eligible `R=128,C=128` cases cover a partial zero tail, nonzero binary rows, an all-zero tensor, and a nonbinary fallback row. Each row is compared with the original `G1::mulVec` MSM using the same row blindings, including serialized point equality; the provider read count is checked. Smaller balanced shapes cover the row and column thresholds.
- `tests/test_t2_capacity_consumer.py`: the real native-CLI-to-Python consumer case asserts schema v10 and requires a nonzero LUT capacity for an eligible L_qry shape.

At the current large-dataset L_qry width `C=131072`, the table bound is `32*C` G1 points, or 603,979,776 bytes (576 MiB). This is standard host-heap memory and is included in the capacity estimate; it is not classified as mapped storage.

## Bounded checks

- `cmake --build build/cpu-tests --target native_masked_hyrax_smoke zkgpt_gat_prover zkgpt_gat_verifier --parallel 2`: passed, exit 0, 40.60 s. Log: `hyrax-lut-build.log`.
- `ctest --test-dir build/cpu-tests -R '^native_masked_hyrax_smoke$' --output-on-failure`: passed, 1/1, 9.21 s. Log: `native-masked-hyrax-smoke.log`.
- `python3 tests/test_t2_capacity_consumer.py`: passed, 1 test, 0.65 s wall time (0.382 s test time), exercising the real native CLI and Python consumer with a nonzero eligible LUT field. Log: `hyrax-lut-capacity-consumer.log`.

No full dataset run or performance benchmark was performed. These checks do not establish PPI/OGBN-Arxiv end-to-end completion or a five-dataset 24-hour total. The formal run still must pass its resource and capacity admission on the target host.

## Detached PPI and OGBN-Arxiv run

From the repository root:

```bash
./scripts/run_formal_detached.sh --datasets ppi ogbn_arxiv
```

The launcher prints a launch ID and writes the console and status under `baseline-local/formal-detached/<launch_id>/`. It starts a detached supervisor, so the SSH session can close after the launcher returns. To follow progress use:

```bash
./scripts/formal_progress.sh
./scripts/formal_progress.sh --once
tail -f baseline-local/formal-detached/<launch_id>/console.log
```

To stop exactly that launch safely:

```bash
python3 scripts/stop_formal.py --launch-id <launch_id>
```

The stop script validates the launch record and recorded supervisor process identity before signaling it. A stale `latest_launch.json` is replaced by the new detached launch; an old `RUNNING` local owner record is recovered as interrupted when its process is gone and the formal lock is available. A still-live recorded owner prevents a second formal owner from starting. No formal process was launched by this bounded check.
