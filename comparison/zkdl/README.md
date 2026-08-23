# zkDL-style GAT formal baseline

This directory contains the baseline-local BLS12-381 implementation of the
relation defined by `GAT-ZKML-zkDL.md`.  It uses hiding Hyrax commitments,
FAC-style re-indexing, zero-knowledge masked Sumcheck, and Hyrax evaluation
openings.  It does not import the main experiment or another baseline's proof
system.

## Environment and build

Python 3 with NumPy is required by the canonical integer witness adapter.  Rust
is built from the locked dependency graph.  The GPU build additionally requires
CUDA `nvcc`, `libcudart`, and a CUDA device supported by the local toolkit.

```bash
python3 -m pip install -r requirements.txt
./scripts/build.sh --cpu
./scripts/build.sh --gpu
```

`--gpu` is fail-closed: the resulting prover probes a real CUDA kernel, runs a
CPU/CUDA known-answer test, and records nonzero CUDA invocation counters for
MSM, field inner products, scalar/point folds, and Sumcheck.  Verification is
always performed by the CPU binary and validates that telemetry.

## Formal workflow

Each row performs preflight bound/table checks, creates or validates the local
transparent Hyrax PP, runs the fixed canonical GAT adapter, compiles and checks
the public statement, registers the model if no registration exists, proves on
the selected device, independently verifies, cross-checks manifests, and emits
`runs/<dataset>/<device>/warm/result.json`.

```bash
python3 scripts/run_single.py --dataset cora --device cpu --project-root /home/pzh/project
python3 scripts/run_single.py --dataset cora --device gpu --project-root /home/pzh/project
./scripts/run_all.sh --project-root /home/pzh/project
python3 scripts/collect_results.py
python3 scripts/validate_final_results.py
```

The full serialized driver is `scripts/run_until_complete.sh`; it builds CPU and
GPU, runs all ten rows, collects results, then validates them.  No runner writes
outside this baseline.  Private witnesses, model opening secrets, proofs, and
generated PP objects are ignored by Git.

The statement binds canonical public features, graph/node and edge pointers,
sorted self-loop-normalized topology, model/checkpoint identity, quantization,
static tables, the certified integer-bound ledger, all relation families, and
the complete public logits.  The registered model must use exactly the same
bound-ledger ID.  Indexed table lookup uses little-endian index bits consistently
in the CPU and CUDA evaluators.

## Protocol-level privacy blocker

The implementation provides the paper-native semi-honest, single-session
hiding proof mechanisms.  The governing protocol requires a separate
multi-session zero-knowledge argument before reusable registered-model
commitments may claim the strict `model_and_trace_hidden` primary profile.  No
such theorem/argument is present.  Consequently every result is explicitly
marked `privacy_profile_status=unsupported_privacy_profile` and
`formal_experiment_eligible=false`; the validator exits with status 2 after a
computationally valid ten-row matrix.  This is an explicit protocol blocker, not
a fallback or skipped verification.

Table-domain preflight is also fail-closed.  A dataset whose authenticated
model bound exceeds its protocol-fixed public table cannot enter proving until
the governing configuration/table artifact is updated by its owner.
