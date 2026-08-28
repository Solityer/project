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
python3 scripts/prepare_resources.py --init
# Place the five dataset caches and checkpoint bundles under resources/.
python3 scripts/prepare_resources.py --check
python3 scripts/setup_resources.py
```

`--gpu` is fail-closed: the resulting prover probes a real CUDA kernel, runs a
CPU/CUDA known-answer test, and records nonzero CUDA invocation counters for
MSM, field inner products, scalar/point folds, and Sumcheck.  Verification is
always performed by the CPU binary and validates that telemetry.

## Formal workflow

The only formal execution sequence is:

```text
build -> prepare_resources (local layout/check) -> setup_resources (PP + authenticated manifest)
      -> register during first cell -> run_single/run_all -> collect_results
      -> validate_final_results
```

Each row performs preflight bound/table checks, uses the authenticated
baseline-local transparent Hyrax PP, runs the fixed canonical GAT adapter,
compiles and checks the public statement, registers the model if no
registration exists, proves on the selected device, independently verifies,
cross-checks manifests, and emits an exact cell under
`runs/<dataset>/<device>/warm/`. The runner refuses external resource paths,
symlinks, hard-links, stale cells, incomplete proofs, and extra cell artifacts.

Before a formal run, place all five resource bundles in the canonical
baseline-local directories.  The preparation command creates those directories
and checks their required files; it never reads any path outside this
directory:

```bash
python3 scripts/prepare_resources.py --init
# Copy/provide resources directly under resources/datasets/ and resources/models/.
python3 scripts/prepare_resources.py --check
python3 scripts/setup_resources.py
```

The five caches belong under `resources/datasets/<dataset>/` (using
`resources/datasets/ogbn_arxiv/` for `ogbn-arxiv`) and the checkpoint bundles
belong under `resources/models/<dataset>_gat/`.  `prepare_resources.py` has no
external path option.  It must be run successfully after the local
resources are present; `setup_resources.py` then authenticates the complete
local inventory and generates the PP-bound manifest.  No resource path may
escape this baseline or be supplied to `run_single.py`.

```bash
python3 scripts/run_single.py --dataset cora --device cpu
python3 scripts/run_single.py --dataset cora --device gpu
./scripts/run_all.sh
python3 scripts/collect_results.py
python3 scripts/validate_final_results.py
```

Run `scripts/build.sh --all` before the matrix if the CPU and GPU binaries are
not already present. The five-dataset matrix runner is `scripts/run_all.sh`;
collection and matrix validation are separate commands and never manufacture
proof or verification success. No runner writes outside this baseline.
Private witnesses, model opening secrets, proofs, and generated PP objects are
ignored by Git.

The statement binds canonical public features, graph/node and edge pointers,
sorted self-loop-normalized topology, checkpoint manifest, model shape,
quantization, static tables, the certified integer-bound ledger, all relation
families, and the complete public logits.  The complete canonical tensor
digest remains a private witness/source binding and is never placed in the
public statement.  The registered model
must use exactly the same authenticated family-set and bound-ledger IDs.
Indexed table lookup uses little-endian index bits consistently in the CPU and
CUDA evaluators.

Registered-model commitments are re-randomized for every proof.  For each
family the prover samples a fresh OS-CSPRNG `delta`, forms
`C_session=C_registry+delta*G_blind(logical_len)`, opens with
`r_registry+delta`, and appends a domain-separated Schnorr proof of that link.
The registry and session commitments are transcript-bound before operation
Sumcheck/FAC/Hyrax evaluation; the verifier checks every link before any
operation proof.  This is the strict
`supported_multi_session_registered_model` profile.

Preflight first computes the conservative integer-bound ledger and then derives
the LReLU/ELU/ExpMap domains and division bit width from those bounds.  The
configured `range_bits` value remains a minimum; it is not a fixed domain gate.
