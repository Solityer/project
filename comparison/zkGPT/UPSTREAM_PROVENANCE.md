# zkGPT GAT adapter provenance

This directory contains the restored zkGPT C++/CMake source tree and the
comparison adapter placed under `gat_adapter/`.  The upstream project named by
the source README is:

- Repository: <https://github.com/security-Anonymous/zkGPT>
- Upstream README description: zkGPT SNARK for GPT-2 inference, with mcl as a
  submodule and references to zkCNN/Lasso.

The local checkout has no recorded upstream tag or commit identity that can be
authenticated as an upstream release.  The local Git commit, if any, is a
workspace snapshot and is intentionally not presented as upstream provenance.
The adapter therefore treats the current checked-in source files as the
source of truth and does not claim a stronger upstream revision.

## Native implementation boundary

The adapter is linked against the official local C++ implementation through
the existing `src/` and `3rd/mcl/` CMake targets.  The production path uses
the native layered-circuit, Hyrax commitment, Sumcheck/GKR/Lasso and matrix
code exposed by that tree.  `gat_adapter/` adds canonical GAT decoding,
relation planning, the native CLI and the separate verifier CLI; it does not
replace the proof protocol with a Python or CPU emulation.

`3rd/mcl/` is a self-contained dependency body staged for this checkout so
the official CMake build does not depend on a main-experiment path.  No mcl
revision is asserted here because the staged dependency does not carry an
authenticated upstream revision record.

## Resources and setup

The five canonical dataset caches, model bundles, configs and quantization
chain are staged below `baseline-local/`.  Their SHA-256 sidecars are part of
the local input contract and are checked before a run.  The checkpoint
`manifest.json` digest is recorded by `prepare_resources.py`; payloads are
never silently regenerated or repaired.

The configs declare `setup_mode = hyrax_native_local` and
`setup_resource_scope = baseline-local`.  No main-experiment SRS is copied,
referenced, or treated as a zkGPT setup artifact.  A missing or unsupported
native setup remains a hard failure.

## Execution and evidence

`gat_zkgpt_experiment.py` launches one explicit native backend and then the
independent verifier process.  It never retries GPU with CPU.  A formal cell
is successful only when the native result, serialized artifact, independent
verifier report, timing record, backend evidence and witness-isolation fields
all agree.  Static builds, resource authentication and preflight output are
not formal proof results.  The current native `masked_full` implementation
must advertise support before the formal runner can proceed.
