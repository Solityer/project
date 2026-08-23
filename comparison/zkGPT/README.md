# zkGPT comparison baseline

This directory retains the upstream zkGPT C++ source, paper, license, and a
research GAT graft.  The governing relation is `GAT-ZKML-zkGPT.md`.

## Formal status

The official path is deliberately fail-closed.  Neither CPU nor GPU may emit a
formal comparison result from the retained graft.  The authoritative machine
readable status is `configs/formal_capabilities.json`; every official entry
point reads that status directly or invokes `scripts/formal_preflight.py`.

The retained Python graft is not promoted to a formal implementation because
it differs from the protocol in material ways: it uses floating-point GAT
forward operations; a seven-segment layout with a private `com_T_H` instead of
the required public `X` and `X|W|Aux|L` four-segment layout; digest/sampled
lookup checks instead of complete native Lasso; residual/product-sum objects
instead of a verifier-replayed ML-friendly GKR; field-linear pseudo
commitments with deterministic blinders instead of native BN254 group Hyrax;
and a CUDA u64 reduction over CPU-generated words instead of GPU proof
primitives.  It also lacks the repaired `11 -> L` selector, final `com_L`
opening, and the required lookup/input consistency relation as one closed
native protocol.

The strict `masked_full` privacy profile is independently blocked: the
governing protocol requires a composable multi-session zero-knowledge argument
covering Lasso, GKR, MatMul, combine, and all three terminal openings while
reusing `com_W`.  That argument is not supplied by the original paper or this
checkout, so the required status is `unsupported_privacy_profile` even after
computational soundness components are considered.

## Fail-closed commands

These commands perform configuration/source/environment checks and report the
blockers.  They never execute the retained graft as a formal experiment:

```bash
python3 scripts/formal_preflight.py
./scripts/build.sh --cpu
./scripts/build.sh --gpu
python3 scripts/gat_zkgpt_experiment.py --datasets all --backends all
python3 scripts/collect_results.py
python3 scripts/validate_final_results.py
```

Preflight, build, run, and validation exit with code 2 while the protocol is
unsupported.  The collector emits ten explicit blocked rows with no proof or
verification success claim.  This prevents old research artifacts, host-side
digest checks, or the CUDA reducer from being mistaken for formal results.

The declared `3rd/mcl` submodule is currently unpopulated.  CMake configuration
remains available for source checking; the native Hyrax reference CLI target is
only defined when the mcl headers and static library are present.  The CUDA
target is named `zkgpt_cuda_reference_reducer` to make its non-formal role
explicit.
