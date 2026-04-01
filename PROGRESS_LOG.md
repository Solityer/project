# Progress Log

## This round
- Copied the minimum official GAT reference files into `project/reference/gat_official/` so formal work no longer depends on mutating or importing from `GAT/`.
- Kept all formal edits inside `project/`; the note file `GAT-ZKML-单层多头.md` was not modified.
- Tightened full-Cora regression coverage in `tests/test_main.cpp`:
  - auto-regenerate reference artifacts
  - check `dst` sorting and self-loops
  - check bias parity
  - check real multi-head formal context with explicit `H_cat` and `H_C`
  - check verifier rejection for metadata mismatch, proof block order mismatch, and wrong `H_C` domain reuse
- Extended proof metadata and proof-order tracking:
  - `PublicMetadata`
  - `Proof::block_order`
  - verifier checks for both
- Promoted `H_cat` and `H_C` to explicit work domains in formal context.
- Removed `project/scripts/__pycache__` as a non-formal runtime residue.

## Local run results
- `gatzk_tests`: pass
- `gatzk_run --config configs/cora_full.cfg`: fail fast in formal multi-head trace construction with an explicit message about missing hidden-head/concat/output formal objects

## Current blocker
- `src/protocol/trace.cpp` still materializes legacy single-head witness names, so the real multi-head formal run reaches trace construction and then aborts on missing objects.
