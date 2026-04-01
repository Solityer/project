# Forward Alignment Report

## Alignment status
- Reference source inside `project/`: `reference/gat_official/execute_cora.py`, `reference/gat_official/models/gat.py`, `reference/gat_official/utils/layers.py`
- Real parameter bundle inside `project/`: `artifacts/checkpoints/cora_gat`
- Full-Cora reference artifacts inside `project/`: `runs/cora_full/reference`

## Verified objects on full Cora
- Hidden heads `0..7`
  - `H_prime`, `E_src`, `E_dst`, `S`, `Z`, `M`, `Delta`, `U`, `Sum`, `inv`, `alpha`, `H_agg`
- Hidden concat
  - `hidden_concat`
- Output head
  - `H_prime`, `E_src`, `E_dst`, `S`, `Z`, `M`, `Delta`, `U`, `Sum`, `inv`, `alpha`, `H_agg`, `Y_lin`, `Y`
- Bias / topology checks
  - bias matrix, `dst`-sorted edges, explicit self-loops

## Important semantic fixes already landed
- Self-loop now enters attention aggregation instead of being treated as a side artifact.
- Hidden heads are concatenated after ELU; output layer remains a single attention head with identity output.
- Full-Cora parity is driven from project-local artifacts only; if reference artifacts are missing, tests regenerate them via `scripts/gat_reference.py`.

## Residual gap
- The neural forward path is aligned.
- The formal witness path is not yet aligned to the note for concat/output-stage proof objects, so formal `cora_full` still stops in multi-head trace construction.
