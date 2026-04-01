# Benchmark Report

## Environment
- Host: local Linux server
- Project root: `project/`
- Build: `/tmp/gatzk_build2`
- Dataset: full Cora from `project/data` and `project/data/cache`
- Checkpoint bundle: `project/artifacts/checkpoints/cora_gat`

## Commands
- Build: `cmake -S /home/pzh/project -B /tmp/gatzk_build2 && cmake --build /tmp/gatzk_build2 -j`
- Test benchmark: `/tmp/gatzk_build2/gatzk_tests`
- Formal entry check: `/tmp/gatzk_build2/gatzk_run --config /home/pzh/project/configs/cora_full.cfg`

## Latest measurements
- `gatzk_tests`
  - status: pass
  - wall time: `106.225 s`
- `gatzk_run --config configs/cora_full.cfg`
  - status: fail
  - last log:
    - `[gatzk] backend_name=mcl benchmark_mode=single route2=msm_fft_packed_kernel_layout_pairing`
    - `[gatzk] building protocol context`
    - `[gatzk] building trace`
    - `fatal: build_trace is still wired to the legacy single-head witness system; the formal multi-head objects for hidden heads, H_cat/H_cat_star, H_C, Y'_star/Y_star, and PSQ_out are not materialized yet`

## Formal benchmark status
- Full-Cora multi-head formal `prove` time: unavailable in this round
- Full-Cora multi-head formal `verify` time: unavailable in this round
- Reason: concat/output-stage formal objects are still missing in `src/protocol/trace.cpp`, so the formal mainline does not reach proof generation yet.

## Bottleneck
- The hard blocker is not FFT/MSM yet; it is missing multi-head formal witness wiring in trace generation.

## Next optimization targets
- After multi-head trace closure:
  - cache transcript-independent static commitments and keys
  - reuse seven-domain FFT plans
  - remove duplicate matrix/padding materialization in trace and commitment setup
