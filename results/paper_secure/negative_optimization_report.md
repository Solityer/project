# Negative Optimization Report

Generated: 2026-06-20

## Rejected Or Non-Main Routes

| route | evidence | decision |
|---|---|---|
| CPU_KZG_MSM_REWRITE using raw `mcl::G1::mulVec` | `results/performance_engineering/benchmarks/msm_stage_prepatch_cpu_route_bench_20260620T121449Z.csv` showed current profiled/chunked MSM is faster than raw and compact raw variants. | rejected |
| `evaluate_t_edge` per-point map removal as main optimization | `t_edge` sampled benchmark improved only about 3-4 percent. | retained only as auxiliary cleanup |
| full-domain `t_edge` prepatch benchmark | `results/performance_engineering/failed_runs/t_edge_eval_prepatch_20260620T122150Z.interrupted.*` was interrupted after exceeding probe budget without a first row. | excluded from speedup claims |
| OGBN LReLU negative-range rebalance | `runs/secure_main_runner/logs/run_cpu_ogbn-arxiv_20260620T175945Z.log` showed the change only moved the failure from the negative side to `left=7553776 right=7553776`; formal raw scan shows the required span is far larger. | reverted and documented in `results/performance_engineering/failed_runs/ogbn_lrelu_static_table_blocker_20260620.md` |

No diagnostic, failed, stale, toy, smoke, or partial run is admitted into `results/paper_secure/latest.csv` or `latest.json`.
