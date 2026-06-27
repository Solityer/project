# Formal Constraint Conflict: OGBN LReLU Table Domain

Generated: 2026-06-27

## Conflict

Under the fixed proof relation, fixed contiguous public LReLU table semantics, fixed quantization/encoding, fixed domain config, and fixed SRS/PK/VK, the full OGBN-arxiv row cannot complete. The current raw attention values require a contiguous LReLU table covering `[-44524211, 108122576]`, span `152646788`, which pads to `2^28`. The fixed secure SRS provides edge-domain Lagrange material for `2^23`.

## Evidence

- dataset integrity audit: `results/performance_engineering/dataset_integrity_audit.json`
- SRS/PK/VK audit: `results/performance_engineering/srs_pk_vk_integrity_audit.json`
- new formal raw diagnostic: `results/performance_engineering/diagnostics/ogbn_lrelu_range_diag_formal_raw_20260627.csv`
- prior failed official row log: `results/performance_engineering/failed_runs/ogbn_cpu_initial_final_runner_lrelu_failure_20260620T172559Z.log`
- prior rebalance failure log: `results/performance_engineering/failed_runs/ogbn_cpu_lrelu_table_after_negative_rebalance_20260620T175945Z.log`
- prior conflict report: `results/performance_engineering/failed_runs/ogbn_lrelu_static_table_blocker_20260620.md`

## Forbidden Escape Hatches Rejected

- change quantization or rescale raw attention values
- use witness-dependent sparse LReLU tables
- split LReLU into multiple lookup relations or domains
- change proof schema, transcript schedule, verifier equations, or label order
- increase or replace SRS/PK/VK/domain-bound Lagrange material
- shrink OGBN dataset, edges, features, heads, layers, proof domain, or lookup coverage
- allow verifier witness reads or private plain values

## Why Engineering Optimization Cannot Resolve It

Engineering optimization can reduce how existing vectors are computed, committed, or opened. It cannot make a fixed contiguous table of `152646788` legal raw values fit into a fixed `2^23` edge/table domain without changing the relation, table semantics, quantization, domain config, or SRS. Expanding to `2^28` is both forbidden by the fixed SRS/PK/VK rule and infeasible on this host for the existing edge-domain witness layout.

## Completed Before Conflict

- initial checkpoint recorded
- baseline 2x/2.2x budget generated
- dataset integrity audit generated
- SRS/PK/VK integrity audit generated
- performance path equivalence audit updated
- runner timeout and completed-over-budget dry-run passed
- final preflight generated and blocked final matrix
- CPU/CUDA builds passed
- CPU/CUDA ctest passed
- static scan passed
- full security regression passed
