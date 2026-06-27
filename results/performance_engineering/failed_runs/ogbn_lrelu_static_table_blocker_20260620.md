# OGBN LReLU Static Table Blocker

Generated: 2026-06-20

## Summary

The final `cpu/ogbn-arxiv` official row cannot complete under the current formal raw trace path and the task constraints without changing proof semantics, static-table semantics, quantization/encoding, or domain/SRS scale.

## Evidence

- Initial final-run failure log: `runs/secure_main_runner/logs/run_cpu_ogbn-arxiv_20260620T172559Z.log`
- Archived initial failure copy: `results/performance_engineering/failed_runs/ogbn_cpu_initial_final_runner_lrelu_failure_20260620T172559Z.log`
- Rebalance canary failure copy: `results/performance_engineering/failed_runs/ogbn_cpu_lrelu_table_after_negative_rebalance_20260620T175945Z.log`
- Formal raw range diagnostic: `results/performance_engineering/diagnostics/ogbn_lrelu_range_diag_formal_raw_20260620.csv`
- Failed canary after temporary negative-range rebalance: `runs/secure_main_runner/logs/run_cpu_ogbn-arxiv_20260620T175945Z.log`

The first failure was:

```text
fatal: pair lookup query escaped static table for P_h0_L left=...76606163 right=...80268843
```

Those field values decode to signed raw values `-4578349` and `-915669`.

A temporary negative-side rebalance was tested and rejected because it only moved the failure to the positive side:

```text
fatal: pair lookup query escaped static table for P_h0_L left=7553776 right=7553776
```

The retained source does not include that temporary rebalance.

The formal raw diagnostic shows the actual OGBN attention score range required by the current trace:

| label | min_s_raw | max_s_raw | span_s_raw | unique_s |
|---|---:|---:|---:|---:|
| hidden[0] | -4578405 | 108122576 | 112700982 | 18816 |
| output[0] | -44524211 | 20775666 | 65299878 | 32500 |

To keep the current fixed public contiguous LReLU table semantics, one table would need to cover at least `[-44524211, 108122576]`, which is `152646788` rows before padding.

## Constraint Conflict

The current large-graph table policy keeps `edge=2^23` (`8388608`) so edge-domain trace columns remain feasible. Expanding the edge domain to the next power of two that can hold the contiguous LReLU table would require `edge=2^28` (`268435456`).

That is not viable under the current proof layout:

- Each edge-domain witness column would become tens of times larger.
- The current machine has 121 GiB RAM; the existing `edge=2^23` OGBN path already used roughly 40-55 GiB during proof attempts.
- `edge=2^28` would push edge-domain vectors and KZG material beyond practical memory for a full proof row.
- The current SRS material header confirms only `67108864` coefficient powers and `8388608` edge Lagrange points.

Rejected alternatives:

- Witness-dependent sparse LReLU table: would change the fixed public activation relation and leak or bind private trace/model-derived table content.
- Non-contiguous hand-picked table windows: would no longer prove the same fixed contiguous LReLU table relation.
- Fixed-point rescaling of raw projection values: would change quantization/encoding and zkMap projection constraints.
- Splitting LReLU into multiple lookup tables or domains: would change proof schema, commitment labels, challenge schedule, and verifier equations.
- Increasing edge domain/SRS to `2^28`: would change `domain_cfg`/SRS scale and is not feasible for the full OGBN row on this host.

## Result

The retained KZG opening optimization remains valid and measured, but the required final 10-row matrix cannot be produced because `cpu/ogbn-arxiv` and therefore `gpu/ogbn-arxiv` cannot complete without violating the stated constraints or exceeding feasible memory.
