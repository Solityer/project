# Baseline Budget 2x

- timing_scope: `proof_core_components`
- proof_core_total_ms_formula: `commitment_ms + prove_ms + verify_ms`
- target_policy: final proof_core_total_ms <= target_2x
- near_acceptance_policy: final proof_core_total_ms may be <= near_2_2x only if global hot-path optimization evidence is complete
- row_over_near_2_2x_policy: abort matrix and continue optimization

| backend | dataset | baseline_core_ms | target_2x_ms | near_2_2x_ms | row_hard_timeout_ms |
| --- | --- | ---: | ---: | ---: | ---: |
| cpu | cora | 2238.275 | 4476.550 | 4924.205 | 147726.150 |
| cpu | citeseer | 1984.201 | 3968.402 | 4365.242 | 130957.260 |
| cpu | pubmed | 8839.821 | 17679.642 | 19447.606 | 583428.180 |
| cpu | ppi | 11275.126 | 22550.252 | 24805.277 | 744158.310 |
| cpu | ogbn-arxiv | 84410.517 | 168821.034 | 185703.137 | 1800000.000 |
| gpu | cora | 1638.292 | 3276.584 | 3604.242 | 120000.000 |
| gpu | citeseer | 1998.596 | 3997.192 | 4396.911 | 131907.330 |
| gpu | pubmed | 4331.885 | 8663.770 | 9530.147 | 285904.410 |
| gpu | ppi | 6455.733 | 12911.466 | 14202.613 | 426078.390 |
| gpu | ogbn-arxiv | 55014.665 | 110029.330 | 121032.263 | 1800000.000 |
