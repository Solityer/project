# Five Dataset Bottleneck Matrix

Generated: 2026-06-27

This matrix is diagnostic and is not a final matrix. `comparison/**` is ignored.

| backend | dataset | proof_core_total_ms | commitment_ms | prove_ms | verify_ms | target_2x | near_2_2x | budget_status | top bottlenecks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| cpu | cora | 959300.554 | 212393.441 | 738237.061 | 8670.052 | 4476.550 | 4924.205 | over_near_2_2x | quotient_build=298523.275; domain_open_witness=287847.362; domain_opening=271168.341; fh_open_witness=255284.519; quotient_t_edge=196739.083 |
| gpu | cora | 959082.874 | 213553.651 | 737223.796 | 8305.427 | 3276.584 | 3604.242 | over_near_2_2x | quotient_build=304041.441; domain_open_witness=280281.150; domain_opening=263088.181; fh_open_witness=250850.232; quotient_t_edge=201922.068 |
| cpu | citeseer | 2576829.793 | 446485.993 | 2110788.051 | 19555.749 | 3968.402 | 4365.242 | over_near_2_2x | quotient_build=994000.723; domain_opening=946294.052; domain_open_witness=910820.931; fh_open_witness=883857.577; quotient_t_FH=640911.104 |
| gpu | citeseer | 2568287.975 | 448712.888 | 2103265.429 | 16309.658 | 3997.192 | 4396.911 | over_near_2_2x | quotient_build=990388.966; domain_opening=942305.619; domain_open_witness=914371.054; fh_open_witness=891872.878; quotient_t_FH=635348.946 |
| cpu | pubmed | 3139857.140 | 704445.587 | 2415164.513 | 20247.040 | 17679.642 | 19447.606 | over_near_2_2x | quotient_build=999271.034; domain_open_witness=917431.826; domain_opening=901523.877; fh_open_witness=866390.442; quotient_t_FH=640714.351 |
| gpu | pubmed | 3148782.811 | 706901.749 | 2425437.894 | 16443.168 | 8663.770 | 9530.147 | over_near_2_2x | quotient_build=1000829.047; domain_open_witness=929292.436; domain_opening=907768.977; fh_open_witness=883469.826; quotient_t_FH=642250.264 |
| cpu | ppi | 2503629.240 | 671430.298 | 1822063.262 | 10135.680 | 22550.252 | 24805.277 | over_near_2_2x | quotient_build=597283.560; dynamic_commit_msm=539927.958; quotient_t_edge=461987.767; domain_open_witness=438887.181; domain_opening=310985.765 |
| gpu | ppi | 2467419.978 | 663228.750 | 1794269.968 | 9921.260 | 12911.466 | 14202.613 | over_near_2_2x | quotient_build=578633.009; dynamic_commit_msm=540866.962; quotient_t_edge=452481.933; domain_open_witness=430887.209; domain_opening=300452.159 |
| cpu | ogbn-arxiv | 0.000 | 0.000 | 0.000 | 0.000 | 168821.034 | 185703.137 | formal_constraint_conflict | OGBN LReLU raw span 152646788 requires edge domain 2^28; fixed edge SRS is 2^23 |
| gpu | ogbn-arxiv | 0.000 | 0.000 | 0.000 | 0.000 | 110029.330 | 121032.263 | formal_constraint_conflict | OGBN LReLU raw span 152646788 requires edge domain 2^28; fixed edge SRS is 2^23 |

## Common Bottlenecks

- KZG/MSM dynamic and quotient commitments dominate across all completed secure diagnostics.
- Domain-grouped opening witness generation dominates FH/edge bundles.
- Quotient materialization remains large for FH and edge domains.
- GPU rows are CUDA-hotspot runs; KZG MSM, pairing, quotient materialization, and opening witness still fall back to CPU.

## Selected Optimization Route

- Retained shared hot-path route from current working tree: `OPENING_WITNESS_REUSE` in `src/crypto/kzg.cpp`, plus smaller `t_edge` evaluation cleanup in `src/protocol/quotients.cpp`.
- This is not Cora-only; evidence files are under `results/performance_engineering/benchmarks/`.
- It is insufficient for final readiness because rows remain far above near-2.2x and OGBN has a formal table-domain conflict.

## PPI / OGBN Notes

- PPI multi-graph metadata is audited in `dataset_integrity_audit.json`; current bottlenecks are dynamic MSM, t_edge quotient, FH opening, and external opening.
- OGBN cannot enter final matrix: fixed contiguous LReLU table semantics require a 2^28 edge/table domain, but the fixed SRS has edge Lagrange material for 2^23.

## Matrix Abort 20260627T131913Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260627T132322Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T050032Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T053720Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T054051Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T055148Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T055504Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T060210Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T060729Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T061702Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T062025Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T062341Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T062836Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T063519Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T063835Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T064143Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T064458Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T100932Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T102127Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T102948Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T104004Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T104759Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T105644Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T110616Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T112210Z

- cpu/cora: row exited with code 1

## Matrix Abort 20260628T112823Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260629T060142Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260629T060754Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260629T061735Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260629T062905Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260629T063907Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260629T065542Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260630T030655Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260630T045906Z

- gpu/cora: row exited with code 124

## Matrix Abort 20260630T051547Z

- gpu/cora: row exited with code 124

## Matrix Abort 20260630T052206Z

- gpu/cora: row exited with code 124

## Matrix Abort 20260630T054004Z

- gpu/cora: row exited with code 124

## Matrix Abort 20260630T054645Z

- gpu/cora: row exited with code 124

## Matrix Abort 20260630T060532Z

- gpu/cora: row exited with code 124

## Matrix Abort 20260630T062155Z

- gpu/cora: row exited with code 124

## Matrix Abort 20260630T063314Z

- gpu/cora: row exited with code 124

## Matrix Abort 20260630T111654Z

- gpu/cora: row exited with code 124

## Matrix Abort 20260630T112217Z

- gpu/cora: row exited with code 124

## Matrix Abort 20260630T113129Z

- gpu/cora: row exited with code 124

## Matrix Abort 20260630T113539Z

- gpu/cora: row exited with code 124

## Matrix Abort 20260630T114505Z

- gpu/cora: row exited with code 124

## Matrix Abort 20260630T115233Z

- gpu/cora: row exited with code 124

## Matrix Abort 20260630T115730Z

- gpu/cora: row exited with code 124

## Matrix Abort 20260630T120311Z

- gpu/cora: row exited with code 124

## Matrix Abort 20260630T121015Z

- gpu/cora: row exited with code 124

## Matrix Abort 20260630T121722Z

- gpu/cora: row exited with code 124

## Matrix Abort 20260630T122333Z

- gpu/cora: row exited with code 124

## Matrix Abort 20260630T123004Z

- gpu/cora: row exited with code 124

## Matrix Abort 20260630T123749Z

- gpu/cora: row exited with code 124

## Matrix Abort 20260630T124827Z

- gpu/cora: row exited with code 124

## Matrix Abort 20260701T031512Z

- gpu/cora: row exited with code 124

## Matrix Abort 20260701T032033Z

- gpu/cora: row exited with code 124

## Matrix Abort 20260701T032601Z

- gpu/cora: row exited with code 124

## Matrix Abort 20260701T033658Z

- gpu/cora: row exited with code 124

## Matrix Abort 20260701T034602Z

- gpu/cora: row exited with code 124

## Matrix Abort 20260701T035055Z

- gpu/cora: row exited with code 124

## Matrix Abort 20260701T035608Z

- gpu/cora: row exited with code 124

## Matrix Abort 20260701T040045Z

- gpu/cora: row exited with code 124

## Matrix Abort 20260701T040737Z

- gpu/cora: row exited with code 124

## Matrix Abort 20260701T041302Z

- gpu/cora: row exited with code 124

## Matrix Abort 20260701T041747Z

- gpu/cora: row exited with code 124

## Matrix Abort 20260701T042005Z

- gpu/cora: row exited with code 124
