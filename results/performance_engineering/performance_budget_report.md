# zkGAT Performance Budget Report

Generated: 2026-06-19 15:55 Asia/Shanghai

This report is diagnostic only. It does not mark `results/paper_secure` complete, does not authorize the final 10-row matrix, and does not authorize a new Cora official row by itself. The latest completed Cora evidence below was generated with `--skip-export`; `results/paper_secure/latest.*` was not regenerated.

## Current Row

Latest completed current-source Cora evidence:

- path: `runs/cora_secure/paper_secure_final/cpu/warm/run_manifest.json`
- runner log: `runs/secure_main_runner/logs/run_cpu_cora_20260619T071854Z.log`
- status: `verified=true`, `exit_code=0`, `timeout=false`, `stale_result=false`
- source tree hash: `6c29826ee43f5209948d4d39c31f8005e36f52b69ce87eb7a00021d9635b63dd`
- CPU binary hash: `c8f7cfa666d217ca4c3a47172984a58ce01ec116b7abd058bad5ddda6b928d09`
- runner hash: `96d993f1103c8960252c97fd2e0596f742ac5e07377108fcbe39e23a5e2aa0c3`
- manifest hash: `80bb593163f2b1b1cb55aa3c6270d5d66331d35eae6a7179e3cd7ef744e2d873`

There is a newer local t_FH/t_edge coset-fusion patch in the working tree. It has not produced a Cora official row and is not treated as the main performance route.

## End-To-End Budget

| field | ms | seconds | note |
|---|---:|---:|---|
| total wall clock | 785,469.068 | 785.469 | row wall clock from `gatzk_run` |
| official runner wall clock | 867,633.034 | 867.633 | Python runner plus subprocess |
| runner overhead | 82,163.966 | 82.164 | orchestration/validation/archive cost |
| context build | 480.420 | 0.480 | warm static context path |
| reference forward | 1,054.138 | 1.054 | public logits/reference check |
| trace generation | 18,193.876 | 18.194 | full-dataset trace |
| commitment | 207,287.233 | 207.287 | dynamic and quotient KZG commitments |
| dynamic commitment | 113,889.402 | 113.889 | dynamic commitment path |
| dynamic commitment MSM | 113,527.543 | 113.528 | SRS MSM inside dynamic commitments |
| prove | 776,603.739 | 776.604 | includes quotient/opening/external opening |
| quotient build | 349,373.872 | 349.374 | dominant quotient materialization |
| quotient t_FH | 253,648.317 | 253.648 | FH identity quotient |
| quotient t_edge | 214,312.876 | 214.313 | edge identity quotient; overlaps with t_FH timing because quotient materialization is parallel |
| quotient commitment | 93,397.831 | 93.398 | KZG commitments for quotient polynomials |
| domain opening | 259,075.690 | 259.076 | domain-grouped batch openings |
| FH domain opening | 255,265.889 | 255.266 | dominant opening bundle |
| FH opening witness | 245,564.321 | 245.564 | SRS opening witness work |
| edge domain opening | 37,122.651 | 37.123 | edge bundle |
| external opening | 34,977.816 | 34.978 | external fold opening |
| verify | 8,354.370 | 8.354 | verifier-side checks |
| payload audit | 0.069 | 0.000 | passed |
| reference consistency | 0.001 | 0.000 | passed |
| tamper tests | 0.097 | 0.000 | passed |
| export artifact | 16.892 | 0.017 | per-row artifact write/validation, not final export |
| profile logging overhead | 1.317 | 0.001 | negligible |

Gate status:

- Cora CPU hard gate: `<60,000 ms`.
- Latest Cora CPU: `785,469.068 ms`, which is `13.09x` over the hard gate and `78.55x` over a 10s target.
- Previous current-source Cora before later optimizations was `925,920.760 ms`, which was `15.43x` over the hard gate and `92.59x` over 10s.
- Readiness: `NOT_PERFORMANCE_READY`.

`prove_time_ms`, commitment subfields, quotient subfields, and opening subfields overlap in the measured implementation; percentages and Amdahl subtractions below are upper-bound counterfactuals, not additive budgets.

## Amdahl Upper-Bound Analysis For Cora

This hard section controls whether a new Cora official row is allowed. It uses the latest completed Cora manifest/profile/log listed above.

### Required Inputs

| item | value |
|---|---:|
| `total_wall_clock_ms` | 785,469.068 |
| `context_build_ms` | 480.420 |
| `trace_generation_ms` | 18,193.876 |
| `commitment_time_ms` | 207,287.233 |
| `commit_dynamic_ms` | 113,889.402 |
| `quotient_build_ms` | 349,373.872 |
| `quotient_t_fh_ms` | 253,648.317 |
| `quotient_t_edge_ms` | 214,312.876 |
| `quotient_commitment_ms` | 93,397.831 |
| `domain_opening_ms` | 259,075.690 |
| `domain_open_FH_ms` | 255,265.889 |
| `fh_open_witness_ms` | 245,564.321 |
| `external_opening_ms` | 34,977.816 |
| `verify_time_ms` | 8,354.370 |
| audit/reference/tamper/export | 17.059 combined |
| runner overhead | 82,163.966 |

Proof/object counts:

- dynamic commitment labels in proof order: `734` from `commitment_order.txt`;
- profiled dynamic KZG objects: `729`;
- quotient commitments: `7` (`t_FH`, `t_edge`, `t_in`, `t_d_h`, `t_cat`, `t_C`, `t_N`);
- proof commitment elements before static public metadata: approximately `741` dynamic plus quotient commitments;
- domain opening bundles: `7` (`FH`, `edge`, `in`, `d_h`, `cat`, `C`, `N`);
- domain opening points: `14` bundle-points, two per domain bundle;
- external opening evaluations for the Cora multihead shape: `55` by `multihead_external_specs` (`8 hidden heads * 6`, one hidden concat, four output-head values, and two output logits values);
- proof block order count: `9` blocks (`M_pub`, `Com_dyn`, `S_route`, `Eval_ext`, `Eval_dom`, `Com_quot`, `Open_dom`, `W_ext`, `Pi_bind`).

Major MSM sizes and object/domain breakdown:

| domain/object class | count | effective scalars | cumulative profile MSM ms | notes |
|---|---:|---:|---:|---|
| dynamic edge objects | 380 | 3,391,496 | 240,111.799 | edge evaluation objects length 131,072 dominate dynamic commitment profile |
| dynamic N objects | 162 | 400,348 | 47,104.553 | node-domain route/trace objects |
| dynamic coefficient objects | 28 | 551,216 | 16,814.266 | matrix/linearized coefficient objects |
| dynamic input objects | 24 | 38,390 | 9,547.886 | input-domain objects |
| dynamic d objects | 120 | 1,223 | 663.975 | hidden-width domain |
| dynamic cat objects | 6 | 512 | 243.328 | concat domain |
| dynamic C objects | 9 | 89 | 49.683 | class domain |
| quotient `t_FH` commitment | 1 | about 8,388,608 coefficients | included in `quotient_commitment_ms` | FH quotient coset multiplier is 2 over FH=4,194,304 |
| quotient `t_edge` commitment | 1 | about 524,288 coefficients | included in `quotient_commitment_ms` | edge quotient coset multiplier is 4 over edge=131,072 |
| FH opening witness | 1 bundle, 2 points | 8,388,608-scale coefficient witness plus FH-domain evaluation fold | 245,564.321 ms | dominant opening cost |

The largest profiled dynamic MSM objects are edge lookup/route objects. Examples from the latest profile include `P_h1_R_ELU` at `2,412.157 ms`, `P_h7_R_ELU` at `2,406.626 ms`, and `P_h1_R_R` at `1,684.299 ms`.

NTT/IFFT counts from the current materialization path:

- FH quotient construction uses four large FH evaluation dependencies in the completed row: `P_Q_tbl_feat`, `P_Row_feat_tbl`, `P_Col_feat_tbl`, and `P_T_H`.
- Each FH dependency in that row requires one FH-domain IFFT of size `4,194,304` and one coset NTT of size `8,388,608`; final quotient coefficients require one more IFFT of size `8,388,608`.
- t_edge prewarm reports `base_count=303` and `derived_count=97` in the Cora log. Base dependencies require target coset NTTs of size `524,288`; evaluation-basis base dependencies also require source-domain IFFTs. Derived dependencies are linear coset combinations and avoid additional NTTs.
- The local unprofiled coset-fusion patch can at most reduce a subset of the FH/t_edge quotient materialization transforms; it does not reduce proof object count, commitment count, opening count, or KZG witness equations.

### Counterfactual Upper Bounds

| counterfactual | remaining total ms | remaining seconds | upper-bound speedup | below 60s? | near 10s? |
|---|---:|---:|---:|---|---|
| `quotient_build_ms` becomes 0 | 436,095.196 | 436.095 | 1.80x | no | no |
| `quotient_commitment_ms` becomes 0 | 692,071.237 | 692.071 | 1.13x | no | no |
| `domain_opening_ms` becomes 0 | 526,393.378 | 526.393 | 1.49x | no | no |
| `commit_dynamic_ms` becomes 0 | 671,579.666 | 671.580 | 1.17x | no | no |
| only t_FH/t_edge coset materialization is optimized | at least 436,095.196 | at least 436.095 | at most 1.80x | no | no |

The last row is deliberately generous: even deleting the entire `quotient_build_ms` critical-path cost, which is more than the local t_FH/t_edge coset fusion can do, leaves Cora at `436.095s`. Therefore this local patch cannot make Cora `<60s`, cannot approach `10s`, and does not meet the `>=2x` end-to-end potential requirement for a new Cora official row.

### Required Speedup

- Latest Cora to `<60s`: `13.09x` end-to-end speedup required.
- Latest Cora to `10s`: `78.55x` end-to-end speedup required.
- Previous `925.921s` Cora to `<60s`: `15.43x` required.
- Previous `925.921s` Cora to `10s`: `92.59x` required.

Single-stage optimization is not sufficient:

- optimizing quotient alone is insufficient because the zero-quotient upper bound is still `436.095s`;
- optimizing opening alone is insufficient because the zero-opening upper bound is still `526.393s`;
- optimizing dynamic commitment alone is insufficient because the zero-dynamic-commitment upper bound is still `671.580s`;
- optimizing quotient commitment alone is insufficient because the zero-quotient-commitment upper bound is still `692.071s`.

Stages that must drop together:

- proof object / commitment / opening count or KZG work must fall substantially, not just per-object scheduling;
- FH quotient/opening work must fall by an order of magnitude;
- edge dynamic commitments and edge quotient materialization must fall together, especially for PubMed/PPI/OGBN scaling;
- external opening and verifier time are secondary but become relevant once the main KZG/quotient costs are reduced;
- runner overhead must be revisited after proof time is reduced, because the current `82.164s` runner overhead alone exceeds the hard gate.

Decision: the current t_FH/t_edge coset-fusion patch is at most an auxiliary local optimization. It is not a main route and does not authorize another Cora official row.

## PubMed Profiling Context

Latest available PubMed full-dataset profiling evidence is stale for final use but useful for route selection:

- path: `runs/pubmed_secure/paper_secure_final/cpu/warm/run_manifest.json`
- log: `runs/secure_main_runner/logs/run_cpu_pubmed_20260616T093124Z.log`
- source tree hash: `091322e843a27d49b983c969df854aed3572daac33280dfd730ad7ac04c44b5e`
- `commitment_time_ms=895,303.823`
- `commit_dynamic_ms=517,208.607`
- `quotient_build_ms=1,476,653.437`
- `quotient_t_fh_ms=1,091,480.221`
- `quotient_t_edge_ms=449,073.490`
- `domain_opening_ms=948,546.527`
- `domain_open_FH_ms=931,796.074`
- `fh_open_witness_ms=916,202.968`
- `external_opening_ms=107,478.948`
- `verify_time_ms=20,015.304`

PubMed amplifies both FH and edge costs: FH domain size is `16,777,216`, edge domain size is `262,144`, and dynamic edge objects account for `1,284,454.809 ms` cumulative profiled MSM time across 380 edge objects.

## Current Bottleneck Conclusion

Performance readiness is `NOT_PERFORMANCE_READY`. The next main route must be structural: reducing proof objects/commitments/opening/materialization counts, adding true GPU MSM/NTT/opening/quotient execution, domain-level aggregation, streaming quotient-to-MSM, cross-object reuse, or public/static precomputation with cold/warm/amortized accounting. Worker caps, local coset linear-combination changes, single-MSM tuning, and copy/move cleanup are not eligible as the next main route unless an Amdahl model first shows at least `2x` end-to-end potential.
