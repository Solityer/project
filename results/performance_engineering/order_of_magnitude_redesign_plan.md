# zkGAT Order-Of-Magnitude Redesign Plan

Generated: 2026-06-19 15:55 Asia/Shanghai

This plan is a performance-engineering gate. It does not authorize final matrix execution, final export, protocol weakening, dataset reduction, fake KZG, runtime tau, verifier witness access, or manifest relabeling.

## Evidence Baseline

Latest completed Cora profiling evidence:

- `total_wall_clock_ms=785,469.068`
- `official_runner_wall_clock_ms=867,633.034`
- `commit_dynamic_ms=113,889.402`
- `quotient_build_ms=349,373.872`
- `quotient_t_fh_ms=253,648.317`
- `quotient_t_edge_ms=214,312.876`
- `quotient_commitment_ms=93,397.831`
- `domain_opening_ms=259,075.690`
- `domain_open_FH_ms=255,265.889`
- `fh_open_witness_ms=245,564.321`
- `external_opening_ms=34,977.816`
- `verify_time_ms=8,354.370`

Latest available PubMed profiling evidence is stale for final use but useful for scaling:

- `commitment_time_ms=895,303.823`
- `commit_dynamic_ms=517,208.607`
- `quotient_build_ms=1,476,653.437`
- `quotient_t_fh_ms=1,091,480.221`
- `quotient_t_edge_ms=449,073.490`
- `domain_opening_ms=948,546.527`
- `domain_open_FH_ms=931,796.074`
- `fh_open_witness_ms=916,202.968`
- PubMed FH domain: `16,777,216`
- PubMed edge domain: `262,144`

Amdahl gate from `performance_budget_report.md`: even deleting all Cora `quotient_build_ms` leaves `436.095s`, so local quotient materialization patches cannot be the main route and cannot trigger a Cora official row.

## Route Priority

Priority order:

1. True GPU backend for KZG MSM, NTT/IFFT, opening witness, and quotient kernels.
2. Domain-grouped multi-opening aggregation and proof-object/domain-level aggregation.
3. Streaming quotient-to-MSM and fused execution for quotient commitment/opening.
4. Cross-object reuse and fused materialization.
5. Public/static artifact precomputation with cold/warm/amortized accounting.
6. Worker/thread/copy/local coset tuning only after a route above proves at least `2x` end-to-end potential.

The next main prototype should be Route B, true GPU backend, because CPU-only local changes cannot show a credible path to Cora `<60s` and OGBN near `100s`. Route D should be designed in parallel because even a fast GPU MSM does not reduce opening count.

## Route A - Proof-Object / Domain-Level Aggregation

1. Acting stage: dynamic commitments, quotient commitments, domain openings, proof payload size.
2. Current stage time: Cora `commit_dynamic_ms=113,889.402`, `quotient_commitment_ms=93,397.831`, `domain_opening_ms=259,075.690`; PubMed `commit_dynamic_ms=517,208.607`, `domain_opening_ms=948,546.527`.
3. Share of Cora wall clock: dynamic commitment 14.5%, quotient commitment 11.9%, domain opening 33.0%.
4. Theoretical maximum saved time: up to about `466s` if these three Cora classes vanished, but they overlap with proof accounting; a realistic aggregation prototype must report measured critical-path savings.
5. Expected end-to-end speedup: potentially `2x-5x` only if it reduces actual commitment/opening counts, not just folds inside the same count.
6. Can make Cora `<60s`: not alone unless combined with GPU or quotient reduction; possible as part of a combined route.
7. Can approach Cora `10s`: no alone.
8. Can make OGBN near `100s`: not alone; OGBN needs GPU-scale acceleration plus fewer openings.
9. Protocol relation: reducing committed objects generally changes proof schema or batched relation encoding. A safe design must prove that the verifier still checks the same underlying GAT constraints via sound random linear combinations.
10. Security/config: must keep full dataset, hidden model/trace, public graph/features/logits, real KZG/SRS, and VK-only verifier.
11. Deterministic transcript/order: must add deterministic aggregation labels and Fiat-Shamir challenges without changing row/dataset semantics.
12. Code paths: `src/protocol/trace.cpp`, `src/protocol/schema.cpp`, `src/protocol/prover.cpp`, `src/protocol/verifier.cpp`, proof serialization, tamper tests.
13. Minimum prototype: aggregate a small family of same-domain lookup accumulator commitments into one domain-packed object while preserving individual equation checks via verifier-side folded constraints.
14. Success criterion: object/commitment/opening count drops measurably, Cora stage model predicts at least `2x` end-to-end speedup before an official row.
15. Failure fallback: if schema-safe aggregation cannot reduce counts without changing relation semantics, move to Route B and D.

Aggregation status:

- Can aggregate: repeated same-domain lookup/route families if verifier equations are rewritten as sound batched constraints.
- Cannot blindly aggregate: objects whose individual commitments are directly referenced by current verifier equations, external evaluations, or transcript order unless those equations are redesigned and revalidated.
- Current code already uses same-domain batch opening within bundles; this route must reduce counts or witnesses beyond the current `domain_grouped_batch_opening`.

## Route B - True GPU Backend

1. Acting stage: KZG G1 MSM, NTT/IFFT, quotient materialization, opening witness generation.
2. Current stage time: Cora dynamic MSM `113,527.543 ms`, quotient build `349,373.872 ms`, quotient commitment `93,397.831 ms`, FH witness `245,564.321 ms`; PubMed corresponding stages are much larger.
3. Share of Cora wall clock: these classes cover most of the critical proof path.
4. Theoretical maximum saved time: if GPU accelerates MSM/NTT/opening/quotient by `10x`, Cora can plausibly move from hundreds of seconds toward the hard gate, but runner overhead must also be reduced/accounted.
5. Expected end-to-end speedup: `5x-15x` for a complete GPU backend; lower if only trace kernels move to GPU.
6. Can make Cora `<60s`: yes only if GPU covers MSM plus NTT/IFFT plus opening witness, not CPU fallback.
7. Can approach Cora `10s`: possible only with mature GPU MSM/NTT and reduced opening/aggregation counts.
8. Can make OGBN near `100s`: this is the only currently credible route, combined with opening aggregation.
9. Protocol relation: unchanged if GPU kernels compute the same KZG MSM, NTT/IFFT, quotient, and witness values.
10. Security/config: unchanged; real SRS points and real KZG openings must be used.
11. Deterministic transcript/order: unchanged; GPU kernels must be deterministic for the configured backend or the proof hashes must document deterministic reduction order.
12. Code paths: `src/crypto/curve.cpp`, `src/crypto/kzg.cpp`, `src/algebra/eval_backend.cpp`, CUDA build files, GPU memory/SRS cache code, runner/backend flags.
13. Minimum prototype: GPU G1 MSM for large SRS-backed MSM sizes `>=2^22`, with host-device transfer timed and correctness checked against CPU MSM.
14. Success criterion: stage benchmark shows at least `8x` on 8M-scale MSM and at least `2x` predicted end-to-end Cora speedup before any official Cora row.
15. Failure fallback: if GPU MSM/NTT cannot be made real quickly, switch to Route D plus Route C to reduce witness/materialization counts on CPU.

Current GPU status:

- Latest Cora and PubMed evidence are CPU rows: `compute_backend_name=cpu`, `crypto_backend_name=mcl`, `gpu_acceleration_status=not_requested_cpu_run`, `gpu_runtime_present=false`.
- Existing CUDA tests do not prove that KZG MSM, NTT/IFFT, opening witness, or quotient materialization are executed on GPU.
- GPU final rows cannot count as performance PASS until fallback status is explicit and the bottlenecks above run on GPU with timed transfers.
- Code audit details are recorded in `results/performance_engineering/gpu_backend_gap_report.md`.

## Route C - Streaming Quotient-To-MSM

1. Acting stage: quotient build, quotient commitment, FH/edge opening witness.
2. Current stage time: Cora `quotient_build_ms=349,373.872`, `quotient_commitment_ms=93,397.831`, `fh_open_witness_ms=245,564.321`.
3. Share of Cora wall clock: quotient build 44.5%, quotient commitment 11.9%, FH witness 31.3%.
4. Theoretical maximum saved time: bounded by quotient build plus some memory traffic around commitment/opening; it does not delete KZG MSM.
5. Expected end-to-end speedup: `1.5x-3x` if fused with GPU/MSM; less than `2x` if it only avoids vector copies.
6. Can make Cora `<60s`: no alone.
7. Can approach Cora `10s`: no alone.
8. Can make OGBN near `100s`: only with GPU and opening aggregation.
9. Protocol relation: unchanged if the final committed quotient polynomial and opening witnesses are algebraically identical to materialized KZG objects.
10. Security/config: unchanged; no fake commitment, no runtime tau, no verifier witness access.
11. Deterministic transcript/order: unchanged if commitment order and challenge order remain fixed.
12. Code paths: `src/protocol/prover.cpp` quotient materializer, `src/crypto/kzg.cpp` commitment/opening API, NTT plan/execution code.
13. Minimum prototype: stream `t_FH` inverse NTT output directly into a KZG MSM sink and verify the resulting commitment equals the materialized path on small and medium domains.
14. Success criterion: stage benchmark removes a full materialization pass and predicts at least `2x` end-to-end when combined with Route B or D.
15. Failure fallback: if memory traffic is not the critical path, prioritize GPU MSM/NTT and opening aggregation.

## Route D - Domain-Grouped Multi-Opening Aggregation

1. Acting stage: `domain_opening_ms`, especially FH opening witness.
2. Current stage time: Cora `domain_opening_ms=259,075.690`, `domain_open_FH_ms=255,265.889`, `fh_open_witness_ms=245,564.321`; PubMed `fh_open_witness_ms=916,202.968`.
3. Share of Cora wall clock: domain opening 33.0%, FH witness 31.3%.
4. Theoretical maximum saved time: deleting all domain opening leaves `526.393s`, so this must combine with other routes.
5. Expected end-to-end speedup: `1.5x` alone at best; `2x-4x` when it reduces witness count and is combined with GPU MSM.
6. Can make Cora `<60s`: no alone; possible with Route B plus quotient reductions.
7. Can approach Cora `10s`: no alone.
8. Can make OGBN near `100s`: possible only if OGBN witness count/work is reduced and GPU-backed.
9. Protocol relation: polynomial equations unchanged; verifier opening equation changes if using a SHPLONK-style aggregated witness or cross-point aggregation.
10. Security/config: unchanged if aggregation challenge is Fiat-Shamir derived and all original claims remain bound.
11. Deterministic transcript/order: must add deterministic aggregation challenge ordering after commitments and before witness construction.
12. Code paths: `src/crypto/kzg.cpp` `open_batch`, verifier batch opening checks, proof schema for aggregated witness metadata.
13. Minimum prototype: aggregate the two FH opening points into one proof object, compare verifier acceptance and tamper rejection against current two-witness path.
14. Success criterion: FH witness stage drops by at least `2x` in a stage benchmark and Amdahl predicts at least `2x` end-to-end with Route B or C.
15. Failure fallback: keep current same-point batching and focus on GPU opening witness.

Current same-point batching status:

- Domain opening already batches labels within each domain and opens two points per domain.
- The missing structural gain is reducing witness count/work across points/domains, not tuning `point_msm_worker_cap`.

## Route E - Cross-Object Reuse / Fused Materialization

1. Acting stage: repeated coset/table/NTT/IFFT/materialized vector construction across heads, layers, and lookup tables.
2. Current stage time: Cora `quotient_t_fh_ms=253,648.317`, `quotient_t_edge_ms=214,312.876`; PubMed `quotient_t_fh_ms=1,091,480.221`, `quotient_t_edge_ms=449,073.490`.
3. Share of Cora wall clock: quotient build is 44.5%.
4. Theoretical maximum saved time: deleting all quotient build leaves `436.095s`, so reuse alone cannot pass the hard gate.
5. Expected end-to-end speedup: below `1.8x` alone; useful only as an auxiliary route.
6. Can make Cora `<60s`: no.
7. Can approach Cora `10s`: no.
8. Can make OGBN near `100s`: no alone.
9. Protocol relation: unchanged if reused values are exactly the same public/table/challenge-derived polynomials.
10. Security/config: unchanged.
11. Deterministic transcript/order: unchanged.
12. Code paths: `src/protocol/prover.cpp` quotient dependency cache, public table cache, trace materialization.
13. Minimum prototype: cache and reuse public/static cosets across quotient and opening evaluation with explicit cold/warm accounting.
14. Success criterion: useful as part of another route only; must not trigger a Cora official row unless combined Amdahl model predicts `>=2x`.
15. Failure fallback: do not repeat local coset fusion; move to Route B/D.

Current t_FH/t_edge coset fusion falls in this route. It is auxiliary only.

## Route F - Public/Static Artifact Precomputation

1. Acting stage: public/static table cosets, SRS/Lagrange SRS loading, NTT plans, immutable lookup tables.
2. Current stage time: not fully separated in the latest manifest; public/static work contributes inside context, quotient, commitment, and opening stages.
3. Share of Cora wall clock: bounded by quotient/public-table and SRS prep portions; not enough alone.
4. Theoretical maximum saved time: cannot exceed the public/static subset of quotient/opening/commitment work and must report cold cost.
5. Expected end-to-end speedup: warm-only improvement; likely below `2x` alone.
6. Can make Cora `<60s`: no alone.
7. Can approach Cora `10s`: no alone.
8. Can make OGBN near `100s`: useful for artifact workflow only if paired with GPU/aggregation.
9. Protocol relation: unchanged if artifacts are immutable public/SRS-derived data.
10. Security/config: unchanged; SRS identity, degree bound, and public table hashes must be manifest-bound.
11. Deterministic transcript/order: unchanged.
12. Code paths: context cache, SRS load/cache, table/coset cache, runner cold/warm artifact accounting.
13. Minimum prototype: serialize public table cosets or Lagrange SRS artifacts with manifest hash, then measure cold build, warm load, and amortized cost.
14. Success criterion: report cold/warm/amortized timing and prove that warm cache is a formal artifact workflow, not hidden preprocessing.
15. Failure fallback: keep public cache local only and focus on Route B/D.

## Next Cora Official Row Gate

A new Cora full-dataset official row is forbidden until all are true:

1. A selected structural route has a prototype or patch.
2. A stage benchmark shows significant target-stage improvement.
3. Amdahl modeling predicts at least `2x` end-to-end speedup.
4. CPU build/tests pass.
5. CUDA/GPU build/tests pass.
6. static and semantic secure scans pass.
7. full security regression passes.
8. source/binary hashes are re-frozen.
9. protocol/security/experiment config and deterministic transcript/order are documented as unchanged or explicitly revalidated if a proof-schema aggregation route is selected.

Until then, the readiness state remains `NOT_PERFORMANCE_READY`.
