# Direction Guard

Updated: 2026-06-29T06:31:35Z

## Patch `CACHE-CORRECT-001`: remove proof-specific persistent caches

- depends on failed-run cache: no
- depends on timeout-run partial artifact: no
- changes proof relation: no
- changes verifier equation: no
- changes transcript/order: no
- changes public/private boundary: no
- Cora-only special case: no
- effective in clean/cold-cache single official row: yes, by preventing invalid cache hits and forcing real hot-path measurement
- if cold-cache canary still times out, continue filling cache: no
- if ineffective, rollback or disable: yes

Files:

```text
src/protocol/trace.cpp
src/protocol/prover.cpp
```

## Patch `HOTPATH-001`: same-row dynamic commitment reuse

- depends on failed-run cache: no
- depends on timeout-run partial artifact: no
- changes proof relation: no
- changes verifier equation: no
- changes transcript/order: no
- changes public/private boundary: no
- Cora-only special case: no
- effective in clean/cold-cache single official row: yes, duplicate polynomials in the same row avoid repeated MSM inputs
- if cold-cache canary still times out, continue filling cache: no
- if ineffective, rollback or disable: yes

Files:

```text
src/protocol/trace.cpp
```

## Patch `HOTPATH-015`: same-run physical polynomial coset alias cache

- timestamp_utc: 2026-06-29T06:45:41Z
- depends on failed-run cache: no
- depends on timeout-run partial artifact: no
- changes proof relation: no
- changes verifier equation: no
- changes transcript/order: no
- changes public/private boundary: no
- Cora-only special case: no
- effective in clean/cold-cache single official row: no, reverted after `runs/secure_main_runner/logs/run_cpu_cora_20260629T065313Z.log` showed no `coset_alias` hits and still timed out during `t_edge` base prewarm
- if cold-cache canary still times out, continue filling cache: no
- if ineffective, rollback or disable: yes, reverted from source after no-hit evidence
- target evidence: `Polynomial::from_shared_evaluations()` keeps shared backing vectors for large shared columns, while `runs/secure_main_runner/logs/run_cpu_cora_20260629T062636Z.log` still shows 276 base edge coset transforms before timeout
- no-hit evidence: `runs/secure_main_runner/logs/run_cpu_cora_20260629T065313Z.log` contains no `coset_alias` progress records; progress improved to hidden head 7 but is not attributable to this patch under current evidence

Files:

```text
src/protocol/prover.cpp
```

## Patch `HOTPATH-007`: increase KZG batch outer scheduling for medium dynamic batches

- depends on failed-run cache: no
- depends on timeout-run partial artifact: no
- changes proof relation: no
- changes verifier equation: no
- changes transcript/order: no
- changes public/private boundary: no
- Cora-only special case: no
- effective in clean/cold-cache single official row: intended, logs show many 32k edge commitments in the same batch each take about 1.4s while the whole batch takes 4.8-5.0s, indicating outer batch waves are still limiting throughput
- if cold-cache canary still times out, continue filling cache: no
- if ineffective, rollback or disable: yes
- target evidence: `runs/secure_main_runner/logs/run_cpu_cora_20260628T104530Z.log` still times out after hidden head 6's size-43 batch, with repeated size-43 batches around 4.8-5.0s and size-17 batches around 1.9s

Files:

```text
src/crypto/kzg.cpp
```

## Patch `HOTPATH-008`: derive PSQ weighted-sum commitments in-batch

- depends on failed-run cache: no
- depends on timeout-run partial artifact: no
- changes proof relation: no
- changes verifier equation: no
- changes transcript/order: no
- changes public/private boundary: no
- Cora-only special case: no
- effective in clean/cold-cache single official row: intended, `w_psq = U + lambda_psq * widehat_v_pre_star` and output `w = U + lambda_out * widehat_y_star` keep their original witness/opening polynomials while their commitment points are derived from already committed same-batch or earlier points
- if cold-cache canary still times out, continue filling cache: no
- if ineffective, rollback or disable: yes
- target evidence: `runs/secure_main_runner/logs/run_cpu_cora_20260628T105415Z.log` reaches all hidden heads but still times out after the output initial batch; hidden size-43 batches still pay one full edge MSM for `w_psq` per head and the output initial batch pays one for `w`

Files:

```text
src/protocol/trace.cpp
```

## Patch `HOTPATH-010`: raise mid-size MSM inner worker threshold

- depends on failed-run cache: no
- depends on timeout-run partial artifact: no
- changes proof relation: no
- changes verifier equation: no
- changes transcript/order: no
- changes public/private boundary: no
- Cora-only special case: no
- effective in clean/cold-cache single official row: intended, `runs/secure_main_runner/logs/run_cpu_cora_20260629T055913Z.log` reaches the output initial batch but repeatedly spends about 1.4s on single 13k-nonzero edge MSMs such as `R_t` and `R_dst`; raising the 8k-16k MSM inner worker count from 2 to 4 targets those batches while large size-43 batches remain capped by outer batch scheduling
- if cold-cache canary still times out, continue filling cache: no
- if ineffective, rollback or disable: yes
- target evidence: `runs/secure_main_runner/logs/run_cpu_cora_20260629T055913Z.log` times out at output initial MSM after hidden heads complete; repeated size-17 and size-5 batches show `P_h*_R_t` and `P_h*_R_dst` at about 1.41-1.42s each

Files:

```text
src/crypto/curve.cpp
```

## Patch `HOTPATH-011`: raise t_edge coset prewarm worker cap

- depends on failed-run cache: no
- depends on timeout-run partial artifact: no
- changes proof relation: no
- changes verifier equation: no
- changes transcript/order: no
- changes public/private boundary: no
- Cora-only special case: no
- effective in clean/cold-cache single official row: intended, `runs/secure_main_runner/logs/run_cpu_cora_20260629T060524Z.log` completes trace and times out inside `prove:quotient:t_edge:coset_prewarm` with 303 independent base coset requests; the current cap starts only 8 workers on a 20-core host, leaving CPU parallelism unused during independent coefficient/coset evaluation work
- if cold-cache canary still times out, continue filling cache: no
- if ineffective, rollback or disable: yes
- target evidence: `runs/secure_main_runner/logs/run_cpu_cora_20260629T060524Z.log` times out after reaching base `P_h3_R_dst`, before derived coset requests or quotient evaluation, so the bottleneck has moved from trace MSM to quotient coset prewarm scheduling

Files:

```text
src/protocol/prover.cpp
```

## Patch `HOTPATH-012`: cap outer quotient materialization around internally parallel t_edge

- depends on failed-run cache: no
- depends on timeout-run partial artifact: no
- changes proof relation: no
- changes verifier equation: no
- changes transcript/order: no
- changes public/private boundary: no
- Cora-only special case: no
- effective in clean/cold-cache single official row: intended, `runs/secure_main_runner/logs/run_cpu_cora_20260629T061506Z.log` shows `t_FH`, `t_edge`, and then `t_in/t_cat/t_C/t_N` materialization running concurrently while `t_edge` already has 15 internal coset-prewarm workers, creating nested CPU and memory-bandwidth contention in the final runner path
- if cold-cache canary still times out, continue filling cache: no
- if ineffective, rollback or disable: yes
- target evidence: the same log confirms `t_edge` worker count increased to 15 and progressed further, but the timeout still occurred before `t_edge` base prewarm completed while other quotient workers were active

Files:

```text
src/protocol/prover.cpp
```

## Patch `HOTPATH-013`: narrow t_edge quotient dependencies and derive Delta cosets

- depends on failed-run cache: no
- depends on timeout-run partial artifact: no
- changes proof relation: no
- changes verifier equation: no
- changes transcript/order: no
- changes public/private boundary: no
- Cora-only special case: no
- effective in clean/cold-cache single official row: intended, `evaluate_t_edge()` only accesses a subset of edge-domain labels returned by `domain_opening_labels(context, "edge")`; narrowing the materializer/memoizer dependency list to the actually evaluated labels and deriving `Delta = M_edge - Z` cosets in-process reduces unnecessary base coset FFTs without changing commitments or opening bundle order
- if cold-cache canary still times out, continue filling cache: no
- if ineffective, rollback or disable: yes
- target evidence: `runs/secure_main_runner/logs/run_cpu_cora_20260629T061506Z.log` still reports `base_count=303:derived_count=97` and times out during base prewarm, so reducing base coset requests is the next measured hot path

Files:

```text
src/protocol/prover.cpp
```

## Patch `HOTPATH-014`: closed-form coefficients for prefix selector cosets

- depends on failed-run cache: no
- depends on timeout-run partial artifact: no
- changes proof relation: no
- changes verifier equation: no
- changes transcript/order: no
- changes public/private boundary: no
- Cora-only special case: no
- effective in clean/cold-cache single official row: no, reverted after `runs/secure_main_runner/logs/run_cpu_cora_20260629T063638Z.log` still failed to complete `t_FH`'s `P_Q_tbl_feat` coset and regressed `t_edge` base-prewarm progress versus HOTPATH-012/013
- if cold-cache canary still times out, continue filling cache: no
- if ineffective, rollback or disable: yes, reverted from source after regression evidence
- target evidence: Cora `FH=4194304` and `edge=131072`; the current materializer converts prefix selectors from dense evaluation vectors via `coefficients_native_from_polynomial()` before coset NTT, so selector labels pay an avoidable inverse NTT on the hot quotient path
- regression evidence: `runs/secure_main_runner/logs/run_cpu_cora_20260629T063638Z.log` shows closed-form selector hooks firing for `P_Q_tbl_feat` and edge selectors, but timeout still occurs during `t_edge` base prewarm and `t_FH` does not advance beyond the selector coset

Files:

```text
src/protocol/prover.cpp
```

## Patch `HOTPATH-009`: edge-lift commitment point precomputation

- depends on failed-run cache: no
- depends on timeout-run partial artifact: no
- changes proof relation: no
- changes verifier equation: no
- changes transcript/order: no
- changes public/private boundary: no
- Cora-only special case: no
- effective in clean/cold-cache single official row: no, after SRS lookup repair it regressed canary progress to hidden head 7's size-43 batch and did not beat HOTPATH-008's output-initial timeout point
- if cold-cache canary still times out, continue filling cache: no
- if ineffective, rollback or disable: yes, reverted from source after regression evidence
- target evidence: `runs/secure_main_runner/logs/run_cpu_cora_20260628T110347Z.log` still times out at the output initial batch; repeated hidden batches spend most time on broadcast/group-target edge commitments such as `T_psq_edge`, `H_agg_*_edge`, `R_*`, and `*_edge`
- regression evidence: `runs/secure_main_runner/logs/run_cpu_cora_20260628T112111Z.log` failed on missing edge-domain SRS lookup; `runs/secure_main_runner/logs/run_cpu_cora_20260628T112554Z.log` passed that lookup but timed out at hidden head 7, showing the per-run edge-lift precompute overhead outweighed removed batch MSM inputs

Files:

```text
src/protocol/trace.cpp
```

## Patch `HOTPATH-006`: in-batch `Delta = M_edge - Z` commitment derivation

- depends on failed-run cache: no
- depends on timeout-run partial artifact: no
- changes proof relation: no
- changes verifier equation: no
- changes transcript/order: no
- changes public/private boundary: no
- Cora-only special case: no
- effective in clean/cold-cache single official row: intended, `Delta` remains in the original batch position with its original witness/opening polynomial while its point is derived from same-batch `M_edge` and `Z`
- if cold-cache canary still times out, continue filling cache: no
- if ineffective, rollback or disable: yes
- target evidence: `runs/secure_main_runner/logs/run_cpu_cora_20260628T103736Z.log` shows HOTPATH-005 advanced the timeout to after hidden head 6's size-43 batch, leaving the initial size-16 and subsequent large batches as active bottlenecks

Files:

```text
src/protocol/trace.cpp
```

## Patch `HOTPATH-005`: in-batch deferred linear commitment derivation

- depends on failed-run cache: no
- depends on timeout-run partial artifact: no
- changes proof relation: no
- changes verifier equation: no
- changes transcript/order: no
- changes public/private boundary: no
- Cora-only special case: no
- effective in clean/cold-cache single official row: intended, same-batch query columns keep their labels and opening polynomials while their commitment points are derived after the batch MSM from already committed same-batch points
- if cold-cache canary still times out, continue filling cache: no
- if ineffective, rollback or disable: yes
- prior negative evidence avoided: HOTPATH-004 split batches and regressed `cora/cpu`; HOTPATH-005 must not split those batches
- target evidence: `runs/secure_main_runner/logs/run_cpu_cora_20260628T102720Z.log` shows `Query_src` and `Query_ELU` still paid MSM cost inside the original large batches

Files:

```text
src/protocol/trace.cpp
```

## Patch `HOTPATH-002`: t_FH same-row dependency reuse

- depends on failed-run cache: no
- depends on timeout-run partial artifact: no
- changes proof relation: no
- changes verifier equation: no
- changes transcript/order: no
- changes public/private boundary: no
- Cora-only special case: no
- effective in clean/cold-cache single official row: yes, it reuses the current trace commitment opening polynomial for `P_Table_feat` instead of rebuilding equivalent table material
- if cold-cache canary still times out, continue filling cache: no
- if ineffective, rollback or disable: yes

Files:

```text
src/protocol/prover.cpp
```

## Patch `HOTPATH-003`: SRS selective loading and MSM worker scheduling

- depends on failed-run cache: no
- depends on timeout-run partial artifact: no
- changes proof relation: no
- changes verifier equation: no
- changes transcript/order: no
- changes public/private boundary: no
- Cora-only special case: no
- effective in clean/cold-cache single official row: yes, it reduces static SRS materialization and avoids small MSM oversubscription
- if cold-cache canary still times out, continue filling cache: no
- if ineffective, rollback or disable: yes

Files:

```text
include/gatzk/crypto/kzg.hpp
src/crypto/kzg.cpp
src/protocol/prover.cpp
```

## Patch `HOTPATH-004`: same-row linear commitment derivation for lookup queries

- depends on failed-run cache: no
- depends on timeout-run partial artifact: no
- changes proof relation: no
- changes verifier equation: no
- changes transcript/order: no
- changes public/private boundary: no
- Cora-only special case: no
- effective in clean/cold-cache single official row: no, this version split large dynamic commitment batches and regressed the Cora CPU canary from timeout at hidden head 6 to timeout at hidden head 4
- if cold-cache canary still times out, continue filling cache: no
- if ineffective, rollback or disable: yes
- keep/revert decision: revert after clean/cold-cache canary regression
- regression evidence: `runs/secure_main_runner/logs/run_cpu_cora_20260628T101858Z.log`

Files:

```text
src/protocol/trace.cpp
```
