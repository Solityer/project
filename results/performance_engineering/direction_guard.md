# Direction Guard

Updated: 2026-06-28T11:13:28Z

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

## Patch `HOTPATH-009`: edge-lift commitment point precomputation

- depends on failed-run cache: no
- depends on timeout-run partial artifact: no
- changes proof relation: no
- changes verifier equation: no
- changes transcript/order: no
- changes public/private boundary: no
- Cora-only special case: no
- effective in clean/cold-cache single official row: intended, edge-domain columns that are public-graph src/dst broadcasts of node-domain witness values keep their original edge-domain witness/opening polynomial while their commitment points are computed from per-run aggregated edge Lagrange SRS buckets
- if cold-cache canary still times out, continue filling cache: no
- if ineffective, rollback or disable: yes
- target evidence: `runs/secure_main_runner/logs/run_cpu_cora_20260628T110347Z.log` still times out at the output initial batch; repeated hidden batches spend most time on broadcast/group-target edge commitments such as `T_psq_edge`, `H_agg_*_edge`, `R_*`, and `*_edge`

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
