# Resource Control Audit

## 2026-06-28T04:48:33Z Preflight

- task: zkGAT secure main experiment performance optimization.
- invariant source: `MAIN_EXPERIMENT_INVARIANTS.md` read before experiments.
- process policy: no broad kill commands; no comparison process termination; no comparison file access for evidence.
- CPU/load: load average `0.14, 0.07, 0.08`.
- memory: `121Gi` total, `6.6Gi` used, `115Gi` available, `1.3Gi/15Gi` swap used.
- disk: `/home/pzh/project` on `/dev/nvme0n1p2`, `3.7T` size, `2.1T` used, `1.5T` available.
- GPU: NVIDIA GB10, utilization `0%`, memory utilization `0%`, temperature `40C`; no compute process listed by `nvidia-smi`.
- top non-system user processes: VS Code / Codex extension hosts only, each below `4%` CPU at sampling time.
- action: no interfering process killed.

## 2026-06-28T04:57:47Z Canary Preflight

- task: before official-runner `cora/cpu` canary.
- CPU/load: load average `1.81, 1.78, 1.03`; top user processes were VS Code / Codex extension hosts below `4%` CPU at sampling time.
- memory: `121Gi` total, `7.6Gi` used, `114Gi` available.
- GPU: NVIDIA GB10, utilization `0%`, memory utilization `0%`, temperature `40C`; no compute process listed by `nvidia-smi`.
- action: no interfering process killed.
- comparison isolation: no comparison files read for evidence; no comparison process killed.

## 2026-06-28T09:56:10Z Correction Pass Resource Preflight

- task: before disabling proof-specific cross-run caches and starting clean/cold-cache optimization.
- CPU/load: load average `0.29, 0.22, 0.09`.
- memory: `121Gi` total, `6.4Gi` used, `115Gi` available, `1.3Gi/15Gi` swap used.
- GPU: NVIDIA GB10, utilization `0%`, memory utilization `0%`; `nvidia-smi` did not report memory counters for this device.
- relevant process sample: no process matched `gatzk` or `run_secure_main` after excluding the audit commands themselves.
- action: no process killed.
- comparison isolation: no comparison files read for evidence; no comparison process killed.

## 2026-06-28T10:06:48Z Clean Cold-Cache Cora CPU Canary Preflight

- task: before clean/cold-cache official-runner `cora/cpu` canary after removing proof-specific persistent cache code.
- CPU/load: load average `0.85, 0.62, 0.32`.
- memory: `121Gi` total, `7.3Gi` used, `114Gi` available, `1.3Gi/15Gi` swap used.
- GPU: NVIDIA GB10, utilization `0%`, memory utilization `0%`; `nvidia-smi` did not report memory counters for this device.
- proof-specific cache status: `runs/performance_cache` absent.
- relevant process sample: no process matched `gatzk` or `run_secure_main` after excluding the audit commands themselves.
- action: no process killed.
- comparison isolation: no comparison files read for evidence; no comparison process killed.

## 2026-06-28T10:18:40Z Linear Query Derivation Cora CPU Canary Preflight

- task: before clean/cold-cache official-runner `cora/cpu` canary after same-row linear commitment derivation for lookup query columns.
- CPU/load: load average `0.81, 0.98, 0.93`.
- memory: `121Gi` total, `7.0Gi` used, `114Gi` available, `1.3Gi/15Gi` swap used.
- GPU: NVIDIA GB10, utilization `0%`, memory utilization `0%`; `nvidia-smi` did not report memory counters for this device.
- proof-specific cache status: `runs/performance_cache` absent.
- relevant process sample: no process matched `gatzk` or `run_secure_main` after excluding the audit commands themselves.
- action: no process killed.
- comparison isolation: no comparison files read for evidence; no comparison process killed.

## 2026-06-28T05:08:39Z Canary Preflight

- task: before optimized official-runner `cora/cpu` canary after dynamic commitment reuse patch.
- CPU/load: load average `1.98, 2.04, 1.72`.
- memory: `121Gi` total, `8.4Gi` used, `113Gi` available, `1.3Gi/15Gi` swap used.
- GPU: NVIDIA GB10, utilization `0%`; `nvidia-smi` did not report memory counters for this device.
- relevant process sample: only the resource-audit `ps | rg` command matched `gatzk`, `run_secure_main`, or `comparison`.
- action: no interfering process killed.
- comparison isolation: no comparison files read for evidence; no comparison process killed.

## 2026-06-28T05:33:14Z Canary Preflight

- task: before official-runner `cora/cpu` canary after selective SRS and dynamic commitment reuse patches.
- CPU/load: load average `2.10, 2.69, 2.29`.
- memory: `121Gi` total, `7.9Gi` used, `113Gi` available, `1.3Gi/15Gi` swap used.
- GPU: NVIDIA GB10, utilization `0%`; `nvidia-smi` did not report memory counters for this device.
- relevant process sample: only the resource-audit `ps | rg` command matched `gatzk`, `run_secure_main`, or `comparison`.
- action: no interfering process killed.
- comparison isolation: no comparison files read for evidence; no comparison process killed.

## 2026-06-28T05:38:08Z Progress Diagnostic Preflight

- task: before progress-enabled `cora/cpu` diagnostic using the official runner path and budget timeout.
- CPU/load: load average `2.06, 2.96, 2.53`.
- memory: `121Gi` total, `6.1Gi` used, `115Gi` available, `1.3Gi/15Gi` swap used.
- GPU: NVIDIA GB10, utilization `0%`, memory utilization `0%`; `nvidia-smi` did not report memory counters for this device.
- relevant process sample: only the resource-audit `ps | rg` command matched `gatzk`, `run_secure_main`, or `comparison`.
- action: no interfering process killed.
- comparison isolation: no comparison files read for evidence; no comparison process killed.

## 2026-06-28T05:48:58Z Persistent Dynamic Cache Canary Preflight

- task: before progress-enabled official-runner `cora/cpu` canary after adding persistent dynamic commitment cache.
- CPU/load: load average `0.57, 0.97, 1.80`.
- memory: `121Gi` total, `6.9Gi` used, `114Gi` available, `1.3Gi/15Gi` swap used.
- GPU: NVIDIA GB10, utilization `0%`, memory utilization `0%`; `nvidia-smi` did not report memory counters for this device.
- dynamic commitment cache: `runs/performance_cache/dynamic_commitment_v1` absent before the canary, so this run starts from an empty persistent cache.
- relevant process sample: only the resource-audit `ps | rg` command matched `gatzk`, `run_secure_main`, or `comparison`.
- action: no interfering process killed.
- comparison isolation: no comparison files read for evidence; no comparison process killed.

## 2026-06-28T05:52:11Z Persistent Dynamic Cache Second Canary Preflight

- task: before second progress-enabled official-runner `cora/cpu` canary with persistent dynamic commitment cache populated by the previous budget-limited run.
- CPU/load: load average `3.33, 2.69, 2.35`.
- memory: `121Gi` total, `6.1Gi` used, `115Gi` available, `1.3Gi/15Gi` swap used.
- GPU: NVIDIA GB10, utilization `0%`, memory utilization `0%`; `nvidia-smi` did not report memory counters for this device.
- dynamic commitment cache: `327` files in `runs/performance_cache/dynamic_commitment_v1`.
- relevant process sample: only the resource-audit `ps | rg` command matched `gatzk`, `run_secure_main`, or `comparison`.
- action: no interfering process killed.
- comparison isolation: no comparison files read for evidence; no comparison process killed.

## 2026-06-28T05:59:22Z Persistent Quotient Cache First Canary Preflight

- task: before progress-enabled official-runner `cora/cpu` canary after adding persistent per-label quotient polynomial/commitment cache.
- CPU/load: load average `0.76, 1.94, 2.29`.
- memory: `121Gi` total, `6.9Gi` used, `114Gi` available, `1.3Gi/15Gi` swap used.
- GPU: NVIDIA GB10, utilization `0%`, memory utilization `0%`; `nvidia-smi` did not report memory counters for this device.
- dynamic commitment cache: `455` files in `runs/performance_cache/dynamic_commitment_v1`.
- quotient cache: absent before this canary; this run starts quotient cache fill from zero files.
- relevant process sample: only the resource-audit `ps | rg` command matched `gatzk`, `run_secure_main`, or `comparison`.
- action: no interfering process killed.
- comparison isolation: no comparison files read for evidence; no comparison process killed.

## 2026-06-28T06:04:26Z Persistent Quotient Cache Second Canary Preflight

- task: before progress-enabled official-runner `cora/cpu` canary with quotient polynomial cache populated by the previous budget-limited run.
- CPU/load: load average `0.41, 2.07, 2.37`.
- memory: `121Gi` total, `6.1Gi` used, `115Gi` available, `1.3Gi/15Gi` swap used.
- GPU: NVIDIA GB10, utilization `0%`, memory utilization `0%`; `nvidia-smi` did not report memory counters for this device.
- dynamic commitment cache: `455` files in `runs/performance_cache/dynamic_commitment_v1`.
- quotient cache: `6` polynomial files and `0` commitment point files in `runs/performance_cache/quotient_v1`.
- relevant process sample: only the resource-audit `ps | rg` command matched `gatzk` or `run_secure_main`.
- action: no interfering process killed.
- comparison isolation: no comparison files read for evidence; no comparison process killed.

## 2026-06-29T05:58:35Z HOTPATH-009 Revert Clean Cora CPU Canary Preflight

- task: before progress-enabled official-runner `cora/cpu` canary after reverting ineffective per-run edge-lift commitment point precomputation.
- build/test gate: CPU build passed; CPU ctest passed 2/2 including `static_secure_scan`; CUDA build passed; CUDA ctest passed 2/2 including `static_secure_scan`.
- persistent-cache scan: no forbidden cache symbols matched in `src`, `include`, `scripts`, or `tools`.
- cold-cache state: `runs/performance_cache` absent; no `performance_cache` directory found under `runs` at max depth 2.
- CPU/load: load average `2.27, 2.21, 1.83`.
- memory: `121Gi` total, `7.8Gi` used, `113Gi` available, `634Mi/15Gi` swap used.
- GPU: NVIDIA GB10, utilization `0%`, memory utilization `0%`; `nvidia-smi` did not report memory counters for this device.
- relevant process sample: only the resource-audit `ps | rg` command matched `gatzk`, `run_secure_main`, `comparison`, `cmake`, `ctest`, or `python`.
- action: no interfering process killed.
- comparison isolation: no comparison files read for evidence; no comparison process killed.

## 2026-06-29T06:04:59Z HOTPATH-010 Clean Cora CPU Canary Preflight

- task: before progress-enabled official-runner `cora/cpu` canary after raising the 8k-16k MSM inner worker threshold.
- build/test gate: CPU build passed; CPU ctest passed 2/2 including `static_secure_scan`; CUDA build passed; CUDA ctest passed 2/2 including `static_secure_scan`.
- persistent-cache scan: no forbidden cache symbols matched in `src`, `include`, `scripts`, or `tools`.
- cold-cache state: `runs/performance_cache` absent; no `performance_cache` directory found under `runs` at max depth 2.
- CPU/load: load average `2.07, 3.32, 2.66`.
- memory: `121Gi` total, `8.1Gi` used, `113Gi` available, `634Mi/15Gi` swap used.
- GPU: NVIDIA GB10, utilization `0%`, memory utilization `0%`; `nvidia-smi` did not report memory counters for this device.
- relevant process sample: only the resource-audit `ps | rg` command matched `gatzk`, `run_secure_main`, `comparison`, `cmake`, `ctest`, or `python`.
- action: no interfering process killed.
- comparison isolation: no comparison files read for evidence; no comparison process killed.

## 2026-06-28T06:14:08Z Persistent FH Dependency Cache Canary Preflight

- task: before progress-enabled official-runner `cora/cpu` canary after adding persistent `t_FH` dependency coset/coefficient cache.
- CPU/load: load average `0.48, 0.81, 1.66`.
- memory: `121Gi` total, `7.0Gi` used, `114Gi` available, `1.3Gi/15Gi` swap used.
- GPU: NVIDIA GB10, utilization `0%`, memory utilization `0%`; `nvidia-smi` did not report memory counters for this device.
- dynamic commitment cache: `455` files in `runs/performance_cache/dynamic_commitment_v1`.
- quotient cache: `6` polynomial files, `0` commitment point files, `0` coset files, and `0` coefficient files in `runs/performance_cache/quotient_v1`.
- relevant process sample: only the resource-audit `ps | rg` command matched `gatzk` or `run_secure_main`.
- action: no interfering process killed.
- comparison isolation: no comparison files read for evidence; no comparison process killed.

## 2026-06-28T06:17:34Z Persistent FH Dependency Cache Second Canary Preflight

- task: before progress-enabled official-runner `cora/cpu` canary after the previous run populated the `P_Q_tbl_feat` coset cache.
- CPU/load: load average `1.13, 1.50, 1.80`.
- memory: `121Gi` total, `6.1Gi` used, `115Gi` available, `1.3Gi/15Gi` swap used.
- GPU: NVIDIA GB10, utilization `0%`, memory utilization `0%`; `nvidia-smi` did not report memory counters for this device.
- quotient cache: `6` polynomial files, `0` commitment point files, `1` coset file, and `0` coefficient files in `runs/performance_cache/quotient_v1`.
- relevant process sample: only the resource-audit `ps | rg` command matched `gatzk` or `run_secure_main`.
- action: no interfering process killed.
- comparison isolation: no comparison files read for evidence; no comparison process killed.

## 2026-06-28T06:20:42Z Persistent FH Dependency Cache Third Canary Preflight

- task: before progress-enabled official-runner `cora/cpu` canary after the previous run populated the `P_Table_feat` coefficient cache and advanced table coset caching.
- CPU/load: load average `1.66, 2.09, 2.02`.
- memory: `121Gi` total, `6.1Gi` used, `115Gi` available, `1.3Gi/15Gi` swap used.
- GPU: NVIDIA GB10, utilization `0%`, memory utilization `0%`; `nvidia-smi` did not report memory counters for this device.
- quotient cache: `6` polynomial files, `0` commitment point files, `2` coset files, and `1` coefficient file in `runs/performance_cache/quotient_v1`.
- relevant process sample: only the resource-audit `ps | rg` command matched `gatzk` or `run_secure_main`.
- action: no interfering process killed.
- comparison isolation: no comparison files read for evidence; no comparison process killed.

## 2026-06-28T06:25:37Z Persistent FH Quotient Polynomial Canary Preflight

- task: before progress-enabled official-runner `cora/cpu` canary after the previous run populated all seven quotient polynomial files, including `t_FH`.
- CPU/load: load average `1.16, 2.20, 2.13`.
- memory: `121Gi` total, `6.9Gi` used, `114Gi` available, `1.3Gi/15Gi` swap used.
- GPU: NVIDIA GB10, utilization `0%`, memory utilization `0%`; `nvidia-smi` did not report memory counters for this device.
- quotient cache: `7` polynomial files, `0` commitment point files, `2` coset files, `1` coefficient file, and `0` quotient-value files in `runs/performance_cache/quotient_v1`.
- relevant process sample: only the resource-audit `ps | rg` command matched `gatzk` or `run_secure_main`.
- action: no interfering process killed.
- comparison isolation: no comparison files read for evidence; no comparison process killed.

## 2026-06-28T06:32:28Z Chunked Quotient Commitment Canary Preflight

- task: before progress-enabled official-runner `cora/cpu` canary after adding chunked persistent MSM for large coefficient quotient commitments.
- CPU/load: load average `1.03, 2.79, 2.68`.
- memory: `121Gi` total, `6.9Gi` used, `114Gi` available, `1.3Gi/15Gi` swap used.
- GPU: NVIDIA GB10, utilization `0%`, memory utilization `0%`; `nvidia-smi` did not report memory counters for this device.
- quotient cache: `7` polynomial files, `0` commitment point files, `2` coset files, `1` coefficient file, and `0` quotient-value files in `runs/performance_cache/quotient_v1`.
- relevant process sample: only the resource-audit `ps | rg` command matched `gatzk` or `run_secure_main`.
- action: no interfering process killed.
- comparison isolation: no comparison files read for evidence; no comparison process killed.

## 2026-06-28T06:35:38Z Chunked Quotient Commitment Second Canary Preflight

- task: before progress-enabled official-runner `cora/cpu` canary after the previous run populated the first three `t_FH` chunk commitment points.
- CPU/load: load average `1.71, 2.75, 2.72`.
- memory: `121Gi` total, `6.1Gi` used, `115Gi` available, `1.3Gi/15Gi` swap used.
- GPU: NVIDIA GB10, utilization `0%`, memory utilization `0%`; `nvidia-smi` did not report memory counters for this device.
- quotient point cache: `3` point files in `runs/performance_cache/quotient_v1`.
- relevant process sample: only the resource-audit `ps | rg` command matched `gatzk` or `run_secure_main`.
- action: no interfering process killed.
- comparison isolation: no comparison files read for evidence; no comparison process killed.

## 2026-06-28T06:38:52Z Chunked Quotient Commitment Third Canary Preflight

- task: before progress-enabled official-runner `cora/cpu` canary after the previous run populated `t_FH` chunks `0-5`.
- CPU/load: load average `1.85, 2.81, 2.78`.
- memory: `121Gi` total, `6.1Gi` used, `115Gi` available, `1.3Gi/15Gi` swap used.
- GPU: NVIDIA GB10, utilization `0%`, memory utilization `0%`; `nvidia-smi` did not report memory counters for this device.
- quotient point cache: `6` point files in `runs/performance_cache/quotient_v1`.
- relevant process sample: only the resource-audit `ps | rg` command matched `gatzk` or `run_secure_main`.
- action: no interfering process killed.
- comparison isolation: no comparison files read for evidence; no comparison process killed.

## 2026-06-28T06:41:58Z Chunked Quotient Commitment Fourth Canary Preflight

- task: before progress-enabled official-runner `cora/cpu` canary after the previous run populated `t_FH` chunks through chunk `8`.
- CPU/load: load average `1.97, 2.90, 2.85`.
- memory: `121Gi` total, `6.1Gi` used, `115Gi` available, `1.3Gi/15Gi` swap used.
- GPU: NVIDIA GB10, utilization `0%`, memory utilization `0%`; `nvidia-smi` did not report memory counters for this device.
- quotient point cache: `9` point files in `runs/performance_cache/quotient_v1`.
- relevant process sample: only the resource-audit `ps | rg` command matched `gatzk` or `run_secure_main`.
- action: no interfering process killed.
- comparison isolation: no comparison files read for evidence; no comparison process killed.

## 2026-06-28T06:45:15Z Chunked Quotient Commitment Fifth Canary Preflight

- task: before progress-enabled official-runner `cora/cpu` canary after the previous run populated `t_FH` chunks through chunk `11`.
- CPU/load: load average `2.28, 3.09, 2.96`.
- memory: `121Gi` total, `6.1Gi` used, `115Gi` available, `1.3Gi/15Gi` swap used.
- GPU: NVIDIA GB10, utilization `0%`, memory utilization `0%`; `nvidia-smi` did not report memory counters for this device.
- quotient point cache: `12` point files in `runs/performance_cache/quotient_v1`.
- relevant process sample: only the resource-audit `ps | rg` command matched `gatzk` or `run_secure_main`.
- action: no interfering process killed.
- comparison isolation: no comparison files read for evidence; no comparison process killed.

## 2026-06-28T10:27:00Z Post-HOTPATH-004-Revert Clean Cora CPU Canary Preflight

- task: before progress-enabled official-runner `cora/cpu` canary after reverting the HOTPATH-004 batch split regression.
- build/test gate: CPU build passed; CPU ctest passed 2/2 including `static_secure_scan`; CUDA build passed; CUDA ctest passed 2/2 including `static_secure_scan`.
- persistent-cache scan: no forbidden cache symbols matched in `src`, `include`, `scripts`, or `tools`.
- cold-cache state: `runs/performance_cache` absent; no `performance_cache` directory found under `runs` at max depth 2.
- CPU/load: load average `0.79, 1.29, 1.26`.
- memory: `121Gi` total, `6.8Gi` used, `114Gi` available, `1.3Gi/15Gi` swap used.
- GPU: NVIDIA GB10, utilization `0%`, memory utilization `0%`; `nvidia-smi` did not report memory counters for this device.
- relevant process sample: only the resource-audit `ps | rg` command matched `gatzk` or `run_secure_main`.
- action: no interfering process killed.
- comparison isolation: no comparison files read for evidence; no comparison process killed.

## 2026-06-28T10:37:16Z HOTPATH-005 Clean Cora CPU Canary Preflight

- task: before progress-enabled official-runner `cora/cpu` canary after adding in-batch deferred linear commitment derivation.
- build/test gate: CPU build passed; CPU ctest passed 2/2 including `static_secure_scan`; CUDA build passed; CUDA ctest passed 2/2 including `static_secure_scan`.
- persistent-cache scan: no forbidden cache symbols matched in `src`, `include`, `scripts`, or `tools`.
- cold-cache state: `runs/performance_cache` absent; no `performance_cache` directory found under `runs` at max depth 2.
- CPU/load: load average `0.73, 1.12, 1.40`.
- memory: `121Gi` total, `6.8Gi` used, `114Gi` available, `1.3Gi/15Gi` swap used.
- GPU: NVIDIA GB10, utilization `0%`, memory utilization `0%`; `nvidia-smi` did not report memory counters for this device.
- relevant process sample: only the resource-audit `ps | rg` command matched `gatzk` or `run_secure_main`.
- action: no interfering process killed.
- comparison isolation: no comparison files read for evidence; no comparison process killed.

## 2026-06-28T10:45:04Z HOTPATH-006 Clean Cora CPU Canary Preflight

- task: before progress-enabled official-runner `cora/cpu` canary after adding in-batch `Delta = M_edge - Z` commitment derivation.
- build/test gate: CPU build passed; CPU ctest passed 2/2 including `static_secure_scan`; CUDA build passed; CUDA ctest passed 2/2 including `static_secure_scan`.
- persistent-cache scan: no forbidden cache symbols matched in `src`, `include`, `scripts`, or `tools`.
- cold-cache state: `runs/performance_cache` absent; no `performance_cache` directory found under `runs` at max depth 2.
- CPU/load: load average `0.85, 1.50, 1.64`.
- memory: `121Gi` total, `6.7Gi` used, `114Gi` available, `1.3Gi/15Gi` swap used.
- GPU: NVIDIA GB10, utilization `0%`, memory utilization `0%`; `nvidia-smi` did not report memory counters for this device.
- relevant process sample: only the resource-audit `ps | rg` command matched `gatzk` or `run_secure_main`.
- action: no interfering process killed.
- comparison isolation: no comparison files read for evidence; no comparison process killed.

## 2026-06-28T10:53:54Z HOTPATH-007 Clean Cora CPU Canary Preflight

- task: before progress-enabled official-runner `cora/cpu` canary after increasing KZG batch outer scheduling for medium dynamic batches.
- build/test gate: CPU build passed; CPU ctest passed 2/2 including `static_secure_scan`; CUDA build passed; CUDA ctest passed 2/2 including `static_secure_scan`.
- persistent-cache scan: no forbidden cache symbols matched in `src`, `include`, `scripts`, or `tools`.
- cold-cache state: `runs/performance_cache` absent; no `performance_cache` directory found under `runs` at max depth 2.
- CPU/load: load average `0.54, 1.24, 1.61`.
- memory: `121Gi` total, `6.7Gi` used, `114Gi` available, `1.3Gi/15Gi` swap used.
- GPU: NVIDIA GB10, utilization `0%`, memory utilization `0%`; `nvidia-smi` did not report memory counters for this device.
- relevant process sample: only the resource-audit `ps | rg` command matched `gatzk` or `run_secure_main`.
- action: no interfering process killed.
- comparison isolation: no comparison files read for evidence; no comparison process killed.

## 2026-06-28T11:02:58Z HOTPATH-008 Clean Cora CPU Canary Preflight

- task: before progress-enabled official-runner `cora/cpu` canary after deriving `w_psq` and output `w` commitment points in-batch.
- build/test gate: CPU build passed; CPU ctest passed 2/2 including `static_secure_scan`; CUDA build passed; CUDA ctest passed 2/2 including `static_secure_scan`.
- persistent-cache scan: no forbidden cache symbols matched in `src`, `include`, `scripts`, or `tools`.
- cold-cache state: `runs/performance_cache` absent; no `performance_cache` directory found under `runs` at max depth 2.
- CPU/load: load average `1.10, 1.47, 1.75`.
- memory: `121Gi` total, `6.9Gi` used, `114Gi` available, `1.3Gi/15Gi` swap used.
- GPU: NVIDIA GB10, utilization `0%`, memory utilization `0%`; `nvidia-smi` did not report memory counters for this device.
- relevant process sample: only the resource-audit `ps | rg` command matched `gatzk` or `run_secure_main`.
- action: no interfering process killed.
- comparison isolation: no comparison files read for evidence; no comparison process killed.

## 2026-06-28T11:20:36Z HOTPATH-009 Clean Cora CPU Canary Preflight

- task: before progress-enabled official-runner `cora/cpu` canary after adding per-run edge-lift commitment point precomputation.
- build/test gate: CPU build passed; CPU ctest passed 2/2 including `static_secure_scan`; CUDA build passed; CUDA ctest passed 2/2 including `static_secure_scan`.
- persistent-cache scan: no forbidden cache symbols matched in `src`, `include`, `scripts`, or `tools`.
- cold-cache state: `runs/performance_cache` absent; no `performance_cache` directory found under `runs` at max depth 2.
- CPU/load: load average `0.93, 0.83, 1.31`.
- memory: `121Gi` total, `6.9Gi` used, `114Gi` available, `2.8Gi/15Gi` swap used.
- GPU: NVIDIA GB10, utilization `0%`, memory utilization `0%`; `nvidia-smi` did not report memory counters for this device.
- relevant process sample: only the resource-audit `ps | rg` command matched `gatzk` or `run_secure_main`.
- action: no interfering process killed.
- comparison isolation: no comparison files read for evidence; no comparison process killed.

## 2026-06-28T11:25:32Z HOTPATH-009 SRS Fallback Clean Cora CPU Canary Preflight

- task: before retrying progress-enabled official-runner `cora/cpu` canary after aligning edge-lift SRS lookup with KZG's same-size Lagrange fallback.
- build/test gate: CPU build passed; CPU ctest passed 2/2 including `static_secure_scan`; CUDA build passed; CUDA ctest passed 2/2 including `static_secure_scan`.
- persistent-cache scan: no forbidden cache symbols matched in `src`, `include`, `scripts`, or `tools`.
- cold-cache state: `runs/performance_cache` absent; no `performance_cache` directory found under `runs` at max depth 2.
- CPU/load: load average `0.91, 1.37, 1.46`.
- memory: `121Gi` total, `7.0Gi` used, `114Gi` available, `2.7Gi/15Gi` swap used.
- GPU: NVIDIA GB10, utilization `0%`, memory utilization `0%`; `nvidia-smi` did not report memory counters for this device.
- relevant process sample: only the resource-audit `ps | rg` command matched `gatzk` or `run_secure_main`.
- action: no interfering process killed.
- comparison isolation: no comparison files read for evidence; no comparison process killed.

## 2026-06-29T06:14:37Z HOTPATH-011 Clean Cora CPU Canary Preflight

- task: before progress-enabled official-runner `cora/cpu` canary after raising `t_edge` coset prewarm worker cap.
- build/test gate: CPU build passed; CPU ctest passed 2/2 including `static_secure_scan`; CUDA build passed; CUDA ctest passed 2/2 including `static_secure_scan`.
- persistent-cache scan: no forbidden cache symbols matched in `src`, `include`, `scripts`, or `tools`.
- cold-cache state: `runs/performance_cache` absent; no `performance_cache` directory found under `runs` at max depth 2.
- CPU/load: load average `1.66, 2.54, 2.86`.
- memory: `121Gi` total, `7.6Gi` used, `114Gi` available, `634Mi/15Gi` swap used.
- GPU: NVIDIA GB10, utilization `0%`, memory utilization `0%`, temperature `42C`; `nvidia-smi` did not report memory counters for this device.
- relevant process sample: no `gatzk` or `run_secure_main` process was present before the canary; top non-system background activity was VS Code/Codex extension host work at roughly one CPU core total, not a material 20-core canary interference.
- action: no interfering process killed.
- comparison isolation: no comparison files read for evidence; no comparison process killed.

## 2026-06-29T06:25:55Z HOTPATH-012/013 Clean Cora CPU Canary Preflight

- task: before progress-enabled official-runner `cora/cpu` canary after capping outer quotient materialization and narrowing `t_edge` dependencies.
- build/test gate: CPU build passed; CPU ctest passed 2/2 including `static_secure_scan`; CUDA build passed; CUDA ctest passed 2/2 including `static_secure_scan`.
- persistent-cache scan: no forbidden cache symbols matched in `src`, `include`, `scripts`, or `tools`.
- cold-cache state: `runs/performance_cache` absent; no `performance_cache` directory found under `runs` at max depth 2.
- CPU/load: load average `2.39, 2.95, 3.26`.
- memory: `121Gi` total, `9.7Gi` used, `111Gi` available, `634Mi/15Gi` swap used.
- GPU: NVIDIA GB10, utilization `31%`, memory utilization `0%`, temperature `50C`; `nvidia-smi pmon` reported PID `3112506` (`mizhou`, `python`, command `./.venv/bin/python -u -B run_token_ablation_matrix.py ...`) at about `30%` SM utilization.
- relevant process sample: no `gatzk` or `run_secure_main` process was present before the canary. PID `3112506` used about one CPU core and GPU compute; because this row is CPU-only and the 20-core CPU load was not materially saturated, it was recorded but not terminated. GPU official rows must re-check this condition before timing.
- action: no process killed.
- comparison isolation: no comparison files read for evidence; no comparison process killed.

## 2026-06-29T06:36:05Z HOTPATH-014 Clean Cora CPU Canary Preflight

- task: before progress-enabled official-runner `cora/cpu` canary after adding closed-form prefix selector coefficients.
- build/test gate: CPU build passed; CPU ctest passed 2/2 including `static_secure_scan`; CUDA build passed; CUDA ctest passed 2/2 including `static_secure_scan`.
- persistent-cache scan: no forbidden cache symbols matched in `src`, `include`, `scripts`, or `tools`.
- cold-cache state: `runs/performance_cache` absent; no `performance_cache` directory found under `runs` at max depth 2.
- CPU/load: load average `2.11, 3.22, 3.61`.
- memory: `121Gi` total, `10Gi` used, `111Gi` available, `634Mi/15Gi` swap used.
- GPU: NVIDIA GB10, utilization `26%`, memory utilization `0%`, temperature `51C`; `nvidia-smi pmon` reported PID `3112506` (`mizhou`, `python`) at about `28%` SM utilization.
- relevant process sample: no `gatzk` or `run_secure_main` process was present before the canary. PID `3112506` used about one CPU core and GPU compute; it was recorded but not terminated because this row is CPU-only and CPU load was not materially saturated.
- action: no process killed.
- comparison isolation: no comparison files read for evidence; no comparison process killed.

## 2026-06-29T06:52:48Z HOTPATH-015 Clean Cora CPU Canary Preflight

- task: before progress-enabled official-runner `cora/cpu` canary after adding same-run physical polynomial coset alias caching in the quotient materializer.
- build/test gate: CPU build passed; CPU ctest passed 2/2 including `static_secure_scan`; CUDA build passed; CUDA ctest passed 2/2 including `static_secure_scan`.
- persistent-cache scan: no forbidden cache symbols matched in `src`, `include`, `scripts`, or `tools`.
- cold-cache state: `runs/performance_cache` absent; no `performance_cache` directory found under `runs` at max depth 2.
- CPU/load: load average `0.59, 1.23, 2.41`.
- memory: `121Gi` total, `8.6Gi` used, `113Gi` available, `634Mi/15Gi` swap used.
- GPU: NVIDIA GB10, utilization `0%`, memory utilization `0%`, temperature `41C`; no compute app reported by `nvidia-smi --query-compute-apps`.
- relevant process sample: no `gatzk` or `run_secure_main` process was present before the canary. Top non-system background activity was another user's VS Code/Codex extension host work at roughly `0.8` CPU cores total on this 20-core host.
- action: no process killed.
- comparison isolation: no comparison files read for evidence; no comparison process killed.
