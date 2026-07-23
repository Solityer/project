# Cora full performance capture

- capture_dir: `/home/pzh/project/results/performance_engineering/cora_capture/20260722T081132Z_2295618`
- trace_file: `/home/pzh/project/results/performance_engineering/cora_capture/20260722T081132Z_2295618/CORA_FULL_PERFORMANCE_TRACE.md`
- start_utc: `2026-07-22T08:11:32Z`
- config: `configs/cora_secure.cfg`
- backend: `cpu` (formal compute backend: `cpu`)
- timeout_seconds: 1
- sample_interval_seconds: 0.2
- requested_threads: 1
- build_jobs: 1
- skip_build: 1
- dry_run: 1

## Execution contract

The formal entry is the current `gatzk_run` warm path. It must complete the
full Cora graph, context, trace, commitments, quotient, openings, canonical
serialization, strict parse, verifier, and artifact export. No fixture,
subgraph, local benchmark, or old SRS fallback is permitted.

## Environment

```text
Linux spark-f1b7 6.17.0-1014-nvidia #14-Ubuntu SMP PREEMPT_DYNAMIC Tue Mar 17 19:01:40 UTC 2026 aarch64 aarch64 aarch64 GNU/Linux
hostname=spark-f1b7
online_cpus=20
affinity=pid 2295618's current affinity list: 0-19
memory=Mem:           121Gi       9.9Gi        79Gi       8.9Mi        33Gi       111Gi
NVIDIA GB10, 580.142, [N/A]
```
## Resource samples
| timestamp_utc | phase | pid | rss_kib | cpu_percent | threads | mem_total/used/available_mib | gpu_util/used/total_mib |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
## SRS preflight
- srs_type: `trusted_srs`
- srs_id: `cora_formal_srs_v2`
- srs_path: `/home/pzh/project/artifacts/srs/cora_formal_srs_v2`
- compiler_artifact: `/home/pzh/project/results/performance_engineering/cora_capture/20260722T081132Z_2295618/srs_preflight/current_protocol_compiler_artifact.txt`
- dry_run: setup-only and Cora execution intentionally skipped
- command: `/home/pzh/project/build/gatzk_srs_setup compile /home/pzh/project/results/performance_engineering/cora_capture/20260722T081132Z_2295618/srs_preflight/current_protocol_compiler_artifact.txt 16777215 /home/pzh/project/configs/srs_cora_max_envelope_v1.cfg C=8 FH=4194304 N=4096 cat=128 d=16 edge=32768 in=2048`
- command: `/home/pzh/project/build/gatzk_srs_setup ceremony /home/pzh/project/results/performance_engineering/cora_capture/20260722T081132Z_2295618/srs_preflight/cora_formal_srs_v2.staging /home/pzh/project/results/performance_engineering/cora_capture/20260722T081132Z_2295618/srs_preflight/current_protocol_compiler_artifact.txt cora_formal_srs_v2`
- command: `/home/pzh/project/build/gatzk_run --config configs/cora_secure.cfg --benchmark-mode warm --compute-backend cpu --fast-msm on --parallel-fft on --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on --export-tag cora_capture/20260722T081132Z`
## Final status
- status: dry_run
- exit_code: 0
- failure_stage: not_available
- failure_or_exception: not_available
## Final metrics
not_available (run_manifest.json was not produced)
- raw resource CSV: `/home/pzh/project/results/performance_engineering/cora_capture/20260722T081132Z_2295618/resource_samples.csv`
- latest: `/home/pzh/project/results/performance_engineering/cora_capture/latest`
