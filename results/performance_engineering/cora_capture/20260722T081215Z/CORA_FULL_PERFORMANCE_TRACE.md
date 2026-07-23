# Cora full performance capture

- capture_dir: `/home/pzh/project/results/performance_engineering/cora_capture/20260722T081215Z`
- trace_file: `/home/pzh/project/results/performance_engineering/cora_capture/20260722T081215Z/CORA_FULL_PERFORMANCE_TRACE.md`
- start_utc: `2026-07-22T08:12:15Z`
- config: `configs/cora_secure.cfg`
- backend: `gpu` (formal compute backend: `cuda_hotspots`)
- timeout_seconds: 1
- sample_interval_seconds: 0.2
- requested_threads: 2
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
affinity=pid 2296058's current affinity list: 0-19
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
- compiler_artifact: `/home/pzh/project/results/performance_engineering/cora_capture/20260722T081215Z/srs_preflight/current_protocol_compiler_artifact.txt`
- dry_run: setup-only and Cora execution intentionally skipped
- command: `/home/pzh/project/build_cuda/gatzk_srs_setup compile /home/pzh/project/results/performance_engineering/cora_capture/20260722T081215Z/srs_preflight/current_protocol_compiler_artifact.txt 16777215 /home/pzh/project/configs/srs_cora_max_envelope_v1.cfg C=8 FH=4194304 N=4096 cat=128 d=16 edge=32768 in=2048`
- command: `/home/pzh/project/build_cuda/gatzk_srs_setup ceremony /home/pzh/project/results/performance_engineering/cora_capture/20260722T081215Z/srs_preflight/cora_formal_srs_v2.staging /home/pzh/project/results/performance_engineering/cora_capture/20260722T081215Z/srs_preflight/current_protocol_compiler_artifact.txt cora_formal_srs_v2`
- command: `taskset -c 0-1 /home/pzh/project/build_cuda/gatzk_run --config configs/cora_secure.cfg --benchmark-mode warm --compute-backend cuda_hotspots --fast-msm on --parallel-fft on --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on --export-tag cora_capture/20260722T081215Z`
## Final status
- status: dry_run
- exit_code: 0
- failure_stage: not_available
- failure_or_exception: not_available
## Final metrics
not_available (run_manifest.json was not produced)
- raw resource CSV: `/home/pzh/project/results/performance_engineering/cora_capture/20260722T081215Z/resource_samples.csv`
- previous_latest_preserved: `/home/pzh/project/results/performance_engineering/cora_capture/latest.previous_20260722T081215Z`
- latest: `/home/pzh/project/results/performance_engineering/cora_capture/latest`
