# Cora full performance capture

- capture_dir: `/home/pzh/project/results/performance_engineering/cora_capture/20260723T033929Z`
- trace_file: `/home/pzh/project/results/performance_engineering/cora_capture/20260723T033929Z/CORA_FULL_PERFORMANCE_TRACE.md`
- start_utc: `2026-07-23T03:39:29Z`
- config: `configs/cora_secure.cfg`
- backend: `cpu` (formal compute backend: `cpu`)
- timeout_seconds: 31536000
- sample_interval_seconds: 5
- requested_threads: 20
- build_jobs: 20
- skip_build: 0
- dry_run: 0

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
affinity=pid 3291156 的当前亲和力列表：0-19
memory=
NVIDIA GB10, 580.142, [N/A]
```
## Resource samples
| timestamp_utc | phase | pid | rss_kib | cpu_percent | threads | mem_total/used/available_mib | gpu_util/used/total_mib |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
## SRS preflight
- srs_type: `trusted_srs`
- srs_id: `cora_formal_srs_v2`
- srs_path: `/home/pzh/project/artifacts/srs/cora_formal_srs_v2`
- compiler_artifact: `/home/pzh/project/results/performance_engineering/cora_capture/20260723T033929Z/srs_preflight/current_protocol_compiler_artifact.txt`
### phase: build
- command: `cmake -S . -B build -DGATZK_ENABLE_CUDA_BACKEND=OFF -DGATZK_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release`
- stdout_log: `/home/pzh/project/results/performance_engineering/cora_capture/20260723T033929Z/logs/build.stdout.log`
- stderr_log: `/home/pzh/project/results/performance_engineering/cora_capture/20260723T033929Z/logs/build.stderr.log`
- live_stdout: `-- GAT-ZKML: using vendored mcl backend from third_party/mcl`
| 2026-07-23T03:39:29Z | build | 3291190 | 6836 | 0.0 | 1 | not_available | 0; [N/A]; [N/A] |
- live_stdout: `-- Configuring done (0.1s)`
- live_stdout: `-- Generating done (0.0s)`
- live_stdout: `-- Build files have been written to: /home/pzh/project/build`
| 2026-07-23T03:39:34Z | build | 3291190 | not_available | not_available | 0 | not_available | 0; [N/A]; [N/A] |
- exit_code: 0
### phase: build_compile
- command: `cmake --build build --target gatzk_run gatzk_srs_setup -j 20`
- stdout_log: `/home/pzh/project/results/performance_engineering/cora_capture/20260723T033929Z/logs/build_compile.stdout.log`
- stderr_log: `/home/pzh/project/results/performance_engineering/cora_capture/20260723T033929Z/logs/build_compile.stderr.log`
- live_stdout: `[  3%] Built target gatzk_mcl_build`
| 2026-07-23T03:39:34Z | build_compile | 3291343 | 2824 | 0.0 | 1 | not_available | 0; [N/A]; [N/A] |
- live_stdout: `[ 92%] Built target gatzkml`
- live_stdout: `[100%] Built target gatzk_run`
- live_stdout: `[  3%] Built target gatzk_mcl_build`
- live_stdout: `[ 92%] Built target gatzkml`
- live_stdout: `[100%] Built target gatzk_srs_setup`
| 2026-07-23T03:39:40Z | build_compile | 3291343 | not_available | not_available | 0 | not_available | 0; [N/A]; [N/A] |
- exit_code: 0
### phase: srs_compile
- command: `/home/pzh/project/build/gatzk_srs_setup compile /home/pzh/project/results/performance_engineering/cora_capture/20260723T033929Z/srs_preflight/current_protocol_compiler_artifact.txt 16777215 /home/pzh/project/configs/srs_cora_max_envelope_v1.cfg C=8 FH=4194304 N=4096 cat=128 d=16 edge=32768 in=2048`
- stdout_log: `/home/pzh/project/results/performance_engineering/cora_capture/20260723T033929Z/logs/srs_compile.stdout.log`
- stderr_log: `/home/pzh/project/results/performance_engineering/cora_capture/20260723T033929Z/logs/srs_compile.stderr.log`
- live_stdout: `wrote protocol compiler artifact "/home/pzh/project/results/performance_engineering/cora_capture/20260723T033929Z/srs_preflight/current_protocol_compiler_artifact.txt"`
| 2026-07-23T03:39:40Z | srs_compile | 3291496 | 5236 | not_available | 0 | not_available | 0; [N/A]; [N/A] |
| 2026-07-23T03:39:45Z | srs_compile | 3291496 | not_available | not_available | 0 | not_available | 0; [N/A]; [N/A] |
- exit_code: 0
- result: existing Cora SRS matches current config, current artifact, domain catalogue, manifest and all material hashes; reused
### SRS validation output
- full_log: `/home/pzh/project/results/performance_engineering/cora_capture/20260723T033929Z/logs/srs_validate.log`
```text
SRS_VALID
```
## Formal Cora execution
- command: `taskset -c 0-19 /home/pzh/project/build/gatzk_run --config configs/cora_secure.cfg --benchmark-mode warm --compute-backend cpu --fast-msm on --parallel-fft on --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on --export-tag cora_capture/20260723T033929Z`
### phase: main
- command: `taskset -c 0-19 /home/pzh/project/build/gatzk_run --config configs/cora_secure.cfg --benchmark-mode warm --compute-backend cpu --fast-msm on --parallel-fft on --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on --export-tag cora_capture/20260723T033929Z`
- stdout_log: `/home/pzh/project/results/performance_engineering/cora_capture/20260723T033929Z/logs/main.stdout.log`
- stderr_log: `/home/pzh/project/results/performance_engineering/cora_capture/20260723T033929Z/logs/main.stderr.log`
- live_stderr: `[progress] light-warmup:context:start`
- live_stderr: `[progress] context:load_static:start`
| 2026-07-23T03:39:46Z | main | 3291631 | 5420 | 100 | 1 | not_available | 0; [N/A]; [N/A] |
- live_stderr: `[progress] context:load_static:done`
- live_stderr: `[progress] context:domains:start`
- live_stderr: `[progress] context:domains:done`
- live_stderr: `[progress] context:srs:start`
- live_stderr: `[progress] kzg:srs:pk_coef:start`
- live_stderr: `[progress] kzg:srs:pk_coef:done`
- live_stderr: `[progress] kzg:srs:lagrange:FH:start`
- live_stderr: `[progress] kzg:srs:lagrange:FH:done`
- live_stderr: `[progress] kzg:srs:lagrange:N:start`
- live_stderr: `[progress] kzg:srs:lagrange:N:done`
- live_stderr: `[progress] kzg:srs:lagrange:cat:start`
- live_stderr: `[progress] kzg:srs:lagrange:cat:done`
- live_stderr: `[progress] kzg:srs:lagrange:d:start`
- live_stderr: `[progress] kzg:srs:lagrange:d:done`
- live_stderr: `[progress] kzg:srs:lagrange:edge:start`
- live_stderr: `[progress] kzg:srs:lagrange:edge:done`
- live_stderr: `[progress] kzg:srs:lagrange:in:start`
- live_stderr: `[progress] kzg:srs:lagrange:in:done`
- live_stderr: `[progress] kzg:srs:vk:start`
- live_stderr: `[progress] kzg:srs:vk:done`
- live_stderr: `[progress] kzg:srs:done`
- live_stderr: `[progress] context:srs:done`
- live_stderr: `[progress] context:public_static:start`
- live_stderr: `[progress] context:public:P_I:start`
- live_stderr: `[progress] context:public:P_I:done`
- live_stderr: `[progress] context:public:P_src:start`
- live_stderr: `[progress] context:public:P_src:done`
- live_stderr: `[progress] context:public:P_dst:start`
- live_stderr: `[progress] context:public:P_dst:done`
- live_stderr: `[progress] context:public:P_Q_new_edge:start`
- live_stderr: `[progress] context:public:P_Q_new_edge:done`
- live_stderr: `[progress] context:public:P_Q_end_edge:start`
- live_stderr: `[progress] context:public:P_Q_end_edge:done`
- live_stderr: `[progress] context:public:P_node_gid:start`
- live_stderr: `[progress] context:public:P_node_gid:done`
- live_stderr: `[progress] context:public:P_edge_gid:start`
- live_stderr: `[progress] context:public:P_edge_gid:done`
- live_stderr: `[progress] context:public:P_Q_graph_new_edge:start`
- live_stderr: `[progress] context:public:P_Q_graph_new_edge:done`
- live_stderr: `[progress] context:public:P_Q_graph_end_edge:start`
- live_stderr: `[progress] context:public:P_Q_graph_end_edge:done`
- live_stderr: `[progress] context:public:P_Q_grp_new_edge:start`
- live_stderr: `[progress] context:public:P_Q_grp_new_edge:done`
- live_stderr: `[progress] context:public:P_Q_grp_end_edge:start`
- live_stderr: `[progress] context:public:P_Q_grp_end_edge:done`
- live_stderr: `[progress] context:public:P_Q_edge_valid:start`
- live_stderr: `[progress] context:public:P_Q_edge_valid:done`
- live_stderr: `[progress] context:public:P_Q_div_h0:start`
- live_stderr: `[progress] context:public:P_Q_div_h0:done`
- live_stderr: `[progress] context:public:P_Q_div_h1:start`
- live_stderr: `[progress] context:public:P_Q_div_h1:done`
- live_stderr: `[progress] context:public:P_Q_div_h2:start`
- live_stderr: `[progress] context:public:P_Q_div_h2:done`
- live_stderr: `[progress] context:public:P_Q_div_h3:start`
- live_stderr: `[progress] context:public:P_Q_div_h3:done`
- live_stderr: `[progress] context:public:P_Q_div_h4:start`
- live_stderr: `[progress] context:public:P_Q_div_h4:done`
- live_stderr: `[progress] context:public:P_Q_div_h5:start`
- live_stderr: `[progress] context:public:P_Q_div_h5:done`
- live_stderr: `[progress] context:public:P_Q_div_h6:start`
- live_stderr: `[progress] context:public:P_Q_div_h6:done`
- live_stderr: `[progress] context:public:P_Q_div_h7:start`
- live_stderr: `[progress] context:public:P_Q_div_h7:done`
- live_stderr: `[progress] context:public:P_Q_div_out0:start`
- live_stderr: `[progress] context:public:P_Q_div_out0:done`
- live_stderr: `[progress] context:public:P_Q_div_avg:start`
- live_stderr: `[progress] context:public:P_Q_div_avg:done`
- live_stderr: `[progress] context:public:P_Q_N:start`
- live_stderr: `[progress] context:public:P_Q_N:done`
- live_stderr: `[progress] context:public:P_AggBind_m_src:start`
- live_stderr: `[progress] context:public:P_AggBind_m_src:done`
- live_stderr: `[progress] context:public:P_AggBind_m_dst:start`
- live_stderr: `[progress] context:public:P_AggBind_m_dst:done`
- live_stderr: `[progress] context:public:P_AggBind_index_N:start`
- live_stderr: `[progress] context:public:P_AggBind_index_N:done`
- live_stderr: `[progress] context:public:P_Q_proj_valid:start`
- live_stderr: `[progress] context:public:P_Q_proj_valid:done`
- live_stderr: `[progress] context:public:P_Q_d_valid:start`
- live_stderr: `[progress] context:public:P_Q_d_valid:done`
- live_stderr: `[progress] context:public:P_Q_cat_valid:start`
- live_stderr: `[progress] context:public:P_Q_cat_valid:done`
- live_stderr: `[progress] context:public:P_Q_C_valid:start`
- live_stderr: `[progress] context:public:P_Q_C_valid:done`
- live_stderr: `[progress] context:public:P_BiasIndex_C:start`
- live_stderr: `[progress] context:public:P_BiasIndex_C:done`
- live_stderr: `[progress] context:public:P_BiasMultiplicity_C:start`
- live_stderr: `[progress] context:public:P_BiasMultiplicity_C:done`
- live_stderr: `[progress] context:public:P_BiasIndex_edge:start`
- live_stderr: `[progress] context:public:P_BiasIndex_edge:done`
- live_stderr: `[progress] context:public:P_out0_BiasBasis_Q_active_0:start`
- live_stderr: `[progress] context:public:P_out0_BiasBasis_Q_active_0:done`
- live_stderr: `[progress] context:public:P_out0_BiasBasis_Q_semantic_0:start`
- live_stderr: `[progress] context:public:P_out0_BiasBasis_Q_semantic_0:done`
- live_stderr: `[progress] context:public:P_out0_BiasBasis_Q_gap_0:start`
- live_stderr: `[progress] context:public:P_out0_BiasBasis_Q_gap_0:done`
- live_stderr: `[progress] context:public:P_out0_BiasBasis_Q_tail_0:start`
- live_stderr: `[progress] context:public:P_out0_BiasBasis_Q_tail_0:done`
- live_stderr: `[progress] context:public:P_Q_tbl_feat:start`
- live_stderr: `[progress] context:public:P_Q_tbl_feat:done`
- live_stderr: `[progress] context:public:P_Row_feat_tbl:start`
- live_stderr: `[progress] context:public:P_Row_feat_tbl:done`
- live_stderr: `[progress] context:public:P_Col_feat_tbl:start`
- live_stderr: `[progress] context:public:P_Col_feat_tbl:done`
- live_stderr: `[progress] context:public_alias:P_Q_qry_feat->P_Q_tbl_feat:start`
- live_stderr: `[progress] context:public_alias:P_Q_qry_feat->P_Q_tbl_feat:done`
- live_stderr: `[progress] context:public_alias:P_Row_feat_qry->P_Row_feat_tbl:start`
- live_stderr: `[progress] context:public_alias:P_Row_feat_qry->P_Row_feat_tbl:done`
- live_stderr: `[progress] context:public_alias:P_Col_feat_qry->P_Col_feat_tbl:start`
- live_stderr: `[progress] context:public_alias:P_Col_feat_qry->P_Col_feat_tbl:done`
- live_stderr: `[progress] context:public_alias:P_I_feat_qry->P_Row_feat_tbl:start`
- live_stderr: `[progress] context:public_alias:P_I_feat_qry->P_Row_feat_tbl:done`
- live_stderr: `[progress] context:public:P_Q_qry_src:start`
- live_stderr: `[progress] context:public:P_Q_qry_src:done`
- live_stderr: `[progress] context:public:P_Q_qry_dst:start`
- live_stderr: `[progress] context:public:P_Q_qry_dst:done`
- live_stderr: `[progress] context:public:P_Q_tbl_L:start`
- live_stderr: `[progress] context:public:P_Q_tbl_L:done`
- live_stderr: `[progress] context:public:P_Q_qry_L:start`
- live_stderr: `[progress] context:public:P_Q_qry_L:done`
- live_stderr: `[progress] context:public:P_Q_tbl_R:start`
- live_stderr: `[progress] context:public:P_Q_tbl_R:done`
- live_stderr: `[progress] context:public:P_Q_qry_R:start`
- live_stderr: `[progress] context:public:P_Q_qry_R:done`
- live_stderr: `[progress] context:public:P_Q_tbl_exp:start`
- live_stderr: `[progress] context:public:P_Q_tbl_exp:done`
- live_stderr: `[progress] context:public:P_Q_qry_exp:start`
- live_stderr: `[progress] context:public:P_Q_qry_exp:done`
- live_stderr: `[progress] context:public:P_Q_tbl_ELU:start`
- live_stderr: `[progress] context:public:P_Q_tbl_ELU:done`
- live_stderr: `[progress] context:public:P_Q_qry_ELU:start`
- live_stderr: `[progress] context:public:P_Q_qry_ELU:done`
- live_stderr: `[progress] context:static:V_T_H:start`
- live_stderr: `[progress] context:static:V_T_H:done`
- live_stderr: `[progress] context:public:P_T_H:start`
- live_stderr: `[progress] context:public:P_T_H:done`
- live_stderr: `[progress] context:public_fh_coeff:P_T_H:start`
| 2026-07-23T03:39:51Z | main | 3291631 | 3297572 | 366 | 20 | not_available | 0; [N/A]; [N/A] |
- live_stderr: `[progress] context:public_fh_coeff:P_T_H:done`
- live_stderr: `[progress] context:public_fh_coeff:P_Q_tbl_feat:start`
- live_stderr: `[progress] context:public_fh_coeff:P_Q_tbl_feat:done`
- live_stderr: `[progress] context:public_fh_coeff:P_Row_feat_tbl:start`
- live_stderr: `[progress] context:public_fh_coeff:P_Row_feat_tbl:done`
- live_stderr: `[progress] context:public_fh_coeff:P_Col_feat_tbl:start`
| 2026-07-23T03:39:56Z | main | 3291631 | 3727960 | 233 | 20 | not_available | 0; [N/A]; [N/A] |
- live_stderr: `[progress] context:public_fh_coeff:P_Col_feat_tbl:done`
- live_stderr: `[progress] context:public_fh_coset:P_Q_tbl_feat:boolean_quotient:start`
| 2026-07-23T03:40:01Z | main | 3291631 | 5038708 | 205 | 21 | not_available | 0; [N/A]; [N/A] |
- live_stderr: `[progress] context:public_fh_coset:P_Q_tbl_feat:boolean_quotient:done`
- live_stderr: `[progress] context:public:P_T_L_x:start`
- live_stderr: `[progress] context:public:P_T_L_x:done`
- live_stderr: `[progress] context:public:P_T_L_y:start`
- live_stderr: `[progress] context:public:P_T_L_y:done`
- live_stderr: `[progress] context:static:V_T_L_x:start`
- live_stderr: `[progress] context:static:V_T_L_x:done`
- live_stderr: `[progress] context:static:V_T_L_y:start`
- live_stderr: `[progress] context:static:V_T_L_y:done`
- live_stderr: `[progress] context:public:P_T_ELU_x:start`
- live_stderr: `[progress] context:public:P_T_ELU_x:done`
- live_stderr: `[progress] context:public:P_T_ELU_y:start`
- live_stderr: `[progress] context:public:P_T_ELU_y:done`
- live_stderr: `[progress] context:static:V_T_ELU_x:start`
- live_stderr: `[progress] context:static:V_T_ELU_x:done`
- live_stderr: `[progress] context:static:V_T_ELU_y:start`
- live_stderr: `[progress] context:static:V_T_ELU_y:done`
- live_stderr: `[progress] context:public:P_T_exp_x:start`
- live_stderr: `[progress] context:public:P_T_exp_x:done`
- live_stderr: `[progress] context:public:P_T_exp_y:start`
- live_stderr: `[progress] context:public:P_T_exp_y:done`
- live_stderr: `[progress] context:static:V_T_exp_x:start`
- live_stderr: `[progress] context:static:V_T_exp_x:done`
- live_stderr: `[progress] context:static:V_T_exp_y:start`
- live_stderr: `[progress] context:static:V_T_exp_y:done`
- live_stderr: `[progress] context:public:P_T_range:start`
- live_stderr: `[progress] context:public:P_T_range:done`
- live_stderr: `[progress] context:static:V_T_range:start`
- live_stderr: `[progress] context:static:V_T_range:done`
- live_stderr: `[progress] context:static:V_h0_W_col0:start`
| 2026-07-23T03:40:06Z | main | 3291631 | not_available | not_available | 0 | not_available | 0; [N/A]; [N/A] |
- exit_code: 1
## Final status
- status: failed
- exit_code: 1
- failure_stage: main
- failure_or_exception: phase exited with code 1
## Final metrics
not_available (run_manifest.json was not produced)
