# EZKL GPU FFT raw-Montgomery bridge closeout — 2026-10-23

## Outcome

This round targets the Prove-time CPU representation conversion around GPU FFTs, not the earlier fused-H evaluator. In `icicle.rs::fft_on_device`, the exact pinned BN254 `Fr` path now copies the active range's raw Montgomery bytes directly to/from ICICLE, while retaining the reduced-value check after device-to-host transfer. Canonical twiddle generation and the linear NTT representation remain unchanged; generic fields and MSM keep their existing paths. Resource counters, budget, and record schemas are unchanged.

The existing GPU forward/inverse/high-water check passed on the changed path, and the final official GPU build and identity checks passed. No separate NTT latency or end-to-end Prove timing was measured after this bridge change. Therefore, this confirms correctness and dispatch, not a wall-clock speedup.

## Source-derived conversion work

The producer runs one initial transform for each advice polynomial and one instance polynomial. For one instance, the removed per-element CPU conversions are estimated as `2 × (advice_columns + 1) × 2^k`, where the factor two counts `to_repr` and `from_repr`.

- Cora: `native_keygen_receipt.json` reports 109 advice columns and `settings.json` reports `run_args.logrows=22`; the estimate is `2 × (109 + 1) × 2^22 = 922,746,880` representation conversions.
- CiteSeer: the corresponding receipt reports 73 advice columns and `run_args.logrows=24`; the estimate is `2 × (73 + 1) × 2^24 = 2,483,027,968` conversions.

The producer loop is in `vendor/halo2/halo2_proofs/src/poly/prover.rs` around lines 1726–1730 and 1744–1759. The inputs are `runs/formal/{cora,citeseer}/native_keygen_receipt.json` (`native_census.preflight.advice_columns`) and each cell's `settings.json` (`run_args.logrows`). These counts are a one-instance initial-transform work estimate, not elapsed time, measured invocation counts, or a complete proof cost. They apply only when GPU dispatch is active and the exact BN254 representation guard succeeds.

## The one runtime check

`formal_gpu_ntt_workspace_reuses_high_water` in `tests/formal_gpu_ntt.rs` ran once with `formal-gpu`, the official release target, `ENABLE_ICICLE_GPU=1`, and the selected GB10. It directly initializes the device, exercises GPU forward/inverse/roundtrip over sizes `2^18 → 2^16 → 2^18`, and compares against the CPU oracle.

The passing test asserts at least six GPU-dispatched NTT calls, at least six CPU-oracle calls, and at least 1,179,648 GPU NTT elements. It also checks the retained `2^18` high-water capacities: 16,777,216 bytes for the two device buffers, 8,390,880 bytes for the domain cache, and a 25,176,288-byte device upper bound. Result: `running 1 test`, `1 passed`, test body 0.52 s. Cargo's 5:43.63 wall time was the first release-target compile plus the check; it is not an NTT or Prove timing.

Exact command:

```bash
cargo test --offline --locked --release --no-default-features \
  --target-dir target/gpu --features formal-gpu \
  --test formal_gpu_ntt -- \
  --exact formal_gpu_ntt_workspace_reuses_high_water --nocapture
```

The persistent log is `rust_ntt_rawbridge_exact.log`.

## Prior fused-H evidence, not evidence for this change

The earlier H-path measurements belong to the source before this raw-FFT bridge and were not rerun against the current source. The k=13 proof had 400 fused-H groups, zero H fallback, a full `coefficient_ifft_and_h_evaluation` phase of 5,185.593 ms, total Prove of 7,631 ms, and independent Verify of 41 ms. Its H adapter-call sum was 669.948 ms, not the full phase. A prior A+B tiny reference was 4.487 s for its H phase and 6.014 s total Prove; these runs were uncontrolled, and the arithmetic differences (+15.6% for the phase, +26.8% total) are not a causal comparison. The cold standalone H fixture also measured 13.083 ms GPU versus 5.577 ms CPU for 65,573 rows. None of these numbers establishes that the present raw-FFT conversion change improves Prove time.

No formal dataset run was started. Cora and the other four datasets do not have current-source Setup → Prove → independent Verify results here; this check does not establish five-dataset completion or performance.

## Final build identity

The final official command was the stored `scripts/build_ezkl.sh --gpu` invocation in `prebuild_identity_attempt4.json`, using aarch64 `nightly-2025-12-01`, CUDA 13.0.88, sm_121, and NVIDIA GB10 UUID `GPU-3cde57ba-ae43-4452-bd59-7b156b36db71`. It completed the native CLI and source-identity checks successfully; no tests were run afterward.

- Source tree SHA-256: `4fa34ed401c2b1cbf071f6cb6c396f9b65153d9e0cd7c45ce6eeb9b8c894d053` (matches receipt).
- `target/gpu/release/ezkl` SHA-256: `c5c76026cd891a61d3b9ebbf94da99388d382f533cb98a5339c3c2e48af3e4b6` (matches receipt).
- `artifacts/build/gpu/build_receipt.json` SHA-256: `663d2404f898f69a0617dcb439b05034cc133d478d097db1fc54f546c136bf1e`.
- Native ICICLE manifest SHA-256: `663b49a83c7cec4c215b1b0cc663ef2771b4a48b0de44dfa3dfa7f6a82357958` (matches receipt).

The fresh formal launcher remains `python3 scripts/run_formal_matrix.py --execute-formal`; it was not executed in this task.

## 五数据集正式运行前检查（未运行正式实验）

本节记录 2026-10-23 的只读生产资源检查。它检查 secure config、canonical resource、table/ledger/checkpoint 与官方 SRS 之间的实际 schema、路径、内容摘要和已有 metadata 绑定；不构成资源 Admission 或任何正式实验阶段的成功证明。

| Dataset | Secure config 与 cache metadata | Resource、checkpoint、table/ledger | NPY 检查 | SRS 绑定 |
|---|---|---|---|---|
| Cora | `validate_config` 通过；canonical text cache 的 manifest 内容摘要和 dataset metadata 匹配 | checkpoint archive 通过；table registry、ledger ID/digest 与 checkpoint/config 一致 | 不适用，输入为 text | 五 entry 共享路径、ID、digest 绑定通过 |
| CiteSeer | `validate_config` 通过；canonical text cache 的 manifest 内容摘要和 dataset metadata 匹配 | checkpoint archive 通过；table registry、ledger ID/digest 与 checkpoint/config 一致 | 不适用，输入为 text | 五 entry 共享路径、ID、digest 绑定通过 |
| PubMed | `validate_config` 通过；canonical text cache 的 manifest 内容摘要和 dataset metadata 匹配 | checkpoint archive 通过；table registry、ledger ID/digest 与 checkpoint/config 一致 | 不适用，输入为 text | 五 entry 共享路径、ID、digest 绑定通过 |
| PPI | `validate_config` 通过；cache、mask/label 文件摘要和 dataset metadata 匹配 | checkpoint archive 通过；table registry、ledger ID/digest 与 checkpoint/config 一致 | `features.npy` shape/dtype/order contract 通过；其余 NPY 项完成 manifest size/SHA-256 校验 | 五 entry 共享路径、ID、digest 绑定通过 |
| OGBN-Arxiv | `validate_config` 通过；canonical cache 的 manifest 内容摘要和 dataset metadata 匹配 | checkpoint archive 通过；table registry、ledger ID/digest 与 checkpoint/config 一致 | `features.npy` shape/dtype/order contract 通过 | 五 entry 共享路径、ID、digest 绑定通过 |

实际执行的只读 validator 链为：对 `DATASETS` 的每个 `resources/canonical/configs/{dataset}_secure.cfg` 调用 `formal_common.validate_config(config, dataset)`；随后调用 `validate_resources.require_complete_resource_set(ROOT, manifest)`、`validate_resources.validate_manifest(ROOT, manifest, False)`、`validate_resources.validate_generated_resource_links(ROOT)`，最后调用 `formal_common.validate_native_srs_manifest(config_values=configs)`。资源 validator 对 69/69 个条目逐一验证 exact schema/path set、大小、内容 SHA-256、provenance、NPY feature contract 和 checkpoint archive；表/账本链接 validator 再将实际 table binary registry 和 ledger digest 与五个 checkpoint/config 绑定。没有调用会写入认证 marker 的 `validate_resources.py --require-complete` CLI。

resource manifest 包含 69 entries、合计 213,654,956 bytes，manifest SHA-256 为 `fdf1c7729efed73266beff9111765549120ea0b32e99132ecf58e212bb317ba5`，与 GPU build receipt 的 `resource_manifest_sha256` 相同。native SRS manifest SHA-256 为 `1da0610a39f65da4778eed0d67a082c909277b4e04d826f1d6447705a9820fbc`。共享 SRS `resources/native_srs/kzg26.srs` 大小为 8,589,934,852 bytes；只读 validator 对完整文件进行流式 SHA-256，结果 `b198a51d48b88181508d8e4ea9dea39db285e4585663b29b7e4ded0c22a94875`，与当前 `src/srs_sha.rs` 的嵌入式 logrows-26 摘要匹配。source URL、k=26、SRS 类型、native-get-srs provenance、五个 manifest entries，以及五份 config 的 `srs_id=official_ezkl_native_shared_kzg26_srs`、路径和 trusted digest 均通过生产校验。

与以上实际纯 validator 执行分开，source-only 审阅确认：五个 dataset 的 base k=24/25 与 extended-to-coefficient k=27/28 对应的 k=26 SRS 和 `Fr::S28` 范围关系符合当前生产实现；失败记录后继续后续数据集的原行为仍在共享路径中；独立 Verify 和四项结果指标汇总链也经过源码核对。这些是静态源码结论，不是动态准入或证明结果。

本检查没有运行 formal launcher，也没有触发 prepare、export、calibrate、native Admission、SRS provision、Setup、witness generation、Prove、Verify 或结果 collector。因此它确认了当前五套输入和官方 SRS 绑定完整，不声称任何 dataset 已完成正式运行。

## Cora late path rejection repair — latest build identity (2026-10-23)

The user-owned session `9962321ff1564503a6d63ae089e55c68` was stopped through the official session controller. Its controller records `EXITED`, exit code `-2`, control PID `3169582`, and runner PID `3169584`; both PIDs were absent at closeout, and no process carrying this session ID remained. The existing run directory and its artifacts were preserved. Cora's `prove.log` records `native_create_proof_total elapsed_ms=3927306.061`, followed by a rejection of the absolute `statement.json` path as non-baseline-local. The Cora `proof.json` is absent: the proof computation completed, but finalization rejected the path before saving the proof. This is not a successful Prove or Verify result.

The shared formal consumer in `src/execute.rs` now resolves statement, public output, artifact manifest, proof output, and proving-key parent against the canonical baseline-local cell root. It passes relative public-reference paths through the existing regular non-symlink guard, then reads and hashes canonical paths. Formal Prove checks the path and PK-parent contract before loading the large PK and computes its full public-instance anchor before proof creation; after proof creation it recomputes and compares that anchor before saving. The existing basename, traversal, containment, symlink, manifest-digest, and TOCTOU checks remain active. See `formal_proof_reference_paths` and `validate_formal_prove_paths` at `src/execute.rs:2723-2797`, the early prove checks at `:7054-7064` and reference anchor at `:7103-7115`, and the post-proof rebind before save at `:7243-7259`.

One production-helper path check used the retained Cora statement, public output, and artifact manifest plus the baseline `Cargo.toml` and existing Cora/CiteSeer cell directories. It exercised relative/absolute equivalence and rejected traversal, outside-baseline run roots, a cross-cell proof parent, a cross-cell PK parent, and a missing formal proof destination. It did not load a PK or create, prove, or verify a proof. Exact command:

```bash
cargo test --offline --locked --release --no-default-features \
  --target-dir target/gpu --features formal-gpu --lib \
  execute::formal_proof_path_contract_tests::retained_cora_paths_are_baseline_bound_before_proving \
  -- --exact --nocapture
```

Only `formal_proof_path_contract_exact_attempt6.log` is the successful check: it reports `running 1 test`, `1 passed`, `0 failed`, `0 ignored`, with a 0.00 s test body. Earlier attempt logs are retained but are not passing evidence. The official GPU build then completed with exit code 0 and passed its native CLI and source-identity stages; its log is `final_official_build_path_contract.log`. No test or formal stage ran after that build.

This latest build supersedes the earlier build hashes above:

- Source tree SHA-256 from the verified receipt: `3674e441c32f5d78ffd24a931f8104e99e46536e00d7b877e7e0a0f777a7d980`.
- `target/gpu/release/ezkl` SHA-256: `07b7eca4d5a58c6bd1bedda5850ce0c805e3121050b6565cad4279a85b9130d0`; it matches `build_receipt.json`.
- `artifacts/build/gpu/build_receipt.json` SHA-256: `1508cdfa511f14c60d4ba284caf5ba1982aec0175782d74aa2a8249371b1e93b`.
- Native ICICLE manifest SHA-256: `663b49a83c7cec4c215b1b0cc663ef2771b4a48b0de44dfa3dfa7f6a82357958`; it matches the receipt.

The fresh official launcher to hand off is `python3 scripts/run_formal_matrix.py --execute-formal`. It was not run here; the old stopped session was not resumed, and no Setup, Prove, or Verify was started by this repair.
