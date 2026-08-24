# ezkl formal GAT-ZKML baseline

本目录在 ezkl 自身的 Halo2/BN254/KZG/SHPLONK 证明系统内实现
`GAT-ZKML-ezkl.md` 的完整固定形状 GAT relation。正式路线不读取预计算 logits，
不使用 host-side equality 代替约束，也没有 tiny/partial/sampled 或 CPU/GPU fallback。

## 证明 relation

公开 statement 是
`(M_pub,T_H,I,src,dst,node_ptr,edge_ptr,Y,H(W),CircuitRef)`；规范整数模型参数和完整
GAT trace 是 private witness。编译计划固定层数、所有 hidden/output heads、张量形状、
CSR topology wiring、量化链、冻结的 canonical exact lookup rows、整数 bound ledger、security
profile 和注册模型 Poseidon hash。

`src/native_gat.rs` 的 Halo2 circuit 约束 projection、edge route、LReLU、stable group
max、exact ExpMap、非零 `Sum` 与 field inverse、`alpha`、field/integer 双聚合、hidden
ELU/concat、output bias、head average，以及协议规定的 recovery/alignment/average
`SignedRoundDiv(nearest_ties_away)`。模型哈希由同一 circuit 的 ezkl Poseidon chip 对
canonical private model cells 计算。

Verifier 从单独的 `public_statement.json` 重建 instances，并检查 settings、compiled
circuit、VK、table digest、integer ledger、model registration 与 `CircuitRef` 的绑定；
proof 内序列化的 instances 只允许与重建值完全一致，不能充当 statement 来源。
settings、compiled circuit、模型注册 bounds、PK/VK 和 witness trace 均使用各自独立的
domain-separated canonical identity；CSR CLI 参数必须指向已登记的 v2 manifest，且该
manifest 的维度、最大入度与 `csr_segment_layout.bin` 必须逐项一致。

## Canonical adapter artifacts

正式 C++ adapter 只编译 integer/formal 路线并输出：

- `node_features.i64.bin`、`model_parameters.i64.txt`
- `csr_segment_layout.bin`（含 absolute node identities）与 `graph_attention_spec.json`
- `formal_quant_config.json`、`formal_table_spec.json`
- `formal_main_tables.i64.bin`（只含协议主表 LReLU/ELU/ExpMap；ID 同时绑定 generator/version/scales/domain/rounding/clipping/order 元数据与 rows）
- `backend_aux_table_spec.json`、`backend_aux_tables.i64.bin`（独立绑定 Halo2 range limb 与 MaxReLU，绝不进入 `static_tables_id`）
- `integer_bound_ledger.json`、`model_registration.json`
- `security_profile.json`（固定 Halo2 commit `01c88842679b...`、BN254-Fr、KZG/SHPLONK、Blake2b/Challenge255，以及 exact ceremony ID/source SRS SHA-256/original max_k）
- `canonical_forward.json`（仅供 adapter 语义检查；prover 会从 canonical inputs 独立生成完整 trace）

旧 f64、approximation-band、attention-bias、preflight candidate 和 final-logits binding
代码不在正式 adapter target 中。

正式 circuit 使用 32 个相互独立的 fixed-program lanes；lane 数进入 compiled identity，
只压缩 Halo2 row 布局，不改变 canonical variable、copy constraint 或 lookup relation。
模型 family bounds 在 compile/setup 前从 canonical 参数重新计算并与 bounds digest、`H(W)`、
architecture 和 quant profile 绑定；inference circuit 不重复逐参数 range proof。

## Build

构建脚本不运行测试、不下载数据或模型。CPU 是默认 target；GPU 使用独立 target 目录
和 ezkl/Halo2 的 `gpu-accelerated`/ICICLE feature：

```bash
scripts/build.sh
EZKL_BUILD_GPU=1 scripts/build.sh
```

CPU binary 为 `target/release/ezkl`；CUDA binary 为
`target-gpu/release/ezkl`。CUDA 入口必须成功加载并选择 CUDA device，否则立即失败；
不会回退到 CPU。CPU runner 也拒绝 GPU-enabled binary。

## 单数据集正式运行

默认外部数据与 checkpoint 路径登记在 `configs/full_datasets.json`，也可通过 CLI
显式覆盖。以下命令依次执行 adapter、settings、compile、setup、full witness、proof 和
independent verify：

首先由可信 ceremony 管理方提供 descriptor（`descriptor_kind=ezkl_trusted_ceremony_srs_v1`、
`ceremony_id`、`source_srs_sha256`、真实的 `source_srs_max_k`）。runner 会在 adapter 之前
校验 SRS bytes 并物化 resolved profile；SRS 的真实 ceremony 身份、摘要和原始 `k` 会贯穿
setup/prove/verify，且必须覆盖编译电路的 `logrows`。

```bash
python3 scripts/run_gat_ezkl_baseline.py \
  --dataset cora_full --device cpu --srs-path /trusted/ceremony.srs \
  --ceremony-descriptor /trusted/ceremony.descriptor.json --clean --run

python3 scripts/run_gat_ezkl_baseline.py \
  --dataset cora_full --device cuda --srs-path /trusted/ceremony.srs \
  --ceremony-descriptor /trusted/ceremony.descriptor.json --clean --run
```

仅生成 canonical adapter inputs（不产生 proof）：

```bash
python3 scripts/run_gat_ezkl_baseline.py --dataset cora_full --device cpu \
  --srs-path /trusted/ceremony.srs --ceremony-descriptor /trusted/ceremony.descriptor.json --clean
```

外部输入覆盖示例：

```bash
python3 scripts/run_gat_ezkl_baseline.py \
  --dataset cora_full --device cpu \
  --config configs/cora_full.cfg \
  --dataset-root /absolute/read-only/data/cache/cora \
  --checkpoint /absolute/read-only/checkpoints/cora_gat \
  --srs-path /trusted/ceremony.srs \
  --ceremony-descriptor /trusted/ceremony.descriptor.json \
  --clean --run
```

## 全量正式入口与结果

```bash
export EZKL_SRS_PATH=/trusted/ceremony.srs
export EZKL_CEREMONY_DESCRIPTOR=/trusted/ceremony.descriptor.json
scripts/run_all.sh
python3 scripts/collect_results.py
python3 scripts/validate_final_results.py
```

`run_all.sh` 默认运行 CPU/CUDA 全矩阵；可用 `EZKL_DEVICES=cpu` 或
`EZKL_DEVICES=cuda` 选择一个正式 backend。只有 native verifier 写出
`verify_passed=true` 的 run 才会进入 `results/formal_results.{json,csv}`；未验证、失败或
中断的 manifest 会被明确排除。`scripts/run_until_complete.sh` 只重试缺失的正式矩阵
cell，不改变 relation、backend 或输入。

CUDA run 还要求独立 GPU build 成功选择 ICICLE CUDA device，并在 proof/verify 后把
`gpu_backend_execution=halo2_icicle_backend_selected_and_proof_verified` 与
`gpu_kernel_telemetry=icicle_cuda_device_initialized` 写入 manifest；CUDA 请求在设备
初始化失败时立即终止，绝不回退到 CPU。

validator 默认要求配置中全部 dataset × `{cpu,cuda}` cells；只检查已经存在的工件时需
显式使用 `python3 scripts/validate_final_results.py --allow-incomplete`。每个正式 run 还会
绑定并重算 source SRS、downsized `pk.srs`/`vk.srs` 的 SHA-256 与原始/目标 `k`。
prove/verify 会各自从 profile 固定的 source SRS 重新执行 deterministic downsize，并将
期望序列化 bytes 与 `pk.srs`、`vk.srs` 逐字节比较，不依赖可修改的 manifest 声明。

本轮没有执行正式 setup/proof/verify 或全数据集实验；资源需求、CUDA runtime 与正式
计时仍需在上述正式命令中验证。

## 运行边界

trusted ceremony descriptor 中的 `source_srs_max_k` 必须是实际 SRS 的原始 `k`，并满足
`source_srs_max_k >= settings.logrows`；代码会逐字节重算原始与 downsized SRS 摘要。
adapter 根据注册模型的 scalar bound ledger 动态生成覆盖全部 family 的 exact LReLU、ELU、
ExpMap 和 MaxReLU 表，`range_bits` 只决定 limb 表的宽度下限，不会截断合法模型的表域。
