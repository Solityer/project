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

本 baseline 的五个正式模型形状已固定为当前 canonical 多层多头结构：`L=3`、
`hidden_profile=[8x8,8x8]`、`K_out=2`。每个 hidden layer 都显式携带
`d_in_profile`；第一项是数据集特征维度，第二项固定为第一隐藏层 concat 后的 `64`。
对应的 `model_arch_id` 为
`gat_family_L3_hid8x8-8x8_kout2_c{C}`，五个 checkpoint 的 model parameter id 与
正式 config 一一匹配。adapter 计算的 `shape_hash` 同时进入 graph spec、settings、
compiled circuit、setup manifest、model registration、public statement、prove/verify
输出、run manifest、collector 和 final validator；任一旧 L2/K1/单层产物或 d_in profile
改动都会在 native 或结果验收边界 fail closed。

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

`build.sh` 只使用当前 `PATH` 中的 `cargo`/`rustc`，不读取开发机 shell 文件或主实验目录；
当前 `rust-toolchain` 对应的 `1.93.0-nightly` compiler release line 不匹配时会在构建前
直接失败。CPU/GPU target、build identity、adapter 和所有输出路径均位于本 baseline 目录内。
正式 Cargo manifest 只暴露 `formal` relation binary；原 ezkl CLI/`execute` 路径及其开发机
lookup-cache 已从正式源码中移除，formal table 每次由当前注册资源生成。

CPU formal binary 为 `target/release/ezkl_formal`；CUDA formal binary 为
`target-gpu/release/ezkl_formal`。CUDA 入口必须成功加载并选择 CUDA device，否则立即失败；
不会回退到 CPU。CPU runner 也拒绝 GPU-enabled binary。

Cargo release 构建有时会把顶层 binary hard-link 到同一 target 下的 `deps/` binary。正式
`build.sh` 会在写入 build identity 前，在相同 baseline-local 目录用 exclusive temporary
regular file 复制并 atomic replace 顶层路径；复制前后都检查路径 containment、symlink/special
entry、内容大小和 executable mode，最终 `nlink` 必须为 `1`。因此 `write_build_identity.py`
和所有 verifier/collector 的严格 hard-link gate 保持不变，直接把未 materialize 的 hard-linked
binary 交给 identity 或 validator 仍会 fail closed。

GPU 构建使用 baseline-local ICICLE CPU/frontend 代码；锁定的公开 ICICLE revision 不包含
CUDA backend source，因此 CUDA native backend 不是由 CMake 联网下载或从 `/opt`、
`LD_LIBRARY_PATH`、`ICICLE_BACKEND_INSTALL_DIR` 等环境路径加载。实验负责人必须先把与锁定
ICICLE commit 和 ABI 匹配的四个共享库放入本目录的
`resources/icicle_cuda_backend/lib/backend/`：
`libicicle_backend_cuda_device.so`、`libicicle_backend_cuda_curve_bn254.so`、
`libicicle_backend_cuda_field_bn254.so`、`libicicle_backend_cuda_hash.so`。
同目录的 `backend.manifest.json` 必须严格包含 `schema`、`backend`、`icicle_commit`、
`abi`、`library_inventory` 五个字段；inventory 按上述顺序记录每个库的 baseline-relative
`path`、正数 `size` 和 `sha256:<64 lowercase hex>`。`prepare_resources.py --backend cuda`
只创建目录和模板，不下载、复制或计算 native library；负责人填入真实 size/digest 后，
以下检查才会通过：

```bash
python3 scripts/prepare_resources.py --backend cuda
# 将四个 ABI-matched DSO 和真实 manifest 放入 resources/icicle_cuda_backend/
python3 scripts/preflight.py --backend cuda --require-staged
EZKL_BUILD_GPU=1 scripts/build.sh
```

GPU build identity、每个 CUDA cell、collector 和 final validator 都会重新校验并绑定这四个
库的 commit、ABI、相对路径、大小和摘要；缺库、额外文件、symlink、hardlink、路径逃逸或
digest 变化均 fail closed。Rust native loader 在每个 DSO 的直接 `dlopen` 前后都会重新
检查 manifest、每个已认证 DSO 的 SHA256/metadata 以及从 baseline 根到文件的全部父路径组件；
底层单文件 FFI 只在 ICICLE 私有 source header 中声明并仅由该认证 wrapper 调用，公共 runtime
header 和 Rust API 均不提供任意路径或递归加载入口。它只接受单个 regular DSO，绝不扫描或递归加载
目录。CPU 构建不需要这些 DSO，也不会尝试加载它们。vendor 目录则由
`configs/formal_vendor_files.txt` 的显式排序清单覆盖，新增或删除任意 vendor 文件会使
source identity 失效。该清单只保留正式 CPU/GPU/proof 路径需要的生产文件；上游
benchmarks、examples、tests、proptest-regressions 以及 halo2curves 的纯 inline
`#[cfg(test)]` 模块均不属于正式依赖闭包，测试 fixture 只位于 `tests/`。

非 vendor production source 也遵守 `ezkl_formal_production_source_filter_v1`：Rust/C/C++/
header、配置和正式 Python/shell 入口使用显式允许的后缀/文件名；任何未知 source 文件
都会使 build identity fail closed。`__pycache__`、`.pytest_cache`、`.mypy_cache`、
`.ruff_cache`、`target`、`build`、`build_gat_adapter`、`tmp`、`logs`、`runs_tmp` 以及
`.pyc`、`.pyo`、`.log`、`.tmp` 是唯一明确排除的生成物，不进入 Python 或 Rust source
manifest；正式 Python 入口同时禁用 bytecode 写入，避免正常 workflow 自己污染 identity。

正式资源必须由实验负责人先放入本目录的 `data/cache/`、`artifacts/checkpoints/` 与
`resources/ceremony/`，不能通过 runner 从主实验目录或外部项目目录读取。资源布局只由下面的
命令检查；该命令不会打开或计算任何完整数据集：

```bash
python3 scripts/prepare_resources.py
python3 scripts/preflight.py --require-staged
```

CUDA 资源检查必须显式选择 CUDA backend（`--backend cuda --require-staged`）；不提供该
选项时只检查通用资源布局，不会把缺失 CUDA native bundle 误报为可运行。任何 dataset、
checkpoint、ceremony 或 native backend 都必须由负责人直接放入 `comparison/ezkl/`，本目录
脚本没有 `--source-root` 或其他外部 project staging 入口。

`resources/ceremony/source.srs` 和
`resources/ceremony/ceremony.descriptor.json` 是唯一接受的正式 ceremony
资源路径。descriptor 必须声明真实的 `source_srs_sha256`、
`source_srs_max_k=28` 和稳定 `ceremony_id`；runner 会对 SRS bytes、descriptor
和所有数据/checkpoint 文件名执行 baseline-local fail-closed 检查。

`prepare_resources.py` 和 `preflight.py` 会在任何写入前逐组件 `lstat` baseline-local
路径；parent、destination、`resources` 或 `resources/ceremony` 的 symlink、hardlink、
special entry 或路径别名都会直接失败，且不会触碰 alias 指向的外部 sentinel。缺失目录
只会在完整 lexical containment 检查之后逐组件安全创建，结果文件使用 exclusive temporary
file + atomic replace。

`resources/resource_manifest.json` 固定五个正式 config、dataset/cache 和 checkpoint
相对路径。`--check`/`--require-staged` 只检查必需文件名存在，内容、维度、checkpoint
身份仍由 native adapter 在正式运行时 fail closed 校验。

`prepare_resources.py --check` 在 ceremony、dataset 或 checkpoint 缺失时输出
`ezkl_formal_resource_check_v1` 的 `status=blocked` JSON 并以非零状态退出，不会以裸
traceback 掩盖缺口；`preflight.py --require-staged` 同样输出结构化 blocked 报告。

shape identity 的生成与校验由 C++ adapter、Rust native circuit 和 Python final validator
共享同一 length-prefixed、domain-separated 编码；因此仅修改 manifest 中的 shape 字段而不
重建 compiled circuit/keys 不能绕过 artifact binding。

## 单数据集正式运行

默认 baseline-local 数据与 checkpoint 路径登记在 `configs/full_datasets.json`，也可通过 CLI
显式覆盖为 baseline 内路径。以下命令依次执行 adapter、settings、compile、setup、full witness、proof 和
independent verify：

首先由可信 ceremony 管理方把 descriptor（`descriptor_kind=ezkl_trusted_ceremony_srs_v1`、
`ceremony_id`、`source_srs_sha256`、真实的 `source_srs_max_k`）放入
`resources/ceremony/ceremony.descriptor.json`，并把对应 SRS 放入
`resources/ceremony/source.srs`。runner 会在 adapter 之前
校验 SRS bytes 并物化 resolved profile；SRS 的真实 ceremony 身份、摘要和原始 `k` 会贯穿
setup/prove/verify，且必须覆盖编译电路的 `logrows`。

```bash
python3 scripts/run_gat_ezkl_baseline.py \
  --dataset cora_full --device cpu --clean --run

python3 scripts/run_gat_ezkl_baseline.py \
  --dataset cora_full --device cuda --clean --run
```

仅生成 canonical adapter inputs（不产生 proof；stage 会在进程结束时清理，不会写入
正式 result cell）：

```bash
python3 scripts/run_gat_ezkl_baseline.py --dataset cora_full --device cpu --clean
```

外部输入覆盖示例：

```bash
python3 scripts/run_gat_ezkl_baseline.py \
  --dataset cora_full --device cpu \
  --config configs/cora_full.cfg \
  --dataset-root data/cache/cora \
  --checkpoint artifacts/checkpoints/cora_gat \
  --clean --run
```

## 全量正式入口与结果

```bash
scripts/run_all.sh
python3 scripts/collect_results.py
python3 scripts/validate_final_results.py
```

`run_all.sh` 是唯一正式入口，并且只运行完整 CPU/CUDA 5×2 矩阵；它不接受 backend 或数据集
子集选择环境变量。只有 native verifier 写出 `verify_passed=true` 的 run 才会进入矩阵
summary；未验证、失败或中断的 manifest 会被明确排除。每个正式 cell 只会在真实 proof 与独立 native verify
成功且通过精确 artifact inventory 后，原子提交到
`results/runs/<backend>/<dataset>/`。临时 setup/prove 工作区在
`results/runs_tmp/`，日志在 `results/logs/`，都不会进入 cell。
runner 不再写 `results/<dataset>.<device>.json` 扁平副产物；正式结果只保留 canonical
cell 和 `results/formal_results.{json,csv}` 矩阵 summary。

当所有十个 cell 已由真实 proof 与独立 native verify 成功后，默认全矩阵入口会执行严格
validator；缺少任一 cell 时 fail closed。单个 cell 的成功仍只由该 cell 的真实 proof +
fresh independent verify 判定，不依赖矩阵 validator。

CUDA run 还要求独立 GPU build 成功选择 ICICLE CUDA device，并在真实 prove scope 内
取得 CUPTI kernel activity（`gpu_backend_execution=cuda_icicle_prove_only`、
`gpu_prove_telemetry`）；verify 始终由独立 CPU binary 执行，manifest 固定
`prover_backend=cuda_icicle`、`verifier_backend=cpu_independent`。CUDA 请求在设备初始化
或 CUPTI 记录失败时立即终止，绝不回退到 CPU，也不声称 GPU verify。

validator 默认要求配置中全部 dataset × `{cpu,cuda}` cells；只检查已经存在的
工件时需显式使用 `python3 scripts/validate_final_results.py --allow-incomplete`。每个正式 run 还会
绑定并重算 source SRS、downsized `pk.srs`/`vk.srs` 的 SHA-256 与原始/目标 `k`。
prove/verify 会各自从 profile 固定的 source SRS 重新执行 deterministic downsize，并将
期望序列化 bytes 与 `pk.srs`、`vk.srs` 逐字节比较，不依赖可修改的 manifest 声明。

本轮没有执行正式 setup/proof/verify 或全数据集实验；资源需求、CUDA runtime 与正式
计时仍需在上述正式命令中验证。

Halo2 的 ICICLE 小型 FFT CPU 边界固定为源码常量 `ICICLE_SMALL_K_FFT = 2`；它不读取
环境变量，避免通过运行时覆盖改变正式 CPU/GPU 路由。

## 运行边界

trusted ceremony descriptor 中的 `source_srs_max_k` 必须是实际 SRS 的原始 `k`，并满足
`source_srs_max_k >= settings.logrows`；代码会逐字节重算原始与 downsized SRS 摘要。
adapter 根据注册模型的 scalar bound ledger 动态生成覆盖全部 family 的 exact LReLU、ELU、
ExpMap 和 MaxReLU 表，`range_bits` 只决定 limb 表的宽度下限，不会截断合法模型的表域。
