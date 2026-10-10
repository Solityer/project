# zkGAT 主实验

本仓库只有一条正式主实验路径：`gatzk_run` 加载当前 canonical 配置、真实训练得到的
PyTorch checkpoint bundle、对应的 trusted SRS 和同一份公开数据，执行统一的
commitment、prove、verify 与结果导出。协议约束以只读文件
`GAT-ZKML-多层多头.md` 和 `PROJECT_CONTEXT_AND_EXECUTION_CONSTRAINTS.md` 为准。

当前五个正式数据集都使用同一模型族：

| 数据集 | `d_in_profile` | `L` | `hidden_profile` | `K_out` | `C` | `range_bits` |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| Cora | `[1433,64]` | 3 | `[8x8,8x8]` | 2 | 7 | 12 |
| Citeseer | `[3703,64]` | 3 | `[8x8,8x8]` | 2 | 6 | 12 |
| Pubmed | `[500,64]` | 3 | `[8x8,8x8]` | 2 | 3 | 12 |
| PPI | `[50,64]` | 3 | `[8x8,8x8]` | 2 | 121 | 14 |
| OGBN-Arxiv | `[128,64]` | 3 | `[8x8,8x8]` | 2 | 40 | 21 |

正式 proof schema 已按层实例化。隐藏头的唯一正式身份是
`(layer, local_head)`，标签形如 `P_h_l<layer>_h<head>`；每个隐藏层分别拥有
`Q_in`、`Q_d_h`、`Q_cat`、`Q_Nd_h`，并且最后隐藏层的 `Q_cat` 与输出层的
`Q_cat` 是两个独立对象。trace、quotient、prover、verifier 和 codec 都按
`[layer][local_head]` 访问这些对象，隐藏层 concat 也始终保留显式层号。
checkpoint、parity、proof 与 transcript 使用同一结构化身份。

当前 canonical 设计固定 instance/proof schema、对象顺序、degree bound 计算和工作域
目录。正式工件必须由当前源码从 canonical 配置重新生成，并通过 compiler artifact、
envelope、domain 与 degree binding 的完整校验；不能仅凭 nominal degree 或 protocol ID
接受不匹配的 material。当前 canonical SRS 目录为空，尚未执行 ceremony；后续必须按下方
完整命令 compile、ceremony 和 validate。

正式 checkpoint、SRS、完整 proof/verify 和 CPU/GPU 十组正式运行需要在本轮之后执行；
训练不启用 strict deterministic algorithms，固定 `seed=11`。同一数据集的 CPU 与 CUDA
运行必须使用同一个 canonical checkpoint、配置和 SRS。

## 依赖

正式训练必须使用 CUDA 版 PyTorch；当前训练器在 CUDA 不可用或传入 CPU 选项时直接失败。
在项目根目录执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --upgrade numpy scipy
.venv/bin/python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
```

## 构建

CPU Release：

```bash
cmake --build build -j10
```

CUDA Release：

```bash
cmake -S . -B build_cuda -DCMAKE_BUILD_TYPE=Release -DGATZK_ENABLE_CUDA_BACKEND=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build_cuda -j10
```

上述正式构建默认不编译 `tests/` 中的 test-only targets。正式构建的短时 schema/static 检查（不加载正式数据、不生成 checkpoint/SRS/run）：

```bash
cmake --build build -j10
cmake --build build_cuda -j10
python3 tools/security/static_scan_no_tau.py
git diff --check -- README.md include src tests
```

测试构建、targeted runtime tests、Planetoid parity 和 PPI diagnostic 配置均位于
`tests/`；独立测试构建与路径说明见 [`tests/README.md`](tests/README.md)。

`--compute-backend cuda_hotspots` 需要 CUDA-enabled binary、可用 CUDA runtime 和至少
一个真实 CUDA kernel。当前 backend 只把已有 size-aware hotspot 路径放到 CUDA，pairing、
部分 NTT/IFFT、quotient 和小型/冷启动工作仍按当前实现留在 CPU；零 kernel 或 runtime
缺失都会失败关闭，不会把 CPU fallback 写成 GPU 结果。

## 数据准备

```bash
.venv/bin/python scripts/prepare_planetoid.py --data-root data --dataset cora --cache-root data/cache
.venv/bin/python scripts/prepare_planetoid.py --data-root data --dataset citeseer --cache-root data/cache
.venv/bin/python scripts/prepare_planetoid.py --data-root data --dataset pubmed --cache-root data/cache
.venv/bin/python scripts/prepare_ppi.py --data-root data/ppi --output-root data/cache/ppi
.venv/bin/python scripts/prepare_ogbn_arxiv.py --project-root . --data-root data
```

PPI cache 保留原始 121 维 multi-label 矩阵 `labels_multi.npy` 及供 C++ formal loader
读取的同序 `labels_multi.txt`，并保存 `train_mask.npy`、`val_mask.npy`、`test_mask.npy`；
不生成单标签 `labels.txt`。图组件、
特征行、边端点、`node_ptr`、图边界、去重和稳定边排序保持现有正式语义。

## 正式 GPU 训练

以下命令必须自然完成或真实失败，不要包裹固定 timeout。每个命令都写入对应 canonical
checkpoint 目录；`--overwrite` 只用于在确认没有并发训练后替换该目录中的本轮新模型。
最佳模型和 early stopping 只依据 validation loss，test split 不参与模型选择。

```bash
.venv/bin/python scripts/train_planetoid_gat.py --config configs/cora_secure.cfg --epochs 500 --patience 100 --learning-rate 0.005 --weight-decay 0.0005 --dropout 0.6 --seed 11 --use-cuda --overwrite
.venv/bin/python scripts/train_planetoid_gat.py --config configs/citeseer_secure.cfg --epochs 500 --patience 100 --learning-rate 0.005 --weight-decay 0.0005 --dropout 0.6 --seed 11 --use-cuda --overwrite
.venv/bin/python scripts/train_planetoid_gat.py --config configs/pubmed_secure.cfg --epochs 500 --patience 100 --learning-rate 0.01 --weight-decay 0.001 --dropout 0.6 --seed 11 --use-cuda --overwrite
.venv/bin/python scripts/train_ppi_gat.py --config configs/ppi_secure.cfg --epochs 500 --patience 100 --learning-rate 0.005 --weight-decay 0 --dropout 0.0 --seed 11 --use-cuda --overwrite
.venv/bin/python scripts/train_ogbn_arxiv_gat.py --config configs/ogbn_arxiv_secure.cfg --epochs 500 --patience 100 --learning-rate 0.005 --weight-decay 0.0005 --dropout 0.5 --seed 11 --use-cuda --overwrite
```

PPI 训练复用 `train_planetoid_gat.py` 的 `PlanetoidGAT`、optimizer、seed 和 checkpoint
链，但使用 `BCEWithLogitsLoss`，输出维度固定为 121；其 test mask 只保存在数据语义中，
不用于最佳模型选择。

## checkpoint export 与严格检查

训练完成后，先把真实 `best_model.pt` 导出到同一 canonical bundle 目录：

```bash
.venv/bin/python scripts/export_torch_gat_bundle.py --checkpoint artifacts/checkpoints/cora_gat/best_model.pt --output-dir artifacts/checkpoints/cora_gat
.venv/bin/python scripts/export_torch_gat_bundle.py --checkpoint artifacts/checkpoints/citeseer_gat/best_model.pt --output-dir artifacts/checkpoints/citeseer_gat
.venv/bin/python scripts/export_torch_gat_bundle.py --checkpoint artifacts/checkpoints/pubmed_gat/best_model.pt --output-dir artifacts/checkpoints/pubmed_gat
.venv/bin/python scripts/import_ppi_bundle.py --torch-checkpoint artifacts/checkpoints/ppi_gat/best_model.pt --output-dir artifacts/checkpoints/ppi_gat
.venv/bin/python scripts/export_torch_gat_bundle.py --checkpoint artifacts/checkpoints/ogbn_arxiv_gat/best_model.pt --output-dir artifacts/checkpoints/ogbn_arxiv_gat
```

然后执行严格的 bundle、source checkpoint、tensor index 和每个 kernel/bias shape 检查：

```bash
.venv/bin/python scripts/check_checkpoint_bundle.py --dataset cora --bundle-dir artifacts/checkpoints/cora_gat --checkpoint artifacts/checkpoints/cora_gat/best_model.pt
.venv/bin/python scripts/check_checkpoint_bundle.py --dataset citeseer --bundle-dir artifacts/checkpoints/citeseer_gat --checkpoint artifacts/checkpoints/citeseer_gat/best_model.pt
.venv/bin/python scripts/check_checkpoint_bundle.py --dataset pubmed --bundle-dir artifacts/checkpoints/pubmed_gat --checkpoint artifacts/checkpoints/pubmed_gat/best_model.pt
.venv/bin/python scripts/check_checkpoint_bundle.py --dataset ppi --bundle-dir artifacts/checkpoints/ppi_gat --checkpoint artifacts/checkpoints/ppi_gat/best_model.pt
.venv/bin/python scripts/check_checkpoint_bundle.py --dataset ogbn_arxiv --bundle-dir artifacts/checkpoints/ogbn_arxiv_gat --checkpoint artifacts/checkpoints/ogbn_arxiv_gat/best_model.pt
```

每个检查都必须报告 `checkpoint bundle shape check passed`。任何不符合 canonical
多层多头结构的 checkpoint 都会因 metadata、state_dict key 或 shape 不匹配而拒绝；禁止
复制、reshape、补权重或改名生成正式模型。

## forward/parity 短检查

三个 Planetoid 数据集的 checkpoint 与导出 bundle CPU forward trace parity 属于
test-only 检查；命令和临时输出约束见 [`tests/README.md`](tests/README.md)。

## SRS 检查与必要重建

当前 canonical secure config 的 SRS envelope 按 `L=3`、16 个 hidden heads、两个输出
heads、concat width 64 和各数据集真实容量定义。SRS material 必须与当前 compiler artifact、
envelope、domain、degree binding 和 canonical `srs_id` 完整一致；任一不匹配都拒绝并重新
生成。当前 canonical SRS 目录为空，尚未 ceremony，后续仍必须执行下面完整流程。下列命令
使用现有 `build/gatzk_srs_setup`、当前 degree bound 和 domain catalog；ceremony 会采样真实
setup，不能在本轮短验证中执行。

先完成每个数据集的 compile、ceremony 和独立 validate；只有 validate 成功后才删除同名
canonical 目录并移动 staging 目录，禁止直接在 canonical 目录中部分写入：

```bash
srs_stage_cora=$(mktemp -d /tmp/zkgat-srs-cora.XXXXXX)
build/gatzk_srs_setup compile "$srs_stage_cora/protocol_compiler_artifact.txt" 16777215 configs/srs_cora_max_envelope.cfg C=16 FH=4194304 N=4096 cat=128 d=16 edge=32768 in=2048
build/gatzk_srs_setup ceremony "$srs_stage_cora/material" "$srs_stage_cora/protocol_compiler_artifact.txt" cora_formal_srs
build/gatzk_srs_setup validate "$srs_stage_cora/material" cora_formal_srs
rm -rf artifacts/srs/cora_formal_srs
mv "$srs_stage_cora/material" artifacts/srs/cora_formal_srs
rm -rf "$srs_stage_cora"

srs_stage_citeseer=$(mktemp -d /tmp/zkgat-srs-citeseer.XXXXXX)
build/gatzk_srs_setup compile "$srs_stage_citeseer/protocol_compiler_artifact.txt" 67108863 configs/srs_citeseer_max_envelope.cfg C=8 FH=16777216 N=4096 cat=128 d=16 edge=32768 in=4096
build/gatzk_srs_setup ceremony "$srs_stage_citeseer/material" "$srs_stage_citeseer/protocol_compiler_artifact.txt" citeseer_formal_srs
build/gatzk_srs_setup validate "$srs_stage_citeseer/material" citeseer_formal_srs
rm -rf artifacts/srs/citeseer_formal_srs
mv "$srs_stage_citeseer/material" artifacts/srs/citeseer_formal_srs
rm -rf "$srs_stage_citeseer"

srs_stage_pubmed=$(mktemp -d /tmp/zkgat-srs-pubmed.XXXXXX)
build/gatzk_srs_setup compile "$srs_stage_pubmed/protocol_compiler_artifact.txt" 67108863 configs/srs_pubmed_max_envelope.cfg C=8 FH=16777216 N=32768 cat=128 d=16 edge=262144 in=512
build/gatzk_srs_setup ceremony "$srs_stage_pubmed/material" "$srs_stage_pubmed/protocol_compiler_artifact.txt" pubmed_formal_srs
build/gatzk_srs_setup validate "$srs_stage_pubmed/material" pubmed_formal_srs
rm -rf artifacts/srs/pubmed_formal_srs
mv "$srs_stage_pubmed/material" artifacts/srs/pubmed_formal_srs
rm -rf "$srs_stage_pubmed"

srs_stage_ppi=$(mktemp -d /tmp/zkgat-srs-ppi.XXXXXX)
build/gatzk_srs_setup compile "$srs_stage_ppi/protocol_compiler_artifact.txt" 33554431 configs/srs_ppi_max_envelope.cfg C=128 FH=4194304 N=65536 cat=64 d=16 edge=8388608 in=64
build/gatzk_srs_setup ceremony "$srs_stage_ppi/material" "$srs_stage_ppi/protocol_compiler_artifact.txt" ppi_formal_srs
build/gatzk_srs_setup validate "$srs_stage_ppi/material" ppi_formal_srs
rm -rf artifacts/srs/ppi_formal_srs
mv "$srs_stage_ppi/material" artifacts/srs/ppi_formal_srs
rm -rf "$srs_stage_ppi"

srs_stage_ogbn=$(mktemp -d /tmp/zkgat-srs-ogbn_arxiv.XXXXXX)
build/gatzk_srs_setup compile "$srs_stage_ogbn/protocol_compiler_artifact.txt" 134217727 configs/srs_ogbn_arxiv_max_envelope.cfg C=64 FH=33554432 N=262144 cat=64 d=16 edge=8388608 in=256
build/gatzk_srs_setup ceremony "$srs_stage_ogbn/material" "$srs_stage_ogbn/protocol_compiler_artifact.txt" ogbn_arxiv_formal_srs
build/gatzk_srs_setup validate "$srs_stage_ogbn/material" ogbn_arxiv_formal_srs
rm -rf artifacts/srs/ogbn_arxiv_formal_srs
mv "$srs_stage_ogbn/material" artifacts/srs/ogbn_arxiv_formal_srs
rm -rf "$srs_stage_ogbn"
```

SRS 生成完成后检查五个 canonical manifest，并执行安全绑定扫描：

```bash
build/gatzk_srs_setup validate artifacts/srs/cora_formal_srs cora_formal_srs
build/gatzk_srs_setup validate artifacts/srs/citeseer_formal_srs citeseer_formal_srs
build/gatzk_srs_setup validate artifacts/srs/pubmed_formal_srs pubmed_formal_srs
build/gatzk_srs_setup validate artifacts/srs/ppi_formal_srs ppi_formal_srs
build/gatzk_srs_setup validate artifacts/srs/ogbn_arxiv_formal_srs ogbn_arxiv_formal_srs
python3 tools/security/static_scan_no_tau.py --require-all-configured-srs
```

## 正式 CPU 与 CUDA runs

完成新 checkpoint、bundle 检查和对应 SRS 后，执行五个 full CPU runs：

```bash
build/gatzk_run --config configs/cora_secure.cfg --compute-backend cpu --export-tag cpu_formal
build/gatzk_run --config configs/citeseer_secure.cfg --compute-backend cpu --export-tag cpu_formal
build/gatzk_run --config configs/pubmed_secure.cfg --compute-backend cpu --export-tag cpu_formal
build/gatzk_run --config configs/ppi_secure.cfg --compute-backend cpu --export-tag cpu_formal
build/gatzk_run --config configs/ogbn_arxiv_secure.cfg --compute-backend cpu --export-tag cpu_formal
```

再执行五个同配置、同 checkpoint、同 SRS 的 CUDA hotspot runs：

```bash
build_cuda/gatzk_run --config configs/cora_secure.cfg --compute-backend cuda_hotspots --export-tag cuda_formal
build_cuda/gatzk_run --config configs/citeseer_secure.cfg --compute-backend cuda_hotspots --export-tag cuda_formal
build_cuda/gatzk_run --config configs/pubmed_secure.cfg --compute-backend cuda_hotspots --export-tag cuda_formal
build_cuda/gatzk_run --config configs/ppi_secure.cfg --compute-backend cuda_hotspots --export-tag cuda_formal
build_cuda/gatzk_run --config configs/ogbn_arxiv_secure.cfg --compute-backend cuda_hotspots --export-tag cuda_formal
```

每个目录中的 `warm/run_manifest.json` 和 `proof.bin` 是结果真源。以下检查必须全部成功；
`range_table_size` 分别对应各配置的 `range_bits`，`private_plain_value_count` 必须为 0，
GPU manifest 还必须记录 runtime 和正数 kernel invocation：

```bash
jq -e '.status == "VERIFY_OK" and .verified == "true" and .full_dataset == "true" and .stale_result == "false" and .verifier_reads_witness == "false" and .private_plain_value_count == "0" and .range_table_size == "4096" and .gpu_backend_requested == "false"' runs/cora_secure/cpu_formal/warm/run_manifest.json
jq -e '.status == "VERIFY_OK" and .verified == "true" and .full_dataset == "true" and .stale_result == "false" and .verifier_reads_witness == "false" and .private_plain_value_count == "0" and .range_table_size == "4096" and .gpu_backend_requested == "false"' runs/citeseer_secure/cpu_formal/warm/run_manifest.json
jq -e '.status == "VERIFY_OK" and .verified == "true" and .full_dataset == "true" and .stale_result == "false" and .verifier_reads_witness == "false" and .private_plain_value_count == "0" and .range_table_size == "4096" and .gpu_backend_requested == "false"' runs/pubmed_secure/cpu_formal/warm/run_manifest.json
jq -e '.status == "VERIFY_OK" and .verified == "true" and .full_dataset == "true" and .stale_result == "false" and .verifier_reads_witness == "false" and .private_plain_value_count == "0" and .range_table_size == "16384" and .gpu_backend_requested == "false"' runs/ppi_secure/cpu_formal/warm/run_manifest.json
jq -e '.status == "VERIFY_OK" and .verified == "true" and .full_dataset == "true" and .stale_result == "false" and .verifier_reads_witness == "false" and .private_plain_value_count == "0" and .range_table_size == "2097152" and .gpu_backend_requested == "false"' runs/ogbn_arxiv_secure/cpu_formal/warm/run_manifest.json

jq -e '.status == "VERIFY_OK" and .verified == "true" and .full_dataset == "true" and .stale_result == "false" and .verifier_reads_witness == "false" and .private_plain_value_count == "0" and .gpu_backend_requested == "true" and .gpu_runtime_present == "true" and ((.gpu_kernel_invocation_count | tonumber) > 0)' runs/cora_secure/cuda_formal/warm/run_manifest.json
jq -e '.status == "VERIFY_OK" and .verified == "true" and .full_dataset == "true" and .stale_result == "false" and .verifier_reads_witness == "false" and .private_plain_value_count == "0" and .gpu_backend_requested == "true" and .gpu_runtime_present == "true" and ((.gpu_kernel_invocation_count | tonumber) > 0)' runs/citeseer_secure/cuda_formal/warm/run_manifest.json
jq -e '.status == "VERIFY_OK" and .verified == "true" and .full_dataset == "true" and .stale_result == "false" and .verifier_reads_witness == "false" and .private_plain_value_count == "0" and .gpu_backend_requested == "true" and .gpu_runtime_present == "true" and ((.gpu_kernel_invocation_count | tonumber) > 0)' runs/pubmed_secure/cuda_formal/warm/run_manifest.json
jq -e '.status == "VERIFY_OK" and .verified == "true" and .full_dataset == "true" and .stale_result == "false" and .verifier_reads_witness == "false" and .private_plain_value_count == "0" and .gpu_backend_requested == "true" and .gpu_runtime_present == "true" and ((.gpu_kernel_invocation_count | tonumber) > 0)' runs/ppi_secure/cuda_formal/warm/run_manifest.json
jq -e '.status == "VERIFY_OK" and .verified == "true" and .full_dataset == "true" and .stale_result == "false" and .verifier_reads_witness == "false" and .private_plain_value_count == "0" and .gpu_backend_requested == "true" and .gpu_runtime_present == "true" and ((.gpu_kernel_invocation_count | tonumber) > 0)' runs/ogbn_arxiv_secure/cuda_formal/warm/run_manifest.json
```

对每个成功 run 还要核验 `proof_hash` 与 `sha256sum proof.bin` 相等，配置、模型、量化和
SRS hash 非空；只有这些 current-source manifest 才能进入结果表。运行过程中可设置
`GATZK_RUN_PROGRESS=1` 观察阶段进度，但不要改变计时边界或添加固定 timeout。

## 结果整理

每个 dataset/backend 的 protocol tamper suite 都必须由用户在独立审阅后提供一份
external attestation 和它的外部 SHA-256 信任锚。`gatzk_protocol_tamper_tests` 的输出
只允许写入仓库 canonical `tests/` 目录（推荐 `tests/attestations/`）；不要把测试
artifact 写入仓库外或 `runs/`、`artifacts/`、`results/`。聚合器使用
`--cpu-attestation dataset=path`、`--cpu-attestation-digest dataset=sha256` 以及对应的
`--cuda-attestation*` 参数逐项认证五组 CPU/CUDA attestation；benchmark exporter 使用
`--attestation dataset=path` 和 `--attestation-digest dataset=sha256`。摘要必须由用户
从 attestation 文件外部计算（例如 `sha256sum`），不能把 attestation 自身的字段当作信任根。
正式 verifier report 中 `tamper_tests_passed=false`、空 map、`tamper_tests_ms=0` 是待补
状态，不是安全通过；只有外部 digest 已校验且所有 provenance/schema/public-input/topology
binding 一致时，consumer 才会输出 tamper pass。

十组正式 run 全部通过上述检查后，生成 CPU 与 CUDA 两份结果目录：

```bash
.venv/bin/python scripts/export_benchmark_table.py --output-dir results/paper_secure/cpu --run cora=runs/cora_secure/cpu_formal/warm --run citeseer=runs/citeseer_secure/cpu_formal/warm --run pubmed=runs/pubmed_secure/cpu_formal/warm --run ppi=runs/ppi_secure/cpu_formal/warm --run ogbn_arxiv=runs/ogbn_arxiv_secure/cpu_formal/warm --attestation cora=tests/attestations/cora_cpu_tamper.json --attestation citeseer=tests/attestations/citeseer_cpu_tamper.json --attestation pubmed=tests/attestations/pubmed_cpu_tamper.json --attestation ppi=tests/attestations/ppi_cpu_tamper.json --attestation ogbn_arxiv=tests/attestations/ogbn_arxiv_cpu_tamper.json --attestation-digest cora=<reviewed-cora-cpu-attestation-sha256> --attestation-digest citeseer=<reviewed-citeseer-cpu-attestation-sha256> --attestation-digest pubmed=<reviewed-pubmed-cpu-attestation-sha256> --attestation-digest ppi=<reviewed-ppi-cpu-attestation-sha256> --attestation-digest ogbn_arxiv=<reviewed-ogbn-arxiv-cpu-attestation-sha256>
.venv/bin/python scripts/export_benchmark_table.py --output-dir results/paper_secure/cuda --run cora=runs/cora_secure/cuda_formal/warm --run citeseer=runs/citeseer_secure/cuda_formal/warm --run pubmed=runs/pubmed_secure/cuda_formal/warm --run ppi=runs/ppi_secure/cuda_formal/warm --run ogbn_arxiv=runs/ogbn_arxiv_secure/cuda_formal/warm --attestation cora=tests/attestations/cora_cuda_tamper.json --attestation citeseer=tests/attestations/citeseer_cuda_tamper.json --attestation pubmed=tests/attestations/pubmed_cuda_tamper.json --attestation ppi=tests/attestations/ppi_cuda_tamper.json --attestation ogbn_arxiv=tests/attestations/ogbn_arxiv_cuda_tamper.json --attestation-digest cora=<reviewed-cora-cuda-attestation-sha256> --attestation-digest citeseer=<reviewed-citeseer-cuda-attestation-sha256> --attestation-digest pubmed=<reviewed-pubmed-cuda-attestation-sha256> --attestation-digest ppi=<reviewed-ppi-cuda-attestation-sha256> --attestation-digest ogbn_arxiv=<reviewed-ogbn-arxiv-cuda-attestation-sha256>
jq -e '.benchmark_mode == "warm" and (.rows | length) == 5' results/paper_secure/cpu/latest.json
jq -e '.benchmark_mode == "warm" and (.rows | length) == 5' results/paper_secure/cuda/latest.json
```

将十个已通过上述逐项核验的 manifest 聚合为唯一根结果集。该命令要求五个 CPU 和五个
CUDA run 全部存在且通过 `VERIFY_OK`、full-dataset、proof hash、配置/模型/量化/SRS
配对和安全字段检查。聚合器还会重新计算 canonical config、checkpoint bundle 的
`manifest.json`/`tensors.txt`/`tensors.npz`/`best_model.pt` 以及 SRS
`srs_manifest.json` 的 SHA-256，并与 run manifest 中的真实记录逐项比对；任一缺失、
篡改或不一致都会 fail-closed，且不会写出根结果文件：

```bash
.venv/bin/python scripts/aggregate_secure_results.py --output-dir results/paper_secure --cpu-run cora=runs/cora_secure/cpu_formal/warm --cpu-run citeseer=runs/citeseer_secure/cpu_formal/warm --cpu-run pubmed=runs/pubmed_secure/cpu_formal/warm --cpu-run ppi=runs/ppi_secure/cpu_formal/warm --cpu-run ogbn_arxiv=runs/ogbn_arxiv_secure/cpu_formal/warm --cuda-run cora=runs/cora_secure/cuda_formal/warm --cuda-run citeseer=runs/citeseer_secure/cuda_formal/warm --cuda-run pubmed=runs/pubmed_secure/cuda_formal/warm --cuda-run ppi=runs/ppi_secure/cuda_formal/warm --cuda-run ogbn_arxiv=runs/ogbn_arxiv_secure/cuda_formal/warm --cpu-attestation cora=tests/attestations/cora_cpu_tamper.json --cpu-attestation citeseer=tests/attestations/citeseer_cpu_tamper.json --cpu-attestation pubmed=tests/attestations/pubmed_cpu_tamper.json --cpu-attestation ppi=tests/attestations/ppi_cpu_tamper.json --cpu-attestation ogbn_arxiv=tests/attestations/ogbn_arxiv_cpu_tamper.json --cpu-attestation-digest cora=<reviewed-cora-cpu-attestation-sha256> --cpu-attestation-digest citeseer=<reviewed-citeseer-cpu-attestation-sha256> --cpu-attestation-digest pubmed=<reviewed-pubmed-cpu-attestation-sha256> --cpu-attestation-digest ppi=<reviewed-ppi-cpu-attestation-sha256> --cpu-attestation-digest ogbn_arxiv=<reviewed-ogbn-arxiv-cpu-attestation-sha256> --cuda-attestation cora=tests/attestations/cora_cuda_tamper.json --cuda-attestation citeseer=tests/attestations/citeseer_cuda_tamper.json --cuda-attestation pubmed=tests/attestations/pubmed_cuda_tamper.json --cuda-attestation ppi=tests/attestations/ppi_cuda_tamper.json --cuda-attestation ogbn_arxiv=tests/attestations/ogbn_arxiv_cuda_tamper.json --cuda-attestation-digest cora=<reviewed-cora-cuda-attestation-sha256> --cuda-attestation-digest citeseer=<reviewed-citeseer-cuda-attestation-sha256> --cuda-attestation-digest pubmed=<reviewed-pubmed-cuda-attestation-sha256> --cuda-attestation-digest ppi=<reviewed-ppi-cuda-attestation-sha256> --cuda-attestation-digest ogbn_arxiv=<reviewed-ogbn-arxiv-cuda-attestation-sha256>
jq -e '.run_count == 10 and (.rows | length) == 10 and .security_tests.all_passed == true' results/paper_secure/latest.json
test -s results/paper_secure/security_tests_report.md
test "$(awk 'END {print NR-1}' results/paper_secure/latest.csv)" -eq 10
```

上述聚合命令成功后，`results/paper_secure/summary.md`、`latest.csv`、`latest.json` 和
`security_tests_report.md` 才是十组正式结果的根级交付面；本轮没有执行正式训练、SRS
ceremony、完整 proof/verify 或十组实验，因此当前不生成这些文件。

当前 README 只描述这套 canonical 状态；正式训练、checkpoint 生成、SRS ceremony、完整
proof/verify 和十组最终结果在执行完成前均不视为已完成。

## 真实图机制级消融

本轮已完成 11 组真实图配置、每组一对 Original/Control，共 22 份 CUDA 机制证明。每个配置只测量一次，因此下表直接列出单次测量值，不报告标准差、置信区间或显著性。数值格式为 Original / Control；Saving = (Control − Original) / Control × 100%；Factor = Control / Original。正 Saving 表示 Original 在该指标上更小。

实验中间值来自正式 GAT 第一隐藏层第 0 个注意力头，宽度为 8。Feature Folding 的 D=1、2、4 使用同一个真实 8 维 H′ 的前 D 个坐标，是受控机制宽度实验，不代表正式模型完整单头计算。Shared Neighborhood 使用该头的有限域 U、Sum、归一化权重、加权压缩边特征和聚合前目标值；聚合值取自后续激活及舍入之前。实验只证明并比较这两项独立机制，不代表完整 GAT 端到端证明性能。

### Feature Folding

| 数据集 | N | E | D | G | Commitment O/C ms；Saving；Factor | Prove O/C ms；Saving；Factor | Verify O/C ms；Saving；Factor | Proof Size O/C bytes；Saving；Factor |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| Cora | 2708 | 13264 | 1 | 1 | 124.879424 / 122.921320; -1.592973%; 0.984320× | 218.130203 / 212.301324; -2.745569%; 0.973278× | 35.973663 / 33.691886; -6.772482%; 0.936571× | 3848 / 3743; -2.805236%; 0.972713× |
| Cora | 2708 | 13264 | 2 | 1 | 124.271522 / 227.543787; 45.385667%; 1.831021× | 217.102000 / 393.137180; 44.777037%; 1.810841× | 36.509525 / 50.775638; 28.096374%; 1.390750× | 3982 / 7105; 43.954961%; 1.784279× |
| Cora | 2708 | 13264 | 4 | 1 | 128.251546 / 423.889607; 69.744116%; 3.305142× | 215.731652 / 750.205907; 71.243675%; 3.477496× | 36.667271 / 79.976278; 54.152316%; 2.181135× | 4250 / 13829; 69.267481%; 3.253882× |
| Cora | 2708 | 13264 | 8 | 1 | 130.864436 / 813.990798; 83.923106%; 6.220107× | 220.331356 / 1468.470055; 84.995856%; 6.664826× | 37.459600 / 137.177366; 72.692580%; 3.662008× | 4786 / 27277; 82.454082%; 5.699331× |
| Citeseer | 3327 | 12431 | 8 | 1 | 135.445006 / 819.076726; 83.463697%; 6.047301× | 217.005365 / 1465.546957; 85.192875%; 6.753506× | 35.591890 / 123.092559; 71.085263%; 3.458444× | 4790 / 27281; 82.441993%; 5.695407× |
| PPI | 1546 | 17699 | 8 | 1 | 4325.626412 / 25648.274180; 83.134825%; 5.929378× | 23062.817338 / 155731.693336; 85.190672%; 6.752501× | 581.435836 / 4774.692892; 87.822550%; 8.211900× | 4785 / 27276; 82.457105%; 5.700313× |
| PPI | 2923 | 33928 | 8 | 2 | 4272.736449 / 25611.597633; 83.317181%; 5.994191× | 22468.499785 / 154619.331276; 85.468505%; 6.881605× | 635.159136 / 4739.317609; 86.598089%; 7.461622× | 4785 / 27276; 82.457105%; 5.700313× |

### Shared Neighborhood

| 数据集 | N | E | D | G | Commitment O/C ms；Saving；Factor | Prove O/C ms；Saving；Factor | Verify O/C ms；Saving；Factor | Proof Size O/C bytes；Saving；Factor |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| Cora | 2708 | 13264 | 8 | 1 | 197.048008 / 320.234547; 38.467598%; 1.625160× | 334.916930 / 620.896177; 46.059109%; 1.853881× | 41.350309 / 69.335738; 40.362200%; 1.676789× | 6429 / 10512; 38.841324%; 1.635091× |
| Citeseer | 3327 | 12431 | 8 | 1 | 196.506448 / 323.183294; 39.196595%; 1.644645× | 319.366942 / 618.184915; 48.337959%; 1.935657× | 45.026568 / 60.843572; 25.996179%; 1.351282× | 6433 / 10516; 38.826550%; 1.634696× |
| PPI | 1546 | 17699 | 8 | 1 | 6564.653727 / 11730.980944; 44.040027%; 1.786992× | 43021.798101 / 82348.166642; 47.756216%; 1.914103× | 1209.115697 / 2462.848683; 50.905807%; 2.036901× | 6428 / 10511; 38.845020%; 1.635190× |
| PPI | 2923 | 33928 | 8 | 2 | 6684.225177 / 11498.205810; 41.867233%; 1.720200× | 44804.879436 / 80414.596513; 44.282653%; 1.794773× | 1370.420488 / 2613.936092; 47.572533%; 1.907397× | 6428 / 10511; 38.845020%; 1.635190× |

PPI 选择规则固定为原始组件 ID 升序，选择 N≥3 且至少一个节点有两个不同非自环入邻居的前两个完整组件。组件 0 的原始节点边界为 [0,1546)，N/E=1546/17699，非自环入度 min/median/max 为 0/5/268；组件 9 的边界为 [1554,2931)，N/E=1377/16229，入度为 0/5/226。G=2 时 node_ptr=[0,1546,2923]、edge_ptr=[0,17699,33928]。两组件间边界保留，未填充孤立节点或截断边。

计时含义和完整逐次结果见 [docs/mechanism_ablation.md](docs/mechanism_ablation.md)。结果目录为 <code>runs/mechanism_ablation_real/mechanism_20261010T025528Z_953a5b0e1617/</code>；其中 <code>cells.csv</code>、<code>summary.json</code>、<code>results_manifest.json</code>、两项机制各自的 Markdown/CSV 表格、独立验证收据及证明工件记录逐项来源和哈希。全量 CPU 独立复核归档在 <code>audit_reverify.json</code>，该检查 22/22 通过且 <code>reverified=true</code>。首次发布完整性流程也已成功：<code>record-publication-docs</code> 返回 0 并写出 <code>publication_document_receipt.json</code>；随后不带 <code>--reverify</code> 的严格 <code>audit</code> 返回 0、报告 <code>RESULT_AUDIT_PASS cells=22 reverified=false</code>；<code>summarize</code> 返回 0 并重生成两份论文表格。确切参数见下方命令。此处的 <code>reverified=false</code> 表示发布审计只检查工件与文档身份，完整的 22 项 CPU 重新验证证据仍在 <code>audit_reverify.json</code>。本次文档补充后，session owner 重新运行代码块中的三条出版收据、严格审计和汇总命令，使新收据绑定当前 README 与实验文档，然后生成 <code>publication_integrity.json</code>。

本轮成功使用的构建、正式启动、全量独立复核和结果生成命令如下。工作目录为 <code>/home/pzh/project</code>。正式启动使用独立后台 launcher，CUDA 设备为 0，每组运行一次且不设证明超时：

~~~bash
cd /home/pzh/project
cmake -S . -B build_cuda -DCMAKE_BUILD_TYPE=Release -DGATZK_ENABLE_CUDA_BACKEND=ON -DCMAKE_CUDA_ARCHITECTURES=native -DGATZK_ENABLE_NATIVE_ARCH=ON -DGATZK_BUILD_CORE_EXECUTION_REGRESSION=ON -DGATZK_BUILD_MECHANISM_SECURITY_REGRESSION=ON -DGATZK_BUILD_REAL_MECHANISM_INPUT_REGRESSION=ON -DGATZK_BUILD_DEFAULT_PROOF_REGRESSION=OFF
cmake --build build_cuda --parallel 4 --target gatzk_mechanism gatzk_core_execution_regression gatzk_mechanism_security_regression gatzk_real_mechanism_input_regression
python3 scripts/run_mechanism_ablation.py launch --binary /home/pzh/project/build_cuda/gatzk_mechanism --device 0 --repetitions 1 --timeout-seconds 0 --prepare-timeout-seconds 0
python3 -u scripts/run_mechanism_ablation.py audit /home/pzh/project/runs/mechanism_ablation_real/mechanism_20261010T025528Z_953a5b0e1617 --reverify
python3 -u scripts/run_mechanism_ablation.py summarize /home/pzh/project/runs/mechanism_ablation_real/mechanism_20261010T025528Z_953a5b0e1617
python3 -u scripts/run_mechanism_ablation.py record-publication-docs /home/pzh/project/runs/mechanism_ablation_real/mechanism_20261010T025528Z_953a5b0e1617
python3 -u scripts/run_mechanism_ablation.py audit /home/pzh/project/runs/mechanism_ablation_real/mechanism_20261010T025528Z_953a5b0e1617
python3 -u scripts/run_mechanism_ablation.py summarize /home/pzh/project/runs/mechanism_ablation_real/mechanism_20261010T025528Z_953a5b0e1617
~~~

正式实验证明从 2026-10-10 10:55:28 到 11:36:33（Asia/Shanghai），墙钟时间 41 分 05 秒。构建环境为 CMake 3.28.3、G++ 13.3.0、CUDA 13.0；设备为 NVIDIA GB10、UUID <code>GPU-3cde57ba-ae43-4452-bd59-7b156b36db71</code>、driver 580.142。22 份证明都实际调用 CUDA NTT、INTT 和 MSM；独立验证器在 CPU 运行。

首次后台尝试在 Cora Feature Folding D=1 的两份独立证明通过后，于重复输入稳定性检查抛出 <code>TypeError: unhashable type: 'dict'</code>：比较逻辑把含嵌套 <code>security_policy</code> 字典的 tuple 放入 set。修复后以规范 JSON 形式比较该策略，并增加显式 <code>--repetitions</code> 参数及单次运行的完整计数、审计和汇总逻辑。旧尝试 session <code>mechanism_20261010T015032Z_16f5013c1802</code> 的工件保留但未混入本轮；本轮采用新源码身份重新完成 22 份证明。

这些收益只适用于两项机制的独立证明成本和 proof size；它们不表示完整 GAT 前向或端到端零知识证明的加速幅度。每组只有一次测量，差异是本轮配对实测值，不具有重复抽样的统计结论。
