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
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGATZK_ENABLE_CUDA_BACKEND=OFF
cmake --build build -j10
```

CUDA Release：

```bash
cmake -S . -B build_cuda -DCMAKE_BUILD_TYPE=Release -DGATZK_ENABLE_CUDA_BACKEND=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build_cuda -j10
```

短时 schema/static 检查（不加载正式数据、不生成 checkpoint/SRS/run）：

```bash
cmake --build build -j10
cmake --build build_cuda -j10
build/gatzk_layered_schema_test
build_cuda/gatzk_layered_schema_test
ctest --test-dir build --output-on-failure
ctest --test-dir build_cuda --output-on-failure
python3 tests/test_layered_schema.py
python3 tools/security/static_scan_no_tau.py
git diff --check -- README.md include src tests
```

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

三个 Planetoid 数据集可以在完整 proof 前做 checkpoint 与导出 bundle 的 CPU forward trace
parity。输出是临时文件，检查后删除，不是正式性能结果：

```bash
parity_cora=$(mktemp /tmp/zkgat-parity-cora.XXXXXX.json)
.venv/bin/python scripts/check_planetoid_parity.py --checkpoint artifacts/checkpoints/cora_gat/best_model.pt --bundle-dir artifacts/checkpoints/cora_gat --data-root data --dataset cora --output "$parity_cora"
jq -e '.passed == true and .comparison_count > 0' "$parity_cora"
unlink "$parity_cora"

parity_citeseer=$(mktemp /tmp/zkgat-parity-citeseer.XXXXXX.json)
.venv/bin/python scripts/check_planetoid_parity.py --checkpoint artifacts/checkpoints/citeseer_gat/best_model.pt --bundle-dir artifacts/checkpoints/citeseer_gat --data-root data --dataset citeseer --output "$parity_citeseer"
jq -e '.passed == true and .comparison_count > 0' "$parity_citeseer"
unlink "$parity_citeseer"

parity_pubmed=$(mktemp /tmp/zkgat-parity-pubmed.XXXXXX.json)
.venv/bin/python scripts/check_planetoid_parity.py --checkpoint artifacts/checkpoints/pubmed_gat/best_model.pt --bundle-dir artifacts/checkpoints/pubmed_gat --data-root data --dataset pubmed --output "$parity_pubmed"
jq -e '.passed == true and .comparison_count > 0' "$parity_pubmed"
unlink "$parity_pubmed"
```

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

十组正式 run 全部通过上述检查后，生成 CPU 与 CUDA 两份结果目录：

```bash
.venv/bin/python scripts/export_benchmark_table.py --output-dir results/paper_secure/cpu --run cora=runs/cora_secure/cpu_formal/warm --run citeseer=runs/citeseer_secure/cpu_formal/warm --run pubmed=runs/pubmed_secure/cpu_formal/warm --run ppi=runs/ppi_secure/cpu_formal/warm --run ogbn_arxiv=runs/ogbn_arxiv_secure/cpu_formal/warm
.venv/bin/python scripts/export_benchmark_table.py --output-dir results/paper_secure/cuda --run cora=runs/cora_secure/cuda_formal/warm --run citeseer=runs/citeseer_secure/cuda_formal/warm --run pubmed=runs/pubmed_secure/cuda_formal/warm --run ppi=runs/ppi_secure/cuda_formal/warm --run ogbn_arxiv=runs/ogbn_arxiv_secure/cuda_formal/warm
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
.venv/bin/python scripts/aggregate_secure_results.py --output-dir results/paper_secure --cpu-run cora=runs/cora_secure/cpu_formal/warm --cpu-run citeseer=runs/citeseer_secure/cpu_formal/warm --cpu-run pubmed=runs/pubmed_secure/cpu_formal/warm --cpu-run ppi=runs/ppi_secure/cpu_formal/warm --cpu-run ogbn_arxiv=runs/ogbn_arxiv_secure/cpu_formal/warm --cuda-run cora=runs/cora_secure/cuda_formal/warm --cuda-run citeseer=runs/citeseer_secure/cuda_formal/warm --cuda-run pubmed=runs/pubmed_secure/cuda_formal/warm --cuda-run ppi=runs/ppi_secure/cuda_formal/warm --cuda-run ogbn_arxiv=runs/ogbn_arxiv_secure/cuda_formal/warm
jq -e '.run_count == 10 and (.rows | length) == 10 and .security_tests.all_passed == true' results/paper_secure/latest.json
test -s results/paper_secure/security_tests_report.md
test "$(awk 'END {print NR-1}' results/paper_secure/latest.csv)" -eq 10
```

上述聚合命令成功后，`results/paper_secure/summary.md`、`latest.csv`、`latest.json` 和
`security_tests_report.md` 才是十组正式结果的根级交付面；本轮没有执行正式训练、SRS
ceremony、完整 proof/verify 或十组实验，因此当前不生成这些文件。

当前 README 只描述这套 canonical 状态；正式训练、checkpoint 生成、SRS ceremony、完整
proof/verify 和十组最终结果在执行完成前均不视为已完成。
