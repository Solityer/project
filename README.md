# zkGAT

**Zero-Knowledge Verifiable Inference for Graph Attention Networks**

zkGAT 对图注意力网络（Graph Attention Network，GAT）的推理计算生成零知识证明，并独立验证证明。图由节点和节点之间的边组成；GAT 根据节点及其邻居的信息计算节点表示和预测结果。本项目面向训练完成后的 GAT 推理，训练脚本用于准备模型参数。

## 主要功能与整体流程

- 支持 Cora、CiteSeer、PubMed、PPI 和 OGBN-Arxiv 数据集。
- 支持 CUDA 加速计算。
- Feature Folding 将多个特征坐标合并到一条证明路径中处理；Shared Neighborhood 在共享的证明路径中处理邻居归一化与聚合约束。
- 生成推理证明、独立验证，并导出证明文件和运行清单。

运行流程：准备数据和模型参数，以及 SRS（结构化参考字符串，证明系统使用的参数文件）→ 执行 GAT 推理 → 生成证明 → 验证证明 → 保存输出。

## 环境要求与构建

构建 CUDA 证明计算路径需要 Linux、NVIDIA GPU、兼容的 CUDA 工具包、C++20 编译器、CMake 3.24 或更高版本、GNU Make、Python 3、OpenSSL 开发文件，以及构建 MCL 组件所需的 LLVM 工具链。CUDA 构建使用固定版本的 ICICLE 子模块。

在仓库根目录初始化 ICICLE 子模块并构建：

```bash
git submodule update --init third_party/icicle
cmake -S . -B build_cuda -DCMAKE_BUILD_TYPE=Release -DGATZK_ENABLE_CUDA_BACKEND=ON
cmake --build build_cuda --parallel 2 --target gatzk_run gatzk_mechanism gatzk_srs_setup gatzk_registry_register
```

## 数据与模型准备

仓库包含五个数据集的原始数据和规范缓存。需要重新生成缓存时，在仓库根目录运行对应命令：

数据准备脚本需要 NumPy；Planetoid 缓存准备还需要 SciPy。训练脚本使用 PyTorch，按本机 Python 和 CUDA 环境安装相应版本。

```bash
python3 scripts/prepare_planetoid.py --data-root data --dataset cora --cache-root data/cache
python3 scripts/prepare_planetoid.py --data-root data --dataset citeseer --cache-root data/cache
python3 scripts/prepare_planetoid.py --data-root data --dataset pubmed --cache-root data/cache
python3 scripts/prepare_ppi.py --data-root data/ppi --output-root data/cache/ppi
python3 scripts/prepare_ogbn_arxiv.py --project-root . --data-root data
```

每个数据集的配置文件指定模型参数包和 SRS 的位置。模型参数包由训练 checkpoint 导出并验证；例如，准备 Cora 参数包：

```bash
python3 scripts/train_planetoid_gat.py --config configs/cora_secure.cfg
python3 scripts/export_torch_gat_bundle.py --checkpoint artifacts/checkpoints/cora_gat/best_model.pt --output-dir artifacts/checkpoints/cora_gat
python3 scripts/check_checkpoint_bundle.py --bundle-dir artifacts/checkpoints/cora_gat --dataset cora
```

以下命令为当前 Cora 配置生成 SRS：

```bash
mkdir -p artifacts/srs
build_cuda/gatzk_srs_setup compile artifacts/srs/cora_protocol_compiler_artifact.txt 16777215 configs/srs_cora_max_envelope.cfg C=16 FH=4194304 N=4096 cat=128 d=16 edge=32768 in=2048
build_cuda/gatzk_srs_setup ceremony artifacts/srs/cora_formal_srs artifacts/srs/cora_protocol_compiler_artifact.txt cora_formal_srs
build_cuda/gatzk_srs_setup validate artifacts/srs/cora_formal_srs cora_formal_srs
```

注册 Cora 验证参数，并将命令输出的 `candidate_registry_digest` 填入 `configs/cora_secure.cfg` 的 `verifier_registry_digest` 项：

```bash
build_cuda/gatzk_registry_register --config configs/cora_secure.cfg
```

其他数据集使用对应的训练脚本和配置。模型参数包保存于 `artifacts/checkpoints/<dataset>_gat/`，SRS 保存于 `artifacts/srs/<dataset>_formal_srs/`；SRS 配置位于 `configs/srs_<dataset>_max_envelope.cfg`。

## 运行方法

准备好 Cora 数据缓存、模型参数包和 SRS，并完成验证器注册与摘要回填后，在仓库根目录运行单个推理证明：

```bash
build_cuda/gatzk_run cora
```

程序显示 `status=VERIFY_OK` 表示该次证明通过验证。证明文件和运行清单保存在 `runs/cora_secure/<自动生成的tag>/single/` 下，文件名分别为 `proof.bin` 和 `run_manifest.json`。

运行五个数据集的主流程，或只运行指定数据集：

```bash
bash scripts/run_paper_matrix.sh
bash scripts/run_paper_matrix.sh ogbn_arxiv
```

运行 Feature Folding 和 Shared Neighborhood 的真实数据 CUDA 消融：

```bash
RUN_ID="$(date +%Y%m%dT%H%M%S)"
python3 scripts/run_mechanism_ablation.py launch --device 0 --repetitions 1 \
  --mechanism all --input-scope all \
  --output-root "results/mechanism_ablation_final/all_47rep1_${RUN_ID}"
```

`--device` 指定 GPU 序号；`--repetitions` 指定每组配置的重复次数。

## 项目结构与代码阅读

```text
src/
├── data/        数据加载
├── model/       GAT 计算
├── mechanism/   Feature Folding 和 Shared Neighborhood
└── protocol/    trace、证明、验证和产物导出
configs/         数据集、证明和 SRS 配置
scripts/         数据准备、模型参数包和运行脚本
include/         C++ 接口
```

按一次推理的执行顺序阅读：

1. [src/main.cpp](src/main.cpp)：从 `main` 查看命令行入口和各阶段的组织。
2. [src/data/loader.cpp](src/data/loader.cpp)：从 `load_dataset` 查看数据集加载。
3. [src/model/gat.cpp](src/model/gat.cpp)：从 `forward_note_style` 查看 GAT 前向计算。
4. [src/protocol/trace.cpp](src/protocol/trace.cpp)：从 `build_trace` 查看证明所需计算轨迹（trace）的构建。
5. [src/protocol/prover.cpp](src/protocol/prover.cpp)：从 `prove` 查看证明生成。
6. [src/protocol/verifier.cpp](src/protocol/verifier.cpp)：从 `verify` 查看独立证明验证。
7. [src/protocol/artifact_export.cpp](src/protocol/artifact_export.cpp)：从 `export_run_artifacts` 查看证明和运行产物的导出。

Feature Folding 和 Shared Neighborhood 的实现分别位于 [src/mechanism/feature_folding.cpp](src/mechanism/feature_folding.cpp) 和 [src/mechanism/shared_neighborhood.cpp](src/mechanism/shared_neighborhood.cpp)。
