# zkGAT 真实图机制级消融

本文件记录真实图机制级消融的最终 22 份 CUDA 证明和单次配对测量。本实验隔离比较 Feature Folding 与 Shared Neighborhood 两项证明机制，不执行完整 GAT 零知识证明。结果是机制级证明开销，不可解释为完整 GAT 端到端性能收益。

## 固定协议和数据配置

机制输入来自正式 GAT 第一隐藏层第 0 个注意力头，正式头宽度为 8。上游只执行生成这两项机制输入所需的正式数据加载、checkpoint 前向与量化；不执行完整 GAT 证明。

Feature Folding 的 Original 使用随机折叠证明，Control 使用逐坐标证明；两侧针对同一真实 Node-to-Edge 关系。D=1、2、4 使用同一个真实 8 维 H′ 的前 D 个坐标，是受控机制宽度实验，不代表正式模型的完整单头计算。

Shared Neighborhood 的输入是该注意力头真实有限域中的 U、Sum、归一化权重 alpha、加权压缩边特征 V=alpha×H′[src]，以及聚合前目标值 Hagg_pre。Hagg_pre 位于后续 signed rounding 和激活之前。Original 使用共享 PSQ/邻域证明，Control 分别证明 denominator 与 weighted aggregate。输入准备时间不计入四项机制指标。输入、checkpoint 和源码哈希用于审计见证来源和本轮一致性，不声称它们本身证明了完整 GAT 前向关系。

Cora 和 Citeseer 使用完整正式预处理图及其边序。PPI 按原始 canonical component ID 升序，选择满足 N≥3 且至少有一个节点具备两个不同非自环入邻居的前两个完整组件。只在所选完整连通组件之间建立图边界，没有截断真实边或添加填充节点。

| 机制 | 数据集 | 实际 N | 实际 E | D | G | PPI 原始组件 ID |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Feature Folding | Cora | 2708 | 13264 | 1 | 1 | — |
| Feature Folding | Cora | 2708 | 13264 | 2 | 1 | — |
| Feature Folding | Cora | 2708 | 13264 | 4 | 1 | — |
| Feature Folding | Cora | 2708 | 13264 | 8 | 1 | — |
| Feature Folding | Citeseer | 3327 | 12431 | 8 | 1 | — |
| Feature Folding | PPI | 1546 | 17699 | 8 | 1 | 0 |
| Feature Folding | PPI | 2923 | 33928 | 8 | 2 | 0, 9 |
| Shared Neighborhood | Cora | 2708 | 13264 | 8 | 1 | — |
| Shared Neighborhood | Citeseer | 3327 | 12431 | 8 | 1 | — |
| Shared Neighborhood | PPI | 1546 | 17699 | 8 | 1 | 0 |
| Shared Neighborhood | PPI | 2923 | 33928 | 8 | 2 | 0, 9 |

PPI 的组件 0 原始节点边界为 [0,1546)，N=1546、E=17699，非自环入度 min/median/max 为 0/5/268；组件 9 原始边界为 [1554,2931)，N=1377、E=16229，非自环入度为 0/5/226。G=2 的本地边界为 node_ptr=[0,1546,2923]、edge_ptr=[0,17699,33928]。PPI 源数据是 data/cache/ppi 及 data/ppi；Planetoid 源数据是 data/cache/cora、data/cache/citeseer 及对应 data 原始文件。输入清单、各数据文件哈希、图组件身份和 prepared bundle 哈希记录在本 session 的 results_manifest.json、configuration_and_srs_manifest.json、summary.json 和 real_inputs 子目录。

使用的正式配置与 checkpoint 为 configs/cora_secure.cfg / artifacts/checkpoints/cora_gat、configs/citeseer_secure.cfg / artifacts/checkpoints/citeseer_gat、configs/ppi_secure.cfg / artifacts/checkpoints/ppi_gat。正式模型 checkpoint bundle 身份如下：

| 数据集 | config SHA-256 | checkpoint bundle SHA-256 | trusted SRS | SRS manifest SHA-256 | 注册 edge domain |
| --- | --- | --- | --- | --- | ---: |
| Cora | d9502db858a2691d58eda5c296785e9fc4e8d442f8c2ba870257d05e4081093a | f81b9c007cbd1b55cdd8914d0f09580bfca0b2558f73aa14631d1eba61d424cd | cora_formal_srs | 0734f359deac2c145bab22ed3ed40de7310eda95e732de9aa45f66919108fcf3 | 32768 |
| Citeseer | 50be73f5569a1e18c743f1b1b5c6fb2730750270cbe4b00b40099037c644235e | c92f42267f8affee1a6888e607e893b88cb26f6c460ea6028f57765319a25f63 | citeseer_formal_srs | b9ebe4d4cba8c333c83ab1d31f2c9b328199a2e1dd20153647631a73a7b4839f | 32768 |
| PPI | ca0d4f29915c258d5a032000835f68025f9b343094b31cd2fd3522448d92d8d2 | 65a226dbdb4999b944241cfc511a577991905a60c1e224bda5d37d9ab397a5df | ppi_formal_srs | 859f59973e1c917303adc7015eac2472250b58cd5508b59dcba62b00b2419e34 | 8388608 |

The experiment used the existing registered trusted SRS and its full domains. No smaller SRS or weaker security setting was substituted.

| Identity | SHA-256 |
| --- | --- |
| Source tree inventory | 0ba5e67caf5313daaef3aafac719df86bd32aa8af0e2f4d7edf6790697c617d1 |
| Source diff | 92288783ac2c9e3635c9ca42e8ef122fb071b539c7d590436c00f59904763b27 |
| CUDA executable | 0ffb246a418f56b792dec47da8215a09fd8b119bed18f7ff21339619cabc65f9 |
| CMake cache | 58cd7db89f92e67345c77389a08ffbd6d7d77e4c9369a001a67c69222710884a |
| Build source manifest | c6d67776eb122cd6c939545c5627621b9ad1bafe003138cd63333e9522e93ce1 |

## Measurement protocol

Each of the 11 configurations ran once as a paired Original/Control measurement, for 11 pairs and 22 CUDA proofs. Pair order alternated to balance which variant ran first. No sample standard deviation, confidence interval or significance test is reported for one measurement per configuration. The independent verifier ran on CPU in a separate process.

The four metrics use disjoint boundaries:

- Commitment Time (ms): cold native KZG commitment, including first-use static-base preparation invoked inside commitment.
- Prove Time (ms): online mechanism proving, Taylor-shift coefficient materialization and canonical proof encoding; excludes commitment and artifact file I/O.
- Verify Time (ms): independent CPU verifier parsing and proof decoding through acceptance; SRS and artifact reads are outside the timer.
- Proof Size (Bytes): actual canonical serialized mechanism proof length.

Upstream real-input preparation is recorded separately and is not included in these measurements. Each formal CUDA proof recorded actual CUDA NTT, INTT and MSM work. The verifier returned independent VERIFY_OK for all 22 cells.

For each metric, O is Original and C is Control. Saving (%) = (C − O) / C × 100%; Improvement Factor = C / O. Negative Saving means Original was slower or larger in this single measured pair. Tables display direct measured values and the derived paired ratios; CSV also contains the exact Original − Control difference.

## 正式结果：Feature Folding

All times are milliseconds and proof sizes are bytes. Each metric cell shows Original / Control; Saving; Factor.

| Dataset | N | E | D | G | Component IDs | Commitment ms: O/C; Saving; Factor | Prove ms: O/C; Saving; Factor | Verify ms: O/C; Saving; Factor | Proof bytes: O/C; Saving; Factor |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| Cora | 2708 | 13264 | 1 | 1 | — | 124.879424 / 122.921320; -1.592973%; 0.984320× | 218.130203 / 212.301324; -2.745569%; 0.973278× | 35.973663 / 33.691886; -6.772482%; 0.936571× | 3848 / 3743; -2.805236%; 0.972713× |
| Cora | 2708 | 13264 | 2 | 1 | — | 124.271522 / 227.543787; 45.385667%; 1.831021× | 217.102000 / 393.137180; 44.777037%; 1.810841× | 36.509525 / 50.775638; 28.096374%; 1.390750× | 3982 / 7105; 43.954961%; 1.784279× |
| Cora | 2708 | 13264 | 4 | 1 | — | 128.251546 / 423.889607; 69.744116%; 3.305142× | 215.731652 / 750.205907; 71.243675%; 3.477496× | 36.667271 / 79.976278; 54.152316%; 2.181135× | 4250 / 13829; 69.267481%; 3.253882× |
| Cora | 2708 | 13264 | 8 | 1 | — | 130.864436 / 813.990798; 83.923106%; 6.220107× | 220.331356 / 1468.470055; 84.995856%; 6.664826× | 37.459600 / 137.177366; 72.692580%; 3.662008× | 4786 / 27277; 82.454082%; 5.699331× |
| Citeseer | 3327 | 12431 | 8 | 1 | — | 135.445006 / 819.076726; 83.463697%; 6.047301× | 217.005365 / 1465.546957; 85.192875%; 6.753506× | 35.591890 / 123.092559; 71.085263%; 3.458444× | 4790 / 27281; 82.441993%; 5.695407× |
| PPI | 1546 | 17699 | 8 | 1 | 0 | 4325.626412 / 25648.274180; 83.134825%; 5.929378× | 23062.817338 / 155731.693336; 85.190672%; 6.752501× | 581.435836 / 4774.692892; 87.822550%; 8.211900× | 4785 / 27276; 82.457105%; 5.700313× |
| PPI | 2923 | 33928 | 8 | 2 | 0, 9 | 4272.736449 / 25611.597633; 83.317181%; 5.994191× | 22468.499785 / 154619.331276; 85.468505%; 6.881605× | 635.159136 / 4739.317609; 86.598089%; 7.461622× | 4785 / 27276; 82.457105%; 5.700313× |

## 正式结果：Shared Neighborhood

| Dataset | N | E | D | G | Component IDs | Commitment ms: O/C; Saving; Factor | Prove ms: O/C; Saving; Factor | Verify ms: O/C; Saving; Factor | Proof bytes: O/C; Saving; Factor |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| Cora | 2708 | 13264 | 8 | 1 | — | 197.048008 / 320.234547; 38.467598%; 1.625160× | 334.916930 / 620.896177; 46.059109%; 1.853881× | 41.350309 / 69.335738; 40.362200%; 1.676789× | 6429 / 10512; 38.841324%; 1.635091× |
| Citeseer | 3327 | 12431 | 8 | 1 | — | 196.506448 / 323.183294; 39.196595%; 1.644645× | 319.366942 / 618.184915; 48.337959%; 1.935657× | 45.026568 / 60.843572; 25.996179%; 1.351282× | 6433 / 10516; 38.826550%; 1.634696× |
| PPI | 1546 | 17699 | 8 | 1 | 0 | 6564.653727 / 11730.980944; 44.040027%; 1.786992× | 43021.798101 / 82348.166642; 47.756216%; 1.914103× | 1209.115697 / 2462.848683; 50.905807%; 2.036901× | 6428 / 10511; 38.845020%; 1.635190× |
| PPI | 2923 | 33928 | 8 | 2 | 0, 9 | 6684.225177 / 11498.205810; 41.867233%; 1.720200× | 44804.879436 / 80414.596513; 44.282653%; 1.794773× | 1370.420488 / 2613.936092; 47.572533%; 1.907397× | 6428 / 10511; 38.845020%; 1.635190× |

Feature Folding D=1 的 Cora 单次测量中，Original 四项指标都略低于 Control，因此 Saving 为负；D=2、4、8 及 Citeseer/PPI D=8 的各项指标则由 Original 获得较低开销。Shared Neighborhood 的四个配置中，Original 的四项指标都低于 Control。这些观察仅描述本轮独立机制证明的单次配对结果。

## 执行、验证与结果工件

正式 session 为 <code>mechanism_20261010T025528Z_953a5b0e1617</code>，绝对路径为：

<code>/home/pzh/project/runs/mechanism_ablation_real/mechanism_20261010T025528Z_953a5b0e1617/</code>

正式运行从 2026-10-10 02:55:28 UTC 至 03:36:33 UTC，即 Asia/Shanghai 10:55:28 至 11:36:33，墙钟时间 41 分 05 秒。执行设备为 NVIDIA GB10，UUID <code>GPU-3cde57ba-ae43-4452-bd59-7b156b36db71</code>，driver 580.142，CUDA runtime 13.0；构建工具为 CMake 3.28.3、G++ 13.3.0、nvcc 13.0 V13.0.88。

22 项原始证明均在各自单元独立验证成功。随后执行全量 <code>audit --reverify</code>，CPU 独立重验 22/22 全部通过，命令退出码为 0，输出 <code>RESULT_AUDIT_PASS</code> 且 <code>reverified=true</code>。该次审计运行 895.85 秒（14 分 55.85 秒）。经原样归档并逐字节比对，<code>audit_reverify.json</code> 与该次全量 <code>audit.json</code> 一致，SHA-256 为 <code>ee37a5dd10ad53f3cd3d626a96f788ace0637ecdf965c7abe1da41aba2fc83dc</code>。结果生成命令随后退出码为 0，输出 <code>PAPER_TABLES_REGENERATED</code>。summary.json SHA-256 为 <code>f9f4af2164502338e0d1574c15409a15ec5a094cdb045168e90d35995643f0b1</code>。

session 中的重要工件：

- <code>cells.csv</code>：22 项逐次 Original/Control 结果及 CUDA、证明、声明、收据和文件哈希。
- <code>summary.json</code>：本轮直接测量、配对统计和输入身份。
- <code>feature_folding_summary.md</code> / <code>feature_folding_summary.csv</code>：Feature Folding 论文表格。
- <code>shared_neighborhood_summary.md</code> / <code>shared_neighborhood_summary.csv</code>：Shared Neighborhood 论文表格。
- <code>results_manifest.json</code> / <code>configuration_and_srs_manifest.json</code>：数据、配置、checkpoint、SRS、源码、构建和 CUDA 身份。
- <code>audit_reverify.json</code> / <code>audit_reverify.log</code> / <code>audit_reverify.time</code>：归档的全量 CPU 独立复核、日志和耗时。
- 每个 <code>mechanism/&lt;config&gt;/rep_001/&lt;original|control&gt;/</code>：规范序列化 proof、公开声明、prover/verifier 日志与机器可读收据。
- <code>session.log</code> 和 <code>/home/pzh/project/runs/mechanism_ablation_real/launchers/cuda_20261010T025528Z_e608af94fb/</code>：正式后台运行编排日志及 launcher 身份。

构建与正式运行成功命令如下，均从项目根目录执行：

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

启动修复记录：旧 launcher <code>cuda_20261010T015032Z_79d4a2b8c0</code> 对应的尝试 session <code>mechanism_20261010T015032Z_16f5013c1802</code> 已生成 Cora FF D=1 Original/Control 两份独立 <code>VERIFY_OK</code> 证明，但重复输入稳定性检查把包含嵌套 security_policy 字典的 tuple 放入 set，导致 <code>TypeError: unhashable type: 'dict'</code>。修复将安全策略转为规范 JSON 后再比较；同时为正式单次测量加入显式 repetitions 配置，并修正单次场景的配对数量、完成条件、审计和结果汇总。旧 session 工件保留但不复用，因为本轮源码/实验身份已经变化；本轮 22 份证明均由同一新身份新鲜执行。执行器将 README 与本实验文档作为单独出版哈希记录，其余执行源码仍纳入来源身份锁定。

发布完整性链第一轮已实际通过：<code>record-publication-docs</code> 返回 0，生成 <code>publication_document_receipt.json</code>；随后不带 <code>--reverify</code> 的严格 <code>audit</code> 返回 0，报告 <code>RESULT_AUDIT_PASS cells=22 reverified=false</code>；接着 <code>summarize</code> 返回 0 并输出 <code>PAPER_TABLES_REGENERATED</code>。相应的完整命令见上方成功命令块。本次补充前的 receipt 绑定的是前一版 README 和本文件，因此文档哈希更新后 session owner 会重新执行这三条出版命令，并生成绑定最终文档哈希的 <code>publication_integrity.json</code>。

两类审计的职责不同：<code>audit_reverify.json</code> 归档了独立 CPU verifier 对 22 份证明逐份重新验证的结果，22/22 通过且 <code>reverified=true</code>；出版收据之后执行的最终严格 <code>audit</code> 检查 session 工件、来源身份及当前文档哈希，不再重新运行 22 个 verifier，因此其 <code>audit.json</code> 显示 <code>reverified=false</code>。应将前者作为完整密码学重验的证据，后者及 <code>publication_document_receipt.json</code> 由 <code>publication_integrity.json</code> 绑定，用于最终发布完整性核对。

本实验的历史合成机制测试记录不进入上述表格和任何正式统计。
