# zkGAT 真实图机制级消融：initial-verification-only 结果

## 状态与范围

正式 session 为 `mechanism_20261010T063825Z_9fe0a93c236b`，位于：

`/home/pzh/project/results/mechanism_ablation_final/mechanism_20261010T063825Z_9fe0a93c236b/`

本次运行覆盖 11 组固定配置、33 对 Original/Control、66 份真实 CUDA 机制证明，每组重复 3 次。运行中的初始独立验证收据为 66/66 `VERIFY_OK`，完整配对为 33/33。正式证明运行从 2026-10-10T06:38:25Z 到 2026-10-10T08:48:32Z，墙钟时间 7807 秒（2 小时 10 分 07 秒）。本文件的性能表来自 session 中的 `report_initial_only.py` 输出，并明确属于 **initial-verification-only** 报告。

全量运行后独立重验由用户取消。重验在 13/66 个 verifier rerun 成功后，正在执行的第 14 项（Feature Folding, PPI components 0+9, D=8, G=2, repetition 1, Control）收到 SIGTERM；审计命令以 exit 2 结束，用时 387.596137 秒。事件及部分收据保存在：

`/home/pzh/project/results/mechanism_ablation_final/incidents/reverify_cancelled_20261010T085441Z_c57bc5f351/`

取消记录确认该次过程不是完整重验，且不表示底层证明无效。没有提交有效的全量 `audit_reverify.json` PASS 归档。之后普通 `audit` 返回 exit 0、`RESULT_AUDIT_PASS cells=66 reverified=false`，耗时约 5.12 秒；它只验证现有收据、文件哈希、清单和身份，没有重新运行 66 个 verifier。strict runner publication receipt 未创建，`publication_integrity.json` 没有 PASS。本报告完整性文件 `initial_only_report_integrity.json` 记录的是初验收据和哈希审计范围，不替代全量密码学重验或 strict publication gate。

## 机制输入、图数据和配置

上游中间值来自正式 GAT 第一隐藏层（layer 0）第 0 个注意力头（head 0），头宽度为 8。只执行生成机制输入所需的数据读取、正式 checkpoint 前向和量化准备，没有执行完整 GAT 零知识证明。输入哈希、checkpoint 和 source provenance 用于追溯真实见证及配对一致性；这些元数据本身不证明完整 GAT 前向关系。

Feature Folding 的 Original 使用随机折叠证明，Control 使用逐坐标证明，二者验证同一个真实 Node-to-Edge 关系。D=1、2、4 取同一真实 8 维 H′ 的前 D 个坐标，是受控机制宽度实验，不是正式模型完整单头计算。Shared Neighborhood 使用该头真实有限域的 U、Sum、归一化权重 α、加权压缩边特征 V=α×H′[src] 和聚合前目标值 Hagg_pre；Hagg_pre 位于后续激活与 signed rounding / 输出舍入之前。Original 使用共享 PSQ/邻域证明，Control 分别证明 denominator 和 weighted aggregate。所有配对共用真实图、实际见证、选定坐标和安全参数。

Cora、Citeseer 使用正式预处理的完整真实图和边序。PPI 在原始 canonical component ID 升序中选择满足 N≥3 且至少一个节点有两个不同非自环入邻居的前两个完整组件。组件 0 原始节点边界为 [0,1546)，N/E=1546/17699，非自环入度 min/median/max=0/5/268；组件 9 边界为 [1554,2931)，N/E=1377/16229，入度为 0/5/226。G=2 时 node_ptr=[0,1546,2923]，edge_ptr=[0,17699,33928]。未添加孤立填充节点、生成拓扑或截断真实边。

| 机制 | 数据集 | N | E | D | G | PPI 原始组件 |
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

图数据来自项目正式原始数据和预处理缓存：Cora、Citeseer 的 `data/cache/cora`, `data/cache/citeseer` 及对应源数据；PPI 的 `data/cache/ppi` 与 `data/ppi`。各输入文件哈希、正式图元数据和 bundle SHA-256 在 session 的 `results_manifest.json`、`summary.json` 及 `real_inputs/` 中。

本次运行实际使用的预处理输入 bundle：Cora full graph SHA-256 `dc20bd6e6d1b420dd039af65875f25e3814c088aa5878be0521ca09bc0369c8d`；Citeseer full graph SHA-256 `9770dc035af209c0076d73367fdcd0117c0d59a02bbf08cd738da0b3e524af04`；PPI component 0 SHA-256 `8206574e15bb6fbd7efc1b985df5d1747a54f9cf18a31d4f859a5d16d181f059`；PPI components 0+9 SHA-256 `7d9e54854960deb75a02c07db3fcad2e67ae5ece063403c220ba3dcc684f7fd4`。

checkpoint 和 SRS provenance 如下。checkpoint 目录分别为 `artifacts/checkpoints/cora_gat`, `artifacts/checkpoints/citeseer_gat`, `artifacts/checkpoints/ppi_gat`；正式配置分别为 `configs/cora_secure.cfg`, `configs/citeseer_secure.cfg`, `configs/ppi_secure.cfg`。使用现有注册的可信 SRS，安全目标保持 120-bit，没有缩小 SRS 或降低安全参数。

| Dataset | Config SHA-256 | Formal checkpoint bundle SHA-256 | Trusted SRS ID | SRS manifest SHA-256 |
| --- | --- | --- | --- | --- |
| Cora | `d9502db858a2691d58eda5c296785e9fc4e8d442f8c2ba870257d05e4081093a` | `f81b9c007cbd1b55cdd8914d0f09580bfca0b2558f73aa14631d1eba61d424cd` | `cora_formal_srs` | `0734f359deac2c145bab22ed3ed40de7310eda95e732de9aa45f66919108fcf3` |
| Citeseer | `50be73f5569a1e18c743f1b1b5c6fb2730750270cbe4b00b40099037c644235e` | `c92f42267f8affee1a6888e607e893b88cb26f6c460ea6028f57765319a25f63` | `citeseer_formal_srs` | `b9ebe4d4cba8c333c83ab1d31f2c9b328199a2e1dd20153647631a73a7b4839f` |
| PPI | `ca0d4f29915c258d5a032000835f68025f9b343094b31cd2fd3522448d92d8d2` | `65a226dbdb4999b944241cfc511a577991905a60c1e224bda5d37d9ab397a5df` | `ppi_formal_srs` | `859f59973e1c917303adc7015eac2472250b58cd5508b59dcba62b00b2419e34` |

## 计时口径和结果

每个配置包含 3 次 Original/Control 配对测量。派生统计先逐对计算，再报告 3 个配对值的均值和样本标准差：Saving (%) = (Control − Original) / Control × 100%；Improvement Factor = Control / Original；配对差为 Original − Control。没有对组均值再求比值，也没有置信区间或统计显著性结论。

- **Commitment Time (ms)**：native KZG commitment；包括 commitment 路径内首次静态基点准备；不含 Taylor-shift coefficient materialization。
- **Prove Time (ms)**：online mechanism proving、Taylor-shift coefficient materialization 和规范 proof encoding；不含 KZG commitment 和 proof 文件 I/O。
- **Verify Time (ms)**：独立 verifier 从解析到接受/拒绝的记录时间。验证器在 CPU 上运行。
- **Proof Size (Bytes)**：规范序列化后的实际机制 proof 文件长度。

上游输入载入与真实中间值准备单独记录，不计入上述四项。正式证明均使用 CUDA 后端，记录了实际 CUDA NTT、INTT 与 MSM 执行。GPU 运行明确使用 shared-GPU 选项；394 个活动监测样本记录显示其他用户计算进程 PID 960497 和 1089566 的活动，没有干预这些进程。运行器将本轮 66 个 native prover PID 与外部进程区分记录。因此结果是真实共享设备上的实测值，不应解释为独占 GPU 基准。

### Feature Folding

| Dataset | N | E | D | G | PPI components | Metric | Original mean | Original sample SD | Control mean | Control sample SD | Paired O−C mean | Paired O−C sample SD | Saving % mean | Saving % sample SD | Control/Original mean | Control/Original sample SD | Repetitions |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cora | 2708 | 13264 | 1 | 1 | 0 | Commitment Time (ms) | 132.196016 | 9.003068 | 129.553191 | 6.699432 | 2.642825 | 2.304019 | -1.982220 | 1.676955 | 0.980740 | 0.016142 | 3/3 |
| cora | 2708 | 13264 | 1 | 1 | 0 | Prove Time (ms) | 214.143930 | 8.569871 | 220.837092 | 12.584712 | -6.693162 | 5.052354 | 2.964141 | 2.075519 | 1.030864 | 0.022254 | 3/3 |
| cora | 2708 | 13264 | 1 | 1 | 0 | Verify Time (ms) | 35.071085 | 0.553868 | 35.768942 | 0.924423 | -0.697857 | 0.642892 | 1.927220 | 1.731698 | 1.019864 | 0.018141 | 3/3 |
| cora | 2708 | 13264 | 1 | 1 | 0 | Proof Size (Bytes) | 3848.00 | 0.00 | 3743.00 | 0.00 | 105.00 | 0.00 | -2.805236 | 0.000000 | 0.972713 | 0.000000 | 3/3 |
| cora | 2708 | 13264 | 2 | 1 | 0 | Commitment Time (ms) | 131.103915 | 5.701119 | 238.891992 | 15.519220 | -107.788078 | 9.871084 | 45.067103 | 1.257460 | 1.821032 | 0.041216 | 3/3 |
| cora | 2708 | 13264 | 2 | 1 | 0 | Prove Time (ms) | 219.425529 | 4.594433 | 407.190561 | 22.725257 | -187.765032 | 18.739912 | 46.035397 | 2.122985 | 1.854953 | 0.071949 | 3/3 |
| cora | 2708 | 13264 | 2 | 1 | 0 | Verify Time (ms) | 36.183942 | 0.117266 | 49.843894 | 0.592779 | -13.659953 | 0.674149 | 27.397477 | 1.023571 | 1.377547 | 0.019580 | 3/3 |
| cora | 2708 | 13264 | 2 | 1 | 0 | Proof Size (Bytes) | 3982.00 | 0.00 | 7105.00 | 0.00 | -3123.00 | 0.00 | 43.954961 | 0.000000 | 1.784279 | 0.000000 | 3/3 |
| cora | 2708 | 13264 | 4 | 1 | 0 | Commitment Time (ms) | 131.693751 | 6.181249 | 450.025080 | 24.400005 | -318.331330 | 18.439424 | 70.727015 | 0.400626 | 3.416549 | 0.047084 | 3/3 |
| cora | 2708 | 13264 | 4 | 1 | 0 | Prove Time (ms) | 219.192538 | 9.187472 | 781.056467 | 29.611887 | -561.863929 | 20.430255 | 71.939295 | 0.119834 | 3.563746 | 0.015255 | 3/3 |
| cora | 2708 | 13264 | 4 | 1 | 0 | Verify Time (ms) | 36.443253 | 0.235660 | 77.345754 | 2.032159 | -40.902500 | 1.796818 | 52.866492 | 0.923996 | 2.122181 | 0.041941 | 3/3 |
| cora | 2708 | 13264 | 4 | 1 | 0 | Proof Size (Bytes) | 4250.00 | 0.00 | 13829.00 | 0.00 | -9579.00 | 0.00 | 69.267481 | 0.000000 | 3.253882 | 0.000000 | 3/3 |
| cora | 2708 | 13264 | 8 | 1 | 0 | Commitment Time (ms) | 144.341810 | 12.020545 | 896.298752 | 94.413616 | -751.956942 | 95.838469 | 83.779615 | 2.124320 | 6.238243 | 0.840091 | 3/3 |
| cora | 2708 | 13264 | 8 | 1 | 0 | Prove Time (ms) | 225.572148 | 5.226329 | 1578.782958 | 141.472482 | -1353.210809 | 138.421237 | 85.649809 | 1.070840 | 6.995653 | 0.544863 | 3/3 |
| cora | 2708 | 13264 | 8 | 1 | 0 | Verify Time (ms) | 37.300290 | 0.212180 | 135.910515 | 2.916121 | -98.610224 | 2.936429 | 72.546599 | 0.624225 | 3.643782 | 0.082232 | 3/3 |
| cora | 2708 | 13264 | 8 | 1 | 0 | Proof Size (Bytes) | 4786.00 | 0.00 | 27277.00 | 0.00 | -22491.00 | 0.00 | 82.454082 | 0.000000 | 5.699331 | 0.000000 | 3/3 |
| citeseer | 3327 | 12431 | 8 | 1 | 0 | Commitment Time (ms) | 143.777858 | 8.727506 | 939.062666 | 105.133799 | -795.284807 | 100.698738 | 84.592176 | 1.488114 | 6.531188 | 0.637403 | 3/3 |
| citeseer | 3327 | 12431 | 8 | 1 | 0 | Prove Time (ms) | 230.110673 | 8.586204 | 1639.147056 | 153.798069 | -1409.036382 | 148.606379 | 85.895367 | 1.132014 | 7.119578 | 0.555609 | 3/3 |
| citeseer | 3327 | 12431 | 8 | 1 | 0 | Verify Time (ms) | 35.661619 | 1.396813 | 131.528486 | 9.996152 | -95.866866 | 8.925349 | 72.821972 | 1.421730 | 3.686114 | 0.191388 | 3/3 |
| citeseer | 3327 | 12431 | 8 | 1 | 0 | Proof Size (Bytes) | 4790.00 | 0.00 | 27281.00 | 0.00 | -22491.00 | 0.00 | 82.441993 | 0.000000 | 5.695407 | 0.000000 | 3/3 |
| ppi | 1546 | 17699 | 8 | 1 | 0 | Commitment Time (ms) | 4841.685374 | 283.546066 | 29139.920117 | 283.265661 | -24298.234743 | 472.855325 | 83.381154 | 1.045684 | 6.033759 | 0.393231 | 3/3 |
| ppi | 1546 | 17699 | 8 | 1 | 0 | Prove Time (ms) | 24803.083967 | 1616.233341 | 169974.396391 | 1270.812674 | -145171.312425 | 2811.463833 | 85.402984 | 1.045324 | 6.875222 | 0.513341 | 3/3 |
| ppi | 1546 | 17699 | 8 | 1 | 0 | Verify Time (ms) | 732.656534 | 50.927608 | 5554.053955 | 145.496373 | -4821.397421 | 101.560590 | 86.817036 | 0.622216 | 7.597069 | 0.366231 | 3/3 |
| ppi | 1546 | 17699 | 8 | 1 | 0 | Proof Size (Bytes) | 4785.00 | 0.00 | 27276.00 | 0.00 | -22491.00 | 0.00 | 82.457105 | 0.000000 | 5.700313 | 0.000000 | 3/3 |
| ppi | 2923 | 33928 | 8 | 2 | 0,9 | Commitment Time (ms) | 4810.942343 | 292.279827 | 28847.553516 | 398.054226 | -24036.611173 | 330.047281 | 83.326213 | 0.906150 | 6.009286 | 0.327189 | 3/3 |
| ppi | 2923 | 33928 | 8 | 2 | 0,9 | Prove Time (ms) | 24924.818230 | 1392.639262 | 169608.465302 | 3804.602370 | -144683.647071 | 3038.083970 | 85.307805 | 0.648493 | 6.815417 | 0.308651 | 3/3 |
| ppi | 2923 | 33928 | 8 | 2 | 0,9 | Verify Time (ms) | 750.784711 | 49.952939 | 5284.680328 | 197.820629 | -4533.895617 | 211.996520 | 85.776355 | 1.165816 | 7.060846 | 0.554592 | 3/3 |
| ppi | 2923 | 33928 | 8 | 2 | 0,9 | Proof Size (Bytes) | 4785.00 | 0.00 | 27276.00 | 0.00 | -22491.00 | 0.00 | 82.457105 | 0.000000 | 5.700313 | 0.000000 | 3/3 |

完整原始报告：[feature_folding_summary_initial_only.md](../results/mechanism_ablation_final/mechanism_20261010T063825Z_9fe0a93c236b/feature_folding_summary_initial_only.md)，CSV：[feature_folding_summary_initial_only.csv](../results/mechanism_ablation_final/mechanism_20261010T063825Z_9fe0a93c236b/feature_folding_summary_initial_only.csv)。

### Shared Neighborhood

| Dataset | N | E | D | G | PPI components | Metric | Original mean | Original sample SD | Control mean | Control sample SD | Paired O−C mean | Paired O−C sample SD | Saving % mean | Saving % sample SD | Control/Original mean | Control/Original sample SD | Repetitions |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cora | 2708 | 13264 | 8 | 1 | 0 | Commitment Time (ms) | 206.702699 | 7.120460 | 353.634640 | 24.946647 | -146.931942 | 21.817175 | 41.410275 | 3.336515 | 1.710607 | 0.100709 | 3/3 |
| cora | 2708 | 13264 | 8 | 1 | 0 | Prove Time (ms) | 345.534078 | 10.972564 | 677.117608 | 50.813144 | -331.583530 | 40.008304 | 48.860268 | 2.215929 | 1.957891 | 0.085334 | 3/3 |
| cora | 2708 | 13264 | 8 | 1 | 0 | Verify Time (ms) | 44.467823 | 1.055529 | 68.511897 | 0.781149 | -24.044074 | 1.024862 | 35.093915 | 1.401223 | 1.541166 | 0.033278 | 3/3 |
| cora | 2708 | 13264 | 8 | 1 | 0 | Proof Size (Bytes) | 6429.00 | 0.00 | 10512.00 | 0.00 | -4083.00 | 0.00 | 38.841324 | 0.000000 | 1.635091 | 0.000000 | 3/3 |
| citeseer | 3327 | 12431 | 8 | 1 | 0 | Commitment Time (ms) | 212.143170 | 19.517307 | 349.323467 | 20.621233 | -137.180297 | 10.696460 | 39.317027 | 3.004326 | 1.650571 | 0.080686 | 3/3 |
| citeseer | 3327 | 12431 | 8 | 1 | 0 | Prove Time (ms) | 351.772467 | 8.066595 | 653.836772 | 23.809918 | -302.064305 | 15.749700 | 46.181228 | 0.721516 | 1.858311 | 0.025001 | 3/3 |
| citeseer | 3327 | 12431 | 8 | 1 | 0 | Verify Time (ms) | 44.449377 | 0.156974 | 67.252298 | 3.114517 | -22.802920 | 3.046451 | 33.812658 | 3.048005 | 1.512947 | 0.067872 | 3/3 |
| citeseer | 3327 | 12431 | 8 | 1 | 0 | Proof Size (Bytes) | 6433.00 | 0.00 | 10516.00 | 0.00 | -4083.00 | 0.00 | 38.826550 | 0.000000 | 1.634696 | 0.000000 | 3/3 |
| ppi | 1546 | 17699 | 8 | 1 | 0 | Commitment Time (ms) | 7102.556514 | 402.214694 | 12725.005279 | 842.770615 | -5622.448765 | 441.443197 | 44.159988 | 0.565220 | 1.790953 | 0.018136 | 3/3 |
| ppi | 1546 | 17699 | 8 | 1 | 0 | Prove Time (ms) | 45863.157337 | 1714.352951 | 86174.995114 | 3064.169239 | -40311.837777 | 1413.986678 | 46.780454 | 0.362589 | 1.879067 | 0.012814 | 3/3 |
| ppi | 1546 | 17699 | 8 | 1 | 0 | Verify Time (ms) | 1439.559956 | 71.144898 | 2883.187509 | 175.350953 | -1443.627553 | 121.731069 | 50.029139 | 1.690104 | 2.002696 | 0.067845 | 3/3 |
| ppi | 1546 | 17699 | 8 | 1 | 0 | Proof Size (Bytes) | 6428.00 | 0.00 | 10511.00 | 0.00 | -4083.00 | 0.00 | 38.845020 | 0.000000 | 1.635190 | 0.000000 | 3/3 |
| ppi | 2923 | 33928 | 8 | 2 | 0,9 | Commitment Time (ms) | 7209.160186 | 414.030242 | 12787.075050 | 766.681156 | -5577.914864 | 352.939877 | 43.615525 | 0.170283 | 1.773549 | 0.005352 | 3/3 |
| ppi | 2923 | 33928 | 8 | 2 | 0,9 | Prove Time (ms) | 46051.176317 | 1091.990995 | 87017.482740 | 2819.556230 | -40966.306422 | 1745.086203 | 47.067963 | 0.504654 | 1.889330 | 0.018042 | 3/3 |
| ppi | 2923 | 33928 | 8 | 2 | 0,9 | Verify Time (ms) | 1440.269591 | 25.401856 | 2832.875936 | 179.566569 | -1392.606345 | 166.596748 | 49.047690 | 2.737668 | 1.966488 | 0.108012 | 3/3 |
| ppi | 2923 | 33928 | 8 | 2 | 0,9 | Proof Size (Bytes) | 6428.00 | 0.00 | 10511.00 | 0.00 | -4083.00 | 0.00 | 38.845020 | 0.000000 | 1.635190 | 0.000000 | 3/3 |

完整原始报告：[shared_neighborhood_summary_initial_only.md](../results/mechanism_ablation_final/mechanism_20261010T063825Z_9fe0a93c236b/shared_neighborhood_summary_initial_only.md)，CSV：[shared_neighborhood_summary_initial_only.csv](../results/mechanism_ablation_final/mechanism_20261010T063825Z_9fe0a93c236b/shared_neighborhood_summary_initial_only.csv)。

在这些观测均值中，Feature Folding 的 Cora D=1 结果方向混合；其余 Feature Folding 配置四项 Original 均值均低于 Control。Shared Neighborhood 的四组配置中，Original 四项均值都低于 Control。三次运行的样本差异描述重复测量分布，且受共享 GPU 活动影响；不作为统计显著性或普遍加速保证。这些是独立机制 proof 的局部收益，不能外推成完整 GAT 端到端前向或零知识证明性能收益。

本实验使用注册的 trusted SRS 物理域：Cora/Citeseer 的 node/edge 域分别为 4096/32768，PPI 为 65536/8388608。PPI 的 G=1 与 G=2 沿用同一注册物理域；表中的实际图 N/E 和组件边界未因物理域而改变。较大的 PPI 物理域为其绝对耗时提供重要背景，但实验没有单独隔离域大小的因果影响，因此不能将全部 PPI 耗时差归因于此。每组仅 n=3，且实验使用共享 GPU，这些测量不支持对域大小作因果结论，也不构成独占 GPU 基准或统计显著性证据。

## 构建、来源身份和复现命令

本次实验源码身份为 source-tree inventory SHA-256 `b014f2e34acf840da37d15d09973c871153a5d3bf873f7a6918c1ef234a22a3b`。README 和本实验文档属于单独 provenance snapshot，不计入可执行 source-tree identity；session 结束时原执行源码、构建、checkpoint 和 SRS 身份保持稳定。主要构建身份如下：

| Provenance item | Identity |
| --- | --- |
| Source tree inventory SHA-256 | `b014f2e34acf840da37d15d09973c871153a5d3bf873f7a6918c1ef234a22a3b` |
| Source diff SHA-256 | `f65a5c4261d51770da710c6bcfd740036603966cc263f7c5aaa71ac23633b1da` |
| CUDA binary SHA-256 | `1410547bd6f8437de3e39f85a84d14a0b7bd964383487581a9f5ede2ac181cf6` |
| CMake cache SHA-256 | `58cd7db89f92e67345c77389a08ffbd6d7d77e4c9369a001a67c69222710884a` |
| Build source manifest SHA-256 | `85c96da5f59bc76689f9fa85c9152254e2a6746a20c246082fb2408cf8f8dfaa` |
| Results manifest SHA-256 | `16a2d58ecc94f9e5348901ff00c467a2facb1bf39bd08a7188f4976fe59f8c4d` |
| Configuration/SRS manifest SHA-256 | `2252dacdfc0ae1bb492d38a1f749e40b80d461ea456ef7573b1db864d718cde2` |

每个正式输入 bundle 的 checkpoint bundle SHA-256、配置 SHA-256 和 SRS manifest SHA-256 记录在上表及每次 prepared-input metadata 中；完整的 source/build/result 清单保存在 session 下的 `source_manifest.json`, `build_source_manifest.json`, `results_manifest.json`, `configuration_and_srs_manifest.json` 和 `summary.json`。证明声明、规范 proof 文件、初次独立验证收据、CUDA activity sample 和各自哈希由 `cells.csv` 与 `results_manifest.json` 索引。

成功执行的命令如下，工作目录均为 `/home/pzh/project`。CMake configure 沿用已验证的 Release/CUDA build cache；这条命令本身没有附加或改写 CUDA/Release flags。构建和执行日志存放在下方列出的 session 绝对路径。

正式 session 中 launcher 复制的 build manifest 需恢复冻结的原始 JSON 字节，避免 raw SHA 检查失败。修复于 2026-10-10T08:51:38Z 成功；该 argv 来自修复脚本自动写入的 record，并以脚本、目标文件与 repair record 的时间戳及预声明 SHA 相互印证，不是独立终端 transcript。操作只恢复构建 manifest 副本：内容语义一致并匹配预声明 SHA；没有改 proof、statement、summary hash 或协议。记录文件为 `results/mechanism_ablation_final/build_manifest_copy_repair_20261010T085138Z.json`。复现到新的正式运行时，应使用新 session 和该次对应的修复记录，不要机械重放旧 session 路径。

~~~bash
cd /home/pzh/project
cmake -S . -B build_cuda
cmake --build build_cuda --config Release --parallel 4 --target gatzk_mechanism gatzk_run gatzk_core_execution_regression gatzk_mechanism_security_regression gatzk_real_mechanism_input_regression
python3 scripts/run_mechanism_ablation.py launch --binary /home/pzh/project/build_cuda/gatzk_mechanism --output-root /home/pzh/project/results/mechanism_ablation_final --device 0 --repetitions 3 --timeout-seconds 0 --prepare-timeout-seconds 0 --allow-shared-gpu
python3 /home/pzh/project/results/mechanism_ablation_final/repair_build_manifest_copy.py --session /home/pzh/project/results/mechanism_ablation_final/mechanism_20261010T063825Z_9fe0a93c236b --build-manifest /home/pzh/project/build_cuda/generated/gatzk/source_manifest.json
python3 scripts/run_mechanism_ablation.py audit /home/pzh/project/results/mechanism_ablation_final/mechanism_20261010T063825Z_9fe0a93c236b
python3 /home/pzh/project/results/mechanism_ablation_final/mechanism_20261010T063825Z_9fe0a93c236b/report_initial_only.py
~~~

## 后续完整运行的自动合并表

当前 runner 对新的 session 自动生成 `combined_summary.csv`、`combined_summary.md` 和 `combined_summary.json`。只有 11 个注册配置的所有配对与初次独立验证均成功、源码/构建身份稳定且导出时工件 SHA-256 检查通过，才会生成合并表；CSV/JSON 为每个配置一行，共 11 行。每行记录机制、数据集、实际 N/E/D/G、PPI 组件、重复次数及四项指标的 Original、Control、Original−Control、Saving、Improvement Factor 统计。Saving 和倍数先对每个配对分别计算，再汇总配对值；重复数至少为 2 时报告均值与样本标准差。单次运行只报告实测值，不生成样本标准差。合并文件的 session 内绝对路径和 SHA-256 写入 `results_manifest.json`。

用于新的三次重复完整运行的统一后台入口如下。`launch` 会创建 detached launcher，日志位于新 session root 下的 `launchers/cuda_*/runner.log`；运行成功后不需要单独的合并命令。以下命令是复现说明，**本轮只实现并测试导出功能，没有启动该正式命令或任何证明**。如果 session 不完整或初验/工件检查失败，合并文件不会生成，失败状态保存在 `summary.json`。合并报告的依据是初次独立 verifier 收据及工件哈希检查；导出器不会执行或宣称全量独立重验。既有 strict `audit --reverify`、`summarize` 和出版完整性门禁未修改。此功能只应用于该 schema 的新 session，不追补或改写历史 session。

~~~bash
cd /home/pzh/project
python3 scripts/run_mechanism_ablation.py launch --binary /home/pzh/project/build_cuda/gatzk_mechanism --output-root /home/pzh/project/results/mechanism_ablation_final --device 0 --repetitions 3 --timeout-seconds 0 --prepare-timeout-seconds 0 --allow-shared-gpu
~~~

CSV 使用 `{metric}_{statistic}_{field}` 列：metric 为 `commitment_time_ms`、`prove_time_ms`、`verify_time_ms`、`proof_size_bytes`；statistic 为 `original`、`control`、`paired_original_minus_control`、`saving_percent`、`improvement_factor`；field 为 `value`、`mean`、`sample_sd`、`n`。三份合并文件分别是 `combined_summary.csv`、`combined_summary.md`、`combined_summary.json`。

GPU/工具链信息实际查询命令为：

~~~bash
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
nvcc --version | tail -n 4
~~~

实际环境为 NVIDIA GB10、UUID `GPU-3cde57ba-ae43-4452-bd59-7b156b36db71`、driver 580.142、CUDA 13.0 / nvcc V13.0.88、CMake 3.28.3、G++ 13.3.0。日志：

- Configure: `/home/pzh/project/results/mechanism_ablation_final/preflight_configure_20261010T0635Z.log`
- Build: `/home/pzh/project/results/mechanism_ablation_final/preflight_build_release_20261010T0635Z.log`
- Detached serial launcher: `/home/pzh/project/results/mechanism_ablation_final/launchers/cuda_20261010T063825Z_289d2209cd/runner.log`
- Formal session and per-cell artifacts: `/home/pzh/project/results/mechanism_ablation_final/mechanism_20261010T063825Z_9fe0a93c236b/`
- Ordinary receipt/hash audit: `.../audit.json` (PASS, `reverified=false`)
- Cancelled full-reverification evidence: `/home/pzh/project/results/mechanism_ablation_final/incidents/reverify_cancelled_20261010T085441Z_c57bc5f351/`

全量重验尝试的准确命令及结果：`python3 /home/pzh/project/scripts/run_mechanism_ablation.py audit /home/pzh/project/results/mechanism_ablation_final/mechanism_20261010T063825Z_9fe0a93c236b --reverify`，exit 2；13 项成功后用户取消正在执行的第 14 项。该命令不是成功的复现步骤，也没有产生完整重验 PASS 归档。`record-publication-docs`、strict `publication_integrity` 和正式 gate 后的 `summarize` 未成功执行；本页面展示的是单独生成并带 initial-only 完整性清单的结果。旧 `runs/` 未删除，因为用户原先设定的“全量重验通过后清理”条件未满足。

## 结果文件

正式数据均位于 session 目录，不与历史合成实验混合：

- `feature_folding_summary_initial_only.md` / `.csv`
- `shared_neighborhood_summary_initial_only.md` / `.csv`
- `initial_only_cells.csv` / `.json`：66 个逐次机制记录
- `initial_only_pairs.csv` / `.json`：33 对配对记录
- `initial_only_statistics.json`
- `initial_only_report_integrity.json` / `.sha256`
- `report_initial_only.py`

独立报告完整性清单中的状态为 `CANCELLED_BY_USER_AFTER_13_OF_66`，验证范围为初次 verifier receipt 与哈希审计；其中 strict runner publication receipt 标记为 `NOT_CREATED`。它没有把初验收据等同于全量重新验证。本文改动后，应使用 session 内的 `report_initial_only.py` 刷新文档 SHA-256 绑定；这只更新 initial-only 报告清单，不执行 proof 或 verifier。
