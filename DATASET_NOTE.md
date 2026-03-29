# Dataset Note

## 1. Available Datasets

当前目录中已有以下数据集：

- `data/cora/`
- `data/citeseer/`
- `data/pubmed/`

这些数据集均采用经典 Planetoid / citation-network 文件格式，典型文件包括：

- `ind.<name>.x`
- `ind.<name>.y`
- `ind.<name>.tx`
- `ind.<name>.ty`
- `ind.<name>.allx`
- `ind.<name>.ally`
- `ind.<name>.graph`
- `ind.<name>.test.index`

实现必须优先兼容这种格式，而不是只支持自定义 edge list / csv / json。

## 2. Priority Requirement

数据集支持优先级如下：

1. 第一优先完整支持 `Cora`；
2. 数据加载层必须同时兼容 `Citeseer` 与 `Pubmed`；
3. 不允许把 loader 写死为只支持单一数据集；
4. 必须实现通用 citation dataset loader。

## 3. Required Loader Behavior

数据加载模块至少必须完成以下任务：

1. 读取 Planetoid 风格文件；
2. 重建节点特征矩阵；
3. 重建标签；
4. 重建图结构；
5. 正确处理测试索引；
6. 提供后续局部子图构造所需接口；
7. 能导出局部节点绝对编号、src、dst 等核心索引对象；
8. 支持按协议要求对边进行排序和重组；
9. 保留可复现的数据预处理路径。

## 4. Toy Dataset Requirement

除真实数据集外，项目内部必须提供一个 toy dataset，用于：

- 数据加载自测；
- 最小前向验证；
- trace / witness 正确性检查；
- prove / verify smoke test；
- 中间变量 dump 调试。

toy dataset 要求：

- 节点数小；
- 边数小；
- 特征维度小；
- 标签数小；
- 能人工检查聚合、中间变量和输出结果。

## 5. Reproducibility Requirement

数据处理流程必须尽量可复现，包括但不限于：

- 明确数据路径；
- 明确数据集名称；
- 固定必要的随机种子；
- 记录预处理和采样配置；
- 支持最小可运行示例。