#pragma once
// 本文件声明数据加载与图规范化入口，
// 负责把外部数据源整理成 protocol 层可直接消费的正式对象。

#include "gatzk/data/dataset.hpp"
#include "gatzk/util/config.hpp"

namespace gatzk::data {

// 按配置读取完整数据集，并完成量化前后的基础缓存加载。
GraphDataset load_dataset(const util::AppConfig& config);
// 将完整数据集裁剪 / 排序 / 补自环后整理为 formal 主线使用的局部图。
LocalGraph normalize_graph_input(const GraphDataset& dataset, const util::AppConfig& config);

}  // namespace gatzk::data
