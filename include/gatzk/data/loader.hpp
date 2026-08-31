#pragma once
// 本文件声明数据加载与图规范化入口，
// 负责把外部数据源整理成 protocol 层可直接消费的正式对象。

#include <cstdint>

#include "gatzk/data/dataset.hpp"
#include "gatzk/util/config.hpp"

namespace gatzk::data
{
    // Canonical registered fixed-point quantizer shared by dataset loading,
    // trace construction, and the independent reference.  It implements
    // nearest-integer rounding with exact half cases away from zero and
    // rejects non-finite inputs, zero scales, and int64 overflow.
    std::int64_t quantize_raw_at_scale(
        double value,
        std::uint64_t scale);
    algebra::FieldElement quantize_field_at_scale(
        double value,
        std::uint64_t scale);

    // 按配置读取完整数据集，并完成量化前后的基础缓存加载。
    GraphDataset load_dataset(const util::AppConfig &config);
    // 将完整数据集裁剪 / 排序 / 补自环后整理为 formal 主线使用的局部图。
    LocalGraph normalize_graph_input(const GraphDataset &dataset, const util::AppConfig &config);
    // Validate the raw and quantized feature views against the registered
    // S_input quantizer before either view can enter the formal trace.
    // Verifier registration artifacts intentionally carry only the public
    // quantized view and therefore do not use this prover-side check.
    void validate_input_feature_views(
        const GraphDataset &dataset,
        const LocalGraph &local,
        std::uint64_t input_scale);
    // 验证者与证明者共用的公开图边界检查。任何指针、图编号、
    // 边排序、跨图边或零入度节点不一致都必须在进入 trace 前失败。
    void validate_normalized_graph_input(const LocalGraph &local);
}
