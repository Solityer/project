#pragma once
// 本文件定义 formal 主线中的 LogUp / lookup 公用构件，
// 服务特征表、激活表、range 表与 exp 表的查询约束。

#include <cstddef>
#include <vector>

#include "gatzk/algebra/field.hpp"

namespace gatzk::protocol
{

    struct LogUpInverseColumns
    {
        std::vector<algebra::FieldElement> table;
        std::vector<algebra::FieldElement> query;
    };

    struct IndexedViewActivePrefixWitness
    {
        std::vector<algebra::FieldElement> inverses;
        std::vector<algebra::FieldElement> accumulator;
    };

    IndexedViewActivePrefixWitness build_indexed_view_active_prefix_witness(
        const std::vector<algebra::FieldElement> &tuples,
        std::size_t active_length,
        const algebra::FieldElement &active_multiplicity,
        const algebra::FieldElement &beta);

    LogUpInverseColumns build_logup_inverse_columns(
        const std::vector<algebra::FieldElement> &table,
        const std::vector<algebra::FieldElement> &query,
        const std::vector<algebra::FieldElement> &q_table,
        const std::vector<algebra::FieldElement> &q_query,
        const algebra::FieldElement &beta);

    std::vector<algebra::FieldElement>
    build_logup_accumulator_from_inverse_columns_with_active_count(
        const std::vector<algebra::FieldElement> &table_inverse,
        const std::vector<algebra::FieldElement> &query_inverse,
        const std::vector<algebra::FieldElement> &multiplicity,
        const std::vector<algebra::FieldElement> &q_table,
        const std::vector<algebra::FieldElement> &q_query,
        std::size_t active_count);

    std::vector<algebra::FieldElement> build_active_prefix_inverse_column(
        const std::vector<algebra::FieldElement> &values,
        std::size_t active_length,
        const algebra::FieldElement &beta);

    // 生成有效区选择器，控制正式列在工作域中的填充边界。
    std::vector<algebra::FieldElement> build_selector(
        std::size_t valid_length,
        std::size_t domain_size);

} // namespace gatzk::protocol