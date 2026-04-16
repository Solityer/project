#pragma once
// 本文件定义 formal 主线中的 LogUp / lookup 公用构件，
// 服务特征表、激活表、range 表与 exp 表的查询约束。

#include <cstddef>
#include <string>
#include <vector>

#include "gatzk/algebra/field.hpp"

namespace gatzk::protocol {

// 一次 lookup 轨迹的标准列布局。
struct LogUpTrace {
    std::vector<algebra::FieldElement> table;
    std::vector<algebra::FieldElement> query;
    std::vector<algebra::FieldElement> multiplicity;
    std::vector<algebra::FieldElement> q_table;
    std::vector<algebra::FieldElement> q_query;
    std::vector<algebra::FieldElement> accumulator;
};

// 生成有效区选择器，控制 table / query 在工作域中的填充边界。
std::vector<algebra::FieldElement> build_selector(std::size_t valid_length, std::size_t domain_size);
// 构造正式的 LogUp 累加器递推。
std::vector<algebra::FieldElement> build_logup_accumulator(
    const std::vector<algebra::FieldElement>& table,
    const std::vector<algebra::FieldElement>& query,
    const std::vector<algebra::FieldElement>& multiplicity,
    const std::vector<algebra::FieldElement>& q_table,
    const std::vector<algebra::FieldElement>& q_query,
    const algebra::FieldElement& beta);
// 只对有效过渡前缀做递推，避免在大域填充区浪费时间。
std::vector<algebra::FieldElement> build_logup_accumulator_with_active_count(
    const std::vector<algebra::FieldElement>& table,
    const std::vector<algebra::FieldElement>& query,
    const std::vector<algebra::FieldElement>& multiplicity,
    const std::vector<algebra::FieldElement>& q_table,
    const std::vector<algebra::FieldElement>& q_query,
    const algebra::FieldElement& beta,
    std::size_t active_count);
// 带缓存版本，供 warm / measured run 之间复用正式允许的执行层结果。
std::vector<algebra::FieldElement> build_logup_accumulator_cached(
    const std::string& cache_key,
    const std::vector<algebra::FieldElement>& table,
    const std::vector<algebra::FieldElement>& query,
    const std::vector<algebra::FieldElement>& multiplicity,
    const std::vector<algebra::FieldElement>& q_table,
    const std::vector<algebra::FieldElement>& q_query,
    const algebra::FieldElement& beta);
// 既限制有效前缀，又允许缓存复用，是大图 lookup 热路径常用入口。
std::vector<algebra::FieldElement> build_logup_accumulator_cached_with_active_count(
    const std::string& cache_key,
    const std::vector<algebra::FieldElement>& table,
    const std::vector<algebra::FieldElement>& query,
    const std::vector<algebra::FieldElement>& multiplicity,
    const std::vector<algebra::FieldElement>& q_table,
    const std::vector<algebra::FieldElement>& q_query,
    const algebra::FieldElement& beta,
    std::size_t active_count);

}  // namespace gatzk::protocol
