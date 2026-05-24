#pragma once

// GAT projection 的固定点语义：输入与权重量化到 scale=16，先做整数乘加，
// 再按 half-away-from-zero 除以 scale，输出重新解释为 scale=16 的 double trace。

std::int64_t zkdl_quantize_scale16_i64(double value)
{
    const double scaled = value * 16.0;
    return static_cast<std::int64_t>(scaled >= 0.0 ? scaled + 0.5 : scaled - 0.5);
}

std::int64_t zkdl_round_half_away_div_i64(std::int64_t numerator, std::int64_t denominator)
{
    if (denominator <= 0) throw std::runtime_error("fixed projection denominator 非法");
    if (numerator >= 0) {
        return (numerator + denominator / 2) / denominator;
    }
    const auto magnitude = static_cast<std::uint64_t>(-(numerator + 1)) + 1ULL;
    return -static_cast<std::int64_t>((magnitude + static_cast<std::uint64_t>(denominator / 2)) / static_cast<std::uint64_t>(denominator));
}

std::int64_t zkdl_checked_i128_to_i64(__int128 value, const std::string& label)
{
    const __int128 lo = static_cast<__int128>(std::numeric_limits<std::int64_t>::min());
    const __int128 hi = static_cast<__int128>(std::numeric_limits<std::int64_t>::max());
    if (value < lo || value > hi) {
        throw std::runtime_error(label + " 超出 int64 fixed-point 范围");
    }
    return static_cast<std::int64_t>(value);
}

std::vector<double> zkdl_fixed_point_projection_scale16(
    const std::vector<double>& input,
    std::size_t rows,
    std::size_t input_dim,
    const std::vector<double>& weights,
    std::size_t output_dim)
{
    if (input.size() != rows * input_dim || weights.size() != input_dim * output_dim) {
        throw std::runtime_error("fixed projection 输入形状不一致");
    }
    std::vector<std::int64_t> q_input(input.size(), 0);
    std::vector<std::int64_t> q_weight(weights.size(), 0);
    for (std::size_t i = 0; i < input.size(); ++i) q_input[i] = zkdl_quantize_scale16_i64(input[i]);
    for (std::size_t i = 0; i < weights.size(); ++i) q_weight[i] = zkdl_quantize_scale16_i64(weights[i]);

    std::vector<double> out(rows * output_dim, 0.0);
    const std::size_t task_count = rows * output_dim;
    const std::size_t cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    const std::size_t worker_count = std::min<std::size_t>(cpu_count, task_count == 0 ? 1 : task_count);
    auto worker = [&](std::size_t begin, std::size_t end) {
        for (std::size_t idx = begin; idx < end; ++idx) {
            const std::size_t row = idx / output_dim;
            const std::size_t col = idx % output_dim;
            __int128 acc = 0;
            for (std::size_t k = 0; k < input_dim; ++k) {
                acc += static_cast<__int128>(q_input[row * input_dim + k])
                    * static_cast<__int128>(q_weight[k * output_dim + col]);
            }
            const auto pre = zkdl_checked_i128_to_i64(acc, "fixed projection pre-rescale product");
            const auto rounded = zkdl_round_half_away_div_i64(pre, 16);
            out[idx] = static_cast<double>(rounded) / 16.0;
        }
    };
    if (worker_count <= 1 || task_count < 4096) {
        worker(0, task_count);
    } else {
        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        const std::size_t chunk = (task_count + worker_count - 1) / worker_count;
        for (std::size_t worker_index = 0; worker_index < worker_count; ++worker_index) {
            const std::size_t begin = worker_index * chunk;
            const std::size_t end = std::min(task_count, begin + chunk);
            if (begin >= end) break;
            workers.emplace_back(worker, begin, end);
        }
        for (auto& thread : workers) thread.join();
    }
    return out;
}
