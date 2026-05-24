#pragma once

// official full-chain proof 共享的 Fr 编码、确定性挑战和有效区 MLE 工具。
// 这些函数服务 CPU/GPU relation record，不写 artifact，也不读取 witness 以外的调试数据。

// 将有符号 int64 映射到 Fr 的标准代表；负数按协议的 p-|x| 语义编码。
Fr_t fr_from_signed_host(std::int64_t value)
{
    Fr_t out = {{0, 0, 0, 0, 0, 0, 0, 0}};
    const std::uint32_t p[8] = {1U, 4294967295U, 4294859774U, 1404937218U, 161601541U, 859428872U, 698187080U, 1944954707U};
    if (value >= 0) {
        std::uint64_t v = static_cast<std::uint64_t>(value);
        out.val[0] = static_cast<std::uint32_t>(v & 0xffffffffULL);
        out.val[1] = static_cast<std::uint32_t>((v >> 32U) & 0xffffffffULL);
        return out;
    }
    std::uint64_t abs_value = static_cast<std::uint64_t>(-value);
    std::uint32_t sub[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    sub[0] = static_cast<std::uint32_t>(abs_value & 0xffffffffULL);
    sub[1] = static_cast<std::uint32_t>((abs_value >> 32U) & 0xffffffffULL);
    std::uint64_t borrow = 0;
    for (int i = 0; i < 8; ++i) {
        const std::uint64_t lhs = p[i];
        const std::uint64_t rhs = static_cast<std::uint64_t>(sub[i]) + borrow;
        if (lhs >= rhs) {
            out.val[i] = static_cast<std::uint32_t>(lhs - rhs);
            borrow = 0;
        } else {
            out.val[i] = static_cast<std::uint32_t>((1ULL << 32U) + lhs - rhs);
            borrow = 1;
        }
    }
    return out;
}

// 输出 Montgomery 形态的有符号 Fr 元素，供 relation prover 直接参与乘加约束。
Fr_t fr_from_signed_mont_host(std::int64_t value)
{
    return Fr_host_mont(fr_from_signed_host(value));
}

// FNV-1a 用于本目录内的可复现实验 ID、挑战标签和 metadata digest。
std::uint64_t fnv1a64(const std::string& value)
{
    std::uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char ch : value) {
        hash ^= ch;
        hash *= 1099511628211ULL;
    }
    return hash;
}

// 从标签生成确定性 Fr raw 向量，保证 CPU/GPU runner 的评值点可复现。
std::vector<Fr_t> deterministic_raw_fr_vec(std::size_t n, const std::string& label)
{
    std::mt19937_64 rng(fnv1a64(label));
    std::vector<Fr_t> out(n);
    for (auto& value : out) {
        value = fr_from_signed_host(0);
        const auto lo = rng();
        value.val[0] = static_cast<std::uint32_t>(lo & 0xffffffffULL);
        value.val[1] = static_cast<std::uint32_t>((lo >> 32U) & 0xffffffffULL);
    }
    return out;
}

// 从标签生成 Montgomery 形态评值点；用于 full-chain relation 的 MLE 末端声明。
std::vector<Fr_t> deterministic_mont_fr_vec(std::size_t n, const std::string& label)
{
    auto raw = deterministic_raw_fr_vec(n, label);
    if (raw.empty()) {
        return raw;
    }
    FrTensor tensor(static_cast<uint>(raw.size()), raw.data());
    tensor.mont();
    cuda_sync("deterministic_mont_fr_vec");
    std::vector<Fr_t> out(raw.size());
    for (uint i = 0; i < tensor.size; ++i) {
        out[i] = tensor(i);
    }
    return out;
}

// 计算小整数在 Fr 中的 Montgomery 逆元，用于 Sum * inv = 1 与 K_out 平均。
Fr_t fr_inverse_uint_mont(std::uint32_t denominator)
{
    if (denominator == 0) throw std::runtime_error("zero denominator inversion");
    if (denominator == 1) return Fr_host_one_mont();
    const Fr_t p = Fr_host_modulus();
    std::uint64_t p_mod = 0;
    for (int i = 7; i >= 0; --i) {
        p_mod = ((p_mod << 32U) + p.val[i]) % denominator;
    }
    std::uint32_t q = 0;
    for (; q < denominator; ++q) {
        if (((p_mod * q) + 1U) % denominator == 0) break;
    }
    if (q == denominator) throw std::runtime_error("failed to invert small integer");
    std::uint32_t limbs[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::uint64_t carry = 1U;
    for (int i = 0; i < 8; ++i) {
        const std::uint64_t cur = static_cast<std::uint64_t>(p.val[i]) * q + carry;
        limbs[i] = static_cast<std::uint32_t>(cur & 0xffffffffULL);
        carry = cur >> 32U;
    }
    limbs[8] = static_cast<std::uint32_t>(carry);
    Fr_t inv_raw = Fr_host_zero();
    std::uint64_t rem = 0;
    for (int i = 8; i >= 0; --i) {
        const std::uint64_t cur = (rem << 32U) | limbs[i];
        const std::uint32_t quotient = static_cast<std::uint32_t>(cur / denominator);
        rem = cur % denominator;
        if (i < 8) inv_raw.val[i] = quotient;
    }
    if (rem != 0) throw std::runtime_error("small integer inverse division left remainder");
    return Fr_host_mont(inv_raw);
}

// 计算一维有效区 selector 的 MLE；验证 padding 区固定为 0 时使用。
Fr_t evaluate_valid_selector_mle(std::size_t valid_count, const std::vector<Fr_t>& point)
{
    const std::size_t padded_count = next_power_of_two(std::max<std::size_t>(1, valid_count));
    std::vector<Fr_t> values(padded_count, Fr_host_zero());
    for (std::size_t i = 0; i < valid_count; ++i) {
        values[i] = Fr_host_one_mont();
    }
    return formal_host_mle_vector(values, point.begin(), point.end());
}

// 计算二维有效区 selector 的 MLE；rescale/rounding 和矩阵 relation 用它绑定真实区。
Fr_t evaluate_valid_matrix_selector_mle(
    std::size_t rows,
    std::size_t cols,
    std::size_t padded_rows,
    std::size_t padded_cols,
    const std::vector<Fr_t>& point)
{
    if (rows > padded_rows || cols > padded_cols || point.size() != ceilLog2(static_cast<uint>(padded_rows * padded_cols))) {
        throw std::runtime_error("valid matrix selector MLE shape mismatch");
    }
    std::vector<Fr_t> values(padded_rows * padded_cols, Fr_host_zero());
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            values[row * padded_cols + col] = Fr_host_one_mont();
        }
    }
    return formal_host_mle_vector(values, point.begin(), point.end());
}

// -1 的 Montgomery 形态；linear relation 用它把 public evaluation 移到等式左侧。
Fr_t fr_neg_one_mont()
{
    return Fr_host_sub(Fr_host_zero(), Fr_host_one_mont());
}

// 与 protocol_context/runner 共用的 scale16 舍入规则，固定为 round_half_away_from_zero。
std::int64_t quantize_scale16_to_i64(double value)
{
    const double scaled = value * 16.0;
    return static_cast<std::int64_t>(scaled >= 0.0 ? scaled + 0.5 : scaled - 0.5);
}
