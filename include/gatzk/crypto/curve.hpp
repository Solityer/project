#pragma once
// 本文件封装 KZG 依赖的群运算、序列化与 pairing 接口，
// 是 protocol 层承诺、opening 与验证方程访问曲线后端的唯一入口。

#include <cstddef>
#include <cstdint>
#include <mcl/bn.hpp>
#include <memory>
#include <string>
#include <vector>

#include "gatzk/algebra/field.hpp"

namespace gatzk::crypto
{
    // G1 上的承诺点与见证点包装。
    struct G1Point
    {
        mcl::G1 value;

        std::string to_string() const;
    };

    // G2 上的 SRS 点包装，主要用于 KZG 验证方程。
    struct G2Point
    {
        mcl::G2 value;

        std::string to_string() const;
    };

    // 预处理后的 G2 点，用于降低 verifier pairing 热路径的重复成本。
    struct PreparedG2
    {
        std::vector<mcl::Fp6> coeffs;
    };

    // 返回当前正式主线使用的曲线后端名称。
    std::string backend_name();

    // 下面这组接口只暴露群运算语义，不向 protocol 层泄漏 mcl 细节。
    G1Point g1_zero();
    G2Point g2_zero();
    G1Point g1_add(const G1Point& lhs, const G1Point& rhs);
    G1Point g1_sub(const G1Point& lhs, const G1Point& rhs);
    G1Point g1_mul(const G1Point& point, const algebra::FieldElement& scalar);
    std::vector<G1Point> g1_mul_same_base_batch(const G1Point& point, const std::vector<algebra::FieldElement>& scalars);
    G2Point g2_add(const G2Point& lhs, const G2Point& rhs);
    G2Point g2_sub(const G2Point& lhs, const G2Point& rhs);
    G2Point g2_mul(const G2Point& point, const algebra::FieldElement& scalar);
    bool operator==(const G1Point& lhs, const G1Point& rhs);
    bool operator==(const G2Point& lhs, const G2Point& rhs);
    std::vector<std::uint8_t> serialize(const G1Point& point);
    std::vector<std::uint8_t> serialize(const G2Point& point);
    std::size_t serialized_size(const G1Point& point);
    std::size_t serialized_size(const G2Point& point);
    PreparedG2 prepare_g2(const G2Point& point);

    // 检查两项 pairing 是否相等，是 KZG batch verification 的基础原语。
    bool pairing_equal(
        const G1Point& lhs_a,
        const G2Point& rhs_a,
        const G1Point& lhs_b,
        const G2Point& rhs_b);
        
    // 将固定 G2 端预处理后参与 pairing，可减少 verifier 热路径的重复准备成本。
    bool pairing_product_is_one_mixed_prepared(
        const G1Point& lhs_dynamic,
        const G2Point& rhs_dynamic,
        const G1Point& lhs_prepared,
        const PreparedG2& rhs_prepared);

}
