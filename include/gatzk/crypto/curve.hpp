#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <mcl/bn.hpp>

#include "gatzk/algebra/field.hpp"

namespace gatzk::crypto {

struct G1Point {
    mcl::G1 value;

    std::string to_string() const;
};

struct G2Point {
    mcl::G2 value;

    std::string to_string() const;
};

struct PreparedG2 {
    std::vector<mcl::Fp6> coeffs;
};

std::string backend_name();

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
bool pairing_equal(
    const G1Point& lhs_a,
    const G2Point& rhs_a,
    const G1Point& lhs_b,
    const G2Point& rhs_b);
bool pairing_product_is_one_mixed_prepared(
    const G1Point& lhs_dynamic,
    const G2Point& rhs_dynamic,
    const G1Point& lhs_prepared,
    const PreparedG2& rhs_prepared);

}  // namespace gatzk::crypto
