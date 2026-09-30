#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mcl/bn.hpp>
#include <memory>
#include <string>
#include <vector>

#include "gatzk/algebra/field.hpp"

namespace gatzk::crypto
{
    struct G1Point
    {
        mcl::G1 value;

        std::string to_string() const;
    };

    struct G2Point
    {
        mcl::G2 value;

        std::string to_string() const;
    };

    struct PreparedG2
    {
        std::vector<mcl::Fp6> coeffs;
    };

    struct G1MSMProfile
    {
        std::size_t input_count = 0;
        std::size_t nonzero_scalar_count = 0;
        std::size_t effective_scalar_count = 0;
        bool zero_scalar_skipping = false;
        bool sparse_compaction = false;
        bool multithreaded = false;
    };

    struct G1MSMExecutionOptions
    {
        std::size_t max_worker_count = 0;

        bool stable_srs_bases = false;

        std::size_t stable_srs_capacity = 0;

        const std::uint64_t *device_montgomery_scalars = nullptr;

        std::size_t device_montgomery_scalar_capacity = 0;

        int device_montgomery_scalar_device = -1;

        bool device_montgomery_scalar_nonzero_count_known = false;

        std::size_t device_montgomery_scalar_nonzero_count = 0;
    };

    std::string backend_name();

    G1Point g1_zero();
    G2Point g2_zero();
    G1Point g1_add(const G1Point &lhs, const G1Point &rhs);
    G1Point g1_sub(const G1Point &lhs, const G1Point &rhs);
    G1Point g1_neg(const G1Point &point);
    G1Point g1_mul(const G1Point &point, const algebra::FieldElement &scalar);
    G1Point g1_multi_scalar_mul(const G1Point *points, const algebra::FieldElement *scalars, std::size_t count);
    G1Point g1_multi_scalar_mul_profiled(const G1Point *points, const algebra::FieldElement *scalars, std::size_t count, G1MSMProfile *profile,
                                         const G1MSMExecutionOptions *options = nullptr);
    G1Point g1_multi_scalar_mul(const std::vector<G1Point> &points, const std::vector<algebra::FieldElement> &scalars);
    void g1_parallel_msm_batch_workers(std::size_t worker_count, const std::function<void(std::size_t)> &worker);
    std::vector<G1Point> g1_mul_same_base_batch(const G1Point &point, const std::vector<algebra::FieldElement> &scalars);
    G2Point g2_add(const G2Point &lhs, const G2Point &rhs);
    G2Point g2_sub(const G2Point &lhs, const G2Point &rhs);
    G2Point g2_mul(const G2Point &point, const algebra::FieldElement &scalar);
    bool operator==(const G1Point &lhs, const G1Point &rhs);
    bool operator==(const G2Point &lhs, const G2Point &rhs);
    std::vector<std::uint8_t> serialize(const G1Point &point);
    std::vector<std::uint8_t> serialize(const G2Point &point);

    std::size_t serialize(const G1Point &point, std::uint8_t *output, std::size_t capacity);

    G1Point deserialize_g1(const std::uint8_t *bytes, std::size_t size);

    G1Point deserialize_g1(const std::vector<std::uint8_t> &bytes);

    G2Point deserialize_g2(const std::vector<std::uint8_t> &bytes);
    std::size_t serialized_size(const G1Point &point);
    std::size_t serialized_size(const G2Point &point);
    PreparedG2 prepare_g2(const G2Point &point);

    bool pairing_equal(const G1Point &lhs_a, const G2Point &rhs_a, const G1Point &lhs_b, const G2Point &rhs_b);

    bool pairing_equal_with_prepared_rhs_a(const G1Point &lhs_a, const PreparedG2 &rhs_a_prepared, const G1Point &lhs_b, const G2Point &rhs_b);

    bool pairing_product_is_one_mixed_prepared(const G1Point &lhs_dynamic, const G2Point &rhs_dynamic, const G1Point &lhs_prepared,
                                               const PreparedG2 &rhs_prepared);

}
