#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "gatzk/algebra/field.hpp"

namespace gatzk::algebra
{
    struct RootOfUnityDomain
    {
        std::string name;
        std::size_t size = 0;
        FieldElement omega = FieldElement::one();
        FieldElement inv_size = FieldElement::one();
        std::vector<FieldElement> points;
        std::vector<FieldElement> points_scaled_by_inv_size;
        bool points_precomputed = true;

        static std::shared_ptr<RootOfUnityDomain> create(const std::string &name, std::size_t size);

        FieldElement point_at(std::size_t index) const;

        FieldElement point_scaled_by_inv_size_at(std::size_t index) const;

        FieldElement zero_polynomial_eval(const FieldElement &x) const;

        FieldElement lagrange_basis_eval(std::size_t index, const FieldElement &x) const;

        std::vector<mcl::Fr> barycentric_weights_native(const FieldElement &x) const;
        std::vector<FieldElement> barycentric_weights(const FieldElement &x) const;
        std::optional<std::size_t> rotation_shift(const FieldElement &from, const FieldElement &to) const;
    };

    class CudaHostTransferGuard
    {
    public:
        virtual ~CudaHostTransferGuard() = default;

        virtual void wait() const = 0;
    };

    struct CudaResidentCoefficientBuffer
    {
        std::shared_ptr<void> allocation;

        const std::uint64_t *device_montgomery_limbs = nullptr;

        std::size_t capacity = 0;

        std::size_t logical_count = 0;

        std::size_t nonzero_count = 0;

        bool nonzero_count_known = true;

        int device = -1;

        std::shared_ptr<const CudaHostTransferGuard> host_transfer;
    };

    enum class PolynomialBasis
    {
        Coefficient,
        Evaluation
    };

    struct Polynomial
    {
        std::string name;

        PolynomialBasis basis = PolynomialBasis::Coefficient;

        std::vector<FieldElement> data;

        std::shared_ptr<const std::vector<FieldElement>> shared_data;

        std::shared_ptr<RootOfUnityDomain> domain;

        std::vector<FieldElement> zero_domain_blinder;

        std::shared_ptr<const CudaResidentCoefficientBuffer> cuda_resident_coefficients;

        static Polynomial from_coefficients(const std::string &name, std::vector<FieldElement> coefficients);
        static Polynomial from_evaluations(const std::string &name, std::vector<FieldElement> evaluations, const std::shared_ptr<RootOfUnityDomain> &domain);
        static Polynomial from_shared_evaluations(const std::string &name, std::shared_ptr<const std::vector<FieldElement>> evaluations,
                                                  const std::shared_ptr<RootOfUnityDomain> &domain);

        const std::vector<FieldElement> &values() const;
        std::size_t size() const;
        const FieldElement &value_at(std::size_t index) const;

        void set_zero_domain_blinder(std::vector<FieldElement> coefficients);
        bool is_zero_domain_blinded() const;
        FieldElement semantic_evaluate(const FieldElement &x) const;

        FieldElement evaluate(const FieldElement &x) const;
    };

    FieldElement horner(const std::vector<FieldElement> &coefficients, const FieldElement &x);
    FieldElement interpolate_at(const std::vector<FieldElement> &points, const std::vector<FieldElement> &values, const FieldElement &x);

    std::vector<FieldElement> materialize_blinded_coefficients(const Polynomial &polynomial, std::size_t configured_degree);

    std::vector<FieldElement> coefficient_taylor_shift(const std::vector<FieldElement> &coefficients, std::size_t configured_degree, const FieldElement &delta);

    void release_taylor_shift_cache();

    std::vector<FieldElement> flatten_matrix_coefficients(const std::vector<std::vector<FieldElement>> &matrix);

}
