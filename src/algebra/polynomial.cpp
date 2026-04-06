#include "gatzk/algebra/polynomial.hpp"

#include <algorithm>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/util/route2.hpp"

namespace gatzk::algebra {
namespace {

FieldElement barycentric_sum_range(
    const std::vector<FieldElement>& values,
    const std::vector<FieldElement>& domain_points,
    const FieldElement& x,
    std::size_t begin,
    std::size_t end) {
    FieldElement sum = FieldElement::zero();
    for (std::size_t i = begin; i < end; ++i) {
        sum += values[i] * domain_points[i] / (x - domain_points[i]);
    }
    return sum;
}

}

std::shared_ptr<RootOfUnityDomain> RootOfUnityDomain::create(const std::string& name, std::size_t size) {
    if (size == 0 || (size & (size - 1U)) != 0U) {
        throw std::runtime_error("domain size must be a non-zero power of two");
    }
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, std::shared_ptr<RootOfUnityDomain>> cache;
    const auto cache_key = name + ":" + std::to_string(size);
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (const auto it = cache.find(cache_key); it != cache.end()) {
            return it->second;
        }
    }

    auto domain = std::make_shared<RootOfUnityDomain>();
    domain->name = name;
    domain->size = size;
    domain->omega = FieldElement::root_of_unity(size);
    domain->inv_size = FieldElement(static_cast<std::uint64_t>(size)).inv();
    domain->points_precomputed = size < (1ULL << 24);
    if (domain->points_precomputed) {
        domain->points.resize(size);
        domain->points_scaled_by_inv_size.resize(size);
        domain->points[0] = FieldElement::one();
        for (std::size_t i = 1; i < size; ++i) {
            domain->points[i] = domain->points[i - 1] * domain->omega;
        }
        for (std::size_t i = 0; i < size; ++i) {
            domain->points_scaled_by_inv_size[i] = domain->points[i] * domain->inv_size;
        }
    }

    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        cache.emplace(cache_key, domain);
    }
    return domain;
}

FieldElement RootOfUnityDomain::point_at(std::size_t index) const {
    if (points_precomputed) {
        return points.at(index);
    }
    return omega.pow(static_cast<std::uint64_t>(index));
}

FieldElement RootOfUnityDomain::point_scaled_by_inv_size_at(std::size_t index) const {
    if (points_precomputed) {
        return points_scaled_by_inv_size.at(index);
    }
    return point_at(index) * inv_size;
}

FieldElement RootOfUnityDomain::zero_polynomial_eval(const FieldElement& x) const {
    return x.pow(size) - FieldElement::one();
}

FieldElement RootOfUnityDomain::lagrange_basis_eval(std::size_t index, const FieldElement& x) const {
    const FieldElement xi = point_at(index);
    const FieldElement numerator = zero_polynomial_eval(x) * xi;
    return numerator * inv_size / (x - xi);
}

std::vector<mcl::Fr> RootOfUnityDomain::barycentric_weights_native(const FieldElement& x) const {
    std::vector<mcl::Fr> denominators(size);
    std::vector<mcl::Fr> prefixes(size);
    mcl::Fr accumulator = 1;
    FieldElement point = FieldElement::one();
    for (std::size_t i = 0; i < size; ++i) {
        const auto current_point = points_precomputed ? points[i] : point;
        mcl::Fr::sub(denominators[i], x.native(), current_point.native());
        prefixes[i] = accumulator;
        mcl::Fr::mul(accumulator, accumulator, denominators[i]);
        if (!points_precomputed) {
            point *= omega;
        }
    }

    mcl::Fr suffix_inverse;
    mcl::Fr::inv(suffix_inverse, accumulator);
    std::vector<mcl::Fr> weights(size);
    const auto zero_eval = zero_polynomial_eval(x).native();
    point = points_precomputed ? FieldElement::zero() : omega.pow(static_cast<std::uint64_t>(size - 1));
    const auto omega_inv = points_precomputed ? FieldElement::one() : omega.inv();
    for (std::size_t i = size; i-- > 0;) {
        mcl::Fr inverse;
        mcl::Fr::mul(inverse, suffix_inverse, prefixes[i]);
        const auto scaled_point = points_precomputed ? points_scaled_by_inv_size[i] : (point * inv_size);
        mcl::Fr::mul(weights[i], zero_eval, scaled_point.native());
        mcl::Fr::mul(weights[i], weights[i], inverse);
        mcl::Fr::mul(suffix_inverse, suffix_inverse, denominators[i]);
        if (!points_precomputed && i > 0) {
            point *= omega_inv;
        }
    }
    return weights;
}

std::vector<FieldElement> RootOfUnityDomain::barycentric_weights(const FieldElement& x) const {
    const auto native_weights = barycentric_weights_native(x);
    std::vector<FieldElement> weights(size, FieldElement::zero());
    for (std::size_t i = 0; i < size; ++i) {
        weights[i] = FieldElement::from_native(native_weights[i]);
    }
    return weights;
}

std::optional<std::size_t> RootOfUnityDomain::rotation_shift(
    const FieldElement& from,
    const FieldElement& to) const {
    if (from.is_zero() || to.is_zero()) {
        return from == to ? std::optional<std::size_t>(0) : std::nullopt;
    }

    
    const auto ratio = to / from;
    FieldElement power = FieldElement::one();
    for (std::size_t shift = 0; shift < size; ++shift) {
        if (power == ratio) {
            return shift;
        }
        power *= omega;
    }
    return std::nullopt;
}

Polynomial Polynomial::from_coefficients(const std::string& name, std::vector<FieldElement> coefficients) {
    Polynomial out;
    out.name = name;
    out.basis = PolynomialBasis::Coefficient;
    out.data = std::move(coefficients);
    return out;
}

Polynomial Polynomial::from_evaluations(
    const std::string& name,
    std::vector<FieldElement> evaluations,
    const std::shared_ptr<RootOfUnityDomain>& domain) {
    if (domain == nullptr || evaluations.size() != domain->size) {
        throw std::runtime_error("evaluation polynomial requires a matching domain");
    }
    Polynomial out;
    out.name = name;
    out.basis = PolynomialBasis::Evaluation;
    out.data = std::move(evaluations);
    out.domain = domain;
    return out;
}

FieldElement Polynomial::evaluate(const FieldElement& x) const {
    if (basis == PolynomialBasis::Coefficient) {
        return horner(data, x);
    }
    if (domain == nullptr) {
        throw std::runtime_error("evaluation polynomial is missing its domain");
    }
    if (domain->points_precomputed) {
        for (std::size_t i = 0; i < domain->points.size(); ++i) {
            if (domain->points[i] == x) {
                return data[i];
            }
        }
    } else if (const auto shift = domain->rotation_shift(FieldElement::one(), x); shift.has_value()) {
        if (*shift < data.size()) {
            return data[*shift];
        }
    }
    const auto& route2 = util::route2_options();
    
    const FieldElement zero_eval = domain->zero_polynomial_eval(x);
    FieldElement sum = FieldElement::zero();
    const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    if (domain->points_precomputed && route2.parallel_fft && data.size() >= 1024 && cpu_count > 1) {
        const auto task_count = std::min<std::size_t>(cpu_count, data.size() / 512);
        const auto chunk_size = task_count > 1 ? (data.size() + task_count - 1) / task_count : data.size();
        if (task_count > 1) {
            std::vector<std::future<FieldElement>> futures;
            futures.reserve(task_count);
            for (std::size_t task = 0; task < task_count; ++task) {
                const auto begin = task * chunk_size;
                const auto end = std::min(data.size(), begin + chunk_size);
                if (begin >= end) {
                    break;
                }
                futures.push_back(std::async(
                    std::launch::async,
                    [&, begin, end]() {
                        return barycentric_sum_range(data, domain->points, x, begin, end);
                    }));
            }
            for (auto& future : futures) {
                sum += future.get();
            }
            return zero_eval * domain->inv_size * sum;
        }
    }
    FieldElement point = FieldElement::one();
    for (std::size_t i = 0; i < data.size(); ++i) {
        const auto current_point = domain->points_precomputed ? domain->points[i] : point;
        sum += data[i] * current_point / (x - current_point);
        if (!domain->points_precomputed) {
            point *= domain->omega;
        }
    }
    return zero_eval * domain->inv_size * sum;
}

FieldElement horner(const std::vector<FieldElement>& coefficients, const FieldElement& x) {
    FieldElement out = FieldElement::zero();
    for (auto it = coefficients.rbegin(); it != coefficients.rend(); ++it) {
        out *= x;
        out += *it;
    }
    return out;
}

FieldElement interpolate_at(
    const std::vector<FieldElement>& points,
    const std::vector<FieldElement>& values,
    const FieldElement& x) {
    if (points.size() != values.size()) {
        throw std::runtime_error("interpolation points/value mismatch");
    }
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (points[i] == x) {
            return values[i];
        }
    }
    FieldElement total = FieldElement::zero();
    for (std::size_t i = 0; i < points.size(); ++i) {
        FieldElement basis = FieldElement::one();
        for (std::size_t j = 0; j < points.size(); ++j) {
            if (i == j) {
                continue;
            }
            basis *= (x - points[j]) / (points[i] - points[j]);
        }
        total += basis * values[i];
    }
    return total;
}

std::vector<FieldElement> flatten_matrix_coefficients(const std::vector<std::vector<FieldElement>>& matrix) {
    std::vector<FieldElement> out;
    for (const auto& row : matrix) {
        out.insert(out.end(), row.begin(), row.end());
    }
    return out;
}

}
