#include "gatzk/crypto/curve.hpp"

#include <array>
#include <algorithm>
#include <future>
#include <stdexcept>
#include <thread>

#include "gatzk/util/route2.hpp"

namespace gatzk::crypto {
namespace {

template <class Point>
std::vector<std::uint8_t> serialize_impl(const Point& point) {
    algebra::ensure_mcl_field_ready();
    std::array<std::uint8_t, 256> buffer{};
    const auto written = point.serialize(buffer.data(), buffer.size());
    if (written == 0) {
        throw std::runtime_error("failed to serialize mcl point");
    }
    return std::vector<std::uint8_t>(buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(written));
}

}  // namespace

std::string G1Point::to_string() const {
    algebra::ensure_mcl_field_ready();
    return value.serializeToHexStr();
}

std::string G2Point::to_string() const {
    algebra::ensure_mcl_field_ready();
    return value.serializeToHexStr();
}

std::string backend_name() {
    return GATZK_CRYPTO_BACKEND_NAME;
}

G1Point g1_zero() {
    G1Point out;
    algebra::ensure_mcl_field_ready();
    out.value.clear();
    return out;
}

G2Point g2_zero() {
    G2Point out;
    algebra::ensure_mcl_field_ready();
    out.value.clear();
    return out;
}

G1Point g1_add(const G1Point& lhs, const G1Point& rhs) {
    G1Point out;
    algebra::ensure_mcl_field_ready();
    mcl::G1::add(out.value, lhs.value, rhs.value);
    return out;
}

G1Point g1_sub(const G1Point& lhs, const G1Point& rhs) {
    G1Point out;
    algebra::ensure_mcl_field_ready();
    mcl::G1::sub(out.value, lhs.value, rhs.value);
    return out;
}

G1Point g1_mul(const G1Point& point, const algebra::FieldElement& scalar) {
    G1Point out;
    algebra::ensure_mcl_field_ready();
    mcl::G1::mul(out.value, point.value, scalar.native());
    return out;
}

std::vector<G1Point> g1_mul_same_base_batch(const G1Point& point, const std::vector<algebra::FieldElement>& scalars) {
    algebra::ensure_mcl_field_ready();
    std::vector<G1Point> out(scalars.size());
    if (scalars.empty()) {
        return out;
    }

    // This route only changes how the same fixed generator is multiplied by a
    // batch of scalars. The resulting commitment points are identical; the
    // feature flag only selects the low-level mcl path.
    if (!util::route2_options().fast_msm) {
        for (std::size_t i = 0; i < scalars.size(); ++i) {
            out[i] = g1_mul(point, scalars[i]);
        }
        return out;
    }

    std::vector<mcl::G1> points(scalars.size(), point.value);
    std::vector<mcl::Fr> native_scalars;
    native_scalars.reserve(scalars.size());
    for (const auto& scalar : scalars) {
        native_scalars.push_back(scalar.native());
    }

    const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    const bool run_parallel = cpu_count > 1 && scalars.size() >= 256;
    if (run_parallel) {
        const auto task_count = std::min<std::size_t>(cpu_count, scalars.size() / 128);
        if (task_count > 1) {
            const auto chunk_size = (scalars.size() + task_count - 1) / task_count;
            std::vector<std::future<std::vector<mcl::G1>>> futures;
            futures.reserve(task_count);
            for (std::size_t task = 0; task < task_count; ++task) {
                const auto begin = task * chunk_size;
                const auto end = std::min<std::size_t>(scalars.size(), begin + chunk_size);
                if (begin >= end) {
                    break;
                }
                futures.push_back(std::async(
                    std::launch::async,
                    [&, begin, end]() {
                        std::vector<mcl::G1> chunk(end - begin, point.value);
                        mcl::G1::mulEach(chunk.data(), native_scalars.data() + begin, chunk.size());
                        return chunk;
                    }));
            }
            std::size_t begin = 0;
            for (auto& future : futures) {
                auto chunk = future.get();
                for (std::size_t i = 0; i < chunk.size(); ++i) {
                    out[begin + i].value = chunk[i];
                }
                begin += chunk.size();
            }
            return out;
        }
    }

    // This still computes the same batch of fixed-base points `[s_i]G`.
    // The route2 fast MSM path only changes the low-level same-base batch
    // multiplication primitive, not the resulting protocol commitments.
    mcl::G1::mulEach(points.data(), native_scalars.data(), points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        out[i].value = points[i];
    }
    return out;
}

G2Point g2_add(const G2Point& lhs, const G2Point& rhs) {
    G2Point out;
    algebra::ensure_mcl_field_ready();
    mcl::G2::add(out.value, lhs.value, rhs.value);
    return out;
}

G2Point g2_sub(const G2Point& lhs, const G2Point& rhs) {
    G2Point out;
    algebra::ensure_mcl_field_ready();
    mcl::G2::sub(out.value, lhs.value, rhs.value);
    return out;
}

G2Point g2_mul(const G2Point& point, const algebra::FieldElement& scalar) {
    G2Point out;
    algebra::ensure_mcl_field_ready();
    mcl::G2::mul(out.value, point.value, scalar.native());
    return out;
}

bool operator==(const G1Point& lhs, const G1Point& rhs) {
    algebra::ensure_mcl_field_ready();
    return lhs.value == rhs.value;
}

bool operator==(const G2Point& lhs, const G2Point& rhs) {
    algebra::ensure_mcl_field_ready();
    return lhs.value == rhs.value;
}

std::vector<std::uint8_t> serialize(const G1Point& point) {
    return serialize_impl(point.value);
}

std::vector<std::uint8_t> serialize(const G2Point& point) {
    return serialize_impl(point.value);
}

std::size_t serialized_size(const G1Point& point) {
    return serialize(point).size();
}

std::size_t serialized_size(const G2Point& point) {
    return serialize(point).size();
}

PreparedG2 prepare_g2(const G2Point& point) {
    algebra::ensure_mcl_field_ready();
    PreparedG2 prepared;
    mcl::precomputeG2(prepared.coeffs, point.value);
    return prepared;
}

bool pairing_equal(
    const G1Point& lhs_a,
    const G2Point& rhs_a,
    const G1Point& lhs_b,
    const G2Point& rhs_b) {
    algebra::ensure_mcl_field_ready();
    mcl::GT eval_a;
    mcl::GT eval_b;
    mcl::pairing(eval_a, lhs_a.value, rhs_a.value);
    mcl::pairing(eval_b, lhs_b.value, rhs_b.value);
    return eval_a == eval_b;
}

bool pairing_product_is_one_mixed_prepared(
    const G1Point& lhs_dynamic,
    const G2Point& rhs_dynamic,
    const G1Point& lhs_prepared,
    const PreparedG2& rhs_prepared) {
    algebra::ensure_mcl_field_ready();
    mcl::Fp12 miller_product;
    mcl::precomputedMillerLoop2mixed(
        miller_product,
        lhs_dynamic.value,
        rhs_dynamic.value,
        lhs_prepared.value,
        rhs_prepared.coeffs);
    mcl::finalExp(miller_product, miller_product);
    return miller_product.isOne();
}

}  // namespace gatzk::crypto
