#include "gatzk/algebra/vector_ops.hpp"

#include <algorithm>
#include <cstdlib>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>

#include "gatzk/util/route2.hpp"

namespace gatzk::algebra {
#if GATZK_ENABLE_CUDA_BACKEND
FieldElement dot_product_cuda(
    const std::vector<FieldElement>& lhs,
    const std::vector<FieldElement>& rhs);
FieldElement dot_product_native_weights_cuda(
    const std::vector<FieldElement>& lhs,
    const PackedFieldBuffer& packed_rhs);
bool cuda_backend_runtime_available();
#endif

namespace {

AlgebraBackend parse_backend_env() {
    const char* value = std::getenv("GATZK_ALGEBRA_BACKEND");
    if (value == nullptr) {
        return AlgebraBackend::Cpu;
    }
    const std::string backend(value);
    if (backend == "cpu") {
        return AlgebraBackend::Cpu;
    }
    if (backend == "cuda") {
        return AlgebraBackend::Cuda;
    }
    throw std::runtime_error("unsupported GATZK_ALGEBRA_BACKEND value: " + backend);
}

bool cuda_dot_products_enabled() {
    const char* value = std::getenv("GATZK_ENABLE_CUDA_DOT_PRODUCTS");
    return value != nullptr && std::string(value) == "1";
}

bool should_use_cuda_dot_product(std::size_t size) {
    return configured_algebra_backend() == AlgebraBackend::Cuda
        && cuda_dot_products_enabled()
        && size >= 1024;
}

FieldElement dot_product_range(
    const std::vector<FieldElement>& lhs,
    const std::vector<FieldElement>& rhs,
    std::size_t begin,
    std::size_t end) {
    mcl::Fr sum;
    sum.clear();
    for (std::size_t i = begin; i < end; ++i) {
        mcl::Fr term;
        mcl::Fr::mul(term, lhs[i].native(), rhs[i].native());
        mcl::Fr::add(sum, sum, term);
    }
    return FieldElement::from_native(sum);
}

FieldElement dot_product_native_range(
    const std::vector<FieldElement>& lhs,
    const std::vector<mcl::Fr>& rhs,
    std::size_t begin,
    std::size_t end) {
    mcl::Fr sum;
    sum.clear();
    for (std::size_t i = begin; i < end; ++i) {
        mcl::Fr term;
        mcl::Fr::mul(term, lhs[i].native(), rhs[i]);
        mcl::Fr::add(sum, sum, term);
    }
    return FieldElement::from_native(sum);
}

}  // namespace

AlgebraBackend configured_algebra_backend() {
    static const auto backend = parse_backend_env();
    return backend;
}

std::string configured_algebra_backend_name() {
    return configured_algebra_backend() == AlgebraBackend::Cuda ? "cuda" : "cpu";
}

bool cuda_backend_available() {
#if GATZK_ENABLE_CUDA_BACKEND
    return cuda_backend_runtime_available();
#else
    return false;
#endif
}

FieldElement dot_product(
    const std::vector<FieldElement>& lhs,
    const std::vector<FieldElement>& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::runtime_error("dot product size mismatch");
    }
    if (lhs.empty()) {
        return FieldElement::zero();
    }

    if (should_use_cuda_dot_product(lhs.size())) {
#if GATZK_ENABLE_CUDA_BACKEND
        return dot_product_cuda(lhs, rhs);
#else
        throw std::runtime_error("CUDA algebra backend requested but this build was compiled without CUDA support");
#endif
    }

    const auto& route2 = util::route2_options();
    const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    if (!route2.parallel_fft || lhs.size() < 1024 || cpu_count == 1) {
        return dot_product_range(lhs, rhs, 0, lhs.size());
    }

    // The current protocol pipeline stores many witness polynomials directly in
    // evaluation form. For this codebase, the safe Route 2 landing point for a
    // "parallel FFT/NTT" switch is therefore the large domain dot-product and
    // sweep backend that dominates opening/quotient time, not a protocol-level
    // change to the domains or polynomial objects.
    const auto task_count = std::min<std::size_t>(cpu_count, lhs.size() / 512);
    if (task_count <= 1) {
        return dot_product_range(lhs, rhs, 0, lhs.size());
    }

    std::vector<std::future<FieldElement>> futures;
    futures.reserve(task_count);
    const auto chunk_size = (lhs.size() + task_count - 1) / task_count;
    for (std::size_t task = 0; task < task_count; ++task) {
        const auto begin = task * chunk_size;
        const auto end = std::min(lhs.size(), begin + chunk_size);
        if (begin >= end) {
            break;
        }
        futures.push_back(std::async(
            std::launch::async,
            [&lhs, &rhs, begin, end]() {
                return dot_product_range(lhs, rhs, begin, end);
            }));
    }

    FieldElement sum = FieldElement::zero();
    for (auto& future : futures) {
        sum += future.get();
    }
    return sum;
}

FieldElement dot_product_native_weights(
    const std::vector<FieldElement>& lhs,
    const std::vector<mcl::Fr>& rhs) {
    if (should_use_cuda_dot_product(lhs.size())) {
#if GATZK_ENABLE_CUDA_BACKEND
        const auto packed_rhs = pack_native_field_elements(rhs);
        return dot_product_native_weights_cuda(lhs, packed_rhs);
#else
        throw std::runtime_error("CUDA algebra backend requested but this build was compiled without CUDA support");
#endif
    }
    return dot_product_packed_native_weights(lhs, rhs, PackedFieldBuffer());
}

FieldElement dot_product_packed_native_weights(
    const std::vector<FieldElement>& lhs,
    const std::vector<mcl::Fr>& rhs,
    const PackedFieldBuffer& packed_rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::runtime_error("dot product size mismatch");
    }
    if (lhs.empty()) {
        return FieldElement::zero();
    }

    if (should_use_cuda_dot_product(lhs.size())) {
#if GATZK_ENABLE_CUDA_BACKEND
        if (packed_rhs.size() != rhs.size()) {
            throw std::runtime_error("dot product packed weight size mismatch");
        }
        return dot_product_native_weights_cuda(lhs, packed_rhs);
#else
        throw std::runtime_error("CUDA algebra backend requested but this build was compiled without CUDA support");
#endif
    }

    const auto& route2 = util::route2_options();
    const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    if (!route2.parallel_fft || lhs.size() < 1024 || cpu_count == 1) {
        return dot_product_native_range(lhs, rhs, 0, lhs.size());
    }

    const auto task_count = std::min<std::size_t>(cpu_count, lhs.size() / 512);
    if (task_count <= 1) {
        return dot_product_native_range(lhs, rhs, 0, lhs.size());
    }

    std::vector<std::future<FieldElement>> futures;
    futures.reserve(task_count);
    const auto chunk_size = (lhs.size() + task_count - 1) / task_count;
    for (std::size_t task = 0; task < task_count; ++task) {
        const auto begin = task * chunk_size;
        const auto end = std::min(lhs.size(), begin + chunk_size);
        if (begin >= end) {
            break;
        }
        futures.push_back(std::async(
            std::launch::async,
            [&lhs, &rhs, begin, end]() {
                return dot_product_native_range(lhs, rhs, begin, end);
            }));
    }

    FieldElement sum = FieldElement::zero();
    for (auto& future : futures) {
        sum += future.get();
    }
    return sum;
}

}  // namespace gatzk::algebra
