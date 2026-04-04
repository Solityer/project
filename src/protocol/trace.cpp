#include "gatzk/protocol/trace.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <future>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/crypto/kzg.hpp"
#include "gatzk/crypto/transcript.hpp"
#include "gatzk/model/gat.hpp"
#include "gatzk/protocol/challenges.hpp"
#include "gatzk/protocol/lookup.hpp"
#include "gatzk/protocol/psq.hpp"
#include "gatzk/protocol/zkmap.hpp"
#include "gatzk/util/route2.hpp"

namespace gatzk::protocol {

#if GATZK_ENABLE_CUDA_BACKEND
std::vector<std::size_t> lookup_histogram_indices_cuda(
    const std::vector<std::size_t>& indices,
    std::size_t domain_size);
std::vector<algebra::FieldElement> build_max_counter_state_cuda(
    const std::vector<algebra::FieldElement>& s_max,
    const std::vector<algebra::FieldElement>& q_new);
#endif

namespace {

using algebra::FieldElement;
using algebra::Polynomial;
using data::Edge;
using model::Matrix;
using Clock = std::chrono::steady_clock;

double elapsed_ms(const Clock::time_point& start, const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::vector<FieldElement> padded_column(const std::vector<FieldElement>& values, std::size_t size) {
    std::vector<FieldElement> out(size, FieldElement::zero());
    std::copy(values.begin(), values.end(), out.begin());
    return out;
}

Polynomial coeff_poly(const std::string& name, const std::vector<FieldElement>& values) {
    return Polynomial::from_coefficients(name, values);
}

void add_metric(double* metric, const Clock::time_point& start) {
    if (metric != nullptr) {
        *metric += elapsed_ms(start, Clock::now());
    }
}

bool cuda_lookup_histogram_enabled() {
#if GATZK_ENABLE_CUDA_BACKEND
    const char* flag = std::getenv("GATZK_ENABLE_CUDA_LOOKUP_HISTOGRAM");
    return flag != nullptr
        && std::string(flag) == "1"
        && algebra::cuda_backend_available();
#else
    return false;
#endif
}

bool cuda_max_counter_enabled() {
#if GATZK_ENABLE_CUDA_BACKEND
    const char* flag = std::getenv("GATZK_ENABLE_CUDA_MAX_COUNTER");
    return flag != nullptr
        && std::string(flag) == "1"
        && algebra::cuda_backend_available();
#else
    return false;
#endif
}

template <typename Fn>
void parallel_for_ranges(std::size_t item_count, std::size_t min_parallel_items, Fn&& fn) {
    const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    const auto task_count =
        (item_count >= min_parallel_items && cpu_count > 1)
        ? std::min<std::size_t>(cpu_count, item_count)
        : 1;
    if (task_count <= 1) {
        fn(0, item_count);
        return;
    }

    const auto chunk_size = (item_count + task_count - 1) / task_count;
    std::vector<std::future<void>> futures;
    futures.reserve(task_count);
    for (std::size_t task = 0; task < task_count; ++task) {
        const auto begin = task * chunk_size;
        const auto end = std::min(item_count, begin + chunk_size);
        if (begin >= end) {
            break;
        }
        futures.push_back(std::async(
            std::launch::async,
            [&fn, begin, end]() {
                fn(begin, end);
            }));
    }
    for (auto& future : futures) {
        future.get();
    }
}

std::vector<FieldElement> powers(const FieldElement& base, std::size_t count) {
    std::vector<FieldElement> out(count, FieldElement::one());
    for (std::size_t i = 1; i < count; ++i) {
        out[i] = out[i - 1] * base;
    }
    return out;
}

std::vector<FieldElement> strided_powers(const FieldElement& base, std::size_t count, std::size_t stride) {
    std::vector<FieldElement> out(count, FieldElement::one());
    const auto step = base.pow(static_cast<std::uint64_t>(stride));
    for (std::size_t i = 1; i < count; ++i) {
        out[i] = out[i - 1] * step;
    }
    return out;
}

std::vector<FieldElement> weighted_column_sum(
    const Matrix& matrix,
    const std::vector<FieldElement>& row_weights) {
    if (matrix.size() != row_weights.size()) {
        throw std::runtime_error("weighted column sum input size mismatch");
    }
    if (matrix.empty()) {
        return {};
    }
    std::vector<mcl::Fr> native_out(matrix.front().size());
    for (auto& value : native_out) {
        value.clear();
    }
    for (std::size_t i = 0; i < matrix.size(); ++i) {
        const auto& weight = row_weights[i].native();
        for (std::size_t j = 0; j < matrix[i].size(); ++j) {
            mcl::Fr term;
            mcl::Fr::mul(term, matrix[i][j].native(), weight);
            mcl::Fr::add(native_out[j], native_out[j], term);
        }
    }
    std::vector<FieldElement> out(matrix.front().size(), FieldElement::zero());
    for (std::size_t j = 0; j < native_out.size(); ++j) {
        out[j] = FieldElement::from_native(native_out[j]);
    }
    return out;
}

const std::vector<mcl::Fr>& cached_barycentric_weights_native(
    const std::shared_ptr<algebra::RootOfUnityDomain>& domain,
    const FieldElement& point) {
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, std::shared_ptr<std::vector<mcl::Fr>>> cache;
    const auto cache_key =
        domain->name + ":" + std::to_string(domain->size) + ":" + point.to_string();
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (const auto it = cache.find(cache_key); it != cache.end()) {
            return *it->second;
        }
    }

    auto weights = std::make_shared<std::vector<mcl::Fr>>(domain->barycentric_weights_native(point));
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        const auto [it, _] = cache.emplace(cache_key, weights);
        return *it->second;
    }
}

std::vector<FieldElement> weighted_column_sum_native(
    const Matrix& matrix,
    const std::vector<mcl::Fr>& row_weights,
    std::size_t row_count) {
    if (matrix.empty()) {
        return {};
    }
    if (row_count > matrix.size() || row_count > row_weights.size()) {
        throw std::runtime_error("weighted column sum native input size mismatch");
    }
    const auto column_count = matrix.front().size();
    std::vector<mcl::Fr> native_out(column_count);
    for (auto& value : native_out) {
        value.clear();
    }

    const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    if (row_count >= 256 && column_count >= 4 && cpu_count > 1) {
        const auto task_count = std::min<std::size_t>(cpu_count, row_count);
        const auto chunk_size = (row_count + task_count - 1) / task_count;
        std::vector<std::future<std::vector<mcl::Fr>>> futures;
        futures.reserve(task_count);
        for (std::size_t task = 0; task < task_count; ++task) {
            const auto begin = task * chunk_size;
            const auto end = std::min(row_count, begin + chunk_size);
            if (begin >= end) {
                break;
            }
            futures.push_back(std::async(
                std::launch::async,
                [&matrix, &row_weights, begin, end, column_count]() {
                    std::vector<mcl::Fr> partial(column_count);
                    for (auto& value : partial) {
                        value.clear();
                    }
                    for (std::size_t row = begin; row < end; ++row) {
                        const auto& weight = row_weights[row];
                        for (std::size_t column = 0; column < column_count; ++column) {
                            mcl::Fr term;
                            mcl::Fr::mul(term, matrix[row][column].native(), weight);
                            mcl::Fr::add(partial[column], partial[column], term);
                        }
                    }
                    return partial;
                }));
        }
        for (auto& future : futures) {
            const auto partial = future.get();
            for (std::size_t column = 0; column < column_count; ++column) {
                mcl::Fr::add(native_out[column], native_out[column], partial[column]);
            }
        }
    } else {
        for (std::size_t row = 0; row < row_count; ++row) {
            const auto& weight = row_weights[row];
            for (std::size_t column = 0; column < column_count; ++column) {
                mcl::Fr term;
                mcl::Fr::mul(term, matrix[row][column].native(), weight);
                mcl::Fr::add(native_out[column], native_out[column], term);
            }
        }
    }

    std::vector<FieldElement> out(column_count, FieldElement::zero());
    for (std::size_t column = 0; column < column_count; ++column) {
        out[column] = FieldElement::from_native(native_out[column]);
    }
    return out;
}

std::vector<FieldElement> linear_form_by_powers(
    const std::vector<std::vector<FieldElement>>& matrix,
    const std::vector<FieldElement>& column_powers) {
    if (matrix.empty()) {
        return {};
    }
    std::vector<FieldElement> out(matrix.size(), FieldElement::zero());
    for (std::size_t i = 0; i < matrix.size(); ++i) {
        mcl::Fr sum;
        sum.clear();
        for (std::size_t j = 0; j < matrix[i].size(); ++j) {
            mcl::Fr term;
            mcl::Fr::mul(term, matrix[i][j].native(), column_powers[j].native());
            mcl::Fr::add(sum, sum, term);
        }
        out[i] = FieldElement::from_native(sum);
    }
    return out;
}

std::vector<FieldElement> build_route_node_accumulator(
    const std::vector<FieldElement>& table,
    const std::vector<FieldElement>& multiplicity,
    std::size_t valid_length,
    std::size_t domain_size,
    const FieldElement& beta) {
    if (table.size() != domain_size || multiplicity.size() != domain_size) {
        throw std::runtime_error("route node accumulator input size mismatch");
    }
    std::vector<FieldElement> out(domain_size, FieldElement::zero());
    if (domain_size == 0 || valid_length == 0) {
        return out;
    }

    std::vector<FieldElement> denominators;
    denominators.reserve(valid_length);
    for (std::size_t i = 0; i < valid_length; ++i) {
        denominators.push_back(table[i] + beta);
    }
    const auto inverses = algebra::batch_invert(denominators);
    mcl::Fr running;
    running.clear();
    for (std::size_t i = 0; i < valid_length; ++i) {
        mcl::Fr term;
        mcl::Fr::mul(term, multiplicity[i].native(), inverses[i].native());
        mcl::Fr::add(running, running, term);
        out[i + 1] = FieldElement::from_native(running);
    }
    for (std::size_t i = valid_length + 1; i < domain_size; ++i) {
        out[i] = out[valid_length];
    }
    return out;
}

std::vector<FieldElement> build_route_edge_accumulator(
    const std::vector<FieldElement>& query,
    std::size_t valid_length,
    std::size_t domain_size,
    const FieldElement& beta) {
    if (query.size() != domain_size) {
        throw std::runtime_error("route edge accumulator input size mismatch");
    }
    std::vector<FieldElement> out(domain_size, FieldElement::zero());
    if (domain_size == 0 || valid_length == 0) {
        return out;
    }

    std::vector<FieldElement> denominators;
    denominators.reserve(valid_length);
    for (std::size_t i = 0; i < valid_length; ++i) {
        denominators.push_back(query[i] + beta);
    }
    const auto inverses = algebra::batch_invert(denominators);
    mcl::Fr running;
    running.clear();
    for (std::size_t i = 0; i < valid_length; ++i) {
        mcl::Fr::add(running, running, inverses[i].native());
        out[i + 1] = FieldElement::from_native(running);
    }
    for (std::size_t i = valid_length + 1; i < domain_size; ++i) {
        out[i] = out[valid_length];
    }
    return out;
}

enum class DynamicCommitmentKind {
    Column,
    Matrix,
};

struct DynamicCommitmentSpec {
    std::string name;
    DynamicCommitmentKind kind = DynamicCommitmentKind::Column;
    std::vector<FieldElement> column_values;
    std::shared_ptr<algebra::RootOfUnityDomain> domain;
    const Matrix* matrix = nullptr;
    bool retain_polynomial = true;
};

struct PreparedDynamicCommitment {
    std::string name;
    DynamicCommitmentKind kind = DynamicCommitmentKind::Column;
    std::optional<std::vector<FieldElement>> column_export;
    std::optional<Matrix> matrix_export;
    std::optional<Polynomial> polynomial;
    std::optional<FieldElement> direct_tau_evaluation;
    crypto::Commitment commitment;
};

DynamicCommitmentSpec column_spec(
    const std::string& name,
    std::vector<FieldElement> values,
    const std::shared_ptr<algebra::RootOfUnityDomain>& domain) {
    DynamicCommitmentSpec spec;
    spec.name = name;
    spec.kind = DynamicCommitmentKind::Column;
    spec.column_values = std::move(values);
    spec.domain = domain;
    return spec;
}

DynamicCommitmentSpec matrix_spec(const std::string& name, const Matrix& matrix) {
    DynamicCommitmentSpec spec;
    spec.name = name;
    spec.kind = DynamicCommitmentKind::Matrix;
    spec.matrix = &matrix;
    return spec;
}

DynamicCommitmentSpec matrix_commitment_only_spec(const std::string& name, const Matrix& matrix) {
    auto spec = matrix_spec(name, matrix);
    // These matrix witnesses are committed and absorbed exactly as before, but
    // they are never opened again as polynomial objects inside the protocol.
    // Under the trace-layout route we can therefore keep only the commitment
    // and tau evaluation, avoiding a transient flattened coefficient buffer
    // without changing any proof-visible object or verifier check.
    spec.retain_polynomial = false;
    return spec;
}

FieldElement row_polynomial_at_point(const std::vector<FieldElement>& row, const FieldElement& point) {
    mcl::Fr out;
    out.clear();
    for (std::size_t column = row.size(); column-- > 0;) {
        mcl::Fr::mul(out, out, point.native());
        mcl::Fr::add(out, out, row[column].native());
    }
    return FieldElement::from_native(out);
}

FieldElement matrix_row_major_evaluation(const Matrix& matrix, const FieldElement& point) {
    if (matrix.empty()) {
        return FieldElement::zero();
    }
    if (matrix.front().empty()) {
        return FieldElement::zero();
    }

    const auto& route2 = util::route2_options();
    const auto row_count = matrix.size();
    const auto column_count = matrix.front().size();
    const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    std::vector<FieldElement> row_evaluations(row_count, FieldElement::zero());

    // This preserves the exact coefficient polynomial defined by row-major
    // flattening in the spec. The engineering change is only that we evaluate
    // each row block directly at the evaluation point and then fold rows by
    // point^cols, so matrix
    // commitments no longer need a temporary flattened coefficient vector.
    if (route2.trace_layout_upgrade && row_count >= 8 && cpu_count > 1) {
        const auto task_count = std::min<std::size_t>(cpu_count, row_count);
        const auto chunk_size = (row_count + task_count - 1) / task_count;
        std::vector<std::future<void>> futures;
        futures.reserve(task_count);
        for (std::size_t task = 0; task < task_count; ++task) {
            const auto begin = task * chunk_size;
            const auto end = std::min(row_count, begin + chunk_size);
            if (begin >= end) {
                break;
            }
            futures.push_back(std::async(
                std::launch::async,
                [&matrix, &row_evaluations, &point, begin, end]() {
                    for (std::size_t row = begin; row < end; ++row) {
                        row_evaluations[row] = row_polynomial_at_point(matrix[row], point);
                    }
                }));
        }
        for (auto& future : futures) {
            future.get();
        }
    } else {
        for (std::size_t row = 0; row < row_count; ++row) {
            row_evaluations[row] = row_polynomial_at_point(matrix[row], point);
        }
    }

    const auto row_stride = point.pow(static_cast<std::uint64_t>(column_count));
    mcl::Fr out;
    out.clear();
    for (std::size_t row = row_count; row-- > 0;) {
        mcl::Fr::mul(out, out, row_stride.native());
        mcl::Fr::add(out, out, row_evaluations[row].native());
    }
    return FieldElement::from_native(out);
}

FieldElement matrix_row_major_tau_evaluation(const Matrix& matrix, const FieldElement& tau) {
    return matrix_row_major_evaluation(matrix, tau);
}

void fill_feature_lookup_rows(
    const std::vector<FieldElement>& row_ids,
    const std::vector<FieldElement>& feature_indices,
    const Matrix& features,
    const FieldElement& eta_feature_index,
    const FieldElement& eta_feature_value,
    std::vector<FieldElement>& out) {
    if (features.empty()) {
        return;
    }
    const auto row_count = features.size();
    const auto row_width = feature_indices.size();
    parallel_for_ranges(
        row_count,
        32,
        [&](std::size_t begin, std::size_t end) {
            for (std::size_t row = begin; row < end; ++row) {
                const auto base = row * row_width;
                const auto& row_id = row_ids[row];
                const auto& feature_row = features[row];
                for (std::size_t column = 0; column < row_width; ++column) {
                    out[base + column] =
                        row_id
                        + eta_feature_index * feature_indices[column]
                        + eta_feature_value * feature_row[column];
                }
            }
        });
}

void fill_repeated_row_values(
    const std::vector<FieldElement>& row_values,
    std::size_t row_width,
    std::vector<FieldElement>& out) {
    if (row_width == 0 || row_values.empty()) {
        return;
    }
    parallel_for_ranges(
        row_values.size(),
        32,
        [&](std::size_t begin, std::size_t end) {
            for (std::size_t row = begin; row < end; ++row) {
                std::fill_n(
                    out.begin() + static_cast<std::ptrdiff_t>(row * row_width),
                    static_cast<std::ptrdiff_t>(row_width),
                    row_values[row]);
            }
        });
}

PreparedDynamicCommitment materialize_commitment(
    DynamicCommitmentSpec spec,
    const crypto::KZGKeyPair& key,
    bool keep_trace_payloads) {
    PreparedDynamicCommitment out;
    out.name = spec.name;
    out.kind = spec.kind;
    if (spec.kind == DynamicCommitmentKind::Column) {
        // Export payload retention is an engineering choice only. When
        // dump_trace=false we still commit to the exact same polynomial, but we
        // skip keeping a second copy of the witness column solely for file dump.
        if (keep_trace_payloads) {
            out.column_export = spec.column_values;
        }
        out.polynomial.emplace(Polynomial::from_evaluations(spec.name, std::move(spec.column_values), spec.domain));
        return out;
    }

    if (spec.matrix == nullptr) {
        throw std::runtime_error("matrix commitment is missing matrix input");
    }
    if (keep_trace_payloads) {
        out.matrix_export = *spec.matrix;
    }
    if (!spec.retain_polynomial && util::route2_options().trace_layout_upgrade) {
        out.direct_tau_evaluation = matrix_row_major_tau_evaluation(*spec.matrix, key.tau);
        return out;
    }
    out.polynomial.emplace(coeff_poly(spec.name, algebra::flatten_matrix_coefficients(*spec.matrix)));
    return out;
}

void record_dynamic_commit_totals(RunMetrics* metrics) {
    if (metrics == nullptr) {
        return;
    }
    metrics->commit_dynamic_ms =
        metrics->dynamic_commit_input_ms
        + metrics->dynamic_polynomial_materialization_ms
        + metrics->dynamic_commit_pack_ms
        + metrics->dynamic_fft_ms
        + metrics->dynamic_domain_convert_ms
        + metrics->dynamic_copy_convert_ms
        + metrics->dynamic_commit_msm_ms
        + metrics->dynamic_bundle_finalize_ms;
    metrics->dynamic_commit_finalize_ms =
        metrics->dynamic_copy_convert_ms + metrics->dynamic_bundle_finalize_ms;
}

void add_dynamic_commitment_batch(
    TraceArtifacts& trace,
    std::vector<DynamicCommitmentSpec> specs,
    const crypto::KZGKeyPair& key,
    bool keep_trace_payloads,
    RunMetrics* metrics = nullptr) {
    if (specs.empty()) {
        return;
    }

    // Commitment order is still finalized in the original transcript order.
    // The only optimization here is to batch independent materialization and
    // fixed-base commit work inside one transcript stage. The protocol-visible
    // commitment set and transcript order are unchanged.
    const auto input_start = Clock::now();
    const bool run_parallel = specs.size() > 1 && std::thread::hardware_concurrency() > 1;
    add_metric(metrics != nullptr ? &metrics->dynamic_commit_input_ms : nullptr, input_start);

    std::vector<PreparedDynamicCommitment> prepared;
    prepared.reserve(specs.size());
    const auto materialize_start = Clock::now();
    if (run_parallel) {
        std::vector<std::future<PreparedDynamicCommitment>> futures;
        futures.reserve(specs.size());
        for (auto& spec : specs) {
            futures.push_back(std::async(
                std::launch::async,
                [spec = std::move(spec), &key, keep_trace_payloads]() mutable {
                    return materialize_commitment(std::move(spec), key, keep_trace_payloads);
                }));
        }
        for (auto& future : futures) {
            prepared.push_back(future.get());
        }
    } else {
        for (auto& spec : specs) {
            prepared.push_back(materialize_commitment(std::move(spec), key, keep_trace_payloads));
        }
    }
    add_metric(metrics != nullptr ? &metrics->dynamic_polynomial_materialization_ms : nullptr, materialize_start);

    const auto pack_start = Clock::now();
    std::vector<std::pair<std::string, const algebra::Polynomial*>> named_polynomials;
    std::vector<std::size_t> polynomial_indices;
    std::vector<std::pair<std::string, FieldElement>> direct_tau_evaluations;
    std::vector<std::size_t> direct_tau_indices;
    named_polynomials.reserve(prepared.size());
    direct_tau_evaluations.reserve(prepared.size());
    for (std::size_t i = 0; i < prepared.size(); ++i) {
        auto& item = prepared[i];
        if (item.polynomial.has_value()) {
            named_polynomials.push_back({item.name, &*item.polynomial});
            polynomial_indices.push_back(i);
        } else if (item.direct_tau_evaluation.has_value()) {
            direct_tau_evaluations.push_back({item.name, *item.direct_tau_evaluation});
            direct_tau_indices.push_back(i);
        } else {
            throw std::runtime_error("dynamic commitment is missing both polynomial and direct tau evaluation");
        }
    }
    add_metric(metrics != nullptr ? &metrics->dynamic_commit_pack_ms : nullptr, pack_start);

    crypto::CommitBatchProfile polynomial_profile;
    crypto::CommitBatchProfile direct_tau_profile;
    if (!named_polynomials.empty()) {
        const auto commitments = crypto::KZG::commit_batch(named_polynomials, key, &polynomial_profile);
        for (std::size_t i = 0; i < commitments.size(); ++i) {
            prepared[polynomial_indices[i]].commitment = commitments[i];
        }
    }
    if (!direct_tau_evaluations.empty()) {
        const auto commitments = crypto::KZG::commit_tau_evaluation_batch(direct_tau_evaluations, key, &direct_tau_profile);
        for (std::size_t i = 0; i < commitments.size(); ++i) {
            prepared[direct_tau_indices[i]].commitment = commitments[i];
        }
    }
    if (metrics != nullptr) {
        metrics->dynamic_domain_convert_ms += polynomial_profile.tau_eval_ms;
        metrics->dynamic_commit_msm_ms += polynomial_profile.msm_ms + direct_tau_profile.msm_ms;
    }

    const auto copy_start = Clock::now();
    for (auto& item : prepared) {
        if (item.column_export.has_value()) {
            trace.columns[item.name] = std::move(*item.column_export);
        }
        if (item.matrix_export.has_value()) {
            trace.matrices[item.name] = std::move(*item.matrix_export);
        }
        if (item.polynomial.has_value()) {
            if ((*item.polynomial).domain != nullptr) {
                trace.polynomial_domains[item.name] = (*item.polynomial).domain->name;
            }
            trace.polynomials[item.name] = std::move(*item.polynomial);
        }
    }
    add_metric(metrics != nullptr ? &metrics->dynamic_copy_convert_ms : nullptr, copy_start);

    const auto finalize_start = Clock::now();
    for (auto& item : prepared) {
        trace.commitments[item.name] = std::move(item.commitment);
        trace.commitment_order.push_back(item.name);
    }
    add_metric(metrics != nullptr ? &metrics->dynamic_bundle_finalize_ms : nullptr, finalize_start);
    record_dynamic_commit_totals(metrics);
}

const std::vector<FieldElement>& eval_data(const ProtocolContext& context, const std::string& name) {
    return context.public_polynomials.at(name).data;
}

FieldElement exp_map(const FieldElement& delta, std::size_t range_size) {
    return FieldElement(range_size - delta.value());
}

FieldElement evaluate_bias_fold(
    const std::vector<FieldElement>& bias,
    std::size_t rows,
    const FieldElement& point) {
    FieldElement out = FieldElement::zero();
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < bias.size(); ++j) {
            out += bias[j] * point.pow(static_cast<std::uint64_t>(i * bias.size() + j));
        }
    }
    return out;
}

FieldElement quantize_float(double value) {
    return FieldElement::from_signed(static_cast<std::int64_t>(
        value >= 0.0 ? value * 16.0 + 0.5 : value * 16.0 - 0.5));
}

std::vector<FieldElement> quantize_vector(const std::vector<double>& values) {
    std::vector<FieldElement> out(values.size(), FieldElement::zero());
    for (std::size_t i = 0; i < values.size(); ++i) {
        out[i] = quantize_float(values[i]);
    }
    return out;
}

Matrix quantize_matrix(const model::FloatMatrix& matrix) {
    Matrix out(matrix.size());
    for (std::size_t row = 0; row < matrix.size(); ++row) {
        out[row] = quantize_vector(matrix[row]);
    }
    return out;
}

struct HiddenHeadStaticQuantizedParameters {
    Matrix seq_kernel;
    std::vector<FieldElement> attn_src;
    std::vector<FieldElement> attn_dst;
};

const HiddenHeadStaticQuantizedParameters& cached_quantized_hidden_head_parameters(
    const ProtocolContext& context,
    std::size_t head_index) {
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, std::shared_ptr<HiddenHeadStaticQuantizedParameters>> cache;
    const auto& head = context.model.hidden_heads.at(head_index);
    const auto cache_key =
        context.config.checkpoint_bundle
        + ":" + context.config.dataset
        + ":" + std::to_string(context.config.seed)
        + ":hidden:" + std::to_string(head_index)
        + ":" + std::to_string(head.seq_kernel_fp.size())
        + ":" + std::to_string(head.seq_kernel_fp.empty() ? 0 : head.seq_kernel_fp.front().size());
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (const auto it = cache.find(cache_key); it != cache.end()) {
            return *it->second;
        }
    }

    auto packed = std::make_shared<HiddenHeadStaticQuantizedParameters>();
    packed->seq_kernel = quantize_matrix(head.seq_kernel_fp);
    packed->attn_src = quantize_vector(head.attn_src_kernel_fp);
    packed->attn_dst = quantize_vector(head.attn_dst_kernel_fp);
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        const auto [it, _] = cache.emplace(cache_key, packed);
        return *it->second;
    }
}

struct HiddenQuantizedWitness {
    Matrix h_prime;
    Matrix h_agg_pre;
    Matrix h_agg;
    std::vector<FieldElement> e_src;
    std::vector<FieldElement> e_dst;
    std::vector<FieldElement> s;
    std::vector<FieldElement> z;
    std::vector<FieldElement> m;
    std::vector<FieldElement> delta;
    std::vector<FieldElement> u;
    std::vector<FieldElement> sum;
    std::vector<FieldElement> inv;
    std::vector<FieldElement> alpha;
    std::vector<FieldElement> e_src_edge;
    std::vector<FieldElement> e_dst_edge;
    std::vector<FieldElement> m_edge;
    std::vector<FieldElement> sum_edge;
    std::vector<FieldElement> inv_edge;
};

HiddenQuantizedWitness quantize_hidden_witness_cpu(
    const model::HeadForwardTrace& fp,
    const std::vector<std::size_t>& edge_src_indices,
    const std::vector<std::size_t>& edge_dst_indices,
    std::size_t edge_domain_size) {
    HiddenQuantizedWitness out;
    const auto node_count = fp.H_prime.size();
    const auto d_h = node_count == 0 ? 0 : fp.H_prime.front().size();
    out.h_prime.assign(node_count, std::vector<FieldElement>(d_h, FieldElement::zero()));
    out.h_agg_pre.assign(node_count, std::vector<FieldElement>(d_h, FieldElement::zero()));
    out.h_agg.assign(node_count, std::vector<FieldElement>(d_h, FieldElement::zero()));
    out.e_src.assign(fp.E_src.size(), FieldElement::zero());
    out.e_dst.assign(fp.E_dst.size(), FieldElement::zero());
    out.m.assign(fp.M.size(), FieldElement::zero());
    out.sum.assign(fp.Sum.size(), FieldElement::zero());
    out.inv.assign(fp.inv.size(), FieldElement::zero());
    out.s.assign(edge_domain_size, FieldElement::zero());
    out.z.assign(edge_domain_size, FieldElement::zero());
    out.delta.assign(edge_domain_size, FieldElement::zero());
    out.u.assign(edge_domain_size, FieldElement::zero());
    out.alpha.assign(edge_domain_size, FieldElement::zero());
    out.e_src_edge.assign(edge_domain_size, FieldElement::zero());
    out.e_dst_edge.assign(edge_domain_size, FieldElement::zero());
    out.m_edge.assign(edge_domain_size, FieldElement::zero());
    out.sum_edge.assign(edge_domain_size, FieldElement::zero());
    out.inv_edge.assign(edge_domain_size, FieldElement::zero());

    parallel_for_ranges(
        node_count,
        32,
        [&](std::size_t begin, std::size_t end) {
            for (std::size_t row = begin; row < end; ++row) {
                out.e_src[row] = quantize_float(fp.E_src[row]);
                out.e_dst[row] = quantize_float(fp.E_dst[row]);
                out.m[row] = quantize_float(fp.M[row]);
                out.sum[row] = quantize_float(fp.Sum[row]);
                out.inv[row] = quantize_float(fp.inv[row]);
                for (std::size_t column = 0; column < d_h; ++column) {
                    out.h_prime[row][column] = quantize_float(fp.H_prime[row][column]);
                    out.h_agg_pre[row][column] = quantize_float(fp.H_agg_pre_bias[row][column]);
                    out.h_agg[row][column] = quantize_float(fp.H_agg[row][column]);
                }
            }
        });

    parallel_for_ranges(
        edge_src_indices.size(),
        128,
        [&](std::size_t begin, std::size_t end) {
            for (std::size_t edge_index = begin; edge_index < end; ++edge_index) {
                out.s[edge_index] = quantize_float(fp.S[edge_index]);
                out.z[edge_index] = quantize_float(fp.Z[edge_index]);
                out.delta[edge_index] = quantize_float(fp.Delta[edge_index]);
                out.u[edge_index] = quantize_float(fp.U[edge_index]);
                out.alpha[edge_index] = quantize_float(fp.alpha[edge_index]);
                const auto src = edge_src_indices[edge_index];
                const auto dst = edge_dst_indices[edge_index];
                out.e_src_edge[edge_index] = out.e_src[src];
                out.e_dst_edge[edge_index] = out.e_dst[dst];
                out.m_edge[edge_index] = out.m[dst];
                out.sum_edge[edge_index] = out.sum[dst];
                out.inv_edge[edge_index] = out.inv[dst];
            }
        });
    return out;
}

std::size_t lookup_active_transition_limit(std::size_t valid_count, std::size_t domain_size) {
    if (domain_size <= 1) {
        return 0;
    }
    return std::min(valid_count, domain_size - 1U);
}

std::vector<FieldElement> compress_rows_with_challenge(const Matrix& matrix, const FieldElement& challenge) {
    if (matrix.empty()) {
        return {};
    }
    const auto column_powers = powers(challenge, matrix.front().size());
    return linear_form_by_powers(matrix, column_powers);
}

std::vector<FieldElement> broadcast_node_values(
    const std::vector<FieldElement>& node_values,
    const std::vector<Edge>& edges,
    bool use_src,
    std::size_t domain_size) {
    std::vector<FieldElement> out(domain_size, FieldElement::zero());
    for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
        out[edge_index] = node_values[use_src ? edges[edge_index].src : edges[edge_index].dst];
    }
    return out;
}

std::vector<FieldElement> count_by_src(const std::vector<Edge>& edges, std::size_t node_count, std::size_t domain_size) {
    std::vector<FieldElement> out(domain_size, FieldElement::zero());
    std::vector<std::size_t> counts(node_count, 0);
    for (const auto& edge : edges) {
        counts[edge.src] += 1;
    }
    for (std::size_t node = 0; node < node_count; ++node) {
        out[node] = FieldElement(counts[node]);
    }
    return out;
}

std::vector<FieldElement> count_by_dst(const std::vector<Edge>& edges, std::size_t node_count, std::size_t domain_size) {
    std::vector<FieldElement> out(domain_size, FieldElement::zero());
    std::vector<std::size_t> counts(node_count, 0);
    for (const auto& edge : edges) {
        counts[edge.dst] += 1;
    }
    for (std::size_t node = 0; node < node_count; ++node) {
        out[node] = FieldElement(counts[node]);
    }
    return out;
}

std::vector<std::pair<std::size_t, std::size_t>> dst_edge_groups(const std::vector<Edge>& edges) {
    std::vector<std::pair<std::size_t, std::size_t>> groups;
    if (edges.empty()) {
        return groups;
    }
    std::size_t group_begin = 0;
    while (group_begin < edges.size()) {
        std::size_t group_end = group_begin;
        while (group_end + 1 < edges.size() && edges[group_end + 1].dst == edges[group_begin].dst) {
            ++group_end;
        }
        groups.emplace_back(group_begin, group_end + 1);
        group_begin = group_end + 1;
    }
    return groups;
}

std::vector<FieldElement> build_group_target(
    const std::vector<FieldElement>& node_values,
    const std::vector<Edge>& edges,
    std::size_t domain_size) {
    return broadcast_node_values(node_values, edges, false, domain_size);
}

struct BindingTrace {
    std::vector<FieldElement> a;
    std::vector<FieldElement> b;
    std::vector<FieldElement> acc;
    FieldElement mu = FieldElement::zero();
};

std::string trace_cache_prefix(const ProtocolContext& context) {
    return context.config.checkpoint_bundle
        + ":" + context.config.dataset
        + ":" + std::to_string(context.local.num_nodes)
        + ":" + std::to_string(context.local.edges.size())
        + ":" + std::to_string(context.config.seed);
}

struct TraceStaticArtifacts {
    std::vector<FieldElement> feature_index_fields;
    std::vector<FieldElement> absolute_id_fields;
    std::vector<FieldElement> dataset_index_fields;
    std::vector<FieldElement> edge_src_fields;
    std::vector<FieldElement> edge_dst_fields;
    std::vector<std::size_t> edge_src_indices;
    std::vector<std::size_t> edge_dst_indices;
    std::vector<FieldElement> src_multiplicity_template;
    std::vector<FieldElement> dst_multiplicity_template;
    std::vector<std::pair<std::size_t, std::size_t>> edge_groups_by_dst;
    bool full_feature_identity = false;
};

const TraceStaticArtifacts& cached_trace_static_artifacts(const ProtocolContext& context) {
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, std::shared_ptr<TraceStaticArtifacts>> cache;
    const auto cache_key =
        trace_cache_prefix(context)
        + ":trace_static:"
        + std::to_string(context.local.num_features)
        + ":" + std::to_string(context.dataset.num_nodes)
        + ":" + std::to_string(context.domains.n->size)
        + ":" + std::to_string(context.domains.edge->size);
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (const auto it = cache.find(cache_key); it != cache.end()) {
            return *it->second;
        }
    }

    auto artifacts = std::make_shared<TraceStaticArtifacts>();
    artifacts->feature_index_fields.assign(context.local.num_features, FieldElement::zero());
    for (std::size_t j = 0; j < context.local.num_features; ++j) {
        artifacts->feature_index_fields[j] = FieldElement(j);
    }

    artifacts->absolute_id_fields.assign(context.local.num_nodes, FieldElement::zero());
    artifacts->full_feature_identity = context.local.num_nodes == context.dataset.num_nodes;
    for (std::size_t i = 0; i < context.local.num_nodes; ++i) {
        artifacts->absolute_id_fields[i] = FieldElement(context.local.absolute_ids[i]);
        artifacts->full_feature_identity =
            artifacts->full_feature_identity && context.local.absolute_ids[i] == i;
    }

    artifacts->dataset_index_fields.assign(context.dataset.num_nodes, FieldElement::zero());
    for (std::size_t i = 0; i < context.dataset.num_nodes; ++i) {
        artifacts->dataset_index_fields[i] = FieldElement(i);
    }

    const auto edge_count = context.local.edges.size();
    artifacts->edge_src_fields.assign(edge_count, FieldElement::zero());
    artifacts->edge_dst_fields.assign(edge_count, FieldElement::zero());
    artifacts->edge_src_indices.assign(edge_count, 0);
    artifacts->edge_dst_indices.assign(edge_count, 0);
    for (std::size_t edge_index = 0; edge_index < edge_count; ++edge_index) {
        artifacts->edge_src_indices[edge_index] = context.local.edges[edge_index].src;
        artifacts->edge_dst_indices[edge_index] = context.local.edges[edge_index].dst;
        artifacts->edge_src_fields[edge_index] = FieldElement(artifacts->edge_src_indices[edge_index]);
        artifacts->edge_dst_fields[edge_index] = FieldElement(artifacts->edge_dst_indices[edge_index]);
    }

    artifacts->src_multiplicity_template =
        count_by_src(context.local.edges, context.local.num_nodes, context.domains.n->size);
    artifacts->dst_multiplicity_template =
        count_by_dst(context.local.edges, context.local.num_nodes, context.domains.n->size);
    artifacts->edge_groups_by_dst = dst_edge_groups(context.local.edges);

    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        const auto [it, _] = cache.emplace(cache_key, artifacts);
        return *it->second;
    }
}

struct FullFeatureLookupArtifacts {
    std::vector<FieldElement> table;
    std::vector<FieldElement> query;
    std::vector<FieldElement> multiplicity;
};

const FullFeatureLookupArtifacts& cached_full_feature_lookup_artifacts(
    const ProtocolContext& context,
    const TraceStaticArtifacts& static_artifacts,
    const FieldElement& eta_feature_index,
    const FieldElement& eta_feature_value) {
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, std::shared_ptr<FullFeatureLookupArtifacts>> cache;
    const auto cache_key =
        trace_cache_prefix(context)
        + ":full_fh_lookup:"
        + eta_feature_index.to_string()
        + ":" + eta_feature_value.to_string()
        + ":" + std::to_string(context.domains.fh->size);
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (const auto it = cache.find(cache_key); it != cache.end()) {
            return *it->second;
        }
    }

    auto artifacts = std::make_shared<FullFeatureLookupArtifacts>();
    artifacts->table.assign(context.domains.fh->size, FieldElement::zero());
    artifacts->query.assign(context.domains.fh->size, FieldElement::zero());
    artifacts->multiplicity.assign(context.domains.fh->size, FieldElement::zero());
    fill_feature_lookup_rows(
        static_artifacts.dataset_index_fields,
        static_artifacts.feature_index_fields,
        context.dataset.features,
        eta_feature_index,
        eta_feature_value,
        artifacts->table);
    artifacts->query = artifacts->table;
    std::fill(
        artifacts->multiplicity.begin(),
        artifacts->multiplicity.begin()
            + static_cast<std::ptrdiff_t>(context.dataset.num_nodes * context.local.num_features),
        FieldElement::one());

    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        const auto [it, _] = cache.emplace(cache_key, artifacts);
        return *it->second;
    }
}

BindingTrace build_binding_trace(
    const Matrix& matrix,
    const std::shared_ptr<algebra::RootOfUnityDomain>& row_domain,
    std::size_t real_row_count,
    const FieldElement& point,
    const std::vector<FieldElement>& parameter_values,
    const std::shared_ptr<algebra::RootOfUnityDomain>& dim_domain,
    std::size_t real_dim_count) {
    BindingTrace out;
    const auto& row_weights = cached_barycentric_weights_native(row_domain, point);
    const auto folded = weighted_column_sum_native(matrix, row_weights, real_row_count);
    out.a.assign(dim_domain->size, FieldElement::zero());
    out.b.assign(dim_domain->size, FieldElement::zero());
    out.acc.assign(dim_domain->size, FieldElement::zero());
    for (std::size_t i = 0; i < real_dim_count; ++i) {
        out.a[i] = folded[i];
        out.b[i] = parameter_values[i];
        out.mu += out.a[i] * out.b[i];
        if (i + 1 < dim_domain->size) {
            out.acc[i + 1] = out.mu;
        }
    }
    for (std::size_t i = real_dim_count + 1; i < dim_domain->size; ++i) {
        out.acc[i] = out.mu;
    }
    return out;
}

template <typename Builder>
const BindingTrace& cached_binding_trace(
    const std::string& cache_key,
    Builder&& builder) {
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, std::shared_ptr<BindingTrace>> cache;
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (const auto it = cache.find(cache_key); it != cache.end()) {
            return *it->second;
        }
    }

    auto binding = std::make_shared<BindingTrace>(builder());
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        const auto [it, _] = cache.emplace(cache_key, binding);
        return *it->second;
    }
}

struct HiddenCompressedTrace {
    std::vector<FieldElement> h_star;
    std::vector<FieldElement> h_star_edge;
    std::vector<FieldElement> h_agg_pre_star;
    std::vector<FieldElement> h_agg_pre_star_edge;
    std::vector<FieldElement> h_agg_star;
    std::vector<FieldElement> h_agg_star_edge;
};

const HiddenCompressedTrace& cached_hidden_compressed_trace(
    const std::string& cache_key,
    const Matrix& h_prime,
    const Matrix& h_agg_pre,
    const Matrix& h_agg,
    const std::vector<Edge>& edges,
    std::size_t edge_domain_size,
    const FieldElement& xi) {
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, std::shared_ptr<HiddenCompressedTrace>> cache;
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (const auto it = cache.find(cache_key); it != cache.end()) {
            return *it->second;
        }
    }

    auto packed = std::make_shared<HiddenCompressedTrace>();
    packed->h_star = compress_rows_with_challenge(h_prime, xi);
    packed->h_star_edge = broadcast_node_values(packed->h_star, edges, true, edge_domain_size);
    packed->h_agg_pre_star = compress_rows_with_challenge(h_agg_pre, xi);
    packed->h_agg_pre_star_edge =
        broadcast_node_values(packed->h_agg_pre_star, edges, false, edge_domain_size);
    packed->h_agg_star = compress_rows_with_challenge(h_agg, xi);
    packed->h_agg_star_edge = broadcast_node_values(packed->h_agg_star, edges, false, edge_domain_size);
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        const auto [it, _] = cache.emplace(cache_key, packed);
        return *it->second;
    }
}

struct RouteTrace {
    std::vector<FieldElement> table;
    std::vector<FieldElement> query;
    std::vector<FieldElement> multiplicity;
    std::vector<FieldElement> node_acc;
    std::vector<FieldElement> edge_acc;
    FieldElement total = FieldElement::zero();
};

template <typename Builder>
const RouteTrace& cached_route_trace(
    const std::string& cache_key,
    Builder&& builder) {
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, std::shared_ptr<RouteTrace>> cache;
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (const auto it = cache.find(cache_key); it != cache.end()) {
            return *it->second;
        }
    }

    auto route = std::make_shared<RouteTrace>(builder());
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        const auto [it, _] = cache.emplace(cache_key, route);
        return *it->second;
    }
}

template <typename Builder>
const std::vector<FieldElement>& cached_state_vector(
    const std::string& cache_key,
    Builder&& builder) {
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, std::shared_ptr<std::vector<FieldElement>>> cache;
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (const auto it = cache.find(cache_key); it != cache.end()) {
            return *it->second;
        }
    }

    auto values = std::make_shared<std::vector<FieldElement>>(builder());
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        const auto [it, _] = cache.emplace(cache_key, values);
        return *it->second;
    }
}

struct HiddenRoutePackArtifacts {
    std::vector<FieldElement> src_table;
    std::vector<FieldElement> src_query;
    std::vector<FieldElement> t_table;
    std::vector<FieldElement> t_query;
    std::vector<FieldElement> dst_table;
    std::vector<FieldElement> dst_query;
};

HiddenRoutePackArtifacts build_hidden_route_pack_artifacts(
    const std::vector<FieldElement>& absolute_id_fields,
    const std::vector<FieldElement>& edge_src_fields,
    const std::vector<FieldElement>& edge_dst_fields,
    const std::vector<FieldElement>& e_src,
    const std::vector<FieldElement>& e_dst,
    const std::vector<FieldElement>& e_dst_edge,
    const std::vector<FieldElement>& h_star_edge,
    const std::vector<FieldElement>& m,
    const std::vector<FieldElement>& m_edge,
    const std::vector<FieldElement>& sum,
    const std::vector<FieldElement>& sum_edge,
    const std::vector<FieldElement>& inv,
    const std::vector<FieldElement>& inv_edge,
    const std::vector<FieldElement>& h_agg_star,
    const std::vector<FieldElement>& h_agg_star_edge,
    const std::vector<FieldElement>& t_psq,
    const std::vector<FieldElement>& t_psq_edge,
    const FieldElement& eta_src_1,
    const FieldElement& eta_t_1,
    const std::array<FieldElement, 5>& eta_dst_terms) {
    HiddenRoutePackArtifacts out;
    out.src_table.assign(absolute_id_fields.size(), FieldElement::zero());
    out.src_query.assign(edge_src_fields.size(), FieldElement::zero());
    out.t_table.assign(absolute_id_fields.size(), FieldElement::zero());
    out.t_query.assign(edge_dst_fields.size(), FieldElement::zero());
    out.dst_table.assign(absolute_id_fields.size(), FieldElement::zero());
    out.dst_query.assign(edge_dst_fields.size(), FieldElement::zero());

    parallel_for_ranges(
        absolute_id_fields.size(),
        64,
        [&](std::size_t begin, std::size_t end) {
            for (std::size_t i = begin; i < end; ++i) {
                const auto absolute_id = absolute_id_fields[i];
                out.src_table[i] = absolute_id + eta_src_1 * e_src[i];
                out.t_table[i] = absolute_id + eta_t_1 * t_psq[i];
                out.dst_table[i] =
                    absolute_id
                    + eta_dst_terms[0] * e_dst[i]
                    + eta_dst_terms[1] * m[i]
                    + eta_dst_terms[2] * sum[i]
                    + eta_dst_terms[3] * inv[i]
                    + eta_dst_terms[4] * h_agg_star[i];
            }
        });

    parallel_for_ranges(
        edge_src_fields.size(),
        128,
        [&](std::size_t begin, std::size_t end) {
            for (std::size_t k = begin; k < end; ++k) {
                out.src_query[k] = edge_src_fields[k] + eta_src_1 * h_star_edge[k];
                out.t_query[k] = edge_dst_fields[k] + eta_t_1 * t_psq_edge[k];
                out.dst_query[k] =
                    edge_dst_fields[k]
                    + eta_dst_terms[0] * e_dst_edge[k]
                    + eta_dst_terms[1] * m_edge[k]
                    + eta_dst_terms[2] * sum_edge[k]
                    + eta_dst_terms[3] * inv_edge[k]
                    + eta_dst_terms[4] * h_agg_star_edge[k];
            }
        });
    return out;
}

RouteTrace build_route_trace(
    const std::vector<FieldElement>& table_values,
    const std::vector<FieldElement>& query_values,
    const std::vector<FieldElement>& multiplicity_values,
    std::size_t real_node_count,
    std::size_t real_edge_count,
    const std::shared_ptr<algebra::RootOfUnityDomain>& node_domain,
    const std::shared_ptr<algebra::RootOfUnityDomain>& edge_domain,
    const FieldElement& beta) {
    RouteTrace out;
    out.table = padded_column(table_values, node_domain->size);
    out.query = padded_column(query_values, edge_domain->size);
    out.multiplicity = padded_column(multiplicity_values, node_domain->size);
    out.node_acc = build_route_node_accumulator(out.table, out.multiplicity, real_node_count, node_domain->size, beta);
    out.edge_acc = build_route_edge_accumulator(out.query, real_edge_count, edge_domain->size, beta);
    out.total = out.node_acc[real_node_count];
    return out;
}

std::uint64_t load_u64_le(const std::uint8_t* bytes) {
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(bytes[i]) << (8U * i);
    }
    return value;
}

struct FieldKey {
    std::array<std::uint64_t, 4> limbs = {0, 0, 0, 0};

    bool operator==(const FieldKey& other) const {
        return limbs == other.limbs;
    }
};

FieldKey pack_field_key(const FieldElement& value) {
    std::array<std::uint8_t, 32> bytes{};
    const auto written = value.native().getLittleEndian(bytes.data(), bytes.size());
    if (written > bytes.size()) {
        throw std::runtime_error("field key little-endian serialization overflow");
    }
    FieldKey out;
    for (std::size_t limb = 0; limb < out.limbs.size(); ++limb) {
        out.limbs[limb] = load_u64_le(bytes.data() + limb * 8U);
    }
    return out;
}

struct FieldPairKey {
    FieldKey first;
    FieldKey second;

    bool operator==(const FieldPairKey& other) const {
        return first == other.first && second == other.second;
    }
};

struct FieldPairKeyHash {
    static std::size_t mix_key(const FieldKey& key) {
        std::size_t out = 0;
        for (const auto limb : key.limbs) {
            out ^= std::hash<std::uint64_t>{}(limb) + 0x9e3779b97f4a7c15ULL + (out << 6U) + (out >> 2U);
        }
        return out;
    }

    std::size_t operator()(const FieldPairKey& key) const {
        return mix_key(key.first) ^ (mix_key(key.second) << 1U);
    }
};

std::unordered_map<std::uint64_t, std::size_t> build_value_index_map(
    const std::vector<FieldElement>& values,
    std::size_t valid_count) {
    std::unordered_map<std::uint64_t, std::size_t> index;
    index.reserve(valid_count);
    for (std::size_t i = 0; i < valid_count; ++i) {
        index.emplace(values[i].value(), i);
    }
    return index;
}

const std::unordered_map<std::uint64_t, std::size_t>& cached_value_index_map(
    const std::string& label,
    const std::vector<FieldElement>& values,
    std::size_t valid_count) {
    static std::mutex cache_mutex;
    static std::unordered_map<
        std::string,
        std::shared_ptr<std::unordered_map<std::uint64_t, std::size_t>>> cache;
    const auto cache_key =
        label + ":" + std::to_string(valid_count) + ":" + std::to_string(values.size());
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (const auto it = cache.find(cache_key); it != cache.end()) {
            return *it->second;
        }
    }
    auto packed = std::make_shared<std::unordered_map<std::uint64_t, std::size_t>>(
        build_value_index_map(values, valid_count));
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        const auto [it, _] = cache.emplace(cache_key, packed);
        return *it->second;
    }
}

std::unordered_map<FieldPairKey, std::size_t, FieldPairKeyHash> build_pair_index_map(
    const std::vector<std::pair<FieldElement, FieldElement>>& values) {
    std::unordered_map<FieldPairKey, std::size_t, FieldPairKeyHash> index;
    index.reserve(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        index.emplace(FieldPairKey{pack_field_key(values[i].first), pack_field_key(values[i].second)}, i);
    }
    return index;
}

const std::unordered_map<FieldPairKey, std::size_t, FieldPairKeyHash>& cached_pair_index_map(
    const std::string& label,
    const std::vector<std::pair<FieldElement, FieldElement>>& values) {
    static std::mutex cache_mutex;
    static std::unordered_map<
        std::string,
        std::shared_ptr<std::unordered_map<FieldPairKey, std::size_t, FieldPairKeyHash>>> cache;
    const auto cache_key = label + ":" + std::to_string(values.size());
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (const auto it = cache.find(cache_key); it != cache.end()) {
            return *it->second;
        }
    }
    auto packed = std::make_shared<std::unordered_map<FieldPairKey, std::size_t, FieldPairKeyHash>>(
        build_pair_index_map(values));
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        const auto [it, _] = cache.emplace(cache_key, packed);
        return *it->second;
    }
}

std::vector<FieldElement> build_lookup_multiplicity(
    const std::unordered_map<std::uint64_t, std::size_t>& index_by_value,
    const std::vector<FieldElement>& query_values,
    std::size_t domain_size,
    std::size_t valid_count,
    const std::string& label) {
    std::vector<FieldElement> multiplicity(domain_size, FieldElement::zero());
    for (std::size_t i = 0; i < valid_count; ++i) {
        const auto it = index_by_value.find(query_values[i].value());
        if (it == index_by_value.end()) {
            throw std::runtime_error(
                "lookup query escaped static table for " + label
                + " value=" + query_values[i].to_string());
        }
        multiplicity[it->second] += FieldElement::one();
    }
    return multiplicity;
}

std::vector<FieldElement> build_pair_lookup_multiplicity(
    const std::unordered_map<FieldPairKey, std::size_t, FieldPairKeyHash>& index_by_value,
    const std::vector<FieldElement>& query_left,
    const std::vector<FieldElement>& query_right,
    std::size_t domain_size,
    std::size_t valid_count,
    const std::string& label) {
    std::vector<FieldElement> multiplicity(domain_size, FieldElement::zero());
    for (std::size_t i = 0; i < valid_count; ++i) {
        const auto it = index_by_value.find(FieldPairKey{pack_field_key(query_left[i]), pack_field_key(query_right[i])});
        if (it == index_by_value.end()) {
            throw std::runtime_error(
                "pair lookup query escaped static table for " + label
                + " left=" + query_left[i].to_string()
                + " right=" + query_right[i].to_string());
        }
        multiplicity[it->second] += FieldElement::one();
    }
    return multiplicity;
}

std::vector<FieldElement> multiplicity_from_indices(
    const std::vector<std::size_t>& indices,
    std::size_t domain_size) {
    std::vector<FieldElement> multiplicity(domain_size, FieldElement::zero());
    if (cuda_lookup_histogram_enabled() && indices.size() >= 4096 && domain_size >= 1024) {
#if GATZK_ENABLE_CUDA_BACKEND
        const auto counts = lookup_histogram_indices_cuda(indices, domain_size);
        for (std::size_t index = 0; index < domain_size; ++index) {
            multiplicity[index] = FieldElement(counts[index]);
        }
        return multiplicity;
#endif
    }
    for (const auto index : indices) {
        multiplicity[index] += FieldElement::one();
    }
    return multiplicity;
}

const std::vector<FieldElement>& cached_single_lookup_table(
    const std::string& label,
    const std::vector<FieldElement>& values,
    std::size_t domain_size) {
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, std::shared_ptr<std::vector<FieldElement>>> cache;
    const auto cache_key =
        label + ":" + std::to_string(domain_size) + ":" + std::to_string(values.size());
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (const auto it = cache.find(cache_key); it != cache.end()) {
            return *it->second;
        }
    }
    auto packed = std::make_shared<std::vector<FieldElement>>(domain_size, FieldElement::zero());
    std::copy(values.begin(), values.end(), packed->begin());
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        const auto [it, _] = cache.emplace(cache_key, packed);
        return *it->second;
    }
}

const std::vector<FieldElement>& cached_pair_lookup_table(
    const std::string& label,
    const std::vector<std::pair<FieldElement, FieldElement>>& values,
    const FieldElement& eta,
    std::size_t domain_size) {
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, std::shared_ptr<std::vector<FieldElement>>> cache;
    const auto cache_key =
        label + ":" + std::to_string(domain_size) + ":" + eta.to_string();
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (const auto it = cache.find(cache_key); it != cache.end()) {
            return *it->second;
        }
    }
    auto packed = std::make_shared<std::vector<FieldElement>>(domain_size, FieldElement::zero());
    for (std::size_t i = 0; i < values.size(); ++i) {
        (*packed)[i] = values[i].first + eta * values[i].second;
    }
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        const auto [it, _] = cache.emplace(cache_key, packed);
        return *it->second;
    }
}

struct SingleLookupArtifacts {
    std::vector<FieldElement> query;
    std::vector<FieldElement> multiplicity;
};

template <typename Builder>
const SingleLookupArtifacts& cached_single_lookup_artifacts(
    const std::string& cache_key,
    Builder&& builder) {
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, std::shared_ptr<SingleLookupArtifacts>> cache;
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (const auto it = cache.find(cache_key); it != cache.end()) {
            return *it->second;
        }
    }

    auto artifacts = std::make_shared<SingleLookupArtifacts>(builder());
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        const auto [it, _] = cache.emplace(cache_key, artifacts);
        return *it->second;
    }
}

SingleLookupArtifacts build_single_lookup_artifacts(
    const std::unordered_map<std::uint64_t, std::size_t>& index_by_value,
    const std::vector<FieldElement>& query_values,
    std::size_t domain_size,
    std::size_t valid_count,
    const std::string& label,
    RunMetrics* metrics) {
    SingleLookupArtifacts out;
    out.query.assign(domain_size, FieldElement::zero());
    auto stage_start = Clock::now();
    std::vector<std::size_t> indices(valid_count, 0);
    for (std::size_t i = 0; i < valid_count; ++i) {
        out.query[i] = query_values[i];
        const auto it = index_by_value.find(query_values[i].value());
        if (it == index_by_value.end()) {
            throw std::runtime_error(
                "lookup query escaped static table for " + label
                + " value=" + query_values[i].to_string());
        }
        indices[i] = it->second;
    }
    add_metric(metrics != nullptr ? &metrics->lookup_query_pack_ms : nullptr, stage_start);
    stage_start = Clock::now();
    out.multiplicity = multiplicity_from_indices(indices, domain_size);
    add_metric(metrics != nullptr ? &metrics->lookup_multiplicity_ms : nullptr, stage_start);
    return out;
}

struct PairLookupArtifacts {
    std::vector<FieldElement> query;
    std::vector<FieldElement> multiplicity;
};

template <typename Builder>
const PairLookupArtifacts& cached_pair_lookup_artifacts(
    const std::string& cache_key,
    Builder&& builder) {
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, std::shared_ptr<PairLookupArtifacts>> cache;
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (const auto it = cache.find(cache_key); it != cache.end()) {
            return *it->second;
        }
    }

    auto artifacts = std::make_shared<PairLookupArtifacts>(builder());
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        const auto [it, _] = cache.emplace(cache_key, artifacts);
        return *it->second;
    }
}

PairLookupArtifacts build_pair_lookup_artifacts(
    const std::unordered_map<FieldPairKey, std::size_t, FieldPairKeyHash>& index_by_value,
    const std::vector<FieldElement>& query_left,
    const std::vector<FieldElement>& query_right,
    const FieldElement& eta,
    std::size_t domain_size,
    std::size_t valid_count,
    const std::string& label,
    RunMetrics* metrics) {
    PairLookupArtifacts out;
    out.query.assign(domain_size, FieldElement::zero());
    auto stage_start = Clock::now();
    std::vector<std::size_t> indices(valid_count, 0);
    for (std::size_t i = 0; i < valid_count; ++i) {
        out.query[i] = query_left[i] + eta * query_right[i];
        const auto it = index_by_value.find(FieldPairKey{pack_field_key(query_left[i]), pack_field_key(query_right[i])});
        if (it == index_by_value.end()) {
            throw std::runtime_error(
                "pair lookup query escaped static table for " + label
                + " left=" + query_left[i].to_string()
                + " right=" + query_right[i].to_string());
        }
        indices[i] = it->second;
    }
    add_metric(metrics != nullptr ? &metrics->lookup_query_pack_ms : nullptr, stage_start);
    stage_start = Clock::now();
    out.multiplicity = multiplicity_from_indices(indices, domain_size);
    add_metric(metrics != nullptr ? &metrics->lookup_multiplicity_ms : nullptr, stage_start);
    return out;
}

std::vector<FieldElement> build_weighted_sum(
    const std::vector<FieldElement>& left,
    const FieldElement& lambda,
    const std::vector<FieldElement>& right,
    std::size_t size) {
    std::vector<FieldElement> out(size, FieldElement::zero());
    for (std::size_t i = 0; i < left.size(); ++i) {
        out[i] = left[i] + lambda * right[i];
    }
    return out;
}

std::string hidden_weight_label(std::size_t head_index) {
    return "V_h" + std::to_string(head_index) + "_W";
}

std::string hidden_src_label(std::size_t head_index) {
    return "V_h" + std::to_string(head_index) + "_a_src";
}

std::string hidden_dst_label(std::size_t head_index) {
    return "V_h" + std::to_string(head_index) + "_a_dst";
}

std::string output_weight_label() {
    return "V_out_W";
}

std::string output_src_label() {
    return "V_out_a_src";
}

std::string output_dst_label() {
    return "V_out_a_dst";
}

TraceArtifacts build_multihead_trace(const ProtocolContext& context, RunMetrics* metrics) {
    if (!model::supports_current_formal_proof_shape(context.model)) {
        throw std::runtime_error(
            "formal multi-head proof currently supports only L=2 with exactly one hidden layer and K_out=1");
    }
    const auto trace_start = Clock::now();
    const auto initial_forward_ms = metrics != nullptr ? metrics->forward_ms : 0.0;
    const auto initial_commit_dynamic_ms = metrics != nullptr ? metrics->commit_dynamic_ms : 0.0;
    TraceArtifacts trace;
    crypto::Transcript transcript("gatzkml");
    const bool keep_trace_payloads = context.config.dump_trace;
    const auto& local = context.local;
    const auto& domains = context.domains;
    const auto& edges = local.edges;
    const std::size_t n_nodes = local.num_nodes;
    const std::size_t n_edges = edges.size();
    const std::size_t d_in = local.num_features;
    const std::size_t d_h = model::attention_head_output_width(context.model.hidden_heads.front());
    const std::size_t d_cat = d_h * context.model.hidden_heads.size();
    const std::size_t n_classes = local.num_classes;
    auto metric_value = [&](double RunMetrics::*member) {
        return metrics != nullptr ? metrics->*member : 0.0;
    };
    auto add_residual_metric = [&](double RunMetrics::*member, double total_ms, std::initializer_list<double> accounted) {
        if (metrics == nullptr) {
            return;
        }
        double accounted_ms = 0.0;
        for (const auto value : accounted) {
            accounted_ms += value;
        }
        metrics->*member += std::max(0.0, total_ms - accounted_ms);
    };

    auto stage_start = Clock::now();
    const auto cache_prefix = trace_cache_prefix(context);
    const auto& static_artifacts = cached_trace_static_artifacts(context);
    const auto& feature_index_fields = static_artifacts.feature_index_fields;
    const auto& absolute_id_fields = static_artifacts.absolute_id_fields;
    const auto& dataset_index_fields = static_artifacts.dataset_index_fields;
    const auto& edge_src_fields = static_artifacts.edge_src_fields;
    const auto& edge_dst_fields = static_artifacts.edge_dst_fields;
    const auto& edge_src_indices = static_artifacts.edge_src_indices;
    const auto& edge_dst_indices = static_artifacts.edge_dst_indices;
    const auto& src_multiplicity_template = static_artifacts.src_multiplicity_template;
    const auto& dst_multiplicity_template = static_artifacts.dst_multiplicity_template;
    const auto& edge_groups_by_dst = static_artifacts.edge_groups_by_dst;
    add_metric(metrics != nullptr ? &metrics->shared_helper_build_ms : nullptr, stage_start);

    stage_start = Clock::now();
    const auto& q_new = eval_data(context, "P_Q_new_edge");
    const auto& q_edge_valid = eval_data(context, "P_Q_edge_valid");
    const auto& q_tbl_feat = eval_data(context, "P_Q_tbl_feat");
    const auto& q_qry_feat = eval_data(context, "P_Q_qry_feat");
    const auto& q_tbl_l = eval_data(context, "P_Q_tbl_L");
    const auto& q_qry_l = eval_data(context, "P_Q_qry_L");
    const auto& q_tbl_r = eval_data(context, "P_Q_tbl_R");
    const auto& q_qry_r = eval_data(context, "P_Q_qry_R");
    const auto& q_tbl_exp = eval_data(context, "P_Q_tbl_exp");
    const auto& q_qry_exp = eval_data(context, "P_Q_qry_exp");
    const auto& q_tbl_elu = eval_data(context, "P_Q_tbl_ELU");
    const auto& q_qry_elu = eval_data(context, "P_Q_qry_ELU");
    add_metric(metrics != nullptr ? &metrics->public_poly_residual_ms : nullptr, stage_start);

    add_dynamic_commitment_batch(
        trace,
        {matrix_commitment_only_spec("P_H", local.features)},
        context.kzg,
        keep_trace_payloads,
        metrics);

    absorb_public_metadata(transcript, canonical_public_metadata(context));
    transcript.absorb_scalar("N", FieldElement(n_nodes));
    transcript.absorb_scalar("E", FieldElement(n_edges));
    transcript.absorb_scalar("d_in", FieldElement(d_in));
    transcript.absorb_scalar("d_h", FieldElement(d_h));
    transcript.absorb_scalar("d_cat", FieldElement(d_cat));
    transcript.absorb_scalar("C", FieldElement(n_classes));
    transcript.absorb_scalar("B", FieldElement(context.config.range_bits));
    transcript.absorb_commitment("P_I", context.public_commitments.at("P_I").point);
    transcript.absorb_commitment("P_src", context.public_commitments.at("P_src").point);
    transcript.absorb_commitment("P_dst", context.public_commitments.at("P_dst").point);
    transcript.absorb_commitment("P_Q_new_edge", context.public_commitments.at("P_Q_new_edge").point);
    transcript.absorb_commitment("P_Q_end_edge", context.public_commitments.at("P_Q_end_edge").point);
    transcript.absorb_commitment("P_Q_edge_valid", context.public_commitments.at("P_Q_edge_valid").point);
    transcript.absorb_commitment("P_Q_N", context.public_commitments.at("P_Q_N").point);
    transcript.absorb_commitment("P_Q_proj_valid", context.public_commitments.at("P_Q_proj_valid").point);
    transcript.absorb_commitment("P_Q_d_valid", context.public_commitments.at("P_Q_d_valid").point);
    transcript.absorb_commitment("P_Q_cat_valid", context.public_commitments.at("P_Q_cat_valid").point);
    transcript.absorb_commitment("P_Q_C_valid", context.public_commitments.at("P_Q_C_valid").point);
    transcript.absorb_commitment("P_H", trace.commitments.at("P_H").point);
    transcript.absorb_commitment("V_T_H", context.static_commitments.at("V_T_H").point);
    trace.challenges["eta_feat"] = transcript.challenge("eta_feat");
    trace.challenges["beta_feat"] = transcript.challenge("beta_feat");

    {
        const auto eta_feat_powers = powers(trace.challenges.at("eta_feat"), 3);
        std::vector<FieldElement> table_feat(domains.fh->size, FieldElement::zero());
        std::vector<FieldElement> query_feat(domains.fh->size, FieldElement::zero());
        std::vector<FieldElement> multiplicity_feat(domains.fh->size, FieldElement::zero());
        const auto eta_feature_index = eta_feat_powers[1];
        const auto eta_feature_value = eta_feat_powers[2];
        const bool full_feature_identity = static_artifacts.full_feature_identity;
        if (full_feature_identity) {
            const auto fh_stage_start = Clock::now();
            const auto& cached_full_lookup = cached_full_feature_lookup_artifacts(
                context,
                static_artifacts,
                eta_feature_index,
                eta_feature_value);
            table_feat = cached_full_lookup.table;
            add_metric(metrics != nullptr ? &metrics->fh_table_materialization_ms : nullptr, fh_stage_start);
            const auto query_stage_start = Clock::now();
            query_feat = cached_full_lookup.query;
            add_metric(metrics != nullptr ? &metrics->fh_query_materialization_ms : nullptr, query_stage_start);
            const auto mult_stage_start = Clock::now();
            multiplicity_feat = cached_full_lookup.multiplicity;
            add_metric(metrics != nullptr ? &metrics->fh_multiplicity_build_ms : nullptr, mult_stage_start);
        } else {
            std::vector<std::size_t> feat_hits(context.dataset.num_nodes, 0);
            auto fh_stage_start = Clock::now();
            for (const auto absolute_id : local.absolute_ids) {
                if (absolute_id < feat_hits.size()) {
                    feat_hits[absolute_id] += 1;
                }
            }
            std::vector<FieldElement> feat_hit_fields(context.dataset.num_nodes, FieldElement::zero());
            for (std::size_t v = 0; v < context.dataset.num_nodes; ++v) {
                feat_hit_fields[v] = FieldElement(feat_hits[v]);
            }
            add_metric(metrics != nullptr ? &metrics->fh_multiplicity_build_ms : nullptr, fh_stage_start);
            fh_stage_start = Clock::now();
            fill_feature_lookup_rows(
                dataset_index_fields,
                feature_index_fields,
                context.dataset.features,
                eta_feature_index,
                eta_feature_value,
                table_feat);
            add_metric(metrics != nullptr ? &metrics->fh_table_materialization_ms : nullptr, fh_stage_start);
            fh_stage_start = Clock::now();
            fill_feature_lookup_rows(
                absolute_id_fields,
                feature_index_fields,
                local.features,
                eta_feature_index,
                eta_feature_value,
                query_feat);
            add_metric(metrics != nullptr ? &metrics->fh_query_materialization_ms : nullptr, fh_stage_start);
            fh_stage_start = Clock::now();
            fill_repeated_row_values(feat_hit_fields, d_in, multiplicity_feat);
            add_metric(metrics != nullptr ? &metrics->fh_multiplicity_build_ms : nullptr, fh_stage_start);
        }
        auto lookup_stage_start = Clock::now();
        auto fh_accumulator_start = Clock::now();
        std::vector<FieldElement> r_feat(domains.fh->size, FieldElement::zero());
        if (!full_feature_identity) {
            r_feat = build_logup_accumulator_cached_with_active_count(
                cache_prefix + ":fh_feat",
                table_feat,
                query_feat,
                multiplicity_feat,
                q_tbl_feat,
                q_qry_feat,
                trace.challenges.at("beta_feat"),
                lookup_active_transition_limit(n_nodes * d_in, domains.fh->size));
        }
        add_metric(metrics != nullptr ? &metrics->fh_accumulator_build_ms : nullptr, fh_accumulator_start);
        add_metric(metrics != nullptr ? &metrics->lookup_trace_ms : nullptr, lookup_stage_start);
        add_dynamic_commitment_batch(
            trace,
            {
                column_spec("P_Table_feat", table_feat, domains.fh),
                column_spec("P_Query_feat", query_feat, domains.fh),
                column_spec("P_m_feat", multiplicity_feat, domains.fh),
                column_spec("P_R_feat", r_feat, domains.fh),
            },
            context.kzg,
            keep_trace_payloads,
            metrics);
    }

    model::ForwardProfile forward_profile;
    const auto forward_start = Clock::now();
    const auto forward = model::forward_note_style(local.features_fp, edges, context.model, &forward_profile);
    if (metrics != nullptr) {
        metrics->forward_ms += elapsed_ms(forward_start, Clock::now());
        metrics->feature_projection_ms += forward_profile.hidden_projection_ms + forward_profile.output_projection_ms;
        metrics->hidden_forward_projection_ms += forward_profile.hidden_projection_ms;
        metrics->hidden_forward_attention_ms += forward_profile.hidden_attention_ms;
        metrics->hidden_forward_activation_ms += forward_profile.hidden_activation_ms;
        metrics->hidden_concat_ms += forward_profile.hidden_concat_ms;
        metrics->output_forward_projection_ms += forward_profile.output_projection_ms;
        metrics->output_forward_attention_ms += forward_profile.output_attention_ms;
        metrics->output_forward_activation_ms += forward_profile.output_activation_ms;
    }

    stage_start = Clock::now();
    const auto& lrelu_index = cached_pair_index_map("lrelu", context.tables.lrelu);
    const auto& range_index = cached_value_index_map("range", context.tables.range, context.tables.range.size());
    const auto& exp_index = cached_pair_index_map("exp", context.tables.exp);
    const auto& elu_index = cached_pair_index_map("elu", context.tables.elu);
    add_metric(metrics != nullptr ? &metrics->lookup_key_build_ms : nullptr, stage_start);

    auto commit_hidden_head = [&](std::size_t head_index) {
        const auto prefix = "P_h" + std::to_string(head_index) + "_";
        const auto head_cache_prefix = cache_prefix + ":hidden:" + std::to_string(head_index);
        const auto& fp = forward.hidden_head_traces[head_index];
        auto stage_start = Clock::now();
        const auto quantized = quantize_hidden_witness_cpu(
            fp,
            edge_src_indices,
            edge_dst_indices,
            domains.edge->size);
        const auto& h_prime = quantized.h_prime;
        const auto& e_src = quantized.e_src;
        const auto& e_dst = quantized.e_dst;
        const auto& s = quantized.s;
        const auto& z = quantized.z;
        const auto& m = quantized.m;
        const auto& delta = quantized.delta;
        const auto& u = quantized.u;
        const auto& sum = quantized.sum;
        const auto& inv = quantized.inv;
        const auto& alpha = quantized.alpha;
        const auto& h_agg_pre = quantized.h_agg_pre;
        const auto& h_agg = quantized.h_agg;
        const auto& e_src_edge = quantized.e_src_edge;
        const auto& e_dst_edge = quantized.e_dst_edge;
        const auto& m_edge = quantized.m_edge;
        const auto& sum_edge = quantized.sum_edge;
        const auto& inv_edge = quantized.inv_edge;
        add_metric(metrics != nullptr ? &metrics->witness_materialization_ms : nullptr, stage_start);
        add_metric(metrics != nullptr ? &metrics->hidden_edge_score_trace_ms : nullptr, stage_start);

        add_dynamic_commitment_batch(
            trace,
            {
                matrix_commitment_only_spec(prefix + "H_prime", h_prime),
                column_spec(prefix + "E_src", padded_column(e_src, domains.n->size), domains.n),
                column_spec(prefix + "E_dst", padded_column(e_dst, domains.n->size), domains.n),
                column_spec(prefix + "S", padded_column(s, domains.edge->size), domains.edge),
                column_spec(prefix + "Z", padded_column(z, domains.edge->size), domains.edge),
                column_spec(prefix + "M", padded_column(m, domains.n->size), domains.n),
                column_spec(prefix + "M_edge", m_edge, domains.edge),
                column_spec(prefix + "Delta", padded_column(delta, domains.edge->size), domains.edge),
                column_spec(prefix + "U", padded_column(u, domains.edge->size), domains.edge),
                column_spec(prefix + "Sum", padded_column(sum, domains.n->size), domains.n),
                column_spec(prefix + "Sum_edge", sum_edge, domains.edge),
                column_spec(prefix + "inv", padded_column(inv, domains.n->size), domains.n),
                column_spec(prefix + "inv_edge", inv_edge, domains.edge),
                column_spec(prefix + "alpha", padded_column(alpha, domains.edge->size), domains.edge),
                matrix_commitment_only_spec(prefix + "H_agg_pre", h_agg_pre),
                matrix_commitment_only_spec(prefix + "H_agg", h_agg),
            },
            context.kzg,
            keep_trace_payloads,
            metrics);

        transcript.absorb_commitment("P_H", trace.commitments.at("P_H").point);
        transcript.absorb_commitment(prefix + "H_prime", trace.commitments.at(prefix + "H_prime").point);
        transcript.absorb_commitment(hidden_weight_label(head_index), context.static_commitments.at(hidden_weight_label(head_index)).point);
        trace.challenges["y_proj_h" + std::to_string(head_index)] = transcript.challenge("y_proj_h" + std::to_string(head_index));
        trace.challenges["xi_h" + std::to_string(head_index)] = transcript.challenge("xi_h" + std::to_string(head_index));
        transcript.absorb_commitment(prefix + "H_prime", trace.commitments.at(prefix + "H_prime").point);
        transcript.absorb_commitment(prefix + "E_src", trace.commitments.at(prefix + "E_src").point);
        transcript.absorb_commitment(hidden_src_label(head_index), context.static_commitments.at(hidden_src_label(head_index)).point);
        trace.challenges["y_src_h" + std::to_string(head_index)] = transcript.challenge("y_src_h" + std::to_string(head_index));
        transcript.absorb_commitment(prefix + "H_prime", trace.commitments.at(prefix + "H_prime").point);
        transcript.absorb_commitment(prefix + "E_dst", trace.commitments.at(prefix + "E_dst").point);
        transcript.absorb_commitment(hidden_dst_label(head_index), context.static_commitments.at(hidden_dst_label(head_index)).point);
        trace.challenges["y_dst_h" + std::to_string(head_index)] = transcript.challenge("y_dst_h" + std::to_string(head_index));

        stage_start = Clock::now();
        const auto xi = trace.challenges.at("xi_h" + std::to_string(head_index));
        const auto& compressed = cached_hidden_compressed_trace(
            head_cache_prefix + ":compressed:" + xi.to_string(),
            h_prime,
            h_agg_pre,
            h_agg,
            edges,
            domains.edge->size,
            xi);
        const auto& h_star = compressed.h_star;
        const auto& h_star_edge = compressed.h_star_edge;
        add_metric(metrics != nullptr ? &metrics->hidden_h_star_trace_ms : nullptr, stage_start);

        stage_start = Clock::now();
        const auto& h_agg_pre_star = compressed.h_agg_pre_star;
        const auto& h_agg_pre_star_edge = compressed.h_agg_pre_star_edge;
        add_metric(metrics != nullptr ? &metrics->hidden_h_agg_pre_star_trace_ms : nullptr, stage_start);

        stage_start = Clock::now();
        const auto& h_agg_star = compressed.h_agg_star;
        const auto& h_agg_star_edge = compressed.h_agg_star_edge;
        add_metric(metrics != nullptr ? &metrics->hidden_h_agg_star_trace_ms : nullptr, stage_start);

        add_dynamic_commitment_batch(
            trace,
            {column_spec(prefix + "H_star", padded_column(h_star, domains.n->size), domains.n)},
            context.kzg,
            keep_trace_payloads,
            metrics);

        stage_start = Clock::now();
        const auto& hidden_parameters = cached_quantized_hidden_head_parameters(context, head_index);
        const auto& head_w = hidden_parameters.seq_kernel;
        const auto y_proj_h = trace.challenges.at("y_proj_h" + std::to_string(head_index));
        const auto y_proj_powers = powers(y_proj_h, d_h);
        const auto proj_b = linear_form_by_powers(head_w, y_proj_powers);
        const auto& proj_binding = cached_binding_trace(
            head_cache_prefix + ":proj:" + y_proj_h.to_string(),
            [&]() {
                return build_binding_trace(
                    local.features,
                    domains.n,
                    n_nodes,
                    y_proj_h,
                    proj_b,
                    domains.in,
                    d_in);
            });
        add_metric(metrics != nullptr ? &metrics->hidden_projection_trace_ms : nullptr, stage_start);

        stage_start = Clock::now();
        const auto y_src_h = trace.challenges.at("y_src_h" + std::to_string(head_index));
        const auto& src_binding = cached_binding_trace(
            head_cache_prefix + ":src:" + y_src_h.to_string(),
            [&]() {
                return build_binding_trace(
                    h_prime,
                    domains.n,
                    n_nodes,
                    y_src_h,
                    hidden_parameters.attn_src,
                    domains.d,
                    d_h);
            });
        add_metric(metrics != nullptr ? &metrics->hidden_src_attention_trace_ms : nullptr, stage_start);

        stage_start = Clock::now();
        const auto y_dst_h = trace.challenges.at("y_dst_h" + std::to_string(head_index));
        const auto& dst_binding = cached_binding_trace(
            head_cache_prefix + ":dst:" + y_dst_h.to_string(),
            [&]() {
                return build_binding_trace(
                    h_prime,
                    domains.n,
                    n_nodes,
                    y_dst_h,
                    hidden_parameters.attn_dst,
                    domains.d,
                    d_h);
            });
        transcript.absorb_commitment(prefix + "H_star", trace.commitments.at(prefix + "H_star").point);
        trace.challenges["y_star_h" + std::to_string(head_index)] = transcript.challenge("y_star_h" + std::to_string(head_index));
        add_metric(metrics != nullptr ? &metrics->hidden_dst_attention_trace_ms : nullptr, stage_start);

        stage_start = Clock::now();
        const auto xi_powers = powers(xi, d_h);
        const auto y_star_h = trace.challenges.at("y_star_h" + std::to_string(head_index));
        const auto& star_binding = cached_binding_trace(
            head_cache_prefix + ":star:" + y_star_h.to_string(),
            [&]() {
                return build_binding_trace(
                    h_prime,
                    domains.n,
                    n_nodes,
                    y_star_h,
                    xi_powers,
                    domains.d,
                    d_h);
            });
        add_metric(metrics != nullptr ? &metrics->hidden_h_star_trace_ms : nullptr, stage_start);

        transcript.absorb_commitment(prefix + "E_src", trace.commitments.at(prefix + "E_src").point);
        transcript.absorb_commitment(prefix + "H_star", trace.commitments.at(prefix + "H_star").point);
        const auto eta_src = transcript.challenge("eta_src_h" + std::to_string(head_index));
        const auto beta_src = transcript.challenge("beta_src_h" + std::to_string(head_index));
        trace.challenges["eta_src_h" + std::to_string(head_index)] = eta_src;
        trace.challenges["beta_src_h" + std::to_string(head_index)] = beta_src;
        const auto eta_src_powers = powers(eta_src, 2);

        transcript.absorb_commitment(prefix + "S", trace.commitments.at(prefix + "S").point);
        transcript.absorb_commitment(prefix + "Z", trace.commitments.at(prefix + "Z").point);
        transcript.absorb_commitment("V_T_L_x", context.static_commitments.at("V_T_L_x").point);
        transcript.absorb_commitment("V_T_L_y", context.static_commitments.at("V_T_L_y").point);
        const auto eta_l = transcript.challenge("eta_L_h" + std::to_string(head_index));
        const auto beta_l = transcript.challenge("beta_L_h" + std::to_string(head_index));
        trace.challenges["eta_L_h" + std::to_string(head_index)] = eta_l;
        trace.challenges["beta_L_h" + std::to_string(head_index)] = beta_l;

        transcript.absorb_commitment(prefix + "M", trace.commitments.at(prefix + "M").point);
        transcript.absorb_commitment(prefix + "M_edge", trace.commitments.at(prefix + "M_edge").point);
        transcript.absorb_commitment(prefix + "Delta", trace.commitments.at(prefix + "Delta").point);
        transcript.absorb_commitment("V_T_range", context.static_commitments.at("V_T_range").point);
        const auto beta_r = transcript.challenge("beta_R_h" + std::to_string(head_index));
        trace.challenges["beta_R_h" + std::to_string(head_index)] = beta_r;

        transcript.absorb_commitment(prefix + "Delta", trace.commitments.at(prefix + "Delta").point);
        transcript.absorb_commitment(prefix + "U", trace.commitments.at(prefix + "U").point);
        transcript.absorb_commitment("V_T_exp_x", context.static_commitments.at("V_T_exp_x").point);
        transcript.absorb_commitment("V_T_exp_y", context.static_commitments.at("V_T_exp_y").point);
        const auto eta_exp = transcript.challenge("eta_exp_h" + std::to_string(head_index));
        const auto beta_exp = transcript.challenge("beta_exp_h" + std::to_string(head_index));
        trace.challenges["eta_exp_h" + std::to_string(head_index)] = eta_exp;
        trace.challenges["beta_exp_h" + std::to_string(head_index)] = beta_exp;

        stage_start = Clock::now();
        const auto initial_lookup_table_ms = metric_value(&RunMetrics::lookup_table_pack_ms);
        const auto initial_lookup_query_ms = metric_value(&RunMetrics::lookup_query_pack_ms);
        const auto initial_lookup_mult_ms = metric_value(&RunMetrics::lookup_multiplicity_ms);
        const auto initial_lookup_acc_ms = metric_value(&RunMetrics::lookup_accumulator_ms);
        std::vector<FieldElement> table_l =
            cached_pair_lookup_table(prefix + "L", context.tables.lrelu, eta_l, domains.edge->size);
        add_metric(metrics != nullptr ? &metrics->lookup_table_pack_ms : nullptr, stage_start);
        const auto& lrelu_artifacts = cached_pair_lookup_artifacts(
            head_cache_prefix + ":lrelu_artifacts:" + eta_l.to_string(),
            [&]() {
                return build_pair_lookup_artifacts(
                    lrelu_index,
                    s,
                    z,
                    eta_l,
                    domains.edge->size,
                    n_edges,
                    prefix + "L",
                    metrics);
            });
        const auto& query_l = lrelu_artifacts.query;
        const auto& m_l = lrelu_artifacts.multiplicity;
        const auto lrelu_acc_start = Clock::now();
        auto r_l = build_logup_accumulator_cached_with_active_count(
            head_cache_prefix + ":lrelu",
            table_l,
            query_l,
            m_l,
            q_tbl_l,
            q_qry_l,
            beta_l,
            lookup_active_transition_limit(n_edges, domains.edge->size));
        add_metric(metrics != nullptr ? &metrics->lookup_accumulator_ms : nullptr, lrelu_acc_start);
        add_residual_metric(
            &RunMetrics::lookup_copy_convert_ms,
            elapsed_ms(stage_start, Clock::now()),
            {
                metric_value(&RunMetrics::lookup_table_pack_ms) - initial_lookup_table_ms,
                metric_value(&RunMetrics::lookup_query_pack_ms) - initial_lookup_query_ms,
                metric_value(&RunMetrics::lookup_multiplicity_ms) - initial_lookup_mult_ms,
                metric_value(&RunMetrics::lookup_accumulator_ms) - initial_lookup_acc_ms,
            });

        stage_start = Clock::now();
        std::vector<FieldElement> s_max(domains.edge->size, FieldElement::zero());
        for (const auto& [group_begin, group_end] : edge_groups_by_dst) {
            for (std::size_t k = group_begin; k < group_end; ++k) {
                if (delta[k].is_zero()) {
                    s_max[k] = FieldElement::one();
                    break;
                }
            }
        }
        add_metric(metrics != nullptr ? &metrics->hidden_edge_score_trace_ms : nullptr, stage_start);
        stage_start = Clock::now();
        const auto& c_max = cached_state_vector(
            head_cache_prefix + ":cmax",
            [&]() {
#if GATZK_ENABLE_CUDA_BACKEND
                if (cuda_max_counter_enabled()) {
                    return build_max_counter_state_cuda(s_max, q_new);
                }
#endif
                return build_max_counter_state(s_max, q_new);
            });
        add_metric(metrics != nullptr ? &metrics->state_machine_trace_ms : nullptr, stage_start);
        stage_start = Clock::now();
        const auto initial_range_table_ms = metric_value(&RunMetrics::lookup_table_pack_ms);
        const auto initial_range_query_ms = metric_value(&RunMetrics::lookup_query_pack_ms);
        const auto initial_range_mult_ms = metric_value(&RunMetrics::lookup_multiplicity_ms);
        const auto initial_range_acc_ms = metric_value(&RunMetrics::lookup_accumulator_ms);
        std::vector<FieldElement> table_r =
            cached_single_lookup_table(prefix + "R", context.tables.range, domains.edge->size);
        add_metric(metrics != nullptr ? &metrics->lookup_table_pack_ms : nullptr, stage_start);
        const auto& range_artifacts = cached_single_lookup_artifacts(
            head_cache_prefix + ":range_artifacts",
            [&]() {
                return build_single_lookup_artifacts(
                    range_index,
                    delta,
                    domains.edge->size,
                    n_edges,
                    prefix + "R",
                    metrics);
            });
        const auto& query_r = range_artifacts.query;
        const auto& m_r = range_artifacts.multiplicity;
        const auto range_acc_start = Clock::now();
        auto r_r = build_logup_accumulator_cached_with_active_count(
            head_cache_prefix + ":range",
            table_r,
            query_r,
            m_r,
            q_tbl_r,
            q_qry_r,
            beta_r,
            lookup_active_transition_limit(n_edges, domains.edge->size));
        add_metric(metrics != nullptr ? &metrics->lookup_accumulator_ms : nullptr, range_acc_start);
        add_residual_metric(
            &RunMetrics::lookup_copy_convert_ms,
            elapsed_ms(stage_start, Clock::now()),
            {
                metric_value(&RunMetrics::lookup_table_pack_ms) - initial_range_table_ms,
                metric_value(&RunMetrics::lookup_query_pack_ms) - initial_range_query_ms,
                metric_value(&RunMetrics::lookup_multiplicity_ms) - initial_range_mult_ms,
                metric_value(&RunMetrics::lookup_accumulator_ms) - initial_range_acc_ms,
            });

        stage_start = Clock::now();
        const auto initial_exp_table_ms = metric_value(&RunMetrics::lookup_table_pack_ms);
        const auto initial_exp_query_ms = metric_value(&RunMetrics::lookup_query_pack_ms);
        const auto initial_exp_mult_ms = metric_value(&RunMetrics::lookup_multiplicity_ms);
        const auto initial_exp_acc_ms = metric_value(&RunMetrics::lookup_accumulator_ms);
        std::vector<FieldElement> table_exp =
            cached_pair_lookup_table(prefix + "exp", context.tables.exp, eta_exp, domains.edge->size);
        add_metric(metrics != nullptr ? &metrics->lookup_table_pack_ms : nullptr, stage_start);
        const auto& exp_artifacts = cached_pair_lookup_artifacts(
            head_cache_prefix + ":exp_artifacts:" + eta_exp.to_string(),
            [&]() {
                return build_pair_lookup_artifacts(
                    exp_index,
                    delta,
                    u,
                    eta_exp,
                    domains.edge->size,
                    n_edges,
                    prefix + "exp",
                    metrics);
            });
        const auto& query_exp = exp_artifacts.query;
        const auto& m_exp = exp_artifacts.multiplicity;
        const auto exp_acc_start = Clock::now();
        auto r_exp = build_logup_accumulator_cached_with_active_count(
            head_cache_prefix + ":exp",
            table_exp,
            query_exp,
            m_exp,
            q_tbl_exp,
            q_qry_exp,
            beta_exp,
            lookup_active_transition_limit(n_edges, domains.edge->size));
        add_metric(metrics != nullptr ? &metrics->lookup_accumulator_ms : nullptr, exp_acc_start);
        add_residual_metric(
            &RunMetrics::lookup_copy_convert_ms,
            elapsed_ms(stage_start, Clock::now()),
            {
                metric_value(&RunMetrics::lookup_table_pack_ms) - initial_exp_table_ms,
                metric_value(&RunMetrics::lookup_query_pack_ms) - initial_exp_query_ms,
                metric_value(&RunMetrics::lookup_multiplicity_ms) - initial_exp_mult_ms,
                metric_value(&RunMetrics::lookup_accumulator_ms) - initial_exp_acc_ms,
            });

        const auto lambda_psq = transcript.challenge("lambda_psq_h" + std::to_string(head_index));
        trace.challenges["lambda_psq_h" + std::to_string(head_index)] = lambda_psq;
        stage_start = Clock::now();
        std::vector<FieldElement> widehat_v_pre_star(domains.edge->size, FieldElement::zero());
        for (std::size_t k = 0; k < n_edges; ++k) {
            widehat_v_pre_star[k] = alpha[k] * h_star_edge[k];
        }
        auto w_psq = build_weighted_sum(u, lambda_psq, widehat_v_pre_star, domains.edge->size);
        std::vector<FieldElement> t_psq(n_nodes, FieldElement::zero());
        for (std::size_t i = 0; i < n_nodes; ++i) {
            t_psq[i] = sum[i] + lambda_psq * h_agg_pre_star[i];
        }
        const auto t_psq_edge = build_group_target(t_psq, edges, domains.edge->size);
        add_metric(metrics != nullptr ? &metrics->hidden_softmax_chain_trace_ms : nullptr, stage_start);
        stage_start = Clock::now();
        const auto& psq = cached_state_vector(
            head_cache_prefix + ":psq:" + lambda_psq.to_string(),
            [&]() {
                return build_group_prefix_state(w_psq, q_new);
            });
        add_metric(metrics != nullptr ? &metrics->psq_trace_ms : nullptr, stage_start);
        add_metric(metrics != nullptr ? &metrics->state_machine_trace_ms : nullptr, stage_start);

        stage_start = Clock::now();
        std::vector<FieldElement> src_table(n_nodes, FieldElement::zero());
        std::vector<FieldElement> src_query(n_edges, FieldElement::zero());
        for (std::size_t i = 0; i < n_nodes; ++i) {
            src_table[i] = absolute_id_fields[i] + eta_src_powers[1] * e_src[i];
        }
        for (std::size_t k = 0; k < n_edges; ++k) {
            src_query[k] = edge_src_fields[k] + eta_src_powers[1] * h_star_edge[k];
        }
        add_metric(metrics != nullptr ? &metrics->hidden_route_trace_ms : nullptr, stage_start);
        stage_start = Clock::now();
        const auto& src_route = cached_route_trace(
            head_cache_prefix + ":src_route:" + eta_src.to_string() + ":" + beta_src.to_string(),
            [&]() {
                return build_route_trace(
                    src_table,
                    src_query,
                    src_multiplicity_template,
                    n_nodes,
                    n_edges,
                    domains.n,
                    domains.edge,
                    beta_src);
            });
        add_metric(metrics != nullptr ? &metrics->route_trace_ms : nullptr, stage_start);

        add_dynamic_commitment_batch(
            trace,
            {
                column_spec(prefix + "a_proj", proj_binding.a, domains.in),
                column_spec(prefix + "b_proj", proj_binding.b, domains.in),
                column_spec(prefix + "Acc_proj", proj_binding.acc, domains.in),
                column_spec(prefix + "a_src", src_binding.a, domains.d),
                column_spec(prefix + "b_src", src_binding.b, domains.d),
                column_spec(prefix + "Acc_src", src_binding.acc, domains.d),
                column_spec(prefix + "a_dst", dst_binding.a, domains.d),
                column_spec(prefix + "b_dst", dst_binding.b, domains.d),
                column_spec(prefix + "Acc_dst", dst_binding.acc, domains.d),
                column_spec(prefix + "a_star", star_binding.a, domains.d),
                column_spec(prefix + "b_star", star_binding.b, domains.d),
                column_spec(prefix + "Acc_star", star_binding.acc, domains.d),
                column_spec(prefix + "E_src_edge", e_src_edge, domains.edge),
                column_spec(prefix + "E_dst_edge", e_dst_edge, domains.edge),
                column_spec(prefix + "H_src_star_edge", h_star_edge, domains.edge),
                column_spec(prefix + "Table_src", src_route.table, domains.n),
                column_spec(prefix + "Query_src", src_route.query, domains.edge),
                column_spec(prefix + "m_src", src_route.multiplicity, domains.n),
                column_spec(prefix + "R_src_node", src_route.node_acc, domains.n),
                column_spec(prefix + "R_src", src_route.edge_acc, domains.edge),
                column_spec(prefix + "Table_L", table_l, domains.edge),
                column_spec(prefix + "Query_L", query_l, domains.edge),
                column_spec(prefix + "m_L", m_l, domains.edge),
                column_spec(prefix + "R_L", r_l, domains.edge),
                column_spec(prefix + "s_max", s_max, domains.edge),
                column_spec(prefix + "C_max", c_max, domains.edge),
                column_spec(prefix + "Table_R", table_r, domains.edge),
                column_spec(prefix + "Query_R", query_r, domains.edge),
                column_spec(prefix + "m_R", m_r, domains.edge),
                column_spec(prefix + "R_R", r_r, domains.edge),
                column_spec(prefix + "Table_exp", table_exp, domains.edge),
                column_spec(prefix + "Query_exp", query_exp, domains.edge),
                column_spec(prefix + "m_exp", m_exp, domains.edge),
                column_spec(prefix + "R_exp", r_exp, domains.edge),
                column_spec(prefix + "H_agg_pre_star", padded_column(h_agg_pre_star, domains.n->size), domains.n),
                column_spec(prefix + "H_agg_pre_star_edge", h_agg_pre_star_edge, domains.edge),
                column_spec(prefix + "widehat_v_pre_star", widehat_v_pre_star, domains.edge),
                column_spec(prefix + "w_psq", w_psq, domains.edge),
                column_spec(prefix + "T_psq", padded_column(t_psq, domains.n->size), domains.n),
                column_spec(prefix + "T_psq_edge", t_psq_edge, domains.edge),
                column_spec(prefix + "PSQ", psq, domains.edge),
                column_spec(prefix + "H_agg_star", padded_column(h_agg_star, domains.n->size), domains.n),
                column_spec(prefix + "H_agg_star_edge", h_agg_star_edge, domains.edge),
            },
            context.kzg,
            keep_trace_payloads,
            metrics);

        transcript.absorb_commitment(prefix + "T_psq", trace.commitments.at(prefix + "T_psq").point);
        transcript.absorb_commitment(prefix + "T_psq_edge", trace.commitments.at(prefix + "T_psq_edge").point);
        const auto eta_t = transcript.challenge("eta_t_h" + std::to_string(head_index));
        const auto beta_t = transcript.challenge("beta_t_h" + std::to_string(head_index));
        trace.challenges["eta_t_h" + std::to_string(head_index)] = eta_t;
        trace.challenges["beta_t_h" + std::to_string(head_index)] = beta_t;
        stage_start = Clock::now();
        const auto eta_t_powers = powers(eta_t, 2);
        std::vector<FieldElement> t_table(n_nodes, FieldElement::zero());
        std::vector<FieldElement> t_query(n_edges, FieldElement::zero());
        for (std::size_t i = 0; i < n_nodes; ++i) {
            t_table[i] = absolute_id_fields[i] + eta_t_powers[1] * t_psq[i];
        }
        for (std::size_t k = 0; k < n_edges; ++k) {
            t_query[k] = edge_dst_fields[k] + eta_t_powers[1] * t_psq_edge[k];
        }
        add_metric(metrics != nullptr ? &metrics->hidden_route_trace_ms : nullptr, stage_start);
        stage_start = Clock::now();
        const auto& t_route = cached_route_trace(
            head_cache_prefix + ":t_route:" + eta_t.to_string() + ":" + beta_t.to_string(),
            [&]() {
                return build_route_trace(
                    t_table,
                    t_query,
                    dst_multiplicity_template,
                    n_nodes,
                    n_edges,
                    domains.n,
                    domains.edge,
                    beta_t);
            });
        add_metric(metrics != nullptr ? &metrics->route_trace_ms : nullptr, stage_start);

        transcript.absorb_commitment(prefix + "H_agg_pre", trace.commitments.at(prefix + "H_agg_pre").point);
        transcript.absorb_commitment(prefix + "H_agg_pre_star", trace.commitments.at(prefix + "H_agg_pre_star").point);
        trace.challenges["y_agg_pre_h" + std::to_string(head_index)] = transcript.challenge("y_agg_pre_h" + std::to_string(head_index));
        transcript.absorb_commitment(prefix + "H_agg_pre", trace.commitments.at(prefix + "H_agg_pre").point);
        transcript.absorb_commitment(prefix + "H_agg", trace.commitments.at(prefix + "H_agg").point);
        transcript.absorb_commitment("V_T_ELU_x", context.static_commitments.at("V_T_ELU_x").point);
        transcript.absorb_commitment("V_T_ELU_y", context.static_commitments.at("V_T_ELU_y").point);
        const auto eta_elu = transcript.challenge("eta_ELU_h" + std::to_string(head_index));
        const auto beta_elu = transcript.challenge("beta_ELU_h" + std::to_string(head_index));
        trace.challenges["eta_ELU_h" + std::to_string(head_index)] = eta_elu;
        trace.challenges["beta_ELU_h" + std::to_string(head_index)] = beta_elu;
        transcript.absorb_commitment(prefix + "H_agg", trace.commitments.at(prefix + "H_agg").point);
        transcript.absorb_commitment(prefix + "H_agg_star", trace.commitments.at(prefix + "H_agg_star").point);
        trace.challenges["y_agg_h" + std::to_string(head_index)] = transcript.challenge("y_agg_h" + std::to_string(head_index));
        const auto eta_dst = transcript.challenge("eta_dst_h" + std::to_string(head_index));
        const auto beta_dst = transcript.challenge("beta_dst_h" + std::to_string(head_index));
        trace.challenges["eta_dst_h" + std::to_string(head_index)] = eta_dst;
        trace.challenges["beta_dst_h" + std::to_string(head_index)] = beta_dst;

        stage_start = Clock::now();
        const auto y_agg_pre_h = trace.challenges.at("y_agg_pre_h" + std::to_string(head_index));
        const auto& agg_pre_binding = cached_binding_trace(
            head_cache_prefix + ":agg_pre:" + y_agg_pre_h.to_string(),
            [&]() {
                return build_binding_trace(
                    h_agg_pre,
                    domains.n,
                    n_nodes,
                    y_agg_pre_h,
                    xi_powers,
                    domains.d,
                    d_h);
            });
        add_metric(metrics != nullptr ? &metrics->hidden_h_agg_pre_star_trace_ms : nullptr, stage_start);

        stage_start = Clock::now();
        const auto y_agg_h = trace.challenges.at("y_agg_h" + std::to_string(head_index));
        const auto& agg_binding = cached_binding_trace(
            head_cache_prefix + ":agg:" + y_agg_h.to_string(),
            [&]() {
                return build_binding_trace(
                    h_agg,
                    domains.n,
                    n_nodes,
                    y_agg_h,
                    xi_powers,
                    domains.d,
                    d_h);
            });
        add_metric(metrics != nullptr ? &metrics->hidden_h_agg_star_trace_ms : nullptr, stage_start);

        stage_start = Clock::now();
        const auto initial_elu_table_ms = metric_value(&RunMetrics::lookup_table_pack_ms);
        const auto initial_elu_query_ms = metric_value(&RunMetrics::lookup_query_pack_ms);
        const auto initial_elu_mult_ms = metric_value(&RunMetrics::lookup_multiplicity_ms);
        const auto initial_elu_acc_ms = metric_value(&RunMetrics::lookup_accumulator_ms);
        std::vector<FieldElement> table_elu =
            cached_pair_lookup_table(prefix + "ELU", context.tables.elu, eta_elu, domains.edge->size);
        add_metric(metrics != nullptr ? &metrics->lookup_table_pack_ms : nullptr, stage_start);
        std::vector<FieldElement> query_elu(domains.edge->size, FieldElement::zero());
        std::vector<FieldElement> agg_pre_flat(domains.edge->size, FieldElement::zero());
        std::vector<FieldElement> agg_flat(domains.edge->size, FieldElement::zero());
        for (std::size_t node = 0; node < n_nodes; ++node) {
            for (std::size_t col = 0; col < d_h; ++col) {
                const auto flat_index = node * d_h + col;
                agg_pre_flat[flat_index] = h_agg_pre[node][col];
                agg_flat[flat_index] = h_agg[node][col];
            }
        }
        const auto& elu_artifacts = cached_pair_lookup_artifacts(
            head_cache_prefix + ":elu_artifacts:" + eta_elu.to_string(),
            [&]() {
                return build_pair_lookup_artifacts(
                    elu_index,
                    agg_pre_flat,
                    agg_flat,
                    eta_elu,
                    domains.edge->size,
                    n_nodes * d_h,
                    prefix + "ELU",
                    metrics);
            });
        query_elu = elu_artifacts.query;
        const auto& m_elu = elu_artifacts.multiplicity;
        const auto elu_acc_start = Clock::now();
        auto r_elu = build_logup_accumulator_cached_with_active_count(
            head_cache_prefix + ":elu",
            table_elu,
            query_elu,
            m_elu,
            q_tbl_elu,
            q_qry_elu,
            beta_elu,
            lookup_active_transition_limit(n_nodes * d_h, domains.edge->size));
        add_metric(metrics != nullptr ? &metrics->lookup_accumulator_ms : nullptr, elu_acc_start);
        add_residual_metric(
            &RunMetrics::lookup_copy_convert_ms,
            elapsed_ms(stage_start, Clock::now()),
            {
                metric_value(&RunMetrics::lookup_table_pack_ms) - initial_elu_table_ms,
                metric_value(&RunMetrics::lookup_query_pack_ms) - initial_elu_query_ms,
                metric_value(&RunMetrics::lookup_multiplicity_ms) - initial_elu_mult_ms,
                metric_value(&RunMetrics::lookup_accumulator_ms) - initial_elu_acc_ms,
            });

        stage_start = Clock::now();
        const auto eta_dst_powers = powers(eta_dst, 6);
        std::vector<FieldElement> dst_table(n_nodes, FieldElement::zero());
        std::vector<FieldElement> dst_query(n_edges, FieldElement::zero());
        for (std::size_t i = 0; i < n_nodes; ++i) {
            dst_table[i] =
                absolute_id_fields[i]
                + eta_dst_powers[1] * e_dst[i]
                + eta_dst_powers[2] * m[i]
                + eta_dst_powers[3] * sum[i]
                + eta_dst_powers[4] * inv[i]
                + eta_dst_powers[5] * h_agg_star[i];
        }
        for (std::size_t k = 0; k < n_edges; ++k) {
            dst_query[k] =
                edge_dst_fields[k]
                + eta_dst_powers[1] * e_dst_edge[k]
                + eta_dst_powers[2] * m_edge[k]
                + eta_dst_powers[3] * sum_edge[k]
                + eta_dst_powers[4] * inv_edge[k]
                + eta_dst_powers[5] * h_agg_star_edge[k];
        }
        add_metric(metrics != nullptr ? &metrics->hidden_route_trace_ms : nullptr, stage_start);
        stage_start = Clock::now();
        const auto& dst_route = cached_route_trace(
            head_cache_prefix + ":dst_route:" + eta_dst.to_string() + ":" + beta_dst.to_string(),
            [&]() {
                return build_route_trace(
                    dst_table,
                    dst_query,
                    dst_multiplicity_template,
                    n_nodes,
                    n_edges,
                    domains.n,
                    domains.edge,
                    beta_dst);
            });
        add_metric(metrics != nullptr ? &metrics->route_trace_ms : nullptr, stage_start);

        add_dynamic_commitment_batch(
            trace,
            {
                column_spec(prefix + "a_agg_pre", agg_pre_binding.a, domains.d),
                column_spec(prefix + "b_agg_pre", agg_pre_binding.b, domains.d),
                column_spec(prefix + "Acc_agg_pre", agg_pre_binding.acc, domains.d),
                column_spec(prefix + "a_agg", agg_binding.a, domains.d),
                column_spec(prefix + "b_agg", agg_binding.b, domains.d),
                column_spec(prefix + "Acc_agg", agg_binding.acc, domains.d),
                column_spec(prefix + "H_agg_pre_flat", agg_pre_flat, domains.edge),
                column_spec(prefix + "H_agg_flat", agg_flat, domains.edge),
                column_spec(prefix + "Table_ELU", table_elu, domains.edge),
                column_spec(prefix + "Query_ELU", query_elu, domains.edge),
                column_spec(prefix + "m_ELU", m_elu, domains.edge),
                column_spec(prefix + "R_ELU", r_elu, domains.edge),
                column_spec(prefix + "Table_t", t_route.table, domains.n),
                column_spec(prefix + "Query_t", t_route.query, domains.edge),
                column_spec(prefix + "m_t", t_route.multiplicity, domains.n),
                column_spec(prefix + "R_t_node", t_route.node_acc, domains.n),
                column_spec(prefix + "R_t", t_route.edge_acc, domains.edge),
            },
            context.kzg,
            keep_trace_payloads,
            metrics);
        add_dynamic_commitment_batch(
            trace,
            {
                column_spec(prefix + "Table_dst", dst_route.table, domains.n),
                column_spec(prefix + "Query_dst", dst_route.query, domains.edge),
                column_spec(prefix + "m_dst", dst_route.multiplicity, domains.n),
                column_spec(prefix + "R_dst_node", dst_route.node_acc, domains.n),
                column_spec(prefix + "R_dst", dst_route.edge_acc, domains.edge),
            },
            context.kzg,
            keep_trace_payloads,
            metrics);
        trace.witness_scalars["S_src_h" + std::to_string(head_index)] = src_route.total;
        trace.witness_scalars["S_dst_h" + std::to_string(head_index)] = dst_route.total;
        trace.witness_scalars["S_t_h" + std::to_string(head_index)] = t_route.total;
        trace.external_evaluations["mu_h" + std::to_string(head_index) + "_proj"] =
            matrix_row_major_evaluation(h_prime, trace.challenges.at("y_proj_h" + std::to_string(head_index)));
        trace.external_evaluations["mu_h" + std::to_string(head_index) + "_src"] =
            trace.polynomials.at(prefix + "E_src").evaluate(trace.challenges.at("y_src_h" + std::to_string(head_index)));
        trace.external_evaluations["mu_h" + std::to_string(head_index) + "_dst"] =
            trace.polynomials.at(prefix + "E_dst").evaluate(trace.challenges.at("y_dst_h" + std::to_string(head_index)));
        trace.external_evaluations["mu_h" + std::to_string(head_index) + "_star"] =
            trace.polynomials.at(prefix + "H_star").evaluate(trace.challenges.at("y_star_h" + std::to_string(head_index)));
        trace.external_evaluations["mu_h" + std::to_string(head_index) + "_agg_pre"] =
            trace.polynomials.at(prefix + "H_agg_pre_star").evaluate(trace.challenges.at("y_agg_pre_h" + std::to_string(head_index)));
        trace.external_evaluations["mu_h" + std::to_string(head_index) + "_agg"] =
            trace.polynomials.at(prefix + "H_agg_star").evaluate(trace.challenges.at("y_agg_h" + std::to_string(head_index)));
    };

    for (std::size_t head_index = 0; head_index < context.model.hidden_heads.size(); ++head_index) {
        commit_hidden_head(head_index);
    }

    stage_start = Clock::now();
    const auto hidden_stack_start = stage_start;
    const auto hidden_stack_initial_zkmap_ms = metric_value(&RunMetrics::zkmap_trace_ms);
    const auto hidden_stack_initial_padding_ms = metric_value(&RunMetrics::padding_selector_trace_ms);
    const auto hidden_stack_initial_field_conversion_ms = metric_value(&RunMetrics::field_conversion_residual_ms);
    const auto hidden_stack_initial_commit_dynamic_ms = metric_value(&RunMetrics::commit_dynamic_ms);
    const auto hidden_stack_initial_object_ms = metric_value(&RunMetrics::hidden_output_object_residual_ms);
    const auto h_cat = quantize_matrix(forward.hidden_concat);
    add_dynamic_commitment_batch(
        trace,
        {matrix_commitment_only_spec("P_H_cat", h_cat)},
        context.kzg,
        keep_trace_payloads,
        metrics);
    transcript.absorb_commitment("P_H_cat", trace.commitments.at("P_H_cat").point);
    trace.challenges["xi_cat"] = transcript.challenge("xi_cat");
    const auto object_stage_start = Clock::now();
    const auto h_cat_star = compress_rows_with_challenge(h_cat, trace.challenges.at("xi_cat"));
    add_metric(metrics != nullptr ? &metrics->hidden_output_object_residual_ms : nullptr, object_stage_start);
    add_metric(metrics != nullptr ? &metrics->witness_materialization_ms : nullptr, stage_start);
    add_metric(metrics != nullptr ? &metrics->field_conversion_residual_ms : nullptr, stage_start);
    add_dynamic_commitment_batch(
        trace,
        {column_spec("P_H_cat_star", padded_column(h_cat_star, domains.n->size), domains.n)},
        context.kzg,
        keep_trace_payloads,
        metrics);
    transcript.absorb_commitment("P_H_cat_star", trace.commitments.at("P_H_cat_star").point);
    trace.challenges["y_cat"] = transcript.challenge("y_cat");
    stage_start = Clock::now();
    const auto y_cat = trace.challenges.at("y_cat");
    const auto xi_cat_powers = powers(trace.challenges.at("xi_cat"), d_cat);
    const auto& cat_binding = cached_binding_trace(
        cache_prefix + ":cat:" + y_cat.to_string(),
        [&]() {
            return build_binding_trace(
                h_cat,
                domains.n,
                n_nodes,
                y_cat,
                xi_cat_powers,
                domains.cat,
                d_cat);
        });
    add_metric(metrics != nullptr ? &metrics->zkmap_trace_ms : nullptr, stage_start);
    add_dynamic_commitment_batch(
        trace,
        {
            column_spec("P_cat_a", cat_binding.a, domains.cat),
            column_spec("P_cat_b", cat_binding.b, domains.cat),
            column_spec("P_cat_Acc", cat_binding.acc, domains.cat),
        },
        context.kzg,
        keep_trace_payloads,
        metrics);
    trace.external_evaluations["mu_cat"] = trace.polynomials.at("P_H_cat_star").evaluate(trace.challenges.at("y_cat"));
    add_residual_metric(
        &RunMetrics::hidden_head_trace_ms,
        elapsed_ms(hidden_stack_start, Clock::now()) - (metric_value(&RunMetrics::commit_dynamic_ms) - hidden_stack_initial_commit_dynamic_ms),
        {
            metric_value(&RunMetrics::zkmap_trace_ms) - hidden_stack_initial_zkmap_ms,
            metric_value(&RunMetrics::padding_selector_trace_ms) - hidden_stack_initial_padding_ms,
            metric_value(&RunMetrics::field_conversion_residual_ms) - hidden_stack_initial_field_conversion_ms,
            metric_value(&RunMetrics::hidden_output_object_residual_ms) - hidden_stack_initial_object_ms,
        });

    stage_start = Clock::now();
    const auto output_trace_start = stage_start;
    const auto output_cache_prefix = cache_prefix + ":output";
    const auto output_initial_lookup_ms = metric_value(&RunMetrics::lookup_trace_ms);
    const auto output_initial_route_ms = metric_value(&RunMetrics::route_trace_ms);
    const auto output_initial_psq_ms = metric_value(&RunMetrics::psq_trace_ms);
    const auto output_initial_zkmap_ms = metric_value(&RunMetrics::zkmap_trace_ms);
    const auto output_initial_state_machine_ms = metric_value(&RunMetrics::state_machine_trace_ms);
    const auto output_initial_padding_ms = metric_value(&RunMetrics::padding_selector_trace_ms);
    const auto output_initial_route_pack_ms = metric_value(&RunMetrics::route_pack_residual_ms);
    const auto output_initial_field_conversion_ms = metric_value(&RunMetrics::field_conversion_residual_ms);
    const auto output_initial_object_ms = metric_value(&RunMetrics::hidden_output_object_residual_ms);
    const auto output_initial_commit_dynamic_ms = metric_value(&RunMetrics::commit_dynamic_ms);
    const auto y_prime = quantize_matrix(forward.output_head_trace.H_prime);
    const auto out_e_src = quantize_vector(forward.output_head_trace.E_src);
    const auto out_e_dst = quantize_vector(forward.output_head_trace.E_dst);
    const auto out_s = quantize_vector(forward.output_head_trace.S);
    const auto out_z = quantize_vector(forward.output_head_trace.Z);
    const auto out_m = quantize_vector(forward.output_head_trace.M);
    const auto out_delta = quantize_vector(forward.output_head_trace.Delta);
    const auto out_u = quantize_vector(forward.output_head_trace.U);
    const auto out_sum = quantize_vector(forward.output_head_trace.Sum);
    const auto out_inv = quantize_vector(forward.output_head_trace.inv);
    const auto out_alpha = quantize_vector(forward.output_head_trace.alpha);
    const auto y_lin_matrix = quantize_matrix(forward.Y_lin);
    auto y_matrix = y_lin_matrix;
    const auto output_bias_quantized = quantize_vector(context.model.output_layer.heads.front().output_bias_fp);
    for (auto& row : y_matrix) {
        for (std::size_t col = 0; col < row.size() && col < output_bias_quantized.size(); ++col) {
            row[col] += output_bias_quantized[col];
        }
    }
    const auto out_m_edge = broadcast_node_values(out_m, edges, false, domains.edge->size);
    const auto out_sum_edge = broadcast_node_values(out_sum, edges, false, domains.edge->size);
    const auto out_inv_edge = broadcast_node_values(out_inv, edges, false, domains.edge->size);
    add_metric(metrics != nullptr ? &metrics->witness_materialization_ms : nullptr, stage_start);
    add_metric(metrics != nullptr ? &metrics->field_conversion_residual_ms : nullptr, stage_start);
    add_dynamic_commitment_batch(
        trace,
        {
            matrix_commitment_only_spec("P_out_Y_prime", y_prime),
            column_spec("P_out_E_src", padded_column(out_e_src, domains.n->size), domains.n),
            column_spec("P_out_E_dst", padded_column(out_e_dst, domains.n->size), domains.n),
            column_spec("P_out_S", padded_column(out_s, domains.edge->size), domains.edge),
            column_spec("P_out_Z", padded_column(out_z, domains.edge->size), domains.edge),
            column_spec("P_out_M", padded_column(out_m, domains.n->size), domains.n),
            column_spec("P_out_M_edge", out_m_edge, domains.edge),
            column_spec("P_out_Delta", padded_column(out_delta, domains.edge->size), domains.edge),
            column_spec("P_out_U", padded_column(out_u, domains.edge->size), domains.edge),
            column_spec("P_out_Sum", padded_column(out_sum, domains.n->size), domains.n),
            column_spec("P_out_Sum_edge", out_sum_edge, domains.edge),
            column_spec("P_out_inv", padded_column(out_inv, domains.n->size), domains.n),
            column_spec("P_out_inv_edge", out_inv_edge, domains.edge),
            column_spec("P_out_alpha", padded_column(out_alpha, domains.edge->size), domains.edge),
            matrix_commitment_only_spec("P_Y_lin", y_lin_matrix),
            matrix_commitment_only_spec("P_Y", y_matrix),
        },
        context.kzg,
        keep_trace_payloads,
        metrics);

    transcript.absorb_commitment("P_H_cat", trace.commitments.at("P_H_cat").point);
    transcript.absorb_commitment("P_out_Y_prime", trace.commitments.at("P_out_Y_prime").point);
    transcript.absorb_commitment(output_weight_label(), context.static_commitments.at(output_weight_label()).point);
    trace.challenges["y_proj_out"] = transcript.challenge("y_proj_out");
    trace.challenges["xi_out"] = transcript.challenge("xi_out");
    transcript.absorb_commitment("P_out_Y_prime", trace.commitments.at("P_out_Y_prime").point);
    transcript.absorb_commitment("P_out_E_src", trace.commitments.at("P_out_E_src").point);
    transcript.absorb_commitment(output_src_label(), context.static_commitments.at(output_src_label()).point);
    trace.challenges["y_src_out"] = transcript.challenge("y_src_out");
    transcript.absorb_commitment("P_out_Y_prime", trace.commitments.at("P_out_Y_prime").point);
    transcript.absorb_commitment("P_out_E_dst", trace.commitments.at("P_out_E_dst").point);
    transcript.absorb_commitment(output_dst_label(), context.static_commitments.at(output_dst_label()).point);
    trace.challenges["y_dst_out"] = transcript.challenge("y_dst_out");
    transcript.absorb_commitment("P_out_S", trace.commitments.at("P_out_S").point);
    transcript.absorb_commitment("P_out_Z", trace.commitments.at("P_out_Z").point);
    transcript.absorb_commitment("V_T_L_x", context.static_commitments.at("V_T_L_x").point);
    transcript.absorb_commitment("V_T_L_y", context.static_commitments.at("V_T_L_y").point);
    trace.challenges["eta_L_out"] = transcript.challenge("eta_L_out");
    trace.challenges["beta_L_out"] = transcript.challenge("beta_L_out");
    transcript.absorb_commitment("P_out_M", trace.commitments.at("P_out_M").point);
    transcript.absorb_commitment("P_out_M_edge", trace.commitments.at("P_out_M_edge").point);
    transcript.absorb_commitment("P_out_Delta", trace.commitments.at("P_out_Delta").point);
    transcript.absorb_commitment("V_T_range", context.static_commitments.at("V_T_range").point);
    trace.challenges["beta_R_out"] = transcript.challenge("beta_R_out");
    transcript.absorb_commitment("P_out_Delta", trace.commitments.at("P_out_Delta").point);
    transcript.absorb_commitment("P_out_U", trace.commitments.at("P_out_U").point);
    transcript.absorb_commitment("V_T_exp_x", context.static_commitments.at("V_T_exp_x").point);
    transcript.absorb_commitment("V_T_exp_y", context.static_commitments.at("V_T_exp_y").point);
    trace.challenges["eta_exp_out"] = transcript.challenge("eta_exp_out");
    trace.challenges["beta_exp_out"] = transcript.challenge("beta_exp_out");

    stage_start = Clock::now();
    const auto y_prime_star = compress_rows_with_challenge(y_prime, trace.challenges.at("xi_out"));
    const auto y_prime_star_edge = broadcast_node_values(y_prime_star, edges, true, domains.edge->size);
    const auto y_star = compress_rows_with_challenge(y_lin_matrix, trace.challenges.at("xi_out"));
    const auto y_star_edge = broadcast_node_values(y_star, edges, false, domains.edge->size);
    add_metric(metrics != nullptr ? &metrics->hidden_output_object_residual_ms : nullptr, stage_start);
    const auto eta_src_out = transcript.challenge("eta_src_out");
    const auto beta_src_out = transcript.challenge("beta_src_out");
    trace.challenges["eta_src_out"] = eta_src_out;
    trace.challenges["beta_src_out"] = beta_src_out;
    stage_start = Clock::now();
    const auto eta_src_out_powers = powers(eta_src_out, 2);
    std::vector<FieldElement> out_src_table(n_nodes, FieldElement::zero());
    std::vector<FieldElement> out_src_query(n_edges, FieldElement::zero());
    for (std::size_t i = 0; i < n_nodes; ++i) {
        out_src_table[i] = absolute_id_fields[i] + eta_src_out_powers[1] * out_e_src[i];
    }
    for (std::size_t k = 0; k < n_edges; ++k) {
        out_src_query[k] = edge_src_fields[k] + eta_src_out_powers[1] * y_prime_star_edge[k];
    }
    add_metric(metrics != nullptr ? &metrics->route_pack_residual_ms : nullptr, stage_start);
    stage_start = Clock::now();
    const auto& out_src_route = cached_route_trace(
        output_cache_prefix + ":src_route:" + eta_src_out.to_string() + ":" + beta_src_out.to_string(),
        [&]() {
            return build_route_trace(
                out_src_table,
                out_src_query,
                src_multiplicity_template,
                n_nodes,
                n_edges,
                domains.n,
                domains.edge,
                beta_src_out);
        });
    add_metric(metrics != nullptr ? &metrics->route_trace_ms : nullptr, stage_start);

    stage_start = Clock::now();
    const auto initial_out_l_table_ms = metric_value(&RunMetrics::lookup_table_pack_ms);
    const auto initial_out_l_query_ms = metric_value(&RunMetrics::lookup_query_pack_ms);
    const auto initial_out_l_mult_ms = metric_value(&RunMetrics::lookup_multiplicity_ms);
    const auto initial_out_l_acc_ms = metric_value(&RunMetrics::lookup_accumulator_ms);
    std::vector<FieldElement> out_table_l =
        cached_pair_lookup_table("P_out_L", context.tables.lrelu, trace.challenges.at("eta_L_out"), domains.edge->size);
    add_metric(metrics != nullptr ? &metrics->lookup_table_pack_ms : nullptr, stage_start);
    const auto& out_l_artifacts = cached_pair_lookup_artifacts(
        output_cache_prefix + ":lrelu_artifacts:" + trace.challenges.at("eta_L_out").to_string(),
        [&]() {
            return build_pair_lookup_artifacts(
                lrelu_index,
                out_s,
                out_z,
                trace.challenges.at("eta_L_out"),
                domains.edge->size,
                n_edges,
                "P_out_L",
                metrics);
        });
    const auto& out_query_l = out_l_artifacts.query;
    const auto& out_m_l = out_l_artifacts.multiplicity;
    const auto out_l_acc_start = Clock::now();
    auto out_r_l = build_logup_accumulator_cached_with_active_count(
        output_cache_prefix + ":lrelu",
        out_table_l,
        out_query_l,
        out_m_l,
        q_tbl_l,
        q_qry_l,
        trace.challenges.at("beta_L_out"),
        lookup_active_transition_limit(n_edges, domains.edge->size));
    add_metric(metrics != nullptr ? &metrics->lookup_accumulator_ms : nullptr, out_l_acc_start);
    add_residual_metric(
        &RunMetrics::lookup_copy_convert_ms,
        elapsed_ms(stage_start, Clock::now()),
        {
            metric_value(&RunMetrics::lookup_table_pack_ms) - initial_out_l_table_ms,
            metric_value(&RunMetrics::lookup_query_pack_ms) - initial_out_l_query_ms,
            metric_value(&RunMetrics::lookup_multiplicity_ms) - initial_out_l_mult_ms,
            metric_value(&RunMetrics::lookup_accumulator_ms) - initial_out_l_acc_ms,
        });

    stage_start = Clock::now();
    std::vector<FieldElement> out_s_max(domains.edge->size, FieldElement::zero());
    for (const auto& [group_begin, group_end] : edge_groups_by_dst) {
        for (std::size_t k = group_begin; k < group_end; ++k) {
            if (out_delta[k].is_zero()) {
                out_s_max[k] = FieldElement::one();
                break;
            }
        }
    }
    const auto& out_c_max = cached_state_vector(
        output_cache_prefix + ":cmax",
        [&]() {
#if GATZK_ENABLE_CUDA_BACKEND
            if (cuda_max_counter_enabled()) {
                return build_max_counter_state_cuda(out_s_max, q_new);
            }
#endif
            return build_max_counter_state(out_s_max, q_new);
        });
    add_metric(metrics != nullptr ? &metrics->state_machine_trace_ms : nullptr, stage_start);
    stage_start = Clock::now();
    const auto initial_out_r_table_ms = metric_value(&RunMetrics::lookup_table_pack_ms);
    const auto initial_out_r_query_ms = metric_value(&RunMetrics::lookup_query_pack_ms);
    const auto initial_out_r_mult_ms = metric_value(&RunMetrics::lookup_multiplicity_ms);
    const auto initial_out_r_acc_ms = metric_value(&RunMetrics::lookup_accumulator_ms);
    std::vector<FieldElement> out_table_r =
        cached_single_lookup_table("P_out_R", context.tables.range, domains.edge->size);
    add_metric(metrics != nullptr ? &metrics->lookup_table_pack_ms : nullptr, stage_start);
    const auto& out_r_artifacts = cached_single_lookup_artifacts(
        output_cache_prefix + ":range_artifacts",
        [&]() {
            return build_single_lookup_artifacts(
                range_index,
                out_delta,
                domains.edge->size,
                n_edges,
                "P_out_R",
                metrics);
        });
    const auto& out_query_r = out_r_artifacts.query;
    const auto& out_m_r = out_r_artifacts.multiplicity;
    const auto out_r_acc_start = Clock::now();
    auto out_r_r = build_logup_accumulator_cached_with_active_count(
        output_cache_prefix + ":range",
        out_table_r,
        out_query_r,
        out_m_r,
        q_tbl_r,
        q_qry_r,
        trace.challenges.at("beta_R_out"),
        lookup_active_transition_limit(n_edges, domains.edge->size));
    add_metric(metrics != nullptr ? &metrics->lookup_accumulator_ms : nullptr, out_r_acc_start);
    add_residual_metric(
        &RunMetrics::lookup_copy_convert_ms,
        elapsed_ms(stage_start, Clock::now()),
        {
            metric_value(&RunMetrics::lookup_table_pack_ms) - initial_out_r_table_ms,
            metric_value(&RunMetrics::lookup_query_pack_ms) - initial_out_r_query_ms,
            metric_value(&RunMetrics::lookup_multiplicity_ms) - initial_out_r_mult_ms,
            metric_value(&RunMetrics::lookup_accumulator_ms) - initial_out_r_acc_ms,
        });

    stage_start = Clock::now();
    const auto initial_out_exp_table_ms = metric_value(&RunMetrics::lookup_table_pack_ms);
    const auto initial_out_exp_query_ms = metric_value(&RunMetrics::lookup_query_pack_ms);
    const auto initial_out_exp_mult_ms = metric_value(&RunMetrics::lookup_multiplicity_ms);
    const auto initial_out_exp_acc_ms = metric_value(&RunMetrics::lookup_accumulator_ms);
    std::vector<FieldElement> out_table_exp =
        cached_pair_lookup_table("P_out_exp", context.tables.exp, trace.challenges.at("eta_exp_out"), domains.edge->size);
    add_metric(metrics != nullptr ? &metrics->lookup_table_pack_ms : nullptr, stage_start);
    const auto& out_exp_artifacts = cached_pair_lookup_artifacts(
        output_cache_prefix + ":exp_artifacts:" + trace.challenges.at("eta_exp_out").to_string(),
        [&]() {
            return build_pair_lookup_artifacts(
                exp_index,
                out_delta,
                out_u,
                trace.challenges.at("eta_exp_out"),
                domains.edge->size,
                n_edges,
                "P_out_exp",
                metrics);
        });
    const auto& out_query_exp = out_exp_artifacts.query;
    const auto& out_m_exp = out_exp_artifacts.multiplicity;
    const auto out_exp_acc_start = Clock::now();
    auto out_r_exp = build_logup_accumulator_cached_with_active_count(
        output_cache_prefix + ":exp",
        out_table_exp,
        out_query_exp,
        out_m_exp,
        q_tbl_exp,
        q_qry_exp,
        trace.challenges.at("beta_exp_out"),
        lookup_active_transition_limit(n_edges, domains.edge->size));
    add_metric(metrics != nullptr ? &metrics->lookup_accumulator_ms : nullptr, out_exp_acc_start);
    add_residual_metric(
        &RunMetrics::lookup_copy_convert_ms,
        elapsed_ms(stage_start, Clock::now()),
        {
            metric_value(&RunMetrics::lookup_table_pack_ms) - initial_out_exp_table_ms,
            metric_value(&RunMetrics::lookup_query_pack_ms) - initial_out_exp_query_ms,
            metric_value(&RunMetrics::lookup_multiplicity_ms) - initial_out_exp_mult_ms,
            metric_value(&RunMetrics::lookup_accumulator_ms) - initial_out_exp_acc_ms,
        });

    const auto lambda_out = transcript.challenge("lambda_out");
    trace.challenges["lambda_out"] = lambda_out;
    stage_start = Clock::now();
    std::vector<FieldElement> widehat_y_star(domains.edge->size, FieldElement::zero());
    for (std::size_t k = 0; k < n_edges; ++k) {
        widehat_y_star[k] = out_alpha[k] * y_prime_star_edge[k];
    }
    auto w_out = build_weighted_sum(quantize_vector(forward.output_head_trace.U), lambda_out, widehat_y_star, domains.edge->size);
    std::vector<FieldElement> t_out(n_nodes, FieldElement::zero());
    for (std::size_t i = 0; i < n_nodes; ++i) {
        t_out[i] = out_sum[i] + lambda_out * y_star[i];
    }
    const auto t_out_edge = build_group_target(t_out, edges, domains.edge->size);
    const auto& psq_out = cached_state_vector(
        output_cache_prefix + ":psq:" + lambda_out.to_string(),
        [&]() {
            return build_group_prefix_state(w_out, q_new);
        });
    add_metric(metrics != nullptr ? &metrics->psq_trace_ms : nullptr, stage_start);
    add_metric(metrics != nullptr ? &metrics->state_machine_trace_ms : nullptr, stage_start);
    add_dynamic_commitment_batch(
        trace,
        {
            column_spec("P_out_E_src_edge", broadcast_node_values(out_e_src, edges, true, domains.edge->size), domains.edge),
            column_spec("P_out_E_dst_edge", broadcast_node_values(out_e_dst, edges, false, domains.edge->size), domains.edge),
            column_spec("P_out_Y_prime_star", padded_column(y_prime_star, domains.n->size), domains.n),
            column_spec("P_out_Y_prime_star_edge", y_prime_star_edge, domains.edge),
            column_spec("P_out_Table_src", out_src_route.table, domains.n),
            column_spec("P_out_Query_src", out_src_route.query, domains.edge),
            column_spec("P_out_m_src", out_src_route.multiplicity, domains.n),
            column_spec("P_out_R_src_node", out_src_route.node_acc, domains.n),
            column_spec("P_out_R_src", out_src_route.edge_acc, domains.edge),
            column_spec("P_out_Table_L", out_table_l, domains.edge),
            column_spec("P_out_Query_L", out_query_l, domains.edge),
            column_spec("P_out_m_L", out_m_l, domains.edge),
            column_spec("P_out_R_L", out_r_l, domains.edge),
            column_spec("P_out_s_max", out_s_max, domains.edge),
            column_spec("P_out_C_max", out_c_max, domains.edge),
            column_spec("P_out_Table_R", out_table_r, domains.edge),
            column_spec("P_out_Query_R", out_query_r, domains.edge),
            column_spec("P_out_m_R", out_m_r, domains.edge),
            column_spec("P_out_R_R", out_r_r, domains.edge),
            column_spec("P_out_Table_exp", out_table_exp, domains.edge),
            column_spec("P_out_Query_exp", out_query_exp, domains.edge),
            column_spec("P_out_m_exp", out_m_exp, domains.edge),
            column_spec("P_out_R_exp", out_r_exp, domains.edge),
            column_spec("P_out_widehat_y_star", widehat_y_star, domains.edge),
            column_spec("P_out_w", w_out, domains.edge),
            column_spec("P_out_T", padded_column(t_out, domains.n->size), domains.n),
            column_spec("P_out_T_edge", t_out_edge, domains.edge),
            column_spec("P_out_PSQ", psq_out, domains.edge),
            column_spec("P_out_Y_star", padded_column(y_star, domains.n->size), domains.n),
            column_spec("P_out_Y_star_edge", y_star_edge, domains.edge),
        },
        context.kzg,
        keep_trace_payloads,
        metrics);
    transcript.absorb_commitment("P_out_T", trace.commitments.at("P_out_T").point);
    transcript.absorb_commitment("P_out_T_edge", trace.commitments.at("P_out_T_edge").point);
    const auto eta_t_out = transcript.challenge("eta_t_out");
    const auto beta_t_out = transcript.challenge("beta_t_out");
    trace.challenges["eta_t_out"] = eta_t_out;
    trace.challenges["beta_t_out"] = beta_t_out;
    stage_start = Clock::now();
    const auto eta_t_out_powers = powers(eta_t_out, 2);
    std::vector<FieldElement> out_t_table(n_nodes, FieldElement::zero());
    std::vector<FieldElement> out_t_query(n_edges, FieldElement::zero());
    for (std::size_t i = 0; i < n_nodes; ++i) {
        out_t_table[i] = absolute_id_fields[i] + eta_t_out_powers[1] * t_out[i];
    }
    for (std::size_t k = 0; k < n_edges; ++k) {
        out_t_query[k] = edge_dst_fields[k] + eta_t_out_powers[1] * t_out_edge[k];
    }
    add_metric(metrics != nullptr ? &metrics->route_pack_residual_ms : nullptr, stage_start);
    stage_start = Clock::now();
    const auto& out_t_route = cached_route_trace(
        output_cache_prefix + ":t_route:" + eta_t_out.to_string() + ":" + beta_t_out.to_string(),
        [&]() {
            return build_route_trace(
                out_t_table,
                out_t_query,
                dst_multiplicity_template,
                n_nodes,
                n_edges,
                domains.n,
                domains.edge,
                beta_t_out);
        });
    add_metric(metrics != nullptr ? &metrics->route_trace_ms : nullptr, stage_start);
    transcript.absorb_commitment("P_Y", trace.commitments.at("P_Y").point);
    transcript.absorb_commitment("P_out_Y_star", trace.commitments.at("P_out_Y_star").point);
    trace.challenges["y_out_star"] = transcript.challenge("y_out_star");
    stage_start = Clock::now();
    const auto y_proj_out = trace.challenges.at("y_proj_out");
    const auto y_src_out = trace.challenges.at("y_src_out");
    const auto y_dst_out = trace.challenges.at("y_dst_out");
    const auto y_out_star = trace.challenges.at("y_out_star");
    const auto output_proj_kernel = quantize_matrix(context.model.output_head.seq_kernel_fp);
    const auto output_proj_b = linear_form_by_powers(output_proj_kernel, powers(y_proj_out, n_classes));
    const auto output_attn_src = quantize_vector(context.model.output_head.attn_src_kernel_fp);
    const auto output_attn_dst = quantize_vector(context.model.output_head.attn_dst_kernel_fp);
    const auto xi_out_powers = powers(trace.challenges.at("xi_out"), n_classes);
    const auto& out_proj_binding = cached_binding_trace(
        output_cache_prefix + ":proj:" + y_proj_out.to_string(),
        [&]() {
            return build_binding_trace(
                h_cat,
                domains.n,
                n_nodes,
                y_proj_out,
                output_proj_b,
                domains.cat,
                d_cat);
        });
    const auto& out_src_binding = cached_binding_trace(
        output_cache_prefix + ":src:" + y_src_out.to_string(),
        [&]() {
            return build_binding_trace(
                y_prime,
                domains.n,
                n_nodes,
                y_src_out,
                output_attn_src,
                domains.c,
                n_classes);
        });
    const auto& out_dst_binding = cached_binding_trace(
        output_cache_prefix + ":dst:" + y_dst_out.to_string(),
        [&]() {
            return build_binding_trace(
                y_prime,
                domains.n,
                n_nodes,
                y_dst_out,
                output_attn_dst,
                domains.c,
                n_classes);
        });
    const auto& out_y_binding = cached_binding_trace(
        output_cache_prefix + ":y:" + y_out_star.to_string(),
        [&]() {
            return build_binding_trace(
                y_lin_matrix,
                domains.n,
                n_nodes,
                y_out_star,
                xi_out_powers,
                domains.c,
                n_classes);
        });
    add_metric(metrics != nullptr ? &metrics->zkmap_trace_ms : nullptr, stage_start);
    const auto eta_dst_out = transcript.challenge("eta_dst_out");
    const auto beta_dst_out = transcript.challenge("beta_dst_out");
    trace.challenges["eta_dst_out"] = eta_dst_out;
    trace.challenges["beta_dst_out"] = beta_dst_out;
    stage_start = Clock::now();
    const auto eta_dst_out_powers = powers(eta_dst_out, 6);
    const auto& eta_dst_out_5 = eta_dst_out_powers[5];
    std::vector<FieldElement> out_dst_table(n_nodes, FieldElement::zero());
    std::vector<FieldElement> out_dst_query(n_edges, FieldElement::zero());
    const auto out_e_dst_edge = broadcast_node_values(out_e_dst, edges, false, domains.edge->size);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        out_dst_table[i] = absolute_id_fields[i]
            + eta_dst_out_powers[1] * out_e_dst[i]
            + eta_dst_out_powers[2] * out_m[i]
            + eta_dst_out_powers[3] * out_sum[i]
            + eta_dst_out_powers[4] * out_inv[i]
            + eta_dst_out_5 * y_star[i];
    }
    for (std::size_t k = 0; k < n_edges; ++k) {
        out_dst_query[k] = edge_dst_fields[k]
            + eta_dst_out_powers[1] * out_e_dst_edge[k]
            + eta_dst_out_powers[2] * out_m_edge[k]
            + eta_dst_out_powers[3] * out_sum_edge[k]
                + eta_dst_out_powers[4] * out_inv_edge[k]
                + eta_dst_out_5 * y_star_edge[k];
    }
    add_metric(metrics != nullptr ? &metrics->route_pack_residual_ms : nullptr, stage_start);
    stage_start = Clock::now();
    const auto& out_dst_route = cached_route_trace(
        output_cache_prefix + ":dst_route:" + eta_dst_out.to_string() + ":" + beta_dst_out.to_string(),
        [&]() {
            return build_route_trace(
                out_dst_table,
                out_dst_query,
                dst_multiplicity_template,
                n_nodes,
                n_edges,
                domains.n,
                domains.edge,
                beta_dst_out);
        });
    add_metric(metrics != nullptr ? &metrics->route_trace_ms : nullptr, stage_start);
    add_dynamic_commitment_batch(
        trace,
        {
            column_spec("P_out_Table_t", out_t_route.table, domains.n),
            column_spec("P_out_Query_t", out_t_route.query, domains.edge),
            column_spec("P_out_m_t", out_t_route.multiplicity, domains.n),
            column_spec("P_out_R_t_node", out_t_route.node_acc, domains.n),
            column_spec("P_out_R_t", out_t_route.edge_acc, domains.edge),
            column_spec("P_out_a_proj", out_proj_binding.a, domains.cat),
            column_spec("P_out_b_proj", out_proj_binding.b, domains.cat),
            column_spec("P_out_Acc_proj", out_proj_binding.acc, domains.cat),
            column_spec("P_out_a_src", out_src_binding.a, domains.c),
            column_spec("P_out_b_src", out_src_binding.b, domains.c),
            column_spec("P_out_Acc_src", out_src_binding.acc, domains.c),
            column_spec("P_out_a_dst", out_dst_binding.a, domains.c),
            column_spec("P_out_b_dst", out_dst_binding.b, domains.c),
            column_spec("P_out_Acc_dst", out_dst_binding.acc, domains.c),
            column_spec("P_out_a_y", out_y_binding.a, domains.c),
            column_spec("P_out_b_y", out_y_binding.b, domains.c),
            column_spec("P_out_Acc_y", out_y_binding.acc, domains.c),
            column_spec("P_out_Table_dst", out_dst_route.table, domains.n),
            column_spec("P_out_Query_dst", out_dst_route.query, domains.edge),
            column_spec("P_out_m_dst", out_dst_route.multiplicity, domains.n),
            column_spec("P_out_R_dst_node", out_dst_route.node_acc, domains.n),
            column_spec("P_out_R_dst", out_dst_route.edge_acc, domains.edge),
        },
        context.kzg,
        keep_trace_payloads,
        metrics);
    trace.witness_scalars["S_src_out"] = out_src_route.total;
    trace.witness_scalars["S_dst_out"] = out_dst_route.total;
    trace.witness_scalars["S_t_out"] = out_t_route.total;
    trace.external_evaluations["mu_out_proj"] = matrix_row_major_evaluation(y_prime, trace.challenges.at("y_proj_out"));
    trace.external_evaluations["mu_out_src"] = trace.polynomials.at("P_out_E_src").evaluate(trace.challenges.at("y_src_out"));
    trace.external_evaluations["mu_out_dst"] = trace.polynomials.at("P_out_E_dst").evaluate(trace.challenges.at("y_dst_out"));
    trace.external_evaluations["mu_out_star"] = trace.polynomials.at("P_out_Y_star").evaluate(trace.challenges.at("y_out_star"));
    transcript.absorb_commitment("P_out_Y_prime", trace.commitments.at("P_out_Y_prime").point);
    transcript.absorb_commitment("P_Y", trace.commitments.at("P_Y").point);
    transcript.absorb_commitment("P_out_Y_star", trace.commitments.at("P_out_Y_star").point);
    transcript.absorb_commitment("P_out_Table_dst", trace.commitments.at("P_out_Table_dst").point);
    transcript.absorb_commitment("P_out_Query_dst", trace.commitments.at("P_out_Query_dst").point);
    trace.challenges["y_out"] = transcript.challenge("y_out");
    trace.external_evaluations["mu_Y_lin"] = matrix_row_major_evaluation(y_lin_matrix, trace.challenges.at("y_out"));
    trace.external_evaluations["mu_out"] = matrix_row_major_evaluation(y_matrix, trace.challenges.at("y_out"));
    add_residual_metric(
        &RunMetrics::output_head_trace_ms,
        elapsed_ms(output_trace_start, Clock::now()) - (metric_value(&RunMetrics::commit_dynamic_ms) - output_initial_commit_dynamic_ms),
        {
            metric_value(&RunMetrics::lookup_trace_ms) - output_initial_lookup_ms,
            metric_value(&RunMetrics::route_trace_ms) - output_initial_route_ms,
            metric_value(&RunMetrics::psq_trace_ms) - output_initial_psq_ms,
            metric_value(&RunMetrics::zkmap_trace_ms) - output_initial_zkmap_ms,
            metric_value(&RunMetrics::state_machine_trace_ms) - output_initial_state_machine_ms,
            metric_value(&RunMetrics::padding_selector_trace_ms) - output_initial_padding_ms,
            metric_value(&RunMetrics::route_pack_residual_ms) - output_initial_route_pack_ms,
            metric_value(&RunMetrics::field_conversion_residual_ms) - output_initial_field_conversion_ms,
            metric_value(&RunMetrics::hidden_output_object_residual_ms) - output_initial_object_ms,
        });

    stage_start = Clock::now();
    trace.challenges["alpha_quot"] = transcript.challenge("alpha_quot");
    trace.challenges["z_FH"] = transcript.challenge("z_FH");
    trace.challenges["z_edge"] = transcript.challenge("z_edge");
    trace.challenges["z_in"] = transcript.challenge("z_in");
    trace.challenges["z_d_h"] = transcript.challenge("z_d_h");
    trace.challenges["z_cat"] = transcript.challenge("z_cat");
    trace.challenges["z_C"] = transcript.challenge("z_C");
    trace.challenges["z_N"] = transcript.challenge("z_N");
    trace.challenges["v_FH"] = transcript.challenge("v_FH");
    trace.challenges["v_edge"] = transcript.challenge("v_edge");
    trace.challenges["v_in"] = transcript.challenge("v_in");
    trace.challenges["v_d_h"] = transcript.challenge("v_d_h");
    trace.challenges["v_cat"] = transcript.challenge("v_cat");
    trace.challenges["v_C"] = transcript.challenge("v_C");
    trace.challenges["v_N"] = transcript.challenge("v_N");
    trace.challenges["rho_ext"] = transcript.challenge("rho_ext");
    add_metric(metrics != nullptr ? &metrics->trace_finalize_ms : nullptr, stage_start);

    trace.commitment_order = dynamic_commitment_labels(context);
    trace.challenges["alpha_quot"] = replay_challenges(context, trace.commitments, {}).at("alpha_quot");
    if (metrics != nullptr) {
        const auto total_trace_ms = elapsed_ms(trace_start, Clock::now());
        const auto forward_delta = metrics->forward_ms - initial_forward_ms;
        const auto commit_dynamic_delta = metrics->commit_dynamic_ms - initial_commit_dynamic_ms;
        metrics->trace_generation_ms += std::max(0.0, total_trace_ms - forward_delta - commit_dynamic_delta);
    }
    return trace;
}

}  // namespace

TraceArtifacts build_trace(const ProtocolContext& context, RunMetrics* metrics) {
    const auto trace_start = Clock::now();
    const auto initial_forward_ms = metrics != nullptr ? metrics->forward_ms : 0.0;
    const auto initial_commit_dynamic_ms = metrics != nullptr ? metrics->commit_dynamic_ms : 0.0;
    if (context.model.has_real_multihead) {
        return build_multihead_trace(context, metrics);
    }
    TraceArtifacts trace;
    crypto::Transcript transcript("gatzkml");
    // The proof object and verifier logic do not depend on whether raw columns
    // are exported to disk. This flag only controls whether benchmark runs keep
    // extra copies of trace payloads for debugging artifacts.
    const bool keep_trace_payloads = context.config.dump_trace;

    const auto& local = context.local;
    const auto& domains = context.domains;
    const std::size_t n_nodes = local.num_nodes;
    const std::size_t n_edges = local.edges.size();
    const std::size_t d_in = local.num_features;
    const std::size_t d_hidden = context.model.a_src.size();
    const std::size_t n_classes = local.num_classes;
    std::vector<FieldElement> feature_index_fields(d_in, FieldElement::zero());
    for (std::size_t j = 0; j < d_in; ++j) {
        feature_index_fields[j] = FieldElement(j);
    }
    std::vector<FieldElement> local_index_fields(n_nodes, FieldElement::zero());
    for (std::size_t i = 0; i < n_nodes; ++i) {
        local_index_fields[i] = FieldElement(i);
    }
    std::vector<FieldElement> absolute_id_fields(n_nodes, FieldElement::zero());
    for (std::size_t i = 0; i < n_nodes; ++i) {
        absolute_id_fields[i] = FieldElement(local.absolute_ids[i]);
    }
    std::vector<FieldElement> dataset_index_fields(context.dataset.num_nodes, FieldElement::zero());
    for (std::size_t i = 0; i < context.dataset.num_nodes; ++i) {
        dataset_index_fields[i] = FieldElement(i);
    }

    const auto& q_new = eval_data(context, "P_Q_new_edge");
    const auto& q_end = eval_data(context, "P_Q_end_edge");
    const auto& q_tbl_feat = eval_data(context, "P_Q_tbl_feat");
    const auto& q_qry_feat = eval_data(context, "P_Q_qry_feat");
    const auto& q_tbl_l = eval_data(context, "P_Q_tbl_L");
    const auto& q_qry_l = eval_data(context, "P_Q_qry_L");
    const auto& q_tbl_r = eval_data(context, "P_Q_tbl_R");
    const auto& q_qry_r = eval_data(context, "P_Q_qry_R");
    const auto& q_tbl_exp = eval_data(context, "P_Q_tbl_exp");
    const auto& q_qry_exp = eval_data(context, "P_Q_qry_exp");

    add_dynamic_commitment_batch(
        trace,
        {matrix_commitment_only_spec("P_H", local.features)},
        context.kzg,
        keep_trace_payloads,
        metrics);

    transcript.absorb_scalar("N", FieldElement(n_nodes));
    transcript.absorb_scalar("E", FieldElement(n_edges));
    transcript.absorb_scalar("d_in", FieldElement(d_in));
    transcript.absorb_scalar("d", FieldElement(d_hidden));
    transcript.absorb_scalar("C", FieldElement(n_classes));
    transcript.absorb_scalar("B", FieldElement(context.config.range_bits));
    transcript.absorb_commitment("P_I", context.public_commitments.at("P_I").point);
    transcript.absorb_commitment("P_src", context.public_commitments.at("P_src").point);
    transcript.absorb_commitment("P_dst", context.public_commitments.at("P_dst").point);
    transcript.absorb_commitment("P_Q_new_edge", context.public_commitments.at("P_Q_new_edge").point);
    transcript.absorb_commitment("P_Q_end_edge", context.public_commitments.at("P_Q_end_edge").point);
    transcript.absorb_commitment("P_Q_edge_valid", context.public_commitments.at("P_Q_edge_valid").point);
    transcript.absorb_commitment("P_Q_N", context.public_commitments.at("P_Q_N").point);
    transcript.absorb_commitment("P_Q_proj_valid", context.public_commitments.at("P_Q_proj_valid").point);
    transcript.absorb_commitment("P_Q_d_valid", context.public_commitments.at("P_Q_d_valid").point);
    transcript.absorb_commitment("P_H", trace.commitments.at("P_H").point);
    transcript.absorb_commitment("V_T_H", context.static_commitments.at("V_T_H").point);
    const auto eta_feat = transcript.challenge("eta_feat");
    const auto beta_feat = transcript.challenge("beta_feat");
    trace.challenges["eta_feat"] = eta_feat;
    trace.challenges["beta_feat"] = beta_feat;

    auto stage_start = Clock::now();
    std::unordered_map<std::size_t, std::size_t> feat_hits;
    for (const auto absolute_id : local.absolute_ids) {
        feat_hits[absolute_id] += 1;
    }
    std::vector<FieldElement> feat_hit_fields(context.dataset.num_nodes, FieldElement::zero());
    for (std::size_t v = 0; v < context.dataset.num_nodes; ++v) {
        feat_hit_fields[v] = FieldElement(feat_hits[v]);
    }

    const auto eta_feat_powers = powers(eta_feat, 3);
    std::vector<FieldElement> table_feat(domains.fh->size, FieldElement::zero());
    std::vector<FieldElement> query_feat(domains.fh->size, FieldElement::zero());
    std::vector<FieldElement> multiplicity_feat(domains.fh->size, FieldElement::zero());
    for (std::size_t v = 0; v < context.dataset.num_nodes; ++v) {
        for (std::size_t j = 0; j < d_in; ++j) {
            const std::size_t index = v * d_in + j;
            table_feat[index] = dataset_index_fields[v]
                + eta_feat_powers[1] * feature_index_fields[j]
                + eta_feat_powers[2] * context.dataset.features[v][j];
            multiplicity_feat[index] = feat_hit_fields[v];
        }
    }
    for (std::size_t i = 0; i < n_nodes; ++i) {
        for (std::size_t j = 0; j < d_in; ++j) {
            const std::size_t index = i * d_in + j;
            query_feat[index] = absolute_id_fields[i]
                + eta_feat_powers[1] * feature_index_fields[j]
                + eta_feat_powers[2] * local.features[i][j];
        }
    }
    auto r_feat = build_logup_accumulator(table_feat, query_feat, multiplicity_feat, q_tbl_feat, q_qry_feat, beta_feat);
    add_metric(metrics != nullptr ? &metrics->lookup_trace_ms : nullptr, stage_start);

    add_dynamic_commitment_batch(
        trace,
        {
            column_spec("P_Table_feat", std::move(table_feat), domains.fh),
            column_spec("P_Query_feat", std::move(query_feat), domains.fh),
            column_spec("P_m_feat", std::move(multiplicity_feat), domains.fh),
            column_spec("P_R_feat", std::move(r_feat), domains.fh),
        },
        context.kzg,
        keep_trace_payloads,
        metrics);

    stage_start = Clock::now();
    const auto h_prime = model::project_features(local.features, context.model.W);
    if (metrics != nullptr) {
        metrics->feature_projection_ms += elapsed_ms(stage_start, Clock::now());
    }
    add_dynamic_commitment_batch(
        trace,
        {matrix_commitment_only_spec("P_H_prime", h_prime)},
        context.kzg,
        keep_trace_payloads,
        metrics);

    transcript.absorb_commitment("P_H", trace.commitments.at("P_H").point);
    transcript.absorb_commitment("P_H_prime", trace.commitments.at("P_H_prime").point);
    transcript.absorb_commitment("V_W", context.static_commitments.at("V_W").point);
    const auto y_proj = transcript.challenge("y_proj");
    trace.challenges["y_proj"] = y_proj;

    stage_start = Clock::now();
    const auto y_proj_row_powers = strided_powers(y_proj, n_nodes, d_hidden);
    const auto y_proj_col_powers = powers(y_proj, d_hidden);
    const auto a_proj = weighted_column_sum(local.features, y_proj_row_powers);
    const auto b_proj = linear_form_by_powers(context.model.W, y_proj_col_powers);
    add_metric(metrics != nullptr ? &metrics->witness_materialization_ms : nullptr, stage_start);

    stage_start = Clock::now();
    auto proj_trace = build_zkmap_trace(a_proj, b_proj, domains.in->size);
    add_metric(metrics != nullptr ? &metrics->zkmap_trace_ms : nullptr, stage_start);
    trace.external_evaluations["mu_proj"] = proj_trace.mu;
    add_dynamic_commitment_batch(
        trace,
        {
            column_spec("P_a_proj", std::move(proj_trace.a_values), domains.in),
            column_spec("P_b_proj", std::move(proj_trace.b_values), domains.in),
            column_spec("P_Acc_proj", std::move(proj_trace.accumulator), domains.in),
        },
        context.kzg,
        keep_trace_payloads,
        metrics);

    transcript.absorb_commitment("P_H_prime", trace.commitments.at("P_H_prime").point);
    const auto xi = transcript.challenge("xi");
    trace.challenges["xi"] = xi;

    stage_start = Clock::now();
    const auto e_src = model::matvec_projection(h_prime, context.model.a_src);
    const auto e_dst = model::matvec_projection(h_prime, context.model.a_dst);
    const auto h_star = model::compress_rows(h_prime, xi);
    if (metrics != nullptr) {
        const auto stage_ms = elapsed_ms(stage_start, Clock::now());
        metrics->feature_projection_ms += stage_ms;
        metrics->forward_ms += stage_ms;
    }
    add_dynamic_commitment_batch(
        trace,
        {
            column_spec("P_E_src", padded_column(e_src, domains.n->size), domains.n),
            column_spec("P_E_dst", padded_column(e_dst, domains.n->size), domains.n),
            column_spec("P_H_star", padded_column(h_star, domains.n->size), domains.n),
        },
        context.kzg,
        keep_trace_payloads,
        metrics);

    transcript.absorb_commitment("P_H_prime", trace.commitments.at("P_H_prime").point);
    transcript.absorb_commitment("P_E_src", trace.commitments.at("P_E_src").point);
    transcript.absorb_commitment("V_a_src", context.static_commitments.at("V_a_src").point);
    const auto y_src = transcript.challenge("y_src");
    trace.challenges["y_src"] = y_src;

    stage_start = Clock::now();
    const auto y_src_powers = powers(y_src, n_nodes);
    const auto a_src = weighted_column_sum(h_prime, y_src_powers);
    add_metric(metrics != nullptr ? &metrics->witness_materialization_ms : nullptr, stage_start);

    stage_start = Clock::now();
    auto src_trace = build_zkmap_trace(a_src, context.model.a_src, domains.d->size);
    add_metric(metrics != nullptr ? &metrics->zkmap_trace_ms : nullptr, stage_start);
    trace.external_evaluations["mu_src"] = src_trace.mu;
    add_dynamic_commitment_batch(
        trace,
        {
            column_spec("P_a_src", std::move(src_trace.a_values), domains.d),
            column_spec("P_b_src", std::move(src_trace.b_values), domains.d),
            column_spec("P_Acc_src", std::move(src_trace.accumulator), domains.d),
        },
        context.kzg,
        keep_trace_payloads,
        metrics);

    transcript.absorb_commitment("P_H_prime", trace.commitments.at("P_H_prime").point);
    transcript.absorb_commitment("P_E_dst", trace.commitments.at("P_E_dst").point);
    transcript.absorb_commitment("V_a_dst", context.static_commitments.at("V_a_dst").point);
    const auto y_dst = transcript.challenge("y_dst");
    trace.challenges["y_dst"] = y_dst;

    stage_start = Clock::now();
    const auto y_dst_powers = powers(y_dst, n_nodes);
    const auto a_dst = weighted_column_sum(h_prime, y_dst_powers);
    add_metric(metrics != nullptr ? &metrics->witness_materialization_ms : nullptr, stage_start);

    stage_start = Clock::now();
    auto dst_trace = build_zkmap_trace(a_dst, context.model.a_dst, domains.d->size);
    add_metric(metrics != nullptr ? &metrics->zkmap_trace_ms : nullptr, stage_start);
    trace.external_evaluations["mu_dst"] = dst_trace.mu;
    add_dynamic_commitment_batch(
        trace,
        {
            column_spec("P_a_dst", std::move(dst_trace.a_values), domains.d),
            column_spec("P_b_dst", std::move(dst_trace.b_values), domains.d),
            column_spec("P_Acc_dst", std::move(dst_trace.accumulator), domains.d),
        },
        context.kzg,
        keep_trace_payloads,
        metrics);

    transcript.absorb_commitment("P_H_prime", trace.commitments.at("P_H_prime").point);
    transcript.absorb_commitment("P_H_star", trace.commitments.at("P_H_star").point);
    const auto y_star = transcript.challenge("y_star");
    trace.challenges["y_star"] = y_star;

    stage_start = Clock::now();
    const auto y_star_powers = powers(y_star, n_nodes);
    const auto a_star = weighted_column_sum(h_prime, y_star_powers);
    const auto b_star = powers(xi, d_hidden);
    add_metric(metrics != nullptr ? &metrics->witness_materialization_ms : nullptr, stage_start);

    stage_start = Clock::now();
    auto star_trace = build_zkmap_trace(a_star, b_star, domains.d->size);
    add_metric(metrics != nullptr ? &metrics->zkmap_trace_ms : nullptr, stage_start);
    trace.external_evaluations["mu_star"] = star_trace.mu;
    add_dynamic_commitment_batch(
        trace,
        {
            column_spec("P_a_star", std::move(star_trace.a_values), domains.d),
            column_spec("P_b_star", std::move(star_trace.b_values), domains.d),
            column_spec("P_Acc_star", std::move(star_trace.accumulator), domains.d),
        },
        context.kzg,
        keep_trace_payloads,
        metrics);

    transcript.absorb_commitment("P_E_src", trace.commitments.at("P_E_src").point);
    transcript.absorb_commitment("P_H_star", trace.commitments.at("P_H_star").point);
    const auto eta_src = transcript.challenge("eta_src");
    const auto beta_src = transcript.challenge("beta_src");
    trace.challenges["eta_src"] = eta_src;
    trace.challenges["beta_src"] = beta_src;

    stage_start = Clock::now();
    std::vector<FieldElement> e_src_edge(domains.edge->size, FieldElement::zero());
    std::vector<FieldElement> h_src_star_edge(domains.edge->size, FieldElement::zero());
    std::vector<FieldElement> table_src(domains.n->size, FieldElement::zero());
    std::vector<FieldElement> query_src(domains.edge->size, FieldElement::zero());
    std::vector<FieldElement> m_src(domains.n->size, FieldElement::zero());
    std::vector<std::size_t> out_degree(n_nodes, 0);
    const auto eta_src_powers = powers(eta_src, 3);
    for (const auto& edge : local.edges) {
        out_degree[edge.src] += 1;
    }
    for (std::size_t i = 0; i < n_nodes; ++i) {
        table_src[i] = local_index_fields[i] + eta_src_powers[1] * e_src[i] + eta_src_powers[2] * h_star[i];
        m_src[i] = FieldElement(out_degree[i]);
    }
    for (std::size_t k = 0; k < n_edges; ++k) {
        e_src_edge[k] = e_src[local.edges[k].src];
        h_src_star_edge[k] = h_star[local.edges[k].src];
        query_src[k] = local_index_fields[local.edges[k].src]
            + eta_src_powers[1] * e_src_edge[k]
            + eta_src_powers[2] * h_src_star_edge[k];
    }
    auto r_src_node = build_route_node_accumulator(table_src, m_src, n_nodes, domains.n->size, beta_src);
    auto r_src = build_route_edge_accumulator(query_src, n_edges, domains.edge->size, beta_src);
    if (r_src_node[n_nodes] != r_src[n_edges]) {
        throw std::runtime_error("src dual accumulators do not agree on terminal route sum");
    }
    trace.witness_scalars["S_src"] = r_src_node[n_nodes];
    add_metric(metrics != nullptr ? &metrics->lookup_trace_ms : nullptr, stage_start);
    add_dynamic_commitment_batch(
        trace,
        {
            column_spec("P_E_src_edge", std::move(e_src_edge), domains.edge),
            column_spec("P_H_src_star_edge", h_src_star_edge, domains.edge),
            column_spec("P_Table_src", std::move(table_src), domains.n),
            column_spec("P_Query_src", std::move(query_src), domains.edge),
            column_spec("P_m_src", std::move(m_src), domains.n),
            column_spec("P_R_src_node", std::move(r_src_node), domains.n),
            column_spec("P_R_src", std::move(r_src), domains.edge),
        },
        context.kzg,
        keep_trace_payloads,
        metrics);

    stage_start = Clock::now();
    std::vector<FieldElement> s(domains.edge->size, FieldElement::zero());
    std::vector<FieldElement> z(domains.edge->size, FieldElement::zero());
    for (std::size_t k = 0; k < n_edges; ++k) {
        s[k] = e_src[local.edges[k].src] + e_dst[local.edges[k].dst];
        z[k] = s[k];
    }
    add_metric(metrics != nullptr ? &metrics->route_trace_ms : nullptr, stage_start);
    add_dynamic_commitment_batch(
        trace,
        {
            column_spec("P_S", s, domains.edge),
            column_spec("P_Z", z, domains.edge),
        },
        context.kzg,
        keep_trace_payloads,
        metrics);

    transcript.absorb_commitment("P_S", trace.commitments.at("P_S").point);
    transcript.absorb_commitment("P_Z", trace.commitments.at("P_Z").point);
    transcript.absorb_commitment("V_T_L_x", context.static_commitments.at("V_T_L_x").point);
    transcript.absorb_commitment("V_T_L_y", context.static_commitments.at("V_T_L_y").point);
    const auto eta_l = transcript.challenge("eta_L");
    const auto beta_l = transcript.challenge("beta_L");
    trace.challenges["eta_L"] = eta_l;
    trace.challenges["beta_L"] = beta_l;

    stage_start = Clock::now();
    std::vector<FieldElement> table_l(domains.edge->size, FieldElement::zero());
    std::vector<FieldElement> query_l(domains.edge->size, FieldElement::zero());
    std::vector<FieldElement> m_l(domains.edge->size, FieldElement::zero());
    for (std::size_t t = 0; t < context.tables.lrelu.size(); ++t) {
        table_l[t] = context.tables.lrelu[t].first + eta_l * context.tables.lrelu[t].second;
    }
    for (std::size_t k = 0; k < n_edges; ++k) {
        query_l[k] = s[k] + eta_l * z[k];
        m_l[s[k].value()] += FieldElement::one();
    }
    auto r_l = build_logup_accumulator(table_l, query_l, m_l, q_tbl_l, q_qry_l, beta_l);
    add_metric(metrics != nullptr ? &metrics->lookup_trace_ms : nullptr, stage_start);
    add_dynamic_commitment_batch(
        trace,
        {
            column_spec("P_Table_L", std::move(table_l), domains.edge),
            column_spec("P_Query_L", std::move(query_l), domains.edge),
            column_spec("P_m_L", std::move(m_l), domains.edge),
            column_spec("P_R_L", std::move(r_l), domains.edge),
        },
        context.kzg,
        keep_trace_payloads,
        metrics);

    stage_start = Clock::now();
    std::vector<FieldElement> m_node(n_nodes, FieldElement::zero());
    std::vector<FieldElement> m_edge(domains.edge->size, FieldElement::zero());
    std::vector<FieldElement> delta(domains.edge->size, FieldElement::zero());
    for (std::size_t k = 0; k < n_edges; ++k) {
        m_node[local.edges[k].dst] = FieldElement(std::max(m_node[local.edges[k].dst].value(), z[k].value()));
    }
    for (std::size_t i = 0; i < n_nodes; ++i) {
        m_node[i] = FieldElement(m_node[i].value());
    }
    for (std::size_t k = 0; k < n_edges; ++k) {
        m_edge[k] = m_node[local.edges[k].dst];
        delta[k] = m_edge[k] - z[k];
    }
    add_metric(metrics != nullptr ? &metrics->route_trace_ms : nullptr, stage_start);
    add_dynamic_commitment_batch(
        trace,
        {
            column_spec("P_M", padded_column(m_node, domains.n->size), domains.n),
            column_spec("P_M_edge", m_edge, domains.edge),
            column_spec("P_Delta", delta, domains.edge),
        },
        context.kzg,
        keep_trace_payloads,
        metrics);

    transcript.absorb_commitment("P_M", trace.commitments.at("P_M").point);
    transcript.absorb_commitment("P_M_edge", trace.commitments.at("P_M_edge").point);
    transcript.absorb_commitment("P_Delta", trace.commitments.at("P_Delta").point);
    transcript.absorb_commitment("V_T_range", context.static_commitments.at("V_T_range").point);
    const auto beta_r = transcript.challenge("beta_R");
    trace.challenges["beta_R"] = beta_r;

    stage_start = Clock::now();
    std::vector<FieldElement> s_max(domains.edge->size, FieldElement::zero());
    std::size_t group_start = 0;
    while (group_start < n_edges) {
        std::size_t group_end = group_start;
        while (group_end + 1 < n_edges && local.edges[group_end + 1].dst == local.edges[group_start].dst) {
            ++group_end;
        }
        for (std::size_t k = group_start; k <= group_end; ++k) {
            if (delta[k].is_zero()) {
                s_max[k] = FieldElement::one();
                break;
            }
        }
        group_start = group_end + 1;
    }
    const auto c_max = build_max_counter_state(s_max, q_new);
    std::vector<FieldElement> table_r(domains.edge->size, FieldElement::zero());
    std::vector<FieldElement> query_r(domains.edge->size, FieldElement::zero());
    std::vector<FieldElement> m_r(domains.edge->size, FieldElement::zero());
    for (std::size_t t = 0; t < context.tables.range.size(); ++t) {
        table_r[t] = context.tables.range[t];
    }
    for (std::size_t k = 0; k < n_edges; ++k) {
        query_r[k] = delta[k];
        m_r[delta[k].value()] += FieldElement::one();
    }
    auto r_r = build_logup_accumulator(table_r, query_r, m_r, q_tbl_r, q_qry_r, beta_r);
    add_metric(metrics != nullptr ? &metrics->lookup_trace_ms : nullptr, stage_start);
    add_dynamic_commitment_batch(
        trace,
        {
            column_spec("P_s_max", std::move(s_max), domains.edge),
            column_spec("P_C_max", c_max, domains.edge),
            column_spec("P_Table_R", std::move(table_r), domains.edge),
            column_spec("P_Query_R", std::move(query_r), domains.edge),
            column_spec("P_m_R", std::move(m_r), domains.edge),
            column_spec("P_R_R", std::move(r_r), domains.edge),
        },
        context.kzg,
        keep_trace_payloads,
        metrics);

    stage_start = Clock::now();
    std::vector<FieldElement> u(domains.edge->size, FieldElement::zero());
    for (std::size_t k = 0; k < n_edges; ++k) {
        u[k] = exp_map(delta[k], context.tables.range.size());
    }
    add_metric(metrics != nullptr ? &metrics->route_trace_ms : nullptr, stage_start);
    add_dynamic_commitment_batch(
        trace,
        {column_spec("P_U", u, domains.edge)},
        context.kzg,
        keep_trace_payloads,
        metrics);

    transcript.absorb_commitment("P_Delta", trace.commitments.at("P_Delta").point);
    transcript.absorb_commitment("P_U", trace.commitments.at("P_U").point);
    transcript.absorb_commitment("V_T_exp_x", context.static_commitments.at("V_T_exp_x").point);
    transcript.absorb_commitment("V_T_exp_y", context.static_commitments.at("V_T_exp_y").point);
    const auto eta_exp = transcript.challenge("eta_exp");
    const auto beta_exp = transcript.challenge("beta_exp");
    trace.challenges["eta_exp"] = eta_exp;
    trace.challenges["beta_exp"] = beta_exp;

    stage_start = Clock::now();
    std::vector<FieldElement> sum(n_nodes, FieldElement::zero());
    for (std::size_t k = 0; k < n_edges; ++k) {
        sum[local.edges[k].dst] += u[k];
    }
    std::vector<FieldElement> inv(n_nodes, FieldElement::zero());
    for (std::size_t i = 0; i < n_nodes; ++i) {
        inv[i] = sum[i].inv();
    }
    std::vector<FieldElement> alpha(domains.edge->size, FieldElement::zero());
    for (std::size_t k = 0; k < n_edges; ++k) {
        alpha[k] = u[k] * inv[local.edges[k].dst];
    }
    add_metric(metrics != nullptr ? &metrics->route_trace_ms : nullptr, stage_start);

    stage_start = Clock::now();
    std::vector<FieldElement> table_exp(domains.edge->size, FieldElement::zero());
    std::vector<FieldElement> query_exp(domains.edge->size, FieldElement::zero());
    std::vector<FieldElement> m_exp(domains.edge->size, FieldElement::zero());
    const auto eta_exp_powers = powers(eta_exp, 2);
    for (std::size_t t = 0; t < context.tables.exp.size(); ++t) {
        table_exp[t] = context.tables.exp[t].first + eta_exp_powers[1] * context.tables.exp[t].second;
    }
    for (std::size_t k = 0; k < n_edges; ++k) {
        query_exp[k] = delta[k] + eta_exp_powers[1] * u[k];
        m_exp[delta[k].value()] += FieldElement::one();
    }
    auto r_exp = build_logup_accumulator(table_exp, query_exp, m_exp, q_tbl_exp, q_qry_exp, beta_exp);
    add_metric(metrics != nullptr ? &metrics->lookup_trace_ms : nullptr, stage_start);
    add_dynamic_commitment_batch(
        trace,
        {
            column_spec("P_Sum", padded_column(sum, domains.n->size), domains.n),
            column_spec("P_inv", padded_column(inv, domains.n->size), domains.n),
            column_spec("P_alpha", alpha, domains.edge),
            column_spec("P_Table_exp", std::move(table_exp), domains.edge),
            column_spec("P_Query_exp", std::move(query_exp), domains.edge),
            column_spec("P_m_exp", std::move(m_exp), domains.edge),
            column_spec("P_R_exp", std::move(r_exp), domains.edge),
        },
        context.kzg,
        keep_trace_payloads,
        metrics);

    stage_start = Clock::now();
    const auto h_agg = model::aggregate_by_edges(h_prime, alpha, local.edges, n_nodes);
    const auto h_agg_star = model::compress_rows(h_agg, xi);
    if (metrics != nullptr) {
        const auto stage_ms = elapsed_ms(stage_start, Clock::now());
        metrics->feature_projection_ms += stage_ms;
        metrics->forward_ms += stage_ms;
    }
    add_dynamic_commitment_batch(
        trace,
        {
            matrix_commitment_only_spec("P_H_agg", h_agg),
            column_spec("P_H_agg_star", padded_column(h_agg_star, domains.n->size), domains.n),
        },
        context.kzg,
        keep_trace_payloads,
        metrics);

    transcript.absorb_commitment("P_H_agg", trace.commitments.at("P_H_agg").point);
    transcript.absorb_commitment("P_H_agg_star", trace.commitments.at("P_H_agg_star").point);
    const auto y_agg = transcript.challenge("y_agg");
    trace.challenges["y_agg"] = y_agg;

    stage_start = Clock::now();
    const auto y_agg_powers = powers(y_agg, n_nodes);
    const auto a_agg = weighted_column_sum(h_agg, y_agg_powers);
    const auto b_agg = powers(xi, d_hidden);
    add_metric(metrics != nullptr ? &metrics->witness_materialization_ms : nullptr, stage_start);

    stage_start = Clock::now();
    auto agg_trace = build_zkmap_trace(a_agg, b_agg, domains.d->size);
    add_metric(metrics != nullptr ? &metrics->zkmap_trace_ms : nullptr, stage_start);
    trace.external_evaluations["mu_agg"] = agg_trace.mu;
    add_dynamic_commitment_batch(
        trace,
        {
            column_spec("P_a_agg", std::move(agg_trace.a_values), domains.d),
            column_spec("P_b_agg", std::move(agg_trace.b_values), domains.d),
            column_spec("P_Acc_agg", std::move(agg_trace.accumulator), domains.d),
        },
        context.kzg,
        keep_trace_payloads,
        metrics);

    stage_start = Clock::now();
    std::vector<FieldElement> e_dst_edge(domains.edge->size, FieldElement::zero());
    std::vector<FieldElement> sum_edge(domains.edge->size, FieldElement::zero());
    std::vector<FieldElement> inv_edge(domains.edge->size, FieldElement::zero());
    std::vector<FieldElement> h_agg_star_edge(domains.edge->size, FieldElement::zero());
    for (std::size_t k = 0; k < n_edges; ++k) {
        e_dst_edge[k] = e_dst[local.edges[k].dst];
        sum_edge[k] = sum[local.edges[k].dst];
        inv_edge[k] = inv[local.edges[k].dst];
        h_agg_star_edge[k] = h_agg_star[local.edges[k].dst];
    }
    add_metric(metrics != nullptr ? &metrics->route_trace_ms : nullptr, stage_start);
    add_dynamic_commitment_batch(
        trace,
        {
            column_spec("P_E_dst_edge", e_dst_edge, domains.edge),
            column_spec("P_Sum_edge", sum_edge, domains.edge),
            column_spec("P_inv_edge", inv_edge, domains.edge),
            column_spec("P_H_agg_star_edge", h_agg_star_edge, domains.edge),
        },
        context.kzg,
        keep_trace_payloads,
        metrics);

    transcript.absorb_commitment("P_E_dst", trace.commitments.at("P_E_dst").point);
    transcript.absorb_commitment("P_M", trace.commitments.at("P_M").point);
    transcript.absorb_commitment("P_Sum", trace.commitments.at("P_Sum").point);
    transcript.absorb_commitment("P_inv", trace.commitments.at("P_inv").point);
    transcript.absorb_commitment("P_H_agg_star", trace.commitments.at("P_H_agg_star").point);
    transcript.absorb_commitment("P_E_dst_edge", trace.commitments.at("P_E_dst_edge").point);
    transcript.absorb_commitment("P_M_edge", trace.commitments.at("P_M_edge").point);
    transcript.absorb_commitment("P_Sum_edge", trace.commitments.at("P_Sum_edge").point);
    transcript.absorb_commitment("P_inv_edge", trace.commitments.at("P_inv_edge").point);
    transcript.absorb_commitment("P_H_agg_star_edge", trace.commitments.at("P_H_agg_star_edge").point);
    const auto eta_dst_route = transcript.challenge("eta_dst");
    const auto beta_dst = transcript.challenge("beta_dst");
    trace.challenges["eta_dst"] = eta_dst_route;
    trace.challenges["beta_dst"] = beta_dst;

    stage_start = Clock::now();
    std::vector<FieldElement> table_dst(domains.n->size, FieldElement::zero());
    std::vector<FieldElement> query_dst(domains.edge->size, FieldElement::zero());
    std::vector<FieldElement> m_dst(domains.n->size, FieldElement::zero());
    std::vector<std::size_t> in_degree(n_nodes, 0);
    const auto eta_dst_powers = powers(eta_dst_route, 6);
    for (const auto& edge : local.edges) {
        in_degree[edge.dst] += 1;
    }
    for (std::size_t i = 0; i < n_nodes; ++i) {
        table_dst[i] = local_index_fields[i]
            + eta_dst_powers[1] * e_dst[i]
            + eta_dst_powers[2] * m_node[i]
            + eta_dst_powers[3] * sum[i]
            + eta_dst_powers[4] * inv[i]
            + eta_dst_powers[5] * h_agg_star[i];
        m_dst[i] = FieldElement(in_degree[i]);
    }
    for (std::size_t k = 0; k < n_edges; ++k) {
        query_dst[k] = local_index_fields[local.edges[k].dst]
            + eta_dst_powers[1] * e_dst_edge[k]
            + eta_dst_powers[2] * m_edge[k]
            + eta_dst_powers[3] * sum_edge[k]
            + eta_dst_powers[4] * inv_edge[k]
            + eta_dst_powers[5] * h_agg_star_edge[k];
    }
    auto r_dst_node = build_route_node_accumulator(table_dst, m_dst, n_nodes, domains.n->size, beta_dst);
    auto r_dst = build_route_edge_accumulator(query_dst, n_edges, domains.edge->size, beta_dst);
    if (r_dst_node[n_nodes] != r_dst[n_edges]) {
        throw std::runtime_error("dst dual accumulators do not agree on terminal route sum");
    }
    trace.witness_scalars["S_dst"] = r_dst_node[n_nodes];
    add_metric(metrics != nullptr ? &metrics->lookup_trace_ms : nullptr, stage_start);
    add_dynamic_commitment_batch(
        trace,
        {
            column_spec("P_Table_dst", std::move(table_dst), domains.n),
            column_spec("P_Query_dst", std::move(query_dst), domains.edge),
            column_spec("P_m_dst", std::move(m_dst), domains.n),
            column_spec("P_R_dst_node", std::move(r_dst_node), domains.n),
            column_spec("P_R_dst", std::move(r_dst), domains.edge),
        },
        context.kzg,
        keep_trace_payloads,
        metrics);

    stage_start = Clock::now();
    std::vector<FieldElement> v_hat(domains.edge->size, FieldElement::zero());
    for (std::size_t k = 0; k < n_edges; ++k) {
        v_hat[k] = alpha[k] * h_src_star_edge[k];
    }
    add_metric(metrics != nullptr ? &metrics->route_trace_ms : nullptr, stage_start);
    add_dynamic_commitment_batch(
        trace,
        {column_spec("P_v_hat", v_hat, domains.edge)},
        context.kzg,
        keep_trace_payloads,
        metrics);

    transcript.absorb_commitment("P_U", trace.commitments.at("P_U").point);
    transcript.absorb_commitment("P_alpha", trace.commitments.at("P_alpha").point);
    transcript.absorb_commitment("P_H_src_star_edge", trace.commitments.at("P_H_src_star_edge").point);
    transcript.absorb_commitment("P_Sum", trace.commitments.at("P_Sum").point);
    transcript.absorb_commitment("P_H_agg_star", trace.commitments.at("P_H_agg_star").point);
    transcript.absorb_commitment("P_H_agg_star_edge", trace.commitments.at("P_H_agg_star_edge").point);
    transcript.absorb_commitment("P_v_hat", trace.commitments.at("P_v_hat").point);
    const auto lambda_psq = transcript.challenge("lambda_psq");
    trace.challenges["lambda_psq"] = lambda_psq;

    stage_start = Clock::now();
    std::vector<FieldElement> w_psq(domains.edge->size, FieldElement::zero());
    std::vector<FieldElement> t_psq_edge(domains.edge->size, FieldElement::zero());
    for (std::size_t k = 0; k < n_edges; ++k) {
        w_psq[k] = u[k] + lambda_psq * v_hat[k];
        t_psq_edge[k] = sum_edge[k] + lambda_psq * h_agg_star_edge[k];
    }
    const auto psq_state = build_group_prefix_state(w_psq, q_new);
    add_metric(metrics != nullptr ? &metrics->psq_trace_ms : nullptr, stage_start);
    add_dynamic_commitment_batch(
        trace,
        {
            column_spec("P_w_psq", std::move(w_psq), domains.edge),
            column_spec("P_T_psq_edge", std::move(t_psq_edge), domains.edge),
            column_spec("P_PSQ", psq_state, domains.edge),
        },
        context.kzg,
        keep_trace_payloads,
        metrics);

    Matrix y_lin;
    stage_start = Clock::now();
    const auto y = model::output_projection(h_agg, context.model.W_out, context.model.b, &y_lin);
    if (metrics != nullptr) {
        const auto stage_ms = elapsed_ms(stage_start, Clock::now());
        metrics->feature_projection_ms += stage_ms;
        metrics->forward_ms += stage_ms;
    }
    add_dynamic_commitment_batch(
        trace,
        {
            matrix_commitment_only_spec("P_Y_lin", y_lin),
            matrix_commitment_only_spec("P_Y", y),
        },
        context.kzg,
        keep_trace_payloads,
        metrics);

    transcript.absorb_commitment("P_H_agg", trace.commitments.at("P_H_agg").point);
    transcript.absorb_commitment("P_Y_lin", trace.commitments.at("P_Y_lin").point);
    transcript.absorb_commitment("P_Y", trace.commitments.at("P_Y").point);
    transcript.absorb_commitment("V_W_out", context.static_commitments.at("V_W_out").point);
    transcript.absorb_commitment("V_b", context.static_commitments.at("V_b").point);
    const auto y_out = transcript.challenge("y_out");
    trace.challenges["y_out"] = y_out;

    stage_start = Clock::now();
    const auto y_out_row_powers = strided_powers(y_out, n_nodes, n_classes);
    const auto y_out_col_powers = powers(y_out, n_classes);
    const auto a_out = weighted_column_sum(h_agg, y_out_row_powers);
    const auto b_out = linear_form_by_powers(context.model.W_out, y_out_col_powers);
    add_metric(metrics != nullptr ? &metrics->witness_materialization_ms : nullptr, stage_start);

    stage_start = Clock::now();
    auto out_trace = build_zkmap_trace(a_out, b_out, domains.d->size);
    add_metric(metrics != nullptr ? &metrics->zkmap_trace_ms : nullptr, stage_start);
    add_dynamic_commitment_batch(
        trace,
        {
            column_spec("P_a_out", std::move(out_trace.a_values), domains.d),
            column_spec("P_b_out", std::move(out_trace.b_values), domains.d),
            column_spec("P_Acc_out", std::move(out_trace.accumulator), domains.d),
        },
        context.kzg,
        keep_trace_payloads,
        metrics);
    trace.external_evaluations["mu_Y_lin"] = out_trace.mu;
    trace.external_evaluations["mu_bias_out"] = evaluate_bias_fold(context.model.b, n_nodes, y_out);
    trace.external_evaluations["mu_out"] = trace.external_evaluations["mu_Y_lin"] + trace.external_evaluations["mu_bias_out"];

    for (const auto& [name, challenge] : transcript.issued_challenges()) {
        trace.challenges[name] = challenge;
    }
    if (metrics != nullptr) {
        metrics->forward_ms = metrics->feature_projection_ms;
        const auto total_trace_ms = elapsed_ms(trace_start, Clock::now());
        const auto forward_delta = metrics->forward_ms - initial_forward_ms;
        const auto commit_dynamic_delta = metrics->commit_dynamic_ms - initial_commit_dynamic_ms;
        metrics->trace_generation_ms += std::max(0.0, total_trace_ms - forward_delta - commit_dynamic_delta);
    }
    return trace;
}

}  // namespace gatzk::protocol
