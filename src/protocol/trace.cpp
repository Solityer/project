#include "gatzk/protocol/trace.hpp"

#include <algorithm>
#include <chrono>
#include <future>
#include <optional>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/crypto/kzg.hpp"
#include "gatzk/crypto/transcript.hpp"
#include "gatzk/model/gat.hpp"
#include "gatzk/protocol/challenges.hpp"
#include "gatzk/protocol/lookup.hpp"
#include "gatzk/protocol/psq.hpp"
#include "gatzk/protocol/zkmap.hpp"
#include "gatzk/util/route2.hpp"

namespace gatzk::protocol {
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

FieldElement row_polynomial_at_tau(const std::vector<FieldElement>& row, const FieldElement& tau) {
    mcl::Fr out;
    out.clear();
    for (std::size_t column = row.size(); column-- > 0;) {
        mcl::Fr::mul(out, out, tau.native());
        mcl::Fr::add(out, out, row[column].native());
    }
    return FieldElement::from_native(out);
}

FieldElement matrix_row_major_tau_evaluation(const Matrix& matrix, const FieldElement& tau) {
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
    // each row block directly at tau and then fold rows by tau^cols, so matrix
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
                [&matrix, &row_evaluations, &tau, begin, end]() {
                    for (std::size_t row = begin; row < end; ++row) {
                        row_evaluations[row] = row_polynomial_at_tau(matrix[row], tau);
                    }
                }));
        }
        for (auto& future : futures) {
            future.get();
        }
    } else {
        for (std::size_t row = 0; row < row_count; ++row) {
            row_evaluations[row] = row_polynomial_at_tau(matrix[row], tau);
        }
    }

    const auto row_stride = tau.pow(static_cast<std::uint64_t>(column_count));
    mcl::Fr out;
    out.clear();
    for (std::size_t row = row_count; row-- > 0;) {
        mcl::Fr::mul(out, out, row_stride.native());
        mcl::Fr::add(out, out, row_evaluations[row].native());
    }
    return FieldElement::from_native(out);
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
        + metrics->dynamic_commit_msm_ms
        + metrics->dynamic_commit_finalize_ms;
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

    const auto commit_start = Clock::now();
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

    if (!named_polynomials.empty()) {
        const auto commitments = crypto::KZG::commit_batch(named_polynomials, key);
        for (std::size_t i = 0; i < commitments.size(); ++i) {
            prepared[polynomial_indices[i]].commitment = commitments[i];
        }
    }
    if (!direct_tau_evaluations.empty()) {
        const auto commitments = crypto::KZG::commit_tau_evaluation_batch(direct_tau_evaluations, key);
        for (std::size_t i = 0; i < commitments.size(); ++i) {
            prepared[direct_tau_indices[i]].commitment = commitments[i];
        }
    }
    add_metric(metrics != nullptr ? &metrics->dynamic_commit_msm_ms : nullptr, commit_start);

    const auto finalize_start = Clock::now();
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
        trace.commitments[item.name] = std::move(item.commitment);
        trace.commitment_order.push_back(item.name);
    }
    add_metric(metrics != nullptr ? &metrics->dynamic_commit_finalize_ms : nullptr, finalize_start);
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

BindingTrace build_binding_trace(
    const Matrix& matrix,
    const std::shared_ptr<algebra::RootOfUnityDomain>& row_domain,
    std::size_t real_row_count,
    const FieldElement& point,
    const std::vector<FieldElement>& parameter_values,
    const std::shared_ptr<algebra::RootOfUnityDomain>& dim_domain,
    std::size_t real_dim_count) {
    BindingTrace out;
    const auto row_weights_full = row_domain->barycentric_weights(point);
    std::vector<FieldElement> row_weights(real_row_count, FieldElement::zero());
    for (std::size_t i = 0; i < real_row_count; ++i) {
        row_weights[i] = row_weights_full[i];
    }
    const auto folded = weighted_column_sum(matrix, row_weights);
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

struct RouteTrace {
    std::vector<FieldElement> table;
    std::vector<FieldElement> query;
    std::vector<FieldElement> multiplicity;
    std::vector<FieldElement> node_acc;
    std::vector<FieldElement> edge_acc;
    FieldElement total = FieldElement::zero();
};

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

std::vector<FieldElement> zero_eval_column(std::size_t size) {
    return std::vector<FieldElement>(size, FieldElement::zero());
}

TraceArtifacts build_multihead_trace(const ProtocolContext& context, RunMetrics* metrics) {
    const auto trace_start = Clock::now();
    TraceArtifacts trace;
    crypto::Transcript transcript("gatzkml");
    const bool keep_trace_payloads = context.config.dump_trace;
    const auto& local = context.local;
    const auto& domains = context.domains;
    const auto& edges = local.edges;
    const std::size_t n_nodes = local.num_nodes;
    const std::size_t n_edges = edges.size();
    const std::size_t d_in = local.num_features;
    const std::size_t d_h = context.model.hidden_heads.front().output_bias_fp.size();
    const std::size_t d_cat = d_h * context.model.hidden_heads.size();
    const std::size_t n_classes = local.num_classes;

    std::vector<FieldElement> feature_index_fields(d_in, FieldElement::zero());
    for (std::size_t j = 0; j < d_in; ++j) {
        feature_index_fields[j] = FieldElement(j);
    }
    std::vector<FieldElement> absolute_id_fields(n_nodes, FieldElement::zero());
    for (std::size_t i = 0; i < n_nodes; ++i) {
        absolute_id_fields[i] = FieldElement(local.absolute_ids[i]);
    }
    std::vector<FieldElement> dataset_index_fields(context.dataset.num_nodes, FieldElement::zero());
    for (std::size_t i = 0; i < context.dataset.num_nodes; ++i) {
        dataset_index_fields[i] = FieldElement(i);
    }
    const auto q_new = eval_data(context, "P_Q_new_edge");
    const auto q_edge_valid = eval_data(context, "P_Q_edge_valid");
    const auto q_tbl_feat = eval_data(context, "P_Q_tbl_feat");
    const auto q_qry_feat = eval_data(context, "P_Q_qry_feat");

    add_dynamic_commitment_batch(
        trace,
        {matrix_commitment_only_spec("P_H", local.features)},
        context.kzg,
        keep_trace_payloads,
        metrics);

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
        std::unordered_map<std::size_t, std::size_t> feat_hits;
        for (const auto absolute_id : local.absolute_ids) {
            feat_hits[absolute_id] += 1;
        }
        std::vector<FieldElement> feat_hit_fields(context.dataset.num_nodes, FieldElement::zero());
        for (std::size_t v = 0; v < context.dataset.num_nodes; ++v) {
            feat_hit_fields[v] = FieldElement(feat_hits[v]);
        }
        const auto eta_feat_powers = powers(trace.challenges.at("eta_feat"), 3);
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
        auto r_feat = build_logup_accumulator(
            table_feat,
            query_feat,
            multiplicity_feat,
            q_tbl_feat,
            q_qry_feat,
            trace.challenges.at("beta_feat"));
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

    const auto forward_start = Clock::now();
    const auto forward = model::forward_reference_style(local.features_fp, edges, context.model);
    if (metrics != nullptr) {
        metrics->forward_ms += elapsed_ms(forward_start, Clock::now());
    }

    auto commit_hidden_head = [&](std::size_t head_index) {
        const auto prefix = "P_h" + std::to_string(head_index) + "_";
        const auto& fp = forward.hidden_head_traces[head_index];
        const auto h_prime = quantize_matrix(fp.H_prime);
        const auto e_src = quantize_vector(fp.E_src);
        const auto e_dst = quantize_vector(fp.E_dst);
        const auto s = quantize_vector(fp.S);
        const auto z = quantize_vector(fp.Z);
        const auto m = quantize_vector(fp.M);
        const auto delta = quantize_vector(fp.Delta);
        const auto u = quantize_vector(fp.U);
        const auto sum = quantize_vector(fp.Sum);
        const auto inv = quantize_vector(fp.inv);
        const auto alpha = quantize_vector(fp.alpha);
        const auto h_agg_pre = quantize_matrix(fp.H_agg_pre_bias);
        const auto h_agg = quantize_matrix(fp.H_agg);
        const auto m_edge = broadcast_node_values(m, edges, false, domains.edge->size);
        const auto sum_edge = broadcast_node_values(sum, edges, false, domains.edge->size);
        const auto inv_edge = broadcast_node_values(inv, edges, false, domains.edge->size);

        add_dynamic_commitment_batch(
            trace,
            {
                matrix_spec(prefix + "H_prime", h_prime),
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
                matrix_spec(prefix + "H_agg_pre", h_agg_pre),
                matrix_spec(prefix + "H_agg", h_agg),
            },
            context.kzg,
            keep_trace_payloads,
            metrics);

        transcript.absorb_commitment(prefix + "H_prime", trace.commitments.at(prefix + "H_prime").point);
        trace.challenges["y_proj_h" + std::to_string(head_index)] = transcript.challenge("y_proj_h" + std::to_string(head_index));
        trace.challenges["xi_h" + std::to_string(head_index)] = transcript.challenge("xi_h" + std::to_string(head_index));
        transcript.absorb_commitment(prefix + "E_src", trace.commitments.at(prefix + "E_src").point);
        trace.challenges["y_src_h" + std::to_string(head_index)] = transcript.challenge("y_src_h" + std::to_string(head_index));
        transcript.absorb_commitment(prefix + "E_dst", trace.commitments.at(prefix + "E_dst").point);
        trace.challenges["y_dst_h" + std::to_string(head_index)] = transcript.challenge("y_dst_h" + std::to_string(head_index));

        const auto xi = trace.challenges.at("xi_h" + std::to_string(head_index));
        const auto h_star = compress_rows_with_challenge(h_prime, xi);
        const auto h_star_edge = broadcast_node_values(h_star, edges, true, domains.edge->size);
        const auto h_agg_pre_star = compress_rows_with_challenge(h_agg_pre, xi);
        const auto h_agg_pre_star_edge = broadcast_node_values(h_agg_pre_star, edges, false, domains.edge->size);
        const auto h_agg_star = compress_rows_with_challenge(h_agg, xi);
        const auto h_agg_star_edge = broadcast_node_values(h_agg_star, edges, false, domains.edge->size);

        const auto head_w = quantize_matrix(context.model.hidden_heads[head_index].seq_kernel_fp);
        const auto proj_binding = build_binding_trace(
            local.features,
            domains.n,
            n_nodes,
            trace.challenges.at("y_proj_h" + std::to_string(head_index)),
            linear_form_by_powers(head_w, powers(trace.challenges.at("y_proj_h" + std::to_string(head_index)), d_h)),
            domains.in,
            d_in);
        const auto src_binding = build_binding_trace(
            h_prime,
            domains.n,
            n_nodes,
            trace.challenges.at("y_src_h" + std::to_string(head_index)),
            quantize_vector(context.model.hidden_heads[head_index].attn_src_kernel_fp),
            domains.d,
            d_h);
        const auto dst_binding = build_binding_trace(
            h_prime,
            domains.n,
            n_nodes,
            trace.challenges.at("y_dst_h" + std::to_string(head_index)),
            quantize_vector(context.model.hidden_heads[head_index].attn_dst_kernel_fp),
            domains.d,
            d_h);
        transcript.absorb_commitment(prefix + "H_star", crypto::KZG::commit(prefix + "H_star", coeff_poly(prefix + "H_star", h_star), context.kzg).point);
        trace.challenges["y_star_h" + std::to_string(head_index)] = transcript.challenge("y_star_h" + std::to_string(head_index));
        const auto star_binding = build_binding_trace(
            h_prime,
            domains.n,
            n_nodes,
            trace.challenges.at("y_star_h" + std::to_string(head_index)),
            powers(xi, d_h),
            domains.d,
            d_h);

        const auto eta_src = transcript.challenge("eta_src_h" + std::to_string(head_index));
        const auto beta_src = transcript.challenge("beta_src_h" + std::to_string(head_index));
        trace.challenges["eta_src_h" + std::to_string(head_index)] = eta_src;
        trace.challenges["beta_src_h" + std::to_string(head_index)] = beta_src;
        const auto eta_src_powers = powers(eta_src, 2);
        std::vector<FieldElement> src_table(n_nodes, FieldElement::zero());
        std::vector<FieldElement> src_query(n_edges, FieldElement::zero());
        for (std::size_t i = 0; i < n_nodes; ++i) {
            src_table[i] = absolute_id_fields[i] + eta_src_powers[1] * e_src[i];
        }
        for (std::size_t k = 0; k < n_edges; ++k) {
            src_query[k] = FieldElement(edges[k].src) + eta_src_powers[1] * h_star_edge[k];
        }
        const auto src_mult = count_by_src(edges, n_nodes, domains.n->size);
        auto src_route = build_route_trace(src_table, src_query, src_mult, n_nodes, n_edges, domains.n, domains.edge, beta_src);

        const auto lambda_psq = transcript.challenge("lambda_psq_h" + std::to_string(head_index));
        trace.challenges["lambda_psq_h" + std::to_string(head_index)] = lambda_psq;
        std::vector<FieldElement> widehat_v_pre_star(domains.edge->size, FieldElement::zero());
        for (std::size_t k = 0; k < n_edges; ++k) {
            widehat_v_pre_star[k] = alpha[k] * h_star_edge[k];
        }
        auto w_psq = build_weighted_sum(quantize_vector(fp.U), lambda_psq, widehat_v_pre_star, domains.edge->size);
        std::vector<FieldElement> t_psq(n_nodes, FieldElement::zero());
        for (std::size_t i = 0; i < n_nodes; ++i) {
            t_psq[i] = sum[i] + lambda_psq * h_agg_pre_star[i];
        }
        const auto t_psq_edge = build_group_target(t_psq, edges, domains.edge->size);
        auto psq = build_group_prefix_state(w_psq, q_new);

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
                column_spec(prefix + "H_star", padded_column(h_star, domains.n->size), domains.n),
                column_spec(prefix + "a_star", star_binding.a, domains.d),
                column_spec(prefix + "b_star", star_binding.b, domains.d),
                column_spec(prefix + "Acc_star", star_binding.acc, domains.d),
                column_spec(prefix + "E_src_edge", broadcast_node_values(e_src, edges, true, domains.edge->size), domains.edge),
                column_spec(prefix + "E_dst_edge", broadcast_node_values(e_dst, edges, false, domains.edge->size), domains.edge),
                column_spec(prefix + "H_src_star_edge", h_star_edge, domains.edge),
                column_spec(prefix + "Table_src", src_route.table, domains.n),
                column_spec(prefix + "Query_src", src_route.query, domains.edge),
                column_spec(prefix + "m_src", src_route.multiplicity, domains.n),
                column_spec(prefix + "R_src_node", src_route.node_acc, domains.n),
                column_spec(prefix + "R_src", src_route.edge_acc, domains.edge),
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

        transcript.absorb_commitment(prefix + "H_agg_pre", trace.commitments.at(prefix + "H_agg_pre").point);
        transcript.absorb_commitment(prefix + "H_agg_pre_star", trace.commitments.at(prefix + "H_agg_pre_star").point);
        trace.challenges["y_agg_pre_h" + std::to_string(head_index)] = transcript.challenge("y_agg_pre_h" + std::to_string(head_index));
        transcript.absorb_commitment(prefix + "H_agg", trace.commitments.at(prefix + "H_agg").point);
        transcript.absorb_commitment(prefix + "H_agg_star", trace.commitments.at(prefix + "H_agg_star").point);
        trace.challenges["y_agg_h" + std::to_string(head_index)] = transcript.challenge("y_agg_h" + std::to_string(head_index));

        const auto agg_binding = build_binding_trace(
            h_agg,
            domains.n,
            n_nodes,
            trace.challenges.at("y_agg_h" + std::to_string(head_index)),
            powers(xi, d_h),
            domains.d,
            d_h);
        const auto eta_dst = transcript.challenge("eta_dst_h" + std::to_string(head_index));
        const auto beta_dst = transcript.challenge("beta_dst_h" + std::to_string(head_index));
        trace.challenges["eta_dst_h" + std::to_string(head_index)] = eta_dst;
        trace.challenges["beta_dst_h" + std::to_string(head_index)] = beta_dst;
        const auto eta_dst_powers = powers(eta_dst, 5);
        std::vector<FieldElement> dst_table(n_nodes, FieldElement::zero());
        std::vector<FieldElement> dst_query(n_edges, FieldElement::zero());
        for (std::size_t i = 0; i < n_nodes; ++i) {
            dst_table[i] = absolute_id_fields[i]
                + eta_dst_powers[1] * e_dst[i]
                + eta_dst_powers[2] * m[i]
                + eta_dst_powers[3] * sum[i]
                + eta_dst_powers[4] * inv[i]
                + eta_dst.pow(5) * h_agg_star[i];
        }
        const auto e_dst_edge = broadcast_node_values(e_dst, edges, false, domains.edge->size);
        for (std::size_t k = 0; k < n_edges; ++k) {
            dst_query[k] = FieldElement(edges[k].dst)
                + eta_dst_powers[1] * e_dst_edge[k]
                + eta_dst_powers[2] * m_edge[k]
                + eta_dst_powers[3] * sum_edge[k]
                + eta_dst_powers[4] * inv_edge[k]
                + eta_dst.pow(5) * h_agg_star_edge[k];
        }
        const auto dst_mult = count_by_dst(edges, n_nodes, domains.n->size);
        auto dst_route = build_route_trace(dst_table, dst_query, dst_mult, n_nodes, n_edges, domains.n, domains.edge, beta_dst);
        add_dynamic_commitment_batch(
            trace,
            {
                column_spec(prefix + "a_agg", agg_binding.a, domains.d),
                column_spec(prefix + "b_agg", agg_binding.b, domains.d),
                column_spec(prefix + "Acc_agg", agg_binding.acc, domains.d),
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
        trace.external_evaluations["mu_h" + std::to_string(head_index) + "_proj"] =
            trace.polynomials.at(prefix + "H_prime").evaluate(trace.challenges.at("y_proj_h" + std::to_string(head_index)));
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

    const auto h_cat = quantize_matrix(forward.hidden_concat);
    add_dynamic_commitment_batch(
        trace,
        {matrix_spec("P_H_cat", h_cat)},
        context.kzg,
        keep_trace_payloads,
        metrics);
    transcript.absorb_commitment("P_H_cat", trace.commitments.at("P_H_cat").point);
    trace.challenges["xi_cat"] = transcript.challenge("xi_cat");
    const auto h_cat_star = compress_rows_with_challenge(h_cat, trace.challenges.at("xi_cat"));
    add_dynamic_commitment_batch(
        trace,
        {column_spec("P_H_cat_star", padded_column(h_cat_star, domains.n->size), domains.n)},
        context.kzg,
        keep_trace_payloads,
        metrics);
    transcript.absorb_commitment("P_H_cat_star", trace.commitments.at("P_H_cat_star").point);
    trace.challenges["y_cat"] = transcript.challenge("y_cat");
    const auto cat_binding = build_binding_trace(
        h_cat,
        domains.n,
        n_nodes,
        trace.challenges.at("y_cat"),
        powers(trace.challenges.at("xi_cat"), d_cat),
        domains.cat,
        d_cat);
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
    const auto y_matrix = quantize_matrix(forward.Y);
    const auto y_lin_matrix = quantize_matrix(forward.Y_lin);
    const auto out_m_edge = broadcast_node_values(out_m, edges, false, domains.edge->size);
    const auto out_sum_edge = broadcast_node_values(out_sum, edges, false, domains.edge->size);
    const auto out_inv_edge = broadcast_node_values(out_inv, edges, false, domains.edge->size);
    add_dynamic_commitment_batch(
        trace,
        {
            matrix_spec("P_out_Y_prime", y_prime),
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
            matrix_spec("P_Y_lin", y_lin_matrix),
            matrix_spec("P_Y", y_matrix),
        },
        context.kzg,
        keep_trace_payloads,
        metrics);

    transcript.absorb_commitment("P_out_Y_prime", trace.commitments.at("P_out_Y_prime").point);
    trace.challenges["y_proj_out"] = transcript.challenge("y_proj_out");
    trace.challenges["xi_out"] = transcript.challenge("xi_out");
    transcript.absorb_commitment("P_out_E_src", trace.commitments.at("P_out_E_src").point);
    trace.challenges["y_src_out"] = transcript.challenge("y_src_out");
    transcript.absorb_commitment("P_out_E_dst", trace.commitments.at("P_out_E_dst").point);
    trace.challenges["y_dst_out"] = transcript.challenge("y_dst_out");

    const auto y_prime_star = compress_rows_with_challenge(y_prime, trace.challenges.at("xi_out"));
    const auto y_prime_star_edge = broadcast_node_values(y_prime_star, edges, true, domains.edge->size);
    const auto y_star = compress_rows_with_challenge(y_matrix, trace.challenges.at("xi_out"));
    const auto y_star_edge = broadcast_node_values(y_star, edges, false, domains.edge->size);
    const auto eta_src_out = transcript.challenge("eta_src_out");
    const auto beta_src_out = transcript.challenge("beta_src_out");
    trace.challenges["eta_src_out"] = eta_src_out;
    trace.challenges["beta_src_out"] = beta_src_out;
    const auto eta_src_out_powers = powers(eta_src_out, 2);
    std::vector<FieldElement> out_src_table(n_nodes, FieldElement::zero());
    std::vector<FieldElement> out_src_query(n_edges, FieldElement::zero());
    for (std::size_t i = 0; i < n_nodes; ++i) {
        out_src_table[i] = absolute_id_fields[i] + eta_src_out_powers[1] * out_e_src[i];
    }
    for (std::size_t k = 0; k < n_edges; ++k) {
        out_src_query[k] = FieldElement(edges[k].src) + eta_src_out_powers[1] * y_prime_star_edge[k];
    }
    const auto out_src_mult = count_by_src(edges, n_nodes, domains.n->size);
    auto out_src_route = build_route_trace(out_src_table, out_src_query, out_src_mult, n_nodes, n_edges, domains.n, domains.edge, beta_src_out);
    const auto lambda_out = transcript.challenge("lambda_out");
    trace.challenges["lambda_out"] = lambda_out;
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
    auto psq_out = build_group_prefix_state(w_out, q_new);
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
    transcript.absorb_commitment("P_Y", trace.commitments.at("P_Y").point);
    transcript.absorb_commitment("P_out_Y_star", trace.commitments.at("P_out_Y_star").point);
    trace.challenges["y_out_star"] = transcript.challenge("y_out_star");
    const auto out_proj_binding = build_binding_trace(
        h_cat,
        domains.n,
        n_nodes,
        trace.challenges.at("y_proj_out"),
        linear_form_by_powers(
            quantize_matrix(context.model.output_head.seq_kernel_fp),
            powers(trace.challenges.at("y_proj_out"), n_classes)),
        domains.cat,
        d_cat);
    const auto out_src_binding = build_binding_trace(
        y_prime,
        domains.n,
        n_nodes,
        trace.challenges.at("y_src_out"),
        quantize_vector(context.model.output_head.attn_src_kernel_fp),
        domains.c,
        n_classes);
    const auto out_dst_binding = build_binding_trace(
        y_prime,
        domains.n,
        n_nodes,
        trace.challenges.at("y_dst_out"),
        quantize_vector(context.model.output_head.attn_dst_kernel_fp),
        domains.c,
        n_classes);
    const auto out_y_binding = build_binding_trace(
        y_matrix,
        domains.n,
        n_nodes,
        trace.challenges.at("y_out_star"),
        powers(trace.challenges.at("xi_out"), n_classes),
        domains.c,
        n_classes);
    const auto eta_dst_out = transcript.challenge("eta_dst_out");
    const auto beta_dst_out = transcript.challenge("beta_dst_out");
    trace.challenges["eta_dst_out"] = eta_dst_out;
    trace.challenges["beta_dst_out"] = beta_dst_out;
    const auto eta_dst_out_powers = powers(eta_dst_out, 5);
    std::vector<FieldElement> out_dst_table(n_nodes, FieldElement::zero());
    std::vector<FieldElement> out_dst_query(n_edges, FieldElement::zero());
    const auto out_e_dst_edge = broadcast_node_values(out_e_dst, edges, false, domains.edge->size);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        out_dst_table[i] = absolute_id_fields[i]
            + eta_dst_out_powers[1] * out_e_dst[i]
            + eta_dst_out_powers[2] * out_m[i]
            + eta_dst_out_powers[3] * out_sum[i]
            + eta_dst_out_powers[4] * out_inv[i]
            + eta_dst_out.pow(5) * y_star[i];
    }
    for (std::size_t k = 0; k < n_edges; ++k) {
        out_dst_query[k] = FieldElement(edges[k].dst)
            + eta_dst_out_powers[1] * out_e_dst_edge[k]
            + eta_dst_out_powers[2] * out_m_edge[k]
            + eta_dst_out_powers[3] * out_sum_edge[k]
            + eta_dst_out_powers[4] * out_inv_edge[k]
            + eta_dst_out.pow(5) * y_star_edge[k];
    }
    const auto out_dst_mult = count_by_dst(edges, n_nodes, domains.n->size);
    auto out_dst_route = build_route_trace(out_dst_table, out_dst_query, out_dst_mult, n_nodes, n_edges, domains.n, domains.edge, beta_dst_out);
    add_dynamic_commitment_batch(
        trace,
        {
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
    trace.external_evaluations["mu_out_proj"] = trace.polynomials.at("P_out_Y_prime").evaluate(trace.challenges.at("y_proj_out"));
    trace.external_evaluations["mu_out_src"] = trace.polynomials.at("P_out_E_src").evaluate(trace.challenges.at("y_src_out"));
    trace.external_evaluations["mu_out_dst"] = trace.polynomials.at("P_out_E_dst").evaluate(trace.challenges.at("y_dst_out"));
    trace.external_evaluations["mu_out_star"] = trace.polynomials.at("P_out_Y_star").evaluate(trace.challenges.at("y_out_star"));
    trace.challenges["y_out"] = transcript.challenge("y_out");
    trace.external_evaluations["mu_out"] = trace.polynomials.at("P_Y").evaluate(trace.challenges.at("y_out"));

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

    trace.commitment_order = dynamic_commitment_labels(context);
    if (metrics != nullptr) {
        metrics->trace_generation_ms += elapsed_ms(trace_start, Clock::now());
    }
    return trace;
}

}  // namespace

TraceArtifacts build_trace(const ProtocolContext& context, RunMetrics* metrics) {
    const auto trace_start = Clock::now();
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

    const auto q_new = eval_data(context, "P_Q_new_edge");
    const auto q_end = eval_data(context, "P_Q_end_edge");
    const auto q_tbl_feat = eval_data(context, "P_Q_tbl_feat");
    const auto q_qry_feat = eval_data(context, "P_Q_qry_feat");
    const auto q_tbl_l = eval_data(context, "P_Q_tbl_L");
    const auto q_qry_l = eval_data(context, "P_Q_qry_L");
    const auto q_tbl_r = eval_data(context, "P_Q_tbl_R");
    const auto q_qry_r = eval_data(context, "P_Q_qry_R");
    const auto q_tbl_exp = eval_data(context, "P_Q_tbl_exp");
    const auto q_qry_exp = eval_data(context, "P_Q_qry_exp");

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
        metrics->trace_generation_ms = elapsed_ms(trace_start, Clock::now());
    }
    return trace;
}

}  // namespace gatzk::protocol
