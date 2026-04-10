#include "gatzk/protocol/prover.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <functional>
#include <iomanip>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include "gatzk/algebra/eval_backend.hpp"
#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/data/loader.hpp"
#include "gatzk/model/gat.hpp"
#include "gatzk/protocol/challenges.hpp"
#include "gatzk/protocol/lookup.hpp"
#include "gatzk/protocol/quotients.hpp"
#include "gatzk/protocol/schema.hpp"
#include "gatzk/util/logging.hpp"
#include "gatzk/util/route2.hpp"

namespace gatzk::protocol {
namespace {

using algebra::FieldElement;
using algebra::Polynomial;
using crypto::Commitment;
using Clock = std::chrono::steady_clock;

double elapsed_ms(const Clock::time_point& start, const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

Polynomial make_eval_poly(
    const std::string& name,
    const std::vector<FieldElement>& values,
    const std::shared_ptr<algebra::RootOfUnityDomain>& domain) {
    return Polynomial::from_evaluations(name, values, domain);
}

Polynomial make_coeff_poly(const std::string& name, const std::vector<FieldElement>& values) {
    return Polynomial::from_coefficients(name, values);
}

void add_public_poly(
    ProtocolContext& context,
    const std::string& name,
    const Polynomial& polynomial) {
    context.public_polynomials[name] = polynomial;
    context.public_commitments[name] = crypto::KZG::commit(name, polynomial, context.kzg);
}

void add_public_tau_commitment(
    ProtocolContext& context,
    const std::string& name,
    const FieldElement& tau_evaluation) {
    context.public_commitments[name] = crypto::KZG::commit_tau_evaluation(name, tau_evaluation, context.kzg);
}

void add_static_commitment(
    ProtocolContext& context,
    const std::string& name,
    const Polynomial& polynomial) {
    context.static_commitments[name] = crypto::KZG::commit(name, polynomial, context.kzg);
}

void add_static_tau_commitment(
    ProtocolContext& context,
    const std::string& name,
    const FieldElement& tau_evaluation) {
    context.static_commitments[name] = crypto::KZG::commit_tau_evaluation(name, tau_evaluation, context.kzg);
}

std::vector<FieldElement> padded(const std::vector<FieldElement>& values, std::size_t size) {
    std::vector<FieldElement> out(size, FieldElement::zero());
    std::copy(values.begin(), values.end(), out.begin());
    return out;
}

std::vector<FieldElement> ids_to_field(const std::vector<std::size_t>& ids, std::size_t size) {
    std::vector<FieldElement> out(size, FieldElement::zero());
    for (std::size_t i = 0; i < ids.size(); ++i) {
        out[i] = FieldElement(ids[i]);
    }
    return out;
}

std::vector<FieldElement> edge_component(const std::vector<data::Edge>& edges, std::size_t size, bool use_src) {
    std::vector<FieldElement> out(size, FieldElement::zero());
    for (std::size_t i = 0; i < edges.size(); ++i) {
        out[i] = FieldElement(use_src ? edges[i].src : edges[i].dst);
    }
    return out;
}

std::vector<FieldElement> feature_table_row_indices(
    std::size_t node_count,
    std::size_t feature_count,
    std::size_t size) {
    std::vector<FieldElement> out(size, FieldElement::zero());
    for (std::size_t v = 0; v < node_count; ++v) {
        for (std::size_t j = 0; j < feature_count; ++j) {
            out[v * feature_count + j] = FieldElement(v);
        }
    }
    return out;
}

std::vector<FieldElement> feature_table_col_indices(
    std::size_t node_count,
    std::size_t feature_count,
    std::size_t size) {
    std::vector<FieldElement> out(size, FieldElement::zero());
    for (std::size_t v = 0; v < node_count; ++v) {
        for (std::size_t j = 0; j < feature_count; ++j) {
            out[v * feature_count + j] = FieldElement(j);
        }
    }
    return out;
}

std::vector<FieldElement> feature_query_row_indices(
    std::size_t local_node_count,
    std::size_t feature_count,
    std::size_t size) {
    std::vector<FieldElement> out(size, FieldElement::zero());
    for (std::size_t i = 0; i < local_node_count; ++i) {
        for (std::size_t j = 0; j < feature_count; ++j) {
            out[i * feature_count + j] = FieldElement(i);
        }
    }
    return out;
}

std::vector<FieldElement> feature_query_col_indices(
    std::size_t local_node_count,
    std::size_t feature_count,
    std::size_t size) {
    return feature_table_col_indices(local_node_count, feature_count, size);
}

std::vector<FieldElement> feature_query_absolute_ids(
    const std::vector<std::size_t>& absolute_ids,
    std::size_t feature_count,
    std::size_t size) {
    std::vector<FieldElement> out(size, FieldElement::zero());
    for (std::size_t i = 0; i < absolute_ids.size(); ++i) {
        for (std::size_t j = 0; j < feature_count; ++j) {
            out[i * feature_count + j] = FieldElement(absolute_ids[i]);
        }
    }
    return out;
}

bool full_graph_feature_identity_enabled(const ProtocolContext& context) {
    if (context.config.batching_rule != "whole_graph_single") {
        return false;
    }
    if (context.local.num_nodes != context.dataset.num_nodes
        || context.local.num_features != context.dataset.num_features) {
        return false;
    }
    for (std::size_t i = 0; i < context.local.absolute_ids.size(); ++i) {
        if (context.local.absolute_ids[i] != i) {
            return false;
        }
    }
    return true;
}

bool lazy_large_fh_public_enabled(const ProtocolContext& context) {
    constexpr std::size_t kMinLazyLargeFhDomainSize = 1ULL << 24;
    return context.config.batching_rule == "whole_graph_single"
        && context.local.num_nodes == context.dataset.num_nodes
        && context.local.num_features == context.dataset.num_features
        && context.domains.fh != nullptr
        && context.domains.fh->size >= kMinLazyLargeFhDomainSize;
}

bool is_lazy_large_fh_public_label(const std::string& name) {
    return name == "P_T_H"
        || name == "P_Row_feat_tbl"
        || name == "P_Col_feat_tbl"
        || name == "P_Row_feat_qry"
        || name == "P_Col_feat_qry"
        || name == "P_I_feat_qry"
        || name == "P_Q_tbl_feat"
        || name == "P_Q_qry_feat";
}

bool lazy_full_feature_lookup_trace_enabled(const ProtocolContext& context) {
    return full_graph_feature_identity_enabled(context);
}

bool is_lazy_full_feature_lookup_trace_label(const std::string& name) {
    return name == "P_Table_feat"
        || name == "P_Query_feat"
        || name == "P_m_feat"
        || name == "P_R_feat";
}

std::size_t label_index_or_npos(
    const std::vector<std::string>& labels,
    std::string_view target) {
    for (std::size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] == target) {
            return i;
        }
    }
    return labels.size();
}

template <typename ValueFn>
FieldElement evaluate_truncated_domain_polynomial(
    const std::shared_ptr<algebra::RootOfUnityDomain>& domain,
    std::size_t valid_count,
    const FieldElement& point,
    ValueFn&& value_fn) {
    if (domain == nullptr) {
        throw std::runtime_error("lazy public polynomial requires a domain");
    }
    if (valid_count == 0) {
        return FieldElement::zero();
    }
    if (const auto shift = domain->rotation_shift(FieldElement::one(), point); shift.has_value()) {
        return *shift < valid_count ? value_fn(*shift) : FieldElement::zero();
    }
    const auto zero_eval = domain->zero_polynomial_eval(point);
    FieldElement sum = FieldElement::zero();
    const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    if (valid_count >= (1ULL << 20) && cpu_count > 1) {
        const auto task_count = std::min<std::size_t>(cpu_count, (valid_count + (1ULL << 20) - 1) / (1ULL << 20));
        const auto chunk_size = (valid_count + task_count - 1) / task_count;
        std::vector<std::future<FieldElement>> futures;
        futures.reserve(task_count);
        for (std::size_t task = 0; task < task_count; ++task) {
            const auto begin = task * chunk_size;
            const auto end = std::min(valid_count, begin + chunk_size);
            if (begin >= end) {
                break;
            }
            futures.push_back(std::async(
                std::launch::async,
                [&, begin, end]() {
                    FieldElement partial = FieldElement::zero();
                    auto omega_power = domain->omega.pow(static_cast<std::uint64_t>(begin));
                    for (std::size_t index = begin; index < end; ++index) {
                        partial += value_fn(index) * omega_power / (point - omega_power);
                        omega_power *= domain->omega;
                    }
                    return partial;
                }));
        }
        for (auto& future : futures) {
            sum += future.get();
        }
    } else {
        FieldElement omega_power = FieldElement::one();
        for (std::size_t index = 0; index < valid_count; ++index) {
            sum += value_fn(index) * omega_power / (point - omega_power);
            omega_power *= domain->omega;
        }
    }
    return zero_eval * domain->inv_size * sum;
}

FieldElement evaluate_lazy_large_fh_public_poly(
    const ProtocolContext& context,
    const std::string& name,
    const FieldElement& point) {
    if (!lazy_large_fh_public_enabled(context) || !is_lazy_large_fh_public_label(name)) {
        throw std::runtime_error("unsupported lazy public polynomial: " + name);
    }
    const bool full_feature_identity = full_graph_feature_identity_enabled(context);
    std::string canonical_name = name;
    if (full_feature_identity) {
        if (canonical_name == "P_Q_qry_feat") {
            canonical_name = "P_Q_tbl_feat";
        } else if (canonical_name == "P_Row_feat_qry" || canonical_name == "P_I_feat_qry") {
            canonical_name = "P_Row_feat_tbl";
        } else if (canonical_name == "P_Col_feat_qry") {
            canonical_name = "P_Col_feat_tbl";
        }
    }

    static std::mutex cache_mutex;
    static std::unordered_map<std::string, FieldElement> cache;
    const auto cache_key =
        context.config.checkpoint_bundle
        + ":" + context.config.dataset
        + ":" + std::to_string(context.dataset.num_nodes)
        + ":" + std::to_string(context.local.num_features)
        + ":" + canonical_name
        + "@" + point.to_string();
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (const auto it = cache.find(cache_key); it != cache.end()) {
            return it->second;
        }
    }

    const auto& domain = context.domains.fh;
    const std::size_t dataset_rows = context.dataset.num_nodes;
    const std::size_t local_rows = context.local.num_nodes;
    const std::size_t feature_count = context.local.num_features;
    const std::size_t dataset_valid = dataset_rows * feature_count;
    const std::size_t local_valid = local_rows * feature_count;
    const auto& feature_matrix = !context.dataset.features.empty() ? context.dataset.features : context.local.features;
    FieldElement value = FieldElement::zero();
    if (canonical_name == "P_Q_tbl_feat") {
        value = evaluate_truncated_domain_polynomial(
            domain,
            dataset_valid,
            point,
            [](std::size_t) { return FieldElement::one(); });
    } else if (canonical_name == "P_Q_qry_feat") {
        value = evaluate_truncated_domain_polynomial(
            domain,
            local_valid,
            point,
            [](std::size_t) { return FieldElement::one(); });
    } else if (canonical_name == "P_Row_feat_tbl") {
        value = evaluate_truncated_domain_polynomial(
            domain,
            dataset_valid,
            point,
            [&](std::size_t index) { return FieldElement(index / feature_count); });
    } else if (canonical_name == "P_Col_feat_tbl") {
        value = evaluate_truncated_domain_polynomial(
            domain,
            dataset_valid,
            point,
            [&](std::size_t index) { return FieldElement(index % feature_count); });
    } else if (canonical_name == "P_Row_feat_qry") {
        value = evaluate_truncated_domain_polynomial(
            domain,
            local_valid,
            point,
            [&](std::size_t index) { return FieldElement(index / feature_count); });
    } else if (canonical_name == "P_Col_feat_qry") {
        value = evaluate_truncated_domain_polynomial(
            domain,
            local_valid,
            point,
            [&](std::size_t index) { return FieldElement(index % feature_count); });
    } else if (canonical_name == "P_I_feat_qry") {
        value = evaluate_truncated_domain_polynomial(
            domain,
            local_valid,
            point,
            [&](std::size_t index) { return FieldElement(context.local.absolute_ids[index / feature_count]); });
    } else if (canonical_name == "P_T_H") {
        value = evaluate_truncated_domain_polynomial(
            domain,
            dataset_valid,
            point,
            [&](std::size_t index) {
                const auto row = index / feature_count;
                const auto column = index % feature_count;
                return feature_matrix[row][column];
            });
    } else {
        throw std::runtime_error("unsupported lazy public polynomial label: " + canonical_name);
    }
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        cache.emplace(cache_key, value);
    }
    return value;
}

FieldElement evaluate_lazy_full_feature_lookup_trace_poly(
    const ProtocolContext& context,
    const std::map<std::string, FieldElement>& challenges,
    const std::string& name,
    const FieldElement& point) {
    if (!lazy_full_feature_lookup_trace_enabled(context) || !is_lazy_full_feature_lookup_trace_label(name)) {
        throw std::runtime_error("unsupported lazy feature lookup trace polynomial: " + name);
    }
    if (name == "P_R_feat") {
        return FieldElement::zero();
    }
    const bool full_feature_identity = full_graph_feature_identity_enabled(context);
    if (name == "P_m_feat") {
        if (full_feature_identity) {
            return evaluate_lazy_large_fh_public_poly(context, "P_Q_tbl_feat", point);
        }
        return evaluate_lazy_large_fh_public_poly(context, "P_Q_qry_feat", point);
    }
    if (name == "P_Query_feat" && full_feature_identity) {
        return evaluate_lazy_full_feature_lookup_trace_poly(context, challenges, "P_Table_feat", point);
    }

    const auto eta_feat = challenges.at("eta_feat");
    const auto row_label = name == "P_Table_feat" ? "P_Row_feat_tbl" : "P_Row_feat_qry";
    const auto col_label = name == "P_Table_feat" ? "P_Col_feat_tbl" : "P_Col_feat_qry";
    const auto row_eval = evaluate_lazy_large_fh_public_poly(context, row_label, point);
    const auto col_eval = evaluate_lazy_large_fh_public_poly(context, col_label, point);
    const auto feat_eval = evaluate_lazy_large_fh_public_poly(context, "P_T_H", point);
    return row_eval + eta_feat * col_eval + eta_feat.pow(2) * feat_eval;
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

FieldElement matrix_row_major_evaluation_with_row_stride(
    const model::Matrix& matrix,
    const FieldElement& point,
    const FieldElement& row_stride) {
    if (matrix.empty() || matrix.front().empty()) {
        return FieldElement::zero();
    }

    mcl::Fr out;
    out.clear();
    for (std::size_t row = matrix.size(); row-- > 0;) {
        const auto row_eval = row_polynomial_at_point(matrix[row], point);
        mcl::Fr::mul(out, out, row_stride.native());
        mcl::Fr::add(out, out, row_eval.native());
    }
    return FieldElement::from_native(out);
}

std::vector<FieldElement> q_new_selector(const std::vector<data::Edge>& edges, std::size_t size) {
    std::vector<FieldElement> out(size, FieldElement::zero());
    for (std::size_t i = 0; i < edges.size(); ++i) {
        const bool is_new = i == 0 || edges[i].dst != edges[i - 1].dst;
        out[i] = FieldElement(is_new);
    }
    return out;
}

std::vector<FieldElement> q_end_selector(const std::vector<data::Edge>& edges, std::size_t size) {
    std::vector<FieldElement> out(size, FieldElement::zero());
    for (std::size_t i = 0; i < edges.size(); ++i) {
        const bool is_end = (i + 1 == edges.size()) || (edges[i].dst != edges[i + 1].dst);
        out[i] = FieldElement(is_end);
    }
    return out;
}

std::vector<FieldElement> make_range_table(std::size_t size) {
    std::vector<FieldElement> out(size, FieldElement::zero());
    for (std::size_t i = 0; i < size; ++i) {
        out[i] = FieldElement(i);
    }
    return out;
}

std::int64_t quantize_float_model_value_signed(double value) {
    return static_cast<std::int64_t>(
        value >= 0.0 ? value * 16.0 + 0.5 : value * 16.0 - 0.5);
}

FieldElement quantize_float_model_value(double value) {
    return FieldElement::from_signed(quantize_float_model_value_signed(value));
}

double dequantize_trace_scalar(std::int64_t value) {
    return static_cast<double>(value) / 16.0;
}

std::vector<std::pair<FieldElement, FieldElement>> make_lrelu_table(std::size_t bound) {
    std::vector<std::pair<FieldElement, FieldElement>> out;
    out.reserve(bound * 2 + 1);
    for (std::int64_t raw = -static_cast<std::int64_t>(bound); raw <= static_cast<std::int64_t>(bound); ++raw) {
        const auto input = dequantize_trace_scalar(raw);
        const auto output = input >= 0.0 ? input : 0.2 * input;
        out.push_back({FieldElement::from_signed(raw), quantize_float_model_value(output)});
    }
    return out;
}

std::size_t elu_table_band() {
    return 1;
}

std::vector<std::pair<FieldElement, FieldElement>> make_elu_table(std::size_t bound) {
    std::vector<std::pair<FieldElement, FieldElement>> out;
    out.reserve((bound * 2 + 1) * (2 * elu_table_band() + 1));
    for (std::int64_t raw = -static_cast<std::int64_t>(bound); raw <= static_cast<std::int64_t>(bound); ++raw) {
        const auto input = dequantize_trace_scalar(raw);
        const auto central_signed = quantize_float_model_value_signed(input >= 0.0 ? input : std::exp(input) - 1.0);
        for (std::int64_t offset = -static_cast<std::int64_t>(elu_table_band());
             offset <= static_cast<std::int64_t>(elu_table_band());
             ++offset) {
            out.push_back({
                FieldElement::from_signed(raw),
                FieldElement::from_signed(central_signed + offset),
            });
        }
    }
    return out;
}

std::size_t exp_table_band() {
    return 2;
}

std::vector<std::pair<FieldElement, FieldElement>> make_exp_table(std::size_t size) {
    std::vector<std::pair<FieldElement, FieldElement>> out;
    out.reserve(size * (2 * exp_table_band() + 1));
    for (std::size_t i = 0; i < size; ++i) {
        const auto input = static_cast<std::int64_t>(i);
        const auto central_signed = quantize_float_model_value_signed(std::exp(-dequantize_trace_scalar(input)));
        for (std::int64_t offset = -static_cast<std::int64_t>(exp_table_band());
             offset <= static_cast<std::int64_t>(exp_table_band());
             ++offset) {
            const auto candidate = std::max<std::int64_t>(0, central_signed + offset);
            out.push_back({
                FieldElement(static_cast<std::uint64_t>(i)),
                FieldElement::from_signed(candidate),
            });
        }
    }
    return out;
}

std::vector<FieldElement> quantize_model_vector(const std::vector<double>& values) {
    std::vector<FieldElement> out(values.size(), FieldElement::zero());
    for (std::size_t i = 0; i < values.size(); ++i) {
        out[i] = quantize_float_model_value(values[i]);
    }
    return out;
}

model::Matrix quantize_model_matrix(const model::FloatMatrix& matrix) {
    model::Matrix out(matrix.size());
    for (std::size_t row = 0; row < matrix.size(); ++row) {
        out[row] = quantize_model_vector(matrix[row]);
    }
    return out;
}

std::vector<std::string> vector_to_lines(const std::vector<FieldElement>& values) {
    std::vector<std::string> lines;
    lines.reserve(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        lines.push_back(std::to_string(i) + " " + values[i].to_string());
    }
    return lines;
}

std::vector<std::string> matrix_to_lines(const model::Matrix& matrix) {
    std::vector<std::string> lines;
    for (std::size_t i = 0; i < matrix.size(); ++i) {
        std::ostringstream row;
        row << i;
        for (const auto& value : matrix[i]) {
            row << ' ' << value.to_string();
        }
        lines.push_back(row.str());
    }
    return lines;
}

void append_note(RunMetrics* metrics, const std::string& note) {
    if (metrics == nullptr || note.empty()) {
        return;
    }
    if (!metrics->notes.empty()) {
        metrics->notes += "; ";
    }
    metrics->notes += note;
}

std::string point_key(const FieldElement& point) {
    return point.to_string();
}

std::string domain_point_key(
    const std::shared_ptr<algebra::RootOfUnityDomain>& domain,
    const FieldElement& point) {
    return domain->name + ":" + std::to_string(domain->size) + ":" + point_key(point);
}

std::string context_cache_key(const util::AppConfig& config) {
    std::ostringstream stream;
    // export_dir, dump_trace and prove_enabled do not affect protocol objects, so
    // they are intentionally excluded from the reusable static context key.
    stream << config.project_root << '|'
           << config.dataset << '|'
           << config.data_root << '|'
           << config.cache_root << '|'
           << config.checkpoint_bundle << '|'
           << config.hidden_dim << '|'
           << config.num_classes << '|'
           << config.range_bits << '|'
           << config.seed << '|'
           << config.local_nodes << '|'
           << config.center_node << '|'
           << config.layer_count << '|'
           << config.K_out << '|'
           << config.batch_graphs << '|'
           << config.task_type << '|'
           << config.report_unit << '|'
           << config.batching_rule << '|'
           << config.subgraph_rule << '|'
           << config.self_loop_rule << '|'
           << config.edge_sort_rule << '|'
           << config.chunking_rule << '|'
           << (config.allow_synthetic_model ? "synthetic" : "formal");
    for (const auto input_dim : config.d_in_profile) {
        stream << "|din=" << input_dim;
    }
    for (const auto& layer : config.hidden_profile) {
        stream << "|hid=" << layer.head_count << 'x' << layer.head_dim;
    }
    return stream.str();
}

std::size_t hidden_head_width(const model::ModelParameters& parameters, const util::AppConfig& config) {
    if (parameters.has_real_multihead) {
        return model::max_hidden_head_dim(parameters);
    }
    return config.hidden_dim;
}

std::size_t concat_width(const model::ModelParameters& parameters, const util::AppConfig& config) {
    if (parameters.has_real_multihead) {
        return model::max_hidden_concat_width(parameters);
    }
    return config.hidden_dim;
}

PublicMetadata build_public_metadata(const ProtocolContext& context) { return canonical_public_metadata(context); }

struct ExternalEvalSpec {
    std::string proof_name;
    std::string label;
    std::string challenge_name;
};

std::vector<ExternalEvalSpec> multihead_external_specs(const ProtocolContext& context) {
    std::vector<ExternalEvalSpec> specs;
    for (std::size_t head_index = 0; head_index < context.model.hidden_heads.size(); ++head_index) {
        const auto suffix = "h" + std::to_string(head_index);
        const auto prefix = "P_h" + std::to_string(head_index) + "_";
        specs.push_back({"mu_" + suffix + "_proj", prefix + "H_prime", "y_proj_h" + std::to_string(head_index)});
        specs.push_back({"mu_" + suffix + "_src", prefix + "E_src", "y_src_h" + std::to_string(head_index)});
        specs.push_back({"mu_" + suffix + "_dst", prefix + "E_dst", "y_dst_h" + std::to_string(head_index)});
        specs.push_back({"mu_" + suffix + "_star", prefix + "H_star", "y_star_h" + std::to_string(head_index)});
        specs.push_back({"mu_" + suffix + "_agg_pre", prefix + "H_agg_pre_star", "y_agg_pre_h" + std::to_string(head_index)});
        specs.push_back({"mu_" + suffix + "_agg", prefix + "H_agg_star", "y_agg_h" + std::to_string(head_index)});
    }
    for (std::size_t layer_index = 0; layer_index < context.model.hidden_layers.size(); ++layer_index) {
        const bool is_final_layer = layer_index + 1 == context.model.hidden_layers.size();
        specs.push_back({
            is_final_layer ? "mu_cat" : "mu_cat_l" + std::to_string(layer_index),
            hidden_layer_concat_star_label(layer_index, is_final_layer),
            hidden_concat_y_name(layer_index, is_final_layer),
        });
    }
    const bool legacy_single_output = context.model.output_layer.heads.size() == 1;
    for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index) {
        const auto prefix = output_head_prefix(head_index, legacy_single_output);
        specs.push_back({
            legacy_single_output ? "mu_out_proj" : output_external_eval_name("proj", head_index, false),
            prefix + "_Y_prime",
            output_challenge_name("y_proj_out", head_index, legacy_single_output),
        });
        specs.push_back({
            legacy_single_output ? "mu_out_src" : output_external_eval_name("src", head_index, false),
            prefix + "_E_src",
            output_challenge_name("y_src_out", head_index, legacy_single_output),
        });
        specs.push_back({
            legacy_single_output ? "mu_out_dst" : output_external_eval_name("dst", head_index, false),
            prefix + "_E_dst",
            output_challenge_name("y_dst_out", head_index, legacy_single_output),
        });
        specs.push_back({
            legacy_single_output ? "mu_out_star" : output_external_eval_name("star", head_index, false),
            prefix + "_Y_star",
            output_challenge_name("y_out_star", head_index, legacy_single_output),
        });
        if (!legacy_single_output) {
            specs.push_back({output_external_eval_name("y_lin", head_index, false), output_y_lin_label(head_index, false), "y_out"});
            specs.push_back({output_external_eval_name("y", head_index, false), output_y_label(head_index, false), "y_out"});
        }
    }
    specs.push_back({"mu_Y_lin", "P_Y_lin", "y_out"});
    specs.push_back({"mu_out", "P_Y", "y_out"});
    return specs;
}

double* domain_open_metric(RunMetrics* metrics, const std::string& bundle_name) {
    if (metrics == nullptr) {
        return nullptr;
    }
    if (bundle_name == "FH") return &metrics->domain_open_fh_ms;
    if (bundle_name == "edge") return &metrics->domain_open_edge_ms;
    if (bundle_name == "in") return &metrics->domain_open_in_ms;
    if (bundle_name == "d_h") return &metrics->domain_open_d_h_ms;
    if (bundle_name == "cat") return &metrics->domain_open_cat_ms;
    if (bundle_name == "C") return &metrics->domain_open_c_ms;
    if (bundle_name == "N") return &metrics->domain_open_n_ms;
    return nullptr;
}

double* quotient_metric(RunMetrics* metrics, const std::string& label) {
    if (metrics == nullptr) {
        return nullptr;
    }
    if (label == "t_FH") return &metrics->quotient_t_fh_ms;
    if (label == "t_edge") return &metrics->quotient_t_edge_ms;
    if (label == "t_in") return &metrics->quotient_t_in_ms;
    if (label == "t_d_h") return &metrics->quotient_t_d_h_ms;
    if (label == "t_cat") return &metrics->quotient_t_cat_ms;
    if (label == "t_C") return &metrics->quotient_t_c_ms;
    if (label == "t_N") return &metrics->quotient_t_n_ms;
    return nullptr;
}

std::filesystem::path resolve_project_path(
    const std::string& project_root,
    const std::string& relative_or_absolute) {
    const std::filesystem::path path(relative_or_absolute);
    if (path.is_absolute()) {
        return path;
    }
    return std::filesystem::path(project_root) / path;
}

struct DomainEvaluationWeights {
    std::optional<std::size_t> direct_index;
    std::vector<mcl::Fr> native_weights;
    algebra::PackedFieldBuffer packed_weights;
};

class ProofDomainWeightCache {
  public:
    const DomainEvaluationWeights& get(
        const std::shared_ptr<algebra::RootOfUnityDomain>& domain,
        const FieldElement& point) {
        const auto cache_key = domain_point_key(domain, point);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (const auto it = entries_.find(cache_key); it != entries_.end()) {
                return it->second;
            }
        }

        DomainEvaluationWeights entry;
        if (domain->points_precomputed) {
            for (std::size_t i = 0; i < domain->points.size(); ++i) {
                if (domain->points[i] == point) {
                    entry.direct_index = i;
                    break;
                }
            }
        } else if (const auto shift = domain->rotation_shift(FieldElement::one(), point); shift.has_value()) {
            entry.direct_index = *shift;
        }

        if (!entry.direct_index.has_value()) {
            entry.native_weights = domain->barycentric_weights_native(point);
            algebra::pack_native_field_elements_into(entry.native_weights, &entry.packed_weights);
        }

        std::lock_guard<std::mutex> lock(mutex_);
        auto [it, inserted] = entries_.emplace(cache_key, std::move(entry));
        (void)inserted;
        return it->second;
    }

  private:
    std::mutex mutex_;
    std::unordered_map<std::string, DomainEvaluationWeights> entries_;
};

std::shared_ptr<ProofDomainWeightCache> shared_proof_domain_weight_cache() {
    static auto cache = std::make_shared<ProofDomainWeightCache>();
    return cache;
}

class SharedFeatureMatrixEvaluationCache {
  public:
    FieldElement get_or_compute(
        const std::string& context_key,
        const model::Matrix& matrix,
        const FieldElement& point,
        RunMetrics* metrics) {
        const auto cache_key = context_key + "|P_H@" + point_key(point);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (const auto it = entries_.find(cache_key); it != entries_.end()) {
                return it->second;
            }
        }

        const auto point_power_start = Clock::now();
        const auto row_stride =
            matrix.empty() || matrix.front().empty()
            ? FieldElement::one()
            : point.pow(static_cast<std::uint64_t>(matrix.front().size()));
        if (metrics != nullptr) {
            metrics->fh_point_powers_ms += elapsed_ms(point_power_start, Clock::now());
        }

        const auto feature_eval_start = Clock::now();
        const auto value = matrix_row_major_evaluation_with_row_stride(matrix, point, row_stride);
        if (metrics != nullptr) {
            const auto feature_eval_ms = elapsed_ms(feature_eval_start, Clock::now());
            metrics->fh_feature_poly_interp_ms += feature_eval_ms;
            metrics->fh_interpolation_ms += feature_eval_ms;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        const auto [it, inserted] = entries_.emplace(cache_key, value);
        (void)inserted;
        return it->second;
    }

  private:
    std::mutex mutex_;
    std::unordered_map<std::string, FieldElement> entries_;
};

std::shared_ptr<SharedFeatureMatrixEvaluationCache> shared_feature_matrix_evaluation_cache() {
    static auto cache = std::make_shared<SharedFeatureMatrixEvaluationCache>();
    return cache;
}

std::string labels_cache_key(const std::vector<std::string>& labels);

class SharedTraceEvaluationCache {
  public:
    bool lookup(
        const std::string& context_key,
        const std::shared_ptr<algebra::RootOfUnityDomain>& domain,
        const std::vector<std::string>& labels,
        const FieldElement& point,
        std::vector<FieldElement>* values) {
        const auto cache_key =
            context_key + "|trace|" + domain_point_key(domain, point) + "|" + labels_cache_key(labels);
        std::lock_guard<std::mutex> lock(mutex_);
        if (const auto it = entries_.find(cache_key); it != entries_.end()) {
            if (values != nullptr) {
                *values = it->second;
            }
            return true;
        }
        return false;
    }

    void store(
        const std::string& context_key,
        const std::shared_ptr<algebra::RootOfUnityDomain>& domain,
        const std::vector<std::string>& labels,
        const FieldElement& point,
        std::vector<FieldElement> values) {
        const auto cache_key =
            context_key + "|trace|" + domain_point_key(domain, point) + "|" + labels_cache_key(labels);
        std::lock_guard<std::mutex> lock(mutex_);
        entries_.emplace(std::move(cache_key), std::move(values));
    }

  private:
    std::mutex mutex_;
    std::unordered_map<std::string, std::vector<FieldElement>> entries_;
};

std::shared_ptr<SharedTraceEvaluationCache> shared_trace_evaluation_cache() {
    static auto cache = std::make_shared<SharedTraceEvaluationCache>();
    return cache;
}

std::string labels_cache_key(const std::vector<std::string>& labels) {
    std::string key;
    for (const auto& label : labels) {
        key += label;
        key.push_back('\x1f');
    }
    return key;
}

std::string quotient_build_cache_key(const ProtocolContext& context, const TraceArtifacts& trace) {
    std::ostringstream stream;
    stream << context_cache_key(context.config) << "|quotients";
    for (const auto& label : trace.commitment_order) {
        const auto it = trace.commitments.find(label);
        if (it == trace.commitments.end()) {
            continue;
        }
        stream << '|' << label << '=' << it->second.tau_evaluation.to_string();
    }
    return stream.str();
}

struct CachedQuotientArtifacts {
    std::vector<std::pair<std::string, FieldElement>> named_tau_values;
    std::unordered_map<std::string, Commitment> quotient_commitments;
};

class SharedQuotientArtifactsCache {
  public:
    bool lookup(const std::string& key, CachedQuotientArtifacts* out) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (const auto it = entries_.find(key); it != entries_.end()) {
            if (out != nullptr) {
                *out = it->second;
            }
            return true;
        }
        return false;
    }

    void store(const std::string& key, CachedQuotientArtifacts value) {
        std::lock_guard<std::mutex> lock(mutex_);
        entries_.emplace(key, std::move(value));
    }

  private:
    std::mutex mutex_;
    std::unordered_map<std::string, CachedQuotientArtifacts> entries_;
};

SharedQuotientArtifactsCache& shared_quotient_artifacts_cache() {
    static SharedQuotientArtifactsCache cache;
    return cache;
}

class SharedPublicEvaluationCache {
  public:
    bool lookup(
        const std::string& context_key,
        const std::shared_ptr<algebra::RootOfUnityDomain>& domain,
        const std::vector<std::string>& labels,
        const FieldElement& point,
        std::vector<FieldElement>* values) {
        const auto cache_key =
            context_key + "|" + domain_point_key(domain, point) + "|" + labels_cache_key(labels);
        std::lock_guard<std::mutex> lock(mutex_);
        if (const auto it = entries_.find(cache_key); it != entries_.end()) {
            if (values != nullptr) {
                *values = it->second;
            }
            return true;
        }
        return false;
    }

    void store(
        const std::string& context_key,
        const std::shared_ptr<algebra::RootOfUnityDomain>& domain,
        const std::vector<std::string>& labels,
        const FieldElement& point,
        std::vector<FieldElement> values) {
        const auto cache_key =
            context_key + "|" + domain_point_key(domain, point) + "|" + labels_cache_key(labels);
        std::lock_guard<std::mutex> lock(mutex_);
        entries_.emplace(std::move(cache_key), std::move(values));
    }

  private:
    std::mutex mutex_;
    std::unordered_map<std::string, std::vector<FieldElement>> entries_;
};

std::shared_ptr<SharedPublicEvaluationCache> shared_public_evaluation_cache() {
    static auto cache = std::make_shared<SharedPublicEvaluationCache>();
    return cache;
}

class ProofEvaluationBackendRegistry {
  public:
    ProofEvaluationBackendRegistry(const ProtocolContext& context, const TraceArtifacts& trace) {
        std::unordered_map<std::string, PolynomialBatchGroup> groups;
        auto collect = [&](const auto& polynomial_map) {
            for (const auto& [name, polynomial] : polynomial_map) {
                if (polynomial.basis != algebra::PolynomialBasis::Evaluation || polynomial.domain == nullptr) {
                    continue;
                }
                const auto domain_key = polynomial.domain->name + ":" + std::to_string(polynomial.domain->size);
                auto& group = groups[domain_key];
                if (group.polynomials.empty()) {
                    group.domain = polynomial.domain;
                }
                group.polynomials.push_back({name, &polynomial});
            }
        };
        collect(context.public_polynomials);
        collect(trace.polynomials);

        for (auto& [domain_key, group] : groups) {
            backends_.emplace(
                domain_key,
                algebra::PackedEvaluationBackend(group.domain, std::move(group.polynomials)));
        }
    }

    const algebra::PackedEvaluationBackend* find(const std::shared_ptr<algebra::RootOfUnityDomain>& domain) const {
        const auto domain_key = domain->name + ":" + std::to_string(domain->size);
        if (const auto it = backends_.find(domain_key); it != backends_.end()) {
            return &it->second;
        }
        return nullptr;
    }

  private:
    struct PolynomialBatchGroup {
        std::shared_ptr<algebra::RootOfUnityDomain> domain;
        std::vector<std::pair<std::string, const Polynomial*>> polynomials;
    };

    std::unordered_map<std::string, algebra::PackedEvaluationBackend> backends_;
};

class SharedProofEvaluationBackendRegistryCache {
  public:
    std::shared_ptr<const ProofEvaluationBackendRegistry> get_or_build(
        const std::string& key,
        const std::function<std::shared_ptr<const ProofEvaluationBackendRegistry>()>& build) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (const auto it = entries_.find(key); it != entries_.end()) {
                return it->second;
            }
        }
        auto registry = build();
        std::lock_guard<std::mutex> lock(mutex_);
        const auto [it, _] = entries_.emplace(key, std::move(registry));
        return it->second;
    }

  private:
    std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<const ProofEvaluationBackendRegistry>> entries_;
};

SharedProofEvaluationBackendRegistryCache& shared_proof_backend_registry_cache() {
    static SharedProofEvaluationBackendRegistryCache cache;
    return cache;
}

FieldElement evaluate_spilled_evaluation_polynomial(
    const TraceArtifacts::SpilledEvaluationPolynomial& spilled,
    const std::shared_ptr<algebra::RootOfUnityDomain>& domain,
    const FieldElement& /*point*/,
    const DomainEvaluationWeights& weight_entry) {
    if (domain == nullptr) {
        throw std::runtime_error("spilled polynomial is missing its domain");
    }
    constexpr std::size_t kFieldBytes = 32;
    std::array<std::uint8_t, kFieldBytes> bytes{};
    std::ifstream stream(spilled.path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("failed to open spilled polynomial file: " + spilled.path);
    }
    if (weight_entry.direct_index.has_value()) {
        if (*weight_entry.direct_index >= spilled.size) {
            return FieldElement::zero();
        }
        const auto offset = static_cast<std::streamoff>(*weight_entry.direct_index * kFieldBytes);
        stream.seekg(offset);
        stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!stream) {
            throw std::runtime_error("failed to read direct spilled polynomial entry: " + spilled.path);
        }
        return FieldElement::from_little_endian_mod(bytes.data(), bytes.size());
    }

    mcl::Fr native_sum;
    native_sum.clear();
    for (std::size_t index = 0; index < spilled.size; ++index) {
        stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!stream) {
            throw std::runtime_error("failed to stream spilled polynomial: " + spilled.path);
        }
        const auto value = FieldElement::from_little_endian_mod(bytes.data(), bytes.size());
        mcl::Fr term;
        mcl::Fr::mul(term, value.native(), weight_entry.native_weights[index]);
        mcl::Fr::add(native_sum, native_sum, term);
    }
    return FieldElement::from_native(native_sum);
}

// Proof-scoped memoization only caches values derived from the current trace and
// current Fiat-Shamir challenges. This matches the spec's semantics while
// avoiding repeated evaluation of the same column or quotient at the same point.
class EvaluationMemoization {
  public:
    EvaluationMemoization(
        const ProtocolContext& context,
        const TraceArtifacts& trace,
        const std::map<std::string, FieldElement>& challenges,
        std::shared_ptr<const ProofEvaluationBackendRegistry> backend_registry = nullptr,
        std::shared_ptr<ProofDomainWeightCache> domain_weight_cache = nullptr,
        RunMetrics* metrics = nullptr)
        : context_(context),
          trace_(trace),
          challenges_(challenges),
          backend_registry_(std::move(backend_registry)),
          domain_weight_cache_(std::move(domain_weight_cache)),
          context_key_(context_cache_key(context.config)),
          metrics_(metrics) {
        if (domain_weight_cache_ == nullptr) {
            domain_weight_cache_ = std::make_shared<ProofDomainWeightCache>();
        }
    }

    FieldElement eval_named(const std::string& name, const FieldElement& point) {
        const auto cache_key = name + "@" + point_key(point);
        if (const auto it = value_cache_.find(cache_key); it != value_cache_.end()) {
            return it->second;
        }

        FieldElement value = FieldElement::zero();
        if (name == "P_H" && lazy_full_feature_lookup_trace_enabled(context_)) {
            value = eval_named("P_T_H", point);
        } else if (name == "P_H") {
            value = shared_feature_matrix_evaluation_cache()->get_or_compute(
                context_key_,
                context_.local.features,
                point,
                metrics_);
        } else if (lazy_full_feature_lookup_trace_enabled(context_)
                   && is_lazy_full_feature_lookup_trace_label(name)) {
            value = evaluate_lazy_full_feature_lookup_trace_poly(context_, challenges_, name, point);
        } else if (trace_.polynomials.contains(name)) {
            value = eval_polynomial(trace_.polynomials.at(name), point);
        } else if (trace_.spilled_polynomials.contains(name)) {
            const auto& spilled = trace_.spilled_polynomials.at(name);
            const auto domain = domain_from_name(spilled.domain_name);
            value = evaluate_spilled_evaluation_polynomial(
                spilled,
                domain,
                point,
                domain_weights(domain, point));
        } else if (lazy_large_fh_public_enabled(context_) && is_lazy_large_fh_public_label(name)) {
            value = evaluate_lazy_large_fh_public_poly(context_, name, point);
        } else if (context_.public_polynomials.contains(name)) {
            value = eval_polynomial(context_.public_polynomials.at(name), point);
        } else if (name == "t_FH") {
            FHQuotientProfile fh_profile;
            value = evaluate_t_fh(
                context_,
                challenges_,
                [&](const std::string& inner_name, const FieldElement& inner_point) {
                    return eval_named(inner_name, inner_point);
                },
                point,
                metrics_ != nullptr ? &fh_profile : nullptr);
            if (metrics_ != nullptr) {
                metrics_->fh_quotient_assembly_ms += fh_profile.assembly_ms;
            }
        } else if (name == "t_edge") {
            value = evaluate_t_edge(
                context_,
                challenges_,
                trace_.witness_scalars,
                [&](const std::string& inner_name, const FieldElement& inner_point) {
                    return eval_named(inner_name, inner_point);
                },
                point);
        } else if (name == "t_N") {
            value = evaluate_t_n(
                context_,
                challenges_,
                trace_.witness_scalars,
                [&](const std::string& inner_name, const FieldElement& inner_point) {
                    return eval_named(inner_name, inner_point);
                },
                point);
        } else if (name == "t_in") {
            value = evaluate_t_in(
                context_,
                challenges_,
                trace_.external_evaluations,
                [&](const std::string& inner_name, const FieldElement& inner_point) {
                    return eval_named(inner_name, inner_point);
                },
                point);
        } else if (name == "t_d" || name == "t_d_h") {
            value = evaluate_t_d(
                context_,
                challenges_,
                trace_.external_evaluations,
                [&](const std::string& inner_name, const FieldElement& inner_point) {
                    return eval_named(inner_name, inner_point);
                },
                point);
        } else if (name == "t_cat") {
            value = evaluate_t_cat(
                context_,
                challenges_,
                trace_.external_evaluations,
                [&](const std::string& inner_name, const FieldElement& inner_point) {
                    return eval_named(inner_name, inner_point);
                },
                point);
        } else if (name == "t_C") {
            value = evaluate_t_c(
                context_,
                challenges_,
                trace_.external_evaluations,
                [&](const std::string& inner_name, const FieldElement& inner_point) {
                    return eval_named(inner_name, inner_point);
                },
                point);
        } else {
            throw std::runtime_error("missing polynomial for evaluation: " + name);
        }

        value_cache_.emplace(cache_key, value);
        return value;
    }

  private:
    std::shared_ptr<algebra::RootOfUnityDomain> domain_from_name(const std::string& name) const {
        if (name == "FH") return context_.domains.fh;
        if (name == "edge") return context_.domains.edge;
        if (name == "in") return context_.domains.in;
        if (name == "d") return context_.domains.d;
        if (name == "cat") return context_.domains.cat;
        if (name == "C") return context_.domains.c;
        if (name == "N") return context_.domains.n;
        throw std::runtime_error("unknown spilled polynomial domain: " + name);
    }

    const Polynomial* lookup_polynomial(const std::string& name) const {
        if (trace_.polynomials.contains(name)) {
            return &trace_.polynomials.at(name);
        }
        if (context_.public_polynomials.contains(name)) {
            return &context_.public_polynomials.at(name);
        }
        return nullptr;
    }

    std::optional<std::vector<std::size_t>> rotated_point_shifts(
        const std::shared_ptr<algebra::RootOfUnityDomain>& domain,
        const std::vector<FieldElement>& points) const {
        if (points.size() <= 1) {
            return std::nullopt;
        }
        std::vector<std::size_t> shifts(points.size(), 0);
        for (std::size_t i = 1; i < points.size(); ++i) {
            const auto shift = domain->rotation_shift(points.front(), points[i]);
            if (!shift.has_value()) {
                return std::nullopt;
            }
            shifts[i] = *shift;
        }
        return shifts;
    }

    std::vector<FieldElement> evaluate_lazy_large_fh_public_group(
        const std::vector<std::string>& labels,
        const FieldElement& point) {
        std::vector<FieldElement> out(labels.size(), FieldElement::zero());
        if (labels.empty()) {
            return out;
        }

        const auto& domain = context_.domains.fh;
        const auto& weight_entry = domain_weights(domain, point);
        const std::size_t dataset_rows = context_.dataset.num_nodes;
        const std::size_t local_rows = context_.local.num_nodes;
        const std::size_t feature_count = context_.local.num_features;
        const std::size_t dataset_valid = dataset_rows * feature_count;
        const std::size_t local_valid = local_rows * feature_count;
        const bool full_feature_identity = full_graph_feature_identity_enabled(context_);
        const auto& feature_matrix =
            !context_.dataset.features.empty() ? context_.dataset.features : context_.local.features;

        const auto q_tbl_index = label_index_or_npos(labels, "P_Q_tbl_feat");
        const auto q_qry_index = label_index_or_npos(labels, "P_Q_qry_feat");
        const auto row_tbl_index = label_index_or_npos(labels, "P_Row_feat_tbl");
        const auto col_tbl_index = label_index_or_npos(labels, "P_Col_feat_tbl");
        const auto row_qry_index = label_index_or_npos(labels, "P_Row_feat_qry");
        const auto col_qry_index = label_index_or_npos(labels, "P_Col_feat_qry");
        const auto abs_qry_index = label_index_or_npos(labels, "P_I_feat_qry");
        const auto th_index = label_index_or_npos(labels, "P_T_H");

        auto direct_value = [&](std::string_view label, std::size_t index) -> FieldElement {
            if (label == "P_Q_tbl_feat") {
                return index < dataset_valid ? FieldElement::one() : FieldElement::zero();
            }
            if (label == "P_Q_qry_feat") {
                return index < local_valid ? FieldElement::one() : FieldElement::zero();
            }
            if (label == "P_Row_feat_tbl") {
                return index < dataset_valid ? FieldElement(index / feature_count) : FieldElement::zero();
            }
            if (label == "P_Col_feat_tbl") {
                return index < dataset_valid ? FieldElement(index % feature_count) : FieldElement::zero();
            }
            if (label == "P_Row_feat_qry") {
                return index < local_valid ? FieldElement(index / feature_count) : FieldElement::zero();
            }
            if (label == "P_Col_feat_qry") {
                return index < local_valid ? FieldElement(index % feature_count) : FieldElement::zero();
            }
            if (label == "P_I_feat_qry") {
                return index < local_valid
                    ? FieldElement(context_.local.absolute_ids[index / feature_count])
                    : FieldElement::zero();
            }
            if (label == "P_T_H") {
                if (index >= dataset_valid) {
                    return FieldElement::zero();
                }
                const auto row = index / feature_count;
                const auto column = index % feature_count;
                return feature_matrix[row][column];
            }
            throw std::runtime_error("unsupported lazy FH public label: " + std::string(label));
        };

        if (weight_entry.direct_index.has_value()) {
            const auto direct_index = *weight_entry.direct_index;
            for (std::size_t i = 0; i < labels.size(); ++i) {
                out[i] = direct_value(labels[i], direct_index);
            }
            return out;
        }

        std::vector<FieldElement> column_values(feature_count, FieldElement::zero());
        for (std::size_t column = 0; column < feature_count; ++column) {
            column_values[column] = FieldElement(column);
        }

        std::vector<mcl::Fr> accumulators(labels.size());
        for (auto& accumulator : accumulators) {
            accumulator.clear();
        }

        auto add_scaled = [&](std::size_t label_index, const FieldElement& value, const mcl::Fr& weight) {
            if (label_index >= labels.size()) {
                return;
            }
            mcl::Fr term;
            mcl::Fr::mul(term, value.native(), weight);
            mcl::Fr::add(accumulators[label_index], accumulators[label_index], term);
        };
        auto add_weight = [&](std::size_t label_index, const mcl::Fr& weight) {
            if (label_index >= labels.size()) {
                return;
            }
            mcl::Fr::add(accumulators[label_index], accumulators[label_index], weight);
        };

        std::size_t index = 0;
        for (std::size_t row = 0; row < dataset_rows; ++row) {
            const FieldElement row_value(row);
            const auto& feature_row = feature_matrix[row];
            for (std::size_t column = 0; column < feature_count; ++column, ++index) {
                const auto& weight = weight_entry.native_weights[index];
                add_weight(q_tbl_index, weight);
                add_scaled(row_tbl_index, row_value, weight);
                add_scaled(col_tbl_index, column_values[column], weight);
                add_scaled(th_index, feature_row[column], weight);
                if (full_feature_identity) {
                    add_weight(q_qry_index, weight);
                    add_scaled(row_qry_index, row_value, weight);
                    add_scaled(col_qry_index, column_values[column], weight);
                    add_scaled(abs_qry_index, row_value, weight);
                }
            }
        }

        if (!full_feature_identity) {
            index = 0;
            for (std::size_t row = 0; row < local_rows; ++row) {
                const FieldElement row_value(row);
                const FieldElement absolute_id(context_.local.absolute_ids[row]);
                for (std::size_t column = 0; column < feature_count; ++column, ++index) {
                    const auto& weight = weight_entry.native_weights[index];
                    add_weight(q_qry_index, weight);
                    add_scaled(row_qry_index, row_value, weight);
                    add_scaled(col_qry_index, column_values[column], weight);
                    add_scaled(abs_qry_index, absolute_id, weight);
                }
            }
        }

        for (std::size_t i = 0; i < labels.size(); ++i) {
            out[i] = FieldElement::from_native(accumulators[i]);
        }
        return out;
    }

    std::vector<std::vector<FieldElement>> evaluate_lazy_large_fh_public_group_points(
        const std::vector<std::string>& labels,
        const std::vector<FieldElement>& points) {
        std::vector<std::vector<FieldElement>> out(
            points.size(),
            std::vector<FieldElement>(labels.size(), FieldElement::zero()));
        if (labels.empty() || points.empty()) {
            return out;
        }

        const auto& domain = context_.domains.fh;
        const std::size_t dataset_rows = context_.dataset.num_nodes;
        const std::size_t local_rows = context_.local.num_nodes;
        const std::size_t feature_count = context_.local.num_features;
        const std::size_t dataset_valid = dataset_rows * feature_count;
        const std::size_t local_valid = local_rows * feature_count;
        const bool full_feature_identity = full_graph_feature_identity_enabled(context_);
        const auto& feature_matrix =
            !context_.dataset.features.empty() ? context_.dataset.features : context_.local.features;

        const auto q_tbl_index = label_index_or_npos(labels, "P_Q_tbl_feat");
        const auto q_qry_index = label_index_or_npos(labels, "P_Q_qry_feat");
        const auto row_tbl_index = label_index_or_npos(labels, "P_Row_feat_tbl");
        const auto col_tbl_index = label_index_or_npos(labels, "P_Col_feat_tbl");
        const auto row_qry_index = label_index_or_npos(labels, "P_Row_feat_qry");
        const auto col_qry_index = label_index_or_npos(labels, "P_Col_feat_qry");
        const auto abs_qry_index = label_index_or_npos(labels, "P_I_feat_qry");
        const auto th_index = label_index_or_npos(labels, "P_T_H");

        auto direct_value = [&](std::string_view label, std::size_t index) -> FieldElement {
            if (label == "P_Q_tbl_feat") {
                return index < dataset_valid ? FieldElement::one() : FieldElement::zero();
            }
            if (label == "P_Q_qry_feat") {
                return index < local_valid ? FieldElement::one() : FieldElement::zero();
            }
            if (label == "P_Row_feat_tbl") {
                return index < dataset_valid ? FieldElement(index / feature_count) : FieldElement::zero();
            }
            if (label == "P_Col_feat_tbl") {
                return index < dataset_valid ? FieldElement(index % feature_count) : FieldElement::zero();
            }
            if (label == "P_Row_feat_qry") {
                return index < local_valid ? FieldElement(index / feature_count) : FieldElement::zero();
            }
            if (label == "P_Col_feat_qry") {
                return index < local_valid ? FieldElement(index % feature_count) : FieldElement::zero();
            }
            if (label == "P_I_feat_qry") {
                return index < local_valid
                    ? FieldElement(context_.local.absolute_ids[index / feature_count])
                    : FieldElement::zero();
            }
            if (label == "P_T_H") {
                if (index >= dataset_valid) {
                    return FieldElement::zero();
                }
                const auto row = index / feature_count;
                const auto column = index % feature_count;
                return feature_matrix[row][column];
            }
            throw std::runtime_error("unsupported lazy FH public label: " + std::string(label));
        };

        struct PointWeightRef {
            std::size_t point_index = 0;
            std::size_t direct_index = 0;
            const DomainEvaluationWeights* weight_entry = nullptr;
            bool direct = false;
        };

        std::vector<PointWeightRef> weighted_points;
        weighted_points.reserve(points.size());
        for (std::size_t point_index = 0; point_index < points.size(); ++point_index) {
            const auto& weight_entry = domain_weights(domain, points[point_index]);
            if (weight_entry.direct_index.has_value()) {
                const auto direct_index = *weight_entry.direct_index;
                for (std::size_t label_index = 0; label_index < labels.size(); ++label_index) {
                    out[point_index][label_index] = direct_value(labels[label_index], direct_index);
                }
                continue;
            }
            weighted_points.push_back(PointWeightRef{
                .point_index = point_index,
                .direct_index = 0,
                .weight_entry = nullptr,
                .direct = false,
            });
        }
        if (weighted_points.empty()) {
            return out;
        }
        for (auto& point_ref : weighted_points) {
            point_ref.weight_entry = &domain_weights(domain, points[point_ref.point_index]);
        }

        struct FhAccumulatorSet {
            mcl::Fr q_tbl;
            mcl::Fr row_tbl;
            mcl::Fr col_tbl;
            mcl::Fr th;
            mcl::Fr q_qry;
            mcl::Fr row_qry;
            mcl::Fr col_qry;
            mcl::Fr abs_qry;
        };
        auto clear_accumulators = [](FhAccumulatorSet* accum) {
            accum->q_tbl.clear();
            accum->row_tbl.clear();
            accum->col_tbl.clear();
            accum->th.clear();
            accum->q_qry.clear();
            accum->row_qry.clear();
            accum->col_qry.clear();
            accum->abs_qry.clear();
        };

        const bool need_q_tbl = q_tbl_index < labels.size();
        const bool need_row_tbl = row_tbl_index < labels.size();
        const bool need_col_tbl = col_tbl_index < labels.size();
        const bool need_th = th_index < labels.size();
        const bool need_q_qry = q_qry_index < labels.size();
        const bool need_row_qry = row_qry_index < labels.size();
        const bool need_col_qry = col_qry_index < labels.size();
        const bool need_abs_qry = abs_qry_index < labels.size();

        std::vector<mcl::Fr> column_values(feature_count);
        for (std::size_t column = 0; column < feature_count; ++column) {
            column_values[column] = FieldElement(column).native();
        }

        auto accumulate_dataset_range = [&](std::size_t begin_row, std::size_t end_row) {
            std::vector<FhAccumulatorSet> partial(weighted_points.size());
            for (auto& accum : partial) {
                clear_accumulators(&accum);
            }
            for (std::size_t row = begin_row; row < end_row; ++row) {
                const auto row_native = FieldElement(row).native();
                const auto& feature_row = feature_matrix[row];
                const auto base = row * feature_count;
                for (std::size_t column = 0; column < feature_count; ++column) {
                    const auto index = base + column;
                    for (std::size_t point_offset = 0; point_offset < weighted_points.size(); ++point_offset) {
                        const auto& weight =
                            weighted_points[point_offset].weight_entry->native_weights[index];
                        auto& accum = partial[point_offset];
                        if (need_q_tbl) {
                            mcl::Fr::add(accum.q_tbl, accum.q_tbl, weight);
                        }
                        if (need_row_tbl) {
                            mcl::Fr term;
                            mcl::Fr::mul(term, row_native, weight);
                            mcl::Fr::add(accum.row_tbl, accum.row_tbl, term);
                        }
                        if (need_col_tbl) {
                            mcl::Fr term;
                            mcl::Fr::mul(term, column_values[column], weight);
                            mcl::Fr::add(accum.col_tbl, accum.col_tbl, term);
                        }
                        if (need_th) {
                            mcl::Fr term;
                            mcl::Fr::mul(term, feature_row[column].native(), weight);
                            mcl::Fr::add(accum.th, accum.th, term);
                        }
                        if (full_feature_identity) {
                            continue;
                        }
                    }
                }
            }
            return partial;
        };

        auto combine_accumulators = [&](std::vector<FhAccumulatorSet>* target,
                                        const std::vector<FhAccumulatorSet>& source) {
            for (std::size_t i = 0; i < target->size(); ++i) {
                mcl::Fr::add((*target)[i].q_tbl, (*target)[i].q_tbl, source[i].q_tbl);
                mcl::Fr::add((*target)[i].row_tbl, (*target)[i].row_tbl, source[i].row_tbl);
                mcl::Fr::add((*target)[i].col_tbl, (*target)[i].col_tbl, source[i].col_tbl);
                mcl::Fr::add((*target)[i].th, (*target)[i].th, source[i].th);
                mcl::Fr::add((*target)[i].q_qry, (*target)[i].q_qry, source[i].q_qry);
                mcl::Fr::add((*target)[i].row_qry, (*target)[i].row_qry, source[i].row_qry);
                mcl::Fr::add((*target)[i].col_qry, (*target)[i].col_qry, source[i].col_qry);
                mcl::Fr::add((*target)[i].abs_qry, (*target)[i].abs_qry, source[i].abs_qry);
            }
        };

        const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        std::vector<FhAccumulatorSet> accumulators(weighted_points.size());
        for (auto& accum : accumulators) {
            clear_accumulators(&accum);
        }
        if (dataset_valid >= (1ULL << 20) && cpu_count > 1) {
            const auto task_count =
                std::min<std::size_t>(cpu_count, std::max<std::size_t>(1, dataset_rows / 2048));
            if (task_count > 1) {
                const auto chunk_size = (dataset_rows + task_count - 1) / task_count;
                std::vector<std::future<std::vector<FhAccumulatorSet>>> futures;
                futures.reserve(task_count);
                for (std::size_t task = 0; task < task_count; ++task) {
                    const auto begin_row = task * chunk_size;
                    const auto end_row = std::min(dataset_rows, begin_row + chunk_size);
                    if (begin_row >= end_row) {
                        break;
                    }
                    futures.push_back(std::async(
                        std::launch::async,
                        [&, begin_row, end_row]() {
                            return accumulate_dataset_range(begin_row, end_row);
                        }));
                }
                for (auto& future : futures) {
                    combine_accumulators(&accumulators, future.get());
                }
            } else {
                combine_accumulators(&accumulators, accumulate_dataset_range(0, dataset_rows));
            }
        } else {
            combine_accumulators(&accumulators, accumulate_dataset_range(0, dataset_rows));
        }

        if (!full_feature_identity) {
            auto accumulate_local_range = [&](std::size_t begin_row, std::size_t end_row) {
                std::vector<FhAccumulatorSet> partial(weighted_points.size());
                for (auto& accum : partial) {
                    clear_accumulators(&accum);
                }
                for (std::size_t row = begin_row; row < end_row; ++row) {
                    const auto row_native = FieldElement(row).native();
                    const auto absolute_id_native = FieldElement(context_.local.absolute_ids[row]).native();
                    const auto base = row * feature_count;
                    for (std::size_t column = 0; column < feature_count; ++column) {
                        const auto index = base + column;
                        for (std::size_t point_offset = 0; point_offset < weighted_points.size(); ++point_offset) {
                            const auto& weight =
                                weighted_points[point_offset].weight_entry->native_weights[index];
                            auto& accum = partial[point_offset];
                            if (need_q_qry) {
                                mcl::Fr::add(accum.q_qry, accum.q_qry, weight);
                            }
                            if (need_row_qry) {
                                mcl::Fr term;
                                mcl::Fr::mul(term, row_native, weight);
                                mcl::Fr::add(accum.row_qry, accum.row_qry, term);
                            }
                            if (need_col_qry) {
                                mcl::Fr term;
                                mcl::Fr::mul(term, column_values[column], weight);
                                mcl::Fr::add(accum.col_qry, accum.col_qry, term);
                            }
                            if (need_abs_qry) {
                                mcl::Fr term;
                                mcl::Fr::mul(term, absolute_id_native, weight);
                                mcl::Fr::add(accum.abs_qry, accum.abs_qry, term);
                            }
                        }
                    }
                }
                return partial;
            };
            if (local_valid >= (1ULL << 20) && cpu_count > 1) {
                const auto task_count =
                    std::min<std::size_t>(cpu_count, std::max<std::size_t>(1, local_rows / 2048));
                if (task_count > 1) {
                    const auto chunk_size = (local_rows + task_count - 1) / task_count;
                    std::vector<std::future<std::vector<FhAccumulatorSet>>> futures;
                    futures.reserve(task_count);
                    for (std::size_t task = 0; task < task_count; ++task) {
                        const auto begin_row = task * chunk_size;
                        const auto end_row = std::min(local_rows, begin_row + chunk_size);
                        if (begin_row >= end_row) {
                            break;
                        }
                        futures.push_back(std::async(
                            std::launch::async,
                            [&, begin_row, end_row]() {
                                return accumulate_local_range(begin_row, end_row);
                            }));
                    }
                    for (auto& future : futures) {
                        combine_accumulators(&accumulators, future.get());
                    }
                } else {
                    combine_accumulators(&accumulators, accumulate_local_range(0, local_rows));
                }
            } else {
                combine_accumulators(&accumulators, accumulate_local_range(0, local_rows));
            }
        }

        for (std::size_t point_offset = 0; point_offset < weighted_points.size(); ++point_offset) {
            const auto point_index = weighted_points[point_offset].point_index;
            const auto& accum = accumulators[point_offset];
            if (q_tbl_index < labels.size()) {
                out[point_index][q_tbl_index] = FieldElement::from_native(accum.q_tbl);
            }
            if (row_tbl_index < labels.size()) {
                out[point_index][row_tbl_index] = FieldElement::from_native(accum.row_tbl);
            }
            if (col_tbl_index < labels.size()) {
                out[point_index][col_tbl_index] = FieldElement::from_native(accum.col_tbl);
            }
            if (th_index < labels.size()) {
                out[point_index][th_index] = FieldElement::from_native(accum.th);
            }
            if (full_feature_identity) {
                if (q_qry_index < labels.size()) {
                    out[point_index][q_qry_index] = FieldElement::from_native(accum.q_tbl);
                }
                if (row_qry_index < labels.size()) {
                    out[point_index][row_qry_index] = FieldElement::from_native(accum.row_tbl);
                }
                if (col_qry_index < labels.size()) {
                    out[point_index][col_qry_index] = FieldElement::from_native(accum.col_tbl);
                }
                if (abs_qry_index < labels.size()) {
                    out[point_index][abs_qry_index] = FieldElement::from_native(accum.row_tbl);
                }
                continue;
            }
            if (q_qry_index < labels.size()) {
                out[point_index][q_qry_index] = FieldElement::from_native(accum.q_qry);
            }
            if (row_qry_index < labels.size()) {
                out[point_index][row_qry_index] = FieldElement::from_native(accum.row_qry);
            }
            if (col_qry_index < labels.size()) {
                out[point_index][col_qry_index] = FieldElement::from_native(accum.col_qry);
            }
            if (abs_qry_index < labels.size()) {
                out[point_index][abs_qry_index] = FieldElement::from_native(accum.abs_qry);
            }
        }
        return out;
    }

    std::vector<FieldElement> evaluate_lazy_full_feature_lookup_trace_group(
        const std::vector<std::string>& labels,
        const FieldElement& point) {
        std::vector<FieldElement> out(labels.size(), FieldElement::zero());
        if (labels.empty()) {
            return out;
        }
        const auto eta_feat = challenges_.at("eta_feat");
        const auto eta_feature_value = eta_feat.pow(2);
        const bool full_feature_identity = full_graph_feature_identity_enabled(context_);
        const auto table_index = label_index_or_npos(labels, "P_Table_feat");
        const auto query_index = label_index_or_npos(labels, "P_Query_feat");
        const auto multiplicity_index = label_index_or_npos(labels, "P_m_feat");
        const auto accumulator_index = label_index_or_npos(labels, "P_R_feat");
        const auto cache_suffix = "@" + point_key(point);
        auto load_cached = [&](const std::string& label, FieldElement* value) -> bool {
            const auto it = value_cache_.find(label + cache_suffix);
            if (it == value_cache_.end()) {
                return false;
            }
            *value = it->second;
            return true;
        };

        std::vector<std::string> missing_public_labels;
        auto ensure_public_value = [&](const std::string& label, FieldElement* value) {
            if (load_cached(label, value)) {
                return;
            }
            if (std::find(missing_public_labels.begin(), missing_public_labels.end(), label)
                == missing_public_labels.end()) {
                missing_public_labels.push_back(label);
            }
        };

        FieldElement row_tbl = FieldElement::zero();
        FieldElement col_tbl = FieldElement::zero();
        FieldElement row_qry = FieldElement::zero();
        FieldElement col_qry = FieldElement::zero();
        FieldElement q_qry = FieldElement::zero();
        FieldElement t_h = FieldElement::zero();

        if (table_index < labels.size()) {
            ensure_public_value("P_Row_feat_tbl", &row_tbl);
            ensure_public_value("P_Col_feat_tbl", &col_tbl);
            ensure_public_value("P_T_H", &t_h);
        }
        if (query_index < labels.size()) {
            if (full_feature_identity) {
                ensure_public_value("P_Row_feat_tbl", &row_tbl);
                ensure_public_value("P_Col_feat_tbl", &col_tbl);
                ensure_public_value("P_T_H", &t_h);
            } else {
                ensure_public_value("P_Row_feat_qry", &row_qry);
                ensure_public_value("P_Col_feat_qry", &col_qry);
                ensure_public_value("P_T_H", &t_h);
            }
        }
        if (multiplicity_index < labels.size()) {
            ensure_public_value(full_feature_identity ? "P_Q_tbl_feat" : "P_Q_qry_feat", &q_qry);
        }

        if (!missing_public_labels.empty()) {
            const auto values = evaluate_lazy_large_fh_public_group(missing_public_labels, point);
            for (std::size_t i = 0; i < missing_public_labels.size(); ++i) {
                value_cache_.emplace(missing_public_labels[i] + cache_suffix, values[i]);
            }
            if (table_index < labels.size()) {
                load_cached("P_Row_feat_tbl", &row_tbl);
                load_cached("P_Col_feat_tbl", &col_tbl);
                load_cached("P_T_H", &t_h);
            }
            if (query_index < labels.size()) {
                if (full_feature_identity) {
                    load_cached("P_Row_feat_tbl", &row_tbl);
                    load_cached("P_Col_feat_tbl", &col_tbl);
                    row_qry = row_tbl;
                    col_qry = col_tbl;
                    load_cached("P_T_H", &t_h);
                } else {
                    load_cached("P_Row_feat_qry", &row_qry);
                    load_cached("P_Col_feat_qry", &col_qry);
                    load_cached("P_T_H", &t_h);
                }
            }
            if (multiplicity_index < labels.size()) {
                load_cached(full_feature_identity ? "P_Q_tbl_feat" : "P_Q_qry_feat", &q_qry);
            }
        }
        if (full_feature_identity && query_index < labels.size()) {
            row_qry = row_tbl;
            col_qry = col_tbl;
        }

        if (table_index < labels.size()) {
            out[table_index] = row_tbl + eta_feat * col_tbl + eta_feature_value * t_h;
        }
        if (query_index < labels.size()) {
            out[query_index] = row_qry + eta_feat * col_qry + eta_feature_value * t_h;
        }
        if (multiplicity_index < labels.size()) {
            out[multiplicity_index] = q_qry;
        }
        if (accumulator_index < labels.size()) {
            out[accumulator_index] = FieldElement::zero();
        }
        return out;
    }

  public:
    std::vector<std::vector<FieldElement>> collect_named_values(
        const std::vector<std::string>& labels,
        const std::vector<FieldElement>& points) {
        std::vector<std::vector<FieldElement>> out(
            labels.size(),
            std::vector<FieldElement>(points.size(), FieldElement::zero()));
        if (labels.empty() || points.empty()) {
            return out;
        }

        precompute_named(labels, points);
        for (std::size_t i = 0; i < labels.size(); ++i) {
            for (std::size_t point_index = 0; point_index < points.size(); ++point_index) {
                out[i][point_index] = eval_named(labels[i], points[point_index]);
            }
        }
        return out;
    }

    std::vector<std::vector<FieldElement>> collect_named_values_from_cache(
        const std::vector<std::string>& labels,
        const std::vector<FieldElement>& points) {
        std::vector<std::vector<FieldElement>> out(
            labels.size(),
            std::vector<FieldElement>(points.size(), FieldElement::zero()));
        for (std::size_t i = 0; i < labels.size(); ++i) {
            for (std::size_t point_index = 0; point_index < points.size(); ++point_index) {
                out[i][point_index] = eval_named(labels[i], points[point_index]);
            }
        }
        return out;
    }

    void precompute_named(
        const std::vector<std::string>& labels,
        const std::vector<FieldElement>& points) {
        const bool enable_backend_precompute = util::route2_options().fft_backend_upgrade;

        auto cache_group_values = [&](const std::vector<std::string>& group_labels,
                                      const FieldElement& point,
                                      const std::vector<FieldElement>& values) {
            const auto cache_suffix = "@" + point_key(point);
            for (std::size_t i = 0; i < group_labels.size(); ++i) {
                value_cache_.emplace(group_labels[i] + cache_suffix, values[i]);
            }
        };

        if (lazy_large_fh_public_enabled(context_)) {
            std::vector<std::string> lazy_public_labels;
            lazy_public_labels.reserve(labels.size());
            for (const auto& label : labels) {
                if (is_lazy_large_fh_public_label(label)) {
                    lazy_public_labels.push_back(label);
                }
            }
            if (!lazy_public_labels.empty()) {
                std::vector<std::size_t> missing_point_indices;
                std::vector<std::vector<FieldElement>> cached_values(points.size());
                missing_point_indices.reserve(points.size());
                for (std::size_t point_index = 0; point_index < points.size(); ++point_index) {
                    if (!shared_public_evaluation_cache()->lookup(
                            context_key_,
                            context_.domains.fh,
                            lazy_public_labels,
                            points[point_index],
                            &cached_values[point_index])) {
                        missing_point_indices.push_back(point_index);
                    }
                }
                if (!missing_point_indices.empty()) {
                    std::vector<FieldElement> missing_points;
                    missing_points.reserve(missing_point_indices.size());
                    for (const auto point_index : missing_point_indices) {
                        missing_points.push_back(points[point_index]);
                    }
                    const auto missing_values =
                        missing_points.size() == 1
                        ? std::vector<std::vector<FieldElement>>{
                              evaluate_lazy_large_fh_public_group(lazy_public_labels, missing_points.front())}
                        : evaluate_lazy_large_fh_public_group_points(lazy_public_labels, missing_points);
                    for (std::size_t i = 0; i < missing_point_indices.size(); ++i) {
                        cached_values[missing_point_indices[i]] = missing_values[i];
                        shared_public_evaluation_cache()->store(
                            context_key_,
                            context_.domains.fh,
                            lazy_public_labels,
                            points[missing_point_indices[i]],
                            cached_values[missing_point_indices[i]]);
                    }
                }
                for (std::size_t point_index = 0; point_index < points.size(); ++point_index) {
                    const auto& point = points[point_index];
                    const auto& values = cached_values[point_index];
                    cache_group_values(lazy_public_labels, point, values);
                    if (full_graph_feature_identity_enabled(context_)) {
                        const auto cache_suffix = "@" + point_key(point);
                        auto seed_alias = [&](std::string_view target, std::string_view source) {
                            if (std::find(lazy_public_labels.begin(), lazy_public_labels.end(), target)
                                == lazy_public_labels.end()) {
                                return;
                            }
                            const auto it = std::find(lazy_public_labels.begin(), lazy_public_labels.end(), source);
                            if (it == lazy_public_labels.end()) {
                                return;
                            }
                            const auto source_index = static_cast<std::size_t>(it - lazy_public_labels.begin());
                            value_cache_.emplace(std::string(target) + cache_suffix, values[source_index]);
                        };
                        seed_alias("P_Q_qry_feat", "P_Q_tbl_feat");
                        seed_alias("P_Row_feat_qry", "P_Row_feat_tbl");
                        seed_alias("P_Col_feat_qry", "P_Col_feat_tbl");
                        seed_alias("P_I_feat_qry", "P_Row_feat_tbl");
                    }
                }
            }
        }

        if (lazy_full_feature_lookup_trace_enabled(context_)) {
            std::vector<std::string> lazy_trace_labels;
            lazy_trace_labels.reserve(labels.size());
            for (const auto& label : labels) {
                if (is_lazy_full_feature_lookup_trace_label(label)) {
                    lazy_trace_labels.push_back(label);
                }
            }
            if (!lazy_trace_labels.empty()) {
                std::vector<std::string> required_public_labels;
                required_public_labels.reserve(8);
                auto append_required_public = [&](std::string_view label) {
                    if (std::find(required_public_labels.begin(), required_public_labels.end(), label)
                        == required_public_labels.end()) {
                        required_public_labels.emplace_back(label);
                    }
                };
                if (std::find(lazy_trace_labels.begin(), lazy_trace_labels.end(), "P_Table_feat")
                    != lazy_trace_labels.end()) {
                    append_required_public("P_Row_feat_tbl");
                    append_required_public("P_Col_feat_tbl");
                    append_required_public("P_T_H");
                }
                if (std::find(lazy_trace_labels.begin(), lazy_trace_labels.end(), "P_Query_feat")
                    != lazy_trace_labels.end()) {
                    append_required_public("P_Row_feat_qry");
                    append_required_public("P_Col_feat_qry");
                    append_required_public("P_T_H");
                }
                if (std::find(lazy_trace_labels.begin(), lazy_trace_labels.end(), "P_m_feat")
                    != lazy_trace_labels.end()) {
                    append_required_public("P_Q_qry_feat");
                }
                if (std::find(labels.begin(), labels.end(), "P_H") != labels.end()) {
                    append_required_public("P_T_H");
                }
                if (!required_public_labels.empty() && !lazy_large_fh_public_enabled(context_)) {
                    precompute_named(required_public_labels, points);
                    for (const auto& point : points) {
                        const auto cache_suffix = "@" + point_key(point);
                        for (const auto& label : required_public_labels) {
                            if (value_cache_.find(label + cache_suffix) == value_cache_.end()) {
                                value_cache_.emplace(label + cache_suffix, eval_named(label, point));
                            }
                        }
                        if (std::find(labels.begin(), labels.end(), "P_H") != labels.end()) {
                            if (value_cache_.find("P_H" + cache_suffix) == value_cache_.end()) {
                                value_cache_.emplace("P_H" + cache_suffix, eval_named("P_T_H", point));
                            }
                        }
                    }
                }
                for (const auto& point : points) {
                    cache_group_values(
                        lazy_trace_labels,
                        point,
                        evaluate_lazy_full_feature_lookup_trace_group(lazy_trace_labels, point));
                }
            }
        }

        if (!enable_backend_precompute) {
            return;
        }

        std::unordered_map<std::string, std::pair<std::shared_ptr<algebra::RootOfUnityDomain>, std::vector<std::string>>> groups;
        for (const auto& label : labels) {
            const auto* polynomial = lookup_polynomial(label);
            if (polynomial == nullptr || polynomial->basis != algebra::PolynomialBasis::Evaluation
                || polynomial->domain == nullptr) {
                continue;
            }
            const auto group_key = polynomial->domain->name + ":" + std::to_string(polynomial->domain->size);
            auto& group = groups[group_key];
            if (group.second.empty()) {
                group.first = polynomial->domain;
            }
            group.second.push_back(label);
        }

        auto evaluate_group_at_point = [&](
                                           const std::shared_ptr<algebra::RootOfUnityDomain>& domain,
                                           const algebra::PackedEvaluationBackend& backend,
                                           const std::vector<std::string>& group_labels,
                                           const FieldElement& point,
                                           bool is_public_group) {
            const auto prep_start = Clock::now();
            const auto weight_fetch_start = Clock::now();
            const auto& weight_entry = domain_weights(domain, point);
            const auto weight_fetch_ms = elapsed_ms(weight_fetch_start, Clock::now());
            if (metrics_ != nullptr && domain->name == "FH") {
                metrics_->fh_barycentric_weight_fetch_ms += weight_fetch_ms;
                const auto prep_ms = elapsed_ms(prep_start, Clock::now());
                metrics_->fh_eval_prep_ms += prep_ms;
                metrics_->fh_opening_eval_prep_ms += prep_ms;
            }
            std::vector<FieldElement> values;
            const auto eval_start = Clock::now();
            if (weight_entry.direct_index.has_value()) {
                values = backend.values_at_direct_index(group_labels, *weight_entry.direct_index);
            } else {
                values = backend.evaluate_with_packed_native_weights(
                    group_labels,
                    weight_entry.native_weights,
                    weight_entry.packed_weights);
            }
            if (metrics_ != nullptr && domain->name == "FH") {
                const auto eval_ms = elapsed_ms(eval_start, Clock::now());
                metrics_->fh_interpolation_ms += eval_ms;
                if (is_public_group) {
                    metrics_->fh_public_poly_interp_ms += eval_ms;
                } else {
                    metrics_->fh_lagrange_eval_ms += eval_ms;
                }
            }
            return values;
        };

        for (auto& [group_key, group] : groups) {
            (void)group_key;
            if (backend_registry_ == nullptr) {
                continue;
            }
            const auto* backend = backend_registry_->find(group.first);
            if (backend == nullptr) {
                continue;
            }
            std::vector<std::string> public_labels;
            std::vector<std::string> trace_labels;
            public_labels.reserve(group.second.size());
            trace_labels.reserve(group.second.size());
            for (const auto& label : group.second) {
                if (context_.public_polynomials.contains(label)) {
                    public_labels.push_back(label);
                } else {
                    trace_labels.push_back(label);
                }
            }
            auto store_group_values = [&](const std::vector<std::string>& group_labels,
                                          const FieldElement& point,
                                          const std::vector<FieldElement>& values) {
                const auto copy_start = Clock::now();
                cache_group_values(group_labels, point, values);
                if (metrics_ != nullptr && group.first->name == "FH") {
                    metrics_->fh_copy_convert_ms += elapsed_ms(copy_start, Clock::now());
                }
            };
            // The rotated-point kernel is a backend-level proving optimization
            // for large root-of-unity domains. Small toy domains do not amortize
            // its setup cost, so we intentionally keep the legacy packed sweep
            // there while sending real hot-path bundles through the fused route.
            if (!public_labels.empty() && group.first->name == "FH") {
                for (const auto& point : points) {
                    std::vector<FieldElement> values;
                    const auto reuse_start = Clock::now();
                    const bool cache_hit = shared_public_evaluation_cache()->lookup(
                            context_key_,
                            group.first,
                            public_labels,
                            point,
                            &values);
                    if (cache_hit) {
                        if (metrics_ != nullptr) {
                            metrics_->fh_public_eval_reuse_ms += elapsed_ms(reuse_start, Clock::now());
                        }
                    } else {
                        values = evaluate_group_at_point(group.first, *backend, public_labels, point, true);
                        shared_public_evaluation_cache()->store(
                            context_key_,
                            group.first,
                            public_labels,
                            point,
                            values);
                    }
                    store_group_values(public_labels, point, values);
                }
            }
            const auto& eval_labels = group.first->name == "FH" ? trace_labels : group.second;
            if (eval_labels.empty()) {
                continue;
            }
            const bool allow_shared_trace_reuse =
                group.first->name == "FH" || group.first->name == "edge";
            if (allow_shared_trace_reuse) {
                bool all_cached = true;
                std::vector<std::vector<FieldElement>> cached_values(points.size());
                for (std::size_t point_index = 0; point_index < points.size(); ++point_index) {
                    if (!shared_trace_evaluation_cache()->lookup(
                            context_key_,
                            group.first,
                            eval_labels,
                            points[point_index],
                            &cached_values[point_index])) {
                        all_cached = false;
                        break;
                    }
                }
                if (all_cached) {
                    for (std::size_t point_index = 0; point_index < points.size(); ++point_index) {
                        store_group_values(eval_labels, points[point_index], cached_values[point_index]);
                    }
                    continue;
                }
            }
            if (util::route2_options().fft_kernel_upgrade && group.first->size >= 256) {
                if (const auto shifts = rotated_point_shifts(group.first, points); shifts.has_value()) {
                    const auto prep_start = Clock::now();
                    const auto weight_fetch_start = Clock::now();
                    const auto& weight_entry = domain_weights(group.first, points.front());
                    if (metrics_ != nullptr && group.first->name == "FH") {
                        metrics_->fh_barycentric_weight_fetch_ms += elapsed_ms(weight_fetch_start, Clock::now());
                        const auto prep_ms = elapsed_ms(prep_start, Clock::now());
                        metrics_->fh_eval_prep_ms += prep_ms;
                        metrics_->fh_opening_eval_prep_ms += prep_ms;
                    }
                    if (weight_entry.direct_index.has_value()) {
                        const auto eval_start = Clock::now();
                        const auto domain_mask = group.first->size - 1U;
                        for (std::size_t point_index = 0; point_index < points.size(); ++point_index) {
                            const auto direct_index = (*weight_entry.direct_index + (*shifts)[point_index]) & domain_mask;
                            const auto values = backend->values_at_direct_index(eval_labels, direct_index);
                            store_group_values(eval_labels, points[point_index], values);
                            if (allow_shared_trace_reuse) {
                                shared_trace_evaluation_cache()->store(
                                    context_key_,
                                    group.first,
                                    eval_labels,
                                    points[point_index],
                                    values);
                            }
                        }
                        if (metrics_ != nullptr && group.first->name == "FH") {
                            const auto eval_ms = elapsed_ms(eval_start, Clock::now());
                            metrics_->fh_interpolation_ms += eval_ms;
                            metrics_->fh_lagrange_eval_ms += eval_ms;
                        }
                        continue;
                    }

                    // This Route 2 kernel upgrade only exploits a root-of-unity
                    // identity inside the proving backend: for the same domain,
                    // evaluations at {z, omega z, ...} correspond to cyclic
                    // shifts of one representative barycentric weight vector.
                    // We therefore fuse these rotated points into one packed
                    // sweep without changing which polynomials are opened, at
                    // which points, or what values enter the proof.
                    const auto eval_start = Clock::now();
                    const auto batched_values =
                        backend->evaluate_with_packed_native_weight_rotations(
                            eval_labels,
                            weight_entry.native_weights,
                            weight_entry.packed_weights,
                            *shifts);
                    if (metrics_ != nullptr && group.first->name == "FH") {
                        const auto eval_ms = elapsed_ms(eval_start, Clock::now());
                        metrics_->fh_interpolation_ms += eval_ms;
                        metrics_->fh_lagrange_eval_ms += eval_ms;
                    }
                    for (std::size_t point_index = 0; point_index < points.size(); ++point_index) {
                        store_group_values(eval_labels, points[point_index], batched_values[point_index]);
                        if (allow_shared_trace_reuse) {
                            shared_trace_evaluation_cache()->store(
                                context_key_,
                                group.first,
                                eval_labels,
                                points[point_index],
                                batched_values[point_index]);
                        }
                    }
                    continue;
                }
            }
            for (const auto& point : points) {
                const auto values = evaluate_group_at_point(group.first, *backend, eval_labels, point, false);
                store_group_values(eval_labels, point, values);
                if (allow_shared_trace_reuse) {
                    shared_trace_evaluation_cache()->store(
                        context_key_,
                        group.first,
                        eval_labels,
                        point,
                        values);
                }
            }
        }
    }

  private:
    FieldElement eval_polynomial(const Polynomial& polynomial, const FieldElement& point) {
        if (polynomial.basis == algebra::PolynomialBasis::Coefficient) {
            return polynomial.evaluate(point);
        }
        if (polynomial.domain == nullptr) {
            throw std::runtime_error("evaluation polynomial is missing its domain");
        }

        const auto& weight_entry = domain_weights(polynomial.domain, point);
        const auto& values = polynomial.values();
        if (weight_entry.direct_index.has_value()) {
            return values.at(*weight_entry.direct_index);
        }
        // The route2 parallel FFT flag only changes how the cached domain
        // weights are reduced. The opened values themselves are identical.
        return algebra::dot_product_packed_native_weights(
            values,
            weight_entry.native_weights,
            weight_entry.packed_weights);
    }

    const DomainEvaluationWeights& domain_weights(
        const std::shared_ptr<algebra::RootOfUnityDomain>& domain,
        const FieldElement& point) {
        return domain_weight_cache_->get(domain, point);
    }

    const ProtocolContext& context_;
    const TraceArtifacts& trace_;
    const std::map<std::string, FieldElement>& challenges_;
    std::shared_ptr<const ProofEvaluationBackendRegistry> backend_registry_;
    std::shared_ptr<ProofDomainWeightCache> domain_weight_cache_;
    std::string context_key_;
    RunMetrics* metrics_ = nullptr;
    std::unordered_map<std::string, FieldElement> value_cache_;
};

std::unordered_map<std::string, Commitment> batch_quotient_commitments(
    const std::vector<std::pair<std::string, FieldElement>>& named_tau_values,
    const crypto::KZGKeyPair& key) {
    std::unordered_map<std::string, Commitment> out;
    out.reserve(named_tau_values.size());
    if (named_tau_values.empty()) {
        return out;
    }

    std::vector<FieldElement> tau_values;
    tau_values.reserve(named_tau_values.size());
    for (const auto& [name, tau_value] : named_tau_values) {
        (void)name;
        tau_values.push_back(tau_value);
    }

    const auto points = crypto::g1_mul_same_base_batch(key.g1_generator, tau_values);
    for (std::size_t i = 0; i < named_tau_values.size(); ++i) {
        out.emplace(
            named_tau_values[i].first,
            Commitment{
                .name = named_tau_values[i].first,
                .point = points[i],
                .tau_evaluation = tau_values[i],
            });
    }
    return out;
}

std::vector<Commitment> collect_commitments(
    const TraceArtifacts& trace,
    const std::unordered_map<std::string, Commitment>& quotient_commitments,
    const std::vector<std::string>& labels) {
    std::vector<Commitment> out;
    out.reserve(labels.size());
    for (const auto& label : labels) {
        if (trace.commitments.contains(label)) {
            out.push_back(trace.commitments.at(label));
        } else {
            out.push_back(quotient_commitments.at(label));
        }
    }
    return out;
}

const std::vector<std::string>& quotient_dependencies_fh() {
    static const std::vector<std::string> labels = {
        "P_H",
        "P_R_feat",
        "P_Table_feat",
        "P_Query_feat",
        "P_T_H",
        "P_Row_feat_tbl",
        "P_Col_feat_tbl",
        "P_Row_feat_qry",
        "P_Col_feat_qry",
        "P_I_feat_qry",
        "P_Q_tbl_feat",
        "P_m_feat",
        "P_Q_qry_feat",
    };
    return labels;
}

void append_unique_labels(
    std::vector<std::string>* target,
    const std::vector<std::string>& labels) {
    for (const auto& label : labels) {
        if (std::find(target->begin(), target->end(), label) == target->end()) {
            target->push_back(label);
        }
    }
}

std::vector<FieldElement> fh_dependency_points(
    const std::shared_ptr<algebra::RootOfUnityDomain>& domain,
    const std::vector<FieldElement>& points) {
    std::vector<FieldElement> out = points;
    out.reserve(points.size() * 2);
    for (const auto& point : points) {
        const auto rotated = point * domain->omega;
        if (std::find(out.begin(), out.end(), rotated) == out.end()) {
            out.push_back(rotated);
        }
    }
    return out;
}

const std::vector<std::string>& quotient_dependencies_edge() {
    static const std::vector<std::string> labels = {
        "P_R_src",
        "P_Query_src",
        "P_Q_qry_src",
        "P_E_src_edge",
        "P_H_src_star_edge",
        "P_R_dst",
        "P_Query_dst",
        "P_Q_qry_dst",
        "P_E_dst_edge",
        "P_M_edge",
        "P_Sum_edge",
        "P_inv_edge",
        "P_H_agg_star_edge",
        "P_R_L",
        "P_Table_L",
        "P_Query_L",
        "P_Q_tbl_L",
        "P_m_L",
        "P_Q_qry_L",
        "P_R_R",
        "P_Table_R",
        "P_Query_R",
        "P_Q_tbl_R",
        "P_m_R",
        "P_Q_qry_R",
        "P_R_exp",
        "P_Table_exp",
        "P_Query_exp",
        "P_Q_tbl_exp",
        "P_m_exp",
        "P_Q_qry_exp",
        "P_M_edge",
        "P_Delta",
        "P_Z",
        "P_s_max",
        "P_C_max",
        "P_Q_edge_valid",
        "P_Q_new_edge",
        "P_Q_end_edge",
        "P_alpha",
        "P_U",
        "P_inv_edge",
        "P_v_hat",
        "P_PSQ",
        "P_w_psq",
        "P_T_psq_edge",
        "P_T_L_x",
        "P_T_L_y",
        "P_T_range",
        "P_T_exp_x",
        "P_T_exp_y",
        "P_src",
        "P_dst",
        "P_S",
    };
    return labels;
}

std::vector<std::string> multihead_edge_dependencies(const ProtocolContext& context) {
    std::vector<std::string> labels = domain_opening_labels(context, "edge");
    append_unique_labels(
        &labels,
        {
            "P_Q_edge_valid",
            "P_Q_new_edge",
            "P_Q_end_edge",
            "P_Q_tbl_L",
            "P_Q_qry_L",
            "P_Q_tbl_R",
            "P_Q_qry_R",
            "P_Q_tbl_exp",
            "P_Q_qry_exp",
            "P_Q_tbl_ELU",
            "P_Q_qry_ELU",
            "P_T_L_x",
            "P_T_L_y",
            "P_T_range",
            "P_T_exp_x",
            "P_T_exp_y",
            "P_T_ELU_x",
            "P_T_ELU_y",
            "P_src",
            "P_dst",
        });
    return labels;
}

std::vector<std::string> multihead_in_dependencies(const ProtocolContext& context) {
    std::vector<std::string> labels = domain_opening_labels(context, "in");
    append_unique_labels(&labels, {"P_Q_proj_valid"});
    return labels;
}

std::vector<std::string> multihead_d_dependencies(const ProtocolContext& context) {
    std::vector<std::string> labels = domain_opening_labels(context, "d");
    append_unique_labels(&labels, {"P_Q_d_valid"});
    return labels;
}

std::vector<std::string> multihead_cat_dependencies(const ProtocolContext& context) {
    std::vector<std::string> labels = domain_opening_labels(context, "cat");
    append_unique_labels(&labels, {"P_Q_cat_valid"});
    return labels;
}

std::vector<std::string> multihead_c_dependencies(const ProtocolContext& context) {
    std::vector<std::string> labels = domain_opening_labels(context, "C");
    append_unique_labels(&labels, {"P_Q_C_valid"});
    return labels;
}

std::vector<std::string> multihead_n_dependencies(const ProtocolContext& context) {
    std::vector<std::string> labels = domain_opening_labels(context, "N");
    append_unique_labels(&labels, {"P_I", "P_Q_N"});
    return labels;
}

const std::vector<std::string>& quotient_dependencies_in() {
    static const std::vector<std::string> labels = {
        "P_Q_proj_valid",
        "P_a_proj",
        "P_b_proj",
        "P_Acc_proj",
    };
    return labels;
}

const std::vector<std::string>& quotient_dependencies_d() {
    static const std::vector<std::string> labels = {
        "P_Q_d_valid",
        "P_a_src",
        "P_b_src",
        "P_Acc_src",
        "P_a_dst",
        "P_b_dst",
        "P_Acc_dst",
        "P_a_star",
        "P_b_star",
        "P_Acc_star",
        "P_a_agg",
        "P_b_agg",
        "P_Acc_agg",
        "P_a_out",
        "P_b_out",
        "P_Acc_out",
    };
    return labels;
}

const std::vector<std::string>& quotient_dependencies_n() {
    static const std::vector<std::string> labels = {
        "P_I",
        "P_Q_N",
        "P_E_src",
        "P_H_star",
        "P_Table_src",
        "P_m_src",
        "P_R_src_node",
        "P_E_dst",
        "P_M",
        "P_Sum",
        "P_inv",
        "P_H_agg_star",
        "P_Table_dst",
        "P_m_dst",
        "P_R_dst_node",
    };
    return labels;
}

std::vector<std::string> quotient_dependency_labels(
    const ProtocolContext& context,
    const std::string& quotient_name) {
    if (!context.model.has_real_multihead) {
        if (quotient_name == "t_FH") return quotient_dependencies_fh();
        if (quotient_name == "t_edge") return quotient_dependencies_edge();
        if (quotient_name == "t_in") return quotient_dependencies_in();
        if (quotient_name == "t_d" || quotient_name == "t_d_h") return quotient_dependencies_d();
        if (quotient_name == "t_N") return quotient_dependencies_n();
        return {};
    }

    if (quotient_name == "t_FH") return quotient_dependencies_fh();
    if (quotient_name == "t_edge") return multihead_edge_dependencies(context);
    if (quotient_name == "t_in") return multihead_in_dependencies(context);
    if (quotient_name == "t_d_h" || quotient_name == "t_d") return multihead_d_dependencies(context);
    if (quotient_name == "t_cat") return multihead_cat_dependencies(context);
    if (quotient_name == "t_C") return multihead_c_dependencies(context);
    if (quotient_name == "t_N") return multihead_n_dependencies(context);
    return {};
}

DomainOpeningBundle make_bundle(
    const ProtocolContext& context,
    const TraceArtifacts& trace,
    EvaluationMemoization& memo,
    const std::unordered_map<std::string, Commitment>& quotient_commitments,
    const std::vector<std::string>& labels,
    const std::vector<FieldElement>& points,
    const std::string& folding_name,
    const std::map<std::string, FieldElement>& challenges,
    RunMetrics* metrics = nullptr,
    double* eval_gather_ms = nullptr,
    double* witness_open_ms = nullptr) {
    DomainOpeningBundle bundle;
    bundle.points = points;
    const auto commitments = collect_commitments(trace, quotient_commitments, labels);
    const auto gather_start = Clock::now();
    std::string quotient_name;
    for (const auto& label : labels) {
        if (!label.empty() && label[0] == 't') {
            quotient_name = label;
            break;
        }
    }
    const bool is_fh_bundle = quotient_name == "t_FH";
    std::vector<std::vector<FieldElement>> values;
    if (!quotient_name.empty()) {
        const auto dependency_labels = quotient_dependency_labels(context, quotient_name);
        std::shared_ptr<algebra::RootOfUnityDomain> domain;
        if (quotient_name == "t_FH") {
            domain = context.domains.fh;
        } else if (quotient_name == "t_edge") {
            domain = context.domains.edge;
        } else if (quotient_name == "t_in") {
            domain = context.domains.in;
        } else if (quotient_name == "t_d" || quotient_name == "t_d_h") {
            domain = context.domains.d;
        } else if (quotient_name == "t_cat") {
            domain = context.domains.cat;
        } else if (quotient_name == "t_C") {
            domain = context.domains.c;
        } else if (quotient_name == "t_N") {
            domain = context.domains.n;
        }
        if (domain != nullptr && !dependency_labels.empty()) {
            memo.precompute_named(dependency_labels, fh_dependency_points(domain, points));
        }
        values = memo.collect_named_values_from_cache(labels, points);
    } else {
        values = memo.collect_named_values(labels, points);
    }
    if (eval_gather_ms != nullptr) {
        *eval_gather_ms += elapsed_ms(gather_start, Clock::now());
    }
    if (metrics != nullptr && is_fh_bundle) {
        metrics->fh_open_gather_ms += elapsed_ms(gather_start, Clock::now());
    }
    for (std::size_t i = 0; i < labels.size(); ++i) {
        bundle.values.push_back({labels[i], values[i]});
    }
    const auto witness_start = Clock::now();
    crypto::BatchOpeningProfile batch_profile;
    bundle.witness =
        crypto::KZG::open_batch(commitments, points, values, challenges.at(folding_name), context.kzg, &batch_profile);
    if (witness_open_ms != nullptr) {
        *witness_open_ms += elapsed_ms(witness_start, Clock::now());
    }
    if (metrics != nullptr && is_fh_bundle) {
        const auto witness_total_ms = elapsed_ms(witness_start, Clock::now());
        metrics->fh_open_witness_ms += std::max(0.0, witness_total_ms - batch_profile.precompute_ms);
        metrics->fh_open_fold_prepare_ms += batch_profile.precompute_ms;
        metrics->fh_fold_prep_ms += batch_profile.precompute_ms;
    }
    return bundle;
}

}  // namespace

ProtocolContext build_context(const util::AppConfig& config, RunMetrics* metrics) {
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, ProtocolContext> cache;
    const bool persist_context_cache =
        config.dataset != "ogbn_arxiv" && config.dataset != "ogbn-arxiv";

    const auto cache_lookup_start = Clock::now();
    const auto cache_key = context_cache_key(config);
    if (persist_context_cache) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (const auto it = cache.find(cache_key); it != cache.end()) {
            // This cache only stores static protocol context material: domains,
            // SRS, public/static polynomials, static commitments and model data.
            // Instance-specific witnesses and challenge-derived values are not
            // shared across proofs.
            auto context = it->second;
            context.config = config;
            if (metrics != nullptr) {
                metrics->load_static_ms += elapsed_ms(cache_lookup_start, Clock::now());
                append_note(metrics, "static_context_cache=hit");
            }
            return context;
        }
    }

    append_note(metrics, "static_context_cache=miss");
    ProtocolContext context;
    context.config = config;

    auto start = Clock::now();
    context.dataset = data::load_dataset(config);
    context.local = data::normalize_graph_input(context.dataset, config);
    if (config.allow_synthetic_model) {
        append_note(metrics, "model_source=synthetic_debug_only");
        context.model = model::build_family_model_parameters(
            config.d_in_profile.empty() ? std::vector<std::size_t>{context.local.num_features} : config.d_in_profile,
            [&]() {
                std::vector<model::HiddenLayerShape> shapes;
                shapes.reserve(config.hidden_profile.size());
                for (const auto& layer : config.hidden_profile) {
                    shapes.push_back({layer.head_count, layer.head_dim});
                }
                return shapes;
            }(),
            config.K_out,
            context.local.num_classes,
            config.seed);
    } else {
        if (config.checkpoint_bundle.empty()) {
            throw std::runtime_error(
                "formal model route requires checkpoint_bundle; refusing to fall back to synthetic parameters");
        }
        const auto checkpoint_root = resolve_project_path(config.project_root, config.checkpoint_bundle);
        context.model = model::load_checkpoint_bundle_parameters(checkpoint_root.string());
        append_note(metrics, "model_source=checkpoint_bundle");
        append_note(metrics, "hidden_head_count=" + std::to_string(model::flattened_hidden_head_count(context.model)));
    }
    if (context.model.has_real_multihead) {
        std::string family_reason;
        if (!model::hidden_family_dimension_chain_is_valid(context.model, &family_reason)) {
            throw std::runtime_error("formal family shape is invalid: " + family_reason);
        }
        bool hidden_profile_matches = context.model.hidden_profile.size() == config.hidden_profile.size();
        for (std::size_t i = 0; hidden_profile_matches && i < config.hidden_profile.size(); ++i) {
            hidden_profile_matches =
                context.model.hidden_profile[i].head_count == config.hidden_profile[i].head_count
                && context.model.hidden_profile[i].head_dim == config.hidden_profile[i].head_dim;
        }
        if (context.model.L != config.layer_count) {
            throw std::runtime_error(
                "config L=" + std::to_string(config.layer_count)
                + " conflicts with model L=" + std::to_string(context.model.L));
        }
        if (!hidden_profile_matches) {
            throw std::runtime_error("config hidden_profile conflicts with checkpoint manifest hidden_profile");
        }
        if (!config.d_in_profile.empty() && context.model.d_in_profile != config.d_in_profile) {
            throw std::runtime_error("config d_in_profile conflicts with checkpoint manifest d_in_profile");
        }
        if (context.model.K_out != config.K_out) {
            throw std::runtime_error(
                "config K_out=" + std::to_string(config.K_out)
                + " conflicts with model K_out=" + std::to_string(context.model.K_out));
        }
        if (context.model.C != config.num_classes) {
            throw std::runtime_error(
                "config num_classes=" + std::to_string(config.num_classes)
                + " conflicts with model C=" + std::to_string(context.model.C));
        }
    }
    auto end = Clock::now();
    if (metrics != nullptr) {
        metrics->load_static_ms += elapsed_ms(start, end);
    }

    start = Clock::now();
    context.kzg = crypto::KZG::setup(config.seed);
    end = Clock::now();
    if (metrics != nullptr) {
        metrics->srs_prepare_ms += elapsed_ms(start, end);
    }

    start = Clock::now();
    const std::size_t fh_size = algebra::next_power_of_two(
        std::max(context.dataset.num_nodes * context.dataset.num_features, context.local.num_nodes * context.local.num_features) + 2);
    const std::size_t configured_range_size = (1ULL << config.range_bits);
    const std::size_t d_h = hidden_head_width(context.model, config);
    const std::size_t d_cat = concat_width(context.model, config);
    const std::size_t max_in = context.model.has_real_multihead
        ? std::max<std::size_t>(context.local.num_features, model::max_hidden_input_dim(context.model))
        : context.local.num_features;
    const bool large_whole_graph_gat =
        (config.dataset == "ogbn_arxiv" || config.dataset == "ogbn-arxiv")
        && config.batching_rule == "whole_graph_single";
    const std::size_t range_size = std::max<std::size_t>(
        configured_range_size,
        large_whole_graph_gat ? (1ULL << 19) : configured_range_size);
    const std::size_t lrelu_bound = std::max<std::size_t>(large_whole_graph_gat ? (1ULL << 20) : 4096, range_size);
    const std::size_t elu_bound = lrelu_bound;
    const std::size_t lrelu_table_size = lrelu_bound * 2 + 1;
    const std::size_t elu_table_size = (elu_bound * 2 + 1) * (2 * elu_table_band() + 1);
    const std::size_t exp_table_size = range_size * (2 * exp_table_band() + 1);
    const std::size_t edge_size = algebra::next_power_of_two(
        std::max<std::size_t>({
            context.local.num_nodes,
            context.local.edges.size(),
            context.local.num_nodes * d_h,
            lrelu_table_size,
            elu_table_size,
            exp_table_size,
            range_size,
        }) + 2);
    const std::size_t in_size = algebra::next_power_of_two(max_in + 2);
    const std::size_t d_size = algebra::next_power_of_two(d_h + 2);
    const std::size_t cat_size = algebra::next_power_of_two(d_cat + 2);
    const std::size_t c_size = algebra::next_power_of_two(context.local.num_classes + 2);
    const std::size_t n_size = algebra::next_power_of_two(context.local.num_nodes + 2);

    context.domains = WorkDomains{
        .fh = algebra::RootOfUnityDomain::create("FH", fh_size),
        .edge = algebra::RootOfUnityDomain::create("edge", edge_size),
        .in = algebra::RootOfUnityDomain::create("in", in_size),
        .d = algebra::RootOfUnityDomain::create("d", d_size),
        .cat = algebra::RootOfUnityDomain::create("cat", cat_size),
        .c = algebra::RootOfUnityDomain::create("C", c_size),
        .n = algebra::RootOfUnityDomain::create("N", n_size),
        .hidden_heads = {},
        .output_head = std::nullopt,
    };
    if (context.model.has_real_multihead) {
        context.domains.hidden_heads.assign(
            context.model.hidden_heads.size(),
            AttentionHeadDomains{
                .in = context.domains.in,
                .d = context.domains.d,
            });
        context.domains.output_head = AttentionHeadDomains{.in = context.domains.cat, .d = context.domains.c};
    }
    end = Clock::now();
    if (metrics != nullptr) {
        metrics->fft_plan_ms += elapsed_ms(start, end);
    }

    start = Clock::now();
    context.tables.range = make_range_table(range_size);
    context.tables.exp = make_exp_table(range_size);
    context.tables.lrelu = make_lrelu_table(lrelu_bound);
    context.tables.elu = make_elu_table(elu_bound);

    const auto edge_valid = build_selector(context.local.edges.size(), edge_size);

    add_public_poly(
        context,
        "P_I",
        make_eval_poly("P_I", ids_to_field(context.local.absolute_ids, n_size), context.domains.n));
    add_public_poly(
        context,
        "P_src",
        make_eval_poly("P_src", edge_component(context.local.edges, edge_size, true), context.domains.edge));
    add_public_poly(
        context,
        "P_dst",
        make_eval_poly("P_dst", edge_component(context.local.edges, edge_size, false), context.domains.edge));
    add_public_poly(
        context,
        "P_Q_new_edge",
        make_eval_poly("P_Q_new_edge", q_new_selector(context.local.edges, edge_size), context.domains.edge));
    add_public_poly(
        context,
        "P_Q_end_edge",
        make_eval_poly("P_Q_end_edge", q_end_selector(context.local.edges, edge_size), context.domains.edge));
    add_public_poly(
        context,
        "P_Q_edge_valid",
        make_eval_poly("P_Q_edge_valid", edge_valid, context.domains.edge));
    add_public_poly(
        context,
        "P_Q_N",
        make_eval_poly("P_Q_N", build_selector(context.local.num_nodes, n_size), context.domains.n));
    add_public_poly(
        context,
        "P_Q_proj_valid",
        make_eval_poly("P_Q_proj_valid", build_selector(max_in, in_size), context.domains.in));
    add_public_poly(
        context,
        "P_Q_d_valid",
        make_eval_poly("P_Q_d_valid", build_selector(d_h, d_size), context.domains.d));
    add_public_poly(
        context,
        "P_Q_cat_valid",
        make_eval_poly("P_Q_cat_valid", build_selector(d_cat, cat_size), context.domains.cat));
    add_public_poly(
        context,
        "P_Q_C_valid",
        make_eval_poly("P_Q_C_valid", build_selector(context.local.num_classes, c_size), context.domains.c));
    if (lazy_large_fh_public_enabled(context)) {
        add_public_tau_commitment(
            context,
            "P_Q_tbl_feat",
            evaluate_lazy_large_fh_public_poly(context, "P_Q_tbl_feat", context.kzg.tau));
        add_public_tau_commitment(
            context,
            "P_Q_qry_feat",
            evaluate_lazy_large_fh_public_poly(context, "P_Q_qry_feat", context.kzg.tau));
        add_public_tau_commitment(
            context,
            "P_Row_feat_tbl",
            evaluate_lazy_large_fh_public_poly(context, "P_Row_feat_tbl", context.kzg.tau));
        add_public_tau_commitment(
            context,
            "P_Col_feat_tbl",
            evaluate_lazy_large_fh_public_poly(context, "P_Col_feat_tbl", context.kzg.tau));
        add_public_tau_commitment(
            context,
            "P_Row_feat_qry",
            evaluate_lazy_large_fh_public_poly(context, "P_Row_feat_qry", context.kzg.tau));
        add_public_tau_commitment(
            context,
            "P_Col_feat_qry",
            evaluate_lazy_large_fh_public_poly(context, "P_Col_feat_qry", context.kzg.tau));
        add_public_tau_commitment(
            context,
            "P_I_feat_qry",
            evaluate_lazy_large_fh_public_poly(context, "P_I_feat_qry", context.kzg.tau));
    } else {
        add_public_poly(
            context,
            "P_Q_tbl_feat",
            make_eval_poly("P_Q_tbl_feat", build_selector(context.dataset.num_nodes * context.dataset.num_features, fh_size), context.domains.fh));
        add_public_poly(
            context,
            "P_Q_qry_feat",
            make_eval_poly("P_Q_qry_feat", build_selector(context.local.num_nodes * context.local.num_features, fh_size), context.domains.fh));
        add_public_poly(
            context,
            "P_Row_feat_tbl",
            make_eval_poly(
                "P_Row_feat_tbl",
                feature_table_row_indices(context.dataset.num_nodes, context.dataset.num_features, fh_size),
                context.domains.fh));
        add_public_poly(
            context,
            "P_Col_feat_tbl",
            make_eval_poly(
                "P_Col_feat_tbl",
                feature_table_col_indices(context.dataset.num_nodes, context.dataset.num_features, fh_size),
                context.domains.fh));
        add_public_poly(
            context,
            "P_Row_feat_qry",
            make_eval_poly(
                "P_Row_feat_qry",
                feature_query_row_indices(context.local.num_nodes, context.local.num_features, fh_size),
                context.domains.fh));
        add_public_poly(
            context,
            "P_Col_feat_qry",
            make_eval_poly(
                "P_Col_feat_qry",
                feature_query_col_indices(context.local.num_nodes, context.local.num_features, fh_size),
                context.domains.fh));
        add_public_poly(
            context,
            "P_I_feat_qry",
            make_eval_poly(
                "P_I_feat_qry",
                feature_query_absolute_ids(context.local.absolute_ids, context.local.num_features, fh_size),
                context.domains.fh));
    }
    add_public_poly(
        context,
        "P_Q_qry_src",
        make_eval_poly("P_Q_qry_src", edge_valid, context.domains.edge));
    add_public_poly(
        context,
        "P_Q_qry_dst",
        make_eval_poly("P_Q_qry_dst", edge_valid, context.domains.edge));
    add_public_poly(
        context,
        "P_Q_tbl_L",
        make_eval_poly("P_Q_tbl_L", build_selector(context.tables.lrelu.size(), edge_size), context.domains.edge));
    add_public_poly(
        context,
        "P_Q_qry_L",
        make_eval_poly("P_Q_qry_L", build_selector(context.local.edges.size(), edge_size), context.domains.edge));
    add_public_poly(
        context,
        "P_Q_tbl_R",
        make_eval_poly("P_Q_tbl_R", build_selector(context.tables.range.size(), edge_size), context.domains.edge));
    add_public_poly(
        context,
        "P_Q_qry_R",
        make_eval_poly("P_Q_qry_R", build_selector(context.local.edges.size(), edge_size), context.domains.edge));
    add_public_poly(
        context,
        "P_Q_tbl_exp",
        make_eval_poly("P_Q_tbl_exp", build_selector(context.tables.exp.size(), edge_size), context.domains.edge));
    add_public_poly(
        context,
        "P_Q_qry_exp",
        make_eval_poly("P_Q_qry_exp", build_selector(context.local.edges.size(), edge_size), context.domains.edge));
    add_public_poly(
        context,
        "P_Q_tbl_ELU",
        make_eval_poly("P_Q_tbl_ELU", build_selector(context.tables.elu.size(), edge_size), context.domains.edge));
    add_public_poly(
        context,
        "P_Q_qry_ELU",
        make_eval_poly("P_Q_qry_ELU", build_selector(context.local.num_nodes * d_h, edge_size), context.domains.edge));

    const auto& feature_matrix_for_h =
        !context.dataset.features.empty() ? context.dataset.features : context.local.features;
    if (lazy_large_fh_public_enabled(context)) {
        const auto tau_row_stride = context.kzg.tau.pow(static_cast<std::uint64_t>(context.local.num_features));
        add_static_tau_commitment(
            context,
            "V_T_H",
            matrix_row_major_evaluation_with_row_stride(
                feature_matrix_for_h,
                context.kzg.tau,
                tau_row_stride));
        add_public_tau_commitment(
            context,
            "P_T_H",
            evaluate_lazy_large_fh_public_poly(context, "P_T_H", context.kzg.tau));
    } else {
        add_static_commitment(
            context,
            "V_T_H",
            make_coeff_poly("V_T_H", algebra::flatten_matrix_coefficients(feature_matrix_for_h)));
        add_public_poly(
            context,
            "P_T_H",
            make_eval_poly(
                "P_T_H",
                padded(algebra::flatten_matrix_coefficients(feature_matrix_for_h), fh_size),
                context.domains.fh));
    }

    std::vector<FieldElement> l_x(edge_size, FieldElement::zero());
    std::vector<FieldElement> l_y(edge_size, FieldElement::zero());
    for (std::size_t i = 0; i < context.tables.lrelu.size(); ++i) {
        l_x[i] = context.tables.lrelu[i].first;
        l_y[i] = context.tables.lrelu[i].second;
    }
    add_public_poly(context, "P_T_L_x", make_eval_poly("P_T_L_x", l_x, context.domains.edge));
    add_public_poly(context, "P_T_L_y", make_eval_poly("P_T_L_y", l_y, context.domains.edge));
    add_static_commitment(context, "V_T_L_x", make_eval_poly("V_T_L_x", l_x, context.domains.edge));
    add_static_commitment(context, "V_T_L_y", make_eval_poly("V_T_L_y", l_y, context.domains.edge));

    std::vector<FieldElement> elu_x(edge_size, FieldElement::zero());
    std::vector<FieldElement> elu_y(edge_size, FieldElement::zero());
    for (std::size_t i = 0; i < context.tables.elu.size(); ++i) {
        elu_x[i] = context.tables.elu[i].first;
        elu_y[i] = context.tables.elu[i].second;
    }
    add_public_poly(context, "P_T_ELU_x", make_eval_poly("P_T_ELU_x", elu_x, context.domains.edge));
    add_public_poly(context, "P_T_ELU_y", make_eval_poly("P_T_ELU_y", elu_y, context.domains.edge));
    add_static_commitment(context, "V_T_ELU_x", make_eval_poly("V_T_ELU_x", elu_x, context.domains.edge));
    add_static_commitment(context, "V_T_ELU_y", make_eval_poly("V_T_ELU_y", elu_y, context.domains.edge));

    std::vector<FieldElement> exp_x(edge_size, FieldElement::zero());
    std::vector<FieldElement> exp_y(edge_size, FieldElement::zero());
    for (std::size_t i = 0; i < context.tables.exp.size(); ++i) {
        exp_x[i] = context.tables.exp[i].first;
        exp_y[i] = context.tables.exp[i].second;
    }
    add_public_poly(context, "P_T_exp_x", make_eval_poly("P_T_exp_x", exp_x, context.domains.edge));
    add_public_poly(context, "P_T_exp_y", make_eval_poly("P_T_exp_y", exp_y, context.domains.edge));
    add_static_commitment(context, "V_T_exp_x", make_eval_poly("V_T_exp_x", exp_x, context.domains.edge));
    add_static_commitment(context, "V_T_exp_y", make_eval_poly("V_T_exp_y", exp_y, context.domains.edge));
    add_public_poly(
        context,
        "P_T_range",
        make_eval_poly("P_T_range", padded(context.tables.range, edge_size), context.domains.edge));
    add_static_commitment(
        context,
        "V_T_range",
        make_eval_poly("V_T_range", padded(context.tables.range, edge_size), context.domains.edge));

    if (context.model.has_real_multihead) {
        for (std::size_t head_index = 0; head_index < context.model.hidden_heads.size(); ++head_index) {
            add_static_commitment(
                context,
                hidden_weight_label(head_index),
                make_coeff_poly(
                    hidden_weight_label(head_index),
                    algebra::flatten_matrix_coefficients(quantize_model_matrix(context.model.hidden_heads[head_index].seq_kernel_fp))));
            add_static_commitment(
                context,
                hidden_src_label(head_index),
                make_coeff_poly(hidden_src_label(head_index), quantize_model_vector(context.model.hidden_heads[head_index].attn_src_kernel_fp)));
            add_static_commitment(
                context,
                hidden_dst_label(head_index),
                make_coeff_poly(hidden_dst_label(head_index), quantize_model_vector(context.model.hidden_heads[head_index].attn_dst_kernel_fp)));
        }
        const bool legacy_single_output = context.model.output_layer.heads.size() == 1;
        for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index) {
            const auto& head = context.model.output_layer.heads[head_index];
            add_static_commitment(
                context,
                output_weight_label(head_index, legacy_single_output),
                make_coeff_poly(
                    output_weight_label(head_index, legacy_single_output),
                    algebra::flatten_matrix_coefficients(quantize_model_matrix(head.seq_kernel_fp))));
            add_static_commitment(
                context,
                output_src_label(head_index, legacy_single_output),
                make_coeff_poly(
                    output_src_label(head_index, legacy_single_output),
                    quantize_model_vector(head.attn_src_kernel_fp)));
            add_static_commitment(
                context,
                output_dst_label(head_index, legacy_single_output),
                make_coeff_poly(
                    output_dst_label(head_index, legacy_single_output),
                    quantize_model_vector(head.attn_dst_kernel_fp)));
        }
    } else {
        add_static_commitment(
            context,
            "V_W",
            make_coeff_poly("V_W", algebra::flatten_matrix_coefficients(context.model.W)));
        add_static_commitment(context, "V_a_src", make_coeff_poly("V_a_src", context.model.a_src));
        add_static_commitment(context, "V_a_dst", make_coeff_poly("V_a_dst", context.model.a_dst));
        add_static_commitment(
            context,
            "V_W_out",
            make_coeff_poly("V_W_out", algebra::flatten_matrix_coefficients(context.model.W_out)));
        add_static_commitment(context, "V_b", make_coeff_poly("V_b", context.model.b));
    }
    end = Clock::now();
    if (metrics != nullptr) {
        metrics->load_static_ms += elapsed_ms(start, end);
    }

    if (full_graph_feature_identity_enabled(context)) {
        context.dataset.features.clear();
        context.dataset.features.shrink_to_fit();
        context.dataset.features_fp.clear();
        context.dataset.features_fp.shrink_to_fit();
    }

    if (persist_context_cache) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        cache[cache_key] = context;
    }
    return context;
}

Proof prove(const ProtocolContext& context, const TraceArtifacts& trace, RunMetrics* metrics) {
    const auto prove_start = Clock::now();
    const auto& route2 = util::route2_options();
    if (trace.commitment_order != dynamic_commitment_labels(context)) {
        throw std::runtime_error("trace commitment order does not match the protocol commitment order");
    }
    if (context.model.has_real_multihead) {
        auto stage_start = Clock::now();
        const auto pre_quotient_challenges = replay_challenges(context, trace.commitments, {});
        const auto quotient_cache_key = quotient_build_cache_key(context, trace);
        std::shared_ptr<const ProofEvaluationBackendRegistry> backend_registry;
        if (route2.fft_backend_upgrade) {
            backend_registry = shared_proof_backend_registry_cache().get_or_build(
                quotient_cache_key + "|backend_registry",
                [&]() {
                    return std::make_shared<ProofEvaluationBackendRegistry>(context, trace);
                });
        }
        auto proof_domain_weight_cache = shared_proof_domain_weight_cache();
        struct QuotientEvalResult {
            std::string label;
            FieldElement value;
            double elapsed_ms = 0.0;
            RunMetrics local_metrics;
        };
        auto compute_quotient = [&](const std::string& label) -> QuotientEvalResult {
            const auto local_start = Clock::now();
            RunMetrics local_metrics;
            EvaluationMemoization quotient_memo(
                context,
                trace,
                pre_quotient_challenges,
                backend_registry,
                proof_domain_weight_cache,
                &local_metrics);
            const auto dependency_labels = quotient_dependency_labels(context, label);
            const auto eval = [&](const std::string& name, const FieldElement& point) {
                return quotient_memo.eval_named(name, point);
            };
            FieldElement value = FieldElement::zero();
            if (label == "t_FH") {
                if (!dependency_labels.empty()) {
                    quotient_memo.precompute_named(
                        dependency_labels,
                        fh_dependency_points(context.domains.fh, {context.kzg.tau}));
                }
                FHQuotientProfile fh_profile;
                value = evaluate_t_fh(context, pre_quotient_challenges, eval, context.kzg.tau, &fh_profile);
                local_metrics.fh_quotient_assembly_ms += fh_profile.assembly_ms;
            } else if (label == "t_edge") {
                if (!dependency_labels.empty()) {
                    quotient_memo.precompute_named(
                        dependency_labels,
                        fh_dependency_points(context.domains.edge, {context.kzg.tau}));
                }
                value = evaluate_t_edge(context, pre_quotient_challenges, trace.witness_scalars, eval, context.kzg.tau);
            } else if (label == "t_in") {
                if (!dependency_labels.empty()) {
                    quotient_memo.precompute_named(
                        dependency_labels,
                        fh_dependency_points(context.domains.in, {context.kzg.tau}));
                }
                value = evaluate_t_in(context, pre_quotient_challenges, trace.external_evaluations, eval, context.kzg.tau);
            } else if (label == "t_d_h") {
                if (!dependency_labels.empty()) {
                    quotient_memo.precompute_named(
                        dependency_labels,
                        fh_dependency_points(context.domains.d, {context.kzg.tau}));
                }
                value = evaluate_t_d(context, pre_quotient_challenges, trace.external_evaluations, eval, context.kzg.tau);
            } else if (label == "t_cat") {
                if (!dependency_labels.empty()) {
                    quotient_memo.precompute_named(
                        dependency_labels,
                        fh_dependency_points(context.domains.cat, {context.kzg.tau}));
                }
                value = evaluate_t_cat(context, pre_quotient_challenges, trace.external_evaluations, eval, context.kzg.tau);
            } else if (label == "t_C") {
                if (!dependency_labels.empty()) {
                    quotient_memo.precompute_named(
                        dependency_labels,
                        fh_dependency_points(context.domains.c, {context.kzg.tau}));
                }
                value = evaluate_t_c(context, pre_quotient_challenges, trace.external_evaluations, eval, context.kzg.tau);
            } else if (label == "t_N") {
                if (!dependency_labels.empty()) {
                    quotient_memo.precompute_named(
                        dependency_labels,
                        fh_dependency_points(context.domains.n, {context.kzg.tau}));
                }
                value = evaluate_t_n(context, pre_quotient_challenges, trace.witness_scalars, eval, context.kzg.tau);
            } else {
                throw std::runtime_error("unknown multi-head quotient label: " + label);
            }
            return {label, value, elapsed_ms(local_start, Clock::now()), std::move(local_metrics)};
        };

        const std::vector<std::string> quotient_labels = {"t_FH", "t_edge", "t_in", "t_d_h", "t_cat", "t_C", "t_N"};
        std::vector<std::pair<std::string, FieldElement>> named_tau_values;
        std::unordered_map<std::string, Commitment> quotient_commitments;
        CachedQuotientArtifacts cached_quotients;
        const bool quotient_cache_hit =
            shared_quotient_artifacts_cache().lookup(quotient_cache_key, &cached_quotients);
        if (quotient_cache_hit) {
            named_tau_values = cached_quotients.named_tau_values;
            quotient_commitments = cached_quotients.quotient_commitments;
            append_note(metrics, "quotient_cache=hit");
        } else {
            append_note(metrics, "quotient_cache=miss");
            std::vector<QuotientEvalResult> quotient_results;
            quotient_results.reserve(quotient_labels.size());
            if (std::thread::hardware_concurrency() > 1) {
                std::vector<std::future<QuotientEvalResult>> futures;
                futures.reserve(quotient_labels.size());
                for (const auto& label : quotient_labels) {
                    futures.push_back(std::async(std::launch::async, compute_quotient, label));
                }
                for (auto& future : futures) {
                    quotient_results.push_back(future.get());
                }
            } else {
                for (const auto& label : quotient_labels) {
                    quotient_results.push_back(compute_quotient(label));
                }
            }
            named_tau_values.reserve(quotient_labels.size());
            for (const auto& label : quotient_labels) {
                const auto it = std::find_if(
                    quotient_results.begin(),
                    quotient_results.end(),
                    [&](const QuotientEvalResult& result) { return result.label == label; });
                if (it == quotient_results.end()) {
                    throw std::runtime_error("missing multi-head quotient result for " + label);
                }
                named_tau_values.push_back({it->label, it->value});
                if (auto* metric = quotient_metric(metrics, it->label); metric != nullptr) {
                    *metric += it->elapsed_ms;
                }
                if (metrics != nullptr && it->label == "t_FH") {
                    metrics->fh_eval_prep_ms += it->local_metrics.fh_eval_prep_ms;
                    metrics->fh_interpolation_ms += it->local_metrics.fh_interpolation_ms;
                    metrics->fh_public_eval_reuse_ms += it->local_metrics.fh_public_eval_reuse_ms;
                    metrics->fh_quotient_assembly_ms += it->local_metrics.fh_quotient_assembly_ms;
                    metrics->quotient_public_eval_ms += it->local_metrics.fh_public_eval_reuse_ms;
                }
            }
            const auto quotient_pack_start = Clock::now();
            quotient_commitments = batch_quotient_commitments(named_tau_values, context.kzg);
            if (metrics != nullptr) {
                metrics->quotient_bundle_pack_ms += elapsed_ms(quotient_pack_start, Clock::now());
            }
            shared_quotient_artifacts_cache().store(
                quotient_cache_key,
                CachedQuotientArtifacts{
                    .named_tau_values = named_tau_values,
                    .quotient_commitments = quotient_commitments,
                });
        }
        const auto challenges = replay_challenges(context, trace.commitments, quotient_commitments);
        if (metrics != nullptr) {
            metrics->quotient_build_ms += elapsed_ms(stage_start, Clock::now());
        }

        Proof proof;
        proof.public_metadata = build_public_metadata(context);
        proof.block_order = proof_block_order();
        proof.challenges = challenges;
        for (const auto& label : dynamic_commitment_labels(context)) {
            proof.dynamic_commitments.push_back({label, trace.commitments.at(label)});
        }
        for (const auto& label : quotient_commitment_labels(context)) {
            proof.quotient_commitments.push_back({label, quotient_commitments.at(label)});
        }

        struct BundleSpec {
            std::string bundle_name;
            std::string trace_domain_name;
            std::shared_ptr<algebra::RootOfUnityDomain> domain;
            std::string z_name;
            std::string v_name;
            std::string quotient_name;
        };
        const std::vector<BundleSpec> bundle_specs = {
            {"FH", "FH", context.domains.fh, "z_FH", "v_FH", "t_FH"},
            {"edge", "edge", context.domains.edge, "z_edge", "v_edge", "t_edge"},
            {"in", "in", context.domains.in, "z_in", "v_in", "t_in"},
            {"d_h", "d", context.domains.d, "z_d_h", "v_d_h", "t_d_h"},
            {"cat", "cat", context.domains.cat, "z_cat", "v_cat", "t_cat"},
            {"C", "C", context.domains.c, "z_C", "v_C", "t_C"},
            {"N", "N", context.domains.n, "z_N", "v_N", "t_N"},
        };
        stage_start = Clock::now();
        struct BundleBuildResult {
            std::string bundle_name;
            DomainOpeningBundle bundle;
            double gather_ms = 0.0;
            double witness_ms = 0.0;
            double total_ms = 0.0;
            RunMetrics local_metrics;
        };
        auto build_bundle = [&](const BundleSpec& spec) -> BundleBuildResult {
            RunMetrics local_metrics;
            EvaluationMemoization opening_memo(
                context,
                trace,
                challenges,
                backend_registry,
                proof_domain_weight_cache,
                &local_metrics);
            auto labels = domain_opening_labels(context, spec.trace_domain_name);
            labels.push_back(spec.quotient_name);
            const std::vector<FieldElement> points = {
                challenges.at(spec.z_name),
                challenges.at(spec.z_name) * spec.domain->omega,
            };
            double gather_ms = 0.0;
            double witness_ms = 0.0;
            const auto local_start = Clock::now();
            auto bundle = make_bundle(
                context,
                trace,
                opening_memo,
                quotient_commitments,
                labels,
                points,
                spec.v_name,
                challenges,
                &local_metrics,
                &gather_ms,
                &witness_ms);
            if (spec.bundle_name == "FH") {
                const auto nested_ms =
                    local_metrics.fh_eval_prep_ms
                    + local_metrics.fh_interpolation_ms
                    + local_metrics.fh_public_eval_reuse_ms
                    + local_metrics.fh_quotient_assembly_ms;
                local_metrics.fh_open_gather_ms =
                    std::max(0.0, local_metrics.fh_open_gather_ms - nested_ms);
            }
            return {
                spec.bundle_name,
                std::move(bundle),
                gather_ms,
                witness_ms,
                elapsed_ms(local_start, Clock::now()),
                std::move(local_metrics),
            };
        };
        std::vector<BundleBuildResult> bundle_results;
        bundle_results.reserve(bundle_specs.size());
        if (std::thread::hardware_concurrency() > 1) {
            std::vector<std::future<BundleBuildResult>> futures;
            futures.reserve(bundle_specs.size());
            for (const auto& spec : bundle_specs) {
                futures.push_back(std::async(std::launch::async, build_bundle, spec));
            }
            for (auto& future : futures) {
                bundle_results.push_back(future.get());
            }
        } else {
            for (const auto& spec : bundle_specs) {
                bundle_results.push_back(build_bundle(spec));
            }
        }
        for (const auto& spec : bundle_specs) {
            const auto it = std::find_if(
                bundle_results.begin(),
                bundle_results.end(),
                [&](const BundleBuildResult& result) { return result.bundle_name == spec.bundle_name; });
            if (it == bundle_results.end()) {
                throw std::runtime_error("missing bundle build result for " + spec.bundle_name);
            }
            proof.domain_openings.push_back({it->bundle_name, it->bundle});
            if (metrics != nullptr) {
                metrics->domain_eval_gather_ms += it->gather_ms;
                metrics->domain_open_witness_ms += it->witness_ms;
                if (auto* metric = domain_open_metric(metrics, it->bundle_name); metric != nullptr) {
                    *metric += it->total_ms;
                }
                if (it->bundle_name == "FH") {
                    metrics->fh_eval_prep_ms += it->local_metrics.fh_eval_prep_ms;
                    metrics->fh_interpolation_ms += it->local_metrics.fh_interpolation_ms;
                    metrics->fh_public_eval_reuse_ms += it->local_metrics.fh_public_eval_reuse_ms;
                    metrics->fh_quotient_assembly_ms += it->local_metrics.fh_quotient_assembly_ms;
                    metrics->fh_open_gather_ms += it->local_metrics.fh_open_gather_ms;
                    metrics->fh_open_witness_ms += it->local_metrics.fh_open_witness_ms;
                    metrics->fh_open_fold_prepare_ms += it->local_metrics.fh_open_fold_prepare_ms;
                }
            }
        }
        if (metrics != nullptr) {
            metrics->domain_opening_ms += elapsed_ms(stage_start, Clock::now());
        }

        const auto ext_specs = multihead_external_specs(context);
        std::vector<std::pair<Commitment, FieldElement>> external_commitments;
        std::vector<FieldElement> external_points;
        stage_start = Clock::now();
        for (const auto& spec : ext_specs) {
            const auto value = trace.external_evaluations.at(spec.proof_name);
            proof.external_evaluations.push_back({spec.proof_name, value});
            external_commitments.push_back({trace.commitments.at(spec.label), value});
            external_points.push_back(challenges.at(spec.challenge_name));
        }
        proof.external_witness = crypto::KZG::open_external_fold(
            external_commitments,
            external_points,
            challenges.at("rho_ext"),
            context.kzg);
        if (metrics != nullptr) {
            metrics->external_opening_ms += elapsed_ms(stage_start, Clock::now());
        }
        for (const auto& [name, value] : trace.witness_scalars) {
            proof.witness_scalars.push_back({name, value});
        }
        if (metrics != nullptr) {
            metrics->prove_time_ms = elapsed_ms(prove_start, Clock::now());
        }
        return proof;
    }

    std::unordered_map<std::string, Commitment> quotient_commitments;
    auto stage_start = Clock::now();
    auto challenges = replay_challenges(context, trace.commitments, quotient_commitments);
    std::shared_ptr<const ProofEvaluationBackendRegistry> backend_registry;
    const auto legacy_quotient_cache_key = quotient_build_cache_key(context, trace);
    if (route2.fft_backend_upgrade) {
        backend_registry = shared_proof_backend_registry_cache().get_or_build(
            legacy_quotient_cache_key + "|backend_registry",
            [&]() {
                return std::make_shared<ProofEvaluationBackendRegistry>(context, trace);
            });
    }
    auto proof_domain_weight_cache = shared_proof_domain_weight_cache();
    // The quotient identities are unchanged. We only overlap the five domain
    // quotient evaluations, which are independent once the trace commitments
    // and Fiat-Shamir challenges have been fixed for this proof instance.
    const bool run_parallel = std::thread::hardware_concurrency() > 1;
    auto build_eval = [&](auto evaluator) {
        return [&, evaluator]() {
            EvaluationMemoization memo(
                context,
                trace,
                challenges,
                backend_registry,
                proof_domain_weight_cache,
                nullptr);
            return evaluator(memo);
        };
    };

    FieldElement t_fh_tau = FieldElement::zero();
    FieldElement t_edge_tau = FieldElement::zero();
    FieldElement t_n_tau = FieldElement::zero();
    FieldElement t_in_tau = FieldElement::zero();
    FieldElement t_d_tau = FieldElement::zero();

    if (run_parallel) {
        auto fh_future = std::async(
            std::launch::async,
            build_eval([&](EvaluationMemoization& memo) {
                if (route2.fft_backend_upgrade) {
                    memo.precompute_named(
                        quotient_dependencies_fh(),
                        {context.kzg.tau, context.kzg.tau * context.domains.fh->omega});
                }
                return evaluate_t_fh(
                    context,
                    challenges,
                    [&](const std::string& name, const FieldElement& point) {
                        return memo.eval_named(name, point);
                    },
                    context.kzg.tau);
            }));
        auto edge_future = std::async(
            std::launch::async,
            build_eval([&](EvaluationMemoization& memo) {
                if (route2.fft_backend_upgrade) {
                    memo.precompute_named(
                        quotient_dependencies_edge(),
                        {context.kzg.tau, context.kzg.tau * context.domains.edge->omega});
                }
                return evaluate_t_edge(
                    context,
                    challenges,
                    trace.witness_scalars,
                    [&](const std::string& name, const FieldElement& point) {
                        return memo.eval_named(name, point);
                    },
                    context.kzg.tau);
            }));
        auto n_future = std::async(
            std::launch::async,
            build_eval([&](EvaluationMemoization& memo) {
                if (route2.fft_backend_upgrade) {
                    memo.precompute_named(quotient_dependencies_n(), {context.kzg.tau});
                }
                return evaluate_t_n(
                    context,
                    challenges,
                    trace.witness_scalars,
                    [&](const std::string& name, const FieldElement& point) {
                        return memo.eval_named(name, point);
                    },
                    context.kzg.tau);
            }));
        auto in_future = std::async(
            std::launch::async,
            build_eval([&](EvaluationMemoization& memo) {
                if (route2.fft_backend_upgrade) {
                    memo.precompute_named(
                        quotient_dependencies_in(),
                        {context.kzg.tau, context.kzg.tau * context.domains.in->omega});
                }
                return evaluate_t_in(
                    context,
                    challenges,
                    trace.external_evaluations,
                    [&](const std::string& name, const FieldElement& point) {
                        return memo.eval_named(name, point);
                    },
                    context.kzg.tau);
            }));
        auto d_future = std::async(
            std::launch::async,
            build_eval([&](EvaluationMemoization& memo) {
                if (route2.fft_backend_upgrade) {
                    memo.precompute_named(
                        quotient_dependencies_d(),
                        {context.kzg.tau, context.kzg.tau * context.domains.d->omega});
                }
                return evaluate_t_d(
                    context,
                    challenges,
                    trace.external_evaluations,
                    [&](const std::string& name, const FieldElement& point) {
                        return memo.eval_named(name, point);
                    },
                    context.kzg.tau);
            }));

        t_fh_tau = fh_future.get();
        t_edge_tau = edge_future.get();
        t_n_tau = n_future.get();
        t_in_tau = in_future.get();
        t_d_tau = d_future.get();
    } else {
        EvaluationMemoization quotient_eval(
            context,
            trace,
            challenges,
            backend_registry,
            proof_domain_weight_cache,
            nullptr);
        if (route2.fft_backend_upgrade) {
            quotient_eval.precompute_named(
                quotient_dependencies_fh(),
                {context.kzg.tau, context.kzg.tau * context.domains.fh->omega});
            quotient_eval.precompute_named(
                quotient_dependencies_edge(),
                {context.kzg.tau, context.kzg.tau * context.domains.edge->omega});
            quotient_eval.precompute_named(
                quotient_dependencies_in(),
                {context.kzg.tau, context.kzg.tau * context.domains.in->omega});
            quotient_eval.precompute_named(
                quotient_dependencies_d(),
                {context.kzg.tau, context.kzg.tau * context.domains.d->omega});
            quotient_eval.precompute_named(quotient_dependencies_n(), {context.kzg.tau});
        }
        FHQuotientProfile fh_profile;
        t_fh_tau = evaluate_t_fh(
            context,
            challenges,
            [&](const std::string& name, const FieldElement& point) {
                return quotient_eval.eval_named(name, point);
            },
            context.kzg.tau,
            &fh_profile);
        if (metrics != nullptr) {
            metrics->fh_quotient_assembly_ms += fh_profile.assembly_ms;
        }
        t_edge_tau = evaluate_t_edge(
            context,
            challenges,
            trace.witness_scalars,
            [&](const std::string& name, const FieldElement& point) {
                return quotient_eval.eval_named(name, point);
            },
            context.kzg.tau);
        t_n_tau = evaluate_t_n(
            context,
            challenges,
            trace.witness_scalars,
            [&](const std::string& name, const FieldElement& point) {
                return quotient_eval.eval_named(name, point);
            },
            context.kzg.tau);
        t_in_tau = evaluate_t_in(
            context,
            challenges,
            trace.external_evaluations,
            [&](const std::string& name, const FieldElement& point) {
                return quotient_eval.eval_named(name, point);
            },
            context.kzg.tau);
        t_d_tau = evaluate_t_d(
            context,
            challenges,
            trace.external_evaluations,
            [&](const std::string& name, const FieldElement& point) {
                return quotient_eval.eval_named(name, point);
            },
            context.kzg.tau);
    }

    quotient_commitments = batch_quotient_commitments(
        {
            {"t_FH", t_fh_tau},
            {"t_edge", t_edge_tau},
            {"t_in", t_in_tau},
            {"t_d", t_d_tau},
            {"t_N", t_n_tau},
        },
        context.kzg);

    challenges = replay_challenges(context, trace.commitments, quotient_commitments);
    if (metrics != nullptr) {
        metrics->quotient_build_ms += elapsed_ms(stage_start, Clock::now());
    }

    Proof proof;
        proof.public_metadata = canonical_public_metadata(context);
    proof.block_order = proof_block_order();
    for (const auto& label : dynamic_commitment_labels(context)) {
        proof.dynamic_commitments.push_back({label, trace.commitments.at(label)});
    }
    for (const auto& label : quotient_commitment_labels(context)) {
        proof.quotient_commitments.push_back({label, quotient_commitments.at(label)});
    }
    proof.challenges = challenges;

    stage_start = Clock::now();
    // Domain opening bundles still open the same object set at the same points
    // as the main spec. The memoizer only reuses identical point evaluations
    // across columns and quotient terms inside this one proof instance.
    struct BundleSpec {
        std::string domain_name;
        std::vector<std::string> labels;
        std::vector<FieldElement> points;
        std::string folding_name;
    };
    const std::vector<BundleSpec> bundle_specs = {
        {
            "FH",
            {"P_Table_feat", "P_Query_feat", "P_m_feat", "P_R_feat", "t_FH"},
            {challenges.at("z_FH"), challenges.at("z_FH") * context.domains.fh->omega},
            "v_FH",
        },
        {
            "edge",
            {
                "P_E_src_edge",
                "P_H_src_star_edge",
                "P_Query_src",
                "P_R_src",
                "P_Table_L",
                "P_Query_L",
                "P_m_L",
                "P_R_L",
                "P_Table_R",
                "P_Query_R",
                "P_m_R",
                "P_R_R",
                "P_Table_exp",
                "P_Query_exp",
                "P_m_exp",
                "P_R_exp",
                "P_E_dst_edge",
                "P_M_edge",
                "P_Query_dst",
                "P_R_dst",
                "P_Delta",
                "P_S",
                "P_Z",
                "P_s_max",
                "P_C_max",
                "P_alpha",
                "P_U",
                "P_Sum_edge",
                "P_inv_edge",
                "P_H_agg_star_edge",
                "P_v_hat",
                "P_PSQ",
                "P_w_psq",
                "P_T_psq_edge",
                "t_edge",
            },
            {challenges.at("z_edge"), challenges.at("z_edge") * context.domains.edge->omega},
            "v_edge",
        },
        {
            "in",
            {"P_a_proj", "P_b_proj", "P_Acc_proj", "t_in"},
            {challenges.at("z_in"), challenges.at("z_in") * context.domains.in->omega},
            "v_in",
        },
        {
            "d",
            {
                "P_a_src",
                "P_b_src",
                "P_Acc_src",
                "P_a_dst",
                "P_b_dst",
                "P_Acc_dst",
                "P_a_star",
                "P_b_star",
                "P_Acc_star",
                "P_a_agg",
                "P_b_agg",
                "P_Acc_agg",
                "P_a_out",
                "P_b_out",
                "P_Acc_out",
                "t_d",
            },
            {challenges.at("z_d"), challenges.at("z_d") * context.domains.d->omega},
            "v_d",
        },
        {
            "N",
            {
                "P_E_src",
                "P_H_star",
                "P_Table_src",
                "P_m_src",
                "P_R_src_node",
                "P_E_dst",
                "P_M",
                "P_Sum",
                "P_inv",
                "P_H_agg_star",
                "P_Table_dst",
                "P_m_dst",
                "P_R_dst_node",
                "t_N",
            },
            {challenges.at("z_N"), challenges.at("z_N") * context.domains.n->omega},
            "v_N",
        },
    };
    std::vector<DomainOpeningBundle> bundle_results(bundle_specs.size());
    if (run_parallel && bundle_specs.size() > 1) {
        std::vector<std::future<DomainOpeningBundle>> futures;
        futures.reserve(bundle_specs.size());
        for (const auto& spec : bundle_specs) {
            futures.push_back(std::async(
                std::launch::async,
                [&, spec]() {
                    EvaluationMemoization memo(
                        context,
                        trace,
                        challenges,
                        backend_registry,
                        proof_domain_weight_cache,
                        nullptr);
                    return make_bundle(
                        context,
                        trace,
                        memo,
                        quotient_commitments,
                        spec.labels,
                        spec.points,
                        spec.folding_name,
                        challenges,
                        nullptr);
                }));
        }
        for (std::size_t i = 0; i < futures.size(); ++i) {
            bundle_results[i] = futures[i].get();
        }
    } else {
        EvaluationMemoization opening_eval(
            context,
            trace,
            challenges,
            backend_registry,
            proof_domain_weight_cache,
            nullptr);
        for (std::size_t i = 0; i < bundle_specs.size(); ++i) {
            bundle_results[i] = make_bundle(
                context,
                trace,
                opening_eval,
                quotient_commitments,
                bundle_specs[i].labels,
                bundle_specs[i].points,
                bundle_specs[i].folding_name,
                challenges,
                nullptr);
        }
    }
    for (std::size_t i = 0; i < bundle_specs.size(); ++i) {
        proof.domain_openings.push_back({bundle_specs[i].domain_name, std::move(bundle_results[i])});
    }
    if (metrics != nullptr) {
        metrics->domain_opening_ms += elapsed_ms(stage_start, Clock::now());
    }

    const std::vector<std::pair<Commitment, FieldElement>> external_commitments = {
        {trace.commitments.at("P_H_prime"), trace.external_evaluations.at("mu_proj")},
        {trace.commitments.at("P_E_src"), trace.external_evaluations.at("mu_src")},
        {trace.commitments.at("P_E_dst"), trace.external_evaluations.at("mu_dst")},
        {trace.commitments.at("P_H_star"), trace.external_evaluations.at("mu_star")},
        {trace.commitments.at("P_H_agg_star"), trace.external_evaluations.at("mu_agg")},
        {trace.commitments.at("P_Y_lin"), trace.external_evaluations.at("mu_Y_lin")},
        {trace.commitments.at("P_Y"), trace.external_evaluations.at("mu_out")},
    };
    const std::vector<FieldElement> external_points = {
        challenges.at("y_proj"),
        challenges.at("y_src"),
        challenges.at("y_dst"),
        challenges.at("y_star"),
        challenges.at("y_agg"),
        challenges.at("y_out"),
        challenges.at("y_out"),
    };
    stage_start = Clock::now();
    proof.external_witness = crypto::KZG::open_external_fold(
        external_commitments,
        external_points,
        challenges.at("rho_ext"),
        context.kzg);
    proof.external_evaluations = {
        {"mu_proj", trace.external_evaluations.at("mu_proj")},
        {"mu_src", trace.external_evaluations.at("mu_src")},
        {"mu_dst", trace.external_evaluations.at("mu_dst")},
        {"mu_star", trace.external_evaluations.at("mu_star")},
        {"mu_agg", trace.external_evaluations.at("mu_agg")},
        {"mu_Y_lin", trace.external_evaluations.at("mu_Y_lin")},
        {"mu_out", trace.external_evaluations.at("mu_out")},
    };
    for (const auto& [name, value] : trace.witness_scalars) {
        proof.witness_scalars.push_back({name, value});
    }
    if (metrics != nullptr) {
        metrics->external_opening_ms += elapsed_ms(stage_start, Clock::now());
        metrics->prove_time_ms = elapsed_ms(prove_start, Clock::now());
    }
    return proof;
}

std::size_t proof_size_bytes(const Proof& proof) {
    constexpr std::size_t kLengthPrefixBytes = sizeof(std::uint64_t);
    constexpr std::size_t kFieldBytes = 32U;
    auto encoded_string_size = [](const std::string& value) {
        return sizeof(std::uint64_t) + value.size();
    };

    std::size_t total = kLengthPrefixBytes;
    for (const auto* value : {
             &proof.public_metadata.protocol_id,
             &proof.public_metadata.dataset_name,
             &proof.public_metadata.task_type,
             &proof.public_metadata.report_unit,
             &proof.public_metadata.graph_count,
             &proof.public_metadata.L,
             &proof.public_metadata.hidden_profile,
             &proof.public_metadata.d_in_profile,
             &proof.public_metadata.K_out,
             &proof.public_metadata.C,
             &proof.public_metadata.batching_rule,
             &proof.public_metadata.subgraph_rule,
             &proof.public_metadata.self_loop_rule,
             &proof.public_metadata.edge_sort_rule,
             &proof.public_metadata.chunking_rule,
             &proof.public_metadata.model_arch_id,
             &proof.public_metadata.model_param_id,
             &proof.public_metadata.static_table_id,
             &proof.public_metadata.quant_cfg_id,
             &proof.public_metadata.domain_cfg,
             &proof.public_metadata.dim_cfg,
             &proof.public_metadata.encoding_id,
             &proof.public_metadata.padding_rule_id,
             &proof.public_metadata.degree_bound_id,
         }) {
        total += encoded_string_size(*value);
    }
    total += kLengthPrefixBytes;
    for (const auto& label : proof.block_order) {
        total += encoded_string_size(label);
    }
    total += kLengthPrefixBytes;
    for (const auto& [name, commitment] : proof.dynamic_commitments) {
        (void)name;
        total += crypto::serialized_size(commitment);
    }
    total += kLengthPrefixBytes;
    for (const auto& [name, commitment] : proof.quotient_commitments) {
        (void)name;
        total += crypto::serialized_size(commitment);
    }
    total += kLengthPrefixBytes;
    for (const auto& [name, bundle] : proof.domain_openings) {
        (void)name;
        total += kLengthPrefixBytes;
        for (const auto& [value_name, values] : bundle.values) {
            (void)value_name;
            total += kLengthPrefixBytes + values.size() * kFieldBytes;
        }
        total += crypto::serialized_size(bundle.witness);
    }
    total += kLengthPrefixBytes;
    for (const auto& [name, value] : proof.external_evaluations) {
        (void)name;
        (void)value;
        total += kFieldBytes;
    }
    total += kLengthPrefixBytes;
    for (const auto& [name, value] : proof.witness_scalars) {
        (void)name;
        (void)value;
        total += kFieldBytes;
    }
    total += crypto::serialized_size(proof.external_witness);
    return total;
}

void export_run_artifacts(
    const ProtocolContext& context,
    const TraceArtifacts& trace,
    const Proof& proof,
    const RunMetrics& metrics,
    bool verified) {
    const auto export_root = (std::filesystem::path(context.config.project_root) / context.config.export_dir).string();
    util::ensure_directory(export_root);

    std::map<std::string, std::string> challenge_strings;
    for (const auto& [name, value] : proof.challenges) {
        challenge_strings[name] = value.to_string();
    }
    util::write_key_values(export_root + "/challenges.txt", challenge_strings);

    const std::map<std::string, std::string> metadata = {
        {"protocol_id", proof.public_metadata.protocol_id},
        {"dataset_name", proof.public_metadata.dataset_name},
        {"task_type", proof.public_metadata.task_type},
        {"report_unit", proof.public_metadata.report_unit},
        {"graph_count", proof.public_metadata.graph_count},
        {"L", proof.public_metadata.L},
        {"hidden_profile", proof.public_metadata.hidden_profile},
        {"d_in_profile", proof.public_metadata.d_in_profile},
        {"K_out", proof.public_metadata.K_out},
        {"C", proof.public_metadata.C},
        {"batching_rule", proof.public_metadata.batching_rule},
        {"subgraph_rule", proof.public_metadata.subgraph_rule},
        {"self_loop_rule", proof.public_metadata.self_loop_rule},
        {"edge_sort_rule", proof.public_metadata.edge_sort_rule},
        {"chunking_rule", proof.public_metadata.chunking_rule},
        {"model_arch_id", proof.public_metadata.model_arch_id},
        {"model_param_id", proof.public_metadata.model_param_id},
        {"static_table_id", proof.public_metadata.static_table_id},
        {"quant_cfg_id", proof.public_metadata.quant_cfg_id},
        {"domain_cfg", proof.public_metadata.domain_cfg},
        {"dim_cfg", proof.public_metadata.dim_cfg},
        {"encoding_id", proof.public_metadata.encoding_id},
        {"padding_rule_id", proof.public_metadata.padding_rule_id},
        {"degree_bound_id", proof.public_metadata.degree_bound_id},
    };
    util::write_key_values(export_root + "/metadata.txt", metadata);
    util::write_lines(export_root + "/proof_block_order.txt", proof.block_order);

    auto format_double = [](double value) {
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(3) << value;
        return stream.str();
    };

    util::write_lines(export_root + "/commitment_order.txt", trace.commitment_order);
    const std::map<std::string, std::string> summary = {
        {"backend", metrics.backend_name},
        {"backend_name", metrics.backend_name},
        {"crypto_backend_name", metrics.crypto_backend_name},
        {"algebra_backend_name", metrics.algebra_backend_name},
        {"compute_backend_name", metrics.compute_backend_name},
        {"config", metrics.config},
        {"dataset", metrics.dataset},
        {"dataset_name", proof.public_metadata.dataset_name},
        {"task_type", proof.public_metadata.task_type},
        {"report_unit", proof.public_metadata.report_unit},
        {"graph_count", proof.public_metadata.graph_count},
        {"L", proof.public_metadata.L},
        {"hidden_profile", proof.public_metadata.hidden_profile},
        {"d_in_profile", proof.public_metadata.d_in_profile},
        {"K_out", proof.public_metadata.K_out},
        {"C", proof.public_metadata.C},
        {"batching_rule", proof.public_metadata.batching_rule},
        {"subgraph_rule", proof.public_metadata.subgraph_rule},
        {"self_loop_rule", proof.public_metadata.self_loop_rule},
        {"edge_sort_rule", proof.public_metadata.edge_sort_rule},
        {"chunking_rule", proof.public_metadata.chunking_rule},
        {"quant_cfg_id", proof.public_metadata.quant_cfg_id},
        {"model_arch_id", proof.public_metadata.model_arch_id},
        {"model_param_id", proof.public_metadata.model_param_id},
        {"static_table_id", proof.public_metadata.static_table_id},
        {"degree_bound_id", proof.public_metadata.degree_bound_id},
        {"benchmark_mode", metrics.benchmark_mode},
        {"enabled_fft_backend_upgrade", metrics.enabled_fft_backend_upgrade ? "true" : "false"},
        {"enabled_fft_kernel_upgrade", metrics.enabled_fft_kernel_upgrade ? "true" : "false"},
        {"enabled_fast_msm", metrics.enabled_fast_msm ? "true" : "false"},
        {"enabled_parallel_fft", metrics.enabled_parallel_fft ? "true" : "false"},
        {"enabled_trace_layout_upgrade", metrics.enabled_trace_layout_upgrade ? "true" : "false"},
        {"enabled_fast_verify_pairing", metrics.enabled_fast_verify_pairing ? "true" : "false"},
        {"enabled_cuda_trace_hotspots", metrics.enabled_cuda_trace_hotspots ? "true" : "false"},
        {"route2_label", metrics.route2_label},
        {"node_count", std::to_string(metrics.node_count)},
        {"edge_count", std::to_string(metrics.edge_count)},
        {"gpu_runtime_present", metrics.gpu_runtime_present ? "true" : "false"},
        {"context_build_ms", format_double(metrics.context_build_ms)},
        {"domain_opening_ms", format_double(metrics.domain_opening_ms)},
        {"external_opening_ms", format_double(metrics.external_opening_ms)},
        {"fft_plan_ms", format_double(metrics.fft_plan_ms)},
        {"fft_backend_route", metrics.fft_backend_route},
        {"feature_projection_ms", format_double(metrics.feature_projection_ms)},
        {"hidden_forward_projection_ms", format_double(metrics.hidden_forward_projection_ms)},
        {"hidden_forward_attention_ms", format_double(metrics.hidden_forward_attention_ms)},
        {"hidden_forward_activation_ms", format_double(metrics.hidden_forward_activation_ms)},
        {"hidden_concat_ms", format_double(metrics.hidden_concat_ms)},
        {"output_forward_projection_ms", format_double(metrics.output_forward_projection_ms)},
        {"output_forward_attention_ms", format_double(metrics.output_forward_attention_ms)},
        {"output_forward_activation_ms", format_double(metrics.output_forward_activation_ms)},
        {"forward_ms", format_double(metrics.forward_ms)},
        {"commit_dynamic_ms", format_double(metrics.commit_dynamic_ms)},
        {"commitment_time_ms", format_double(metrics.commitment_time_ms)},
        {"dynamic_commit_finalize_ms", format_double(metrics.dynamic_commit_finalize_ms)},
        {"dynamic_commit_input_ms", format_double(metrics.dynamic_commit_input_ms)},
        {"dynamic_commit_pack_ms", format_double(metrics.dynamic_commit_pack_ms)},
        {"dynamic_fft_ms", format_double(metrics.dynamic_fft_ms)},
        {"dynamic_domain_convert_ms", format_double(metrics.dynamic_domain_convert_ms)},
        {"dynamic_copy_convert_ms", format_double(metrics.dynamic_copy_convert_ms)},
        {"dynamic_commit_msm_ms", format_double(metrics.dynamic_commit_msm_ms)},
        {"dynamic_bundle_finalize_ms", format_double(metrics.dynamic_bundle_finalize_ms)},
        {"dynamic_polynomial_materialization_ms", format_double(metrics.dynamic_polynomial_materialization_ms)},
        {"is_cold_run", metrics.is_cold_run ? "true" : "false"},
        {"is_full_dataset", metrics.is_full_dataset ? "true" : "false"},
        {"local_edges", std::to_string(context.local.edges.size())},
        {"local_nodes", std::to_string(context.local.num_nodes)},
        {"N_total", std::to_string(context.local.public_input.N_total)},
        {"G_batch", std::to_string(context.local.public_input.G_batch)},
        {"node_ptr", [&]() {
            std::ostringstream out;
            for (std::size_t i = 0; i < context.local.node_ptr.size(); ++i) {
                if (i != 0) out << ',';
                out << context.local.node_ptr[i];
            }
            return out.str();
        }()},
        {"edge_ptr", [&]() {
            std::ostringstream out;
            for (std::size_t i = 0; i < context.local.edge_ptr.size(); ++i) {
                if (i != 0) out << ',';
                out << context.local.edge_ptr[i];
            }
            return out.str();
        }()},
        {"load_static_ms", format_double(metrics.load_static_ms)},
        {"notes", metrics.notes},
        {"proof_size_bytes", std::to_string(metrics.proof_size_bytes)},
        {"prove_accounted_ms", format_double(metrics.prove_accounted_ms)},
        {"prove_accounting_gap_ms", format_double(metrics.prove_accounting_gap_ms)},
        {"prove_time_ms", format_double(metrics.prove_time_ms)},
        {"prove_finalize_ms", format_double(metrics.prove_finalize_ms)},
        {"quotient_build_ms", format_double(metrics.quotient_build_ms)},
        {"quotient_t_fh_ms", format_double(metrics.quotient_t_fh_ms)},
        {"quotient_t_edge_ms", format_double(metrics.quotient_t_edge_ms)},
        {"quotient_t_in_ms", format_double(metrics.quotient_t_in_ms)},
        {"quotient_t_d_h_ms", format_double(metrics.quotient_t_d_h_ms)},
        {"quotient_t_cat_ms", format_double(metrics.quotient_t_cat_ms)},
        {"quotient_t_C_ms", format_double(metrics.quotient_t_c_ms)},
        {"quotient_t_N_ms", format_double(metrics.quotient_t_n_ms)},
        {"quotient_public_eval_ms", format_double(metrics.quotient_public_eval_ms)},
        {"quotient_bundle_pack_ms", format_double(metrics.quotient_bundle_pack_ms)},
        {"quotient_fold_prepare_ms", format_double(metrics.quotient_fold_prepare_ms)},
        {"quotient_copy_convert_ms", format_double(metrics.quotient_copy_convert_ms)},
        {"srs_prepare_ms", format_double(metrics.srs_prepare_ms)},
        {"trace_generation_ms", format_double(metrics.trace_generation_ms)},
        {"trace_misc_ms", format_double(metrics.trace_misc_ms)},
        {"witness_materialization_ms", format_double(metrics.witness_materialization_ms)},
        {"lookup_trace_ms", format_double(metrics.lookup_trace_ms)},
        {"lookup_table_pack_ms", format_double(metrics.lookup_table_pack_ms)},
        {"lookup_query_pack_ms", format_double(metrics.lookup_query_pack_ms)},
        {"lookup_key_build_ms", format_double(metrics.lookup_key_build_ms)},
        {"lookup_multiplicity_ms", format_double(metrics.lookup_multiplicity_ms)},
        {"lookup_accumulator_ms", format_double(metrics.lookup_accumulator_ms)},
        {"lookup_state_machine_ms", format_double(metrics.lookup_state_machine_ms)},
        {"lookup_selector_mask_ms", format_double(metrics.lookup_selector_mask_ms)},
        {"lookup_public_helper_ms", format_double(metrics.lookup_public_helper_ms)},
        {"lookup_copy_convert_ms", format_double(metrics.lookup_copy_convert_ms)},
        {"route_trace_ms", format_double(metrics.route_trace_ms)},
        {"psq_trace_ms", format_double(metrics.psq_trace_ms)},
        {"zkmap_trace_ms", format_double(metrics.zkmap_trace_ms)},
        {"state_machine_trace_ms", format_double(metrics.state_machine_trace_ms)},
        {"padding_selector_trace_ms", format_double(metrics.padding_selector_trace_ms)},
        {"public_poly_trace_ms", format_double(metrics.public_poly_trace_ms)},
        {"hidden_head_trace_ms", format_double(metrics.hidden_head_trace_ms)},
        {"hidden_projection_trace_ms", format_double(metrics.hidden_projection_trace_ms)},
        {"hidden_src_attention_trace_ms", format_double(metrics.hidden_src_attention_trace_ms)},
        {"hidden_dst_attention_trace_ms", format_double(metrics.hidden_dst_attention_trace_ms)},
        {"hidden_edge_score_trace_ms", format_double(metrics.hidden_edge_score_trace_ms)},
        {"hidden_softmax_chain_trace_ms", format_double(metrics.hidden_softmax_chain_trace_ms)},
        {"hidden_h_star_trace_ms", format_double(metrics.hidden_h_star_trace_ms)},
        {"hidden_h_agg_pre_star_trace_ms", format_double(metrics.hidden_h_agg_pre_star_trace_ms)},
        {"hidden_h_agg_star_trace_ms", format_double(metrics.hidden_h_agg_star_trace_ms)},
        {"hidden_route_trace_ms", format_double(metrics.hidden_route_trace_ms)},
        {"hidden_copy_convert_ms", format_double(metrics.hidden_copy_convert_ms)},
        {"output_head_trace_ms", format_double(metrics.output_head_trace_ms)},
        {"route_pack_residual_ms", format_double(metrics.route_pack_residual_ms)},
        {"selector_padding_residual_ms", format_double(metrics.selector_padding_residual_ms)},
        {"public_poly_residual_ms", format_double(metrics.public_poly_residual_ms)},
        {"hidden_output_object_residual_ms", format_double(metrics.hidden_output_object_residual_ms)},
        {"shared_helper_build_ms", format_double(metrics.shared_helper_build_ms)},
        {"field_conversion_residual_ms", format_double(metrics.field_conversion_residual_ms)},
        {"copy_move_residual_ms", format_double(metrics.copy_move_residual_ms)},
        {"trace_finalize_ms", format_double(metrics.trace_finalize_ms)},
        {"fh_table_materialization_ms", format_double(metrics.fh_table_materialization_ms)},
        {"fh_query_materialization_ms", format_double(metrics.fh_query_materialization_ms)},
        {"fh_multiplicity_build_ms", format_double(metrics.fh_multiplicity_build_ms)},
        {"fh_accumulator_build_ms", format_double(metrics.fh_accumulator_build_ms)},
        {"fh_interpolation_ms", format_double(metrics.fh_interpolation_ms)},
        {"fh_lagrange_eval_ms", format_double(metrics.fh_lagrange_eval_ms)},
        {"fh_barycentric_weight_fetch_ms", format_double(metrics.fh_barycentric_weight_fetch_ms)},
        {"fh_point_powers_ms", format_double(metrics.fh_point_powers_ms)},
        {"fh_public_poly_interp_ms", format_double(metrics.fh_public_poly_interp_ms)},
        {"fh_feature_poly_interp_ms", format_double(metrics.fh_feature_poly_interp_ms)},
        {"fh_fold_prep_ms", format_double(metrics.fh_fold_prep_ms)},
        {"fh_opening_eval_prep_ms", format_double(metrics.fh_opening_eval_prep_ms)},
        {"fh_copy_convert_ms", format_double(metrics.fh_copy_convert_ms)},
        {"fh_eval_prep_ms", format_double(metrics.fh_eval_prep_ms)},
        {"fh_public_eval_reuse_ms", format_double(metrics.fh_public_eval_reuse_ms)},
        {"fh_quotient_assembly_ms", format_double(metrics.fh_quotient_assembly_ms)},
        {"fh_open_gather_ms", format_double(metrics.fh_open_gather_ms)},
        {"fh_open_witness_ms", format_double(metrics.fh_open_witness_ms)},
        {"fh_open_fold_prepare_ms", format_double(metrics.fh_open_fold_prepare_ms)},
        {"domain_eval_gather_ms", format_double(metrics.domain_eval_gather_ms)},
        {"domain_open_witness_ms", format_double(metrics.domain_open_witness_ms)},
        {"domain_open_FH_ms", format_double(metrics.domain_open_fh_ms)},
        {"domain_open_edge_ms", format_double(metrics.domain_open_edge_ms)},
        {"domain_open_in_ms", format_double(metrics.domain_open_in_ms)},
        {"domain_open_d_h_ms", format_double(metrics.domain_open_d_h_ms)},
        {"domain_open_cat_ms", format_double(metrics.domain_open_cat_ms)},
        {"domain_open_C_ms", format_double(metrics.domain_open_c_ms)},
        {"domain_open_N_ms", format_double(metrics.domain_open_n_ms)},
        {"verified", verified ? "true" : "false"},
        {"verify_time_ms", format_double(metrics.verify_time_ms)},
        {"verify_metadata_ms", format_double(metrics.verify_metadata_ms)},
        {"verify_transcript_ms", format_double(metrics.verify_transcript_ms)},
        {"verify_domain_opening_ms", format_double(metrics.verify_domain_opening_ms)},
        {"verify_quotient_ms", format_double(metrics.verify_quotient_ms)},
        {"verify_external_fold_ms", format_double(metrics.verify_external_fold_ms)},
        {"verify_misc_ms", format_double(metrics.verify_misc_ms)},
        {"verify_accounted_ms", format_double(metrics.verify_accounted_ms)},
        {"verify_accounting_gap_ms", format_double(metrics.verify_accounting_gap_ms)},
        {"verify_FH_ms", format_double(metrics.verify_fh_ms)},
        {"verify_edge_ms", format_double(metrics.verify_edge_ms)},
        {"verify_in_ms", format_double(metrics.verify_in_ms)},
        {"verify_d_h_ms", format_double(metrics.verify_d_h_ms)},
        {"verify_cat_ms", format_double(metrics.verify_cat_ms)},
        {"verify_C_ms", format_double(metrics.verify_c_ms)},
        {"verify_N_ms", format_double(metrics.verify_n_ms)},
        {"verify_public_eval_ms", format_double(metrics.verify_public_eval_ms)},
        {"verify_bundle_lookup_ms", format_double(metrics.verify_bundle_lookup_ms)},
        {"verify_fold_ms", format_double(metrics.verify_fold_ms)},
        {"verify_copy_convert_ms", format_double(metrics.verify_copy_convert_ms)},
    };
    util::write_key_values(export_root + "/benchmark.txt", summary);
    util::write_json_object(export_root + "/run_manifest.json", summary);

    if (!context.config.dump_trace) {
        return;
    }

    util::ensure_directory(export_root + "/columns");
    util::ensure_directory(export_root + "/matrices");
    for (const auto& [name, column] : trace.columns) {
        util::write_lines(export_root + "/columns/" + name + ".txt", vector_to_lines(column));
    }
    for (const auto& [name, matrix] : trace.matrices) {
        util::write_lines(export_root + "/matrices/" + name + ".txt", matrix_to_lines(matrix));
    }
}

}  // namespace gatzk::protocol
