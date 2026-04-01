#include "gatzk/protocol/prover.hpp"

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <future>
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

void add_static_commitment(
    ProtocolContext& context,
    const std::string& name,
    const Polynomial& polynomial) {
    context.static_commitments[name] = crypto::KZG::commit(name, polynomial, context.kzg);
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
           << (config.allow_synthetic_model ? "synthetic" : "formal");
    return stream.str();
}

std::size_t hidden_head_width(const model::ModelParameters& parameters, const util::AppConfig& config) {
    if (parameters.has_real_multihead && !parameters.hidden_heads.empty()) {
        return parameters.hidden_heads.front().output_bias_fp.size();
    }
    return config.hidden_dim;
}

std::size_t concat_width(const model::ModelParameters& parameters, const util::AppConfig& config) {
    if (parameters.has_real_multihead && !parameters.hidden_heads.empty()) {
        return hidden_head_width(parameters, config) * parameters.hidden_heads.size();
    }
    return config.hidden_dim;
}

PublicMetadata build_public_metadata(const ProtocolContext& context) {
    PublicMetadata metadata;
    metadata.protocol_id = "gatzkml";
    metadata.model_arch_id = context.model.has_real_multihead
        ? "single_layer_gat_hidden8_output1"
        : "legacy_single_head_debug";
    metadata.model_param_id = context.model.has_real_multihead
        ? ("checkpoint_bundle:" + context.config.checkpoint_bundle)
        : ("synthetic_seed:" + std::to_string(context.config.seed));
    metadata.static_table_id = "tables:lrelu+exp+range";
    metadata.quant_cfg_id = "range_bits=" + std::to_string(context.config.range_bits);
    metadata.domain_cfg =
        "FH=" + context.domains.fh->name + ":" + std::to_string(context.domains.fh->size)
        + ",edge=" + context.domains.edge->name + ":" + std::to_string(context.domains.edge->size)
        + ",in=" + context.domains.in->name + ":" + std::to_string(context.domains.in->size)
        + ",d_h=" + context.domains.d->name + ":" + std::to_string(context.domains.d->size)
        + ",cat=" + context.domains.cat->name + ":" + std::to_string(context.domains.cat->size)
        + ",C=" + context.domains.c->name + ":" + std::to_string(context.domains.c->size)
        + ",N=" + context.domains.n->name + ":" + std::to_string(context.domains.n->size);
    metadata.dim_cfg =
        "N=" + std::to_string(context.local.num_nodes)
        + ",E=" + std::to_string(context.local.edges.size())
        + ",d_in=" + std::to_string(context.local.num_features)
        + ",d_h=" + std::to_string(hidden_head_width(context.model, context.config))
        + ",d_cat=" + std::to_string(concat_width(context.model, context.config))
        + ",C=" + std::to_string(context.local.num_classes);
    metadata.encoding_id = "project-fixed-order-v1";
    metadata.padding_rule_id = "zero-pad+selector-mask";
    metadata.degree_bound_id = context.model.has_real_multihead
        ? "note-target:FH,edge,in,d_h,cat,C,N"
        : "legacy:FH,edge,in,d,N";
    return metadata;
}

std::vector<std::string> fixed_proof_block_order() {
    return {"M_pub", "Com_dyn", "S_route", "Eval_ext", "Eval_dom", "Com_quot", "Open_dom", "W_ext", "Pi_bind"};
}

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
    specs.push_back({"mu_cat", "P_H_cat_star", "y_cat"});
    specs.push_back({"mu_out_proj", "P_out_Y_prime", "y_proj_out"});
    specs.push_back({"mu_out_src", "P_out_E_src", "y_src_out"});
    specs.push_back({"mu_out_dst", "P_out_E_dst", "y_dst_out"});
    specs.push_back({"mu_out_star", "P_out_Y_star", "y_out_star"});
    specs.push_back({"mu_out", "P_Y", "y_out"});
    return specs;
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
        for (std::size_t i = 0; i < domain->points.size(); ++i) {
            if (domain->points[i] == point) {
                entry.direct_index = i;
                break;
            }
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
        std::shared_ptr<ProofDomainWeightCache> domain_weight_cache = nullptr)
        : context_(context),
          trace_(trace),
          challenges_(challenges),
          backend_registry_(std::move(backend_registry)),
          domain_weight_cache_(std::move(domain_weight_cache)) {
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
        if (trace_.polynomials.contains(name)) {
            value = eval_polynomial(trace_.polynomials.at(name), point);
        } else if (context_.public_polynomials.contains(name)) {
            value = eval_polynomial(context_.public_polynomials.at(name), point);
        } else if (name == "t_FH") {
            value = evaluate_t_fh(
                context_,
                challenges_,
                [&](const std::string& inner_name, const FieldElement& inner_point) {
                    return eval_named(inner_name, inner_point);
                },
                point);
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

  public:
    std::vector<std::vector<FieldElement>> collect_named_values(
        const std::vector<std::string>& labels,
        const std::vector<FieldElement>& points) {
        std::vector<std::vector<FieldElement>> out(
            labels.size(),
            std::vector<FieldElement>(points.size(), FieldElement::zero()));
        std::vector<bool> filled(labels.size(), false);
        if (labels.empty() || points.empty()) {
            return out;
        }

        const bool use_cuda_bundle_collect =
            util::route2_options().fft_backend_upgrade
            && backend_registry_ != nullptr
            && algebra::configured_algebra_backend() == algebra::AlgebraBackend::Cuda
            && algebra::cuda_backend_available();
        if (!use_cuda_bundle_collect) {
            precompute_named(labels, points);
            for (std::size_t i = 0; i < labels.size(); ++i) {
                for (std::size_t point_index = 0; point_index < points.size(); ++point_index) {
                    out[i][point_index] = eval_named(labels[i], points[point_index]);
                }
            }
            return out;
        }

        using Group = std::pair<std::shared_ptr<algebra::RootOfUnityDomain>, std::vector<std::pair<std::size_t, std::string>>>;
        std::unordered_map<std::string, Group> groups;
        for (std::size_t i = 0; i < labels.size(); ++i) {
            const auto* polynomial = lookup_polynomial(labels[i]);
            if (polynomial == nullptr || polynomial->basis != algebra::PolynomialBasis::Evaluation
                || polynomial->domain == nullptr) {
                continue;
            }
            const auto group_key = polynomial->domain->name + ":" + std::to_string(polynomial->domain->size);
            auto& group = groups[group_key];
            if (group.second.empty()) {
                group.first = polynomial->domain;
            }
            group.second.push_back({i, labels[i]});
        }

        auto cache_value = [&](std::size_t label_index, std::size_t point_index, const FieldElement& value) {
            out[label_index][point_index] = value;
            value_cache_.emplace(labels[label_index] + "@" + point_key(points[point_index]), value);
            filled[label_index] = true;
        };

        for (auto& [group_key, group] : groups) {
            (void)group_key;
            const auto* backend = backend_registry_->find(group.first);
            if (backend == nullptr) {
                continue;
            }

            std::vector<std::string> group_labels;
            group_labels.reserve(group.second.size());
            for (const auto& [label_index, label] : group.second) {
                (void)label_index;
                group_labels.push_back(label);
            }

            bool handled_rotated = false;
            if (util::route2_options().fft_kernel_upgrade && group.first->size >= 256) {
                if (const auto shifts = rotated_point_shifts(group.first, points); shifts.has_value()) {
                    const auto& weight_entry = domain_weights(group.first, points.front());
                    if (weight_entry.direct_index.has_value()) {
                        const auto domain_mask = group.first->size - 1U;
                        for (std::size_t point_index = 0; point_index < points.size(); ++point_index) {
                            const auto direct_index =
                                (*weight_entry.direct_index + (*shifts)[point_index]) & domain_mask;
                            const auto values = backend->values_at_direct_index(group_labels, direct_index);
                            for (std::size_t i = 0; i < group.second.size(); ++i) {
                                cache_value(group.second[i].first, point_index, values[i]);
                            }
                        }
                        handled_rotated = true;
                    } else {
                        const auto device_values = backend->evaluate_device_with_packed_native_weight_rotations(
                            group_labels,
                            weight_entry.native_weights,
                            weight_entry.packed_weights,
                            *shifts);
                        const auto values = backend->materialize_device_rotation_result(device_values);
                        for (std::size_t point_index = 0; point_index < points.size(); ++point_index) {
                            for (std::size_t i = 0; i < group.second.size(); ++i) {
                                cache_value(group.second[i].first, point_index, values[point_index][i]);
                            }
                        }
                        handled_rotated = true;
                    }
                }
            }
            if (handled_rotated) {
                continue;
            }

            for (std::size_t point_index = 0; point_index < points.size(); ++point_index) {
                const auto& weight_entry = domain_weights(group.first, points[point_index]);
                std::vector<FieldElement> values;
                if (weight_entry.direct_index.has_value()) {
                    values = backend->values_at_direct_index(group_labels, *weight_entry.direct_index);
                } else {
                    const auto device_values = backend->evaluate_device_with_packed_native_weights(
                        group_labels,
                        weight_entry.native_weights,
                        weight_entry.packed_weights);
                    values = backend->materialize_device_result(device_values);
                }
                for (std::size_t i = 0; i < group.second.size(); ++i) {
                    cache_value(group.second[i].first, point_index, values[i]);
                }
            }
        }

        for (std::size_t label_index = 0; label_index < labels.size(); ++label_index) {
            if (filled[label_index]) {
                continue;
            }
            for (std::size_t point_index = 0; point_index < points.size(); ++point_index) {
                out[label_index][point_index] = eval_named(labels[label_index], points[point_index]);
            }
        }
        return out;
    }

    void precompute_named(
        const std::vector<std::string>& labels,
        const std::vector<FieldElement>& points) {
        if (!util::route2_options().fft_backend_upgrade) {
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

        for (auto& [group_key, group] : groups) {
            (void)group_key;
            if (backend_registry_ == nullptr) {
                continue;
            }
            const auto* backend = backend_registry_->find(group.first);
            if (backend == nullptr) {
                continue;
            }
            // The rotated-point kernel is a backend-level proving optimization
            // for large root-of-unity domains. Small toy domains do not amortize
            // its setup cost, so we intentionally keep the legacy packed sweep
            // there while sending real hot-path bundles through the fused route.
            if (util::route2_options().fft_kernel_upgrade && group.first->size >= 256) {
                if (const auto shifts = rotated_point_shifts(group.first, points); shifts.has_value()) {
                    const auto& weight_entry = domain_weights(group.first, points.front());
                    auto cache_suffix = [](const FieldElement& point) {
                        return "@" + point_key(point);
                    };
                    if (weight_entry.direct_index.has_value()) {
                        const auto domain_mask = group.first->size - 1U;
                        for (std::size_t point_index = 0; point_index < points.size(); ++point_index) {
                            const auto direct_index = (*weight_entry.direct_index + (*shifts)[point_index]) & domain_mask;
                            const auto values = backend->values_at_direct_index(group.second, direct_index);
                            for (std::size_t i = 0; i < group.second.size(); ++i) {
                                value_cache_.emplace(group.second[i] + cache_suffix(points[point_index]), values[i]);
                            }
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
                    const auto batched_values =
                        backend->evaluate_with_packed_native_weight_rotations(
                            group.second,
                            weight_entry.native_weights,
                            weight_entry.packed_weights,
                            *shifts);
                    for (std::size_t point_index = 0; point_index < points.size(); ++point_index) {
                        for (std::size_t i = 0; i < group.second.size(); ++i) {
                            value_cache_.emplace(
                                group.second[i] + cache_suffix(points[point_index]),
                                batched_values[point_index][i]);
                        }
                    }
                    continue;
                }
            }
            for (const auto& point : points) {
                const auto& weight_entry = domain_weights(group.first, point);
                std::vector<FieldElement> values;
                if (weight_entry.direct_index.has_value()) {
                    values = backend->values_at_direct_index(group.second, *weight_entry.direct_index);
                } else {
                    values = backend->evaluate_with_packed_native_weights(
                        group.second,
                        weight_entry.native_weights,
                        weight_entry.packed_weights);
                }
                const auto cache_suffix = "@" + point_key(point);
                for (std::size_t i = 0; i < group.second.size(); ++i) {
                    value_cache_.emplace(group.second[i] + cache_suffix, values[i]);
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
        if (weight_entry.direct_index.has_value()) {
            return polynomial.data.at(*weight_entry.direct_index);
        }
        // The route2 parallel FFT flag only changes how the cached domain
        // weights are reduced. The opened values themselves are identical.
        return algebra::dot_product_packed_native_weights(
            polynomial.data,
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
        "P_R_feat",
        "P_Table_feat",
        "P_Query_feat",
        "P_Q_tbl_feat",
        "P_m_feat",
        "P_Q_qry_feat",
    };
    return labels;
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

#if GATZK_ENABLE_CUDA_BACKEND
struct DeviceQuotientWeights {
    std::vector<mcl::Fr> native_weights;
    algebra::PackedFieldBuffer packed_weights;
};

DeviceQuotientWeights make_device_quotient_weights(
    const std::shared_ptr<algebra::RootOfUnityDomain>& domain,
    const FieldElement& point,
    ProofDomainWeightCache& weight_cache) {
    const auto& shared = weight_cache.get(domain, point);
    if (!shared.native_weights.empty()) {
        return DeviceQuotientWeights{
            .native_weights = shared.native_weights,
            .packed_weights = shared.packed_weights,
        };
    }

    DeviceQuotientWeights out;
    out.native_weights.resize(domain->size);
    out.native_weights[*shared.direct_index] = FieldElement::one().native();
    algebra::pack_native_field_elements_into(out.native_weights, &out.packed_weights);
    return out;
}

algebra::PackedEvaluationDeviceResult evaluate_device_bundle(
    const algebra::PackedEvaluationBackend& backend,
    const std::vector<std::string>& labels,
    const FieldElement& representative_point,
    const std::vector<std::size_t>& rotations,
    ProofDomainWeightCache& weight_cache) {
    const auto weights = make_device_quotient_weights(backend.domain(), representative_point, weight_cache);
    if (rotations.empty()) {
        return backend.evaluate_device_with_packed_native_weights(
            labels,
            weights.native_weights,
            weights.packed_weights);
    }
    return backend.evaluate_device_with_packed_native_weight_rotations(
        labels,
        weights.native_weights,
        weights.packed_weights,
        rotations);
}

bool debug_compare_cuda_quotients_enabled();

bool evaluate_cuda_fused_quotients(
    const ProtocolContext& context,
    const TraceArtifacts& trace,
    const std::map<std::string, FieldElement>& challenges,
    const ProofEvaluationBackendRegistry& backend_registry,
    ProofDomainWeightCache& weight_cache,
    FieldElement* t_fh_tau,
    FieldElement* t_edge_tau,
    FieldElement* t_n_tau,
    FieldElement* t_in_tau,
    FieldElement* t_d_tau) {
    const auto* fh_backend = backend_registry.find(context.domains.fh);
    const auto* edge_backend = backend_registry.find(context.domains.edge);
    const auto* n_backend = backend_registry.find(context.domains.n);
    const auto* in_backend = backend_registry.find(context.domains.in);
    const auto* d_backend = backend_registry.find(context.domains.d);
    if (fh_backend == nullptr || edge_backend == nullptr || n_backend == nullptr || in_backend == nullptr
        || d_backend == nullptr) {
        return false;
    }

    const auto fh_shift = context.domains.fh->rotation_shift(context.kzg.tau, context.kzg.tau * context.domains.fh->omega);
    const auto edge_shift =
        context.domains.edge->rotation_shift(context.kzg.tau, context.kzg.tau * context.domains.edge->omega);
    const auto in_shift = context.domains.in->rotation_shift(context.kzg.tau, context.kzg.tau * context.domains.in->omega);
    const auto d_shift = context.domains.d->rotation_shift(context.kzg.tau, context.kzg.tau * context.domains.d->omega);
    const auto n_shift = context.domains.n->rotation_shift(context.kzg.tau, context.kzg.tau * context.domains.n->omega);
    if (!fh_shift.has_value() || !edge_shift.has_value() || !n_shift.has_value() || !in_shift.has_value()
        || !d_shift.has_value()) {
        return false;
    }

    const auto fh_evaluations =
        evaluate_device_bundle(*fh_backend, quotient_dependencies_fh(), context.kzg.tau, {0, *fh_shift}, weight_cache);
    const auto edge_evaluations =
        evaluate_device_bundle(*edge_backend, quotient_dependencies_edge(), context.kzg.tau, {0, *edge_shift}, weight_cache);
    const auto n_evaluations =
        evaluate_device_bundle(*n_backend, quotient_dependencies_n(), context.kzg.tau, {0, *n_shift}, weight_cache);
    const auto in_evaluations =
        evaluate_device_bundle(*in_backend, quotient_dependencies_in(), context.kzg.tau, {0, *in_shift}, weight_cache);
    const auto d_evaluations =
        evaluate_device_bundle(*d_backend, quotient_dependencies_d(), context.kzg.tau, {0, *d_shift}, weight_cache);

    *t_fh_tau = evaluate_t_fh_device_cuda(context, challenges, fh_evaluations, context.kzg.tau);
    *t_edge_tau =
        evaluate_t_edge_device_cuda(context, challenges, trace.witness_scalars, edge_evaluations, context.kzg.tau);
    *t_n_tau = evaluate_t_n_device_cuda(context, challenges, trace.witness_scalars, n_evaluations, context.kzg.tau);
    *t_in_tau =
        evaluate_t_in_device_cuda(context, challenges, trace.external_evaluations, in_evaluations, context.kzg.tau);
    *t_d_tau =
        evaluate_t_d_device_cuda(context, challenges, trace.external_evaluations, d_evaluations, context.kzg.tau);
    if (debug_compare_cuda_quotients_enabled()) {
        const auto materialized = d_backend->materialize_device_rotation_result(d_evaluations);
        const auto& labels = quotient_dependencies_d();
        std::unordered_map<std::string, std::size_t> row_by_label;
        row_by_label.reserve(labels.size());
        for (std::size_t i = 0; i < labels.size(); ++i) {
            row_by_label.emplace(labels[i], i);
        }
        const auto tau_omega = context.kzg.tau * context.domains.d->omega;
        const auto cpu_from_materialized = evaluate_t_d(
            context,
            challenges,
            trace.external_evaluations,
            [&](const std::string& name, const FieldElement& point) {
                const auto it = row_by_label.find(name);
                if (it == row_by_label.end()) {
                    throw std::runtime_error("missing materialized d quotient label: " + name);
                }
                std::size_t point_index = 0;
                if (point == tau_omega) {
                    point_index = 1;
                } else if (point != context.kzg.tau) {
                    throw std::runtime_error("unexpected d quotient evaluation point");
                }
                return materialized.at(point_index).at(it->second);
            },
            context.kzg.tau);
        if (cpu_from_materialized != *t_d_tau) {
            std::cerr << "[gatzk][cuda-d-materialized-mismatch] cpu_from_materialized=" << cpu_from_materialized
                      << " gpu=" << *t_d_tau << '\n';
        } else {
            std::cerr << "[gatzk][cuda-d-materialized-match]\n";
        }

        auto lookup_polynomial = [&](const std::string& name) -> const Polynomial& {
            if (trace.polynomials.contains(name)) {
                return trace.polynomials.at(name);
            }
            if (context.public_polynomials.contains(name)) {
                return context.public_polynomials.at(name);
            }
            throw std::runtime_error("missing polynomial for d quotient debug: " + name);
        };
        for (std::size_t row = 0; row < labels.size(); ++row) {
            const auto& polynomial = lookup_polynomial(labels[row]);
            const auto cpu_tau = polynomial.evaluate(context.kzg.tau);
            const auto cpu_tau_omega = polynomial.evaluate(tau_omega);
            if (cpu_tau != materialized.at(0).at(row)) {
                std::cerr << "[gatzk][cuda-d-eval-mismatch] label=" << labels[row]
                          << " point=tau cpu=" << cpu_tau
                          << " gpu=" << materialized.at(0).at(row) << '\n';
            }
            if (cpu_tau_omega != materialized.at(1).at(row)) {
                std::cerr << "[gatzk][cuda-d-eval-mismatch] label=" << labels[row]
                          << " point=tau_omega cpu=" << cpu_tau_omega
                          << " gpu=" << materialized.at(1).at(row) << '\n';
            }
        }
    }
    return true;
}

bool debug_compare_cuda_quotients_enabled() {
    const char* flag = std::getenv("GATZK_DEBUG_COMPARE_CUDA_QUOTIENTS");
    return flag != nullptr && std::string(flag) != "0";
}

void debug_compare_cuda_quotients(
    const ProtocolContext& context,
    const TraceArtifacts& trace,
    const std::map<std::string, FieldElement>& challenges,
    const std::shared_ptr<ProofDomainWeightCache>& proof_domain_weight_cache,
    const FieldElement& t_fh_tau,
    const FieldElement& t_edge_tau,
    const FieldElement& t_n_tau,
    const FieldElement& t_in_tau,
    const FieldElement& t_d_tau) {
    if (!debug_compare_cuda_quotients_enabled()) {
        return;
    }

    EvaluationMemoization memo(
        context,
        trace,
        challenges,
        nullptr,
        proof_domain_weight_cache);
    memo.precompute_named(
        quotient_dependencies_fh(),
        {context.kzg.tau, context.kzg.tau * context.domains.fh->omega});
    memo.precompute_named(
        quotient_dependencies_edge(),
        {context.kzg.tau, context.kzg.tau * context.domains.edge->omega});
    memo.precompute_named(quotient_dependencies_n(), {context.kzg.tau});
    memo.precompute_named(
        quotient_dependencies_in(),
        {context.kzg.tau, context.kzg.tau * context.domains.in->omega});
    memo.precompute_named(
        quotient_dependencies_d(),
        {context.kzg.tau, context.kzg.tau * context.domains.d->omega});

    auto eval = [&](const std::string& name, const FieldElement& point) {
        return memo.eval_named(name, point);
    };
    const auto cpu_t_fh = evaluate_t_fh(context, challenges, eval, context.kzg.tau);
    const auto cpu_t_edge = evaluate_t_edge(context, challenges, trace.witness_scalars, eval, context.kzg.tau);
    const auto cpu_t_n = evaluate_t_n(context, challenges, trace.witness_scalars, eval, context.kzg.tau);
    const auto cpu_t_in = evaluate_t_in(context, challenges, trace.external_evaluations, eval, context.kzg.tau);
    const auto cpu_t_d = evaluate_t_d(context, challenges, trace.external_evaluations, eval, context.kzg.tau);

    auto report = [&](std::string_view label, const FieldElement& cpu_value, const FieldElement& gpu_value) {
        if (cpu_value != gpu_value) {
            std::cerr << "[gatzk][cuda-quotient-mismatch] " << label
                      << " cpu=" << cpu_value
                      << " gpu=" << gpu_value << '\n';
        } else {
            std::cerr << "[gatzk][cuda-quotient-match] " << label << '\n';
        }
    };

    report("t_FH", cpu_t_fh, t_fh_tau);
    report("t_edge", cpu_t_edge, t_edge_tau);
    report("t_N", cpu_t_n, t_n_tau);
    report("t_in", cpu_t_in, t_in_tau);
    report("t_d", cpu_t_d, t_d_tau);
}
#endif

DomainOpeningBundle make_bundle(
    const ProtocolContext& context,
    const TraceArtifacts& trace,
    EvaluationMemoization& memo,
    const std::unordered_map<std::string, Commitment>& quotient_commitments,
    const std::vector<std::string>& labels,
    const std::vector<FieldElement>& points,
    const std::string& folding_name,
    const std::map<std::string, FieldElement>& challenges) {
    DomainOpeningBundle bundle;
    bundle.points = points;
    const auto commitments = collect_commitments(trace, quotient_commitments, labels);
    // For CUDA-backed opening gather, keep packed eval / rotated eval results on
    // device until this PCS boundary, then materialize once for open_batch.
    const auto values = memo.collect_named_values(labels, points);
    for (std::size_t i = 0; i < labels.size(); ++i) {
        bundle.values.push_back({labels[i], values[i]});
    }
    bundle.witness = crypto::KZG::open_batch(commitments, points, values, challenges.at(folding_name), context.kzg);
    return bundle;
}

}  // namespace

ProtocolContext build_context(const util::AppConfig& config, RunMetrics* metrics) {
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, ProtocolContext> cache;

    const auto cache_lookup_start = Clock::now();
    const auto cache_key = context_cache_key(config);
    {
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
    context.local = data::extract_local_subgraph(context.dataset, config.center_node, config.local_nodes);
    if (config.allow_synthetic_model) {
        append_note(metrics, "model_source=synthetic_debug_only");
        context.model = model::build_model_parameters(
            context.local.num_features,
            config.hidden_dim,
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
        append_note(metrics, "hidden_head_count=" + std::to_string(context.model.hidden_heads.size()));
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
    const std::size_t range_size = (1ULL << config.range_bits);
    const std::size_t lrelu_bound = std::max<std::size_t>(4096, range_size);
    const std::size_t edge_size = algebra::next_power_of_two(
        std::max<std::size_t>({
            context.local.num_nodes,
            context.local.edges.size(),
            lrelu_bound + 1,
            range_size,
        }) + 2);
    const std::size_t in_size = algebra::next_power_of_two(context.local.num_features + 2);
    const std::size_t d_h = hidden_head_width(context.model, config);
    const std::size_t d_cat = concat_width(context.model, config);
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
        context.domains.output_head = AttentionHeadDomains{
            .in = context.domains.cat,
            .d = context.domains.c,
        };
    }
    end = Clock::now();
    if (metrics != nullptr) {
        metrics->fft_plan_ms += elapsed_ms(start, end);
    }

    start = Clock::now();
    context.tables.range = make_range_table(range_size);
    context.tables.exp.reserve(range_size);
    for (std::size_t i = 0; i < range_size; ++i) {
        context.tables.exp.push_back({FieldElement(i), FieldElement(range_size - i)});
    }
    context.tables.lrelu.reserve(lrelu_bound + 1);
    for (std::size_t i = 0; i <= lrelu_bound; ++i) {
        context.tables.lrelu.push_back({FieldElement(i), FieldElement(i)});
    }

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
        make_eval_poly("P_Q_proj_valid", build_selector(context.local.num_features, in_size), context.domains.in));
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

    add_static_commitment(
        context,
        "V_T_H",
        make_coeff_poly("V_T_H", algebra::flatten_matrix_coefficients(context.dataset.features)));

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

    if (!context.model.has_real_multihead) {
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

    {
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
        std::shared_ptr<const ProofEvaluationBackendRegistry> backend_registry;
        if (route2.fft_backend_upgrade) {
            backend_registry = std::make_shared<ProofEvaluationBackendRegistry>(context, trace);
        }
        auto proof_domain_weight_cache = std::make_shared<ProofDomainWeightCache>();
        EvaluationMemoization quotient_memo(
            context,
            trace,
            pre_quotient_challenges,
            backend_registry,
            proof_domain_weight_cache);
        const auto eval = [&](const std::string& name, const FieldElement& point) {
            return quotient_memo.eval_named(name, point);
        };
        const std::vector<std::pair<std::string, FieldElement>> named_tau_values = {
            {"t_FH", evaluate_t_fh(context, pre_quotient_challenges, eval, context.kzg.tau)},
            {"t_edge", evaluate_t_edge(context, pre_quotient_challenges, trace.witness_scalars, eval, context.kzg.tau)},
            {"t_in", evaluate_t_in(context, pre_quotient_challenges, trace.external_evaluations, eval, context.kzg.tau)},
            {"t_d_h", evaluate_t_d(context, pre_quotient_challenges, trace.external_evaluations, eval, context.kzg.tau)},
            {"t_cat", evaluate_t_cat(context, pre_quotient_challenges, trace.external_evaluations, eval, context.kzg.tau)},
            {"t_C", evaluate_t_c(context, pre_quotient_challenges, trace.external_evaluations, eval, context.kzg.tau)},
            {"t_N", evaluate_t_n(context, pre_quotient_challenges, trace.witness_scalars, eval, context.kzg.tau)},
        };
        const auto quotient_commitments = batch_quotient_commitments(named_tau_values, context.kzg);
        const auto challenges = replay_challenges(context, trace.commitments, quotient_commitments);
        if (metrics != nullptr) {
            metrics->quotient_build_ms += elapsed_ms(stage_start, Clock::now());
        }

        Proof proof;
        proof.public_metadata = build_public_metadata(context);
        proof.block_order = fixed_proof_block_order();
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
        EvaluationMemoization opening_memo(
            context,
            trace,
            challenges,
            backend_registry,
            proof_domain_weight_cache);
        for (const auto& spec : bundle_specs) {
            auto labels = domain_opening_labels(context, spec.trace_domain_name);
            labels.push_back(spec.quotient_name);
            const std::vector<FieldElement> points = {
                challenges.at(spec.z_name),
                challenges.at(spec.z_name) * spec.domain->omega,
            };
            proof.domain_openings.push_back(
                {spec.bundle_name,
                 make_bundle(
                     context,
                     trace,
                     opening_memo,
                     quotient_commitments,
                     labels,
                     points,
                     spec.v_name,
                     challenges)});
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
    if (route2.fft_backend_upgrade) {
        backend_registry = std::make_shared<ProofEvaluationBackendRegistry>(context, trace);
    }
    auto proof_domain_weight_cache = std::make_shared<ProofDomainWeightCache>();
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
                proof_domain_weight_cache);
            return evaluator(memo);
        };
    };

    FieldElement t_fh_tau = FieldElement::zero();
    FieldElement t_edge_tau = FieldElement::zero();
    FieldElement t_n_tau = FieldElement::zero();
    FieldElement t_in_tau = FieldElement::zero();
    FieldElement t_d_tau = FieldElement::zero();

    bool used_cuda_fused_quotients = false;
#if GATZK_ENABLE_CUDA_BACKEND
    used_cuda_fused_quotients =
        route2.experimental_cuda_quotients
        && route2.fft_backend_upgrade
        && backend_registry != nullptr
        && algebra::configured_algebra_backend() == algebra::AlgebraBackend::Cuda
        && algebra::cuda_backend_available()
        && evaluate_cuda_fused_quotients(
            context,
            trace,
            challenges,
            *backend_registry,
            *proof_domain_weight_cache,
            &t_fh_tau,
            &t_edge_tau,
            &t_n_tau,
            &t_in_tau,
            &t_d_tau);
    if (used_cuda_fused_quotients) {
        debug_compare_cuda_quotients(
            context,
            trace,
            challenges,
            proof_domain_weight_cache,
            t_fh_tau,
            t_edge_tau,
            t_n_tau,
            t_in_tau,
            t_d_tau);
    }
#endif

    if (!used_cuda_fused_quotients && run_parallel) {
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
    } else if (!used_cuda_fused_quotients) {
        EvaluationMemoization quotient_eval(
            context,
            trace,
            challenges,
            backend_registry,
            proof_domain_weight_cache);
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
        t_fh_tau = evaluate_t_fh(
            context,
            challenges,
            [&](const std::string& name, const FieldElement& point) {
                return quotient_eval.eval_named(name, point);
            },
            context.kzg.tau);
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
    proof.public_metadata = build_public_metadata(context);
    proof.block_order = fixed_proof_block_order();
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
                        proof_domain_weight_cache);
                    return make_bundle(
                        context,
                        trace,
                        memo,
                        quotient_commitments,
                        spec.labels,
                        spec.points,
                        spec.folding_name,
                        challenges);
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
            proof_domain_weight_cache);
        for (std::size_t i = 0; i < bundle_specs.size(); ++i) {
            bundle_results[i] = make_bundle(
                context,
                trace,
                opening_eval,
                quotient_commitments,
                bundle_specs[i].labels,
                bundle_specs[i].points,
                bundle_specs[i].folding_name,
                challenges);
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
    std::size_t total =
        proof.public_metadata.protocol_id.size()
        + proof.public_metadata.model_arch_id.size()
        + proof.public_metadata.model_param_id.size()
        + proof.public_metadata.static_table_id.size()
        + proof.public_metadata.quant_cfg_id.size()
        + proof.public_metadata.domain_cfg.size()
        + proof.public_metadata.dim_cfg.size()
        + proof.public_metadata.encoding_id.size()
        + proof.public_metadata.padding_rule_id.size()
        + proof.public_metadata.degree_bound_id.size();
    for (const auto& label : proof.block_order) {
        total += label.size();
    }
    for (const auto& [name, commitment] : proof.dynamic_commitments) {
        (void)name;
        total += crypto::serialized_size(commitment);
    }
    for (const auto& [name, commitment] : proof.quotient_commitments) {
        (void)name;
        total += crypto::serialized_size(commitment);
    }
    for (const auto& [name, bundle] : proof.domain_openings) {
        (void)name;
        total += crypto::serialized_size(bundle.witness);
    }
    total += crypto::serialized_size(proof.external_witness);
    total += proof.witness_scalars.size() * 32U;
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
        {"config", metrics.config},
        {"dataset", metrics.dataset},
        {"enabled_fft_backend_upgrade", metrics.enabled_fft_backend_upgrade ? "true" : "false"},
        {"enabled_fft_kernel_upgrade", metrics.enabled_fft_kernel_upgrade ? "true" : "false"},
        {"enabled_fast_msm", metrics.enabled_fast_msm ? "true" : "false"},
        {"enabled_parallel_fft", metrics.enabled_parallel_fft ? "true" : "false"},
        {"enabled_trace_layout_upgrade", metrics.enabled_trace_layout_upgrade ? "true" : "false"},
        {"enabled_fast_verify_pairing", metrics.enabled_fast_verify_pairing ? "true" : "false"},
        {"node_count", std::to_string(metrics.node_count)},
        {"edge_count", std::to_string(metrics.edge_count)},
        {"domain_opening_ms", format_double(metrics.domain_opening_ms)},
        {"external_opening_ms", format_double(metrics.external_opening_ms)},
        {"fft_plan_ms", format_double(metrics.fft_plan_ms)},
        {"fft_backend_route", metrics.fft_backend_route},
        {"feature_projection_ms", format_double(metrics.feature_projection_ms)},
        {"forward_ms", format_double(metrics.forward_ms)},
        {"commit_dynamic_ms", format_double(metrics.commit_dynamic_ms)},
        {"dynamic_commit_finalize_ms", format_double(metrics.dynamic_commit_finalize_ms)},
        {"dynamic_commit_input_ms", format_double(metrics.dynamic_commit_input_ms)},
        {"dynamic_commit_msm_ms", format_double(metrics.dynamic_commit_msm_ms)},
        {"dynamic_polynomial_materialization_ms", format_double(metrics.dynamic_polynomial_materialization_ms)},
        {"is_cold_run", metrics.is_cold_run ? "true" : "false"},
        {"is_full_dataset", metrics.is_full_dataset ? "true" : "false"},
        {"local_edges", std::to_string(context.local.edges.size())},
        {"local_nodes", std::to_string(context.local.num_nodes)},
        {"load_static_ms", format_double(metrics.load_static_ms)},
        {"notes", metrics.notes},
        {"proof_size_bytes", std::to_string(metrics.proof_size_bytes)},
        {"prove_time_ms", format_double(metrics.prove_time_ms)},
        {"quotient_build_ms", format_double(metrics.quotient_build_ms)},
        {"srs_prepare_ms", format_double(metrics.srs_prepare_ms)},
        {"trace_generation_ms", format_double(metrics.trace_generation_ms)},
        {"witness_materialization_ms", format_double(metrics.witness_materialization_ms)},
        {"lookup_trace_ms", format_double(metrics.lookup_trace_ms)},
        {"route_trace_ms", format_double(metrics.route_trace_ms)},
        {"psq_trace_ms", format_double(metrics.psq_trace_ms)},
        {"zkmap_trace_ms", format_double(metrics.zkmap_trace_ms)},
        {"verified", verified ? "true" : "false"},
        {"verify_time_ms", format_double(metrics.verify_time_ms)},
    };
    util::write_key_values(export_root + "/summary.txt", summary);
    util::write_key_values(export_root + "/benchmark.txt", summary);

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
