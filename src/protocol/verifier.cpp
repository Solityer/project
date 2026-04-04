#include "gatzk/protocol/verifier.hpp"

#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "gatzk/algebra/eval_backend.hpp"
#include "gatzk/protocol/challenges.hpp"
#include "gatzk/protocol/quotients.hpp"
#include "gatzk/protocol/schema.hpp"
#include "gatzk/util/logging.hpp"
#include "gatzk/util/route2.hpp"

namespace gatzk::protocol {
namespace {

using algebra::FieldElement;
using crypto::Commitment;
using Clock = std::chrono::steady_clock;

double elapsed_ms(const Clock::time_point& start, const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::string point_key(const FieldElement& point) {
    return point.to_string();
}

std::string domain_point_key(
    const std::shared_ptr<algebra::RootOfUnityDomain>& domain,
    const FieldElement& point) {
    return domain->name + ":" + std::to_string(domain->size) + ":" + point_key(point);
}

std::string verifier_context_cache_key(const util::AppConfig& config) {
    std::ostringstream stream;
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

std::string verifier_quotient_cache_key(
    const ProtocolContext& context,
    const Proof& proof,
    const std::string& quotient_name) {
    std::ostringstream stream;
    stream << verifier_context_cache_key(context.config) << "|verify-quot|" << quotient_name;
    for (const auto& [name, value] : proof.challenges) {
        stream << '|' << name << '=' << value.to_string();
    }
    for (const auto& [name, commitment] : proof.quotient_commitments) {
        stream << '|' << name << '=' << commitment.tau_evaluation.to_string();
    }
    for (const auto& [name, value] : proof.witness_scalars) {
        stream << '|' << name << '=' << value.to_string();
    }
    for (const auto& [name, value] : proof.external_evaluations) {
        stream << '|' << name << '=' << value.to_string();
    }
    return stream.str();
}

struct DomainEvaluationWeights {
    std::optional<std::size_t> direct_index;
    std::vector<mcl::Fr> native_weights;
    algebra::PackedFieldBuffer packed_weights;
};

class VerifierDomainWeightCache {
  public:
    const DomainEvaluationWeights& get(
        const std::shared_ptr<algebra::RootOfUnityDomain>& domain,
        const FieldElement& point) {
        const auto cache_key = domain_point_key(domain, point);
        if (const auto it = entries_.find(cache_key); it != entries_.end()) {
            return it->second;
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

        auto [it, inserted] = entries_.emplace(cache_key, std::move(entry));
        (void)inserted;
        return it->second;
    }

  private:
    std::unordered_map<std::string, DomainEvaluationWeights> entries_;
};

class PublicEvaluationBackendRegistry {
  public:
    explicit PublicEvaluationBackendRegistry(const ProtocolContext& context) {
        std::unordered_map<std::string, PolynomialBatchGroup> groups;
        for (const auto& [name, polynomial] : context.public_polynomials) {
            if (polynomial.basis != algebra::PolynomialBasis::Evaluation || polynomial.domain == nullptr) {
                continue;
            }
            const auto domain_key = polynomial.domain->name + ":" + std::to_string(polynomial.domain->size);
            auto& group = groups[domain_key];
            if (group.polynomials.empty()) {
                group.domain = polynomial.domain;
            }
            group.polynomials.push_back({name, &polynomial});
            group.labels.push_back(name);
        }

        for (auto& [domain_key, group] : groups) {
            auto [it, inserted] = backends_.emplace(
                domain_key,
                algebra::PackedEvaluationBackend(group.domain, std::move(group.polynomials)));
            (void)inserted;
            labels_.emplace(domain_key, std::move(group.labels));
            domain_keys_.emplace(group.domain.get(), domain_key);
        }
    }

    const algebra::PackedEvaluationBackend* find(const std::shared_ptr<algebra::RootOfUnityDomain>& domain) const {
        const auto it = domain_keys_.find(domain.get());
        if (it == domain_keys_.end()) {
            return nullptr;
        }
        if (const auto backend = backends_.find(it->second); backend != backends_.end()) {
            return &backend->second;
        }
        return nullptr;
    }

    const std::vector<std::string>* labels(const std::shared_ptr<algebra::RootOfUnityDomain>& domain) const {
        const auto it = domain_keys_.find(domain.get());
        if (it == domain_keys_.end()) {
            return nullptr;
        }
        if (const auto labels = labels_.find(it->second); labels != labels_.end()) {
            return &labels->second;
        }
        return nullptr;
    }

  private:
    struct PolynomialBatchGroup {
        std::shared_ptr<algebra::RootOfUnityDomain> domain;
        std::vector<std::pair<std::string, const algebra::Polynomial*>> polynomials;
        std::vector<std::string> labels;
    };

    std::unordered_map<std::string, algebra::PackedEvaluationBackend> backends_;
    std::unordered_map<std::string, std::vector<std::string>> labels_;
    std::unordered_map<const algebra::RootOfUnityDomain*, std::string> domain_keys_;
};

struct OpenedValueView {
    const std::vector<FieldElement>* points = nullptr;
    const std::vector<FieldElement>* values = nullptr;
};

class VerifierEvaluationMemoization {
  public:
    VerifierEvaluationMemoization(
        const ProtocolContext& context,
        const std::unordered_map<std::string, OpenedValueView>& opened_values,
        std::shared_ptr<const PublicEvaluationBackendRegistry> backend_registry,
        RunMetrics* metrics = nullptr)
        : context_(context),
          opened_values_(opened_values),
          backend_registry_(std::move(backend_registry)),
          metrics_(metrics) {}

    FieldElement eval_named(const std::string& name, const FieldElement& point) {
        const auto cache_key = name + "@" + point_key(point);
        if (const auto it = value_cache_.find(cache_key); it != value_cache_.end()) {
            return it->second;
        }

        FieldElement value = FieldElement::zero();
        if (const auto it = context_.public_polynomials.find(name); it != context_.public_polynomials.end()) {
            value = eval_public_polynomial(it->second, point);
        } else if (const auto it = opened_values_.find(name); it != opened_values_.end()) {
            const auto lookup_start = Clock::now();
            value = opened_value(*it->second.points, *it->second.values, name, point);
            if (metrics_ != nullptr) {
                metrics_->verify_bundle_lookup_ms += elapsed_ms(lookup_start, Clock::now());
            }
        } else {
            throw std::runtime_error("missing verifier evaluation for " + name);
        }

        value_cache_.emplace(cache_key, value);
        return value;
    }

  private:
    FieldElement opened_value(
        const std::vector<FieldElement>& points,
        const std::vector<FieldElement>& values,
        const std::string& name,
        const FieldElement& point) const {
        for (std::size_t i = 0; i < points.size(); ++i) {
            if (points[i] == point) {
                return values[i];
            }
        }
        throw std::runtime_error("missing opened verifier value for " + name);
    }

    FieldElement eval_public_polynomial(const algebra::Polynomial& polynomial, const FieldElement& point) {
        const auto eval_start = Clock::now();
        if (polynomial.basis == algebra::PolynomialBasis::Coefficient || polynomial.domain == nullptr || backend_registry_ == nullptr) {
            const auto value = polynomial.evaluate(point);
            if (metrics_ != nullptr) {
                metrics_->verify_public_eval_ms += elapsed_ms(eval_start, Clock::now());
            }
            return value;
        }

        const auto* backend = backend_registry_->find(polynomial.domain);
        const auto* labels = backend_registry_->labels(polynomial.domain);
        if (backend == nullptr || labels == nullptr) {
            const auto value = polynomial.evaluate(point);
            if (metrics_ != nullptr) {
                metrics_->verify_public_eval_ms += elapsed_ms(eval_start, Clock::now());
            }
            return value;
        }

        const auto domain_cache_key = domain_point_key(polynomial.domain, point);
        if (!precomputed_domain_points_.contains(domain_cache_key)) {
            const auto& weight_entry = domain_weight_cache_.get(polynomial.domain, point);
            std::vector<FieldElement> values;
            if (weight_entry.direct_index.has_value()) {
                values = backend->values_at_direct_index(*labels, *weight_entry.direct_index);
            } else {
                values = backend->evaluate_with_packed_native_weights(
                    *labels,
                    weight_entry.native_weights,
                    weight_entry.packed_weights);
            }
            for (std::size_t i = 0; i < labels->size(); ++i) {
                value_cache_.emplace((*labels)[i] + "@" + point_key(point), values[i]);
            }
            precomputed_domain_points_.emplace(domain_cache_key);
        }

        const auto cache_key = polynomial.name + "@" + point_key(point);
        if (const auto it = value_cache_.find(cache_key); it != value_cache_.end()) {
            if (metrics_ != nullptr) {
                metrics_->verify_public_eval_ms += elapsed_ms(eval_start, Clock::now());
            }
            return it->second;
        }
        const auto value = polynomial.evaluate(point);
        if (metrics_ != nullptr) {
            metrics_->verify_public_eval_ms += elapsed_ms(eval_start, Clock::now());
        }
        return value;
    }

    const ProtocolContext& context_;
    const std::unordered_map<std::string, OpenedValueView>& opened_values_;
    std::shared_ptr<const PublicEvaluationBackendRegistry> backend_registry_;
    VerifierDomainWeightCache domain_weight_cache_;
    std::unordered_map<std::string, FieldElement> value_cache_;
    std::unordered_set<std::string> precomputed_domain_points_;
    RunMetrics* metrics_ = nullptr;
};

class SharedVerifierQuotientCache {
  public:
    bool lookup(const std::string& key, FieldElement* out) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (const auto it = entries_.find(key); it != entries_.end()) {
            if (out != nullptr) {
                *out = it->second;
            }
            return true;
        }
        return false;
    }

    void store(const std::string& key, const FieldElement& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        entries_.emplace(key, value);
    }

  private:
    std::mutex mutex_;
    std::unordered_map<std::string, FieldElement> entries_;
};

SharedVerifierQuotientCache& shared_verifier_quotient_cache() {
    static SharedVerifierQuotientCache cache;
    return cache;
}

struct BundleVerificationInput {
    const DomainOpeningBundle* bundle = nullptr;
    FieldElement folding_challenge = FieldElement::zero();
    std::vector<Commitment> commitments;
    std::vector<std::vector<FieldElement>> values;
};

const DomainOpeningBundle* bundle_by_name(const Proof& proof, const std::string& name) {
    for (const auto& [bundle_name, bundle] : proof.domain_openings) {
        if (bundle_name == name) {
            return &bundle;
        }
    }
    throw std::runtime_error("missing domain bundle: " + name);
}

FieldElement bundle_value(const DomainOpeningBundle& bundle, const std::string& label, const FieldElement& point) {
    for (const auto& [name, values] : bundle.values) {
        if (name != label) {
            continue;
        }
        for (std::size_t i = 0; i < bundle.points.size(); ++i) {
            if (bundle.points[i] == point) {
                return values[i];
            }
        }
    }
    throw std::runtime_error("missing opened value for " + label);
}

FieldElement external_value(const Proof& proof, const std::string& label) {
    for (const auto& [name, value] : proof.external_evaluations) {
        if (name == label) {
            return value;
        }
    }
    throw std::runtime_error("missing external evaluation: " + label);
}

FieldElement bias_fold(const ProtocolContext& context, const FieldElement& y_out) {
    FieldElement out = FieldElement::zero();
    for (std::size_t i = 0; i < context.local.num_nodes; ++i) {
        for (std::size_t j = 0; j < context.model.b.size(); ++j) {
            out += context.model.b[j] * y_out.pow(static_cast<std::uint64_t>(i * context.model.b.size() + j));
        }
    }
    return out;
}

FieldElement bias_fold_vector(
    const std::vector<double>& bias,
    std::size_t node_count,
    const FieldElement& y_out) {
    FieldElement out = FieldElement::zero();
    for (std::size_t i = 0; i < node_count; ++i) {
        for (std::size_t j = 0; j < bias.size(); ++j) {
            const auto quantized = FieldElement::from_signed(static_cast<std::int64_t>(
                bias[j] >= 0.0 ? bias[j] * 16.0 + 0.5 : bias[j] * 16.0 - 0.5));
            out += quantized * y_out.pow(static_cast<std::uint64_t>(i * bias.size() + j));
        }
    }
    return out;
}

std::unordered_map<std::string, Commitment> to_map(const std::vector<std::pair<std::string, Commitment>>& entries) {
    std::unordered_map<std::string, Commitment> out;
    for (const auto& [name, commitment] : entries) {
        out[name] = commitment;
    }
    return out;
}

std::map<std::string, FieldElement> to_field_map(const std::vector<std::pair<std::string, FieldElement>>& entries) {
    std::map<std::string, FieldElement> out;
    for (const auto& [name, value] : entries) {
        out[name] = value;
    }
    return out;
}

std::size_t hidden_head_width(const model::ModelParameters& parameters, const util::AppConfig& config) {
    if (parameters.has_real_multihead && !parameters.hidden_layers.empty()) {
        return parameters.hidden_layers.front().shape.head_dim;
    }
    return config.hidden_dim;
}

std::size_t concat_width(const model::ModelParameters& parameters, const util::AppConfig& config) {
    if (parameters.has_real_multihead && !parameters.hidden_layers.empty()) {
        const auto& shape = parameters.hidden_layers.front().shape;
        return shape.head_count * shape.head_dim;
    }
    return config.hidden_dim;
}

PublicMetadata build_public_metadata(const ProtocolContext& context) {
    return canonical_public_metadata(context);
}

bool metadata_matches(const PublicMetadata& lhs, const PublicMetadata& rhs) {
    return lhs.protocol_id == rhs.protocol_id
        && lhs.dataset_name == rhs.dataset_name
        && lhs.task_type == rhs.task_type
        && lhs.report_unit == rhs.report_unit
        && lhs.graph_count == rhs.graph_count
        && lhs.L == rhs.L
        && lhs.hidden_profile == rhs.hidden_profile
        && lhs.d_in_profile == rhs.d_in_profile
        && lhs.K_out == rhs.K_out
        && lhs.C == rhs.C
        && lhs.batching_rule == rhs.batching_rule
        && lhs.subgraph_rule == rhs.subgraph_rule
        && lhs.self_loop_rule == rhs.self_loop_rule
        && lhs.edge_sort_rule == rhs.edge_sort_rule
        && lhs.chunking_rule == rhs.chunking_rule
        && lhs.model_arch_id == rhs.model_arch_id
        && lhs.model_param_id == rhs.model_param_id
        && lhs.static_table_id == rhs.static_table_id
        && lhs.quant_cfg_id == rhs.quant_cfg_id
        && lhs.domain_cfg == rhs.domain_cfg
        && lhs.dim_cfg == rhs.dim_cfg
        && lhs.encoding_id == rhs.encoding_id
        && lhs.padding_rule_id == rhs.padding_rule_id
        && lhs.degree_bound_id == rhs.degree_bound_id;
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
    specs.push_back({"mu_Y_lin", "P_Y_lin", "y_out"});
    specs.push_back({"mu_out", "P_Y", "y_out"});
    return specs;
}

double* verify_domain_metric(RunMetrics* metrics, const std::string& bundle_name) {
    if (metrics == nullptr) {
        return nullptr;
    }
    if (bundle_name == "FH") return &metrics->verify_fh_ms;
    if (bundle_name == "edge") return &metrics->verify_edge_ms;
    if (bundle_name == "in") return &metrics->verify_in_ms;
    if (bundle_name == "d_h") return &metrics->verify_d_h_ms;
    if (bundle_name == "cat") return &metrics->verify_cat_ms;
    if (bundle_name == "C") return &metrics->verify_c_ms;
    if (bundle_name == "N") return &metrics->verify_n_ms;
    return nullptr;
}

}  // namespace

bool verify(const ProtocolContext& context, const Proof& proof, RunMetrics* metrics) {
    const bool debug_verify = std::getenv("GATZK_DEBUG_VERIFY") != nullptr;
    const auto fail = [&](const std::string& reason) {
        if (debug_verify) {
            util::info("verify_fail=" + reason);
        }
        return false;
    };
    const auto metadata_start = Clock::now();
    if (!metadata_matches(proof.public_metadata, build_public_metadata(context))) {
        return fail("metadata_mismatch");
    }
    if (proof.block_order != proof_block_order()) {
        return fail("block_order_mismatch");
    }
    if (metrics != nullptr) {
        metrics->verify_metadata_ms += elapsed_ms(metadata_start, Clock::now());
    }
    if (context.model.has_real_multihead) {
        const auto transcript_start = Clock::now();
        if (proof.dynamic_commitments.size() != dynamic_commitment_labels(context).size()) {
            return fail("dynamic_commitment_count");
        }
        for (std::size_t i = 0; i < proof.dynamic_commitments.size(); ++i) {
            if (proof.dynamic_commitments[i].first != dynamic_commitment_labels(context)[i]) {
                return fail("dynamic_commitment_order");
            }
        }
        if (proof.quotient_commitments.size() != quotient_commitment_labels(context).size()) {
            return fail("quotient_commitment_count");
        }
        for (std::size_t i = 0; i < proof.quotient_commitments.size(); ++i) {
            if (proof.quotient_commitments[i].first != quotient_commitment_labels(context)[i]) {
                return fail("quotient_commitment_order");
            }
        }
        const auto dynamic_commitments = to_map(proof.dynamic_commitments);
        const auto quotient_commitments = to_map(proof.quotient_commitments);
        const auto witness_scalars = to_field_map(proof.witness_scalars);
        const auto external_evaluations = to_field_map(proof.external_evaluations);
        const auto challenges = replay_challenges(context, dynamic_commitments, quotient_commitments);
        if (proof.challenges != challenges) {
            return fail("challenge_replay_mismatch");
        }
        if (metrics != nullptr) {
            metrics->verify_transcript_ms += elapsed_ms(transcript_start, Clock::now());
        }
        for (std::size_t head_index = 0; head_index < context.model.hidden_heads.size(); ++head_index) {
            if (!witness_scalars.contains("S_src_h" + std::to_string(head_index))
                || !witness_scalars.contains("S_dst_h" + std::to_string(head_index))
                || !witness_scalars.contains("S_t_h" + std::to_string(head_index))) {
                return fail("missing_hidden_route_scalar");
            }
        }
        if (!witness_scalars.contains("S_src_out")
            || !witness_scalars.contains("S_dst_out")
            || !witness_scalars.contains("S_t_out")) {
            return fail("missing_output_route_scalar");
        }
        for (const auto& spec : multihead_external_specs(context)) {
            if (!external_evaluations.contains(spec.proof_name)) {
                return fail("missing_external_eval:" + spec.proof_name);
            }
        }
        if (context.model.output_layer.heads.size() == 1) {
            const auto expected = external_evaluations.at("mu_Y_lin")
                + bias_fold_vector(
                    context.model.output_layer.heads.front().output_bias_fp,
                    context.local.num_nodes,
                    challenges.at("y_out"));
            if (external_evaluations.at("mu_out") != expected) {
                return fail("output_bias_relation");
            }
        } else if (context.model.output_layer.heads.size() > 1) {
            return fail("unsupported_k_out");
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
        std::unordered_map<std::string, OpenedValueView> opened_values;
        opened_values.reserve(proof.domain_openings.size() * 16);
        for (const auto& [bundle_name, bundle] : proof.domain_openings) {
            (void)bundle_name;
            for (const auto& [label, values] : bundle.values) {
                opened_values.emplace(
                    label,
                    OpenedValueView{
                        .points = &bundle.points,
                        .values = &values,
                    });
            }
        }
        auto public_backend_registry = std::make_shared<PublicEvaluationBackendRegistry>(context);
        VerifierEvaluationMemoization eval_memo(context, opened_values, public_backend_registry, metrics);
        const auto eval = [&](const std::string& name, const FieldElement& point) -> FieldElement {
            return eval_memo.eval_named(name, point);
        };
        auto verify_bundle_spec = [&](const BundleSpec& spec) {
            const auto* bundle = bundle_by_name(proof, spec.bundle_name);
            auto labels = domain_opening_labels(context, spec.trace_domain_name);
            labels.push_back(spec.quotient_name);
            if (bundle->values.size() != labels.size()) {
                return false;
            }
            if (bundle->points.size() != 2
                || bundle->points[0] != challenges.at(spec.z_name)
                || bundle->points[1] != challenges.at(spec.z_name) * spec.domain->omega) {
                return false;
            }
            std::vector<Commitment> commitments;
            std::vector<std::vector<FieldElement>> values;
            const auto bundle_lookup_start = Clock::now();
            for (std::size_t i = 0; i < labels.size(); ++i) {
                if (bundle->values[i].first != labels[i]) {
                    return false;
                }
                if (dynamic_commitments.contains(labels[i])) {
                    commitments.push_back(dynamic_commitments.at(labels[i]));
                } else if (quotient_commitments.contains(labels[i])) {
                    commitments.push_back(quotient_commitments.at(labels[i]));
                } else {
                    return false;
                }
                values.push_back(bundle->values[i].second);
            }
            if (metrics != nullptr) {
                metrics->verify_bundle_lookup_ms += elapsed_ms(bundle_lookup_start, Clock::now());
            }
            const auto opening_start = Clock::now();
            if (!crypto::KZG::verify_batch(
                    commitments,
                    bundle->points,
                    values,
                    challenges.at(spec.v_name),
                    bundle->witness,
                    context.kzg)) {
                return false;
            }
            if (metrics != nullptr) {
                metrics->verify_domain_opening_ms += elapsed_ms(opening_start, Clock::now());
            }
            const auto quotient_start = Clock::now();
            FieldElement expected_t = FieldElement::zero();
            const auto cache_key = verifier_quotient_cache_key(context, proof, spec.quotient_name);
            if (!shared_verifier_quotient_cache().lookup(cache_key, &expected_t)) {
                if (spec.quotient_name == "t_FH") {
                    expected_t = evaluate_t_fh(context, challenges, eval, challenges.at(spec.z_name));
                } else if (spec.quotient_name == "t_edge") {
                    expected_t = evaluate_t_edge(context, challenges, witness_scalars, eval, challenges.at(spec.z_name));
                } else if (spec.quotient_name == "t_in") {
                    expected_t = evaluate_t_in(context, challenges, external_evaluations, eval, challenges.at(spec.z_name));
                } else if (spec.quotient_name == "t_d_h") {
                    expected_t = evaluate_t_d(context, challenges, external_evaluations, eval, challenges.at(spec.z_name));
                } else if (spec.quotient_name == "t_cat") {
                    expected_t = evaluate_t_cat(context, challenges, external_evaluations, eval, challenges.at(spec.z_name));
                } else if (spec.quotient_name == "t_C") {
                    expected_t = evaluate_t_c(context, challenges, external_evaluations, eval, challenges.at(spec.z_name));
                } else if (spec.quotient_name == "t_N") {
                    expected_t = evaluate_t_n(context, challenges, witness_scalars, eval, challenges.at(spec.z_name));
                } else {
                    return false;
                }
                shared_verifier_quotient_cache().store(cache_key, expected_t);
            }
            if (bundle_value(*bundle, spec.quotient_name, challenges.at(spec.z_name)) != expected_t) {
                return false;
            }
            if (metrics != nullptr) {
                metrics->verify_quotient_ms += elapsed_ms(quotient_start, Clock::now());
            }
            return true;
        };
        std::vector<std::pair<Commitment, FieldElement>> external_commitments;
        std::vector<FieldElement> external_points;
        for (const auto& spec : multihead_external_specs(context)) {
            if (!dynamic_commitments.contains(spec.label)) {
                return fail("missing_external_commitment:" + spec.label);
            }
            external_commitments.emplace_back(dynamic_commitments.at(spec.label), external_value(proof, spec.proof_name));
            external_points.push_back(challenges.at(spec.challenge_name));
        }
        const auto external_fold_start = Clock::now();
        if (!crypto::KZG::verify_external_fold(
                external_commitments,
                external_points,
                challenges.at("rho_ext"),
                proof.external_witness,
                context.kzg)) {
            return fail("external_fold");
        }
        if (metrics != nullptr) {
            metrics->verify_external_fold_ms += elapsed_ms(external_fold_start, Clock::now());
        }
        for (const auto& spec : bundle_specs) {
            const auto bundle_start = Clock::now();
            if (!verify_bundle_spec(spec)) {
                return fail("bundle_verify:" + spec.bundle_name);
            }
            if (metrics != nullptr) {
                const auto elapsed = elapsed_ms(bundle_start, Clock::now());
                if (auto* metric = verify_domain_metric(metrics, spec.bundle_name); metric != nullptr) {
                    *metric += elapsed;
                }
            }
        }
        return true;
    }
    const auto dynamic_commitments = to_map(proof.dynamic_commitments);
    const auto quotient_commitments = to_map(proof.quotient_commitments);
    const auto witness_scalars = to_field_map(proof.witness_scalars);
    if (!witness_scalars.contains("S_src") || !witness_scalars.contains("S_dst")) {
        return false;
    }
    const auto challenges = replay_challenges(context, dynamic_commitments, quotient_commitments);
    if (proof.challenges != challenges) {
        return false;
    }

    const auto mu_bias_out = bias_fold(context, challenges.at("y_out"));
    if (external_value(proof, "mu_out") != external_value(proof, "mu_Y_lin") + mu_bias_out) {
        return false;
    }

    const std::vector<std::pair<Commitment, FieldElement>> external_commitments = {
        {dynamic_commitments.at("P_H_prime"), external_value(proof, "mu_proj")},
        {dynamic_commitments.at("P_E_src"), external_value(proof, "mu_src")},
        {dynamic_commitments.at("P_E_dst"), external_value(proof, "mu_dst")},
        {dynamic_commitments.at("P_H_star"), external_value(proof, "mu_star")},
        {dynamic_commitments.at("P_H_agg_star"), external_value(proof, "mu_agg")},
        {dynamic_commitments.at("P_Y_lin"), external_value(proof, "mu_Y_lin")},
        {dynamic_commitments.at("P_Y"), external_value(proof, "mu_out")},
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

    const auto* bundle_fh = bundle_by_name(proof, "FH");
    const auto* bundle_edge = bundle_by_name(proof, "edge");
    const auto* bundle_in = bundle_by_name(proof, "in");
    const auto* bundle_d = bundle_by_name(proof, "d");
    const auto* bundle_n = bundle_by_name(proof, "N");

    const auto eval = [&](const std::string& name, const FieldElement& point) -> FieldElement {
        if (context.public_polynomials.contains(name)) {
            return context.public_polynomials.at(name).evaluate(point);
        }
        if (name == "t_FH" || name == "P_Table_feat" || name == "P_Query_feat" || name == "P_m_feat" || name == "P_R_feat") {
            return bundle_value(*bundle_fh, name, point);
        }
        if (name == "t_edge"
            || name == "P_E_src_edge"
            || name == "P_H_src_star_edge"
            || name == "P_Query_src"
            || name == "P_R_src"
            || name == "P_E_dst_edge"
            || name == "P_M_edge"
            || name == "P_Sum_edge"
            || name == "P_inv_edge"
            || name == "P_H_agg_star_edge"
            || name == "P_Query_dst"
            || name == "P_R_dst"
            || name == "P_Table_L"
            || name == "P_Query_L"
            || name == "P_m_L"
            || name == "P_R_L"
            || name == "P_Table_R"
            || name == "P_Query_R"
            || name == "P_m_R"
            || name == "P_R_R"
            || name == "P_Table_exp"
            || name == "P_Query_exp"
            || name == "P_m_exp"
            || name == "P_R_exp"
            || name == "P_Delta"
            || name == "P_S"
            || name == "P_Z"
            || name == "P_s_max"
            || name == "P_C_max"
            || name == "P_alpha"
            || name == "P_U"
            || name == "P_v_hat"
            || name == "P_PSQ"
            || name == "P_w_psq"
            || name == "P_T_psq_edge") {
            return bundle_value(*bundle_edge, name, point);
        }
        if (name == "t_in" || name == "P_a_proj" || name == "P_b_proj" || name == "P_Acc_proj") {
            return bundle_value(*bundle_in, name, point);
        }
        if (name == "t_d"
            || name == "P_a_src"
            || name == "P_b_src"
            || name == "P_Acc_src"
            || name == "P_a_dst"
            || name == "P_b_dst"
            || name == "P_Acc_dst"
            || name == "P_a_star"
            || name == "P_b_star"
            || name == "P_Acc_star"
            || name == "P_a_agg"
            || name == "P_b_agg"
            || name == "P_Acc_agg"
            || name == "P_a_out"
            || name == "P_b_out"
            || name == "P_Acc_out") {
            return bundle_value(*bundle_d, name, point);
        }
        if (name == "t_N"
            || name == "P_E_src"
            || name == "P_H_star"
            || name == "P_Table_src"
            || name == "P_m_src"
            || name == "P_R_src_node"
            || name == "P_E_dst"
            || name == "P_M"
            || name == "P_Sum"
            || name == "P_inv"
            || name == "P_H_agg_star"
            || name == "P_Table_dst"
            || name == "P_m_dst"
            || name == "P_R_dst_node") {
            return bundle_value(*bundle_n, name, point);
        }
        throw std::runtime_error("missing evaluation in verifier: " + name);
    };

    const std::map<std::string, FieldElement> external_evals = {
        {"mu_proj", external_value(proof, "mu_proj")},
        {"mu_src", external_value(proof, "mu_src")},
        {"mu_dst", external_value(proof, "mu_dst")},
        {"mu_star", external_value(proof, "mu_star")},
        {"mu_agg", external_value(proof, "mu_agg")},
        {"mu_Y_lin", external_value(proof, "mu_Y_lin")},
        {"mu_out", external_value(proof, "mu_out")},
    };

    if (evaluate_t_fh(context, challenges, eval, challenges.at("z_FH")) != eval("t_FH", challenges.at("z_FH"))) {
        return false;
    }
    if (evaluate_t_edge(context, challenges, witness_scalars, eval, challenges.at("z_edge")) != eval("t_edge", challenges.at("z_edge"))) {
        return false;
    }
    if (evaluate_t_n(context, challenges, witness_scalars, eval, challenges.at("z_N")) != eval("t_N", challenges.at("z_N"))) {
        return false;
    }
    if (evaluate_t_in(context, challenges, external_evals, eval, challenges.at("z_in")) != eval("t_in", challenges.at("z_in"))) {
        return false;
    }
    if (evaluate_t_d(context, challenges, external_evals, eval, challenges.at("z_d")) != eval("t_d", challenges.at("z_d"))) {
        return false;
    }

    auto prepare_bundle = [&](const DomainOpeningBundle& bundle, const std::string& challenge_name) {
        BundleVerificationInput input;
        input.bundle = &bundle;
        input.folding_challenge = challenges.at(challenge_name);
        input.commitments.reserve(bundle.values.size());
        input.values.reserve(bundle.values.size());
        for (const auto& [label, opened_values] : bundle.values) {
            if (dynamic_commitments.contains(label)) {
                input.commitments.push_back(dynamic_commitments.at(label));
            } else {
                input.commitments.push_back(quotient_commitments.at(label));
            }
            input.values.push_back(opened_values);
        }
        return input;
    };

    auto verify_bundle = [&](const BundleVerificationInput& input) {
        return crypto::KZG::verify_batch(
            input.commitments,
            input.bundle->points,
            input.values,
            input.folding_challenge,
            input.bundle->witness,
            context.kzg);
    };

    const auto prepared_fh = prepare_bundle(*bundle_fh, "v_FH");
    const auto prepared_edge = prepare_bundle(*bundle_edge, "v_edge");
    const auto prepared_in = prepare_bundle(*bundle_in, "v_in");
    const auto prepared_d = prepare_bundle(*bundle_d, "v_d");
    const auto prepared_n = prepare_bundle(*bundle_n, "v_N");

    const bool run_parallel_verification = std::thread::hardware_concurrency() > 1;
    if (run_parallel_verification) {
        auto external_future = std::async(
            std::launch::async,
            [&]() {
                return crypto::KZG::verify_external_fold(
                    external_commitments,
                    external_points,
                    challenges.at("rho_ext"),
                    proof.external_witness,
                    context.kzg);
            });
        auto fh_future = std::async(std::launch::async, [&]() { return verify_bundle(prepared_fh); });
        auto edge_future = std::async(std::launch::async, [&]() { return verify_bundle(prepared_edge); });
        auto in_future = std::async(std::launch::async, [&]() { return verify_bundle(prepared_in); });
        auto d_future = std::async(std::launch::async, [&]() { return verify_bundle(prepared_d); });
        auto n_future = std::async(std::launch::async, [&]() { return verify_bundle(prepared_n); });
        if (!external_future.get() || !fh_future.get() || !edge_future.get() || !in_future.get() || !d_future.get() || !n_future.get()) {
            return false;
        }
    } else {
        if (!crypto::KZG::verify_external_fold(
                external_commitments,
                external_points,
                challenges.at("rho_ext"),
                proof.external_witness,
                context.kzg)) {
            return false;
        }
        if (!verify_bundle(prepared_fh)
            || !verify_bundle(prepared_edge)
            || !verify_bundle(prepared_in)
            || !verify_bundle(prepared_d)
            || !verify_bundle(prepared_n)) {
            return false;
        }
    }

    return true;
}

}  // namespace gatzk::protocol
