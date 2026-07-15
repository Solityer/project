#include "gatzk/protocol/verifier.hpp"

#include <algorithm>
#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "gatzk/algebra/domain_evaluation.hpp"
#include "gatzk/algebra/eval_backend.hpp"
#include "gatzk/data/loader.hpp"
#include "gatzk/protocol/challenges.hpp"
#include "gatzk/protocol/quotients.hpp"
#include "gatzk/protocol/schema.hpp"
#include "gatzk/util/logging.hpp"
#include "gatzk/util/route2.hpp"

namespace gatzk::protocol
{
    // 本文件实现 verifier 主链：
    // 它会重放 transcript、检查公开元数据契约、批量校验各域 opening，并重算 quotient 与外点关系。
    // 这里允许做求值缓存和并行检查，但任何优化都不能改变验证方程或公开对象集合。

    namespace
    {
        using algebra::FieldElement;
        using crypto::Commitment;
        using Clock = std::chrono::steady_clock;

        bool public_u64_column_matches(
            const ProtocolContext &context,
            const std::string &label,
            const std::vector<std::size_t> &expected)
        {
            const auto it = context.public_polynomials.find(label);
            if (it == context.public_polynomials.end() || it->second.size() < expected.size())
            {
                return false;
            }
            const auto &values = it->second.values();
            for (std::size_t i = 0; i < expected.size(); ++i)
            {
                if (!(values[i] == FieldElement(static_cast<std::uint64_t>(expected[i]))))
                {
                    return false;
                }
            }
            return true;
        }

        bool public_statement_matches_registered_columns(const ProtocolContext &context)
        {
            std::vector<std::size_t> src;
            std::vector<std::size_t> dst;
            src.reserve(context.local.public_input.src.size());
            dst.reserve(context.local.public_input.dst.size());
            for (const auto value : context.local.public_input.src) src.push_back(value);
            for (const auto value : context.local.public_input.dst) dst.push_back(value);
            return public_u64_column_matches(context, "P_I", context.local.public_input.I)
                && public_u64_column_matches(context, "P_src", src)
                && public_u64_column_matches(context, "P_dst", dst);
        }

        double elapsed_ms(const Clock::time_point &start, const Clock::time_point &end)
        {
            return std::chrono::duration<double, std::milli>(end - start).count();
        }

        std::string verifier_context_cache_key(const util::AppConfig &config)
        {
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
                   << config.srs_id << '|'
                   << config.srs_type << '|'
                   << config.privacy_policy << '|'
                   << config.input_privacy << '|'
                   << config.output_mode << '|'
                   << config.opening_mode << '|'
                   << config.binding_mode << '|'
                   << config.proof_task << '|'
                   << (config.allow_synthetic_model ? "synthetic" : "formal");
            for (const auto input_dim : config.d_in_profile)
            {
                stream << "|din=" << input_dim;
            }
            for (const auto &layer : config.hidden_profile)
            {
                stream << "|hid=" << layer.head_count << 'x' << layer.head_dim;
            }
            return stream.str();
        }

        std::string verifier_quotient_cache_key(
            const ProtocolContext &context,
            const Proof &proof,
            const std::string &quotient_name)
        {
            std::ostringstream stream;
            stream << verifier_context_cache_key(context.config) << "|verify-quot|" << quotient_name;
            for (const auto &[name, value] : proof.challenges)
            {
                stream << '|' << name << '=' << value.to_string();
            }
            for (const auto &[name, commitment] : proof.quotient_commitments)
            {
                stream << '|' << name << '=' << commitment.point.to_string();
            }
            for (const auto &[name, value] : proof.witness_scalars)
            {
                stream << '|' << name << '=' << value.to_string();
            }
            for (const auto &[name, value] : proof.external_evaluations)
            {
                stream << '|' << name << '=' << value.to_string();
            }
            return stream.str();
        }

        bool full_graph_feature_identity_enabled(const ProtocolContext &context)
        {
            if (context.local.num_nodes != context.dataset.num_nodes || context.local.num_features != context.dataset.num_features)
            {
                return false;
            }
            for (std::size_t i = 0; i < context.local.absolute_ids.size(); ++i)
            {
                if (context.local.absolute_ids[i] != i)
                {
                    return false;
                }
            }
            return true;
        }

        bool lazy_large_fh_public_enabled(const ProtocolContext &context)
        {
            constexpr std::size_t kMinLazyLargeFhDomainSize = 1ULL << 24;
            return context.config.batching_rule == "whole_graph_single" && context.local.num_nodes == context.dataset.num_nodes && context.local.num_features == context.dataset.num_features && context.domains.fh != nullptr && context.domains.fh->size >= kMinLazyLargeFhDomainSize;
        }

        bool is_lazy_large_fh_public_label(const std::string &name)
        {
            return name == "P_T_H" || name == "P_Row_feat_tbl" || name == "P_Col_feat_tbl" || name == "P_Row_feat_qry" || name == "P_Col_feat_qry" || name == "P_I_feat_qry" || name == "P_Q_tbl_feat" || name == "P_Q_qry_feat";
        }

        template <typename ValueFn>
        FieldElement evaluate_truncated_domain_polynomial(
            const std::shared_ptr<algebra::RootOfUnityDomain> &domain,
            std::size_t valid_count,
            const FieldElement &point,
            ValueFn &&value_fn)
        {
            if (domain == nullptr)
            {
                throw std::runtime_error("lazy verifier public polynomial requires a domain");
            }
            if (valid_count == 0)
            {
                return FieldElement::zero();
            }
            if (const auto shift = domain->rotation_shift(FieldElement::one(), point); shift.has_value())
            {
                return *shift < valid_count ? value_fn(*shift) : FieldElement::zero();
            }
            const auto zero_eval = domain->zero_polynomial_eval(point);
            FieldElement sum = FieldElement::zero();
            const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
            if (valid_count >= (1ULL << 20) && cpu_count > 1)
            {
                const auto task_count = std::min<std::size_t>(cpu_count, (valid_count + (1ULL << 20) - 1) / (1ULL << 20));
                const auto chunk_size = (valid_count + task_count - 1) / task_count;
                std::vector<std::future<FieldElement>> futures;
                futures.reserve(task_count);
                for (std::size_t task = 0; task < task_count; ++task)
                {
                    const auto begin = task * chunk_size;
                    const auto end = std::min(valid_count, begin + chunk_size);
                    if (begin >= end)
                    {
                        break;
                    }
                    futures.push_back(std::async(
                        std::launch::async,
                        [&, begin, end]()
                        {
                            FieldElement partial = FieldElement::zero();
                            auto omega_power = domain->omega.pow(static_cast<std::uint64_t>(begin));
                            for (std::size_t index = begin; index < end; ++index)
                            {
                                partial += value_fn(index) * omega_power / (point - omega_power);
                                omega_power *= domain->omega;
                            }
                            return partial;
                        }));
                }
                for (auto &future : futures)
                {
                    sum += future.get();
                }
            }
            else
            {
                FieldElement omega_power = FieldElement::one();
                for (std::size_t index = 0; index < valid_count; ++index)
                {
                    sum += value_fn(index) * omega_power / (point - omega_power);
                    omega_power *= domain->omega;
                }
            }
            return zero_eval * domain->inv_size * sum;
        }

        FieldElement evaluate_lazy_large_fh_public_poly(
            const ProtocolContext &context,
            const std::string &name,
            const FieldElement &point)
        {
            if (!lazy_large_fh_public_enabled(context) || !is_lazy_large_fh_public_label(name))
            {
                throw std::runtime_error("unsupported verifier lazy public polynomial: " + name);
            }
            const auto &domain = context.domains.fh;
            const std::size_t dataset_rows = context.dataset.num_nodes;
            const std::size_t local_rows = context.local.num_nodes;
            const std::size_t feature_count = context.local.num_features;
            const std::size_t dataset_valid = dataset_rows * feature_count;
            const std::size_t local_valid = local_rows * feature_count;
            const auto &feature_matrix = !context.dataset.features.empty() ? context.dataset.features : context.local.features;
            if (name == "P_Q_tbl_feat")
            {
                return evaluate_truncated_domain_polynomial(domain, dataset_valid, point, [](std::size_t)
                                                            { return FieldElement::one(); });
            }
            if (name == "P_Q_qry_feat")
            {
                return evaluate_truncated_domain_polynomial(domain, local_valid, point, [](std::size_t)
                                                            { return FieldElement::one(); });
            }
            if (name == "P_Row_feat_tbl")
            {
                return evaluate_truncated_domain_polynomial(domain, dataset_valid, point, [&](std::size_t index)
                                                            { return FieldElement(index / feature_count); });
            }
            if (name == "P_Col_feat_tbl")
            {
                return evaluate_truncated_domain_polynomial(domain, dataset_valid, point, [&](std::size_t index)
                                                            { return FieldElement(index % feature_count); });
            }
            if (name == "P_Row_feat_qry")
            {
                return evaluate_truncated_domain_polynomial(domain, local_valid, point, [&](std::size_t index)
                                                            { return FieldElement(index / feature_count); });
            }
            if (name == "P_Col_feat_qry")
            {
                return evaluate_truncated_domain_polynomial(domain, local_valid, point, [&](std::size_t index)
                                                            { return FieldElement(index % feature_count); });
            }
            if (name == "P_I_feat_qry")
            {
                return evaluate_truncated_domain_polynomial(domain, local_valid, point, [&](std::size_t index)
                                                            { return FieldElement(context.local.absolute_ids[index / feature_count]); });
            }
            if (name == "P_T_H")
            {
                return evaluate_truncated_domain_polynomial(domain, dataset_valid, point, [&](std::size_t index)
                                                            {
            const auto row = index / feature_count;
            const auto column = index % feature_count;
            return feature_matrix[row][column]; });
            }
            throw std::runtime_error("unsupported verifier lazy public polynomial label: " + name);
        }

        class PublicEvaluationBackendRegistry
        {
        public:
            explicit PublicEvaluationBackendRegistry(const ProtocolContext &context)
            {
                std::unordered_map<std::string, PolynomialBatchGroup> groups;
                for (const auto &[name, polynomial] : context.public_polynomials)
                {
                    if (polynomial.basis != algebra::PolynomialBasis::Evaluation || polynomial.domain == nullptr)
                    {
                        continue;
                    }
                    const auto domain_key = polynomial.domain->name + ":" + std::to_string(polynomial.domain->size);
                    auto &group = groups[domain_key];
                    if (group.polynomials.empty())
                    {
                        group.domain = polynomial.domain;
                    }
                    group.polynomials.push_back({name, &polynomial});
                    group.labels.push_back(name);
                }

                for (auto &[domain_key, group] : groups)
                {
                    auto [it, inserted] = backends_.emplace(
                        domain_key,
                        algebra::PackedEvaluationBackend(group.domain, std::move(group.polynomials)));
                    (void)inserted;
                    labels_.emplace(domain_key, std::move(group.labels));
                    domain_keys_.emplace(group.domain.get(), domain_key);
                }
            }

            const algebra::PackedEvaluationBackend *find(const std::shared_ptr<algebra::RootOfUnityDomain> &domain) const
            {
                const auto it = domain_keys_.find(domain.get());
                if (it == domain_keys_.end())
                {
                    return nullptr;
                }
                if (const auto backend = backends_.find(it->second); backend != backends_.end())
                {
                    return &backend->second;
                }
                return nullptr;
            }

            const std::vector<std::string> *labels(const std::shared_ptr<algebra::RootOfUnityDomain> &domain) const
            {
                const auto it = domain_keys_.find(domain.get());
                if (it == domain_keys_.end())
                {
                    return nullptr;
                }
                if (const auto labels = labels_.find(it->second); labels != labels_.end())
                {
                    return &labels->second;
                }
                return nullptr;
            }

        private:
            struct PolynomialBatchGroup
            {
                std::shared_ptr<algebra::RootOfUnityDomain> domain;
                std::vector<std::pair<std::string, const algebra::Polynomial *>> polynomials;
                std::vector<std::string> labels;
            };

            std::unordered_map<std::string, algebra::PackedEvaluationBackend> backends_;
            std::unordered_map<std::string, std::vector<std::string>> labels_;
            std::unordered_map<const algebra::RootOfUnityDomain *, std::string> domain_keys_;
        };

        class SharedVerifierBackendRegistryCache
        {
        public:
            std::shared_ptr<const PublicEvaluationBackendRegistry> get_or_build(
                const std::string &key,
                const std::function<std::shared_ptr<const PublicEvaluationBackendRegistry>()> &build)
            {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (const auto it = entries_.find(key); it != entries_.end())
                    {
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
            std::unordered_map<std::string, std::shared_ptr<const PublicEvaluationBackendRegistry>> entries_;
        };

        SharedVerifierBackendRegistryCache &shared_verifier_backend_registry_cache()
        {
            static SharedVerifierBackendRegistryCache cache;
            return cache;
        }

        struct OpenedValueView
        {
            const std::vector<FieldElement> *points = nullptr;
            const std::vector<FieldElement> *values = nullptr;
        };

        class VerifierEvaluationMemoization
        {
        public:
            VerifierEvaluationMemoization(
                const ProtocolContext &context,
                const std::unordered_map<std::string, OpenedValueView> &opened_values,
                std::shared_ptr<const PublicEvaluationBackendRegistry> backend_registry,
                RunMetrics *metrics = nullptr)
                : context_(context),
                  opened_values_(opened_values),
                  backend_registry_(std::move(backend_registry)),
                  metrics_(metrics) {}

            void prefetch_public_domain_points(
                const std::shared_ptr<algebra::RootOfUnityDomain> &domain,
                const std::vector<FieldElement> &points)
            {
                if (domain == nullptr || points.empty() || backend_registry_ == nullptr)
                {
                    return;
                }

                const auto *backend = backend_registry_->find(domain);
                const auto *labels = backend_registry_->labels(domain);
                if (backend == nullptr || labels == nullptr)
                {
                    return;
                }

                bool all_precomputed = true;
                for (const auto &point : points)
                {
                    if (!precomputed_domain_points_.contains(algebra::domain_point_cache_key(domain, point)))
                    {
                        all_precomputed = false;
                        break;
                    }
                }
                if (all_precomputed)
                {
                    return;
                }

                const auto prefetch_start = Clock::now();
                std::vector<std::size_t> rotations;
                rotations.reserve(points.size());
                bool all_rotations = true;
                for (const auto &point : points)
                {
                    const auto shift = domain->rotation_shift(points.front(), point);
                    if (!shift.has_value())
                    {
                        all_rotations = false;
                        break;
                    }
                    rotations.push_back(*shift);
                }

                if (all_rotations)
                {
                    const auto &weight_entry = domain_weight_cache_.get(domain, points.front());
                    if (!weight_entry.direct_index.has_value())
                    {
                        const auto rotated_values = backend->evaluate_with_native_weight_rotations(
                            *labels,
                            weight_entry.native_weights,
                            rotations);
                        for (std::size_t point_index = 0; point_index < points.size(); ++point_index)
                        {
                            const auto point_key = algebra::point_cache_key(points[point_index]);
                            for (std::size_t label_index = 0; label_index < labels->size(); ++label_index)
                            {
                                value_cache_.emplace((*labels)[label_index] + "@" + point_key, rotated_values[point_index][label_index]);
                            }
                            precomputed_domain_points_.emplace(algebra::domain_point_cache_key(domain, points[point_index]));
                        }
                        if (metrics_ != nullptr)
                        {
                            metrics_->verify_public_eval_ms += elapsed_ms(prefetch_start, Clock::now());
                        }
                        return;
                    }
                }

                for (const auto &point : points)
                {
                    precompute_public_domain_point(domain, *backend, *labels, point);
                }
                if (metrics_ != nullptr)
                {
                    metrics_->verify_public_eval_ms += elapsed_ms(prefetch_start, Clock::now());
                }
            }

            FieldElement eval_named(const std::string &name, const FieldElement &point)
            {
                const auto cache_key = name + "@" + algebra::point_cache_key(point);
                if (const auto it = value_cache_.find(cache_key); it != value_cache_.end())
                {
                    return it->second;
                }

                FieldElement value = FieldElement::zero();
                if (const auto it = context_.public_polynomials.find(name); it != context_.public_polynomials.end())
                {
                    value = eval_public_polynomial(it->second, point);
                }
                else if (lazy_large_fh_public_enabled(context_) && is_lazy_large_fh_public_label(name))
                {
                    batch_evaluate_lazy_fh_labels(point);
                    if (const auto it2 = value_cache_.find(cache_key); it2 != value_cache_.end())
                    {
                        return it2->second;
                    }
                    value = evaluate_lazy_large_fh_public_poly(context_, name, point);
                }
                else if (const auto it = opened_values_.find(name); it != opened_values_.end())
                {
                    const auto lookup_start = Clock::now();
                    value = opened_value(*it->second.points, *it->second.values, name, point);
                    if (metrics_ != nullptr)
                    {
                        metrics_->verify_bundle_lookup_ms += elapsed_ms(lookup_start, Clock::now());
                    }
                }
                else
                {
                    throw std::runtime_error("missing verifier evaluation for " + name);
                }

                value_cache_.emplace(cache_key, value);
                return value;
            }

        private:
            void batch_evaluate_lazy_fh_labels(const FieldElement &point)
            {
                const auto point_key = algebra::point_cache_key(point);
                const auto batch_key = "lazy_fh_batch_done@" + point_key;
                if (lazy_fh_batch_done_.contains(batch_key))
                {
                    return;
                }
                lazy_fh_batch_done_.emplace(batch_key);

                const auto &domain = context_.domains.fh;
                if (domain == nullptr)
                {
                    return;
                }
                const std::size_t dataset_rows = context_.dataset.num_nodes;
                const std::size_t feature_count = context_.local.num_features;
                const std::size_t dataset_valid = dataset_rows * feature_count;
                const bool full_feature_identity = full_graph_feature_identity_enabled(context_);
                const auto &feature_matrix =
                    !context_.dataset.features.empty() ? context_.dataset.features : context_.local.features;

                if (const auto shift = domain->rotation_shift(FieldElement::one(), point); shift.has_value())
                {
                    const bool in_range = *shift < dataset_valid;
                    const auto row = in_range ? *shift / feature_count : 0;
                    const auto column = in_range ? *shift % feature_count : 0;
                    const auto q_value = in_range ? FieldElement::one() : FieldElement::zero();
                    const auto row_value = in_range ? FieldElement(row) : FieldElement::zero();
                    const auto col_value = in_range ? FieldElement(column) : FieldElement::zero();
                    const auto id_value = in_range
                                              ? (full_feature_identity ? FieldElement(row) : FieldElement(context_.local.absolute_ids[row]))
                                              : FieldElement::zero();
                    const auto feature_value = in_range ? feature_matrix[row][column] : FieldElement::zero();
                    value_cache_.emplace("P_Q_tbl_feat@" + point_key, q_value);
                    value_cache_.emplace("P_Q_qry_feat@" + point_key, q_value);
                    value_cache_.emplace("P_Row_feat_tbl@" + point_key, row_value);
                    value_cache_.emplace("P_Row_feat_qry@" + point_key, row_value);
                    value_cache_.emplace("P_Col_feat_tbl@" + point_key, col_value);
                    value_cache_.emplace("P_Col_feat_qry@" + point_key, col_value);
                    value_cache_.emplace("P_I_feat_qry@" + point_key, id_value);
                    value_cache_.emplace("P_T_H@" + point_key, feature_value);
                    return;
                }

                const auto weights = domain->barycentric_weights_native(point);

                std::vector<mcl::Fr> column_fields(feature_count);
                for (std::size_t column = 0; column < feature_count; ++column)
                {
                    column_fields[column] = static_cast<long long>(column);
                }

                constexpr std::size_t kAccumulatorCount = 5;
                std::vector<mcl::Fr> results(kAccumulatorCount);
                for (auto &value : results)
                {
                    value.clear();
                }

                const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
                const bool run_parallel = dataset_valid >= (1ULL << 18) && dataset_rows >= 64 && cpu_count > 1;

                auto accumulate_rows = [&](std::size_t begin_row, std::size_t end_row)
                {
                    std::vector<mcl::Fr> partial(kAccumulatorCount);
                    for (auto &value : partial)
                    {
                        value.clear();
                    }
                    for (std::size_t row = begin_row; row < end_row; ++row)
                    {
                        mcl::Fr row_weight_sum;
                        mcl::Fr column_weight_sum;
                        mcl::Fr feature_weight_sum;
                        row_weight_sum.clear();
                        column_weight_sum.clear();
                        feature_weight_sum.clear();

                        const auto base = row * feature_count;
                        const auto &feature_row = feature_matrix[row];
                        for (std::size_t column = 0; column < feature_count; ++column)
                        {
                            const auto &weight = weights[base + column];
                            mcl::Fr::add(row_weight_sum, row_weight_sum, weight);

                            mcl::Fr column_term;
                            mcl::Fr::mul(column_term, column_fields[column], weight);
                            mcl::Fr::add(column_weight_sum, column_weight_sum, column_term);

                            mcl::Fr feature_term;
                            mcl::Fr::mul(feature_term, feature_row[column].native(), weight);
                            mcl::Fr::add(feature_weight_sum, feature_weight_sum, feature_term);
                        }

                        mcl::Fr::add(partial[0], partial[0], row_weight_sum);

                        mcl::Fr row_term;
                        row_term = static_cast<long long>(row);
                        mcl::Fr::mul(row_term, row_term, row_weight_sum);
                        mcl::Fr::add(partial[1], partial[1], row_term);

                        mcl::Fr::add(partial[2], partial[2], column_weight_sum);

                        if (!full_feature_identity)
                        {
                            mcl::Fr id_term;
                            id_term = static_cast<long long>(context_.local.absolute_ids[row]);
                            mcl::Fr::mul(id_term, id_term, row_weight_sum);
                            mcl::Fr::add(partial[3], partial[3], id_term);
                        }

                        mcl::Fr::add(partial[4], partial[4], feature_weight_sum);
                    }
                    return partial;
                };

                if (run_parallel)
                {
                    const auto task_count = std::min<std::size_t>(cpu_count, dataset_rows);
                    const auto chunk_size = (dataset_rows + task_count - 1) / task_count;
                    std::vector<std::future<std::vector<mcl::Fr>>> futures;
                    futures.reserve(task_count);
                    for (std::size_t t = 0; t < task_count; ++t)
                    {
                        const auto begin = t * chunk_size;
                        const auto end = std::min(dataset_rows, begin + chunk_size);
                        if (begin >= end)
                        {
                            break;
                        }
                        futures.push_back(std::async(std::launch::async, accumulate_rows, begin, end));
                    }
                    for (auto &f : futures)
                    {
                        const auto partial = f.get();
                        for (std::size_t k = 0; k < results.size(); ++k)
                        {
                            mcl::Fr::add(results[k], results[k], partial[k]);
                        }
                    }
                }
                else
                {
                    results = accumulate_rows(0, dataset_rows);
                }

                const auto q_value = FieldElement::from_native(results[0]);
                const auto row_value = FieldElement::from_native(results[1]);
                const auto col_value = FieldElement::from_native(results[2]);
                const auto id_value = full_feature_identity
                                          ? row_value
                                          : FieldElement::from_native(results[3]);
                const auto feature_value = FieldElement::from_native(results[4]);

                value_cache_.emplace("P_Q_tbl_feat@" + point_key, q_value);
                value_cache_.emplace("P_Q_qry_feat@" + point_key, q_value);
                value_cache_.emplace("P_Row_feat_tbl@" + point_key, row_value);
                value_cache_.emplace("P_Row_feat_qry@" + point_key, row_value);
                value_cache_.emplace("P_Col_feat_tbl@" + point_key, col_value);
                value_cache_.emplace("P_Col_feat_qry@" + point_key, col_value);
                value_cache_.emplace("P_I_feat_qry@" + point_key, id_value);
                value_cache_.emplace("P_T_H@" + point_key, feature_value);
            }

            void precompute_public_domain_point(
                const std::shared_ptr<algebra::RootOfUnityDomain> &domain,
                const algebra::PackedEvaluationBackend &backend,
                const std::vector<std::string> &labels,
                const FieldElement &point)
            {
                const auto domain_cache_key = algebra::domain_point_cache_key(domain, point);
                if (precomputed_domain_points_.contains(domain_cache_key))
                {
                    return;
                }

                const auto &weight_entry = domain_weight_cache_.get(domain, point);
                std::vector<FieldElement> values;
                if (weight_entry.direct_index.has_value())
                {
                    values = backend.values_at_direct_index(labels, *weight_entry.direct_index);
                }
                else
                {
                    values = backend.evaluate_with_native_weights(labels, weight_entry.native_weights);
                }
                const auto point_key = algebra::point_cache_key(point);
                for (std::size_t i = 0; i < labels.size(); ++i)
                {
                    value_cache_.emplace(labels[i] + "@" + point_key, values[i]);
                }
                precomputed_domain_points_.emplace(domain_cache_key);
            }

            FieldElement opened_value(
                const std::vector<FieldElement> &points,
                const std::vector<FieldElement> &values,
                const std::string &name,
                const FieldElement &point) const
            {
                for (std::size_t i = 0; i < points.size(); ++i)
                {
                    if (points[i] == point)
                    {
                        return values[i];
                    }
                }
                throw std::runtime_error("missing opened verifier value for " + name);
            }

            FieldElement eval_public_polynomial(const algebra::Polynomial &polynomial, const FieldElement &point)
            {
                const auto eval_start = Clock::now();
                if (polynomial.basis == algebra::PolynomialBasis::Coefficient || polynomial.domain == nullptr || backend_registry_ == nullptr)
                {
                    const auto value = polynomial.evaluate(point);
                    if (metrics_ != nullptr)
                    {
                        metrics_->verify_public_eval_ms += elapsed_ms(eval_start, Clock::now());
                    }
                    return value;
                }

                const auto *backend = backend_registry_->find(polynomial.domain);
                const auto *labels = backend_registry_->labels(polynomial.domain);
                if (backend == nullptr || labels == nullptr)
                {
                    const auto value = polynomial.evaluate(point);
                    if (metrics_ != nullptr)
                    {
                        metrics_->verify_public_eval_ms += elapsed_ms(eval_start, Clock::now());
                    }
                    return value;
                }

                const auto domain_cache_key = algebra::domain_point_cache_key(polynomial.domain, point);
                if (!precomputed_domain_points_.contains(domain_cache_key))
                {
                    precompute_public_domain_point(polynomial.domain, *backend, *labels, point);
                }

                const auto cache_key = polynomial.name + "@" + algebra::point_cache_key(point);
                if (const auto it = value_cache_.find(cache_key); it != value_cache_.end())
                {
                    if (metrics_ != nullptr)
                    {
                        metrics_->verify_public_eval_ms += elapsed_ms(eval_start, Clock::now());
                    }
                    return it->second;
                }
                const auto value = polynomial.evaluate(point);
                if (metrics_ != nullptr)
                {
                    metrics_->verify_public_eval_ms += elapsed_ms(eval_start, Clock::now());
                }
                return value;
            }

            const ProtocolContext &context_;
            const std::unordered_map<std::string, OpenedValueView> &opened_values_;
            std::shared_ptr<const PublicEvaluationBackendRegistry> backend_registry_;
            algebra::DomainEvaluationWeightCache domain_weight_cache_;
            std::unordered_map<std::string, FieldElement> value_cache_;
            std::unordered_set<std::string> precomputed_domain_points_;
            std::unordered_set<std::string> lazy_fh_batch_done_;
            RunMetrics *metrics_ = nullptr;
        };

        class SharedVerifierQuotientCache
        {
        public:
            bool lookup(const std::string &key, FieldElement *out)
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (const auto it = entries_.find(key); it != entries_.end())
                {
                    if (out != nullptr)
                    {
                        *out = it->second;
                    }
                    return true;
                }
                return false;
            }

            void store(const std::string &key, const FieldElement &value)
            {
                std::lock_guard<std::mutex> lock(mutex_);
                entries_.emplace(key, value);
            }

        private:
            std::mutex mutex_;
            std::unordered_map<std::string, FieldElement> entries_;
        };

        SharedVerifierQuotientCache &shared_verifier_quotient_cache()
        {
            static SharedVerifierQuotientCache cache;
            return cache;
        }

        struct BundleVerificationInput
        {
            const DomainOpeningBundle *bundle = nullptr;
            FieldElement folding_challenge = FieldElement::zero();
            std::vector<Commitment> commitments;
            std::vector<std::vector<FieldElement>> values;
        };

        struct IndexedBundleView
        {
            const DomainOpeningBundle *bundle = nullptr;
            std::unordered_map<std::string, const std::vector<FieldElement> *> values_by_label;
        };

        using IndexedBundleMap = std::unordered_map<std::string, IndexedBundleView>;

        IndexedBundleMap index_bundles(const Proof &proof)
        {
            IndexedBundleMap index;
            index.reserve(proof.domain_openings.size());
            for (const auto &[bundle_name, bundle] : proof.domain_openings)
            {
                IndexedBundleView view;
                view.bundle = &bundle;
                view.values_by_label.reserve(bundle.values.size());
                for (const auto &[label, values] : bundle.values)
                {
                    view.values_by_label.emplace(label, &values);
                }
                index.emplace(bundle_name, std::move(view));
            }
            return index;
        }

        const IndexedBundleView &bundle_by_name(const IndexedBundleMap &index, const std::string &name)
        {
            if (const auto it = index.find(name); it != index.end())
            {
                return it->second;
            }
            throw std::runtime_error("missing domain bundle: " + name);
        }

        FieldElement bundle_value(const IndexedBundleView &bundle, const std::string &label, const FieldElement &point)
        {
            const auto values_it = bundle.values_by_label.find(label);
            if (values_it == bundle.values_by_label.end())
            {
                throw std::runtime_error("missing opened value for " + label);
            }
            const auto &values = *values_it->second;
            for (std::size_t i = 0; i < bundle.bundle->points.size(); ++i)
            {
                if (bundle.bundle->points[i] == point)
                {
                    return values[i];
                }
            }
            throw std::runtime_error("missing opened value for " + label);
        }

        FieldElement external_value(const Proof &proof, const std::string &label)
        {
            for (const auto &[name, value] : proof.external_evaluations)
            {
                if (name == label)
                {
                    return value;
                }
            }
            throw std::runtime_error("missing external evaluation: " + label);
        }

        FieldElement bias_fold(const ProtocolContext &context, const FieldElement &y_out)
        {
            FieldElement out = FieldElement::zero();
            FieldElement power = FieldElement::one();
            for (std::size_t i = 0; i < context.local.num_nodes; ++i)
            {
                for (std::size_t j = 0; j < context.model.b.size(); ++j)
                {
                    out += context.model.b[j] * power;
                    power *= y_out;
                }
            }
            return out;
        }

        FieldElement bias_fold_vector(
            const std::vector<double> &bias,
            std::size_t node_count,
            const FieldElement &y_out)
        {
            std::vector<FieldElement> quantized_bias;
            quantized_bias.reserve(bias.size());
            for (const auto value : bias)
            {
                quantized_bias.push_back(FieldElement::from_signed(static_cast<std::int64_t>(
                    value >= 0.0 ? value * 16.0 + 0.5 : value * 16.0 - 0.5)));
            }
            const auto total_iter = node_count * quantized_bias.size();
            const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
            if (total_iter >= (1ULL << 18) && cpu_count > 1)
            {
                const auto step = quantized_bias.size();
                const auto task_count = std::min<std::size_t>(cpu_count, (node_count + 255) / 256);
                const auto chunk = (node_count + task_count - 1) / task_count;
                std::vector<std::future<FieldElement>> futures;
                futures.reserve(task_count);
                for (std::size_t t = 0; t < task_count; ++t)
                {
                    const auto begin = t * chunk;
                    const auto end = std::min(node_count, begin + chunk);
                    if (begin >= end)
                    {
                        break;
                    }
                    futures.push_back(std::async(
                        std::launch::async,
                        [&, begin, end]()
                        {
                            FieldElement partial = FieldElement::zero();
                            FieldElement power = y_out.pow(static_cast<std::uint64_t>(begin * step));
                            for (std::size_t i = begin; i < end; ++i)
                            {
                                for (const auto &quantized : quantized_bias)
                                {
                                    partial += quantized * power;
                                    power *= y_out;
                                }
                            }
                            return partial;
                        }));
                }
                FieldElement out = FieldElement::zero();
                for (auto &f : futures)
                {
                    out += f.get();
                }
                return out;
            }
            FieldElement out = FieldElement::zero();
            FieldElement power = FieldElement::one();
            for (std::size_t i = 0; i < node_count; ++i)
            {
                for (const auto &quantized : quantized_bias)
                {
                    out += quantized * power;
                    power *= y_out;
                }
            }
            return out;
        }

        std::unordered_map<std::string, Commitment> to_map(const std::vector<std::pair<std::string, Commitment>> &entries)
        {
            std::unordered_map<std::string, Commitment> out;
            for (const auto &[name, commitment] : entries)
            {
                out[name] = commitment;
            }
            return out;
        }

        std::map<std::string, FieldElement> to_field_map(const std::vector<std::pair<std::string, FieldElement>> &entries)
        {
            std::map<std::string, FieldElement> out;
            for (const auto &[name, value] : entries)
            {
                out[name] = value;
            }
            return out;
        }

        FieldElement evaluate_coefficient_vector(
            const std::vector<FieldElement> &coefficients,
            const FieldElement &point)
        {
            FieldElement out = FieldElement::zero();
            for (auto it = coefficients.rbegin(); it != coefficients.rend(); ++it)
            {
                out *= point;
                out += *it;
            }
            return out;
        }

        bool verify_public_output_binding(const ProtocolContext &context, const Proof &proof)
        {
            if (proof.public_logits.size() != context.local.num_nodes * context.local.num_classes)
            {
                return false;
            }
            if (proof.public_metadata.public_output != canonical_public_output_descriptor(
                    proof.public_logits,
                    context.local.num_nodes,
                    context.local.num_classes))
            {
                return false;
            }
            const auto output_it = std::find_if(
                proof.dynamic_commitments.begin(),
                proof.dynamic_commitments.end(),
                [](const auto &entry) { return entry.first == "P_Y"; });
            if (output_it == proof.dynamic_commitments.end())
            {
                return false;
            }
            (void)derive_public_output_challenges(
                proof.public_metadata,
                output_it->second,
                proof.public_logits,
                context.local.num_nodes,
                context.local.num_classes);
            return true;
        }

        bool binding_relation_meta_matches(
            const BindingRelationMeta &lhs,
            const BindingRelationMeta &rhs)
        {
            auto indexed_view_matches = [](
                const IndexedViewRelationMeta &a,
                const IndexedViewRelationMeta &b)
            {
                return a.relation_id == b.relation_id
                    && a.canonical_object == b.canonical_object
                    && a.view_object == b.view_object
                    && a.canonical_domain == b.canonical_domain
                    && a.view_domain == b.view_domain
                    && a.canonical_length == b.canonical_length
                    && a.view_length == b.view_length
                    && a.index_map_id == b.index_map_id
                    && a.canonical_selector_id == b.canonical_selector_id
                    && a.query_selector_id == b.query_selector_id
                    && a.multiplicity_id == b.multiplicity_id
                    && a.eta_challenge_label == b.eta_challenge_label
                    && a.beta_challenge_label == b.beta_challenge_label
                    && a.commitment_order == b.commitment_order;
            };
            auto shift_relation_matches = [](
                const KZGShiftRelationMeta &a,
                const KZGShiftRelationMeta &b)
            {
                return a.relation_id == b.relation_id
                    && a.source_linear_maps == b.source_linear_maps
                    && a.source_domains == b.source_domains
                    && a.canonical_points == b.canonical_points
                    && a.lambda0 == b.lambda0
                    && a.deltas == b.deltas
                    && a.source_value_kinds == b.source_value_kinds
                    && a.challenge_label == b.challenge_label
                    && a.exclusion_degree == b.exclusion_degree
                    && a.max_difference_degree == b.max_difference_degree
                    && a.configured_degrees == b.configured_degrees
                    && a.shift_domain_sizes == b.shift_domain_sizes
                    && a.opening_claim_order == b.opening_claim_order;
            };
            const auto indexed_equal = lhs.indexed_views.size() == rhs.indexed_views.size()
                && std::equal(
                    lhs.indexed_views.begin(), lhs.indexed_views.end(),
                    rhs.indexed_views.begin(), indexed_view_matches);
            const auto shift_equal = lhs.shift_relations.size() == rhs.shift_relations.size()
                && std::equal(
                    lhs.shift_relations.begin(), lhs.shift_relations.end(),
                    rhs.shift_relations.begin(), shift_relation_matches);
            return lhs.relation_id == rhs.relation_id
                && lhs.object_kind == rhs.object_kind
                && lhs.layer_id == rhs.layer_id
                && lhs.head_id == rhs.head_id
                && lhs.source_objects == rhs.source_objects
                && lhs.target_objects == rhs.target_objects
                && lhs.work_domains == rhs.work_domains
                && lhs.real_lengths == rhs.real_lengths
                && lhs.quotient_relations == rhs.quotient_relations
                && indexed_equal
                && shift_equal;
        }

        bool validate_pi_bind_model_reblinds(
            const ProtocolContext &context,
            const Proof &proof,
            const std::unordered_map<std::string, crypto::Commitment> &top_level_dynamic,
            std::unordered_map<std::string, crypto::Commitment> *combined)
        {
            if (combined == nullptr) return false;
            *combined = top_level_dynamic;
            const auto expected_meta = build_binding_relation_metadata(context);
            if (proof.pi_bind.subproofs.size() != expected_meta.size()) return false;

            std::size_t global_hidden_head = 0;
            std::unordered_set<std::string> model_object_ids;
            for (std::size_t subproof_index = 0;
                 subproof_index < expected_meta.size(); ++subproof_index)
            {
                const auto &subproof = proof.pi_bind.subproofs[subproof_index];
                if (!binding_relation_meta_matches(
                        subproof.relation_meta, expected_meta[subproof_index]))
                    return false;
                std::vector<std::pair<std::string, std::string>> expected_models;
                if (subproof.relation_meta.object_kind == "hidden")
                {
                    const auto layer = subproof.relation_meta.layer_id.value() - 1;
                    const auto width = context.model.hidden_layers.at(layer).shape.head_dim;
                    for (std::size_t column = 0; column < width; ++column)
                        expected_models.push_back({
                            hidden_weight_column_label(global_hidden_head, column),
                            context.domains.in->name});
                    expected_models.push_back({hidden_src_label(global_hidden_head), context.domains.d->name});
                    expected_models.push_back({hidden_dst_label(global_hidden_head), context.domains.d->name});
                    ++global_hidden_head;
                }
                else if (subproof.relation_meta.object_kind == "output")
                {
                    const auto head = subproof.relation_meta.head_id.value();
                    for (std::size_t column = 0; column < context.local.num_classes; ++column)
                        expected_models.push_back({
                            output_weight_column_label(head, column),
                            context.domains.cat->name});
                    expected_models.push_back({output_src_label(head, false), context.domains.c->name});
                    expected_models.push_back({output_dst_label(head, false), context.domains.c->name});
                    expected_models.push_back({output_bias_label(head, false), context.domains.c->name});
                }
                if (subproof.model_reblinds.size() != expected_models.size()) return false;
                for (std::size_t model_index = 0;
                     model_index < expected_models.size(); ++model_index)
                {
                    const auto &[expected_anchor, expected_domain] = expected_models[model_index];
                    const auto &entry = subproof.model_reblinds[model_index];
                    const auto anchor_it = context.static_commitments.find(expected_anchor);
                    if (entry.object_id != expected_anchor
                        || entry.domain != expected_domain
                        || entry.vk_model_anchor.object_id != expected_anchor
                        || entry.proof_local_commitment.name != model_proof_local_label(expected_anchor)
                        || entry.blinder_commitment.name != model_reblind_witness_label(expected_anchor)
                        || anchor_it == context.static_commitments.end()
                        || entry.blinder_commitment.point == crypto::g1_zero()
                        || !model_object_ids.insert(expected_anchor).second
                        || !crypto::KZG::verify_domain_divisibility(
                            entry.proof_local_commitment,
                            anchor_it->second,
                            entry.blinder_commitment,
                            expected_domain,
                            context.kzg.vk)
                        || !combined->emplace(
                            entry.proof_local_commitment.name,
                            entry.proof_local_commitment).second
                        || !combined->emplace(
                            entry.blinder_commitment.name,
                            entry.blinder_commitment).second)
                    {
                        return false;
                    }
                }
                if (subproof.relation_meta.object_kind == "output")
                {
                    const auto head = subproof.relation_meta.head_id.value();
                    const auto &view_meta = subproof.relation_meta.indexed_views.front();
                    const auto *terminal_meta_ptr = find_unique_shift_relation_by_kind(
                        subproof.relation_meta, "output-bias-terminal");
                    if (terminal_meta_ptr == nullptr) return false;
                    const auto &terminal_meta = *terminal_meta_ptr;
                    const bool same_point = terminal_meta.configured_degrees.empty();
                    const std::vector<std::string> expected_view_references = {
                        view_meta.canonical_object,
                        view_meta.view_object,
                        "public:" + view_meta.canonical_selector_id,
                        "public:" + view_meta.query_selector_id,
                    };
                    std::vector<std::tuple<
                        std::string, std::string, std::size_t, PayloadValueKind>>
                        expected_evaluations = {
                            {output_bias_view_label(head, "m_can") + ":public_bind",
                             "relation", 1, PayloadValueKind::PublicValue},
                            {model_proof_local_label(output_bias_label(head, false)),
                             "C", 2, PayloadValueKind::MaskedPrivateValue},
                            {output_bias_view_label(head, "Table_can"),
                             "C", 2, PayloadValueKind::MaskedPrivateValue},
                            {output_bias_view_label(head, "m_can"),
                             "C", 2, PayloadValueKind::PublicValue},
                            {output_bias_view_label(head, "J_can"),
                             "C", 2, PayloadValueKind::MaskedPrivateValue},
                            {output_bias_view_label(head, "R_can"),
                             "C", 2, PayloadValueKind::MaskedPrivateValue},
                            {output_bias_view_label(head, "Query"),
                             "edge", 2, PayloadValueKind::MaskedPrivateValue},
                            {output_bias_view_label(head, "J_qry"),
                             "edge", 2, PayloadValueKind::MaskedPrivateValue},
                            {output_bias_view_label(head, "R_qry"),
                             "edge", 2, PayloadValueKind::MaskedPrivateValue},
                        };
                    if (!same_point)
                        expected_evaluations.insert(
                            expected_evaluations.begin(),
                            {output_bias_view_label(head, "R_shifted") + ":u_rel",
                             "relation", 1, PayloadValueKind::MaskedPrivateValue});
                    const bool evaluations_match =
                        subproof.evaluations.size() == expected_evaluations.size()
                        && std::equal(
                            subproof.evaluations.begin(),
                            subproof.evaluations.end(),
                            expected_evaluations.begin(),
                            [](const BindingEvaluation &evaluation, const auto &expected)
                            {
                                return evaluation.object_id == std::get<0>(expected)
                                    && evaluation.domain == std::get<1>(expected)
                                    && evaluation.values.size() == std::get<2>(expected)
                                    && evaluation.kind == std::get<3>(expected);
                            });
                    if (subproof.view_references.size()
                            != expected_view_references.size()
                        || !std::equal(
                            subproof.view_references.begin(),
                            subproof.view_references.end(),
                            expected_view_references.begin(),
                            [](const MetadataReference &reference,
                               const std::string &expected)
                            { return reference.object_id == expected; })
                        || !subproof.view_commitments.empty()
                        || !subproof.aggregation_commitments.empty()
                        || !subproof.basis_commitments.empty()
                        || !subproof.crpc_commitments.empty()
                        || subproof.view_logup_commitments.size()
                            != view_meta.commitment_order.size()
                        || !std::equal(
                            subproof.view_logup_commitments.begin(),
                            subproof.view_logup_commitments.end(),
                            view_meta.commitment_order.begin(),
                            [](const auto &entry, const std::string &expected)
                            { return entry.first == expected; })
                        || (same_point
                            ? !subproof.relation_commitments.empty()
                            : (subproof.relation_commitments.size() != 1
                               || subproof.relation_commitments.front().first
                                  != output_bias_view_label(head, "R_shifted")))
                        || !evaluations_match)
                    {
                        return false;
                    }
                }
                else if (!subproof.relation_meta.indexed_views.empty()
                    || !subproof.relation_meta.shift_relations.empty()
                    || !subproof.view_references.empty()
                    || !subproof.view_commitments.empty()
                    || !subproof.view_logup_commitments.empty()
                    || !subproof.aggregation_commitments.empty()
                    || !subproof.basis_commitments.empty()
                    || !subproof.crpc_commitments.empty()
                    || !subproof.relation_commitments.empty()
                    || !subproof.evaluations.empty()
                    || !subproof.opening_references.empty())
                {
                    return false;
                }
                for (const auto* group : {
                         &subproof.view_commitments,
                         &subproof.view_logup_commitments,
                         &subproof.aggregation_commitments,
                         &subproof.basis_commitments,
                         &subproof.crpc_commitments,
                         &subproof.relation_commitments})
                {
                    for (const auto& [name, commitment] : *group)
                    {
                        if (name != commitment.name
                            || !combined->emplace(name, commitment).second)
                            return false;
                    }
                }
            }
            return true;
        }

        bool verify_model_binding_relations(
            const ProtocolContext &context,
            const std::unordered_map<std::string, crypto::Commitment> &dynamic,
            const std::map<std::string, FieldElement> &challenges)
        {
            auto required_dynamic = [&](const std::string &label) -> const crypto::Commitment *
            {
                const auto it = dynamic.find(label);
                return it == dynamic.end() ? nullptr : &it->second;
            };
            auto verify_relation = [&]
            (
                const std::string &source_label,
                const crypto::Commitment &anchor,
                const std::string &blinder_label,
                const std::string &domain_name)
            {
                const auto *source = required_dynamic(source_label);
                const auto *blinder = required_dynamic(blinder_label);
                return source != nullptr && blinder != nullptr
                    && crypto::KZG::verify_domain_divisibility(
                        *source, anchor, *blinder, domain_name, context.kzg.vk);
            };
            auto folded_anchor = [&]
            (
                const std::string &name,
                const std::vector<std::string> &anchor_labels,
                const FieldElement &folding) -> std::optional<crypto::Commitment>
            {
                crypto::Commitment out;
                out.name = name;
                out.point = crypto::g1_zero();
                auto power = FieldElement::one();
                for (const auto &anchor_label : anchor_labels)
                {
                    const auto it = dynamic.find(model_proof_local_label(anchor_label));
                    if (it == dynamic.end()) return std::nullopt;
                    out.point = crypto::g1_add(
                        out.point, crypto::g1_mul(it->second.point, power));
                    power *= folding;
                }
                return out;
            };

            for (std::size_t head = 0; head < context.model.hidden_heads.size(); ++head)
            {
                std::vector<std::string> weight_columns;
                for (std::size_t column = 0;; ++column)
                {
                    const auto label = hidden_weight_column_label(head, column);
                    if (!context.static_commitments.contains(label)) break;
                    weight_columns.push_back(label);
                }
                const auto challenge_it = challenges.find("y_proj_h" + std::to_string(head));
                if (weight_columns.empty() || challenge_it == challenges.end()) return false;
                const auto anchor = folded_anchor(
                    "folded_hidden_weight_proof_" + std::to_string(head),
                    weight_columns, challenge_it->second);
                const auto src_anchor = required_dynamic(
                    model_proof_local_label(hidden_src_label(head)));
                const auto dst_anchor = required_dynamic(
                    model_proof_local_label(hidden_dst_label(head)));
                const auto prefix = hidden_head_prefix(head) + "_";
                if (!anchor.has_value() || src_anchor == nullptr || dst_anchor == nullptr
                    || !verify_relation(prefix + "b_proj", *anchor,
                        hidden_model_binding_blinder_label(head, "proj"), context.domains.in->name)
                    || !verify_relation(prefix + "b_src", *src_anchor,
                        hidden_model_binding_blinder_label(head, "src"), context.domains.d->name)
                    || !verify_relation(prefix + "b_dst", *dst_anchor,
                        hidden_model_binding_blinder_label(head, "dst"), context.domains.d->name))
                    return false;
            }
            for (std::size_t head = 0; head < context.model.output_layer.heads.size(); ++head)
            {
                std::vector<std::string> weight_columns;
                weight_columns.reserve(context.local.num_classes);
                for (std::size_t column = 0; column < context.local.num_classes; ++column)
                {
                    weight_columns.push_back(output_weight_column_label(head, column));
                }
                const auto challenge_it = challenges.find(
                    output_challenge_name("y_proj_out", head, false));
                if (challenge_it == challenges.end()) return false;
                const auto anchor = folded_anchor(
                    "folded_output_weight_proof_" + std::to_string(head),
                    weight_columns,
                    challenge_it->second);
                const auto src_anchor = required_dynamic(
                    model_proof_local_label(output_src_label(head, false)));
                const auto dst_anchor = required_dynamic(
                    model_proof_local_label(output_dst_label(head, false)));
                const auto prefix = output_head_prefix(head, false) + "_";
                if (!anchor.has_value() || src_anchor == nullptr || dst_anchor == nullptr
                    || !verify_relation(
                        prefix + "b_proj", *anchor,
                        output_model_binding_blinder_label(head, "proj"),
                        context.domains.cat->name)
                    || !verify_relation(
                        prefix + "b_src", *src_anchor,
                        output_model_binding_blinder_label(head, "src"),
                        context.domains.c->name)
                    || !verify_relation(
                        prefix + "b_dst", *dst_anchor,
                        output_model_binding_blinder_label(head, "dst"),
                        context.domains.c->name))
                {
                    return false;
                }
            }
            return true;
        }

        PublicMetadata build_public_metadata(const ProtocolContext &context)
        {
            return canonical_public_metadata(context);
        }

        bool metadata_matches(const PublicMetadata &lhs, const PublicMetadata &rhs)
        {
            const auto is_lower_hex_256 = [](const std::string &value)
            {
                return value.size() == 64 && std::all_of(value.begin(), value.end(), [](unsigned char ch)
                {
                    return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
                });
            };
            return lhs.protocol_name == rhs.protocol_name
                && is_lower_hex_256(lhs.proof_nonce)
                && lhs.proof_task == rhs.proof_task
                && lhs.srs_id == rhs.srs_id
                && lhs.protocol_schema_id == rhs.protocol_schema_id
                && is_lower_hex_256(lhs.protocol_schema_id)
                && lhs.instance_schema_hash == rhs.instance_schema_hash
                && is_lower_hex_256(lhs.instance_schema_hash)
                && lhs.srs_type == rhs.srs_type
                && lhs.srs_degree_bound == rhs.srs_degree_bound
                && lhs.kzg_assumption_profile == rhs.kzg_assumption_profile
                && lhs.privacy_policy == rhs.privacy_policy
                && lhs.input_privacy == rhs.input_privacy
                && lhs.output_mode == rhs.output_mode
                && lhs.opening_mode == rhs.opening_mode
                && lhs.binding_mode == rhs.binding_mode
                && lhs.model_arch == rhs.model_arch
                && lhs.model_params == rhs.model_params
                && lhs.static_tables == rhs.static_tables
                && lhs.quant_config == rhs.quant_config
                && lhs.domain_cfg == rhs.domain_cfg
                && lhs.dim_cfg == rhs.dim_cfg
                && lhs.encoding == rhs.encoding
                && lhs.padding_rule == rhs.padding_rule
                && lhs.degree_bound == rhs.degree_bound
                && lhs.security_budget == rhs.security_budget
                && !lhs.public_output.empty();
        }

        struct ExternalEvalSpec
        {
            std::string proof_name;
            std::string label;
            std::string challenge_name;
            bool static_commitment = false;
        };

        std::vector<ExternalEvalSpec> multihead_external_specs(const ProtocolContext &context)
        {
            std::vector<ExternalEvalSpec> specs;
            for (std::size_t head_index = 0; head_index < context.model.hidden_heads.size(); ++head_index)
            {
                const auto suffix = "h" + std::to_string(head_index);
                const auto prefix = "P_h" + std::to_string(head_index) + "_";
                specs.push_back({"mu_" + suffix + "_proj", prefix + "H_prime", "y_proj_h" + std::to_string(head_index)});
                specs.push_back({"mu_" + suffix + "_src", prefix + "E_src", "y_src_h" + std::to_string(head_index)});
                specs.push_back({"mu_" + suffix + "_dst", prefix + "E_dst", "y_dst_h" + std::to_string(head_index)});
                specs.push_back({"mu_" + suffix + "_star", prefix + "H_star", "y_star_h" + std::to_string(head_index)});
                specs.push_back({"mu_" + suffix + "_agg_pre", prefix + "H_agg_pre_star", "y_agg_pre_h" + std::to_string(head_index)});
                specs.push_back({"mu_" + suffix + "_agg", prefix + "H_agg_star", "y_agg_h" + std::to_string(head_index)});
            }
            for (std::size_t layer_index = 0; layer_index < context.model.hidden_layers.size(); ++layer_index)
            {
                const bool is_final_layer = layer_index + 1 == context.model.hidden_layers.size();
                specs.push_back({
                    is_final_layer ? "mu_cat" : "mu_cat_l" + std::to_string(layer_index),
                    hidden_layer_concat_star_label(layer_index, is_final_layer),
                    hidden_concat_y_name(layer_index, is_final_layer),
                });
            }
            const bool legacy_single_output = false;
            for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index)
            {
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
                if (!legacy_single_output)
                {
                    specs.push_back({output_external_eval_name("y_lin", head_index, false), output_y_lin_label(head_index, false), "y_out"});
                    specs.push_back({output_external_eval_name("y", head_index, false), output_y_label(head_index, false), "y_out"});
                }
            }
            specs.push_back({"mu_Y_lin", "P_Y_lin", "y_out"});
            specs.push_back({"mu_out", "P_Y", "y_out"});
            return specs;
        }

        double *verify_domain_metric(RunMetrics *metrics, const std::string &bundle_name)
        {
            if (metrics == nullptr)
            {
                return nullptr;
            }
            if (bundle_name == "FH")
                return &metrics->verify_fh_ms;
            if (bundle_name == "edge")
                return &metrics->verify_edge_ms;
            if (bundle_name == "in")
                return &metrics->verify_in_ms;
            if (bundle_name == "d_h")
                return &metrics->verify_d_h_ms;
            if (bundle_name == "cat")
                return &metrics->verify_cat_ms;
            if (bundle_name == "C")
                return &metrics->verify_c_ms;
            if (bundle_name == "N")
                return &metrics->verify_n_ms;
            return nullptr;
        }

        PayloadValueKind expected_payload_kind_for_label(const ProtocolContext &context, const std::string &label)
        {
            return polynomial_visibility(context, label) == PolynomialVisibility::PrivateWitness
                       ? PayloadValueKind::MaskedPrivateValue
                       : PayloadValueKind::PublicValue;
        }

        bool audit_payload_values(
            const ProtocolContext &context,
            const Proof &proof,
            std::size_t *private_plain_count,
            std::string *reason)
        {
            std::size_t count = 0;
            auto fail = [&](const std::string &value)
            {
                if (private_plain_count != nullptr)
                {
                    *private_plain_count = count;
                }
                if (reason != nullptr)
                {
                    *reason = value;
                }
                return false;
            };
            auto check = [&](const std::string &label, PayloadValueKind actual)
            {
                const auto expected = expected_payload_kind_for_label(context, label);
                if (expected == PayloadValueKind::MaskedPrivateValue && actual == PayloadValueKind::PublicValue)
                {
                    ++count;
                    return false;
                }
                return expected == actual;
            };
            for (const auto &[bundle_name, bundle] : proof.domain_openings)
            {
                (void)bundle_name;
                if (bundle.value_kinds.size() != bundle.values.size())
                {
                    return fail("domain opening payload kind count mismatch");
                }
                for (std::size_t i = 0; i < bundle.values.size(); ++i)
                {
                    if (!check(bundle.values[i].first, bundle.value_kinds[i]))
                    {
                        return fail("domain opening payload kind mismatch: " + bundle.values[i].first);
                    }
                }
            }
            for (const auto &subproof : proof.pi_bind.subproofs)
            {
                for (const auto &evaluation : subproof.evaluations)
                {
                    const bool public_multiplicity =
                        evaluation.object_id.ends_with(
                            "_BiasView_m_can:public_bind");
                    if (public_multiplicity)
                    {
                        if (evaluation.kind != PayloadValueKind::PublicValue)
                            return fail(
                                "Pi_bind public multiplicity kind mismatch");
                    }
                    else
                    {
                        auto source_label = evaluation.object_id;
                        if (source_label.ends_with(":u_rel"))
                            source_label.resize(source_label.size() - 6);
                        if (!check(source_label, evaluation.kind))
                            return fail(
                                "Pi_bind evaluation payload kind mismatch: "
                                + evaluation.object_id);
                    }
                }
            }
            if (proof.external_evaluation_kinds.size() != proof.external_evaluations.size())
            {
                return fail("external evaluation payload kind count mismatch");
            }
            std::unordered_map<std::string, std::string> external_sources;
            if (context.model.has_real_multihead)
            {
                for (const auto &spec : multihead_external_specs(context))
                {
                    external_sources.emplace(spec.proof_name, spec.label);
                }
            }
            else
            {
                external_sources.emplace("mu_proj", "P_H_prime");
                external_sources.emplace("mu_src", "P_E_src");
                external_sources.emplace("mu_dst", "P_E_dst");
                external_sources.emplace("mu_star", "P_H_star");
                external_sources.emplace("mu_agg", "P_H_agg_star");
                external_sources.emplace("mu_Y_lin", "P_Y_lin");
                external_sources.emplace("mu_out", "P_Y");
            }
            for (std::size_t i = 0; i < proof.external_evaluations.size(); ++i)
            {
                const auto &name = proof.external_evaluations[i].first;
                const auto it = external_sources.find(name);
                if (it == external_sources.end())
                {
                    return fail("unknown external evaluation payload: " + name);
                }
                if (!check(it->second, proof.external_evaluation_kinds[i]))
                {
                    return fail("external evaluation payload kind mismatch: " + name);
                }
            }
            if (proof.witness_scalar_kinds.size() != proof.witness_scalars.size())
            {
                return fail("witness scalar payload kind count mismatch");
            }
            for (std::size_t i = 0; i < proof.witness_scalars.size(); ++i)
            {
                if (proof.witness_scalar_kinds[i] == PayloadValueKind::PublicValue)
                {
                    ++count;
                    return fail("private witness scalar exposed as public: " + proof.witness_scalars[i].first);
                }
            }
            if (private_plain_count != nullptr)
            {
                *private_plain_count = count;
            }
            if (reason != nullptr)
            {
                reason->clear();
            }
            return true;
        }

    } // namespace

    std::size_t private_plain_value_count(const ProtocolContext &context, const Proof &proof)
    {
        std::size_t count = 0;
        std::string reason;
        if (!audit_payload_values(context, proof, &count, &reason))
        {
            return count == 0 ? std::numeric_limits<std::size_t>::max() : count;
        }
        return count;
    }

    // verifier 入口：重放 transcript、批量校验 opening，并确认 quotient / external relation 全部成立。
    bool verify(const ProtocolContext &context, const Proof &proof, RunMetrics *metrics)
    {
        // verify 必须重放 transcript，并按固定 block 顺序检查公开元数据、quotient 与 opening 关系。
        const bool debug_verify = std::getenv("GATZK_DEBUG_VERIFY") != nullptr;
        const auto fail = [&](const std::string &reason)
        {
            if (debug_verify)
            {
                util::info("verify_fail=" + reason);
            }
            return false;
        };
        const auto metadata_start = Clock::now();
        if (!context.model.has_real_multihead)
        {
            return fail("removed_single_output_compatibility_protocol");
        }
        if (context.compiler_artifact == nullptr)
        {
            return fail("missing_protocol_compiler_artifact");
        }
        std::string compiler_artifact_reason;
        if (!validate_protocol_compiler_artifact(
                context,
                *context.compiler_artifact,
                &compiler_artifact_reason))
        {
            return fail("protocol_compiler_artifact:" + compiler_artifact_reason);
        }
        try
        {
            data::validate_normalized_graph_input(context.local);
        }
        catch (const std::exception &)
        {
            return fail("public_graph_boundary");
        }
        if (!public_statement_matches_registered_columns(context))
        {
            return fail("public_input_commitment_mismatch");
        }
        if (!metadata_matches(proof.public_metadata, build_public_metadata(context)))
        {
            return fail("metadata_mismatch");
        }
        if (proof.public_metadata.protocol_name != "zkgat" || proof.public_metadata.proof_task != "node_classification_full_gat" || proof.public_metadata.srs_type != "trusted_srs" || proof.public_metadata.privacy_policy != "model_and_trace_hidden" || proof.public_metadata.input_privacy != "public_graph_public_features" || proof.public_metadata.output_mode != "public_logits" || proof.public_metadata.opening_mode != "point_grouped_batch_opening" || proof.public_metadata.binding_mode != "zk_subproof_for_crpc")
        {
            return fail("metadata_security_profile");
        }
        if (proof.block_order != proof_block_order())
        {
            return fail("block_order_mismatch");
        }
        std::size_t audited_private_plain_count = 0;
        std::string payload_audit_reason;
        if (!audit_payload_values(context, proof, &audited_private_plain_count, &payload_audit_reason))
        {
            return fail("proof_payload_audit:" + payload_audit_reason);
        }
        if (audited_private_plain_count != 0)
        {
            return fail("private_plain_value_count");
        }
        if (!verify_public_output_binding(context, proof))
        {
            return fail("public_output_binding");
        }
        if (metrics != nullptr)
        {
            metrics->verify_metadata_ms += elapsed_ms(metadata_start, Clock::now());
        }
        if (context.model.has_real_multihead)
        {
            const auto transcript_start = Clock::now();
            if (proof.dynamic_commitments.size() != dynamic_commitment_labels(context).size())
            {
                return fail("dynamic_commitment_count");
            }
            for (std::size_t i = 0; i < proof.dynamic_commitments.size(); ++i)
            {
                if (proof.dynamic_commitments[i].first != dynamic_commitment_labels(context)[i])
                {
                    return fail("dynamic_commitment_order");
                }
            }
            if (proof.quotient_commitments.size() != quotient_commitment_labels(context).size())
            {
                return fail("quotient_commitment_count");
            }
            for (std::size_t i = 0; i < proof.quotient_commitments.size(); ++i)
            {
                if (proof.quotient_commitments[i].first != quotient_commitment_labels(context)[i])
                {
                    return fail("quotient_commitment_order");
                }
            }
            const auto top_level_dynamic_commitments = to_map(proof.dynamic_commitments);
            const auto quotient_commitments = to_map(proof.quotient_commitments);
            const auto witness_scalars = to_field_map(proof.witness_scalars);
            const auto external_evaluations = to_field_map(proof.external_evaluations);
            std::unordered_map<std::string, Commitment> dynamic_commitments;
            if (!validate_pi_bind_model_reblinds(
                    context,
                    proof,
                    top_level_dynamic_commitments,
                    &dynamic_commitments))
            {
                return fail("pi_bind_model_reblind");
            }
            const auto challenges = replay_challenges(
                context, dynamic_commitments, quotient_commitments, &proof.public_metadata);
            if (proof.challenges != challenges)
            {
                return fail("challenge_replay_mismatch");
            }
            for (std::size_t head = 0;
                 head < context.model.output_layer.heads.size();
                 ++head)
            {
                const auto relation_id =
                    "bind:output:" + std::to_string(head);
                const auto subproof_it = std::find_if(
                    proof.pi_bind.subproofs.begin(), proof.pi_bind.subproofs.end(),
                    [&](const BindingSubproof &subproof)
                    { return subproof.relation_meta.relation_id == relation_id; });
                if (subproof_it == proof.pi_bind.subproofs.end())
                    return fail("missing_output_binding_open_refs");
                const auto *relation_ptr = find_unique_shift_relation_by_kind(
                    subproof_it->relation_meta, "output-bias-terminal");
                if (relation_ptr == nullptr)
                    return fail("output_binding_shift_relation_registry");
                const auto &relation = *relation_ptr;
                const auto y_rel = challenges.at(relation.challenge_label);
                std::vector<std::string> expected_references;
                auto append_unique_reference = [&](std::string reference)
                {
                    if (std::find(
                            expected_references.begin(),
                            expected_references.end(), reference)
                        == expected_references.end())
                        expected_references.push_back(std::move(reference));
                };
                if (relation.configured_degrees.empty())
                {
                    append_unique_reference("W_ext:" + y_rel.to_string());
                    append_unique_reference(
                        "W_ext:" + relation.canonical_points.front().to_string());
                }
                else
                {
                    for (const auto &point : std::vector<FieldElement>{
                             y_rel,
                             y_rel + relation.deltas.at(1),
                             relation.canonical_points.at(0)})
                        append_unique_reference("W_ext:" + point.to_string());
                }
                for (const auto &[domain, point, omega] : std::vector<std::tuple<
                         std::string, FieldElement, FieldElement>>{
                         {"C", challenges.at("z_C"), context.domains.c->omega},
                         {"edge", challenges.at("z_edge"), context.domains.edge->omega}})
                {
                    append_unique_reference(
                        "Open_dom:" + domain + ":" + point.to_string());
                    append_unique_reference(
                        "Open_dom:" + domain + ":" + (point * omega).to_string());
                }
                if (subproof_it->opening_references.size()
                        != expected_references.size()
                    || !std::equal(
                        subproof_it->opening_references.begin(),
                        subproof_it->opening_references.end(),
                        expected_references.begin(),
                        [](const MetadataReference &reference,
                           const std::string &expected)
                        { return reference.object_id == expected; }))
                    return fail("output_binding_open_ref_registry");
            }
            if (!verify_model_binding_relations(context, dynamic_commitments, challenges))
            {
                return fail("model_binding_relation");
            }
            if (metrics != nullptr)
            {
                metrics->verify_transcript_ms += elapsed_ms(transcript_start, Clock::now());
            }
            for (std::size_t head_index = 0; head_index < context.model.hidden_heads.size(); ++head_index)
            {
                if (!witness_scalars.contains("S_src_h" + std::to_string(head_index)) || !witness_scalars.contains("S_dst_h" + std::to_string(head_index)) || !witness_scalars.contains("S_t_h" + std::to_string(head_index)))
                {
                    return fail("missing_hidden_route_scalar");
                }
            }
            const bool legacy_single_output = false;
            for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index)
            {
                if (!witness_scalars.contains(output_witness_scalar_name("src", head_index, legacy_single_output)) || !witness_scalars.contains(output_witness_scalar_name("dst", head_index, legacy_single_output)) || !witness_scalars.contains(output_witness_scalar_name("t", head_index, legacy_single_output)))
                {
                    return fail("missing_output_route_scalar");
                }
            }
            for (const auto &spec : multihead_external_specs(context))
            {
                if (!external_evaluations.contains(spec.proof_name))
                {
                    return fail("missing_external_eval:" + spec.proof_name);
                }
            }
            struct BundleSpec
            {
                std::string bundle_name;
                std::string trace_domain_name;
                std::shared_ptr<algebra::RootOfUnityDomain> domain;
                std::string z_name;
                std::string v_name;
                std::string quotient_name;
            };
            // 每个 bundle 都对应一个工作域的 opening 子证明；名字、点顺序和标签集合都必须与 schema 完全一致。
            const std::vector<BundleSpec> bundle_specs = {
                {"FH", "FH", context.domains.fh, "z_FH", "v_FH", "t_FH"},
                {"edge", "edge", context.domains.edge, "z_edge", "v_edge", "t_edge"},
                {"in", "in", context.domains.in, "z_in", "v_in", "t_in"},
                {"d_h", "d", context.domains.d, "z_d_h", "v_d_h", "t_d_h"},
                {"cat", "cat", context.domains.cat, "z_cat", "v_cat", "t_cat"},
                {"C", "C", context.domains.c, "z_C", "v_C", "t_C"},
                {"N", "N", context.domains.n, "z_N", "v_N", "t_N"},
            };
            const auto indexed_bundles = index_bundles(proof);
            std::unordered_map<std::string, OpenedValueView> opened_values;
            opened_values.reserve(proof.domain_openings.size() * 16);
            for (const auto &[bundle_name, indexed_bundle] : indexed_bundles)
            {
                (void)bundle_name;
                for (const auto &[label, values] : indexed_bundle.values_by_label)
                {
                    opened_values.emplace(
                        label,
                        OpenedValueView{
                            .points = &indexed_bundle.bundle->points,
                            .values = values,
                        });
                }
            }
            auto public_backend_registry = shared_verifier_backend_registry_cache().get_or_build(
                verifier_context_cache_key(context.config),
                [&]()
                {
                    return std::make_shared<PublicEvaluationBackendRegistry>(context);
                });
            VerifierEvaluationMemoization eval_memo(context, opened_values, public_backend_registry, metrics);
            const auto eval = [&](const std::string &name, const FieldElement &point) -> FieldElement
            {
                return eval_memo.eval_named(name, point);
            };
            std::string bundle_failure_reason;
            auto verify_bundle_spec = [&](const BundleSpec &spec)
            {
                // 每个 bundle 都要同时满足：标签集合正确、打开点正确、KZG witness 正确、quotient 重算正确。
                auto bundle_fail = [&](const std::string &reason)
                {
                    bundle_failure_reason = spec.bundle_name + ":" + reason;
                    if (debug_verify)
                    {
                        util::info("bundle_fail=" + bundle_failure_reason);
                    }
                    return false;
                };
                const auto &indexed_bundle = bundle_by_name(indexed_bundles, spec.bundle_name);
                const auto &bundle = *indexed_bundle.bundle;
                auto labels = domain_opening_labels(context, spec.trace_domain_name);
                labels.push_back(spec.quotient_name);
                const auto quotient_mask_name = quotient_mask_label(spec.quotient_name);
                labels.push_back(quotient_mask_name);
                if (bundle.values.size() != labels.size())
                {
                    return bundle_fail("value_count");
                }
                if (bundle.points.size() != 2 || bundle.points[0] != challenges.at(spec.z_name) || bundle.points[1] != challenges.at(spec.z_name) * spec.domain->omega)
                {
                    return bundle_fail("points");
                }
                std::vector<Commitment> commitments;
                std::vector<std::vector<FieldElement>> values;
                const auto bundle_lookup_start = Clock::now();
                for (std::size_t i = 0; i < labels.size(); ++i)
                {
                    if (bundle.values[i].first != labels[i])
                    {
                        return bundle_fail("label:" + bundle.values[i].first + "!=" + labels[i]);
                    }
                    if (dynamic_commitments.contains(labels[i]))
                    {
                        commitments.push_back(dynamic_commitments.at(labels[i]));
                    }
                    else if (quotient_commitments.contains(labels[i]))
                    {
                        commitments.push_back(quotient_commitments.at(labels[i]));
                    }
                    else
                    {
                        return bundle_fail("missing_commitment:" + labels[i]);
                    }
                    values.push_back(bundle.values[i].second);
                }
                if (metrics != nullptr)
                {
                    metrics->verify_bundle_lookup_ms += elapsed_ms(bundle_lookup_start, Clock::now());
                }
                const auto opening_start = Clock::now();
                if (!crypto::KZG::verify_batch(
                        commitments,
                        bundle.points,
                        values,
                        derive_batch_opening_folding_challenge(
                            proof.public_metadata,
                            spec.v_name,
                            commitments,
                            bundle.points,
                            values),
                        bundle.proof,
                        context.kzg.vk))
                {
                    return bundle_fail("kzg_batch");
                }
                if (metrics != nullptr)
                {
                    metrics->verify_domain_opening_ms += elapsed_ms(opening_start, Clock::now());
                }
                const auto quotient_start = Clock::now();
                FieldElement expected_t = FieldElement::zero();
                const auto cache_key = verifier_quotient_cache_key(context, proof, spec.quotient_name);
                const bool quotient_cache_hit = false;
                if (!quotient_cache_hit)
                {
                    if (context.config.batching_rule == "whole_graph_single" && context.local.num_nodes >= (1ULL << 16) && spec.domain != nullptr && spec.domain->size >= (1ULL << 20))
                    {
                        eval_memo.prefetch_public_domain_points(spec.domain, bundle.points);
                    }
                    if (spec.quotient_name == "t_FH")
                    {
                        expected_t = evaluate_t_fh(context, challenges, eval, challenges.at(spec.z_name));
                    }
                    else if (spec.quotient_name == "t_edge")
                    {
                        expected_t = evaluate_t_edge(context, challenges, witness_scalars, external_evaluations, eval, challenges.at(spec.z_name));
                    }
                    else if (spec.quotient_name == "t_in")
                    {
                        expected_t = evaluate_t_in(context, challenges, external_evaluations, eval, challenges.at(spec.z_name));
                    }
                    else if (spec.quotient_name == "t_d_h")
                    {
                        expected_t = evaluate_t_d(context, challenges, external_evaluations, eval, challenges.at(spec.z_name));
                    }
                    else if (spec.quotient_name == "t_cat")
                    {
                        expected_t = evaluate_t_cat(context, challenges, external_evaluations, eval, challenges.at(spec.z_name));
                    }
                    else if (spec.quotient_name == "t_C")
                    {
                        expected_t = evaluate_t_c(context, challenges, external_evaluations, eval, challenges.at(spec.z_name));
                    }
                    else if (spec.quotient_name == "t_N")
                    {
                        expected_t = evaluate_t_n(context, challenges, witness_scalars, eval, challenges.at(spec.z_name));
                    }
                    else
                    {
                        return bundle_fail("unknown_quotient:" + spec.quotient_name);
                    }
                    // A fresh verification recomputes the quotient identity. Reusing an
                    // earlier proof's acceptance-side value is reserved for an explicit
                    // repeat_same_proof_verify benchmark, not this path.
                }
                const auto quotient_point = challenges.at(spec.z_name);
                const auto opened_mask = bundle_value(
                    indexed_bundle,
                    quotient_mask_name,
                    quotient_point);
                const auto expected_masked_quotient =
                    expected_t + spec.domain->zero_polynomial_eval(quotient_point) * opened_mask;
                if (bundle_value(indexed_bundle, spec.quotient_name, quotient_point)
                    != expected_masked_quotient)
                {
                    return bundle_fail("quotient_mismatch:" + spec.quotient_name);
                }
                if (metrics != nullptr)
                {
                    metrics->verify_quotient_ms += elapsed_ms(quotient_start, Clock::now());
                }
                return true;
            };
            std::vector<std::pair<Commitment, FieldElement>> external_commitments;
            std::vector<FieldElement> external_points;
            for (const auto &spec : multihead_external_specs(context))
            {
                if ((!spec.static_commitment && !dynamic_commitments.contains(spec.label))
                    || (spec.static_commitment && !context.static_commitments.contains(spec.label)))
                {
                    return fail("missing_external_commitment:" + spec.label);
                }
                external_commitments.emplace_back(
                    spec.static_commitment ? context.static_commitments.at(spec.label) : dynamic_commitments.at(spec.label),
                    external_value(proof, spec.proof_name));
                external_points.push_back(challenges.at(spec.challenge_name));
            }
            for (std::size_t head_index = 0;
                 head_index < context.model.output_layer.heads.size();
                 ++head_index)
            {
                const auto relation_id =
                    "bind:output:" + std::to_string(head_index);
                const auto subproof_it = std::find_if(
                    proof.pi_bind.subproofs.begin(), proof.pi_bind.subproofs.end(),
                    [&](const BindingSubproof &subproof)
                    { return subproof.relation_meta.relation_id == relation_id; });
                if (subproof_it == proof.pi_bind.subproofs.end())
                    return fail("missing_output_bias_shift_relation");
                const auto *relation_ptr = find_unique_shift_relation_by_kind(
                    subproof_it->relation_meta, "output-bias-terminal");
                if (relation_ptr == nullptr)
                    return fail("missing_output_bias_shift_relation");
                const auto &relation = *relation_ptr;
                if (relation.configured_degrees.empty())
                {
                    const auto r_can_label = output_bias_view_label(
                        head_index, "R_can");
                    const auto r_qry_label = output_bias_view_label(
                        head_index, "R_qry");
                    if (relation.canonical_points.size() != 1
                        || !dynamic_commitments.contains(r_can_label)
                        || !dynamic_commitments.contains(r_qry_label))
                        return fail("invalid_same_point_output_bias_relation");
                    const auto &c_bundle = indexed_bundles.at("C");
                    for (const auto &point : c_bundle.bundle->points)
                    {
                        if (bundle_value(
                                c_bundle,
                                output_bias_view_label(head_index, "m_can"),
                                point)
                            != eval_memo.eval_named(
                                "P_BiasMultiplicity_C", point))
                            return fail("public_bias_multiplicity_domain_value");
                    }
                    const Commitment relation_commitment{
                        .name = output_bias_view_label(
                            head_index, "terminal_relation_virtual"),
                        .point = crypto::g1_add(
                            dynamic_commitments.at(r_can_label).point,
                            crypto::g1_neg(
                                dynamic_commitments.at(r_qry_label).point)),
                    };
                    external_commitments.push_back(
                        {relation_commitment, FieldElement::zero()});
                    external_points.push_back(relation.canonical_points.front());
                    const auto public_m_id = output_bias_view_label(
                        head_index, "m_can") + ":public_bind";
                    const auto public_m_it = std::find_if(
                        subproof_it->evaluations.begin(),
                        subproof_it->evaluations.end(),
                        [&](const BindingEvaluation &evaluation)
                        { return evaluation.object_id == public_m_id; });
                    const auto y_public = challenges.at(relation.challenge_label);
                    if (public_m_it == subproof_it->evaluations.end()
                        || public_m_it->values.size() != 1
                        || public_m_it->kind != PayloadValueKind::PublicValue
                        || public_m_it->values.front()
                            != eval_memo.eval_named(
                                output_bias_view_label(head_index, "m_can"),
                                y_public))
                        return fail("invalid_same_point_public_bias_multiplicity");
                    external_commitments.push_back({
                        dynamic_commitments.at(output_bias_view_label(
                            head_index, "m_can")),
                        public_m_it->values.front()});
                    external_points.push_back(y_public);
                    continue;
                }
                const auto shifted_label =
                    output_bias_view_label(head_index, "R_shifted");
                const auto u_id = shifted_label + ":u_rel";
                const auto u_it = std::find_if(
                    subproof_it->evaluations.begin(), subproof_it->evaluations.end(),
                    [&](const BindingEvaluation &evaluation)
                    { return evaluation.object_id == u_id; });
                if (u_it == subproof_it->evaluations.end()
                    || u_it->domain != "relation"
                    || u_it->values.size() != 1
                    || u_it->kind != PayloadValueKind::MaskedPrivateValue
                    || !dynamic_commitments.contains(shifted_label))
                    return fail("invalid_output_bias_shift_value");
                const bool canonical_is_first =
                    relation.source_domains.at(0) == context.domains.c->name;
                const auto r_can_label =
                    output_bias_view_label(head_index, "R_can");
                const auto r_qry_label =
                    output_bias_view_label(head_index, "R_qry");
                if (!dynamic_commitments.contains(r_can_label)
                    || !dynamic_commitments.contains(r_qry_label))
                    return fail("missing_output_bias_terminal_commitment");
                const auto &c_bundle = indexed_bundles.at("C");
                for (const auto &point : c_bundle.bundle->points)
                {
                    if (bundle_value(
                            c_bundle,
                            output_bias_view_label(head_index, "m_can"),
                            point)
                        != eval_memo.eval_named("P_BiasMultiplicity_C", point))
                        return fail("public_bias_multiplicity_domain_value");
                }
                const auto first_point = canonical_is_first
                    ? dynamic_commitments.at(r_can_label).point
                    : crypto::g1_neg(dynamic_commitments.at(r_qry_label).point);
                const auto second_point = canonical_is_first
                    ? crypto::g1_neg(dynamic_commitments.at(r_qry_label).point)
                    : dynamic_commitments.at(r_can_label).point;
                const Commitment second_source{
                    .name = output_bias_view_prefix(head_index) + "F2",
                    .point = second_point,
                };
                const Commitment relation_commitment{
                    .name = output_bias_view_label(
                        head_index, "terminal_relation_virtual"),
                    .point = crypto::g1_add(
                        first_point, dynamic_commitments.at(shifted_label).point),
                };
                const auto y_rel = challenges.at(relation.challenge_label);
                external_commitments.push_back(
                    {dynamic_commitments.at(shifted_label), u_it->values.front()});
                external_points.push_back(y_rel);
                external_commitments.push_back(
                    {second_source, u_it->values.front()});
                external_points.push_back(y_rel + relation.deltas.at(1));
                external_commitments.push_back(
                    {relation_commitment, FieldElement::zero()});
                external_points.push_back(relation.canonical_points.at(0));

                const auto public_m_id =
                    output_bias_view_label(head_index, "m_can") + ":public_bind";
                const auto public_m_it = std::find_if(
                    subproof_it->evaluations.begin(), subproof_it->evaluations.end(),
                    [&](const BindingEvaluation &evaluation)
                    { return evaluation.object_id == public_m_id; });
                const auto public_poly_it = context.public_commitments.find(
                    "P_BiasMultiplicity_C");
                if (public_m_it == subproof_it->evaluations.end()
                    || public_m_it->domain != "relation"
                    || public_m_it->values.size() != 1
                    || public_m_it->kind != PayloadValueKind::PublicValue
                    || public_poly_it == context.public_commitments.end()
                    || public_poly_it->second.opening_polynomial == nullptr
                    || public_m_it->values.front()
                        != public_poly_it->second.opening_polynomial->evaluate(y_rel))
                    return fail("invalid_public_bias_multiplicity_binding");
                external_commitments.push_back({
                    dynamic_commitments.at(
                        output_bias_view_label(head_index, "m_can")),
                    public_m_it->values.front()});
                external_points.push_back(y_rel);
            }
            const auto output_binding = derive_public_output_challenges(
                proof.public_metadata,
                dynamic_commitments.at("P_Y"),
                proof.public_logits,
                context.local.num_nodes,
                context.local.num_classes);
            external_commitments.emplace_back(
                dynamic_commitments.at("P_Y"),
                evaluate_coefficient_vector(proof.public_logits, output_binding.point));
            external_points.push_back(output_binding.point);
            const auto external_fold_start = Clock::now();
            if (!crypto::KZG::verify_external_fold(
                    external_commitments,
                    external_points,
                    derive_external_opening_folding_challenge(
                        proof.public_metadata,
                        external_commitments,
                        external_points),
                    proof.external_proof,
                    context.kzg.vk))
            {
                return fail("external_fold");
            }
            if (metrics != nullptr)
            {
                metrics->verify_external_fold_ms += elapsed_ms(external_fold_start, Clock::now());
            }
            for (const auto &spec : bundle_specs)
            {
                const auto bundle_start = Clock::now();
                if (!verify_bundle_spec(spec))
                {
                    return fail("bundle_verify:" + (bundle_failure_reason.empty() ? spec.bundle_name : bundle_failure_reason));
                }
                if (metrics != nullptr)
                {
                    const auto elapsed = elapsed_ms(bundle_start, Clock::now());
                    if (auto *metric = verify_domain_metric(metrics, spec.bundle_name); metric != nullptr)
                    {
                        *metric += elapsed;
                    }
                }
            }
            return true;
        }
        return fail("removed_single_output_compatibility_protocol");
        return true;
    }

} // namespace gatzk::protocol
