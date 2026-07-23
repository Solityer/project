#include "gatzk/protocol/prover.hpp"

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <string_view>
#include <thread>
#include <unordered_map>

#include <mcl/ntt.hpp>
#include <openssl/rand.h>

#include "gatzk/algebra/domain_evaluation.hpp"
#include "gatzk/algebra/eval_backend.hpp"
#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/data/loader.hpp"
#include "gatzk/model/gat.hpp"
#include "gatzk/protocol/challenges.hpp"
#include "gatzk/protocol/lookup.hpp"
#include "gatzk/protocol/opening_registry.hpp"
#include "gatzk/protocol/quotients.hpp"
#include "gatzk/protocol/proof_codec.hpp"
#include "gatzk/protocol/schema.hpp"
#include "gatzk/util/logging.hpp"
#include "gatzk/util/route2.hpp"
#include "gatzk/protocol/verifier.hpp"

namespace gatzk::protocol
{
    namespace
    {
        using algebra::FieldElement;
        using algebra::Polynomial;
        using crypto::Commitment;
        using Clock = std::chrono::steady_clock;

        double elapsed_ms(const Clock::time_point &start, const Clock::time_point &end)
        {
            return std::chrono::duration<double, std::milli>(end - start).count();
        }

        bool progress_enabled()
        {
            return std::getenv("GATZK_RUN_PROGRESS") != nullptr;
        }

        void progress_note(const std::string &label)
        {
            if (progress_enabled())
            {
                std::cerr << "[progress] " << label << '\n';
            }
        }

        PayloadValueKind payload_kind_for_label(const ProtocolContext &context, const std::string &label)
        {
            return polynomial_visibility(context, label) == PolynomialVisibility::PrivateWitness
                       ? PayloadValueKind::MaskedPrivateValue
                       : PayloadValueKind::PublicValue;
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

        void attach_public_output_statement(
            const ProtocolContext &context,
            const TraceArtifacts &trace,
            Proof *proof)
        {
            if (proof == nullptr || trace.public_logits.empty())
            {
                throw std::runtime_error("public output binding is missing logits");
            }
            if (!trace.commitments.contains("P_Y"))
            {
                throw std::runtime_error("public output binding is missing P_Y commitment");
            }
            proof->public_logits = trace.public_logits;
            proof->public_metadata.public_output = canonical_public_output_descriptor(
                proof->public_logits,
                context.local.num_nodes,
                context.local.num_classes);
        }

        void strip_prover_material_from_proof(Proof *proof)
        {
            if (proof == nullptr)
            {
                return;
            }
            for (auto &[name, commitment] : proof->dynamic_commitments)
            {
                (void)name;
                commitment.opening_polynomial.reset();
            }
            for (auto &[name, commitment] : proof->quotient_commitments)
            {
                (void)name;
                commitment.opening_polynomial.reset();
            }
            for (auto &subproof : proof->pi_bind.subproofs)
            {
                for (auto &reblind : subproof.model_reblinds)
                {
                    reblind.proof_local_commitment.opening_polynomial.reset();
                    reblind.blinder_commitment.opening_polynomial.reset();
                }
                for (auto *group : {
                         &subproof.view_commitments,
                         &subproof.view_logup_commitments,
                         &subproof.aggregation_commitments,
                         &subproof.basis_commitments,
                         &subproof.crpc_commitments,
                         &subproof.relation_commitments})
                {
                    for (auto &[name, commitment] : *group)
                    {
                        (void)name;
                        commitment.opening_polynomial.reset();
                    }
                }
            }
        }


        Polynomial make_eval_poly(
            const std::string &name,
            std::vector<FieldElement> values,
            const std::shared_ptr<algebra::RootOfUnityDomain> &domain)
        {
            return Polynomial::from_evaluations(name, std::move(values), domain);
        }

        Polynomial make_shared_eval_poly(
            const std::string &name,
            std::vector<FieldElement> values,
            const std::shared_ptr<algebra::RootOfUnityDomain> &domain)
        {
            return Polynomial::from_shared_evaluations(
                name,
                std::make_shared<const std::vector<FieldElement>>(std::move(values)),
                domain);
        }

        Polynomial make_coeff_poly(const std::string &name, std::vector<FieldElement> values)
        {
            return Polynomial::from_coefficients(name, std::move(values));
        }

        void add_public_poly(
            ProtocolContext &context,
            const std::string &name,
            Polynomial polynomial)
        {
            progress_note("context:public:" + name + ":start");
            context.public_commitments[name] = crypto::KZG::commit(name, polynomial, context.kzg);
            polynomial.name = name;
            context.public_polynomials[name] = std::move(polynomial);
            progress_note("context:public:" + name + ":done");
        }

        bool full_graph_feature_identity_enabled(const ProtocolContext &context);

        bool same_polynomial_domain(
            const std::shared_ptr<algebra::RootOfUnityDomain> &lhs,
            const std::shared_ptr<algebra::RootOfUnityDomain> &rhs)
        {
            if (lhs == nullptr || rhs == nullptr)
            {
                return lhs == nullptr && rhs == nullptr;
            }
            return lhs->name == rhs->name && lhs->size == rhs->size && lhs->omega == rhs->omega;
        }

        bool same_polynomial_material(const Polynomial &lhs, const Polynomial &rhs)
        {
            return lhs.basis == rhs.basis
                && same_polynomial_domain(lhs.domain, rhs.domain)
                && lhs.values() == rhs.values();
        }

        void add_public_poly_alias_from_source_checked(
            ProtocolContext &context,
            const std::string &name,
            const std::string &source_name)
        {
            if (!full_graph_feature_identity_enabled(context))
            {
                throw std::runtime_error("public polynomial source alias requires full feature identity: " + name);
            }
            const auto polynomial_it = context.public_polynomials.find(source_name);
            if (polynomial_it == context.public_polynomials.end())
            {
                throw std::runtime_error("missing public polynomial alias source: " + source_name);
            }
            const auto commitment_it = context.public_commitments.find(source_name);
            if (commitment_it == context.public_commitments.end())
            {
                throw std::runtime_error("missing public commitment alias source: " + source_name);
            }
            if (commitment_it->second.opening_polynomial == nullptr)
            {
                throw std::runtime_error("public commitment alias source is missing opening material: " + source_name);
            }
            const auto &source_polynomial = polynomial_it->second;
            const auto &source_material = *commitment_it->second.opening_polynomial;
            const bool same_shared_material =
                source_polynomial.shared_data != nullptr
                && source_material.shared_data != nullptr
                && source_polynomial.shared_data.get() == source_material.shared_data.get()
                && source_polynomial.basis == source_material.basis
                && same_polynomial_domain(source_polynomial.domain, source_material.domain);
            if (!same_shared_material && !same_polynomial_material(source_material, source_polynomial))
            {
                throw std::runtime_error("public commitment alias opening material mismatch: " + name + " -> " + source_name);
            }
            progress_note("context:public_alias:" + name + "->" + source_name + ":start");
            auto alias_polynomial = source_polynomial;
            alias_polynomial.name = name;
            context.public_polynomials[name] = std::move(alias_polynomial);
            auto commitment = commitment_it->second;
            commitment.name = name;
            context.public_commitments[name] = std::move(commitment);
            progress_note("context:public_alias:" + name + "->" + source_name + ":done");
        }

        void trim_coefficients(std::vector<FieldElement> *coefficients)
        {
            while (coefficients->size() > 1 && coefficients->back() == FieldElement::zero())
            {
                coefficients->pop_back();
            }
        }

        std::shared_ptr<const Polynomial> public_fh_coefficient_material(
            const ProtocolContext &context,
            const std::string &name)
        {
            const auto it = context.public_polynomials.find(name);
            if (it == context.public_polynomials.end())
            {
                throw std::runtime_error("missing public FH polynomial for coefficient material: " + name);
            }
            const auto &polynomial = it->second;
            if (polynomial.basis == algebra::PolynomialBasis::Coefficient)
            {
                return std::make_shared<Polynomial>(polynomial);
            }
            if (polynomial.domain == nullptr
                || context.domains.fh == nullptr
                || polynomial.domain->size != context.domains.fh->size
                || !(polynomial.domain->omega == context.domains.fh->omega)
                || polynomial.values().size() != context.domains.fh->size)
            {
                throw std::runtime_error("public FH coefficient material domain mismatch: " + name);
            }

            auto coefficients = polynomial.values();
            auto *native_coefficients = FieldElement::mutable_native_array_data(coefficients.data());
            algebra::ntt_plan(coefficients.size())->intt(native_coefficients);
            trim_coefficients(&coefficients);
            return std::make_shared<Polynomial>(Polynomial::from_coefficients(name, std::move(coefficients)));
        }

        void cache_public_fh_coefficient_material(ProtocolContext &context, const std::string &name)
        {
            progress_note("context:public_fh_coeff:" + name + ":start");
            context.public_fh_coefficient_materials[name] = public_fh_coefficient_material(context, name);
            progress_note("context:public_fh_coeff:" + name + ":done");
        }

        void rebind_public_fh_opening_material(
            ProtocolContext &context,
            const std::string &commitment_name,
            const std::string &material_name)
        {
            const auto material_it = context.public_fh_coefficient_materials.find(material_name);
            if (material_it == context.public_fh_coefficient_materials.end() || material_it->second == nullptr)
            {
                throw std::runtime_error("missing public FH coefficient material for opening rebind: " + material_name);
            }
            const auto commitment_it = context.public_commitments.find(commitment_name);
            if (commitment_it == context.public_commitments.end())
            {
                throw std::runtime_error("missing public FH commitment for opening rebind: " + commitment_name);
            }
            commitment_it->second.opening_polynomial = material_it->second;
        }

        void add_static_commitment(
            ProtocolContext &context,
            const std::string &name,
            Polynomial polynomial)
        {
            progress_note("context:static:" + name + ":start");
            context.static_commitments[name] = crypto::KZG::commit(name, polynomial, context.kzg);
            progress_note("context:static:" + name + ":done");
        }

        std::vector<FieldElement> padded(const std::vector<FieldElement> &values, std::size_t size)
        {
            std::vector<FieldElement> out(size, FieldElement::zero());
            std::copy(values.begin(), values.end(), out.begin());
            return out;
        }

        std::vector<FieldElement> ids_to_field(const std::vector<std::size_t> &ids, std::size_t size)
        {
            std::vector<FieldElement> out(size, FieldElement::zero());
            for (std::size_t i = 0; i < ids.size(); ++i)
            {
                out[i] = FieldElement(ids[i]);
            }
            return out;
        }

        std::vector<FieldElement> edge_component(const std::vector<data::Edge> &edges, std::size_t size, bool use_src)
        {
            std::vector<FieldElement> out(size, FieldElement::zero());
            for (std::size_t i = 0; i < edges.size(); ++i)
            {
                out[i] = FieldElement(use_src ? edges[i].src : edges[i].dst);
            }
            return out;
        }

        std::vector<FieldElement> feature_table_row_indices(
            std::size_t node_count,
            std::size_t feature_count,
            std::size_t size)
        {
            std::vector<FieldElement> out(size, FieldElement::zero());
            for (std::size_t v = 0; v < node_count; ++v)
            {
                for (std::size_t j = 0; j < feature_count; ++j)
                {
                    out[v * feature_count + j] = FieldElement(v);
                }
            }
            return out;
        }

        std::vector<FieldElement> feature_table_col_indices(
            std::size_t node_count,
            std::size_t feature_count,
            std::size_t size)
        {
            std::vector<FieldElement> out(size, FieldElement::zero());
            for (std::size_t v = 0; v < node_count; ++v)
            {
                for (std::size_t j = 0; j < feature_count; ++j)
                {
                    out[v * feature_count + j] = FieldElement(j);
                }
            }
            return out;
        }

        std::vector<FieldElement> feature_query_row_indices(
            std::size_t local_node_count,
            std::size_t feature_count,
            std::size_t size)
        {
            std::vector<FieldElement> out(size, FieldElement::zero());
            for (std::size_t i = 0; i < local_node_count; ++i)
            {
                for (std::size_t j = 0; j < feature_count; ++j)
                {
                    out[i * feature_count + j] = FieldElement(i);
                }
            }
            return out;
        }

        std::vector<FieldElement> feature_query_col_indices(
            std::size_t local_node_count,
            std::size_t feature_count,
            std::size_t size)
        {
            return feature_table_col_indices(local_node_count, feature_count, size);
        }

        std::vector<FieldElement> feature_query_absolute_ids(
            const std::vector<std::size_t> &absolute_ids,
            std::size_t feature_count,
            std::size_t size)
        {
            std::vector<FieldElement> out(size, FieldElement::zero());
            for (std::size_t i = 0; i < absolute_ids.size(); ++i)
            {
                for (std::size_t j = 0; j < feature_count; ++j)
                {
                    out[i * feature_count + j] = FieldElement(absolute_ids[i]);
                }
            }
            return out;
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

        bool lazy_full_feature_lookup_trace_enabled(const ProtocolContext &context)
        {
            return full_graph_feature_identity_enabled(context);
        }

        bool is_lazy_full_feature_lookup_trace_label(const std::string &name)
        {
            return name == "P_Table_feat" || name == "P_Query_feat" || name == "P_m_feat" || name == "P_R_feat";
        }

        std::size_t label_index_or_npos(
            const std::vector<std::string> &labels,
            std::string_view target)
        {
            for (std::size_t i = 0; i < labels.size(); ++i)
            {
                if (labels[i] == target)
                {
                    return i;
                }
            }
            return labels.size();
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
                throw std::runtime_error("lazy public polynomial requires a domain");
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
                throw std::runtime_error("unsupported lazy public polynomial: " + name);
            }
            const bool full_feature_identity = full_graph_feature_identity_enabled(context);
            std::string canonical_name = name;
            if (full_feature_identity)
            {
                if (canonical_name == "P_Q_qry_feat")
                {
                    canonical_name = "P_Q_tbl_feat";
                }
                else if (canonical_name == "P_Row_feat_qry" || canonical_name == "P_I_feat_qry")
                {
                    canonical_name = "P_Row_feat_tbl";
                }
                else if (canonical_name == "P_Col_feat_qry")
                {
                    canonical_name = "P_Col_feat_tbl";
                }
            }

            static std::mutex cache_mutex;
            static std::unordered_map<std::string, FieldElement> cache;
            const auto cache_key =
                context.config.checkpoint_bundle + ":" + context.config.dataset + ":" + std::to_string(context.dataset.num_nodes) + ":" + std::to_string(context.local.num_features) + ":" + canonical_name + "@" + point.to_string();
            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                if (const auto it = cache.find(cache_key); it != cache.end())
                {
                    return it->second;
                }
            }

            const auto &domain = context.domains.fh;
            const std::size_t dataset_rows = context.dataset.num_nodes;
            const std::size_t local_rows = context.local.num_nodes;
            const std::size_t feature_count = context.local.num_features;
            const std::size_t dataset_valid = dataset_rows * feature_count;
            const std::size_t local_valid = local_rows * feature_count;
            const auto &feature_matrix = !context.dataset.features.empty() ? context.dataset.features : context.local.features;
            FieldElement value = FieldElement::zero();
            if (canonical_name == "P_Q_tbl_feat")
            {
                value = evaluate_truncated_domain_polynomial(
                    domain,
                    dataset_valid,
                    point,
                    [](std::size_t)
                    { return FieldElement::one(); });
            }
            else if (canonical_name == "P_Q_qry_feat")
            {
                value = evaluate_truncated_domain_polynomial(
                    domain,
                    local_valid,
                    point,
                    [](std::size_t)
                    { return FieldElement::one(); });
            }
            else if (canonical_name == "P_Row_feat_tbl")
            {
                value = evaluate_truncated_domain_polynomial(
                    domain,
                    dataset_valid,
                    point,
                    [&](std::size_t index)
                    { return FieldElement(index / feature_count); });
            }
            else if (canonical_name == "P_Col_feat_tbl")
            {
                value = evaluate_truncated_domain_polynomial(
                    domain,
                    dataset_valid,
                    point,
                    [&](std::size_t index)
                    { return FieldElement(index % feature_count); });
            }
            else if (canonical_name == "P_Row_feat_qry")
            {
                value = evaluate_truncated_domain_polynomial(
                    domain,
                    local_valid,
                    point,
                    [&](std::size_t index)
                    { return FieldElement(index / feature_count); });
            }
            else if (canonical_name == "P_Col_feat_qry")
            {
                value = evaluate_truncated_domain_polynomial(
                    domain,
                    local_valid,
                    point,
                    [&](std::size_t index)
                    { return FieldElement(index % feature_count); });
            }
            else if (canonical_name == "P_I_feat_qry")
            {
                value = evaluate_truncated_domain_polynomial(
                    domain,
                    local_valid,
                    point,
                    [&](std::size_t index)
                    { return FieldElement(context.local.absolute_ids[index / feature_count]); });
            }
            else if (canonical_name == "P_T_H")
            {
                value = evaluate_truncated_domain_polynomial(
                    domain,
                    dataset_valid,
                    point,
                    [&](std::size_t index)
                    {
                        const auto row = index / feature_count;
                        const auto column = index % feature_count;
                        return feature_matrix[row][column];
                    });
            }
            else
            {
                throw std::runtime_error("unsupported lazy public polynomial label: " + canonical_name);
            }
            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                cache.emplace(cache_key, value);
            }
            return value;
        }

        FieldElement evaluate_lazy_full_feature_lookup_trace_poly(
            const ProtocolContext &context,
            const std::map<std::string, FieldElement> &challenges,
            const std::string &name,
            const FieldElement &point)
        {
            if (!lazy_full_feature_lookup_trace_enabled(context) || !is_lazy_full_feature_lookup_trace_label(name))
            {
                throw std::runtime_error("unsupported lazy feature lookup trace polynomial: " + name);
            }
            if (name == "P_R_feat")
            {
                return FieldElement::zero();
            }
            const bool full_feature_identity = full_graph_feature_identity_enabled(context);
            if (name == "P_m_feat")
            {
                if (full_feature_identity)
                {
                    return evaluate_lazy_large_fh_public_poly(context, "P_Q_tbl_feat", point);
                }
                return evaluate_lazy_large_fh_public_poly(context, "P_Q_qry_feat", point);
            }
            if (name == "P_Query_feat" && full_feature_identity)
            {
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

        FieldElement row_polynomial_at_point(const std::vector<FieldElement> &row, const FieldElement &point)
        {
            mcl::Fr out;
            out.clear();
            for (std::size_t column = row.size(); column-- > 0;)
            {
                mcl::Fr::mul(out, out, point.native());
                mcl::Fr::add(out, out, row[column].native());
            }
            return FieldElement::from_native(out);
        }

        FieldElement matrix_row_major_evaluation_with_row_stride(
            const model::Matrix &matrix,
            const FieldElement &point,
            const FieldElement &row_stride)
        {
            if (matrix.empty() || matrix.front().empty())
            {
                return FieldElement::zero();
            }

            mcl::Fr out;
            out.clear();
            for (std::size_t row = matrix.size(); row-- > 0;)
            {
                const auto row_eval = row_polynomial_at_point(matrix[row], point);
                mcl::Fr::mul(out, out, row_stride.native());
                mcl::Fr::add(out, out, row_eval.native());
            }
            return FieldElement::from_native(out);
        }

        std::vector<FieldElement> q_new_selector(const std::vector<data::Edge> &edges, std::size_t size)
        {
            std::vector<FieldElement> out(size, FieldElement::zero());
            for (std::size_t i = 0; i < edges.size(); ++i)
            {
                const bool is_new = i == 0 || edges[i].dst != edges[i - 1].dst;
                out[i] = FieldElement(is_new);
            }
            return out;
        }

        std::vector<FieldElement> q_end_selector(const std::vector<data::Edge> &edges, std::size_t size)
        {
            std::vector<FieldElement> out(size, FieldElement::zero());
            for (std::size_t i = 0; i < edges.size(); ++i)
            {
                const bool is_end = (i + 1 == edges.size()) || (edges[i].dst != edges[i + 1].dst);
                out[i] = FieldElement(is_end);
            }
            return out;
        }

        std::vector<FieldElement> edge_gid_column(const std::vector<data::Edge> &edges, std::size_t size)
        {
            std::vector<FieldElement> out(size, FieldElement::zero());
            for (std::size_t i = 0; i < edges.size(); ++i)
            {
                out[i] = FieldElement(edges[i].graph_id);
            }
            return out;
        }

        std::vector<FieldElement> node_gid_column(
            const std::vector<std::size_t> &node_ptr,
            std::size_t node_count,
            std::size_t size)
        {
            std::vector<FieldElement> out(size, FieldElement::zero());
            if (node_ptr.size() < 2 || node_ptr.back() != node_count)
            {
                throw std::runtime_error("invalid node_ptr while building node_gid");
            }
            for (std::size_t graph_id = 0; graph_id + 1 < node_ptr.size(); ++graph_id)
            {
                for (std::size_t node = node_ptr[graph_id]; node < node_ptr[graph_id + 1]; ++node)
                {
                    out[node] = FieldElement(graph_id);
                }
            }
            return out;
        }

        std::vector<FieldElement> q_graph_new_selector(const std::vector<data::Edge> &edges, std::size_t size)
        {
            std::vector<FieldElement> out(size, FieldElement::zero());
            for (std::size_t i = 0; i < edges.size(); ++i)
            {
                out[i] = FieldElement(i == 0 || edges[i].graph_id != edges[i - 1].graph_id);
            }
            return out;
        }

        std::vector<FieldElement> q_graph_end_selector(const std::vector<data::Edge> &edges, std::size_t size)
        {
            std::vector<FieldElement> out(size, FieldElement::zero());
            for (std::size_t i = 0; i < edges.size(); ++i)
            {
                out[i] = FieldElement(i + 1 == edges.size() || edges[i].graph_id != edges[i + 1].graph_id);
            }
            return out;
        }

        std::vector<FieldElement> q_group_selector(
            const std::vector<FieldElement> &graph_boundary,
            const std::vector<FieldElement> &destination_boundary)
        {
            if (graph_boundary.size() != destination_boundary.size())
            {
                throw std::runtime_error("group selector input size mismatch");
            }
            std::vector<FieldElement> out(graph_boundary.size(), FieldElement::zero());
            for (std::size_t i = 0; i < out.size(); ++i)
            {
                out[i] = FieldElement::one()
                    - (FieldElement::one() - graph_boundary[i])
                        * (FieldElement::one() - destination_boundary[i]);
            }
            return out;
        }

        std::vector<FieldElement> make_range_table(std::size_t size)
        {
            std::vector<FieldElement> out(size, FieldElement::zero());
            for (std::size_t i = 0; i < size; ++i)
            {
                out[i] = FieldElement(i);
            }
            return out;
        }

        std::int64_t quantize_float_model_value_signed(double value)
        {
            return static_cast<std::int64_t>(
                value >= 0.0 ? value * 16.0 + 0.5 : value * 16.0 - 0.5);
        }

        FieldElement quantize_float_model_value(double value)
        {
            return FieldElement::from_signed(quantize_float_model_value_signed(value));
        }

        double dequantize_trace_scalar(std::int64_t value)
        {
            return static_cast<double>(value) / 16.0;
        }

        std::size_t signed_table_size(std::int64_t min_raw, std::int64_t max_raw, std::string_view label)
        {
            if (min_raw > max_raw)
            {
                throw std::runtime_error("invalid signed table range: " + std::string(label));
            }
            const auto span =
                static_cast<std::uint64_t>(max_raw) - static_cast<std::uint64_t>(min_raw) + 1ULL;
            if (span == 0 || span > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
            {
                throw std::runtime_error("signed table range is too large: " + std::string(label));
            }
            return static_cast<std::size_t>(span);
        }

        std::vector<std::pair<FieldElement, FieldElement>> make_lrelu_table(std::int64_t min_raw, std::int64_t max_raw)
        {
            constexpr double kAttentionScale = 256.0;
            std::vector<std::pair<FieldElement, FieldElement>> out;
            out.reserve(signed_table_size(min_raw, max_raw, "lrelu"));
            for (std::int64_t raw = min_raw; raw <= max_raw; ++raw)
            {
                const auto input = static_cast<double>(raw) / kAttentionScale;
                const auto output = input >= 0.0 ? input : 0.2 * input;
                const auto output_raw = static_cast<std::int64_t>(
                    output >= 0.0 ? output * kAttentionScale + 0.5 : output * kAttentionScale - 0.5);
                out.push_back({FieldElement::from_signed(raw), FieldElement::from_signed(output_raw)});
                if (raw == max_raw)
                {
                    break;
                }
            }
            return out;
        }

        std::vector<std::pair<FieldElement, FieldElement>> make_elu_table(std::size_t bound)
        {
            std::vector<std::pair<FieldElement, FieldElement>> out;
            out.reserve(bound * 2 + 1);
            for (std::int64_t raw = -static_cast<std::int64_t>(bound); raw <= static_cast<std::int64_t>(bound); ++raw)
            {
                const auto input = dequantize_trace_scalar(raw);
                const auto central_signed = quantize_float_model_value_signed(input >= 0.0 ? input : std::exp(input) - 1.0);
                out.push_back({
                    FieldElement::from_signed(raw),
                    FieldElement::from_signed(central_signed),
                });
            }
            return out;
        }

        std::vector<std::pair<FieldElement, FieldElement>> make_exp_table(std::size_t size)
        {
            constexpr double kAttentionScale = 256.0;
            std::vector<std::pair<FieldElement, FieldElement>> out;
            out.reserve(size);
            for (std::size_t i = 0; i < size; ++i)
            {
                const auto input = static_cast<std::int64_t>(i);
                const auto central_signed = quantize_float_model_value_signed(
                    std::exp(-static_cast<double>(input) / kAttentionScale));
                const auto candidate = std::max<std::int64_t>(0, central_signed);
                out.push_back({
                    FieldElement(static_cast<std::uint64_t>(i)),
                    FieldElement::from_signed(candidate),
                });
            }
            return out;
        }

        void require_functional_pair_table(
            const std::vector<std::pair<FieldElement, FieldElement>> &table,
            std::string_view label)
        {
            std::unordered_map<std::string, FieldElement> outputs;
            outputs.reserve(table.size());
            for (const auto &[input, output] : table)
            {
                const auto key = input.to_string();
                const auto [it, inserted] = outputs.emplace(key, output);
                if (!inserted && !(it->second == output))
                {
                    throw std::runtime_error(std::string(label) + " table is not a function");
                }
            }
        }

        std::vector<FieldElement> quantize_model_vector(const std::vector<double> &values)
        {
            std::vector<FieldElement> out(values.size(), FieldElement::zero());
            for (std::size_t i = 0; i < values.size(); ++i)
            {
                out[i] = quantize_float_model_value(values[i]);
            }
            return out;
        }

        std::vector<FieldElement> quantize_model_vector_at_scale(
            const std::vector<double> &values,
            double scale)
        {
            std::vector<FieldElement> out(values.size(), FieldElement::zero());
            for (std::size_t i = 0; i < values.size(); ++i)
            {
                const auto scaled = values[i] * scale;
                out[i] = FieldElement::from_signed(static_cast<std::int64_t>(
                    scaled >= 0.0 ? scaled + 0.5 : scaled - 0.5));
            }
            return out;
        }

        model::Matrix quantize_model_matrix(const model::FloatMatrix &matrix)
        {
            model::Matrix out(matrix.size());
            for (std::size_t row = 0; row < matrix.size(); ++row)
            {
                out[row] = quantize_model_vector(matrix[row]);
            }
            return out;
        }


        void append_note(RunMetrics *metrics, const std::string &note)
        {
            if (metrics == nullptr || note.empty())
            {
                return;
            }
            if (!metrics->notes.empty())
            {
                metrics->notes += "; ";
            }
            metrics->notes += note;
        }

        std::string context_cache_key(const util::AppConfig &config)
        {
            std::ostringstream stream;
            // `export_dir` 与 `dump_trace` 只影响 artifact 侧行为，不影响可复用的正式协议对象。
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

        std::size_t hidden_head_width(const model::ModelParameters &parameters, const util::AppConfig &config)
        {
            if (parameters.has_real_multihead)
            {
                return model::max_hidden_head_dim(parameters);
            }
            return config.hidden_dim;
        }

        std::size_t concat_width(const model::ModelParameters &parameters, const util::AppConfig &config)
        {
            if (parameters.has_real_multihead)
            {
                return model::max_hidden_concat_width(parameters);
            }
            return config.hidden_dim;
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
                specs.push_back({
                    hidden_score_mask_eval_name(head_index, "src"),
                    hidden_score_mask_label(head_index, "src"),
                    "y_src_h" + std::to_string(head_index)});
                specs.push_back({"mu_" + suffix + "_dst", prefix + "E_dst", "y_dst_h" + std::to_string(head_index)});
                specs.push_back({
                    hidden_score_mask_eval_name(head_index, "dst"),
                    hidden_score_mask_label(head_index, "dst"),
                    "y_dst_h" + std::to_string(head_index)});
                specs.push_back({"mu_" + suffix + "_star", prefix + "H_star", "y_star_h" + std::to_string(head_index)});
                specs.push_back({
                    external_mask_eval_name("mu_" + suffix + "_star"),
                    external_mask_label(prefix + "H_star"),
                    "y_star_h" + std::to_string(head_index)});
                specs.push_back({"mu_" + suffix + "_agg_pre", prefix + "H_agg_pre_star", "y_agg_pre_h" + std::to_string(head_index)});
                specs.push_back({
                    external_mask_eval_name("mu_" + suffix + "_agg_pre"),
                    external_mask_label(prefix + "H_agg_pre_star"),
                    "y_agg_pre_h" + std::to_string(head_index)});
                specs.push_back({"mu_" + suffix + "_agg", prefix + "H_agg_star", "y_agg_h" + std::to_string(head_index)});
                specs.push_back({
                    external_mask_eval_name("mu_" + suffix + "_agg"),
                    external_mask_label(prefix + "H_agg_star"),
                    "y_agg_h" + std::to_string(head_index)});
            }
            for (std::size_t layer_index = 0; layer_index < context.model.hidden_layers.size(); ++layer_index)
            {
                const bool is_final_layer = layer_index + 1 == context.model.hidden_layers.size();
                specs.push_back({
                    is_final_layer ? "mu_cat" : "mu_cat_l" + std::to_string(layer_index),
                    hidden_layer_concat_star_label(layer_index, is_final_layer),
                    hidden_concat_y_name(layer_index, is_final_layer),
                });
                const auto concat_eval_name =
                    is_final_layer ? std::string("mu_cat")
                                   : "mu_cat_l" + std::to_string(layer_index);
                specs.push_back({
                    external_mask_eval_name(concat_eval_name),
                    external_mask_label(hidden_layer_concat_star_label(
                        layer_index, is_final_layer)),
                    hidden_concat_y_name(layer_index, is_final_layer),
                });
            }
            for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index)
            {
                const auto prefix = output_head_prefix(head_index);
                specs.push_back({
                    output_external_eval_name("proj", head_index),
                    prefix + "_Y_prime",
                    output_challenge_name("y_proj_out", head_index),
                });
                specs.push_back({
                    output_external_eval_name("src", head_index),
                    prefix + "_E_src",
                    output_challenge_name("y_src_out", head_index),
                });
                specs.push_back({
                    output_score_mask_eval_name(head_index, "src"),
                    output_score_mask_label(head_index, "src"),
                    output_challenge_name("y_src_out", head_index),
                });
                specs.push_back({
                    output_external_eval_name("dst", head_index),
                    prefix + "_E_dst",
                    output_challenge_name("y_dst_out", head_index),
                });
                specs.push_back({
                    output_score_mask_eval_name(head_index, "dst"),
                    output_score_mask_label(head_index, "dst"),
                    output_challenge_name("y_dst_out", head_index),
                });
                specs.push_back({
                    output_external_eval_name("star", head_index),
                    prefix + "_Y_star",
                    output_challenge_name("y_out_star", head_index),
                });
                const auto output_star_eval_name =
                    output_external_eval_name("star", head_index);
                specs.push_back({
                    external_mask_eval_name(output_star_eval_name),
                    external_mask_label(prefix + "_Y_star"),
                    output_challenge_name("y_out_star", head_index),
                });
                specs.push_back({output_external_eval_name("y_lin", head_index), output_y_lin_label(head_index), "y_out"});
                specs.push_back({output_external_eval_name("y", head_index), output_y_label(head_index), "y_out"});
            }
            specs.push_back({"mu_Y_lin", "P_Y_lin", "y_out"});
            specs.push_back({"mu_out", "P_Y", "y_out"});
            return specs;
        }

        double *domain_open_metric(RunMetrics *metrics, const std::string &bundle_name)
        {
            if (metrics == nullptr)
            {
                return nullptr;
            }
            if (bundle_name == "FH")
                return &metrics->domain_open_fh_ms;
            if (bundle_name == "edge")
                return &metrics->domain_open_edge_ms;
            if (bundle_name == "in")
                return &metrics->domain_open_in_ms;
            if (bundle_name == "d_h")
                return &metrics->domain_open_d_h_ms;
            if (bundle_name == "cat")
                return &metrics->domain_open_cat_ms;
            if (bundle_name == "C")
                return &metrics->domain_open_c_ms;
            if (bundle_name == "N")
                return &metrics->domain_open_n_ms;
            return nullptr;
        }

        double *quotient_metric(RunMetrics *metrics, const std::string &label)
        {
            if (metrics == nullptr)
            {
                return nullptr;
            }
            if (label == "t_FH")
                return &metrics->quotient_t_fh_ms;
            if (label == "t_edge")
                return &metrics->quotient_t_edge_ms;
            if (label == "t_in")
                return &metrics->quotient_t_in_ms;
            if (label == "t_d_h")
                return &metrics->quotient_t_d_h_ms;
            if (label == "t_cat")
                return &metrics->quotient_t_cat_ms;
            if (label == "t_C")
                return &metrics->quotient_t_c_ms;
            if (label == "t_N")
                return &metrics->quotient_t_n_ms;
            return nullptr;
        }

        std::filesystem::path resolve_project_path(
            const std::string &project_root,
            const std::string &relative_or_absolute)
        {
            const std::filesystem::path path(relative_or_absolute);
            if (path.is_absolute())
            {
                return path;
            }
            return std::filesystem::path(project_root) / path;
        }

        std::shared_ptr<algebra::DomainEvaluationWeightCache> shared_proof_domain_weight_cache()
        {
            static auto cache = std::make_shared<algebra::DomainEvaluationWeightCache>();
            return cache;
        }

        class SharedFeatureMatrixEvaluationCache
        {
        public:
            FieldElement get_or_compute(
                const std::string &context_key,
                const model::Matrix &matrix,
                const FieldElement &point,
                RunMetrics *metrics)
            {
                const auto cache_key = context_key + "|P_H@" + algebra::point_cache_key(point);
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (const auto it = entries_.find(cache_key); it != entries_.end())
                    {
                        return it->second;
                    }
                }

                const auto point_power_start = Clock::now();
                const auto row_stride =
                    matrix.empty() || matrix.front().empty()
                        ? FieldElement::one()
                        : point.pow(static_cast<std::uint64_t>(matrix.front().size()));
                if (metrics != nullptr)
                {
                    metrics->fh_point_powers_ms += elapsed_ms(point_power_start, Clock::now());
                }

                const auto feature_eval_start = Clock::now();
                const auto value = matrix_row_major_evaluation_with_row_stride(matrix, point, row_stride);
                if (metrics != nullptr)
                {
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

        std::shared_ptr<SharedFeatureMatrixEvaluationCache> shared_feature_matrix_evaluation_cache()
        {
            static auto cache = std::make_shared<SharedFeatureMatrixEvaluationCache>();
            return cache;
        }

        std::string labels_cache_key(const std::vector<std::string> &labels);

        class SharedTraceEvaluationCache
        {
        public:
            std::shared_ptr<const std::vector<FieldElement>> lookup(
                const std::string &context_key,
                const std::shared_ptr<algebra::RootOfUnityDomain> &domain,
                const std::vector<std::string> &labels,
                const FieldElement &point)
            {
                const auto cache_key =
                    context_key + "|trace|" + algebra::domain_point_cache_key(domain, point) + "|" + labels_cache_key(labels);
                std::lock_guard<std::mutex> lock(mutex_);
                if (const auto it = entries_.find(cache_key); it != entries_.end())
                {
                    return it->second;
                }
                return nullptr;
            }

            std::shared_ptr<const std::vector<FieldElement>> store(
                const std::string &context_key,
                const std::shared_ptr<algebra::RootOfUnityDomain> &domain,
                const std::vector<std::string> &labels,
                const FieldElement &point,
                std::vector<FieldElement> values)
            {
                const auto cache_key =
                    context_key + "|trace|" + algebra::domain_point_cache_key(domain, point) + "|" + labels_cache_key(labels);
                auto shared_values = std::make_shared<const std::vector<FieldElement>>(std::move(values));
                std::lock_guard<std::mutex> lock(mutex_);
                const auto [it, inserted] = entries_.emplace(std::move(cache_key), std::move(shared_values));
                (void)inserted;
                return it->second;
            }

        private:
            std::mutex mutex_;
            std::unordered_map<std::string, std::shared_ptr<const std::vector<FieldElement>>> entries_;
        };

        std::shared_ptr<SharedTraceEvaluationCache> shared_trace_evaluation_cache()
        {
            static auto cache = std::make_shared<SharedTraceEvaluationCache>();
            return cache;
        }

        std::string labels_cache_key(const std::vector<std::string> &labels)
        {
            std::string key;
            for (const auto &label : labels)
            {
                key += label;
                key.push_back('\x1f');
            }
            return key;
        }

        std::string quotient_build_cache_key(const ProtocolContext &context, const TraceArtifacts &trace)
        {
            std::ostringstream stream;
            stream << context_cache_key(context.config) << "|quotients";
            for (const auto &label : trace.commitment_order)
            {
                const auto it = trace.commitments.find(label);
                if (it == trace.commitments.end())
                {
                    continue;
                }
                stream << '|' << label << '=' << it->second.point.to_string();
            }
            return stream.str();
        }


        class SharedPublicEvaluationCache
        {
        public:
            std::shared_ptr<const std::vector<FieldElement>> lookup(
                const std::string &context_key,
                const std::shared_ptr<algebra::RootOfUnityDomain> &domain,
                const std::vector<std::string> &labels,
                const FieldElement &point)
            {
                const auto cache_key =
                    context_key + "|" + algebra::domain_point_cache_key(domain, point) + "|" + labels_cache_key(labels);
                std::lock_guard<std::mutex> lock(mutex_);
                if (const auto it = entries_.find(cache_key); it != entries_.end())
                {
                    return it->second;
                }
                return nullptr;
            }

            std::shared_ptr<const std::vector<FieldElement>> store(
                const std::string &context_key,
                const std::shared_ptr<algebra::RootOfUnityDomain> &domain,
                const std::vector<std::string> &labels,
                const FieldElement &point,
                std::vector<FieldElement> values)
            {
                const auto cache_key =
                    context_key + "|" + algebra::domain_point_cache_key(domain, point) + "|" + labels_cache_key(labels);
                auto shared_values = std::make_shared<const std::vector<FieldElement>>(std::move(values));
                std::lock_guard<std::mutex> lock(mutex_);
                const auto [it, inserted] = entries_.emplace(std::move(cache_key), std::move(shared_values));
                (void)inserted;
                return it->second;
            }

        private:
            std::mutex mutex_;
            std::unordered_map<std::string, std::shared_ptr<const std::vector<FieldElement>>> entries_;
        };

        std::shared_ptr<SharedPublicEvaluationCache> shared_public_evaluation_cache()
        {
            static auto cache = std::make_shared<SharedPublicEvaluationCache>();
            return cache;
        }

        class ProofEvaluationBackendRegistry
        {
        public:
            ProofEvaluationBackendRegistry(const ProtocolContext &context, const TraceArtifacts &trace)
            {
                std::unordered_map<std::string, PolynomialBatchGroup> groups;
                auto collect = [&](const auto &polynomial_map)
                {
                    for (const auto &[name, polynomial] : polynomial_map)
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
                    }
                };
                collect(context.public_polynomials);
                collect(trace.polynomials);

                for (auto &[domain_key, group] : groups)
                {
                    backends_.emplace(
                        domain_key,
                        algebra::PackedEvaluationBackend(group.domain, std::move(group.polynomials)));
                }
            }

            const algebra::PackedEvaluationBackend *find(const std::shared_ptr<algebra::RootOfUnityDomain> &domain) const
            {
                const auto domain_key = domain->name + ":" + std::to_string(domain->size);
                if (const auto it = backends_.find(domain_key); it != backends_.end())
                {
                    return &it->second;
                }
                return nullptr;
            }

        private:
            struct PolynomialBatchGroup
            {
                std::shared_ptr<algebra::RootOfUnityDomain> domain;
                std::vector<std::pair<std::string, const Polynomial *>> polynomials;
            };

            std::unordered_map<std::string, algebra::PackedEvaluationBackend> backends_;
        };

        class SharedProofEvaluationBackendRegistryCache
        {
        public:
            std::shared_ptr<const ProofEvaluationBackendRegistry> get_or_build(
                const std::string &key,
                const std::function<std::shared_ptr<const ProofEvaluationBackendRegistry>()> &build)
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
            std::unordered_map<std::string, std::shared_ptr<const ProofEvaluationBackendRegistry>> entries_;
        };

        SharedProofEvaluationBackendRegistryCache &shared_proof_backend_registry_cache()
        {
            static SharedProofEvaluationBackendRegistryCache cache;
            return cache;
        }

        FieldElement evaluate_spilled_evaluation_polynomial(
            const TraceArtifacts::SpilledEvaluationPolynomial &spilled,
            const std::shared_ptr<algebra::RootOfUnityDomain> &domain,
            const FieldElement &point,
            const algebra::DomainEvaluationWeights &weight_entry)
        {
            if (domain == nullptr)
            {
                throw std::runtime_error("spilled polynomial is missing its domain");
            }
            constexpr std::size_t kFieldBytes = 32;
            std::array<std::uint8_t, kFieldBytes> bytes{};
            std::ifstream stream(spilled.path, std::ios::binary);
            if (!stream)
            {
                throw std::runtime_error("failed to open spilled polynomial file: " + spilled.path);
            }
            if (weight_entry.direct_index.has_value())
            {
                if (*weight_entry.direct_index >= spilled.size)
                {
                    return FieldElement::zero();
                }
                const auto offset = static_cast<std::streamoff>(*weight_entry.direct_index * kFieldBytes);
                stream.seekg(offset);
                stream.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
                if (!stream)
                {
                    throw std::runtime_error("failed to read direct spilled polynomial entry: " + spilled.path);
                }
                return FieldElement::from_little_endian_mod(bytes.data(), bytes.size());
            }

            mcl::Fr native_sum;
            native_sum.clear();
            for (std::size_t index = 0; index < spilled.size; ++index)
            {
                stream.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
                if (!stream)
                {
                    throw std::runtime_error("failed to stream spilled polynomial: " + spilled.path);
                }
                const auto value = FieldElement::from_little_endian_mod(bytes.data(), bytes.size());
                mcl::Fr term;
                mcl::Fr::mul(term, value.native(), weight_entry.native_weights[index]);
                mcl::Fr::add(native_sum, native_sum, term);
            }
            auto value = FieldElement::from_native(native_sum);
            if (!spilled.zero_domain_blinder.empty())
            {
                value += domain->zero_polynomial_eval(point)
                    * algebra::horner(spilled.zero_domain_blinder, point);
            }
            return value;
        }

        // proof 级 memoization 只缓存“当前 trace + 当前 Fiat-Shamir 挑战”派生出的值。
        // 这样既能避免同一点重复求值，又不会把不同 proof 实例的状态混在一起。
        class EvaluationMemoization
        {
        public:
            EvaluationMemoization(
                const ProtocolContext &context,
                const TraceArtifacts &trace,
                const std::map<std::string, FieldElement> &challenges,
                std::shared_ptr<const ProofEvaluationBackendRegistry> backend_registry = nullptr,
                std::shared_ptr<algebra::DomainEvaluationWeightCache> domain_weight_cache = nullptr,
                RunMetrics *metrics = nullptr)
                : context_(context),
                  trace_(trace),
                  challenges_(challenges),
                  backend_registry_(std::move(backend_registry)),
                  domain_weight_cache_(std::move(domain_weight_cache)),
                  context_key_(context_cache_key(context.config)),
                  metrics_(metrics)
            {
                if (domain_weight_cache_ == nullptr)
                {
                    domain_weight_cache_ = std::make_shared<algebra::DomainEvaluationWeightCache>();
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
                if (name == "P_H" && lazy_full_feature_lookup_trace_enabled(context_))
                {
                    value = eval_named("P_T_H", point);
                }
                else if (name == "P_H")
                {
                    value = shared_feature_matrix_evaluation_cache()->get_or_compute(
                        context_key_,
                        context_.local.features,
                        point,
                        metrics_);
                }
                else if (lazy_full_feature_lookup_trace_enabled(context_) && is_lazy_full_feature_lookup_trace_label(name))
                {
                    value = evaluate_lazy_full_feature_lookup_trace_poly(context_, challenges_, name, point);
                    if (const auto polynomial_it = trace_.polynomials.find(name);
                        polynomial_it != trace_.polynomials.end()
                        && polynomial_it->second.is_zero_domain_blinded())
                    {
                        value += polynomial_it->second.domain->zero_polynomial_eval(point)
                            * algebra::horner(
                                polynomial_it->second.zero_domain_blinder,
                                point);
                    }
                }
                else if (trace_.polynomials.contains(name))
                {
                    value = eval_polynomial(trace_.polynomials.at(name), point);
                }
                else if (trace_.spilled_polynomials.contains(name))
                {
                    const auto &spilled = trace_.spilled_polynomials.at(name);
                    const auto domain = domain_from_name(spilled.domain_name);
                    value = evaluate_spilled_evaluation_polynomial(
                        spilled,
                        domain,
                        point,
                        domain_weights(domain, point));
                }
                else if (lazy_large_fh_public_enabled(context_) && is_lazy_large_fh_public_label(name))
                {
                    value = evaluate_lazy_large_fh_public_poly(context_, name, point);
                }
                else if (context_.public_polynomials.contains(name))
                {
                    value = eval_polynomial(context_.public_polynomials.at(name), point);
                }
                else if (name == "t_FH")
                {
                    FHQuotientProfile fh_profile;
                    value = evaluate_t_fh(
                        context_,
                        challenges_,
                        [&](const std::string &inner_name, const FieldElement &inner_point)
                        {
                            return eval_named(inner_name, inner_point);
                        },
                        point,
                        metrics_ != nullptr ? &fh_profile : nullptr);
                    if (metrics_ != nullptr)
                    {
                        metrics_->fh_quotient_assembly_ms += fh_profile.assembly_ms;
                    }
                }
                else if (name == "t_edge")
                {
                    value = evaluate_t_edge(
                        context_,
                        challenges_,
                        trace_.external_evaluations,
                        [&](const std::string &inner_name, const FieldElement &inner_point)
                        {
                            return eval_named(inner_name, inner_point);
                        },
                        point);
                }
                else if (name == "t_N")
                {
                    value = evaluate_t_n(
                        context_,
                        challenges_,
                        [&](const std::string &inner_name, const FieldElement &inner_point)
                        {
                            return eval_named(inner_name, inner_point);
                        },
                        point);
                }
                else if (name == "t_in")
                {
                    value = evaluate_t_in(
                        context_,
                        challenges_,
                        trace_.external_evaluations,
                        [&](const std::string &inner_name, const FieldElement &inner_point)
                        {
                            return eval_named(inner_name, inner_point);
                        },
                        point);
                }
                else if (name == "t_d" || name == "t_d_h")
                {
                    value = evaluate_t_d(
                        context_,
                        challenges_,
                        trace_.external_evaluations,
                        [&](const std::string &inner_name, const FieldElement &inner_point)
                        {
                            return eval_named(inner_name, inner_point);
                        },
                        point);
                }
                else if (name == "t_cat")
                {
                    value = evaluate_t_cat(
                        context_,
                        challenges_,
                        trace_.external_evaluations,
                        [&](const std::string &inner_name, const FieldElement &inner_point)
                        {
                            return eval_named(inner_name, inner_point);
                        },
                        point);
                }
                else if (name == "t_C")
                {
                    value = evaluate_t_c(
                        context_,
                        challenges_,
                        trace_.external_evaluations,
                        [&](const std::string &inner_name, const FieldElement &inner_point)
                        {
                            return eval_named(inner_name, inner_point);
                        },
                        point);
                }
                else
                {
                    throw std::runtime_error("missing polynomial for evaluation: " + name);
                }

                value_cache_.emplace(cache_key, value);
                return value;
            }

        private:
            std::shared_ptr<algebra::RootOfUnityDomain> domain_from_name(const std::string &name) const
            {
                if (name == "FH")
                    return context_.domains.fh;
                if (name == "edge")
                    return context_.domains.edge;
                if (name == "in")
                    return context_.domains.in;
                if (name == "d")
                    return context_.domains.d;
                if (name == "cat")
                    return context_.domains.cat;
                if (name == "C")
                    return context_.domains.c;
                if (name == "N")
                    return context_.domains.n;
                throw std::runtime_error("unknown spilled polynomial domain: " + name);
            }

            const Polynomial *lookup_polynomial(const std::string &name) const
            {
                if (trace_.polynomials.contains(name))
                {
                    return &trace_.polynomials.at(name);
                }
                if (context_.public_polynomials.contains(name))
                {
                    return &context_.public_polynomials.at(name);
                }
                return nullptr;
            }

            std::optional<std::vector<std::size_t>> rotated_point_shifts(
                const std::shared_ptr<algebra::RootOfUnityDomain> &domain,
                const std::vector<FieldElement> &points) const
            {
                if (points.size() <= 1)
                {
                    return std::nullopt;
                }
                std::vector<std::size_t> shifts(points.size(), 0);
                for (std::size_t i = 1; i < points.size(); ++i)
                {
                    const auto shift = domain->rotation_shift(points.front(), points[i]);
                    if (!shift.has_value())
                    {
                        return std::nullopt;
                    }
                    shifts[i] = *shift;
                }
                return shifts;
            }

            std::vector<FieldElement> evaluate_lazy_large_fh_public_group(
                const std::vector<std::string> &labels,
                const FieldElement &point)
            {
                std::vector<FieldElement> out(labels.size(), FieldElement::zero());
                if (labels.empty())
                {
                    return out;
                }

                const auto &domain = context_.domains.fh;
                const auto &weight_entry = domain_weights(domain, point);
                const std::size_t dataset_rows = context_.dataset.num_nodes;
                const std::size_t local_rows = context_.local.num_nodes;
                const std::size_t feature_count = context_.local.num_features;
                const std::size_t dataset_valid = dataset_rows * feature_count;
                const std::size_t local_valid = local_rows * feature_count;
                const bool full_feature_identity = full_graph_feature_identity_enabled(context_);
                const auto &feature_matrix =
                    !context_.dataset.features.empty() ? context_.dataset.features : context_.local.features;

                const auto q_tbl_index = label_index_or_npos(labels, "P_Q_tbl_feat");
                const auto q_qry_index = label_index_or_npos(labels, "P_Q_qry_feat");
                const auto row_tbl_index = label_index_or_npos(labels, "P_Row_feat_tbl");
                const auto col_tbl_index = label_index_or_npos(labels, "P_Col_feat_tbl");
                const auto row_qry_index = label_index_or_npos(labels, "P_Row_feat_qry");
                const auto col_qry_index = label_index_or_npos(labels, "P_Col_feat_qry");
                const auto abs_qry_index = label_index_or_npos(labels, "P_I_feat_qry");
                const auto th_index = label_index_or_npos(labels, "P_T_H");

                auto direct_value = [&](std::string_view label, std::size_t index) -> FieldElement
                {
                    if (label == "P_Q_tbl_feat")
                    {
                        return index < dataset_valid ? FieldElement::one() : FieldElement::zero();
                    }
                    if (label == "P_Q_qry_feat")
                    {
                        return index < local_valid ? FieldElement::one() : FieldElement::zero();
                    }
                    if (label == "P_Row_feat_tbl")
                    {
                        return index < dataset_valid ? FieldElement(index / feature_count) : FieldElement::zero();
                    }
                    if (label == "P_Col_feat_tbl")
                    {
                        return index < dataset_valid ? FieldElement(index % feature_count) : FieldElement::zero();
                    }
                    if (label == "P_Row_feat_qry")
                    {
                        return index < local_valid ? FieldElement(index / feature_count) : FieldElement::zero();
                    }
                    if (label == "P_Col_feat_qry")
                    {
                        return index < local_valid ? FieldElement(index % feature_count) : FieldElement::zero();
                    }
                    if (label == "P_I_feat_qry")
                    {
                        return index < local_valid
                                   ? FieldElement(context_.local.absolute_ids[index / feature_count])
                                   : FieldElement::zero();
                    }
                    if (label == "P_T_H")
                    {
                        if (index >= dataset_valid)
                        {
                            return FieldElement::zero();
                        }
                        const auto row = index / feature_count;
                        const auto column = index % feature_count;
                        return feature_matrix[row][column];
                    }
                    throw std::runtime_error("unsupported lazy FH public label: " + std::string(label));
                };

                if (weight_entry.direct_index.has_value())
                {
                    const auto direct_index = *weight_entry.direct_index;
                    for (std::size_t i = 0; i < labels.size(); ++i)
                    {
                        out[i] = direct_value(labels[i], direct_index);
                    }
                    return out;
                }

                std::vector<FieldElement> column_values(feature_count, FieldElement::zero());
                for (std::size_t column = 0; column < feature_count; ++column)
                {
                    column_values[column] = FieldElement(column);
                }

                std::vector<mcl::Fr> accumulators(labels.size());
                for (auto &accumulator : accumulators)
                {
                    accumulator.clear();
                }

                auto add_scaled = [&](std::size_t label_index, const FieldElement &value, const mcl::Fr &weight)
                {
                    if (label_index >= labels.size())
                    {
                        return;
                    }
                    mcl::Fr term;
                    mcl::Fr::mul(term, value.native(), weight);
                    mcl::Fr::add(accumulators[label_index], accumulators[label_index], term);
                };
                auto add_weight = [&](std::size_t label_index, const mcl::Fr &weight)
                {
                    if (label_index >= labels.size())
                    {
                        return;
                    }
                    mcl::Fr::add(accumulators[label_index], accumulators[label_index], weight);
                };

                std::size_t index = 0;
                for (std::size_t row = 0; row < dataset_rows; ++row)
                {
                    const FieldElement row_value(row);
                    const auto &feature_row = feature_matrix[row];
                    for (std::size_t column = 0; column < feature_count; ++column, ++index)
                    {
                        const auto &weight = weight_entry.native_weights[index];
                        add_weight(q_tbl_index, weight);
                        add_scaled(row_tbl_index, row_value, weight);
                        add_scaled(col_tbl_index, column_values[column], weight);
                        add_scaled(th_index, feature_row[column], weight);
                        if (full_feature_identity)
                        {
                            add_weight(q_qry_index, weight);
                            add_scaled(row_qry_index, row_value, weight);
                            add_scaled(col_qry_index, column_values[column], weight);
                            add_scaled(abs_qry_index, row_value, weight);
                        }
                    }
                }

                if (!full_feature_identity)
                {
                    index = 0;
                    for (std::size_t row = 0; row < local_rows; ++row)
                    {
                        const FieldElement row_value(row);
                        const FieldElement absolute_id(context_.local.absolute_ids[row]);
                        for (std::size_t column = 0; column < feature_count; ++column, ++index)
                        {
                            const auto &weight = weight_entry.native_weights[index];
                            add_weight(q_qry_index, weight);
                            add_scaled(row_qry_index, row_value, weight);
                            add_scaled(col_qry_index, column_values[column], weight);
                            add_scaled(abs_qry_index, absolute_id, weight);
                        }
                    }
                }

                for (std::size_t i = 0; i < labels.size(); ++i)
                {
                    out[i] = FieldElement::from_native(accumulators[i]);
                }
                return out;
            }

            std::vector<std::vector<FieldElement>> evaluate_lazy_large_fh_public_group_points(
                const std::vector<std::string> &labels,
                const std::vector<FieldElement> &points)
            {
                std::vector<std::vector<FieldElement>> out(
                    points.size(),
                    std::vector<FieldElement>(labels.size(), FieldElement::zero()));
                if (labels.empty() || points.empty())
                {
                    return out;
                }

                const auto &domain = context_.domains.fh;
                const std::size_t dataset_rows = context_.dataset.num_nodes;
                const std::size_t local_rows = context_.local.num_nodes;
                const std::size_t feature_count = context_.local.num_features;
                const std::size_t dataset_valid = dataset_rows * feature_count;
                const std::size_t local_valid = local_rows * feature_count;
                const bool full_feature_identity = full_graph_feature_identity_enabled(context_);
                const auto &feature_matrix =
                    !context_.dataset.features.empty() ? context_.dataset.features : context_.local.features;

                const auto q_tbl_index = label_index_or_npos(labels, "P_Q_tbl_feat");
                const auto q_qry_index = label_index_or_npos(labels, "P_Q_qry_feat");
                const auto row_tbl_index = label_index_or_npos(labels, "P_Row_feat_tbl");
                const auto col_tbl_index = label_index_or_npos(labels, "P_Col_feat_tbl");
                const auto row_qry_index = label_index_or_npos(labels, "P_Row_feat_qry");
                const auto col_qry_index = label_index_or_npos(labels, "P_Col_feat_qry");
                const auto abs_qry_index = label_index_or_npos(labels, "P_I_feat_qry");
                const auto th_index = label_index_or_npos(labels, "P_T_H");

                auto direct_value = [&](std::string_view label, std::size_t index) -> FieldElement
                {
                    if (label == "P_Q_tbl_feat")
                    {
                        return index < dataset_valid ? FieldElement::one() : FieldElement::zero();
                    }
                    if (label == "P_Q_qry_feat")
                    {
                        return index < local_valid ? FieldElement::one() : FieldElement::zero();
                    }
                    if (label == "P_Row_feat_tbl")
                    {
                        return index < dataset_valid ? FieldElement(index / feature_count) : FieldElement::zero();
                    }
                    if (label == "P_Col_feat_tbl")
                    {
                        return index < dataset_valid ? FieldElement(index % feature_count) : FieldElement::zero();
                    }
                    if (label == "P_Row_feat_qry")
                    {
                        return index < local_valid ? FieldElement(index / feature_count) : FieldElement::zero();
                    }
                    if (label == "P_Col_feat_qry")
                    {
                        return index < local_valid ? FieldElement(index % feature_count) : FieldElement::zero();
                    }
                    if (label == "P_I_feat_qry")
                    {
                        return index < local_valid
                                   ? FieldElement(context_.local.absolute_ids[index / feature_count])
                                   : FieldElement::zero();
                    }
                    if (label == "P_T_H")
                    {
                        if (index >= dataset_valid)
                        {
                            return FieldElement::zero();
                        }
                        const auto row = index / feature_count;
                        const auto column = index % feature_count;
                        return feature_matrix[row][column];
                    }
                    throw std::runtime_error("unsupported lazy FH public label: " + std::string(label));
                };

                struct PointWeightRef
                {
                    std::size_t point_index = 0;
                    std::size_t direct_index = 0;
                    const algebra::DomainEvaluationWeights *weight_entry = nullptr;
                    bool direct = false;
                };

                std::vector<PointWeightRef> weighted_points;
                weighted_points.reserve(points.size());
                for (std::size_t point_index = 0; point_index < points.size(); ++point_index)
                {
                    const auto &weight_entry = domain_weights(domain, points[point_index]);
                    if (weight_entry.direct_index.has_value())
                    {
                        const auto direct_index = *weight_entry.direct_index;
                        for (std::size_t label_index = 0; label_index < labels.size(); ++label_index)
                        {
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
                if (weighted_points.empty())
                {
                    return out;
                }
                for (auto &point_ref : weighted_points)
                {
                    point_ref.weight_entry = &domain_weights(domain, points[point_ref.point_index]);
                }

                struct FhAccumulatorSet
                {
                    mcl::Fr q_tbl;
                    mcl::Fr row_tbl;
                    mcl::Fr col_tbl;
                    mcl::Fr th;
                    mcl::Fr q_qry;
                    mcl::Fr row_qry;
                    mcl::Fr col_qry;
                    mcl::Fr abs_qry;
                };
                auto clear_accumulators = [](FhAccumulatorSet *accum)
                {
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
                for (std::size_t column = 0; column < feature_count; ++column)
                {
                    column_values[column] = FieldElement(column).native();
                }

                auto accumulate_dataset_range = [&](std::size_t begin_row, std::size_t end_row)
                {
                    std::vector<FhAccumulatorSet> partial(weighted_points.size());
                    for (auto &accum : partial)
                    {
                        clear_accumulators(&accum);
                    }
                    for (std::size_t row = begin_row; row < end_row; ++row)
                    {
                        const auto row_native = FieldElement(row).native();
                        const auto &feature_row = feature_matrix[row];
                        const auto base = row * feature_count;
                        for (std::size_t column = 0; column < feature_count; ++column)
                        {
                            const auto index = base + column;
                            for (std::size_t point_offset = 0; point_offset < weighted_points.size(); ++point_offset)
                            {
                                const auto &weight =
                                    weighted_points[point_offset].weight_entry->native_weights[index];
                                auto &accum = partial[point_offset];
                                if (need_q_tbl)
                                {
                                    mcl::Fr::add(accum.q_tbl, accum.q_tbl, weight);
                                }
                                if (need_row_tbl)
                                {
                                    mcl::Fr term;
                                    mcl::Fr::mul(term, row_native, weight);
                                    mcl::Fr::add(accum.row_tbl, accum.row_tbl, term);
                                }
                                if (need_col_tbl)
                                {
                                    mcl::Fr term;
                                    mcl::Fr::mul(term, column_values[column], weight);
                                    mcl::Fr::add(accum.col_tbl, accum.col_tbl, term);
                                }
                                if (need_th)
                                {
                                    mcl::Fr term;
                                    mcl::Fr::mul(term, feature_row[column].native(), weight);
                                    mcl::Fr::add(accum.th, accum.th, term);
                                }
                                if (full_feature_identity)
                                {
                                    continue;
                                }
                            }
                        }
                    }
                    return partial;
                };

                auto combine_accumulators = [&](std::vector<FhAccumulatorSet> *target,
                                                const std::vector<FhAccumulatorSet> &source)
                {
                    for (std::size_t i = 0; i < target->size(); ++i)
                    {
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
                for (auto &accum : accumulators)
                {
                    clear_accumulators(&accum);
                }
                if (dataset_valid >= (1ULL << 20) && cpu_count > 1)
                {
                    const auto task_count =
                        std::min<std::size_t>(cpu_count, std::max<std::size_t>(1, dataset_rows / 2048));
                    if (task_count > 1)
                    {
                        const auto chunk_size = (dataset_rows + task_count - 1) / task_count;
                        std::vector<std::future<std::vector<FhAccumulatorSet>>> futures;
                        futures.reserve(task_count);
                        for (std::size_t task = 0; task < task_count; ++task)
                        {
                            const auto begin_row = task * chunk_size;
                            const auto end_row = std::min(dataset_rows, begin_row + chunk_size);
                            if (begin_row >= end_row)
                            {
                                break;
                            }
                            futures.push_back(std::async(
                                std::launch::async,
                                [&, begin_row, end_row]()
                                {
                                    return accumulate_dataset_range(begin_row, end_row);
                                }));
                        }
                        for (auto &future : futures)
                        {
                            combine_accumulators(&accumulators, future.get());
                        }
                    }
                    else
                    {
                        combine_accumulators(&accumulators, accumulate_dataset_range(0, dataset_rows));
                    }
                }
                else
                {
                    combine_accumulators(&accumulators, accumulate_dataset_range(0, dataset_rows));
                }

                if (!full_feature_identity)
                {
                    auto accumulate_local_range = [&](std::size_t begin_row, std::size_t end_row)
                    {
                        std::vector<FhAccumulatorSet> partial(weighted_points.size());
                        for (auto &accum : partial)
                        {
                            clear_accumulators(&accum);
                        }
                        for (std::size_t row = begin_row; row < end_row; ++row)
                        {
                            const auto row_native = FieldElement(row).native();
                            const auto absolute_id_native = FieldElement(context_.local.absolute_ids[row]).native();
                            const auto base = row * feature_count;
                            for (std::size_t column = 0; column < feature_count; ++column)
                            {
                                const auto index = base + column;
                                for (std::size_t point_offset = 0; point_offset < weighted_points.size(); ++point_offset)
                                {
                                    const auto &weight =
                                        weighted_points[point_offset].weight_entry->native_weights[index];
                                    auto &accum = partial[point_offset];
                                    if (need_q_qry)
                                    {
                                        mcl::Fr::add(accum.q_qry, accum.q_qry, weight);
                                    }
                                    if (need_row_qry)
                                    {
                                        mcl::Fr term;
                                        mcl::Fr::mul(term, row_native, weight);
                                        mcl::Fr::add(accum.row_qry, accum.row_qry, term);
                                    }
                                    if (need_col_qry)
                                    {
                                        mcl::Fr term;
                                        mcl::Fr::mul(term, column_values[column], weight);
                                        mcl::Fr::add(accum.col_qry, accum.col_qry, term);
                                    }
                                    if (need_abs_qry)
                                    {
                                        mcl::Fr term;
                                        mcl::Fr::mul(term, absolute_id_native, weight);
                                        mcl::Fr::add(accum.abs_qry, accum.abs_qry, term);
                                    }
                                }
                            }
                        }
                        return partial;
                    };
                    if (local_valid >= (1ULL << 20) && cpu_count > 1)
                    {
                        const auto task_count =
                            std::min<std::size_t>(cpu_count, std::max<std::size_t>(1, local_rows / 2048));
                        if (task_count > 1)
                        {
                            const auto chunk_size = (local_rows + task_count - 1) / task_count;
                            std::vector<std::future<std::vector<FhAccumulatorSet>>> futures;
                            futures.reserve(task_count);
                            for (std::size_t task = 0; task < task_count; ++task)
                            {
                                const auto begin_row = task * chunk_size;
                                const auto end_row = std::min(local_rows, begin_row + chunk_size);
                                if (begin_row >= end_row)
                                {
                                    break;
                                }
                                futures.push_back(std::async(
                                    std::launch::async,
                                    [&, begin_row, end_row]()
                                    {
                                        return accumulate_local_range(begin_row, end_row);
                                    }));
                            }
                            for (auto &future : futures)
                            {
                                combine_accumulators(&accumulators, future.get());
                            }
                        }
                        else
                        {
                            combine_accumulators(&accumulators, accumulate_local_range(0, local_rows));
                        }
                    }
                    else
                    {
                        combine_accumulators(&accumulators, accumulate_local_range(0, local_rows));
                    }
                }

                for (std::size_t point_offset = 0; point_offset < weighted_points.size(); ++point_offset)
                {
                    const auto point_index = weighted_points[point_offset].point_index;
                    const auto &accum = accumulators[point_offset];
                    if (q_tbl_index < labels.size())
                    {
                        out[point_index][q_tbl_index] = FieldElement::from_native(accum.q_tbl);
                    }
                    if (row_tbl_index < labels.size())
                    {
                        out[point_index][row_tbl_index] = FieldElement::from_native(accum.row_tbl);
                    }
                    if (col_tbl_index < labels.size())
                    {
                        out[point_index][col_tbl_index] = FieldElement::from_native(accum.col_tbl);
                    }
                    if (th_index < labels.size())
                    {
                        out[point_index][th_index] = FieldElement::from_native(accum.th);
                    }
                    if (full_feature_identity)
                    {
                        if (q_qry_index < labels.size())
                        {
                            out[point_index][q_qry_index] = FieldElement::from_native(accum.q_tbl);
                        }
                        if (row_qry_index < labels.size())
                        {
                            out[point_index][row_qry_index] = FieldElement::from_native(accum.row_tbl);
                        }
                        if (col_qry_index < labels.size())
                        {
                            out[point_index][col_qry_index] = FieldElement::from_native(accum.col_tbl);
                        }
                        if (abs_qry_index < labels.size())
                        {
                            out[point_index][abs_qry_index] = FieldElement::from_native(accum.row_tbl);
                        }
                        continue;
                    }
                    if (q_qry_index < labels.size())
                    {
                        out[point_index][q_qry_index] = FieldElement::from_native(accum.q_qry);
                    }
                    if (row_qry_index < labels.size())
                    {
                        out[point_index][row_qry_index] = FieldElement::from_native(accum.row_qry);
                    }
                    if (col_qry_index < labels.size())
                    {
                        out[point_index][col_qry_index] = FieldElement::from_native(accum.col_qry);
                    }
                    if (abs_qry_index < labels.size())
                    {
                        out[point_index][abs_qry_index] = FieldElement::from_native(accum.abs_qry);
                    }
                }
                return out;
            }

            std::vector<FieldElement> evaluate_lazy_full_feature_lookup_trace_group(
                const std::vector<std::string> &labels,
                const FieldElement &point)
            {
                std::vector<FieldElement> out(labels.size(), FieldElement::zero());
                if (labels.empty())
                {
                    return out;
                }
                const auto eta_feat = challenges_.at("eta_feat");
                const auto eta_feature_value = eta_feat.pow(2);
                const bool full_feature_identity = full_graph_feature_identity_enabled(context_);
                const auto table_index = label_index_or_npos(labels, "P_Table_feat");
                const auto query_index = label_index_or_npos(labels, "P_Query_feat");
                const auto multiplicity_index = label_index_or_npos(labels, "P_m_feat");
                const auto accumulator_index = label_index_or_npos(labels, "P_R_feat");
                const auto cache_suffix = "@" + algebra::point_cache_key(point);
                auto load_cached = [&](const std::string &label, FieldElement *value) -> bool
                {
                    const auto it = value_cache_.find(label + cache_suffix);
                    if (it == value_cache_.end())
                    {
                        return false;
                    }
                    *value = it->second;
                    return true;
                };

                std::vector<std::string> missing_public_labels;
                auto ensure_public_value = [&](const std::string &label, FieldElement *value)
                {
                    if (load_cached(label, value))
                    {
                        return;
                    }
                    if (std::find(missing_public_labels.begin(), missing_public_labels.end(), label) == missing_public_labels.end())
                    {
                        missing_public_labels.push_back(label);
                    }
                };

                FieldElement row_tbl = FieldElement::zero();
                FieldElement col_tbl = FieldElement::zero();
                FieldElement row_qry = FieldElement::zero();
                FieldElement col_qry = FieldElement::zero();
                FieldElement q_qry = FieldElement::zero();
                FieldElement t_h = FieldElement::zero();

                if (table_index < labels.size())
                {
                    ensure_public_value("P_Row_feat_tbl", &row_tbl);
                    ensure_public_value("P_Col_feat_tbl", &col_tbl);
                    ensure_public_value("P_T_H", &t_h);
                }
                if (query_index < labels.size())
                {
                    if (full_feature_identity)
                    {
                        ensure_public_value("P_Row_feat_tbl", &row_tbl);
                        ensure_public_value("P_Col_feat_tbl", &col_tbl);
                        ensure_public_value("P_T_H", &t_h);
                    }
                    else
                    {
                        ensure_public_value("P_Row_feat_qry", &row_qry);
                        ensure_public_value("P_Col_feat_qry", &col_qry);
                        ensure_public_value("P_T_H", &t_h);
                    }
                }
                if (multiplicity_index < labels.size())
                {
                    ensure_public_value(full_feature_identity ? "P_Q_tbl_feat" : "P_Q_qry_feat", &q_qry);
                }

                if (!missing_public_labels.empty())
                {
                    const auto values = evaluate_lazy_large_fh_public_group(missing_public_labels, point);
                    for (std::size_t i = 0; i < missing_public_labels.size(); ++i)
                    {
                        value_cache_.emplace(missing_public_labels[i] + cache_suffix, values[i]);
                    }
                    if (table_index < labels.size())
                    {
                        load_cached("P_Row_feat_tbl", &row_tbl);
                        load_cached("P_Col_feat_tbl", &col_tbl);
                        load_cached("P_T_H", &t_h);
                    }
                    if (query_index < labels.size())
                    {
                        if (full_feature_identity)
                        {
                            load_cached("P_Row_feat_tbl", &row_tbl);
                            load_cached("P_Col_feat_tbl", &col_tbl);
                            row_qry = row_tbl;
                            col_qry = col_tbl;
                            load_cached("P_T_H", &t_h);
                        }
                        else
                        {
                            load_cached("P_Row_feat_qry", &row_qry);
                            load_cached("P_Col_feat_qry", &col_qry);
                            load_cached("P_T_H", &t_h);
                        }
                    }
                    if (multiplicity_index < labels.size())
                    {
                        load_cached(full_feature_identity ? "P_Q_tbl_feat" : "P_Q_qry_feat", &q_qry);
                    }
                }
                if (full_feature_identity && query_index < labels.size())
                {
                    row_qry = row_tbl;
                    col_qry = col_tbl;
                }

                if (table_index < labels.size())
                {
                    out[table_index] = row_tbl + eta_feat * col_tbl + eta_feature_value * t_h;
                }
                if (query_index < labels.size())
                {
                    out[query_index] = row_qry + eta_feat * col_qry + eta_feature_value * t_h;
                }
                if (multiplicity_index < labels.size())
                {
                    out[multiplicity_index] = q_qry;
                }
                if (accumulator_index < labels.size())
                {
                    out[accumulator_index] = FieldElement::zero();
                }
                for (std::size_t i = 0; i < labels.size(); ++i)
                {
                    const auto polynomial_it = trace_.polynomials.find(labels[i]);
                    if (polynomial_it == trace_.polynomials.end()
                        || !polynomial_it->second.is_zero_domain_blinded())
                        continue;
                    out[i] += polynomial_it->second.domain->zero_polynomial_eval(point)
                        * algebra::horner(
                            polynomial_it->second.zero_domain_blinder, point);
                }
                return out;
            }

        public:
            std::vector<std::vector<FieldElement>> collect_named_values(
                const std::vector<std::string> &labels,
                const std::vector<FieldElement> &points)
            {
                std::vector<std::vector<FieldElement>> out(
                    labels.size(),
                    std::vector<FieldElement>(points.size(), FieldElement::zero()));
                if (labels.empty() || points.empty())
                {
                    return out;
                }

                precompute_named(labels, points);
                for (std::size_t i = 0; i < labels.size(); ++i)
                {
                    for (std::size_t point_index = 0; point_index < points.size(); ++point_index)
                    {
                        out[i][point_index] = eval_named(labels[i], points[point_index]);
                    }
                }
                return out;
            }

            std::vector<std::vector<FieldElement>> collect_named_values_from_cache(
                const std::vector<std::string> &labels,
                const std::vector<FieldElement> &points)
            {
                std::vector<std::vector<FieldElement>> out(
                    labels.size(),
                    std::vector<FieldElement>(points.size(), FieldElement::zero()));
                for (std::size_t i = 0; i < labels.size(); ++i)
                {
                    for (std::size_t point_index = 0; point_index < points.size(); ++point_index)
                    {
                        out[i][point_index] = eval_named(labels[i], points[point_index]);
                    }
                }
                return out;
            }

            void precompute_named(
                const std::vector<std::string> &labels,
                const std::vector<FieldElement> &points)
            {
                const bool enable_backend_precompute = util::route2_options().fft_backend_upgrade;

                auto cache_group_values = [&](const std::vector<std::string> &group_labels,
                                              const FieldElement &point,
                                              const std::vector<FieldElement> &values)
                {
                    const auto cache_suffix = "@" + algebra::point_cache_key(point);
                    for (std::size_t i = 0; i < group_labels.size(); ++i)
                    {
                        value_cache_.emplace(group_labels[i] + cache_suffix, values[i]);
                    }
                };

                if (lazy_large_fh_public_enabled(context_))
                {
                    std::vector<std::string> lazy_public_labels;
                    lazy_public_labels.reserve(labels.size());
                    for (const auto &label : labels)
                    {
                        if (is_lazy_large_fh_public_label(label))
                        {
                            lazy_public_labels.push_back(label);
                        }
                    }
                    if (!lazy_public_labels.empty())
                    {
                        std::vector<std::size_t> missing_point_indices;
                        std::vector<std::shared_ptr<const std::vector<FieldElement>>> cached_values(points.size());
                        missing_point_indices.reserve(points.size());
                        for (std::size_t point_index = 0; point_index < points.size(); ++point_index)
                        {
                            cached_values[point_index] = shared_public_evaluation_cache()->lookup(
                                context_key_,
                                context_.domains.fh,
                                lazy_public_labels,
                                points[point_index]);
                            if (cached_values[point_index] == nullptr)
                            {
                                missing_point_indices.push_back(point_index);
                            }
                        }
                        if (!missing_point_indices.empty())
                        {
                            std::vector<FieldElement> missing_points;
                            missing_points.reserve(missing_point_indices.size());
                            for (const auto point_index : missing_point_indices)
                            {
                                missing_points.push_back(points[point_index]);
                            }
                            const auto missing_values =
                                missing_points.size() == 1
                                    ? std::vector<std::vector<FieldElement>>{
                                          evaluate_lazy_large_fh_public_group(lazy_public_labels, missing_points.front())}
                                    : evaluate_lazy_large_fh_public_group_points(lazy_public_labels, missing_points);
                            for (std::size_t i = 0; i < missing_point_indices.size(); ++i)
                            {
                                cached_values[missing_point_indices[i]] = shared_public_evaluation_cache()->store(
                                    context_key_,
                                    context_.domains.fh,
                                    lazy_public_labels,
                                    points[missing_point_indices[i]],
                                    missing_values[i]);
                            }
                        }
                        for (std::size_t point_index = 0; point_index < points.size(); ++point_index)
                        {
                            const auto &point = points[point_index];
                            const auto &values = *cached_values[point_index];
                            cache_group_values(lazy_public_labels, point, values);
                            if (full_graph_feature_identity_enabled(context_))
                            {
                                const auto cache_suffix = "@" + algebra::point_cache_key(point);
                                auto seed_alias = [&](std::string_view target, std::string_view source)
                                {
                                    if (std::find(lazy_public_labels.begin(), lazy_public_labels.end(), target) == lazy_public_labels.end())
                                    {
                                        return;
                                    }
                                    const auto it = std::find(lazy_public_labels.begin(), lazy_public_labels.end(), source);
                                    if (it == lazy_public_labels.end())
                                    {
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

                if (lazy_full_feature_lookup_trace_enabled(context_))
                {
                    std::vector<std::string> lazy_trace_labels;
                    lazy_trace_labels.reserve(labels.size());
                    for (const auto &label : labels)
                    {
                        if (is_lazy_full_feature_lookup_trace_label(label))
                        {
                            lazy_trace_labels.push_back(label);
                        }
                    }
                    if (!lazy_trace_labels.empty())
                    {
                        std::vector<std::string> required_public_labels;
                        required_public_labels.reserve(8);
                        auto append_required_public = [&](std::string_view label)
                        {
                            if (std::find(required_public_labels.begin(), required_public_labels.end(), label) == required_public_labels.end())
                            {
                                required_public_labels.emplace_back(label);
                            }
                        };
                        if (std::find(lazy_trace_labels.begin(), lazy_trace_labels.end(), "P_Table_feat") != lazy_trace_labels.end())
                        {
                            append_required_public("P_Row_feat_tbl");
                            append_required_public("P_Col_feat_tbl");
                            append_required_public("P_T_H");
                        }
                        if (std::find(lazy_trace_labels.begin(), lazy_trace_labels.end(), "P_Query_feat") != lazy_trace_labels.end())
                        {
                            append_required_public("P_Row_feat_qry");
                            append_required_public("P_Col_feat_qry");
                            append_required_public("P_T_H");
                        }
                        if (std::find(lazy_trace_labels.begin(), lazy_trace_labels.end(), "P_m_feat") != lazy_trace_labels.end())
                        {
                            append_required_public("P_Q_qry_feat");
                        }
                        if (std::find(labels.begin(), labels.end(), "P_H") != labels.end())
                        {
                            append_required_public("P_T_H");
                        }
                        if (!required_public_labels.empty() && !lazy_large_fh_public_enabled(context_))
                        {
                            precompute_named(required_public_labels, points);
                            for (const auto &point : points)
                            {
                                const auto cache_suffix = "@" + algebra::point_cache_key(point);
                                for (const auto &label : required_public_labels)
                                {
                                    if (value_cache_.find(label + cache_suffix) == value_cache_.end())
                                    {
                                        value_cache_.emplace(label + cache_suffix, eval_named(label, point));
                                    }
                                }
                                if (std::find(labels.begin(), labels.end(), "P_H") != labels.end())
                                {
                                    if (value_cache_.find("P_H" + cache_suffix) == value_cache_.end())
                                    {
                                        value_cache_.emplace("P_H" + cache_suffix, eval_named("P_T_H", point));
                                    }
                                }
                            }
                        }
                        for (const auto &point : points)
                        {
                            cache_group_values(
                                lazy_trace_labels,
                                point,
                                evaluate_lazy_full_feature_lookup_trace_group(lazy_trace_labels, point));
                        }
                    }
                }

                if (!enable_backend_precompute)
                {
                    return;
                }

                std::unordered_map<std::string, std::pair<std::shared_ptr<algebra::RootOfUnityDomain>, std::vector<std::string>>> groups;
                for (const auto &label : labels)
                {
                    const auto *polynomial = lookup_polynomial(label);
                    if (polynomial == nullptr || polynomial->basis != algebra::PolynomialBasis::Evaluation || polynomial->domain == nullptr)
                    {
                        continue;
                    }
                    const auto group_key = polynomial->domain->name + ":" + std::to_string(polynomial->domain->size);
                    auto &group = groups[group_key];
                    if (group.second.empty())
                    {
                        group.first = polynomial->domain;
                    }
                    group.second.push_back(label);
                }

                auto evaluate_group_at_point = [&](
                                                   const std::shared_ptr<algebra::RootOfUnityDomain> &domain,
                                                   const algebra::PackedEvaluationBackend &backend,
                                                   const std::vector<std::string> &group_labels,
                                                   const FieldElement &point,
                                                   bool is_public_group)
                {
                    const auto prep_start = Clock::now();
                    const auto weight_fetch_start = Clock::now();
                    const auto &weight_entry = domain_weights(domain, point);
                    const auto weight_fetch_ms = elapsed_ms(weight_fetch_start, Clock::now());
                    if (metrics_ != nullptr && domain->name == "FH")
                    {
                        metrics_->fh_barycentric_weight_fetch_ms += weight_fetch_ms;
                        const auto prep_ms = elapsed_ms(prep_start, Clock::now());
                        metrics_->fh_eval_prep_ms += prep_ms;
                        metrics_->fh_opening_eval_prep_ms += prep_ms;
                    }
                    std::vector<FieldElement> values;
                    const auto eval_start = Clock::now();
                    if (weight_entry.direct_index.has_value())
                    {
                        values = backend.values_at_direct_index(group_labels, *weight_entry.direct_index);
                    }
                    else
                    {
                        values = backend.evaluate_with_native_weights(
                            group_labels,
                            weight_entry.native_weights);
                    }
                    if (metrics_ != nullptr && domain->name == "FH")
                    {
                        const auto eval_ms = elapsed_ms(eval_start, Clock::now());
                        metrics_->fh_interpolation_ms += eval_ms;
                        if (is_public_group)
                        {
                            metrics_->fh_public_poly_interp_ms += eval_ms;
                        }
                        else
                        {
                            metrics_->fh_lagrange_eval_ms += eval_ms;
                        }
                    }
                    return values;
                };

                for (auto &[group_key, group] : groups)
                {
                    (void)group_key;
                    if (backend_registry_ == nullptr)
                    {
                        continue;
                    }
                    const auto *backend = backend_registry_->find(group.first);
                    if (backend == nullptr)
                    {
                        continue;
                    }
                    std::vector<std::string> public_labels;
                    std::vector<std::string> trace_labels;
                    public_labels.reserve(group.second.size());
                    trace_labels.reserve(group.second.size());
                    for (const auto &label : group.second)
                    {
                        if (context_.public_polynomials.contains(label))
                        {
                            public_labels.push_back(label);
                        }
                        else
                        {
                            trace_labels.push_back(label);
                        }
                    }
                    auto store_group_values = [&](const std::vector<std::string> &group_labels,
                                                  const FieldElement &point,
                                                  const std::vector<FieldElement> &values)
                    {
                        const auto copy_start = Clock::now();
                        auto proof_values = values;
                        for (std::size_t i = 0; i < group_labels.size(); ++i)
                        {
                            const auto *polynomial = lookup_polynomial(group_labels[i]);
                            if (polynomial != nullptr && polynomial->is_zero_domain_blinded())
                            {
                                proof_values[i] += polynomial->domain->zero_polynomial_eval(point)
                                    * algebra::horner(polynomial->zero_domain_blinder, point);
                            }
                        }
                        cache_group_values(group_labels, point, proof_values);
                        if (metrics_ != nullptr && group.first->name == "FH")
                        {
                            metrics_->fh_copy_convert_ms += elapsed_ms(copy_start, Clock::now());
                        }
                    };
                    // 旋转点 kernel 只是大域 proving 热点的执行层优化。
                    // 小域无法摊薄初始化成本，因此仍保留原始打包扫掠路线。
                    if (!public_labels.empty() && group.first->name == "FH")
                    {
                        for (const auto &point : points)
                        {
                            const auto reuse_start = Clock::now();
                            auto cached_values = shared_public_evaluation_cache()->lookup(
                                context_key_,
                                group.first,
                                public_labels,
                                point);
                            if (cached_values != nullptr)
                            {
                                if (metrics_ != nullptr)
                                {
                                    metrics_->fh_public_eval_reuse_ms += elapsed_ms(reuse_start, Clock::now());
                                }
                            }
                            else
                            {
                                cached_values = shared_public_evaluation_cache()->store(
                                    context_key_,
                                    group.first,
                                    public_labels,
                                    point,
                                    evaluate_group_at_point(group.first, *backend, public_labels, point, true));
                            }
                            store_group_values(public_labels, point, *cached_values);
                        }
                    }
                    const auto &eval_labels = group.first->name == "FH" ? trace_labels : group.second;
                    if (eval_labels.empty())
                    {
                        continue;
                    }
                    const bool allow_shared_trace_reuse =
                        group.first->name == "FH" || group.first->name == "edge";
                    if (allow_shared_trace_reuse)
                    {
                        bool all_cached = true;
                        std::vector<std::shared_ptr<const std::vector<FieldElement>>> cached_values(points.size());
                        for (std::size_t point_index = 0; point_index < points.size(); ++point_index)
                        {
                            cached_values[point_index] = shared_trace_evaluation_cache()->lookup(
                                context_key_,
                                group.first,
                                eval_labels,
                                points[point_index]);
                            if (cached_values[point_index] == nullptr)
                            {
                                all_cached = false;
                                break;
                            }
                        }
                        if (all_cached)
                        {
                            for (std::size_t point_index = 0; point_index < points.size(); ++point_index)
                            {
                                store_group_values(eval_labels, points[point_index], *cached_values[point_index]);
                            }
                            continue;
                        }
                    }
                    if (util::route2_options().fft_kernel_upgrade && group.first->size >= 256)
                    {
                        if (const auto shifts = rotated_point_shifts(group.first, points); shifts.has_value())
                        {
                            const auto prep_start = Clock::now();
                            const auto weight_fetch_start = Clock::now();
                            const auto &weight_entry = domain_weights(group.first, points.front());
                            if (metrics_ != nullptr && group.first->name == "FH")
                            {
                                metrics_->fh_barycentric_weight_fetch_ms += elapsed_ms(weight_fetch_start, Clock::now());
                                const auto prep_ms = elapsed_ms(prep_start, Clock::now());
                                metrics_->fh_eval_prep_ms += prep_ms;
                                metrics_->fh_opening_eval_prep_ms += prep_ms;
                            }
                            if (weight_entry.direct_index.has_value())
                            {
                                const auto eval_start = Clock::now();
                                const auto domain_mask = group.first->size - 1U;
                                for (std::size_t point_index = 0; point_index < points.size(); ++point_index)
                                {
                                    const auto direct_index = (*weight_entry.direct_index + (*shifts)[point_index]) & domain_mask;
                                    const auto values = backend->values_at_direct_index(eval_labels, direct_index);
                                    store_group_values(eval_labels, points[point_index], values);
                                    if (allow_shared_trace_reuse)
                                    {
                                        shared_trace_evaluation_cache()->store(
                                            context_key_,
                                            group.first,
                                            eval_labels,
                                            points[point_index],
                                            values);
                                    }
                                }
                                if (metrics_ != nullptr && group.first->name == "FH")
                                {
                                    const auto eval_ms = elapsed_ms(eval_start, Clock::now());
                                    metrics_->fh_interpolation_ms += eval_ms;
                                    metrics_->fh_lagrange_eval_ms += eval_ms;
                                }
                                continue;
                            }

                            // Route2 这里只利用单位根域上的循环位移恒等式：
                            // 同一域内 `{z, omega z, ...}` 的权重向量彼此只是循环平移。
                            // 因此可以把多个点打包求值，而不改变 proof 中实际打开的对象和点位。
                            const auto eval_start = Clock::now();
                            const auto batched_values =
                                backend->evaluate_with_native_weight_rotations(
                                    eval_labels,
                                    weight_entry.native_weights,
                                    *shifts);
                            if (metrics_ != nullptr && group.first->name == "FH")
                            {
                                const auto eval_ms = elapsed_ms(eval_start, Clock::now());
                                metrics_->fh_interpolation_ms += eval_ms;
                                metrics_->fh_lagrange_eval_ms += eval_ms;
                            }
                            for (std::size_t point_index = 0; point_index < points.size(); ++point_index)
                            {
                                store_group_values(eval_labels, points[point_index], batched_values[point_index]);
                                if (allow_shared_trace_reuse)
                                {
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
                    for (const auto &point : points)
                    {
                        const auto values = evaluate_group_at_point(group.first, *backend, eval_labels, point, false);
                        store_group_values(eval_labels, point, values);
                        if (allow_shared_trace_reuse)
                        {
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
            FieldElement eval_polynomial(const Polynomial &polynomial, const FieldElement &point)
            {
                if (polynomial.basis == algebra::PolynomialBasis::Coefficient)
                {
                    return polynomial.evaluate(point);
                }
                if (polynomial.domain == nullptr)
                {
                    throw std::runtime_error("evaluation polynomial is missing its domain");
                }

                const auto &weight_entry = domain_weights(polynomial.domain, point);
                const auto &values = polynomial.values();
                if (weight_entry.direct_index.has_value())
                {
                    return polynomial.evaluate(point);
                }
                // `parallel_fft` 只改变域权重的归约方式，opened value 本身必须保持不变。
                auto value = algebra::dot_product_native_weights(values, weight_entry.native_weights);
                if (polynomial.is_zero_domain_blinded())
                {
                    value += polynomial.domain->zero_polynomial_eval(point)
                        * algebra::horner(polynomial.zero_domain_blinder, point);
                }
                return value;
            }

            const algebra::DomainEvaluationWeights &domain_weights(
                const std::shared_ptr<algebra::RootOfUnityDomain> &domain,
                const FieldElement &point)
            {
                return domain_weight_cache_->get(domain, point);
            }

            const ProtocolContext &context_;
            const TraceArtifacts &trace_;
            const std::map<std::string, FieldElement> &challenges_;
            std::shared_ptr<const ProofEvaluationBackendRegistry> backend_registry_;
            std::shared_ptr<algebra::DomainEvaluationWeightCache> domain_weight_cache_;
            std::string context_key_;
            RunMetrics *metrics_ = nullptr;
            std::unordered_map<std::string, FieldElement> value_cache_;
        };

        std::shared_ptr<const mcl::Ntt<mcl::Fr>> ntt_plan(std::size_t size)
        {
            return algebra::ntt_plan(size);
        }

        template <class Fn>
        void parallel_for_chunks_static(
            std::size_t count,
            std::size_t min_parallel_count,
            std::size_t max_worker_count,
            Fn &&fn)
        {
            const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
            const auto available_workers =
                max_worker_count == 0 ? cpu_count : std::min<std::size_t>(cpu_count, max_worker_count);
            if (count < min_parallel_count || available_workers <= 1)
            {
                fn(0, count);
                return;
            }
            const auto task_count = std::min<std::size_t>(
                available_workers,
                std::max<std::size_t>(2, count / min_parallel_count));
            const auto chunk_size = (count + task_count - 1) / task_count;
            std::vector<std::future<void>> futures;
            futures.reserve(task_count);
            for (std::size_t task = 0; task < task_count; ++task)
            {
                const auto begin = task * chunk_size;
                const auto end = std::min(count, begin + chunk_size);
                if (begin >= end)
                {
                    break;
                }
                futures.push_back(std::async(std::launch::async, [&, begin, end]()
                                             { fn(begin, end); }));
            }
            for (auto &future : futures)
            {
                future.get();
            }
        }

        std::shared_ptr<const std::vector<mcl::Fr>> inverse_shift_powers(
            std::size_t size,
            const FieldElement &shift)
        {
            static std::mutex mutex;
            static std::unordered_map<std::string, std::shared_ptr<const std::vector<mcl::Fr>>> cache;
            const auto key = std::to_string(size) + ":" + shift.to_string();
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (const auto it = cache.find(key); it != cache.end())
                {
                    return it->second;
                }
            }

            auto powers = std::make_shared<std::vector<mcl::Fr>>(size);
            if (size != 0)
            {
                const auto inv_shift = shift.inv();
                const auto inv_shift_native = inv_shift.native();
                parallel_for_chunks_static(size, 1ULL << 12, 0, [&](std::size_t begin, std::size_t end)
                                           {
                    auto power = inv_shift.pow(static_cast<std::uint64_t>(begin)).native();
                    for (std::size_t i = begin; i < end; ++i) {
                        (*powers)[i] = power;
                        mcl::Fr::mul(power, power, inv_shift_native);
                    } });
            }

            std::lock_guard<std::mutex> lock(mutex);
            const auto [it, inserted] = cache.emplace(key, powers);
            (void)inserted;
            return it->second;
        }

        void copy_field_values_to_native(
            const std::vector<FieldElement> &values,
            std::vector<mcl::Fr> *out)
        {
            if (values.empty())
            {
                return;
            }
            if (out == nullptr || out->size() < values.size())
            {
                throw std::runtime_error("native copy target is smaller than field vector");
            }
            const auto *begin = FieldElement::native_array_data(values.data());
            std::copy(begin, begin + values.size(), out->data());
        }

        void copy_evaluation_values_to_native_domain(
            const std::vector<FieldElement> &values,
            std::size_t domain_size,
            std::vector<mcl::Fr> *out,
            const std::string &label)
        {
            if (out == nullptr)
            {
                throw std::runtime_error("native domain copy received null output for " + label);
            }
            if (values.size() > domain_size)
            {
                throw std::runtime_error("evaluation polynomial exceeds domain while materializing quotient: " + label);
            }
            out->resize(domain_size);
            if (!values.empty())
            {
                const auto *begin = FieldElement::native_array_data(values.data());
                std::copy(begin, begin + values.size(), out->data());
            }
            for (std::size_t i = values.size(); i < domain_size; ++i)
            {
                (*out)[i].clear();
            }
        }

        void trim_native_coefficients(std::vector<mcl::Fr> *coefficients)
        {
            while (coefficients->size() > 1 && coefficients->back().isZero())
            {
                coefficients->pop_back();
            }
        }

        void trim_field_coefficients(std::vector<FieldElement> *coefficients)
        {
            while (coefficients->size() > 1 && coefficients->back() == FieldElement::zero())
            {
                coefficients->pop_back();
            }
        }

        void batch_invert_native_in_place(std::vector<mcl::Fr> *values)
        {
            if (values == nullptr || values->empty())
            {
                return;
            }

            const auto count = values->size();
            const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
            if (count >= (1ULL << 14) && cpu_count > 1)
            {
                const auto chunk_count = std::min<std::size_t>(
                    cpu_count,
                    std::max<std::size_t>(2, count / (1ULL << 15)));
                const auto chunk_size = (count + chunk_count - 1) / chunk_count;
                std::vector<mcl::Fr> prefixes(count);
                std::vector<mcl::Fr> chunk_products(chunk_count);

                parallel_for_chunks_static(chunk_count, 1, 0, [&](std::size_t chunk_begin, std::size_t chunk_end)
                                           {
                    for (std::size_t chunk = chunk_begin; chunk < chunk_end; ++chunk) {
                        const auto begin = chunk * chunk_size;
                        const auto end = std::min(count, begin + chunk_size);
                        mcl::Fr accumulator;
                        accumulator = 1;
                        for (std::size_t i = begin; i < end; ++i) {
                            if ((*values)[i].isZero()) {
                                throw std::runtime_error("native batch inversion encountered zero");
                            }
                            prefixes[i] = accumulator;
                            mcl::Fr::mul(accumulator, accumulator, (*values)[i]);
                        }
                        chunk_products[chunk] = accumulator;
                    } });

                batch_invert_native_in_place(&chunk_products);

                parallel_for_chunks_static(chunk_count, 1, 0, [&](std::size_t chunk_begin, std::size_t chunk_end)
                                           {
                    for (std::size_t chunk = chunk_begin; chunk < chunk_end; ++chunk) {
                        const auto begin = chunk * chunk_size;
                        const auto end = std::min(count, begin + chunk_size);
                        auto suffix_inverse = chunk_products[chunk];
                        for (std::size_t i = end; i-- > begin;) {
                            const auto original = (*values)[i];
                            mcl::Fr::mul((*values)[i], suffix_inverse, prefixes[i]);
                            mcl::Fr::mul(suffix_inverse, suffix_inverse, original);
                        }
                    } });
                return;
            }

            std::vector<mcl::Fr> prefixes(values->size());
            mcl::Fr accumulator;
            accumulator = 1;
            for (std::size_t i = 0; i < values->size(); ++i)
            {
                if ((*values)[i].isZero())
                {
                    throw std::runtime_error("native batch inversion encountered zero");
                }
                prefixes[i] = accumulator;
                mcl::Fr::mul(accumulator, accumulator, (*values)[i]);
            }

            mcl::Fr suffix_inverse;
            mcl::Fr::inv(suffix_inverse, accumulator);
            for (std::size_t i = values->size(); i-- > 0;)
            {
                const auto original = (*values)[i];
                mcl::Fr::mul((*values)[i], suffix_inverse, prefixes[i]);
                mcl::Fr::mul(suffix_inverse, suffix_inverse, original);
            }
        }

        std::vector<mcl::Fr> prefix_selector_coefficients_native(
            const algebra::RootOfUnityDomain &domain,
            std::size_t valid_count,
            const std::string &label)
        {
            if (domain.size == 0)
            {
                throw std::runtime_error("prefix selector coefficient path requires a non-empty domain: " + label);
            }
            if (valid_count == 0)
            {
                std::vector<mcl::Fr> coefficients(1);
                coefficients[0].clear();
                return coefficients;
            }
            if (valid_count >= domain.size)
            {
                std::vector<mcl::Fr> coefficients(1);
                coefficients[0] = 1;
                return coefficients;
            }

            std::vector<mcl::Fr> coefficients(domain.size);
            std::vector<mcl::Fr> denominators(domain.size - 1);
            const auto valid_count_field = FieldElement(static_cast<std::uint64_t>(valid_count));
            const auto omega_inv_field = domain.omega.inv();
            const auto omega_inv_valid_field = domain.omega.pow(static_cast<std::uint64_t>(domain.size - valid_count));
            const mcl::Fr inv_size = domain.inv_size.native();
            const mcl::Fr omega_inv = omega_inv_field.native();
            const mcl::Fr omega_inv_valid = omega_inv_valid_field.native();

            mcl::Fr::mul(coefficients[0], valid_count_field.native(), inv_size);

            mcl::Fr one;
            one = 1;
            progress_note("prove:quotient:t_FH:selector:" + label + ":coefficients:series:start");
            parallel_for_chunks_static(domain.size - 1, 1ULL << 12, 0, [&](std::size_t begin, std::size_t end)
                                       {
                auto denominator_power = omega_inv_field.pow(static_cast<std::uint64_t>(begin + 1)).native();
                auto numerator_power = omega_inv_valid_field.pow(static_cast<std::uint64_t>(begin + 1)).native();
                for (std::size_t offset = begin; offset < end; ++offset) {
                    const auto k = offset + 1;
                    mcl::Fr::sub(denominators[offset], one, denominator_power);
                    mcl::Fr::sub(coefficients[k], one, numerator_power);
                    mcl::Fr::mul(denominator_power, denominator_power, omega_inv);
                    mcl::Fr::mul(numerator_power, numerator_power, omega_inv_valid);
                } });
            progress_note("prove:quotient:t_FH:selector:" + label + ":coefficients:series:done");
            progress_note("prove:quotient:t_FH:selector:" + label + ":coefficients:inverses:start");
            batch_invert_native_in_place(&denominators);
            progress_note("prove:quotient:t_FH:selector:" + label + ":coefficients:inverses:done");
            progress_note("prove:quotient:t_FH:selector:" + label + ":coefficients:scale:start");
            parallel_for_chunks_static(domain.size - 1, 1ULL << 12, 0, [&](std::size_t begin, std::size_t end)
                                       {
                for (std::size_t offset = begin; offset < end; ++offset) {
                    const auto k = offset + 1;
                    mcl::Fr::mul(coefficients[k], coefficients[k], denominators[offset]);
                    mcl::Fr::mul(coefficients[k], coefficients[k], inv_size);
                } });
            progress_note("prove:quotient:t_FH:selector:" + label + ":coefficients:scale:done");
            return coefficients;
        }

        std::vector<FieldElement> load_spilled_values(const TraceArtifacts::SpilledEvaluationPolynomial &spilled)
        {
            constexpr std::size_t kFieldBytes = 32;
            std::ifstream stream(spilled.path, std::ios::binary);
            if (!stream)
            {
                throw std::runtime_error("failed to open spilled quotient dependency: " + spilled.path);
            }
            std::array<std::uint8_t, kFieldBytes> bytes{};
            std::vector<FieldElement> values(spilled.size, FieldElement::zero());
            for (std::size_t i = 0; i < spilled.size; ++i)
            {
                stream.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
                if (!stream)
                {
                    throw std::runtime_error("failed to read spilled quotient dependency: " + spilled.path);
                }
                values[i] = FieldElement::from_little_endian_mod(bytes.data(), bytes.size());
            }
            return values;
        }

        std::shared_ptr<algebra::RootOfUnityDomain> context_domain_by_name(
            const ProtocolContext &context,
            const std::string &name)
        {
            if (name == "FH")
                return context.domains.fh;
            if (name == "edge")
                return context.domains.edge;
            if (name == "in")
                return context.domains.in;
            if (name == "d" || name == "d_h")
                return context.domains.d;
            if (name == "cat")
                return context.domains.cat;
            if (name == "C")
                return context.domains.c;
            if (name == "N")
                return context.domains.n;
            throw std::runtime_error("unknown quotient materialization domain: " + name);
        }

        std::shared_ptr<algebra::RootOfUnityDomain> quotient_domain(
            const ProtocolContext &context,
            const std::string &label)
        {
            if (label == "t_FH")
                return context.domains.fh;
            if (label == "t_edge")
                return context.domains.edge;
            if (label == "t_in")
                return context.domains.in;
            if (label == "t_d" || label == "t_d_h")
                return context.domains.d;
            if (label == "t_cat")
                return context.domains.cat;
            if (label == "t_C")
                return context.domains.c;
            if (label == "t_N")
                return context.domains.n;
            throw std::runtime_error("unknown quotient label: " + label);
        }

        std::optional<std::size_t> quotient_degree_bound_after_division(
            const std::string &label,
            std::size_t domain_size)
        {
            if (domain_size < 2)
            {
                return std::nullopt;
            }
            if (label == "t_FH" || label == "t_edge")
            {
                // FH/edge lookup relations multiply as many as three blinded
                // columns.  Each private column includes a registered
                // Z_D*R tail, so the old 2|D| interpolation coset aliased the
                // highest quotient coefficients.  A 4|D| coset covers the
                // compiled tail budget.  FH's three registered blinder
                // coefficients give degree at most 3|D|+2 after division.
                // The edge relation reaches that same bound because its
                // blinded lookup/state columns are multiplied by degree-|D|
                // selectors; the minimal security regression attains degree 3|D|+2.
                return 3 * domain_size + 2;
            }
            return std::nullopt;
        }

        std::size_t quotient_coset_multiplier(const std::string &label)
        {
            (void)label;
            return 4;
        }

        FieldElement quotient_coset_shift(const algebra::RootOfUnityDomain &domain)
        {
            auto shift = FieldElement(7);
            const auto one = FieldElement::one();
            const auto minus_one = -one;
            for (std::size_t attempt = 0; attempt < 64; ++attempt)
            {
                const auto shifted_domain_power = shift.pow(domain.size);
                if (shifted_domain_power != one && shifted_domain_power != minus_one)
                {
                    return shift;
                }
                shift += one;
            }
            throw std::runtime_error("failed to find non-overlapping quotient coset");
        }

        std::vector<mcl::Fr> coefficients_native_from_polynomial(const algebra::Polynomial &polynomial)
        {
            if (polynomial.basis == algebra::PolynomialBasis::Coefficient)
            {
                std::vector<mcl::Fr> out(polynomial.size());
                const auto &values = polynomial.values();
                copy_field_values_to_native(values, &out);
                if (out.empty())
                {
                    out.push_back(mcl::Fr(0));
                }
                trim_native_coefficients(&out);
                return out;
            }

            if (polynomial.domain == nullptr)
            {
                throw std::runtime_error("evaluation polynomial is missing its domain: " + polynomial.name);
            }
            std::vector<mcl::Fr> work;
            const auto &values = polynomial.values();
            copy_evaluation_values_to_native_domain(values, polynomial.domain->size, &work, polynomial.name);
            algebra::intt_in_place(work.data(), work.size());
            trim_native_coefficients(&work);
            return work;
        }

        template <class CoefficientAt>
        std::vector<FieldElement> coset_evaluate_coefficients_impl(
            std::size_t coefficient_count,
            std::size_t target_size,
            const FieldElement &shift,
            const std::string &label,
            const std::vector<mcl::Fr> *shift_powers,
            CoefficientAt &&coefficient_at)
        {
            if (coefficient_count > target_size)
            {
                throw std::runtime_error("polynomial degree exceeds quotient coset size: " + label);
            }
            std::vector<FieldElement> out(target_size);
            if (coefficient_count == 0)
            {
                return out;
            }
            if (shift_powers != nullptr && shift_powers->size() < coefficient_count)
            {
                throw std::runtime_error("insufficient quotient coset shift powers for " + label);
            }

            if (target_size >= 2 && coefficient_count <= target_size / 2)
            {
                const auto half_size = target_size / 2;
                std::vector<mcl::Fr> even_work(half_size);
                std::vector<mcl::Fr> odd_work(half_size);
                if (shift_powers != nullptr)
                {
                    const auto root = FieldElement::root_of_unity(target_size);
                    const auto root_native = root.native();
                    mcl::Fr root_power;
                    root_power = 1;
                    for (std::size_t i = 0; i < coefficient_count; ++i)
                    {
                        mcl::Fr::mul(even_work[i], coefficient_at(i), (*shift_powers)[i]);
                        mcl::Fr odd_shift_power;
                        mcl::Fr::mul(odd_shift_power, (*shift_powers)[i], root_power);
                        mcl::Fr::mul(odd_work[i], coefficient_at(i), odd_shift_power);
                        mcl::Fr::mul(root_power, root_power, root_native);
                    }
                }
                else
                {
                    mcl::Fr even_shift_power;
                    even_shift_power = 1;
                    mcl::Fr odd_shift_power;
                    odd_shift_power = 1;
                    const auto root = FieldElement::root_of_unity(target_size);
                    const auto even_shift_native = shift.native();
                    const auto odd_shift_native = (shift * root).native();
                    for (std::size_t i = 0; i < coefficient_count; ++i)
                    {
                        mcl::Fr::mul(even_work[i], coefficient_at(i), even_shift_power);
                        mcl::Fr::mul(odd_work[i], coefficient_at(i), odd_shift_power);
                        mcl::Fr::mul(even_shift_power, even_shift_power, even_shift_native);
                        mcl::Fr::mul(odd_shift_power, odd_shift_power, odd_shift_native);
                    }
                }
                for (std::size_t i = coefficient_count; i < half_size; ++i)
                {
                    even_work[i].clear();
                    odd_work[i].clear();
                }
                algebra::ntt_pair_in_place(
                    even_work.data(), odd_work.data(), half_size);
                auto *out_native = FieldElement::mutable_native_array_data(out.data());
                for (std::size_t i = 0; i < half_size; ++i)
                {
                    out_native[2 * i] = even_work[i];
                    out_native[2 * i + 1] = odd_work[i];
                }
                return out;
            }

            auto *work = FieldElement::mutable_native_array_data(out.data());
            if (shift_powers != nullptr)
            {
                for (std::size_t i = 0; i < coefficient_count; ++i)
                {
                    mcl::Fr::mul(work[i], coefficient_at(i), (*shift_powers)[i]);
                }
            }
            else
            {
                mcl::Fr shift_power;
                shift_power = 1;
                const auto shift_native = shift.native();
                for (std::size_t i = 0; i < coefficient_count; ++i)
                {
                    mcl::Fr::mul(work[i], coefficient_at(i), shift_power);
                    mcl::Fr::mul(shift_power, shift_power, shift_native);
                }
            }
            ntt_plan(target_size)->ntt(work);
            return out;
        }

        std::vector<FieldElement> coset_evaluate_coefficients(
            const std::vector<mcl::Fr> &coefficients,
            std::size_t target_size,
            const FieldElement &shift,
            const std::string &label,
            const std::vector<mcl::Fr> *shift_powers = nullptr)
        {
            return coset_evaluate_coefficients_impl(
                coefficients.size(),
                target_size,
                shift,
                label,
                shift_powers,
                [&](std::size_t index) -> const mcl::Fr &
                {
                    return coefficients[index];
                });
        }

        std::vector<FieldElement> coset_evaluate_polynomial(
            const algebra::Polynomial &polynomial,
            std::size_t target_size,
            const FieldElement &shift,
            const std::string &label,
            const std::vector<mcl::Fr> *shift_powers = nullptr)
        {
            thread_local std::vector<mcl::Fr> coefficient_work;

            auto active_coefficient_size = [](const std::vector<mcl::Fr> &values)
            {
                std::size_t active = values.size();
                while (active > 1 && values[active - 1].isZero())
                {
                    --active;
                }
                return active;
            };

            std::size_t coefficient_count = 0;
            if (polynomial.basis == algebra::PolynomialBasis::Coefficient)
            {
                const auto &coefficients = polynomial.values();
                coefficient_count = coefficients.size();
                while (coefficient_count > 1 && coefficients[coefficient_count - 1].is_zero())
                {
                    --coefficient_count;
                }
                if (coefficient_count > target_size)
                {
                    throw std::runtime_error("polynomial degree exceeds quotient coset size: " + label);
                }
                return coset_evaluate_coefficients_impl(
                    coefficient_count,
                    target_size,
                    shift,
                    label,
                    shift_powers,
                    [&](std::size_t index) -> const mcl::Fr &
                    {
                        return coefficients[index].native();
                    });
            }
            else
            {
                if (polynomial.domain == nullptr)
                {
                    throw std::runtime_error("evaluation polynomial is missing its domain: " + polynomial.name);
                }
                const auto &values = polynomial.values();
                copy_evaluation_values_to_native_domain(values, polynomial.domain->size, &coefficient_work, polynomial.name);
                algebra::intt_in_place(
                    coefficient_work.data(), coefficient_work.size());

                if (polynomial.is_zero_domain_blinded())
                {
                    const auto domain_size = polynomial.domain->size;
                    const auto required_size = domain_size + polynomial.zero_domain_blinder.size();
                    if (required_size > target_size)
                    {
                        throw std::runtime_error(
                            "zero-domain-blinded polynomial exceeds quotient coset size: " + label);
                    }
                    coefficient_work.resize(required_size);
                    // `mcl::Fr` default construction does not establish a
                    // canonical zero value.  These are new high-degree
                    // coefficient slots, so they must be cleared before the
                    // Z_D*R tail is accumulated into them.
                    for (std::size_t i = domain_size; i < required_size; ++i)
                    {
                        coefficient_work[i].clear();
                    }
                    for (std::size_t i = 0; i < polynomial.zero_domain_blinder.size(); ++i)
                    {
                        mcl::Fr::sub(
                            coefficient_work[i],
                            coefficient_work[i],
                            polynomial.zero_domain_blinder[i].native());
                        mcl::Fr::add(
                            coefficient_work[domain_size + i],
                            coefficient_work[domain_size + i],
                            polynomial.zero_domain_blinder[i].native());
                    }
                }

                coefficient_count = active_coefficient_size(coefficient_work);
                if (coefficient_count > target_size)
                {
                    throw std::runtime_error("polynomial degree exceeds quotient coset size: " + label);
                }
                return coset_evaluate_coefficients_impl(
                    coefficient_count,
                    target_size,
                    shift,
                    label,
                    shift_powers,
                    [&](std::size_t index) -> const mcl::Fr &
                    {
                        return coefficient_work[index];
                    });
            }
        }

        std::vector<FieldElement> coefficients_from_coset_evaluations_in_place(
            std::vector<FieldElement> values,
            const FieldElement &shift,
            std::string_view progress_prefix = {})
        {
            auto *work = FieldElement::mutable_native_array_data(values.data());
            if (!progress_prefix.empty())
            {
                progress_note(std::string(progress_prefix) + ":intt:start");
            }
            algebra::intt_in_place(work, values.size());
            if (!progress_prefix.empty())
            {
                progress_note(std::string(progress_prefix) + ":intt:done");
                progress_note(std::string(progress_prefix) + ":shift_scale:start");
            }

            const auto powers = inverse_shift_powers(values.size(), shift);
            parallel_for_chunks_static(values.size(), 1ULL << 12, 0, [&](std::size_t begin, std::size_t end)
                                       {
                for (std::size_t i = begin; i < end; ++i) {
                    mcl::Fr::mul(work[i], work[i], (*powers)[i]);
                } });
            if (!progress_prefix.empty())
            {
                progress_note(std::string(progress_prefix) + ":shift_scale:done");
            }
            trim_field_coefficients(&values);
            return values;
        }

        std::string fh_selector_boolean_quotient_coset_key(std::string_view selector_name)
        {
            return std::string(selector_name) + ":boolean_quotient_coset";
        }

        std::shared_ptr<const std::vector<FieldElement>> build_fh_selector_boolean_quotient_coset_material(
            const ProtocolContext &context,
            const std::string &selector_name)
        {
            if (context.domains.fh == nullptr)
            {
                throw std::runtime_error("missing FH domain for selector quotient coset material: " + selector_name);
            }
            const auto material_it = context.public_fh_coefficient_materials.find(selector_name);
            if (material_it == context.public_fh_coefficient_materials.end() || material_it->second == nullptr)
            {
                throw std::runtime_error("missing public FH selector coefficient material: " + selector_name);
            }
            const auto &domain = context.domains.fh;
            const auto target_size = domain->size * quotient_coset_multiplier("t_FH");
            const auto shift = quotient_coset_shift(*domain);
            auto q_coset = coset_evaluate_polynomial(*material_it->second, target_size, shift, selector_name);
            if (q_coset.size() != target_size)
            {
                throw std::runtime_error("FH selector quotient coset material size mismatch: " + selector_name);
            }

            const auto root = FieldElement::root_of_unity(target_size);
            const auto zero_cycle = target_size / domain->size;
            if (zero_cycle == 0 || target_size % domain->size != 0)
            {
                throw std::runtime_error("invalid FH selector quotient coset cycle: " + selector_name);
            }
            std::vector<mcl::Fr> inverse_zero(zero_cycle);
            auto zero_value = shift.pow(domain->size);
            const auto zero_step = root.pow(domain->size);
            const auto one = FieldElement::one();
            for (std::size_t i = 0; i < zero_cycle; ++i)
            {
                inverse_zero[i] = (zero_value - one).inv().native();
                zero_value *= zero_step;
            }

            auto *q_native = FieldElement::mutable_native_array_data(q_coset.data());
            const auto &one_native = one.native();
            parallel_for_chunks_static(q_coset.size(), 1ULL << 15, 0, [&](std::size_t begin, std::size_t end)
                                       {
                mcl::Fr one_minus_q;
                mcl::Fr q_boolean;
                for (std::size_t i = begin; i < end; ++i) {
                    mcl::Fr::sub(one_minus_q, one_native, q_native[i]);
                    mcl::Fr::mul(q_boolean, q_native[i], one_minus_q);
                    mcl::Fr::mul(q_native[i], q_boolean, inverse_zero[i % zero_cycle]);
                } });
            return std::make_shared<const std::vector<FieldElement>>(std::move(q_coset));
        }

        std::vector<std::string> quotient_dependency_labels(
            const ProtocolContext &context,
            const std::string &quotient_name);

        class QuotientPolynomialMaterializer
        {
        public:
            QuotientPolynomialMaterializer(
                const ProtocolContext &context,
                const TraceArtifacts &trace,
                const std::map<std::string, FieldElement> &challenges,
                std::string label)
                : context_(context),
                  trace_(trace),
                  challenges_(challenges),
                  label_(std::move(label)),
                  domain_(quotient_domain(context_, label_)),
                  target_size_(domain_->size * quotient_coset_multiplier(label_)),
                  rotation_step_(target_size_ / domain_->size),
                  shift_(quotient_coset_shift(*domain_))
            {
                if (domain_ == nullptr)
                {
                    throw std::runtime_error("quotient materialization is missing domain: " + label_);
                }
                if (rotation_step_ == 0 || target_size_ % domain_->size != 0)
                {
                    throw std::runtime_error("invalid quotient coset/domain size for " + label_);
                }
                if (const auto degree_bound = quotient_degree_bound_after_division(label_, domain_->size);
                    degree_bound.has_value() && *degree_bound >= target_size_)
                {
                    throw std::runtime_error(
                        "quotient coset is too small for derived degree bound: " + label_
                        + " degree_bound=" + std::to_string(*degree_bound)
                        + " target_size=" + std::to_string(target_size_));
                }
                const auto edge_head_count = context_.model.hidden_heads.size()
                    + context_.model.output_layer.heads.size();
                if (label_ == "t_edge" && edge_head_count >= 2)
                {
                    const auto power_count = std::size_t{2048}
                        + std::size_t{1024} * edge_head_count;
                    quotient_alpha_powers_.reserve(power_count);
                    auto power = FieldElement::one();
                    const auto alpha = challenges_.at("alpha_quot");
                    for (std::size_t i = 0; i < power_count; ++i)
                    {
                        quotient_alpha_powers_.push_back(power);
                        power *= alpha;
                    }
                }
                if (label_ == "t_edge")
                {
                    edge_evaluation_plan_ = build_t_edge_evaluation_plan(
                        context_, challenges_);
                }
            }

            algebra::Polynomial materialize(const crypto::KZGKeyPair &key)
            {
                if (target_size_ > key.pk.degree_bound + 1)
                {
                    throw std::runtime_error(
                        "SRS degree bound is too small for quotient polynomial " + label_ + ": need coefficients=" + std::to_string(target_size_) + " but degree_bound+1=" + std::to_string(key.pk.degree_bound + 1));
                }
                const auto root = FieldElement::root_of_unity(target_size_);
                auto evaluate_at_index = [&] (
                    std::size_t i,
                    const FieldElement &z,
                    const QuotientBoundaryValues *boundary_values = nullptr)
                {
                    const auto omega_z = z * domain_->omega;
                    const auto rotated_index = (i + rotation_step_) & (target_size_ - 1U);
                    const auto eval = [&](const std::string &name, const FieldElement &point) -> FieldElement
                    {
                        if (point == z)
                        {
                            return coset_value_at(name, i);
                        }
                        if (point == omega_z)
                        {
                            return coset_value_at(name, rotated_index);
                        }
                        throw std::runtime_error("quotient materializer received unsupported evaluation point for " + name);
                    };

                    if (label_ == "t_FH")
                    {
                        return evaluate_t_fh(context_, challenges_, eval, z);
                    }
                    else if (label_ == "t_edge")
                    {
                        return evaluate_t_edge(
                            context_,
                            challenges_,
                            trace_.external_evaluations,
                            eval,
                            z,
                            false,
                            nullptr,
                            nullptr,
                            quotient_alpha_powers_.empty()
                                ? nullptr
                                : &quotient_alpha_powers_,
                            edge_evaluation_plan_.get(),
                            boundary_values);
                    }
                    else if (label_ == "t_N")
                    {
                        return evaluate_t_n(context_, challenges_, eval, z);
                    }
                    else if (label_ == "t_in")
                    {
                        return evaluate_t_in(context_, challenges_, trace_.external_evaluations, eval, z);
                    }
                    else if (label_ == "t_d" || label_ == "t_d_h")
                    {
                        return evaluate_t_d(context_, challenges_, trace_.external_evaluations, eval, z);
                    }
                    else if (label_ == "t_cat")
                    {
                        return evaluate_t_cat(context_, challenges_, trace_.external_evaluations, eval, z);
                    }
                    else if (label_ == "t_C")
                    {
                        return evaluate_t_c(context_, challenges_, trace_.external_evaluations, eval, z);
                    }
                    else
                    {
                        throw std::runtime_error("unknown quotient label: " + label_);
                    }
                };

                std::vector<FieldElement> quotient_values(target_size_, FieldElement::zero());
                // Warm the dependency cache before parallel point evaluation; after this the worker
                // threads only read the cached coset vectors.
                if (label_ == "t_edge")
                {
                    const auto prewarm_start = Clock::now();
                    const auto dependency_labels = quotient_dependency_labels(context_, label_);
                    std::vector<std::string> prewarm_labels;
                    prewarm_labels.reserve(dependency_labels.size());
                    for (const auto &dependency_label : dependency_labels)
                    {
                        if (append_t_edge_materializer_dependency(&prewarm_labels, dependency_label))
                        {
                            continue;
                        }
                        prewarm_labels.push_back(dependency_label);
                    }
                    progress_note("prove:quotient:t_edge:coset_prewarm:start");
                    prewarm_coset_values(prewarm_labels);
                    seed_fast_coset_lookup(prewarm_labels);
                    progress_note(
                        "prove:quotient:t_edge:coset_prewarm:done:ms="
                        + std::to_string(elapsed_ms(prewarm_start, Clock::now())));
                    // The generic evaluator is the protocol source of truth.  The
                    // former hand-unrolled route did not reproduce evaluate_t_edge
                    // after the formal integer-scale constraints changed, producing
                    // a commitment whose polynomial disagreed with verifier claims.
                    // Keep it out of the formal path until equivalence is proved by
                    // a pointwise regression over every registered constraint term.
                }
                if (label_ == "t_FH"
                    && std::getenv("GATZK_DEBUG_QUOTIENT_SELF_CHECK") != nullptr)
                {
                    if (root.pow(static_cast<std::uint64_t>(rotation_step_))
                        != domain_->omega)
                    {
                        throw std::runtime_error(
                            "FH quotient coset rotation is incompatible with domain omega");
                    }
                    for (const auto &name : quotient_dependency_labels(
                             context_, label_))
                    {
                        const auto *polynomial = lookup_polynomial(name);
                        if (polynomial == nullptr && name == "P_H"
                            && lazy_full_feature_lookup_trace_enabled(context_))
                            polynomial = lookup_polynomial("P_T_H");
                        if (polynomial == nullptr)
                            continue;
                        const auto materialized_values = coset_values_ptr(name);
                        auto point = shift_;
                        for (std::size_t index = 0;
                             index < target_size_; ++index)
                        {
                            const auto direct = polynomial->evaluate(point);
                            if (materialized_values->at(index) != direct)
                            {
                                throw std::runtime_error(
                                    "FH quotient coset dependency mismatch: "
                                    + name + " index=" + std::to_string(index)
                                    + " materialized="
                                    + materialized_values->at(index).to_string()
                                    + " direct=" + direct.to_string());
                            }
                            point *= root;
                        }
                    }
                }
                if (label_ == "t_edge"
                    && std::getenv("GATZK_DEBUG_QUOTIENT_SELF_CHECK") != nullptr)
                {
                    EvaluationMemoization direct_memo(
                        context_, trace_, challenges_);
                    const auto dependencies = quotient_dependency_labels(
                        context_, label_);
                    direct_memo.precompute_named(
                        dependencies,
                        {shift_, shift_ * domain_->omega});
                    for (const auto &name : dependencies)
                    {
                        const auto materialized_values = coset_values_ptr(name);
                        for (const auto &[point, index] : std::array<
                                 std::pair<FieldElement, std::size_t>, 2>{
                                 std::pair{shift_, std::size_t{0}},
                                 std::pair{
                                     shift_ * domain_->omega,
                                     rotation_step_}})
                        {
                            const auto direct_dependency =
                                direct_memo.eval_named(name, point);
                            const auto materialized_dependency =
                                materialized_values->at(index);
                            if (materialized_dependency != direct_dependency)
                            {
                                throw std::runtime_error(
                                    "edge quotient coset dependency mismatch: "
                                    + name + " index=" + std::to_string(index)
                                    + " materialized="
                                    + materialized_dependency.to_string()
                                    + " direct=" + direct_dependency.to_string());
                            }
                        }
                    }
                    const auto direct = evaluate_t_edge(
                        context_,
                        challenges_,
                        trace_.external_evaluations,
                        [&](const std::string &name, const FieldElement &point)
                        {
                            return direct_memo.eval_named(name, point);
                        },
                        shift_);
                    const auto materialized = evaluate_at_index(0, shift_);
                    if (materialized != direct)
                    {
                        throw std::runtime_error(
                            "edge quotient coset evaluator mismatch at index 0: "
                            "materialized=" + materialized.to_string()
                            + " direct=" + direct.to_string());
                    }
                }
                if (label_ == "t_edge")
                {
                    progress_note("prove:quotient:t_edge:evaluate:start");
                }
                const auto point_evaluation_start = Clock::now();
                const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
                const bool run_parallel = cpu_count > 1 && target_size_ >= 4096;
                if (run_parallel)
                {
                    const auto min_points_per_task =
                        label_ == "t_edge" ? std::size_t{1024} : std::size_t{32768};
                    const auto task_count = std::min<std::size_t>(
                        std::min<std::size_t>(cpu_count, 32),
                        std::max<std::size_t>(1, target_size_ / min_points_per_task));
                    const auto chunk_size = (target_size_ + task_count - 1) / task_count;
                    std::vector<std::future<void>> futures;
                    futures.reserve(task_count);
                    for (std::size_t task = 0; task < task_count; ++task)
                    {
                        const auto begin = task * chunk_size;
                        const auto end = std::min(target_size_, begin + chunk_size);
                        if (begin >= end)
                        {
                            break;
                        }
                        futures.push_back(std::async(
                            std::launch::async,
                            [&, begin, end]()
                            {
                                auto z = shift_ * root.pow(static_cast<std::uint64_t>(begin));
                                if (label_ == "t_edge")
                                {
                                    const auto one = FieldElement::one();
                                    const auto last_point =
                                        domain_->point_at(domain_->size - 1);
                                    const auto zero_step = root.pow(domain_->size);
                                    auto zero_power = shift_.pow(domain_->size)
                                        * zero_step.pow(
                                            static_cast<std::uint64_t>(begin));
                                    constexpr std::size_t kInverseBlockSize = 8192;
                                    for (std::size_t block_begin = begin;
                                         block_begin < end;
                                         block_begin += kInverseBlockSize)
                                    {
                                        const auto block_end = std::min(
                                            end, block_begin + kInverseBlockSize);
                                        std::vector<mcl::Fr> denominator_products(
                                            block_end - block_begin);
                                        auto denominator_z = z;
                                        for (std::size_t i = block_begin;
                                             i < block_end; ++i)
                                        {
                                            mcl::Fr first_denominator;
                                            mcl::Fr last_denominator;
                                            mcl::Fr::sub(
                                                first_denominator,
                                                denominator_z.native(),
                                                one.native());
                                            mcl::Fr::sub(
                                                last_denominator,
                                                denominator_z.native(),
                                                last_point.native());
                                            mcl::Fr::mul(
                                                denominator_products[
                                                    i - block_begin],
                                                first_denominator,
                                                last_denominator);
                                            denominator_z *= root;
                                        }
                                        batch_invert_native_in_place(
                                            &denominator_products);
                                        for (std::size_t i = block_begin;
                                             i < block_end; ++i)
                                        {
                                            const auto first_denominator =
                                                z - one;
                                            const auto last_denominator =
                                                z - last_point;
                                            const auto inverse_product =
                                                FieldElement::from_native(
                                                    denominator_products[
                                                        i - block_begin]);
                                            const auto zero = zero_power - one;
                                            const auto zero_scaled =
                                                zero * domain_->inv_size;
                                            const QuotientBoundaryValues boundary{
                                                .first = zero_scaled
                                                    * last_denominator
                                                    * inverse_product,
                                                .last = zero_scaled
                                                    * last_point
                                                    * first_denominator
                                                    * inverse_product,
                                                .zero = zero,
                                            };
                                            quotient_values[i] =
                                                evaluate_at_index(
                                                    i, z, &boundary);
                                            z *= root;
                                            zero_power *= zero_step;
                                        }
                                    }
                                    return;
                                }
                                for (std::size_t i = begin; i < end; ++i)
                                {
                                    quotient_values[i] = evaluate_at_index(i, z);
                                    z *= root;
                                }
                            }));
                    }
                    for (auto &future : futures)
                    {
                        future.get();
                    }
                }
                else
                {
                    auto z = shift_;
                    for (std::size_t i = 0; i < target_size_; ++i)
                    {
                        quotient_values[i] = evaluate_at_index(i, z);
                        z *= root;
                    }
                }
                if (label_ == "t_edge")
                {
                    progress_note(
                        "prove:quotient:t_edge:evaluate:done:ms="
                        + std::to_string(elapsed_ms(
                            point_evaluation_start, Clock::now())));
                    progress_note("prove:quotient:t_edge:coefficients:start");
                }

                const auto coefficient_start = Clock::now();
                auto coefficients = coefficients_from_coset_evaluations_in_place(
                    std::move(quotient_values), shift_);
                if (std::getenv("GATZK_DEBUG_QUOTIENT_SELF_CHECK") != nullptr)
                {
                    std::cerr << "[quotient-debug] " << label_
                              << " coefficient_count="
                              << coefficients.size() << " target_size="
                              << target_size_ << '\n';
                }
                if (label_ == "t_edge")
                {
                    progress_note(
                        "prove:quotient:t_edge:coefficients:done:ms="
                        + std::to_string(elapsed_ms(
                            coefficient_start, Clock::now())));
                }
                if (coefficients.size() > key.pk.degree_bound + 1)
                {
                    throw std::runtime_error("materialized quotient exceeds SRS degree bound: " + label_);
                }
                return algebra::Polynomial::from_coefficients(label_, std::move(coefficients));
            }

        private:

            algebra::Polynomial materialize_full_identity_fh()
            {
                const auto eta = challenges_.at("eta_feat");
                const auto eta_sq = eta * eta;
                const std::vector<FieldElement> *selector_quotient = nullptr;
                const std::vector<FieldElement> *q = nullptr;
                if (const auto selector_it = context_.public_fh_coset_materials.find(
                        fh_selector_boolean_quotient_coset_key("P_Q_tbl_feat"));
                    selector_it != context_.public_fh_coset_materials.end() && selector_it->second != nullptr)
                {
                    selector_quotient = selector_it->second.get();
                    progress_note("prove:quotient:t_FH:identity:selector_quotient_coset:cache_hit");
                }
                else
                {
                    progress_note("prove:quotient:t_FH:identity:coset:P_Q_tbl_feat:start");
                    q = &coset_values("P_Q_tbl_feat");
                    progress_note("prove:quotient:t_FH:identity:coset:P_Q_tbl_feat:done");
                }
                progress_note("prove:quotient:t_FH:identity:coset:P_Table_feat_fused:start");
                const auto table_values = fused_feature_table_coset_values(eta, eta_sq);
                progress_note("prove:quotient:t_FH:identity:cosets:done");
                if ((selector_quotient != nullptr && selector_quotient->size() != target_size_)
                    || (q != nullptr && q->size() != target_size_)
                    || table_values->size() != target_size_)
                {
                    throw std::runtime_error("full-identity FH quotient coset value size mismatch");
                }

                const auto beta = challenges_.at("beta_feat");
                const auto alpha = challenges_.at("alpha_quot");
                const auto one = FieldElement::one();
                const auto &beta_native = beta.native();
                const auto &alpha_native = alpha.native();
                const auto &one_native = one.native();
                std::vector<mcl::Fr> inverse_zero;
                std::size_t zero_cycle = 0;
                if (selector_quotient == nullptr)
                {
                    const auto root = FieldElement::root_of_unity(target_size_);
                    zero_cycle = target_size_ / domain_->size;
                    if (zero_cycle == 0 || target_size_ % domain_->size != 0)
                    {
                        throw std::runtime_error("invalid full-identity FH quotient coset cycle");
                    }
                    inverse_zero.resize(zero_cycle);
                    auto zero_value = shift_.pow(domain_->size);
                    const auto zero_step = root.pow(domain_->size);
                    for (std::size_t i = 0; i < zero_cycle; ++i)
                    {
                        inverse_zero[i] = (zero_value - one).inv().native();
                        zero_value *= zero_step;
                    }
                }

                std::vector<FieldElement> quotient_values(target_size_, FieldElement::zero());
                auto *quotient_native = FieldElement::mutable_native_array_data(quotient_values.data());
                const auto *q_native = q != nullptr ? FieldElement::native_array_data(q->data()) : nullptr;
                const auto *selector_quotient_native =
                    selector_quotient != nullptr ? FieldElement::native_array_data(selector_quotient->data()) : nullptr;
                const auto *table_native = FieldElement::native_array_data(table_values->data());
                progress_note("prove:quotient:t_FH:identity:assemble:start");
                const auto fill_range = [&](std::size_t begin, std::size_t end)
                {
                    mcl::Fr one_minus_q;
                    mcl::Fr q_boolean;
                    mcl::Fr table_plus_beta;
                    mcl::Fr value;
                    for (std::size_t i = begin; i < end; ++i)
                    {
                        mcl::Fr::add(table_plus_beta, table_native[i], beta_native);
                        if (selector_quotient_native != nullptr)
                        {
                            mcl::Fr::mul(value, selector_quotient_native[i], table_plus_beta);
                        }
                        else
                        {
                            mcl::Fr::sub(one_minus_q, one_native, q_native[i]);
                            mcl::Fr::mul(q_boolean, q_native[i], one_minus_q);
                            mcl::Fr::mul(value, q_boolean, table_plus_beta);
                            mcl::Fr::mul(value, value, inverse_zero[i % zero_cycle]);
                        }
                        mcl::Fr::mul(quotient_native[i], alpha_native, value);
                    }
                };
                parallel_for_chunks(target_size_, 1ULL << 15, 0, fill_range);
                progress_note("prove:quotient:t_FH:identity:assemble:done");
                progress_note("prove:quotient:t_FH:identity:coefficients:start");
                auto coefficients = coefficients_from_coset_evaluations_in_place(std::move(quotient_values), shift_);
                progress_note("prove:quotient:t_FH:identity:coefficients:done");
                return algebra::Polynomial::from_coefficients(label_, std::move(coefficients));
            }

            template <class Fn>
            void parallel_for_chunks(
                std::size_t count,
                std::size_t min_parallel_count,
                std::size_t max_worker_count,
                Fn &&fn) const
            {
                const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
                const auto available_workers =
                    max_worker_count == 0 ? cpu_count : std::min<std::size_t>(cpu_count, max_worker_count);
                if (count < min_parallel_count || available_workers <= 1)
                {
                    fn(0, count);
                    return;
                }
                const auto task_count = std::min<std::size_t>(
                    available_workers,
                    std::max<std::size_t>(2, count / min_parallel_count));
                const auto chunk_size = (count + task_count - 1) / task_count;
                std::vector<std::future<void>> futures;
                futures.reserve(task_count);
                for (std::size_t task = 0; task < task_count; ++task)
                {
                    const auto begin = task * chunk_size;
                    const auto end = std::min(count, begin + chunk_size);
                    if (begin >= end)
                    {
                        break;
                    }
                    futures.push_back(std::async(std::launch::async, [&, begin, end]()
                                                 { fn(begin, end); }));
                }
                for (auto &future : futures)
                {
                    future.get();
                }
            }

            std::shared_ptr<const std::vector<FieldElement>> fused_feature_table_coset_values(
                const FieldElement &eta,
                const FieldElement &eta_sq)
            {
                if (label_ != "t_FH" || !full_graph_feature_identity_enabled(context_))
                {
                    throw std::runtime_error("fused FH feature table coset requested outside full identity");
                }
                if (const auto it = trace_.commitments.find("P_Table_feat");
                    it != trace_.commitments.end() && it->second.opening_polynomial != nullptr)
                {
                    progress_note("prove:quotient:t_FH:identity:coset:P_Table_feat:direct_eval:start");
                    auto values = std::make_shared<const std::vector<FieldElement>>(
                        coset_evaluate_polynomial(
                            *it->second.opening_polynomial,
                            target_size_,
                            shift_,
                            "P_Table_feat",
                            &shift_powers()));
                    progress_note("prove:quotient:t_FH:identity:coset:P_Table_feat:direct_eval:done");
                    return values;
                }
                const auto *row_poly = lookup_polynomial("P_Row_feat_tbl");
                const auto *col_poly = lookup_polynomial("P_Col_feat_tbl");
                const auto *feature_poly = lookup_polynomial("P_T_H");
                if (row_poly == nullptr || col_poly == nullptr || feature_poly == nullptr)
                {
                    throw std::runtime_error("missing FH public columns for fused table coset");
                }
                if (row_poly->basis != algebra::PolynomialBasis::Evaluation || col_poly->basis != algebra::PolynomialBasis::Evaluation || feature_poly->basis != algebra::PolynomialBasis::Evaluation || row_poly->domain.get() != domain_.get() || col_poly->domain.get() != domain_.get() || feature_poly->domain.get() != domain_.get())
                {
                    throw std::runtime_error("fused FH table coset requires FH evaluation-basis public columns");
                }

                const auto &rows = row_poly->values();
                const auto &cols = col_poly->values();
                const auto &features = feature_poly->values();
                if (rows.size() > domain_->size || cols.size() > domain_->size || features.size() > domain_->size)
                {
                    throw std::runtime_error("fused FH table dependency exceeds FH domain");
                }

                std::vector<FieldElement> table(domain_->size, FieldElement::zero());
                parallel_for_chunks(domain_->size, 1ULL << 16, 4, [&](std::size_t begin, std::size_t end)
                                    {
            for (std::size_t i = begin; i < end; ++i) {
                const auto row_value = i < rows.size() ? rows[i] : FieldElement::zero();
                const auto col_value = i < cols.size() ? cols[i] : FieldElement::zero();
                const auto feature_value = i < features.size() ? features[i] : FieldElement::zero();
                table[i] = row_value + eta * col_value + eta_sq * feature_value;
            } });

                auto table_poly = algebra::Polynomial::from_evaluations(
                    "P_Table_feat_fused",
                    std::move(table),
                    domain_);
                auto values = std::make_shared<const std::vector<FieldElement>>(
                    coset_evaluate_polynomial(
                        table_poly,
                        target_size_,
                        shift_,
                        "P_Table_feat_fused",
                        &shift_powers()));
                return values;
            }

            const std::vector<FieldElement> &coset_values(const std::string &requested_name)
            {
                return *coset_values_ptr(requested_name);
            }

            void seed_fast_coset_lookup(const std::vector<std::string> &labels)
            {
                if (label_ != "t_edge")
                {
                    return;
                }
                fast_coset_lookup_.clear();
                fast_coset_lookup_.reserve(labels.size() * 2);
                for (const auto &label : labels)
                {
                    if (fast_coset_lookup_.find(label) != fast_coset_lookup_.end())
                    {
                        continue;
                    }
                    const auto values = coset_values_ptr(label);
                    fast_coset_lookup_.emplace(label, values.get());
                }
            }

            const std::vector<FieldElement> &fast_coset_values(const std::string &requested_name)
            {
                if (!fast_coset_lookup_.empty())
                {
                    if (const auto it = fast_coset_lookup_.find(requested_name); it != fast_coset_lookup_.end())
                    {
                        return *it->second;
                    }
                }
                return coset_values(requested_name);
            }

            bool is_direct_t_edge_coset_request(const std::string &name) const
            {
                if (!context_.model.has_real_multihead || label_ != "t_edge")
                {
                    return false;
                }
                for (std::size_t head_index = 0; head_index < context_.model.hidden_heads.size(); ++head_index)
                {
                    const auto prefix = "P_h" + std::to_string(head_index) + "_";
                    if (name.rfind(prefix, 0) == 0)
                    {
                        const auto suffix = std::string_view(name).substr(prefix.size());
                        return suffix == "w_psq" || suffix == "T_psq_edge";
                    }
                }
                for (std::size_t head_index = 0; head_index < context_.model.output_layer.heads.size(); ++head_index)
                {
                    const auto prefix = output_head_prefix(head_index) + "_";
                    if (name.rfind(prefix, 0) == 0)
                    {
                        const auto suffix = std::string_view(name).substr(prefix.size());
                        return suffix == "w";
                    }
                }
                return false;
            }

            bool append_t_edge_materializer_dependency(
                std::vector<std::string> *labels,
                const std::string &name) const
            {
                if (!context_.model.has_real_multihead || label_ != "t_edge")
                {
                    return false;
                }
                // A committed column carries its own registered Z_edge*R
                // blinder.  Its semantic expression is equal only on the
                // edge domain and cannot replace that polynomial on a coset.
                if (lookup_polynomial(name) != nullptr
                    || trace_.spilled_polynomials.contains(name))
                {
                    return false;
                }
                for (std::size_t head_index = 0; head_index < context_.model.hidden_heads.size(); ++head_index)
                {
                    const auto prefix = "P_h" + std::to_string(head_index) + "_";
                    if (name.rfind(prefix, 0) != 0)
                    {
                        continue;
                    }
                    const auto suffix = std::string_view(name).substr(prefix.size());
                    if (suffix == "w_psq")
                    {
                        labels->push_back(prefix + "U");
                        labels->push_back(prefix + "widehat_v_pre_star");
                        return true;
                    }
                    if (suffix == "T_psq_edge")
                    {
                        labels->push_back(prefix + "Sum_edge");
                        labels->push_back(prefix + "H_agg_pre_star_edge");
                        return true;
                    }
                    if (suffix == "Query_ELU")
                    {
                        labels->push_back(name);
                        return true;
                    }
                    if (suffix == "H_agg_pre_flat" || suffix == "H_agg_flat")
                    {
                        return true;
                    }
                    return is_edge_derived_coset_request(name);
                }

                for (std::size_t head_index = 0; head_index < context_.model.output_layer.heads.size(); ++head_index)
                {
                    const auto prefix = output_head_prefix(head_index) + "_";
                    if (name.rfind(prefix, 0) != 0)
                    {
                        continue;
                    }
                    const auto suffix = std::string_view(name).substr(prefix.size());
                    if (suffix == "w")
                    {
                        labels->push_back(prefix + "U");
                        labels->push_back(prefix + "widehat_y_star");
                        return true;
                    }
                    if (suffix == "T_edge")
                    {
                        labels->push_back(prefix + "Sum_edge");
                        labels->push_back(prefix + "Y_star_edge");
                        return true;
                    }
                    return is_edge_derived_coset_request(name);
                }

                return false;
            }

            FieldElement coset_value_at(const std::string &requested_name, std::size_t index)
            {
                if (label_ == "t_edge")
                {
                    if (const auto it = fast_coset_lookup_.find(requested_name);
                        it != fast_coset_lookup_.end())
                    {
                        return (*it->second)[index];
                    }
                    if (const auto derived = t_edge_derived_value_at(requested_name, index); derived.has_value())
                    {
                        return *derived;
                    }
                }
                return fast_coset_values(requested_name)[index];
            }

            std::optional<FieldElement> t_edge_derived_value_at(const std::string &name, std::size_t index)
            {
                if (!context_.model.has_real_multihead || label_ != "t_edge")
                {
                    return std::nullopt;
                }
                if (lookup_polynomial(name) != nullptr
                    || trace_.spilled_polynomials.contains(name))
                {
                    return std::nullopt;
                }
                for (std::size_t head_index = 0; head_index < context_.model.hidden_heads.size(); ++head_index)
                {
                    const auto prefix = "P_h" + std::to_string(head_index) + "_";
                    if (name.rfind(prefix, 0) != 0)
                    {
                        continue;
                    }
                    const auto suffix = std::string_view(name).substr(prefix.size());
                    const auto value = [&](const std::string &label)
                    {
                        return coset_value_at(label, index);
                    };
                    const auto head_value = [&](std::string_view inner_suffix)
                    {
                        return value(prefix + std::string(inner_suffix));
                    };
                    if (suffix == "S")
                    {
                        return head_value("E_src_edge") + head_value("E_dst_edge");
                    }
                    if (suffix == "Delta")
                    {
                        return head_value("M_edge") - head_value("Z");
                    }
                    const auto eta_l = challenges_.at("eta_L_h" + std::to_string(head_index));
                    if (suffix == "Table_L")
                    {
                        return value("P_T_L_x") + eta_l * value("P_T_L_y");
                    }
                    if (suffix == "Query_L")
                    {
                        return head_value("S") + eta_l * head_value("Z");
                    }
                    if (suffix == "Table_R")
                    {
                        return value("P_T_range");
                    }
                    if (suffix == "Query_R")
                    {
                        return head_value("Delta");
                    }
                    const auto eta_exp = challenges_.at("eta_exp_h" + std::to_string(head_index));
                    if (suffix == "Table_exp")
                    {
                        return value("P_T_exp_x") + eta_exp * value("P_T_exp_y");
                    }
                    if (suffix == "Query_exp")
                    {
                        return head_value("Delta") + eta_exp * head_value("U");
                    }
                    const auto eta_elu = challenges_.at("eta_ELU_h" + std::to_string(head_index));
                    if (suffix == "Table_ELU")
                    {
                        return value("P_T_ELU_x") + eta_elu * value("P_T_ELU_y");
                    }
                    if (suffix == "Query_ELU")
                    {
                        return head_value("H_agg_pre_flat") + eta_elu * head_value("H_agg_flat");
                    }
                    const auto eta_src = challenges_.at("eta_src_h" + std::to_string(head_index));
                    if (suffix == "Query_src")
                    {
                        const auto eta_src_sq = eta_src * eta_src;
                        return value("P_src") + eta_src * head_value("E_src_edge") + eta_src_sq * head_value("H_src_star_edge");
                    }
                    if (suffix == "w_psq")
                    {
                        if (lookup_polynomial(name) != nullptr || trace_.spilled_polynomials.contains(name))
                        {
                            return std::nullopt;
                        }
                        const auto lambda_psq = challenges_.at("lambda_psq_h" + std::to_string(head_index));
                        return head_value("U") + lambda_psq * head_value("widehat_v_pre_star");
                    }
                    if (suffix == "T_psq_edge")
                    {
                        if (lookup_polynomial(name) != nullptr || trace_.spilled_polynomials.contains(name))
                        {
                            return std::nullopt;
                        }
                        const auto lambda_psq = challenges_.at("lambda_psq_h" + std::to_string(head_index));
                        return head_value("Sum_edge") + lambda_psq * head_value("H_agg_pre_star_edge");
                    }
                    const auto eta_t = challenges_.at("eta_t_h" + std::to_string(head_index));
                    if (suffix == "Query_t")
                    {
                        return value("P_dst") + eta_t * head_value("T_psq_edge");
                    }
                    const auto eta_dst = challenges_.at("eta_dst_h" + std::to_string(head_index));
                    if (suffix == "Query_dst")
                    {
                        const auto eta_dst_sq = eta_dst * eta_dst;
                        const auto eta_dst_cu = eta_dst_sq * eta_dst;
                        const auto eta_dst_4 = eta_dst_cu * eta_dst;
                        const auto eta_dst_5 = eta_dst_4 * eta_dst;
                        return value("P_dst")
                               + eta_dst * head_value("E_dst_edge")
                               + eta_dst_sq * head_value("M_edge")
                               + eta_dst_cu * head_value("Sum_edge")
                               + eta_dst_4 * head_value("inv_edge")
                               + eta_dst_5 * head_value("H_agg_star_edge");
                    }
                    return std::nullopt;
                }

                for (std::size_t head_index = 0; head_index < context_.model.output_layer.heads.size(); ++head_index)
                {
                    const auto prefix = output_head_prefix(head_index) + "_";
                    if (name.rfind(prefix, 0) != 0)
                    {
                        continue;
                    }
                    const auto suffix = std::string_view(name).substr(prefix.size());
                    const auto value = [&](const std::string &label)
                    {
                        return coset_value_at(label, index);
                    };
                    const auto head_value = [&](std::string_view inner_suffix)
                    {
                        return value(prefix + std::string(inner_suffix));
                    };
                    if (suffix == "S")
                    {
                        return head_value("E_src_edge") + head_value("E_dst_edge");
                    }
                    if (suffix == "Delta")
                    {
                        return head_value("M_edge") - head_value("Z");
                    }
                    const auto eta_l = challenges_.at(output_challenge_name("eta_L_out", head_index));
                    if (suffix == "Table_L")
                    {
                        return value("P_T_L_x") + eta_l * value("P_T_L_y");
                    }
                    if (suffix == "Query_L")
                    {
                        return head_value("S") + eta_l * head_value("Z");
                    }
                    if (suffix == "Table_R")
                    {
                        return value("P_T_range");
                    }
                    if (suffix == "Query_R")
                    {
                        return head_value("Delta");
                    }
                    const auto eta_exp = challenges_.at(output_challenge_name("eta_exp_out", head_index));
                    if (suffix == "Table_exp")
                    {
                        return value("P_T_exp_x") + eta_exp * value("P_T_exp_y");
                    }
                    if (suffix == "Query_exp")
                    {
                        return head_value("Delta") + eta_exp * head_value("U");
                    }
                    const auto eta_src = challenges_.at(output_challenge_name("eta_src_out", head_index));
                    if (suffix == "Query_src")
                    {
                        return value("P_src") + eta_src * head_value("E_src_edge");
                    }
                    if (suffix == "w")
                    {
                        if (lookup_polynomial(name) != nullptr || trace_.spilled_polynomials.contains(name))
                        {
                            return std::nullopt;
                        }
                        const auto lambda_out = challenges_.at(output_challenge_name("lambda_out", head_index));
                        return head_value("U") + lambda_out * head_value("widehat_y_star");
                    }
                    if (suffix == "T_edge")
                    {
                        const auto lambda_out = challenges_.at(output_challenge_name("lambda_out", head_index));
                        return head_value("Sum_edge") + lambda_out * head_value("Y_star_edge");
                    }
                    const auto eta_t = challenges_.at(output_challenge_name("eta_t_out", head_index));
                    if (suffix == "Query_t")
                    {
                        return value("P_dst") + eta_t * head_value("T_edge");
                    }
                    const auto eta_dst = challenges_.at(output_challenge_name("eta_dst_out", head_index));
                    if (suffix == "Query_dst")
                    {
                        const auto eta_dst_sq = eta_dst * eta_dst;
                        const auto eta_dst_cu = eta_dst_sq * eta_dst;
                        const auto eta_dst_4 = eta_dst_cu * eta_dst;
                        const auto eta_dst_5 = eta_dst_4 * eta_dst;
                        return value("P_dst")
                               + eta_dst * head_value("E_dst_edge")
                               + eta_dst_sq * head_value("M_edge")
                               + eta_dst_cu * head_value("Sum_edge")
                               + eta_dst_4 * head_value("inv_edge")
                               + eta_dst_5 * head_value("Y_star_edge");
                    }
                    return std::nullopt;
                }
                return std::nullopt;
            }

            std::vector<FieldElement> quotient_alpha_powers_;

            struct SemanticCosetGroup
            {
                const algebra::Polynomial* owner = nullptr;
                std::string owner_label;
                std::vector<std::string> labels;
                std::mutex mutex;
                std::shared_ptr<const std::vector<FieldElement>> base_values;
                std::size_t remaining = 0;
            };

            std::uint64_t semantic_values_hash(
                const std::vector<FieldElement>& values) const
            {
                std::uint64_t hash = 1469598103934665603ULL;
                for (const auto& value : values)
                {
                    const auto* units = value.native().getUnit();
                    for (std::size_t unit = 0;
                         unit < mcl::Fr::getUnitSize(); ++unit)
                    {
                        hash ^= static_cast<std::uint64_t>(units[unit]);
                        hash *= 1099511628211ULL;
                    }
                }
                hash ^= values.size();
                hash *= 1099511628211ULL;
                return hash;
            }

            void seed_semantic_coset_groups(
                const std::vector<std::string>& wave)
            {
                semantic_coset_group_by_label_.clear();
                semantic_coset_groups_.clear();
                std::unordered_map<
                    std::uint64_t,
                    std::vector<std::shared_ptr<SemanticCosetGroup>>>
                    groups_by_hash;
                std::size_t semantic_objects = 0;
                for (const auto& item_label : wave)
                {
                    const auto* polynomial = lookup_polynomial(item_label);
                    if (polynomial == nullptr
                        || polynomial->basis
                            != algebra::PolynomialBasis::Evaluation
                        || polynomial->domain.get() != domain_.get())
                    {
                        continue;
                    }
                    ++semantic_objects;
                    const auto& values = polynomial->values();
                    auto& candidates = groups_by_hash[
                        semantic_values_hash(values)];
                    auto group = std::find_if(
                        candidates.begin(),
                        candidates.end(),
                        [&](const auto& candidate)
                        {
                            return candidate->owner->values() == values;
                        });
                    if (group == candidates.end())
                    {
                        auto created = std::make_shared<SemanticCosetGroup>();
                        created->owner = polynomial;
                        created->owner_label = item_label;
                        created->labels.push_back(item_label);
                        candidates.push_back(created);
                    }
                    else
                    {
                        (*group)->labels.push_back(item_label);
                    }
                }

                std::size_t semantic_unique = 0;
                for (auto& [_, candidates] : groups_by_hash)
                {
                    semantic_unique += candidates.size();
                    for (auto& group : candidates)
                    {
                        if (group->labels.size() <= 1)
                        {
                            continue;
                        }
                        group->remaining = group->labels.size();
                        semantic_coset_groups_.push_back(group);
                        for (const auto& item_label : group->labels)
                        {
                            semantic_coset_group_by_label_.emplace(
                                item_label, group);
                        }
                    }
                }

                if (!semantic_coset_groups_.empty())
                {
                    semantic_coset_points_.resize(target_size_);
                    semantic_coset_zero_values_.resize(target_size_);
                    const auto root = FieldElement::root_of_unity(target_size_);
                    const auto zero_step = root.pow(domain_->size);
                    auto point = shift_;
                    auto zero_power = shift_.pow(domain_->size);
                    const auto one = FieldElement::one();
                    for (std::size_t index = 0; index < target_size_; ++index)
                    {
                        semantic_coset_points_[index] = point;
                        semantic_coset_zero_values_[index] = zero_power - one;
                        point *= root;
                        zero_power *= zero_step;
                    }
                }

                if (progress_enabled())
                {
                    progress_note(
                        "prove:quotient:t_edge:coset_prewarm:semantic_objects="
                        + std::to_string(semantic_objects)
                        + ":semantic_unique="
                        + std::to_string(semantic_unique)
                        + ":semantic_duplicates="
                        + std::to_string(semantic_objects - semantic_unique)
                        + ":semantic_reuse_groups="
                        + std::to_string(semantic_coset_groups_.size()));
                }
            }

            std::shared_ptr<const std::vector<FieldElement>>
            reused_semantic_coset_values(
                const std::string& requested_name,
                const algebra::Polynomial& polynomial)
            {
                const auto group_it = semantic_coset_group_by_label_.find(
                    requested_name);
                if (group_it == semantic_coset_group_by_label_.end())
                {
                    return nullptr;
                }
                const auto& group = group_it->second;
                std::shared_ptr<const std::vector<FieldElement>> base_values;
                {
                    std::lock_guard<std::mutex> lock(group->mutex);
                    if (group->base_values == nullptr)
                    {
                        auto semantic_polynomial = *group->owner;
                        semantic_polynomial.zero_domain_blinder.clear();
                        group->base_values =
                            std::make_shared<const std::vector<FieldElement>>(
                                coset_evaluate_polynomial(
                                    semantic_polynomial,
                                    target_size_,
                                    shift_,
                                    group->owner_label,
                                    &shift_powers()));
                    }
                    base_values = group->base_values;
                }

                auto values = std::make_shared<std::vector<FieldElement>>(
                    *base_values);
                if (!polynomial.zero_domain_blinder.empty())
                {
                    auto* out = FieldElement::mutable_native_array_data(
                        values->data());
                    for (std::size_t index = 0; index < target_size_; ++index)
                    {
                        mcl::Fr blinder_value =
                            polynomial.zero_domain_blinder.back().native();
                        for (std::size_t coefficient =
                                 polynomial.zero_domain_blinder.size() - 1;
                             coefficient-- > 0;)
                        {
                            mcl::Fr::mul(
                                blinder_value,
                                blinder_value,
                                semantic_coset_points_[index].native());
                            mcl::Fr::add(
                                blinder_value,
                                blinder_value,
                                polynomial.zero_domain_blinder[coefficient]
                                    .native());
                        }
                        mcl::Fr contribution;
                        mcl::Fr::mul(
                            contribution,
                            blinder_value,
                            semantic_coset_zero_values_[index].native());
                        mcl::Fr::add(out[index], out[index], contribution);
                    }
                }

                {
                    std::lock_guard<std::mutex> lock(group->mutex);
                    if (group->remaining != 0)
                    {
                        --group->remaining;
                        if (group->remaining == 0)
                        {
                            group->base_values.reset();
                        }
                    }
                }
                return values;
            }

            void prewarm_coset_values(const std::vector<std::string> &labels)
            {
                std::vector<std::string> unique_labels;
                unique_labels.reserve(labels.size());
                for (const auto &label : labels)
                {
                    if (std::find(unique_labels.begin(), unique_labels.end(), label) != unique_labels.end())
                    {
                        continue;
                    }
                    {
                        std::lock_guard<std::mutex> lock(coset_cache_mutex_);
                        if (coset_cache_.find(label) != coset_cache_.end())
                        {
                            continue;
                        }
                    }
                    unique_labels.push_back(label);
                }
                if (unique_labels.empty())
                {
                    return;
                }
                seed_shared_source_coset_aliases(&unique_labels);

                auto derived_begin = std::stable_partition(
                    unique_labels.begin(),
                    unique_labels.end(),
                    [&](const std::string &label)
                    {
                        return !is_edge_derived_coset_request(label);
                    });
                progress_note(
                    "prove:quotient:" + label_ + ":coset_prewarm:base_count=" + std::to_string(static_cast<std::size_t>(std::distance(unique_labels.begin(), derived_begin))) + ":derived_count=" + std::to_string(static_cast<std::size_t>(std::distance(derived_begin, unique_labels.end()))));

                auto prewarm_range = [&](std::string_view wave_name, auto begin_it, auto end_it)
                {
                    const auto wave_size = static_cast<std::size_t>(std::distance(begin_it, end_it));
                    if (wave_size == 0)
                    {
                        return;
                    }
                    std::vector<std::string> wave(begin_it, end_it);
                    if (wave_name == std::string_view("base"))
                    {
                        seed_semantic_coset_groups(wave);
                        seed_canonical_coset_aliases(wave);
                    }
                    progress_note(
                        "prove:quotient:" + label_ + ":coset_prewarm:" + std::string(wave_name) + ":start:size=" + std::to_string(wave.size()));
                    const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
                    std::size_t worker_cap = 4;
                    if (label_ == "t_edge" && target_size_ <= (1ULL << 22) && cpu_count >= 8)
                    {
                        worker_cap = std::min<std::size_t>(
                            24,
                            std::max<std::size_t>(8, cpu_count));
                    }
                    const auto worker_count = std::min<std::size_t>(
                        std::min<std::size_t>(cpu_count, worker_cap),
                        wave.size());
                    const bool log_items =
                        !(label_ == "t_edge"
                          && wave_name == std::string_view("base")
                          && wave.size() > 64
                          && std::getenv("GATZK_VERBOSE_T_EDGE_PREWARM") == nullptr);
                    progress_note(
                        "prove:quotient:" + label_ + ":coset_prewarm:" + std::string(wave_name) + ":workers=" + std::to_string(worker_count));
                    std::atomic<std::size_t> next{0};
                    std::vector<std::future<void>> futures;
                    futures.reserve(worker_count);
                    for (std::size_t worker = 0; worker < worker_count; ++worker)
                    {
                        futures.push_back(std::async(std::launch::async, [&, wave_name]()
                                                     {
                    while (true) {
                        const auto index = next.fetch_add(1);
                        if (index >= wave.size()) {
                            break;
                        }
                        if (log_items) {
                            progress_note(
                                "prove:quotient:" + label_ + ":coset_prewarm:"
                                + std::string(wave_name) + ":" + wave[index] + ":start");
                        }
                        coset_values_ptr(wave[index]);
                        if (log_items) {
                            progress_note(
                                "prove:quotient:" + label_ + ":coset_prewarm:"
                                + std::string(wave_name) + ":" + wave[index] + ":done");
                        }
                    } }));
                    }
                    for (auto &future : futures)
                    {
                        future.get();
                    }
                    progress_note(
                        "prove:quotient:" + label_ + ":coset_prewarm:" + std::string(wave_name) + ":done");
                };

                prewarm_range("base", unique_labels.begin(), derived_begin);
                prewarm_range("derived", derived_begin, unique_labels.end());
            }

            void seed_canonical_coset_aliases(const std::vector<std::string> &wave)
            {
                if (label_ != "t_edge")
                {
                    return;
                }
                const auto has_label = [&](const std::string &name)
                {
                    return std::find(wave.begin(), wave.end(), name) != wave.end();
                };
                if (!has_label("P_Q_edge_valid"))
                {
                    return;
                }

                std::vector<std::string> aliases;
                for (const auto &name : {"P_Q_qry_src", "P_Q_qry_dst", "P_Q_qry_L", "P_Q_qry_R", "P_Q_qry_exp"})
                {
                    if (has_label(name))
                    {
                        aliases.emplace_back(name);
                    }
                }
                if (aliases.empty())
                {
                    return;
                }

                const auto values = coset_values_ptr("P_Q_edge_valid");
                std::lock_guard<std::mutex> lock(coset_cache_mutex_);
                for (const auto &alias : aliases)
                {
                    coset_cache_.emplace(alias, values);
                }
            }

            std::string coset_source_key(const std::string &label) const
            {
                const auto *polynomial = lookup_polynomial(label);
                if (polynomial == nullptr
                    || polynomial->basis != algebra::PolynomialBasis::Evaluation
                    || polynomial->domain == nullptr
                    || polynomial->shared_data == nullptr)
                {
                    return {};
                }
                return polynomial->domain->name
                       + ":" + std::to_string(polynomial->domain->size)
                       + ":" + std::to_string(reinterpret_cast<std::uintptr_t>(polynomial->shared_data.get()));
            }

            void seed_shared_source_coset_aliases(std::vector<std::string> *labels)
            {
                if (labels == nullptr || labels->size() <= 1)
                {
                    return;
                }
                std::unordered_map<std::string, std::string> owner_by_source;
                owner_by_source.reserve(labels->size());
                std::vector<std::pair<std::string, std::string>> aliases;
                aliases.reserve(labels->size());
                std::vector<std::string> filtered;
                filtered.reserve(labels->size());
                for (const auto &label : *labels)
                {
                    const auto source_key = coset_source_key(label);
                    if (source_key.empty())
                    {
                        filtered.push_back(label);
                        continue;
                    }
                    const auto [it, inserted] = owner_by_source.emplace(source_key, label);
                    if (inserted)
                    {
                        filtered.push_back(label);
                    }
                    else
                    {
                        aliases.push_back({label, it->second});
                    }
                }
                if (aliases.empty())
                {
                    return;
                }
                {
                    std::lock_guard<std::mutex> lock(coset_cache_mutex_);
                    for (const auto &[alias, owner] : aliases)
                    {
                        coset_aliases_[alias] = owner;
                    }
                }
                labels->swap(filtered);
                if (label_ == "t_edge")
                {
                    progress_note(
                        "prove:quotient:t_edge:coset_prewarm:shared_source_aliases="
                        + std::to_string(aliases.size()));
                }
            }

            bool is_edge_derived_coset_request(const std::string &name) const
            {
                if (!context_.model.has_real_multihead || label_ != "t_edge")
                {
                    return false;
                }
                auto is_hidden_derived_suffix = [](std::string_view suffix)
                {
                    return suffix == "S" || suffix == "Delta" || suffix == "Table_L" || suffix == "Query_L" || suffix == "Table_R" || suffix == "Query_R" || suffix == "Table_exp" || suffix == "Query_exp" || suffix == "Table_ELU" || suffix == "Query_ELU" || suffix == "Query_src" || suffix == "Query_t" || suffix == "Query_dst" || suffix == "w_psq" || suffix == "T_psq_edge";
                };
                for (std::size_t head_index = 0; head_index < context_.model.hidden_heads.size(); ++head_index)
                {
                    const auto prefix = "P_h" + std::to_string(head_index) + "_";
                    if (name.rfind(prefix, 0) == 0)
                    {
                        return is_hidden_derived_suffix(name.substr(prefix.size()));
                    }
                }
                auto is_output_derived_suffix = [](std::string_view suffix)
                {
                    return suffix == "S" || suffix == "Delta" || suffix == "Table_L" || suffix == "Query_L" || suffix == "Table_R" || suffix == "Query_R" || suffix == "Table_exp" || suffix == "Query_exp" || suffix == "Query_src" || suffix == "Query_t" || suffix == "Query_dst" || suffix == "w" || suffix == "T_edge";
                };
                for (std::size_t head_index = 0; head_index < context_.model.output_layer.heads.size(); ++head_index)
                {
                    const auto prefix = output_head_prefix(head_index) + "_";
                    if (name.rfind(prefix, 0) == 0)
                    {
                        return is_output_derived_suffix(name.substr(prefix.size()));
                    }
                }
                return false;
            }

            std::shared_ptr<const std::vector<FieldElement>> scaled_coset_sum(
                const std::vector<std::pair<std::string, FieldElement>> &terms)
            {
                struct ResolvedTerm
                {
                    std::shared_ptr<const std::vector<FieldElement>> values;
                    mcl::Fr scale;
                    bool scale_is_one = false;
                };

                std::vector<ResolvedTerm> resolved_terms;
                resolved_terms.reserve(terms.size());
                for (const auto &[label, scale] : terms)
                {
                    if (scale.is_zero())
                    {
                        continue;
                    }
                    const auto values = coset_values_ptr(label);
                    if (values->size() != target_size_)
                    {
                        throw std::runtime_error("linear coset dependency size mismatch: " + label);
                    }
                    resolved_terms.push_back({
                        .values = values,
                        .scale = scale.native(),
                        .scale_is_one = scale == FieldElement::one(),
                    });
                }
                if (resolved_terms.empty())
                {
                    return std::make_shared<const std::vector<FieldElement>>(target_size_, FieldElement::zero());
                }
                if (resolved_terms.size() == 1 && resolved_terms.front().scale_is_one)
                {
                    return resolved_terms.front().values;
                }

                auto out = std::make_shared<std::vector<FieldElement>>(target_size_, FieldElement::zero());
                parallel_for_chunks(target_size_, 1ULL << 15, 2, [&](std::size_t begin, std::size_t end)
                                    {
                mcl::Fr product;
                for (std::size_t i = begin; i < end; ++i) {
                    mcl::Fr acc;
                    acc.clear();
                    for (const auto &term : resolved_terms) {
                        if (term.scale_is_one) {
                            mcl::Fr::add(acc, acc, (*term.values)[i].native());
                        } else {
                            mcl::Fr::mul(product, (*term.values)[i].native(), term.scale);
                            mcl::Fr::add(acc, acc, product);
                        }
                    }
                    (*out)[i] = FieldElement::from_native(acc);
                } });
                return out;
            }

            std::shared_ptr<const std::vector<FieldElement>> scaled_domain_sum_coset(
                const std::string &derived_label,
                const std::vector<std::pair<std::string, FieldElement>> &terms)
            {
                struct ResolvedTerm
                {
                    const std::vector<FieldElement> *values = nullptr;
                    mcl::Fr scale;
                    bool scale_is_one = false;
                };

                std::vector<ResolvedTerm> resolved_terms;
                resolved_terms.reserve(terms.size());
                for (const auto &[label, scale] : terms)
                {
                    if (scale.is_zero())
                    {
                        continue;
                    }
                    const auto *polynomial = lookup_polynomial(label);
                    if (polynomial == nullptr
                        || polynomial->basis != algebra::PolynomialBasis::Evaluation
                        || polynomial->domain.get() != domain_.get())
                    {
                        return scaled_coset_sum(terms);
                    }
                    const auto &values = polynomial->values();
                    if (values.size() > domain_->size)
                    {
                        throw std::runtime_error("derived t_edge domain dependency exceeds edge domain: " + label);
                    }
                    resolved_terms.push_back({
                        .values = &values,
                        .scale = scale.native(),
                        .scale_is_one = scale == FieldElement::one(),
                    });
                }
                if (resolved_terms.empty())
                {
                    return std::make_shared<const std::vector<FieldElement>>(target_size_, FieldElement::zero());
                }

                std::vector<FieldElement> domain_values(domain_->size, FieldElement::zero());
                auto *domain_native = FieldElement::mutable_native_array_data(domain_values.data());
                parallel_for_chunks(domain_->size, 1ULL << 15, 4, [&](std::size_t begin, std::size_t end)
                                    {
                    mcl::Fr product;
                    for (std::size_t i = begin; i < end; ++i) {
                        mcl::Fr acc;
                        acc.clear();
                        for (const auto &term : resolved_terms) {
                            if (i >= term.values->size()) {
                                continue;
                            }
                            if (term.scale_is_one) {
                                mcl::Fr::add(acc, acc, (*term.values)[i].native());
                            } else {
                                mcl::Fr::mul(product, (*term.values)[i].native(), term.scale);
                                mcl::Fr::add(acc, acc, product);
                            }
                        }
                        domain_native[i] = acc;
                    } });

                auto polynomial = algebra::Polynomial::from_evaluations(
                    derived_label,
                    std::move(domain_values),
                    domain_);
                return std::make_shared<const std::vector<FieldElement>>(
                    coset_evaluate_polynomial(
                        polynomial,
                        target_size_,
                        shift_,
                        derived_label,
                        &shift_powers()));
            }

            std::shared_ptr<const std::vector<FieldElement>> derived_hidden_edge_coset(const std::string &name)
            {
                if (!context_.model.has_real_multihead || label_ != "t_edge")
                {
                    return nullptr;
                }
                for (std::size_t head_index = 0; head_index < context_.model.hidden_heads.size(); ++head_index)
                {
                    const auto prefix = "P_h" + std::to_string(head_index) + "_";
                    if (name.rfind(prefix, 0) != 0)
                    {
                        continue;
                    }
                    const auto suffix = name.substr(prefix.size());
                    const auto eta_l = challenges_.at("eta_L_h" + std::to_string(head_index));
                    const auto eta_exp = challenges_.at("eta_exp_h" + std::to_string(head_index));
                    const auto eta_elu = challenges_.at("eta_ELU_h" + std::to_string(head_index));
                    const auto eta_src = challenges_.at("eta_src_h" + std::to_string(head_index));
                    const auto eta_dst = challenges_.at("eta_dst_h" + std::to_string(head_index));
                    const auto eta_t = challenges_.at("eta_t_h" + std::to_string(head_index));
                    if (suffix == "Table_L")
                    {
                        return scaled_coset_sum({{"P_T_L_x", FieldElement::one()}, {"P_T_L_y", eta_l}});
                    }
                    if (suffix == "Query_L")
                    {
                        return scaled_coset_sum({{prefix + "S", FieldElement::one()}, {prefix + "Z", eta_l}});
                    }
                    if (suffix == "S")
                    {
                        return scaled_coset_sum({{prefix + "E_src_edge", FieldElement::one()}, {prefix + "E_dst_edge", FieldElement::one()}});
                    }
                    if (suffix == "Delta")
                    {
                        return scaled_coset_sum({{prefix + "M_edge", FieldElement::one()}, {prefix + "Z", -FieldElement::one()}});
                    }
                    if (suffix == "Table_R")
                    {
                        return coset_values_ptr("P_T_range");
                    }
                    if (suffix == "Query_R")
                    {
                        return coset_values_ptr(prefix + "Delta");
                    }
                    if (suffix == "Table_exp")
                    {
                        return scaled_coset_sum({{"P_T_exp_x", FieldElement::one()}, {"P_T_exp_y", eta_exp}});
                    }
                    if (suffix == "Query_exp")
                    {
                        return scaled_coset_sum({{prefix + "Delta", FieldElement::one()}, {prefix + "U", eta_exp}});
                    }
                    if (suffix == "w_psq")
                    {
                        if (lookup_polynomial(name) != nullptr || trace_.spilled_polynomials.contains(name))
                        {
                            return nullptr;
                        }
                        const auto lambda_psq = challenges_.at("lambda_psq_h" + std::to_string(head_index));
                        return scaled_coset_sum({{prefix + "U", FieldElement::one()}, {prefix + "widehat_v_pre_star", lambda_psq}});
                    }
                    if (suffix == "T_psq_edge")
                    {
                        if (lookup_polynomial(name) != nullptr || trace_.spilled_polynomials.contains(name))
                        {
                            return nullptr;
                        }
                        const auto lambda_psq = challenges_.at("lambda_psq_h" + std::to_string(head_index));
                        return scaled_coset_sum({{prefix + "Sum_edge", FieldElement::one()}, {prefix + "H_agg_pre_star_edge", lambda_psq}});
                    }
                    if (suffix == "Table_ELU")
                    {
                        return scaled_coset_sum({{"P_T_ELU_x", FieldElement::one()}, {"P_T_ELU_y", eta_elu}});
                    }
                    if (suffix == "Query_ELU")
                    {
                        return scaled_domain_sum_coset(
                            name,
                            {{prefix + "H_agg_pre_flat", FieldElement::one()}, {prefix + "H_agg_flat", eta_elu}});
                    }
                    if (suffix == "Query_src")
                    {
                        const auto eta_src_sq = eta_src * eta_src;
                        return scaled_coset_sum({
                            {"P_src", FieldElement::one()},
                            {prefix + "E_src_edge", eta_src},
                            {prefix + "H_src_star_edge", eta_src_sq},
                        });
                    }
                    if (suffix == "Query_t")
                    {
                        return scaled_coset_sum({{"P_dst", FieldElement::one()}, {prefix + "T_psq_edge", eta_t}});
                    }
                    if (suffix == "Query_dst")
                    {
                        const auto eta_dst_sq = eta_dst * eta_dst;
                        const auto eta_dst_cu = eta_dst_sq * eta_dst;
                        const auto eta_dst_4 = eta_dst_cu * eta_dst;
                        const auto eta_dst_5 = eta_dst_4 * eta_dst;
                        return scaled_coset_sum({
                            {"P_dst", FieldElement::one()},
                            {prefix + "E_dst_edge", eta_dst},
                            {prefix + "M_edge", eta_dst_sq},
                            {prefix + "Sum_edge", eta_dst_cu},
                            {prefix + "inv_edge", eta_dst_4},
                            {prefix + "H_agg_star_edge", eta_dst_5},
                        });
                    }
                    return nullptr;
                }
                return nullptr;
            }

            std::shared_ptr<const std::vector<FieldElement>> derived_output_edge_coset(const std::string &name)
            {
                if (!context_.model.has_real_multihead || label_ != "t_edge")
                {
                    return nullptr;
                }
                for (std::size_t head_index = 0; head_index < context_.model.output_layer.heads.size(); ++head_index)
                {
                    const auto prefix = output_head_prefix(head_index) + "_";
                    if (name.rfind(prefix, 0) != 0)
                    {
                        continue;
                    }
                    const auto suffix = name.substr(prefix.size());
                    const auto eta_l = challenges_.at(output_challenge_name("eta_L_out", head_index));
                    const auto eta_exp = challenges_.at(output_challenge_name("eta_exp_out", head_index));
                    const auto eta_src = challenges_.at(output_challenge_name("eta_src_out", head_index));
                    const auto eta_dst = challenges_.at(output_challenge_name("eta_dst_out", head_index));
                    const auto eta_t = challenges_.at(output_challenge_name("eta_t_out", head_index));
                    if (suffix == "Table_L")
                    {
                        return scaled_coset_sum({{"P_T_L_x", FieldElement::one()}, {"P_T_L_y", eta_l}});
                    }
                    if (suffix == "Query_L")
                    {
                        return scaled_coset_sum({{prefix + "S", FieldElement::one()}, {prefix + "Z", eta_l}});
                    }
                    if (suffix == "S")
                    {
                        return scaled_coset_sum({{prefix + "E_src_edge", FieldElement::one()}, {prefix + "E_dst_edge", FieldElement::one()}});
                    }
                    if (suffix == "Delta")
                    {
                        return scaled_coset_sum({{prefix + "M_edge", FieldElement::one()}, {prefix + "Z", -FieldElement::one()}});
                    }
                    if (suffix == "Table_R")
                    {
                        return coset_values_ptr("P_T_range");
                    }
                    if (suffix == "Query_R")
                    {
                        return coset_values_ptr(prefix + "Delta");
                    }
                    if (suffix == "Table_exp")
                    {
                        return scaled_coset_sum({{"P_T_exp_x", FieldElement::one()}, {"P_T_exp_y", eta_exp}});
                    }
                    if (suffix == "Query_exp")
                    {
                        return scaled_coset_sum({{prefix + "Delta", FieldElement::one()}, {prefix + "U", eta_exp}});
                    }
                    if (suffix == "w")
                    {
                        if (lookup_polynomial(name) != nullptr || trace_.spilled_polynomials.contains(name))
                        {
                            return nullptr;
                        }
                        const auto lambda_out = challenges_.at(output_challenge_name("lambda_out", head_index));
                        return scaled_coset_sum({{prefix + "U", FieldElement::one()}, {prefix + "widehat_y_star", lambda_out}});
                    }
                    if (suffix == "T_edge")
                    {
                        const auto lambda_out = challenges_.at(output_challenge_name("lambda_out", head_index));
                        return scaled_coset_sum({{prefix + "Sum_edge", FieldElement::one()}, {prefix + "Y_star_edge", lambda_out}});
                    }
                    if (suffix == "Query_src")
                    {
                        return scaled_coset_sum({{"P_src", FieldElement::one()}, {prefix + "E_src_edge", eta_src}});
                    }
                    if (suffix == "Query_t")
                    {
                        return scaled_coset_sum({{"P_dst", FieldElement::one()}, {prefix + "T_edge", eta_t}});
                    }
                    if (suffix == "Query_dst")
                    {
                        const auto eta_dst_sq = eta_dst * eta_dst;
                        const auto eta_dst_cu = eta_dst_sq * eta_dst;
                        const auto eta_dst_4 = eta_dst_cu * eta_dst;
                        const auto eta_dst_5 = eta_dst_4 * eta_dst;
                        return scaled_coset_sum({
                            {"P_dst", FieldElement::one()},
                            {prefix + "E_dst_edge", eta_dst},
                            {prefix + "M_edge", eta_dst_sq},
                            {prefix + "Sum_edge", eta_dst_cu},
                            {prefix + "inv_edge", eta_dst_4},
                            {prefix + "Y_star_edge", eta_dst_5},
                        });
                    }
                    return nullptr;
                }
                return nullptr;
            }

            std::string canonical_coset_label(const std::string &name) const
            {
                if (label_ == "t_edge"
                    && (name == "P_Q_qry_src"
                        || name == "P_Q_qry_dst"
                        || name == "P_Q_qry_L"
                        || name == "P_Q_qry_R"
                        || name == "P_Q_qry_exp"))
                {
                    return "P_Q_edge_valid";
                }
                return name;
            }

            std::shared_ptr<const std::vector<FieldElement>> feature_selector_coset(const std::string &requested_name)
            {
                if (label_ != "t_FH" || domain_ != context_.domains.fh)
                {
                    return nullptr;
                }
                if (requested_name != "P_Q_tbl_feat" && requested_name != "P_Q_qry_feat")
                {
                    return nullptr;
                }
                const auto *polynomial = lookup_polynomial(requested_name);
                if (polynomial == nullptr)
                {
                    return nullptr;
                }
                if (polynomial->basis != algebra::PolynomialBasis::Evaluation
                    || polynomial->domain == nullptr
                    || polynomial->domain->size != domain_->size
                    || !(polynomial->domain->omega == domain_->omega)
                    || polynomial->values().size() != domain_->size)
                {
                    throw std::runtime_error("FH selector fast path domain mismatch: " + requested_name);
                }
                if (target_size_ != domain_->size * quotient_coset_multiplier(label_) || rotation_step_ != target_size_ / domain_->size)
                {
                    throw std::runtime_error("FH selector fast path target coset mismatch: " + requested_name);
                }
                const auto one = FieldElement::one();
                if (!(domain_->point_at(0) == one) || !(domain_->point_at(1) == domain_->omega))
                {
                    throw std::runtime_error("FH selector fast path requires canonical domain order: " + requested_name);
                }
                const auto shifted_zero = shift_.pow(domain_->size);
                if (shifted_zero == one || shifted_zero == -one)
                {
                    throw std::runtime_error("FH selector fast path requires non-overlapping shifted coset: " + requested_name);
                }

                const std::size_t row_count =
                    requested_name == "P_Q_tbl_feat" ? context_.dataset.num_nodes : context_.local.num_nodes;
                const std::size_t feature_count =
                    requested_name == "P_Q_tbl_feat" ? context_.dataset.num_features : context_.local.num_features;
                if (feature_count != 0 && row_count > std::numeric_limits<std::size_t>::max() / feature_count)
                {
                    throw std::runtime_error("feature selector size overflow while materializing " + requested_name);
                }
                const auto valid_count = row_count * feature_count;

                progress_note("prove:quotient:t_FH:selector:" + requested_name + ":coefficients:start");
                auto coefficients = prefix_selector_coefficients_native(*domain_, valid_count, requested_name);
                progress_note("prove:quotient:t_FH:selector:" + requested_name + ":coefficients:done");
                progress_note("prove:quotient:t_FH:selector:" + requested_name + ":coset:start");
                auto values = std::make_shared<const std::vector<FieldElement>>(
                    coset_evaluate_coefficients(coefficients, target_size_, shift_, requested_name, &shift_powers()));
                progress_note("prove:quotient:t_FH:selector:" + requested_name + ":coset:done");
                return values;
            }

            std::shared_ptr<const std::vector<FieldElement>> cached_public_fh_material_coset(const std::string &requested_name)
            {
                if (label_ != "t_FH" || domain_ != context_.domains.fh)
                {
                    return nullptr;
                }
                const auto it = context_.public_fh_coefficient_materials.find(requested_name);
                if (it == context_.public_fh_coefficient_materials.end() || it->second == nullptr)
                {
                    return nullptr;
                }
                progress_note("prove:quotient:t_FH:public_coeff:" + requested_name + ":coset:start");
                auto values = std::make_shared<const std::vector<FieldElement>>(
                    coset_evaluate_polynomial(*it->second, target_size_, shift_, requested_name, &shift_powers()));
                progress_note("prove:quotient:t_FH:public_coeff:" + requested_name + ":coset:done");
                return values;
            }

            std::shared_ptr<const std::vector<FieldElement>> coset_values_ptr(const std::string &requested_name)
            {
                std::string alias_owner;
                {
                    std::lock_guard<std::mutex> lock(coset_cache_mutex_);
                    if (const auto it = coset_cache_.find(requested_name); it != coset_cache_.end())
                    {
                        return it->second;
                    }
                    if (const auto alias_it = coset_aliases_.find(requested_name); alias_it != coset_aliases_.end())
                    {
                        alias_owner = alias_it->second;
                    }
                }
                if (const auto *private_polynomial = lookup_polynomial(requested_name);
                    private_polynomial != nullptr && private_polynomial->is_zero_domain_blinded())
                {
                    auto direct_values = reused_semantic_coset_values(
                        requested_name, *private_polynomial);
                    if (direct_values == nullptr)
                    {
                        direct_values =
                            std::make_shared<const std::vector<FieldElement>>(
                                coset_evaluate_polynomial(
                                    *private_polynomial,
                                    target_size_,
                                    shift_,
                                    requested_name,
                                    &shift_powers()));
                    }
                    std::lock_guard<std::mutex> lock(coset_cache_mutex_);
                    const auto [it, inserted] = coset_cache_.emplace(requested_name, direct_values);
                    return inserted ? direct_values : it->second;
                }
                if (!alias_owner.empty() && alias_owner != requested_name)
                {
                    auto values = coset_values_ptr(alias_owner);
                    std::lock_guard<std::mutex> lock(coset_cache_mutex_);
                    const auto [it, inserted] = coset_cache_.emplace(requested_name, values);
                    return inserted ? values : it->second;
                }
                std::shared_ptr<const std::vector<FieldElement>> values;
                const auto canonical_name = canonical_coset_label(requested_name);
                if (canonical_name != requested_name)
                {
                    values = coset_values_ptr(canonical_name);
                }
                else if (full_graph_feature_identity_enabled(context_) && requested_name == "P_Q_qry_feat")
                {
                    values = coset_values_ptr("P_Q_tbl_feat");
                }
                else if (full_graph_feature_identity_enabled(context_) && (requested_name == "P_Row_feat_qry" || requested_name == "P_I_feat_qry"))
                {
                    values = coset_values_ptr("P_Row_feat_tbl");
                }
                else if (full_graph_feature_identity_enabled(context_) && requested_name == "P_Col_feat_qry")
                {
                    values = coset_values_ptr("P_Col_feat_tbl");
                }
                else if (requested_name == "P_H" && lazy_full_feature_lookup_trace_enabled(context_))
                {
                    values = coset_values_ptr("P_T_H");
                }
                else if (lazy_full_feature_lookup_trace_enabled(context_) && is_lazy_full_feature_lookup_trace_label(requested_name))
                {
                    values = materialize_lazy_feature_lookup_trace(requested_name);
                }
                else if (auto public_material = cached_public_fh_material_coset(requested_name); public_material != nullptr)
                {
                    values = std::move(public_material);
                }
                else if (auto selector = feature_selector_coset(requested_name); selector != nullptr)
                {
                    values = std::move(selector);
                }
                else if (auto derived = derived_hidden_edge_coset(requested_name); derived != nullptr)
                {
                    values = std::move(derived);
                }
                else if (auto derived = derived_output_edge_coset(requested_name); derived != nullptr)
                {
                    values = std::move(derived);
                }
                else if (const auto *polynomial = lookup_polynomial(requested_name); polynomial != nullptr)
                {
                    values = std::make_shared<const std::vector<FieldElement>>(
                        coset_evaluate_polynomial(*polynomial, target_size_, shift_, requested_name, &shift_powers()));
                }
                else if (const auto it = trace_.spilled_polynomials.find(requested_name); it != trace_.spilled_polynomials.end())
                {
                    auto polynomial = algebra::Polynomial::from_evaluations(
                        requested_name,
                        load_spilled_values(it->second),
                        context_domain_by_name(context_, it->second.domain_name));
                    if (!it->second.zero_domain_blinder.empty())
                    {
                        polynomial.set_zero_domain_blinder(it->second.zero_domain_blinder);
                    }
                    values = std::make_shared<const std::vector<FieldElement>>(
                        coset_evaluate_polynomial(polynomial, target_size_, shift_, requested_name, &shift_powers()));
                }
                else
                {
                    throw std::runtime_error("missing quotient dependency polynomial: " + requested_name);
                }
                {
                    std::lock_guard<std::mutex> lock(coset_cache_mutex_);
                    const auto [it, inserted] = coset_cache_.emplace(requested_name, values);
                    return inserted ? values : it->second;
                }
            }

            std::shared_ptr<const std::vector<FieldElement>> materialize_lazy_feature_lookup_trace(const std::string &name)
            {
                if (name == "P_R_feat")
                {
                    return std::make_shared<const std::vector<FieldElement>>(target_size_, FieldElement::zero());
                }
                if (name == "P_m_feat")
                {
                    return full_graph_feature_identity_enabled(context_)
                               ? coset_values_ptr("P_Q_tbl_feat")
                               : coset_values_ptr("P_Q_qry_feat");
                }
                if (name == "P_Query_feat" && full_graph_feature_identity_enabled(context_))
                {
                    return coset_values_ptr("P_Table_feat");
                }

                const auto eta_feat = challenges_.at("eta_feat");
                const auto eta_feat_sq = eta_feat * eta_feat;
                const auto &rows = coset_values(name == "P_Table_feat" ? "P_Row_feat_tbl" : "P_Row_feat_qry");
                const auto &cols = coset_values(name == "P_Table_feat" ? "P_Col_feat_tbl" : "P_Col_feat_qry");
                const auto &features = coset_values("P_T_H");
                auto out = std::make_shared<std::vector<FieldElement>>(target_size_, FieldElement::zero());
                for (std::size_t i = 0; i < target_size_; ++i)
                {
                    (*out)[i] = rows[i] + eta_feat * cols[i] + eta_feat_sq * features[i];
                }
                return out;
            }

            std::vector<mcl::Fr> dependency_coefficients(
                const std::string &name,
                const algebra::Polynomial &polynomial) const
            {
                (void)name;
                return coefficients_native_from_polynomial(polynomial);
            }

            const std::vector<mcl::Fr> &shift_powers()
            {
                if (!shift_powers_.empty())
                {
                    return shift_powers_;
                }
                shift_powers_.resize(target_size_);
                if (target_size_ == 0)
                {
                    return shift_powers_;
                }
                shift_powers_[0] = 1;
                const auto shift_native = shift_.native();
                for (std::size_t i = 1; i < shift_powers_.size(); ++i)
                {
                    mcl::Fr::mul(shift_powers_[i], shift_powers_[i - 1], shift_native);
                }
                return shift_powers_;
            }

            const algebra::Polynomial *lookup_polynomial(const std::string &name) const
            {
                if (const auto it = trace_.polynomials.find(name); it != trace_.polynomials.end())
                {
                    return &it->second;
                }
                if (const auto it = context_.public_polynomials.find(name); it != context_.public_polynomials.end())
                {
                    return &it->second;
                }
                return nullptr;
            }

            const ProtocolContext &context_;
            const TraceArtifacts &trace_;
            const std::map<std::string, FieldElement> &challenges_;
            std::string label_;
            std::shared_ptr<algebra::RootOfUnityDomain> domain_;
            std::size_t target_size_ = 0;
            std::size_t rotation_step_ = 0;
            FieldElement shift_;
            std::vector<mcl::Fr> shift_powers_;
            std::shared_ptr<const TEdgeEvaluationPlan> edge_evaluation_plan_;
            std::mutex coset_cache_mutex_;
            std::unordered_map<std::string, std::shared_ptr<const std::vector<FieldElement>>> coset_cache_;
            std::unordered_map<std::string, std::string> coset_aliases_;
            std::unordered_map<std::string, const std::vector<FieldElement> *> fast_coset_lookup_;
            std::vector<std::shared_ptr<SemanticCosetGroup>> semantic_coset_groups_;
            std::unordered_map<std::string, std::shared_ptr<SemanticCosetGroup>>
                semantic_coset_group_by_label_;
            std::vector<FieldElement> semantic_coset_points_;
            std::vector<FieldElement> semantic_coset_zero_values_;
        };

        std::vector<std::pair<std::string, algebra::Polynomial>> materialize_quotient_polynomials(
            const ProtocolContext &context,
            const TraceArtifacts &trace,
            const std::map<std::string, FieldElement> &challenges,
            const std::vector<std::string> &labels,
            RunMetrics *metrics)
        {
            std::vector<std::pair<std::string, algebra::Polynomial>> out;
            out.reserve(labels.size());

            auto worker_count_for_labels = [&]()
            {
                const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
                if (labels.size() <= 1 || cpu_count < 8 || !context.model.has_real_multihead)
                {
                    return std::size_t{1};
                }
                const auto fh_size = context.domains.fh != nullptr ? context.domains.fh->size : 0;
                const auto edge_size = context.domains.edge != nullptr ? context.domains.edge->size : 0;
                if (fh_size >= (1ULL << 25) || edge_size >= (1ULL << 22))
                {
                    return std::size_t{1};
                }
                if (fh_size >= (1ULL << 23) || edge_size >= (1ULL << 18))
                {
                    return std::min<std::size_t>(2, labels.size());
                }
                if (std::find(labels.begin(), labels.end(), "t_edge") != labels.end() && edge_size >= (1ULL << 15))
                {
                    return std::min<std::size_t>(2, labels.size());
                }
                return std::min<std::size_t>(3, labels.size());
            };

            const auto materialize_one = [&](const std::string &label)
            {
                const auto start = Clock::now();
                progress_note("prove:quotient:" + label + ":start");
                QuotientPolynomialMaterializer materializer(context, trace, challenges, label);
                auto polynomial = materializer.materialize(context.kzg);
                progress_note("prove:quotient:" + label + ":done");
                if (auto *metric = quotient_metric(metrics, label); metric != nullptr)
                {
                    *metric += elapsed_ms(start, Clock::now());
                }
                return std::pair<std::string, algebra::Polynomial>{label, std::move(polynomial)};
            };

            const auto worker_count = worker_count_for_labels();
            const auto t_edge_it = std::find(labels.begin(), labels.end(), "t_edge");
            const auto edge_size = context.domains.edge != nullptr ? context.domains.edge->size : 0;
            if (worker_count > 1
                && context.model.has_real_multihead
                && t_edge_it != labels.end()
                && edge_size >= (1ULL << 15))
            {
                append_note(metrics, "quotient_materialization_t_edge_priority=1");
                std::vector<std::optional<std::pair<std::string, algebra::Polynomial>>> results(labels.size());
                const auto t_edge_index = static_cast<std::size_t>(std::distance(labels.begin(), t_edge_it));
                results[t_edge_index] = materialize_one("t_edge");

                std::vector<std::size_t> remaining_indices;
                remaining_indices.reserve(labels.size() - 1);
                for (std::size_t i = 0; i < labels.size(); ++i)
                {
                    if (i != t_edge_index)
                    {
                        remaining_indices.push_back(i);
                    }
                }
                const auto remaining_worker_count = std::min<std::size_t>(worker_count, remaining_indices.size());
                if (remaining_worker_count <= 1)
                {
                    for (const auto index : remaining_indices)
                    {
                        results[index] = materialize_one(labels[index]);
                    }
                }
                else
                {
                    append_note(metrics, "parallel_quotient_materialization_workers_after_t_edge=" + std::to_string(remaining_worker_count));
                    std::atomic<std::size_t> next{0};
                    std::vector<std::future<void>> futures;
                    futures.reserve(remaining_worker_count);
                    for (std::size_t worker = 0; worker < remaining_worker_count; ++worker)
                    {
                        futures.push_back(std::async(std::launch::async, [&]()
                                                     {
                    while (true) {
                        const auto offset = next.fetch_add(1);
                        if (offset >= remaining_indices.size()) {
                            break;
                        }
                        const auto index = remaining_indices[offset];
                        results[index] = materialize_one(labels[index]);
                    } }));
                    }
                    for (auto &future : futures)
                    {
                        future.get();
                    }
                }
                for (auto &result : results)
                {
                    if (!result.has_value())
                    {
                        throw std::runtime_error("missing t_edge-priority quotient materialization result");
                    }
                    out.push_back(std::move(*result));
                }
                return out;
            }
            if (worker_count <= 1)
            {
                for (const auto &label : labels)
                {
                    out.push_back(materialize_one(label));
                }
                return out;
            }

            append_note(metrics, "parallel_quotient_materialization_workers=" + std::to_string(worker_count));
            std::vector<std::optional<std::pair<std::string, algebra::Polynomial>>> results(labels.size());
            std::atomic<std::size_t> next{0};
            std::vector<std::future<void>> futures;
            futures.reserve(worker_count);
            for (std::size_t worker = 0; worker < worker_count; ++worker)
            {
                futures.push_back(std::async(std::launch::async, [&]()
                                             {
            while (true) {
                const auto index = next.fetch_add(1);
                if (index >= labels.size()) {
                    break;
                }
                results[index] = materialize_one(labels[index]);
            } }));
            }
            for (auto &future : futures)
            {
                future.get();
            }
            for (auto &result : results)
            {
                if (!result.has_value())
                {
                    throw std::runtime_error("missing parallel quotient materialization result");
                }
                out.push_back(std::move(*result));
            }
            return out;
        }

        std::vector<FieldElement> sample_quotient_mask(
            const ProtocolContext &context,
            const std::string &label)
        {
            const auto kMaskCoefficientCount =
                registered_blinder_coefficient_count(
                    context, quotient_mask_label(label));
            std::vector<FieldElement> mask(kMaskCoefficientCount, FieldElement::zero());
            std::array<std::uint8_t, 64> bytes{};
            for (auto &coefficient : mask)
            {
                if (RAND_bytes(bytes.data(), static_cast<int>(bytes.size())) != 1)
                {
                    throw std::runtime_error("CSPRNG failed while sampling quotient mask: " + label);
                }
                coefficient = FieldElement::from_little_endian_mod(bytes.data(), bytes.size());
            }
            while (std::all_of(mask.begin(), mask.end(), [](const FieldElement &value)
                   { return value.is_zero(); }))
            {
                if (RAND_bytes(bytes.data(), static_cast<int>(bytes.size())) != 1)
                {
                    throw std::runtime_error("CSPRNG failed while resampling quotient mask: " + label);
                }
                mask.front() = FieldElement::from_little_endian_mod(bytes.data(), bytes.size());
            }
            return mask;
        }

        std::size_t quotient_domain_size(
            const ProtocolContext &context,
            const std::string &label)
        {
            if (label == "t_FH") return context.domains.fh->size;
            if (label == "t_edge") return context.domains.edge->size;
            if (label == "t_in") return context.domains.in->size;
            if (label == "t_d" || label == "t_d_h") return context.domains.d->size;
            if (label == "t_cat") return context.domains.cat->size;
            if (label == "t_C") return context.domains.c->size;
            if (label == "t_N") return context.domains.n->size;
            throw std::runtime_error("unknown quotient domain: " + label);
        }

        std::vector<std::pair<std::string, algebra::Polynomial>> blind_quotient_polynomials(
            const ProtocolContext &context,
            std::vector<std::pair<std::string, algebra::Polynomial>> &&quotients)
        {
            std::vector<std::pair<std::string, algebra::Polynomial>> out;
            out.reserve(quotients.size() * 2);
            for (auto &[label, quotient] : quotients)
            {
                if (quotient.basis != algebra::PolynomialBasis::Coefficient)
                {
                    throw std::runtime_error("quotient masking requires coefficient basis: " + label);
                }
                auto mask = sample_quotient_mask(context, label);
                const auto domain_size = quotient_domain_size(context, label);
                const auto required_size = domain_size + mask.size();
                if (required_size > context.kzg.pk.degree_bound + 1)
                {
                    throw std::runtime_error("masked quotient exceeds SRS degree bound: " + label);
                }
                if (quotient.data.size() < required_size)
                {
                    quotient.data.resize(required_size, FieldElement::zero());
                }
                for (std::size_t i = 0; i < mask.size(); ++i)
                {
                    quotient.data[i] -= mask[i];
                    quotient.data[domain_size + i] += mask[i];
                }
                auto mask_polynomial = algebra::Polynomial::from_coefficients(
                    quotient_mask_label(label),
                    std::move(mask));
                out.push_back({label, std::move(quotient)});
                out.push_back({mask_polynomial.name, std::move(mask_polynomial)});
            }
            return out;
        }

        std::unordered_map<std::string, Commitment> batch_quotient_commitments(
            std::vector<std::pair<std::string, algebra::Polynomial>> &&named_quotient_polynomials,
            const crypto::KZGKeyPair &key,
            crypto::CommitBatchProfile *profile)
        {
            std::unordered_map<std::string, Commitment> out;
            out.reserve(named_quotient_polynomials.size());
            if (named_quotient_polynomials.empty())
            {
                return out;
            }

            auto commitments = crypto::KZG::commit_owned_batch(
                std::move(named_quotient_polynomials),
                key,
                profile,
                "prove:quotient:commit_batch");
            for (const auto &commitment : commitments)
            {
                out.emplace(commitment.name, commitment);
            }
            return out;
        }

        std::vector<Commitment> collect_commitments(
            const TraceArtifacts &trace,
            const std::unordered_map<std::string, Commitment> &quotient_commitments,
            const std::vector<std::string> &labels)
        {
            std::vector<Commitment> out;
            out.reserve(labels.size());
            for (const auto &label : labels)
            {
                if (trace.commitments.contains(label))
                {
                    out.push_back(trace.commitments.at(label));
                }
                else
                {
                    out.push_back(quotient_commitments.at(label));
                }
            }
            return out;
        }

        const std::vector<std::string> &quotient_dependencies_fh()
        {
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
            std::vector<std::string> *target,
            const std::vector<std::string> &labels)
        {
            for (const auto &label : labels)
            {
                if (std::find(target->begin(), target->end(), label) == target->end())
                {
                    target->push_back(label);
                }
            }
        }

        std::vector<FieldElement> fh_dependency_points(
            const std::shared_ptr<algebra::RootOfUnityDomain> &domain,
            const std::vector<FieldElement> &points)
        {
            std::vector<FieldElement> out = points;
            out.reserve(points.size() * 2);
            for (const auto &point : points)
            {
                const auto rotated = point * domain->omega;
                if (std::find(out.begin(), out.end(), rotated) == out.end())
                {
                    out.push_back(rotated);
                }
            }
            return out;
        }

        const std::vector<std::string> &quotient_dependencies_edge()
        {
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
                "P_Q_grp_new_edge",
                "P_Q_grp_end_edge",
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

        std::vector<std::string> multihead_edge_dependencies(const ProtocolContext &context)
        {
            std::vector<std::string> labels = {
                "P_Q_edge_valid",
                "P_Q_grp_new_edge",
                "P_Q_grp_end_edge",
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
                "P_BiasIndex_edge",
            };
            auto append_label = [&](std::string label)
            {
                if (std::find(labels.begin(), labels.end(), label) == labels.end())
                {
                    labels.push_back(std::move(label));
                }
            };
            auto append_prefixed = [&](const std::string &prefix, const std::vector<std::string> &suffixes)
            {
                for (const auto &suffix : suffixes)
                {
                    append_label(prefix + suffix);
                }
            };

            const std::vector<std::string> hidden_suffixes = {
                "R_src",
                "Query_src",
                "E_src_edge",
                "E_dst_edge",
                "H_src_star_edge",
                "S",
                "R_L",
                "Table_L",
                "Query_L",
                "m_L",
                "Z",
                "Delta",
                "M_edge",
                "s_max",
                "C_max",
                "R_R",
                "Table_R",
                "Query_R",
                "m_R",
                "alpha",
                "U",
                "inv_edge",
                "R_exp",
                "Table_exp",
                "Query_exp",
                "m_exp",
                "PSQ",
                "w_psq",
                "T_psq_edge",
                "R_t",
                "Query_t",
                "R_ELU",
                "Table_ELU",
                "Query_ELU",
                "m_ELU",
                "H_agg_pre_flat",
                "H_agg_flat",
                "R_dst",
                "Query_dst",
                "Sum_edge",
                "H_agg_star_edge",
            };
            const std::vector<std::string> division_suffixes = {
                "Div_Active", "Div_X", "Div_D", "Div_Y", "Div_Sigma", "Div_Zero",
                "Div_A", "Div_A_inv", "Div_D_inv", "Div_Q", "Div_R", "Div_Delta",
            };
            auto append_division_range_dependencies = [&](const std::string &division_prefix)
            {
                const auto limb_count = signed_division_limb_count(context.config.range_bits);
                for (const auto *value_name : {"A", "DMinus1", "Q", "R", "Delta"})
                {
                    for (std::size_t limb = 0; limb < limb_count; ++limb)
                    {
                        append_label(signed_division_limb_label(
                            division_prefix, value_name, limb));
                        for (const auto *column : {"Table", "Query", "m", "J_tbl", "J_qry", "R"})
                        {
                            append_label(signed_division_range_label(
                                division_prefix, value_name, limb, column));
                        }
                    }
                }
            };
            for (std::size_t head_index = 0; head_index < context.model.hidden_heads.size(); ++head_index)
            {
                append_prefixed("P_h" + std::to_string(head_index) + "_", hidden_suffixes);
                append_prefixed("P_h" + std::to_string(head_index) + "_", division_suffixes);
                append_division_range_dependencies(
                    hidden_head_prefix(head_index) + "_Div_");
                append_label("P_Q_div_h" + std::to_string(head_index));
            }

            const std::vector<std::string> output_suffixes = {
                "R_src",
                "Query_src",
                "E_src_edge",
                "E_dst_edge",
                "S",
                "R_L",
                "Table_L",
                "Query_L",
                "m_L",
                "Z",
                "Delta",
                "M_edge",
                "s_max",
                "C_max",
                "R_R",
                "Table_R",
                "Query_R",
                "m_R",
                "alpha",
                "U",
                "inv_edge",
                "R_exp",
                "Table_exp",
                "Query_exp",
                "m_exp",
                "PSQ",
                "w",
                "T_edge",
                "R_t",
                "Query_t",
                "R_dst",
                "Query_dst",
                "Sum_edge",
                "Y_star_edge",
                "Y_pre_flat",
                "Bias_flat",
                "Y_flat",
                "Bind_a",
                "Bind_b",
                "Bind_Acc",
            };
            for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index)
            {
                append_prefixed(output_head_prefix(head_index) + "_", output_suffixes);
                append_prefixed(output_head_prefix(head_index) + "_", division_suffixes);
                append_division_range_dependencies(
                    output_head_prefix(head_index) + "_Div_");
                append_label("P_Q_div_out" + std::to_string(head_index));
                const auto metadata = build_binding_relation_metadata(context);
                const auto relation_id =
                    "bind:output:" + std::to_string(head_index);
                const auto relation = std::find_if(
                    metadata.begin(), metadata.end(),
                    [&](const BindingRelationMeta &candidate)
                    { return candidate.relation_id == relation_id; });
                if (relation == metadata.end()
                    || relation->coefficient_basis_relations.size() != 1)
                    throw std::runtime_error(
                        "edge quotient dependency registry is missing coefficient basis metadata");
                const auto &basis =
                    relation->coefficient_basis_relations.front();
                for (const auto &chunk : basis.chunks)
                {
                    append_label(chunk.object_id);
                    for (const auto *region : {
                             "active", "semantic", "gap", "tail"})
                        append_label(output_bias_basis_region_selector_label(
                            head_index, chunk.chunk_index, region));
                }
                // Gamma powers are a verifier-reconstructible virtual column,
                // not a committed/opened Pi_bind object.  The prover keeps the
                // interpolant only as quotient-construction material.
                append_label(basis.power_object);
                for (const auto &chunk : basis.chunks)
                    append_label(chunk.accumulator_object);
            }
            for (const auto &suffix : division_suffixes)
            {
                append_label("P_avg_" + suffix);
            }
            append_division_range_dependencies("P_avg_Div_");
            append_label("P_Q_div_avg");
            append_label("P_avg_Bind_a");
            append_label("P_avg_Bind_b");
            append_label("P_avg_Bind_Acc");
            return labels;
        }

        std::vector<std::string> multihead_in_dependencies(const ProtocolContext &context)
        {
            std::vector<std::string> labels = domain_opening_labels(context, "in");
            append_unique_labels(&labels, {"P_Q_proj_valid"});
            return labels;
        }

        std::vector<std::string> multihead_d_dependencies(const ProtocolContext &context)
        {
            std::vector<std::string> labels = domain_opening_labels(context, "d");
            append_unique_labels(&labels, {"P_Q_d_valid"});
            return labels;
        }

        std::vector<std::string> multihead_cat_dependencies(const ProtocolContext &context)
        {
            std::vector<std::string> labels = domain_opening_labels(context, "cat");
            append_unique_labels(&labels, {"P_Q_cat_valid"});
            return labels;
        }

        std::vector<std::string> multihead_c_dependencies(const ProtocolContext &context)
        {
            std::vector<std::string> labels = domain_opening_labels(context, "C");
            append_unique_labels(
                &labels,
                {"P_Q_C_valid", "P_BiasIndex_C", "P_BiasMultiplicity_C"});
            return labels;
        }

        std::vector<std::string> multihead_n_dependencies(const ProtocolContext &context)
        {
            std::vector<std::string> labels = domain_opening_labels(context, "N");
            append_unique_labels(
                &labels,
                {"P_I", "P_Q_N", "P_AggBind_index_N",
                 "P_AggBind_m_src", "P_AggBind_m_dst"});
            return labels;
        }

        const std::vector<std::string> &quotient_dependencies_in()
        {
            static const std::vector<std::string> labels = {
                "P_Q_proj_valid",
                "P_a_proj",
                "P_b_proj",
                "P_Acc_proj",
            };
            return labels;
        }

        const std::vector<std::string> &quotient_dependencies_d()
        {
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

        const std::vector<std::string> &quotient_dependencies_n()
        {
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
            const ProtocolContext &context,
            const std::string &quotient_name)
        {
            if (!context.model.has_real_multihead)
            {
                if (quotient_name == "t_FH")
                    return quotient_dependencies_fh();
                if (quotient_name == "t_edge")
                    return quotient_dependencies_edge();
                if (quotient_name == "t_in")
                    return quotient_dependencies_in();
                if (quotient_name == "t_d" || quotient_name == "t_d_h")
                    return quotient_dependencies_d();
                if (quotient_name == "t_N")
                    return quotient_dependencies_n();
                return {};
            }

            if (quotient_name == "t_FH")
                return quotient_dependencies_fh();
            if (quotient_name == "t_edge")
                return multihead_edge_dependencies(context);
            if (quotient_name == "t_in")
                return multihead_in_dependencies(context);
            if (quotient_name == "t_d_h" || quotient_name == "t_d")
                return multihead_d_dependencies(context);
            if (quotient_name == "t_cat")
                return multihead_cat_dependencies(context);
            if (quotient_name == "t_C")
                return multihead_c_dependencies(context);
            if (quotient_name == "t_N")
                return multihead_n_dependencies(context);
            return {};
        }

        DomainOpeningBundle make_bundle(
            const ProtocolContext &context,
            const TraceArtifacts &trace,
            EvaluationMemoization &memo,
            const std::unordered_map<std::string, Commitment> &quotient_commitments,
            const std::vector<std::string> &labels,
            const std::vector<FieldElement> &points,
            RunMetrics *metrics = nullptr,
            double *eval_gather_ms = nullptr,
            const std::string &progress_prefix = "")
        {
            // 一个 bundle 对应 formal schema 中某个工作域的一次 opening。
            // 这里必须严格复用 `domain_opening_labels()` 给出的标签顺序，
            // 因为 prover 组包、verifier 取值和 proof_size_bytes 统计都依赖同一份 shape。
            DomainOpeningBundle bundle;
            bundle.points = points;
            const auto commitments = collect_commitments(trace, quotient_commitments, labels);
            const auto gather_start = Clock::now();
            std::string quotient_name;
            for (const auto &label : labels)
            {
                if (!label.empty() && label[0] == 't')
                {
                    quotient_name = label;
                    break;
                }
            }
            const bool is_fh_bundle = quotient_name == "t_FH";
            std::vector<std::vector<FieldElement>> values;
            if (!progress_prefix.empty())
            {
                progress_note(progress_prefix + ":gather:start");
            }
            if (!quotient_name.empty())
            {
                const auto dependency_labels = quotient_dependency_labels(context, quotient_name);
                std::shared_ptr<algebra::RootOfUnityDomain> domain;
                if (quotient_name == "t_FH")
                {
                    domain = context.domains.fh;
                }
                else if (quotient_name == "t_edge")
                {
                    domain = context.domains.edge;
                }
                else if (quotient_name == "t_in")
                {
                    domain = context.domains.in;
                }
                else if (quotient_name == "t_d" || quotient_name == "t_d_h")
                {
                    domain = context.domains.d;
                }
                else if (quotient_name == "t_cat")
                {
                    domain = context.domains.cat;
                }
                else if (quotient_name == "t_C")
                {
                    domain = context.domains.c;
                }
                else if (quotient_name == "t_N")
                {
                    domain = context.domains.n;
                }
                if (domain != nullptr && !dependency_labels.empty())
                {
                    memo.precompute_named(dependency_labels, fh_dependency_points(domain, points));
                }
                values.reserve(labels.size());
                for (const auto &label : labels)
                {
                    if (const auto commitment_it = quotient_commitments.find(label);
                        commitment_it != quotient_commitments.end())
                    {
                        if (commitment_it->second.opening_polynomial == nullptr)
                        {
                            throw std::runtime_error(
                                "quotient opening material is unavailable: " + label);
                        }
                        std::vector<FieldElement> opened;
                        opened.reserve(points.size());
                        for (const auto &point : points)
                        {
                            opened.push_back(
                                commitment_it->second.opening_polynomial->evaluate(point));
                        }
                        values.push_back(std::move(opened));
                    }
                    else
                    {
                        values.push_back(
                            memo.collect_named_values_from_cache({label}, points).front());
                    }
                }
            }
            else
            {
                values = memo.collect_named_values(labels, points);
            }
            if (eval_gather_ms != nullptr)
            {
                *eval_gather_ms += elapsed_ms(gather_start, Clock::now());
            }
            if (!progress_prefix.empty())
            {
                progress_note(progress_prefix + ":gather:done");
            }
            if (metrics != nullptr && is_fh_bundle)
            {
                metrics->fh_open_gather_ms += elapsed_ms(gather_start, Clock::now());
            }
            for (std::size_t i = 0; i < labels.size(); ++i)
            {
                bundle.values.push_back({labels[i], values[i]});
                bundle.value_kinds.push_back(payload_kind_for_label(context, labels[i]));
            }
            return bundle;
        }

    } // namespace

    // 构建一次 formal run 的静态上下文，产出数据集、模型、工作域、公开列与静态承诺。
    // 这里禁止注入任何依赖单次 proof 的 witness 或挑战状态。
    ProtocolContext build_context(const util::AppConfig &config, RunMetrics *metrics)
    {
        static std::mutex cache_mutex;
        static std::unordered_map<std::string, ProtocolContext> cache;
        // 这里只复用静态上下文对象，见证和挑战相关状态必须按每次正式证明重新生成。
        const bool allow_static_context_cache = true;

        const auto cache_lookup_start = Clock::now();
        const auto cache_key = context_cache_key(config);
        if (allow_static_context_cache)
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            if (const auto it = cache.find(cache_key); it != cache.end())
            {
                // 这里只缓存静态上下文：工作域、SRS、公开列、静态承诺和模型数据。
                // 每次 proof 的 witness 与挑战派生值都必须重新生成。
                auto node = cache.extract(it);
                auto context = std::move(node.mapped());
                context.config = config;
                if (metrics != nullptr)
                {
                    metrics->load_static_ms += elapsed_ms(cache_lookup_start, Clock::now());
                    append_note(metrics, "static_context_cache=hit_move");
                }
                return context;
            }
        }

        append_note(metrics, "static_context_cache=miss");
        ProtocolContext context;
        context.config = config;

        progress_note("context:load_static:start");
        auto start = Clock::now();
        // 先把数据集和 checkpoint bundle 收束成 formal shape；这一段还不接触任何 challenge 或 witness。
        context.dataset = data::load_dataset(config);
        context.local = data::normalize_graph_input(context.dataset, config);
        if (config.allow_synthetic_model)
        {
            append_note(metrics, "model_source=synthetic_model");
            context.model = model::build_family_model_parameters(
                config.d_in_profile.empty() ? std::vector<std::size_t>{context.local.num_features} : config.d_in_profile,
                [&]()
                {
                    std::vector<model::HiddenLayerShape> shapes;
                    shapes.reserve(config.hidden_profile.size());
                    for (const auto &layer : config.hidden_profile)
                    {
                        shapes.push_back({layer.head_count, layer.head_dim});
                    }
                    return shapes;
                }(),
                config.K_out,
                context.local.num_classes,
                config.seed);
        }
        else
        {
            if (config.checkpoint_bundle.empty())
            {
                throw std::runtime_error(
                    "formal model route requires checkpoint_bundle; refusing to fall back to synthetic parameters");
            }
            const auto checkpoint_root = resolve_project_path(config.project_root, config.checkpoint_bundle);
            context.model = model::load_checkpoint_bundle_parameters(checkpoint_root.string());
            append_note(metrics, "model_source=checkpoint_bundle");
            append_note(metrics, "hidden_head_count=" + std::to_string(model::flattened_hidden_head_count(context.model)));
        }
        if (!context.model.has_real_multihead)
        {
            throw std::runtime_error(
                "formal protocol requires the indexed multi-layer multi-head model representation");
        }
        if (context.model.has_real_multihead)
        {
            std::string family_reason;
            if (!model::hidden_family_dimension_chain_is_valid(context.model, &family_reason))
            {
                throw std::runtime_error("formal family shape is invalid: " + family_reason);
            }
            bool hidden_profile_matches = context.model.hidden_profile.size() == config.hidden_profile.size();
            for (std::size_t i = 0; hidden_profile_matches && i < config.hidden_profile.size(); ++i)
            {
                hidden_profile_matches =
                    context.model.hidden_profile[i].head_count == config.hidden_profile[i].head_count && context.model.hidden_profile[i].head_dim == config.hidden_profile[i].head_dim;
            }
            if (context.model.L != config.layer_count)
            {
                throw std::runtime_error(
                    "config L=" + std::to_string(config.layer_count) + " conflicts with model L=" + std::to_string(context.model.L));
            }
            if (!hidden_profile_matches)
            {
                throw std::runtime_error("config hidden_profile conflicts with checkpoint manifest hidden_profile");
            }
            if (!config.d_in_profile.empty() && context.model.d_in_profile != config.d_in_profile)
            {
                throw std::runtime_error("config d_in_profile conflicts with checkpoint manifest d_in_profile");
            }
            if (context.model.K_out != config.K_out)
            {
                throw std::runtime_error(
                    "config K_out=" + std::to_string(config.K_out) + " conflicts with model K_out=" + std::to_string(context.model.K_out));
            }
            if (context.model.C != config.num_classes)
            {
                throw std::runtime_error(
                    "config num_classes=" + std::to_string(config.num_classes) + " conflicts with model C=" + std::to_string(context.model.C));
            }
        }
        auto end = Clock::now();
        if (metrics != nullptr)
        {
            metrics->load_static_ms += elapsed_ms(start, end);
        }
        progress_note("context:load_static:done");

        progress_note("context:domains:start");
        start = Clock::now();
        // 工作域大小一次性按特征表、边域、隐藏维和查表规模展开，后面 trace / quotient / opening 共用这套布局。
        const std::size_t fh_size = algebra::next_power_of_two(
            std::max(context.dataset.num_nodes * context.dataset.num_features, context.local.num_nodes * context.local.num_features) + 2);
        const std::size_t configured_range_size = (1ULL << config.range_bits);
        const std::size_t d_h = hidden_head_width(context.model, config);
        const std::size_t d_cat = concat_width(context.model, config);
        const std::size_t max_in = context.model.has_real_multihead
                                       ? std::max<std::size_t>(context.local.num_features, model::max_hidden_input_dim(context.model))
                                       : context.local.num_features;
        const bool large_whole_graph_gat =
            config.batching_rule == "whole_graph_single" && context.local.num_nodes == context.dataset.num_nodes && (context.local.edges.size() >= (1ULL << 18) || context.dataset.num_nodes * context.dataset.num_features >= (1ULL << 24));
        const std::size_t default_activation_table_bound = configured_range_size;
        std::size_t range_size = std::max<std::size_t>(
            configured_range_size,
            large_whole_graph_gat ? (1ULL << 19) : default_activation_table_bound);
        std::int64_t lrelu_min_raw = -static_cast<std::int64_t>(std::max<std::size_t>(
            large_whole_graph_gat ? (1ULL << 20) : default_activation_table_bound,
            range_size));
        std::int64_t lrelu_max_raw = -lrelu_min_raw;
        if (large_whole_graph_gat)
        {
            constexpr std::size_t kLargeEdgeDomainSize = 1ULL << 23;
            constexpr std::size_t kLargeLReluNegativeRows = 3ULL << 18;
            const std::size_t max_rows_in_edge_domain = kLargeEdgeDomainSize - 2;
            range_size = std::max<std::size_t>(
                range_size,
                max_rows_in_edge_domain);
            lrelu_min_raw = -static_cast<std::int64_t>(kLargeLReluNegativeRows);
            lrelu_max_raw = static_cast<std::int64_t>(
                max_rows_in_edge_domain - kLargeLReluNegativeRows - 1);
        }
        const std::size_t elu_bound = std::max<std::size_t>(
            large_whole_graph_gat ? (1ULL << 20) : default_activation_table_bound,
            configured_range_size);
        const std::size_t lrelu_table_size = signed_table_size(lrelu_min_raw, lrelu_max_raw, "lrelu");
        const std::size_t elu_table_size = elu_bound * 2 + 1;
        const std::size_t exp_table_size = range_size;
        const std::size_t edge_size = algebra::next_power_of_two(
            std::max<std::size_t>({
                context.local.num_nodes,
                context.local.edges.size(),
                context.local.num_nodes * d_h,
                lrelu_table_size,
                elu_table_size,
                exp_table_size,
                range_size,
            }) +
            2);
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
        if (context.model.has_real_multihead)
        {
            context.domains.hidden_heads.assign(
                context.model.hidden_heads.size(),
                AttentionHeadDomains{
                    .in = context.domains.in,
                    .d = context.domains.d,
                });
            context.domains.output_head = AttentionHeadDomains{.in = context.domains.cat, .d = context.domains.c};
        }
        end = Clock::now();
        if (metrics != nullptr)
        {
            metrics->fft_plan_ms += elapsed_ms(start, end);
        }
        progress_note("context:domains:done");

        progress_note("context:srs:start");
        start = Clock::now();
        const std::size_t required_coefficient_count = std::max<std::size_t>({
            fh_size * 2,
            edge_size * 4,
            in_size * 4,
            d_size * 4,
            cat_size * 4,
            c_size * 4,
            n_size * 4,
        });
        context.kzg = crypto::KZG::load_keys_for_domains(
            resolve_project_path(config.project_root, config.srs_path).string(),
            config.srs_type,
            config.srs_id,
            {
                {"FH", fh_size},
                {"edge", edge_size},
                {"in", in_size},
                {"d", d_size},
                {"cat", cat_size},
                {"C", c_size},
                {"N", n_size},
            },
            required_coefficient_count);
        std::string envelope_reason;
        if (!instance_within_authenticated_max_envelope(
                context, &envelope_reason))
            throw std::runtime_error(envelope_reason);
        end = Clock::now();
        if (metrics != nullptr)
        {
            metrics->srs_prepare_ms += elapsed_ms(start, end);
        }
        progress_note("context:srs:done");

        progress_note("context:public_static:start");
        start = Clock::now();
        // 这里开始布置公开列、静态表和静态承诺；它们会直接进入 transcript 与 verifier 的公共视图。
        context.tables.range = make_range_table(range_size);
        context.tables.exp = make_exp_table(range_size);
        context.tables.lrelu = make_lrelu_table(lrelu_min_raw, lrelu_max_raw);
        context.tables.elu = make_elu_table(elu_bound);
        require_functional_pair_table(context.tables.lrelu, "LeakyReLU");
        require_functional_pair_table(context.tables.elu, "ELU");
        require_functional_pair_table(context.tables.exp, "ExpMap");

        const auto edge_valid = build_selector(context.local.edges.size(), edge_size);

        const auto q_new_edge = q_new_selector(context.local.edges, edge_size);
        const auto q_end_edge = q_end_selector(context.local.edges, edge_size);
        const auto q_graph_new_edge = q_graph_new_selector(context.local.edges, edge_size);
        const auto q_graph_end_edge = q_graph_end_selector(context.local.edges, edge_size);
        const auto q_grp_new_edge = q_group_selector(q_graph_new_edge, q_new_edge);
        const auto q_grp_end_edge = q_group_selector(q_graph_end_edge, q_end_edge);

        add_public_poly(
            context,
            "P_I",
            make_shared_eval_poly("P_I", ids_to_field(context.local.absolute_ids, n_size), context.domains.n));
        add_public_poly(
            context,
            "P_src",
            make_shared_eval_poly("P_src", edge_component(context.local.edges, edge_size, true), context.domains.edge));
        add_public_poly(
            context,
            "P_dst",
            make_shared_eval_poly("P_dst", edge_component(context.local.edges, edge_size, false), context.domains.edge));
        add_public_poly(
            context,
            "P_Q_new_edge",
            make_shared_eval_poly("P_Q_new_edge", q_new_edge, context.domains.edge));
        add_public_poly(
            context,
            "P_Q_end_edge",
            make_shared_eval_poly("P_Q_end_edge", q_end_edge, context.domains.edge));
        add_public_poly(
            context,
            "P_node_gid",
            make_shared_eval_poly(
                "P_node_gid",
                node_gid_column(context.local.node_ptr, context.local.num_nodes, n_size),
                context.domains.n));
        add_public_poly(
            context,
            "P_edge_gid",
            make_shared_eval_poly("P_edge_gid", edge_gid_column(context.local.edges, edge_size), context.domains.edge));
        add_public_poly(
            context,
            "P_Q_graph_new_edge",
            make_shared_eval_poly("P_Q_graph_new_edge", q_graph_new_edge, context.domains.edge));
        add_public_poly(
            context,
            "P_Q_graph_end_edge",
            make_shared_eval_poly("P_Q_graph_end_edge", q_graph_end_edge, context.domains.edge));
        add_public_poly(
            context,
            "P_Q_grp_new_edge",
            make_shared_eval_poly("P_Q_grp_new_edge", q_grp_new_edge, context.domains.edge));
        add_public_poly(
            context,
            "P_Q_grp_end_edge",
            make_shared_eval_poly("P_Q_grp_end_edge", q_grp_end_edge, context.domains.edge));
        add_public_poly(
            context,
            "P_Q_edge_valid",
            make_shared_eval_poly("P_Q_edge_valid", edge_valid, context.domains.edge));
        if (context.model.has_real_multihead)
        {
            for (std::size_t head_index = 0; head_index < context.model.hidden_heads.size(); ++head_index)
            {
                const auto width = context.model.hidden_heads[head_index].seq_kernel_fp.empty()
                                       ? 0
                                       : context.model.hidden_heads[head_index].seq_kernel_fp.front().size();
                const auto label = "P_Q_div_h" + std::to_string(head_index);
                add_public_poly(
                    context,
                    label,
                    make_shared_eval_poly(
                        label,
                        build_selector(context.local.num_nodes * width, edge_size),
                        context.domains.edge));
            }
            for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index)
            {
                const auto label = "P_Q_div_out" + std::to_string(head_index);
                add_public_poly(
                    context,
                    label,
                    make_shared_eval_poly(
                        label,
                        build_selector(context.local.num_nodes * context.local.num_classes, edge_size),
                        context.domains.edge));
            }
            add_public_poly(
                context,
                "P_Q_div_avg",
                make_shared_eval_poly(
                    "P_Q_div_avg",
                    build_selector(context.local.num_nodes * context.local.num_classes, edge_size),
                    context.domains.edge));
        }
        add_public_poly(
            context,
            "P_Q_N",
            make_shared_eval_poly("P_Q_N", build_selector(context.local.num_nodes, n_size), context.domains.n));
        std::vector<FieldElement> aggregation_src_multiplicity(
            n_size, FieldElement::zero());
        std::vector<FieldElement> aggregation_dst_multiplicity(
            n_size, FieldElement::zero());
        for (const auto &edge : context.local.edges)
        {
            if (edge.src >= context.local.num_nodes
                || edge.dst >= context.local.num_nodes)
                throw std::runtime_error(
                    "aggregation multiplicity edge escapes local node range");
            aggregation_src_multiplicity[edge.src] += FieldElement::one();
            aggregation_dst_multiplicity[edge.dst] += FieldElement::one();
        }
        add_public_poly(
            context,
            "P_AggBind_m_src",
            make_shared_eval_poly(
                "P_AggBind_m_src", std::move(aggregation_src_multiplicity),
                context.domains.n));
        add_public_poly(
            context,
            "P_AggBind_m_dst",
            make_shared_eval_poly(
                "P_AggBind_m_dst", std::move(aggregation_dst_multiplicity),
                context.domains.n));
        std::vector<FieldElement> aggregation_node_index(
            n_size, FieldElement::zero());
        for (std::size_t node = 0; node < context.local.num_nodes; ++node)
            aggregation_node_index[node] = FieldElement(node);
        add_public_poly(
            context,
            "P_AggBind_index_N",
            make_shared_eval_poly(
                "P_AggBind_index_N", std::move(aggregation_node_index),
                context.domains.n));
        add_public_poly(
            context,
            "P_Q_proj_valid",
            make_shared_eval_poly("P_Q_proj_valid", build_selector(max_in, in_size), context.domains.in));
        add_public_poly(
            context,
            "P_Q_d_valid",
            make_shared_eval_poly("P_Q_d_valid", build_selector(d_h, d_size), context.domains.d));
        add_public_poly(
            context,
            "P_Q_cat_valid",
            make_shared_eval_poly("P_Q_cat_valid", build_selector(d_cat, cat_size), context.domains.cat));
        add_public_poly(
            context,
            "P_Q_C_valid",
            make_shared_eval_poly("P_Q_C_valid", build_selector(context.local.num_classes, c_size), context.domains.c));
        std::vector<FieldElement> bias_index_c(c_size, FieldElement::zero());
        std::vector<FieldElement> bias_multiplicity_c(c_size, FieldElement::zero());
        for (std::size_t column = 0; column < context.local.num_classes; ++column)
        {
            bias_index_c[column] = FieldElement(column);
            bias_multiplicity_c[column] = FieldElement(context.local.num_nodes);
        }
        add_public_poly(
            context,
            "P_BiasIndex_C",
            make_shared_eval_poly(
                "P_BiasIndex_C", std::move(bias_index_c), context.domains.c));
        add_public_poly(
            context,
            "P_BiasMultiplicity_C",
            make_shared_eval_poly(
                "P_BiasMultiplicity_C", std::move(bias_multiplicity_c), context.domains.c));
        std::vector<FieldElement> bias_index_edge(edge_size, FieldElement::zero());
        const auto output_flat_count =
            context.local.num_nodes * context.local.num_classes;
        for (std::size_t index = 0; index < output_flat_count; ++index)
            bias_index_edge[index] = FieldElement(index % context.local.num_classes);
        add_public_poly(
            context,
            "P_BiasIndex_edge",
            make_shared_eval_poly(
                "P_BiasIndex_edge", std::move(bias_index_edge), context.domains.edge));
        constexpr std::size_t kCoefficientTailBaseObservationCount = 3;
        const auto bias_basis_chunks = coefficient_chunk_layout(
            context.local.num_classes + kCoefficientTailBaseObservationCount,
            edge_size);
        for (std::size_t head = 0;
             head < context.model.output_layer.heads.size(); ++head)
        {
            for (const auto &chunk : bias_basis_chunks)
            {
                std::vector<FieldElement> active(
                    edge_size, FieldElement::zero());
                std::vector<FieldElement> semantic(
                    edge_size, FieldElement::zero());
                std::vector<FieldElement> gap(
                    edge_size, FieldElement::zero());
                std::vector<FieldElement> tail(
                    edge_size, FieldElement::zero());
                for (std::size_t row = 0; row < chunk.length; ++row)
                {
                    const auto global_index = chunk.offset + row;
                    active[row] = FieldElement::one();
                    if (global_index < context.local.num_classes)
                        semantic[row] = FieldElement::one();
                    else
                        tail[row] = FieldElement::one();
                }
                for (const auto &[region, values] :
                     std::vector<std::pair<std::string,
                         std::vector<FieldElement>>>{
                         {"active", std::move(active)},
                         {"semantic", std::move(semantic)},
                         {"gap", std::move(gap)},
                         {"tail", std::move(tail)}})
                {
                    const auto label =
                        output_bias_basis_region_selector_label(
                            head, chunk.chunk_index, region);
                    add_public_poly(
                        context, label,
                        make_shared_eval_poly(
                            label, values, context.domains.edge));
                }
            }
        }
        if (lazy_large_fh_public_enabled(context))
        {
            add_public_poly(
                context,
                "P_Q_tbl_feat",
                make_shared_eval_poly("P_Q_tbl_feat", build_selector(context.dataset.num_nodes * context.dataset.num_features, fh_size), context.domains.fh));
            add_public_poly(
                context,
                "P_Row_feat_tbl",
                make_shared_eval_poly(
                    "P_Row_feat_tbl",
                    feature_table_row_indices(context.dataset.num_nodes, context.dataset.num_features, fh_size),
                    context.domains.fh));
            add_public_poly(
                context,
                "P_Col_feat_tbl",
                make_shared_eval_poly(
                    "P_Col_feat_tbl",
                    feature_table_col_indices(context.dataset.num_nodes, context.dataset.num_features, fh_size),
                    context.domains.fh));
            if (full_graph_feature_identity_enabled(context))
            {
                add_public_poly_alias_from_source_checked(context, "P_Q_qry_feat", "P_Q_tbl_feat");
                add_public_poly_alias_from_source_checked(context, "P_Row_feat_qry", "P_Row_feat_tbl");
                add_public_poly_alias_from_source_checked(context, "P_Col_feat_qry", "P_Col_feat_tbl");
                add_public_poly_alias_from_source_checked(context, "P_I_feat_qry", "P_Row_feat_tbl");
            }
            else
            {
                add_public_poly(
                    context,
                    "P_Q_qry_feat",
                    make_shared_eval_poly("P_Q_qry_feat", build_selector(context.local.num_nodes * context.local.num_features, fh_size), context.domains.fh));
                add_public_poly(
                    context,
                    "P_Row_feat_qry",
                    make_shared_eval_poly(
                        "P_Row_feat_qry",
                        feature_query_row_indices(context.local.num_nodes, context.local.num_features, fh_size),
                        context.domains.fh));
                add_public_poly(
                    context,
                    "P_Col_feat_qry",
                    make_shared_eval_poly(
                        "P_Col_feat_qry",
                        feature_query_col_indices(context.local.num_nodes, context.local.num_features, fh_size),
                        context.domains.fh));
                add_public_poly(
                    context,
                    "P_I_feat_qry",
                    make_shared_eval_poly(
                        "P_I_feat_qry",
                        feature_query_absolute_ids(context.local.absolute_ids, context.local.num_features, fh_size),
                        context.domains.fh));
            }
        }
        else
        {
            add_public_poly(
                context,
                "P_Q_tbl_feat",
                make_shared_eval_poly("P_Q_tbl_feat", build_selector(context.dataset.num_nodes * context.dataset.num_features, fh_size), context.domains.fh));
            add_public_poly(
                context,
                "P_Row_feat_tbl",
                make_shared_eval_poly(
                    "P_Row_feat_tbl",
                    feature_table_row_indices(context.dataset.num_nodes, context.dataset.num_features, fh_size),
                    context.domains.fh));
            add_public_poly(
                context,
                "P_Col_feat_tbl",
                make_shared_eval_poly(
                    "P_Col_feat_tbl",
                    feature_table_col_indices(context.dataset.num_nodes, context.dataset.num_features, fh_size),
                    context.domains.fh));
            if (full_graph_feature_identity_enabled(context))
            {
                add_public_poly_alias_from_source_checked(context, "P_Q_qry_feat", "P_Q_tbl_feat");
                add_public_poly_alias_from_source_checked(context, "P_Row_feat_qry", "P_Row_feat_tbl");
                add_public_poly_alias_from_source_checked(context, "P_Col_feat_qry", "P_Col_feat_tbl");
                add_public_poly_alias_from_source_checked(context, "P_I_feat_qry", "P_Row_feat_tbl");
            }
            else
            {
                add_public_poly(
                    context,
                    "P_Q_qry_feat",
                    make_shared_eval_poly("P_Q_qry_feat", build_selector(context.local.num_nodes * context.local.num_features, fh_size), context.domains.fh));
                add_public_poly(
                    context,
                    "P_Row_feat_qry",
                    make_shared_eval_poly(
                        "P_Row_feat_qry",
                        feature_query_row_indices(context.local.num_nodes, context.local.num_features, fh_size),
                        context.domains.fh));
                add_public_poly(
                    context,
                    "P_Col_feat_qry",
                    make_shared_eval_poly(
                        "P_Col_feat_qry",
                        feature_query_col_indices(context.local.num_nodes, context.local.num_features, fh_size),
                        context.domains.fh));
                add_public_poly(
                    context,
                    "P_I_feat_qry",
                    make_shared_eval_poly(
                        "P_I_feat_qry",
                        feature_query_absolute_ids(context.local.absolute_ids, context.local.num_features, fh_size),
                        context.domains.fh));
            }
        }
        add_public_poly(
            context,
            "P_Q_qry_src",
            make_shared_eval_poly("P_Q_qry_src", edge_valid, context.domains.edge));
        add_public_poly(
            context,
            "P_Q_qry_dst",
            make_shared_eval_poly("P_Q_qry_dst", edge_valid, context.domains.edge));
        add_public_poly(
            context,
            "P_Q_tbl_L",
            make_shared_eval_poly("P_Q_tbl_L", build_selector(context.tables.lrelu.size(), edge_size), context.domains.edge));
        add_public_poly(
            context,
            "P_Q_qry_L",
            make_shared_eval_poly("P_Q_qry_L", build_selector(context.local.edges.size(), edge_size), context.domains.edge));
        add_public_poly(
            context,
            "P_Q_tbl_R",
            make_shared_eval_poly("P_Q_tbl_R", build_selector(context.tables.range.size(), edge_size), context.domains.edge));
        add_public_poly(
            context,
            "P_Q_qry_R",
            make_shared_eval_poly("P_Q_qry_R", build_selector(context.local.edges.size(), edge_size), context.domains.edge));
        add_public_poly(
            context,
            "P_Q_tbl_exp",
            make_shared_eval_poly("P_Q_tbl_exp", build_selector(context.tables.exp.size(), edge_size), context.domains.edge));
        add_public_poly(
            context,
            "P_Q_qry_exp",
            make_shared_eval_poly("P_Q_qry_exp", build_selector(context.local.edges.size(), edge_size), context.domains.edge));
        add_public_poly(
            context,
            "P_Q_tbl_ELU",
            make_shared_eval_poly("P_Q_tbl_ELU", build_selector(context.tables.elu.size(), edge_size), context.domains.edge));
        add_public_poly(
            context,
            "P_Q_qry_ELU",
            make_shared_eval_poly("P_Q_qry_ELU", build_selector(context.local.num_nodes * d_h, edge_size), context.domains.edge));

        const auto &feature_matrix_for_h =
            !context.dataset.features.empty() ? context.dataset.features : context.local.features;
        if (lazy_large_fh_public_enabled(context))
        {
            add_static_commitment(
                context,
                "V_T_H",
                make_coeff_poly("V_T_H", algebra::flatten_matrix_coefficients(feature_matrix_for_h)));
            add_public_poly(
                context,
                "P_T_H",
                make_shared_eval_poly(
                    "P_T_H",
                    padded(algebra::flatten_matrix_coefficients(feature_matrix_for_h), fh_size),
                    context.domains.fh));
        }
        else
        {
            add_static_commitment(
                context,
                "V_T_H",
                make_coeff_poly("V_T_H", algebra::flatten_matrix_coefficients(feature_matrix_for_h)));
            add_public_poly(
                context,
                "P_T_H",
                make_shared_eval_poly(
                    "P_T_H",
                    padded(algebra::flatten_matrix_coefficients(feature_matrix_for_h), fh_size),
                    context.domains.fh));
        }

        if (full_graph_feature_identity_enabled(context))
        {
            cache_public_fh_coefficient_material(context, "P_T_H");
            cache_public_fh_coefficient_material(context, "P_Q_tbl_feat");
            cache_public_fh_coefficient_material(context, "P_Row_feat_tbl");
            cache_public_fh_coefficient_material(context, "P_Col_feat_tbl");
            rebind_public_fh_opening_material(context, "P_T_H", "P_T_H");
            rebind_public_fh_opening_material(context, "P_Q_tbl_feat", "P_Q_tbl_feat");
            rebind_public_fh_opening_material(context, "P_Q_qry_feat", "P_Q_tbl_feat");
            rebind_public_fh_opening_material(context, "P_Row_feat_tbl", "P_Row_feat_tbl");
            rebind_public_fh_opening_material(context, "P_Row_feat_qry", "P_Row_feat_tbl");
            rebind_public_fh_opening_material(context, "P_I_feat_qry", "P_Row_feat_tbl");
            rebind_public_fh_opening_material(context, "P_Col_feat_tbl", "P_Col_feat_tbl");
            rebind_public_fh_opening_material(context, "P_Col_feat_qry", "P_Col_feat_tbl");
            progress_note("context:public_fh_coset:P_Q_tbl_feat:boolean_quotient:start");
            context.public_fh_coset_materials[fh_selector_boolean_quotient_coset_key("P_Q_tbl_feat")] =
                build_fh_selector_boolean_quotient_coset_material(context, "P_Q_tbl_feat");
            progress_note("context:public_fh_coset:P_Q_tbl_feat:boolean_quotient:done");
        }

        std::vector<FieldElement> l_x(edge_size, FieldElement::zero());
        std::vector<FieldElement> l_y(edge_size, FieldElement::zero());
        for (std::size_t i = 0; i < context.tables.lrelu.size(); ++i)
        {
            l_x[i] = context.tables.lrelu[i].first;
            l_y[i] = context.tables.lrelu[i].second;
        }
        add_public_poly(context, "P_T_L_x", make_shared_eval_poly("P_T_L_x", l_x, context.domains.edge));
        add_public_poly(context, "P_T_L_y", make_shared_eval_poly("P_T_L_y", l_y, context.domains.edge));
        add_static_commitment(context, "V_T_L_x", make_eval_poly("V_T_L_x", l_x, context.domains.edge));
        add_static_commitment(context, "V_T_L_y", make_eval_poly("V_T_L_y", l_y, context.domains.edge));

        std::vector<FieldElement> elu_x(edge_size, FieldElement::zero());
        std::vector<FieldElement> elu_y(edge_size, FieldElement::zero());
        for (std::size_t i = 0; i < context.tables.elu.size(); ++i)
        {
            elu_x[i] = context.tables.elu[i].first;
            elu_y[i] = context.tables.elu[i].second;
        }
        add_public_poly(context, "P_T_ELU_x", make_shared_eval_poly("P_T_ELU_x", elu_x, context.domains.edge));
        add_public_poly(context, "P_T_ELU_y", make_shared_eval_poly("P_T_ELU_y", elu_y, context.domains.edge));
        add_static_commitment(context, "V_T_ELU_x", make_eval_poly("V_T_ELU_x", elu_x, context.domains.edge));
        add_static_commitment(context, "V_T_ELU_y", make_eval_poly("V_T_ELU_y", elu_y, context.domains.edge));

        std::vector<FieldElement> exp_x(edge_size, FieldElement::zero());
        std::vector<FieldElement> exp_y(edge_size, FieldElement::zero());
        for (std::size_t i = 0; i < context.tables.exp.size(); ++i)
        {
            exp_x[i] = context.tables.exp[i].first;
            exp_y[i] = context.tables.exp[i].second;
        }
        add_public_poly(context, "P_T_exp_x", make_shared_eval_poly("P_T_exp_x", exp_x, context.domains.edge));
        add_public_poly(context, "P_T_exp_y", make_shared_eval_poly("P_T_exp_y", exp_y, context.domains.edge));
        add_static_commitment(context, "V_T_exp_x", make_eval_poly("V_T_exp_x", exp_x, context.domains.edge));
        add_static_commitment(context, "V_T_exp_y", make_eval_poly("V_T_exp_y", exp_y, context.domains.edge));
        add_public_poly(
            context,
            "P_T_range",
            make_shared_eval_poly("P_T_range", padded(context.tables.range, edge_size), context.domains.edge));
        add_static_commitment(
            context,
            "V_T_range",
            make_eval_poly("V_T_range", padded(context.tables.range, edge_size), context.domains.edge));

        if (context.model.has_real_multihead)
        {
            // 模型参数在 context 阶段就固定成静态承诺，后面的 trace 只引用标签，不再重复构造权重对象。
            for (std::size_t head_index = 0; head_index < context.model.hidden_heads.size(); ++head_index)
            {
                const auto weight = quantize_model_matrix(
                    context.model.hidden_heads[head_index].seq_kernel_fp);
                const auto column_count = weight.empty() ? 0 : weight.front().size();
                for (std::size_t column = 0; column < column_count; ++column)
                {
                    std::vector<FieldElement> values(context.domains.in->size, FieldElement::zero());
                    for (std::size_t row = 0; row < weight.size(); ++row) values[row] = weight[row][column];
                    const auto label = hidden_weight_column_label(head_index, column);
                    add_static_commitment(context, label, make_eval_poly(label, values, context.domains.in));
                }
                add_static_commitment(
                    context,
                    hidden_src_label(head_index),
                    make_eval_poly(
                        hidden_src_label(head_index),
                        padded(quantize_model_vector(context.model.hidden_heads[head_index].attn_src_kernel_fp), context.domains.d->size),
                        context.domains.d));
                add_static_commitment(
                    context,
                    hidden_dst_label(head_index),
                    make_eval_poly(
                        hidden_dst_label(head_index),
                        padded(quantize_model_vector(context.model.hidden_heads[head_index].attn_dst_kernel_fp), context.domains.d->size),
                        context.domains.d));
            }
            for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index)
            {
                const auto &head = context.model.output_layer.heads[head_index];
                const auto weight = quantize_model_matrix(head.seq_kernel_fp);
                const auto column_count = weight.empty() ? 0 : weight.front().size();
                for (std::size_t column = 0; column < column_count; ++column)
                {
                    std::vector<FieldElement> values(context.domains.cat->size, FieldElement::zero());
                    for (std::size_t row = 0; row < weight.size(); ++row) values[row] = weight[row][column];
                    const auto label = output_weight_column_label(head_index, column);
                    add_static_commitment(context, label, make_eval_poly(label, values, context.domains.cat));
                }
                add_static_commitment(
                    context,
                    output_src_label(head_index),
                    make_eval_poly(
                        output_src_label(head_index),
                        padded(quantize_model_vector_at_scale(head.attn_src_kernel_fp, 1.0), context.domains.c->size),
                        context.domains.c));
                add_static_commitment(
                    context,
                    output_dst_label(head_index),
                    make_eval_poly(
                        output_dst_label(head_index),
                        padded(quantize_model_vector_at_scale(head.attn_dst_kernel_fp, 1.0), context.domains.c->size),
                        context.domains.c));
                add_static_commitment(
                    context,
                    output_bias_label(head_index),
                    make_eval_poly(
                        output_bias_label(head_index),
                        padded(
                            quantize_model_vector(head.output_bias_fp),
                            context.domains.c->size),
                        context.domains.c));
            }
        }
        end = Clock::now();
        if (metrics != nullptr)
        {
            metrics->load_static_ms += elapsed_ms(start, end);
        }
        progress_note("context:public_static:done");

        auto compiler_artifact = std::make_shared<ProtocolCompilerArtifact>(
            build_protocol_compiler_artifact(context));
        std::string compiler_artifact_reason;
        if (!validate_protocol_compiler_artifact(
                context, *compiler_artifact, &compiler_artifact_reason))
        {
            throw std::runtime_error(
                "instance ProtocolCompilerArtifact validation failed: "
                + compiler_artifact_reason);
        }
        context.compiler_artifact = std::move(compiler_artifact);
        context.validated_compiler_artifact_identity =
            context.compiler_artifact.get();

        if (full_graph_feature_identity_enabled(context))
        {
            context.dataset.features.clear();
            context.dataset.features.shrink_to_fit();
            context.dataset.features_fp.clear();
            context.dataset.features_fp.shrink_to_fit();
        }

        const bool cacheable_static_context =
            context.domains.fh != nullptr && context.domains.edge != nullptr && context.domains.fh->size < (1ULL << 24) && context.domains.edge->size < (1ULL << 22);
        if (allow_static_context_cache && cacheable_static_context)
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            cache[cache_key] = context;
        }
        return context;
    }

    // 消费 trace 生成最终 proof，对外保证 commitment 顺序、block 顺序与 opening 结构稳定。
    Proof prove(const ProtocolContext &context, const TraceArtifacts &trace, RunMetrics *metrics)
    {
        const auto &route2 = util::route2_options();
        if (!context.model.has_real_multihead)
        {
            throw std::runtime_error(
                "formal prover rejects the removed single-output compatibility protocol");
        }
        // prove 入口只消费 trace 和固定挑战顺序，不在这里改写 proof schema 或 block 顺序。
        if (trace.commitment_order != dynamic_commitment_labels(context))
        {
            throw std::runtime_error("trace commitment order does not match the protocol commitment order");
        }
        if (context.model.has_real_multihead)
        {
            auto stage_start = Clock::now();
            progress_note("prove:quotient:start");
            // 先重放动态承诺之前的 transcript，再在这套固定挑战上构造各工作域 quotient。
            const auto pre_quotient_challenges = replay_challenges(context, trace.commitments, {}, &trace.public_metadata);
            const auto check_fh_domain_constraints = [&]()
            {
                const auto fh_value = [&](const std::string &name, std::size_t row)
                    -> FieldElement
                {
                    if (name == "P_H"
                        && lazy_full_feature_lookup_trace_enabled(context))
                        return context.public_polynomials.at("P_T_H")
                            .values().at(row);
                    if (const auto it = trace.polynomials.find(name);
                        it != trace.polynomials.end())
                        return it->second.values().at(row);
                    if (const auto it = context.public_polynomials.find(name);
                        it != context.public_polynomials.end())
                        return it->second.values().at(row);
                    throw std::runtime_error(
                        "FH domain self-check is missing polynomial: " + name);
                };
                const auto one = FieldElement::one();
                const auto eta = pre_quotient_challenges.at("eta_feat");
                const auto eta_squared = eta * eta;
                const auto beta = pre_quotient_challenges.at("beta_feat");
                for (std::size_t row = 0; row < context.domains.fh->size; ++row)
                {
                    const auto next = (row + 1) % context.domains.fh->size;
                    const auto q_table = fh_value("P_Q_tbl_feat", row);
                    const auto q_query = fh_value("P_Q_qry_feat", row);
                    const auto table = fh_value("P_Table_feat", row);
                    const auto query = fh_value("P_Query_feat", row);
                    const auto multiplicity = fh_value("P_m_feat", row);
                    const auto j_table = fh_value("P_J_tbl_feat", row);
                    const auto j_query = fh_value("P_J_qry_feat", row);
                    const auto accumulator = fh_value("P_R_feat", row);
                    const std::vector<std::pair<std::string, FieldElement>> terms = {
                        {"first_accumulator", row == 0
                            ? accumulator : FieldElement::zero()},
                        {"recurrence", row + 1 < context.domains.fh->size
                            ? fh_value("P_R_feat", next) - accumulator
                                - q_table * multiplicity * j_table
                                + q_query * j_query
                            : FieldElement::zero()},
                        {"last_accumulator", row + 1 == context.domains.fh->size
                            ? accumulator : FieldElement::zero()},
                        {"table_inverse", q_table
                            * ((table + beta) * j_table - one)},
                        {"table_padding", (one - q_table) * j_table},
                        {"query_inverse", q_query
                            * ((query + beta) * j_query - one)},
                        {"query_padding", (one - q_query) * j_query},
                        {"table_encoding", table
                            - (fh_value("P_Row_feat_tbl", row)
                                + eta * fh_value("P_Col_feat_tbl", row)
                                + eta_squared * fh_value("P_T_H", row))},
                        {"query_encoding", query
                            - (fh_value("P_I_feat_qry", row)
                                + eta * fh_value("P_Col_feat_qry", row)
                                + eta_squared * fh_value("P_H", row))},
                    };
                    for (const auto &[term_name, term] : terms)
                    {
                        if (!term.is_zero())
                        {
                            throw std::runtime_error(
                                "FH domain constraint violation: row="
                                + std::to_string(row) + " term=" + term_name
                                + " value=" + term.to_string());
                        }
                    }
                }
            };
            if (std::getenv("GATZK_DEBUG_QUOTIENT_SELF_CHECK") != nullptr)
            {
                check_fh_domain_constraints();
                std::size_t max_edge_blinder = 0;
                std::vector<std::string> max_edge_blinder_labels;
                for (const auto &[name, polynomial] : trace.polynomials)
                {
                    if (polynomial.domain != context.domains.edge)
                        continue;
                    const auto size = polynomial.zero_domain_blinder.size();
                    if (size > max_edge_blinder)
                    {
                        max_edge_blinder = size;
                        max_edge_blinder_labels = {name};
                    }
                    else if (size == max_edge_blinder && size != 0)
                    {
                        max_edge_blinder_labels.push_back(name);
                    }
                }
                std::cerr << "[quotient-debug] edge max_blinder="
                          << max_edge_blinder << " labels=";
                for (std::size_t i = 0;
                     i < std::min<std::size_t>(
                         max_edge_blinder_labels.size(), 8); ++i)
                {
                    if (i != 0) std::cerr << ',';
                    std::cerr << max_edge_blinder_labels[i];
                }
                std::cerr << '\n';

                std::vector<std::size_t> sample_rows = {
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                    context.local.edges.size(),
                    context.local.num_nodes * context.local.num_classes,
                    context.tables.range.size(),
                    context.domains.edge->size - 2,
                    context.domains.edge->size - 1,
                };
                std::sort(sample_rows.begin(), sample_rows.end());
                sample_rows.erase(
                    std::remove_if(
                        sample_rows.begin(), sample_rows.end(),
                        [&](std::size_t row)
                        { return row >= context.domains.edge->size; }),
                    sample_rows.end());
                sample_rows.erase(
                    std::unique(sample_rows.begin(), sample_rows.end()),
                    sample_rows.end());
                std::vector<FieldElement> sample_points;
                sample_points.reserve(sample_rows.size() * 2);
                for (const auto row : sample_rows)
                {
                    const auto point = context.domains.edge->point_at(row);
                    sample_points.push_back(point);
                    sample_points.push_back(point * context.domains.edge->omega);
                }
                EvaluationMemoization edge_domain_memo(
                    context, trace, pre_quotient_challenges);
                edge_domain_memo.precompute_named(
                    quotient_dependency_labels(context, "t_edge"),
                    sample_points);
                for (const auto row : sample_rows)
                {
                    const auto point = context.domains.edge->point_at(row);
                    std::size_t first_nonzero_term =
                        std::numeric_limits<std::size_t>::max();
                    FieldElement first_nonzero_value = FieldElement::zero();
                    const auto numerator = evaluate_t_edge(
                        context,
                        pre_quotient_challenges,
                        trace.external_evaluations,
                        [&](const std::string &name, const FieldElement &at)
                        {
                            return edge_domain_memo.eval_named(name, at);
                        },
                        point,
                        true,
                        &first_nonzero_term,
                        &first_nonzero_value);
                    if (!numerator.is_zero())
                    {
                        throw std::runtime_error(
                            "edge domain constraint violation: row="
                            + std::to_string(row)
                            + " first_term="
                            + std::to_string(first_nonzero_term)
                            + " first_value="
                            + first_nonzero_value.to_string()
                            + " numerator=" + numerator.to_string());
                    }
                }

                std::vector<FieldElement> d_points;
                d_points.reserve(context.domains.d->size * 2);
                for (std::size_t row = 0; row < context.domains.d->size; ++row)
                {
                    const auto point = context.domains.d->point_at(row);
                    d_points.push_back(point);
                    d_points.push_back(point * context.domains.d->omega);
                }
                EvaluationMemoization d_domain_memo(
                    context, trace, pre_quotient_challenges);
                d_domain_memo.precompute_named(
                    quotient_dependency_labels(context, "t_d_h"), d_points);
                for (std::size_t row = 0; row < context.domains.d->size; ++row)
                {
                    const auto point = context.domains.d->point_at(row);
                    std::size_t first_nonzero_term =
                        std::numeric_limits<std::size_t>::max();
                    FieldElement first_nonzero_value = FieldElement::zero();
                    const auto numerator = evaluate_t_d(
                        context,
                        pre_quotient_challenges,
                        trace.external_evaluations,
                        [&](const std::string &name, const FieldElement &at)
                        {
                            return d_domain_memo.eval_named(name, at);
                        },
                        point,
                        true,
                        &first_nonzero_term,
                        &first_nonzero_value);
                    if (!numerator.is_zero())
                    {
                        throw std::runtime_error(
                            "d_h domain constraint violation: row="
                            + std::to_string(row)
                            + " first_term="
                            + std::to_string(first_nonzero_term)
                            + " first_value="
                            + first_nonzero_value.to_string()
                            + " numerator=" + numerator.to_string());
                    }
                }
            }
            const auto quotient_cache_key = quotient_build_cache_key(context, trace);
            std::shared_ptr<const ProofEvaluationBackendRegistry> backend_registry;
            if (route2.fft_backend_upgrade)
            {
                backend_registry = shared_proof_backend_registry_cache().get_or_build(
                    quotient_cache_key + "|backend_registry",
                    [&]()
                    {
                        return std::make_shared<ProofEvaluationBackendRegistry>(context, trace);
                    });
            }
            auto proof_domain_weight_cache = shared_proof_domain_weight_cache();
            const std::vector<std::string> quotient_labels = {"t_FH", "t_edge", "t_in", "t_d_h", "t_cat", "t_C", "t_N"};
            append_note(metrics, "quotient_cache=miss");
            auto named_quotient_polynomials = materialize_quotient_polynomials(
                context,
                trace,
                pre_quotient_challenges,
                quotient_labels,
                metrics);
            named_quotient_polynomials = blind_quotient_polynomials(
                context,
                std::move(named_quotient_polynomials));
            progress_note("prove:quotient:commit_batch:start");
            crypto::CommitBatchProfile quotient_commit_profile;
            auto quotient_commitments = batch_quotient_commitments(
                std::move(named_quotient_polynomials),
                context.kzg,
                metrics != nullptr ? &quotient_commit_profile : nullptr);
            progress_note("prove:quotient:commit_batch:done");
            if (metrics != nullptr)
            {
                metrics->quotient_commitment_ms +=
                    quotient_commit_profile.msm_ms;
            }
            // Quotient polynomials and commitments are proof-specific and must not be
            // reused by a later fresh-proof measurement.
            const auto challenges = replay_challenges(context, trace.commitments, quotient_commitments, &trace.public_metadata);
            if (std::getenv("GATZK_DEBUG_QUOTIENT_SELF_CHECK") != nullptr)
            {
                const auto z = challenges.at("z_FH");
                EvaluationMemoization self_check_memo(
                    context,
                    trace,
                    challenges,
                    backend_registry,
                    proof_domain_weight_cache,
                    nullptr);
                self_check_memo.precompute_named(
                    quotient_dependency_labels(context, "t_FH"),
                    {z, z * context.domains.fh->omega});
                const auto expected = evaluate_t_fh(
                    context,
                    challenges,
                    [&](const std::string &name, const FieldElement &point)
                    {
                        return self_check_memo.eval_named(name, point);
                    },
                    z);
                const auto quotient_it = quotient_commitments.find("t_FH");
                const auto mask_it = quotient_commitments.find(
                    quotient_mask_label("t_FH"));
                if (quotient_it == quotient_commitments.end()
                    || mask_it == quotient_commitments.end()
                    || quotient_it->second.opening_polynomial == nullptr
                    || mask_it->second.opening_polynomial == nullptr)
                {
                    throw std::runtime_error(
                        "FH quotient self-check is missing opening material");
                }
                const auto opened =
                    quotient_it->second.opening_polynomial->evaluate(z);
                const auto mask =
                    mask_it->second.opening_polynomial->evaluate(z);
                const auto expected_masked = expected
                    + context.domains.fh->zero_polynomial_eval(z) * mask;
                if (opened != expected_masked)
                {
                    throw std::runtime_error(
                        "FH quotient prover self-check mismatch: opened="
                        + opened.to_string() + " expected=" + expected.to_string()
                        + " mask=" + mask.to_string()
                        + " expected_masked=" + expected_masked.to_string());
                }
                progress_note("prove:quotient:t_FH:self_check:ok");
            }
            if (metrics != nullptr)
            {
                metrics->quotient_build_ms += elapsed_ms(stage_start, Clock::now());
            }
            progress_note("prove:quotient:done");

            // 组装 proof 时必须严格按 schema 的 label 顺序回填，
            // 否则 serializer / parser 与 verifier 会同时失配。
            Proof proof;
            proof.public_metadata = trace.public_metadata;
            proof.block_order = proof_block_order();
            proof.challenges = challenges;
            proof.route_references =
                route_opening_references(context, proof.challenges);
            if (trace.pi_bind == nullptr)
            {
                throw std::runtime_error("trace is missing Pi_bind proof material");
            }
            proof.pi_bind = *trace.pi_bind;
            for (const auto &label : dynamic_commitment_labels(context))
            {
                proof.dynamic_commitments.push_back({label, trace.commitments.at(label)});
            }
            for (const auto &label : quotient_commitment_labels(context))
            {
                proof.quotient_commitments.push_back({label, quotient_commitments.at(label)});
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
            progress_note("prove:domain_opening:start");
            struct BundleBuildResult
            {
                std::string bundle_name;
                DomainOpeningBundle bundle;
                double gather_ms = 0.0;
                double total_ms = 0.0;
                RunMetrics local_metrics;
            };
            auto build_bundle = [&](const BundleSpec &spec) -> BundleBuildResult
            {
                const auto progress_prefix = "prove:domain_opening:" + spec.bundle_name;
                progress_note(progress_prefix + ":start");
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
                labels.push_back(quotient_mask_label(spec.quotient_name));
                const std::vector<FieldElement> points = {
                    challenges.at(spec.z_name),
                    challenges.at(spec.z_name) * spec.domain->omega,
                };
                double gather_ms = 0.0;
                const auto local_start = Clock::now();
                auto bundle = make_bundle(
                    context,
                    trace,
                    opening_memo,
                    quotient_commitments,
                    labels,
                    points,
                    &local_metrics,
                    &gather_ms,
                    progress_prefix);
                if (spec.bundle_name == "FH")
                {
                    const auto nested_ms =
                        local_metrics.fh_eval_prep_ms + local_metrics.fh_interpolation_ms + local_metrics.fh_public_eval_reuse_ms + local_metrics.fh_quotient_assembly_ms;
                    local_metrics.fh_open_gather_ms =
                        std::max(0.0, local_metrics.fh_open_gather_ms - nested_ms);
                }
                progress_note(progress_prefix + ":done");
                return {
                    spec.bundle_name,
                    std::move(bundle),
                    gather_ms,
                    elapsed_ms(local_start, Clock::now()),
                    std::move(local_metrics),
                };
            };
            std::vector<BundleBuildResult> bundle_results;
            bundle_results.reserve(bundle_specs.size());
            // 并行构造 opening bundle 之前先预热 barycentric 权重和 FH 公开值缓存。
            // 大域权重向量体量极大，若在多个 bundle 中同时构造，会明显放大缺页和带宽争抢。
            if (std::thread::hardware_concurrency() > 1)
            {
                // 只预热最重的 FH / edge 两个域，并且只针对正式 opening 用到的挑战点。
                for (const auto &spec : bundle_specs)
                {
                    if (spec.bundle_name == "FH" || spec.bundle_name == "edge")
                    {
                        const auto z = challenges.at(spec.z_name);
                        proof_domain_weight_cache->get(spec.domain, z);
                        proof_domain_weight_cache->get(spec.domain, z * spec.domain->omega);
                    }
                }
                std::vector<std::future<BundleBuildResult>> futures;
                futures.reserve(bundle_specs.size());
                for (const auto &spec : bundle_specs)
                {
                    futures.push_back(std::async(std::launch::async, build_bundle, spec));
                }
                for (auto &future : futures)
                {
                    bundle_results.push_back(future.get());
                }
            }
            else
            {
                for (const auto &spec : bundle_specs)
                {
                    bundle_results.push_back(build_bundle(spec));
                }
            }
            for (const auto &spec : bundle_specs)
            {
                const auto it = std::find_if(
                    bundle_results.begin(),
                    bundle_results.end(),
                    [&](const BundleBuildResult &result)
                    { return result.bundle_name == spec.bundle_name; });
                if (it == bundle_results.end())
                {
                    throw std::runtime_error("missing bundle build result for " + spec.bundle_name);
                }
                proof.domain_openings.push_back({it->bundle_name, it->bundle});
                if (metrics != nullptr)
                {
                    metrics->domain_eval_gather_ms += it->gather_ms;
                    metrics->domain_open_denominator_inverse_ms += it->local_metrics.domain_open_denominator_inverse_ms;
                    metrics->domain_open_denominator_inverse_reuse_ms +=
                        it->local_metrics.domain_open_denominator_inverse_reuse_ms;
                    metrics->domain_open_denominator_inverse_reuse_hits +=
                        it->local_metrics.domain_open_denominator_inverse_reuse_hits;
                    metrics->domain_open_denominator_inverse_cache_misses +=
                        it->local_metrics.domain_open_denominator_inverse_cache_misses;
                    metrics->domain_open_point_parallel_batches +=
                        it->local_metrics.domain_open_point_parallel_batches;
                    metrics->domain_open_point_parallel_workers +=
                        it->local_metrics.domain_open_point_parallel_workers;
                    metrics->domain_open_point_msm_worker_cap +=
                        it->local_metrics.domain_open_point_msm_worker_cap;
                    if (auto *metric = domain_open_metric(metrics, it->bundle_name); metric != nullptr)
                    {
                        *metric += it->total_ms;
                    }
                    if (it->bundle_name == "FH")
                    {
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
            if (metrics != nullptr)
            {
                metrics->domain_opening_ms += elapsed_ms(stage_start, Clock::now());
            }
            progress_note("prove:domain_opening:done");

            // Eval_bind owns every Pi_bind claim value.  The same claims remain
            // inside the in-memory domain bundle solely so Open_dom can fold one
            // global point group; the wire codec omits them from Eval_dom.
            for (std::size_t head_index = 0;
                 head_index < context.model.output_layer.heads.size();
                 ++head_index)
            {
                const auto relation_id =
                    "bind:output:" + std::to_string(head_index);
                auto subproof_it = std::find_if(
                    proof.pi_bind.subproofs.begin(), proof.pi_bind.subproofs.end(),
                    [&](const BindingSubproof &subproof)
                    { return subproof.relation_meta.relation_id == relation_id; });
                if (subproof_it == proof.pi_bind.subproofs.end())
                    throw std::runtime_error("missing output Eval_bind owner");
                for (const auto *expected_bundle_name : {"C", "edge"})
                {
                    const auto bundle_it = std::find_if(
                        proof.domain_openings.begin(),
                        proof.domain_openings.end(),
                        [&](const auto &entry)
                        { return entry.first == expected_bundle_name; });
                    if (bundle_it == proof.domain_openings.end())
                        throw std::runtime_error(
                            "missing Pi_bind global opening bundle: "
                            + std::string(expected_bundle_name));
                    const auto &bundle_name = bundle_it->first;
                    const auto &bundle = bundle_it->second;
                    const bool is_c = bundle_name == "C";
                    std::vector<std::string> owned_labels;
                    if (is_c)
                    {
                        owned_labels = {
                            model_proof_local_label(output_bias_label(head_index)),
                            output_bias_view_label(head_index, "Table_can"),
                            output_bias_view_label(head_index, "m_can"),
                            output_bias_view_label(head_index, "J_can"),
                            output_bias_view_label(head_index, "R_can"),
                        };
                    }
                    else
                    {
                        owned_labels = {
                            output_bias_view_label(head_index, "Query"),
                            output_bias_view_label(head_index, "J_qry"),
                            output_bias_view_label(head_index, "R_qry"),
                        };
                        if (subproof_it->relation_meta
                                .coefficient_basis_relations.size() != 1)
                            throw std::runtime_error(
                                "missing output coefficient basis Eval_bind metadata");
                        const auto &basis = subproof_it->relation_meta
                            .coefficient_basis_relations.front();
                        for (const auto &chunk : basis.chunks)
                        {
                            owned_labels.push_back(chunk.object_id);
                            owned_labels.push_back(chunk.accumulator_object);
                        }
                    }
                    for (const auto &label : owned_labels)
                    {
                        const auto value_it = std::find_if(
                            bundle.values.begin(), bundle.values.end(),
                            [&](const auto &entry) { return entry.first == label; });
                        if (value_it == bundle.values.end())
                            throw std::runtime_error(
                                "Pi_bind domain value missing from global opening: " + label);
                        subproof_it->evaluations.push_back(BindingEvaluation{
                            .domain = bundle_name,
                            .object_id = label,
                            .values = value_it->second,
                            .kind = label.ends_with("_BiasView_m_can")
                                ? PayloadValueKind::PublicValue
                                : PayloadValueKind::MaskedPrivateValue,
                        });
                    }
                    for (const auto &point : bundle.points)
                    {
                        const auto group_id =
                            "Open_dom:" + bundle_name + ":" + point.to_string();
                        if (std::none_of(
                                subproof_it->opening_references.begin(),
                                subproof_it->opening_references.end(),
                                [&](const MetadataReference &reference)
                                { return reference.object_id == group_id; }))
                            subproof_it->opening_references.push_back({group_id});
                    }
                }
            }
            const auto n_bundle_it = std::find_if(
                proof.domain_openings.begin(), proof.domain_openings.end(),
                [](const auto &entry) { return entry.first == "N"; });
            if (n_bundle_it == proof.domain_openings.end())
                throw std::runtime_error(
                    "missing canonical matrix N-domain opening bundle");
            for (auto &subproof : proof.pi_bind.subproofs)
            {
                if (subproof.relation_meta.canonical_matrices.empty())
                    continue;
                for (const auto &matrix :
                     subproof.relation_meta.canonical_matrices)
                {
                    auto owned_objects = matrix.column_objects;
                    owned_objects.push_back(matrix.compressed_object);
                    for (const auto &column : owned_objects)
                    {
                        const auto value_it = std::find_if(
                            n_bundle_it->second.values.begin(),
                            n_bundle_it->second.values.end(),
                            [&](const auto &entry)
                            { return entry.first == column; });
                        if (value_it == n_bundle_it->second.values.end())
                            throw std::runtime_error(
                                "canonical matrix value missing from N opening: "
                                + column);
                        subproof.evaluations.push_back(BindingEvaluation{
                            .domain = "N",
                            .object_id = column,
                            .values = value_it->second,
                            .kind = PayloadValueKind::MaskedPrivateValue,
                        });
                    }
                }
                for (const auto &point : n_bundle_it->second.points)
                {
                    const auto group_id =
                        "Open_dom:N:" + point.to_string();
                    if (std::none_of(
                            subproof.opening_references.begin(),
                            subproof.opening_references.end(),
                            [&](const MetadataReference &reference)
                            { return reference.object_id == group_id; }))
                        subproof.opening_references.push_back({group_id});
                }
            }
            const auto append_binding_domain_values = [&]
            (
                BindingSubproof &subproof,
                const std::string &bundle_name,
                const std::vector<std::string> &objects)
            {
                const auto bundle_it = std::find_if(
                    proof.domain_openings.begin(),
                    proof.domain_openings.end(),
                    [&](const auto &entry)
                    { return entry.first == bundle_name; });
                if (bundle_it == proof.domain_openings.end())
                    throw std::runtime_error(
                        "aggregation binding opening bundle missing: "
                        + bundle_name);
                for (const auto &object : objects)
                {
                    const auto value_it = std::find_if(
                        bundle_it->second.values.begin(),
                        bundle_it->second.values.end(),
                        [&](const auto &entry)
                        { return entry.first == object; });
                    if (value_it == bundle_it->second.values.end())
                        throw std::runtime_error(
                            "aggregation binding domain value missing: "
                            + object);
                    subproof.evaluations.push_back(BindingEvaluation{
                        .domain = bundle_name,
                        .object_id = object,
                        .values = value_it->second,
                        .kind = PayloadValueKind::MaskedPrivateValue,
                    });
                }
                for (const auto &point : bundle_it->second.points)
                {
                    const auto group_id = "Open_dom:" + bundle_name + ":"
                        + point.to_string();
                    if (std::none_of(
                            subproof.opening_references.begin(),
                            subproof.opening_references.end(),
                            [&](const MetadataReference &reference)
                            { return reference.object_id == group_id; }))
                        subproof.opening_references.push_back({group_id});
                }
            };
            for (auto &subproof : proof.pi_bind.subproofs)
            {
                for (const auto &relation :
                     subproof.relation_meta.aggregation_relations)
                {
                    std::vector<std::string> node_objects = {
                        relation.source_node_object,
                        relation.target_node_object,
                    };
                    std::vector<std::string> edge_objects = {
                        relation.source_edge_object,
                        relation.target_edge_object,
                        relation.product_object,
                        relation.psq_object,
                    };
                    for (const auto *route : {
                             &relation.source_route,
                             &relation.target_route})
                    {
                        node_objects.push_back(route->table_object);
                        node_objects.push_back(route->node_inverse_object);
                        node_objects.push_back(
                            route->node_accumulator_object);
                        edge_objects.push_back(route->query_object);
                        edge_objects.push_back(route->edge_inverse_object);
                        edge_objects.push_back(
                            route->edge_accumulator_object);
                    }
                    append_binding_domain_values(
                        subproof, relation.node_domain, node_objects);
                    append_binding_domain_values(
                        subproof, relation.edge_domain, edge_objects);
                }
            }

            const auto point_group_registry_start = Clock::now();
            StandardPointGroupRegistry opening_registry;
            for (const auto& spec : bundle_specs)
            {
                const auto bundle_it = std::find_if(
                    proof.domain_openings.begin(),
                    proof.domain_openings.end(),
                    [&](const auto& entry)
                    { return entry.first == spec.bundle_name; });
                if (bundle_it == proof.domain_openings.end())
                {
                    throw std::runtime_error(
                        "missing domain bundle while building global point groups: "
                        + spec.bundle_name);
                }
                auto labels = domain_opening_labels(
                    context, spec.trace_domain_name);
                labels.push_back(spec.quotient_name);
                labels.push_back(quotient_mask_label(spec.quotient_name));
                add_domain_bundle_claims(
                    opening_registry,
                    spec.bundle_name,
                    bundle_it->second,
                    collect_commitments(trace, quotient_commitments, labels));
            }

            attach_public_output_statement(context, trace, &proof);
            const auto ext_specs = multihead_external_specs(context);
            std::vector<std::pair<Commitment, FieldElement>> external_commitments;
            std::vector<FieldElement> external_points;
            stage_start = Clock::now();
            progress_note("prove:external_opening:start");
            for (const auto &spec : ext_specs)
            {
                const auto value = trace.external_evaluations.at(spec.proof_name);
                proof.external_evaluations.push_back({spec.proof_name, value});
                proof.external_evaluation_kinds.push_back(payload_kind_for_label(context, spec.label));
                external_commitments.push_back({
                    spec.static_commitment ? context.static_commitments.at(spec.label) : trace.commitments.at(spec.label),
                    value});
                external_points.push_back(challenges.at(spec.challenge_name));
            }
            for (const auto &subproof : proof.pi_bind.subproofs)
            {
                for (const auto &relation :
                     subproof.relation_meta.route_terminal_relations)
                {
                    const auto evaluation_id =
                        relation.shifted_accumulator_object + ":u_rel";
                    const auto evaluation = std::find_if(
                        subproof.evaluations.begin(),
                        subproof.evaluations.end(),
                        [&](const BindingEvaluation &candidate)
                        { return candidate.object_id == evaluation_id; });
                    if (evaluation == subproof.evaluations.end()
                        || evaluation->values.size() != 1
                        || evaluation->kind
                            != PayloadValueKind::MaskedPrivateValue)
                        throw std::runtime_error(
                            "missing route terminal shift value: "
                            + relation.relation_id);
                    const auto y_rel = challenges.at(
                        relation.challenge_label);
                    external_commitments.push_back({
                        trace.commitments.at(
                            relation.shifted_accumulator_object),
                        evaluation->values.front()});
                    external_points.push_back(y_rel);
                    external_commitments.push_back({
                        trace.commitments.at(
                            relation.edge_accumulator_object),
                        evaluation->values.front()});
                    external_points.push_back(
                        y_rel + relation.terminal_delta);
                    external_commitments.push_back({
                        trace.commitments.at(
                            relation.relation_virtual_object),
                        FieldElement::zero()});
                    external_points.push_back(
                        relation.node_terminal_point);
                }
                for (const auto &relation :
                     subproof.relation_meta.aggregation_relations)
                {
                    for (const auto *route : {
                             &relation.source_route,
                             &relation.target_route})
                    {
                        const auto evaluation_id =
                            route->shifted_accumulator_object + ":u_rel";
                        const auto evaluation = std::find_if(
                            subproof.evaluations.begin(),
                            subproof.evaluations.end(),
                            [&](const BindingEvaluation &candidate)
                            { return candidate.object_id == evaluation_id; });
                        if (evaluation == subproof.evaluations.end()
                            || evaluation->values.size() != 1
                            || evaluation->kind
                                != PayloadValueKind::MaskedPrivateValue)
                            throw std::runtime_error(
                                "missing aggregation terminal shift value: "
                                + route->relation_id);
                        const auto y_rel = challenges.at(
                            route->terminal_challenge_label);
                        external_commitments.push_back({
                            trace.commitments.at(
                                route->shifted_accumulator_object),
                            evaluation->values.front()});
                        external_points.push_back(y_rel);
                        external_commitments.push_back({
                            trace.commitments.at(
                                route->edge_accumulator_object),
                            evaluation->values.front()});
                        external_points.push_back(
                            y_rel + route->terminal_delta);
                        external_commitments.push_back({
                            trace.commitments.at(
                                route->relation_virtual_object),
                            FieldElement::zero()});
                        external_points.push_back(
                            route->node_terminal_point);
                    }
                }
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
                    throw std::runtime_error("missing output bias shift proof");
                const auto *relation_ptr = find_unique_shift_relation_by_kind(
                    subproof_it->relation_meta, "output-bias-terminal");
                if (relation_ptr == nullptr)
                    throw std::runtime_error(
                        "output bias shift proof relation is not unique");
                const auto &relation = *relation_ptr;
                if (relation.configured_degrees.empty())
                {
                    const auto y_public = challenges.at(relation.challenge_label);
                    external_commitments.push_back({
                        trace.commitments.at(output_bias_view_label(
                            head_index, "terminal_relation_virtual")),
                        FieldElement::zero()});
                    external_points.push_back(relation.canonical_points.front());
                    const auto public_m_id = output_bias_view_label(
                        head_index, "m_can") + ":public_bind";
                    const auto public_m_it = std::find_if(
                        subproof_it->evaluations.begin(),
                        subproof_it->evaluations.end(),
                        [&](const BindingEvaluation &evaluation)
                        { return evaluation.object_id == public_m_id; });
                    if (public_m_it == subproof_it->evaluations.end()
                        || public_m_it->values.size() != 1
                        || public_m_it->kind != PayloadValueKind::PublicValue)
                        throw std::runtime_error(
                            "missing same-point public bias multiplicity binding");
                    external_commitments.push_back({
                        trace.commitments.at(output_bias_view_label(
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
                    || u_it->values.size() != 1)
                    throw std::runtime_error("missing output bias shift check value");
                const auto y_rel = challenges.at(relation.challenge_label);
                external_commitments.push_back(
                    {trace.commitments.at(shifted_label), u_it->values.front()});
                external_points.push_back(y_rel);
                const auto virtual_prefix =
                    output_head_prefix(head_index) + "_BiasView_";
                external_commitments.push_back(
                    {trace.commitments.at(virtual_prefix + "F2"), u_it->values.front()});
                external_points.push_back(y_rel + relation.deltas.at(1));
                external_commitments.push_back({
                    trace.commitments.at(output_bias_view_label(
                        head_index, "terminal_relation_virtual")),
                    FieldElement::zero()});
                external_points.push_back(relation.canonical_points.at(0));

                const auto public_m_id =
                    output_bias_view_label(head_index, "m_can") + ":public_bind";
                const auto public_m_it = std::find_if(
                    subproof_it->evaluations.begin(), subproof_it->evaluations.end(),
                    [&](const BindingEvaluation &evaluation)
                    { return evaluation.object_id == public_m_id; });
                if (public_m_it == subproof_it->evaluations.end()
                    || public_m_it->values.size() != 1
                    || public_m_it->kind != PayloadValueKind::PublicValue)
                    throw std::runtime_error("missing public bias multiplicity binding");
                external_commitments.push_back({
                    trace.commitments.at(output_bias_view_label(head_index, "m_can")),
                    public_m_it->values.front()});
                external_points.push_back(y_rel);
            }
            for (std::size_t head_index = 0;
                 head_index < context.model.output_layer.heads.size();
                 ++head_index)
            {
                const auto relation_id =
                    "bind:output:" + std::to_string(head_index);
                const auto subproof_it = std::find_if(
                    proof.pi_bind.subproofs.begin(),
                    proof.pi_bind.subproofs.end(),
                    [&](const BindingSubproof &subproof)
                    { return subproof.relation_meta.relation_id == relation_id; });
                if (subproof_it == proof.pi_bind.subproofs.end()
                    || subproof_it->relation_meta
                           .coefficient_basis_relations.size() != 1)
                    throw std::runtime_error(
                        "missing output coefficient basis external proof");
                const auto &basis = subproof_it->relation_meta
                    .coefficient_basis_relations.front();
                const auto gamma = challenges.at(
                    basis.gamma_challenge_label);
                const auto y_rel = challenges.at(
                    basis.relation_challenge_label);
                for (const auto &chunk : basis.chunks)
                {
                    const auto u_id = chunk.shifted_accumulator_object
                        + ":u_rel";
                    const auto u_it = std::find_if(
                        subproof_it->evaluations.begin(),
                        subproof_it->evaluations.end(),
                        [&](const BindingEvaluation &evaluation)
                        { return evaluation.object_id == u_id; });
                    if (u_it == subproof_it->evaluations.end()
                        || u_it->values.size() != 1
                        || u_it->kind
                            != PayloadValueKind::MaskedPrivateValue)
                        throw std::runtime_error(
                            "missing output coefficient chunk shift value");
                    external_commitments.push_back({
                        trace.commitments.at(
                            chunk.shifted_accumulator_object),
                        u_it->values.front()});
                    external_points.push_back(y_rel);
                    external_commitments.push_back({
                        trace.commitments.at(chunk.accumulator_object),
                        u_it->values.front()});
                    external_points.push_back(
                        y_rel + chunk.terminal_point - gamma);
                }
                external_commitments.push_back({
                    trace.commitments.at(output_bias_basis_label(
                        head_index, "relation_virtual")),
                    FieldElement::zero()});
                external_points.push_back(gamma);
            }
            const auto output_binding = derive_public_output_challenges(
                proof.public_metadata,
                trace.commitments.at("P_Y"),
                proof.public_logits,
                context.local.num_nodes,
                context.local.num_classes);
            external_commitments.push_back({
                trace.commitments.at("P_Y"),
                evaluate_coefficient_vector(proof.public_logits, output_binding.point)});
            external_points.push_back(output_binding.point);
            add_external_claims(
                opening_registry,
                external_commitments,
                external_points,
                proof.external_evaluation_kinds);
            opening_registry.finalize();
            if (metrics != nullptr)
            {
                metrics->point_group_registry_ms += elapsed_ms(
                    point_group_registry_start, Clock::now());
            }
            for (auto& subproof : proof.pi_bind.subproofs)
            {
                std::vector<MetadataReference> normalized;
                normalized.reserve(subproof.opening_references.size());
                for (auto reference : subproof.opening_references)
                {
                    for (const auto& group : opening_registry.groups())
                    {
                        if (reference.object_id
                            == "W_ext:" + group.point.to_string())
                        {
                            reference.object_id = group.group_id;
                            break;
                        }
                    }
                    if (std::none_of(
                            normalized.begin(), normalized.end(),
                            [&](const MetadataReference& existing)
                            { return existing == reference; }))
                    {
                        normalized.push_back(std::move(reference));
                    }
                }
                subproof.opening_references = std::move(normalized);
            }

            struct PointGroupOpeningResult
            {
                crypto::G1Point witness;
                crypto::BatchOpeningProfile profile;
                double elapsed_ms = 0.0;
            };
            const auto& point_groups = opening_registry.groups();
            std::vector<PointGroupOpeningResult> opening_results(
                point_groups.size());
            const auto open_group_indices = [&]
            (const std::vector<std::size_t>& indices)
            {
                if (indices.empty()) return 0.0;
                const auto phase_start = Clock::now();
                auto work_order = indices;
                const auto estimated_work = [&](std::size_t index)
                {
                    std::size_t total = 0;
                    for (const auto &claim : point_groups[index].claims)
                    {
                        if (claim.commitment.opening_polynomial != nullptr)
                        {
                            total += claim.commitment
                                .opening_polynomial->size();
                        }
                    }
                    return total;
                };
                std::stable_sort(
                    work_order.begin(), work_order.end(),
                    [&](std::size_t lhs, std::size_t rhs)
                    {
                        return estimated_work(lhs) > estimated_work(rhs);
                    });
                const auto hardware_workers = std::max<unsigned>(
                    1, std::thread::hardware_concurrency());
                const bool external_phase =
                    point_groups[work_order.front()].owner_block == "W_ext";
                const auto outer_worker_limit = external_phase
                    ? std::min<std::size_t>(
                        20, hardware_workers)
                    : std::min<std::size_t>(
                        8, std::max<unsigned>(
                            1, hardware_workers / 4));
                const auto worker_count = std::min<std::size_t>(
                    work_order.size(),
                    outer_worker_limit);
                const auto base_inner_worker_count = worker_count > 1
                    ? std::max<std::size_t>(
                        1, hardware_workers / worker_count)
                    : 0;
                std::atomic<std::size_t> next{0};
                crypto::g1_parallel_msm_batch_workers(
                    worker_count,
                    [&](std::size_t)
                    {
                        while (true)
                        {
                            const auto position = next.fetch_add(1);
                            if (position >= work_order.size()) break;
                            const auto group_index = work_order[position];
                            const auto& group = point_groups[group_index];
                            std::vector<Commitment> commitments;
                            std::vector<FieldElement> values;
                            commitments.reserve(group.claims.size());
                            values.reserve(group.claims.size());
                            for (const auto& claim : group.claims)
                            {
                                commitments.push_back(claim.commitment);
                                values.push_back(claim.value);
                            }
                            const auto inner_worker_count =
                                worker_count > 1
                                && !external_phase
                                && estimated_work(group_index)
                                    >= (1ULL << 18)
                                ? std::min<std::size_t>(
                                    8, hardware_workers)
                                : base_inner_worker_count;
                            const crypto::PointGroupOpeningExecutionOptions
                                execution_options{
                                    .max_field_worker_count =
                                        inner_worker_count,
                                    .msm = {
                                        .max_worker_count =
                                            inner_worker_count,
                                    },
                                };
                            const auto local_start = Clock::now();
                            opening_results[group_index].witness =
                                crypto::KZG::open_point_group(
                                    commitments,
                                    group.point,
                                    values,
                                    derive_point_group_challenge(
                                        proof.public_metadata, group),
                                    context.kzg,
                                    &opening_results[group_index].profile,
                                    crypto::BatchOpeningClaimValidation::
                                        AlreadyComputedFromOpeningPolynomials,
                                    worker_count > 1
                                        ? &execution_options
                                        : nullptr);
                            opening_results[group_index].profile
                                .point_parallel_batches = 1;
                            opening_results[group_index].profile
                                .point_parallel_workers = worker_count;
                            opening_results[group_index].elapsed_ms =
                                elapsed_ms(local_start, Clock::now());
                        }
                    });
                return elapsed_ms(phase_start, Clock::now());
            };

            std::vector<std::size_t> domain_group_indices;
            std::vector<std::size_t> external_group_indices;
            for (std::size_t i = 0; i < point_groups.size(); ++i)
            {
                (point_groups[i].owner_block == "W_ext"
                    ? external_group_indices
                    : domain_group_indices).push_back(i);
            }
            const auto external_claim_assembly_ms = elapsed_ms(
                stage_start, Clock::now());
            const auto domain_witness_wall_ms = open_group_indices(
                domain_group_indices);
            const auto external_witness_wall_ms = open_group_indices(
                external_group_indices);

            proof.opening_groups.reserve(point_groups.size());
            for (std::size_t i = 0; i < point_groups.size(); ++i)
            {
                proof.opening_groups.push_back(PointGroupOpening{
                    .group_id = point_groups[i].group_id,
                    .owner_block = point_groups[i].owner_block,
                    .point = point_groups[i].point,
                    .witness = opening_results[i].witness,
                });
                if (metrics == nullptr) continue;
                const auto& profile = opening_results[i].profile;
                metrics->domain_open_denominator_inverse_ms +=
                    profile.denominator_inverse_ms;
                metrics->domain_open_denominator_inverse_reuse_ms +=
                    profile.denominator_inverse_reuse_ms;
                metrics->domain_open_denominator_inverse_reuse_hits +=
                    profile.denominator_inverse_reuse_hits;
                metrics->domain_open_denominator_inverse_cache_misses +=
                    profile.denominator_inverse_cache_misses;
                if (point_groups[i].owner_block.rfind(
                        "Open_dom:", 0) == 0)
                {
                    const auto domain = point_groups[i].owner_block.substr(9);
                    metrics->domain_open_witness_ms +=
                        opening_results[i].elapsed_ms;
                    if (auto* metric = domain_open_metric(metrics, domain);
                        metric != nullptr)
                    {
                        *metric += opening_results[i].elapsed_ms;
                    }
                    if (domain == "FH")
                    {
                        metrics->fh_open_fold_prepare_ms += profile.precompute_ms;
                        metrics->fh_fold_prep_ms += profile.precompute_ms;
                        metrics->fh_open_witness_ms += std::max(
                            0.0,
                            opening_results[i].elapsed_ms
                                - profile.precompute_ms);
                    }
                }
            }
            if (metrics != nullptr)
            {
                metrics->domain_opening_ms += domain_witness_wall_ms;
                metrics->external_opening_ms +=
                    external_claim_assembly_ms + external_witness_wall_ms;
            }
            progress_note("prove:external_opening:done");
            strip_prover_material_from_proof(&proof);
            return proof;
        }

        throw std::runtime_error(
            "formal prover reached the removed single-output compatibility protocol");
    }

} // namespace gatzk::protocol
