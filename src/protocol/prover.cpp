#include "gatzk/protocol/prover.hpp"

#include <algorithm>
#include <atomic>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <barrier>
#include <exception>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <mcl/ntt.hpp>
#include <openssl/sha.h>
#include <openssl/rand.h>

#include "gatzk/algebra/domain_evaluation.hpp"
#include "gatzk/algebra/eval_backend.hpp"
#include "gatzk/algebra/field_io.hpp"
#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/algebra/cuda_weighted_session.hpp"
#include "gatzk/crypto/cuda_msm.hpp"
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
#include "gatzk/protocol/cuda_t_edge.hpp"
#include "gatzk/protocol/t_edge_program.hpp"

namespace gatzk::protocol
{
    namespace
    {
        using algebra::FieldElement;
        using algebra::Polynomial;
        using crypto::Commitment;
        using Clock = std::chrono::steady_clock;

        thread_local std::size_t
            quotient_transform_worker_cap = 0;

        thread_local std::vector<mcl::Fr>
            quotient_coefficient_work;

        class ScopedQuotientCoefficientScratchRelease
        {
        public:
            ScopedQuotientCoefficientScratchRelease() =
                default;

            ScopedQuotientCoefficientScratchRelease(
                const ScopedQuotientCoefficientScratchRelease &) =
                delete;

            ScopedQuotientCoefficientScratchRelease &
            operator=(
                const ScopedQuotientCoefficientScratchRelease &) =
                delete;

            ~ScopedQuotientCoefficientScratchRelease()
            {
                std::vector<mcl::Fr>().swap(
                    quotient_coefficient_work);
            }
        };

        class ScopedQuotientTransformWorkerCap
        {
        public:
            explicit ScopedQuotientTransformWorkerCap(
                std::size_t worker_cap)
                : previous_(
                      quotient_transform_worker_cap)
            {
                quotient_transform_worker_cap =
                    worker_cap;
            }

            ~ScopedQuotientTransformWorkerCap()
            {
                quotient_transform_worker_cap =
                    previous_;
            }

            ScopedQuotientTransformWorkerCap(
                const ScopedQuotientTransformWorkerCap &) = delete;

            ScopedQuotientTransformWorkerCap &
            operator=(
                const ScopedQuotientTransformWorkerCap &) = delete;

        private:
            std::size_t previous_ = 0;
        };

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
        class ScopedCudaProverCacheRelease
        {
        public:
            ScopedCudaProverCacheRelease() = default;

            ~ScopedCudaProverCacheRelease() noexcept
            {
                if (!armed_)
                {
                    return;
                }

                try
                {
                    crypto::
                        release_cuda_g1_msm_resident_cache();
                }
                catch (...)
                {
                }

                try
                {
                    algebra::
                        release_cuda_ntt_workspace();
                }
                catch (...)
                {
                }

                try
                {
                    algebra::
                        release_cuda_weighted_dot_cache();
                }
                catch (...)
                {
                }

                try
                {
                    release_cuda_t_edge_executor_cache();
                }
                catch (...)
                {
                }
            }

            ScopedCudaProverCacheRelease(
                const ScopedCudaProverCacheRelease &) =
                delete;

            ScopedCudaProverCacheRelease &
            operator=(
                const ScopedCudaProverCacheRelease &) =
                delete;

            void dismiss() noexcept
            {
                armed_ = false;
            }

        private:
            bool armed_ = true;
        };

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
            return lhs.basis == rhs.basis && same_polynomial_domain(lhs.domain, rhs.domain) && lhs.values() == rhs.values();
        }

        void add_public_poly_from_existing_material_checked(
            ProtocolContext &context,
            const std::string &name,
            const std::string &source_name)
        {
            if (!full_graph_feature_identity_enabled(context))
            {
                throw std::runtime_error("public polynomial source reuse requires full feature identity: " + name);
            }
            const auto polynomial_it = context.public_polynomials.find(source_name);
            if (polynomial_it == context.public_polynomials.end())
            {
                throw std::runtime_error("missing public polynomial source for reuse: " + source_name);
            }
            const auto commitment_it = context.public_commitments.find(source_name);
            if (commitment_it == context.public_commitments.end())
            {
                throw std::runtime_error("missing public commitment source for reuse: " + source_name);
            }
            if (commitment_it->second.opening_polynomial == nullptr)
            {
                throw std::runtime_error("public commitment source is missing opening material: " + source_name);
            }
            const auto &source_polynomial = polynomial_it->second;
            const auto &source_material = *commitment_it->second.opening_polynomial;
            const bool same_shared_material =
                source_polynomial.shared_data != nullptr && source_material.shared_data != nullptr && source_polynomial.shared_data.get() == source_material.shared_data.get() && source_polynomial.basis == source_material.basis && same_polynomial_domain(source_polynomial.domain, source_material.domain);
            if (!same_shared_material && !same_polynomial_material(source_material, source_polynomial))
            {
                throw std::runtime_error("public commitment source material mismatch: " + name + " -> " + source_name);
            }
            progress_note("context:public_reuse:" + name + "<-" + source_name + ":start");
            auto reused_polynomial = source_polynomial;
            reused_polynomial.name = name;
            context.public_polynomials[name] = std::move(reused_polynomial);
            auto commitment = commitment_it->second;
            commitment.name = name;
            context.public_commitments[name] = std::move(commitment);
            progress_note("context:public_reuse:" + name + "<-" + source_name + ":done");
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
            if (polynomial.domain == nullptr || context.domains.fh == nullptr || polynomial.domain->size != context.domains.fh->size || !(polynomial.domain->omega == context.domains.fh->omega) || polynomial.values().size() != context.domains.fh->size)
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

        bool complete_public_feature_source(const ProtocolContext &context)
        {
            if (context.dataset.num_nodes == 0 || context.dataset.num_features == 0 || context.local.num_nodes != context.dataset.num_nodes || context.local.num_features != context.dataset.num_features || context.local.graph_count != context.dataset.graph_count || context.local.absolute_ids.size() != context.dataset.num_nodes || context.local.features.size() != context.dataset.num_nodes || context.local.public_input.N_total != context.dataset.num_nodes)
                return false;
            for (std::size_t row = 0; row < context.dataset.num_nodes; ++row)
            {
                if (context.local.absolute_ids[row] != row || context.local.features[row].size() != context.dataset.num_features)
                    return false;
            }
            return true;
        }

        bool lazy_fh_public_enabled(const ProtocolContext &context)
        {
            return complete_public_feature_source(context) && context.domains.fh != nullptr;
        }

        bool is_lazy_fh_public_label(const std::string &name)
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

        FieldElement evaluate_lazy_fh_public_poly(
            const ProtocolContext &context,
            const std::string &name,
            const FieldElement &point)
        {
            if (!lazy_fh_public_enabled(context) || !is_lazy_fh_public_label(name))
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
            if (feature_count == 0)
                throw std::runtime_error("feature table has zero feature columns");
            if (dataset_rows > std::numeric_limits<std::size_t>::max() / feature_count)
                throw std::runtime_error("feature table size overflows size_t");
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
                    return evaluate_lazy_fh_public_poly(context, "P_Q_tbl_feat", point);
                }
                return evaluate_lazy_fh_public_poly(context, "P_Q_qry_feat", point);
            }
            if (name == "P_Query_feat" && full_feature_identity)
            {
                return evaluate_lazy_full_feature_lookup_trace_poly(context, challenges, "P_Table_feat", point);
            }

            const auto eta_feat = challenges.at("eta_feat");
            const auto row_label = name == "P_Table_feat" ? "P_Row_feat_tbl" : "P_Row_feat_qry";
            const auto col_label = name == "P_Table_feat" ? "P_Col_feat_tbl" : "P_Col_feat_qry";
            const auto row_eval = evaluate_lazy_fh_public_poly(context, row_label, point);
            const auto col_eval = evaluate_lazy_fh_public_poly(context, col_label, point);
            const auto feat_eval = evaluate_lazy_fh_public_poly(context, "P_T_H", point);
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
                out[i] = FieldElement::one() - (FieldElement::one() - graph_boundary[i]) * (FieldElement::one() - destination_boundary[i]);
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

        std::int64_t quantize_float_model_value_signed(
            double value,
            std::uint64_t scale)
        {
            return data::quantize_raw_at_scale(value, scale);
        }

        FieldElement quantize_float_model_value(double value, std::uint64_t scale)
        {
            return FieldElement::from_signed(
                quantize_float_model_value_signed(value, scale));
        }

        double dequantize_trace_scalar(std::int64_t value, std::uint64_t scale)
        {
            return static_cast<double>(value) / static_cast<double>(scale);
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

        std::vector<std::pair<FieldElement, FieldElement>> make_lrelu_table(
            std::int64_t min_raw,
            std::int64_t max_raw,
            std::uint64_t input_scale,
            std::uint64_t output_scale)
        {
            std::vector<std::pair<FieldElement, FieldElement>> out;
            out.reserve(signed_table_size(min_raw, max_raw, "lrelu"));
            for (std::int64_t raw = min_raw; raw <= max_raw; ++raw)
            {
                const auto input = static_cast<double>(raw) / static_cast<double>(input_scale);
                const auto output = input >= 0.0 ? input : 0.2 * input;
                const auto output_raw = quantize_float_model_value_signed(output, output_scale);
                out.push_back({FieldElement::from_signed(raw), FieldElement::from_signed(output_raw)});
                if (raw == max_raw)
                {
                    break;
                }
            }
            return out;
        }

        std::vector<std::pair<FieldElement, FieldElement>> make_elu_table(
            std::size_t bound,
            std::uint64_t input_scale,
            std::uint64_t output_scale)
        {
            std::vector<std::pair<FieldElement, FieldElement>> out;
            out.reserve(bound * 2 + 1);
            for (std::int64_t raw = -static_cast<std::int64_t>(bound); raw <= static_cast<std::int64_t>(bound); ++raw)
            {
                const auto input = dequantize_trace_scalar(raw, input_scale);
                const auto central_signed = quantize_float_model_value_signed(
                    input >= 0.0 ? input : std::exp(input) - 1.0,
                    output_scale);
                out.push_back({
                    FieldElement::from_signed(raw),
                    FieldElement::from_signed(central_signed),
                });
            }
            return out;
        }

        std::vector<std::pair<FieldElement, FieldElement>> make_exp_table(
            std::size_t size,
            std::uint64_t input_scale,
            std::uint64_t output_scale)
        {
            if (size == 0 || input_scale == 0 || output_scale == 0)
            {
                throw std::runtime_error(
                    "ExpMap table requires a non-empty domain and positive scales");
            }
            std::vector<std::pair<FieldElement, FieldElement>> out;
            out.reserve(size);
            std::int64_t previous = std::numeric_limits<std::int64_t>::max();
            for (std::size_t i = 0; i < size; ++i)
            {
                const auto input = static_cast<std::int64_t>(i);
                const auto central_signed = quantize_float_model_value_signed(
                    std::exp(-static_cast<double>(input) / static_cast<double>(input_scale)),
                    output_scale);
                // ExpMap is a positive fixed-point weight.  Rounding the tail
                // of exp(-delta/S_Delta) to zero would make the denominator
                // relation degenerate and would violate the formal table
                // contract, so the registered table saturates at one.
                const auto candidate = std::max<std::int64_t>(1, central_signed);
                if (candidate > previous)
                {
                    throw std::runtime_error(
                        "ExpMap table is not non-increasing at delta=" +
                        std::to_string(i));
                }
                if (candidate < 1)
                {
                    throw std::runtime_error(
                        "ExpMap table contains a non-positive output at delta=" +
                        std::to_string(i));
                }
                out.push_back({
                    FieldElement(static_cast<std::uint64_t>(i)),
                    FieldElement::from_signed(candidate),
                });
                previous = candidate;
            }
            if (out.front().second.value() == 0 ||
                out.front().second.value() < out.back().second.value())
            {
                throw std::runtime_error(
                    "ExpMap table endpoint invariant failed");
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

        void require_exp_map_table(
            const std::vector<std::pair<FieldElement, FieldElement>> &table)
        {
            if (table.empty())
            {
                throw std::runtime_error("ExpMap table must be non-empty");
            }
            std::uint64_t previous = std::numeric_limits<std::uint64_t>::max();
            for (std::size_t index = 0; index < table.size(); ++index)
            {
                const auto expected_input =
                    FieldElement(static_cast<std::uint64_t>(index));
                if (!(table[index].first == expected_input))
                {
                    throw std::runtime_error(
                        "ExpMap table input order/gap at delta=" +
                        std::to_string(index));
                }
                const auto output = table[index].second.value();
                if (output == 0)
                {
                    throw std::runtime_error(
                        "ExpMap table output must be >= 1 at delta=" +
                        std::to_string(index));
                }
                if (index != 0 && output > previous)
                {
                    throw std::runtime_error(
                        "ExpMap table must be non-increasing at delta=" +
                        std::to_string(index));
                }
                previous = output;
            }
            if (table.front().second.value() < table.back().second.value())
            {
                throw std::runtime_error(
                    "ExpMap(0) must be the maximum registered output");
            }
        }

        void require_canonical_range_table(
            const std::vector<FieldElement> &table,
            std::size_t range_bits)
        {
            if (range_bits == 0 || range_bits >= 63)
            {
                throw std::runtime_error(
                    "T_range requires range_bits in [1,62]");
            }
            const auto expected_size = std::size_t{1} << range_bits;
            if (table.size() != expected_size)
            {
                throw std::runtime_error(
                    "T_range cardinality must be exactly 2^B");
            }
            for (std::size_t index = 0; index < table.size(); ++index)
            {
                if (table[index].value() != index)
                {
                    throw std::runtime_error(
                        "T_range must contain canonical rows 0..2^B-1");
                }
            }
        }

        std::vector<FieldElement> quantize_model_vector(
            const std::vector<double> &values,
            std::uint64_t scale)
        {
            std::vector<FieldElement> out(values.size(), FieldElement::zero());
            for (std::size_t i = 0; i < values.size(); ++i)
            {
                out[i] = quantize_float_model_value(values[i], scale);
            }
            return out;
        }

        std::vector<FieldElement> quantize_model_vector_at_scale(
            const std::vector<double> &values,
            std::uint64_t scale)
        {
            std::vector<FieldElement> out(values.size(), FieldElement::zero());
            for (std::size_t i = 0; i < values.size(); ++i)
            {
                out[i] = FieldElement::from_signed(
                    quantize_float_model_value_signed(values[i], scale));
            }
            return out;
        }

        model::Matrix quantize_model_matrix(
            const model::FloatMatrix &matrix,
            std::uint64_t scale)
        {
            model::Matrix out(matrix.size());
            for (std::size_t row = 0; row < matrix.size(); ++row)
            {
                out[row] = quantize_model_vector(matrix[row], scale);
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

        std::string cache_sha256_hex(std::string_view bytes)
        {
            std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
            SHA256(
                reinterpret_cast<const unsigned char *>(bytes.data()),
                bytes.size(),
                digest.data());
            std::ostringstream out;
            out << std::hex << std::setfill('0');
            for (const auto byte : digest)
                out << std::setw(2) << static_cast<unsigned int>(byte);
            return out.str();
        }

        void append_cache_length_delimited(
            std::string *out,
            std::string_view value)
        {
            out->append(std::to_string(value.size()));
            out->push_back(':');
            out->append(value.data(), value.size());
            out->push_back('|');
        }

        std::filesystem::path resolve_project_path(
            const std::string &project_root,
            const std::string &relative_or_absolute);

        std::string cache_file_sha256(const std::filesystem::path &path)
        {
            std::ifstream input(path, std::ios::binary);
            if (!input)
                throw std::runtime_error(
                    "static context cache cannot read identity file: " +
                    path.string());
            SHA256_CTX digest_context{};
            SHA256_Init(&digest_context);
            std::array<char, 1U << 20> buffer{};
            while (input)
            {
                input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
                const auto count = input.gcount();
                if (count > 0)
                    SHA256_Update(
                        &digest_context,
                        buffer.data(),
                        static_cast<std::size_t>(count));
            }
            if (!input.eof())
                throw std::runtime_error(
                    "static context cache failed while reading identity file: " +
                    path.string());
            std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
            SHA256_Final(digest.data(), &digest_context);
            std::ostringstream out;
            out << std::hex << std::setfill('0');
            for (const auto byte : digest)
                out << std::setw(2) << static_cast<unsigned int>(byte);
            return out.str();
        }

        // Hash the complete content-addressed file set for a private runtime
        // resource.  Paths are included in sorted order so replacing a file at
        // the same path cannot preserve a stale cache entry.  This helper is
        // deliberately used only by the static-context cache; it never enters
        // public proof metadata or exported artifacts.
        std::string cache_file_tree_identity(const std::filesystem::path &root)
        {
            if (!std::filesystem::exists(root))
                return "missing:" + root.lexically_normal().generic_string();
            if (std::filesystem::is_regular_file(root))
                return "file:" + root.lexically_normal().generic_string() +
                       ":" + cache_file_sha256(root);
            if (!std::filesystem::is_directory(root))
                throw std::runtime_error(
                    "static context cache identity root is not a file or directory: " +
                    root.string());

            std::vector<std::filesystem::path> files;
            std::error_code error;
            for (std::filesystem::recursive_directory_iterator it(
                     root,
                     std::filesystem::directory_options::skip_permission_denied,
                     error),
                 end;
                 it != end;
                 it.increment(error))
            {
                if (error)
                    throw std::runtime_error(
                        "static context cache identity traversal failed under: " +
                        root.string());
                if (it->is_regular_file(error))
                    files.push_back(it->path());
                if (error)
                    throw std::runtime_error(
                        "static context cache identity stat failed under: " +
                        root.string());
            }
            std::sort(
                files.begin(), files.end(),
                [&](const auto &lhs, const auto &rhs)
                {
                    return std::filesystem::relative(lhs, root).generic_string() <
                           std::filesystem::relative(rhs, root).generic_string();
                });
            std::string canonical;
            append_cache_length_delimited(
                &canonical,
                root.lexically_normal().generic_string());
            for (const auto &file : files)
            {
                append_cache_length_delimited(
                    &canonical,
                    std::filesystem::relative(file, root).generic_string());
                append_cache_length_delimited(
                    &canonical,
                    cache_file_sha256(file));
            }
            return "sha256:" + cache_sha256_hex(canonical);
        }

        std::string context_cache_key(const util::AppConfig &config)
        {
            std::string canonical = "zkgat:static-context-cache|";
            const auto append = [&](std::string_view label, std::string_view value)
            {
                append_cache_length_delimited(&canonical, label);
                append_cache_length_delimited(&canonical, value);
            };
            const auto append_u64 = [&](std::string_view label, std::uint64_t value)
            {
                append(label, std::to_string(value));
            };
            append("project_root", config.project_root);
            append("dataset", config.dataset);
            append("data_root", config.data_root);
            append("cache_root", config.cache_root);
            append("checkpoint_bundle", config.checkpoint_bundle);
            append("task_type", config.task_type);
            append("report_unit", config.report_unit);
            append("batching_rule", config.batching_rule);
            append("subgraph_rule", config.subgraph_rule);
            append("self_loop_rule", config.self_loop_rule);
            append("edge_sort_rule", config.edge_sort_rule);
            append("chunking_rule", config.chunking_rule);
            append("quant_cfg_id", config.quant_cfg_id);
            append("quantization", config.quantization.canonical_serialization());
            append("model_arch_id", config.model_arch_id);
            append("model_param_id", config.model_param_id);
            append("input_preprocess_id", config.input_preprocess_id);
            append("static_table_id", config.static_table_id);
            append("degree_bound_id", config.degree_bound_id);
            append("srs_path", config.srs_path);
            append("srs_id", config.srs_id);
            append("srs_type", config.srs_type);
            append("verifier_registry_path", config.verifier_registry_path);
            append("verifier_registry_digest", config.verifier_registry_digest);
            append("privacy_policy", config.privacy_policy);
            append("input_privacy", config.input_privacy);
            append("output_mode", config.output_mode);
            append("opening_mode", config.opening_mode);
            append("binding_mode", config.binding_mode);
            append("proof_task", config.proof_task);
            append("security_profile_id", config.security_profile_id);
            append_u64("num_classes", config.num_classes);
            append_u64("range_bits", config.range_bits);
            append_u64("seed", config.seed);
            append_u64("layer_count", config.layer_count);
            append_u64("K_out", config.K_out);
            append_u64("batch_graphs", config.batch_graphs);
            append_u64("symmetrize_edges", config.symmetrize_edges ? 1U : 0U);
            append_u64("deduplicate_edges", config.deduplicate_edges ? 1U : 0U);
            append_u64("auto_prepare_dataset", config.auto_prepare_dataset ? 1U : 0U);
            for (const auto input_dim : config.d_in_profile)
                append_u64("d_in_profile", input_dim);
            for (const auto &layer : config.hidden_profile)
            {
                append_u64("hidden_head_count", layer.head_count);
                append_u64("hidden_head_dim", layer.head_dim);
            }

            const auto project_root = std::filesystem::path(config.project_root);
            const auto configured_resource_identity =
                [&](const std::string &configured_path)
            {
                if (configured_path.empty())
                    return std::string("missing-config");
                return cache_file_tree_identity(
                    resolve_project_path(config.project_root, configured_path));
            };
            // These private runtime identities bind the actual model registry,
            // the complete checkpoint bundle tree (manifest, tensors.txt,
            // tensors.npz, and every other regular file), dataset
            // feature/topology source, SRS manifest and all PK/VK material,
            // plus the compiler source that defines the artifact.  The two
            // tensor encodings are both required; neither is a fallback.
            append(
                "checkpoint_bundle_files",
                configured_resource_identity(config.checkpoint_bundle));
            append(
                "dataset_features_topology_public_input_files",
                cache_file_tree_identity(
                    project_root / config.cache_root / config.dataset));
            append(
                "srs_manifest_vk_pk_files",
                configured_resource_identity(config.srs_path));
            append(
                "compiler_sources",
                cache_file_tree_identity(project_root / "include") +
                    "+" + cache_file_tree_identity(project_root / "src"));
            append(
                "protocol_compiler_template",
                cache_sha256_hex(protocol_template_canonical_bytes()));
            return "sha256:" + cache_sha256_hex(canonical);
        }

        // The evaluation memoization cache is proof-execution local.  Keep its
        // key independent from the static-context file scan so opening a proof
        // does not repeatedly reread large private SRS files.
        std::string execution_context_key(const util::AppConfig &config)
        {
            std::ostringstream stream;
            stream << "zkgat:execution-context|"
                   << config.project_root << '|'
                   << config.dataset << '|'
                   << config.num_classes << '|'
                   << config.range_bits << '|'
                   << config.seed << '|'
                   << config.layer_count << '|'
                   << config.K_out << '|'
                   << config.batch_graphs << '|'
                   << config.batching_rule << '|'
                   << config.srs_id << '|'
                   << config.srs_type << '|'
                   << config.quant_cfg_id << '|'
                   << config.input_preprocess_id << '|'
                   << config.quantization.canonical_serialization();
            for (const auto input_dim : config.d_in_profile)
                stream << "|din=" << input_dim;
            for (const auto &layer : config.hidden_profile)
                stream << "|hid=" << layer.head_count << 'x' << layer.head_dim;
            return stream.str();
        }

        std::size_t hidden_head_width(const model::ModelParameters &parameters)
        {
            return model::max_hidden_head_dim(parameters);
        }

        std::size_t concat_width(const model::ModelParameters &parameters)
        {
            return model::max_hidden_concat_width(parameters);
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
            for (std::size_t layer_index = 0;
                 layer_index < context.model.hidden_layers.size();
                 ++layer_index)
                for (std::size_t head_index = 0;
                     head_index < context.model.hidden_layers[layer_index].heads.size();
                     ++head_index)
                {
                    const auto suffix = "l" + std::to_string(layer_index) + "_h" + std::to_string(head_index);
                    const auto prefix = hidden_head_prefix(layer_index, head_index) + "_";
                    const auto challenge = [&](std::string_view base)
                    {
                        return hidden_challenge_name(base, layer_index, head_index);
                    };
                    specs.push_back({"mu_" + suffix + "_proj", prefix + "H_prime", challenge("y_proj_h")});
                    specs.push_back({"mu_" + suffix + "_src", prefix + "E_src", challenge("y_src_h")});
                    specs.push_back({hidden_score_mask_eval_name(layer_index, head_index, "src"),
                                     hidden_score_mask_label(layer_index, head_index, "src"),
                                     challenge("y_src_h")});
                    specs.push_back({"mu_" + suffix + "_dst", prefix + "E_dst", challenge("y_dst_h")});
                    specs.push_back({hidden_score_mask_eval_name(layer_index, head_index, "dst"),
                                     hidden_score_mask_label(layer_index, head_index, "dst"),
                                     challenge("y_dst_h")});
                    specs.push_back({"mu_" + suffix + "_star", prefix + "H_star", challenge("y_star_h")});
                    specs.push_back({external_mask_eval_name("mu_" + suffix + "_star"),
                                     external_mask_label(prefix + "H_star"),
                                     challenge("y_star_h")});
                    specs.push_back({"mu_" + suffix + "_agg_pre", prefix + "H_agg_pre_star", challenge("y_agg_pre_h")});
                    specs.push_back({external_mask_eval_name("mu_" + suffix + "_agg_pre"),
                                     external_mask_label(prefix + "H_agg_pre_star"),
                                     challenge("y_agg_pre_h")});
                    specs.push_back({"mu_" + suffix + "_agg", prefix + "H_agg_star", challenge("y_agg_h")});
                    specs.push_back({external_mask_eval_name("mu_" + suffix + "_agg"),
                                     external_mask_label(prefix + "H_agg_star"),
                                     challenge("y_agg_h")});
                }
            for (std::size_t layer_index = 0; layer_index < context.model.hidden_layers.size(); ++layer_index)
            {
                specs.push_back({
                    "mu_cat_l" + std::to_string(layer_index),
                    hidden_layer_concat_star_label(layer_index),
                    hidden_concat_y_name(layer_index),
                });
                const auto concat_eval_name =
                    "mu_cat_l" + std::to_string(layer_index);
                specs.push_back({
                    external_mask_eval_name(concat_eval_name),
                    external_mask_label(hidden_layer_concat_star_label(layer_index)),
                    hidden_concat_y_name(layer_index),
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

        std::shared_ptr<algebra::DomainEvaluationWeightCache>
        shared_proof_domain_weight_cache()
        {
            // Share one cache across all EvaluationMemoization instances belonging
            // to this prove() invocation, but never retain challenge-point weights
            // across proofs. Large FH/edge weights are proof-specific and keeping
            // them in a process-static cache unnecessarily extends their lifetime
            // into later opening working sets.
            return std::make_shared<
                algebra::DomainEvaluationWeightCache>();
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

        std::vector<FieldElement> evaluate_spilled_evaluation_polynomial_points_impl(
            const TraceArtifacts::SpilledEvaluationPolynomial &spilled,
            const std::shared_ptr<algebra::RootOfUnityDomain> &domain,
            const std::vector<FieldElement> &points,
            const std::vector<const algebra::DomainEvaluationWeights *> &weights)
        {
            if (domain == nullptr)
            {
                throw std::runtime_error(
                    "spilled polynomial is missing its domain");
            }

            if (points.size() != weights.size())
            {
                throw std::runtime_error(
                    "spilled polynomial point/weight count mismatch");
            }

            std::vector<FieldElement> values(
                points.size(),
                FieldElement::zero());

            if (points.empty())
            {
                return values;
            }

            std::vector<std::size_t> interpolation_indices;
            std::vector<std::size_t> direct_indices;

            interpolation_indices.reserve(points.size());
            direct_indices.reserve(points.size());

            for (std::size_t point_index = 0;
                 point_index < points.size();
                 ++point_index)
            {
                if (weights[point_index] == nullptr)
                {
                    throw std::runtime_error(
                        "spilled polynomial received null domain weights");
                }

                if (weights[point_index]
                        ->direct_index
                        .has_value())
                {
                    direct_indices.push_back(
                        point_index);
                    continue;
                }

                if (weights[point_index]
                        ->native_weights.size() <
                    spilled.size)
                {
                    throw std::runtime_error(
                        "spilled polynomial exceeds its domain weights");
                }

                interpolation_indices.push_back(
                    point_index);
            }

            constexpr std::size_t kFieldBytes = 32;

            std::ifstream stream(
                spilled.path,
                std::ios::binary);

            if (!stream)
            {
                throw std::runtime_error(
                    "failed to open spilled polynomial file: " +
                    spilled.path);
            }

            // Direct domain hits are sparse. Keep the O(number_of_points)
            // seek route so a direct-only opening never scans the whole spill.
            if (interpolation_indices.empty())
            {
                std::array<std::uint8_t, kFieldBytes> bytes{};

                for (const auto point_index :
                     direct_indices)
                {
                    const auto direct_index =
                        *weights[point_index]
                             ->direct_index;

                    if (direct_index >= spilled.size)
                    {
                        continue;
                    }

                    const auto offset =
                        static_cast<std::streamoff>(
                            direct_index *
                            kFieldBytes);

                    stream.seekg(offset);

                    stream.read(
                        reinterpret_cast<char *>(
                            bytes.data()),
                        static_cast<std::streamsize>(
                            bytes.size()));

                    if (!stream)
                    {
                        throw std::runtime_error(
                            "failed to read direct spilled polynomial entry: " +
                            spilled.path);
                    }

                    values[point_index] =
                        FieldElement::
                            from_little_endian_mod(
                                bytes.data(),
                                bytes.size());
                }

                return values;
            }

            const bool use_cuda_stream =
                util::route2_options()
                    .cuda_field_hotspots &&
                algebra::
                    cuda_field_hotspots_available() &&
                spilled.size >=
                    (std::size_t{1} << 18);

            constexpr std::size_t kStreamChunkSize =
                std::size_t{1} << 20;

            std::vector<FieldElement> chunk;
            chunk.reserve(
                std::min(
                    kStreamChunkSize,
                    spilled.size));

            auto read_chunk =
                [&](std::size_t count)
            {
                chunk.resize(count);

                try
                {
                    algebra::read_field_elements_binary_into(
                        stream,
                        std::span<FieldElement>(
                            chunk.data(),
                            chunk.size()));
                }
                catch (const std::exception &error)
                {
                    throw std::runtime_error(
                        "failed to stream spilled polynomial: " +
                        spilled.path +
                        ": " +
                        error.what());
                }
            };

            auto collect_direct_values =
                [&](std::size_t offset,
                    std::size_t count)
            {
                for (const auto point_index :
                     direct_indices)
                {
                    const auto direct_index =
                        *weights[point_index]
                             ->direct_index;

                    if (direct_index < offset ||
                        direct_index >=
                            offset + count)
                    {
                        continue;
                    }

                    values[point_index] =
                        chunk[direct_index -
                              offset];
                }
            };

            if (use_cuda_stream)
            {
                std::vector<const mcl::Fr *>
                    interpolation_weights;

                interpolation_weights.reserve(
                    interpolation_indices.size());

                for (const auto point_index :
                     interpolation_indices)
                {
                    interpolation_weights.push_back(
                        weights[point_index]
                            ->native_weights.data());
                }

                auto session =
                    algebra::
                        create_cuda_weighted_dot_session(
                            interpolation_weights,
                            spilled.size,
                            kStreamChunkSize);

                if (session == nullptr)
                {
                    throw std::runtime_error(
                        "CUDA spilled evaluation route was selected but unavailable");
                }

                for (std::size_t offset = 0;
                     offset < spilled.size;)
                {
                    const auto count =
                        std::min(
                            kStreamChunkSize,
                            spilled.size - offset);

                    read_chunk(count);
                    collect_direct_values(
                        offset,
                        count);

                    session->accumulate(
                        FieldElement::
                            native_array_data(
                                chunk.data()),
                        offset,
                        count);

                    offset += count;
                }

                const auto native_results =
                    session->finish();

                if (native_results.size() !=
                    interpolation_indices.size())
                {
                    throw std::runtime_error(
                        "CUDA spilled evaluation result count mismatch");
                }

                for (std::size_t result_index = 0;
                     result_index <
                     interpolation_indices.size();
                     ++result_index)
                {
                    const auto point_index =
                        interpolation_indices[result_index];

                    values[point_index] =
                        FieldElement::from_native(
                            native_results[result_index]);

                    if (!spilled
                             .zero_domain_blinder
                             .empty())
                    {
                        values[point_index] +=
                            domain->zero_polynomial_eval(
                                points[point_index]) *
                            algebra::horner(
                                spilled
                                    .zero_domain_blinder,
                                points[point_index]);
                    }
                }

                return values;
            }

            std::vector<mcl::Fr> native_sums(
                points.size());

            for (auto &sum : native_sums)
            {
                sum.clear();
            }

            for (std::size_t offset = 0;
                 offset < spilled.size;)
            {
                const auto count =
                    std::min(
                        kStreamChunkSize,
                        spilled.size - offset);

                read_chunk(count);
                collect_direct_values(
                    offset,
                    count);

                const auto *chunk_native =
                    FieldElement::
                        native_array_data(
                            chunk.data());

                for (std::size_t local = 0;
                     local < count;
                     ++local)
                {
                    const auto global_index =
                        offset + local;

                    for (const auto point_index :
                         interpolation_indices)
                    {
                        mcl::Fr term;

                        mcl::Fr::mul(
                            term,
                            chunk_native[local],
                            weights[point_index]
                                ->native_weights[global_index]);

                        mcl::Fr::add(
                            native_sums[point_index],
                            native_sums[point_index],
                            term);
                    }
                }

                offset += count;
            }

            for (const auto point_index :
                 interpolation_indices)
            {
                values[point_index] =
                    FieldElement::from_native(
                        native_sums[point_index]);

                if (!spilled
                         .zero_domain_blinder
                         .empty())
                {
                    values[point_index] +=
                        domain->zero_polynomial_eval(
                            points[point_index]) *
                        algebra::horner(
                            spilled
                                .zero_domain_blinder,
                            points[point_index]);
                }
            }

            return values;
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
                  context_key_(execution_context_key(context.config)),
                  proof_cache_key_(context_key_),
                  metrics_(metrics)
            {
                if (domain_weight_cache_ == nullptr)
                {
                    domain_weight_cache_ = std::make_shared<algebra::DomainEvaluationWeightCache>();
                }
                // Trace evaluations include the proof-local zero-domain
                // blinding polynomial.  A point/label hit from another proof
                // is therefore not semantically reusable, even if its point
                // happens to coincide.  Require the fresh session nonce and
                // bind it into the process-shared cache key.
                if (!trace.public_metadata.proof_nonce.empty())
                {
                    proof_cache_key_ +=
                        "|proof_nonce=" + trace.public_metadata.proof_nonce;
                    trace_cache_reuse_enabled_ = true;
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
                        polynomial_it != trace_.polynomials.end() && polynomial_it->second.is_zero_domain_blinded())
                    {
                        value += polynomial_it->second.domain->zero_polynomial_eval(point) * algebra::horner(
                                                                                                 polynomial_it->second.zero_domain_blinder,
                                                                                                 point);
                    }
                    else if (const auto spilled_it =
                                 trace_.spilled_polynomials.find(name);
                             spilled_it != trace_.spilled_polynomials.end() && !spilled_it->second
                                                                                    .zero_domain_blinder.empty())
                    {
                        const auto domain =
                            domain_from_name(
                                spilled_it->second.domain_name);
                        value += domain->zero_polynomial_eval(point) * algebra::horner(
                                                                           spilled_it->second
                                                                               .zero_domain_blinder,
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
                else if (lazy_fh_public_enabled(context_) && is_lazy_fh_public_label(name))
                {
                    value = evaluate_lazy_fh_public_poly(context_, name, point);
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

            std::vector<FieldElement> evaluate_lazy_fh_public_group(
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
                if (feature_count == 0)
                    throw std::runtime_error("feature table has zero feature columns");
                if (dataset_rows > std::numeric_limits<std::size_t>::max() / feature_count)
                    throw std::runtime_error("feature table size overflows size_t");
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

            std::vector<std::vector<FieldElement>> evaluate_lazy_fh_public_group_points(
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
                if (feature_count == 0)
                    throw std::runtime_error("feature table has zero feature columns");
                if (dataset_rows > std::numeric_limits<std::size_t>::max() / feature_count)
                    throw std::runtime_error("feature table size overflows size_t");
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
                    const auto values = evaluate_lazy_fh_public_group(missing_public_labels, point);
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
                    if (polynomial_it != trace_.polynomials.end() && polynomial_it->second.is_zero_domain_blinded())
                    {
                        out[i] += polynomial_it->second.domain
                                      ->zero_polynomial_eval(point) *
                                  algebra::horner(
                                      polynomial_it->second
                                          .zero_domain_blinder,
                                      point);
                        continue;
                    }
                    const auto spilled_it =
                        trace_.spilled_polynomials.find(labels[i]);
                    if (spilled_it == trace_.spilled_polynomials.end() || spilled_it->second.zero_domain_blinder.empty())
                    {
                        continue;
                    }
                    const auto domain = domain_from_name(
                        spilled_it->second.domain_name);
                    out[i] += domain->zero_polynomial_eval(point) * algebra::horner(
                                                                        spilled_it->second.zero_domain_blinder,
                                                                        point);
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
                const bool enable_backend_precompute = util::route2_options().packed_fft_backend;

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

                if (lazy_fh_public_enabled(context_))
                {
                    std::vector<std::string> lazy_public_labels;
                    lazy_public_labels.reserve(labels.size());
                    for (const auto &label : labels)
                    {
                        if (is_lazy_fh_public_label(label))
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
                                          evaluate_lazy_fh_public_group(lazy_public_labels, missing_points.front())}
                                    : evaluate_lazy_fh_public_group_points(lazy_public_labels, missing_points);
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
                        if (!required_public_labels.empty() && !lazy_fh_public_enabled(context_))
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

                // Spilled evaluation polynomials are not registered in the
                // packed in-memory backend. Seed their ordinary label@point
                // entries with one file pass per label instead of letting
                // eval_named rescan the file independently for every point.
                std::vector<SpilledEvaluationRequest> spilled_requests;
                std::vector<std::vector<std::string>> spilled_cache_keys;
                std::vector<std::string> scheduled_spilled_labels;
                spilled_requests.reserve(labels.size());
                spilled_cache_keys.reserve(labels.size());
                scheduled_spilled_labels.reserve(labels.size());
                for (const auto &label : labels)
                {
                    const auto spilled_it = trace_.spilled_polynomials.find(label);
                    if (spilled_it == trace_.spilled_polynomials.end() || trace_.polynomials.contains(label) || std::find(scheduled_spilled_labels.begin(), scheduled_spilled_labels.end(), label) != scheduled_spilled_labels.end())
                    {
                        continue;
                    }
                    std::vector<FieldElement> missing_points;
                    std::vector<const algebra::DomainEvaluationWeights *> missing_weights;
                    std::vector<std::string> missing_cache_keys;
                    missing_points.reserve(points.size());
                    missing_weights.reserve(points.size());
                    missing_cache_keys.reserve(points.size());
                    const auto domain = domain_from_name(spilled_it->second.domain_name);
                    for (const auto &point : points)
                    {
                        const auto cache_key = label + "@" + algebra::point_cache_key(point);
                        if (value_cache_.contains(cache_key) || std::find(
                                                                    missing_cache_keys.begin(),
                                                                    missing_cache_keys.end(),
                                                                    cache_key) != missing_cache_keys.end())
                        {
                            continue;
                        }
                        missing_points.push_back(point);
                        missing_weights.push_back(&domain_weights(domain, point));
                        missing_cache_keys.push_back(cache_key);
                    }
                    if (missing_points.empty())
                    {
                        continue;
                    }
                    scheduled_spilled_labels.push_back(label);
                    spilled_requests.push_back(SpilledEvaluationRequest{
                        .spilled = &spilled_it->second,
                        .domain = domain,
                        .points = std::move(missing_points),
                        .weights = std::move(missing_weights),
                    });
                    spilled_cache_keys.push_back(std::move(missing_cache_keys));
                }
                const auto spilled_values =
                    evaluate_spilled_evaluation_requests(
                        spilled_requests, 4);
                for (std::size_t job_index = 0;
                     job_index < spilled_requests.size();
                     ++job_index)
                {
                    for (std::size_t point_index = 0;
                         point_index < spilled_requests[job_index].points.size();
                         ++point_index)
                    {
                        value_cache_.emplace(
                            std::move(spilled_cache_keys[job_index][point_index]),
                            spilled_values[job_index][point_index]);
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
                                proof_values[i] += polynomial->domain->zero_polynomial_eval(point) * algebra::horner(polynomial->zero_domain_blinder, point);
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
                        trace_cache_reuse_enabled_ &&
                        (group.first->name == "FH" || group.first->name == "edge");
                    if (allow_shared_trace_reuse)
                    {
                        bool all_cached = true;
                        std::vector<std::shared_ptr<const std::vector<FieldElement>>> cached_values(points.size());
                        for (std::size_t point_index = 0; point_index < points.size(); ++point_index)
                        {
                            cached_values[point_index] = shared_trace_evaluation_cache()->lookup(
                                proof_cache_key_,
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
                    if (util::route2_options().rotated_fft_kernel && group.first->size >= 256)
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
                                            proof_cache_key_,
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
                                        proof_cache_key_,
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
                                proof_cache_key_,
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
                if (values.size() > weight_entry.native_weights.size())
                {
                    throw std::runtime_error(
                        "evaluation polynomial exceeds its domain weights");
                }
                auto value = FieldElement::zero();
                if (values.size() == weight_entry.native_weights.size())
                {
                    value = algebra::dot_product_native_weights(
                        values, weight_entry.native_weights);
                }
                else
                {
                    const std::vector<mcl::Fr> active_weights(
                        weight_entry.native_weights.begin(),
                        weight_entry.native_weights.begin() + static_cast<std::ptrdiff_t>(
                                                                  values.size()));
                    value = algebra::dot_product_native_weights(
                        values, active_weights);
                }
                if (polynomial.is_zero_domain_blinded())
                {
                    value += polynomial.domain->zero_polynomial_eval(point) * algebra::horner(polynomial.zero_domain_blinder, point);
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
            std::string proof_cache_key_;
            bool trace_cache_reuse_enabled_ = false;
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
                algebra::geometric_powers_native(
                    powers->data(), powers->size(), inv_shift);
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
            std::ifstream stream(spilled.path, std::ios::binary);
            if (!stream)
            {
                throw std::runtime_error("failed to open spilled quotient dependency: " + spilled.path);
            }
            try
            {
                return algebra::read_field_elements_binary(
                    stream, spilled.size);
            }
            catch (const std::exception &error)
            {
                throw std::runtime_error(
                    "failed to read spilled quotient dependency: " + spilled.path + ": " + error.what());
            }
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

        std::shared_ptr<const algebra::Polynomial>
        load_spilled_opening_polynomial(
            const ProtocolContext &context,
            const TraceArtifacts::SpilledEvaluationPolynomial &spilled,
            const std::string &name)
        {
            auto polynomial = algebra::Polynomial::from_shared_evaluations(
                name,
                std::make_shared<const std::vector<FieldElement>>(
                    load_spilled_values(spilled)),
                context_domain_by_name(context, spilled.domain_name));
            polynomial.set_zero_domain_blinder(
                spilled.zero_domain_blinder);
            return std::make_shared<const algebra::Polynomial>(
                std::move(polynomial));
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
                // Z_D*R tail, so the interpolation coset must retain the
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

        void coset_evaluate_coefficients_into(
            const mcl::Fr *coefficients,
            std::size_t coefficient_count,
            std::size_t target_size,
            const FieldElement &shift,
            const std::string &label,
            const std::vector<mcl::Fr> *shift_powers,
            FieldElement *output,
            bool preserve_even_odd_split_output = false)
        {
            if (coefficient_count > target_size)
            {
                throw std::runtime_error(
                    "polynomial degree exceeds quotient coset size: " + label);
            }

            if (target_size == 0)
            {
                return;
            }

            if (output == nullptr)
            {
                throw std::runtime_error(
                    "quotient coset evaluation received null output: " + label);
            }

            auto *work =
                FieldElement::mutable_native_array_data(output);

            if (coefficient_count == 0)
            {
                for (std::size_t i = 0; i < target_size; ++i)
                {
                    work[i].clear();
                }
                return;
            }

            if (coefficients == nullptr)
            {
                throw std::runtime_error(
                    "quotient coset evaluation received null coefficients: " + label);
            }

            const auto *usable_shift_powers =
                shift_powers != nullptr &&
                        shift_powers->size() >= coefficient_count
                    ? shift_powers
                    : nullptr;

            if (target_size >= 2 && coefficient_count <= target_size / 2)
            {
                const auto half_size = target_size / 2;

                std::vector<mcl::Fr> even_storage;
                std::vector<mcl::Fr> odd_storage;

                mcl::Fr *even_work = nullptr;
                mcl::Fr *odd_work = nullptr;

                if (preserve_even_odd_split_output)
                {
                    even_work = work;
                    odd_work = work + half_size;
                }
                else
                {
                    even_storage.resize(half_size);
                    odd_storage.resize(half_size);

                    even_work = even_storage.data();
                    odd_work = odd_storage.data();
                }

                if (usable_shift_powers != nullptr)
                {
                    const auto root =
                        FieldElement::root_of_unity(
                            target_size);

                    algebra::split_coset_coefficients_native(
                        coefficients,
                        usable_shift_powers->data(),
                        coefficient_count,
                        root,
                        even_work,
                        odd_work,
                        quotient_transform_worker_cap);
                }
                else
                {
                    const auto root =
                        FieldElement::root_of_unity(
                            target_size);

                    // Fuse coefficient copy and geometric scaling directly
                    // into the two paired-transform lanes.  This retains the
                    // existing bounded parallel implementation and avoids a
                    // persistent multi-GiB shift-power table.
                    algebra::
                        copy_scaled_by_geometric_powers_native(
                            coefficients,
                            even_work,
                            coefficient_count,
                            shift,
                            quotient_transform_worker_cap);

                    algebra::
                        copy_scaled_by_geometric_powers_native(
                            coefficients,
                            odd_work,
                            coefficient_count,
                            shift * root,
                            quotient_transform_worker_cap);
                }

                for (std::size_t i = coefficient_count;
                     i < half_size;
                     ++i)
                {
                    even_work[i].clear();
                    odd_work[i].clear();
                }

                algebra::ntt_pair_in_place(
                    even_work,
                    odd_work,
                    half_size,
                    quotient_transform_worker_cap);

                if (!preserve_even_odd_split_output)
                {
                    algebra::interleave_native_lanes(
                        even_work,
                        odd_work,
                        half_size,
                        work,
                        quotient_transform_worker_cap);
                }

                return;
            }

            if (preserve_even_odd_split_output)
            {
                throw std::runtime_error(
                    "split-lane quotient output did not enter the paired transform path: " +
                    label);
            }

            if (usable_shift_powers != nullptr)
            {
                parallel_for_chunks_static(
                    coefficient_count,
                    std::size_t{1} << 18,
                    quotient_transform_worker_cap,
                    [&](std::size_t begin,
                        std::size_t end)
                    {
                        for (std::size_t i = begin;
                             i < end;
                             ++i)
                        {
                            mcl::Fr::mul(
                                work[i],
                                coefficients[i],
                                (*usable_shift_powers)[i]);
                        }
                    });
            }
            else
            {
                algebra::
                    copy_scaled_by_geometric_powers_native(
                        coefficients,
                        work,
                        coefficient_count,
                        shift,
                        quotient_transform_worker_cap);
            }

            for (std::size_t i = coefficient_count;
                 i < target_size;
                 ++i)
            {
                work[i].clear();
            }

            ntt_plan(target_size)->ntt(work);
        }

        void coset_evaluate_polynomial_into(
            const algebra::Polynomial &polynomial,
            std::size_t target_size,
            const FieldElement &shift,
            const std::string &label,
            const std::vector<mcl::Fr> *shift_powers,
            FieldElement *output,
            bool preserve_even_odd_split_output = false)
        {
            auto &coefficient_work =
                quotient_coefficient_work;

            auto active_coefficient_size =
                [](const std::vector<mcl::Fr> &values)
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
                const auto &coefficients =
                    polynomial.values();

                coefficient_count =
                    coefficients.size();

                while (coefficient_count > 1 && coefficients[coefficient_count - 1]
                                                    .is_zero())
                {
                    --coefficient_count;
                }

                if (coefficient_count > target_size)
                {
                    throw std::runtime_error(
                        "polynomial degree exceeds quotient coset size: " + label);
                }

                coset_evaluate_coefficients_into(
                    FieldElement::native_array_data(
                        coefficients.data()),
                    coefficient_count,
                    target_size,
                    shift,
                    label,
                    shift_powers,
                    output,
                    preserve_even_odd_split_output);

                return;
            }

            if (polynomial.domain == nullptr)
            {
                throw std::runtime_error(
                    "evaluation polynomial is missing its domain: " + polynomial.name);
            }

            const auto &values =
                polynomial.values();

            copy_evaluation_values_to_native_domain(
                values,
                polynomial.domain->size,
                &coefficient_work,
                polynomial.name);

            algebra::intt_in_place(
                coefficient_work.data(),
                coefficient_work.size(),
                quotient_transform_worker_cap);

            if (polynomial.is_zero_domain_blinded())
            {
                const auto domain_size =
                    polynomial.domain->size;

                const auto required_size =
                    domain_size + polynomial.zero_domain_blinder.size();

                if (required_size > target_size)
                {
                    throw std::runtime_error(
                        "zero-domain-blinded polynomial exceeds "
                        "quotient coset size: " +
                        label);
                }

                coefficient_work.resize(
                    required_size);

                for (std::size_t i = domain_size;
                     i < required_size;
                     ++i)
                {
                    coefficient_work[i].clear();
                }

                for (std::size_t i = 0;
                     i < polynomial.zero_domain_blinder.size();
                     ++i)
                {
                    mcl::Fr::sub(
                        coefficient_work[i],
                        coefficient_work[i],
                        polynomial
                            .zero_domain_blinder[i]
                            .native());

                    mcl::Fr::add(
                        coefficient_work[domain_size + i],
                        coefficient_work[domain_size + i],
                        polynomial
                            .zero_domain_blinder[i]
                            .native());
                }
            }

            coefficient_count =
                active_coefficient_size(
                    coefficient_work);

            if (coefficient_count > target_size)
            {
                throw std::runtime_error(
                    "polynomial degree exceeds quotient coset size: " + label);
            }

            coset_evaluate_coefficients_into(
                coefficient_work.data(),
                coefficient_count,
                target_size,
                shift,
                label,
                shift_powers,
                output,
                preserve_even_odd_split_output);
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

            constexpr std::size_t
                kStreamingInverseScaleThreshold =
                    std::size_t{1} << 23;

            if (values.size() >=
                kStreamingInverseScaleThreshold)
            {
                algebra::
                    scale_by_geometric_powers_native(
                        work,
                        values.size(),
                        shift.inv(),
                        quotient_transform_worker_cap);
            }
            else
            {
                const auto powers =
                    inverse_shift_powers(
                        values.size(),
                        shift);

                parallel_for_chunks_static(
                    values.size(),
                    1ULL << 12,
                    quotient_transform_worker_cap,
                    [&](std::size_t begin,
                        std::size_t end)
                    {
                        for (std::size_t i = begin;
                             i < end;
                             ++i)
                        {
                            mcl::Fr::mul(
                                work[i],
                                work[i],
                                (*powers)[i]);
                        }
                    });
            }
            if (!progress_prefix.empty())
            {
                progress_note(std::string(progress_prefix) + ":shift_scale:done");
            }
            trim_field_coefficients(&values);
            return values;
        }

        std::vector<std::string> quotient_dependency_labels(
            const ProtocolContext &context,
            const std::string &quotient_name);

        struct QuotientCosetMemoryBudget
        {
            std::size_t total_bytes = 0;
            std::size_t available_bytes = 0;
            std::size_t reserve_bytes = 0;
            std::size_t budget_bytes = 0;
        };

        QuotientCosetMemoryBudget
        quotient_coset_memory_budget()
        {
            constexpr std::size_t
                kFallbackMaximumMemoryCosetBytes =
                    std::size_t{8} << 30;

            constexpr std::size_t
                kMaximumMemoryCosetBytes =
                    std::size_t{24} << 30;

            constexpr std::size_t
                kMinimumHostReserveBytes =
                    std::size_t{16} << 30;

            QuotientCosetMemoryBudget result;

            const auto kib_to_bytes =
                [](unsigned long long value_kib)
                -> std::size_t
            {
                const auto maximum =
                    std::numeric_limits<
                        std::size_t>::max();

                if (value_kib > static_cast<unsigned long long>(
                                    maximum / 1024ULL))
                {
                    return maximum;
                }

                return static_cast<std::size_t>(
                    value_kib * 1024ULL);
            };

            std::ifstream input("/proc/meminfo");

            if (input)
            {
                std::string line;

                while (std::getline(input, line))
                {
                    std::istringstream row(line);

                    std::string key;
                    unsigned long long value_kib = 0;
                    std::string unit;

                    if (!(row >> key >> value_kib))
                    {
                        continue;
                    }

                    row >> unit;

                    if (!unit.empty() && unit != "kB")
                    {
                        continue;
                    }

                    if (key == "MemTotal:")
                    {
                        result.total_bytes =
                            kib_to_bytes(value_kib);
                    }
                    else if (key == "MemAvailable:")
                    {
                        result.available_bytes =
                            kib_to_bytes(value_kib);
                    }
                }
            }

            if (result.total_bytes == 0)
            {
                const auto physical_pages =
                    ::sysconf(_SC_PHYS_PAGES);

                const auto page_size =
                    ::sysconf(_SC_PAGESIZE);

                if (physical_pages > 0 && page_size > 0)
                {
                    const auto pages =
                        static_cast<unsigned long long>(
                            physical_pages);

                    const auto bytes_per_page =
                        static_cast<unsigned long long>(
                            page_size);

                    const auto maximum =
                        static_cast<unsigned long long>(
                            std::numeric_limits<
                                std::size_t>::max());

                    result.total_bytes =
                        pages > maximum / bytes_per_page
                            ? std::numeric_limits<
                                  std::size_t>::max()
                            : static_cast<std::size_t>(
                                  pages * bytes_per_page);
                }
            }

            if (result.available_bytes == 0)
            {
                if (result.total_bytes != 0)
                {
                    result.budget_bytes =
                        std::min(
                            kFallbackMaximumMemoryCosetBytes,
                            result.total_bytes / 8);
                }

                return result;
            }

            result.reserve_bytes =
                std::max(
                    kMinimumHostReserveBytes,
                    result.total_bytes == 0
                        ? kMinimumHostReserveBytes
                        : result.total_bytes / 4);

            if (result.available_bytes <= result.reserve_bytes)
            {
                return result;
            }

            const auto safely_usable_bytes =
                result.available_bytes - result.reserve_bytes;

            result.budget_bytes =
                std::min(
                    kMaximumMemoryCosetBytes,
                    safely_usable_bytes / 2);

            return result;
        }

        std::vector<std::string> quotient_materialization_dependency_labels(
            const ProtocolContext &context,
            const std::string &quotient_name);

        class QuotientCosetSpillArena
        {
        public:
            struct Allocation
            {
                FieldElement *data = nullptr;
            };

            QuotientCosetSpillArena()
            {
                const auto root =
                    std::filesystem::current_path() / "runs" / ".quotient_coset_spill";
                std::filesystem::create_directories(root);
                if (::chmod(root.c_str(), S_IRWXU) != 0)
                {
                    throw std::runtime_error(
                        "failed to secure quotient coset spill directory: " + std::string(std::strerror(errno)));
                }
                auto template_path =
                    (root / "arena_XXXXXX").string();
                std::vector<char> path(
                    template_path.begin(), template_path.end());
                path.push_back('\0');
                fd_ = ::mkstemp(path.data());
                if (fd_ < 0)
                {
                    throw std::runtime_error(
                        "failed to create quotient coset spill arena: " + std::string(std::strerror(errno)));
                }
                if (::fchmod(fd_, S_IRUSR | S_IWUSR) != 0)
                {
                    const auto error = errno;
                    ::close(fd_);
                    fd_ = -1;
                    ::unlink(path.data());
                    throw std::runtime_error(
                        "failed to secure quotient coset spill arena: " + std::string(std::strerror(error)));
                }
                if (::unlink(path.data()) != 0)
                {
                    const auto error = errno;
                    ::close(fd_);
                    fd_ = -1;
                    throw std::runtime_error(
                        "failed to unlink quotient coset spill arena: " + std::string(std::strerror(error)));
                }
                const auto page_size_raw = ::sysconf(_SC_PAGESIZE);
                if (page_size_raw <= 0)
                {
                    ::close(fd_);
                    fd_ = -1;
                    throw std::runtime_error(
                        "failed to determine quotient spill page size");
                }
                page_size_ = static_cast<std::size_t>(page_size_raw);
            }

            QuotientCosetSpillArena(
                const QuotientCosetSpillArena &) = delete;
            QuotientCosetSpillArena &operator=(
                const QuotientCosetSpillArena &) = delete;

            ~QuotientCosetSpillArena()
            {
                for (const auto &chunk : chunks_)
                {
                    if (chunk.mapping != nullptr)
                    {
                        ::munmap(chunk.mapping, chunk.capacity);
                    }
                }
                if (fd_ >= 0)
                {
                    ::close(fd_);
                }
            }

            Allocation allocate(
                std::size_t byte_count,
                const std::string &label)
            {
                if (byte_count == 0)
                {
                    return {};
                }
                std::lock_guard<std::mutex> lock(mutex_);
                const auto aligned_bytes = align_up(byte_count);
                if (chunks_.empty() || align_up(chunks_.back().used) + aligned_bytes > chunks_.back().capacity)
                {
                    add_chunk(aligned_bytes, label);
                }
                auto &chunk = chunks_.back();
                const auto begin = align_up(chunk.used);
                chunk.used = begin + aligned_bytes;
                return {
                    .data = reinterpret_cast<FieldElement *>(
                        static_cast<std::uint8_t *>(chunk.mapping) + begin),
                };
            }

        private:
            struct Chunk
            {
                void *mapping = nullptr;
                std::size_t capacity = 0;
                std::size_t used = 0;
            };

            std::size_t align_up(std::size_t value) const
            {
                if (value > std::numeric_limits<std::size_t>::max() - (page_size_ - 1))
                {
                    throw std::runtime_error(
                        "quotient coset spill allocation overflow");
                }
                return (value + page_size_ - 1) / page_size_ * page_size_;
            }

            void add_chunk(
                std::size_t minimum_capacity,
                const std::string &label)
            {
                constexpr std::size_t kChunkBytes =
                    std::size_t{1} << 30;
                const auto capacity =
                    align_up(std::max(kChunkBytes, minimum_capacity));
                const auto offset = file_size_;
                if (capacity > static_cast<std::size_t>(
                                   std::numeric_limits<off_t>::max()) ||
                    offset > static_cast<std::size_t>(
                                 std::numeric_limits<off_t>::max()) -
                                 capacity)
                {
                    throw std::runtime_error(
                        "quotient coset spill arena exceeds off_t for " + label);
                }
                const auto allocation_error = ::posix_fallocate(
                    fd_,
                    static_cast<off_t>(offset),
                    static_cast<off_t>(capacity));
                if (allocation_error != 0)
                {
                    throw std::runtime_error(
                        "insufficient quotient coset spill arena resource for " + label + ": " + std::string(std::strerror(allocation_error)));
                }
                auto *mapping = ::mmap(
                    nullptr,
                    capacity,
                    PROT_READ | PROT_WRITE,
                    MAP_SHARED,
                    fd_,
                    static_cast<off_t>(offset));
                if (mapping == MAP_FAILED)
                {
                    const auto error = errno;
                    throw std::runtime_error(
                        "failed to map quotient coset spill arena for " + label + ": " + std::string(std::strerror(error)));
                }
                (void)::madvise(mapping, capacity, MADV_SEQUENTIAL);
                chunks_.push_back(Chunk{
                    .mapping = mapping,
                    .capacity = capacity,
                    .used = 0,
                });
                file_size_ += capacity;
            }

            std::mutex mutex_;
            std::vector<Chunk> chunks_;
            int fd_ = -1;
            std::size_t page_size_ = 0;
            std::size_t file_size_ = 0;
        };

        class QuotientCosetValues
        {
        public:
            struct PureZeroSemanticTag
            {
            };

            QuotientCosetValues(
                std::vector<FieldElement> values,
                std::shared_ptr<QuotientCosetSpillArena> spill_arena,
                const std::string &label)
                : spill_arena_(std::move(spill_arena)),
                  size_(values.size())
            {
                static_assert(
                    std::is_standard_layout_v<FieldElement>,
                    "file-backed quotient cosets require standard-layout field elements");

                static_assert(
                    std::is_trivially_destructible_v<FieldElement>,
                    "file-backed quotient cosets require trivially destructible field elements");

                if (spill_arena_ == nullptr || values.empty())
                {
                    memory_ = std::move(values);
                    return;
                }

                const auto byte_count =
                    size_ * sizeof(FieldElement);

                mapping_ =
                    spill_arena_->allocate(
                                    byte_count,
                                    label)
                        .data;

                std::uninitialized_copy(
                    values.begin(),
                    values.end(),
                    static_cast<FieldElement *>(mapping_));

                // Keep freshly produced pages eligible for the kernel's
                // adaptive file-cache policy. A synchronous flush followed
                // by MADV_DONTNEED forced every quotient dependency through
                // storage even when the complete working set fit in RAM.
                // Large instances remain bounded because these are unlinked
                // MAP_SHARED pages: the kernel can write back and reclaim
                // them under pressure, and the point loop explicitly drops
                // each consumed prefix.
                (void)::madvise(
                    mapping_,
                    byte_count,
                    MADV_SEQUENTIAL);

                std::vector<FieldElement>().swap(values);
            }

            QuotientCosetValues(
                std::size_t size,
                std::shared_ptr<QuotientCosetSpillArena> spill_arena,
                const std::string &label,
                bool even_odd_split_layout = false)
                : spill_arena_(std::move(spill_arena)),
                  size_(size),
                  split_lane_size_(
                      even_odd_split_layout
                          ? size / 2
                          : 0)
            {
                if (even_odd_split_layout &&
                    (size_ < 2 || (size_ & 1U) != 0U))
                {
                    throw std::runtime_error(
                        "split-lane quotient coset requires an even non-zero size");
                }

                static_assert(
                    std::is_standard_layout_v<FieldElement>,
                    "file-backed quotient cosets require standard-layout field elements");

                static_assert(
                    std::is_trivially_destructible_v<FieldElement>,
                    "file-backed quotient cosets require trivially destructible field elements");

                if (size_ == 0)
                {
                    return;
                }

                if (spill_arena_ == nullptr)
                {
                    memory_.resize(size_);
                    return;
                }

                const auto byte_count =
                    size_ * sizeof(FieldElement);

                mapping_ =
                    spill_arena_->allocate(
                                    byte_count,
                                    label)
                        .data;

                auto *elements =
                    static_cast<FieldElement *>(mapping_);

                std::uninitialized_default_construct_n(
                    elements,
                    size_);

                (void)::madvise(
                    mapping_,
                    byte_count,
                    MADV_SEQUENTIAL);
            }

            QuotientCosetValues(
                std::shared_ptr<const QuotientCosetValues> semantic_values,
                std::vector<FieldElement> zero_domain_blinder,
                std::shared_ptr<const std::vector<FieldElement>> coset_points,
                std::vector<FieldElement> zero_cycle)
                : semantic_values_(std::move(semantic_values)),
                  zero_domain_blinder_(
                      std::move(zero_domain_blinder)),
                  coset_points_(std::move(coset_points)),
                  zero_cycle_(std::move(zero_cycle)),
                  size_(semantic_values_ != nullptr
                            ? semantic_values_->size()
                            : 0)
            {
                if (semantic_values_ == nullptr || zero_domain_blinder_.empty() || zero_cycle_.empty())
                {
                    throw std::runtime_error(
                        "invalid lazy quotient blinder view");
                }
                if (zero_domain_blinder_.size() > 1 && (coset_points_ == nullptr || coset_points_->size() != size_))
                {
                    throw std::runtime_error(
                        "lazy quotient blinder view is missing coset points");
                }
            }

            QuotientCosetValues(
                PureZeroSemanticTag,
                std::size_t size,
                std::vector<FieldElement> zero_domain_blinder,
                std::shared_ptr<const std::vector<FieldElement>> coset_points,
                std::vector<FieldElement> zero_cycle)
                : pure_zero_semantic_(true),
                  zero_domain_blinder_(
                      std::move(zero_domain_blinder)),
                  coset_points_(std::move(coset_points)),
                  zero_cycle_(std::move(zero_cycle)),
                  size_(size)
            {
                if (size_ == 0 || zero_cycle_.empty() || size_ % zero_cycle_.size() != 0)
                {
                    throw std::runtime_error(
                        "invalid pure-zero lazy quotient blinder view");
                }
                if (zero_domain_blinder_.size() > 1 && (coset_points_ == nullptr || coset_points_->size() != size_))
                {
                    throw std::runtime_error(
                        "pure-zero lazy quotient blinder view is missing coset points");
                }
            }

            QuotientCosetValues(
                std::vector<std::pair<
                    std::shared_ptr<const QuotientCosetValues>,
                    FieldElement>>
                    linear_terms,
                std::vector<FieldElement> zero_domain_blinder_correction,
                std::shared_ptr<const std::vector<FieldElement>> coset_points,
                std::vector<FieldElement> zero_cycle)
                : linear_terms_(std::move(linear_terms)),
                  zero_domain_blinder_(
                      std::move(zero_domain_blinder_correction)),
                  coset_points_(std::move(coset_points)),
                  zero_cycle_(std::move(zero_cycle)),
                  size_(!linear_terms_.empty()
                            ? linear_terms_.front().first->size()
                            : 0)
            {
                if (linear_terms_.empty() || size_ == 0)
                {
                    throw std::runtime_error(
                        "invalid lazy linear quotient coset view");
                }
                linear_scale_is_one_.reserve(linear_terms_.size());
                for (const auto &[values, _] : linear_terms_)
                {
                    if (values == nullptr || values->size() != size_)
                    {
                        throw std::runtime_error(
                            "lazy linear quotient coset size mismatch");
                    }
                }
                for (const auto &[_, scale] : linear_terms_)
                {
                    linear_scale_is_one_.push_back(
                        scale == FieldElement::one());
                }
                if (!zero_domain_blinder_.empty() && zero_cycle_.empty())
                {
                    throw std::runtime_error(
                        "lazy linear quotient coset is missing zero cycle");
                }
                if (zero_domain_blinder_.size() > 1 && (coset_points_ == nullptr || coset_points_->size() != size_))
                {
                    throw std::runtime_error(
                        "lazy linear quotient coset is missing coset points");
                }
            }

            QuotientCosetValues(
                const QuotientCosetValues &) = delete;
            QuotientCosetValues &operator=(
                const QuotientCosetValues &) = delete;

            ~QuotientCosetValues() = default;

            std::size_t size() const
            {
                return size_;
            }

            FieldElement *mutable_data()
            {
                if (pure_zero_semantic_ || semantic_values_ != nullptr || !linear_terms_.empty())
                {
                    throw std::runtime_error(
                        "lazy quotient coset view "
                        "has no mutable contiguous data");
                }

                return mapping_ != nullptr
                           ? static_cast<FieldElement *>(
                                 mapping_)
                           : memory_.data();
            }

            const FieldElement *data() const
            {
                if (pure_zero_semantic_ ||
                    semantic_values_ != nullptr ||
                    !linear_terms_.empty())
                {
                    throw std::runtime_error(
                        "lazy quotient coset view has no contiguous data");
                }

                if (split_lane_size_ != 0)
                {
                    throw std::runtime_error(
                        "split-lane quotient coset has no logical contiguous data");
                }

                return mapping_ != nullptr
                           ? static_cast<const FieldElement *>(mapping_)
                           : memory_.data();
            }
            FieldElement operator[](
                std::size_t index) const
            {
                if (!pure_zero_semantic_ &&
                    semantic_values_ == nullptr &&
                    linear_terms_.empty())
                {
                    const auto *physical =
                        mapping_ != nullptr
                            ? static_cast<const FieldElement *>(mapping_)
                            : memory_.data();

                    if (split_lane_size_ != 0)
                    {
                        const auto physical_index =
                            index / 2 +
                            ((index & 1U) != 0U
                                 ? split_lane_size_
                                 : 0);

                        return physical[physical_index];
                    }

                    return physical[index];
                }
                FieldElement value = FieldElement::zero();
                if (semantic_values_ != nullptr)
                {
                    value = (*semantic_values_)[index];
                }
                else if (!pure_zero_semantic_)
                {
                    mcl::Fr acc;
                    acc.clear();
                    mcl::Fr product;
                    for (std::size_t term_index = 0;
                         term_index < linear_terms_.size();
                         ++term_index)
                    {
                        const auto &[values, scale] =
                            linear_terms_[term_index];
                        if (linear_scale_is_one_[term_index])
                        {
                            mcl::Fr::add(
                                acc,
                                acc,
                                (*values)[index].native());
                        }
                        else
                        {
                            mcl::Fr::mul(
                                product,
                                (*values)[index].native(),
                                scale.native());
                            mcl::Fr::add(acc, acc, product);
                        }
                    }
                    value = FieldElement::from_native(acc);
                }
                if (zero_domain_blinder_.empty())
                {
                    return value;
                }
                mcl::Fr blinder =
                    zero_domain_blinder_.back().native();
                if (zero_domain_blinder_.size() > 1)
                {
                    const auto &point = (*coset_points_)[index];
                    for (std::size_t coefficient =
                             zero_domain_blinder_.size() - 1;
                         coefficient-- > 0;)
                    {
                        mcl::Fr::mul(
                            blinder, blinder, point.native());
                        mcl::Fr::add(
                            blinder,
                            blinder,
                            zero_domain_blinder_[coefficient]
                                .native());
                    }
                }
                mcl::Fr contribution;
                mcl::Fr::mul(
                    contribution,
                    blinder,
                    zero_cycle_[index % zero_cycle_.size()]
                        .native());
                mcl::Fr out;
                mcl::Fr::add(
                    out, value.native(), contribution);
                return FieldElement::from_native(out);
            }

            FieldElement at(
                std::size_t index) const
            {
                if (index >= size_)
                {
                    throw std::out_of_range(
                        "quotient coset index out of range");
                }
                return (*this)[index];
            }

            void copy_logical_range(
                std::size_t begin,
                std::size_t count,
                FieldElement *output) const
            {
                if (output == nullptr ||
                    begin > size_ ||
                    count > size_ - begin)
                {
                    throw std::runtime_error(
                        "invalid quotient coset logical range");
                }

                if (count == 0)
                {
                    return;
                }

                if (!pure_zero_semantic_ &&
                    semantic_values_ == nullptr &&
                    linear_terms_.empty() &&
                    split_lane_size_ == 0)
                {
                    const auto *source =
                        mapping_ != nullptr
                            ? static_cast<
                                  const FieldElement *>(
                                  mapping_)
                            : memory_.data();

                    std::copy_n(
                        source + begin,
                        count,
                        output);

                    return;
                }

                parallel_for_chunks_static(
                    count,
                    std::size_t{1} << 14,
                    0,
                    [&](std::size_t local_begin,
                        std::size_t local_end)
                    {
                        for (std::size_t local =
                                 local_begin;
                             local < local_end;
                             ++local)
                        {
                            output[local] =
                                (*this)[begin + local];
                        }
                    });
            }

            std::vector<FieldElement> copy_values() const
            {
                if (!pure_zero_semantic_ &&
                    semantic_values_ == nullptr &&
                    linear_terms_.empty() &&
                    split_lane_size_ == 0)
                {
                    return std::vector<FieldElement>(
                        data(), data() + size_);
                }
                std::vector<FieldElement> out(
                    size_, FieldElement::zero());
                for (std::size_t index = 0;
                     index < size_; ++index)
                {
                    out[index] = (*this)[index];
                }
                return out;
            }

            bool file_backed() const
            {
                if (pure_zero_semantic_)
                {
                    return false;
                }
                if (mapping_ != nullptr || (semantic_values_ != nullptr && semantic_values_->file_backed()))
                {
                    return true;
                }
                return std::any_of(
                    linear_terms_.begin(),
                    linear_terms_.end(),
                    [](const auto &term)
                    {
                        return term.first->file_backed();
                    });
            }

            void discard_prefix(
                std::size_t end_index) const
            {
                if (pure_zero_semantic_)
                {
                    return;
                }
                if (semantic_values_ != nullptr)
                {
                    semantic_values_->discard_prefix(end_index);
                    return;
                }
                if (!linear_terms_.empty())
                {
                    for (const auto &[values, _] : linear_terms_)
                    {
                        values->discard_prefix(end_index);
                    }
                    return;
                }
                if (mapping_ == nullptr || end_index == 0)
                {
                    return;
                }
                const auto page_size_raw =
                    ::sysconf(_SC_PAGESIZE);
                if (page_size_raw <= 0)
                {
                    return;
                }
                const auto page_size =
                    static_cast<std::size_t>(page_size_raw);
                if (split_lane_size_ != 0)
                {
                    const auto logical_end =
                        std::min(end_index, size_);

                    const std::array<std::size_t, 2>
                        consumed_elements{
                            (logical_end + 1) / 2,
                            logical_end / 2,
                        };

                    const auto lane_bytes =
                        split_lane_size_ * sizeof(FieldElement);

                    if (lane_bytes % page_size != 0)
                    {
                        // MADV_DONTNEED is only an execution-side memory-pressure hint.
                        // Never discard a partially shared page.
                        return;
                    }

                    for (std::size_t lane = 0;
                         lane < 2;
                         ++lane)
                    {
                        const auto consumed_bytes =
                            consumed_elements[lane] *
                            sizeof(FieldElement);

                        const auto aligned_end =
                            consumed_bytes -
                            (consumed_bytes % page_size);

                        if (aligned_end <=
                            discarded_split_lane_bytes_[lane])
                        {
                            continue;
                        }

                        auto *begin =
                            static_cast<std::uint8_t *>(mapping_) +
                            lane * lane_bytes +
                            discarded_split_lane_bytes_[lane];

                        (void)::madvise(
                            begin,
                            aligned_end -
                                discarded_split_lane_bytes_[lane],
                            MADV_DONTNEED);

                        discarded_split_lane_bytes_[lane] =
                            aligned_end;
                    }

                    return;
                }
                const auto end_bytes =
                    std::min(end_index, size_) * sizeof(FieldElement);
                const auto aligned_end =
                    end_bytes - (end_bytes % page_size);
                if (aligned_end <= discarded_bytes_)
                {
                    return;
                }
                auto *begin =
                    static_cast<std::uint8_t *>(mapping_) + discarded_bytes_;
                (void)::madvise(
                    begin,
                    aligned_end - discarded_bytes_,
                    MADV_DONTNEED);
                discarded_bytes_ = aligned_end;
            }

        private:
            std::vector<FieldElement> memory_;
            std::shared_ptr<QuotientCosetSpillArena> spill_arena_;
            bool pure_zero_semantic_ = false;
            std::shared_ptr<const QuotientCosetValues>
                semantic_values_;
            std::vector<std::pair<
                std::shared_ptr<const QuotientCosetValues>,
                FieldElement>>
                linear_terms_;
            std::vector<bool> linear_scale_is_one_;
            std::vector<FieldElement> zero_domain_blinder_;
            std::shared_ptr<const std::vector<FieldElement>>
                coset_points_;
            std::vector<FieldElement> zero_cycle_;
            void *mapping_ = nullptr;
            std::size_t size_ = 0;

            // Non-zero only for the large CUDA t_edge physical layout:
            // [even lane][odd lane].
            std::size_t split_lane_size_ = 0;

            mutable std::size_t discarded_bytes_ = 0;

            mutable std::array<std::size_t, 2>
                discarded_split_lane_bytes_{0, 0};
        };

        using QuotientCosetValuesPtr =
            std::shared_ptr<const QuotientCosetValues>;

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
                        "quotient coset is too small for derived degree bound: " + label_ + " degree_bound=" + std::to_string(*degree_bound) + " target_size=" + std::to_string(target_size_));
                }
                // This is only a sizing bound for the edge alpha-power cache;
                // derive it from the formal nested family, never from the
                // checkpoint/parity nested family representation.
                std::size_t edge_head_count =
                    context_.model.output_layer.heads.size();
                for (const auto &layer : context_.model.hidden_layers)
                    edge_head_count += layer.heads.size();
                if (label_ == "t_edge" && edge_head_count >= 2)
                {
                    const auto power_count = std::size_t{2048} + std::size_t{1024} * edge_head_count;
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
                if (label_ == "t_FH")
                {
                    fh_parameters_.emplace(FHQuotientParameters{
                        .alpha_quot = challenges_.at("alpha_quot"),
                        .beta_feat = challenges_.at("beta_feat"),
                        .eta_feat = challenges_.at("eta_feat"),
                    });
                    fh_feature_encoding_terms_are_identically_zero_ =
                        full_graph_feature_identity_enabled(context_);
                }
                if (use_file_backed_coset_storage())
                {
                    const auto memory_budget =
                        quotient_coset_memory_budget();

                    coset_host_total_bytes_ =
                        memory_budget.total_bytes;

                    coset_host_available_at_start_bytes_ =
                        memory_budget.available_bytes;

                    coset_host_reserve_bytes_ =
                        memory_budget.reserve_bytes;

                    in_memory_coset_budget_bytes_ =
                        memory_budget.budget_bytes;

                    // Large CUDA t_edge already carries several GiB of
                    // transform scratch plus the CUDA NTT workspace on UMA.
                    // Do not consume the full generic 24 GiB coset allowance:
                    // leave headroom for transform staging and kernel/page-cache
                    // activity while retaining a bounded hot coset set.
                    if (label_ == "t_edge" &&
                        target_size_ >= (std::size_t{1} << 25) &&
                        util::route2_options().cuda_field_hotspots)
                    {
                        constexpr std::size_t
                            kCudaTedgeWorkspaceReserveBytes =
                                std::size_t{6} << 30;

                        in_memory_coset_budget_bytes_ =
                            in_memory_coset_budget_bytes_ >
                                    kCudaTedgeWorkspaceReserveBytes
                                ? in_memory_coset_budget_bytes_ -
                                      kCudaTedgeWorkspaceReserveBytes
                                : 0;

                        progress_note(
                            "prove:quotient:t_edge:"
                            "cuda_host_coset_budget_bytes=" +
                            std::to_string(
                                in_memory_coset_budget_bytes_) +
                            ":cuda_workspace_reserve_bytes=" +
                            std::to_string(
                                kCudaTedgeWorkspaceReserveBytes));
                    }

                    spill_arena_ =
                        std::make_shared<
                            QuotientCosetSpillArena>();
                }
            }

            static std::size_t
            cuda_t_edge_leaf_cache_slot_count(
                const TEdgeBoundProgram &program)
            {
                constexpr std::size_t
                    kBaselineLeafCacheSlots = 8;

                std::size_t slot_count =
                    kBaselineLeafCacheSlots;

                for (const auto &term :
                     program.terms)
                {
                    std::vector<std::string>
                        labels;

                    labels.reserve(
                        term.required_accesses
                            .size());

                    for (const auto access_index :
                         term.required_accesses)
                    {
                        if (access_index >=
                            program.accesses.size())
                        {
                            throw std::runtime_error(
                                "CUDA t_edge program "
                                "access index overflow");
                        }

                        const auto &label =
                            program
                                .accesses[access_index]
                                .label;

                        if (std::find(
                                labels.begin(),
                                labels.end(),
                                label) ==
                            labels.end())
                        {
                            labels.push_back(
                                label);
                        }
                    }

                    slot_count =
                        std::max(
                            slot_count,
                            labels.size());
                }

                return slot_count;
            }
            bool should_use_cuda_t_edge_engine()
                const
            {
                if (label_ != "t_edge" ||
                    !util::route2_options()
                         .cuda_field_hotspots ||
                    !algebra::
                        cuda_field_hotspots_available() ||
                    !cuda_t_edge_executor_available())
                {
                    return false;
                }

                if (edge_evaluation_plan_ ==
                    nullptr)
                {
                    throw std::runtime_error(
                        "CUDA t_edge route "
                        "is missing evaluation plan");
                }

                const auto program =
                    build_t_edge_bound_program(
                        context_,
                        challenges_,
                        trace_
                            .external_evaluations,
                        *edge_evaluation_plan_);

                const auto cache_slot_count =
                    cuda_t_edge_leaf_cache_slot_count(
                        program);

                if (!cuda_t_edge_executor_can_admit(
                        target_size_,
                        cache_slot_count))
                {
                    return false;
                }

                return target_size_ >=
                       (std::size_t{1} << 25);
            }

            void reset_cuda_t_edge_transient_leaf_state()
            {
                fast_coset_lookup_.clear();
                edge_access_sequence_.clear();

                {
                    std::lock_guard<std::mutex>
                        lock(
                            coset_cache_mutex_);

                    coset_cache_.clear();
                }

                semantic_coset_groups_.clear();

                semantic_coset_group_by_label_
                    .clear();

                in_memory_coset_bytes_.store(
                    0,
                    std::memory_order_relaxed);
            }

            std::vector<FieldElement>
            materialize_cuda_t_edge_coefficients()
            {
                if (edge_evaluation_plan_ ==
                    nullptr)
                {
                    throw std::runtime_error(
                        "CUDA t_edge route "
                        "is missing evaluation "
                        "plan");
                }

                auto program =
                    build_t_edge_bound_program(
                        context_,
                        challenges_,
                        trace_
                            .external_evaluations,
                        *edge_evaluation_plan_);

                const auto leaf_cache_slots =
                    cuda_t_edge_leaf_cache_slot_count(
                        program);

                auto executor =
                    create_cuda_t_edge_executor(
                        domain_->size,
                        target_size_,
                        shift_,
                        domain_->omega,
                        leaf_cache_slots);

                if (executor == nullptr)
                {
                    throw std::runtime_error(
                        "CUDA t_edge route "
                        "was selected but "
                        "executor is unavailable");
                }

                executor->prepare_program(
                    program);

                // CUDA t_edge materializes registered derived
                // linear views directly instead of going through
                // the CPU coset-prewarm path.  Those views must
                // use the same zero-domain blinder correction
                // geometry as the canonical materializer.
                seed_lazy_blinder_geometry();

                const auto invalid_slot =
                    std::numeric_limits<
                        std::uint32_t>::max();

                std::vector<std::uint32_t>
                    access_slots(
                        program.accesses.size(),
                        invalid_slot);

                std::vector<std::string>
                    slot_labels(
                        executor->cache_slot_count());

                std::vector<
                    std::vector<std::string>>
                    term_labels(
                        program.terms.size());

                std::unordered_map<
                    std::string,
                    std::size_t>
                    remaining_uses;

                std::unordered_map<
                    std::string,
                    std::vector<std::size_t>>
                    leaf_use_terms;

                std::unordered_map<
                    std::string,
                    std::size_t>
                    leaf_use_cursors;

                for (std::size_t term_index = 0;
                     term_index <
                     program.terms.size();
                     ++term_index)
                {
                    auto &labels =
                        term_labels[term_index];

                    for (const auto access_index :
                         program
                             .terms[term_index]
                             .required_accesses)
                    {
                        if (access_index >=
                            program.accesses
                                .size())
                        {
                            throw std::runtime_error(
                                "CUDA t_edge "
                                "program access "
                                "index overflow");
                        }

                        const auto &label =
                            program
                                .accesses[access_index]
                                .label;

                        if (std::find(
                                labels.begin(),
                                labels.end(),
                                label) ==
                            labels.end())
                        {
                            labels.push_back(
                                label);

                            ++remaining_uses[label];

                            leaf_use_terms[label]
                                .push_back(
                                    term_index);
                        }
                    }

                    if (labels.size() >
                        executor->cache_slot_count())
                    {
                        throw std::runtime_error(
                            "CUDA t_edge term "
                            "requires more leaf "
                            "slots than cache "
                            "capacity");
                    }
                }

                constexpr std::size_t
                    kUploadChunkElements =
                        std::size_t{1} << 20;

                std::vector<FieldElement>
                    staging(
                        std::min(
                            target_size_,
                            kUploadChunkElements));

                const auto clear_label_mapping =
                    [&](const std::string
                            &label)
                {
                    for (std::size_t access = 0;
                         access <
                         program.accesses
                             .size();
                         ++access)
                    {
                        if (program
                                .accesses[access]
                                .label ==
                            label)
                        {
                            access_slots[access] =
                                invalid_slot;
                        }
                    }
                };

                const auto bind_label_mapping =
                    [&](const std::string &label,
                        std::size_t slot)
                {
                    for (std::size_t access = 0;
                         access <
                         program.accesses
                             .size();
                         ++access)
                    {
                        if (program
                                .accesses[access]
                                .label ==
                            label)
                        {
                            access_slots[access] =
                                static_cast<
                                    std::uint32_t>(
                                    slot);
                        }
                    }
                };

                const auto
                    try_materialize_direct_gpu_leaf =
                        [&](const std::string &label,
                            std::size_t slot)
                    -> bool
                {
                    if (const auto *polynomial =
                            lookup_polynomial(
                                label);
                        polynomial != nullptr)
                    {
                        if (polynomial->basis ==
                                algebra::
                                    PolynomialBasis::
                                        Evaluation &&
                            polynomial->domain !=
                                nullptr &&
                            polynomial->domain->size ==
                                domain_->size &&
                            polynomial->domain->omega ==
                                domain_->omega)
                        {
                            const auto &values =
                                polynomial->values();

                            const auto *blinder =
                                polynomial
                                        ->zero_domain_blinder
                                        .empty()
                                    ? nullptr
                                    : FieldElement::
                                          native_array_data(
                                              polynomial
                                                  ->zero_domain_blinder
                                                  .data());

                            return executor->materialize_evaluation_leaf(
                                slot,
                                FieldElement::
                                    native_array_data(
                                        values.data()),
                                values.size(),
                                blinder,
                                polynomial
                                    ->zero_domain_blinder
                                    .size());
                        }

                        if (polynomial->basis ==
                            algebra::
                                PolynomialBasis::
                                    Coefficient)
                        {
                            const auto &coefficients =
                                polynomial->values();

                            return executor->materialize_coefficient_leaf(
                                slot,
                                FieldElement::
                                    native_array_data(
                                        coefficients.data()),
                                coefficients.size());
                        }
                    }

                    const auto spilled_it =
                        trace_.spilled_polynomials
                            .find(label);

                    if (spilled_it !=
                            trace_
                                .spilled_polynomials
                                .end() &&
                        spilled_it->second
                                .domain_name ==
                            domain_->name)
                    {
                        auto source_values =
                            load_spilled_values(
                                spilled_it->second);

                        const auto *blinder =
                            spilled_it->second
                                    .zero_domain_blinder
                                    .empty()
                                ? nullptr
                                : FieldElement::
                                      native_array_data(
                                          spilled_it
                                              ->second
                                              .zero_domain_blinder
                                              .data());

                        return executor->materialize_evaluation_leaf(
                            slot,
                            FieldElement::
                                native_array_data(
                                    source_values
                                        .data()),
                            source_values.size(),
                            blinder,
                            spilled_it->second
                                .zero_domain_blinder
                                .size());
                    }

                    return false;
                };
                const auto materialize_label =
                    [&](const std::string &label,
                        std::size_t slot)
                {
                    if (label !=
                            "__t_edge_boundary_l0" &&
                        try_materialize_direct_gpu_leaf(
                            label,
                            slot))
                    {
                        reset_cuda_t_edge_transient_leaf_state();
                        return;
                    }

                    QuotientCosetValuesPtr values;

                    if (label ==
                        "__t_edge_boundary_l0")
                    {
                        std::vector<FieldElement>
                            boundary_values(
                                domain_->size,
                                FieldElement::zero());

                        boundary_values[0] =
                            FieldElement::one();

                        auto boundary_polynomial =
                            algebra::Polynomial::
                                from_evaluations(
                                    label,
                                    std::move(
                                        boundary_values),
                                    domain_);

                        values =
                            materialize_polynomial_coset(
                                boundary_polynomial,
                                label);
                    }
                    else
                    {
                        values =
                            coset_values_ptr(
                                label);
                    }

                    if (values == nullptr ||
                        values->size() !=
                            target_size_)
                    {
                        throw std::runtime_error(
                            "CUDA t_edge "
                            "materialized leaf "
                            "has wrong size: " +
                            label);
                    }

                    for (std::size_t offset = 0;
                         offset < target_size_;)
                    {
                        const auto count =
                            std::min(
                                staging.size(),
                                target_size_ -
                                    offset);

                        values->copy_logical_range(
                            offset,
                            count,
                            staging.data());

                        executor->upload_leaf_chunk(
                            slot,
                            offset,
                            FieldElement::
                                native_array_data(
                                    staging
                                        .data()),
                            count);

                        offset += count;
                    }

                    values.reset();

                    reset_cuda_t_edge_transient_leaf_state();
                };

                cuda_t_edge_transient_leaf_mode_ =
                    true;

                try
                {
                    for (std::size_t term_index = 0;
                         term_index <
                         program.terms.size();
                         ++term_index)
                    {
                        const auto &required =
                            term_labels[term_index];

                        for (const auto &label :
                             required)
                        {
                            std::size_t
                                existing_slot =
                                    slot_labels
                                        .size();

                            for (std::size_t slot = 0;
                                 slot <
                                 slot_labels
                                     .size();
                                 ++slot)
                            {
                                if (slot_labels[slot] ==
                                    label)
                                {
                                    existing_slot =
                                        slot;

                                    break;
                                }
                            }

                            if (existing_slot <
                                slot_labels.size())
                            {
                                bind_label_mapping(
                                    label,
                                    existing_slot);

                                continue;
                            }

                            std::size_t selected_slot =
                                slot_labels.size();

                            for (std::size_t slot = 0;
                                 slot <
                                 slot_labels
                                     .size();
                                 ++slot)
                            {
                                if (slot_labels[slot]
                                        .empty())
                                {
                                    selected_slot =
                                        slot;

                                    break;
                                }
                            }

                            if (selected_slot ==
                                slot_labels.size())
                            {
                                std::size_t
                                    selected_next_use = 0;

                                for (std::size_t slot = 0;
                                     slot <
                                     slot_labels
                                         .size();
                                     ++slot)
                                {
                                    const auto
                                        &candidate =
                                            slot_labels[slot];

                                    if (std::find(
                                            required
                                                .begin(),
                                            required
                                                .end(),
                                            candidate) !=
                                        required.end())
                                    {
                                        continue;
                                    }

                                    std::size_t
                                        candidate_next_use =
                                            std::numeric_limits<
                                                std::size_t>::
                                                max();

                                    const auto use_it =
                                        leaf_use_terms.find(
                                            candidate);

                                    if (use_it !=
                                        leaf_use_terms.end())
                                    {
                                        const auto cursor_it =
                                            leaf_use_cursors.find(
                                                candidate);

                                        const auto cursor =
                                            cursor_it ==
                                                    leaf_use_cursors
                                                        .end()
                                                ? std::size_t{0}
                                                : cursor_it
                                                      ->second;

                                        if (cursor <
                                            use_it->second
                                                .size())
                                        {
                                            candidate_next_use =
                                                use_it->second[cursor];
                                        }
                                    }

                                    if (selected_slot ==
                                            slot_labels
                                                .size() ||
                                        candidate_next_use >
                                            selected_next_use)
                                    {
                                        selected_slot =
                                            slot;

                                        selected_next_use =
                                            candidate_next_use;
                                    }
                                }
                            }

                            if (selected_slot ==
                                slot_labels.size())
                            {
                                throw std::runtime_error(
                                    "CUDA t_edge "
                                    "leaf cache cannot "
                                    "evict a slot");
                            }

                            if (!slot_labels[selected_slot]
                                     .empty())
                            {
                                clear_label_mapping(
                                    slot_labels[selected_slot]);

                                executor->release_leaf(
                                    selected_slot);
                            }

                            materialize_label(
                                label,
                                selected_slot);

                            slot_labels[selected_slot] =
                                label;

                            bind_label_mapping(
                                label,
                                selected_slot);
                        }

                        executor->accumulate_term(
                            term_index,
                            access_slots);

                        for (const auto &label :
                             required)
                        {
                            auto found =
                                remaining_uses
                                    .find(label);

                            if (found !=
                                    remaining_uses
                                        .end() &&
                                found->second != 0)
                            {
                                --found->second;
                            }

                            const auto use_it =
                                leaf_use_terms.find(
                                    label);

                            if (use_it ==
                                leaf_use_terms.end())
                            {
                                throw std::runtime_error(
                                    "CUDA t_edge leaf "
                                    "use schedule is missing "
                                    "a required label");
                            }

                            auto &cursor =
                                leaf_use_cursors[label];

                            if (cursor >=
                                    use_it->second
                                        .size() ||
                                use_it->second[cursor] !=
                                    term_index)
                            {
                                throw std::runtime_error(
                                    "CUDA t_edge leaf "
                                    "use schedule is out "
                                    "of sync");
                            }

                            ++cursor;
                        }
                    }

                    std::vector<FieldElement>
                        output(
                            target_size_,
                            FieldElement::zero());

                    executor->finalize_coefficients(
                        output.data(),
                        output.size());

                    trim_field_coefficients(
                        &output);

                    cuda_t_edge_transient_leaf_mode_ =
                        false;

                    reset_cuda_t_edge_transient_leaf_state();
                    return output;
                }
                catch (...)
                {
                    cuda_t_edge_transient_leaf_mode_ =
                        false;

                    reset_cuda_t_edge_transient_leaf_state();

                    throw;
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
                auto evaluate_at_index = [&](
                                             std::size_t i,
                                             const FieldElement &z,
                                             const QuotientBoundaryValues *boundary_values = nullptr)
                {
                    const auto omega_z = z * domain_->omega;
                    const auto rotated_index = (i + rotation_step_) & (target_size_ - 1U);
                    std::size_t edge_access_cursor = 0;
                    const auto eval = [&](const std::string &name, const FieldElement &point) -> FieldElement
                    {
                        if (label_ == "t_edge")
                        {
                            if (record_edge_access_sequence_)
                            {
                                const auto rotated = point == omega_z;
                                if (!rotated && point != z)
                                {
                                    throw std::runtime_error(
                                        "t_edge quotient access-plan received "
                                        "an unsupported evaluation point for " +
                                        name);
                                }
                                const auto access_index =
                                    rotated ? rotated_index : i;
                                const auto *direct_values =
                                    bind_t_edge_fast_coset(name);
                                edge_access_sequence_.push_back(
                                    TEdgeCosetAccess{
                                        .name = name,
                                        .values = direct_values,
                                        .rotated = rotated,
                                    });
                                return (*direct_values)[access_index];
                            }
                            if (!edge_access_sequence_.empty())
                            {
                                if (edge_access_cursor >= edge_access_sequence_.size())
                                {
                                    throw std::runtime_error(
                                        "t_edge quotient access-plan overflow");
                                }
                                const auto &access =
                                    edge_access_sequence_[edge_access_cursor++];
                                const auto access_index =
                                    access.rotated ? rotated_index : i;
                                if (access.values == nullptr || access.values->size() != target_size_)
                                {
                                    throw std::runtime_error(
                                        "t_edge quotient access-plan contains an invalid bound dependency: " + access.name);
                                }
                                return (*access.values)[access_index];
                            }
                        }
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
                        if (fh_coset_slots_.ready())
                        {
                            const auto boundary = boundary_values != nullptr
                                                      ? *boundary_values
                                                      : fh_boundary_values(z);
                            const FHQuotientValues values{
                                .q_table =
                                    (*fh_coset_slots_.q_table)[i],
                                .q_query =
                                    (*fh_coset_slots_.q_query)[i],
                                .table = (*fh_coset_slots_.table)[i],
                                .query = (*fh_coset_slots_.query)[i],
                                .multiplicity =
                                    (*fh_coset_slots_.multiplicity)[i],
                                .j_table =
                                    (*fh_coset_slots_.j_table)[i],
                                .j_query =
                                    (*fh_coset_slots_.j_query)[i],
                                .accumulator =
                                    (*fh_coset_slots_.accumulator)[i],
                                .accumulator_next =
                                    (*fh_coset_slots_.accumulator)[rotated_index],
                            };
                            std::optional<FHFeatureEncodingValues>
                                feature_encoding_values;
                            if (!fh_feature_encoding_terms_are_identically_zero_)
                            {
                                feature_encoding_values.emplace(
                                    FHFeatureEncodingValues{
                                        .table_row =
                                            (*fh_coset_slots_.table_row)[i],
                                        .table_column =
                                            (*fh_coset_slots_.table_column)[i],
                                        .table_feature =
                                            (*fh_coset_slots_.table_feature)[i],
                                        .query_row =
                                            (*fh_coset_slots_.query_row)[i],
                                        .query_column =
                                            (*fh_coset_slots_.query_column)[i],
                                        .query_feature =
                                            (*fh_coset_slots_.query_feature)[i],
                                    });
                            }
                            return evaluate_t_fh_from_values(
                                *fh_parameters_,
                                values,
                                boundary,
                                feature_encoding_values.has_value()
                                    ? &*feature_encoding_values
                                    : nullptr);
                        }
                        return evaluate_t_fh(
                            context_,
                            challenges_,
                            eval,
                            z,
                            nullptr,
                            boundary_values,
                            full_graph_feature_identity_enabled(context_));
                    }
                    else if (label_ == "t_edge")
                    {
                        const auto value = evaluate_t_edge(
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
                        if (!record_edge_access_sequence_ && !edge_access_sequence_.empty() && edge_access_cursor != edge_access_sequence_.size())
                        {
                            throw std::runtime_error(
                                "t_edge quotient access-plan underflow");
                        }
                        return value;
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

                const bool cuda_t_edge_route =
                    should_use_cuda_t_edge_engine();

                std::vector<FieldElement>
                    quotient_values;

                std::vector<FieldElement>
                    cuda_t_edge_coefficients;

                if (cuda_t_edge_route)
                {
                    progress_note(
                        "prove:quotient:t_edge:"
                        "cuda_device_resident:start");

                    cuda_t_edge_coefficients =
                        materialize_cuda_t_edge_coefficients();

                    progress_note(
                        "prove:quotient:t_edge:"
                        "cuda_device_resident:done");
                }
                else
                {
                    quotient_values.assign(
                        target_size_,
                        FieldElement::zero());
                }
                // Warm the dependency cache before parallel point evaluation; after this the worker
                // threads only read the cached coset vectors.
                if (label_ == "t_edge" &&
                    !cuda_t_edge_route)
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
                    // Bind the evaluator's original dependency names after
                    // their source columns are warm.  Derived edge names
                    // resolve here to the same lazy, blinder-corrected coset
                    // views used by the generic resolver, so the point loop
                    // can follow direct pointers instead of reparsing
                    // layer/head suffixes and challenge names at every point.
                    seed_fast_coset_lookup(dependency_labels);
                    require_t_edge_fast_coset_lookup(dependency_labels);
                    edge_access_sequence_.clear();
                    edge_access_plan_late_bind_count_ = 0;
                    record_edge_access_sequence_ = true;
                    try
                    {
                        (void)evaluate_at_index(0, shift_);
                    }
                    catch (...)
                    {
                        record_edge_access_sequence_ = false;
                        edge_access_sequence_.clear();
                        throw;
                    }
                    record_edge_access_sequence_ = false;
                    if (edge_access_sequence_.empty())
                    {
                        throw std::runtime_error(
                            "t_edge quotient access-plan is empty");
                    }
                    const auto null_access_count = static_cast<std::size_t>(
                        std::count_if(
                            edge_access_sequence_.begin(),
                            edge_access_sequence_.end(),
                            [](const TEdgeCosetAccess &access)
                            { return access.values == nullptr; }));
                    if (null_access_count != 0)
                    {
                        throw std::runtime_error(
                            "t_edge quotient access-plan retained null derived fallbacks: " + std::to_string(null_access_count));
                    }
                    progress_note(
                        "prove:quotient:t_edge:access_plan:entries=" + std::to_string(edge_access_sequence_.size()) + ":bound=" + std::to_string(edge_access_sequence_.size()) + ":late_bound=" + std::to_string(edge_access_plan_late_bind_count_) + ":null_derived_fallback=0");
                    progress_note(
                        "prove:quotient:t_edge:coset_prewarm:done:ms=" + std::to_string(elapsed_ms(prewarm_start, Clock::now())));
                    // The generic evaluator is the protocol source of truth. A
                    // hand-unrolled route must reproduce evaluate_t_edge exactly
                    // before it can enter the formal path; otherwise its
                    // commitment polynomial could disagree with verifier claims.
                }
                else if (label_ == "t_FH")
                {
                    // The point evaluator runs up to 32 workers.  Letting those
                    // workers discover a cold FH dependency independently causes
                    // a cache-miss stampede: every worker can start the same
                    // full-domain NTT before the first result reaches
                    // coset_cache_.  Materialize the small canonical dependency
                    // registry in order first.  Individual NTTs retain their
                    // internal parallelism, while the evaluation workers become
                    // read-only cache consumers.
                    const auto prewarm_start = Clock::now();
                    const auto dependency_labels =
                        quotient_materialization_dependency_labels(
                            context_, label_);
                    progress_note("prove:quotient:t_FH:coset_prewarm:start");
                    for (const auto &dependency_label : dependency_labels)
                    {
                        (void)coset_values_ptr(dependency_label);
                    }
                    seed_fast_coset_lookup(dependency_labels);
                    bind_fh_coset_slots();
                    progress_note(
                        "prove:quotient:t_FH:coset_prewarm:done:ms=" + std::to_string(
                                                                           elapsed_ms(prewarm_start, Clock::now())));
                }
                else if (label_ != "t_edge")
                {
                    // Small-domain quotient evaluators repeatedly request the
                    // same named coset vectors.  Resolve every authenticated
                    // dependency once before the point loop so evaluation is
                    // a read-only pointer lookup instead of millions of
                    // mutex-protected string-map probes.
                    const auto prewarm_start = Clock::now();
                    const auto dependency_labels =
                        quotient_dependency_labels(context_, label_);
                    if (label_ == "t_N")
                    {
                        progress_note(
                            "prove:quotient:t_N:coset_prewarm:start");
                    }
                    for (const auto &dependency_label : dependency_labels)
                    {
                        (void)coset_values_ptr(dependency_label);
                    }
                    seed_fast_coset_lookup(dependency_labels);
                    if (label_ == "t_N")
                    {
                        progress_note(
                            "prove:quotient:t_N:coset_prewarm:done:ms=" + std::to_string(elapsed_ms(
                                                                              prewarm_start, Clock::now())));
                    }
                }
                if (label_ == "t_edge" &&
                    !cuda_t_edge_route)
                {
                    progress_note(
                        "prove:quotient:t_edge:"
                        "evaluate:start");
                }
                else if (label_ == "t_FH")
                {
                    progress_note("prove:quotient:t_FH:evaluate:start");
                }
                else if (label_ == "t_N")
                {
                    progress_note("prove:quotient:t_N:evaluate:start");
                }
                const auto point_evaluation_start =
                    Clock::now();

                if (!cuda_t_edge_route)
                {
                    std::vector<FieldElement>
                        quotient_zero_inverses;

                    const auto boundary_one =
                        FieldElement::one();

                    const auto boundary_last_point =
                        domain_->point_at(domain_->size - 1);

                    const auto boundary_zero_step =
                        root.pow(domain_->size);

                    const auto boundary_shifted_zero =
                        shift_.pow(domain_->size);

                    if (label_ == "t_edge" || label_ == "t_FH")
                    {
                        if (target_size_ % domain_->size != 0)
                        {
                            throw std::runtime_error(
                                "quotient coset size is not a domain multiple: " + label_);
                        }

                        const auto zero_cycle_size =
                            target_size_ / domain_->size;

                        quotient_zero_inverses.reserve(
                            zero_cycle_size);

                        auto zero_power =
                            boundary_shifted_zero;

                        for (std::size_t residue = 0;
                             residue < zero_cycle_size;
                             ++residue)
                        {
                            quotient_zero_inverses.push_back(
                                (zero_power - boundary_one).inv());

                            zero_power *= boundary_zero_step;
                        }
                    }

                    constexpr std::size_t kInverseBlockSize = 8192;

                    const auto evaluate_range =
                        [&](std::size_t begin,
                            std::size_t end,
                            std::vector<mcl::Fr> *denominator_products)
                    {
                        if (begin >= end)
                        {
                            return;
                        }

                        auto z =
                            shift_ * root.pow(
                                         static_cast<std::uint64_t>(
                                             begin));

                        if (label_ == "t_edge" || label_ == "t_FH")
                        {
                            if (denominator_products == nullptr)
                            {
                                throw std::runtime_error(
                                    "quotient evaluation is missing inverse scratch");
                            }

                            auto zero_power =
                                boundary_shifted_zero * boundary_zero_step.pow(
                                                            static_cast<std::uint64_t>(
                                                                begin));

                            for (std::size_t block_begin = begin;
                                 block_begin < end;
                                 block_begin += kInverseBlockSize)
                            {
                                const auto block_end =
                                    std::min(
                                        end,
                                        block_begin + kInverseBlockSize);

                                denominator_products->resize(
                                    block_end - block_begin);

                                auto denominator_z = z;

                                for (std::size_t i = block_begin;
                                     i < block_end;
                                     ++i)
                                {
                                    mcl::Fr first_denominator;
                                    mcl::Fr last_denominator;

                                    mcl::Fr::sub(
                                        first_denominator,
                                        denominator_z.native(),
                                        boundary_one.native());

                                    mcl::Fr::sub(
                                        last_denominator,
                                        denominator_z.native(),
                                        boundary_last_point.native());

                                    mcl::Fr::mul(
                                        (*denominator_products)[i - block_begin],
                                        first_denominator,
                                        last_denominator);

                                    denominator_z *= root;
                                }

                                batch_invert_native_in_place(
                                    denominator_products);

                                for (std::size_t i = block_begin;
                                     i < block_end;
                                     ++i)
                                {
                                    const auto first_denominator =
                                        z - boundary_one;

                                    const auto last_denominator =
                                        z - boundary_last_point;

                                    const auto inverse_product =
                                        FieldElement::from_native(
                                            (*denominator_products)[i - block_begin]);

                                    const auto zero =
                                        zero_power - boundary_one;

                                    const auto zero_scaled =
                                        zero * domain_->inv_size;

                                    const QuotientBoundaryValues boundary{
                                        .first =
                                            zero_scaled * last_denominator * inverse_product,

                                        .last =
                                            zero_scaled * boundary_last_point * first_denominator * inverse_product,

                                        .zero = zero,

                                        .zero_inverse =
                                            quotient_zero_inverses[i % quotient_zero_inverses
                                                                           .size()],
                                    };

                                    quotient_values[i] =
                                        evaluate_at_index(
                                            i,
                                            z,
                                            &boundary);

                                    z *= root;
                                    zero_power *=
                                        boundary_zero_step;
                                }
                            }

                            return;
                        }

                        for (std::size_t i = begin;
                             i < end;
                             ++i)
                        {
                            quotient_values[i] =
                                evaluate_at_index(i, z);

                            z *= root;
                        }
                    };

                    const auto cpu_count =
                        std::max<std::size_t>(
                            1,
                            std::thread::hardware_concurrency());

                    const bool run_parallel =
                        cpu_count > 1 && target_size_ >= 4096;

                    if (run_parallel)
                    {
                        const bool bounded_file_backed_evaluation =
                            file_backed_coset_count_.load(
                                std::memory_order_relaxed) != 0;

                        const auto evaluation_block_size =
                            bounded_file_backed_evaluation
                                ? std::size_t{1} << 15
                                : target_size_;

                        if (use_file_backed_coset_storage())
                        {
                            progress_note(
                                "prove:quotient:" + label_ + ":coset_storage:tiered" + ":memory_bytes=" + std::to_string(in_memory_coset_bytes_.load(std::memory_order_relaxed)) + ":memory_budget=" + std::to_string(in_memory_coset_budget_bytes_) + ":host_total_at_start=" + std::to_string(coset_host_total_bytes_) + ":host_available_at_start=" + std::to_string(coset_host_available_at_start_bytes_) + ":host_reserve=" + std::to_string(coset_host_reserve_bytes_) + ":files=" + std::to_string(file_backed_coset_count_.load(std::memory_order_relaxed)) + ":bytes=" + std::to_string(file_backed_coset_bytes_.load(std::memory_order_relaxed)) + ":lazy_blinder_views=" + std::to_string(lazy_blinder_coset_count_.load(std::memory_order_relaxed)) + ":lazy_blinder_bytes_avoided=" + std::to_string(lazy_blinder_coset_bytes_avoided_.load(std::memory_order_relaxed)) + ":split_lane_cosets=" +
                                std::to_string(
                                    split_lane_coset_count_.load(
                                        std::memory_order_relaxed)) +
                                ":split_lane_temp_bytes_avoided=" +
                                std::to_string(
                                    split_lane_temporary_bytes_avoided_.load(
                                        std::memory_order_relaxed)) +
                                ":split_lane_interleave_bytes_avoided=" +
                                std::to_string(
                                    split_lane_interleave_bytes_avoided_.load(
                                        std::memory_order_relaxed)) +
                                ":evaluation_block=" + std::to_string(evaluation_block_size));
                        }

                        const auto min_points_per_task =
                            bounded_file_backed_evaluation
                                ? (label_ == "t_edge"
                                       ? std::size_t{1024}
                                       : std::size_t{2048})
                                : (label_ == "t_edge"
                                       ? std::size_t{1024}
                                       : std::size_t{32768});

                        if (bounded_file_backed_evaluation)
                        {
                            const auto persistent_worker_count =
                                std::min<std::size_t>(
                                    std::min<std::size_t>(
                                        cpu_count,
                                        32),
                                    std::max<std::size_t>(
                                        1,
                                        evaluation_block_size / min_points_per_task));

                            const auto evaluation_block_count =
                                (target_size_ + evaluation_block_size - 1) / evaluation_block_size;

                            const auto discard_set =
                                file_backed_coset_discard_set();

                            // file_backed_coset_count_ is a cumulative execution counter.
                            // A file-backed value may have been created and released before
                            // this live cache snapshot.  An empty discard set is therefore
                            // valid and simply means that there are no currently reachable
                            // file-backed pages to discard.
                            progress_note(
                                "prove:quotient:" + label_ + ":evaluation_scheduler" + ":mode=persistent_block_team" + ":workers=" + std::to_string(persistent_worker_count) + ":blocks=" + std::to_string(evaluation_block_count) + ":discard_views=" + std::to_string(discard_set.size()));

                            std::barrier<> block_barrier(
                                static_cast<std::ptrdiff_t>(
                                    persistent_worker_count));

                            std::atomic<int> start_state{0};
                            std::atomic<bool> worker_failed{false};

                            std::mutex worker_error_mutex;
                            std::exception_ptr worker_error;

                            std::vector<std::thread> workers;
                            workers.reserve(
                                persistent_worker_count);

                            try
                            {
                                for (std::size_t worker = 0;
                                     worker < persistent_worker_count;
                                     ++worker)
                                {
                                    workers.emplace_back(
                                        [&, worker]()
                                        {
                                            while (start_state.load(
                                                       std::memory_order_acquire) == 0)
                                            {
                                                std::this_thread::yield();
                                            }

                                            if (start_state.load(
                                                    std::memory_order_acquire) < 0)
                                            {
                                                return;
                                            }

                                            std::vector<mcl::Fr>
                                                denominator_products;

                                            for (std::size_t evaluation_begin = 0;
                                                 evaluation_begin < target_size_;
                                                 evaluation_begin +=
                                                 evaluation_block_size)
                                            {
                                                const auto evaluation_end =
                                                    std::min(
                                                        target_size_,
                                                        evaluation_begin + evaluation_block_size);

                                                const auto evaluation_count =
                                                    evaluation_end - evaluation_begin;

                                                const auto chunk_size =
                                                    (evaluation_count + persistent_worker_count - 1) / persistent_worker_count;

                                                const auto begin =
                                                    evaluation_begin + worker * chunk_size;

                                                const auto end =
                                                    std::min(
                                                        evaluation_end,
                                                        begin + chunk_size);

                                                if (!worker_failed.load(
                                                        std::memory_order_acquire) &&
                                                    begin < end)
                                                {
                                                    try
                                                    {
                                                        evaluate_range(
                                                            begin,
                                                            end,
                                                            &denominator_products);
                                                    }
                                                    catch (...)
                                                    {
                                                        {
                                                            std::lock_guard<std::mutex>
                                                                lock(
                                                                    worker_error_mutex);

                                                            if (worker_error == nullptr)
                                                            {
                                                                worker_error =
                                                                    std::current_exception();
                                                            }
                                                        }

                                                        worker_failed.store(
                                                            true,
                                                            std::memory_order_release);
                                                    }
                                                }

                                                block_barrier.arrive_and_wait();

                                                if (worker == 0 && !worker_failed.load(
                                                                       std::memory_order_acquire))
                                                {
                                                    discard_consumed_coset_pages(
                                                        discard_set,
                                                        evaluation_end);
                                                }

                                                block_barrier.arrive_and_wait();

                                                if (worker_failed.load(
                                                        std::memory_order_acquire))
                                                {
                                                    break;
                                                }
                                            }
                                        });
                                }
                            }
                            catch (...)
                            {
                                start_state.store(
                                    -1,
                                    std::memory_order_release);

                                for (auto &worker : workers)
                                {
                                    if (worker.joinable())
                                    {
                                        worker.join();
                                    }
                                }

                                throw;
                            }

                            start_state.store(
                                1,
                                std::memory_order_release);

                            for (auto &worker : workers)
                            {
                                worker.join();
                            }

                            if (worker_error != nullptr)
                            {
                                std::rethrow_exception(
                                    worker_error);
                            }
                        }
                        else
                        {
                            for (std::size_t evaluation_begin = 0;
                                 evaluation_begin < target_size_;
                                 evaluation_begin +=
                                 evaluation_block_size)
                            {
                                const auto evaluation_end =
                                    std::min(
                                        target_size_,
                                        evaluation_begin + evaluation_block_size);

                                const auto evaluation_count =
                                    evaluation_end - evaluation_begin;

                                const auto task_count =
                                    std::min<std::size_t>(
                                        std::min<std::size_t>(
                                            cpu_count,
                                            32),
                                        std::max<std::size_t>(
                                            1,
                                            evaluation_count / min_points_per_task));

                                const auto chunk_size =
                                    (evaluation_count + task_count - 1) / task_count;

                                std::vector<std::future<void>> futures;
                                futures.reserve(task_count);

                                for (std::size_t task = 0;
                                     task < task_count;
                                     ++task)
                                {
                                    const auto begin =
                                        evaluation_begin + task * chunk_size;

                                    const auto end =
                                        std::min(
                                            evaluation_end,
                                            begin + chunk_size);

                                    if (begin >= end)
                                    {
                                        break;
                                    }

                                    futures.push_back(
                                        std::async(
                                            std::launch::async,
                                            [&, begin, end]()
                                            {
                                                std::vector<mcl::Fr>
                                                    denominator_products;

                                                evaluate_range(
                                                    begin,
                                                    end,
                                                    &denominator_products);
                                            }));
                                }

                                for (auto &future : futures)
                                {
                                    future.get();
                                }
                            }
                        }
                    }
                    else
                    {
                        auto z = shift_;

                        for (std::size_t i = 0;
                             i < target_size_;
                             ++i)
                        {
                            quotient_values[i] =
                                evaluate_at_index(i, z);

                            z *= root;
                        }
                    }
                }

                if (label_ == "t_edge")
                {
                    if (!cuda_t_edge_route)
                    {
                        progress_note(
                            "prove:quotient:t_edge:"
                            "evaluate:done:ms=" +
                            std::to_string(
                                elapsed_ms(
                                    point_evaluation_start,
                                    Clock::now())));
                    }

                    progress_note(
                        "prove:quotient:t_edge:"
                        "coefficients:start");
                }
                else if (label_ == "t_FH")
                {
                    progress_note(
                        "prove:quotient:t_FH:evaluate:done:ms=" + std::to_string(elapsed_ms(
                                                                      point_evaluation_start, Clock::now())));
                    progress_note(
                        "prove:quotient:t_FH:coefficients:start");
                }
                else if (label_ == "t_N")
                {
                    progress_note(
                        "prove:quotient:t_N:evaluate:done:ms=" + std::to_string(elapsed_ms(
                                                                     point_evaluation_start, Clock::now())));
                    progress_note("prove:quotient:t_N:coefficients:start");
                }

                const auto coefficient_start =
                    Clock::now();

                std::vector<FieldElement>
                    coefficients;

                if (cuda_t_edge_route)
                {
                    coefficients =
                        std::move(
                            cuda_t_edge_coefficients);
                }
                else
                {
                    coefficients =
                        coefficients_from_coset_evaluations_in_place(
                            std::move(
                                quotient_values),
                            shift_);
                }
                if (label_ == "t_edge")
                {
                    progress_note(
                        "prove:quotient:t_edge:coefficients:done:ms=" + std::to_string(elapsed_ms(
                                                                            coefficient_start, Clock::now())));
                }
                else if (label_ == "t_FH")
                {
                    progress_note(
                        "prove:quotient:t_FH:coefficients:done:ms=" + std::to_string(elapsed_ms(
                                                                          coefficient_start, Clock::now())));
                }
                else if (label_ == "t_N")
                {
                    progress_note(
                        "prove:quotient:t_N:coefficients:done:ms=" + std::to_string(elapsed_ms(
                                                                         coefficient_start, Clock::now())));
                }
                if (coefficients.size() > key.pk.degree_bound + 1)
                {
                    throw std::runtime_error("materialized quotient exceeds SRS degree bound: " + label_);
                }
                return algebra::Polynomial::from_coefficients(label_, std::move(coefficients));
            }

        private:
            struct FHQuotientCosetSlots
            {
                const QuotientCosetValues *q_table = nullptr;
                const QuotientCosetValues *q_query = nullptr;
                const QuotientCosetValues *table = nullptr;
                const QuotientCosetValues *query = nullptr;
                const QuotientCosetValues *multiplicity = nullptr;
                const QuotientCosetValues *j_table = nullptr;
                const QuotientCosetValues *j_query = nullptr;
                const QuotientCosetValues *accumulator = nullptr;
                const QuotientCosetValues *table_row = nullptr;
                const QuotientCosetValues *table_column = nullptr;
                const QuotientCosetValues *table_feature = nullptr;
                const QuotientCosetValues *query_row = nullptr;
                const QuotientCosetValues *query_column = nullptr;
                const QuotientCosetValues *query_feature = nullptr;

                bool ready() const
                {
                    return q_table != nullptr && q_query != nullptr && table != nullptr && query != nullptr && multiplicity != nullptr && j_table != nullptr && j_query != nullptr && accumulator != nullptr;
                }
            };

            struct TEdgeCosetAccess
            {
                std::string name;
                const QuotientCosetValues *values = nullptr;
                bool rotated = false;
            };

            bool use_file_backed_coset_storage() const
            {
                return (label_ == "t_edge" || label_ == "t_FH") && target_size_ >= (1ULL << 17);
            }

            std::size_t coset_byte_count(
                std::size_t element_count,
                const std::string &label) const
            {
                if (element_count > std::numeric_limits<
                                        std::size_t>::max() /
                                        sizeof(FieldElement))
                {
                    throw std::runtime_error(
                        "quotient coset byte-size "
                        "overflow: " +
                        label);
                }

                return element_count * sizeof(FieldElement);
            }

            bool try_reserve_in_memory_coset_bytes(
                std::size_t byte_count)
            {
                if (byte_count > in_memory_coset_budget_bytes_)
                {
                    return false;
                }

                auto retained_bytes =
                    in_memory_coset_bytes_.load(
                        std::memory_order_relaxed);

                while (true)
                {
                    if (retained_bytes > in_memory_coset_budget_bytes_ || byte_count > in_memory_coset_budget_bytes_ - retained_bytes)
                    {
                        return false;
                    }

                    if (in_memory_coset_bytes_
                            .compare_exchange_weak(
                                retained_bytes,
                                retained_bytes + byte_count,
                                std::memory_order_relaxed))
                    {
                        return true;
                    }
                }
            }

            bool new_coset_is_file_backed(
                std::size_t byte_count)
            {
                if (cuda_t_edge_transient_leaf_mode_)
                {
                    return false;
                }

                if (!use_file_backed_coset_storage())
                {
                    return false;
                }

                return !try_reserve_in_memory_coset_bytes(
                    byte_count);
            }

            void record_file_backed_coset(
                std::size_t byte_count)
            {
                file_backed_coset_count_.fetch_add(
                    1,
                    std::memory_order_relaxed);

                file_backed_coset_bytes_.fetch_add(
                    byte_count,
                    std::memory_order_relaxed);
            }
            QuotientCosetValuesPtr retain_coset_values(
                std::vector<FieldElement> values,
                const std::string &label)
            {
                const auto byte_count =
                    coset_byte_count(
                        values.size(),
                        label);

                const bool file_backed =
                    new_coset_is_file_backed(
                        byte_count);

                auto retained =
                    std::make_shared<
                        const QuotientCosetValues>(
                        std::move(values),
                        file_backed
                            ? spill_arena_
                            : nullptr,
                        label);

                if (file_backed)
                {
                    record_file_backed_coset(
                        byte_count);
                }

                return retained;
            }

            template <typename FillFn>
            QuotientCosetValuesPtr
            materialize_coset_values(
                std::size_t element_count,
                const std::string &label,
                FillFn &&fill,
                bool even_odd_split_layout = false)
            {
                const auto byte_count =
                    coset_byte_count(
                        element_count,
                        label);

                const bool file_backed =
                    new_coset_is_file_backed(
                        byte_count);

                auto retained =
                    std::make_shared<QuotientCosetValues>(
                        element_count,
                        file_backed
                            ? spill_arena_
                            : nullptr,
                        label,
                        even_odd_split_layout);

                fill(retained->mutable_data());

                if (file_backed)
                {
                    record_file_backed_coset(
                        byte_count);
                }

                return retained;
            }
            bool can_preserve_split_lane_output(
                const algebra::Polynomial &polynomial) const
            {
                // t_edge already has its dedicated device-resident fused
                // executor.  Enable physical split-lane storage only for the
                // genuinely large FH family where the generic host/tiered
                // transform otherwise allocates two full half-size
                // temporaries and then interleaves them into another full
                // output vector.
                if (label_ != "t_FH" ||
                    target_size_ <
                        (std::size_t{1} << 26) ||
                    target_size_ < 2)
                {
                    return false;
                }

                const auto half_size =
                    target_size_ / 2;

                if (polynomial.basis ==
                    algebra::
                        PolynomialBasis::
                            Coefficient)
                {
                    return polynomial.size() <=
                           half_size;
                }

                if (polynomial.domain == nullptr)
                {
                    return false;
                }

                const auto domain_size =
                    polynomial.domain->size;

                if (domain_size >
                    half_size)
                {
                    return false;
                }

                if (polynomial.zero_domain_blinder.size() >
                    half_size - domain_size)
                {
                    return false;
                }

                return true;
            }
            QuotientCosetValuesPtr
            materialize_polynomial_coset(
                const algebra::Polynomial &polynomial,
                const std::string &label)
            {
                const bool preserve_split_lane =
                    can_preserve_split_lane_output(
                        polynomial);

                auto retained =
                    materialize_coset_values(
                        target_size_,
                        label,
                        [&](FieldElement *output)
                        {
                            coset_evaluate_polynomial_into(
                                polynomial,
                                target_size_,
                                shift_,
                                label,
                                &shift_powers(),
                                output,
                                preserve_split_lane);
                        },
                        preserve_split_lane);

                if (preserve_split_lane)
                {
                    const auto byte_count =
                        coset_byte_count(
                            target_size_,
                            label);

                    split_lane_coset_count_.fetch_add(
                        1,
                        std::memory_order_relaxed);

                    split_lane_temporary_bytes_avoided_.fetch_add(
                        byte_count,
                        std::memory_order_relaxed);

                    split_lane_interleave_bytes_avoided_.fetch_add(
                        byte_count,
                        std::memory_order_relaxed);
                }

                return retained;
            }
            QuotientCosetValuesPtr
            materialize_coefficient_coset(
                const std::vector<mcl::Fr>
                    &coefficients,
                const std::string &label)
            {
                return materialize_coset_values(
                    target_size_,
                    label,
                    [&](FieldElement *output)
                    {
                        coset_evaluate_coefficients_into(
                            coefficients.data(),
                            coefficients.size(),
                            target_size_,
                            shift_,
                            label,
                            &shift_powers(),
                            output);
                    });
            }

            QuotientCosetValuesPtr retain_blinded_coset_view(
                QuotientCosetValuesPtr semantic_values,
                const std::vector<FieldElement> &zero_domain_blinder)
            {
                auto retained =
                    std::make_shared<const QuotientCosetValues>(
                        std::move(semantic_values),
                        zero_domain_blinder,
                        semantic_coset_points_,
                        semantic_coset_zero_cycle_);
                lazy_blinder_coset_count_.fetch_add(
                    1, std::memory_order_relaxed);
                lazy_blinder_coset_bytes_avoided_.fetch_add(
                    target_size_ * sizeof(FieldElement),
                    std::memory_order_relaxed);
                return retained;
            }

            QuotientCosetValuesPtr retain_blinded_coset_view(
                QuotientCosetValuesPtr semantic_values,
                const algebra::Polynomial &polynomial)
            {
                return retain_blinded_coset_view(
                    std::move(semantic_values),
                    polynomial.zero_domain_blinder);
            }

            std::vector<QuotientCosetValuesPtr>
            file_backed_coset_discard_set()
            {
                std::unordered_map<
                    const QuotientCosetValues *,
                    QuotientCosetValuesPtr>
                    unique;

                {
                    std::lock_guard<std::mutex> lock(
                        coset_cache_mutex_);

                    unique.reserve(coset_cache_.size());

                    for (const auto &[_, values] :
                         coset_cache_)
                    {
                        if (values != nullptr && values->file_backed())
                        {
                            unique.emplace(
                                values.get(),
                                values);
                        }
                    }
                }

                std::vector<QuotientCosetValuesPtr> out;
                out.reserve(unique.size());

                for (const auto &[_, values] : unique)
                {
                    out.push_back(values);
                }

                return out;
            }

            void discard_consumed_coset_pages(
                const std::vector<
                    QuotientCosetValuesPtr> &discard_set,
                std::size_t end_index) const
            {
                for (const auto &values : discard_set)
                {
                    if (values != nullptr)
                    {
                        values->discard_prefix(end_index);
                    }
                }
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

            const QuotientCosetValues &coset_values(const std::string &requested_name)
            {
                return *coset_values_ptr(requested_name);
            }

            void seed_fast_coset_lookup(const std::vector<std::string> &labels)
            {
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

            void require_t_edge_fast_coset_lookup(
                const std::vector<std::string> &labels) const
            {
                if (label_ != "t_edge")
                {
                    throw std::runtime_error(
                        "t_edge fast lookup validation requested for " + label_);
                }
                std::size_t derived_count = 0;
                for (const auto &name : labels)
                {
                    const auto it = fast_coset_lookup_.find(name);
                    if (it == fast_coset_lookup_.end() || it->second == nullptr || it->second->size() != target_size_)
                    {
                        throw std::runtime_error(
                            "t_edge quotient fast lookup failed to bind dependency: " + name);
                    }
                    if (is_edge_derived_coset_request(name))
                    {
                        ++derived_count;
                    }
                }
                progress_note(
                    "prove:quotient:t_edge:fast_lookup:dependencies=" + std::to_string(labels.size()) + ":derived_views=" + std::to_string(derived_count) + ":missing_or_null=0");
            }

            const QuotientCosetValues *bind_t_edge_fast_coset(
                const std::string &name)
            {
                if (label_ != "t_edge")
                {
                    throw std::runtime_error(
                        "t_edge fast lookup binding requested for " + label_);
                }
                auto it = fast_coset_lookup_.find(name);
                if (it == fast_coset_lookup_.end())
                {
                    // The dependency registry is intentionally a prewarm
                    // hint, not an alternate protocol equation.  Bind any
                    // evaluator-discovered name once through the generic
                    // resolver while recording the access plan.  The formal
                    // point loop below remains a pointer-only replay path.
                    const auto values = coset_values_ptr(name);
                    it = fast_coset_lookup_.emplace(name, values.get()).first;
                    ++edge_access_plan_late_bind_count_;
                }
                if (it->second == nullptr || it->second->size() != target_size_)
                {
                    throw std::runtime_error(
                        "t_edge quotient access-plan dependency is not bound to a full coset view: " + name);
                }
                return it->second;
            }

            void bind_fh_coset_slots()
            {
                if (label_ != "t_FH")
                {
                    throw std::runtime_error(
                        "FH coset slots requested for non-FH quotient");
                }
                const auto bind = [&](const char *name)
                    -> const QuotientCosetValues *
                {
                    const auto it = fast_coset_lookup_.find(name);
                    if (it == fast_coset_lookup_.end() || it->second == nullptr || it->second->size() != target_size_)
                    {
                        throw std::runtime_error(
                            "FH fixed coset slot is missing or has the wrong size: " + std::string(name));
                    }
                    return it->second;
                };
                fh_coset_slots_.q_table = bind("P_Q_tbl_feat");
                fh_coset_slots_.q_query = bind("P_Q_qry_feat");
                fh_coset_slots_.table = bind("P_Table_feat");
                fh_coset_slots_.query = bind("P_Query_feat");
                fh_coset_slots_.multiplicity = bind("P_m_feat");
                fh_coset_slots_.j_table = bind("P_J_tbl_feat");
                fh_coset_slots_.j_query = bind("P_J_qry_feat");
                fh_coset_slots_.accumulator = bind("P_R_feat");
                if (!fh_feature_encoding_terms_are_identically_zero_)
                {
                    fh_coset_slots_.table_row = bind("P_Row_feat_tbl");
                    fh_coset_slots_.table_column = bind("P_Col_feat_tbl");
                    fh_coset_slots_.table_feature = bind("P_T_H");
                    fh_coset_slots_.query_row = bind("P_I_feat_qry");
                    fh_coset_slots_.query_column = bind("P_Col_feat_qry");
                    fh_coset_slots_.query_feature = bind("P_H");
                }
                if (!fh_coset_slots_.ready())
                {
                    throw std::runtime_error(
                        "FH fixed coset slots were not fully initialized");
                }
            }

            QuotientBoundaryValues fh_boundary_values(
                const FieldElement &z) const
            {
                const auto one = FieldElement::one();
                const auto last_point =
                    domain_->point_at(domain_->size - 1);
                const auto zero = domain_->zero_polynomial_eval(z);
                if (zero.is_zero())
                {
                    return QuotientBoundaryValues{
                        .first = z == one
                                     ? one
                                     : FieldElement::zero(),
                        .last = z == last_point
                                    ? one
                                    : FieldElement::zero(),
                        .zero = zero,
                        .zero_inverse = std::nullopt,
                    };
                }
                const auto first_denominator = z - one;
                const auto last_denominator = z - last_point;
                const auto denominator_product_inverse =
                    (first_denominator * last_denominator).inv();
                const auto zero_scaled = zero * domain_->inv_size;
                return QuotientBoundaryValues{
                    .first = zero_scaled * last_denominator * denominator_product_inverse,
                    .last = zero_scaled * last_point * first_denominator * denominator_product_inverse,
                    .zero = zero,
                    .zero_inverse = std::nullopt,
                };
            }

            const QuotientCosetValues &fast_coset_values(const std::string &requested_name)
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
                if (label_ != "t_edge")
                {
                    return false;
                }
                for (std::size_t layer_index = 0;
                     layer_index < context_.model.hidden_layers.size();
                     ++layer_index)
                    for (std::size_t local_head_index = 0;
                         local_head_index < context_.model.hidden_layers[layer_index].heads.size();
                         ++local_head_index)
                    {
                        const auto prefix = hidden_head_prefix(layer_index, local_head_index) + "_";
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
                if (label_ != "t_edge")
                {
                    return false;
                }
                // A committed column carries its own registered Z_edge*R
                // blinder.  Its semantic expression is equal only on the
                // edge domain and cannot replace that polynomial on a coset.
                if (lookup_polynomial(name) != nullptr || trace_.spilled_polynomials.contains(name))
                {
                    return false;
                }
                for (std::size_t layer_index = 0;
                     layer_index < context_.model.hidden_layers.size();
                     ++layer_index)
                    for (std::size_t local_head_index = 0;
                         local_head_index < context_.model.hidden_layers[layer_index].heads.size();
                         ++local_head_index)
                    {
                        const auto prefix = hidden_head_prefix(layer_index, local_head_index) + "_";
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
                if (label_ != "t_edge")
                {
                    return std::nullopt;
                }
                if (lookup_polynomial(name) != nullptr || trace_.spilled_polynomials.contains(name))
                {
                    return std::nullopt;
                }
                for (std::size_t layer_index = 0;
                     layer_index < context_.model.hidden_layers.size();
                     ++layer_index)
                    for (std::size_t local_head_index = 0;
                         local_head_index < context_.model.hidden_layers[layer_index].heads.size();
                         ++local_head_index)
                    {
                        const auto prefix = hidden_head_prefix(layer_index, local_head_index) + "_";
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
                        const auto eta_l = challenges_.at(
                            hidden_challenge_name("eta_L_h", layer_index, local_head_index));
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
                        const auto eta_exp = challenges_.at(
                            hidden_challenge_name("eta_exp_h", layer_index, local_head_index));
                        if (suffix == "Table_exp")
                        {
                            return value("P_T_exp_x") + eta_exp * value("P_T_exp_y");
                        }
                        if (suffix == "Query_exp")
                        {
                            return head_value("Delta") + eta_exp * head_value("U");
                        }
                        const auto eta_elu = challenges_.at(
                            hidden_challenge_name("eta_ELU_h", layer_index, local_head_index));
                        if (suffix == "Table_ELU")
                        {
                            return value("P_T_ELU_x") + eta_elu * value("P_T_ELU_y");
                        }
                        if (suffix == "Query_ELU")
                        {
                            return head_value("H_ELU_in_flat") + eta_elu * head_value("H_agg_flat");
                        }
                        const auto eta_src = challenges_.at(
                            hidden_challenge_name("eta_src_h", layer_index, local_head_index));
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
                            const auto lambda_psq = challenges_.at(
                                hidden_challenge_name("lambda_psq_h", layer_index, local_head_index));
                            return head_value("U") + lambda_psq * head_value("widehat_v_pre_star");
                        }
                        if (suffix == "T_psq_edge")
                        {
                            if (lookup_polynomial(name) != nullptr || trace_.spilled_polynomials.contains(name))
                            {
                                return std::nullopt;
                            }
                            const auto lambda_psq = challenges_.at(
                                hidden_challenge_name("lambda_psq_h", layer_index, local_head_index));
                            return head_value("Sum_edge") + lambda_psq * head_value("H_agg_pre_star_edge");
                        }
                        const auto eta_t = challenges_.at(
                            hidden_challenge_name("eta_t_h", layer_index, local_head_index));
                        if (suffix == "Query_t")
                        {
                            return value("P_dst") + eta_t * head_value("T_psq_edge");
                        }
                        const auto eta_dst = challenges_.at(
                            hidden_challenge_name("eta_dst_h", layer_index, local_head_index));
                        if (suffix == "Query_dst")
                        {
                            const auto eta_dst_sq = eta_dst * eta_dst;
                            const auto eta_dst_cu = eta_dst_sq * eta_dst;
                            const auto eta_dst_4 = eta_dst_cu * eta_dst;
                            const auto eta_dst_5 = eta_dst_4 * eta_dst;
                            return value("P_dst") + eta_dst * head_value("E_dst_edge") + eta_dst_sq * head_value("M_edge") + eta_dst_cu * head_value("Sum_edge") + eta_dst_4 * head_value("inv_edge") + eta_dst_5 * head_value("H_agg_star_edge");
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
                        return value("P_dst") + eta_dst * head_value("E_dst_edge") + eta_dst_sq * head_value("M_edge") + eta_dst_cu * head_value("Sum_edge") + eta_dst_4 * head_value("inv_edge") + eta_dst_5 * head_value("Y_star_edge");
                    }
                    return std::nullopt;
                }
                return std::nullopt;
            }

            std::vector<FieldElement> quotient_alpha_powers_;

            struct SemanticCosetGroup
            {
                const algebra::Polynomial *owner = nullptr;
                const TraceArtifacts::SpilledEvaluationPolynomial *
                    spilled_owner = nullptr;
                std::string owner_label;
                std::vector<std::string> labels;
                std::mutex mutex;
                QuotientCosetValuesPtr base_values;
                std::size_t remaining = 0;
            };

            std::uint64_t semantic_values_hash(
                const std::vector<FieldElement> &values) const
            {
                std::uint64_t hash = 1469598103934665603ULL;
                for (const auto &value : values)
                {
                    const auto *units = value.native().getUnit();
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
                const std::vector<std::string> &wave)
            {
                semantic_coset_group_by_label_.clear();
                semantic_coset_groups_.clear();
                std::unordered_map<
                    std::uint64_t,
                    std::vector<std::shared_ptr<SemanticCosetGroup>>>
                    groups_by_hash;
                std::unordered_map<
                    std::string,
                    std::shared_ptr<SemanticCosetGroup>>
                    spilled_groups_by_source;
                std::size_t semantic_objects = 0;
                for (const auto &item_label : wave)
                {
                    const auto *polynomial = lookup_polynomial(item_label);
                    if (polynomial != nullptr && polynomial->basis == algebra::PolynomialBasis::Evaluation && polynomial->domain.get() == domain_.get())
                    {
                        ++semantic_objects;
                        const auto &values = polynomial->values();
                        auto &candidates = groups_by_hash[semantic_values_hash(values)];
                        auto group = std::find_if(
                            candidates.begin(),
                            candidates.end(),
                            [&](const auto &candidate)
                            {
                                return candidate->owner->values() == values;
                            });
                        if (group == candidates.end())
                        {
                            auto created =
                                std::make_shared<SemanticCosetGroup>();
                            created->owner = polynomial;
                            created->owner_label = item_label;
                            created->labels.push_back(item_label);
                            candidates.push_back(created);
                        }
                        else
                        {
                            (*group)->labels.push_back(item_label);
                        }
                        continue;
                    }

                    const auto spilled_it =
                        trace_.spilled_polynomials.find(item_label);
                    if (spilled_it == trace_.spilled_polynomials.end() || spilled_it->second.semantic_source_id == 0 || spilled_it->second.domain_name != domain_->name)
                    {
                        continue;
                    }
                    if (spilled_it->second.size > domain_->size)
                    {
                        throw std::runtime_error(
                            "spilled semantic coset source exceeds domain: " + item_label);
                    }
                    if (spilled_it->second.size == 0)
                    {
                        continue;
                    }
                    ++semantic_objects;
                    const auto source_key =
                        spilled_it->second.domain_name + ":" + std::to_string(spilled_it->second.size) + ":" + std::to_string(spilled_it->second.semantic_source_id);
                    const auto [group_it, inserted] =
                        spilled_groups_by_source.emplace(
                            source_key, nullptr);
                    if (inserted)
                    {
                        auto created =
                            std::make_shared<SemanticCosetGroup>();
                        created->spilled_owner = &spilled_it->second;
                        created->owner_label = item_label;
                        created->labels.push_back(item_label);
                        group_it->second = std::move(created);
                    }
                    else
                    {
                        group_it->second->labels.push_back(item_label);
                    }
                }

                std::size_t semantic_unique = 0;
                const auto register_group = [&](const auto &group)
                {
                    if (group->labels.size() <= 1)
                    {
                        return;
                    }
                    group->remaining = group->labels.size();
                    semantic_coset_groups_.push_back(group);
                    for (const auto &item_label : group->labels)
                    {
                        semantic_coset_group_by_label_.emplace(
                            item_label, group);
                    }
                };
                for (auto &[_, candidates] : groups_by_hash)
                {
                    semantic_unique += candidates.size();
                    for (auto &group : candidates)
                    {
                        register_group(group);
                    }
                }
                semantic_unique += spilled_groups_by_source.size();
                for (auto &[_, group] : spilled_groups_by_source)
                {
                    register_group(group);
                }

                if (!semantic_coset_groups_.empty())
                {
                    seed_lazy_blinder_geometry();
                }

                if (progress_enabled())
                {
                    progress_note(
                        "prove:quotient:" + label_ + ":coset_prewarm:semantic_objects=" + std::to_string(semantic_objects) + ":semantic_unique=" + std::to_string(semantic_unique) + ":semantic_duplicates=" + std::to_string(semantic_objects - semantic_unique) + ":semantic_reuse_groups=" + std::to_string(semantic_coset_groups_.size()));
                }
            }

            void seed_lazy_blinder_geometry()
            {
                if (semantic_coset_points_ != nullptr)
                {
                    return;
                }
                auto points =
                    std::make_shared<std::vector<FieldElement>>(
                        target_size_, FieldElement::zero());
                semantic_coset_zero_cycle_.resize(
                    rotation_step_, FieldElement::zero());
                const auto root = FieldElement::root_of_unity(target_size_);
                const auto zero_step = root.pow(domain_->size);
                auto point = shift_;
                auto zero_power = shift_.pow(domain_->size);
                const auto one = FieldElement::one();
                for (std::size_t index = 0; index < target_size_; ++index)
                {
                    (*points)[index] = point;
                    point *= root;
                }
                for (std::size_t index = 0;
                     index < rotation_step_; ++index)
                {
                    semantic_coset_zero_cycle_[index] =
                        zero_power - one;
                    zero_power *= zero_step;
                }
                semantic_coset_points_ = std::move(points);
            }

            QuotientCosetValuesPtr
            reused_semantic_coset_group_values(
                const std::string &requested_name,
                const std::vector<FieldElement> &zero_domain_blinder)
            {
                const auto group_it = semantic_coset_group_by_label_.find(
                    requested_name);
                if (group_it == semantic_coset_group_by_label_.end())
                {
                    return nullptr;
                }
                const auto &group = group_it->second;
                QuotientCosetValuesPtr base_values;
                {
                    std::lock_guard<std::mutex> lock(group->mutex);
                    if (group->base_values == nullptr)
                    {
                        algebra::Polynomial semantic_polynomial;
                        if (group->owner != nullptr)
                        {
                            semantic_polynomial = *group->owner;
                        }
                        else if (group->spilled_owner != nullptr)
                        {
                            semantic_polynomial =
                                algebra::Polynomial::from_evaluations(
                                    group->owner_label,
                                    load_spilled_values(
                                        *group->spilled_owner),
                                    context_domain_by_name(
                                        context_,
                                        group->spilled_owner
                                            ->domain_name));
                        }
                        else
                        {
                            throw std::runtime_error(
                                "semantic coset group has no owner: " + requested_name);
                        }
                        semantic_polynomial.zero_domain_blinder.clear();
                        group->base_values =
                            materialize_polynomial_coset(
                                semantic_polynomial,
                                group->owner_label);
                    }
                    base_values = group->base_values;
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
                if (zero_domain_blinder.empty())
                {
                    return base_values;
                }
                return retain_blinded_coset_view(
                    std::move(base_values), zero_domain_blinder);
            }

            QuotientCosetValuesPtr
            reused_semantic_coset_values(
                const std::string &requested_name,
                const algebra::Polynomial &polynomial)
            {
                return reused_semantic_coset_group_values(
                    requested_name,
                    polynomial.zero_domain_blinder);
            }

            QuotientCosetValuesPtr
            reused_spilled_semantic_coset_values(
                const std::string &requested_name,
                const TraceArtifacts::SpilledEvaluationPolynomial &spilled)
            {
                return reused_semantic_coset_group_values(
                    requested_name,
                    spilled.zero_domain_blinder);
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

                if (label_ == "t_edge")
                {
                    // Registered derived columns can be evaluated as lazy
                    // linear views of their authenticated sources plus the
                    // exact difference between source and target blinders.
                    // Seed the shared point geometry before parallel prewarm.
                    seed_lazy_blinder_geometry();
                }

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
                    const auto cpu_count =
                        std::max<std::size_t>(
                            1,
                            std::thread::hardware_concurrency());

                    std::size_t worker_cap = 4;

                    if (
                        label_ == "t_edge" &&
                        target_size_ <= (std::size_t{1} << 22) &&
                        cpu_count >= 8)
                    {
                        worker_cap =
                            std::min<std::size_t>(
                                24,
                                std::max<std::size_t>(
                                    8,
                                    cpu_count));
                    }
                    auto worker_count =
                        std::min<std::size_t>(
                            std::min<std::size_t>(
                                cpu_count,
                                worker_cap),
                            wave.size());

                    if (use_file_backed_coset_storage() && coset_host_available_at_start_bytes_ != 0)
                    {
                        const auto coset_bytes =
                            coset_byte_count(
                                target_size_,
                                label_);

                        const auto maximum =
                            std::numeric_limits<
                                std::size_t>::max();

                        const auto estimated_worker_bytes =
                            coset_bytes > maximum / 3
                                ? maximum
                                : coset_bytes * 3;

                        const auto safe_available =
                            coset_host_available_at_start_bytes_ > coset_host_reserve_bytes_
                                ? coset_host_available_at_start_bytes_ - coset_host_reserve_bytes_
                                : std::size_t{0};

                        if (estimated_worker_bytes != 0)
                        {
                            const auto memory_workers =
                                std::max<std::size_t>(
                                    1,
                                    safe_available / estimated_worker_bytes);

                            worker_count =
                                std::min(
                                    worker_count,
                                    memory_workers);
                        }
                    }

                    const auto transform_worker_cap =
                        std::max<std::size_t>(
                            1,
                            cpu_count / std::max<std::size_t>(
                                            1,
                                            worker_count));

                    const bool log_items =
                        !(label_ == "t_edge" && wave_name == std::string_view("base") && wave.size() > 64 && std::getenv("GATZK_VERBOSE_T_EDGE_PREWARM") == nullptr);
                    progress_note(
                        "prove:quotient:" + label_ + ":coset_prewarm:" + std::string(wave_name) + ":workers=" + std::to_string(worker_count));
                    std::atomic<std::size_t> next{0};
                    std::vector<std::future<void>> futures;
                    futures.reserve(worker_count);
                    for (std::size_t worker = 0;
                         worker < worker_count;
                         ++worker)
                    {
                        futures.push_back(
                            std::async(
                                std::launch::async,
                                [&, wave_name]()
                                {
                                    const ScopedQuotientTransformWorkerCap
                                        transform_worker_scope(
                                            transform_worker_cap);

                                    while (true)
                                    {
                                        const auto index =
                                            next.fetch_add(1);

                                        if (index >= wave.size())
                                        {
                                            break;
                                        }

                                        if (log_items)
                                        {
                                            progress_note(
                                                "prove:quotient:" + label_ + ":coset_prewarm:" + std::string(wave_name) + ":" + wave[index] + ":start");
                                        }

                                        coset_values_ptr(
                                            wave[index]);

                                        if (log_items)
                                        {
                                            progress_note(
                                                "prove:quotient:" + label_ + ":coset_prewarm:" + std::string(wave_name) + ":" + wave[index] + ":done");
                                        }
                                    }
                                }));
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
                if (polynomial == nullptr || polynomial->basis != algebra::PolynomialBasis::Evaluation || polynomial->domain == nullptr || polynomial->shared_data == nullptr)
                {
                    return {};
                }
                return polynomial->domain->name + ":" + std::to_string(polynomial->domain->size) + ":" + std::to_string(reinterpret_cast<std::uintptr_t>(polynomial->shared_data.get()));
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
                    const auto *polynomial = lookup_polynomial(label);
                    // A shared evaluation buffer does not imply a shared
                    // formal polynomial: private trace objects carry
                    // independently sampled Z_domain blinders.  Keep those
                    // labels in the materialization wave so semantic coset
                    // reuse can share only the canonical evaluations and add
                    // each registered blinder lazily.  Direct aliases are
                    // valid only for unblinded columns.
                    if (polynomial != nullptr && polynomial->is_zero_domain_blinded())
                    {
                        filtered.push_back(label);
                        continue;
                    }
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
                        "prove:quotient:t_edge:coset_prewarm:shared_source_aliases=" + std::to_string(aliases.size()));
                }
            }

            bool is_edge_derived_coset_request(const std::string &name) const
            {
                if (label_ != "t_edge")
                {
                    return false;
                }
                auto is_hidden_derived_suffix = [](std::string_view suffix)
                {
                    return suffix == "S" || suffix == "Delta" || suffix == "Table_L" || suffix == "Query_L" || suffix == "Table_R" || suffix == "Query_R" || suffix == "Table_exp" || suffix == "Query_exp" || suffix == "Table_ELU" || suffix == "Query_ELU" || suffix == "Query_src" || suffix == "Query_t" || suffix == "Query_dst" || suffix == "w_psq" || suffix == "T_psq_edge";
                };
                for (std::size_t layer_index = 0;
                     layer_index < context_.model.hidden_layers.size();
                     ++layer_index)
                    for (std::size_t local_head_index = 0;
                         local_head_index < context_.model.hidden_layers[layer_index].heads.size();
                         ++local_head_index)
                    {
                        const auto prefix = hidden_head_prefix(layer_index, local_head_index) + "_";
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

            std::vector<FieldElement> registered_blinder_for_label(
                const std::string &label) const
            {
                if (const auto *polynomial = lookup_polynomial(label);
                    polynomial != nullptr)
                {
                    return polynomial->zero_domain_blinder;
                }
                if (const auto spilled_it =
                        trace_.spilled_polynomials.find(label);
                    spilled_it != trace_.spilled_polynomials.end())
                {
                    return spilled_it->second.zero_domain_blinder;
                }
                const auto canonical = canonical_coset_label(label);
                if (canonical != label)
                {
                    return registered_blinder_for_label(canonical);
                }
                return {};
            }

            QuotientCosetValuesPtr scaled_coset_sum(
                const std::string &derived_label,
                const std::vector<std::pair<std::string, FieldElement>> &terms)
            {
                std::vector<std::pair<
                    QuotientCosetValuesPtr,
                    FieldElement>>
                    resolved_terms;
                resolved_terms.reserve(terms.size());
                auto correction =
                    registered_blinder_for_label(derived_label);
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
                    resolved_terms.push_back({values, scale});
                    const auto source_blinder =
                        registered_blinder_for_label(label);
                    if (correction.size() < source_blinder.size())
                    {
                        correction.resize(
                            source_blinder.size(),
                            FieldElement::zero());
                    }
                    for (std::size_t coefficient = 0;
                         coefficient < source_blinder.size();
                         ++coefficient)
                    {
                        correction[coefficient] -=
                            scale * source_blinder[coefficient];
                    }
                }
                trim_field_coefficients(&correction);
                if (correction.size() == 1 && correction.front().is_zero())
                {
                    correction.clear();
                }
                if (resolved_terms.empty())
                {
                    throw std::runtime_error(
                        "derived t_edge linear view has no terms: " + derived_label);
                }
                if (resolved_terms.size() == 1 && resolved_terms.front().second == FieldElement::one() && correction.empty())
                {
                    return resolved_terms.front().first;
                }
                if (!correction.empty() && semantic_coset_zero_cycle_.empty())
                {
                    throw std::runtime_error(
                        "derived t_edge linear view is missing blinder geometry: " + derived_label);
                }
                auto retained =
                    std::make_shared<const QuotientCosetValues>(
                        std::move(resolved_terms),
                        std::move(correction),
                        semantic_coset_points_,
                        semantic_coset_zero_cycle_);
                lazy_blinder_coset_count_.fetch_add(
                    1, std::memory_order_relaxed);
                lazy_blinder_coset_bytes_avoided_.fetch_add(
                    target_size_ * sizeof(FieldElement),
                    std::memory_order_relaxed);
                return retained;
            }

            QuotientCosetValuesPtr derived_hidden_edge_coset(const std::string &name)
            {
                if (label_ != "t_edge")
                {
                    return nullptr;
                }
                for (std::size_t layer_index = 0;
                     layer_index < context_.model.hidden_layers.size();
                     ++layer_index)
                    for (std::size_t local_head_index = 0;
                         local_head_index < context_.model.hidden_layers[layer_index].heads.size();
                         ++local_head_index)
                    {
                        const auto prefix = hidden_head_prefix(layer_index, local_head_index) + "_";
                        if (name.rfind(prefix, 0) != 0)
                        {
                            continue;
                        }
                        const auto suffix = name.substr(prefix.size());
                        const auto eta_l = challenges_.at(
                            hidden_challenge_name("eta_L_h", layer_index, local_head_index));
                        const auto eta_exp = challenges_.at(
                            hidden_challenge_name("eta_exp_h", layer_index, local_head_index));
                        const auto eta_elu = challenges_.at(
                            hidden_challenge_name("eta_ELU_h", layer_index, local_head_index));
                        const auto eta_src = challenges_.at(
                            hidden_challenge_name("eta_src_h", layer_index, local_head_index));
                        const auto eta_dst = challenges_.at(
                            hidden_challenge_name("eta_dst_h", layer_index, local_head_index));
                        const auto eta_t = challenges_.at(
                            hidden_challenge_name("eta_t_h", layer_index, local_head_index));
                        if (suffix == "Table_L")
                        {
                            return scaled_coset_sum(name, {{"P_T_L_x", FieldElement::one()}, {"P_T_L_y", eta_l}});
                        }
                        if (suffix == "Query_L")
                        {
                            return scaled_coset_sum(name, {{prefix + "S", FieldElement::one()}, {prefix + "Z", eta_l}});
                        }
                        if (suffix == "S")
                        {
                            return scaled_coset_sum(name, {{prefix + "E_src_edge", FieldElement::one()}, {prefix + "E_dst_edge", FieldElement::one()}});
                        }
                        if (suffix == "Delta")
                        {
                            return scaled_coset_sum(name, {{prefix + "M_edge", FieldElement::one()}, {prefix + "Z", -FieldElement::one()}});
                        }
                        if (suffix == "Table_R")
                        {
                            return scaled_coset_sum(
                                name,
                                {{"P_T_range", FieldElement::one()}});
                        }
                        if (suffix == "Query_R")
                        {
                            return scaled_coset_sum(
                                name,
                                {{prefix + "Delta", FieldElement::one()}});
                        }
                        if (suffix == "Table_exp")
                        {
                            return scaled_coset_sum(name, {{"P_T_exp_x", FieldElement::one()}, {"P_T_exp_y", eta_exp}});
                        }
                        if (suffix == "Query_exp")
                        {
                            return scaled_coset_sum(name, {{prefix + "Delta", FieldElement::one()}, {prefix + "U", eta_exp}});
                        }
                        if (suffix == "w_psq")
                        {
                            const auto lambda_psq = challenges_.at(
                                hidden_challenge_name("lambda_psq_h", layer_index, local_head_index));
                            return scaled_coset_sum(name, {{prefix + "U", FieldElement::one()}, {prefix + "widehat_v_pre_star", lambda_psq}});
                        }
                        if (suffix == "T_psq_edge")
                        {
                            const auto lambda_psq = challenges_.at(
                                hidden_challenge_name("lambda_psq_h", layer_index, local_head_index));
                            return scaled_coset_sum(name, {{prefix + "Sum_edge", FieldElement::one()}, {prefix + "H_agg_pre_star_edge", lambda_psq}});
                        }
                        if (suffix == "Table_ELU")
                        {
                            return scaled_coset_sum(name, {{"P_T_ELU_x", FieldElement::one()}, {"P_T_ELU_y", eta_elu}});
                        }
                        if (suffix == "Query_ELU")
                        {
                            return scaled_coset_sum(
                                name,
                                {{prefix + "H_ELU_in_flat", FieldElement::one()}, {prefix + "H_agg_flat", eta_elu}});
                        }
                        if (suffix == "Query_src")
                        {
                            const auto eta_src_sq = eta_src * eta_src;
                            return scaled_coset_sum(name, {
                                                              {"P_src", FieldElement::one()},
                                                              {prefix + "E_src_edge", eta_src},
                                                              {prefix + "H_src_star_edge", eta_src_sq},
                                                          });
                        }
                        if (suffix == "Query_t")
                        {
                            return scaled_coset_sum(name, {{"P_dst", FieldElement::one()}, {prefix + "T_psq_edge", eta_t}});
                        }
                        if (suffix == "Query_dst")
                        {
                            const auto eta_dst_sq = eta_dst * eta_dst;
                            const auto eta_dst_cu = eta_dst_sq * eta_dst;
                            const auto eta_dst_4 = eta_dst_cu * eta_dst;
                            const auto eta_dst_5 = eta_dst_4 * eta_dst;
                            return scaled_coset_sum(name, {
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

            QuotientCosetValuesPtr derived_output_edge_coset(const std::string &name)
            {
                if (label_ != "t_edge")
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
                        return scaled_coset_sum(name, {{"P_T_L_x", FieldElement::one()}, {"P_T_L_y", eta_l}});
                    }
                    if (suffix == "Query_L")
                    {
                        return scaled_coset_sum(name, {{prefix + "S", FieldElement::one()}, {prefix + "Z", eta_l}});
                    }
                    if (suffix == "S")
                    {
                        return scaled_coset_sum(name, {{prefix + "E_src_edge", FieldElement::one()}, {prefix + "E_dst_edge", FieldElement::one()}});
                    }
                    if (suffix == "Delta")
                    {
                        return scaled_coset_sum(name, {{prefix + "M_edge", FieldElement::one()}, {prefix + "Z", -FieldElement::one()}});
                    }
                    if (suffix == "Table_R")
                    {
                        return scaled_coset_sum(
                            name,
                            {{"P_T_range", FieldElement::one()}});
                    }
                    if (suffix == "Query_R")
                    {
                        return scaled_coset_sum(
                            name,
                            {{prefix + "Delta", FieldElement::one()}});
                    }
                    if (suffix == "Table_exp")
                    {
                        return scaled_coset_sum(name, {{"P_T_exp_x", FieldElement::one()}, {"P_T_exp_y", eta_exp}});
                    }
                    if (suffix == "Query_exp")
                    {
                        return scaled_coset_sum(name, {{prefix + "Delta", FieldElement::one()}, {prefix + "U", eta_exp}});
                    }
                    if (suffix == "w")
                    {
                        const auto lambda_out = challenges_.at(output_challenge_name("lambda_out", head_index));
                        return scaled_coset_sum(name, {{prefix + "U", FieldElement::one()}, {prefix + "widehat_y_star", lambda_out}});
                    }
                    if (suffix == "T_edge")
                    {
                        const auto lambda_out = challenges_.at(output_challenge_name("lambda_out", head_index));
                        return scaled_coset_sum(name, {{prefix + "Sum_edge", FieldElement::one()}, {prefix + "Y_star_edge", lambda_out}});
                    }
                    if (suffix == "Query_src")
                    {
                        return scaled_coset_sum(name, {{"P_src", FieldElement::one()}, {prefix + "E_src_edge", eta_src}});
                    }
                    if (suffix == "Query_t")
                    {
                        return scaled_coset_sum(name, {{"P_dst", FieldElement::one()}, {prefix + "T_edge", eta_t}});
                    }
                    if (suffix == "Query_dst")
                    {
                        const auto eta_dst_sq = eta_dst * eta_dst;
                        const auto eta_dst_cu = eta_dst_sq * eta_dst;
                        const auto eta_dst_4 = eta_dst_cu * eta_dst;
                        const auto eta_dst_5 = eta_dst_4 * eta_dst;
                        return scaled_coset_sum(name, {
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
                if (label_ == "t_edge" && (name == "P_Q_qry_src" || name == "P_Q_qry_dst" || name == "P_Q_qry_L" || name == "P_Q_qry_R" || name == "P_Q_qry_exp"))
                {
                    return "P_Q_edge_valid";
                }
                return name;
            }

            QuotientCosetValuesPtr feature_selector_coset(const std::string &requested_name)
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
                if (polynomial->basis != algebra::PolynomialBasis::Evaluation || polynomial->domain == nullptr || polynomial->domain->size != domain_->size || !(polynomial->domain->omega == domain_->omega) || polynomial->values().size() != domain_->size)
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
                auto values =
                    materialize_coefficient_coset(
                        coefficients,
                        requested_name);
                progress_note("prove:quotient:t_FH:selector:" + requested_name + ":coset:done");
                return values;
            }

            QuotientCosetValuesPtr cached_public_fh_material_coset(const std::string &requested_name)
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
                auto values =
                    materialize_polynomial_coset(
                        *it->second,
                        requested_name);
                progress_note("prove:quotient:t_FH:public_coeff:" + requested_name + ":coset:done");
                return values;
            }

            std::vector<FieldElement> blinder_difference(
                const std::vector<FieldElement> &left,
                const std::vector<FieldElement> &right) const
            {
                const auto size = std::max(left.size(), right.size());
                std::vector<FieldElement> difference(
                    size, FieldElement::zero());
                for (std::size_t i = 0; i < size; ++i)
                {
                    const auto left_value =
                        i < left.size()
                            ? left[i]
                            : FieldElement::zero();
                    const auto right_value =
                        i < right.size()
                            ? right[i]
                            : FieldElement::zero();
                    difference[i] = left_value - right_value;
                }
                trim_field_coefficients(&difference);
                if (difference.size() == 1 && difference.front().is_zero())
                {
                    difference.clear();
                }
                return difference;
            }

            QuotientCosetValuesPtr retain_coset_with_blinder_correction(
                QuotientCosetValuesPtr base_values,
                std::vector<FieldElement> semantic_values,
                const std::vector<FieldElement> &blinder_correction,
                const std::string &label)
            {
                if (semantic_values.empty())
                {
                    if (base_values != nullptr && blinder_correction.empty())
                    {
                        return base_values;
                    }
                    semantic_values.resize(
                        target_size_, FieldElement::zero());
                }
                if (semantic_values.size() != target_size_ || (base_values != nullptr && base_values->size() != target_size_))
                {
                    throw std::runtime_error(
                        "lazy spilled FH semantic coset size mismatch: " + label);
                }

                const auto root =
                    FieldElement::root_of_unity(target_size_);
                const auto zero_step =
                    root.pow(domain_->size);
                const auto shifted_zero =
                    shift_.pow(domain_->size);
                const auto one = FieldElement::one();
                parallel_for_chunks(
                    target_size_,
                    std::size_t{1} << 15,
                    0,
                    [&](std::size_t begin, std::size_t end)
                    {
                        auto point = shift_ * root.pow(
                                                  static_cast<std::uint64_t>(begin));
                        auto zero_power = shifted_zero * zero_step.pow(
                                                             static_cast<std::uint64_t>(begin));
                        for (std::size_t i = begin; i < end; ++i)
                        {
                            auto value = base_values != nullptr
                                             ? (*base_values)[i]
                                             : semantic_values[i];
                            if (!blinder_correction.empty())
                            {
                                value += (zero_power - one) * algebra::horner(
                                                                  blinder_correction,
                                                                  point);
                            }
                            semantic_values[i] = value;
                            point *= root;
                            zero_power *= zero_step;
                        }
                    });
                return retain_coset_values(
                    std::move(semantic_values), label);
            }

            QuotientCosetValuesPtr retain_lazy_coset_with_blinder_correction(
                QuotientCosetValuesPtr base_values,
                const std::vector<FieldElement> &blinder_correction,
                const std::string &label)
            {
                seed_lazy_blinder_geometry();
                if (semantic_coset_points_ == nullptr || semantic_coset_points_->size() != target_size_ || semantic_coset_zero_cycle_.empty() || semantic_coset_zero_cycle_.size() != rotation_step_ || target_size_ % semantic_coset_zero_cycle_.size() != 0)
                {
                    throw std::runtime_error(
                        "lazy spilled quotient blinder geometry mismatch: " + label);
                }
                if (base_values != nullptr)
                {
                    if (base_values->size() != target_size_)
                    {
                        throw std::runtime_error(
                            "lazy spilled quotient semantic coset size mismatch: " + label);
                    }
                    if (blinder_correction.empty())
                    {
                        return base_values;
                    }
                    return retain_blinded_coset_view(
                        std::move(base_values),
                        blinder_correction);
                }
                auto retained =
                    std::make_shared<const QuotientCosetValues>(
                        QuotientCosetValues::PureZeroSemanticTag{},
                        target_size_,
                        blinder_correction,
                        semantic_coset_points_,
                        semantic_coset_zero_cycle_);
                lazy_blinder_coset_count_.fetch_add(
                    1, std::memory_order_relaxed);
                lazy_blinder_coset_bytes_avoided_.fetch_add(
                    target_size_ * sizeof(FieldElement),
                    std::memory_order_relaxed);
                return retained;
            }

            QuotientCosetValuesPtr
            spilled_pure_zero_semantic_coset(
                const std::string &requested_name,
                const TraceArtifacts::SpilledEvaluationPolynomial
                    &spilled)
            {
                if (!spilled.compact_semantic_source_all_zero)
                {
                    return nullptr;
                }
                const bool signed_division_compact_source =
                    requested_name.find("_Div_Limb_") != std::string::npos || (requested_name.find("_Div_Range_") != std::string::npos && requested_name.ends_with("_Query"));
                if (label_ != "t_edge" || !signed_division_compact_source || spilled.semantic_source_id == 0 || spilled.domain_name != domain_->name || spilled.size == 0 || spilled.size >= domain_->size || spilled.zero_domain_blinder.empty())
                {
                    throw std::runtime_error(
                        "invalid pure-zero compact spilled quotient metadata: " + requested_name);
                }
                return retain_lazy_coset_with_blinder_correction(
                    nullptr,
                    spilled.zero_domain_blinder,
                    requested_name);
            }

            QuotientCosetValuesPtr
            materialize_spilled_lazy_feature_coset(
                const std::string &requested_name,
                const TraceArtifacts::SpilledEvaluationPolynomial
                    &spilled)
            {
                const bool full_feature_identity =
                    lazy_full_feature_lookup_trace_enabled(context_);
                const bool identical_feature_inverse =
                    requested_name == "P_J_tbl_feat" || requested_name == "P_J_qry_feat";
                if (!full_feature_identity || (!is_lazy_full_feature_lookup_trace_label(
                                                   requested_name) &&
                                               !identical_feature_inverse))
                {
                    return nullptr;
                }
                // Table is already the authenticated evaluation column written
                // by the trace.  Transform it once directly.  Reconstructing
                // it from Row/Col/T_H performed three full 4x-FH transforms
                // merely to recover the same committed polynomial.
                if (requested_name == "P_Table_feat" || requested_name == "P_J_tbl_feat")
                {
                    return nullptr;
                }
                progress_note(
                    "prove:quotient:t_FH:spilled_lazy:" + requested_name + ":start");
                QuotientCosetValuesPtr base_values;
                std::vector<FieldElement> semantic_values;
                auto correction = spilled.zero_domain_blinder;
                if (requested_name == "P_Query_feat")
                {
                    base_values =
                        coset_values_ptr("P_Table_feat");
                    const auto table_it =
                        trace_.spilled_polynomials.find(
                            "P_Table_feat");
                    if (table_it == trace_.spilled_polynomials.end())
                    {
                        throw std::runtime_error(
                            "spilled FH query is missing its "
                            "full-identity table source");
                    }
                    correction = blinder_difference(
                        spilled.zero_domain_blinder,
                        table_it->second.zero_domain_blinder);
                }
                else if (requested_name == "P_J_qry_feat")
                {
                    base_values =
                        coset_values_ptr("P_J_tbl_feat");
                    const auto table_inverse_it =
                        trace_.spilled_polynomials.find(
                            "P_J_tbl_feat");
                    if (table_inverse_it == trace_.spilled_polynomials.end())
                    {
                        throw std::runtime_error(
                            "spilled FH query inverse is missing its "
                            "identical table inverse source");
                    }
                    correction = blinder_difference(
                        spilled.zero_domain_blinder,
                        table_inverse_it->second.zero_domain_blinder);
                }
                else if (requested_name == "P_m_feat")
                {
                    base_values =
                        coset_values_ptr("P_Q_tbl_feat");
                }
                else if (requested_name != "P_R_feat")
                {
                    return nullptr;
                }
                const bool use_lazy_blinder_view =
                    label_ == "t_FH" && target_size_ >= (std::size_t{1} << 26);
                auto values = use_lazy_blinder_view
                                  ? retain_lazy_coset_with_blinder_correction(
                                        std::move(base_values),
                                        correction,
                                        requested_name)
                                  : retain_coset_with_blinder_correction(
                                        std::move(base_values),
                                        std::move(semantic_values),
                                        correction,
                                        requested_name);
                progress_note(
                    "prove:quotient:t_FH:spilled_lazy:" + requested_name + ":done");
                return values;
            }

            QuotientCosetValuesPtr coset_values_ptr(const std::string &requested_name)
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
                if (is_edge_derived_coset_request(requested_name))
                {
                    auto values =
                        derived_hidden_edge_coset(requested_name);
                    if (values == nullptr)
                    {
                        values =
                            derived_output_edge_coset(requested_name);
                    }
                    if (values != nullptr)
                    {
                        std::lock_guard<std::mutex> lock(
                            coset_cache_mutex_);
                        const auto [it, inserted] =
                            coset_cache_.emplace(
                                requested_name, values);
                        return inserted ? values : it->second;
                    }
                }
                if (const auto *polynomial =
                        lookup_polynomial(requested_name);
                    polynomial != nullptr)
                {
                    auto direct_values =
                        reused_semantic_coset_values(
                            requested_name, *polynomial);
                    if (direct_values == nullptr && polynomial
                                                        ->is_zero_domain_blinded())
                    {
                        direct_values =
                            materialize_polynomial_coset(
                                *polynomial,
                                requested_name);
                    }
                    if (direct_values != nullptr)
                    {
                        std::lock_guard<std::mutex> lock(
                            coset_cache_mutex_);
                        const auto [it, inserted] =
                            coset_cache_.emplace(
                                requested_name, direct_values);
                        return inserted ? direct_values : it->second;
                    }
                }
                // A spilled trace object is still the committed private
                // polynomial, including its independently sampled
                // Z_domain blinder.  Resolve it before public/lazy semantic
                // aliases: those aliases reproduce only the canonical
                // evaluation column and would silently drop the registered
                // blinding tail after the object becomes file-backed.
                if (const auto spilled_it =
                        trace_.spilled_polynomials.find(requested_name);
                    spilled_it != trace_.spilled_polynomials.end())
                {
                    if (auto zero_values =
                            spilled_pure_zero_semantic_coset(
                                requested_name,
                                spilled_it->second);
                        zero_values != nullptr)
                    {
                        std::lock_guard<std::mutex> lock(
                            coset_cache_mutex_);
                        const auto [it, inserted] =
                            coset_cache_.emplace(
                                requested_name,
                                zero_values);
                        return inserted
                                   ? zero_values
                                   : it->second;
                    }
                    if (auto shared_values =
                            reused_spilled_semantic_coset_values(
                                requested_name,
                                spilled_it->second);
                        shared_values != nullptr)
                    {
                        std::lock_guard<std::mutex> lock(
                            coset_cache_mutex_);
                        const auto [it, inserted] =
                            coset_cache_.emplace(
                                requested_name,
                                shared_values);
                        return inserted
                                   ? shared_values
                                   : it->second;
                    }
                    if (auto lazy_values =
                            materialize_spilled_lazy_feature_coset(
                                requested_name,
                                spilled_it->second);
                        lazy_values != nullptr)
                    {
                        std::lock_guard<std::mutex> lock(
                            coset_cache_mutex_);
                        const auto [it, inserted] =
                            coset_cache_.emplace(
                                requested_name,
                                lazy_values);
                        return inserted
                                   ? lazy_values
                                   : it->second;
                    }
                    auto polynomial =
                        algebra::Polynomial::from_evaluations(
                            requested_name,
                            load_spilled_values(spilled_it->second),
                            context_domain_by_name(
                                context_,
                                spilled_it->second.domain_name));
                    if (!spilled_it->second.zero_domain_blinder.empty())
                    {
                        polynomial.set_zero_domain_blinder(
                            spilled_it->second.zero_domain_blinder);
                    }
                    auto values =
                        materialize_polynomial_coset(
                            polynomial,
                            requested_name);
                    std::lock_guard<std::mutex> lock(
                        coset_cache_mutex_);
                    const auto [it, inserted] =
                        coset_cache_.emplace(
                            requested_name, values);
                    return inserted ? values : it->second;
                }
                if (!alias_owner.empty() && alias_owner != requested_name)
                {
                    auto values = coset_values_ptr(alias_owner);
                    std::lock_guard<std::mutex> lock(coset_cache_mutex_);
                    const auto [it, inserted] = coset_cache_.emplace(requested_name, values);
                    return inserted ? values : it->second;
                }
                QuotientCosetValuesPtr values;
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
                else if (const auto *polynomial =
                             lookup_polynomial(
                                 requested_name);
                         polynomial != nullptr)
                {
                    values =
                        materialize_polynomial_coset(
                            *polynomial,
                            requested_name);
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

            QuotientCosetValuesPtr
            materialize_lazy_feature_lookup_trace(
                const std::string &name)
            {
                if (name == "P_R_feat")
                {
                    return materialize_coset_values(
                        target_size_,
                        name,
                        [](FieldElement *) {});
                }

                if (name == "P_m_feat")
                {
                    return full_graph_feature_identity_enabled(
                               context_)
                               ? coset_values_ptr(
                                     "P_Q_tbl_feat")
                               : coset_values_ptr(
                                     "P_Q_qry_feat");
                }

                if (name == "P_Query_feat" && full_graph_feature_identity_enabled(
                                                  context_))
                {
                    return coset_values_ptr(
                        "P_Table_feat");
                }

                const auto eta_feat =
                    challenges_.at("eta_feat");

                const auto eta_feat_sq =
                    eta_feat * eta_feat;

                const auto &rows =
                    coset_values(
                        name == "P_Table_feat"
                            ? "P_Row_feat_tbl"
                            : "P_Row_feat_qry");

                const auto &cols =
                    coset_values(
                        name == "P_Table_feat"
                            ? "P_Col_feat_tbl"
                            : "P_Col_feat_qry");

                const auto &features =
                    coset_values("P_T_H");

                return materialize_coset_values(
                    target_size_,
                    name,
                    [&](FieldElement *output)
                    {
                        parallel_for_chunks_static(
                            target_size_,
                            std::size_t{1} << 15,
                            0,
                            [&](std::size_t begin,
                                std::size_t end)
                            {
                                for (std::size_t i = begin;
                                     i < end;
                                     ++i)
                                {
                                    output[i] =
                                        rows[i] + eta_feat * cols[i] + eta_feat_sq * features[i];
                                }
                            });
                    });
            }

            std::vector<mcl::Fr> dependency_coefficients(
                const std::string &name,
                const algebra::Polynomial &polynomial) const
            {
                (void)name;
                return coefficients_native_from_polynomial(polynomial);
            }
            const std::vector<mcl::Fr> &
            shift_powers()
            {
                std::call_once(
                    shift_powers_once_,
                    [&]()
                    {
                        // Quotient cosets are 4x their authenticated domain.
                        // A retained power table is useful on medium shapes
                        // because many dependency transforms reuse it, but
                        // 1-2 GiB tables on FH=2^24/2^25 cause more unified-
                        // memory pressure than the saved geometric recurrence.
                        constexpr std::size_t
                            kMaximumRetainedShiftPowerBytes =
                                std::size_t{512} << 20;

                        const auto power_count =
                            target_size_ >= 2
                                ? target_size_ / 2
                                : target_size_;

                        if (power_count == 0)
                        {
                            return;
                        }

                        const auto maximum_count =
                            kMaximumRetainedShiftPowerBytes /
                            sizeof(mcl::Fr);

                        if (power_count >
                            maximum_count)
                        {
                            // Empty means that
                            // coset_evaluate_coefficients_into()
                            // uses its bounded streaming geometric-scaling
                            // path instead of retaining a giant lookup table.
                            progress_note(
                                "prove:quotient:" +
                                label_ +
                                ":shift_powers:"
                                "streaming:count=" +
                                std::to_string(
                                    power_count));

                            return;
                        }

                        shift_powers_.resize(
                            power_count);

                        algebra::geometric_powers_native(
                            shift_powers_.data(),
                            shift_powers_.size(),
                            shift_,
                            quotient_transform_worker_cap);
                    });

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
            std::once_flag shift_powers_once_;
            std::vector<mcl::Fr> shift_powers_;
            std::shared_ptr<const TEdgeEvaluationPlan> edge_evaluation_plan_;
            std::optional<FHQuotientParameters> fh_parameters_;
            FHQuotientCosetSlots fh_coset_slots_;
            bool fh_feature_encoding_terms_are_identically_zero_ = false;
            std::shared_ptr<QuotientCosetSpillArena> spill_arena_;
            std::vector<TEdgeCosetAccess> edge_access_sequence_;
            bool record_edge_access_sequence_ = false;
            std::size_t edge_access_plan_late_bind_count_ = 0;
            std::mutex coset_cache_mutex_;
            std::unordered_map<std::string, QuotientCosetValuesPtr> coset_cache_;
            std::unordered_map<std::string, std::string> coset_aliases_;
            std::unordered_map<std::string, const QuotientCosetValues *> fast_coset_lookup_;
            std::vector<std::shared_ptr<SemanticCosetGroup>> semantic_coset_groups_;
            std::unordered_map<std::string, std::shared_ptr<SemanticCosetGroup>>
                semantic_coset_group_by_label_;
            std::shared_ptr<const std::vector<FieldElement>>
                semantic_coset_points_;
            std::vector<FieldElement> semantic_coset_zero_cycle_;
            std::size_t coset_host_total_bytes_ = 0;
            std::size_t
                coset_host_available_at_start_bytes_ = 0;
            std::size_t coset_host_reserve_bytes_ = 0;
            std::size_t in_memory_coset_budget_bytes_ = 0;
            std::atomic<std::size_t> in_memory_coset_bytes_{0};
            std::atomic<std::size_t> file_backed_coset_count_{0};
            std::atomic<std::size_t> file_backed_coset_bytes_{0};
            std::atomic<std::size_t> lazy_blinder_coset_count_{0};
            std::atomic<std::size_t>
                lazy_blinder_coset_bytes_avoided_{0};
            std::atomic<std::size_t>
                split_lane_coset_count_{0};

            std::atomic<std::size_t>
                split_lane_temporary_bytes_avoided_{0};

            std::atomic<std::size_t>
                split_lane_interleave_bytes_avoided_{0};

            bool cuda_t_edge_transient_leaf_mode_ = false;
        };

        std::vector<std::pair<std::string, algebra::Polynomial>>
        materialize_quotient_polynomials(
            const ProtocolContext &context,
            const TraceArtifacts &trace,
            const std::map<std::string, FieldElement> &challenges,
            const std::vector<std::string> &labels,
            RunMetrics *metrics)
        {
            // One scratch lifetime for the complete quotient family.
            //
            // Sequential large-family materialization can now reuse the
            // coefficient capacity produced by t_edge/FH instead of returning
            // multi-hundred-MiB/GiB buffers to the allocator after every
            // quotient. Worker-thread scratch remains thread_local and is
            // naturally destroyed when each async worker exits.
            const ScopedQuotientCoefficientScratchRelease
                family_coefficient_scratch_release;

            std::vector<
                std::pair<
                    std::string,
                    algebra::Polynomial>>
                out;

            out.reserve(
                labels.size());

            auto worker_count_for_labels = [&]()
            {
                const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
                if (labels.size() <= 1 || cpu_count < 8)
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
                QuotientPolynomialMaterializer materializer(
                    context,
                    trace,
                    challenges,
                    label);
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
            if (worker_count > 1 && t_edge_it != labels.end() && edge_size >= (1ULL << 15))
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
                const auto cpu_count =
                    std::max<std::size_t>(
                        1,
                        std::thread::hardware_concurrency());

                const auto fh_size =
                    context.domains.fh != nullptr
                        ? context.domains.fh->size
                        : 0;

                const auto edge_size =
                    context.domains.edge != nullptr
                        ? context.domains.edge->size
                        : 0;

                const bool large_family_serial_policy =
                    labels.size() > 1 &&
                    (fh_size >= (1ULL << 25) ||
                     edge_size >= (1ULL << 22));

                if (!large_family_serial_policy)
                {
                    for (const auto &label : labels)
                    {
                        out.push_back(
                            materialize_one(label));
                    }

                    return out;
                }

                const auto domain_size_for_label =
                    [&](const std::string &label)
                    -> std::size_t
                {
                    if (label == "t_FH")
                    {
                        return context.domains.fh != nullptr
                                   ? context.domains.fh->size
                                   : 0;
                    }

                    if (label == "t_edge")
                    {
                        return context.domains.edge != nullptr
                                   ? context.domains.edge->size
                                   : 0;
                    }

                    if (label == "t_in")
                    {
                        return context.domains.in != nullptr
                                   ? context.domains.in->size
                                   : 0;
                    }

                    if (label == "t_d" ||
                        label == "t_d_h")
                    {
                        return context.domains.d != nullptr
                                   ? context.domains.d->size
                                   : 0;
                    }

                    if (label == "t_cat")
                    {
                        return context.domains.cat != nullptr
                                   ? context.domains.cat->size
                                   : 0;
                    }

                    if (label == "t_C")
                    {
                        return context.domains.c != nullptr
                                   ? context.domains.c->size
                                   : 0;
                    }

                    if (label == "t_N")
                    {
                        return context.domains.n != nullptr
                                   ? context.domains.n->size
                                   : 0;
                    }

                    return 0;
                };

                constexpr std::size_t
                    kSerialQuotientDomainSize =
                        1ULL << 22;

                std::vector<std::size_t>
                    heavy_indices;

                std::vector<std::size_t>
                    light_indices;

                heavy_indices.reserve(
                    labels.size());

                light_indices.reserve(
                    labels.size());

                for (std::size_t index = 0;
                     index < labels.size();
                     ++index)
                {
                    if (domain_size_for_label(
                            labels[index]) >=
                        kSerialQuotientDomainSize)
                    {
                        heavy_indices.push_back(
                            index);
                    }
                    else
                    {
                        light_indices.push_back(
                            index);
                    }
                }

                std::vector<
                    std::optional<
                        std::pair<
                            std::string,
                            algebra::Polynomial>>>
                    results(
                        labels.size());

                const auto run_one =
                    [&](std::size_t index)
                {
                    results[index] =
                        materialize_one(
                            labels[index]);
                };

                const auto t_edge_index =
                    [&]() -> std::optional<std::size_t>
                {
                    const auto it =
                        std::find(
                            labels.begin(),
                            labels.end(),
                            "t_edge");

                    if (it == labels.end())
                    {
                        return std::nullopt;
                    }

                    return static_cast<std::size_t>(
                        std::distance(
                            labels.begin(),
                            it));
                }();

                if (t_edge_index.has_value() &&
                    std::find(
                        heavy_indices.begin(),
                        heavy_indices.end(),
                        *t_edge_index) !=
                        heavy_indices.end())
                {
                    append_note(
                        metrics,
                        "quotient_large_family_t_edge_first=1");

                    run_one(
                        *t_edge_index);

                    heavy_indices.erase(
                        std::remove(
                            heavy_indices.begin(),
                            heavy_indices.end(),
                            *t_edge_index),
                        heavy_indices.end());
                }

                std::stable_sort(
                    heavy_indices.begin(),
                    heavy_indices.end(),
                    [&](std::size_t lhs,
                        std::size_t rhs)
                    {
                        return domain_size_for_label(
                                   labels[lhs]) >
                               domain_size_for_label(
                                   labels[rhs]);
                    });

                for (const auto index :
                     heavy_indices)
                {
                    run_one(index);
                }

                const auto light_worker_count =
                    std::min<std::size_t>(
                        light_indices.size(),
                        std::min<std::size_t>(
                            3,
                            std::max<std::size_t>(
                                1,
                                cpu_count / 2)));

                if (!light_indices.empty())
                {
                    append_note(
                        metrics,
                        "parallel_quotient_light_wave_workers=" +
                            std::to_string(
                                light_worker_count));
                }

                if (light_worker_count <= 1)
                {
                    for (const auto index :
                         light_indices)
                    {
                        run_one(index);
                    }
                }
                else
                {
                    std::atomic<std::size_t>
                        next_light{0};

                    std::vector<
                        std::future<void>>
                        futures;

                    futures.reserve(
                        light_worker_count);

                    for (std::size_t worker = 0;
                         worker <
                         light_worker_count;
                         ++worker)
                    {
                        futures.push_back(
                            std::async(
                                std::launch::async,
                                [&]()
                                {
                                    while (true)
                                    {
                                        const auto position =
                                            next_light
                                                .fetch_add(
                                                    1);

                                        if (position >=
                                            light_indices
                                                .size())
                                        {
                                            break;
                                        }

                                        run_one(
                                            light_indices[position]);
                                    }
                                }));
                    }

                    for (auto &future :
                         futures)
                    {
                        future.get();
                    }
                }

                for (auto &result :
                     results)
                {
                    if (!result.has_value())
                    {
                        throw std::runtime_error(
                            "missing large-family quotient "
                            "materialization result");
                    }

                    out.push_back(
                        std::move(
                            *result));
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
            if (label == "t_FH")
                return context.domains.fh->size;
            if (label == "t_edge")
                return context.domains.edge->size;
            if (label == "t_in")
                return context.domains.in->size;
            if (label == "t_d" || label == "t_d_h")
                return context.domains.d->size;
            if (label == "t_cat")
                return context.domains.cat->size;
            if (label == "t_C")
                return context.domains.c->size;
            if (label == "t_N")
                return context.domains.n->size;
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
                "P_J_tbl_feat",
                "P_J_qry_feat",
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
                "Div_Active",
                "Div_X",
                "Div_D",
                "Div_Y",
                "Div_Sigma",
                "Div_Zero",
                "Div_A",
                "Div_A_inv",
                "Div_D_inv",
                "Div_Q",
                "Div_R",
                "Div_Delta",
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
            for (std::size_t layer_index = 0;
                 layer_index < context.model.hidden_layers.size();
                 ++layer_index)
                for (std::size_t head_index = 0;
                     head_index < context.model.hidden_layers[layer_index].heads.size();
                     ++head_index)
                {
                    const auto prefix = hidden_head_prefix(layer_index, head_index) + "_";
                    append_prefixed(prefix, hidden_suffixes);
                    append_prefixed(prefix, division_suffixes);
                    append_division_range_dependencies(prefix + "Div_");
                    append_label(hidden_q_nd_h_selector_label(layer_index));
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
                if (relation == metadata.end() || relation->coefficient_basis_relations.size() != 1)
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
            for (std::size_t layer_index = 0;
                 layer_index < context.model.hidden_layers.size();
                 ++layer_index)
                append_unique_labels(
                    &labels, {hidden_q_in_selector_label(layer_index)});
            return labels;
        }

        std::vector<std::string> multihead_d_dependencies(const ProtocolContext &context)
        {
            std::vector<std::string> labels = domain_opening_labels(context, "d");
            for (std::size_t layer_index = 0;
                 layer_index < context.model.hidden_layers.size();
                 ++layer_index)
                append_unique_labels(
                    &labels, {hidden_q_d_h_selector_label(layer_index)});
            return labels;
        }

        std::vector<std::string> multihead_cat_dependencies(const ProtocolContext &context)
        {
            std::vector<std::string> labels = domain_opening_labels(context, "cat");
            for (std::size_t layer_index = 0;
                 layer_index < context.model.hidden_layers.size();
                 ++layer_index)
                append_unique_labels(
                    &labels, {hidden_q_cat_selector_label(layer_index)});
            append_unique_labels(&labels, {output_q_cat_selector_label()});
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

        std::vector<std::string> quotient_dependency_labels(
            const ProtocolContext &context,
            const std::string &quotient_name)
        {
            if (quotient_name == "t_FH")
                return quotient_dependencies_fh();
            if (quotient_name == "t_edge")
                return multihead_edge_dependencies(context);
            if (quotient_name == "t_in")
                return multihead_in_dependencies(context);
            if (quotient_name == "t_d_h")
                return multihead_d_dependencies(context);
            if (quotient_name == "t_cat")
                return multihead_cat_dependencies(context);
            if (quotient_name == "t_C")
                return multihead_c_dependencies(context);
            if (quotient_name == "t_N")
                return multihead_n_dependencies(context);
            return {};
        }

        std::vector<std::string> quotient_materialization_dependency_labels(
            const ProtocolContext &context,
            const std::string &quotient_name)
        {
            auto labels =
                quotient_dependency_labels(context, quotient_name);
            if (quotient_name != "t_FH" || !full_graph_feature_identity_enabled(context))
            {
                return labels;
            }
            static const std::array<std::string_view, 7>
                identically_cancelled_dependencies = {
                    "P_H",
                    "P_T_H",
                    "P_Row_feat_tbl",
                    "P_Col_feat_tbl",
                    "P_Row_feat_qry",
                    "P_Col_feat_qry",
                    "P_I_feat_qry",
                };
            std::erase_if(
                labels,
                [&](const std::string &label)
                {
                    return std::find(
                               identically_cancelled_dependencies.begin(),
                               identically_cancelled_dependencies.end(),
                               label) != identically_cancelled_dependencies.end();
                });
            return labels;
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

    std::vector<algebra::FieldElement>
    evaluate_spilled_evaluation_polynomial_points(
        const TraceArtifacts::SpilledEvaluationPolynomial &spilled,
        const std::shared_ptr<algebra::RootOfUnityDomain> &domain,
        const std::vector<algebra::FieldElement> &points,
        const std::vector<const algebra::DomainEvaluationWeights *> &weights)
    {
        return evaluate_spilled_evaluation_polynomial_points_impl(
            spilled,
            domain,
            points,
            weights);
    }

    algebra::FieldElement evaluate_spilled_evaluation_polynomial(
        const TraceArtifacts::SpilledEvaluationPolynomial &spilled,
        const std::shared_ptr<algebra::RootOfUnityDomain> &domain,
        const algebra::FieldElement &point,
        const algebra::DomainEvaluationWeights &weights)
    {
        return evaluate_spilled_evaluation_polynomial_points_impl(
                   spilled,
                   domain,
                   std::vector<algebra::FieldElement>{point},
                   std::vector<const algebra::DomainEvaluationWeights *>{&weights})
            .front();
    }

    std::vector<std::vector<algebra::FieldElement>>
    evaluate_spilled_evaluation_requests(
        const std::vector<SpilledEvaluationRequest> &requests,
        std::size_t max_workers)
    {
        constexpr std::size_t kWorkerCeiling = 4;
        constexpr std::size_t kMinParallelFieldElements = 1U << 18U;

        std::vector<std::vector<algebra::FieldElement>> results(requests.size());
        if (requests.empty())
        {
            return results;
        }

        std::size_t bounded_work = 0;
        std::size_t sequential_job_count = 0;
        for (const auto &request : requests)
        {
            if (request.spilled == nullptr)
            {
                throw std::runtime_error("spilled evaluation batch received null request");
            }
            if (request.domain == nullptr || request.points.size() != request.weights.size())
            {
                throw std::runtime_error("spilled evaluation batch received invalid point weights");
            }
            bool requires_sequential_scan = false;
            for (const auto *weights : request.weights)
            {
                if (weights == nullptr)
                {
                    throw std::runtime_error("spilled evaluation batch received null point weights");
                }
                requires_sequential_scan = requires_sequential_scan || !weights->direct_index.has_value();
            }
            if (!requires_sequential_scan)
            {
                continue;
            }
            ++sequential_job_count;
            const auto remaining = kMinParallelFieldElements - bounded_work;
            bounded_work += std::min(request.spilled->size, remaining);
        }
        const auto available_workers = std::max<std::size_t>(
            1, std::thread::hardware_concurrency());
        const bool use_parallel = sequential_job_count >= 2 && bounded_work >= kMinParallelFieldElements && max_workers >= 2 && available_workers >= 2;
        const auto worker_count = use_parallel
                                      ? std::min({kWorkerCeiling,
                                                  max_workers,
                                                  available_workers,
                                                  requests.size()})
                                      : std::size_t{1};

        auto run_job = [&](std::size_t job_index)
        {
            const auto &request = requests[job_index];
            results[job_index] =
                evaluate_spilled_evaluation_polynomial_points_impl(
                    *request.spilled,
                    request.domain,
                    request.points,
                    request.weights);
        };
        if (worker_count == 1)
        {
            for (std::size_t job_index = 0;
                 job_index < requests.size();
                 ++job_index)
            {
                run_job(job_index);
            }
            return results;
        }

        std::atomic<std::size_t> next_job{0};
        std::vector<std::future<void>> workers;
        workers.reserve(worker_count);
        for (std::size_t worker = 0; worker < worker_count; ++worker)
        {
            workers.push_back(std::async(std::launch::async, [&]()
                                         {
                while (true)
                {
                    const auto job_index = next_job.fetch_add(1);
                    if (job_index >= requests.size())
                    {
                        return;
                    }
                    run_job(job_index);
                } }));
        }
        std::exception_ptr first_error;
        for (auto &worker : workers)
        {
            try
            {
                worker.get();
            }
            catch (...)
            {
                if (first_error == nullptr)
                {
                    first_error = std::current_exception();
                }
            }
        }
        if (first_error != nullptr)
        {
            std::rethrow_exception(first_error);
        }
        return results;
    }

    // 构建一次 formal run 的静态上下文，产出数据集、模型、工作域、公开列与静态承诺。
    // 这里禁止注入任何依赖单次 proof 的 witness 或挑战状态。
    struct StaticContextCachePayload
    {
        ProtocolContext context;
        // The cache key is retained in the payload so every hit can re-check
        // the current private resource identity instead of trusting a map
        // lookup alone.
        std::string identity;
        std::string compiler_artifact_identity;
    };

    std::string static_context_compiler_artifact_identity(
        const ProtocolContext &context)
    {
        if (context.compiler_artifact == nullptr || context.validated_compiler_artifact_identity != context.compiler_artifact.get())
            return "missing";
        return "sha256:" + cache_sha256_hex(
                               context.compiler_artifact->protocol_canonical_bytes +
                               context.compiler_artifact->instance_canonical_bytes);
    }

    ProtocolContext build_context(const util::AppConfig &config, RunMetrics *metrics)
    {
        // Resolve the registered scale chain before cache lookup so every
        // context identity includes the complete typed chain, not only an ID.
        auto canonical_config = config;
        canonical_config.quantization = util::canonical_quantization_scale_chain(
            config.quant_cfg_id,
            config.hidden_profile,
            config.K_out,
            (std::filesystem::path(config.project_root) / "configs/quantization_scale_chain.cfg").string());
        static std::mutex cache_mutex;
        static std::unordered_map<std::string, StaticContextCachePayload> cache;
        // 这里只复用静态上下文对象，见证和挑战相关状态必须按每次正式证明重新生成。
        const bool allow_static_context_cache = true;

        const auto cache_lookup_start = Clock::now();
        const auto cache_key = context_cache_key(canonical_config);
        if (allow_static_context_cache)
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            if (const auto it = cache.find(cache_key); it != cache.end())
            {
                const auto &payload = it->second;
                const bool compiler_identity_ok =
                    payload.compiler_artifact_identity != "missing" &&
                    payload.compiler_artifact_identity ==
                        static_context_compiler_artifact_identity(payload.context);
                if (payload.identity == cache_key && compiler_identity_ok)
                {
                    // 这里只缓存静态上下文：工作域、SRS、公开列、静态承诺和模型数据。
                    // 每次 proof 的 witness 与挑战派生值都必须重新生成。
                    auto node = cache.extract(it);
                    auto context = std::move(node.mapped().context);
                    context.config = canonical_config;
                    if (metrics != nullptr)
                    {
                        metrics->load_static_ms += elapsed_ms(cache_lookup_start, Clock::now());
                        append_note(metrics, "static_context_cache=hit_identity_revalidated");
                    }
                    return context;
                }
                // A same-path replacement, missing resource, or corrupted
                // cached payload is a hard miss. Never expose a cached static
                // context to a new proof statement.
                cache.erase(it);
                append_note(metrics, "static_context_cache=stale_identity_rejected");
            }
        }

        append_note(metrics, "static_context_cache=miss");
        ProtocolContext context;
        context.config = canonical_config;
        // All model commitments below use the canonical, typed,
        // layer/head-aware scale chain selected above.  Context construction
        // selects one typed scale chain for the formal registry; no
        // unvalidated scale source can enter this path.
        const auto &scales = context.config.quantization;

        progress_note("context:load_static:start");
        auto start = Clock::now();

        // 先把数据集和 checkpoint bundle 收束成 formal shape；
        // 这一段还不接触任何 challenge 或 witness。
        context.dataset = data::load_dataset(canonical_config);
        context.local =
            data::normalize_graph_input(
                context.dataset,
                canonical_config);
        data::validate_input_feature_views(
            context.dataset,
            context.local,
            scales.s_input,
            canonical_config.input_preprocess_id);

        if (config.checkpoint_bundle.empty())
        {
            throw std::runtime_error(
                "formal model route requires checkpoint_bundle; "
                "refusing to fall back to synthetic parameters");
        }

        const auto checkpoint_root =
            resolve_project_path(
                config.project_root,
                config.checkpoint_bundle);

        context.model =
            model::load_checkpoint_bundle_parameters(
                checkpoint_root.string());

        if (context.model.quantization_scale_chain != scales.canonical_serialization())
        {
            throw std::runtime_error(
                "checkpoint bundle quantization scale chain conflicts with formal config");
        }
        if (context.model.input_preprocess_id != canonical_config.input_preprocess_id)
        {
            throw std::runtime_error(
                "checkpoint bundle input_preprocess_id conflicts with formal config");
        }

        append_note(
            metrics,
            "model_source=checkpoint_bundle");

        std::size_t formal_hidden_head_count = 0;
        for (const auto &layer : context.model.hidden_layers)
            formal_hidden_head_count += layer.heads.size();
        append_note(
            metrics,
            "hidden_head_count=" +
                std::to_string(formal_hidden_head_count));

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
        if (context.model.model_arch_id != config.model_arch_id)
        {
            throw std::runtime_error(
                "config model_arch_id conflicts with checkpoint manifest model_arch_id");
        }
        if (context.model.model_param_id != config.model_param_id)
        {
            throw std::runtime_error(
                "config model_param_id conflicts with checkpoint manifest model_param_id");
        }
        if (context.model.quant_cfg_id != config.quant_cfg_id)
        {
            throw std::runtime_error(
                "config quant_cfg_id conflicts with checkpoint manifest quant_cfg_id");
        }
        if (context.model.static_table_id != config.static_table_id)
        {
            throw std::runtime_error(
                "config static_table_id conflicts with checkpoint manifest static_table_id");
        }
        if (context.model.degree_bound_id != config.degree_bound_id)
        {
            throw std::runtime_error(
                "config degree_bound_id conflicts with checkpoint manifest degree_bound_id");
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
            [&]()
            {
                const auto checked_product = [](std::size_t left,
                                                std::size_t right,
                                                std::string_view label)
                {
                    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right)
                        throw std::runtime_error(
                            "feature table size overflows size_t: " + std::string(label));
                    return left * right;
                };
                return std::max(
                           checked_product(
                               context.dataset.num_nodes,
                               context.dataset.num_features,
                               "dataset"),
                           checked_product(
                               context.local.num_nodes,
                               context.local.num_features,
                               "local")) +
                       2;
            }());
        const auto dataset_feature_count = context.dataset.num_nodes * context.dataset.num_features;
        const std::size_t d_h = hidden_head_width(context.model);
        const std::size_t d_cat = concat_width(context.model);
        // The in-domain is a shared power-of-two capacity.  It must not be
        // confused with any formal selector: each Q_in below uses its own
        // layer.input_dim support.
        const std::size_t in_domain_capacity =
            std::max<std::size_t>(
                context.local.num_features,
                model::max_hidden_input_dim(context.model));
        const auto minimum_edge_rows = minimum_edge_trace_length(
            context.local.num_nodes,
            context.local.edges.size(),
            d_h,
            context.local.num_classes);
        // The same pure planner drives canonical table generation and this
        // pre-SRS domain calculation.  In particular, ExpMap rows are derived
        // from the certified LReLU output bound; T_range remains exactly 2^B.
        const auto table_shape = plan_canonical_static_table_shape(context);
        const std::size_t edge_size = algebra::next_power_of_two(
            std::max<std::size_t>({
                minimum_edge_rows,
                table_shape.lrelu_table_size,
                table_shape.elu_table_size,
                table_shape.exp_table_size,
                table_shape.range_table_size,
            }) +
            2);
        const std::size_t in_size = algebra::next_power_of_two(in_domain_capacity + 2);
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
            .hidden_layers = {},
            .output_head = std::nullopt,
        };
        context.domains.hidden_layers.resize(
            context.model.hidden_layers.size());
        for (std::size_t layer_index = 0;
             layer_index < context.model.hidden_layers.size();
             ++layer_index)
        {
            auto &layer_domains = context.domains.hidden_layers[layer_index];
            layer_domains.resize(
                context.model.hidden_layers[layer_index].heads.size(),
                AttentionHeadDomains{
                    .in = context.domains.in,
                    .d = context.domains.d,
                });
        }
        context.domains.output_head = AttentionHeadDomains{.in = context.domains.cat, .d = context.domains.c};
        end = Clock::now();
        if (metrics != nullptr)
        {
            metrics->fft_plan_ms += elapsed_ms(start, end);
        }
        progress_note("context:domains:done");

        progress_note("context:srs:start");
        start = Clock::now();
        // Every registered quotient is materialized on a 4|D| coset below.
        // Check the complete coefficient resource here so an undersized SRS
        // fails during context construction rather than midway through prove.
        const std::size_t required_coefficient_count = std::max<std::size_t>({
            fh_size * 4,
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
        // The canonical producer lives in the schema module and is also used
        // by the independent verifier registration loader.  Keeping one
        // producer ensures that a table with the right shape cannot diverge in
        // one row from the table that was formally generated and committed.
        context.tables = build_canonical_static_tables(context);
        std::string static_table_reason;
        if (!validate_static_tables(context, &static_table_reason))
            throw std::runtime_error(
                "canonical static table validation failed: " + static_table_reason);

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
        // Each hidden layer owns its own valid selectors.  The selector
        // support is the layer's real shape; the surrounding work domain
        // may still be a shared power-of-two capacity.
        for (std::size_t layer_index = 0;
             layer_index < context.model.hidden_layers.size();
             ++layer_index)
        {
            const auto &layer = context.model.hidden_layers[layer_index];
            const auto q_in_label = hidden_q_in_selector_label(layer_index);
            const auto q_d_label = hidden_q_d_h_selector_label(layer_index);
            const auto q_cat_label = hidden_q_cat_selector_label(layer_index);
            const auto q_nd_h_label = hidden_q_nd_h_selector_label(layer_index);
            add_public_poly(
                context,
                q_in_label,
                make_shared_eval_poly(
                    q_in_label,
                    build_selector(layer.input_dim, in_size),
                    context.domains.in));
            add_public_poly(
                context,
                q_d_label,
                make_shared_eval_poly(
                    q_d_label,
                    build_selector(layer.shape.head_dim, d_size),
                    context.domains.d));
            add_public_poly(
                context,
                q_cat_label,
                make_shared_eval_poly(
                    q_cat_label,
                    build_selector(layer.shape.head_count * layer.shape.head_dim, cat_size),
                    context.domains.cat));
            add_public_poly(
                context,
                q_nd_h_label,
                make_shared_eval_poly(
                    q_nd_h_label,
                    build_selector(context.local.num_nodes * layer.shape.head_dim, edge_size),
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
            if (edge.src >= context.local.num_nodes || edge.dst >= context.local.num_nodes)
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
            output_q_cat_selector_label(),
            make_shared_eval_poly(
                output_q_cat_selector_label(),
                build_selector(context.model.output_layer.input_dim, cat_size),
                context.domains.cat));
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
        if (lazy_fh_public_enabled(context))
        {
            add_public_poly(
                context,
                "P_Q_tbl_feat",
                make_shared_eval_poly("P_Q_tbl_feat", build_selector(dataset_feature_count, fh_size), context.domains.fh));
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
                add_public_poly_from_existing_material_checked(context, "P_Q_qry_feat", "P_Q_tbl_feat");
                add_public_poly_from_existing_material_checked(context, "P_Row_feat_qry", "P_Row_feat_tbl");
                add_public_poly_from_existing_material_checked(context, "P_Col_feat_qry", "P_Col_feat_tbl");
                add_public_poly_from_existing_material_checked(context, "P_I_feat_qry", "P_Row_feat_tbl");
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
                make_shared_eval_poly("P_Q_tbl_feat", build_selector(dataset_feature_count, fh_size), context.domains.fh));
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
                add_public_poly_from_existing_material_checked(context, "P_Q_qry_feat", "P_Q_tbl_feat");
                add_public_poly_from_existing_material_checked(context, "P_Row_feat_qry", "P_Row_feat_tbl");
                add_public_poly_from_existing_material_checked(context, "P_Col_feat_qry", "P_Col_feat_tbl");
                add_public_poly_from_existing_material_checked(context, "P_I_feat_qry", "P_Row_feat_tbl");
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
        const auto &feature_matrix_for_h =
            !context.dataset.features.empty() ? context.dataset.features : context.local.features;
        if (lazy_fh_public_enabled(context))
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

        // 模型参数在 context 阶段就固定成静态承诺，后面的 trace 只引用标签，不再重复构造权重对象。
        for (std::size_t layer_index = 0;
             layer_index < context.model.hidden_layers.size();
             ++layer_index)
        {
            const auto &layer = context.model.hidden_layers[layer_index];
            for (std::size_t head_index = 0;
                 head_index < layer.heads.size();
                 ++head_index)
            {
                const auto &head = layer.heads[head_index];
                const auto weight = quantize_model_matrix(
                    head.seq_kernel_fp,
                    scales.s_w_hidden.at(layer_index).at(head_index));
                const auto column_count = weight.empty() ? 0 : weight.front().size();
                for (std::size_t column = 0; column < column_count; ++column)
                {
                    std::vector<FieldElement> values(
                        context.domains.in->size, FieldElement::zero());
                    for (std::size_t row = 0; row < weight.size(); ++row)
                        values[row] = weight[row][column];
                    const auto label = hidden_weight_column_label(
                        layer_index, head_index, column);
                    add_static_commitment(
                        context, label,
                        make_eval_poly(label, values, context.domains.in));
                }
                const auto src_label = hidden_src_label(layer_index, head_index);
                add_static_commitment(
                    context, src_label,
                    make_eval_poly(
                        src_label,
                        padded(quantize_model_vector_at_scale(
                                   head.attn_src_kernel_fp,
                                   scales.s_a_src_hidden.at(layer_index).at(head_index)),
                               context.domains.d->size),
                        context.domains.d));
                const auto dst_label = hidden_dst_label(layer_index, head_index);
                add_static_commitment(
                    context, dst_label,
                    make_eval_poly(
                        dst_label,
                        padded(quantize_model_vector_at_scale(
                                   head.attn_dst_kernel_fp,
                                   scales.s_a_dst_hidden.at(layer_index).at(head_index)),
                               context.domains.d->size),
                        context.domains.d));
            }
        }
        for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index)
        {
            const auto &head = context.model.output_layer.heads[head_index];
            const auto weight = quantize_model_matrix(
                head.seq_kernel_fp,
                scales.s_w_output.at(head_index));
            const auto column_count = weight.empty() ? 0 : weight.front().size();
            for (std::size_t column = 0; column < column_count; ++column)
            {
                std::vector<FieldElement> values(context.domains.cat->size, FieldElement::zero());
                for (std::size_t row = 0; row < weight.size(); ++row)
                    values[row] = weight[row][column];
                const auto label = output_weight_column_label(head_index, column);
                add_static_commitment(context, label, make_eval_poly(label, values, context.domains.cat));
            }
            add_static_commitment(
                context,
                output_src_label(head_index),
                make_eval_poly(
                    output_src_label(head_index),
                    padded(quantize_model_vector_at_scale(
                               head.attn_src_kernel_fp,
                               scales.s_a_src_output.at(head_index)),
                           context.domains.c->size),
                    context.domains.c));
            add_static_commitment(
                context,
                output_dst_label(head_index),
                make_eval_poly(
                    output_dst_label(head_index),
                    padded(quantize_model_vector_at_scale(
                               head.attn_dst_kernel_fp,
                               scales.s_a_dst_output.at(head_index)),
                           context.domains.c->size),
                    context.domains.c));
            add_static_commitment(
                context,
                output_bias_label(head_index),
                make_eval_poly(
                    output_bias_label(head_index),
                    padded(
                        quantize_model_vector(
                            head.output_bias_fp,
                            scales.s_b_output.at(head_index)),
                        context.domains.c->size),
                    context.domains.c));
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
                "instance ProtocolCompilerArtifact validation failed: " + compiler_artifact_reason);
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

        const auto final_cache_key = context_cache_key(context.config);
        if (final_cache_key != cache_key)
            throw std::runtime_error(
                "static context identity changed while loading private resources");
        const bool cacheable_static_context =
            context.domains.fh != nullptr && context.domains.edge != nullptr && context.domains.fh->size < (1ULL << 24) && context.domains.edge->size < (1ULL << 22);
        if (allow_static_context_cache && cacheable_static_context)
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            cache[cache_key] = StaticContextCachePayload{
                .context = context,
                .identity = final_cache_key,
                .compiler_artifact_identity =
                    static_context_compiler_artifact_identity(context),
            };
        }
        return context;
    }

    // 消费 trace 生成最终 proof，对外保证 commitment 顺序、block 顺序与 opening 结构稳定。
    Proof prove(const ProtocolContext &context, const TraceArtifacts &trace, RunMetrics *metrics)
    {
        if (context.verifier_shape_only)
            throw std::runtime_error(
                "prover rejects verifier shape-only execution context");
        // Prove is the commitment-producing boundary.  Revalidate the input
        // views here as well because a caller may retain a trace and mutate
        // or replace the context after build_trace returned.
        data::validate_input_feature_views(
            context.dataset,
            context.local,
            context.config.quantization.s_input,
            context.config.input_preprocess_id);
        const auto &route2 = util::route2_options();
        // prove 入口只消费 trace 和固定挑战顺序，不在这里改写 proof schema 或 block 顺序。
        if (trace.commitment_order != dynamic_commitment_labels(context))
        {
            throw std::runtime_error("trace commitment order does not match the protocol commitment order");
        }

        ScopedCudaProverCacheRelease
            cuda_cache_release_guard;

        auto stage_start = Clock::now();
        progress_note("prove:quotient:start");
        // 先重放动态承诺之前的 transcript，再在这套固定挑战上构造各工作域 quotient。
        const auto pre_quotient_challenges = replay_challenges(context, trace.commitments, {}, &trace.public_metadata);
        std::shared_ptr<const ProofEvaluationBackendRegistry> backend_registry;
        if (route2.packed_fft_backend)
        {
            // Packed backends retain non-owning polynomial pointers.
            // Keep this registry proof-local so those pointers never
            // outlive the current context or trace.  Construction only
            // indexes labels; subset materialization remains lazy.
            backend_registry =
                std::make_shared<ProofEvaluationBackendRegistry>(
                    context, trace);
        }
        auto proof_domain_weight_cache = shared_proof_domain_weight_cache();
        const std::vector<std::string> quotient_labels = {"t_FH", "t_edge", "t_in", "t_d_h", "t_cat", "t_C", "t_N"};
        append_note(metrics, "quotient_cache=miss");

        // On integrated-memory CUDA systems a large resident SRS cache
        // competes directly with the file-backed quotient working set.
        //
        // Keep the cache across phases for ordinary workloads, where its
        // reuse is beneficial. Release it only for quotient shapes that
        // already enter the large edge/FH working-set class.
        const bool large_quotient_working_set =
            (context.domains.edge != nullptr &&
             context.domains.edge->size >= (1ULL << 23)) ||
            (context.domains.fh != nullptr &&
             context.domains.fh->size >= (1ULL << 25));

        if (route2.cuda_field_hotspots &&
            large_quotient_working_set)
        {
            // Scalar conversion buffers and streamed-MSM staging are not
            // useful during quotient construction. Release those first
            // without throwing away immutable SRS bases.
            crypto::
                release_cuda_g1_msm_execution_workspace();

            constexpr std::size_t
                kMinimumRetainedCudaMsmBaseBytes =
                    std::size_t{512} << 20;

            constexpr std::size_t
                kMaximumIntegratedRetainedCudaMsmBaseBytes =
                    std::size_t{4} << 30;

            std::size_t
                retained_cuda_msm_base_bytes =
                    kMinimumRetainedCudaMsmBaseBytes;

            if (crypto::
                    cuda_g1_msm_integrated_memory_device())
            {
                const auto host_budget =
                    quotient_coset_memory_budget();

                if (host_budget.available_bytes >
                    host_budget.reserve_bytes)
                {
                    const auto live_after_reserve =
                        host_budget.available_bytes -
                        host_budget.reserve_bytes;

                    // Give only 1/16 of currently safe host/unified
                    // headroom to cross-phase SRS residency.  Quotient
                    // cosets retain the dominant share while useful
                    // PPI/OGBN Lagrange prefixes can survive.
                    retained_cuda_msm_base_bytes =
                        std::clamp(
                            live_after_reserve /
                                std::size_t{16},
                            kMinimumRetainedCudaMsmBaseBytes,
                            kMaximumIntegratedRetainedCudaMsmBaseBytes);
                }
            }

            progress_note(
                "prove:quotient:"
                "trim_cuda_msm_cache:start:"
                "maximum_bytes=" +
                std::to_string(
                    retained_cuda_msm_base_bytes));

            crypto::
                trim_cuda_g1_msm_resident_cache(
                    retained_cuda_msm_base_bytes);

            progress_note(
                "prove:quotient:"
                "trim_cuda_msm_cache:done:"
                "maximum_bytes=" +
                std::to_string(
                    retained_cuda_msm_base_bytes));
        }

        auto named_quotient_polynomials = materialize_quotient_polynomials(
            context,
            trace,
            pre_quotient_challenges,
            quotient_labels,
            metrics);
        named_quotient_polynomials = blind_quotient_polynomials(
            context,
            std::move(named_quotient_polynomials));

        if (route2.cuda_field_hotspots)
        {
            progress_note(
                "prove:quotient:"
                "release_cuda_transform_transients:start");

            // Quotient coefficients are now canonical host-side prover
            // material. t_edge execution state and CUDA NTT workspace are
            // dead, so release them before coefficient-SRS KZG waves.
            release_cuda_t_edge_executor_cache();

            algebra::release_cuda_ntt_workspace();

            // Scalar/stream MSM staging is also phase-local.  Preserve
            // resident immutable SRS bases: they are useful again for
            // quotient commitments and, later, folded openings.
            crypto::
                release_cuda_g1_msm_execution_workspace();

            progress_note(
                "prove:quotient:"
                "release_cuda_transform_transients:done");
        }

        progress_note("prove:quotient:commit_batch:start");
        const auto quotient_commit_batch_start = Clock::now();
        crypto::CommitBatchProfile quotient_commit_profile;
        auto quotient_commitments = [&]()
        {
            // The owned batch may use CUDA waves or CPU worker
            // dispatch. Measure the synchronously joined call as one
            // wall-clock commitment span; never sum per-worker MSM
            // profiles into the formal total.
            timing::ScopedCommitment commitment_scope(metrics);
            return batch_quotient_commitments(
                std::move(named_quotient_polynomials),
                context.kzg,
                metrics != nullptr ? &quotient_commit_profile : nullptr);
        }();
        progress_note(
            "prove:quotient:commit_batch:done:ms=" + std::to_string(elapsed_ms(
                                                         quotient_commit_batch_start, Clock::now())));
        if (metrics != nullptr)
        {
            metrics->quotient_commitment_ms +=
                quotient_commit_profile.msm_ms;
        }
        // Quotient polynomials and commitments are proof-specific and must not be
        // reused by a later fresh-proof measurement.
        const auto challenges = replay_challenges(context, trace.commitments, quotient_commitments, &trace.public_metadata);
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
            const auto total_ms = elapsed_ms(local_start, Clock::now());
            progress_note(
                progress_prefix + ":done:gather_ms=" + std::to_string(gather_ms) + ":total_ms=" + std::to_string(total_ms));
            return {
                spec.bundle_name,
                std::move(bundle),
                gather_ms,
                total_ms,
                std::move(local_metrics),
            };
        };
        std::vector<BundleBuildResult> bundle_results;
        bundle_results.reserve(bundle_specs.size());
        // 并行构造 opening bundle 之前先预热 barycentric 权重和 FH 公开值缓存。
        // 大域权重向量体量极大，若在多个 bundle 中同时构造，会明显放大缺页和带宽争抢。

        const auto spilled_bytes_for_domain =
            [&](const std::string &domain_name)
        {
            std::size_t bytes = 0;

            for (const auto &[name, spilled] :
                 trace.spilled_polynomials)
            {
                (void)name;

                if (spilled.domain_name !=
                    domain_name)
                {
                    continue;
                }

                if (spilled.size >
                    std::numeric_limits<
                        std::size_t>::max() /
                        sizeof(FieldElement))
                {
                    throw std::runtime_error(
                        "domain opening spill "
                        "byte overflow");
                }

                const auto added =
                    spilled.size *
                    sizeof(FieldElement);

                if (bytes >
                    std::numeric_limits<
                        std::size_t>::max() -
                        added)
                {
                    throw std::runtime_error(
                        "domain opening spill "
                        "byte overflow");
                }

                bytes += added;
            }

            return bytes;
        };

        constexpr std::size_t
            kLargeOpeningSpillBytes =
                std::size_t{2} << 30;

        std::vector<std::size_t>
            large_indices;

        std::vector<std::size_t>
            small_indices;

        for (std::size_t index = 0;
             index < bundle_specs.size();
             ++index)
        {
            const auto spill_bytes =
                spilled_bytes_for_domain(
                    bundle_specs[index]
                        .trace_domain_name);

            if (route2.cuda_field_hotspots &&
                spill_bytes >=
                    kLargeOpeningSpillBytes)
            {
                large_indices.push_back(
                    index);
            }
            else
            {
                small_indices.push_back(
                    index);
            }
        }

        std::sort(
            large_indices.begin(),
            large_indices.end(),
            [&](std::size_t lhs,
                std::size_t rhs)
            {
                return spilled_bytes_for_domain(
                           bundle_specs[lhs]
                               .trace_domain_name) >
                       spilled_bytes_for_domain(
                           bundle_specs[rhs]
                               .trace_domain_name);
            });

        for (const auto index :
             large_indices)
        {
            const auto &spec =
                bundle_specs[index];

            const auto z =
                challenges.at(
                    spec.z_name);

            proof_domain_weight_cache->get(
                spec.domain,
                z);

            proof_domain_weight_cache->get(
                spec.domain,
                z * spec.domain->omega);

            bundle_results.push_back(
                build_bundle(spec));
        }

        if (std::thread::hardware_concurrency() >
            1)
        {
            std::vector<
                std::future<
                    BundleBuildResult>>
                futures;

            futures.reserve(
                small_indices.size());

            for (const auto index :
                 small_indices)
            {
                futures.push_back(
                    std::async(
                        std::launch::async,
                        build_bundle,
                        bundle_specs[index]));
            }

            for (auto &future :
                 futures)
            {
                bundle_results.push_back(
                    future.get());
            }
        }
        else
        {
            for (const auto index :
                 small_indices)
            {
                bundle_results.push_back(
                    build_bundle(
                        bundle_specs[index]));
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
        const auto domain_bundle_wall_ms =
            elapsed_ms(stage_start, Clock::now());
        if (metrics != nullptr)
        {
            metrics->domain_opening_ms += domain_bundle_wall_ms;
        }
        progress_note(
            "prove:domain_opening:done:ms=" + std::to_string(domain_bundle_wall_ms));

        // Every domain-opening scalar value has now been gathered into
        // proof.domain_openings. The barycentric weight vectors are not
        // needed by point-group witness construction below.
        //
        // In particular, large FH/edge instances can retain multiple
        // GiB of challenge-specific weight vectors. Release them before
        // folded opening plans allocate their own full-domain buffers.
        proof_domain_weight_cache.reset();

        progress_note(
            "prove:domain_opening:weight_cache:released");

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
                        "missing Pi_bind global opening bundle: " + std::string(expected_bundle_name));
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
                        [&](const auto &entry)
                        { return entry.first == label; });
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
            [](const auto &entry)
            { return entry.first == "N"; });
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
                            "canonical matrix value missing from N opening: " + column);
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
        const auto append_binding_domain_values = [&](
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
                    "aggregation binding opening bundle missing: " + bundle_name);
            for (const auto &object : objects)
            {
                const auto value_it = std::find_if(
                    bundle_it->second.values.begin(),
                    bundle_it->second.values.end(),
                    [&](const auto &entry)
                    { return entry.first == object; });
                if (value_it == bundle_it->second.values.end())
                    throw std::runtime_error(
                        "aggregation binding domain value missing: " + object);
                subproof.evaluations.push_back(BindingEvaluation{
                    .domain = bundle_name,
                    .object_id = object,
                    .values = value_it->second,
                    .kind = PayloadValueKind::MaskedPrivateValue,
                });
            }
            for (const auto &point : bundle_it->second.points)
            {
                const auto group_id = "Open_dom:" + bundle_name + ":" + point.to_string();
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
        for (const auto &spec : bundle_specs)
        {
            const auto bundle_it = std::find_if(
                proof.domain_openings.begin(),
                proof.domain_openings.end(),
                [&](const auto &entry)
                { return entry.first == spec.bundle_name; });
            if (bundle_it == proof.domain_openings.end())
            {
                throw std::runtime_error(
                    "missing domain bundle while building global point groups: " + spec.bundle_name);
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
            external_commitments.push_back({spec.static_commitment ? context.static_commitments.at(spec.label) : trace.commitments.at(spec.label),
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
                if (evaluation == subproof.evaluations.end() || evaluation->values.size() != 1 || evaluation->kind != PayloadValueKind::MaskedPrivateValue)
                    throw std::runtime_error(
                        "missing route terminal shift value: " + relation.relation_id);
                const auto y_rel = challenges.at(
                    relation.challenge_label);
                external_commitments.push_back({trace.commitments.at(
                                                    relation.shifted_accumulator_object),
                                                evaluation->values.front()});
                external_points.push_back(y_rel);
                external_commitments.push_back({trace.commitments.at(
                                                    relation.edge_accumulator_object),
                                                evaluation->values.front()});
                external_points.push_back(
                    y_rel + relation.terminal_delta);
                external_commitments.push_back({trace.commitments.at(
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
                    if (evaluation == subproof.evaluations.end() || evaluation->values.size() != 1 || evaluation->kind != PayloadValueKind::MaskedPrivateValue)
                        throw std::runtime_error(
                            "missing aggregation terminal shift value: " + route->relation_id);
                    const auto y_rel = challenges.at(
                        route->terminal_challenge_label);
                    external_commitments.push_back({trace.commitments.at(
                                                        route->shifted_accumulator_object),
                                                    evaluation->values.front()});
                    external_points.push_back(y_rel);
                    external_commitments.push_back({trace.commitments.at(
                                                        route->edge_accumulator_object),
                                                    evaluation->values.front()});
                    external_points.push_back(
                        y_rel + route->terminal_delta);
                    external_commitments.push_back({trace.commitments.at(
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
                external_commitments.push_back({trace.commitments.at(output_bias_view_label(
                                                    head_index, "terminal_relation_virtual")),
                                                FieldElement::zero()});
                external_points.push_back(relation.canonical_points.front());
                const auto public_m_id = output_bias_view_label(
                                             head_index, "m_can") +
                                         ":public_bind";
                const auto public_m_it = std::find_if(
                    subproof_it->evaluations.begin(),
                    subproof_it->evaluations.end(),
                    [&](const BindingEvaluation &evaluation)
                    { return evaluation.object_id == public_m_id; });
                if (public_m_it == subproof_it->evaluations.end() || public_m_it->values.size() != 1 || public_m_it->kind != PayloadValueKind::PublicValue)
                    throw std::runtime_error(
                        "missing same-point public bias multiplicity binding");
                external_commitments.push_back({trace.commitments.at(output_bias_view_label(
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
            if (u_it == subproof_it->evaluations.end() || u_it->values.size() != 1)
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
            external_commitments.push_back({trace.commitments.at(output_bias_view_label(
                                                head_index, "terminal_relation_virtual")),
                                            FieldElement::zero()});
            external_points.push_back(relation.canonical_points.at(0));

            const auto public_m_id =
                output_bias_view_label(head_index, "m_can") + ":public_bind";
            const auto public_m_it = std::find_if(
                subproof_it->evaluations.begin(), subproof_it->evaluations.end(),
                [&](const BindingEvaluation &evaluation)
                { return evaluation.object_id == public_m_id; });
            if (public_m_it == subproof_it->evaluations.end() || public_m_it->values.size() != 1 || public_m_it->kind != PayloadValueKind::PublicValue)
                throw std::runtime_error("missing public bias multiplicity binding");
            external_commitments.push_back({trace.commitments.at(output_bias_view_label(head_index, "m_can")),
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
            if (subproof_it == proof.pi_bind.subproofs.end() || subproof_it->relation_meta
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
                const auto u_id = chunk.shifted_accumulator_object + ":u_rel";
                const auto u_it = std::find_if(
                    subproof_it->evaluations.begin(),
                    subproof_it->evaluations.end(),
                    [&](const BindingEvaluation &evaluation)
                    { return evaluation.object_id == u_id; });
                if (u_it == subproof_it->evaluations.end() || u_it->values.size() != 1 || u_it->kind != PayloadValueKind::MaskedPrivateValue)
                    throw std::runtime_error(
                        "missing output coefficient chunk shift value");
                external_commitments.push_back({trace.commitments.at(
                                                    chunk.shifted_accumulator_object),
                                                u_it->values.front()});
                external_points.push_back(y_rel);
                external_commitments.push_back({trace.commitments.at(chunk.accumulator_object),
                                                u_it->values.front()});
                external_points.push_back(
                    y_rel + chunk.terminal_point - gamma);
            }
            external_commitments.push_back({trace.commitments.at(output_bias_basis_label(
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
        external_commitments.push_back({trace.commitments.at("P_Y"),
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
        for (auto &subproof : proof.pi_bind.subproofs)
        {
            std::vector<MetadataReference> normalized;
            normalized.reserve(subproof.opening_references.size());
            for (auto reference : subproof.opening_references)
            {
                for (const auto &group : opening_registry.groups())
                {
                    if (reference.object_id == "W_ext:" + group.point.to_string())
                    {
                        reference.object_id = group.group_id;
                        break;
                    }
                }
                if (std::none_of(
                        normalized.begin(), normalized.end(),
                        [&](const MetadataReference &existing)
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
        const auto &point_groups = opening_registry.groups();
        std::vector<PointGroupOpeningResult> opening_results(
            point_groups.size());
        std::mutex file_backed_opening_cache_mutex;
        std::unordered_map<
            std::string,
            std::weak_ptr<const algebra::Polynomial>>
            file_backed_opening_cache;

        // CPU opening for a very large file-backed evaluation domain has a much
        // larger transient host-memory footprint than the corresponding CUDA MSM
        // route. Keep ordinary/PPI-sized file-backed point groups parallel, but
        // prevent two 2^25-or-larger CPU point groups from constructing their
        // folded values, quotient buffers and MSM working sets simultaneously.
        std::mutex large_cpu_file_backed_opening_mutex;
        constexpr std::size_t kLargeCpuFileBackedOpeningValueCount =
            1ULL << 25;

        file_backed_opening_cache.reserve(
            trace.spilled_polynomials.size());
        const crypto::OpeningPolynomialResolver
            file_backed_opening_resolver =
                [&](const Commitment &commitment)
            -> std::shared_ptr<const algebra::Polynomial>
        {
            if (commitment.opening_polynomial != nullptr)
            {
                return commitment.opening_polynomial;
            }
            const auto spilled_it =
                trace.spilled_polynomials.find(
                    commitment.name);
            if (spilled_it == trace.spilled_polynomials.end())
            {
                throw std::runtime_error(
                    "file-backed opening material is missing: " + commitment.name);
            }
            // A standard domain has two independently folded point
            // groups.  Let concurrent groups share the same decoded
            // source, but do not keep every large source alive for
            // the rest of the proof.  build_streamed_folded_opening_plan
            // consumes one source at a time, so a weak cache releases
            // it as soon as the active folds advance.  A strong cache
            // grows with every spilled edge-domain polynomial and can
            // exhaust memory before external openings complete.
            std::lock_guard<std::mutex> lock(
                file_backed_opening_cache_mutex);
            if (const auto cached =
                    file_backed_opening_cache.find(
                        commitment.name);
                cached != file_backed_opening_cache.end())
            {
                if (auto material = cached->second.lock();
                    material != nullptr)
                {
                    return material;
                }
            }
            auto material = load_spilled_opening_polynomial(
                context,
                spilled_it->second,
                commitment.name);
            file_backed_opening_cache.insert_or_assign(
                commitment.name,
                std::weak_ptr<const algebra::Polynomial>(material));
            return material;
        };

        const crypto::OpeningSemanticSourceResolver
            file_backed_opening_semantic_source_resolver =
                [&](const Commitment &commitment)
            -> std::optional<
                crypto::OpeningSemanticSourceInfo>
        {
            if (commitment.opening_polynomial != nullptr)
            {
                return std::nullopt;
            }

            const auto spilled_it =
                trace.spilled_polynomials.find(
                    commitment.name);

            if (spilled_it == trace.spilled_polynomials.end())
            {
                return std::nullopt;
            }

            const auto &spilled =
                spilled_it->second;

            if (spilled.semantic_source_id == 0)
            {
                return std::nullopt;
            }

            const auto domain =
                context_domain_by_name(
                    context,
                    spilled.domain_name);

            if (domain == nullptr)
            {
                throw std::runtime_error(
                    "file-backed opening semantic source has unknown domain: " + commitment.name);
            }

            return crypto::OpeningSemanticSourceInfo{
                .key =
                    spilled.domain_name + ":" + std::to_string(spilled.size) + ":" + std::to_string(spilled.semantic_source_id),
                .domain = domain,
                .value_count = spilled.size,

                .zero_domain_blinder =
                    spilled.zero_domain_blinder,

                .file_path =
                    spilled.path,

                .file_backed =
                    true,
            };
        };

        const auto estimate_point_group_work =
            [&](std::size_t index)
        {
            std::size_t total = 0;

            for (const auto &claim :
                 point_groups[index].claims)
            {
                if (claim.commitment
                        .opening_polynomial !=
                    nullptr)
                {
                    total +=
                        claim.commitment
                            .opening_polynomial
                            ->size();
                }
                else if (
                    const auto spilled_it =
                        trace.spilled_polynomials
                            .find(
                                claim.commitment.name);
                    spilled_it !=
                    trace.spilled_polynomials
                        .end())
                {
                    total +=
                        spilled_it->second.size;
                }
            }

            return total;
        };

        const auto
            shared_file_backed_semantic_source_count =
                [&](std::size_t first_index,
                    std::size_t second_index)
        {
            // Keep small/medium resident opening paths unchanged.
            // The paired source scan is reserved for genuinely large
            // streamed sources where memory traffic dominates.
            constexpr std::size_t
                kMinimumSharedStreamSourceValues =
                    1ULL << 22;

            std::unordered_map<
                std::string,
                std::string>
                first_sources;

            for (const auto &claim :
                 point_groups[first_index].claims)
            {
                if (claim.commitment
                        .opening_polynomial !=
                    nullptr)
                {
                    continue;
                }

                const auto spilled_it =
                    trace.spilled_polynomials.find(
                        claim.commitment.name);

                if (spilled_it ==
                    trace.spilled_polynomials.end())
                {
                    continue;
                }

                const auto &spilled =
                    spilled_it->second;

                if (spilled.semantic_source_id == 0 ||
                    spilled.size <
                        kMinimumSharedStreamSourceValues)
                {
                    continue;
                }

                const auto key =
                    spilled.domain_name + ":" +
                    std::to_string(spilled.size) + ":" +
                    std::to_string(
                        spilled.semantic_source_id);

                first_sources.try_emplace(
                    key,
                    claim.commitment.name);
            }

            std::size_t shared = 0;

            std::unordered_map<
                std::string,
                bool>
                counted;

            for (const auto &claim :
                 point_groups[second_index].claims)
            {
                if (claim.commitment
                        .opening_polynomial !=
                    nullptr)
                {
                    continue;
                }

                const auto spilled_it =
                    trace.spilled_polynomials.find(
                        claim.commitment.name);

                if (spilled_it ==
                    trace.spilled_polynomials.end())
                {
                    continue;
                }

                const auto &spilled =
                    spilled_it->second;

                if (spilled.semantic_source_id == 0 ||
                    spilled.size <
                        kMinimumSharedStreamSourceValues)
                {
                    continue;
                }

                const auto key =
                    spilled.domain_name + ":" +
                    std::to_string(spilled.size) + ":" +
                    std::to_string(
                        spilled.semantic_source_id);

                const auto first =
                    first_sources.find(key);

                if (first ==
                        first_sources.end() ||
                    first->second !=
                        claim.commitment.name)
                {
                    continue;
                }

                if (counted.emplace(
                               key,
                               true)
                        .second)
                {
                    ++shared;
                }
            }

            return shared;
        };

        const auto open_group_indices = [&](const std::vector<std::size_t> &indices)
        {
            if (indices.empty())
                return 0.0;
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
                    else if (const auto spilled_it =
                                 trace.spilled_polynomials.find(
                                     claim.commitment.name);
                             spilled_it != trace.spilled_polynomials.end())
                    {
                        total += spilled_it->second.size;
                    }
                }
                return total;
            };
            const auto largest_file_backed_value_count =
                [&](std::size_t index)
            {
                std::size_t largest = 0;

                for (const auto &claim :
                     point_groups[index].claims)
                {
                    if (claim.commitment.opening_polynomial != nullptr)
                    {
                        continue;
                    }

                    if (const auto spilled_it =
                            trace.spilled_polynomials.find(
                                claim.commitment.name);
                        spilled_it != trace.spilled_polynomials.end())
                    {
                        largest = std::max(
                            largest,
                            spilled_it->second.size);
                    }
                }

                return largest;
            };
            std::stable_sort(
                work_order.begin(), work_order.end(),
                [&](std::size_t lhs, std::size_t rhs)
                {
                    return estimated_work(lhs) > estimated_work(rhs);
                });
            const auto hardware_workers =
                std::max<unsigned>(
                    1,
                    std::thread::hardware_concurrency());

            const bool external_phase =
                point_groups[work_order.front()]
                    .owner_block == "W_ext";

            const bool has_file_backed_openings =
                std::any_of(
                    work_order.begin(),
                    work_order.end(),
                    [&](std::size_t index)
                    {
                        return std::any_of(
                            point_groups[index].claims.begin(),
                            point_groups[index].claims.end(),
                            [&](const auto &claim)
                            {
                                return claim.commitment
                                               .opening_polynomial ==
                                           nullptr &&
                                       trace.spilled_polynomials
                                           .contains(
                                               claim.commitment.name);
                            });
                    });

            // A 2^25-or-larger CPU file-backed group already needs
            // several full-domain vectors while folding and constructing
            // its witness. Do not overlap another point group with that
            // working set. Smaller/PPI-sized file-backed phases retain
            // the existing two-group pipeline.
            const bool
                has_large_cpu_file_backed_opening =
                    !util::route2_options()
                         .cuda_field_hotspots &&
                    std::any_of(
                        work_order.begin(),
                        work_order.end(),
                        [&](std::size_t index)
                        {
                            return largest_file_backed_value_count(
                                       index) >=
                                   kLargeCpuFileBackedOpeningValueCount;
                        });

            const auto outer_worker_limit =
                external_phase
                    ? std::min<std::size_t>(
                          20,
                          hardware_workers)
                    : std::min<std::size_t>(
                          8,
                          std::max<unsigned>(
                              1,
                              hardware_workers / 4));

            const auto worker_count =
                std::min<std::size_t>(
                    work_order.size(),
                    has_large_cpu_file_backed_opening
                        ? std::size_t{1}
                    : has_file_backed_openings
                        ? std::min<std::size_t>(
                              2,
                              outer_worker_limit)
                        : outer_worker_limit);

            const auto base_inner_worker_count =
                worker_count > 1
                    ? std::max<std::size_t>(
                          1,
                          hardware_workers /
                              worker_count)
                    : std::size_t{0};

            if (has_large_cpu_file_backed_opening)
            {
                progress_note(
                    "prove:opening:"
                    "phase_policy="
                    "large_cpu_file_backed_outer_serial:"
                    "groups=" +
                    std::to_string(
                        work_order.size()));
            }
            std::atomic<std::size_t> next{0};
            crypto::g1_parallel_msm_batch_workers(
                worker_count,
                [&](std::size_t)
                {
                    while (true)
                    {
                        const auto position = next.fetch_add(1);
                        if (position >= work_order.size())
                            break;
                        const auto group_index = work_order[position];
                        const auto &group = point_groups[group_index];
                        std::vector<Commitment> commitments;
                        std::vector<FieldElement> values;
                        commitments.reserve(group.claims.size());
                        values.reserve(group.claims.size());
                        for (const auto &claim : group.claims)
                        {
                            commitments.push_back(claim.commitment);
                            values.push_back(claim.value);
                        }
                        const auto largest_file_backed_count =
                            largest_file_backed_value_count(
                                group_index);

                        const bool
                            serialize_large_cpu_file_backed_opening =
                                !util::route2_options()
                                     .cuda_field_hotspots &&
                                largest_file_backed_count >=
                                    kLargeCpuFileBackedOpeningValueCount;

                        // Preserve bounded inner parallelism even when the
                        // phase itself is outer-serial. Otherwise worker_count
                        // == 1 would drop execution_options entirely and the
                        // largest opening could fall back to an unconstrained
                        // inner worker policy.
                        const auto inner_worker_count =
                            serialize_large_cpu_file_backed_opening
                                ? std::min<std::size_t>(
                                      16,
                                      hardware_workers)
                            : worker_count > 1 &&
                                    !external_phase &&
                                    estimated_work(
                                        group_index) >=
                                        (1ULL << 18)
                                ? std::min<std::size_t>(
                                      16,
                                      hardware_workers)
                                : base_inner_worker_count;

                        const crypto::
                            PointGroupOpeningExecutionOptions
                                execution_options{
                                    .max_field_worker_count =
                                        inner_worker_count,
                                    .msm = {
                                        .max_worker_count =
                                            inner_worker_count,
                                    },
                                };
                        std::unique_lock<std::mutex>
                            large_cpu_file_backed_opening_guard;

                        if (serialize_large_cpu_file_backed_opening)
                        {
                            large_cpu_file_backed_opening_guard =
                                std::unique_lock<std::mutex>(
                                    large_cpu_file_backed_opening_mutex);

                            progress_note(
                                "prove:opening:"
                                "large_cpu_file_backed_serial:" +
                                group.owner_block + ":values=" + std::to_string(largest_file_backed_count));
                        }

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
                                inner_worker_count != 0
                                    ? &execution_options
                                    : nullptr,
                                trace.spilled_polynomials.empty()
                                    ? nullptr
                                    : &file_backed_opening_resolver,
                                trace.spilled_polynomials.empty()
                                    ? nullptr
                                    : &file_backed_opening_semantic_source_resolver);
                        opening_results[group_index].profile.point_parallel_batches = 1;
                        opening_results[group_index].profile.point_parallel_workers =
                            serialize_large_cpu_file_backed_opening
                                ? 1
                                : worker_count;
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
                 : domain_group_indices)
                .push_back(i);
        }
        struct SharedStreamPairSelection
        {
            std::array<std::size_t, 2>
                groups{0, 0};

            std::size_t shared_source_count = 0;

            std::size_t estimated_work = 0;
        };

        const auto select_shared_stream_pairs =
            [&](const std::vector<std::size_t> &candidates,
                bool require_same_owner)
        {
            std::vector<
                SharedStreamPairSelection>
                selections;

            if (!util::route2_options()
                     .cuda_field_hotspots ||
                candidates.size() < 2)
            {
                return selections;
            }

            std::unordered_set<std::size_t>
                consumed;

            while (true)
            {
                bool found = false;

                SharedStreamPairSelection
                    best;

                for (std::size_t lhs_position = 0;
                     lhs_position <
                     candidates.size();
                     ++lhs_position)
                {
                    const auto lhs =
                        candidates[lhs_position];

                    if (consumed.contains(lhs))
                    {
                        continue;
                    }

                    for (std::size_t rhs_position =
                             lhs_position + 1;
                         rhs_position <
                         candidates.size();
                         ++rhs_position)
                    {
                        const auto rhs =
                            candidates[rhs_position];

                        if (consumed.contains(rhs))
                        {
                            continue;
                        }

                        if (require_same_owner &&
                            point_groups[lhs]
                                    .owner_block !=
                                point_groups[rhs]
                                    .owner_block)
                        {
                            continue;
                        }

                        const auto shared =
                            shared_file_backed_semantic_source_count(
                                lhs,
                                rhs);

                        if (shared == 0)
                        {
                            continue;
                        }

                        const auto work =
                            estimate_point_group_work(
                                lhs) +
                            estimate_point_group_work(
                                rhs);

                        if (!found ||
                            shared >
                                best.shared_source_count ||
                            (shared ==
                                 best.shared_source_count &&
                             work >
                                 best.estimated_work))
                        {
                            best.groups = {
                                lhs,
                                rhs,
                            };

                            best.shared_source_count =
                                shared;

                            best.estimated_work =
                                work;

                            found = true;
                        }
                    }
                }

                if (!found)
                {
                    break;
                }

                consumed.insert(
                    best.groups[0]);

                consumed.insert(
                    best.groups[1]);

                selections.push_back(
                    best);
            }

            return selections;
        };

        const auto
            shared_stream_domain_pairs =
                select_shared_stream_pairs(
                    domain_group_indices,
                    true);

        const auto
            shared_stream_external_pairs =
                select_shared_stream_pairs(
                    external_group_indices,
                    false);
        const auto external_claim_assembly_ms =
            elapsed_ms(
                stage_start,
                Clock::now());

        const auto domain_witness_start =
            Clock::now();

        auto open_shared_stream_pair =
            [&](const SharedStreamPairSelection
                    &selection)
        {
            const auto pair =
                selection.groups;

            std::array<
                std::vector<Commitment>,
                2>
                pair_commitments;

            std::array<
                std::vector<FieldElement>,
                2>
                pair_values;

            std::array<FieldElement, 2>
                pair_points{
                    point_groups[pair[0]].point,
                    point_groups[pair[1]].point,
                };

            std::array<FieldElement, 2>
                pair_challenges{
                    derive_point_group_challenge(
                        proof.public_metadata,
                        point_groups[pair[0]]),
                    derive_point_group_challenge(
                        proof.public_metadata,
                        point_groups[pair[1]]),
                };

            for (std::size_t slot = 0;
                 slot < 2;
                 ++slot)
            {
                const auto &group =
                    point_groups[pair[slot]];

                pair_commitments[slot]
                    .reserve(
                        group.claims.size());

                pair_values[slot]
                    .reserve(
                        group.claims.size());

                for (const auto &claim :
                     group.claims)
                {
                    pair_commitments[slot]
                        .push_back(
                            claim.commitment);

                    pair_values[slot]
                        .push_back(
                            claim.value);
                }
            }

            const auto hardware_workers =
                std::max<unsigned>(
                    1,
                    std::thread::
                        hardware_concurrency());

            const auto inner_worker_count =
                std::min<std::size_t>(
                    16,
                    hardware_workers);

            const std::array<
                crypto::
                    PointGroupOpeningExecutionOptions,
                2>
                pair_execution_options{
                    crypto::
                        PointGroupOpeningExecutionOptions{
                            .max_field_worker_count =
                                inner_worker_count,
                            .msm = {
                                .max_worker_count =
                                    inner_worker_count,
                            },
                        },
                    crypto::PointGroupOpeningExecutionOptions{
                        .max_field_worker_count = inner_worker_count,
                        .msm = {
                            .max_worker_count = inner_worker_count,
                        },
                    },
                };

            std::array<
                crypto::BatchOpeningProfile,
                2>
                pair_profiles;

            const auto pair_start =
                Clock::now();

            const auto witnesses =
                crypto::KZG::
                    open_point_group_pair_shared_stream(
                        pair_commitments,
                        pair_points,
                        pair_values,
                        pair_challenges,
                        context.kzg,
                        &pair_profiles,
                        &pair_execution_options,
                        trace.spilled_polynomials
                                .empty()
                            ? nullptr
                            : &file_backed_opening_resolver,
                        trace.spilled_polynomials
                                .empty()
                            ? nullptr
                            : &file_backed_opening_semantic_source_resolver);

            const auto pair_wall_ms =
                elapsed_ms(
                    pair_start,
                    Clock::now());

            for (std::size_t slot = 0;
                 slot < 2;
                 ++slot)
            {
                const auto index =
                    pair[slot];

                pair_profiles[slot]
                    .point_parallel_batches = 1;

                pair_profiles[slot]
                    .point_parallel_workers = 2;

                opening_results[index]
                    .witness =
                    witnesses[slot];

                opening_results[index]
                    .profile =
                    pair_profiles[slot];

                opening_results[index]
                    .elapsed_ms =
                    pair_profiles[slot]
                        .precompute_ms +
                    pair_profiles[slot]
                        .finalize_ms;
            }

            progress_note(
                "prove:opening:"
                "shared_stream_pair:"
                "first_owner=" +
                point_groups[pair[0]]
                    .owner_block +
                ":second_owner=" +
                point_groups[pair[1]]
                    .owner_block +
                ":shared_sources=" +
                std::to_string(
                    selection
                        .shared_source_count) +
                ":ms=" +
                std::to_string(
                    pair_wall_ms));
        };

        const auto erase_opened_pair =
            [](std::vector<std::size_t> *indices,
               const std::array<std::size_t, 2>
                   &pair)
        {
            indices->erase(
                std::remove_if(
                    indices->begin(),
                    indices->end(),
                    [&](std::size_t index)
                    {
                        return index ==
                                   pair[0] ||
                               index ==
                                   pair[1];
                    }),
                indices->end());
        };

        auto remaining_domain_group_indices =
            domain_group_indices;

        for (const auto &selection :
             shared_stream_domain_pairs)
        {
            open_shared_stream_pair(
                selection);

            erase_opened_pair(
                &remaining_domain_group_indices,
                selection.groups);
        }

        (void)open_group_indices(
            remaining_domain_group_indices);

        const auto domain_witness_wall_ms =
            elapsed_ms(
                domain_witness_start,
                Clock::now());

        for (const auto group_index :
             domain_group_indices)
        {
            const auto &group =
                point_groups[group_index];

            const auto &profile =
                opening_results[group_index]
                    .profile;

            progress_note(
                "prove:domain_witness:group:" +
                group.owner_block +
                ":ms=" +
                std::to_string(
                    opening_results[group_index]
                        .elapsed_ms) +
                ":precompute_ms=" +
                std::to_string(
                    profile.precompute_ms) +
                ":inverse_ms=" +
                std::to_string(
                    profile
                        .denominator_inverse_ms) +
                ":assembly_ms=" +
                std::to_string(
                    profile
                        .quotient_assembly_ms) +
                ":witness_msm_ms=" +
                std::to_string(
                    profile
                        .witness_msm_ms) +
                ":blinder_ms=" +
                std::to_string(
                    profile
                        .blinder_witness_ms) +
                ":workers=" +
                std::to_string(
                    profile
                        .point_msm_worker_cap) +
                ":outer_workers=" +
                std::to_string(
                    profile
                        .point_parallel_workers));
        }

        progress_note(
            "prove:domain_witness:done:ms=" +
            std::to_string(
                domain_witness_wall_ms));

        auto remaining_external_group_indices =
            external_group_indices;

        const auto external_witness_start =
            Clock::now();

        for (const auto &selection :
             shared_stream_external_pairs)
        {
            open_shared_stream_pair(
                selection);

            erase_opened_pair(
                &remaining_external_group_indices,
                selection.groups);
        }

        (void)open_group_indices(
            remaining_external_group_indices);

        const auto external_witness_wall_ms =
            elapsed_ms(
                external_witness_start,
                Clock::now());

        proof.opening_groups.reserve(point_groups.size());
        for (std::size_t i = 0; i < point_groups.size(); ++i)
        {
            proof.opening_groups.push_back(PointGroupOpening{
                .group_id = point_groups[i].group_id,
                .owner_block = point_groups[i].owner_block,
                .point = point_groups[i].point,
                .witness = opening_results[i].witness,
            });
            if (metrics == nullptr)
                continue;
            const auto &profile = opening_results[i].profile;
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
                if (auto *metric = domain_open_metric(metrics, domain);
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
                        opening_results[i].elapsed_ms - profile.precompute_ms);
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
        progress_note(
            "prove:cuda_cache_release:start");

        crypto::
            release_cuda_g1_msm_resident_cache();

        algebra::
            release_cuda_ntt_workspace();

        algebra::
            release_cuda_weighted_dot_cache();

        release_cuda_t_edge_executor_cache();

        progress_note(
            "prove:cuda_cache_release:done");

        cuda_cache_release_guard.dismiss();

        return proof;
    }

} // namespace gatzk::protocol
