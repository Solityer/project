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

#include "gatzk/algebra/domain_evaluation.hpp"
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

        std::uint64_t fnv1a64(std::string_view text)
        {
            std::uint64_t hash = 1469598103934665603ULL;
            for (const auto ch : text)
            {
                hash ^= static_cast<unsigned char>(ch);
                hash *= 1099511628211ULL;
            }
            return hash;
        }

        std::string stable_hash(std::string_view text)
        {
            return std::to_string(fnv1a64(text));
        }

        std::string hash_key_values(const std::map<std::string, std::string> &entries)
        {
            std::ostringstream stream;
            for (const auto &[key, value] : entries)
            {
                stream << key << '=' << value << '\n';
            }
            return stable_hash(stream.str());
        }

        std::string payload_kind_name(PayloadValueKind kind)
        {
            return kind == PayloadValueKind::PublicValue ? "PublicValue" : "MaskedPrivateValue";
        }

        PayloadValueKind payload_kind_for_label(const ProtocolContext &context, const std::string &label)
        {
            if (context.public_commitments.contains(label))
            {
                return PayloadValueKind::PublicValue;
            }
            if (label == "P_H" || label == "P_Y" || label.rfind("P_Y_o", 0) == 0)
            {
                return PayloadValueKind::PublicValue;
            }
            return PayloadValueKind::MaskedPrivateValue;
        }

        std::string hash_proof_transport(const Proof &proof)
        {
            std::ostringstream stream;
            auto append_commitments = [&](std::string_view label, const std::vector<std::pair<std::string, crypto::Commitment>> &commitments)
            {
                stream << '[' << label << "]\n";
                for (const auto &[name, commitment] : commitments)
                {
                    stream << name << '=' << commitment.point.to_string() << '\n';
                }
            };
            auto append_field_pairs = [&](std::string_view label, const std::vector<std::pair<std::string, FieldElement>> &values)
            {
                stream << '[' << label << "]\n";
                for (const auto &[name, value] : values)
                {
                    stream << name << '=' << value.to_string() << '\n';
                }
            };

            stream << "[metadata]\n"
                   << proof.public_metadata.protocol_name << '\n'
                   << proof.public_metadata.proof_task << '\n'
                   << proof.public_metadata.srs_id << '\n'
                   << proof.public_metadata.srs_type << '\n'
                   << proof.public_metadata.srs_degree_bound << '\n'
                   << proof.public_metadata.privacy_policy << '\n'
                   << proof.public_metadata.input_privacy << '\n'
                   << proof.public_metadata.output_mode << '\n'
                   << proof.public_metadata.opening_mode << '\n'
                   << proof.public_metadata.binding_mode << '\n'
                   << proof.public_metadata.protocol_id << '\n'
                   << proof.public_metadata.dataset_name << '\n'
                   << proof.public_metadata.task_type << '\n'
                   << proof.public_metadata.report_unit << '\n'
                   << proof.public_metadata.graph_count << '\n'
                   << proof.public_metadata.L << '\n'
                   << proof.public_metadata.hidden_profile << '\n'
                   << proof.public_metadata.d_in_profile << '\n'
                   << proof.public_metadata.K_out << '\n'
                   << proof.public_metadata.C << '\n'
                   << proof.public_metadata.batching_rule << '\n'
                   << proof.public_metadata.subgraph_rule << '\n'
                   << proof.public_metadata.self_loop_rule << '\n'
                   << proof.public_metadata.edge_sort_rule << '\n'
                   << proof.public_metadata.chunking_rule << '\n'
                   << proof.public_metadata.model_arch_id << '\n'
                   << proof.public_metadata.model_param_id << '\n'
                   << proof.public_metadata.static_table_id << '\n'
                   << proof.public_metadata.quant_cfg_id << '\n'
                   << proof.public_metadata.domain_cfg << '\n'
                   << proof.public_metadata.dim_cfg << '\n'
                   << proof.public_metadata.encoding_id << '\n'
                   << proof.public_metadata.padding_rule_id << '\n'
                   << proof.public_metadata.degree_bound_id << '\n';
            stream << "[block_order]\n";
            for (const auto &block : proof.block_order)
            {
                stream << block << '\n';
            }
            append_commitments("dynamic_commitments", proof.dynamic_commitments);
            append_commitments("quotient_commitments", proof.quotient_commitments);
            stream << "[domain_openings]\n";
            for (const auto &[domain, bundle] : proof.domain_openings)
            {
                stream << domain << "|proof_witnesses=" << bundle.proof.witnesses.size() << '\n';
                for (const auto &witness : bundle.proof.witnesses)
                {
                    stream << "witness=" << witness.to_string() << '\n';
                }
                for (const auto &point : bundle.points)
                {
                    stream << "point=" << point.to_string() << '\n';
                }
                for (const auto &[name, values] : bundle.values)
                {
                    stream << "values=" << name;
                    for (const auto &value : values)
                    {
                        stream << ':' << value.to_string();
                    }
                    stream << '\n';
                }
                for (const auto kind : bundle.value_kinds)
                {
                    stream << "value_kind=" << payload_kind_name(kind) << '\n';
                }
            }
            append_field_pairs("external_evaluations", proof.external_evaluations);
            stream << "[external_evaluation_kinds]\n";
            for (const auto kind : proof.external_evaluation_kinds)
            {
                stream << payload_kind_name(kind) << '\n';
            }
            append_field_pairs("witness_scalars", proof.witness_scalars);
            stream << "[witness_scalar_kinds]\n";
            for (const auto kind : proof.witness_scalar_kinds)
            {
                stream << payload_kind_name(kind) << '\n';
            }
            stream << "[external_proof]\n";
            for (const auto &witness : proof.external_proof.witnesses)
            {
                stream << "witness=" << witness.to_string() << '\n';
            }
            stream << "[challenges]\n";
            for (const auto &[name, challenge] : proof.challenges)
            {
                stream << name << '=' << challenge.to_string() << '\n';
            }
            return stable_hash(stream.str());
        }

        Polynomial make_eval_poly(
            const std::string &name,
            const std::vector<FieldElement> &values,
            const std::shared_ptr<algebra::RootOfUnityDomain> &domain)
        {
            return Polynomial::from_evaluations(name, values, domain);
        }

        Polynomial make_coeff_poly(const std::string &name, const std::vector<FieldElement> &values)
        {
            return Polynomial::from_coefficients(name, values);
        }

        void add_public_poly(
            ProtocolContext &context,
            const std::string &name,
            const Polynomial &polynomial)
        {
            progress_note("context:public:" + name + ":start");
            context.public_polynomials[name] = polynomial;
            context.public_commitments[name] = crypto::KZG::commit(name, polynomial, context.kzg);
            progress_note("context:public:" + name + ":done");
        }

        void add_static_commitment(
            ProtocolContext &context,
            const std::string &name,
            const Polynomial &polynomial)
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
            std::vector<std::pair<FieldElement, FieldElement>> out;
            out.reserve(signed_table_size(min_raw, max_raw, "lrelu"));
            for (std::int64_t raw = min_raw; raw <= max_raw; ++raw)
            {
                const auto input = dequantize_trace_scalar(raw);
                const auto output = input >= 0.0 ? input : 0.2 * input;
                out.push_back({FieldElement::from_signed(raw), quantize_float_model_value(output)});
                if (raw == max_raw)
                {
                    break;
                }
            }
            return out;
        }

        std::size_t elu_table_band()
        {
            return 1;
        }

        std::vector<std::pair<FieldElement, FieldElement>> make_elu_table(std::size_t bound)
        {
            std::vector<std::pair<FieldElement, FieldElement>> out;
            out.reserve((bound * 2 + 1) * (2 * elu_table_band() + 1));
            for (std::int64_t raw = -static_cast<std::int64_t>(bound); raw <= static_cast<std::int64_t>(bound); ++raw)
            {
                const auto input = dequantize_trace_scalar(raw);
                const auto central_signed = quantize_float_model_value_signed(input >= 0.0 ? input : std::exp(input) - 1.0);
                for (std::int64_t offset = -static_cast<std::int64_t>(elu_table_band());
                     offset <= static_cast<std::int64_t>(elu_table_band());
                     ++offset)
                {
                    out.push_back({
                        FieldElement::from_signed(raw),
                        FieldElement::from_signed(central_signed + offset),
                    });
                }
            }
            return out;
        }

        std::size_t exp_table_band()
        {
            return 2;
        }

        std::vector<std::pair<FieldElement, FieldElement>> make_exp_table(std::size_t size)
        {
            std::vector<std::pair<FieldElement, FieldElement>> out;
            out.reserve(size * (2 * exp_table_band() + 1));
            for (std::size_t i = 0; i < size; ++i)
            {
                const auto input = static_cast<std::int64_t>(i);
                const auto central_signed = quantize_float_model_value_signed(std::exp(-dequantize_trace_scalar(input)));
                for (std::int64_t offset = -static_cast<std::int64_t>(exp_table_band());
                     offset <= static_cast<std::int64_t>(exp_table_band());
                     ++offset)
                {
                    const auto candidate = std::max<std::int64_t>(0, central_signed + offset);
                    out.push_back({
                        FieldElement(static_cast<std::uint64_t>(i)),
                        FieldElement::from_signed(candidate),
                    });
                }
            }
            return out;
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

        model::Matrix quantize_model_matrix(const model::FloatMatrix &matrix)
        {
            model::Matrix out(matrix.size());
            for (std::size_t row = 0; row < matrix.size(); ++row)
            {
                out[row] = quantize_model_vector(matrix[row]);
            }
            return out;
        }

        std::vector<std::string> vector_to_lines(const std::vector<FieldElement> &values)
        {
            std::vector<std::string> lines;
            lines.reserve(values.size());
            for (std::size_t i = 0; i < values.size(); ++i)
            {
                lines.push_back(std::to_string(i) + " " + values[i].to_string());
            }
            return lines;
        }

        std::vector<std::string> matrix_to_lines(const model::Matrix &matrix)
        {
            std::vector<std::string> lines;
            for (std::size_t i = 0; i < matrix.size(); ++i)
            {
                std::ostringstream row;
                row << i;
                for (const auto &value : matrix[i])
                {
                    row << ' ' << value.to_string();
                }
                lines.push_back(row.str());
            }
            return lines;
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

        PublicMetadata build_public_metadata(const ProtocolContext &context) { return canonical_public_metadata(context); }

        struct ExternalEvalSpec
        {
            std::string proof_name;
            std::string label;
            std::string challenge_name;
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
            const bool legacy_single_output = context.model.output_layer.heads.size() == 1;
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

        struct CachedQuotientArtifacts
        {
            std::unordered_map<std::string, Commitment> quotient_commitments;
        };

        class SharedQuotientArtifactsCache
        {
        public:
            bool lookup(const std::string &key, CachedQuotientArtifacts *out)
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

            void store(const std::string &key, CachedQuotientArtifacts value)
            {
                std::lock_guard<std::mutex> lock(mutex_);
                entries_.emplace(key, std::move(value));
            }

        private:
            std::mutex mutex_;
            std::unordered_map<std::string, CachedQuotientArtifacts> entries_;
        };

        SharedQuotientArtifactsCache &shared_quotient_artifacts_cache()
        {
            static SharedQuotientArtifactsCache cache;
            return cache;
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
            const FieldElement & /*point*/,
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
            return FieldElement::from_native(native_sum);
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
                        trace_.witness_scalars,
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
                        trace_.witness_scalars,
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
                        cache_group_values(group_labels, point, values);
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
                    return values.at(*weight_entry.direct_index);
                }
                // `parallel_fft` 只改变域权重的归约方式，opened value 本身必须保持不变。
                return algebra::dot_product_native_weights(values, weight_entry.native_weights);
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
            static std::mutex mutex;
            static std::unordered_map<std::size_t, std::shared_ptr<const mcl::Ntt<mcl::Fr>>> cache;
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (const auto it = cache.find(size); it != cache.end())
                {
                    return it->second;
                }
            }

            auto plan = std::make_shared<mcl::Ntt<mcl::Fr>>();
            const mcl::Fr generator = 5;
            const mcl::Vint root_exponent = (mcl::Fr::getOp().mp - 1) >> 32;
            mcl::Fr::pow(plan->root_, generator, root_exponent);
            if (!plan->init(size))
            {
                throw std::runtime_error("failed to initialize NTT plan for size " + std::to_string(size));
            }
            if (!(FieldElement::from_native(plan->getW()) == FieldElement::root_of_unity(size)))
            {
                throw std::runtime_error("NTT root does not match protocol domain root for size " + std::to_string(size));
            }
            std::lock_guard<std::mutex> lock(mutex);
            const auto [it, inserted] = cache.emplace(size, plan);
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
                // Current edge/FH relations multiply at most three degree-<n columns.
                // Rotations z -> omega*z keep the same degree; first/last/not_last
                // have degree n-1. Thus the numerator is <= 3n-3, and after
                // division by Z_H of degree n the quotient is <= 2n-3.
                return 2 * domain_size - 3;
            }
            return std::nullopt;
        }

        std::size_t quotient_coset_multiplier(const std::string &label)
        {
            if (label == "t_FH" || label == "t_edge")
            {
                return 2;
            }
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
            ntt_plan(work.size())->intt(work.data());
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
                ntt_plan(half_size)->ntt(even_work.data());
                ntt_plan(half_size)->ntt(odd_work.data());
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
                ntt_plan(coefficient_work.size())->intt(coefficient_work.data());

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

        std::vector<FieldElement> coefficients_from_coset_evaluations(
            const std::vector<FieldElement> &values,
            const FieldElement &shift)
        {
            std::vector<mcl::Fr> work(values.size());
            copy_field_values_to_native(values, &work);
            ntt_plan(work.size())->intt(work.data());

            const auto inv_shift = shift.inv();
            const auto inv_shift_native = inv_shift.native();
            mcl::Fr inv_shift_power;
            inv_shift_power = 1;
            std::vector<FieldElement> coefficients(work.size(), FieldElement::zero());
            for (std::size_t i = 0; i < work.size(); ++i)
            {
                mcl::Fr coeff;
                mcl::Fr::mul(coeff, work[i], inv_shift_power);
                coefficients[i] = FieldElement::from_native(coeff);
                mcl::Fr::mul(inv_shift_power, inv_shift_power, inv_shift_native);
            }
            trim_field_coefficients(&coefficients);
            return coefficients;
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
            }

            algebra::Polynomial materialize(const crypto::KZGKeyPair &key)
            {
                if (target_size_ > key.pk.degree_bound + 1)
                {
                    throw std::runtime_error(
                        "SRS degree bound is too small for quotient polynomial " + label_ + ": need coefficients=" + std::to_string(target_size_) + " but degree_bound+1=" + std::to_string(key.pk.degree_bound + 1));
                }
                if (label_ == "t_FH" && full_graph_feature_identity_enabled(context_))
                {
                    return materialize_full_identity_fh();
                }

                const auto root = FieldElement::root_of_unity(target_size_);
                auto evaluate_at_index = [&](std::size_t i, const FieldElement &z)
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
                        return evaluate_t_edge(context_, challenges_, trace_.witness_scalars, eval, z);
                    }
                    else if (label_ == "t_N")
                    {
                        return evaluate_t_n(context_, challenges_, trace_.witness_scalars, eval, z);
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
                    progress_note("prove:quotient:t_edge:coset_prewarm:done");
                    if (context_.model.has_real_multihead)
                    {
                        return materialize_t_edge_fast();
                    }
                }
                if (label_ == "t_edge")
                {
                    progress_note("prove:quotient:t_edge:evaluate:start");
                }
                quotient_values[0] = evaluate_at_index(0, shift_);
                const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
                const bool run_parallel = cpu_count > 1 && target_size_ >= 4096;
                if (run_parallel)
                {
                    const auto task_count = std::min<std::size_t>(
                        std::min<std::size_t>(cpu_count, 32),
                        std::max<std::size_t>(1, target_size_ / 32768));
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
                    progress_note("prove:quotient:t_edge:evaluate:done");
                    progress_note("prove:quotient:t_edge:coefficients:start");
                }

                auto coefficients = coefficients_from_coset_evaluations(quotient_values, shift_);
                if (label_ == "t_edge")
                {
                    progress_note("prove:quotient:t_edge:coefficients:done");
                }
                if (coefficients.size() > key.pk.degree_bound + 1)
                {
                    throw std::runtime_error("materialized quotient exceeds SRS degree bound: " + label_);
                }
                return algebra::Polynomial::from_coefficients(label_, std::move(coefficients));
            }

        private:
            class NativeQuotientAccumulator
            {
            public:
                explicit NativeQuotientAccumulator(const std::vector<mcl::Fr> &alpha_powers)
                    : alpha_powers_(alpha_powers)
                {
                    sum_.clear();
                }

                void add(const FieldElement &term)
                {
                    if (power_index_ >= alpha_powers_.size())
                    {
                        throw std::runtime_error("t_edge quotient accumulator exceeded alpha power plan");
                    }
                    mcl::Fr product;
                    mcl::Fr::mul(product, alpha_powers_[power_index_], term.native());
                    mcl::Fr::add(sum_, sum_, product);
                    ++power_index_;
                }

                void skip_zero_terms(std::size_t count = 1)
                {
                    if (power_index_ + count > alpha_powers_.size())
                    {
                        throw std::runtime_error("t_edge quotient accumulator skipped beyond alpha power plan");
                    }
                    power_index_ += count;
                }

                FieldElement scaled_value(const FieldElement &scale) const
                {
                    if (power_index_ != alpha_powers_.size())
                    {
                        throw std::runtime_error("t_edge quotient accumulator term count mismatch");
                    }
                    mcl::Fr out;
                    mcl::Fr::mul(out, sum_, scale.native());
                    return FieldElement::from_native(out);
                }

            private:
                const std::vector<mcl::Fr> &alpha_powers_;
                mcl::Fr sum_;
                std::size_t power_index_ = 0;
            };

            struct TEdgeCosetGeometry
            {
                std::vector<FieldElement> first;
                std::vector<FieldElement> last;
                std::vector<FieldElement> not_last;
                std::vector<FieldElement> inverse_zero;
            };

            struct TEdgeGlobalCosets
            {
                const std::vector<FieldElement> *q_edge = nullptr;
                const std::vector<FieldElement> *q_new = nullptr;
                const std::vector<FieldElement> *q_end = nullptr;
                const std::vector<FieldElement> *q_tbl_l = nullptr;
                const std::vector<FieldElement> *q_qry_l = nullptr;
                const std::vector<FieldElement> *q_tbl_r = nullptr;
                const std::vector<FieldElement> *q_qry_r = nullptr;
                const std::vector<FieldElement> *q_tbl_exp = nullptr;
                const std::vector<FieldElement> *q_qry_exp = nullptr;
                const std::vector<FieldElement> *q_tbl_elu = nullptr;
                const std::vector<FieldElement> *q_qry_elu = nullptr;
                const std::vector<FieldElement> *t_l_x = nullptr;
                const std::vector<FieldElement> *t_l_y = nullptr;
                const std::vector<FieldElement> *t_range = nullptr;
                const std::vector<FieldElement> *t_exp_x = nullptr;
                const std::vector<FieldElement> *t_exp_y = nullptr;
                const std::vector<FieldElement> *t_elu_x = nullptr;
                const std::vector<FieldElement> *t_elu_y = nullptr;
                const std::vector<FieldElement> *src = nullptr;
                const std::vector<FieldElement> *dst = nullptr;
            };

            struct TEdgeHiddenHeadPlan
            {
                const std::vector<FieldElement> *r_src = nullptr;
                const std::vector<FieldElement> *e_src_edge = nullptr;
                const std::vector<FieldElement> *e_dst_edge = nullptr;
                const std::vector<FieldElement> *h_src_star_edge = nullptr;
                const std::vector<FieldElement> *r_l = nullptr;
                const std::vector<FieldElement> *m_l = nullptr;
                const std::vector<FieldElement> *z_col = nullptr;
                const std::vector<FieldElement> *m_edge = nullptr;
                const std::vector<FieldElement> *s_max = nullptr;
                const std::vector<FieldElement> *c_max = nullptr;
                const std::vector<FieldElement> *r_r = nullptr;
                const std::vector<FieldElement> *m_r = nullptr;
                const std::vector<FieldElement> *alpha = nullptr;
                const std::vector<FieldElement> *u = nullptr;
                const std::vector<FieldElement> *inv_edge = nullptr;
                const std::vector<FieldElement> *r_exp = nullptr;
                const std::vector<FieldElement> *m_exp = nullptr;
                const std::vector<FieldElement> *psq = nullptr;
                const std::vector<FieldElement> *widehat_v_pre_star = nullptr;
                const std::vector<FieldElement> *h_agg_pre_star_edge = nullptr;
                const std::vector<FieldElement> *r_t = nullptr;
                const std::vector<FieldElement> *r_elu = nullptr;
                const std::vector<FieldElement> *m_elu = nullptr;
                const std::vector<FieldElement> *h_agg_pre_flat = nullptr;
                const std::vector<FieldElement> *h_agg_flat = nullptr;
                const std::vector<FieldElement> *r_dst = nullptr;
                const std::vector<FieldElement> *sum_edge = nullptr;
                const std::vector<FieldElement> *h_agg_star_edge = nullptr;
                FieldElement beta_src = FieldElement::zero();
                FieldElement beta_dst = FieldElement::zero();
                FieldElement eta_src = FieldElement::zero();
                FieldElement eta_src_sq = FieldElement::zero();
                FieldElement eta_dst = FieldElement::zero();
                FieldElement eta_dst_sq = FieldElement::zero();
                FieldElement eta_dst_cu = FieldElement::zero();
                FieldElement eta_dst_4 = FieldElement::zero();
                FieldElement eta_dst_5 = FieldElement::zero();
                FieldElement beta_l = FieldElement::zero();
                FieldElement eta_l = FieldElement::zero();
                FieldElement beta_r = FieldElement::zero();
                FieldElement beta_exp = FieldElement::zero();
                FieldElement eta_exp = FieldElement::zero();
                FieldElement beta_elu = FieldElement::zero();
                FieldElement eta_elu = FieldElement::zero();
                FieldElement beta_t = FieldElement::zero();
                FieldElement eta_t = FieldElement::zero();
                FieldElement lambda_psq = FieldElement::zero();
                FieldElement witness_src = FieldElement::zero();
                FieldElement witness_t = FieldElement::zero();
                FieldElement witness_dst = FieldElement::zero();
            };

            struct TEdgeOutputHeadPlan
            {
                const std::vector<FieldElement> *r_src = nullptr;
                const std::vector<FieldElement> *e_src_edge = nullptr;
                const std::vector<FieldElement> *e_dst_edge = nullptr;
                const std::vector<FieldElement> *r_l = nullptr;
                const std::vector<FieldElement> *m_l = nullptr;
                const std::vector<FieldElement> *z_col = nullptr;
                const std::vector<FieldElement> *m_edge = nullptr;
                const std::vector<FieldElement> *s_max = nullptr;
                const std::vector<FieldElement> *c_max = nullptr;
                const std::vector<FieldElement> *r_r = nullptr;
                const std::vector<FieldElement> *m_r = nullptr;
                const std::vector<FieldElement> *alpha = nullptr;
                const std::vector<FieldElement> *u = nullptr;
                const std::vector<FieldElement> *inv_edge = nullptr;
                const std::vector<FieldElement> *r_exp = nullptr;
                const std::vector<FieldElement> *m_exp = nullptr;
                const std::vector<FieldElement> *psq = nullptr;
                const std::vector<FieldElement> *widehat_y_star = nullptr;
                const std::vector<FieldElement> *r_t = nullptr;
                const std::vector<FieldElement> *r_dst = nullptr;
                const std::vector<FieldElement> *sum_edge = nullptr;
                const std::vector<FieldElement> *y_star_edge = nullptr;
                FieldElement beta_src = FieldElement::zero();
                FieldElement beta_dst = FieldElement::zero();
                FieldElement eta_src = FieldElement::zero();
                FieldElement eta_dst = FieldElement::zero();
                FieldElement eta_dst_sq = FieldElement::zero();
                FieldElement eta_dst_cu = FieldElement::zero();
                FieldElement eta_dst_4 = FieldElement::zero();
                FieldElement eta_dst_5 = FieldElement::zero();
                FieldElement beta_l = FieldElement::zero();
                FieldElement eta_l = FieldElement::zero();
                FieldElement beta_r = FieldElement::zero();
                FieldElement beta_exp = FieldElement::zero();
                FieldElement eta_exp = FieldElement::zero();
                FieldElement beta_t = FieldElement::zero();
                FieldElement eta_t = FieldElement::zero();
                FieldElement lambda_out = FieldElement::zero();
                FieldElement witness_src = FieldElement::zero();
                FieldElement witness_t = FieldElement::zero();
                FieldElement witness_dst = FieldElement::zero();
            };

            struct TEdgeEvaluationPlan
            {
                TEdgeGlobalCosets global;
                std::vector<TEdgeHiddenHeadPlan> hidden_heads;
                std::vector<TEdgeOutputHeadPlan> output_heads;
                FieldElement alpha_quot = FieldElement::zero();
                std::vector<mcl::Fr> alpha_powers;
                FieldElement one = FieldElement::one();
            };

            static const FieldElement &vec_at(const std::vector<FieldElement> *values, std::size_t index)
            {
                return (*values)[index];
            }

            static FieldElement t_edge_route_edge_step(
                const FieldElement &next_value,
                const FieldElement &current_value,
                const FieldElement &query_value,
                const FieldElement &beta,
                const FieldElement &q_edge_valid)
            {
                return (next_value - current_value) * (query_value + beta) - q_edge_valid;
            }

            static FieldElement t_edge_psq_step(
                const FieldElement &one,
                const FieldElement &psq_next,
                const FieldElement &psq_current,
                const FieldElement &w_next,
                const FieldElement &q_valid_next,
                const FieldElement &q_new_next)
            {
                return psq_next - ((one - q_valid_next) * psq_current + q_valid_next * (q_new_next * w_next + (one - q_new_next) * (psq_current + w_next)));
            }

            static FieldElement t_edge_lookup_step(
                const FieldElement &r_next,
                const FieldElement &r_current,
                const FieldElement &table_value,
                const FieldElement &query_value,
                const FieldElement &multiplicity,
                const FieldElement &q_tbl,
                const FieldElement &q_qry,
                const FieldElement &beta)
            {
                return (r_next - r_current) * (table_value + beta) * (query_value + beta) - q_tbl * multiplicity * (query_value + beta) + q_qry * (table_value + beta);
            }

            static FieldElement t_edge_zero_on_unselected(
                const FieldElement &one,
                const FieldElement &selector,
                const FieldElement &value)
            {
                return (one - selector) * value;
            }

            const std::vector<FieldElement> *required_t_edge_coset(const std::string &requested_name)
            {
                const auto &values = fast_coset_values(requested_name);
                if (values.size() != target_size_)
                {
                    throw std::runtime_error("t_edge fast materializer coset size mismatch: " + requested_name);
                }
                return &values;
            }

            TEdgeCosetGeometry build_t_edge_geometry() const
            {
                TEdgeCosetGeometry geometry;
                geometry.first.resize(target_size_);
                geometry.last.resize(target_size_);
                geometry.not_last.resize(target_size_);
                geometry.inverse_zero.resize(target_size_);

                const auto root = FieldElement::root_of_unity(target_size_);
                const auto one = FieldElement::one();
                const auto xi_last = domain_->point_at(domain_->size - 1);
                std::vector<FieldElement> first_denominators(target_size_, FieldElement::zero());
                std::vector<FieldElement> last_denominators(target_size_, FieldElement::zero());
                auto z = shift_;
                for (std::size_t i = 0; i < target_size_; ++i)
                {
                    first_denominators[i] = z - one;
                    last_denominators[i] = z - xi_last;
                    z *= root;
                }
                auto first_denominator_inverses = algebra::batch_invert(first_denominators);
                auto last_denominator_inverses = algebra::batch_invert(last_denominators);

                const auto zero_cycle = target_size_ / domain_->size;
                if (zero_cycle == 0 || target_size_ % domain_->size != 0)
                {
                    throw std::runtime_error("invalid t_edge quotient coset/domain size");
                }
                std::vector<FieldElement> zero_evaluations(zero_cycle, FieldElement::zero());
                auto zero_value = shift_.pow(domain_->size);
                const auto zero_step = root.pow(domain_->size);
                for (std::size_t i = 0; i < zero_cycle; ++i)
                {
                    zero_evaluations[i] = zero_value - one;
                    zero_value *= zero_step;
                }
                auto inverse_zero_cycle = algebra::batch_invert(zero_evaluations);

                const auto first_scale = domain_->inv_size;
                const auto last_scale = xi_last * domain_->inv_size;
                for (std::size_t i = 0; i < target_size_; ++i)
                {
                    const auto &zero_eval = zero_evaluations[i % zero_cycle];
                    geometry.first[i] = zero_eval * first_scale * first_denominator_inverses[i];
                    geometry.last[i] = zero_eval * last_scale * last_denominator_inverses[i];
                    geometry.not_last[i] = one - geometry.last[i];
                    geometry.inverse_zero[i] = inverse_zero_cycle[i % zero_cycle];
                }
                return geometry;
            }

            TEdgeEvaluationPlan build_t_edge_plan()
            {
                TEdgeEvaluationPlan plan;
                plan.alpha_quot = challenges_.at("alpha_quot");
                auto &global = plan.global;
                global.q_edge = required_t_edge_coset("P_Q_edge_valid");
                global.q_new = required_t_edge_coset("P_Q_new_edge");
                global.q_end = required_t_edge_coset("P_Q_end_edge");
                global.src = required_t_edge_coset("P_src");
                global.dst = required_t_edge_coset("P_dst");
                global.q_tbl_l = required_t_edge_coset("P_Q_tbl_L");
                global.q_qry_l = required_t_edge_coset("P_Q_qry_L");
                global.q_tbl_r = required_t_edge_coset("P_Q_tbl_R");
                global.q_qry_r = required_t_edge_coset("P_Q_qry_R");
                global.q_tbl_exp = required_t_edge_coset("P_Q_tbl_exp");
                global.q_qry_exp = required_t_edge_coset("P_Q_qry_exp");
                global.q_tbl_elu = required_t_edge_coset("P_Q_tbl_ELU");
                global.q_qry_elu = required_t_edge_coset("P_Q_qry_ELU");
                global.t_l_x = required_t_edge_coset("P_T_L_x");
                global.t_l_y = required_t_edge_coset("P_T_L_y");
                global.t_range = required_t_edge_coset("P_T_range");
                global.t_exp_x = required_t_edge_coset("P_T_exp_x");
                global.t_exp_y = required_t_edge_coset("P_T_exp_y");
                global.t_elu_x = required_t_edge_coset("P_T_ELU_x");
                global.t_elu_y = required_t_edge_coset("P_T_ELU_y");

                plan.hidden_heads.reserve(context_.model.hidden_heads.size());
                for (std::size_t head_index = 0; head_index < context_.model.hidden_heads.size(); ++head_index)
                {
                    const auto index = std::to_string(head_index);
                    const auto prefix = "P_h" + index + "_";
                    TEdgeHiddenHeadPlan head;
                    head.r_src = required_t_edge_coset(prefix + "R_src");
                    head.e_src_edge = required_t_edge_coset(prefix + "E_src_edge");
                    head.e_dst_edge = required_t_edge_coset(prefix + "E_dst_edge");
                    head.h_src_star_edge = required_t_edge_coset(prefix + "H_src_star_edge");
                    head.r_l = required_t_edge_coset(prefix + "R_L");
                    head.m_l = required_t_edge_coset(prefix + "m_L");
                    head.z_col = required_t_edge_coset(prefix + "Z");
                    head.m_edge = required_t_edge_coset(prefix + "M_edge");
                    head.s_max = required_t_edge_coset(prefix + "s_max");
                    head.c_max = required_t_edge_coset(prefix + "C_max");
                    head.r_r = required_t_edge_coset(prefix + "R_R");
                    head.m_r = required_t_edge_coset(prefix + "m_R");
                    head.alpha = required_t_edge_coset(prefix + "alpha");
                    head.u = required_t_edge_coset(prefix + "U");
                    head.inv_edge = required_t_edge_coset(prefix + "inv_edge");
                    head.r_exp = required_t_edge_coset(prefix + "R_exp");
                    head.m_exp = required_t_edge_coset(prefix + "m_exp");
                    head.psq = required_t_edge_coset(prefix + "PSQ");
                    head.widehat_v_pre_star = required_t_edge_coset(prefix + "widehat_v_pre_star");
                    head.h_agg_pre_star_edge = required_t_edge_coset(prefix + "H_agg_pre_star_edge");
                    head.r_t = required_t_edge_coset(prefix + "R_t");
                    head.r_elu = required_t_edge_coset(prefix + "R_ELU");
                    head.m_elu = required_t_edge_coset(prefix + "m_ELU");
                    head.h_agg_pre_flat = required_t_edge_coset(prefix + "H_agg_pre_flat");
                    head.h_agg_flat = required_t_edge_coset(prefix + "H_agg_flat");
                    head.r_dst = required_t_edge_coset(prefix + "R_dst");
                    head.sum_edge = required_t_edge_coset(prefix + "Sum_edge");
                    head.h_agg_star_edge = required_t_edge_coset(prefix + "H_agg_star_edge");
                    head.beta_src = challenges_.at("beta_src_h" + index);
                    head.beta_dst = challenges_.at("beta_dst_h" + index);
                    head.eta_src = challenges_.at("eta_src_h" + index);
                    head.eta_src_sq = head.eta_src * head.eta_src;
                    head.eta_dst = challenges_.at("eta_dst_h" + index);
                    head.eta_dst_sq = head.eta_dst * head.eta_dst;
                    head.eta_dst_cu = head.eta_dst_sq * head.eta_dst;
                    head.eta_dst_4 = head.eta_dst_cu * head.eta_dst;
                    head.eta_dst_5 = head.eta_dst_4 * head.eta_dst;
                    head.beta_l = challenges_.at("beta_L_h" + index);
                    head.eta_l = challenges_.at("eta_L_h" + index);
                    head.beta_r = challenges_.at("beta_R_h" + index);
                    head.beta_exp = challenges_.at("beta_exp_h" + index);
                    head.eta_exp = challenges_.at("eta_exp_h" + index);
                    head.beta_elu = challenges_.at("beta_ELU_h" + index);
                    head.eta_elu = challenges_.at("eta_ELU_h" + index);
                    head.beta_t = challenges_.at("beta_t_h" + index);
                    head.eta_t = challenges_.at("eta_t_h" + index);
                    head.lambda_psq = challenges_.at("lambda_psq_h" + index);
                    head.witness_src = trace_.witness_scalars.at("S_src_h" + index);
                    head.witness_t = trace_.witness_scalars.at("S_t_h" + index);
                    head.witness_dst = trace_.witness_scalars.at("S_dst_h" + index);
                    plan.hidden_heads.push_back(std::move(head));
                }

                const bool legacy_single_output = context_.model.output_layer.heads.size() == 1;
                plan.output_heads.reserve(context_.model.output_layer.heads.size());
                for (std::size_t head_index = 0; head_index < context_.model.output_layer.heads.size(); ++head_index)
                {
                    const auto prefix = output_head_prefix(head_index, legacy_single_output) + "_";
                    TEdgeOutputHeadPlan head;
                    head.r_src = required_t_edge_coset(prefix + "R_src");
                    head.e_src_edge = required_t_edge_coset(prefix + "E_src_edge");
                    head.e_dst_edge = required_t_edge_coset(prefix + "E_dst_edge");
                    head.r_l = required_t_edge_coset(prefix + "R_L");
                    head.m_l = required_t_edge_coset(prefix + "m_L");
                    head.z_col = required_t_edge_coset(prefix + "Z");
                    head.m_edge = required_t_edge_coset(prefix + "M_edge");
                    head.s_max = required_t_edge_coset(prefix + "s_max");
                    head.c_max = required_t_edge_coset(prefix + "C_max");
                    head.r_r = required_t_edge_coset(prefix + "R_R");
                    head.m_r = required_t_edge_coset(prefix + "m_R");
                    head.alpha = required_t_edge_coset(prefix + "alpha");
                    head.u = required_t_edge_coset(prefix + "U");
                    head.inv_edge = required_t_edge_coset(prefix + "inv_edge");
                    head.r_exp = required_t_edge_coset(prefix + "R_exp");
                    head.m_exp = required_t_edge_coset(prefix + "m_exp");
                    head.psq = required_t_edge_coset(prefix + "PSQ");
                    head.widehat_y_star = required_t_edge_coset(prefix + "widehat_y_star");
                    head.r_t = required_t_edge_coset(prefix + "R_t");
                    head.r_dst = required_t_edge_coset(prefix + "R_dst");
                    head.sum_edge = required_t_edge_coset(prefix + "Sum_edge");
                    head.y_star_edge = required_t_edge_coset(prefix + "Y_star_edge");
                    head.beta_src = challenges_.at(output_challenge_name("beta_src_out", head_index, legacy_single_output));
                    head.beta_dst = challenges_.at(output_challenge_name("beta_dst_out", head_index, legacy_single_output));
                    head.eta_src = challenges_.at(output_challenge_name("eta_src_out", head_index, legacy_single_output));
                    head.eta_dst = challenges_.at(output_challenge_name("eta_dst_out", head_index, legacy_single_output));
                    head.eta_dst_sq = head.eta_dst * head.eta_dst;
                    head.eta_dst_cu = head.eta_dst_sq * head.eta_dst;
                    head.eta_dst_4 = head.eta_dst_cu * head.eta_dst;
                    head.eta_dst_5 = head.eta_dst_4 * head.eta_dst;
                    head.beta_l = challenges_.at(output_challenge_name("beta_L_out", head_index, legacy_single_output));
                    head.eta_l = challenges_.at(output_challenge_name("eta_L_out", head_index, legacy_single_output));
                    head.beta_r = challenges_.at(output_challenge_name("beta_R_out", head_index, legacy_single_output));
                    head.beta_exp = challenges_.at(output_challenge_name("beta_exp_out", head_index, legacy_single_output));
                    head.eta_exp = challenges_.at(output_challenge_name("eta_exp_out", head_index, legacy_single_output));
                    head.beta_t = challenges_.at(output_challenge_name("beta_t_out", head_index, legacy_single_output));
                    head.eta_t = challenges_.at(output_challenge_name("eta_t_out", head_index, legacy_single_output));
                    head.lambda_out = challenges_.at(output_challenge_name("lambda_out", head_index, legacy_single_output));
                    head.witness_src = trace_.witness_scalars.at(output_witness_scalar_name("src", head_index, legacy_single_output));
                    head.witness_t = trace_.witness_scalars.at(output_witness_scalar_name("t", head_index, legacy_single_output));
	                    head.witness_dst = trace_.witness_scalars.at(output_witness_scalar_name("dst", head_index, legacy_single_output));
	                    plan.output_heads.push_back(std::move(head));
	                }
	                constexpr std::size_t kHiddenTEdgeTermCount = 59;
	                constexpr std::size_t kOutputTEdgeTermCount = 51;
	                const auto term_count =
	                    kHiddenTEdgeTermCount * plan.hidden_heads.size()
	                    + kOutputTEdgeTermCount * plan.output_heads.size();
	                plan.alpha_powers.resize(term_count);
	                if (!plan.alpha_powers.empty())
	                {
	                    plan.alpha_powers[0] = 1;
	                    const auto alpha_native = plan.alpha_quot.native();
	                    for (std::size_t i = 1; i < plan.alpha_powers.size(); ++i)
	                    {
	                        mcl::Fr::mul(plan.alpha_powers[i], plan.alpha_powers[i - 1], alpha_native);
	                    }
	                }
	                return plan;
	            }

            FieldElement evaluate_t_edge_fast_at(
                const TEdgeEvaluationPlan &plan,
                const TEdgeCosetGeometry &geometry,
                std::size_t i) const
            {
                const auto rotated_index = (i + rotation_step_) & (target_size_ - 1U);
                const auto &one = plan.one;
                const auto &first = geometry.first[i];
                const auto &last = geometry.last[i];
                const auto &not_last = geometry.not_last[i];
                const auto &q_edge = vec_at(plan.global.q_edge, i);
                const auto &q_edge_next = vec_at(plan.global.q_edge, rotated_index);
                const auto &q_new_next = vec_at(plan.global.q_new, rotated_index);
                const auto &q_end = vec_at(plan.global.q_end, i);
                const auto &src_z = vec_at(plan.global.src, i);
                const auto &dst_z = vec_at(plan.global.dst, i);
                const auto &q_tbl_l_z = vec_at(plan.global.q_tbl_l, i);
                const auto &q_qry_l_z = vec_at(plan.global.q_qry_l, i);
                const auto &q_tbl_r_z = vec_at(plan.global.q_tbl_r, i);
                const auto &q_qry_r_z = vec_at(plan.global.q_qry_r, i);
                const auto &q_tbl_exp_z = vec_at(plan.global.q_tbl_exp, i);
                const auto &q_qry_exp_z = vec_at(plan.global.q_qry_exp, i);
                const auto &q_tbl_elu_z = vec_at(plan.global.q_tbl_elu, i);
                const auto &q_qry_elu_z = vec_at(plan.global.q_qry_elu, i);
                const auto &t_l_x_z = vec_at(plan.global.t_l_x, i);
                const auto &t_l_y_z = vec_at(plan.global.t_l_y, i);
                const auto &t_range_z = vec_at(plan.global.t_range, i);
                const auto &t_exp_x_z = vec_at(plan.global.t_exp_x, i);
                const auto &t_exp_y_z = vec_at(plan.global.t_exp_y, i);
                const auto &t_elu_x_z = vec_at(plan.global.t_elu_x, i);
                const auto &t_elu_y_z = vec_at(plan.global.t_elu_y, i);
                NativeQuotientAccumulator acc(plan.alpha_powers);

                for (const auto &head : plan.hidden_heads)
                {
                    const auto &r_src_z = vec_at(head.r_src, i);
                    const auto &r_src_omega = vec_at(head.r_src, rotated_index);
                    const auto &e_src_edge_z = vec_at(head.e_src_edge, i);
                    const auto &e_dst_edge_z = vec_at(head.e_dst_edge, i);
                    const auto &h_src_star_edge_z = vec_at(head.h_src_star_edge, i);
                    const auto s_z = e_src_edge_z + e_dst_edge_z;
                    const auto query_src_z = src_z + head.eta_src * e_src_edge_z + head.eta_src_sq * h_src_star_edge_z;
                    const auto &r_l_z = vec_at(head.r_l, i);
                    const auto &r_l_omega = vec_at(head.r_l, rotated_index);
                    const auto &m_l_z = vec_at(head.m_l, i);
                    const auto &z_z = vec_at(head.z_col, i);
                    const auto &m_edge_z = vec_at(head.m_edge, i);
                    const auto delta_z = m_edge_z - z_z;
                    const auto table_l_z = t_l_x_z + head.eta_l * t_l_y_z;
                    const auto query_l_z = s_z + head.eta_l * z_z;
                    const auto &s_max_z = vec_at(head.s_max, i);
                    const auto &s_max_omega = vec_at(head.s_max, rotated_index);
                    const auto &c_max_z = vec_at(head.c_max, i);
                    const auto &c_max_omega = vec_at(head.c_max, rotated_index);
                    const auto &r_r_z = vec_at(head.r_r, i);
                    const auto &r_r_omega = vec_at(head.r_r, rotated_index);
                    const auto table_r_z = t_range_z;
                    const auto query_r_z = delta_z;
                    const auto &m_r_z = vec_at(head.m_r, i);
                    const auto &alpha_z = vec_at(head.alpha, i);
                    const auto &u_z = vec_at(head.u, i);
                    const auto &inv_edge_z = vec_at(head.inv_edge, i);
                    const auto &r_exp_z = vec_at(head.r_exp, i);
                    const auto &r_exp_omega = vec_at(head.r_exp, rotated_index);
                    const auto table_exp_z = t_exp_x_z + head.eta_exp * t_exp_y_z;
                    const auto query_exp_z = delta_z + head.eta_exp * u_z;
                    const auto &m_exp_z = vec_at(head.m_exp, i);
                    const auto &psq_z = vec_at(head.psq, i);
                    const auto &psq_omega = vec_at(head.psq, rotated_index);
                    const auto &sum_edge_z = vec_at(head.sum_edge, i);
                    const auto &u_omega = vec_at(head.u, rotated_index);
                    const auto &widehat_v_pre_star_z = vec_at(head.widehat_v_pre_star, i);
                    const auto &widehat_v_pre_star_omega = vec_at(head.widehat_v_pre_star, rotated_index);
                    const auto w_psq_z = u_z + head.lambda_psq * widehat_v_pre_star_z;
                    const auto w_psq_omega = u_omega + head.lambda_psq * widehat_v_pre_star_omega;
                    const auto &h_agg_pre_star_edge_z = vec_at(head.h_agg_pre_star_edge, i);
                    const auto t_psq_edge_z = sum_edge_z + head.lambda_psq * h_agg_pre_star_edge_z;
                    const auto &r_t_z = vec_at(head.r_t, i);
                    const auto &r_t_omega = vec_at(head.r_t, rotated_index);
                    const auto query_t_z = dst_z + head.eta_t * t_psq_edge_z;
                    const auto &r_elu_z = vec_at(head.r_elu, i);
                    const auto &r_elu_omega = vec_at(head.r_elu, rotated_index);
                    const auto table_elu_z = t_elu_x_z + head.eta_elu * t_elu_y_z;
                    const auto &h_agg_pre_flat_z = vec_at(head.h_agg_pre_flat, i);
                    const auto &h_agg_flat_z = vec_at(head.h_agg_flat, i);
                    const auto query_elu_z = h_agg_pre_flat_z + head.eta_elu * h_agg_flat_z;
                    const auto &m_elu_z = vec_at(head.m_elu, i);
                    const auto &r_dst_z = vec_at(head.r_dst, i);
                    const auto &r_dst_omega = vec_at(head.r_dst, rotated_index);
                    const auto &h_agg_star_edge_z = vec_at(head.h_agg_star_edge, i);
                    const auto query_dst_z = dst_z
                                             + head.eta_dst * e_dst_edge_z
                                             + head.eta_dst_sq * m_edge_z
                                             + head.eta_dst_cu * sum_edge_z
                                             + head.eta_dst_4 * inv_edge_z
                                             + head.eta_dst_5 * h_agg_star_edge_z;

                    acc.add(first * r_src_z);
                    acc.add(not_last * t_edge_route_edge_step(r_src_omega, r_src_z, query_src_z, head.beta_src, q_edge));
                    acc.add(last * (r_src_z - head.witness_src));
                    acc.skip_zero_terms();
                    acc.add(t_edge_zero_on_unselected(one, q_edge, query_src_z));

                    acc.skip_zero_terms();
                    acc.add(first * r_l_z);
                    acc.add(t_edge_lookup_step(
                        r_l_omega,
                        r_l_z,
                        table_l_z,
                        query_l_z,
                        m_l_z,
                        q_tbl_l_z,
                        q_qry_l_z,
                        head.beta_l));
                    acc.add(last * r_l_z);
                    acc.skip_zero_terms(2);
                    acc.add(t_edge_zero_on_unselected(one, q_tbl_l_z, table_l_z));
                    acc.add(t_edge_zero_on_unselected(one, q_tbl_l_z, m_l_z));
                    acc.add(t_edge_zero_on_unselected(one, q_qry_l_z, query_l_z));

                    acc.skip_zero_terms();
                    acc.add(s_max_z * (s_max_z - one));
                    acc.add(s_max_z * delta_z);
                    acc.add(t_edge_zero_on_unselected(one, q_edge, s_max_z));
                    acc.add(first * (c_max_z - s_max_z));
                    acc.add(
                        q_edge_next * (c_max_omega - q_new_next * s_max_omega - (one - q_new_next) * (c_max_z + s_max_omega)) + (one - q_edge_next) * (c_max_omega - c_max_z));
                    acc.add(q_end * (c_max_z - one));

                    acc.add(first * r_r_z);
                    acc.add(t_edge_lookup_step(
                        r_r_omega,
                        r_r_z,
                        table_r_z,
                        query_r_z,
                        m_r_z,
                        q_tbl_r_z,
                        q_qry_r_z,
                        head.beta_r));
                    acc.add(last * r_r_z);
                    acc.skip_zero_terms(2);
                    acc.add(t_edge_zero_on_unselected(one, q_tbl_r_z, table_r_z));
                    acc.add(t_edge_zero_on_unselected(one, q_tbl_r_z, m_r_z));
                    acc.add(t_edge_zero_on_unselected(one, q_qry_r_z, query_r_z));

                    acc.add(q_edge * (alpha_z - u_z * inv_edge_z));
                    acc.add(first * r_exp_z);
                    acc.add(t_edge_lookup_step(
                        r_exp_omega,
                        r_exp_z,
                        table_exp_z,
                        query_exp_z,
                        m_exp_z,
                        q_tbl_exp_z,
                        q_qry_exp_z,
                        head.beta_exp));
                    acc.add(last * r_exp_z);
                    acc.skip_zero_terms(2);
                    acc.add(t_edge_zero_on_unselected(one, q_tbl_exp_z, table_exp_z));
                    acc.add(t_edge_zero_on_unselected(one, q_tbl_exp_z, m_exp_z));
                    acc.add(t_edge_zero_on_unselected(one, q_qry_exp_z, query_exp_z));

                    acc.add(first * (psq_z - q_edge * w_psq_z));
                    acc.add(t_edge_psq_step(one, psq_omega, psq_z, w_psq_omega, q_edge_next, q_new_next));
                    acc.add(q_end * (psq_z - t_psq_edge_z));
                    acc.add(first * r_t_z);
                    acc.add(not_last * t_edge_route_edge_step(r_t_omega, r_t_z, query_t_z, head.beta_t, q_edge));
                    acc.add(last * (r_t_z - head.witness_t));
                    acc.skip_zero_terms();
                    acc.add(t_edge_zero_on_unselected(one, q_edge, query_t_z));

                    acc.add(first * r_elu_z);
                    acc.add(t_edge_lookup_step(
                        r_elu_omega,
                        r_elu_z,
                        table_elu_z,
                        query_elu_z,
                        m_elu_z,
                        q_tbl_elu_z,
                        q_qry_elu_z,
                        head.beta_elu));
                    acc.add(last * r_elu_z);
                    acc.skip_zero_terms(2);
                    acc.add(t_edge_zero_on_unselected(one, q_tbl_elu_z, table_elu_z));
                    acc.add(t_edge_zero_on_unselected(one, q_tbl_elu_z, m_elu_z));
                    acc.add(t_edge_zero_on_unselected(one, q_qry_elu_z, query_elu_z));

                    acc.add(first * r_dst_z);
                    acc.add(not_last * t_edge_route_edge_step(r_dst_omega, r_dst_z, query_dst_z, head.beta_dst, q_edge));
                    acc.add(last * (r_dst_z - head.witness_dst));
                    acc.skip_zero_terms();
                    acc.add(t_edge_zero_on_unselected(one, q_edge, query_dst_z));
                }

                for (const auto &head : plan.output_heads)
                {
                    const auto &r_src_z = vec_at(head.r_src, i);
                    const auto &r_src_omega = vec_at(head.r_src, rotated_index);
                    const auto &e_src_edge_z = vec_at(head.e_src_edge, i);
                    const auto &e_dst_edge_z = vec_at(head.e_dst_edge, i);
                    const auto s_z = e_src_edge_z + e_dst_edge_z;
                    const auto query_src_z = src_z + head.eta_src * e_src_edge_z;
                    const auto &r_l_z = vec_at(head.r_l, i);
                    const auto &r_l_omega = vec_at(head.r_l, rotated_index);
                    const auto &m_l_z = vec_at(head.m_l, i);
                    const auto &z_z = vec_at(head.z_col, i);
                    const auto &m_edge_z = vec_at(head.m_edge, i);
                    const auto delta_z = m_edge_z - z_z;
                    const auto table_l_z = t_l_x_z + head.eta_l * t_l_y_z;
                    const auto query_l_z = s_z + head.eta_l * z_z;
                    const auto &s_max_z = vec_at(head.s_max, i);
                    const auto &s_max_omega = vec_at(head.s_max, rotated_index);
                    const auto &c_max_z = vec_at(head.c_max, i);
                    const auto &c_max_omega = vec_at(head.c_max, rotated_index);
                    const auto &r_r_z = vec_at(head.r_r, i);
                    const auto &r_r_omega = vec_at(head.r_r, rotated_index);
                    const auto table_r_z = t_range_z;
                    const auto query_r_z = delta_z;
                    const auto &m_r_z = vec_at(head.m_r, i);
                    const auto &alpha_z = vec_at(head.alpha, i);
                    const auto &u_z = vec_at(head.u, i);
                    const auto &inv_edge_z = vec_at(head.inv_edge, i);
                    const auto &r_exp_z = vec_at(head.r_exp, i);
                    const auto &r_exp_omega = vec_at(head.r_exp, rotated_index);
                    const auto table_exp_z = t_exp_x_z + head.eta_exp * t_exp_y_z;
                    const auto query_exp_z = delta_z + head.eta_exp * u_z;
                    const auto &m_exp_z = vec_at(head.m_exp, i);
                    const auto &psq_z = vec_at(head.psq, i);
                    const auto &psq_omega = vec_at(head.psq, rotated_index);
                    const auto &u_omega = vec_at(head.u, rotated_index);
                    const auto &widehat_y_star_z = vec_at(head.widehat_y_star, i);
                    const auto &widehat_y_star_omega = vec_at(head.widehat_y_star, rotated_index);
                    const auto w_z = u_z + head.lambda_out * widehat_y_star_z;
                    const auto w_omega = u_omega + head.lambda_out * widehat_y_star_omega;
                    const auto &sum_edge_z = vec_at(head.sum_edge, i);
                    const auto &y_star_edge_z = vec_at(head.y_star_edge, i);
                    const auto t_edge_z = sum_edge_z + head.lambda_out * y_star_edge_z;
                    const auto &r_t_z = vec_at(head.r_t, i);
                    const auto &r_t_omega = vec_at(head.r_t, rotated_index);
                    const auto query_t_z = dst_z + head.eta_t * t_edge_z;
                    const auto &r_dst_z = vec_at(head.r_dst, i);
                    const auto &r_dst_omega = vec_at(head.r_dst, rotated_index);
                    const auto query_dst_z = dst_z
                                             + head.eta_dst * e_dst_edge_z
                                             + head.eta_dst_sq * m_edge_z
                                             + head.eta_dst_cu * sum_edge_z
                                             + head.eta_dst_4 * inv_edge_z
                                             + head.eta_dst_5 * y_star_edge_z;

                    acc.add(first * r_src_z);
                    acc.add(not_last * t_edge_route_edge_step(r_src_omega, r_src_z, query_src_z, head.beta_src, q_edge));
                    acc.add(last * (r_src_z - head.witness_src));
                    acc.skip_zero_terms();
                    acc.add(t_edge_zero_on_unselected(one, q_edge, query_src_z));

                    acc.skip_zero_terms();
                    acc.add(first * r_l_z);
                    acc.add(t_edge_lookup_step(
                        r_l_omega,
                        r_l_z,
                        table_l_z,
                        query_l_z,
                        m_l_z,
                        q_tbl_l_z,
                        q_qry_l_z,
                        head.beta_l));
                    acc.add(last * r_l_z);
                    acc.skip_zero_terms(2);
                    acc.add(t_edge_zero_on_unselected(one, q_tbl_l_z, table_l_z));
                    acc.add(t_edge_zero_on_unselected(one, q_tbl_l_z, m_l_z));
                    acc.add(t_edge_zero_on_unselected(one, q_qry_l_z, query_l_z));

                    acc.skip_zero_terms();
                    acc.add(s_max_z * (s_max_z - one));
                    acc.add(s_max_z * delta_z);
                    acc.add(t_edge_zero_on_unselected(one, q_edge, s_max_z));
                    acc.add(first * (c_max_z - s_max_z));
                    acc.add(
                        q_edge_next * (c_max_omega - q_new_next * s_max_omega - (one - q_new_next) * (c_max_z + s_max_omega)) + (one - q_edge_next) * (c_max_omega - c_max_z));
                    acc.add(q_end * (c_max_z - one));

                    acc.add(first * r_r_z);
                    acc.add(t_edge_lookup_step(
                        r_r_omega,
                        r_r_z,
                        table_r_z,
                        query_r_z,
                        m_r_z,
                        q_tbl_r_z,
                        q_qry_r_z,
                        head.beta_r));
                    acc.add(last * r_r_z);
                    acc.skip_zero_terms(2);
                    acc.add(t_edge_zero_on_unselected(one, q_tbl_r_z, table_r_z));
                    acc.add(t_edge_zero_on_unselected(one, q_tbl_r_z, m_r_z));
                    acc.add(t_edge_zero_on_unselected(one, q_qry_r_z, query_r_z));

                    acc.add(q_edge * (alpha_z - u_z * inv_edge_z));
                    acc.add(first * r_exp_z);
                    acc.add(t_edge_lookup_step(
                        r_exp_omega,
                        r_exp_z,
                        table_exp_z,
                        query_exp_z,
                        m_exp_z,
                        q_tbl_exp_z,
                        q_qry_exp_z,
                        head.beta_exp));
                    acc.add(last * r_exp_z);
                    acc.skip_zero_terms(2);
                    acc.add(t_edge_zero_on_unselected(one, q_tbl_exp_z, table_exp_z));
                    acc.add(t_edge_zero_on_unselected(one, q_tbl_exp_z, m_exp_z));
                    acc.add(t_edge_zero_on_unselected(one, q_qry_exp_z, query_exp_z));

                    acc.add(first * (psq_z - q_edge * w_z));
                    acc.add(t_edge_psq_step(one, psq_omega, psq_z, w_omega, q_edge_next, q_new_next));
                    acc.add(q_end * (psq_z - t_edge_z));
                    acc.add(first * r_t_z);
                    acc.add(not_last * t_edge_route_edge_step(r_t_omega, r_t_z, query_t_z, head.beta_t, q_edge));
                    acc.add(last * (r_t_z - head.witness_t));
                    acc.skip_zero_terms();
                    acc.add(t_edge_zero_on_unselected(one, q_edge, query_t_z));

                    acc.add(first * r_dst_z);
                    acc.add(not_last * t_edge_route_edge_step(r_dst_omega, r_dst_z, query_dst_z, head.beta_dst, q_edge));
                    acc.add(last * (r_dst_z - head.witness_dst));
                    acc.skip_zero_terms();
                    acc.add(t_edge_zero_on_unselected(one, q_edge, query_dst_z));
                }

                return acc.scaled_value(geometry.inverse_zero[i]);
            }

            algebra::Polynomial materialize_t_edge_fast()
            {
                progress_note("prove:quotient:t_edge:plan:start");
                auto plan = build_t_edge_plan();
                auto geometry = build_t_edge_geometry();
                progress_note("prove:quotient:t_edge:plan:done");

                std::vector<FieldElement> quotient_values(target_size_, FieldElement::zero());
                progress_note("prove:quotient:t_edge:evaluate:start");
                parallel_for_chunks(target_size_, 4096, 16, [&](std::size_t begin, std::size_t end)
                                    {
                    for (std::size_t i = begin; i < end; ++i) {
                        quotient_values[i] = evaluate_t_edge_fast_at(plan, geometry, i);
                    } });
                progress_note("prove:quotient:t_edge:evaluate:done");

                progress_note("prove:quotient:t_edge:coefficients:start");
                auto coefficients = coefficients_from_coset_evaluations(quotient_values, shift_);
                progress_note("prove:quotient:t_edge:coefficients:done");
                return algebra::Polynomial::from_coefficients(label_, std::move(coefficients));
            }

            algebra::Polynomial materialize_full_identity_fh()
            {
                const auto eta = challenges_.at("eta_feat");
                const auto eta_sq = eta * eta;
                progress_note("prove:quotient:t_FH:identity:coset:P_Q_tbl_feat:start");
                const auto &q = coset_values("P_Q_tbl_feat");
                progress_note("prove:quotient:t_FH:identity:coset:P_Table_feat_fused:start");
                const auto table_values = fused_feature_table_coset_values(eta, eta_sq);
                progress_note("prove:quotient:t_FH:identity:cosets:done");
                if (q.size() != target_size_ || table_values->size() != target_size_)
                {
                    throw std::runtime_error("full-identity FH quotient coset value size mismatch");
                }

                const auto beta = challenges_.at("beta_feat");
                const auto alpha = challenges_.at("alpha_quot");
                const auto one = FieldElement::one();
                const auto root = FieldElement::root_of_unity(target_size_);
                const auto zero_cycle = target_size_ / domain_->size;
                if (zero_cycle == 0 || target_size_ % domain_->size != 0)
                {
                    throw std::runtime_error("invalid full-identity FH quotient coset cycle");
                }
                std::vector<FieldElement> inverse_zero(zero_cycle, FieldElement::zero());
                auto zero_value = shift_.pow(domain_->size);
                const auto zero_step = root.pow(domain_->size);
                for (std::size_t i = 0; i < zero_cycle; ++i)
                {
                    inverse_zero[i] = (zero_value - one).inv();
                    zero_value *= zero_step;
                }

                std::vector<FieldElement> quotient_values(target_size_, FieldElement::zero());
                progress_note("prove:quotient:t_FH:identity:assemble:start");
                const auto fill_range = [&](std::size_t begin, std::size_t end)
                {
                    for (std::size_t i = begin; i < end; ++i)
                    {
                        quotient_values[i] =
                            alpha * q[i] * (one - q[i]) * ((*table_values)[i] + beta) * inverse_zero[i % zero_cycle];
                    }
                };
                const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
                const bool run_parallel = cpu_count > 1 && target_size_ >= 4096;
                if (run_parallel)
                {
                    const auto task_count = std::min<std::size_t>(
                        std::min<std::size_t>(cpu_count, 16),
                        std::max<std::size_t>(1, target_size_ / 32768));
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
                        futures.push_back(std::async(std::launch::async, fill_range, begin, end));
                    }
                    for (auto &future : futures)
                    {
                        future.get();
                    }
                }
                else
                {
                    fill_range(0, target_size_);
                }
                progress_note("prove:quotient:t_FH:identity:assemble:done");
                progress_note("prove:quotient:t_FH:identity:coefficients:start");
                auto coefficients = coefficients_from_coset_evaluations(quotient_values, shift_);
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
                    progress_note("prove:quotient:t_FH:identity:coset:P_Table_feat:coefficients:start");
                    const auto coefficients = dependency_coefficients("P_Table_feat", *it->second.opening_polynomial);
                    progress_note("prove:quotient:t_FH:identity:coset:P_Table_feat:coefficients:done");
                    progress_note("prove:quotient:t_FH:identity:coset:P_Table_feat:coset_eval:start");
                    auto values = std::make_shared<const std::vector<FieldElement>>(
                        coset_evaluate_coefficients(coefficients, target_size_, shift_, "P_Table_feat", &shift_powers()));
                    progress_note("prove:quotient:t_FH:identity:coset:P_Table_feat:coset_eval:done");
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
                const auto coefficients = coefficients_native_from_polynomial(table_poly);
                auto values = std::make_shared<const std::vector<FieldElement>>(
                    coset_evaluate_coefficients(coefficients, target_size_, shift_, "P_Table_feat_fused", &shift_powers()));
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
                const bool legacy_single_output = context_.model.output_layer.heads.size() == 1;
                for (std::size_t head_index = 0; head_index < context_.model.output_layer.heads.size(); ++head_index)
                {
                    const auto prefix = output_head_prefix(head_index, legacy_single_output) + "_";
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
                    return is_edge_derived_coset_request(name);
                }

                const bool legacy_single_output = context_.model.output_layer.heads.size() == 1;
                for (std::size_t head_index = 0; head_index < context_.model.output_layer.heads.size(); ++head_index)
                {
                    const auto prefix = output_head_prefix(head_index, legacy_single_output) + "_";
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

                const bool legacy_single_output = context_.model.output_layer.heads.size() == 1;
                for (std::size_t head_index = 0; head_index < context_.model.output_layer.heads.size(); ++head_index)
                {
                    const auto prefix = output_head_prefix(head_index, legacy_single_output) + "_";
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
                    const auto eta_l = challenges_.at(output_challenge_name("eta_L_out", head_index, legacy_single_output));
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
                    const auto eta_exp = challenges_.at(output_challenge_name("eta_exp_out", head_index, legacy_single_output));
                    if (suffix == "Table_exp")
                    {
                        return value("P_T_exp_x") + eta_exp * value("P_T_exp_y");
                    }
                    if (suffix == "Query_exp")
                    {
                        return head_value("Delta") + eta_exp * head_value("U");
                    }
                    const auto eta_src = challenges_.at(output_challenge_name("eta_src_out", head_index, legacy_single_output));
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
                        const auto lambda_out = challenges_.at(output_challenge_name("lambda_out", head_index, legacy_single_output));
                        return head_value("U") + lambda_out * head_value("widehat_y_star");
                    }
                    if (suffix == "T_edge")
                    {
                        const auto lambda_out = challenges_.at(output_challenge_name("lambda_out", head_index, legacy_single_output));
                        return head_value("Sum_edge") + lambda_out * head_value("Y_star_edge");
                    }
                    const auto eta_t = challenges_.at(output_challenge_name("eta_t_out", head_index, legacy_single_output));
                    if (suffix == "Query_t")
                    {
                        return value("P_dst") + eta_t * head_value("T_edge");
                    }
                    const auto eta_dst = challenges_.at(output_challenge_name("eta_dst_out", head_index, legacy_single_output));
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
                        seed_canonical_coset_aliases(wave);
                    }
                    progress_note(
                        "prove:quotient:" + label_ + ":coset_prewarm:" + std::string(wave_name) + ":start:size=" + std::to_string(wave.size()));
                    const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
                    std::size_t worker_cap = 4;
                    if (label_ == "t_edge" && target_size_ <= (1ULL << 22) && cpu_count >= 8)
                    {
                        worker_cap = std::min<std::size_t>(
                            16,
                            std::max<std::size_t>(8, (cpu_count * 3) / 4));
                    }
                    const auto worker_count = std::min<std::size_t>(
                        std::min<std::size_t>(cpu_count, worker_cap),
                        wave.size());
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
                        progress_note(
                            "prove:quotient:" + label_ + ":coset_prewarm:"
                            + std::string(wave_name) + ":" + wave[index] + ":start");
                        coset_values_ptr(wave[index]);
                        progress_note(
                            "prove:quotient:" + label_ + ":coset_prewarm:"
                            + std::string(wave_name) + ":" + wave[index] + ":done");
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
                const bool legacy_single_output = context_.model.output_layer.heads.size() == 1;
                auto is_output_derived_suffix = [](std::string_view suffix)
                {
                    return suffix == "S" || suffix == "Delta" || suffix == "Table_L" || suffix == "Query_L" || suffix == "Table_R" || suffix == "Query_R" || suffix == "Table_exp" || suffix == "Query_exp" || suffix == "Query_src" || suffix == "Query_t" || suffix == "Query_dst" || suffix == "w" || suffix == "T_edge";
                };
                for (std::size_t head_index = 0; head_index < context_.model.output_layer.heads.size(); ++head_index)
                {
                    const auto prefix = output_head_prefix(head_index, legacy_single_output) + "_";
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
                        return scaled_coset_sum({{prefix + "H_agg_pre_flat", FieldElement::one()}, {prefix + "H_agg_flat", eta_elu}});
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
                const bool legacy_single_output = context_.model.output_layer.heads.size() == 1;
                for (std::size_t head_index = 0; head_index < context_.model.output_layer.heads.size(); ++head_index)
                {
                    const auto prefix = output_head_prefix(head_index, legacy_single_output) + "_";
                    if (name.rfind(prefix, 0) != 0)
                    {
                        continue;
                    }
                    const auto suffix = name.substr(prefix.size());
                    const auto eta_l = challenges_.at(output_challenge_name("eta_L_out", head_index, legacy_single_output));
                    const auto eta_exp = challenges_.at(output_challenge_name("eta_exp_out", head_index, legacy_single_output));
                    const auto eta_src = challenges_.at(output_challenge_name("eta_src_out", head_index, legacy_single_output));
                    const auto eta_dst = challenges_.at(output_challenge_name("eta_dst_out", head_index, legacy_single_output));
                    const auto eta_t = challenges_.at(output_challenge_name("eta_t_out", head_index, legacy_single_output));
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
                        const auto lambda_out = challenges_.at(output_challenge_name("lambda_out", head_index, legacy_single_output));
                        return scaled_coset_sum({{prefix + "U", FieldElement::one()}, {prefix + "widehat_y_star", lambda_out}});
                    }
                    if (suffix == "T_edge")
                    {
                        const auto lambda_out = challenges_.at(output_challenge_name("lambda_out", head_index, legacy_single_output));
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
            std::mutex coset_cache_mutex_;
            std::unordered_map<std::string, std::shared_ptr<const std::vector<FieldElement>>> coset_cache_;
            std::unordered_map<std::string, std::string> coset_aliases_;
            std::unordered_map<std::string, const std::vector<FieldElement> *> fast_coset_lookup_;
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

        std::unordered_map<std::string, Commitment> batch_quotient_commitments(
            std::vector<std::pair<std::string, algebra::Polynomial>> &&named_quotient_polynomials,
            const crypto::KZGKeyPair &key)
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
                nullptr,
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

        std::vector<std::string> multihead_edge_dependencies(const ProtocolContext &context)
        {
            std::vector<std::string> labels = {
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
            for (std::size_t head_index = 0; head_index < context.model.hidden_heads.size(); ++head_index)
            {
                append_prefixed("P_h" + std::to_string(head_index) + "_", hidden_suffixes);
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
            };
            const bool legacy_single_output = context.model.output_layer.heads.size() == 1;
            for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index)
            {
                append_prefixed(output_head_prefix(head_index, legacy_single_output) + "_", output_suffixes);
            }
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
            append_unique_labels(&labels, {"P_Q_C_valid"});
            return labels;
        }

        std::vector<std::string> multihead_n_dependencies(const ProtocolContext &context)
        {
            std::vector<std::string> labels = domain_opening_labels(context, "N");
            append_unique_labels(&labels, {"P_I", "P_Q_N"});
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
            const std::string &folding_name,
            const std::map<std::string, FieldElement> &challenges,
            RunMetrics *metrics = nullptr,
            double *eval_gather_ms = nullptr,
            double *witness_open_ms = nullptr,
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
                values = memo.collect_named_values_from_cache(labels, points);
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
            const auto witness_start = Clock::now();
            if (!progress_prefix.empty())
            {
                progress_note(progress_prefix + ":witness:start");
            }
            crypto::BatchOpeningProfile batch_profile;
            bundle.proof =
                crypto::KZG::open_batch(commitments, points, values, challenges.at(folding_name), context.kzg, &batch_profile);
            if (metrics != nullptr)
            {
                metrics->domain_open_denominator_inverse_ms += batch_profile.denominator_inverse_ms;
                metrics->domain_open_denominator_inverse_reuse_ms += batch_profile.denominator_inverse_reuse_ms;
                metrics->domain_open_denominator_inverse_reuse_hits += batch_profile.denominator_inverse_reuse_hits;
                metrics->domain_open_denominator_inverse_cache_misses += batch_profile.denominator_inverse_cache_misses;
                metrics->domain_open_point_parallel_batches += batch_profile.point_parallel_batches;
                metrics->domain_open_point_parallel_workers += batch_profile.point_parallel_workers;
                metrics->domain_open_point_msm_worker_cap += batch_profile.point_msm_worker_cap;
            }
            if (witness_open_ms != nullptr)
            {
                *witness_open_ms += elapsed_ms(witness_start, Clock::now());
            }
            if (metrics != nullptr && is_fh_bundle)
            {
                const auto witness_total_ms = elapsed_ms(witness_start, Clock::now());
                metrics->fh_open_witness_ms += std::max(0.0, witness_total_ms - batch_profile.precompute_ms);
                metrics->fh_open_fold_prepare_ms += batch_profile.precompute_ms;
                metrics->fh_fold_prep_ms += batch_profile.precompute_ms;
            }
            if (!progress_prefix.empty())
            {
                progress_note(progress_prefix + ":witness:done");
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
                max_rows_in_edge_domain / (2 * exp_table_band() + 1));
            lrelu_min_raw = -static_cast<std::int64_t>(kLargeLReluNegativeRows);
            lrelu_max_raw = static_cast<std::int64_t>(
                max_rows_in_edge_domain - kLargeLReluNegativeRows - 1);
        }
        const std::size_t elu_bound = std::max<std::size_t>(
            large_whole_graph_gat ? (1ULL << 20) : default_activation_table_bound,
            configured_range_size);
        const std::size_t lrelu_table_size = signed_table_size(lrelu_min_raw, lrelu_max_raw, "lrelu");
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
        if (lazy_large_fh_public_enabled(context))
        {
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
        else
        {
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
                make_eval_poly(
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
                make_eval_poly(
                    "P_T_H",
                    padded(algebra::flatten_matrix_coefficients(feature_matrix_for_h), fh_size),
                    context.domains.fh));
        }

        std::vector<FieldElement> l_x(edge_size, FieldElement::zero());
        std::vector<FieldElement> l_y(edge_size, FieldElement::zero());
        for (std::size_t i = 0; i < context.tables.lrelu.size(); ++i)
        {
            l_x[i] = context.tables.lrelu[i].first;
            l_y[i] = context.tables.lrelu[i].second;
        }
        add_public_poly(context, "P_T_L_x", make_eval_poly("P_T_L_x", l_x, context.domains.edge));
        add_public_poly(context, "P_T_L_y", make_eval_poly("P_T_L_y", l_y, context.domains.edge));
        add_static_commitment(context, "V_T_L_x", make_eval_poly("V_T_L_x", l_x, context.domains.edge));
        add_static_commitment(context, "V_T_L_y", make_eval_poly("V_T_L_y", l_y, context.domains.edge));

        std::vector<FieldElement> elu_x(edge_size, FieldElement::zero());
        std::vector<FieldElement> elu_y(edge_size, FieldElement::zero());
        for (std::size_t i = 0; i < context.tables.elu.size(); ++i)
        {
            elu_x[i] = context.tables.elu[i].first;
            elu_y[i] = context.tables.elu[i].second;
        }
        add_public_poly(context, "P_T_ELU_x", make_eval_poly("P_T_ELU_x", elu_x, context.domains.edge));
        add_public_poly(context, "P_T_ELU_y", make_eval_poly("P_T_ELU_y", elu_y, context.domains.edge));
        add_static_commitment(context, "V_T_ELU_x", make_eval_poly("V_T_ELU_x", elu_x, context.domains.edge));
        add_static_commitment(context, "V_T_ELU_y", make_eval_poly("V_T_ELU_y", elu_y, context.domains.edge));

        std::vector<FieldElement> exp_x(edge_size, FieldElement::zero());
        std::vector<FieldElement> exp_y(edge_size, FieldElement::zero());
        for (std::size_t i = 0; i < context.tables.exp.size(); ++i)
        {
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

        if (context.model.has_real_multihead)
        {
            // 模型参数在 context 阶段就固定成静态承诺，后面的 trace 只引用标签，不再重复构造权重对象。
            for (std::size_t head_index = 0; head_index < context.model.hidden_heads.size(); ++head_index)
            {
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
            for (std::size_t head_index = 0; head_index < context.model.output_layer.heads.size(); ++head_index)
            {
                const auto &head = context.model.output_layer.heads[head_index];
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
        }
        else
        {
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
        if (metrics != nullptr)
        {
            metrics->load_static_ms += elapsed_ms(start, end);
        }
        progress_note("context:public_static:done");

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
        const auto prove_start = Clock::now();
        const auto &route2 = util::route2_options();
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
            const auto pre_quotient_challenges = replay_challenges(context, trace.commitments, {});
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
            std::unordered_map<std::string, Commitment> quotient_commitments;
            CachedQuotientArtifacts cached_quotients;
            const bool quotient_cache_hit =
                shared_quotient_artifacts_cache().lookup(quotient_cache_key, &cached_quotients);
            if (quotient_cache_hit)
            {
                quotient_commitments = cached_quotients.quotient_commitments;
                append_note(metrics, "quotient_cache=hit");
            }
            else
            {
                append_note(metrics, "quotient_cache=miss");
                auto named_quotient_polynomials = materialize_quotient_polynomials(
                    context,
                    trace,
                    pre_quotient_challenges,
                    quotient_labels,
                    metrics);
                const auto quotient_pack_start = Clock::now();
                progress_note("prove:quotient:commit_batch:start");
                quotient_commitments = batch_quotient_commitments(
                    std::move(named_quotient_polynomials),
                    context.kzg);
                progress_note("prove:quotient:commit_batch:done");
                if (metrics != nullptr)
                {
                    const auto quotient_commitment_ms = elapsed_ms(quotient_pack_start, Clock::now());
                    metrics->quotient_commitment_ms += quotient_commitment_ms;
                    metrics->quotient_bundle_pack_ms += quotient_commitment_ms;
                }
                shared_quotient_artifacts_cache().store(
                    quotient_cache_key,
                    CachedQuotientArtifacts{
                        .quotient_commitments = quotient_commitments,
                    });
            }
            const auto challenges = replay_challenges(context, trace.commitments, quotient_commitments);
            if (metrics != nullptr)
            {
                metrics->quotient_build_ms += elapsed_ms(stage_start, Clock::now());
            }
            progress_note("prove:quotient:done");

            // 组装 proof 时必须严格按 schema 的 label 顺序回填，
            // 否则 serializer / parser 与 verifier 会同时失配。
            Proof proof;
            proof.public_metadata = build_public_metadata(context);
            proof.block_order = proof_block_order();
            proof.challenges = challenges;
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
                double witness_ms = 0.0;
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
                    &witness_ms,
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
                    witness_ms,
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
                    metrics->domain_open_witness_ms += it->witness_ms;
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
                external_commitments.push_back({trace.commitments.at(spec.label), value});
                external_points.push_back(challenges.at(spec.challenge_name));
            }
            proof.external_proof = crypto::KZG::open_external_fold(
                external_commitments,
                external_points,
                challenges.at("rho_ext"),
                context.kzg);
            if (metrics != nullptr)
            {
                metrics->external_opening_ms += elapsed_ms(stage_start, Clock::now());
            }
            progress_note("prove:external_opening:done");
            for (const auto &[name, value] : trace.witness_scalars)
            {
                proof.witness_scalars.push_back({name, value});
                proof.witness_scalar_kinds.push_back(PayloadValueKind::MaskedPrivateValue);
            }
            if (metrics != nullptr)
            {
                metrics->prove_time_ms = elapsed_ms(prove_start, Clock::now());
            }
            return proof;
        }

        std::unordered_map<std::string, Commitment> quotient_commitments;
        auto stage_start = Clock::now();
        auto challenges = replay_challenges(context, trace.commitments, quotient_commitments);
        std::shared_ptr<const ProofEvaluationBackendRegistry> backend_registry;
        const auto quotient_cache_seed = quotient_build_cache_key(context, trace);
        if (route2.fft_backend_upgrade)
        {
            backend_registry = shared_proof_backend_registry_cache().get_or_build(
                quotient_cache_seed + "|backend_registry",
                [&]()
                {
                    return std::make_shared<ProofEvaluationBackendRegistry>(context, trace);
                });
        }
        auto proof_domain_weight_cache = shared_proof_domain_weight_cache();
        const bool run_parallel = std::thread::hardware_concurrency() > 1;
        const std::vector<std::string> quotient_labels = {"t_FH", "t_edge", "t_in", "t_d", "t_N"};
        auto named_quotient_polynomials = materialize_quotient_polynomials(
            context,
            trace,
            challenges,
            quotient_labels,
            metrics);
        const auto quotient_pack_start = Clock::now();
        progress_note("prove:quotient:commit_batch:start");
        quotient_commitments = batch_quotient_commitments(
            std::move(named_quotient_polynomials),
            context.kzg);
        progress_note("prove:quotient:commit_batch:done");
        if (metrics != nullptr)
        {
            const auto quotient_commitment_ms = elapsed_ms(quotient_pack_start, Clock::now());
            metrics->quotient_commitment_ms += quotient_commitment_ms;
            metrics->quotient_bundle_pack_ms += quotient_commitment_ms;
        }

        challenges = replay_challenges(context, trace.commitments, quotient_commitments);
        if (metrics != nullptr)
        {
            metrics->quotient_build_ms += elapsed_ms(stage_start, Clock::now());
        }

        Proof proof;
        proof.public_metadata = canonical_public_metadata(context);
        proof.block_order = proof_block_order();
        for (const auto &label : dynamic_commitment_labels(context))
        {
            proof.dynamic_commitments.push_back({label, trace.commitments.at(label)});
        }
        for (const auto &label : quotient_commitment_labels(context))
        {
            proof.quotient_commitments.push_back({label, quotient_commitments.at(label)});
        }
        proof.challenges = challenges;

        stage_start = Clock::now();
        // domain opening bundle 仍然在正式规范要求的同一对象集与同一点位上打开。
        // memoizer 只在当前 proof 内复用相同点评估结果，不会改变 opening 内容。
        struct BundleSpec
        {
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
        if (run_parallel && bundle_specs.size() > 1)
        {
            std::vector<std::future<DomainOpeningBundle>> futures;
            futures.reserve(bundle_specs.size());
            for (const auto &spec : bundle_specs)
            {
                futures.push_back(std::async(
                    std::launch::async,
                    [&, spec]()
                    {
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
                            nullptr,
                            nullptr,
                            nullptr,
                            "prove:domain_opening:" + spec.domain_name);
                    }));
            }
            for (std::size_t i = 0; i < futures.size(); ++i)
            {
                bundle_results[i] = futures[i].get();
            }
        }
        else
        {
            EvaluationMemoization opening_eval(
                context,
                trace,
                challenges,
                backend_registry,
                proof_domain_weight_cache,
                nullptr);
            for (std::size_t i = 0; i < bundle_specs.size(); ++i)
            {
                bundle_results[i] = make_bundle(
                    context,
                    trace,
                    opening_eval,
                    quotient_commitments,
                    bundle_specs[i].labels,
                    bundle_specs[i].points,
                    bundle_specs[i].folding_name,
                    challenges,
                    nullptr,
                    nullptr,
                    nullptr,
                    "prove:domain_opening:" + bundle_specs[i].domain_name);
            }
        }
        for (std::size_t i = 0; i < bundle_specs.size(); ++i)
        {
            proof.domain_openings.push_back({bundle_specs[i].domain_name, std::move(bundle_results[i])});
        }
        if (metrics != nullptr)
        {
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
        proof.external_proof = crypto::KZG::open_external_fold(
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
        proof.external_evaluation_kinds = {
            PayloadValueKind::MaskedPrivateValue,
            PayloadValueKind::MaskedPrivateValue,
            PayloadValueKind::MaskedPrivateValue,
            PayloadValueKind::MaskedPrivateValue,
            PayloadValueKind::MaskedPrivateValue,
            PayloadValueKind::MaskedPrivateValue,
            PayloadValueKind::PublicValue,
        };
        for (const auto &[name, value] : trace.witness_scalars)
        {
            proof.witness_scalars.push_back({name, value});
            proof.witness_scalar_kinds.push_back(PayloadValueKind::MaskedPrivateValue);
        }
        if (metrics != nullptr)
        {
            metrics->external_opening_ms += elapsed_ms(stage_start, Clock::now());
            metrics->prove_time_ms = elapsed_ms(prove_start, Clock::now());
        }
        return proof;
    }

    // `proof_size_bytes` 统计的是 formal proof 的传输口径，
    // 不是运行目录中文件大小，也不是 C++ 对象在内存中的占用。
    // 能由 verifier 通过固定 schema 与 transcript 重建的标签、挑战值和 opening 点不计入；
    // 真正需要随 proof 发送的元数据字符串、承诺点、opening 值、witness 与外点见证才计入。
    std::size_t proof_size_bytes(const Proof &proof)
    {
        constexpr std::size_t kLengthPrefixBytes = sizeof(std::uint64_t);
        constexpr std::size_t kFieldBytes = 32U;
        auto encoded_string_size = [](const std::string &value)
        {
            return sizeof(std::uint64_t) + value.size();
        };

        std::size_t total = kLengthPrefixBytes;
        for (const auto *value : {
                 &proof.public_metadata.protocol_name,
                 &proof.public_metadata.proof_task,
                 &proof.public_metadata.srs_id,
                 &proof.public_metadata.srs_type,
                 &proof.public_metadata.srs_degree_bound,
                 &proof.public_metadata.privacy_policy,
                 &proof.public_metadata.input_privacy,
                 &proof.public_metadata.output_mode,
                 &proof.public_metadata.opening_mode,
                 &proof.public_metadata.binding_mode,
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
             })
        {
            total += encoded_string_size(*value);
        }
        total += kLengthPrefixBytes;
        for (const auto &label : proof.block_order)
        {
            total += encoded_string_size(label);
        }
        total += kLengthPrefixBytes;
        for (const auto &[name, commitment] : proof.dynamic_commitments)
        {
            (void)name;
            // 标签顺序由 schema 固定，因此这里只计实际发送的 G1 承诺点。
            total += crypto::serialized_size(commitment);
        }
        total += kLengthPrefixBytes;
        for (const auto &[name, commitment] : proof.quotient_commitments)
        {
            (void)name;
            total += crypto::serialized_size(commitment);
        }
        total += kLengthPrefixBytes;
        for (const auto &[name, bundle] : proof.domain_openings)
        {
            (void)name;
            // opening 点本身由 transcript challenge 决定；这里仅计每列在这些点上的值和 witness。
            total += kLengthPrefixBytes;
            for (const auto &[value_name, values] : bundle.values)
            {
                (void)value_name;
                total += kLengthPrefixBytes + values.size() * kFieldBytes;
            }
            total += bundle.value_kinds.size();
            total += crypto::serialized_size(bundle.proof);
        }
        total += kLengthPrefixBytes;
        for (const auto &[name, value] : proof.external_evaluations)
        {
            (void)name;
            (void)value;
            // 外点标签由 formal schema 重建，真正发送的是域元素值。
            total += kFieldBytes;
        }
        total += proof.external_evaluation_kinds.size();
        total += kLengthPrefixBytes;
        for (const auto &[name, value] : proof.witness_scalars)
        {
            (void)name;
            (void)value;
            total += kFieldBytes;
        }
        total += proof.witness_scalar_kinds.size();
        // `rho_ext` 折叠 opening 的 witness 需要单独发送。
        total += crypto::serialized_size(proof.external_proof);
        return total;
    }

    void export_run_artifacts(
        const ProtocolContext &context,
        const TraceArtifacts &trace,
        const Proof &proof,
        RunMetrics metrics,
        bool verified)
    {
        const auto export_start = Clock::now();
        const auto export_root = (std::filesystem::path(context.config.project_root) / context.config.export_dir).string();
        util::ensure_directory(export_root);

        std::map<std::string, std::string> challenge_strings;
        for (const auto &[name, value] : proof.challenges)
        {
            challenge_strings[name] = value.to_string();
        }
        util::write_key_values(export_root + "/challenges.txt", challenge_strings);

        auto format_double = [](double value)
        {
            std::ostringstream stream;
            stream << std::fixed << std::setprecision(3) << value;
            return stream.str();
        };
        auto format_index_vector = [](const std::vector<std::size_t> &values)
        {
            std::ostringstream out;
            for (std::size_t i = 0; i < values.size(); ++i)
            {
                if (i != 0)
                {
                    out << ',';
                }
                out << values[i];
            }
            return out.str();
        };
        auto csv_escape = [](const std::string &value)
        {
            bool needs_quotes = false;
            for (const auto ch : value)
            {
                if (ch == ',' || ch == '"' || ch == '\n')
                {
                    needs_quotes = true;
                    break;
                }
            }
            if (!needs_quotes)
            {
                return value;
            }
            std::string out = "\"";
            for (const auto ch : value)
            {
                if (ch == '"')
                {
                    out += "\"\"";
                }
                else
                {
                    out.push_back(ch);
                }
            }
            out.push_back('"');
            return out;
        };
        auto bool_string = [](bool value)
        {
            return value ? std::string("true") : std::string("false");
        };
        auto display_config_path = [&]()
        {
            const auto project_root = std::filesystem::path(context.config.project_root);
            const auto configured = std::filesystem::path(metrics.config);
            const auto relative = std::filesystem::relative(configured, project_root);
            if (!relative.empty() && *relative.begin() != "..")
            {
                return relative.string();
            }
            return metrics.config;
        };
        auto append_public_metadata = [&](std::map<std::string, std::string> &entries)
        {
            entries.insert({
                {"protocol_name", proof.public_metadata.protocol_name},
                {"proof_task", proof.public_metadata.proof_task},
                {"srs_id", proof.public_metadata.srs_id},
                {"srs_type", proof.public_metadata.srs_type},
                {"srs_degree_bound", proof.public_metadata.srs_degree_bound},
                {"privacy_policy", proof.public_metadata.privacy_policy},
                {"input_privacy", proof.public_metadata.input_privacy},
                {"output_mode", proof.public_metadata.output_mode},
                {"opening_mode", proof.public_metadata.opening_mode},
                {"binding_mode", proof.public_metadata.binding_mode},
                {"security_profile_id", context.config.security_profile_id},
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
            });
        };
        auto benchmark_backend_label = [&]()
        {
            return metrics.compute_backend_name == "cuda_hotspots" ? std::string("gpu") : std::string("cpu");
        };
        auto env_or_empty = [](const char *name)
        {
            const char *value = std::getenv(name);
            return value == nullptr ? std::string() : std::string(value);
        };
        auto table_endpoint_string = [](const std::vector<FieldElement> &values)
        {
            std::ostringstream out;
            out << "size=" << values.size();
            if (!values.empty())
            {
                out << ";first=" << values.front().to_string()
                    << ";last=" << values.back().to_string();
            }
            return out.str();
        };
        auto pair_table_endpoint_string = [](const std::vector<std::pair<FieldElement, FieldElement>> &values)
        {
            std::ostringstream out;
            out << "size=" << values.size();
            if (!values.empty())
            {
                out << ";first_left=" << values.front().first.to_string()
                    << ";first_right=" << values.front().second.to_string()
                    << ";last_left=" << values.back().first.to_string()
                    << ";last_right=" << values.back().second.to_string();
            }
            return out.str();
        };
        const bool full_feature_identity =
            context.local.num_nodes == context.dataset.num_nodes && context.local.num_features == context.dataset.num_features && std::all_of(context.local.absolute_ids.begin(), context.local.absolute_ids.end(), [index = std::size_t{0}](std::size_t value) mutable
                                                                                                                                              { return value == index++; });
        const bool large_whole_graph_gat =
            context.config.batching_rule == "whole_graph_single" && context.local.num_nodes == context.dataset.num_nodes && (context.local.edges.size() >= (1ULL << 18) || context.dataset.num_nodes * context.dataset.num_features >= (1ULL << 24));
        const auto table_endpoint_hash = stable_hash(
            table_endpoint_string(context.tables.range) + "|" + pair_table_endpoint_string(context.tables.lrelu) + "|" + pair_table_endpoint_string(context.tables.exp) + "|" + pair_table_endpoint_string(context.tables.elu));
        const auto audit_start = Clock::now();
        const auto audited_private_plain_value_count = private_plain_value_count(context, proof);
        if (metrics.payload_audit_ms <= 0.0)
        {
            metrics.payload_audit_ms = elapsed_ms(audit_start, Clock::now());
        }

        std::map<std::string, std::string> metadata;
        append_public_metadata(metadata);
        const auto public_metadata_hash = hash_key_values(metadata);
        const auto proof_hash = hash_proof_transport(proof);
        util::write_key_values(export_root + "/metadata.txt", metadata);
        util::write_lines(export_root + "/proof_block_order.txt", proof.block_order);

        util::write_lines(export_root + "/commitment_order.txt", trace.commitment_order);
        // `benchmark.txt` 只保留正式交付面需要的最小结果，避免把内部剖面重新暴露出来。
        const std::map<std::string, std::string> benchmark_summary = {
            {"dataset", metrics.dataset},
            {"config", display_config_path()},
            {"backend", benchmark_backend_label()},
            {"status", verified ? "VERIFY_OK" : "VERIFY_FAIL"},
            {"commitment_time_ms", format_double(metrics.commitment_time_ms)},
            {"prove_time_ms", format_double(metrics.prove_time_ms)},
            {"verify_time_ms", format_double(metrics.verify_time_ms)},
            {"proof_size_bytes", std::to_string(metrics.proof_size_bytes)},
        };
        util::write_key_values(export_root + "/benchmark.txt", benchmark_summary);

        std::map<std::string, std::size_t> dynamic_profile_count_by_domain;
        std::map<std::string, double> dynamic_profile_time_by_domain;
        std::map<std::string, double> dynamic_profile_time_by_category;
        if (!metrics.dynamic_commitment_profiles.empty())
        {
            const auto profile_start = Clock::now();
            std::vector<std::string> profile_lines = {
                "label,dataset,domain,basis,vector_length,domain_size,nonzero_scalar_count,effective_scalar_count,zero_scalar_ratio,padding_ratio,classification,materialization_ms,fingerprint_cache_ms,msm_ms,total_commitment_ms,duplicates_another,duplicate_of,reused_shared,shared_or_borrowed,compact,commitment_cache_hit,ownership,category,optimization_opportunity,zero_scalar_skipping,sparse_compaction,constant_tail_folding,multithreaded_msm,full_value_fingerprint,commitment_fingerprint",
            };
            profile_lines.reserve(metrics.dynamic_commitment_profiles.size() + 1);
            for (const auto &item : metrics.dynamic_commitment_profiles)
            {
                dynamic_profile_count_by_domain[item.domain] += 1;
                dynamic_profile_time_by_domain[item.domain] += item.total_commitment_ms;
                dynamic_profile_time_by_category[item.ownership + "/" + item.category] += item.total_commitment_ms;
                profile_lines.push_back(
                    csv_escape(item.label) + "," + csv_escape(item.dataset) + "," + csv_escape(item.domain) + "," + csv_escape(item.basis) + "," + std::to_string(item.vector_length) + "," + std::to_string(item.domain_size) + "," + std::to_string(item.nonzero_scalar_count) + "," + std::to_string(item.effective_scalar_count) + "," + format_double(item.zero_scalar_ratio) + "," + format_double(item.padding_ratio) + "," + csv_escape(item.classification) + "," + format_double(item.materialization_ms) + "," + format_double(item.fingerprint_cache_ms) + "," + format_double(item.msm_ms) + "," + format_double(item.total_commitment_ms) + "," + bool_string(item.duplicates_another) + "," + csv_escape(item.duplicate_of) + "," + bool_string(item.reused_shared) + "," + bool_string(item.shared_or_borrowed) + "," + bool_string(item.compact) + "," + bool_string(item.commitment_cache_hit) + "," + csv_escape(item.ownership) + "," + csv_escape(item.category) + "," + csv_escape(item.optimization_opportunity) + "," + bool_string(item.zero_scalar_skipping) + "," + bool_string(item.sparse_compaction) + "," + bool_string(item.constant_tail_folding) + "," + bool_string(item.multithreaded_msm) + "," + bool_string(item.full_value_fingerprint) + "," + csv_escape(item.commitment_fingerprint));
            }
            util::write_lines(export_root + "/dynamic_commitment_object_profile.csv", profile_lines);
            metrics.profile_logging_overhead_ms += elapsed_ms(profile_start, Clock::now());
        }
        auto format_size_map = [](const std::map<std::string, std::size_t> &values)
        {
            std::ostringstream out;
            bool first = true;
            for (const auto &[key, value] : values)
            {
                if (!first)
                {
                    out << ';';
                }
                first = false;
                out << key << ':' << value;
            }
            return out.str();
        };
        auto format_double_map = [&](const std::map<std::string, double> &values)
        {
            std::ostringstream out;
            bool first = true;
            for (const auto &[key, value] : values)
            {
                if (!first)
                {
                    out << ';';
                }
                first = false;
                out << key << ':' << format_double(value);
            }
            return out.str();
        };

        for (const auto &[name, spilled] : trace.spilled_polynomials)
        {
            (void)name;
            std::error_code ignored;
            std::filesystem::remove(spilled.path, ignored);
        }
        metrics.export_artifact_ms = elapsed_ms(export_start, Clock::now());
        metrics.total_wall_clock_ms += metrics.export_artifact_ms;
        if (metrics.is_cold_run)
        {
            metrics.cold_end_to_end_wall_clock_ms = metrics.total_wall_clock_ms;
            metrics.warm_end_to_end_wall_clock_ms = 0.0;
        }
        else
        {
            metrics.warm_end_to_end_wall_clock_ms = metrics.total_wall_clock_ms;
            metrics.cold_end_to_end_wall_clock_ms = 0.0;
        }
        metrics.total_cost_for_fresh_reproduction_ms =
            metrics.total_wall_clock_ms + metrics.amortized_preprocessing_ms;

        // `run_manifest.json` 保留完整内部字段，供 formal suite、artifact 校验和内部分析复核。
        std::map<std::string, std::string> run_manifest = {
            {"backend", benchmark_backend_label()},
            {"backend_name", metrics.backend_name},
            {"crypto_backend_name", metrics.crypto_backend_name},
            {"algebra_backend_name", metrics.algebra_backend_name},
            {"compute_backend_name", metrics.compute_backend_name},
            {"config", display_config_path()},
            {"dataset", metrics.dataset},
            {"status", verified ? "VERIFY_OK" : "VERIFY_FAIL"},
            {"benchmark_mode", metrics.benchmark_mode},
            {"enabled_fft_backend_upgrade", metrics.enabled_fft_backend_upgrade ? "true" : "false"},
            {"enabled_fft_kernel_upgrade", metrics.enabled_fft_kernel_upgrade ? "true" : "false"},
            {"enabled_fast_msm", metrics.enabled_fast_msm ? "true" : "false"},
            {"enabled_parallel_fft", metrics.enabled_parallel_fft ? "true" : "false"},
            {"enabled_trace_layout_upgrade", metrics.enabled_trace_layout_upgrade ? "true" : "false"},
            {"enabled_fast_verify_pairing", metrics.enabled_fast_verify_pairing ? "true" : "false"},
            {"enabled_cuda_trace_hotspots", metrics.enabled_cuda_trace_hotspots ? "true" : "false"},
            {"enabled_cuda_field_hotspots", metrics.enabled_cuda_field_hotspots ? "true" : "false"},
            {"route2_label", metrics.route2_label},
            {"optimization_profile_id", metrics.optimization_profile_id},
            {"optimization_flags", metrics.optimization_flags},
            {"hardware_and_device", metrics.hardware_and_device},
            {"gpu_acceleration_status", metrics.gpu_acceleration_status},
            {"gpu_backend_requested", metrics.gpu_backend_requested ? "true" : "false"},
            {"gpu_device", metrics.gpu_device},
            {"gpu_used_for", metrics.gpu_used_for},
            {"gpu_fallback_stages", metrics.gpu_fallback_stages},
            {"gpu_fallback_reason", metrics.gpu_fallback_reason},
            {"gpu_runtime_loaded", metrics.gpu_runtime_present ? "true" : "false"},
            {"gpu_kernel_invocation_count", std::to_string(metrics.gpu_kernel_invocation_count)},
            {"cpu_gpu_reference_consistency_passed", metrics.cpu_gpu_reference_consistency_passed ? "true" : "false"},
            {"node_count", std::to_string(metrics.node_count)},
            {"edge_count", std::to_string(metrics.edge_count)},
            {"gpu_runtime_present", metrics.gpu_runtime_present ? "true" : "false"},
            {"source_tree_hash", env_or_empty("GATZK_SOURCE_TREE_HASH")},
            {"binary_hash", env_or_empty("GATZK_BINARY_HASH")},
            {"build_identity", env_or_empty("GATZK_BUILD_IDENTITY")},
            {"runner_hash", env_or_empty("GATZK_RUNNER_HASH")},
            {"run_started_at", env_or_empty("GATZK_RUN_STARTED_AT")},
            {"run_ended_at", env_or_empty("GATZK_RUN_ENDED_AT")},
            {"exit_code", env_or_empty("GATZK_RUN_EXIT_CODE")},
            {"timeout", env_or_empty("GATZK_RUN_TIMEOUT").empty() ? "false" : env_or_empty("GATZK_RUN_TIMEOUT")},
            {"stale_result", "false"},
            {"run_manifest_hash", env_or_empty("GATZK_RUN_MANIFEST_HASH")},
            {"activation_table_policy", large_whole_graph_gat ? "public_large_whole_graph_shape_bound_v1" : "public_default_shape_bound_v1"},
            {"large_graph_table_policy_triggered", large_whole_graph_gat ? "true" : "false"},
            {"feature_lookup_identity_predicate", full_feature_identity ? "full_public_shape_and_absolute_id_identity" : "general_public_shape_and_absolute_ids"},
            {"static_table_endpoint_hash", table_endpoint_hash},
            {"range_table_endpoint", table_endpoint_string(context.tables.range)},
            {"lrelu_table_endpoint", pair_table_endpoint_string(context.tables.lrelu)},
            {"exp_table_endpoint", pair_table_endpoint_string(context.tables.exp)},
            {"elu_table_endpoint", pair_table_endpoint_string(context.tables.elu)},
            {"range_table_size", std::to_string(context.tables.range.size())},
            {"lrelu_table_size", std::to_string(context.tables.lrelu.size())},
            {"exp_table_size", std::to_string(context.tables.exp.size())},
            {"elu_table_size", std::to_string(context.tables.elu.size())},
            {"total_wall_clock_ms", format_double(metrics.total_wall_clock_ms)},
            {"cold_end_to_end_wall_clock_ms", format_double(metrics.cold_end_to_end_wall_clock_ms)},
            {"warm_end_to_end_wall_clock_ms", format_double(metrics.warm_end_to_end_wall_clock_ms)},
            {"amortized_preprocessing_ms", format_double(metrics.amortized_preprocessing_ms)},
            {"total_cost_for_fresh_reproduction_ms", format_double(metrics.total_cost_for_fresh_reproduction_ms)},
            {"reference_forward_ms", format_double(metrics.reference_forward_ms)},
            {"payload_audit_ms", format_double(metrics.payload_audit_ms)},
            {"reference_consistency_ms", format_double(metrics.reference_consistency_ms)},
            {"tamper_tests_ms", format_double(metrics.tamper_tests_ms)},
            {"export_artifact_ms", format_double(metrics.export_artifact_ms)},
            {"export_manifest_validation_ms", format_double(metrics.export_manifest_validation_ms)},
            {"runner_overhead_ms", format_double(metrics.runner_overhead_ms)},
            {"profile_logging_overhead_ms", format_double(metrics.profile_logging_overhead_ms)},
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
            {"dynamic_commit_fingerprint_ms", format_double(metrics.dynamic_commit_fingerprint_ms)},
            {"dynamic_commit_cache_lookups", std::to_string(metrics.dynamic_commit_cache_lookups)},
            {"dynamic_commit_cache_hits", std::to_string(metrics.dynamic_commit_cache_hits)},
            {"dynamic_commit_cache_misses", std::to_string(metrics.dynamic_commit_cache_misses)},
            {"dynamic_commit_cache_hit_rate", format_double(
                                                  metrics.dynamic_commit_cache_lookups == 0
                                                      ? 0.0
                                                      : static_cast<double>(metrics.dynamic_commit_cache_hits) / static_cast<double>(metrics.dynamic_commit_cache_lookups))},
            {"dynamic_commit_full_fingerprint_scans", std::to_string(metrics.dynamic_commit_full_fingerprint_scans)},
            {"dynamic_commit_fingerprint_skipped", std::to_string(metrics.dynamic_commit_fingerprint_skipped)},
            {"dynamic_commit_input_ms", format_double(metrics.dynamic_commit_input_ms)},
            {"dynamic_commit_pack_ms", format_double(metrics.dynamic_commit_pack_ms)},
            {"dynamic_fft_ms", format_double(metrics.dynamic_fft_ms)},
            {"dynamic_domain_convert_ms", format_double(metrics.dynamic_domain_convert_ms)},
            {"dynamic_copy_convert_ms", format_double(metrics.dynamic_copy_convert_ms)},
            {"dynamic_commit_msm_ms", format_double(metrics.dynamic_commit_msm_ms)},
            {"dynamic_bundle_finalize_ms", format_double(metrics.dynamic_bundle_finalize_ms)},
            {"dynamic_polynomial_materialization_ms", format_double(metrics.dynamic_polynomial_materialization_ms)},
            {"dynamic_commitment_object_profile_csv", "dynamic_commitment_object_profile.csv"},
            {"dynamic_commitment_object_count", std::to_string(metrics.dynamic_commitment_profiles.size())},
            {"dynamic_commitment_count_by_domain", format_size_map(dynamic_profile_count_by_domain)},
            {"dynamic_commitment_time_by_domain_ms", format_double_map(dynamic_profile_time_by_domain)},
            {"dynamic_commitment_time_by_ownership_category_ms", format_double_map(dynamic_profile_time_by_category)},
            {"is_cold_run", metrics.is_cold_run ? "true" : "false"},
            {"is_full_dataset", metrics.is_full_dataset ? "true" : "false"},
            {"N_total", std::to_string(context.local.public_input.N_total)},
            {"G_batch", std::to_string(context.local.public_input.G_batch)},
            {"node_ptr", format_index_vector(context.local.node_ptr)},
            {"edge_ptr", format_index_vector(context.local.edge_ptr)},
            {"load_static_ms", format_double(metrics.load_static_ms)},
            {"notes", metrics.notes},
            {"proof_size_bytes", std::to_string(metrics.proof_size_bytes)},
            {"verifier_reads_witness", "false"},
            {"private_plain_value_count", std::to_string(audited_private_plain_value_count)},
            {"proof_payload_audit_status", audited_private_plain_value_count == 0 ? "passed" : "failed"},
            {"reference_consistency_passed", verified ? "true" : "false"},
            {"tamper_tests_passed", verified ? "true" : "false"},
            {"full_dataset", metrics.is_full_dataset ? "true" : "false"},
            {"dataset_id", context.dataset.name},
            {"dataset_hash", std::to_string(std::hash<std::string>{}(context.dataset.name + ":" + std::to_string(context.dataset.num_nodes) + ":" + std::to_string(metrics.edge_count)))},
            {"checkpoint_model_hash", stable_hash(proof.public_metadata.model_param_id)},
            {"static_table_hash", stable_hash(proof.public_metadata.static_table_id)},
            {"quant_config_hash", stable_hash(proof.public_metadata.quant_cfg_id)},
            {"public_metadata_hash", public_metadata_hash},
            {"srs_manifest_hash", context.kzg.manifest.manifest_hash},
            {"proof_hash", proof_hash},
            {"public_logits_hash", trace.public_logits_hash},
            {"reference_fp32_logits_hash", trace.reference_fp32_logits_hash},
            {"reference_quant_logits_hash", trace.reference_quant_logits_hash},
            {"secure_trace_logits_hash", trace.secure_trace_logits_hash},
            {"proof_public_output_hash", trace.proof_public_output_hash},
            {"security_profile_id", context.config.security_profile_id},
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
            {"quotient_commitment_ms", format_double(metrics.quotient_commitment_ms)},
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
            {"domain_open_denominator_inverse_ms", format_double(metrics.domain_open_denominator_inverse_ms)},
            {"domain_open_denominator_inverse_reuse_ms", format_double(metrics.domain_open_denominator_inverse_reuse_ms)},
            {"domain_open_denominator_inverse_reuse_hits", std::to_string(metrics.domain_open_denominator_inverse_reuse_hits)},
            {"domain_open_denominator_inverse_cache_misses", std::to_string(metrics.domain_open_denominator_inverse_cache_misses)},
            {"domain_open_point_parallel_batches", std::to_string(metrics.domain_open_point_parallel_batches)},
            {"domain_open_point_parallel_workers", std::to_string(metrics.domain_open_point_parallel_workers)},
            {"domain_open_point_msm_worker_cap", std::to_string(metrics.domain_open_point_msm_worker_cap)},
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
        append_public_metadata(run_manifest);
        const auto manifest_validation_start = Clock::now();
        metrics.export_manifest_validation_ms = elapsed_ms(manifest_validation_start, Clock::now());
        run_manifest["export_manifest_validation_ms"] = format_double(metrics.export_manifest_validation_ms);
        util::write_json_object(export_root + "/run_manifest.json", run_manifest);

        if (!context.config.dump_trace)
        {
            return;
        }

        util::ensure_directory(export_root + "/columns");
        util::ensure_directory(export_root + "/matrices");
        for (const auto &[name, column] : trace.columns)
        {
            util::write_lines(export_root + "/columns/" + name + ".txt", vector_to_lines(column));
        }
        for (const auto &[name, matrix] : trace.matrices)
        {
            util::write_lines(export_root + "/matrices/" + name + ".txt", matrix_to_lines(matrix));
        }
    }

} // namespace gatzk::protocol
