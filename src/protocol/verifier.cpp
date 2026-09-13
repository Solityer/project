#include "gatzk/protocol/verifier.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "gatzk/algebra/domain_evaluation.hpp"
#include "gatzk/algebra/eval_backend.hpp"
#include "gatzk/data/loader.hpp"
#include "gatzk/protocol/challenges.hpp"
#include "gatzk/protocol/opening_registry.hpp"
#include "gatzk/protocol/quotients.hpp"
#include "gatzk/protocol/schema.hpp"
#include "gatzk/util/route2.hpp"
#include "gatzk/protocol/verifier_context.hpp"
#include "protocol/internal/verifier_public_polynomial.hpp"

namespace gatzk::protocol
{
    // 本文件实现 verifier 主链：
    // 它会重放 transcript、检查公开元数据契约、批量校验各域 opening，并重算 quotient 与外点关系。
    // 这里允许做求值缓存和并行检查，但任何优化都不能改变验证方程或公开对象集合。

    bool final_formal_run_accepted(
        bool verifier_accepted,
        bool reference_consistent,
        bool security_budget_passed)
    {
        return verifier_accepted && reference_consistent && security_budget_passed;
    }

    namespace
    {
        using algebra::FieldElement;
        using crypto::Commitment;
        using Clock = std::chrono::steady_clock;

        std::size_t formal_hidden_head_count(const ProtocolContext &context)
        {
            std::size_t count = 0;
            for (const auto &layer : context.config.hidden_profile)
            {
                count += layer.head_count;
            }
            return count;
        }

        bool has_formal_public_architecture(const ProtocolContext &context)
        {
            return context.config.layer_count != 0 && context.config.layer_count == context.config.hidden_profile.size() + 1 && context.config.d_in_profile.size() == context.config.hidden_profile.size() && formal_hidden_head_count(context) != 0 && context.config.K_out != 0;
        }

        bool public_statement_matches_registered_columns(const ProtocolContext &context)
        {
            const auto has_descriptor = [&](std::string_view name,
                                            std::string_view domain,
                                            std::size_t size)
            {
                const auto it = context.public_polynomial_descriptors.find(
                    std::string(name));
                return it != context.public_polynomial_descriptors.end() && it->second.name == name && it->second.basis == algebra::PolynomialBasis::Evaluation && it->second.domain == domain && it->second.size == size;
            };
            return has_descriptor("P_I", "N", context.domains.n->size) && has_descriptor("P_src", "edge", context.domains.edge->size) && has_descriptor("P_dst", "edge", context.domains.edge->size);
        }

        double elapsed_ms(const Clock::time_point &start, const Clock::time_point &end)
        {
            return std::chrono::duration<double, std::milli>(end - start).count();
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
            if (context.dataset.num_nodes == 0 || context.dataset.num_features == 0 || context.local.num_nodes != context.dataset.num_nodes || context.local.num_features != context.dataset.num_features || context.local.graph_count != context.dataset.graph_count || context.local.absolute_ids.size() != context.dataset.num_nodes || context.local.features.size() != context.dataset.num_nodes)
                return false;
            for (std::size_t row = 0; row < context.local.absolute_ids.size(); ++row)
            {
                if (context.local.absolute_ids[row] != row || context.local.features[row].size() != context.local.num_features)
                    return false;
            }
            return context.local.public_input.N_total == context.dataset.num_nodes;
        }

        bool lazy_fh_public_enabled(const ProtocolContext &context)
        {
            return complete_public_feature_source(context) && context.domains.fh != nullptr;
        }

        const std::vector<std::vector<FieldElement>> &public_feature_source(
            const ProtocolContext &context)
        {
            if (!complete_public_feature_source(context))
                throw std::runtime_error(
                    "verifier lazy feature evaluator requires complete public feature coverage");
            if (context.local.num_features == 0)
                throw std::runtime_error(
                    "verifier lazy feature evaluator requires non-empty feature columns");
            if (!context.dataset.features.empty())
            {
                if (context.dataset.features.size() != context.dataset.num_nodes)
                    throw std::runtime_error(
                        "verifier dataset feature source has an invalid row count");
                for (const auto &row : context.dataset.features)
                    if (row.size() != context.dataset.num_features)
                        throw std::runtime_error(
                            "verifier dataset feature source has an invalid row width");
                return context.dataset.features;
            }
            return context.local.features;
        }

        bool is_lazy_fh_public_label(const std::string &name)
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

        FieldElement evaluate_lazy_fh_public_poly(
            const ProtocolContext &context,
            const std::string &name,
            const FieldElement &point)
        {
            if (!lazy_fh_public_enabled(context) || !is_lazy_fh_public_label(name))
            {
                throw std::runtime_error("unsupported verifier lazy public polynomial: " + name);
            }
            const auto &domain = context.domains.fh;
            const std::size_t dataset_rows = context.dataset.num_nodes;
            const std::size_t local_rows = context.local.num_nodes;
            const std::size_t feature_count = context.local.num_features;
            if (feature_count == 0)
                throw std::runtime_error(
                    "verifier feature table has zero feature columns");
            if (dataset_rows > std::numeric_limits<std::size_t>::max() / feature_count)
                throw std::runtime_error("verifier feature table size overflows size_t");
            const std::size_t dataset_valid = dataset_rows * feature_count;
            const std::size_t local_valid = local_rows * feature_count;
            const auto &feature_matrix = public_feature_source(context);
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
                const bool full_feature_identity = full_graph_feature_identity_enabled(context);
                return evaluate_truncated_domain_polynomial(domain, local_valid, point, [&](std::size_t index)
                                                            {
                                                                const auto row = index / feature_count;
                                                                return full_feature_identity
                                                                    ? FieldElement(row)
                                                                    : FieldElement(context.local.absolute_ids[row]); });
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

        std::shared_ptr<algebra::RootOfUnityDomain> public_descriptor_domain(
            const ProtocolContext &context,
            const VerifierPublicPolynomialDescriptor &descriptor)
        {
            for (const auto &domain : {
                     context.domains.fh, context.domains.edge,
                     context.domains.in, context.domains.d,
                     context.domains.cat, context.domains.c,
                     context.domains.n})
            {
                if (domain != nullptr && domain->name == descriptor.domain)
                    return domain;
            }
            return nullptr;
        }

        FieldElement evaluate_deterministic_public_polynomial_impl(
            const ProtocolContext &context,
            const VerifierPublicPolynomialDescriptor &descriptor,
            const FieldElement &point,
            const std::vector<FieldElement> &aggregation_src_multiplicity,
            const std::vector<FieldElement> &aggregation_dst_multiplicity)
        {
            if (descriptor.basis != algebra::PolynomialBasis::Evaluation)
                throw std::runtime_error(
                    "unsupported coefficient public polynomial descriptor: " + descriptor.name);
            const auto domain = public_descriptor_domain(context, descriptor);
            if (domain == nullptr || descriptor.size != domain->size)
                throw std::runtime_error(
                    "public polynomial descriptor domain mismatch: " + descriptor.name);
            const auto checked_product = [](std::size_t left, std::size_t right,
                                            std::string_view label)
            {
                if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right)
                    throw std::runtime_error("public polynomial size overflow: " + std::string(label));
                return left * right;
            };
            const auto edge_size = context.domains.edge->size;
            const auto n_size = context.domains.n->size;
            const auto c_size = context.domains.c->size;
            const auto cat_size = context.domains.cat->size;
            const auto in_size = context.domains.in->size;
            const auto d_size = context.domains.d->size;
            const auto edge_count = context.local.edges.size();
            const auto node_count = context.local.num_nodes;
            const auto class_count = context.local.num_classes;
            const auto selector = [&](std::size_t valid_count)
            {
                if (valid_count > domain->size)
                    throw std::runtime_error("public selector exceeds its domain: " + descriptor.name);
                return evaluate_truncated_domain_polynomial(
                    domain, valid_count, point,
                    [](std::size_t)
                    { return FieldElement::one(); });
            };
            const auto edge_value = [&](auto &&value_fn, std::size_t valid_count)
            {
                if (valid_count > edge_size || domain->name != "edge")
                    throw std::runtime_error("public edge polynomial descriptor mismatch: " + descriptor.name);
                return evaluate_truncated_domain_polynomial(
                    domain, valid_count, point, std::forward<decltype(value_fn)>(value_fn));
            };
            const auto node_value = [&](auto &&value_fn, std::size_t valid_count)
            {
                if (valid_count > n_size || domain->name != "N")
                    throw std::runtime_error("public node polynomial descriptor mismatch: " + descriptor.name);
                return evaluate_truncated_domain_polynomial(
                    domain, valid_count, point, std::forward<decltype(value_fn)>(value_fn));
            };
            const auto cat_value = [&](auto &&value_fn, std::size_t valid_count)
            {
                if (valid_count > cat_size || domain->name != "cat")
                    throw std::runtime_error("public category polynomial descriptor mismatch: " + descriptor.name);
                return evaluate_truncated_domain_polynomial(
                    domain, valid_count, point, std::forward<decltype(value_fn)>(value_fn));
            };
            const auto in_value = [&](auto &&value_fn, std::size_t valid_count)
            {
                if (valid_count > in_size || domain->name != "in")
                    throw std::runtime_error("public input polynomial descriptor mismatch: " + descriptor.name);
                return evaluate_truncated_domain_polynomial(
                    domain, valid_count, point, std::forward<decltype(value_fn)>(value_fn));
            };
            const auto d_value = [&](auto &&value_fn, std::size_t valid_count)
            {
                if (valid_count > d_size || domain->name != "d")
                    throw std::runtime_error("public head polynomial descriptor mismatch: " + descriptor.name);
                return evaluate_truncated_domain_polynomial(
                    domain, valid_count, point, std::forward<decltype(value_fn)>(value_fn));
            };
            const auto c_value = [&](auto &&value_fn, std::size_t valid_count)
            {
                if (valid_count > c_size || domain->name != "C")
                    throw std::runtime_error("public class polynomial descriptor mismatch: " + descriptor.name);
                return evaluate_truncated_domain_polynomial(
                    domain, valid_count, point, std::forward<decltype(value_fn)>(value_fn));
            };

            if (descriptor.name == "P_I")
                return node_value(
                    [&](std::size_t index)
                    { return FieldElement(context.local.absolute_ids.at(index)); },
                    node_count);
            if (descriptor.name == "P_src" || descriptor.name == "P_dst")
                return edge_value(
                    [&](std::size_t index)
                    {
                        const auto &edge = context.local.edges.at(index);
                        return FieldElement(descriptor.name == "P_src" ? edge.src : edge.dst);
                    },
                    edge_count);
            if (descriptor.name == "P_edge_gid")
                return edge_value(
                    [&](std::size_t index)
                    { return FieldElement(context.local.edges.at(index).graph_id); },
                    edge_count);
            if (descriptor.name == "P_Q_new_edge")
                return edge_value(
                    [&](std::size_t index)
                    { return FieldElement(index == 0 || context.local.edges[index].dst != context.local.edges[index - 1].dst); },
                    edge_count);
            if (descriptor.name == "P_Q_end_edge")
                return edge_value(
                    [&](std::size_t index)
                    { return FieldElement(index + 1 == edge_count || context.local.edges[index].dst != context.local.edges[index + 1].dst); },
                    edge_count);
            if (descriptor.name == "P_Q_graph_new_edge")
                return edge_value(
                    [&](std::size_t index)
                    { return FieldElement(index == 0 || context.local.edges[index].graph_id != context.local.edges[index - 1].graph_id); },
                    edge_count);
            if (descriptor.name == "P_Q_graph_end_edge")
                return edge_value(
                    [&](std::size_t index)
                    { return FieldElement(index + 1 == edge_count || context.local.edges[index].graph_id != context.local.edges[index + 1].graph_id); },
                    edge_count);
            if (descriptor.name == "P_Q_grp_new_edge" || descriptor.name == "P_Q_grp_end_edge")
                return edge_value(
                    [&](std::size_t index)
                    {
                        const bool graph_boundary = descriptor.name == "P_Q_grp_new_edge"
                                                        ? index == 0 || context.local.edges[index].graph_id != context.local.edges[index - 1].graph_id
                                                        : index + 1 == edge_count || context.local.edges[index].graph_id != context.local.edges[index + 1].graph_id;
                        const bool destination_boundary = descriptor.name == "P_Q_grp_new_edge"
                                                              ? index == 0 || context.local.edges[index].dst != context.local.edges[index - 1].dst
                                                              : index + 1 == edge_count || context.local.edges[index].dst != context.local.edges[index + 1].dst;
                        return FieldElement(graph_boundary || destination_boundary);
                    },
                    edge_count);
            if (descriptor.name == "P_Q_edge_valid" || descriptor.name == "P_Q_qry_src" || descriptor.name == "P_Q_qry_dst")
                return edge_value([](std::size_t)
                                  { return FieldElement::one(); },
                                  edge_count);
            if (descriptor.name == "P_node_gid")
                return node_value(
                    [&](std::size_t index)
                    {
                        const auto it = std::upper_bound(
                            context.local.node_ptr.begin(), context.local.node_ptr.end(), index);
                        if (it == context.local.node_ptr.begin())
                            throw std::runtime_error("node gid column has invalid node_ptr");
                        return FieldElement(static_cast<std::size_t>(
                            std::distance(context.local.node_ptr.begin(), it) - 1));
                    },
                    node_count);
            if (descriptor.name == "P_Q_N")
                return selector(node_count);
            if (descriptor.name == "P_AggBind_index_N")
                return node_value(
                    [](std::size_t index)
                    { return FieldElement(index); },
                    node_count);
            if (descriptor.name == "P_AggBind_m_src" || descriptor.name == "P_AggBind_m_dst")
            {
                const auto &multiplicity = descriptor.name == "P_AggBind_m_src"
                                               ? aggregation_src_multiplicity
                                               : aggregation_dst_multiplicity;
                if (multiplicity.size() != n_size)
                    throw std::runtime_error(
                        "aggregation multiplicity cache has invalid size");
                return node_value(
                    [&](std::size_t index)
                    { return multiplicity.at(index); },
                    node_count);
            }
            if (descriptor.name == output_q_cat_selector_label())
                return selector(context.model.output_layer.input_dim);
            if (descriptor.name == "P_Q_C_valid")
                return c_value([](std::size_t)
                               { return FieldElement::one(); },
                               class_count);
            if (descriptor.name == "P_BiasIndex_C")
                return c_value(
                    [](std::size_t index)
                    { return FieldElement(index); },
                    class_count);
            if (descriptor.name == "P_BiasMultiplicity_C")
                return c_value(
                    [&](std::size_t)
                    { return FieldElement(node_count); },
                    class_count);
            if (descriptor.name == "P_BiasIndex_edge")
                return edge_value(
                    [&](std::size_t index)
                    { return FieldElement(index % class_count); },
                    checked_product(node_count, class_count, descriptor.name));

            for (std::size_t layer = 0; layer < context.model.hidden_layers.size(); ++layer)
            {
                const auto &shape = context.model.hidden_layers.at(layer);
                if (descriptor.name == hidden_q_in_selector_label(layer))
                    return in_value([](std::size_t)
                                    { return FieldElement::one(); },
                                    shape.input_dim);
                if (descriptor.name == hidden_q_d_h_selector_label(layer))
                    return d_value([](std::size_t)
                                   { return FieldElement::one(); },
                                   shape.shape.head_dim);
                if (descriptor.name == hidden_q_cat_selector_label(layer))
                    return cat_value([](std::size_t)
                                     { return FieldElement::one(); },
                                     checked_product(shape.shape.head_count,
                                                     shape.shape.head_dim,
                                                     descriptor.name));
                if (descriptor.name == hidden_q_nd_h_selector_label(layer))
                    return edge_value([](std::size_t)
                                      { return FieldElement::one(); },
                                      checked_product(node_count,
                                                      shape.shape.head_dim,
                                                      descriptor.name));
            }
            for (std::size_t head = 0; head < context.model.output_layer.heads.size(); ++head)
            {
                if (descriptor.name == "P_Q_div_out" + std::to_string(head) || descriptor.name == "P_Q_div_avg")
                    return edge_value([](std::size_t)
                                      { return FieldElement::one(); },
                                      checked_product(node_count,
                                                      class_count,
                                                      descriptor.name));
                const auto chunks = coefficient_chunk_layout(
                    class_count + 3, edge_size);
                for (const auto &chunk : chunks)
                {
                    for (const auto region : {"active", "semantic", "gap", "tail"})
                    {
                        if (descriptor.name != output_bias_basis_region_selector_label(
                                                   head, chunk.chunk_index, region))
                            continue;
                        return edge_value(
                            [&](std::size_t index)
                            {
                                const auto global_index = chunk.offset + index;
                                if (std::string_view(region) == "active")
                                    return FieldElement::one();
                                if (std::string_view(region) == "semantic")
                                    return FieldElement(global_index < class_count);
                                if (std::string_view(region) == "tail")
                                    return FieldElement(global_index >= class_count);
                                return FieldElement::zero();
                            },
                            chunk.length);
                    }
                }
            }
            if (descriptor.name == "P_Q_tbl_L" || descriptor.name == "P_Q_qry_L" || descriptor.name == "P_Q_tbl_R" || descriptor.name == "P_Q_qry_R" || descriptor.name == "P_Q_tbl_exp" || descriptor.name == "P_Q_qry_exp" || descriptor.name == "P_Q_tbl_ELU")
            {
                const bool table = descriptor.name.find("tbl") != std::string::npos;
                std::size_t count = edge_count;
                if (descriptor.name.ends_with("_L"))
                    count = table ? context.tables.lrelu.size() : edge_count;
                else if (descriptor.name.ends_with("_R"))
                    count = table ? context.tables.range.size() : edge_count;
                else if (descriptor.name.ends_with("_exp"))
                    count = table ? context.tables.exp.size() : edge_count;
                else if (descriptor.name.ends_with("_ELU"))
                    count = context.tables.elu.size();
                return edge_value([](std::size_t)
                                  { return FieldElement::one(); },
                                  count);
            }
            if (descriptor.name == "P_T_L_x" || descriptor.name == "P_T_L_y" || descriptor.name == "P_T_ELU_x" || descriptor.name == "P_T_ELU_y" || descriptor.name == "P_T_exp_x" || descriptor.name == "P_T_exp_y" || descriptor.name == "P_T_range")
            {
                std::size_t count = 0;
                if (descriptor.name == "P_T_L_x" || descriptor.name == "P_T_L_y")
                    count = context.tables.lrelu.size();
                else if (descriptor.name == "P_T_ELU_x" || descriptor.name == "P_T_ELU_y")
                    count = context.tables.elu.size();
                else if (descriptor.name == "P_T_exp_x" || descriptor.name == "P_T_exp_y")
                    count = context.tables.exp.size();
                else
                    count = context.tables.range.size();
                return edge_value(
                    [&](std::size_t index)
                    {
                        if (descriptor.name == "P_T_L_x")
                            return context.tables.lrelu.at(index).first;
                        if (descriptor.name == "P_T_L_y")
                            return context.tables.lrelu.at(index).second;
                        if (descriptor.name == "P_T_ELU_x")
                            return context.tables.elu.at(index).first;
                        if (descriptor.name == "P_T_ELU_y")
                            return context.tables.elu.at(index).second;
                        if (descriptor.name == "P_T_exp_x")
                            return context.tables.exp.at(index).first;
                        if (descriptor.name == "P_T_exp_y")
                            return context.tables.exp.at(index).second;
                        return context.tables.range.at(index);
                    },
                    count);
            }
            if (is_lazy_fh_public_label(descriptor.name))
                return evaluate_lazy_fh_public_poly(context, descriptor.name, point);
            throw std::runtime_error(
                "unsupported deterministic public polynomial descriptor: " + descriptor.name);
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
                  metrics_(metrics)
            {
                aggregation_multiplicity_ =
                    internal::build_aggregation_multiplicity_cache(
                        context_.local,
                        context_.domains.n == nullptr
                            ? 0
                            : context_.domains.n->size);
            }

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
                        // The largest full-graph FH working set also drives
                        // verifier prefetches for its large edge domain.
                        // Scanning those pageable public columns on CPU avoids
                        // the unified-memory CUDA zero-copy penalty while
                        // leaving every other backend caller on automatic
                        const auto rotated_values =
                            backend
                                ->evaluate_with_native_weight_rotations(
                                    *labels,
                                    weight_entry.native_weights,
                                    rotations,
                                    algebra::
                                        EvaluationExecutionPolicy::
                                            Automatic);

                        if (lazy_fh_public_enabled(context_) &&
                            domain->size >=
                                (1ULL << 20) &&
                            metrics_ != nullptr &&
                            !cpu_large_fh_route_noted_)
                        {
                            if (!metrics_->notes.empty())
                            {
                                metrics_->notes +=
                                    "; ";
                            }

                            metrics_->notes +=
                                "verifier_public_eval_route="
                                "automatic_gpu_direct_columns";

                            cpu_large_fh_route_noted_ =
                                true;
                        }
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
                if (const auto it = context_.public_polynomial_descriptors.find(name);
                    it != context_.public_polynomial_descriptors.end())
                {
                    if (lazy_fh_public_enabled(context_) && is_lazy_fh_public_label(name))
                    {
                        batch_evaluate_lazy_fh_labels(point);
                        if (const auto it2 = value_cache_.find(cache_key);
                            it2 != value_cache_.end())
                            return it2->second;
                        value = evaluate_lazy_fh_public_poly(context_, name, point);
                    }
                    else
                    {
                        value = internal::evaluate_deterministic_public_polynomial(
                            context_, it->second, point, aggregation_multiplicity_);
                    }
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
                if (feature_count == 0)
                    throw std::runtime_error(
                        "verifier feature table has zero feature columns");
                if (dataset_rows > std::numeric_limits<std::size_t>::max() / feature_count)
                    throw std::runtime_error("verifier feature table size overflows size_t");
                const std::size_t dataset_valid = dataset_rows * feature_count;
                const bool full_feature_identity = full_graph_feature_identity_enabled(context_);
                const auto &feature_matrix = public_feature_source(context_);

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
            internal::AggregationMultiplicityCache aggregation_multiplicity_;
            RunMetrics *metrics_ = nullptr;
            bool cpu_large_fh_route_noted_ = false;
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
                [](const auto &entry)
                { return entry.first == "P_Y"; });
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
                return a.relation_id == b.relation_id && a.canonical_object == b.canonical_object && a.view_object == b.view_object && a.canonical_domain == b.canonical_domain && a.view_domain == b.view_domain && a.canonical_length == b.canonical_length && a.view_length == b.view_length && a.index_map_id == b.index_map_id && a.canonical_selector_id == b.canonical_selector_id && a.query_selector_id == b.query_selector_id && a.multiplicity_id == b.multiplicity_id && a.eta_challenge_label == b.eta_challenge_label && a.beta_challenge_label == b.beta_challenge_label && a.commitment_order == b.commitment_order;
            };
            auto shift_relation_matches = [](
                                              const KZGShiftRelationMeta &a,
                                              const KZGShiftRelationMeta &b)
            {
                return a.relation_id == b.relation_id && a.source_linear_maps == b.source_linear_maps && a.source_domains == b.source_domains && a.canonical_points == b.canonical_points && a.lambda0 == b.lambda0 && a.deltas == b.deltas && a.source_value_kinds == b.source_value_kinds && a.challenge_label == b.challenge_label && a.exclusion_degree == b.exclusion_degree && a.max_difference_degree == b.max_difference_degree && a.configured_degrees == b.configured_degrees && a.shift_domain_sizes == b.shift_domain_sizes && a.opening_claim_order == b.opening_claim_order;
            };
            const auto indexed_equal = lhs.indexed_views.size() == rhs.indexed_views.size() && std::equal(
                                                                                                   lhs.indexed_views.begin(), lhs.indexed_views.end(),
                                                                                                   rhs.indexed_views.begin(), indexed_view_matches);
            const auto shift_equal = lhs.shift_relations.size() == rhs.shift_relations.size() && std::equal(
                                                                                                     lhs.shift_relations.begin(), lhs.shift_relations.end(),
                                                                                                     rhs.shift_relations.begin(), shift_relation_matches);
            return lhs.relation_id == rhs.relation_id && lhs.object_kind == rhs.object_kind && lhs.layer_id == rhs.layer_id && lhs.head_id == rhs.head_id && lhs.source_objects == rhs.source_objects && lhs.target_objects == rhs.target_objects && lhs.work_domains == rhs.work_domains && lhs.real_lengths == rhs.real_lengths && lhs.quotient_relations == rhs.quotient_relations && lhs.canonical_matrices == rhs.canonical_matrices && lhs.aggregation_relations == rhs.aggregation_relations && lhs.route_terminal_relations == rhs.route_terminal_relations && indexed_equal && lhs.coefficient_basis_relations == rhs.coefficient_basis_relations && shift_equal;
        }

        bool validate_pi_bind_model_reblinds(
            const ProtocolContext &context,
            const Proof &proof,
            const std::unordered_map<std::string, crypto::Commitment> &top_level_dynamic,
            std::unordered_map<std::string, crypto::Commitment> *combined)
        {
            if (combined == nullptr)
                return false;
            *combined = top_level_dynamic;
            const auto expected_meta = build_binding_relation_metadata(context);
            if (proof.pi_bind.subproofs.size() != expected_meta.size())
                return false;

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
                    const auto head = subproof.relation_meta.head_id.value();
                    const auto width = context.config.hidden_profile.at(layer).head_dim;
                    for (std::size_t column = 0; column < width; ++column)
                        expected_models.push_back({hidden_weight_column_label(layer, head, column),
                                                   context.domains.in->name});
                    expected_models.push_back({hidden_src_label(layer, head), context.domains.d->name});
                    expected_models.push_back({hidden_dst_label(layer, head), context.domains.d->name});
                }
                else if (subproof.relation_meta.object_kind == "output")
                {
                    const auto head = subproof.relation_meta.head_id.value();
                    for (std::size_t column = 0; column < context.local.num_classes; ++column)
                        expected_models.push_back({output_weight_column_label(head, column),
                                                   context.domains.cat->name});
                    expected_models.push_back({output_src_label(head), context.domains.c->name});
                    expected_models.push_back({output_dst_label(head), context.domains.c->name});
                    expected_models.push_back({output_bias_label(head), context.domains.c->name});
                }
                if (subproof.model_reblinds.size() != expected_models.size())
                    return false;
                for (std::size_t model_index = 0;
                     model_index < expected_models.size(); ++model_index)
                {
                    const auto &[expected_anchor, expected_domain] = expected_models[model_index];
                    const auto &entry = subproof.model_reblinds[model_index];
                    const auto anchor_it = context.static_commitments.find(expected_anchor);
                    if (entry.object_id != expected_anchor || entry.domain != expected_domain || entry.vk_model_anchor.object_id != expected_anchor || entry.proof_local_commitment.name != model_proof_local_label(expected_anchor) || entry.blinder_commitment.name != model_reblind_witness_label(expected_anchor) || anchor_it == context.static_commitments.end())
                    {
                        return false;
                    }
                    if (entry.blinder_commitment.point == crypto::g1_zero())
                    {
                        return false;
                    }
                    if (!model_object_ids.insert(expected_anchor).second)
                    {
                        return false;
                    }
                    if (!crypto::KZG::verify_domain_divisibility(
                            entry.proof_local_commitment,
                            anchor_it->second,
                            entry.blinder_commitment,
                            expected_domain,
                            context.kzg.vk))
                    {
                        return false;
                    }
                    if (!combined->emplace(
                                     entry.proof_local_commitment.name,
                                     entry.proof_local_commitment)
                             .second)
                    {
                        return false;
                    }
                    if (!combined->emplace(
                                     entry.blinder_commitment.name,
                                     entry.blinder_commitment)
                             .second)
                    {
                        return false;
                    }
                }
                std::vector<std::string> expected_canonical_references;
                std::vector<std::string> expected_canonical_commitments;
                std::vector<std::tuple<
                    std::string, std::string, std::size_t,
                    PayloadValueKind>>
                    expected_canonical_evaluations;
                for (const auto &matrix :
                     subproof.relation_meta.canonical_matrices)
                {
                    expected_canonical_references.push_back(
                        matrix.matrix_object);
                    for (const auto &column : matrix.column_objects)
                    {
                        expected_canonical_commitments.push_back(column);
                        expected_canonical_evaluations.push_back({column, "N", 2,
                                                                  PayloadValueKind::MaskedPrivateValue});
                    }
                    expected_canonical_commitments.push_back(
                        matrix.compressed_object);
                    expected_canonical_evaluations.push_back({matrix.compressed_object, "N", 2,
                                                              PayloadValueKind::MaskedPrivateValue});
                }
                const auto canonical_commitments_match = [&]
                {
                    return subproof.view_commitments.size() == expected_canonical_commitments.size() && std::equal(
                                                                                                            subproof.view_commitments.begin(),
                                                                                                            subproof.view_commitments.end(),
                                                                                                            expected_canonical_commitments.begin(),
                                                                                                            [](const auto &actual,
                                                                                                               const std::string &expected)
                                                                                                            { return actual.first == expected; });
                };
                std::vector<std::string> expected_aggregation_commitments;
                std::vector<std::tuple<
                    std::string, std::string, std::size_t,
                    PayloadValueKind>>
                    expected_aggregation_relation_evaluations;
                std::vector<std::string> expected_route_commitments;
                for (const auto &route :
                     subproof.relation_meta.route_terminal_relations)
                {
                    expected_route_commitments.push_back(
                        route.shifted_accumulator_object);
                    expected_aggregation_relation_evaluations.push_back({route.shifted_accumulator_object + ":u_rel",
                                                                         "relation", 1,
                                                                         PayloadValueKind::MaskedPrivateValue});
                }
                std::vector<std::tuple<
                    std::string, std::string, std::size_t,
                    PayloadValueKind>>
                    expected_aggregation_domain_evaluations;
                for (const auto &aggregation :
                     subproof.relation_meta.aggregation_relations)
                {
                    expected_aggregation_commitments.insert(
                        expected_aggregation_commitments.end(),
                        aggregation.commitment_order.begin(),
                        aggregation.commitment_order.end());
                    for (const auto *route : {
                             &aggregation.source_route,
                             &aggregation.target_route})
                        expected_aggregation_relation_evaluations.push_back({route->shifted_accumulator_object + ":u_rel",
                                                                             "relation", 1,
                                                                             PayloadValueKind::MaskedPrivateValue});
                    for (const auto *object : {
                             &aggregation.source_node_object,
                             &aggregation.target_node_object})
                        expected_aggregation_domain_evaluations.push_back({*object, aggregation.node_domain, 2,
                                                                           PayloadValueKind::MaskedPrivateValue});
                    for (const auto *route : {
                             &aggregation.source_route,
                             &aggregation.target_route})
                        for (const auto *object : {
                                 &route->table_object,
                                 &route->node_inverse_object,
                                 &route->node_accumulator_object})
                            expected_aggregation_domain_evaluations.push_back({*object, aggregation.node_domain, 2,
                                                                               PayloadValueKind::MaskedPrivateValue});
                    for (const auto *object : {
                             &aggregation.source_edge_object,
                             &aggregation.target_edge_object,
                             &aggregation.product_object,
                             &aggregation.psq_object})
                        expected_aggregation_domain_evaluations.push_back({*object, aggregation.edge_domain, 2,
                                                                           PayloadValueKind::MaskedPrivateValue});
                    for (const auto *route : {
                             &aggregation.source_route,
                             &aggregation.target_route})
                        for (const auto *object : {
                                 &route->query_object,
                                 &route->edge_inverse_object,
                                 &route->edge_accumulator_object})
                            expected_aggregation_domain_evaluations.push_back({*object, aggregation.edge_domain, 2,
                                                                               PayloadValueKind::MaskedPrivateValue});
                }
                const bool aggregation_commitments_match =
                    subproof.aggregation_commitments.size() == expected_aggregation_commitments.size() && std::equal(
                                                                                                              subproof.aggregation_commitments.begin(),
                                                                                                              subproof.aggregation_commitments.end(),
                                                                                                              expected_aggregation_commitments.begin(),
                                                                                                              [](const auto &actual, const std::string &expected)
                                                                                                              { return actual.first == expected; });
                if (subproof.relation_meta.object_kind == "output")
                {
                    const auto head = subproof.relation_meta.head_id.value();
                    const auto &view_meta = subproof.relation_meta.indexed_views.front();
                    const auto *terminal_meta_ptr = find_unique_shift_relation_by_kind(
                        subproof.relation_meta, "output-bias-terminal");
                    if (terminal_meta_ptr == nullptr)
                        return false;
                    const auto &terminal_meta = *terminal_meta_ptr;
                    const bool same_point = terminal_meta.configured_degrees.empty();
                    if (subproof.relation_meta
                            .coefficient_basis_relations.size() != 1)
                        return false;
                    const auto &basis = subproof.relation_meta
                                            .coefficient_basis_relations.front();
                    auto expected_view_references =
                        expected_canonical_references;
                    expected_view_references.insert(
                        expected_view_references.end(),
                        {
                            view_meta.canonical_object,
                            view_meta.view_object,
                            "public:" + view_meta.canonical_selector_id,
                            "public:" + view_meta.query_selector_id,
                        });
                    std::vector<std::tuple<
                        std::string, std::string, std::size_t, PayloadValueKind>>
                        expected_evaluations = {
                            {output_bias_view_label(head, "m_can") + ":public_bind",
                             "relation", 1, PayloadValueKind::PublicValue},
                            {model_proof_local_label(output_bias_label(head)),
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
                    expected_evaluations.insert(
                        expected_evaluations.begin(),
                        expected_aggregation_relation_evaluations.begin(),
                        expected_aggregation_relation_evaluations.end());
                    if (!same_point)
                        expected_evaluations.insert(
                            expected_evaluations.begin() + static_cast<std::ptrdiff_t>(
                                                               expected_aggregation_relation_evaluations
                                                                   .size()),
                            {output_bias_view_label(head, "R_shifted") + ":u_rel",
                             "relation", 1, PayloadValueKind::MaskedPrivateValue});
                    auto basis_eval_position = expected_evaluations.begin() + static_cast<std::ptrdiff_t>(
                                                                                  expected_aggregation_relation_evaluations.size() + (same_point ? 1 : 2));
                    for (const auto &chunk : basis.chunks)
                        basis_eval_position = expected_evaluations.insert(
                                                  basis_eval_position,
                                                  {chunk.shifted_accumulator_object + ":u_rel",
                                                   "relation", 1,
                                                   PayloadValueKind::MaskedPrivateValue}) +
                                              1;
                    for (const auto &chunk : basis.chunks)
                    {
                        expected_evaluations.push_back({chunk.object_id, "edge", 2,
                                                        PayloadValueKind::MaskedPrivateValue});
                        expected_evaluations.push_back({chunk.accumulator_object, "edge", 2,
                                                        PayloadValueKind::MaskedPrivateValue});
                    }
                    expected_evaluations.insert(
                        expected_evaluations.end(),
                        expected_canonical_evaluations.begin(),
                        expected_canonical_evaluations.end());
                    expected_evaluations.insert(
                        expected_evaluations.end(),
                        expected_aggregation_domain_evaluations.begin(),
                        expected_aggregation_domain_evaluations.end());
                    const bool evaluations_match =
                        subproof.evaluations.size() == expected_evaluations.size() && std::equal(
                                                                                          subproof.evaluations.begin(),
                                                                                          subproof.evaluations.end(),
                                                                                          expected_evaluations.begin(),
                                                                                          [](const BindingEvaluation &evaluation, const auto &expected)
                                                                                          {
                                                                                              return evaluation.object_id == std::get<0>(expected) && evaluation.domain == std::get<1>(expected) && evaluation.values.size() == std::get<2>(expected) && evaluation.kind == std::get<3>(expected);
                                                                                          });
                    bool domain_evaluations_match = true;
                    for (const auto &evaluation : subproof.evaluations)
                    {
                        if (evaluation.domain == "relation")
                            continue;
                        const auto bundle_it = std::find_if(
                            proof.domain_openings.begin(),
                            proof.domain_openings.end(),
                            [&](const auto &entry)
                            { return entry.first == evaluation.domain; });
                        if (bundle_it == proof.domain_openings.end())
                        {
                            domain_evaluations_match = false;
                            break;
                        }
                        const auto &bundle = bundle_it->second;
                        const auto value_it = std::find_if(
                            bundle.values.begin(), bundle.values.end(),
                            [&](const auto &entry)
                            { return entry.first == evaluation.object_id; });
                        if (value_it == bundle.values.end())
                        {
                            domain_evaluations_match = false;
                            break;
                        }
                        const auto value_index = static_cast<std::size_t>(
                            std::distance(bundle.values.begin(), value_it));
                        if (value_index >= bundle.value_kinds.size() || value_it->second != evaluation.values || bundle.value_kinds[value_index] != evaluation.kind)
                        {
                            domain_evaluations_match = false;
                            break;
                        }
                    }
                    if (subproof.view_references.size() != expected_view_references.size() || !std::equal(subproof.view_references.begin(), subproof.view_references.end(), expected_view_references.begin(), [](const MetadataReference &reference, const std::string &expected)
                                                                                                          { return reference.object_id == expected; }) ||
                        !canonical_commitments_match() || !aggregation_commitments_match || !subproof.crpc_commitments.empty() || subproof.basis_commitments.size() != basis.commitment_order.size() || !std::equal(subproof.basis_commitments.begin(), subproof.basis_commitments.end(), basis.commitment_order.begin(), [](const auto &entry, const std::string &expected)
                                                                                                                                                                                                                    { return entry.first == expected; }) ||
                        subproof.view_logup_commitments.size() != view_meta.commitment_order.size() || !std::equal(subproof.view_logup_commitments.begin(), subproof.view_logup_commitments.end(), view_meta.commitment_order.begin(), [](const auto &entry, const std::string &expected)
                                                                                                                   { return entry.first == expected; }) ||
                        subproof.relation_commitments.size() != expected_route_commitments.size() + basis.chunks.size() + (same_point ? 0 : 1) || !std::equal(subproof.relation_commitments.begin(), subproof.relation_commitments.begin() + static_cast<std::ptrdiff_t>(expected_route_commitments.size()), expected_route_commitments.begin(), [](const auto &entry, const std::string &expected)
                                                                                                                                                              { return entry.first == expected; }) ||
                        (!same_point && subproof.relation_commitments[expected_route_commitments.size()].first != output_bias_view_label(head, "R_shifted")) || !std::equal(subproof.relation_commitments.end() - static_cast<std::ptrdiff_t>(basis.chunks.size()), subproof.relation_commitments.end(), basis.chunks.begin(), [](const auto &actual, const auto &chunk)
                                                                                                                                                                            { return actual.first == chunk.shifted_accumulator_object; }) ||
                        !evaluations_match || !domain_evaluations_match)
                    {
                        return false;
                    }
                }
                else
                {
                    auto expected_evaluations =
                        expected_aggregation_relation_evaluations;
                    expected_evaluations.insert(
                        expected_evaluations.end(),
                        expected_canonical_evaluations.begin(),
                        expected_canonical_evaluations.end());
                    expected_evaluations.insert(
                        expected_evaluations.end(),
                        expected_aggregation_domain_evaluations.begin(),
                        expected_aggregation_domain_evaluations.end());
                    const bool references_match =
                        subproof.view_references.size() == expected_canonical_references.size() && std::equal(
                                                                                                       subproof.view_references.begin(),
                                                                                                       subproof.view_references.end(),
                                                                                                       expected_canonical_references.begin(),
                                                                                                       [](const MetadataReference &actual,
                                                                                                          const std::string &expected)
                                                                                                       { return actual.object_id == expected; });
                    const bool evaluations_match =
                        subproof.evaluations.size() == expected_evaluations.size() && std::equal(
                                                                                          subproof.evaluations.begin(),
                                                                                          subproof.evaluations.end(),
                                                                                          expected_evaluations.begin(),
                                                                                          [](const BindingEvaluation &actual,
                                                                                             const auto &expected)
                                                                                          {
                                                                                              return actual.object_id == std::get<0>(expected) && actual.domain == std::get<1>(expected) && actual.values.size() == std::get<2>(expected) && actual.kind == std::get<3>(expected);
                                                                                          });
                    bool domain_values_match = true;
                    for (const auto &evaluation : subproof.evaluations)
                    {
                        if (evaluation.domain == "relation")
                            continue;
                        const auto bundle_it = std::find_if(
                            proof.domain_openings.begin(),
                            proof.domain_openings.end(),
                            [&](const auto &entry)
                            { return entry.first == evaluation.domain; });
                        if (bundle_it == proof.domain_openings.end())
                        {
                            domain_values_match = false;
                            break;
                        }
                        const auto value_it = std::find_if(
                            bundle_it->second.values.begin(),
                            bundle_it->second.values.end(),
                            [&](const auto &entry)
                            { return entry.first == evaluation.object_id; });
                        if (value_it == bundle_it->second.values.end() || value_it->second != evaluation.values)
                        {
                            domain_values_match = false;
                            break;
                        }
                    }
                    std::vector<std::string> expected_openings;
                    const auto append_unique_opening = [&](std::string value)
                    {
                        if (std::find(
                                expected_openings.begin(),
                                expected_openings.end(), value) == expected_openings.end())
                            expected_openings.push_back(std::move(value));
                    };
                    for (const auto &route :
                         subproof.relation_meta.route_terminal_relations)
                    {
                        const auto y_rel = proof.challenges.at(
                            route.challenge_label);
                        append_unique_opening(
                            opening_group_reference(
                                context, proof.challenges, y_rel));
                        append_unique_opening(
                            opening_group_reference(
                                context,
                                proof.challenges,
                                y_rel + route.terminal_delta));
                        append_unique_opening(
                            opening_group_reference(
                                context,
                                proof.challenges,
                                route.node_terminal_point));
                    }
                    for (const auto &aggregation :
                         subproof.relation_meta.aggregation_relations)
                    {
                        for (const auto *route : {
                                 &aggregation.source_route,
                                 &aggregation.target_route})
                        {
                            const auto y_rel = proof.challenges.at(
                                route->terminal_challenge_label);
                            append_unique_opening(
                                opening_group_reference(
                                    context, proof.challenges, y_rel));
                            append_unique_opening(
                                opening_group_reference(
                                    context,
                                    proof.challenges,
                                    y_rel + route->terminal_delta));
                            append_unique_opening(
                                opening_group_reference(
                                    context,
                                    proof.challenges,
                                    route->node_terminal_point));
                        }
                    }
                    if (!subproof.relation_meta.canonical_matrices.empty())
                    {
                        const auto point = proof.challenges.at("z_N");
                        append_unique_opening(
                            "Open_dom:N:" + point.to_string());
                        append_unique_opening(
                            "Open_dom:N:" + (point * context.domains.n->omega).to_string());
                    }
                    if (!subproof.relation_meta.aggregation_relations.empty())
                    {
                        const auto point = proof.challenges.at("z_edge");
                        append_unique_opening(
                            "Open_dom:edge:" + point.to_string());
                        append_unique_opening(
                            "Open_dom:edge:" + (point * context.domains.edge->omega).to_string());
                    }
                    const bool opening_references_match =
                        subproof.opening_references.size() == expected_openings.size() && std::equal(
                                                                                              subproof.opening_references.begin(),
                                                                                              subproof.opening_references.end(),
                                                                                              expected_openings.begin(),
                                                                                              [](const MetadataReference &actual,
                                                                                                 const std::string &expected)
                                                                                              { return actual.object_id == expected; });
                    if (!references_match || !canonical_commitments_match() || !evaluations_match || !domain_values_match || !opening_references_match || !subproof.relation_meta.indexed_views.empty() || !subproof.relation_meta.coefficient_basis_relations.empty() || !subproof.relation_meta.shift_relations.empty() || !subproof.view_logup_commitments.empty() || !aggregation_commitments_match || !subproof.basis_commitments.empty() || !subproof.crpc_commitments.empty() || subproof.relation_commitments.size() != expected_route_commitments.size() || !std::equal(subproof.relation_commitments.begin(), subproof.relation_commitments.end(), expected_route_commitments.begin(), [](const auto &entry, const std::string &expected)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 { return entry.first == expected; }))
                    {
                        return false;
                    }
                }
                for (const auto *group : {
                         &subproof.view_commitments,
                         &subproof.view_logup_commitments,
                         &subproof.aggregation_commitments,
                         &subproof.basis_commitments,
                         &subproof.crpc_commitments,
                         &subproof.relation_commitments})
                {
                    for (const auto &[name, commitment] : *group)
                    {
                        if (name != commitment.name || !combined->emplace(name, commitment).second)
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
            struct DivisibilityCheck
            {
                crypto::Commitment proof_local;
                crypto::Commitment anchor;
                crypto::Commitment blinder;
                std::string domain_name;
            };
            std::vector<DivisibilityCheck> checks;
            checks.reserve((formal_hidden_head_count(context) + context.config.K_out) * 3);
            auto required_dynamic = [&](const std::string &label) -> const crypto::Commitment *
            {
                const auto it = dynamic.find(label);
                return it == dynamic.end() ? nullptr : &it->second;
            };
            auto append_relation = [&](
                                       const std::string &source_label,
                                       const crypto::Commitment &anchor,
                                       const std::string &blinder_label,
                                       const std::string &domain_name)
            {
                const auto *source = required_dynamic(source_label);
                const auto *blinder = required_dynamic(blinder_label);
                if (source == nullptr || blinder == nullptr)
                {
                    return false;
                }
                checks.push_back(DivisibilityCheck{
                    .proof_local = *source,
                    .anchor = anchor,
                    .blinder = *blinder,
                    .domain_name = domain_name,
                });
                return true;
            };
            auto folded_anchor = [&](
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
                    if (it == dynamic.end())
                        return std::nullopt;
                    out.point = crypto::g1_add(
                        out.point, crypto::g1_mul(it->second.point, power));
                    power *= folding;
                }
                return out;
            };

            for (std::size_t layer = 0;
                 layer < context.model.hidden_layers.size(); ++layer)
                for (std::size_t head = 0;
                     head < context.model.hidden_layers[layer].heads.size(); ++head)
                {
                    std::vector<std::string> weight_columns;
                    for (std::size_t column = 0;; ++column)
                    {
                        const auto label = hidden_weight_column_label(layer, head, column);
                        if (!context.static_commitments.contains(label))
                            break;
                        weight_columns.push_back(label);
                    }
                    const auto challenge_it = challenges.find(
                        hidden_challenge_name("xi_h", layer, head));
                    if (weight_columns.empty() || challenge_it == challenges.end())
                        return false;
                    const auto anchor = folded_anchor(
                        "folded_hidden_weight_proof_l" + std::to_string(layer) + "_h" + std::to_string(head),
                        weight_columns, challenge_it->second);
                    const auto src_anchor = required_dynamic(
                        model_proof_local_label(hidden_src_label(layer, head)));
                    const auto dst_anchor = required_dynamic(
                        model_proof_local_label(hidden_dst_label(layer, head)));
                    const auto prefix = hidden_head_prefix(layer, head) + "_";
                    if (!anchor.has_value() || src_anchor == nullptr || dst_anchor == nullptr || !append_relation(prefix + "b_proj", *anchor, hidden_model_binding_blinder_label(layer, head, "proj"), context.domains.in->name) || !append_relation(prefix + "b_src", *src_anchor, hidden_model_binding_blinder_label(layer, head, "src"), context.domains.d->name) || !append_relation(prefix + "b_dst", *dst_anchor, hidden_model_binding_blinder_label(layer, head, "dst"), context.domains.d->name))
                        return false;
                }
            for (std::size_t head = 0; head < context.config.K_out; ++head)
            {
                std::vector<std::string> weight_columns;
                weight_columns.reserve(context.local.num_classes);
                for (std::size_t column = 0; column < context.local.num_classes; ++column)
                {
                    weight_columns.push_back(output_weight_column_label(head, column));
                }
                const auto challenge_it =
                    challenges.find("xi_out");
                if (challenge_it == challenges.end())
                    return false;
                const auto anchor = folded_anchor(
                    "folded_output_weight_proof_" + std::to_string(head),
                    weight_columns,
                    challenge_it->second);
                const auto src_anchor = required_dynamic(
                    model_proof_local_label(output_src_label(head)));
                const auto dst_anchor = required_dynamic(
                    model_proof_local_label(output_dst_label(head)));
                const auto prefix = output_head_prefix(head) + "_";
                if (!anchor.has_value() || src_anchor == nullptr || dst_anchor == nullptr || !append_relation(prefix + "b_proj", *anchor, output_model_binding_blinder_label(head, "proj"), context.domains.cat->name) || !append_relation(prefix + "b_src", *src_anchor, output_model_binding_blinder_label(head, "src"), context.domains.c->name) || !append_relation(prefix + "b_dst", *dst_anchor, output_model_binding_blinder_label(head, "dst"), context.domains.c->name))
                {
                    return false;
                }
            }
            std::vector<std::uint8_t> results(checks.size(), 0);
            auto verify_range = [&](std::size_t begin, std::size_t end)
            {
                for (std::size_t index = begin; index < end; ++index)
                {
                    const auto &check = checks[index];
                    results[index] = crypto::KZG::verify_domain_divisibility(
                                         check.proof_local,
                                         check.anchor,
                                         check.blinder,
                                         check.domain_name,
                                         context.kzg.vk)
                                         ? 1
                                         : 0;
                }
            };
            const auto cpu_count = std::max<std::size_t>(
                1, std::thread::hardware_concurrency());
            const auto worker_count = std::min(checks.size(), cpu_count);
            if (worker_count <= 1)
            {
                verify_range(0, checks.size());
            }
            else
            {
                const auto chunk_size = (checks.size() + worker_count - 1) / worker_count;
                std::vector<std::future<void>> futures;
                futures.reserve(worker_count);
                for (std::size_t worker = 0; worker < worker_count; ++worker)
                {
                    const auto begin = worker * chunk_size;
                    const auto end = std::min(checks.size(), begin + chunk_size);
                    if (begin >= end)
                        break;
                    futures.push_back(std::async(
                        std::launch::async, verify_range, begin, end));
                }
                for (auto &future : futures)
                    future.get();
            }
            return std::all_of(
                results.begin(), results.end(),
                [](std::uint8_t result)
                { return result != 0; });
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
                                                         { return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'); });
            };
            return lhs.protocol_name == rhs.protocol_name && is_lower_hex_256(lhs.proof_nonce) && lhs.proof_task == rhs.proof_task && lhs.srs_id == rhs.srs_id && lhs.protocol_schema_id == rhs.protocol_schema_id && is_lower_hex_256(lhs.protocol_schema_id) && lhs.instance_schema_hash == rhs.instance_schema_hash && is_lower_hex_256(lhs.instance_schema_hash) && lhs.srs_type == rhs.srs_type && lhs.srs_degree_bound == rhs.srs_degree_bound && lhs.kzg_assumption_profile == rhs.kzg_assumption_profile && lhs.privacy_policy == rhs.privacy_policy && lhs.input_privacy == rhs.input_privacy && lhs.output_mode == rhs.output_mode && lhs.opening_mode == rhs.opening_mode && lhs.binding_mode == rhs.binding_mode && lhs.model_arch == rhs.model_arch && lhs.model_params == rhs.model_params && lhs.input_preprocess_id == rhs.input_preprocess_id && lhs.static_tables == rhs.static_tables && lhs.quant_config == rhs.quant_config && lhs.domain_cfg == rhs.domain_cfg && lhs.dim_cfg == rhs.dim_cfg && lhs.encoding == rhs.encoding && lhs.padding_rule == rhs.padding_rule && lhs.degree_bound == rhs.degree_bound && lhs.security_budget == rhs.security_budget && !lhs.public_output.empty();
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
                    specs.push_back({
                        "mu_" + suffix + "_proj",
                        prefix + "H_star",
                        challenge("rho_proj_h"),
                    });

                    specs.push_back({
                        external_mask_eval_name(
                            "mu_" + suffix + "_proj"),
                        external_mask_label(
                            prefix + "H_star"),
                        challenge("rho_proj_h"),
                    });
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
            for (std::size_t layer_index = 0;
                 layer_index < context.config.hidden_profile.size(); ++layer_index)
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
            for (std::size_t head_index = 0;
                 head_index < context.config.K_out; ++head_index)
            {
                const auto prefix = output_head_prefix(head_index);
                specs.push_back({
                    output_external_eval_name(
                        "proj",
                        head_index),
                    prefix + "_Y_prime_star",
                    output_challenge_name(
                        "rho_proj_out",
                        head_index),
                });
                specs.push_back({
                    external_mask_eval_name(
                        output_external_eval_name(
                            "proj",
                            head_index)),
                    external_mask_label(
                        prefix + "_Y_prime_star"),
                    output_challenge_name(
                        "rho_proj_out",
                        head_index),
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
                const auto expected =
                    expected_payload_kind_for_label(context, label);

                if (expected == PayloadValueKind::MaskedPrivateValue &&
                    actual == PayloadValueKind::PublicValue)
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
                    return fail(
                        "domain opening payload kind count mismatch");
                }

                for (std::size_t i = 0; i < bundle.values.size(); ++i)
                {
                    if (!check(
                            bundle.values[i].first,
                            bundle.value_kinds[i]))
                    {
                        return fail(
                            "domain opening payload kind mismatch: " +
                            bundle.values[i].first);
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
                        if (evaluation.kind !=
                            PayloadValueKind::PublicValue)
                        {
                            return fail(
                                "Pi_bind public multiplicity kind mismatch");
                        }
                    }
                    else
                    {
                        auto source_label = evaluation.object_id;

                        if (source_label.ends_with(":u_rel"))
                        {
                            source_label.resize(
                                source_label.size() - 6);
                        }

                        if (!check(
                                source_label,
                                evaluation.kind))
                        {
                            return fail(
                                "Pi_bind evaluation payload kind mismatch: " +
                                evaluation.object_id);
                        }
                    }
                }
            }

            if (proof.external_evaluation_kinds.size() !=
                proof.external_evaluations.size())
            {
                return fail(
                    "external evaluation payload kind count mismatch");
            }

            std::unordered_map<std::string, std::string>
                external_sources;

            for (const auto &spec : multihead_external_specs(context))
            {
                external_sources.emplace(
                    spec.proof_name,
                    spec.label);
            }

            for (std::size_t i = 0;
                 i < proof.external_evaluations.size();
                 ++i)
            {
                const auto &name =
                    proof.external_evaluations[i].first;

                const auto it =
                    external_sources.find(name);

                if (it == external_sources.end())
                {
                    return fail(
                        "unknown external evaluation payload: " +
                        name);
                }

                if (!check(
                        it->second,
                        proof.external_evaluation_kinds[i]))
                {
                    return fail(
                        "external evaluation payload kind mismatch: " +
                        name);
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

    namespace internal
    {
        AggregationMultiplicityCache build_aggregation_multiplicity_cache(
            const data::LocalGraph &local,
            std::size_t n_domain_size)
        {
            if (n_domain_size == 0)
                throw std::runtime_error(
                    "aggregation multiplicity cache requires N domain");
            if (local.num_nodes > n_domain_size)
                throw std::runtime_error(
                    "aggregation multiplicity node count exceeds N domain");

            AggregationMultiplicityCache cache;
            cache.src.assign(n_domain_size, FieldElement::zero());
            cache.dst.assign(n_domain_size, FieldElement::zero());
            for (const auto &edge : local.edges)
            {
                if (edge.src >= local.num_nodes || edge.dst >= local.num_nodes)
                    throw std::runtime_error(
                        "aggregation multiplicity edge escapes local node range");
                cache.src[edge.src] += FieldElement::one();
                cache.dst[edge.dst] += FieldElement::one();
            }
            return cache;
        }

        algebra::FieldElement evaluate_deterministic_public_polynomial(
            const ProtocolContext &context,
            const VerifierPublicPolynomialDescriptor &descriptor,
            const algebra::FieldElement &point,
            const AggregationMultiplicityCache &multiplicity)
        {
            return evaluate_deterministic_public_polynomial_impl(
                context,
                descriptor,
                point,
                multiplicity.src,
                multiplicity.dst);
        }
    } // namespace internal

    std::size_t private_plain_value_count_internal(const ProtocolContext &context, const Proof &proof)
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
    bool verify_with_execution_context(const ProtocolContext &context, const Proof &proof, RunMetrics *metrics)
    {
        const auto metadata_start = Clock::now();
        if (!has_formal_public_architecture(context))
        {
            return false;
        }
        if (context.compiler_artifact == nullptr || context.validated_compiler_artifact_identity != context.compiler_artifact.get())
        {
            return false;
        }
        try
        {
            data::validate_normalized_graph_input(context.local);
        }
        catch (const std::exception &)
        {
            return false;
        }
        catch (...)
        {
            return false;
        }
        if (!public_statement_matches_registered_columns(context))
        {
            return false;
        }
        if (!metadata_matches(proof.public_metadata, build_public_metadata(context)))
        {
            return false;
        }
        if (proof.public_metadata.protocol_name != "zkgat" || proof.public_metadata.proof_task != "node_classification_full_gat" || proof.public_metadata.srs_type != "trusted_srs" || proof.public_metadata.privacy_policy != "model_and_trace_hidden" || proof.public_metadata.input_privacy != "public_graph_public_features" || proof.public_metadata.output_mode != "public_logits" || proof.public_metadata.opening_mode != "point_grouped_batch_opening" || proof.public_metadata.binding_mode != "zk_subproof_for_crpc")
        {
            return false;
        }
        if (proof.block_order != proof_block_order())
        {
            return false;
        }
        std::size_t audited_private_plain_count = 0;
        std::string payload_audit_reason;
        if (!audit_payload_values(context, proof, &audited_private_plain_count, &payload_audit_reason))
        {
            return false;
        }
        if (audited_private_plain_count != 0)
        {
            return false;
        }
        if (!verify_public_output_binding(context, proof))
        {
            return false;
        }
        if (metrics != nullptr)
        {
            metrics->verify_metadata_ms += elapsed_ms(metadata_start, Clock::now());
        }
        {
            const auto transcript_start = Clock::now();
            const auto expected_dynamic_labels = dynamic_commitment_labels(context);
            if (proof.dynamic_commitments.size() != expected_dynamic_labels.size())
            {
                return false;
            }
            for (std::size_t i = 0; i < proof.dynamic_commitments.size(); ++i)
            {
                if (proof.dynamic_commitments[i].first != expected_dynamic_labels[i])
                {
                    return false;
                }
            }
            const auto expected_quotient_labels = quotient_commitment_labels(context);
            if (proof.quotient_commitments.size() != expected_quotient_labels.size())
            {
                return false;
            }
            for (std::size_t i = 0; i < proof.quotient_commitments.size(); ++i)
            {
                if (proof.quotient_commitments[i].first != expected_quotient_labels[i])
                {
                    return false;
                }
            }
            const auto top_level_dynamic_commitments = to_map(proof.dynamic_commitments);
            const auto quotient_commitments = to_map(proof.quotient_commitments);
            const auto external_evaluations = to_field_map(proof.external_evaluations);
            std::unordered_map<std::string, Commitment> dynamic_commitments;
            if (!validate_pi_bind_model_reblinds(
                    context,
                    proof,
                    top_level_dynamic_commitments,
                    &dynamic_commitments))
            {
                return false;
            }
            const auto challenges = replay_challenges(
                context, dynamic_commitments, quotient_commitments, &proof.public_metadata);
            if (proof.challenges != challenges)
            {
                return false;
            }
            for (std::size_t head = 0;
                 head < context.config.K_out;
                 ++head)
            {
                const auto relation_id =
                    "bind:output:" + std::to_string(head);
                const auto subproof_it = std::find_if(
                    proof.pi_bind.subproofs.begin(), proof.pi_bind.subproofs.end(),
                    [&](const BindingSubproof &subproof)
                    { return subproof.relation_meta.relation_id == relation_id; });
                if (subproof_it == proof.pi_bind.subproofs.end())
                    return false;
                const auto *relation_ptr = find_unique_shift_relation_by_kind(
                    subproof_it->relation_meta, "output-bias-terminal");
                if (relation_ptr == nullptr)
                    return false;
                const auto &relation = *relation_ptr;
                const auto y_rel = challenges.at(relation.challenge_label);
                std::vector<std::string> expected_references;
                auto append_unique_reference = [&](std::string reference)
                {
                    if (std::find(
                            expected_references.begin(),
                            expected_references.end(), reference) == expected_references.end())
                        expected_references.push_back(std::move(reference));
                };
                for (const auto &route :
                     subproof_it->relation_meta.route_terminal_relations)
                {
                    const auto route_y = challenges.at(
                        route.challenge_label);
                    append_unique_reference(
                        opening_group_reference(
                            context, challenges, route_y));
                    append_unique_reference(
                        opening_group_reference(
                            context,
                            challenges,
                            route_y + route.terminal_delta));
                    append_unique_reference(
                        opening_group_reference(
                            context,
                            challenges,
                            route.node_terminal_point));
                }
                for (const auto &aggregation :
                     subproof_it->relation_meta.aggregation_relations)
                {
                    for (const auto *route : {
                             &aggregation.source_route,
                             &aggregation.target_route})
                    {
                        const auto aggregation_y = challenges.at(
                            route->terminal_challenge_label);
                        append_unique_reference(
                            opening_group_reference(
                                context, challenges, aggregation_y));
                        append_unique_reference(
                            opening_group_reference(
                                context,
                                challenges,
                                aggregation_y + route->terminal_delta));
                        append_unique_reference(
                            opening_group_reference(
                                context,
                                challenges,
                                route->node_terminal_point));
                    }
                }
                if (relation.configured_degrees.empty())
                {
                    append_unique_reference(opening_group_reference(
                        context, challenges, y_rel));
                    append_unique_reference(
                        opening_group_reference(
                            context,
                            challenges,
                            relation.canonical_points.front()));
                }
                else
                {
                    for (const auto &point : std::vector<FieldElement>{
                             y_rel,
                             y_rel + relation.deltas.at(1),
                             relation.canonical_points.at(0)})
                        append_unique_reference(opening_group_reference(
                            context, challenges, point));
                }
                if (subproof_it->relation_meta
                        .coefficient_basis_relations.size() != 1)
                    return false;
                const auto &basis = subproof_it->relation_meta
                                        .coefficient_basis_relations.front();
                const auto gamma = challenges.at(
                    basis.gamma_challenge_label);
                const auto basis_y = challenges.at(
                    basis.relation_challenge_label);
                append_unique_reference(opening_group_reference(
                    context, challenges, basis_y));
                for (const auto &chunk : basis.chunks)
                    append_unique_reference(
                        opening_group_reference(
                            context,
                            challenges,
                            basis_y + chunk.terminal_point - gamma));
                append_unique_reference(opening_group_reference(
                    context, challenges, gamma));
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
                const auto n_point = challenges.at("z_N");
                append_unique_reference(
                    "Open_dom:N:" + n_point.to_string());
                append_unique_reference(
                    "Open_dom:N:" + (n_point * context.domains.n->omega).to_string());
                if (subproof_it->opening_references.size() != expected_references.size() || !std::equal(
                                                                                                subproof_it->opening_references.begin(),
                                                                                                subproof_it->opening_references.end(),
                                                                                                expected_references.begin(),
                                                                                                [](const MetadataReference &reference,
                                                                                                   const std::string &expected)
                                                                                                { return reference.object_id == expected; }))
                    return false;
            }
            if (!verify_model_binding_relations(context, dynamic_commitments, challenges))
            {
                return false;
            }
            if (metrics != nullptr)
            {
                metrics->verify_transcript_ms += elapsed_ms(transcript_start, Clock::now());
            }
            if (proof.route_references != route_opening_references(context, challenges))
                return false;
            for (const auto &spec : multihead_external_specs(context))
            {
                if (!external_evaluations.contains(spec.proof_name))
                {
                    return false;
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
            // The lazy packed backend reads canonical public columns through
            // non-owning pointers, so its lifetime must remain within this
            // verification context.
            auto public_backend_registry =
                std::make_shared<PublicEvaluationBackendRegistry>(context);
            VerifierEvaluationMemoization eval_memo(context, opened_values, public_backend_registry, metrics);
            std::unordered_map<std::string, FieldElement> verifier_local_power_columns;
            for (const auto &binding_meta : build_binding_relation_metadata(context))
            {
                for (const auto &basis : binding_meta.coefficient_basis_relations)
                {
                    if (basis.carrier_domain != context.domains.edge->name)
                        throw std::runtime_error(
                            "unsupported coefficient basis carrier domain: " + basis.carrier_domain);
                    verifier_local_power_columns.emplace(
                        basis.power_object,
                        challenges.at(basis.gamma_challenge_label));
                }
            }
            const auto eval = [&](const std::string &name, const FieldElement &point) -> FieldElement
            {
                if (const auto it = verifier_local_power_columns.find(name);
                    it != verifier_local_power_columns.end())
                    return evaluate_geometric_power_column(
                        *context.domains.edge, it->second, point);
                return eval_memo.eval_named(name, point);
            };
            const auto edge_evaluation_plan =
                build_t_edge_evaluation_plan(context, challenges);
            const auto point_group_registry_start = Clock::now();
            StandardPointGroupRegistry opening_registry;
            for (std::size_t spec_index = 0;
                 spec_index < bundle_specs.size(); ++spec_index)
            {
                const auto &spec = bundle_specs[spec_index];
                const auto &indexed_bundle = bundle_by_name(indexed_bundles, spec.bundle_name);
                const auto &bundle = *indexed_bundle.bundle;
                auto labels = domain_opening_labels(context, spec.trace_domain_name);
                labels.push_back(spec.quotient_name);
                const auto quotient_mask_name = quotient_mask_label(spec.quotient_name);
                labels.push_back(quotient_mask_name);
                if (bundle.values.size() != labels.size())
                {
                    return false;
                }
                if (bundle.points.size() != 2 || bundle.points[0] != challenges.at(spec.z_name) || bundle.points[1] != challenges.at(spec.z_name) * spec.domain->omega)
                {
                    return false;
                }
                std::vector<Commitment> commitments;
                commitments.reserve(labels.size());
                const auto bundle_lookup_start = Clock::now();
                for (std::size_t i = 0; i < labels.size(); ++i)
                {
                    if (bundle.values[i].first != labels[i])
                    {
                        return false;
                    }
                    if (dynamic_commitments.contains(labels[i]))
                    {
                        commitments.push_back(
                            dynamic_commitments.at(labels[i]));
                    }
                    else if (quotient_commitments.contains(labels[i]))
                    {
                        commitments.push_back(
                            quotient_commitments.at(labels[i]));
                    }
                    else
                    {
                        return false;
                    }
                }
                add_domain_bundle_claims(
                    opening_registry,
                    spec.bundle_name,
                    bundle,
                    commitments);
                if (metrics != nullptr)
                {
                    metrics->verify_bundle_lookup_ms += elapsed_ms(bundle_lookup_start, Clock::now());
                }
            }
            auto verify_bundle_spec = [&](const BundleSpec &spec)
            {
                // Opening shape and KZG checks were completed before quotient
                // evaluation so the seven independent pairings can run in
                // parallel without sharing verifier evaluation caches.
                const auto &indexed_bundle = bundle_by_name(
                    indexed_bundles, spec.bundle_name);
                const auto &bundle = *indexed_bundle.bundle;
                const auto quotient_mask_name = quotient_mask_label(
                    spec.quotient_name);
                const auto quotient_start = Clock::now();
                FieldElement expected_t = FieldElement::zero();
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
                    expected_t = evaluate_t_edge(
                        context,
                        challenges,
                        external_evaluations,
                        eval,
                        challenges.at(spec.z_name),
                        false,
                        nullptr,
                        nullptr,
                        nullptr,
                        edge_evaluation_plan.get());
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
                    expected_t = evaluate_t_n(context, challenges, eval, challenges.at(spec.z_name));
                }
                else
                {
                    return false;
                }
                const auto quotient_point = challenges.at(spec.z_name);
                const auto opened_mask = bundle_value(
                    indexed_bundle,
                    quotient_mask_name,
                    quotient_point);
                const auto expected_masked_quotient =
                    expected_t + spec.domain->zero_polynomial_eval(quotient_point) * opened_mask;
                const auto opened_quotient = bundle_value(
                    indexed_bundle, spec.quotient_name, quotient_point);
                if (opened_quotient != expected_masked_quotient)
                {
                    return false;
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
                if ((!spec.static_commitment && !dynamic_commitments.contains(spec.label)) || (spec.static_commitment && !context.static_commitments.contains(spec.label)))
                {
                    return false;
                }
                external_commitments.emplace_back(
                    spec.static_commitment ? context.static_commitments.at(spec.label) : dynamic_commitments.at(spec.label),
                    external_value(proof, spec.proof_name));
                external_points.push_back(challenges.at(spec.challenge_name));
            }
            for (const auto &subproof : proof.pi_bind.subproofs)
            {
                for (const auto &relation :
                     subproof.relation_meta.route_terminal_relations)
                {
                    if (!dynamic_commitments.contains(
                            relation.shifted_accumulator_object) ||
                        !dynamic_commitments.contains(
                            relation.edge_accumulator_object) ||
                        !dynamic_commitments.contains(
                            relation.node_accumulator_object))
                        return false;
                    const auto evaluation_id =
                        relation.shifted_accumulator_object + ":u_rel";
                    const auto evaluation = std::find_if(
                        subproof.evaluations.begin(),
                        subproof.evaluations.end(),
                        [&](const BindingEvaluation &candidate)
                        { return candidate.object_id == evaluation_id; });
                    if (evaluation == subproof.evaluations.end() || evaluation->domain != "relation" || evaluation->values.size() != 1 || evaluation->kind != PayloadValueKind::MaskedPrivateValue)
                        return false;
                    const auto y_rel = challenges.at(
                        relation.challenge_label);
                    external_commitments.push_back({dynamic_commitments.at(
                                                        relation.shifted_accumulator_object),
                                                    evaluation->values.front()});
                    external_points.push_back(y_rel);
                    external_commitments.push_back({dynamic_commitments.at(
                                                        relation.edge_accumulator_object),
                                                    evaluation->values.front()});
                    external_points.push_back(
                        y_rel + relation.terminal_delta);
                    external_commitments.push_back({Commitment{
                                                        .name = relation.relation_virtual_object,
                                                        .point = crypto::g1_add(
                                                            dynamic_commitments.at(
                                                                                   relation.node_accumulator_object)
                                                                .point,
                                                            crypto::g1_neg(
                                                                dynamic_commitments.at(
                                                                                       relation.shifted_accumulator_object)
                                                                    .point)),
                                                    },
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
                        if (!dynamic_commitments.contains(
                                route->shifted_accumulator_object) ||
                            !dynamic_commitments.contains(
                                route->edge_accumulator_object) ||
                            !dynamic_commitments.contains(
                                route->node_accumulator_object))
                            return false;
                        const auto evaluation_id =
                            route->shifted_accumulator_object + ":u_rel";
                        const auto evaluation = std::find_if(
                            subproof.evaluations.begin(),
                            subproof.evaluations.end(),
                            [&](const BindingEvaluation &candidate)
                            { return candidate.object_id == evaluation_id; });
                        if (evaluation == subproof.evaluations.end() || evaluation->values.size() != 1 || evaluation->kind != PayloadValueKind::MaskedPrivateValue)
                            return false;
                        const auto y_rel = challenges.at(
                            route->terminal_challenge_label);
                        external_commitments.push_back({dynamic_commitments.at(
                                                            route->shifted_accumulator_object),
                                                        evaluation->values.front()});
                        external_points.push_back(y_rel);
                        external_commitments.push_back({dynamic_commitments.at(
                                                            route->edge_accumulator_object),
                                                        evaluation->values.front()});
                        external_points.push_back(
                            y_rel + route->terminal_delta);
                        external_commitments.push_back({Commitment{
                                                            .name = route->relation_virtual_object,
                                                            .point = crypto::g1_add(
                                                                dynamic_commitments.at(
                                                                                       route->node_accumulator_object)
                                                                    .point,
                                                                crypto::g1_neg(
                                                                    dynamic_commitments.at(
                                                                                           route->shifted_accumulator_object)
                                                                        .point)),
                                                        },
                                                        FieldElement::zero()});
                        external_points.push_back(
                            route->node_terminal_point);
                    }
                }
            }
            for (std::size_t head_index = 0;
                 head_index < context.config.K_out;
                 ++head_index)
            {
                const auto relation_id =
                    "bind:output:" + std::to_string(head_index);
                const auto subproof_it = std::find_if(
                    proof.pi_bind.subproofs.begin(), proof.pi_bind.subproofs.end(),
                    [&](const BindingSubproof &subproof)
                    { return subproof.relation_meta.relation_id == relation_id; });
                if (subproof_it == proof.pi_bind.subproofs.end())
                    return false;
                const auto *relation_ptr = find_unique_shift_relation_by_kind(
                    subproof_it->relation_meta, "output-bias-terminal");
                if (relation_ptr == nullptr)
                    return false;
                const auto &relation = *relation_ptr;
                if (relation.configured_degrees.empty())
                {
                    const auto r_can_label = output_bias_view_label(
                        head_index, "R_can");
                    const auto r_qry_label = output_bias_view_label(
                        head_index, "R_qry");
                    if (relation.canonical_points.size() != 1 || !dynamic_commitments.contains(r_can_label) || !dynamic_commitments.contains(r_qry_label))
                        return false;
                    const auto &c_bundle = indexed_bundles.at("C");
                    for (const auto &point : c_bundle.bundle->points)
                    {
                        if (bundle_value(
                                c_bundle,
                                output_bias_view_label(head_index, "m_can"),
                                point) != eval_memo.eval_named("P_BiasMultiplicity_C", point))
                            return false;
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
                                                 head_index, "m_can") +
                                             ":public_bind";
                    const auto public_m_it = std::find_if(
                        subproof_it->evaluations.begin(),
                        subproof_it->evaluations.end(),
                        [&](const BindingEvaluation &evaluation)
                        { return evaluation.object_id == public_m_id; });
                    const auto y_public = challenges.at(relation.challenge_label);
                    if (public_m_it == subproof_it->evaluations.end() || public_m_it->values.size() != 1 || public_m_it->kind != PayloadValueKind::PublicValue || public_m_it->values.front() != eval_memo.eval_named(output_bias_view_label(head_index, "m_can"), y_public))
                        return false;
                    external_commitments.push_back({dynamic_commitments.at(output_bias_view_label(
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
                if (u_it == subproof_it->evaluations.end() || u_it->domain != "relation" || u_it->values.size() != 1 || u_it->kind != PayloadValueKind::MaskedPrivateValue || !dynamic_commitments.contains(shifted_label))
                    return false;
                const bool canonical_is_first =
                    relation.source_domains.at(0) == context.domains.c->name;
                const auto r_can_label =
                    output_bias_view_label(head_index, "R_can");
                const auto r_qry_label =
                    output_bias_view_label(head_index, "R_qry");
                if (!dynamic_commitments.contains(r_can_label) || !dynamic_commitments.contains(r_qry_label))
                    return false;
                const auto &c_bundle = indexed_bundles.at("C");
                for (const auto &point : c_bundle.bundle->points)
                {
                    if (bundle_value(
                            c_bundle,
                            output_bias_view_label(head_index, "m_can"),
                            point) != eval_memo.eval_named("P_BiasMultiplicity_C", point))
                        return false;
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
                if (public_m_it == subproof_it->evaluations.end() || public_m_it->domain != "relation" || public_m_it->values.size() != 1 || public_m_it->kind != PayloadValueKind::PublicValue || public_poly_it == context.public_commitments.end() || public_m_it->values.front() != eval_memo.eval_named("P_BiasMultiplicity_C", y_rel))
                    return false;
                external_commitments.push_back({dynamic_commitments.at(
                                                    output_bias_view_label(head_index, "m_can")),
                                                public_m_it->values.front()});
                external_points.push_back(y_rel);
            }
            for (std::size_t head_index = 0;
                 head_index < context.config.K_out;
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
                    return false;
                const auto &basis = subproof_it->relation_meta
                                        .coefficient_basis_relations.front();
                if (!dynamic_commitments.contains(basis.coefficient_object))
                    return false;
                const auto gamma = challenges.at(
                    basis.gamma_challenge_label);
                const auto y_rel = challenges.at(
                    basis.relation_challenge_label);
                auto relation_point = dynamic_commitments.at(
                                                             basis.coefficient_object)
                                          .point;
                for (const auto &chunk : basis.chunks)
                {
                    const auto u_id = chunk.shifted_accumulator_object + ":u_rel";
                    const auto u_it = std::find_if(
                        subproof_it->evaluations.begin(),
                        subproof_it->evaluations.end(),
                        [&](const BindingEvaluation &evaluation)
                        { return evaluation.object_id == u_id; });
                    if (u_it == subproof_it->evaluations.end() || u_it->domain != "relation" || u_it->values.size() != 1 || u_it->kind != PayloadValueKind::MaskedPrivateValue || !dynamic_commitments.contains(chunk.accumulator_object) || !dynamic_commitments.contains(chunk.shifted_accumulator_object))
                        return false;
                    const auto scale = gamma.pow(chunk.offset);
                    relation_point = crypto::g1_add(
                        relation_point,
                        crypto::g1_neg(crypto::g1_mul(
                            dynamic_commitments.at(
                                                   chunk.shifted_accumulator_object)
                                .point,
                            scale)));
                    external_commitments.push_back({dynamic_commitments.at(
                                                        chunk.shifted_accumulator_object),
                                                    u_it->values.front()});
                    external_points.push_back(y_rel);
                    external_commitments.push_back({dynamic_commitments.at(chunk.accumulator_object),
                                                    u_it->values.front()});
                    external_points.push_back(
                        y_rel + chunk.terminal_point - gamma);
                }
                const Commitment relation_commitment{
                    .name = output_bias_basis_label(
                        head_index, "relation_virtual"),
                    .point = relation_point,
                };
                external_commitments.push_back({relation_commitment, FieldElement::zero()});
                external_points.push_back(gamma);
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
            add_external_claims(
                opening_registry,
                external_commitments,
                external_points,
                proof.external_evaluation_kinds);
            opening_registry.finalize();
            const auto &point_groups = opening_registry.groups();
            if (metrics != nullptr)
            {
                metrics->verify_point_group_registry_ms += elapsed_ms(
                    point_group_registry_start, Clock::now());
            }
            try
            {
                require_opening_layout(proof.opening_groups, point_groups);
            }
            catch (const std::exception &)
            {
                return false;
            }
            catch (...)
            {
                return false;
            }

            std::vector<std::uint8_t> group_results(point_groups.size(), 0);
            std::vector<double> group_elapsed_ms(point_groups.size(), 0.0);
            const auto verify_group_indices = [&](const std::vector<std::size_t> &indices)
            {
                if (indices.empty())
                    return 0.0;
                const auto phase_start = Clock::now();
                const auto hardware_workers = std::max<unsigned>(
                    1, std::thread::hardware_concurrency());
                const auto worker_count = std::min<std::size_t>(
                    indices.size(), std::min<std::size_t>(
                                        16, hardware_workers));
                std::atomic<std::size_t> next{0};
                std::vector<std::future<void>> futures;
                futures.reserve(worker_count);
                for (std::size_t worker = 0; worker < worker_count; ++worker)
                {
                    futures.push_back(std::async(
                        std::launch::async,
                        [&]
                        {
                            while (true)
                            {
                                const auto position = next.fetch_add(1);
                                if (position >= indices.size())
                                    break;
                                const auto group_index = indices[position];
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
                                const auto local_start = Clock::now();
                                group_results[group_index] =
                                    crypto::KZG::verify_point_group(
                                        commitments,
                                        group.point,
                                        values,
                                        derive_point_group_challenge(
                                            proof.public_metadata, group),
                                        proof.opening_groups[group_index].witness,
                                        context.kzg.vk)
                                        ? 1
                                        : 0;
                                group_elapsed_ms[group_index] = elapsed_ms(
                                    local_start, Clock::now());
                            }
                        }));
                }
                for (auto &future : futures)
                    future.get();
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
            double domain_opening_wall_ms = 0.0;
            double external_opening_wall_ms = 0.0;
            try
            {
                std::vector<std::size_t> all_group_indices;
                all_group_indices.reserve(point_groups.size());
                for (std::size_t i = 0; i < point_groups.size(); ++i)
                    all_group_indices.push_back(i);
                const auto point_group_wall_ms = verify_group_indices(
                    all_group_indices);
                double domain_group_cpu_ms = 0.0;
                double external_group_cpu_ms = 0.0;
                for (const auto group_index : domain_group_indices)
                    domain_group_cpu_ms += group_elapsed_ms[group_index];
                for (const auto group_index : external_group_indices)
                    external_group_cpu_ms += group_elapsed_ms[group_index];
                const auto group_cpu_ms =
                    domain_group_cpu_ms + external_group_cpu_ms;
                if (group_cpu_ms > 0.0)
                {
                    domain_opening_wall_ms = point_group_wall_ms * domain_group_cpu_ms / group_cpu_ms;
                    external_opening_wall_ms =
                        point_group_wall_ms - domain_opening_wall_ms;
                }
                else
                {
                    domain_opening_wall_ms = point_group_wall_ms;
                }
            }
            catch (const std::exception &)
            {
                return false;
            }
            catch (...)
            {
                return false;
            }
            const auto failed_group = std::find(
                group_results.begin(), group_results.end(), 0);
            if (failed_group != group_results.end())
            {
                return false;
            }
            std::unordered_map<std::string, double>
                domain_opening_elapsed_by_name;
            for (const auto group_index : domain_group_indices)
            {
                domain_opening_elapsed_by_name[point_groups[group_index].owner_block.substr(9)] += group_elapsed_ms[group_index];
            }
            if (metrics != nullptr)
            {
                metrics->verify_domain_opening_ms += domain_opening_wall_ms;
                metrics->verify_external_opening_ms += external_opening_wall_ms;
            }
            for (std::size_t spec_index = 0;
                 spec_index < bundle_specs.size(); ++spec_index)
            {
                const auto &spec = bundle_specs[spec_index];
                const auto bundle_start = Clock::now();
                if (!verify_bundle_spec(spec))
                {
                    return false;
                }
                if (metrics != nullptr)
                {
                    const auto elapsed = elapsed_ms(bundle_start, Clock::now());
                    if (auto *metric = verify_domain_metric(metrics, spec.bundle_name); metric != nullptr)
                    {
                        *metric += domain_opening_elapsed_by_name[spec.bundle_name] + elapsed;
                    }
                }
            }
            return true;
        }
    }

    namespace
    {
        bool proof_contains_prover_material(const Proof &proof)
        {
            const auto has_material = [](const crypto::Commitment &commitment)
            {
                return commitment.opening_polynomial != nullptr;
            };
            for (const auto &[name, commitment] : proof.dynamic_commitments)
            {
                (void)name;
                if (has_material(commitment))
                    return true;
            }
            for (const auto &[name, commitment] : proof.quotient_commitments)
            {
                (void)name;
                if (has_material(commitment))
                    return true;
            }
            for (const auto &subproof : proof.pi_bind.subproofs)
            {
                for (const auto &reblind : subproof.model_reblinds)
                    if (has_material(reblind.proof_local_commitment) || has_material(reblind.blinder_commitment))
                        return true;
                for (const auto *group : {
                         &subproof.view_commitments,
                         &subproof.view_logup_commitments,
                         &subproof.aggregation_commitments,
                         &subproof.basis_commitments,
                         &subproof.crpc_commitments,
                         &subproof.relation_commitments})
                    for (const auto &[name, commitment] : *group)
                    {
                        (void)name;
                        if (has_material(commitment))
                            return true;
                    }
            }
            return false;
        }
    } // namespace

    std::size_t private_plain_value_count(
        const VerifierContext &context,
        const Proof &proof)
    {
        if (proof_contains_prover_material(proof) || context.compiler_artifact == nullptr || !validate_verifier_context(context))
            return std::numeric_limits<std::size_t>::max();
        try
        {
            auto execution_context =
                detail::make_verifier_execution_context(context);
            return private_plain_value_count_internal(execution_context, proof);
        }
        catch (const std::exception &)
        {
            return std::numeric_limits<std::size_t>::max();
        }
        catch (...)
        {
            return std::numeric_limits<std::size_t>::max();
        }
    }

    bool verify(
        const VerifierContext &context,
        const Proof &proof,
        RunMetrics *metrics)
    {
        // Only an authenticated registry context can enter formal verification.
        if (context.compiler_artifact == nullptr || !validate_verifier_context(context) || proof_contains_prover_material(proof))
            return false;
        try
        {
            auto execution_context =
                detail::make_verifier_execution_context(context);
            return verify_with_execution_context(
                execution_context,
                proof,
                metrics);
        }
        catch (const std::exception &)
        {
            return false;
        }
        catch (...)
        {
            return false;
        }
    }

} // namespace gatzk::protocol
