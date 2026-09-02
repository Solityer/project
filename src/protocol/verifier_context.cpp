#include "gatzk/protocol/verifier_context.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <openssl/sha.h>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

#include "gatzk/protocol/challenges.hpp"
#include "gatzk/data/loader.hpp"
#include "gatzk/protocol/schema.hpp"

namespace gatzk::protocol
{
    namespace
    {
        using algebra::FieldElement;
        constexpr std::string_view kStaticTableGeneratorId =
            "zkgat_static_table_generator";
        constexpr std::size_t kMaxRegistryEntries = 1ULL << 20;

        void append_u64(std::string *out, std::uint64_t value)
        {
            for (std::size_t i = 0; i < sizeof(value); ++i)
                out->push_back(static_cast<char>((value >> (8U * i)) & 0xffU));
        }

        void append_i64(std::string *out, std::int64_t value)
        {
            append_u64(out, static_cast<std::uint64_t>(value));
        }

        void append_string(std::string *out, std::string_view value)
        {
            append_u64(out, value.size());
            out->append(value.data(), value.size());
        }

        std::string sha256_hex(std::string_view bytes)
        {
            std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
            SHA256(
                reinterpret_cast<const unsigned char *>(bytes.data()),
                bytes.size(), digest.data());
            std::ostringstream out;
            out << std::hex << std::setfill('0');
            for (const auto byte : digest)
                out << std::setw(2) << static_cast<unsigned int>(byte);
            return out.str();
        }

        bool lower_hex_256(const std::string &value)
        {
            return value.size() == 64
                && std::all_of(
                    value.begin(), value.end(), [](unsigned char ch)
                    { return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'); });
        }

        void append_registry_entry(
            std::string *out,
            const VerifierRegistryEntry &entry)
        {
            append_string(out, entry.label);
            append_string(out, entry.domain);
            append_u64(out, entry.degree_bound);
            append_u64(out, entry.scale);
            append_i64(out, entry.range_min);
            append_i64(out, entry.range_max);
            const auto point = crypto::serialize(entry.point);
            append_string(
                out,
                std::string_view(
                    reinterpret_cast<const char *>(point.data()), point.size()));
            append_u64(out, entry.bound_certificate.present ? 1U : 0U);
            if (entry.bound_certificate.present)
            {
                const auto &certificate = entry.bound_certificate;
                append_string(out, certificate.label);
                append_u64(out, certificate.layer_index);
                append_u64(out, certificate.local_head_index);
                append_string(out, certificate.domain);
                append_u64(out, certificate.degree_bound);
                append_u64(out, certificate.scale);
                append_u64(out, certificate.absolute_bound);
            }
        }

        std::string model_layout_serialization(
            const model::ModelParameters &model)
        {
            std::ostringstream out;
            out << "model_layout|arch=" << model.model_arch_id
                << "|param=" << model.model_param_id
                << "|input_preprocess=" << model.input_preprocess_id
                << "|L=" << model.L
                << "|K_out=" << model.K_out
                << "|C=" << model.C;
            for (const auto &layer : model.hidden_layers)
            {
                out << "|layer=" << layer.layer_index
                    << ",input=" << layer.input_dim
                    << ",heads=" << layer.shape.head_count
                    << ",head_dim=" << layer.shape.head_dim
                    << ",head_vector=" << layer.heads.size();
            }
            out << "|output_input=" << model.output_layer.input_dim
                << "|output_dim=" << model.output_layer.output_dim
                << "|output_heads=" << model.output_layer.head_count
                << ",vector=" << model.output_layer.heads.size();
            return out.str();
        }

        std::string model_layout_serialization(
            const VerifierModelShape &model)
        {
            std::ostringstream out;
            out << "model_layout|arch=" << model.model_arch_id
                << "|param=" << model.model_param_id
                << "|input_preprocess=" << model.input_preprocess_id
                << "|L=" << model.L
                << "|K_out=" << model.K_out
                << "|C=" << model.C;
            for (const auto &layer : model.hidden_layers)
            {
                out << "|layer=" << layer.layer_index
                    << ",input=" << layer.input_dim
                    << ",heads=" << layer.shape.head_count
                    << ",head_dim=" << layer.shape.head_dim
                    << ",head_vector=" << layer.heads.size();
            }
            out << "|output_input=" << model.output_layer.input_dim
                << "|output_dim=" << model.output_layer.output_dim
                << "|output_heads=" << model.output_layer.head_count
                << ",vector=" << model.output_layer.heads.size();
            return out.str();
        }

        std::string static_table_serialization(const StaticTables &tables)
        {
            std::ostringstream out;
            out << "static_tables|range=" << tables.range.size();
            for (const auto &value : tables.range) out << '|' << value.to_string();
            out << "|lrelu=" << tables.lrelu.size();
            for (const auto &[x, y] : tables.lrelu)
                out << '|' << x.to_string() << ',' << y.to_string();
            out << "|elu=" << tables.elu.size();
            for (const auto &[x, y] : tables.elu)
                out << '|' << x.to_string() << ',' << y.to_string();
            out << "|exp=" << tables.exp.size();
            for (const auto &[x, y] : tables.exp)
                out << '|' << x.to_string() << ',' << y.to_string();
            return out.str();
        }

        void append_static_table_shape(
            std::string *bytes,
            const CanonicalStaticTableShape &shape)
        {
            append_string(bytes, "canonical_static_table_shape");
            append_u64(bytes, shape.range_table_size);
            append_u64(bytes, static_cast<std::uint64_t>(shape.lrelu_min_raw));
            append_u64(bytes, static_cast<std::uint64_t>(shape.lrelu_max_raw));
            append_u64(bytes, shape.lrelu_table_size);
            append_u64(bytes, shape.elu_bound);
            append_u64(bytes, shape.elu_table_size);
            append_u64(bytes, shape.lrelu_output_bound);
            append_u64(bytes, shape.exp_input_bound);
            append_u64(bytes, shape.exp_table_size);
            append_u64(bytes, shape.table_edge_domain_size);
            append_u64(bytes, shape.large_whole_graph ? 1U : 0U);
        }

        std::string static_table_spec_serialization(
            const ProtocolContext &context)
        {
            // This is the producer's deterministic table specification.  The
            // digest binds the exact table bytes as well as the rules that
            // determine their interpretation; a marker-only certificate is
            // intentionally not sufficient.
            std::string bytes;
            append_string(&bytes, "static_table_spec");
            append_string(&bytes, kStaticTableGeneratorId);
            append_string(&bytes, context.config.static_table_id);
            append_u64(&bytes, context.config.range_bits);
            append_string(&bytes, context.config.quantization.canonical_serialization());
            append_string(
                &bytes,
                "range=ascending-u64;LReLU=canonical-signed-round;"
                "ELU=exp-minus-one-signed-round;"
                "ExpMap=exp-negative-delta-positive-saturate-one;"
                "pairs=ordered-functional;field=canonical");
            append_static_table_shape(
                &bytes, plan_canonical_static_table_shape(context));
            append_string(&bytes, static_table_serialization(context.tables));
            return bytes;
        }

        std::string range_scale_certificate_serialization(
            const ProtocolContext &context)
        {
            std::string bytes;
            append_string(&bytes, "range_scale_certificate");
            append_u64(&bytes, context.config.range_bits);
            append_string(&bytes, context.config.quantization.canonical_serialization());
            append_static_table_shape(
                &bytes, plan_canonical_static_table_shape(context));
            append_string(&bytes, static_table_serialization(context.tables));
            if (context.compiler_artifact == nullptr)
                throw std::runtime_error(
                    "range/scale certificate requires ProtocolCompilerArtifact");
            append_string(
                &bytes,
                integer_bound_ledger_canonical_bytes(
                    context.compiler_artifact->integer_bound_ledger));
            return bytes;
        }

        std::string static_table_spec_serialization(
            const VerifierContext &context)
        {
            std::string bytes;
            append_string(&bytes, "static_table_spec");
            append_string(&bytes, kStaticTableGeneratorId);
            append_string(&bytes, context.config.static_table_id);
            append_u64(&bytes, context.config.range_bits);
            append_string(&bytes, context.config.quantization.canonical_serialization());
            append_string(
                &bytes,
                "range=ascending-u64;LReLU=canonical-signed-round;"
                "ELU=exp-minus-one-signed-round;"
                "ExpMap=exp-negative-delta-positive-saturate-one;"
                "pairs=ordered-functional;field=canonical");
            append_static_table_shape(
                &bytes, plan_canonical_static_table_shape(context));
            append_string(&bytes, static_table_serialization(context.tables));
            return bytes;
        }

        std::string range_scale_certificate_serialization(
            const VerifierContext &context)
        {
            if (context.compiler_artifact == nullptr)
                throw std::runtime_error(
                    "range/scale certificate requires ProtocolCompilerArtifact");
            std::string bytes;
            append_string(&bytes, "range_scale_certificate");
            append_u64(&bytes, context.config.range_bits);
            append_string(&bytes, context.config.quantization.canonical_serialization());
            append_static_table_shape(
                &bytes, plan_canonical_static_table_shape(context));
            append_string(&bytes, static_table_serialization(context.tables));
            append_string(
                &bytes,
                integer_bound_ledger_canonical_bytes(
                    context.compiler_artifact->integer_bound_ledger));
            return bytes;
        }

        void append_size_vector(
            std::string *out,
            const std::vector<std::size_t> &values)
        {
            append_u64(out, values.size());
            for (const auto value : values)
                append_u64(out, value);
        }

        std::string public_input_serialization(const data::LocalGraph &local)
        {
            std::string bytes;
            append_string(&bytes, "public_input");
            append_string(&bytes, local.name);
            append_string(&bytes, local.input_preprocess_id);
            append_u64(&bytes, local.num_nodes);
            append_u64(&bytes, local.num_features);
            append_u64(&bytes, local.num_classes);
            append_u64(&bytes, local.graph_count);
            append_string(&bytes, local.task_type);
            append_string(&bytes, local.report_unit);
            append_string(&bytes, local.batching_rule);
            append_string(&bytes, local.subgraph_rule);
            append_string(&bytes, local.self_loop_rule);
            append_string(&bytes, local.edge_sort_rule);
            append_string(&bytes, local.chunking_rule);
            append_size_vector(&bytes, local.absolute_ids);
            append_size_vector(&bytes, local.node_ptr);
            append_size_vector(&bytes, local.edge_ptr);
            append_u64(&bytes, local.edges.size());
            for (const auto &edge : local.edges)
            {
                append_u64(&bytes, edge.src);
                append_u64(&bytes, edge.dst);
                append_u64(&bytes, edge.graph_id);
                append_u64(&bytes, edge.stable_index);
            }
            append_u64(&bytes, local.public_input.N_total);
            append_u64(&bytes, local.public_input.N);
            append_u64(&bytes, local.public_input.E);
            append_u64(&bytes, local.public_input.G_batch);
            append_size_vector(&bytes, local.public_input.I);
            append_size_vector(&bytes, local.public_input.src);
            append_size_vector(&bytes, local.public_input.dst);
            append_size_vector(&bytes, local.public_input.node_ptr);
            append_size_vector(&bytes, local.public_input.edge_ptr);
            append_u64(&bytes, local.features.size());
            for (const auto &row : local.features)
            {
                append_u64(&bytes, row.size());
                for (const auto &value : row)
                    append_string(&bytes, value.to_string());
            }
            return bytes;
        }

        std::string topology_serialization(const data::LocalGraph &local)
        {
            std::string bytes;
            append_string(&bytes, "topology");
            append_u64(&bytes, local.num_nodes);
            append_u64(&bytes, local.num_features);
            append_u64(&bytes, local.num_classes);
            append_u64(&bytes, local.graph_count);
            append_size_vector(&bytes, local.absolute_ids);
            append_size_vector(&bytes, local.node_ptr);
            append_size_vector(&bytes, local.edge_ptr);
            append_u64(&bytes, local.edges.size());
            for (const auto &edge : local.edges)
            {
                append_u64(&bytes, edge.src);
                append_u64(&bytes, edge.dst);
                append_u64(&bytes, edge.graph_id);
                append_u64(&bytes, edge.stable_index);
            }
            const auto &input = local.public_input;
            append_u64(&bytes, input.N_total);
            append_u64(&bytes, input.N);
            append_u64(&bytes, input.E);
            append_u64(&bytes, input.G_batch);
            append_size_vector(&bytes, input.I);
            append_size_vector(&bytes, input.src);
            append_size_vector(&bytes, input.dst);
            append_size_vector(&bytes, input.node_ptr);
            append_size_vector(&bytes, input.edge_ptr);
            return bytes;
        }

        std::vector<VerifierRegistryEntry> model_registry_entries(
            const ProtocolContext &context)
        {
            const auto &scales = context.config.quantization;
            const auto degree = context.kzg.manifest.degree_bound;
            if (context.config.range_bits < 1 || context.config.range_bits > 62)
                throw std::runtime_error("verifier registry range_bits must be between 1 and 62");
            const auto range_magnitude =
                std::int64_t{1} << context.config.range_bits;
            const auto range_min = -range_magnitude;
            const auto range_max = -range_min - 1;
            std::vector<VerifierRegistryEntry> entries;
            if (context.compiler_artifact == nullptr)
                throw std::runtime_error(
                    "model registry requires the concrete IntegerBoundLedger");
            const auto bound_for = [&](const std::string &id) -> std::uint64_t
            {
                const auto it = std::find_if(
                    context.compiler_artifact->integer_bound_ledger.begin(),
                    context.compiler_artifact->integer_bound_ledger.end(),
                    [&](const auto &entry) { return entry.id == id; });
                if (it == context.compiler_artifact->integer_bound_ledger.end())
                    throw std::runtime_error(
                        "model registry is missing IntegerBoundLedger entry: " + id);
                return it->absolute_bound;
            };
            auto add = [&](const std::string &label,
                           const std::string &domain,
                           std::uint64_t scale,
                           std::size_t layer,
                           std::size_t head,
                           const std::string &bound_id)
            {
                const auto it = context.static_commitments.find(label);
                if (it == context.static_commitments.end())
                    throw std::runtime_error("verifier registry missing model commitment: " + label);
                entries.push_back(VerifierRegistryEntry{
                    .label = label,
                    .domain = domain,
                    .degree_bound = degree,
                    .scale = scale,
                    .range_min = range_min,
                    .range_max = range_max,
                    .point = it->second.point,
                    .bound_certificate = ModelBoundCertificate{
                        .present = true,
                        .label = label,
                        .layer_index = layer,
                        .local_head_index = head,
                        .domain = domain,
                        .degree_bound = degree,
                        .scale = scale,
                        .absolute_bound = bound_for(bound_id)}});
            };
            for (std::size_t layer = 0; layer < context.model.hidden_layers.size(); ++layer)
            {
                const auto &heads = context.model.hidden_layers[layer].heads;
                for (std::size_t head = 0; head < heads.size(); ++head)
                {
                    const auto &scale = scales.s_w_hidden.at(layer).at(head);
                    for (std::size_t column = 0;
                         column < context.model.hidden_layers[layer].shape.head_dim;
                         ++column)
                        add(hidden_weight_column_label(layer, head, column),
                            "in", scale, layer, head,
                            "hidden:l" + std::to_string(layer) + ":h"
                                + std::to_string(head) + ".model.W");
                    add(hidden_src_label(layer, head),
                        "d", scales.s_a_src_hidden.at(layer).at(head), layer, head,
                        "hidden:l" + std::to_string(layer) + ":h"
                            + std::to_string(head) + ".model.a_src");
                    add(hidden_dst_label(layer, head),
                        "d", scales.s_a_dst_hidden.at(layer).at(head), layer, head,
                        "hidden:l" + std::to_string(layer) + ":h"
                            + std::to_string(head) + ".model.a_dst");
                }
            }
            for (std::size_t head = 0; head < context.model.output_layer.heads.size(); ++head)
            {
                for (std::size_t column = 0;
                     column < context.model.output_layer.output_dim;
                     ++column)
                    add(output_weight_column_label(head, column),
                        "cat", scales.s_w_output.at(head), 0, head,
                        "output:h" + std::to_string(head) + ".model.W");
                add(output_src_label(head),
                    "C", scales.s_a_src_output.at(head), 0, head,
                    "output:h" + std::to_string(head) + ".model.a_src");
                add(output_dst_label(head),
                    "C", scales.s_a_dst_output.at(head), 0, head,
                    "output:h" + std::to_string(head) + ".model.a_dst");
                add(output_bias_label(head),
                    "C", scales.s_b_output.at(head), 0, head,
                    "output:h" + std::to_string(head) + ".model.bias");
            }
            return entries;
        }

        std::vector<VerifierRegistryEntry> static_registry_entries(
            const ProtocolContext &context)
        {
            const auto degree = context.kzg.manifest.degree_bound;
            if (context.config.range_bits < 1 || context.config.range_bits > 62)
                throw std::runtime_error("verifier registry range_bits must be between 1 and 62");
            const auto range_magnitude =
                std::int64_t{1} << context.config.range_bits;
            const auto range_min = -range_magnitude;
            const auto range_max = -range_min - 1;
            const auto &scales = context.config.quantization;
            const auto table_shape = plan_canonical_static_table_shape(context);
            const auto declared_range = [&](const std::string &label)
                -> std::pair<std::int64_t, std::int64_t>
            {
                if (label == "V_T_exp_x")
                    return {0, static_cast<std::int64_t>(
                                  table_shape.exp_input_bound)};
                if (label == "V_T_exp_y")
                    return {1, static_cast<std::int64_t>(scales.s_exp)};
                if (label == "V_T_range")
                    return {0, static_cast<std::int64_t>(
                                  table_shape.range_table_size - 1U)};
                return {range_min, range_max};
            };
            const std::array<std::pair<const char *, std::uint64_t>, 8> spec = {{
                {"V_T_H", scales.s_input},
                {"V_T_L_x", scales.s_delta},
                {"V_T_L_y", scales.s_delta},
                {"V_T_ELU_x", scales.s_agg_pre.at(0)},
                {"V_T_ELU_y", scales.s_agg.at(0)},
                {"V_T_exp_x", scales.s_delta},
                {"V_T_exp_y", scales.s_exp},
                {"V_T_range", 1},
            }};
            std::vector<VerifierRegistryEntry> entries;
            entries.reserve(spec.size());
            for (const auto &[label, scale] : spec)
            {
                const auto it = context.static_commitments.find(label);
                if (it == context.static_commitments.end())
                    throw std::runtime_error(std::string("verifier registry missing static table commitment: ") + label);
                const auto [declared_min, declared_max] = declared_range(label);
                entries.push_back(VerifierRegistryEntry{
                    .label = label,
                    .domain = std::string(label) == "V_T_H" ? "FH" : "edge",
                    .degree_bound = degree,
                    .scale = scale,
                    .range_min = declared_min,
                    .range_max = declared_max,
                    .point = it->second.point,
                    .bound_certificate = ModelBoundCertificate{}});
            }
            return entries;
        }

        VerifierPublicPolynomialDescriptor describe_public_polynomial(
            const algebra::Polynomial &source,
            const std::string &name)
        {
            if (!source.zero_domain_blinder.empty())
                throw std::runtime_error(
                    "public polynomial unexpectedly contains private blinding material: "
                    + name);
            if (source.basis == algebra::PolynomialBasis::Evaluation)
            {
                if (source.domain == nullptr)
                    throw std::runtime_error(
                        "public evaluation polynomial has no domain: " + name);
                return VerifierPublicPolynomialDescriptor{
                    .name = name,
                    .basis = source.basis,
                    .domain = source.domain->name,
                    .size = source.values().size()};
            }
            return VerifierPublicPolynomialDescriptor{
                .name = name,
                .basis = source.basis,
                .domain = {},
                .size = source.values().size()};
        }

        bool has_complete_public_feature_source(
            const data::GraphDataset &dataset,
            const data::LocalGraph &local)
        {
            if (dataset.num_nodes == 0
                || dataset.num_features == 0
                || local.num_nodes != dataset.num_nodes
                || local.num_features != dataset.num_features
                || local.graph_count != dataset.graph_count
                || local.absolute_ids.size() != dataset.num_nodes
                || local.features.size() != dataset.num_nodes
                || local.public_input.N_total != dataset.num_nodes)
                return false;
            for (std::size_t row = 0; row < dataset.num_nodes; ++row)
            {
                if (local.absolute_ids[row] != row
                    || local.features[row].size() != dataset.num_features)
                    return false;
            }
            return true;
        }

        AuthenticatedVerifierRegistry registry_from_context(
            const ProtocolContext &context)
        {
            if (context.verifier_shape_only)
                throw std::runtime_error(
                    "verifier shape-only context cannot produce a registry");
            if (context.compiler_artifact == nullptr
                || context.validated_compiler_artifact_identity != context.compiler_artifact.get())
                throw std::runtime_error("verifier registry requires an authenticated compiler artifact");
            AuthenticatedVerifierRegistry registry;
            registry.registry_id = context.config.model_param_id;
            registry.model_arch_id = context.config.model_arch_id;
            registry.model_param_id = context.config.model_param_id;
            registry.input_preprocess_id = context.config.input_preprocess_id;
            registry.quant_cfg_id = context.config.quant_cfg_id;
            registry.static_table_id = context.config.static_table_id;
            registry.srs_id = context.kzg.manifest.srs_id;
            registry.protocol_schema_id = context.kzg.manifest.protocol_schema_id;
            registry.instance_schema_hash = sha256_hex(context.compiler_artifact->instance_canonical_bytes);
            registry.srs_manifest_hash = context.kzg.manifest.manifest_hash;
            registry.srs_degree_bound = context.kzg.manifest.degree_bound;
            registry.srs_g1_generator = context.kzg.vk.g1_generator;
            registry.srs_generator_id = context.kzg.manifest.generator_id;
            registry.srs_encoding = context.kzg.manifest.encoding;
            registry.srs_spec_id =
                context.kzg.manifest.srs_type + ":"
                + context.kzg.manifest.kzg_assumption_profile_id;
            registry.srs_protocol_compiler_artifact_hash =
                context.kzg.manifest.protocol_compiler_artifact_hash;
            registry.model_layout_digest =
                sha256_hex(model_layout_serialization(context.model));
            registry.quant_chain_digest =
                sha256_hex(context.config.quantization.canonical_serialization());
            registry.static_table_digest =
                sha256_hex(static_table_serialization(context.tables));
            registry.static_table_generator_id =
                std::string(kStaticTableGeneratorId);
            registry.static_table_spec_hash =
                sha256_hex(static_table_spec_serialization(context));
            registry.range_and_scale_certificate =
                sha256_hex(range_scale_certificate_serialization(context));
            registry.integer_bound_ledger_digest =
                sha256_hex(integer_bound_ledger_canonical_bytes(
                    context.compiler_artifact->integer_bound_ledger));
            registry.integer_bound_ledger =
                context.compiler_artifact->integer_bound_ledger;
            const auto security_budget = evaluate_security_budget(
                context, *context.compiler_artifact);
            if (!security_budget.passed
                || !lower_hex_256(security_budget.ledger_digest)
                || security_budget.exact_budget.empty())
                throw std::runtime_error(
                    "verifier registry requires a valid SecurityBudgetLedger");
            registry.security_budget_ledger_digest =
                security_budget.ledger_digest;
            registry.security_budget_exact = security_budget.exact_budget;
            registry.public_input_digest =
                sha256_hex(public_input_serialization(context.local));
            registry.topology_digest =
                sha256_hex(topology_serialization(context.local));
            registry.model_entries = model_registry_entries(context);
            registry.static_entries = static_registry_entries(context);
            registry.compiler_artifact_digest =
                sha256_hex(protocol_instance_canonical_bytes(
                    *context.compiler_artifact));
            return registry;
        }

        VerifierConfig public_config(const util::AppConfig &config)
        {
            VerifierConfig out;
            out.dataset = config.dataset;
            out.task_type = config.task_type;
            out.report_unit = config.report_unit;
            out.batching_rule = config.batching_rule;
            out.subgraph_rule = config.subgraph_rule;
            out.self_loop_rule = config.self_loop_rule;
            out.edge_sort_rule = config.edge_sort_rule;
            out.chunking_rule = config.chunking_rule;
            out.quant_cfg_id = config.quant_cfg_id;
            out.quantization = config.quantization;
            out.model_arch_id = config.model_arch_id;
            out.model_param_id = config.model_param_id;
            out.input_preprocess_id = config.input_preprocess_id;
            out.static_table_id = config.static_table_id;
            out.degree_bound_id = config.degree_bound_id;
            out.srs_type = config.srs_type;
            out.srs_id = config.srs_id;
            out.privacy_policy = config.privacy_policy;
            out.input_privacy = config.input_privacy;
            out.output_mode = config.output_mode;
            out.opening_mode = config.opening_mode;
            out.binding_mode = config.binding_mode;
            out.proof_task = config.proof_task;
            out.security_profile_id = config.security_profile_id;
            out.num_classes = config.num_classes;
            out.range_bits = config.range_bits;
            out.layer_count = config.layer_count;
            out.K_out = config.K_out;
            out.batch_graphs = config.batch_graphs;
            out.hidden_profile = config.hidden_profile;
            out.d_in_profile = config.d_in_profile;
            return out;
        }

        VerifierModelShape public_model_shape(const model::ModelParameters &model)
        {
            VerifierModelShape out;
            out.model_arch_id = model.model_arch_id;
            out.model_param_id = model.model_param_id;
            out.input_preprocess_id = model.input_preprocess_id;
            out.quant_cfg_id = model.quant_cfg_id;
            out.static_table_id = model.static_table_id;
            out.degree_bound_id = model.degree_bound_id;
            out.L = model.L;
            out.d_in_profile = model.d_in_profile;
            out.hidden_profile = model.hidden_profile;
            out.K_out = model.K_out;
            out.C = model.C;
            out.hidden_layers.reserve(model.hidden_layers.size());
            for (const auto &layer : model.hidden_layers)
            {
                VerifierHiddenLayerShape shape;
                shape.layer_index = layer.layer_index;
                shape.input_dim = layer.input_dim;
                shape.shape = layer.shape;
                shape.heads.resize(layer.heads.size());
                out.hidden_layers.push_back(std::move(shape));
            }
            out.output_layer.input_dim = model.output_layer.input_dim;
            out.output_layer.output_dim = model.output_layer.output_dim;
            out.output_layer.head_count = model.output_layer.head_count;
            out.output_layer.heads.resize(model.output_layer.heads.size());
            return out;
        }

        VerifierSRSMetadata public_srs(const crypto::SRSManifest &manifest)
        {
            VerifierSRSMetadata out;
            out.manifest_format = manifest.manifest_format;
            out.srs_id = manifest.srs_id;
            out.curve_id = manifest.curve_id;
            out.field_modulus_id = manifest.field_modulus_id;
            out.srs_type = manifest.srs_type;
            out.ceremony_transcript_hash = manifest.ceremony_transcript_hash;
            out.degree_bound = manifest.degree_bound;
            out.domain_sizes = manifest.domain_sizes;
            out.domain_generators = manifest.domain_generators;
            out.root_derivation_id = manifest.root_derivation_id;
            out.row_order = manifest.row_order;
            out.protocol_compiler_artifact_hash = manifest.protocol_compiler_artifact_hash;
            out.max_instance_envelope_id = manifest.max_instance_envelope_id;
            out.max_instance_envelope_hash = manifest.max_instance_envelope_hash;
            out.max_instance_envelope = manifest.max_instance_envelope;
            out.proof_schema_template_hash = manifest.proof_schema_template_hash;
            out.observation_ledger_hash = manifest.observation_ledger_hash;
            out.random_identity_ledger_hash = manifest.random_identity_ledger_hash;
            out.security_budget_ledger_hash = manifest.security_budget_ledger_hash;
            out.vk_hash = manifest.vk_hash;
            out.vanishing_g2_hash = manifest.vanishing_g2_hash;
            out.srs_audit_report_hash = manifest.srs_audit_report_hash;
            out.srs_consistency_hash = manifest.srs_consistency_hash;
            out.manifest_hash = manifest.manifest_hash;
            out.generator_id = manifest.generator_id;
            out.encoding = manifest.encoding;
            out.protocol_schema_id = manifest.protocol_schema_id;
            out.kzg_assumption_profile_id = manifest.kzg_assumption_profile_id;
            out.kzg_assumption_degree_bound = manifest.kzg_assumption_degree_bound;
            out.kzg_assumption_security_bits = manifest.kzg_assumption_security_bits;
            out.kzg_assumption_domain_catalog_hash = manifest.kzg_assumption_domain_catalog_hash;
            out.domain_generator_catalog_hash = manifest.domain_generator_catalog_hash;
            out.security_level = manifest.security_level;
            return out;
        }

        void copy_point_map(
            const std::unordered_map<std::string, crypto::Commitment> &source,
            std::unordered_map<std::string, VerifierPointCommitment> *target,
            bool reject_opening_material)
        {
            target->reserve(source.size());
            for (const auto &[name, commitment] : source)
            {
                if (commitment.name != name
                    || (reject_opening_material
                        && commitment.opening_polynomial != nullptr))
                    throw std::runtime_error("prover material cannot enter VerifierContext");
                target->emplace(name, VerifierPointCommitment{name, commitment.point});
            }
        }

        struct RegistryEntrySpec
        {
            std::string label;
            std::string domain;
            std::uint64_t scale = 0;
            bool has_declared_range = false;
            std::int64_t range_min = 0;
            std::int64_t range_max = 0;
        };

        std::vector<RegistryEntrySpec> verifier_model_entry_specs(
            const VerifierContext &context)
        {
            std::vector<RegistryEntrySpec> specs;
            for (std::size_t layer = 0; layer < context.model.hidden_layers.size(); ++layer)
            {
                const auto &layer_shape = context.model.hidden_layers.at(layer);
                const auto head_count = layer_shape.heads.size();
                for (std::size_t head = 0; head < head_count; ++head)
                {
                    const auto weight_scale = context.config.quantization.s_w_hidden.at(layer).at(head);
                    for (std::size_t column = 0;
                         column < layer_shape.shape.head_dim;
                         ++column)
                    {
                        specs.push_back({
                            hidden_weight_column_label(layer, head, column),
                            "in",
                            weight_scale});
                    }
                    specs.push_back({
                        hidden_src_label(layer, head),
                        "d",
                        context.config.quantization.s_a_src_hidden.at(layer).at(head)});
                    specs.push_back({
                        hidden_dst_label(layer, head),
                        "d",
                        context.config.quantization.s_a_dst_hidden.at(layer).at(head)});
                }
            }
            for (std::size_t head = 0; head < context.model.output_layer.heads.size(); ++head)
            {
                for (std::size_t column = 0;
                     column < context.model.output_layer.output_dim;
                     ++column)
                {
                    specs.push_back({
                        output_weight_column_label(head, column),
                        "cat",
                        context.config.quantization.s_w_output.at(head)});
                }
                specs.push_back({
                    output_src_label(head),
                    "C",
                    context.config.quantization.s_a_src_output.at(head)});
                specs.push_back({
                    output_dst_label(head),
                    "C",
                    context.config.quantization.s_a_dst_output.at(head)});
                specs.push_back({
                    output_bias_label(head),
                    "C",
                    context.config.quantization.s_b_output.at(head)});
            }
            return specs;
        }

        std::vector<RegistryEntrySpec> verifier_static_entry_specs(
            const VerifierContext &context)
        {
            const auto &scales = context.config.quantization;
            if (context.tables.exp.empty())
                throw std::runtime_error(
                    "verifier registry requires a non-empty ExpMap table");
            const auto exp_input_bound = context.tables.exp.back().first.value();
            const auto exp_output_bound = context.tables.exp.front().second.value();
            if (exp_input_bound
                    > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())
                || exp_output_bound
                    > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
                throw std::runtime_error(
                    "verifier registry ExpMap range exceeds signed metadata capacity");
            if (context.config.range_bits == 0 || context.config.range_bits >= 63)
                throw std::runtime_error(
                    "verifier registry range_bits must be in [1,62]");
            const auto range_size = std::uint64_t{1} << context.config.range_bits;
            return {
                {"V_T_H", "FH", scales.s_input},
                {"V_T_L_x", "edge", scales.s_delta},
                {"V_T_L_y", "edge", scales.s_delta},
                {"V_T_ELU_x", "edge", scales.s_agg_pre.at(0)},
                {"V_T_ELU_y", "edge", scales.s_agg.at(0)},
                {"V_T_exp_x", "edge", scales.s_delta, true, 0,
                 static_cast<std::int64_t>(exp_input_bound)},
                {"V_T_exp_y", "edge", scales.s_exp, true, 1,
                 static_cast<std::int64_t>(exp_output_bound)},
                {"V_T_range", "edge", 1, true, 0,
                 static_cast<std::int64_t>(range_size - 1U)},
            };
        }

        bool validate_registry_entries(
            const VerifierContext &context,
            const std::vector<VerifierRegistryEntry> &entries,
            const std::vector<RegistryEntrySpec> &specs,
            std::unordered_set<std::string> *all_labels,
            bool require_bound_certificate)
        {
            if (entries.size() != specs.size())
                return false;
            const auto range_bits = context.config.range_bits;
            if (range_bits >= 63)
                return false;
            const auto range_magnitude = std::int64_t{1} << range_bits;
            const auto expected_min = -range_magnitude;
            const auto expected_max = range_magnitude - 1;
            for (std::size_t index = 0; index < entries.size(); ++index)
            {
                const auto &entry = entries[index];
                const auto &spec = specs[index];
                const auto declared_min = spec.has_declared_range
                    ? spec.range_min
                    : expected_min;
                const auto declared_max = spec.has_declared_range
                    ? spec.range_max
                    : expected_max;
                if (entry.label != spec.label || entry.domain != spec.domain
                    || entry.degree_bound != context.srs.degree_bound
                    || entry.scale != spec.scale
                    || entry.range_min != declared_min
                    || entry.range_max != declared_max
                    || entry.label.empty()
                    || !all_labels->insert(entry.label).second)
                    return false;
                const auto &certificate = entry.bound_certificate;
                if (require_bound_certificate)
                {
                    if (!certificate.present
                        || certificate.label != entry.label
                        || certificate.domain != entry.domain
                        || certificate.degree_bound != entry.degree_bound
                        || certificate.scale != entry.scale)
                        return false;
                }
                else if (certificate.present)
                    return false;
                const auto commitment = context.static_commitments.find(entry.label);
                if (commitment == context.static_commitments.end()
                    || commitment->second.name != entry.label
                    || commitment->second.point != entry.point)
                    return false;
            }
            return true;
        }

        bool validate_integer_bound_ledger(const VerifierContext &context)
        {
            if (context.compiler_artifact == nullptr
                || context.registry.integer_bound_ledger.empty()
                || context.registry.integer_bound_ledger
                       != context.compiler_artifact->integer_bound_ledger)
                return false;
            try
            {
                // Shape-only validation is not sufficient.  Rebuild every
                // recurrence from public input/topology, authenticated tables,
                // the registered scale chain and point-only model certificates.
                return rebuild_integer_bound_ledger(context)
                    == context.registry.integer_bound_ledger;
            }
            catch (const std::exception &)
            {
                return false;
            }
        }

        std::string canonical_field_hex(const FieldElement &value)
        {
            std::array<std::uint8_t, 32> bytes{};
            if (value.write_little_endian(bytes.data(), bytes.size()) != bytes.size())
                throw std::runtime_error("failed to encode canonical SRS field value");
            static constexpr char hex[] = "0123456789abcdef";
            std::string out;
            out.reserve(bytes.size() * 2);
            for (const auto byte : bytes)
            {
                out.push_back(hex[(byte >> 4U) & 0x0fU]);
                out.push_back(hex[byte & 0x0fU]);
            }
            return out;
        }

        std::string point_hex(const crypto::G2Point &point)
        {
            const auto bytes = crypto::serialize(point);
            std::ostringstream out;
            out << std::hex << std::setfill('0');
            for (const auto byte : bytes)
                out << std::setw(2) << static_cast<unsigned int>(byte);
            return out.str();
        }

        std::string srs_domain_catalog_hash(
            const VerifierSRSMetadata &srs)
        {
            std::vector<std::pair<std::string, std::size_t>> ordered(
                srs.domain_sizes.begin(), srs.domain_sizes.end());
            std::sort(ordered.begin(), ordered.end());
            std::ostringstream canonical;
            canonical << "zkgat:domain-size-catalog";
            for (const auto &[name, size] : ordered)
                canonical << '|' << name << '=' << size;
            return sha256_hex(canonical.str());
        }

        std::string srs_generator_catalog_hash(
            const VerifierSRSMetadata &srs)
        {
            std::vector<std::pair<std::string, std::size_t>> ordered(
                srs.domain_sizes.begin(), srs.domain_sizes.end());
            std::sort(ordered.begin(), ordered.end());
            std::ostringstream canonical;
            canonical << "zkgat:domain-generator-catalog"
                      << "|root_derivation_id=" << srs.root_derivation_id
                      << "|row_order=" << srs.row_order;
            for (const auto &[name, size] : ordered)
            {
                const auto it = srs.domain_generators.find(name);
                if (it == srs.domain_generators.end())
                    throw std::runtime_error("SRS domain generator catalogue is incomplete");
                canonical << '|' << name << ':' << size << ':' << it->second;
            }
            return sha256_hex(canonical.str());
        }

        std::string verifier_key_bytes(const crypto::KZGVerifierKey &vk)
        {
            std::string out = "GATZK-VK\n";
            const auto append_point = [&](const auto &point)
            {
                const auto bytes = crypto::serialize(point);
                append_u64(&out, bytes.size());
                out.append(
                    reinterpret_cast<const char *>(bytes.data()), bytes.size());
            };
            append_point(vk.g2_one);
            append_point(vk.g2_tau);
            std::vector<std::string> names;
            names.reserve(vk.domain_zero_g2.size());
            for (const auto &[name, _] : vk.domain_zero_g2)
            {
                (void)_;
                names.push_back(name);
            }
            std::sort(names.begin(), names.end());
            append_u64(&out, names.size());
            for (const auto &name : names)
            {
                append_string(&out, name);
                append_point(vk.domain_zero_g2.at(name));
            }
            return out;
        }

        std::string vanishing_g2_catalog_hash(
            const VerifierSRSMetadata &srs,
            const crypto::KZGVerifierKey &vk)
        {
            std::vector<std::pair<std::string, std::size_t>> ordered(
                srs.domain_sizes.begin(), srs.domain_sizes.end());
            std::sort(ordered.begin(), ordered.end());
            std::ostringstream canonical;
            canonical << "zkgat:vanishing-g2-catalog";
            for (const auto &[name, size] : ordered)
            {
                const auto it = vk.domain_zero_g2.find(name);
                if (it == vk.domain_zero_g2.end())
                    throw std::runtime_error("SRS VK domain-zero catalogue is incomplete");
                canonical << '|' << name << ':' << size << ':' << point_hex(it->second);
            }
            return sha256_hex(canonical.str());
        }

        bool validate_domain_shape(const VerifierContext &context)
        {
            const std::array<std::pair<std::string_view,
                                       const std::shared_ptr<algebra::RootOfUnityDomain> *>,
                             7>
                required = {{
                    {"FH", &context.domains.fh},
                    {"edge", &context.domains.edge},
                    {"in", &context.domains.in},
                    {"d", &context.domains.d},
                    {"cat", &context.domains.cat},
                    {"C", &context.domains.c},
                    {"N", &context.domains.n},
                }};
            if (context.srs.domain_sizes.size() != required.size()
                || context.srs.domain_generators.size() != required.size())
                return false;
            for (const auto &[name, domain_ptr] : required)
            {
                if (*domain_ptr == nullptr || (*domain_ptr)->name != name
                    || (*domain_ptr)->size == 0
                    || !context.srs.domain_sizes.contains(std::string(name))
                    || context.srs.domain_sizes.at(std::string(name)) != (*domain_ptr)->size)
                    return false;
            }
            if (context.model.hidden_layers.size() + 1 != context.model.L
                || context.domains.hidden_layers.size()
                       != context.model.hidden_layers.size()
                || !context.domains.output_head.has_value())
                return false;
            const auto same_domain = [](const auto &actual, const auto &expected)
            {
                return actual != nullptr && expected != nullptr
                    && actual->name == expected->name
                    && actual->size == expected->size
                    && actual->omega == expected->omega
                    && actual->inv_size == expected->inv_size;
            };
            for (std::size_t layer = 0;
                 layer < context.model.hidden_layers.size(); ++layer)
            {
                const auto &shape = context.model.hidden_layers.at(layer);
                const auto &head_domains = context.domains.hidden_layers.at(layer);
                if (shape.layer_index != layer
                    || head_domains.size() != shape.heads.size()
                    || shape.heads.size() != shape.shape.head_count)
                    return false;
                for (const auto &head : head_domains)
                {
                    if (!same_domain(head.in, context.domains.in)
                        || !same_domain(head.d, context.domains.d))
                        return false;
                }
            }
            if (!same_domain(context.domains.output_head->in, context.domains.cat)
                || !same_domain(context.domains.output_head->d, context.domains.c))
                return false;
            const auto allowed_names = std::unordered_set<std::string>{
                "FH", "edge", "in", "d", "cat", "C", "N"};
            for (const auto &[name, size] : context.srs.domain_sizes)
            {
                if (!allowed_names.contains(name) || size == 0)
                    return false;
            }
            return true;
        }

        bool validate_model_shape(const VerifierContext &context)
        {
            const auto &model = context.model;
            const auto &config = context.config;
            bool profile_matches = model.hidden_profile.size() == config.hidden_profile.size();
            for (std::size_t index = 0; profile_matches && index < model.hidden_profile.size(); ++index)
            {
                profile_matches =
                    model.hidden_profile.at(index).head_count
                        == config.hidden_profile.at(index).head_count
                    && model.hidden_profile.at(index).head_dim
                           == config.hidden_profile.at(index).head_dim;
            }
            if (model.L < 2
                || model.L != config.layer_count
                || model.input_preprocess_id.empty()
                || model.input_preprocess_id != config.input_preprocess_id
                || model.L != model.hidden_layers.size() + 1
                || model.hidden_layers.size() != config.hidden_profile.size()
                || !profile_matches
                || model.d_in_profile != config.d_in_profile
                || model.d_in_profile.size() != model.hidden_layers.size()
                || model.K_out != config.K_out
                || model.K_out != model.output_layer.head_count
                || model.output_layer.heads.size() != model.K_out
                || model.C != config.num_classes
                || model.C != model.output_layer.output_dim
                || context.local.num_features != model.d_in_profile.front()
                || context.local.num_classes != model.C)
                return false;
            std::size_t expected_input = 0;
            for (std::size_t layer_index = 0;
                 layer_index < model.hidden_layers.size(); ++layer_index)
            {
                const auto &layer = model.hidden_layers.at(layer_index);
                const auto &profile = model.hidden_profile.at(layer_index);
                if (layer.layer_index != layer_index
                    || layer.input_dim != model.d_in_profile.at(layer_index)
                    || layer.input_dim != (layer_index == 0
                                               ? context.local.num_features
                                               : expected_input)
                    || layer.shape.head_count != profile.head_count
                    || layer.shape.head_dim != profile.head_dim
                    || layer.shape.head_count == 0
                    || layer.shape.head_dim == 0
                    || layer.heads.size() != layer.shape.head_count)
                    return false;
                expected_input = layer.shape.head_count * layer.shape.head_dim;
            }
            return model.output_layer.input_dim == expected_input
                && model.output_layer.head_count == model.K_out
                && model.output_layer.output_dim == model.C;
        }

        bool validate_dataset_and_normalized_input(const VerifierContext &context)
        {
            const auto &dataset = context.dataset;
            const auto &local = context.local;
            if (dataset.name.empty() || local.name.empty()
                || local.name != dataset.name + "_normalized"
                || dataset.input_preprocess_id.empty()
                || local.input_preprocess_id != dataset.input_preprocess_id
                || context.config.input_preprocess_id != dataset.input_preprocess_id
                || context.config.input_preprocess_id
                       != util::canonical_input_preprocess_id(context.config.dataset)
                || dataset.num_nodes == 0
                || dataset.num_features == 0
                || dataset.num_classes == 0
                || dataset.graph_count == 0
                || dataset.node_ptr.size() != dataset.graph_count + 1
                || dataset.edge_ptr.size() != dataset.graph_count + 1
                || dataset.node_ptr.front() != 0
                || dataset.node_ptr.back() != dataset.num_nodes
                || dataset.edge_ptr.front() != 0
                || dataset.edge_ptr.back() != dataset.edges.size()
                || local.num_nodes == 0
                || local.num_nodes > dataset.num_nodes
                || local.num_features != dataset.num_features
                || local.num_classes != dataset.num_classes
                || local.graph_count == 0
                || local.graph_count > dataset.graph_count
                || local.graph_count
                       != std::min(
                           std::max<std::size_t>(1, context.config.batch_graphs),
                           dataset.graph_count)
                || local.absolute_ids.size() != local.num_nodes
                || local.features.size() != local.num_nodes
                || local.node_ptr.size() != local.graph_count + 1
                || local.edge_ptr.size() != local.graph_count + 1
                || local.node_ptr.front() != 0
                || local.node_ptr.back() != local.num_nodes
                || local.edge_ptr.front() != 0
                || local.edge_ptr.back() != local.edges.size()
                || local.public_input.N_total != dataset.num_nodes
                || local.public_input.N != local.num_nodes
                || local.public_input.E != local.edges.size()
                || local.public_input.G_batch != local.graph_count
                || local.public_input.I != local.absolute_ids
                || local.public_input.src.size() != local.edges.size()
                || local.public_input.dst.size() != local.edges.size())
                return false;
            for (std::size_t index = 1; index < dataset.node_ptr.size(); ++index)
                if (dataset.node_ptr[index - 1] >= dataset.node_ptr[index])
                    return false;
            for (std::size_t index = 1; index < dataset.edge_ptr.size(); ++index)
                if (dataset.edge_ptr[index - 1] >= dataset.edge_ptr[index])
                    return false;
            std::unordered_set<std::size_t> absolute_ids;
            absolute_ids.reserve(local.absolute_ids.size());
            for (const auto id : local.absolute_ids)
                if (id >= dataset.num_nodes || !absolute_ids.insert(id).second)
                    return false;
            for (const auto &row : local.features)
                if (row.size() != local.num_features)
                    return false;
            for (const auto &edge : dataset.edges)
            {
                if (edge.graph_id >= dataset.graph_count
                    || edge.src < dataset.node_ptr.at(edge.graph_id)
                    || edge.src >= dataset.node_ptr.at(edge.graph_id + 1)
                    || edge.dst < dataset.node_ptr.at(edge.graph_id)
                    || edge.dst >= dataset.node_ptr.at(edge.graph_id + 1))
                    return false;
            }
            for (std::size_t graph = 0; graph < local.graph_count; ++graph)
            {
                const auto local_begin = local.node_ptr.at(graph);
                const auto local_end = local.node_ptr.at(graph + 1);
                const auto dataset_begin = dataset.node_ptr.at(graph);
                const auto dataset_end = dataset.node_ptr.at(graph + 1);
                if (local_end - local_begin != dataset_end - dataset_begin)
                    return false;
                for (std::size_t index = 0; index < local_end - local_begin; ++index)
                    if (local.absolute_ids.at(local_begin + index)
                        != dataset_begin + index)
                        return false;
                const auto edge_begin = local.edge_ptr.at(graph);
                const auto edge_end = local.edge_ptr.at(graph + 1);
                for (std::size_t edge_index = edge_begin;
                     edge_index < edge_end;
                     ++edge_index)
                {
                    const auto &edge = local.edges.at(edge_index);
                    if (edge.graph_id != graph
                        || edge.src < local_begin
                        || edge.src >= local_end
                        || edge.dst < local_begin
                        || edge.dst >= local_end
                        || local.public_input.src.at(edge_index) != edge.src
                        || local.public_input.dst.at(edge_index) != edge.dst)
                        return false;
                }
            }
            try
            {
                data::validate_normalized_graph_input(local);

                // Re-run the canonical public-input normalizer against the
                // decoded dataset/topology.  The inexpensive structural
                // checks above establish local consistency; this second pass
                // establishes that local is actually the deterministic
                // normalization of dataset (including deduplication, loops,
                // local ids, stable destination ordering and all pointer
                // cardinalities).  Raw feature provenance is intentionally
                // absent from a verifier artifact, so use the already public
                // quantized feature matrix only for the normalizer's shape
                // lookup and compare the resulting public graph structure.
                auto normalization_dataset = dataset;
                normalization_dataset.features_fp.assign(
                    dataset.num_nodes,
                    std::vector<double>(dataset.num_features, 0.0));
                if (dataset.features.size() < local.num_nodes)
                    return false;
                normalization_dataset.features = dataset.features;
                util::AppConfig normalization_config;
                normalization_config.task_type = context.config.task_type;
                normalization_config.report_unit = context.config.report_unit;
                normalization_config.batching_rule = context.config.batching_rule;
                normalization_config.subgraph_rule = context.config.subgraph_rule;
                normalization_config.self_loop_rule = context.config.self_loop_rule;
                normalization_config.edge_sort_rule = context.config.edge_sort_rule;
                normalization_config.chunking_rule = context.config.chunking_rule;
                normalization_config.batch_graphs = context.config.batch_graphs;
                normalization_config.input_preprocess_id =
                    context.config.input_preprocess_id;
                // These are the only accepted mainline normalization
                // semantics and are deliberately not caller-controlled by a
                // verifier artifact that lacks the corresponding raw-data
                // options.
                normalization_config.deduplicate_edges = true;
                normalization_config.symmetrize_edges = false;
                const auto normalized = data::normalize_graph_input(
                    normalization_dataset, normalization_config);
                const auto same_edge = [](const data::Edge &lhs,
                                          const data::Edge &rhs)
                {
                    return lhs.src == rhs.src && lhs.dst == rhs.dst
                        && lhs.graph_id == rhs.graph_id
                        && lhs.stable_index == rhs.stable_index;
                };
                if (normalized.name != local.name
                    || normalized.num_nodes != local.num_nodes
                    || normalized.num_features != local.num_features
                    || normalized.num_classes != local.num_classes
                    || normalized.graph_count != local.graph_count
                    || normalized.absolute_ids != local.absolute_ids
                    || normalized.node_ptr != local.node_ptr
                    || normalized.edge_ptr != local.edge_ptr
                    || normalized.public_input.N_total
                           != local.public_input.N_total
                    || normalized.public_input.N != local.public_input.N
                    || normalized.public_input.E != local.public_input.E
                    || normalized.public_input.G_batch
                           != local.public_input.G_batch
                    || normalized.public_input.I != local.public_input.I
                    || normalized.public_input.src != local.public_input.src
                    || normalized.public_input.dst != local.public_input.dst
                    || normalized.public_input.node_ptr
                           != local.public_input.node_ptr
                    || normalized.public_input.edge_ptr
                           != local.public_input.edge_ptr
                    || normalized.edges.size() != local.edges.size()
                    || !std::equal(
                        normalized.edges.begin(), normalized.edges.end(),
                        local.edges.begin(), same_edge))
                    return false;
            }
            catch (const std::exception &)
            {
                return false;
            }
            return true;
        }

        bool validate_srs_metadata(const VerifierContext &context)
        {
            const auto &srs = context.srs;
            if (srs.manifest_format != "zkgat_srs_manifest"
                || srs.curve_id != "bls12_381"
                || srs.field_modulus_id != "bls12_381_fr"
                || srs.srs_type != "trusted_srs"
                || srs.generator_id != "zkgat_srs_setup_tool"
                || srs.encoding != "canonical_mcl_bls12_381_fixed"
                || srs.root_derivation_id != "bls12_381_fr_generator_5"
                || srs.row_order != "ascending_powers_from_one"
                || srs.security_level != "120-bit"
                || srs.kzg_assumption_profile_id != "shifted_kzg_divisibility"
                || srs.degree_bound == 0
                || srs.kzg_assumption_degree_bound != srs.degree_bound
                || srs.kzg_assumption_security_bits != 128
                || srs.domain_sizes.size() != 7
                || srs.domain_sizes.size() != srs.domain_generators.size()
                || srs.max_instance_envelope.empty()
                || srs.max_instance_envelope_id.empty()
                || srs.max_instance_envelope_id != srs.max_instance_envelope_hash
                || !lower_hex_256(srs.manifest_hash)
                || !lower_hex_256(srs.ceremony_transcript_hash)
                || !lower_hex_256(srs.protocol_compiler_artifact_hash)
                || !lower_hex_256(srs.protocol_schema_id)
                || !lower_hex_256(srs.max_instance_envelope_id)
                || !lower_hex_256(srs.proof_schema_template_hash)
                || !lower_hex_256(srs.observation_ledger_hash)
                || !lower_hex_256(srs.random_identity_ledger_hash)
                || !lower_hex_256(srs.security_budget_ledger_hash)
                || !lower_hex_256(srs.vk_hash)
                || !lower_hex_256(srs.vanishing_g2_hash)
                || !lower_hex_256(srs.srs_audit_report_hash)
                || !lower_hex_256(srs.srs_consistency_hash)
                || srs.srs_audit_report_hash != srs.srs_consistency_hash
                || !lower_hex_256(srs.kzg_assumption_domain_catalog_hash)
                || !lower_hex_256(srs.domain_generator_catalog_hash))
                return false;
            if (srs.kzg_assumption_domain_catalog_hash
                    != srs_domain_catalog_hash(srs)
                || srs.domain_generator_catalog_hash
                       != srs_generator_catalog_hash(srs))
                return false;

            const std::array<std::pair<std::string, std::shared_ptr<algebra::RootOfUnityDomain>>, 7>
                required_domains = {{
                    {"FH", context.domains.fh},
                    {"edge", context.domains.edge},
                    {"in", context.domains.in},
                    {"d", context.domains.d},
                    {"cat", context.domains.cat},
                    {"C", context.domains.c},
                    {"N", context.domains.n},
                }};
            for (const auto &[name, domain] : required_domains)
            {
                if (domain == nullptr || domain->name != name || domain->size == 0
                    || !srs.domain_sizes.contains(name)
                    || srs.domain_sizes.at(name) != domain->size
                    || !srs.domain_generators.contains(name)
                    || srs.domain_generators.at(name) != canonical_field_hex(domain->omega))
                    return false;
            }
            if (context.kzg_vk.domain_zero_g2.size() != srs.domain_sizes.size())
                return false;
            for (const auto &[name, size] : srs.domain_sizes)
            {
                if (size == 0 || size > srs.degree_bound
                    || !context.kzg_vk.domain_zero_g2.contains(name)
                    || context.kzg_vk.domain_zero_g2.at(name).value.isZero())
                    return false;
            }
            if (context.kzg_vk.g2_one.value.isZero()
                || context.kzg_vk.g2_tau.value.isZero()
                || context.kzg_vk.g1_generator.value.isZero()
                || srs.vanishing_g2_hash
                       != vanishing_g2_catalog_hash(srs, context.kzg_vk)
                || srs.vk_hash != sha256_hex(verifier_key_bytes(context.kzg_vk)))
                return false;
            return true;
        }

        bool validate_public_polynomial_descriptors(const VerifierContext &context)
        {
            const auto domain_for = [&](const std::string &name)
                -> std::shared_ptr<algebra::RootOfUnityDomain>
            {
                for (const auto &domain : {
                         context.domains.fh, context.domains.edge,
                         context.domains.in, context.domains.d,
                         context.domains.cat, context.domains.c,
                         context.domains.n})
                {
                    if (domain != nullptr && domain->name == name)
                        return domain;
                }
                return nullptr;
            };
            std::unordered_map<std::string, VerifierPublicPolynomialDescriptor> expected;
            const auto add_expected = [&](const std::string &name,
                                          const std::string &domain_name)
            {
                const auto domain = domain_for(domain_name);
                if (domain == nullptr)
                    throw std::runtime_error("missing public polynomial descriptor domain");
                expected.emplace(
                    name,
                    VerifierPublicPolynomialDescriptor{
                        .name = name,
                        .basis = algebra::PolynomialBasis::Evaluation,
                        .domain = domain_name,
                        .size = domain->size});
            };
            add_expected("P_I", "N");
            add_expected("P_src", "edge");
            add_expected("P_dst", "edge");
            add_expected("P_Q_new_edge", "edge");
            add_expected("P_Q_end_edge", "edge");
            add_expected("P_node_gid", "N");
            add_expected("P_edge_gid", "edge");
            add_expected("P_Q_graph_new_edge", "edge");
            add_expected("P_Q_graph_end_edge", "edge");
            add_expected("P_Q_grp_new_edge", "edge");
            add_expected("P_Q_grp_end_edge", "edge");
            add_expected("P_Q_edge_valid", "edge");
            for (std::size_t layer = 0; layer < context.model.hidden_layers.size(); ++layer)
            {
                add_expected(hidden_q_in_selector_label(layer), "in");
                add_expected(hidden_q_d_h_selector_label(layer), "d");
                add_expected(hidden_q_cat_selector_label(layer), "cat");
                add_expected(hidden_q_nd_h_selector_label(layer), "edge");
            }
            for (std::size_t head = 0; head < context.model.output_layer.heads.size(); ++head)
                add_expected("P_Q_div_out" + std::to_string(head), "edge");
            add_expected("P_Q_div_avg", "edge");
            add_expected("P_Q_N", "N");
            add_expected("P_AggBind_m_src", "N");
            add_expected("P_AggBind_m_dst", "N");
            add_expected("P_AggBind_index_N", "N");
            add_expected(output_q_cat_selector_label(), "cat");
            add_expected("P_Q_C_valid", "C");
            add_expected("P_BiasIndex_C", "C");
            add_expected("P_BiasMultiplicity_C", "C");
            add_expected("P_BiasIndex_edge", "edge");
            const auto chunks = coefficient_chunk_layout(
                context.local.num_classes + 3, context.domains.edge->size);
            for (std::size_t head = 0; head < context.model.output_layer.heads.size(); ++head)
                for (const auto &chunk : chunks)
                    for (const auto region : {"active", "semantic", "gap", "tail"})
                        add_expected(
                            output_bias_basis_region_selector_label(
                                head, chunk.chunk_index, region),
                            "edge");
            add_expected("P_Q_tbl_feat", "FH");
            add_expected("P_Row_feat_tbl", "FH");
            add_expected("P_Col_feat_tbl", "FH");
            add_expected("P_Q_qry_feat", "FH");
            add_expected("P_Row_feat_qry", "FH");
            add_expected("P_Col_feat_qry", "FH");
            add_expected("P_I_feat_qry", "FH");
            add_expected("P_T_H", "FH");
            add_expected("P_Q_qry_src", "edge");
            add_expected("P_Q_qry_dst", "edge");
            add_expected("P_Q_tbl_L", "edge");
            add_expected("P_Q_qry_L", "edge");
            add_expected("P_Q_tbl_R", "edge");
            add_expected("P_Q_qry_R", "edge");
            add_expected("P_Q_tbl_exp", "edge");
            add_expected("P_Q_qry_exp", "edge");
            add_expected("P_Q_tbl_ELU", "edge");
            add_expected("P_T_L_x", "edge");
            add_expected("P_T_L_y", "edge");
            add_expected("P_T_ELU_x", "edge");
            add_expected("P_T_ELU_y", "edge");
            add_expected("P_T_exp_x", "edge");
            add_expected("P_T_exp_y", "edge");
            add_expected("P_T_range", "edge");

            if (context.public_polynomial_descriptors.size() != expected.size()
                || context.public_commitments.size() != expected.size())
                return false;
            for (const auto &[name, expected_descriptor] : expected)
            {
                const auto descriptor = context.public_polynomial_descriptors.find(name);
                const auto commitment = context.public_commitments.find(name);
                if (descriptor == context.public_polynomial_descriptors.end()
                    || descriptor->second != expected_descriptor
                    || commitment == context.public_commitments.end()
                    || commitment->second.name != name)
                    return false;
            }
            for (const auto &[name, _] : context.public_polynomial_descriptors)
                if (!expected.contains(name))
                    return false;
            return true;
        }

        bool validate_point_map(
            const std::unordered_map<std::string, VerifierPointCommitment> &points)
        {
            for (const auto &[name, commitment] : points)
            {
                if (name.empty() || commitment.name != name)
                    return false;
            }
            return true;
        }

        bool validate_registry_shape(const VerifierContext &context)
        {
            if (context.config.range_bits < 1 || context.config.range_bits > 62)
                return false;
            // The registry is a formal mainline input, not a generic proof
            // viewer.  Re-resolve the one registered typed scale chain and
            // enforce the privacy/protocol policy at the loader boundary so a
            // self-consistent artifact cannot downgrade model privacy or
            // switch away from the formal proof mode.
            if (context.config.srs_type != "trusted_srs"
                || context.config.privacy_policy != "model_and_trace_hidden"
                || context.config.input_privacy
                       != "public_graph_public_features"
                || context.config.output_mode != "public_logits"
                || context.config.opening_mode
                       != "point_grouped_batch_opening"
                || context.config.binding_mode != "zk_subproof_for_crpc"
                || context.config.proof_task
                       != "node_classification_full_gat"
                || context.config.security_profile_id
                       != "zkgat_secure_profile")
                return false;
            try
            {
                const auto canonical_scales =
                    util::canonical_quantization_scale_chain(
                        context.config.quant_cfg_id,
                        context.config.hidden_profile,
                        context.config.K_out);
                if (context.config.quantization.canonical_serialization()
                    != canonical_scales.canonical_serialization())
                    return false;
            }
            catch (const std::exception &)
            {
                return false;
            }
            if (!validate_srs_metadata(context)
                || !validate_public_polynomial_descriptors(context)
                || !validate_point_map(context.public_commitments)
                || !validate_point_map(context.static_commitments)
                || !validate_model_shape(context)
                || !validate_domain_shape(context)
                || !validate_dataset_and_normalized_input(context))
                return false;
            if (!context.dataset.features_fp.empty()
                || !context.dataset.labels.empty()
                || !context.dataset.labels_multi.empty()
                || !context.local.features_fp.empty()
                || !context.local.labels.empty()
                || !context.local.labels_multi.empty()
                || context.local.features.empty()
                || context.local.features.size() != context.local.num_nodes
                || context.local.absolute_ids.size() != context.local.num_nodes)
                return false;
            if (!has_complete_public_feature_source(
                    context.dataset, context.local))
                return false;
            for (const auto &row : context.local.features)
                if (row.size() != context.local.num_features)
                    return false;
            if (context.dataset.name.empty()
                || context.local.name.empty()
                || context.config.dataset != context.dataset.name
                || context.config.input_preprocess_id.empty()
                || context.config.input_preprocess_id
                       != util::canonical_input_preprocess_id(context.config.dataset)
                || context.dataset.input_preprocess_id
                       != context.config.input_preprocess_id
                || context.local.input_preprocess_id
                       != context.config.input_preprocess_id
                || context.dataset.task_type != context.config.task_type
                || context.dataset.report_unit != context.config.report_unit
                || context.local.task_type != context.config.task_type
                || context.local.report_unit != context.config.report_unit
                || context.local.batching_rule != context.config.batching_rule
                || context.local.subgraph_rule != context.config.subgraph_rule
                || context.local.self_loop_rule != context.config.self_loop_rule
                || context.local.edge_sort_rule != context.config.edge_sort_rule
                || context.local.chunking_rule != context.config.chunking_rule)
                return false;
            if (context.local.public_input.N_total != context.dataset.num_nodes
                || context.local.public_input.N != context.local.num_nodes
                || context.local.public_input.E != context.local.edges.size()
                || context.local.public_input.I.size()
                       != context.local.public_input.N
                || context.local.public_input.src.size()
                       != context.local.public_input.E
                || context.local.public_input.dst.size()
                       != context.local.public_input.E
                || context.local.public_input.node_ptr.size()
                       != context.local.graph_count + 1
                || context.local.public_input.edge_ptr.size()
                       != context.local.graph_count + 1)
                return false;
            for (const auto &edge : context.local.edges)
                if (edge.src >= context.local.num_nodes
                    || edge.dst >= context.local.num_nodes)
                    return false;
            if (context.compiler_artifact == nullptr
                || sha256_hex(context.compiler_artifact->protocol_canonical_bytes)
                       != context.srs.protocol_schema_id
                || sha256_hex(protocol_proof_schema_template_canonical_bytes())
                       != context.srs.proof_schema_template_hash
                || sha256_hex(protocol_observation_ledger_template_canonical_bytes())
                       != context.srs.observation_ledger_hash
                || sha256_hex(protocol_random_identity_ledger_template_canonical_bytes())
                       != context.srs.random_identity_ledger_hash
                || sha256_hex(protocol_security_budget_ledger_template_canonical_bytes())
                       != context.srs.security_budget_ledger_hash)
                return false;
            if (!lower_hex_256(context.trusted_registry_digest)
                || context.trusted_registry_digest
                       != context.authenticated_registry_digest)
                return false;
            if (context.srs.srs_type != "trusted_srs"
                || context.kzg_vk.srs_id != context.srs.srs_id
                || context.kzg_vk.srs_type != context.srs.srs_type
                || context.kzg_vk.degree_bound != context.srs.degree_bound
                || context.kzg_vk.manifest_hash != context.srs.manifest_hash)
                return false;
            if (context.registry.format != "zkgat_verifier_registry"
                || context.registry.registry_id.empty()
                || context.registry.registry_id != context.config.model_param_id
                || context.registry.model_arch_id != context.config.model_arch_id
                || context.registry.model_param_id != context.config.model_param_id
                || context.registry.input_preprocess_id.empty()
                || context.registry.input_preprocess_id
                       != context.config.input_preprocess_id
                || context.registry.quant_cfg_id != context.config.quant_cfg_id
                || context.registry.static_table_id != context.config.static_table_id
                || context.registry.srs_id != context.srs.srs_id
                || context.registry.protocol_schema_id != context.srs.protocol_schema_id
                || context.compiler_artifact == nullptr
                || !lower_hex_256(context.registry.instance_schema_hash)
                || context.registry.srs_manifest_hash != context.srs.manifest_hash
                || context.registry.srs_degree_bound != context.srs.degree_bound
                || context.registry.srs_g1_generator != context.kzg_vk.g1_generator
                || context.registry.srs_generator_id != context.srs.generator_id
                || context.registry.srs_encoding != context.srs.encoding
                || context.registry.srs_spec_id
                       != context.srs.srs_type + ":"
                              + context.srs.kzg_assumption_profile_id
                || !lower_hex_256(
                       context.registry.srs_protocol_compiler_artifact_hash)
                || context.registry.srs_protocol_compiler_artifact_hash
                       != context.srs.protocol_compiler_artifact_hash
                || !lower_hex_256(context.registry.compiler_artifact_digest)
                || context.registry.compiler_artifact_digest
                       != sha256_hex(protocol_instance_canonical_bytes(
                           *context.compiler_artifact))
                || !lower_hex_256(context.registry.model_layout_digest)
                || context.registry.model_layout_digest
                       != sha256_hex(model_layout_serialization(context.model))
                || !lower_hex_256(context.registry.quant_chain_digest)
                || context.registry.quant_chain_digest
                       != sha256_hex(context.config.quantization.canonical_serialization())
                || !lower_hex_256(context.registry.static_table_digest)
                || context.registry.static_table_digest
                       != sha256_hex(static_table_serialization(context.tables))
                || context.registry.static_table_generator_id
                       != kStaticTableGeneratorId
                || !lower_hex_256(context.registry.static_table_spec_hash)
                || context.registry.static_table_spec_hash
                       != sha256_hex(static_table_spec_serialization(context))
                || !lower_hex_256(context.registry.range_and_scale_certificate)
                || context.registry.range_and_scale_certificate
                       != sha256_hex(range_scale_certificate_serialization(context))
                || context.registry.integer_bound_ledger_digest.empty()
                || !lower_hex_256(context.registry.integer_bound_ledger_digest)
                || context.compiler_artifact->integer_bound_ledger.empty()
                || context.registry.integer_bound_ledger
                       != context.compiler_artifact->integer_bound_ledger
                || context.registry.integer_bound_ledger_digest
                       != sha256_hex(integer_bound_ledger_canonical_bytes(
                           context.compiler_artifact->integer_bound_ledger))
                || !lower_hex_256(context.registry.public_input_digest)
                || context.registry.public_input_digest
                       != sha256_hex(public_input_serialization(context.local))
                || !lower_hex_256(context.registry.topology_digest)
                || context.registry.topology_digest
                       != sha256_hex(topology_serialization(context.local))
                || !lower_hex_256(context.registry.security_budget_ledger_digest)
                || context.registry.security_budget_exact.empty()
                || sha256_hex(context.compiler_artifact->instance_canonical_bytes)
                       != context.registry.instance_schema_hash
                || context.compiler_artifact->instance_canonical_bytes
                       != protocol_instance_canonical_bytes(
                           *context.compiler_artifact))
                return false;
            try
            {
                auto execution_context =
                    detail::make_verifier_execution_context(context);
                std::string artifact_reason;
                if (!validate_protocol_compiler_artifact(
                        execution_context,
                        *context.compiler_artifact,
                        &artifact_reason)
                    || !validate_integer_bound_ledger(context)
                    || !validate_static_tables(execution_context, nullptr))
                    return false;
                const auto budget = evaluate_security_budget(
                    execution_context, *context.compiler_artifact);
                if (!budget.passed
                    || budget.ledger_digest
                           != context.registry.security_budget_ledger_digest
                    || budget.exact_budget != context.registry.security_budget_exact)
                    return false;
            }
            catch (const std::exception &)
            {
                return false;
            }
            if (!lower_hex_256(context.authenticated_registry_digest)
                || !lower_hex_256(context.trusted_registry_digest)
                || context.authenticated_registry_digest
                       != context.trusted_registry_digest)
                return false;
            if (context.static_commitments.empty()
                || context.static_commitments.size()
                       != context.registry.model_entries.size()
                           + context.registry.static_entries.size())
                return false;
            std::unordered_set<std::string> labels;
            if (!validate_registry_entries(
                    context,
                    context.registry.model_entries,
                    verifier_model_entry_specs(context),
                    &labels,
                    true)
                || !validate_registry_entries(
                    context,
                    context.registry.static_entries,
                    verifier_static_entry_specs(context),
                    &labels,
                    false))
                return false;
            for (const auto &[name, commitment] : context.static_commitments)
            {
                if (name.empty() || commitment.name != name
                    || !labels.contains(name))
                    return false;
            }
            return true;
        }
    } // namespace

    std::vector<IntegerBoundLedgerEntry> rebuild_integer_bound_ledger(
        const VerifierContext &context)
    {
        // Keep the verifier independent of optional multiprecision headers.  The
        // ledger deliberately stores u64 absolute bounds, so all values that
        // survive this routine are below UINT64_MAX.  FieldElement::to_string()
        // is canonical decimal; the two decimal cutoffs below let us reject a
        // field representative whose signed magnitude cannot be represented by
        // the typed ledger without ever constructing a host-side big integer.
        constexpr std::string_view kScalarModulus =
            "52435875175126190479447740508185965837690552500527637822603658699938581184513";
        constexpr std::string_view kSignedCapacity =
            "26217937587563095239723870254092982918845276250263818911301829349969290592256";
        constexpr std::string_view kMaxU64 = "18446744073709551615";
        constexpr std::string_view kScalarMinusMaxU64 =
            "52435875175126190479447740508185965837690552500527637822585211955864871632898";
        if (context.config.range_bits == 0 || context.config.range_bits >= 63)
            throw std::runtime_error(
                "IntegerBoundLedger verifier range_bits must be in [1,62]");
        if (context.compiler_artifact == nullptr
            || context.registry.model_entries.empty())
            throw std::runtime_error(
                "IntegerBoundLedger verifier requires authenticated model certificates");
        const auto decimal_less_equal = [](std::string_view lhs,
                                            std::string_view rhs)
        {
            const auto trim = [](std::string_view value)
            {
                const auto first = value.find_first_not_of('0');
                return first == std::string_view::npos
                    ? std::string_view("0")
                    : value.substr(first);
            };
            lhs = trim(lhs);
            rhs = trim(rhs);
            return lhs.size() != rhs.size() ? lhs.size() < rhs.size()
                                            : lhs <= rhs;
        };
        const auto decimal_subtract_to_u64 = [&](std::string_view lhs,
                                                 std::string_view rhs)
            -> std::uint64_t
        {
            // This helper is only called for lhs-rhs in [0, UINT64_MAX].
            // Decimal subtraction avoids relying on compiler-specific 128-bit
            // or third-party multiprecision types in the verifier binary.
            std::array<unsigned char, 96> digits{};
            std::size_t position = digits.size();
            int borrow = 0;
            std::size_t li = lhs.size();
            std::size_t ri = rhs.size();
            while (li != 0)
            {
                const int left = static_cast<int>(lhs[--li] - '0');
                const int right = ri == 0 ? 0 : static_cast<int>(rhs[--ri] - '0');
                int digit = left - right - borrow;
                if (digit < 0)
                {
                    digit += 10;
                    borrow = 1;
                }
                else
                    borrow = 0;
                if (position == 0)
                    throw std::runtime_error(
                        "IntegerBoundLedger verifier decimal subtraction overflow");
                digits[--position] = static_cast<unsigned char>(digit);
            }
            if (borrow != 0 || ri != 0)
                throw std::runtime_error(
                    "IntegerBoundLedger verifier decimal subtraction underflow");
            std::uint64_t value = 0;
            for (; position < digits.size(); ++position)
            {
                const auto digit = static_cast<std::uint64_t>(digits[position]);
                if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10U)
                    throw std::runtime_error(
                        "IntegerBoundLedger verifier decimal result exceeds u64");
                value = value * 10U + digit;
            }
            return value;
        };
        const auto field_abs = [&](const FieldElement &value) -> std::uint64_t
        {
            const auto canonical = value.to_string();
            if (canonical.empty() || canonical == kScalarModulus)
                throw std::runtime_error(
                    "IntegerBoundLedger verifier field representative is invalid");
            if (decimal_less_equal(canonical, kSignedCapacity))
            {
                if (!decimal_less_equal(canonical, kMaxU64))
                    throw std::runtime_error(
                        "IntegerBoundLedger verifier field magnitude exceeds u64");
                std::uint64_t result = 0;
                for (const auto digit : canonical)
                {
                    const auto value_digit = static_cast<std::uint64_t>(digit - '0');
                    if (result > (std::numeric_limits<std::uint64_t>::max() - value_digit) / 10U)
                        throw std::runtime_error(
                            "IntegerBoundLedger verifier field magnitude exceeds u64");
                    result = result * 10U + value_digit;
                }
                return result;
            }
            if (!decimal_less_equal(kScalarMinusMaxU64, canonical))
                throw std::runtime_error(
                    "IntegerBoundLedger verifier field magnitude exceeds u64");
            return decimal_subtract_to_u64(kScalarModulus, canonical);
        };
        const auto checked_add = [](std::uint64_t lhs,
                                    std::uint64_t rhs,
                                    std::string_view label)
            -> std::uint64_t
        {
            if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs)
                throw std::runtime_error(
                    "IntegerBoundLedger verifier addition overflow: "
                    + std::string(label));
            return lhs + rhs;
        };
        const auto checked_mul = [](std::uint64_t lhs,
                                    std::uint64_t rhs,
                                    std::string_view label)
            -> std::uint64_t
        {
            if (lhs != 0
                && rhs > std::numeric_limits<std::uint64_t>::max() / lhs)
                throw std::runtime_error(
                    "IntegerBoundLedger verifier multiplication overflow: "
                    + std::string(label));
            return lhs * rhs;
        };
        const auto minimum_limbs = [&](std::uint64_t bound) -> std::size_t
        {
            std::size_t count = 1;
            auto remaining = bound;
            while (remaining >= (std::uint64_t{1} << context.config.range_bits))
            {
                remaining >>= context.config.range_bits;
                if (count == std::numeric_limits<std::size_t>::max())
                    throw std::runtime_error(
                        "IntegerBoundLedger verifier limb count overflow");
                ++count;
            }
            return count;
        };
        const auto add_limb_capacity_check = [&](std::uint64_t bound,
                                                 std::size_t limbs,
                                                 std::string_view label)
        {
            if (limbs == 0
                || limbs > std::numeric_limits<std::size_t>::max()
                    / context.config.range_bits)
                throw std::runtime_error(
                    "IntegerBoundLedger verifier limb width overflow: "
                    + std::string(label));
            const auto total_bits = limbs * context.config.range_bits;
            const auto representation_max = total_bits >= 64
                ? std::numeric_limits<std::uint64_t>::max()
                : ((std::uint64_t{1} << total_bits) - 1U);
            if (bound > representation_max)
                throw std::runtime_error(
                    "IntegerBoundLedger verifier limb capacity failure: "
                    + std::string(label));
        };
        std::vector<IntegerBoundLedgerEntry> rebuilt;
        rebuilt.reserve(context.registry.integer_bound_ledger.size());
        const auto add = [&](std::string id,
                             std::string stage,
                             std::string source,
                             std::uint64_t bound)
        {
            const auto limbs = minimum_limbs(bound);
            add_limb_capacity_check(bound, limbs, id);
            rebuilt.push_back({
                .id = std::move(id),
                .stage = std::move(stage),
                .source = std::move(source),
                .absolute_bound = bound,
                .limb_count = limbs});
        };
        const auto table_component_bound = [&](const auto &table, bool first)
            -> std::uint64_t
        {
            if (table.empty())
                throw std::runtime_error(
                    "IntegerBoundLedger verifier static table is empty");
            std::uint64_t bound = 0;
            for (const auto &row : table)
                bound = std::max(bound, field_abs(first ? row.first : row.second));
            return bound;
        };
        const auto model_certificate = [&](const std::string &label,
                                           const std::string &domain,
                                           std::size_t layer,
                                           std::size_t head,
                                           std::uint64_t scale)
            -> std::uint64_t
        {
            const auto it = std::find_if(
                context.registry.model_entries.begin(),
                context.registry.model_entries.end(),
                [&](const auto &entry) { return entry.label == label; });
            if (it == context.registry.model_entries.end())
                throw std::runtime_error(
                    "IntegerBoundLedger verifier missing model certificate: "
                    + label);
            const auto &certificate = it->bound_certificate;
            if (!certificate.present || certificate.label != label
                || certificate.layer_index != layer
                || certificate.local_head_index != head
                || certificate.domain != domain
                || certificate.degree_bound != context.srs.degree_bound
                || certificate.scale != scale)
                throw std::runtime_error(
                    "IntegerBoundLedger verifier model certificate metadata mismatch: "
                    + label);
            return certificate.absolute_bound;
        };

        if (context.local.features.empty())
            throw std::runtime_error(
                "IntegerBoundLedger verifier requires public input features");
        std::uint64_t input_bound = 0;
        for (const auto &row : context.local.features)
            for (const auto &value : row)
                input_bound = std::max(input_bound, field_abs(value));
        add("input.features", "input",
            "max_abs(round(features_fp*S_H[0]))", input_bound);

        std::size_t max_indegree_size = 0;
        std::vector<std::size_t> indegree(context.local.num_nodes, 0);
        for (const auto &edge : context.local.edges)
        {
            if (edge.dst >= indegree.size())
                throw std::runtime_error(
                    "IntegerBoundLedger verifier edge destination is out of range");
            if (indegree[edge.dst] == std::numeric_limits<std::size_t>::max())
                throw std::runtime_error(
                    "IntegerBoundLedger verifier indegree overflow");
            ++indegree[edge.dst];
            max_indegree_size = std::max(max_indegree_size, indegree[edge.dst]);
        }
        if (max_indegree_size > std::numeric_limits<std::uint64_t>::max())
            throw std::runtime_error(
                "IntegerBoundLedger verifier indegree exceeds u64");
        const auto max_indegree = static_cast<std::uint64_t>(max_indegree_size);
        add("graph.max_indegree", "topology",
            "max_dst_in_degree(public_edges)", max_indegree);

        const auto lrelu_input_bound = table_component_bound(context.tables.lrelu, true);
        const auto lrelu_output_bound = table_component_bound(context.tables.lrelu, false);
        const auto elu_input_bound = table_component_bound(context.tables.elu, true);
        const auto elu_output_bound = table_component_bound(context.tables.elu, false);
        const auto exp_input_bound = table_component_bound(context.tables.exp, true);
        const auto exp_output_bound = table_component_bound(context.tables.exp, false);
        add("table.LReLU.input", "static_table", "max_abs(T_LReLU[:,0])",
            lrelu_input_bound);
        add("table.LReLU.output", "static_table", "max_abs(T_LReLU[:,1])",
            lrelu_output_bound);
        add("table.ELU.input", "static_table", "max_abs(T_ELU[:,0])",
            elu_input_bound);
        add("table.ELU.output", "static_table", "max_abs(T_ELU[:,1])",
            elu_output_bound);
        add("table.ExpMap.input", "static_table", "max_abs(T_exp[:,0])",
            exp_input_bound);
        add("table.ExpMap.output", "static_table", "max_abs(T_exp[:,1])",
            exp_output_bound);
        add("table.Range.output", "static_table", "max(T_range)",
            context.tables.range.empty() ? 0 : context.tables.range.back().value());

        const auto require_table_cover = [&](std::uint64_t required,
                                             std::uint64_t registered,
                                             std::string_view name)
        {
            if (required > registered)
                throw std::runtime_error(
                    "IntegerBoundLedger verifier " + std::string(name)
                    + " input bound exceeds registered table interval");
        };
        const auto add_division_bounds = [&](const std::string &prefix,
                                             std::uint64_t numerator,
                                             std::uint64_t denominator)
        {
            const auto q = checked_add(numerator, 1, prefix + ".q");
            const auto remainder = checked_add(
                checked_mul(2, numerator, prefix + ".r:2A"),
                denominator, prefix + ".r");
            const auto delta = checked_mul(2, denominator, prefix + ".delta");
            add(prefix + ".A", "division", "abs(numerator)", numerator);
            add(prefix + ".D", "division", "positive denominator", denominator);
            add(prefix + ".q", "division",
                "SignedRoundDiv quotient bound <= A+1", q);
            add(prefix + ".r", "division",
                "SignedRoundDiv remainder bound <= 2A+D", remainder);
            add(prefix + ".delta", "division",
                "SignedRoundDiv delta bound <= 2D", delta);
        };

        std::uint64_t input_layer_bound = input_bound;
        for (std::size_t layer_index = 0;
             layer_index < context.model.hidden_layers.size();
             ++layer_index)
        {
            const auto &layer = context.model.hidden_layers.at(layer_index);
            std::uint64_t concat_bound = 0;
            for (std::size_t head_index = 0;
                 head_index < layer.heads.size(); ++head_index)
            {
                const auto prefix = "hidden:l" + std::to_string(layer_index)
                    + ":h" + std::to_string(head_index);
                std::uint64_t w_bound = 0;
                for (std::size_t column = 0;
                     column < layer.shape.head_dim; ++column)
                    w_bound = std::max(
                        w_bound,
                        model_certificate(
                            hidden_weight_column_label(
                                layer_index, head_index, column),
                            "in", layer_index, head_index,
                            context.config.quantization.s_w_hidden
                                .at(layer_index).at(head_index)));
                const auto a_src_bound = model_certificate(
                    hidden_src_label(layer_index, head_index), "d",
                    layer_index, head_index,
                    context.config.quantization.s_a_src_hidden
                        .at(layer_index).at(head_index));
                const auto a_dst_bound = model_certificate(
                    hidden_dst_label(layer_index, head_index), "d",
                    layer_index, head_index,
                    context.config.quantization.s_a_dst_hidden
                        .at(layer_index).at(head_index));
                add(prefix + ".model.W", "model",
                    "max_abs(round(W_fp*S_W_hidden))", w_bound);
                add(prefix + ".model.a_src", "model",
                    "max_abs(round(a_src_fp*S_a_src_hidden))", a_src_bound);
                add(prefix + ".model.a_dst", "model",
                    "max_abs(round(a_dst_fp*S_a_dst_hidden))", a_dst_bound);
                const auto h_prime = checked_mul(
                    checked_mul(static_cast<std::uint64_t>(layer.input_dim),
                                input_layer_bound, prefix + ".H_prime:dimension"),
                    w_bound, prefix + ".H_prime");
                const auto e_src = checked_mul(
                    static_cast<std::uint64_t>(layer.shape.head_dim),
                    checked_mul(h_prime, a_src_bound,
                                prefix + ".E_src:product"),
                    prefix + ".E_src");
                const auto e_dst = checked_mul(
                    static_cast<std::uint64_t>(layer.shape.head_dim),
                    checked_mul(h_prime, a_dst_bound,
                                prefix + ".E_dst:product"),
                    prefix + ".E_dst");
                const auto score = checked_add(e_src, e_dst, prefix + ".S");
                const auto delta = checked_mul(2, lrelu_output_bound,
                                               prefix + ".Delta");
                const auto sum = checked_mul(max_indegree, exp_output_bound,
                                             prefix + ".Sum");
                const auto numerator = checked_mul(sum, h_prime,
                                                   prefix + ".H_num");
                const auto scaled_numerator = checked_mul(
                    context.config.quantization.s_agg_pre.at(layer_index),
                    numerator, prefix + ".H_num_scaled");
                const auto scaled_denominator = checked_mul(
                    context.config.quantization.s_h_prime
                        .at(layer_index).at(head_index),
                    sum, prefix + ".Sum_scaled");
                require_table_cover(score, lrelu_input_bound, "LReLU");
                require_table_cover(delta, exp_input_bound, "ExpMap");
                require_table_cover(
                    checked_add(scaled_numerator, 1, prefix + ".ELU_input"),
                    elu_input_bound, "ELU");
                add(prefix + ".H_prime", "projection",
                    "d_in*B_H_in*B_W", h_prime);
                add(prefix + ".E_src", "attention",
                    "d_h*B_H_prime*B_a_src", e_src);
                add(prefix + ".E_dst", "attention",
                    "d_h*B_H_prime*B_a_dst", e_dst);
                add(prefix + ".S", "attention", "B_E_src+B_E_dst", score);
                add(prefix + ".LReLU", "lookup",
                    "max_abs(T_LReLU[:,1])", lrelu_output_bound);
                add(prefix + ".M", "attention",
                    "group maximum of LReLU values", lrelu_output_bound);
                add(prefix + ".Delta", "lookup",
                    "|M-Z| <= 2*B_LReLU_output", delta);
                add(prefix + ".ExpMap", "lookup",
                    "max_abs(T_exp[:,1])", exp_output_bound);
                add(prefix + ".Sum", "aggregation",
                    "Delta_in_max*B_U", sum);
                add(prefix + ".H_num", "aggregation",
                    "Delta_in_max*B_U*B_H_prime", numerator);
                add(prefix + ".H_num_scaled", "aggregation",
                    "S_agg_pre*B_H_num", scaled_numerator);
                add(prefix + ".Sum_scaled", "aggregation",
                    "S_H_prime*B_Sum", scaled_denominator);
                add_division_bounds(prefix + ".SignedRoundDiv",
                                    scaled_numerator, scaled_denominator);
                add(prefix + ".ELU_input", "activation",
                    "SignedRoundDiv output <= A+1",
                    checked_add(scaled_numerator, 1,
                                prefix + ".ELU_input"));
                add(prefix + ".ELU_output", "activation",
                    "max_abs(T_ELU[:,1])", elu_output_bound);
                concat_bound = std::max(concat_bound, elu_output_bound);
            }
            add("hidden:l" + std::to_string(layer_index) + ".concat", "concat",
                "max_abs(concat(head_ELU_outputs))", concat_bound);
            input_layer_bound = concat_bound;
        }

        std::uint64_t output_head_bound = 0;
        for (std::size_t head_index = 0;
             head_index < context.model.output_layer.heads.size();
             ++head_index)
        {
            const auto &head = context.model.output_layer;
            const auto prefix = "output:h" + std::to_string(head_index);
            std::uint64_t w_bound = 0;
            for (std::size_t column = 0;
                 column < head.output_dim; ++column)
                w_bound = std::max(
                    w_bound,
                    model_certificate(
                        output_weight_column_label(head_index, column),
                        "cat", 0, head_index,
                        context.config.quantization.s_w_output.at(head_index)));
            const auto a_src_bound = model_certificate(
                output_src_label(head_index), "C", 0, head_index,
                context.config.quantization.s_a_src_output.at(head_index));
            const auto a_dst_bound = model_certificate(
                output_dst_label(head_index), "C", 0, head_index,
                context.config.quantization.s_a_dst_output.at(head_index));
            const auto bias_bound = model_certificate(
                output_bias_label(head_index), "C", 0, head_index,
                context.config.quantization.s_b_output.at(head_index));
            add(prefix + ".model.W", "model",
                "max_abs(round(W_fp*S_W_output))", w_bound);
            add(prefix + ".model.a_src", "model",
                "max_abs(round(a_src_fp*S_a_src_output))", a_src_bound);
            add(prefix + ".model.a_dst", "model",
                "max_abs(round(a_dst_fp*S_a_dst_output))", a_dst_bound);
            add(prefix + ".model.bias", "model",
                "max_abs(round(bias_fp*S_b_output))", bias_bound);
            const auto y_prime = checked_mul(
                checked_mul(static_cast<std::uint64_t>(head.input_dim),
                            input_layer_bound, prefix + ".Y_prime:dimension"),
                w_bound, prefix + ".Y_prime");
            const auto e_src = checked_mul(
                static_cast<std::uint64_t>(head.output_dim),
                checked_mul(y_prime, a_src_bound,
                            prefix + ".E_src:product"),
                prefix + ".E_src");
            const auto e_dst = checked_mul(
                static_cast<std::uint64_t>(head.output_dim),
                checked_mul(y_prime, a_dst_bound,
                            prefix + ".E_dst:product"),
                prefix + ".E_dst");
            const auto score = checked_add(e_src, e_dst, prefix + ".S");
            const auto delta = checked_mul(2, lrelu_output_bound,
                                           prefix + ".Delta");
            const auto sum = checked_mul(max_indegree, exp_output_bound,
                                         prefix + ".Sum");
            const auto numerator = checked_mul(sum, y_prime,
                                               prefix + ".Y_num");
            const auto scaled_numerator = checked_mul(
                context.config.quantization.s_out_pre, numerator,
                prefix + ".Y_num_scaled");
            const auto scaled_denominator = checked_mul(
                context.config.quantization.s_y_prime.at(head_index), sum,
                prefix + ".Sum_scaled");
            require_table_cover(score, lrelu_input_bound, "LReLU");
            require_table_cover(delta, exp_input_bound, "ExpMap");
            add(prefix + ".Y_prime", "projection",
                "d_in*B_H_cat*B_W_output", y_prime);
            add(prefix + ".E_src", "attention",
                "d_C*B_Y_prime*B_a_src", e_src);
            add(prefix + ".E_dst", "attention",
                "d_C*B_Y_prime*B_a_dst", e_dst);
            add(prefix + ".S", "attention", "B_E_src+B_E_dst", score);
            add(prefix + ".LReLU", "lookup",
                "max_abs(T_LReLU[:,1])", lrelu_output_bound);
            add(prefix + ".M", "attention",
                "group maximum of LReLU values", lrelu_output_bound);
            add(prefix + ".Delta", "lookup",
                "|M-Z| <= 2*B_LReLU_output", delta);
            add(prefix + ".ExpMap", "lookup",
                "max_abs(T_exp[:,1])", exp_output_bound);
            add(prefix + ".Sum", "aggregation",
                "Delta_in_max*B_U", sum);
            add(prefix + ".Y_num", "aggregation",
                "Delta_in_max*B_U*B_Y_prime", numerator);
            add(prefix + ".Y_num_scaled", "aggregation",
                "S_out_pre*B_Y_num", scaled_numerator);
            add(prefix + ".Sum_scaled", "aggregation",
                "S_Y_prime*B_Sum", scaled_denominator);
            add_division_bounds(prefix + ".SignedRoundDiv",
                                scaled_numerator, scaled_denominator);
            const auto pre_bias = checked_add(
                scaled_numerator, 1, prefix + ".pre_bias");
            const auto post_bias = checked_add(
                pre_bias, bias_bound, prefix + ".post_bias");
            add(prefix + ".pre_bias", "output_bias",
                "SignedRoundDiv output <= A+1", pre_bias);
            add(prefix + ".post_bias", "output_bias",
                "B_pre_bias+B_bias", post_bias);
            add(prefix + ".concat", "output",
                "one output head post-bias value", post_bias);
            output_head_bound = std::max(output_head_bound, post_bias);
        }
        const auto output_head_sum_bound = checked_mul(
            static_cast<std::uint64_t>(context.model.output_layer.heads.size()),
            output_head_bound, "output.average.head_sum");
        add("output.average.head_sum", "output_average",
            "K_out*B_output_head", output_head_sum_bound);
        add_division_bounds("output.average.SignedRoundDiv",
                            output_head_sum_bound,
                            static_cast<std::uint64_t>(
                                context.model.output_layer.heads.size()));
        add("output.average.final", "output_average",
            "average SignedRoundDiv output <= A+1",
            checked_add(output_head_sum_bound, 1,
                        "output.average.final"));
        if (rebuilt.empty())
            throw std::runtime_error("IntegerBoundLedger verifier rebuilt no rows");
        return rebuilt;
    }

    std::string AuthenticatedVerifierRegistry::canonical_serialization() const
    {
        std::string bytes;
        bytes.reserve(1024 + (model_entries.size() + static_entries.size()) * 128);
        append_string(&bytes, format);
        append_string(&bytes, registry_id);
        append_string(&bytes, model_arch_id);
        append_string(&bytes, model_param_id);
        append_string(&bytes, input_preprocess_id);
        append_string(&bytes, quant_cfg_id);
        append_string(&bytes, static_table_id);
        append_string(&bytes, srs_id);
        append_string(&bytes, protocol_schema_id);
        append_string(&bytes, instance_schema_hash);
        append_string(&bytes, srs_manifest_hash);
        append_u64(&bytes, srs_degree_bound);
        const auto generator = crypto::serialize(srs_g1_generator);
        append_string(
            &bytes,
            std::string_view(
                reinterpret_cast<const char *>(generator.data()), generator.size()));
        append_string(&bytes, srs_generator_id);
        append_string(&bytes, srs_encoding);
        append_string(&bytes, srs_spec_id);
        append_string(&bytes, srs_protocol_compiler_artifact_hash);
        append_string(&bytes, compiler_artifact_digest);
        append_string(&bytes, model_layout_digest);
        append_string(&bytes, quant_chain_digest);
        append_string(&bytes, static_table_digest);
        append_string(&bytes, static_table_generator_id);
        append_string(&bytes, static_table_spec_hash);
        append_string(&bytes, range_and_scale_certificate);
        append_string(&bytes, integer_bound_ledger_digest);
        append_string(&bytes, public_input_digest);
        append_string(&bytes, topology_digest);
        append_string(&bytes, security_budget_ledger_digest);
        append_string(&bytes, security_budget_exact);
        append_u64(&bytes, integer_bound_ledger.size());
        for (const auto &entry : integer_bound_ledger)
        {
            append_string(&bytes, entry.id);
            append_string(&bytes, entry.stage);
            append_string(&bytes, entry.source);
            append_u64(&bytes, entry.absolute_bound);
            append_u64(&bytes, entry.limb_count);
        }
        append_u64(&bytes, model_entries.size());
        for (const auto &entry : model_entries) append_registry_entry(&bytes, entry);
        append_u64(&bytes, static_entries.size());
        for (const auto &entry : static_entries) append_registry_entry(&bytes, entry);
        return bytes;
    }

    std::string AuthenticatedVerifierRegistry::canonical_digest() const
    {
        return sha256_hex(canonical_serialization());
    }

    VerifierRegistrationArtifact make_verifier_registration_artifact(
        const ProtocolContext &prover_context)
    {
        if (prover_context.verifier_shape_only)
            throw std::runtime_error(
                "candidate verifier registration requires a proving context");
        if (prover_context.compiler_artifact == nullptr
            || prover_context.validated_compiler_artifact_identity
                   != prover_context.compiler_artifact.get())
            throw std::runtime_error(
                "candidate verifier registration requires an authenticated compiler artifact");
        if (!has_complete_public_feature_source(
                prover_context.dataset, prover_context.local))
            throw std::runtime_error(
                "candidate verifier registration requires complete public feature coverage");
        VerifierRegistrationArtifact artifact;
        artifact.config = public_config(prover_context.config);
        artifact.dataset = prover_context.dataset;
        // Labels and raw floating-point features are training-side material and
        // must never cross the registration boundary.  The complete quantized
        // public feature matrix is carried once by LocalGraph below.
        artifact.dataset.features_fp.clear();
        artifact.dataset.features.clear();
        artifact.dataset.labels.clear();
        artifact.dataset.labels_multi.clear();
        artifact.local = prover_context.local;
        artifact.local.features_fp.clear();
        artifact.local.labels.clear();
        artifact.local.labels_multi.clear();
        artifact.model = public_model_shape(prover_context.model);
        artifact.tables = prover_context.tables;
        artifact.domains = prover_context.domains;
        artifact.srs = public_srs(prover_context.kzg.manifest);
        artifact.kzg_vk = prover_context.kzg.vk;
        // Prepared pairing data are execution caches, not registration data;
        // the independent loader derives them from the decoded points.
        artifact.kzg_vk.g2_one_prepared.reset();
        artifact.kzg_vk.g2_tau_prepared.reset();
        for (const auto &[name, polynomial] : prover_context.public_polynomials)
            artifact.public_polynomial_descriptors.emplace(
                name, describe_public_polynomial(polynomial, name));
        copy_point_map(
            prover_context.public_commitments,
            &artifact.public_commitments,
            false);
        copy_point_map(
            prover_context.static_commitments,
            &artifact.static_commitments,
            false);
        artifact.compiler_artifact = *prover_context.compiler_artifact;
        artifact.registry = registry_from_context(prover_context);
        return artifact;
    }

    VerifierContext load_verifier_context_from_registry(
        const std::string &registry_path,
        const std::string &trusted_registry_digest)
    {
        if (!lower_hex_256(trusted_registry_digest))
            throw std::runtime_error(
                "formal verifier registry load requires an out-of-band trusted SHA-256 digest");
        std::ifstream input(registry_path, std::ios::binary);
        if (!input)
            throw std::runtime_error(
                "missing authenticated verifier registration artifact; run candidate registration and review its digest: "
                + registry_path);
        const auto file_size = std::filesystem::file_size(registry_path);
        if (file_size > (2ULL << 30))
            throw std::runtime_error(
                "verifier registration artifact exceeds the 2 GiB cap");
        std::vector<std::uint8_t> bytes(static_cast<std::size_t>(file_size));
        input.read(
            reinterpret_cast<char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
        if (input.gcount() != static_cast<std::streamsize>(bytes.size()))
            throw std::runtime_error(
                "failed to read verifier registration artifact: " + registry_path);
        const auto artifact = parse_verifier_registration_artifact(bytes);
        if (artifact.canonical_digest() != trusted_registry_digest)
            throw std::runtime_error(
                "verifier registration artifact digest does not match the trusted registry root");
        VerifierContext out;
        out.config = artifact.config;
        out.dataset = artifact.dataset;
        out.local = artifact.local;
        // The verifier needs only the quantized public feature table.  Keep a
        // single owned copy and never reconstruct a training-side dataset.
        out.dataset.features = out.local.features;
        out.model = artifact.model;
        out.tables = artifact.tables;
        out.domains = artifact.domains;
        out.srs = artifact.srs;
        out.kzg_vk = artifact.kzg_vk;
        out.kzg_vk.g2_one_prepared = std::make_shared<crypto::PreparedG2>(
            crypto::prepare_g2(out.kzg_vk.g2_one));
        out.kzg_vk.g2_tau_prepared = std::make_shared<crypto::PreparedG2>(
            crypto::prepare_g2(out.kzg_vk.g2_tau));
        out.public_polynomial_descriptors = artifact.public_polynomial_descriptors;
        out.public_commitments = artifact.public_commitments;
        out.static_commitments = artifact.static_commitments;
        out.compiler_artifact = std::make_shared<const ProtocolCompilerArtifact>(
            artifact.compiler_artifact);
        out.registry = artifact.registry;
        out.authenticated_registry_digest = artifact.canonical_digest();
        out.trusted_registry_digest = trusted_registry_digest;
        if (!validate_registry_shape(out))
            throw std::runtime_error(
                "authenticated verifier registration failed independent context validation");
        return out;
    }

    bool validate_verifier_context(const VerifierContext &verifier_context)
    {
        try
        {
            return validate_registry_shape(verifier_context);
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

    void write_verifier_registry(
        const ProtocolContext &prover_context,
        const std::string &registry_path)
    {
        const auto artifact = make_verifier_registration_artifact(prover_context);
        const auto body = serialize_verifier_registration_artifact(artifact);
        const auto parent = std::filesystem::path(registry_path).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
        std::ofstream output(registry_path, std::ios::binary | std::ios::trunc);
        if (!output)
            throw std::runtime_error("failed to create verifier registry: " + registry_path);
        output.write(
            reinterpret_cast<const char *>(body.data()),
            static_cast<std::streamsize>(body.size()));
        if (!output)
            throw std::runtime_error("failed to write verifier registry: " + registry_path);
    }

    std::string canonical_verifier_registry_path(
        const ProtocolContext &prover_context)
    {
        if (!prover_context.config.verifier_registry_path.empty())
            return prover_context.config.verifier_registry_path;
        return "artifacts/verifier_registry/"
            + prover_context.dataset.name + "/verifier_registry.txt";
    }

    std::string canonical_verifier_registry_path(
        const VerifierContext &verifier_context)
    {
        return "artifacts/verifier_registry/"
            + verifier_context.dataset.name + "/verifier_registry.txt";
    }

    namespace detail
    {
        ProtocolContext make_verifier_execution_context(
            const VerifierContext &verifier_context)
        {
            ProtocolContext out;
            out.verifier_shape_only = true;
            out.config.dataset = verifier_context.config.dataset;
            out.config.task_type = verifier_context.config.task_type;
            out.config.report_unit = verifier_context.config.report_unit;
            out.config.batching_rule = verifier_context.config.batching_rule;
            out.config.subgraph_rule = verifier_context.config.subgraph_rule;
            out.config.self_loop_rule = verifier_context.config.self_loop_rule;
            out.config.edge_sort_rule = verifier_context.config.edge_sort_rule;
            out.config.chunking_rule = verifier_context.config.chunking_rule;
            out.config.quant_cfg_id = verifier_context.config.quant_cfg_id;
            out.config.quantization = verifier_context.config.quantization;
            out.config.model_arch_id = verifier_context.config.model_arch_id;
            out.config.model_param_id = verifier_context.config.model_param_id;
            out.config.input_preprocess_id = verifier_context.config.input_preprocess_id;
            out.config.static_table_id = verifier_context.config.static_table_id;
            out.config.degree_bound_id = verifier_context.config.degree_bound_id;
            out.config.srs_type = verifier_context.config.srs_type;
            out.config.srs_id = verifier_context.config.srs_id;
            out.config.privacy_policy = verifier_context.config.privacy_policy;
            out.config.input_privacy = verifier_context.config.input_privacy;
            out.config.output_mode = verifier_context.config.output_mode;
            out.config.opening_mode = verifier_context.config.opening_mode;
            out.config.binding_mode = verifier_context.config.binding_mode;
            out.config.proof_task = verifier_context.config.proof_task;
            out.config.security_profile_id = verifier_context.config.security_profile_id;
            out.config.num_classes = verifier_context.config.num_classes;
            out.config.range_bits = verifier_context.config.range_bits;
            out.config.layer_count = verifier_context.config.layer_count;
            out.config.K_out = verifier_context.config.K_out;
            out.config.batch_graphs = verifier_context.config.batch_graphs;
            out.config.hidden_profile = verifier_context.config.hidden_profile;
            out.config.d_in_profile = verifier_context.config.d_in_profile;

            out.dataset = verifier_context.dataset;
            out.local = verifier_context.local;
            out.tables = verifier_context.tables;
            out.domains = verifier_context.domains;

            // This is a shape-only execution object.  Populate only empty
            // tensor shells; any accidental parameter read fails closed rather
            // than observing a forged zero-valued model.
            out.model.model_arch_id = verifier_context.model.model_arch_id;
            out.model.model_param_id = verifier_context.model.model_param_id;
            out.model.input_preprocess_id = verifier_context.model.input_preprocess_id;
            out.model.quant_cfg_id = verifier_context.model.quant_cfg_id;
            out.model.static_table_id = verifier_context.model.static_table_id;
            out.model.degree_bound_id = verifier_context.model.degree_bound_id;
            out.model.L = verifier_context.model.L;
            out.model.d_in_profile = verifier_context.model.d_in_profile;
            out.model.hidden_profile = verifier_context.model.hidden_profile;
            out.model.K_out = verifier_context.model.K_out;
            out.model.C = verifier_context.model.C;
            out.model.hidden_layers.reserve(verifier_context.model.hidden_layers.size());
            for (const auto &layer : verifier_context.model.hidden_layers)
            {
                model::HiddenLayerParameters shape;
                shape.layer_index = layer.layer_index;
                shape.input_dim = layer.input_dim;
                shape.shape = layer.shape;
                shape.heads.resize(layer.heads.size());
                out.model.hidden_layers.push_back(std::move(shape));
            }
            out.model.output_layer.input_dim = verifier_context.model.output_layer.input_dim;
            out.model.output_layer.output_dim = verifier_context.model.output_layer.output_dim;
            out.model.output_layer.head_count = verifier_context.model.output_layer.head_count;
            out.model.output_layer.heads.resize(verifier_context.model.output_layer.heads.size());

            out.kzg.manifest.srs_id = verifier_context.srs.srs_id;
            out.kzg.manifest.srs_type = verifier_context.srs.srs_type;
            out.kzg.manifest.degree_bound = verifier_context.srs.degree_bound;
            out.kzg.manifest.manifest_hash = verifier_context.srs.manifest_hash;
            out.kzg.manifest.protocol_schema_id = verifier_context.srs.protocol_schema_id;
            out.kzg.manifest.kzg_assumption_profile_id = verifier_context.srs.kzg_assumption_profile_id;
            out.kzg.manifest.kzg_assumption_degree_bound = verifier_context.srs.kzg_assumption_degree_bound;
            out.kzg.manifest.kzg_assumption_security_bits = verifier_context.srs.kzg_assumption_security_bits;
            out.kzg.manifest.kzg_assumption_domain_catalog_hash = verifier_context.srs.kzg_assumption_domain_catalog_hash;
            out.kzg.vk = verifier_context.kzg_vk;
            out.kzg.g1_generator = verifier_context.kzg_vk.g1_generator;
            out.kzg.g2_one = verifier_context.kzg_vk.g2_one;
            out.kzg.g2_tau = verifier_context.kzg_vk.g2_tau;
            out.kzg.g2_one_prepared = verifier_context.kzg_vk.g2_one_prepared;
            out.kzg.g2_tau_prepared = verifier_context.kzg_vk.g2_tau_prepared;

            out.public_polynomial_descriptors =
                verifier_context.public_polynomial_descriptors;
            auto copy_points = [](const auto &source, auto *target)
            {
                target->reserve(source.size());
                for (const auto &[name, commitment] : source)
                {
                    target->emplace(
                        name,
                        crypto::Commitment{
                            .name = commitment.name,
                            .point = commitment.point,
                            .opening_polynomial = nullptr});
                }
            };
            copy_points(verifier_context.public_commitments, &out.public_commitments);
            copy_points(verifier_context.static_commitments, &out.static_commitments);
            out.compiler_artifact = verifier_context.compiler_artifact;
            out.validated_compiler_artifact_identity = out.compiler_artifact.get();
            return out;
        }
    } // namespace detail

} // namespace gatzk::protocol
