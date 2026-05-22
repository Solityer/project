#pragma once

// GAT-ZKML-zkDL 的 public metadata、VK_static/VK_model、table context 与量化/rounding contract。
// 这些内容来自主实验的 proof-system-neutral 语义；不包含 KZG、quotient、opening rule 或 transcript challenge。

struct ZkdlGatQuantContract {
    std::string quant_cfg_id;
    std::string encoding_id = "signed_fixed_point_scale_16_to_Fr";
    std::string rounding_policy = "round_half_away_from_zero";
    std::string rescale_policy = "trace_contract_recorded_no_full_rescale_proof_yet";
    std::size_t scale = 16;
    std::size_t range_bits = 12;
    std::size_t exp_band = 2;
    std::size_t elu_band = 1;
    std::string contract_id;
};

struct ZkdlGatTableContext {
    std::string static_table_id;
    std::size_t range_size = 0;
    std::size_t lrelu_entry_count = 0;
    std::size_t elu_entry_count = 0;
    std::size_t exp_entry_count = 0;
    std::string range_digest;
    std::string lrelu_digest;
    std::string elu_digest;
    std::string exp_digest;
    std::string context_id;
};

struct ZkdlGatPublicMetadata {
    std::string protocol_id = "GAT-ZKML-zkDL";
    std::string proof_scope_id = "zkdl_gat_full";
    std::string dataset;
    std::string task_type;
    std::string report_unit;
    std::string batching_rule;
    std::string self_loop_rule = "per_node";
    std::string edge_sort_rule = "edge_gid_then_dst_stable";
    std::string subgraph_rule = "whole_graph";
    std::string model_arch_id;
    std::string model_param_id;
    std::string quant_cfg_id;
    std::string static_table_id;
    std::string domain_cfg;
    std::string dim_cfg;
    std::string padding_rule_id = "next_power_of_two_per_domain";
    std::string tensor_bound_id = "full_dataset_no_tensor_payload_in_proof";
    std::string commitment_backend = "zkDL_Hyrax_style_tensor_commitment";
    std::size_t node_count = 0;
    std::size_t edge_count = 0;
    std::size_t feature_count = 0;
    std::size_t class_count = 0;
    std::size_t graph_count = 0;
    std::size_t model_layer_count = 0;
    std::size_t hidden_layer_count = 0;
    std::size_t hidden_head_count = 0;
    std::size_t output_head_count = 0;
    std::string hidden_profile;
    std::string output_average_rule;
    std::string metadata_id;
};

std::int64_t zkdl_round_half_away_from_zero(double value)
{
    return static_cast<std::int64_t>(value >= 0.0 ? std::floor(value + 0.5) : std::ceil(value - 0.5));
}

std::string hash_u64_hex(std::uint64_t value)
{
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << value;
    return out.str();
}

std::string hash_text_id(const std::string& prefix, const std::string& text)
{
    return prefix + ":" + hash_u64_hex(fnv1a64(text));
}

std::string hidden_profile_string(const ModelParams& model)
{
    std::ostringstream out;
    for (std::size_t i = 0; i < model.hidden_profile.size(); ++i) {
        if (i != 0) out << ",";
        out << model.hidden_profile[i].first << "x" << model.hidden_profile[i].second;
    }
    return out.str();
}

ZkdlGatQuantContract build_zkdl_gat_quant_contract(const DatasetRunSpec& spec, const ModelParams& model)
{
    ZkdlGatQuantContract contract;
    contract.quant_cfg_id = model.quant_cfg_id;
    contract.range_bits = spec.range_bits;
    std::ostringstream canonical;
    canonical << "quant_cfg_id=" << contract.quant_cfg_id
              << ";encoding=" << contract.encoding_id
              << ";scale=" << contract.scale
              << ";rounding=" << contract.rounding_policy
              << ";rescale=" << contract.rescale_policy
              << ";range_bits=" << contract.range_bits
              << ";exp_band=" << contract.exp_band
              << ";elu_band=" << contract.elu_band;
    contract.contract_id = hash_text_id("zkdl_quant_contract", canonical.str());
    return contract;
}

ZkdlGatTableContext build_zkdl_gat_table_context(const DatasetRunSpec& spec, const ModelParams& model)
{
    const auto quant = build_zkdl_gat_quant_contract(spec, model);
    ZkdlGatTableContext context;
    context.static_table_id = model.static_table_id;
    context.range_size = std::size_t{1} << spec.range_bits;

    std::ostringstream range_canonical;
    for (std::size_t value = 0; value < context.range_size; ++value) {
        range_canonical << value << ';';
    }
    context.range_digest = hash_text_id("range", range_canonical.str());

    const auto signed_bound = static_cast<std::int64_t>(context.range_size - 1);
    std::ostringstream lrelu_canonical;
    for (std::int64_t x = -signed_bound; x <= signed_bound; ++x) {
        const auto y = zkdl_round_half_away_from_zero((x >= 0 ? static_cast<double>(x) : 0.2 * static_cast<double>(x)));
        lrelu_canonical << x << ':' << y << ';';
        ++context.lrelu_entry_count;
    }
    context.lrelu_digest = hash_text_id("lrelu", lrelu_canonical.str());

    std::ostringstream elu_canonical;
    for (std::int64_t x = -signed_bound; x <= signed_bound; ++x) {
        for (std::int64_t offset = -static_cast<std::int64_t>(quant.elu_band);
             offset <= static_cast<std::int64_t>(quant.elu_band);
             ++offset) {
            const auto shifted = static_cast<double>(x + offset) / static_cast<double>(quant.scale);
            const auto y = shifted >= 0.0
                ? zkdl_round_half_away_from_zero(shifted * static_cast<double>(quant.scale))
                : zkdl_round_half_away_from_zero((std::exp(shifted) - 1.0) * static_cast<double>(quant.scale));
            elu_canonical << x << ':' << offset << ':' << y << ';';
            ++context.elu_entry_count;
        }
    }
    context.elu_digest = hash_text_id("elu", elu_canonical.str());

    std::ostringstream exp_canonical;
    for (std::size_t x = 0; x < context.range_size; ++x) {
        for (std::int64_t offset = -static_cast<std::int64_t>(quant.exp_band);
             offset <= static_cast<std::int64_t>(quant.exp_band);
             ++offset) {
            const auto shifted = (static_cast<double>(x) + static_cast<double>(offset)) / static_cast<double>(quant.scale);
            const auto y = zkdl_round_half_away_from_zero(std::exp(-shifted) * static_cast<double>(quant.scale));
            exp_canonical << x << ':' << offset << ':' << y << ';';
            ++context.exp_entry_count;
        }
    }
    context.exp_digest = hash_text_id("exp", exp_canonical.str());

    std::ostringstream canonical;
    canonical << "static_table_id=" << context.static_table_id
              << ";range=" << context.range_size << ":" << context.range_digest
              << ";lrelu=" << context.lrelu_entry_count << ":" << context.lrelu_digest
              << ";elu=" << context.elu_entry_count << ":" << context.elu_digest
              << ";exp=" << context.exp_entry_count << ":" << context.exp_digest;
    context.context_id = hash_text_id("zkdl_table_context", canonical.str());
    return context;
}

ZkdlGatPublicMetadata build_zkdl_gat_public_metadata(
    const DatasetRunSpec& spec,
    const GraphDataset& dataset,
    const LocalGraph& graph,
    const ModelParams& model)
{
    ZkdlGatPublicMetadata metadata;
    metadata.dataset = spec.dataset;
    metadata.task_type = spec.task_type;
    metadata.report_unit = spec.report_unit;
    metadata.batching_rule = spec.batching_rule;
    metadata.model_arch_id = model.model_arch_id;
    metadata.model_param_id = model.model_param_id;
    metadata.quant_cfg_id = model.quant_cfg_id;
    metadata.static_table_id = model.static_table_id;
    metadata.node_count = graph.num_nodes;
    metadata.edge_count = graph.edges.size();
    metadata.feature_count = graph.num_features;
    metadata.class_count = graph.num_classes;
    metadata.graph_count = dataset.graph_count;
    metadata.model_layer_count = model.L;
    metadata.hidden_layer_count = model.hidden_profile.size();
    metadata.hidden_head_count = model.hidden_heads.size();
    metadata.output_head_count = model.output_heads.size();
    metadata.hidden_profile = hidden_profile_string(model);
    metadata.output_average_rule = model.output_average_rule;
    metadata.domain_cfg = "N=" + std::to_string(next_power_of_two(graph.num_nodes))
        + ";E=" + std::to_string(next_power_of_two(graph.edges.size()))
        + ";F=" + std::to_string(next_power_of_two(graph.num_features))
        + ";C=" + std::to_string(next_power_of_two(graph.num_classes));
    metadata.dim_cfg = "L=" + std::to_string(metadata.model_layer_count)
        + ";hidden_profile=" + metadata.hidden_profile
        + ";K_out=" + std::to_string(model.K_out)
        + ";C=" + std::to_string(model.C);
    std::ostringstream canonical;
    canonical << "protocol=" << metadata.protocol_id
              << ";scope=" << metadata.proof_scope_id
              << ";dataset=" << metadata.dataset
              << ";task=" << metadata.task_type
              << ";report=" << metadata.report_unit
              << ";nodes=" << metadata.node_count
              << ";edges=" << metadata.edge_count
              << ";features=" << metadata.feature_count
              << ";classes=" << metadata.class_count
              << ";graphs=" << metadata.graph_count
              << ";L=" << metadata.model_layer_count
              << ";batching=" << metadata.batching_rule
              << ";self_loop=" << metadata.self_loop_rule
              << ";edge_sort=" << metadata.edge_sort_rule
              << ";arch=" << metadata.model_arch_id
              << ";params=" << metadata.model_param_id
              << ";quant=" << metadata.quant_cfg_id
              << ";tables=" << metadata.static_table_id
              << ";domain=" << metadata.domain_cfg
              << ";dim=" << metadata.dim_cfg
              << ";padding=" << metadata.padding_rule_id
              << ";commitment=" << metadata.commitment_backend;
    metadata.metadata_id = hash_text_id("zkdl_public_metadata", canonical.str());
    return metadata;
}

void write_zkdl_gat_protocol_context_json(
    const fs::path& path,
    const ZkdlGatPublicMetadata& metadata,
    const ZkdlGatQuantContract& quant,
    const ZkdlGatTableContext& table)
{
    std::ofstream out(path);
    out << "{\n";
    out << "  \"protocol_id\": \"" << json_escape(metadata.protocol_id) << "\",\n";
    out << "  \"proof_scope_id\": \"" << json_escape(metadata.proof_scope_id) << "\",\n";
    out << "  \"metadata_id\": \"" << json_escape(metadata.metadata_id) << "\",\n";
    out << "  \"dataset\": \"" << json_escape(metadata.dataset) << "\",\n";
    out << "  \"public_input\": {\n";
    out << "    \"M_pub\": \"" << json_escape(metadata.dataset + ":features/labels/public-split-metadata") << "\",\n";
    out << "    \"Topo\": \"" << json_escape(metadata.edge_sort_rule + ";self_loop=" + metadata.self_loop_rule) << "\",\n";
    out << "    \"VK_static\": \"" << json_escape(table.context_id) << "\",\n";
    out << "    \"VK_model\": \"" << json_escape(metadata.model_param_id) << "\",\n";
    out << "    \"Y\": \"public_logits_commitment_target\"\n";
    out << "  },\n";
    out << "  \"dataset_shape\": {\n";
    out << "    \"node_count\": " << metadata.node_count << ",\n";
    out << "    \"edge_count\": " << metadata.edge_count << ",\n";
    out << "    \"feature_count\": " << metadata.feature_count << ",\n";
    out << "    \"class_count\": " << metadata.class_count << ",\n";
    out << "    \"graph_count\": " << metadata.graph_count << "\n";
    out << "  },\n";
    out << "  \"model_shape\": {\n";
    out << "    \"L\": " << metadata.model_layer_count << ",\n";
    out << "    \"hidden_profile\": \"" << json_escape(metadata.hidden_profile) << "\",\n";
    out << "    \"hidden_head_count\": " << metadata.hidden_head_count << ",\n";
    out << "    \"output_head_count\": " << metadata.output_head_count << ",\n";
    out << "    \"output_average_rule\": \"" << json_escape(metadata.output_average_rule) << "\"\n";
    out << "  },\n";
    out << "  \"VK_model\": {\n";
    out << "    \"model_arch_id\": \"" << json_escape(metadata.model_arch_id) << "\",\n";
    out << "    \"model_param_id\": \"" << json_escape(metadata.model_param_id) << "\",\n";
    out << "    \"dim_cfg\": \"" << json_escape(metadata.dim_cfg) << "\"\n";
    out << "  },\n";
    out << "  \"VK_static\": {\n";
    out << "    \"static_table_id\": \"" << json_escape(metadata.static_table_id) << "\",\n";
    out << "    \"table_context_id\": \"" << json_escape(table.context_id) << "\",\n";
    out << "    \"range_size\": " << table.range_size << ",\n";
    out << "    \"lrelu_entry_count\": " << table.lrelu_entry_count << ",\n";
    out << "    \"elu_entry_count\": " << table.elu_entry_count << ",\n";
    out << "    \"exp_entry_count\": " << table.exp_entry_count << ",\n";
    out << "    \"range_digest\": \"" << json_escape(table.range_digest) << "\",\n";
    out << "    \"lrelu_digest\": \"" << json_escape(table.lrelu_digest) << "\",\n";
    out << "    \"elu_digest\": \"" << json_escape(table.elu_digest) << "\",\n";
    out << "    \"exp_digest\": \"" << json_escape(table.exp_digest) << "\"\n";
    out << "  },\n";
    out << "  \"quantization_contract\": {\n";
    out << "    \"contract_id\": \"" << json_escape(quant.contract_id) << "\",\n";
    out << "    \"quant_cfg_id\": \"" << json_escape(quant.quant_cfg_id) << "\",\n";
    out << "    \"encoding_id\": \"" << json_escape(quant.encoding_id) << "\",\n";
    out << "    \"scale\": " << quant.scale << ",\n";
    out << "    \"range_bits\": " << quant.range_bits << ",\n";
    out << "    \"rounding_policy\": \"" << json_escape(quant.rounding_policy) << "\",\n";
    out << "    \"rescale_policy\": \"" << json_escape(quant.rescale_policy) << "\"\n";
    out << "  },\n";
    out << "  \"commitment_backend\": \"" << json_escape(metadata.commitment_backend) << "\",\n";
    out << "  \"说明\": \"该文件只记录 full-chain public metadata/VK/table/rounding contract；不是 proof object，也不会写入 official latest。\"\n";
    out << "}\n";
}
