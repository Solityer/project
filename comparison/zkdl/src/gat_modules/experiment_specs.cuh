#pragma once

// 五个正式全量数据集的唯一入口配置，顺序必须与主实验结果表一致。

std::vector<DatasetRunSpec> default_dataset_run_specs()
{
    return {
        {"cora", "cora", "cora_gat", "cora_full.cfg", "transductive_node_classification", "node", "whole_graph_single", 1, 12},
        {"citeseer", "citeseer", "citeseer_gat", "citeseer_full.cfg", "transductive_node_classification", "node", "whole_graph_single", 1, 12},
        {"pubmed", "pubmed", "pubmed_gat", "pubmed_full.cfg", "transductive_node_classification", "node", "whole_graph_single", 1, 12},
        {"ppi", "ppi", "ppi_gat", "ppi_batch_formal.cfg", "inductive_multi_graph_node_classification", "graph", "multi_graph_batch", 295, 12},
        {"ogbn-arxiv", "ogbn_arxiv", "ogbn_arxiv_gat", "ogbn_arxiv_full.cfg", "transductive_node_classification", "node", "whole_graph_single", 1, 19},
    };
}

DatasetRunSpec dataset_spec_or_throw(
    const std::vector<DatasetRunSpec>& specs,
    const std::string& dataset,
    const std::string& mode)
{
    for (const auto& spec : specs) {
        if (spec.dataset == dataset) {
            return spec;
        }
    }
    throw std::runtime_error("unknown dataset for " + mode + ": " + dataset);
}
