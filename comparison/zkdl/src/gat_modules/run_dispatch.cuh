#pragma once

// official 调度层只选择 dataset/device，并把 full-chain proof 生成交给对应后端。
// 这里不保留 partial/debug 入口，避免 relation-level artifact 被误当成 official。

// 运行单个 dataset/device 的完整 GAT proof，并写 full proof/verifier manifest。
// 输入只含公开路径和调度参数；副作用限定在 output_root 的该 dataset/device run 目录。
void run_selected_full_chain(
    const fs::path& project_root,
    const fs::path& output_root,
    const std::vector<DatasetRunSpec>& specs,
    const std::string& dataset,
    const std::string& device,
    std::size_t requested_commitment_block_size)
{
    if (dataset.empty()) {
        throw std::runtime_error("--formal-full-chain requires --dataset");
    }
    const auto& spec = dataset_spec_or_throw(specs, dataset, "formal full-chain runner");
    if (device == "gpu") {
        (void)run_gpu_full_chain_proof(project_root, output_root, spec, requested_commitment_block_size);
    } else if (device == "cpu") {
        (void)run_cpu_full_chain_proof(project_root, output_root, spec, requested_commitment_block_size);
    } else {
        throw std::runtime_error("--formal-full-chain requires --device cpu|gpu");
    }
}
