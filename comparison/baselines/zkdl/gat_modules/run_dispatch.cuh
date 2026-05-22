#pragma once

// CLI 调度层只选择 dataset/device，并把执行交给独立 CPU/GPU runner。

void run_selected_cpu_formal_chain(
    const fs::path& project_root,
    const fs::path& output_root,
    const std::vector<DatasetRunSpec>& specs,
    const std::string& dataset,
    std::size_t requested_commitment_block_size)
{
    if (dataset.empty()) {
        throw std::runtime_error("--formal-cpu-chain requires --dataset");
    }
    const auto& spec = dataset_spec_or_throw(specs, dataset, "CPU formal chain");
    run_cpu_formal_chain(project_root, output_root, spec, requested_commitment_block_size);
}

void run_selected_sparse_grouped(
    const fs::path& project_root,
    const fs::path& output_root,
    const std::vector<DatasetRunSpec>& specs,
    const std::string& dataset,
    std::size_t requested_commitment_block_size)
{
    if (dataset.empty()) {
        throw std::runtime_error("--formal-sparse-grouped requires --dataset");
    }
    const auto& spec = dataset_spec_or_throw(specs, dataset, "formal sparse grouped gadget");
    run_formal_sparse_grouped_gadget(project_root, output_root, spec, requested_commitment_block_size);
}

void run_selected_softmax_aggregation(
    const fs::path& project_root,
    const fs::path& output_root,
    const std::vector<DatasetRunSpec>& specs,
    const std::string& dataset,
    std::size_t max_feature_columns,
    std::size_t requested_commitment_block_size)
{
    if (dataset.empty()) {
        throw std::runtime_error("--formal-softmax-aggregation requires --dataset");
    }
    const auto& spec = dataset_spec_or_throw(specs, dataset, "formal softmax aggregation gadget");
    run_formal_softmax_aggregation_gadget(
        project_root, output_root, spec, max_feature_columns, requested_commitment_block_size);
}

void run_selected_output_binding(
    const fs::path& project_root,
    const fs::path& output_root,
    const std::vector<DatasetRunSpec>& specs,
    const std::string& dataset,
    std::size_t requested_commitment_block_size)
{
    if (dataset.empty()) {
        throw std::runtime_error("--formal-output-binding requires --dataset");
    }
    const auto& spec = dataset_spec_or_throw(specs, dataset, "formal output binding gadget");
    run_formal_output_binding_gadget(project_root, output_root, spec, requested_commitment_block_size);
}

void run_selected_full_chain_scope(
    const fs::path& project_root,
    const fs::path& output_root,
    const std::vector<DatasetRunSpec>& specs,
    const std::string& dataset,
    const std::string& device)
{
    if (dataset.empty()) {
        throw std::runtime_error("--formal-full-chain-scope requires --dataset");
    }
    if (device != "cpu" && device != "gpu") {
        throw std::runtime_error("--formal-full-chain-scope requires --device cpu|gpu");
    }
    const auto& spec = dataset_spec_or_throw(specs, dataset, "formal full-chain scope");
    run_formal_full_chain_scope(project_root, output_root, spec, device);
}

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
