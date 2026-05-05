#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace zkdl_gat {

struct DatasetConfig {
    std::string dataset_id;
    std::string dataset_name;
    std::string source_config;
    std::string cache_dir;
    std::string checkpoint_dir;
    std::string main_cpu_manifest;
    std::string main_gpu_manifest;
    std::string batching_rule;
    std::string self_loop_rule;
    std::string edge_sort_rule;
    std::string chunking_rule;
    std::string task_type;
    std::string report_unit;
    std::string hidden_profile;
    std::string d_in_profile;
    std::size_t layer_count = 0;
    std::size_t k_out = 0;
    std::size_t num_classes = 0;
    std::size_t batch_graphs = 1;
    bool deduplicate_edges = true;
};

struct GpuProbe {
    bool gpu_backend_active = false;
    std::string gpu_device_name;
    std::size_t gpu_device_count = 0;
    std::string gpu_backend_name;
    bool cpu_fallback_used = false;
    std::string primitive_probe_status;
    std::optional<double> primitive_commitment_time_ms;
    std::optional<double> primitive_prove_time_ms;
    std::optional<double> primitive_verify_time_ms;
    std::optional<std::size_t> primitive_proof_size_bytes;
};

struct Manifest {
    std::map<std::string, std::string> string_values;
    std::map<std::string, bool> bool_values;
    std::map<std::string, std::size_t> size_values;
    std::map<std::string, std::optional<double>> double_values;
    std::map<std::string, std::optional<std::size_t>> optional_size_values;
    std::map<std::string, std::string> dataset_stats;
    std::map<std::string, std::string> source_files;
    std::map<std::string, std::string> source_sha256;
    std::vector<std::string> capability_gaps;
};

DatasetConfig load_dataset_config(const std::string& path);
GpuProbe load_gpu_probe(const std::string& path);
Manifest build_manifest(const DatasetConfig& config, const std::string& route, const GpuProbe& gpu_probe);
void write_manifest_json(const Manifest& manifest, const std::string& path);

}  // namespace zkdl_gat
