#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace zkdl_gat {

struct DatasetConfig {
    std::map<std::string, std::string> values;
    std::string path;
};

struct GpuProbe {
    bool loaded = false;
    bool gpu_backend_active = false;
    bool cpu_fallback_used = false;
    int gpu_device_count = 0;
    std::string gpu_device_name;
    std::string gpu_backend_name;
    std::string primitive_probe_status;
};

struct Manifest {
    std::string dataset;
    std::string route_id;
    std::string backend;
    std::string status;
    bool strict_public_input_equivalent = false;
    bool full_gat_forward_in_zkdl_proof = false;
    bool final_logits_derivation_complete = false;
    bool no_precomputed_logits_bypass = true;
    std::optional<double> commitment_time_ms;
    std::optional<double> prove_time_ms;
    std::optional<double> verify_time_ms;
    std::optional<unsigned long long> proof_size_bytes;
    std::string block_reason;
    std::string failure_step;
    bool gpu_backend_active = false;
    std::string gpu_device_name;
    int gpu_device_count = 0;
    std::string gpu_backend_name;
    bool cpu_fallback_used = false;
    std::vector<std::string> capability_gaps;
    std::map<std::string, std::string> source_files;
    std::map<std::string, std::string> source_sha256;
    std::map<std::string, std::string> dataset_stats;
};

DatasetConfig load_dataset_config(const std::string& path);
GpuProbe load_gpu_probe(const std::string& path);
Manifest build_manifest(
    const DatasetConfig& config,
    const std::string& route,
    const std::string& experiment_root,
    const GpuProbe& gpu_probe);
void write_manifest(const Manifest& manifest, const std::string& output_root);
int run_cli(int argc, char** argv);

}  // namespace zkdl_gat
