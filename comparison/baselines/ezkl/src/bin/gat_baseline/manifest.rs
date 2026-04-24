use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize)]
struct ManifestFile {
    datasets: BTreeMap<String, DatasetSpec>,
}

#[derive(Debug, Clone, Deserialize)]
struct DatasetSpec {
    dataset: String,
    output_relpath: String,
    config_relpath: String,
    cache_relpath: String,
    checkpoint_relpath: String,
    reference_run_manifest_relpath: String,
    task_type: String,
    report_unit: String,
    batching_rule: String,
    self_loop_rule: String,
    edge_sort_rule: String,
    chunking_rule: String,
    deduplicate_edges: bool,
    symmetrize_edges: bool,
    batch_graphs: usize,
    protocol_id: String,
    preprocess_id: String,
    attestation_mode: String,
    encoding_id: String,
    proof_system_tag: String,
    quant_cfg_id: String,
    degree_bound_id: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ResolvedDatasetSpec {
    pub dataset: String,
    pub output_root: PathBuf,
    pub reference_run_manifest_path: PathBuf,
    pub task_type: String,
    pub report_unit: String,
    pub batching_rule: String,
    pub self_loop_rule: String,
    pub edge_sort_rule: String,
    pub chunking_rule: String,
    pub deduplicate_edges: bool,
    pub symmetrize_edges: bool,
    pub batch_graphs: usize,
    pub protocol_id: String,
    pub preprocess_id: String,
    pub attestation_mode: String,
    pub encoding_id: String,
    pub proof_system_tag: String,
    pub quant_cfg_id: String,
    pub degree_bound_id: String,
    pub config_path: PathBuf,
    pub cache_path: PathBuf,
    pub checkpoint_path: PathBuf,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

pub fn load_dataset_spec(dataset: &str) -> Result<ResolvedDatasetSpec, String> {
    let manifest_path = repo_root().join("manifests/gat_formal_datasets.json");
    let text = fs::read_to_string(&manifest_path)
        .map_err(|err| format!("failed to read {}: {err}", manifest_path.display()))?;
    let manifest: ManifestFile = serde_json::from_str(&text)
        .map_err(|err| format!("failed to parse dataset manifest: {err}"))?;
    let spec = manifest
        .datasets
        .get(dataset)
        .ok_or_else(|| format!("unknown dataset '{dataset}'"))?;
    let root = repo_root();

    Ok(ResolvedDatasetSpec {
        dataset: spec.dataset.clone(),
        output_root: root.join(&spec.output_relpath),
        reference_run_manifest_path: root.join(&spec.reference_run_manifest_relpath),
        task_type: spec.task_type.clone(),
        report_unit: spec.report_unit.clone(),
        batching_rule: spec.batching_rule.clone(),
        self_loop_rule: spec.self_loop_rule.clone(),
        edge_sort_rule: spec.edge_sort_rule.clone(),
        chunking_rule: spec.chunking_rule.clone(),
        deduplicate_edges: spec.deduplicate_edges,
        symmetrize_edges: spec.symmetrize_edges,
        batch_graphs: spec.batch_graphs,
        protocol_id: spec.protocol_id.clone(),
        preprocess_id: spec.preprocess_id.clone(),
        attestation_mode: spec.attestation_mode.clone(),
        encoding_id: spec.encoding_id.clone(),
        proof_system_tag: spec.proof_system_tag.clone(),
        quant_cfg_id: spec.quant_cfg_id.clone(),
        degree_bound_id: spec.degree_bound_id.clone(),
        config_path: root.join(&spec.config_relpath),
        cache_path: root.join(&spec.cache_relpath),
        checkpoint_path: root.join(&spec.checkpoint_relpath),
    })
}
