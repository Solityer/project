//! Rust-only loader for the packaged canonical GAT resources.
//!
//! The loader is deliberately independent of Python/NumPy. It authenticates
//! the checked-in resource manifest first, parses the five dataset caches and
//! checkpoint tensor archive, applies canonical ties-away quantization, and
//! constructs the exact tables/ledger consumed by the native graph op.

use crate::fieldutils::integer_rep_to_felt;
use crate::gat_adapter::manifest::{AuthenticatedResource, ResourceManifest, ResourceSourceSnapshot,};
use crate::gat_adapter::schema::{
    canonical_tensor_archive_key, multiplier_to_scale_exponent, sha256_digest, CanonicalGat, CanonicalGraph, CanonicalModel, DigestIdentity, ExactTable,
    HeadWeights, HiddenLayerSpec, IntegerBoundLedger, ModelBinding, ScaleChain, StaticTables, GAT_SCHEMA_VERSION,
};
use crate::graph::modules::native_poseidon_hash_string;
use halo2curves::bn256::Fr;
use serde::Deserialize;
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};

/// A parsed canonical dataset/checkpoint bundle.
#[derive(Clone, Debug)]
pub struct LoadedGat {
    /// Fully validated canonical adapter input.
    pub spec: CanonicalGat,
    /// Dataset name.
    pub dataset: String,
    /// Digest of the authenticated resource manifest.
    pub resource_manifest_sha256: String,
    /// Baseline root from which the bundle was loaded.
    pub baseline_root: PathBuf,
    /// Producer snapshot copied from the authenticated resource manifest.
    pub source_snapshot: ResourceSourceSnapshot,
}

/// Authenticated public relation resources used by formal verification.
///
/// This path intentionally parses only the manifest, secure registry
/// configuration, static tables and integer ledger. It does not load
/// checkpoint tensors or reconstruct the private model.
#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub struct AuthenticatedRelationResources {
    pub tables: StaticTables,
    pub ledger: IntegerBoundLedger,
    pub quantization: DigestIdentity,
    pub resource_manifest_sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorMeta {
    archive_key: String,
    dtype: String,
    shape: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)]
struct HiddenLayerMeta {
    head_count: usize,
    head_dim: usize,
    input_dim: usize,
    layer_index: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HiddenHeadSpec {
    attn_dst_kernel: String,
    attn_src_kernel: String,
    layer_index: usize,
    local_head_index: usize,
    seq_kernel: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HiddenProfileMeta {
    head_count: usize,
    head_dim: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OutputHeadSpec {
    attn_dst_kernel: String,
    attn_src_kernel: String,
    head_index: usize,
    output_bias: String,
    seq_kernel: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)]
struct CheckpointManifest {
    #[serde(rename = "C")]
    classes: usize,
    #[serde(rename = "K_out")]
    output_heads: usize,
    #[serde(rename = "L")]
    layers: usize,
    batching_rule: String,
    checkpoint_prefix: String,
    d_in_profile: Vec<usize>,
    dataset: String,
    degree_bound_id: String,
    static_table_digest: String,
    degree_bound_digest: String,
    edge_sort_rule: String,
    family_schema: String,
    hidden_head_specs: Vec<HiddenHeadSpec>,
    hidden_layers: Vec<HiddenLayerMeta>,
    hidden_profile: Vec<HiddenProfileMeta>,
    input_preprocess_id: String,
    label_format: String,
    loss: String,
    model_arch_id: String,
    model_param_id: String,
    output_average_rule: String,
    output_head_specs: Vec<OutputHeadSpec>,
    quant_cfg_id: String,
    quantization_scale_chain: String,
    report_unit: String,
    selection_metric: String,
    self_loop_rule: String,
    split_format: String,
    static_table_id: String,
    subgraph_rule: String,
    task_type: String,
    tensor_count: usize,
    tensor_index: BTreeMap<String, TensorMeta>,
}

#[derive(Clone, Debug)]
struct DatasetMeta {
    name: String,
    num_nodes: usize,
    num_features: usize,
    num_classes: usize,
    graph_count: usize,
    input_preprocess_id: String,
    feature_format: Option<String>,
    feature_dtype: Option<String>,
    feature_shape: Option<Vec<usize>>,
    feature_order: Option<String>,
    feature_length: Option<usize>,
    feature_sha256: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StaticTableRegistryMeta {
    format: String,
    version: u32,
    static_tables_id: String,
    backend_aux_tables_id: String,
    static_tables_digest: String,
    table_order: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StaticTableArchiveMeta {
    table_id: String,
    input_scale: i32,
    output_scale: i32,
    function_id: String,
    domain_id: String,
    generator_id: String,
    input_min: i128,
    input_max: i128,
    row_count: u64,
    digest: String,
    canonical_len: u64,
}

#[derive(Clone, Debug)]
struct ParsedTensor {
    shape: Vec<usize>,
    values: Vec<f64>,
}

#[derive(Clone, Debug)]
struct ParsedNpy {
    shape: Vec<usize>,
    dtype: String,
    order: String,
    /// Quantized values are materialized directly while the authenticated
    /// payload is streamed.  Keeping only this representation avoids the
    /// historical `Vec<f64>` plus `Vec<i128>` peak for PPI/OGBN features.
    values: Vec<i128>,
}

/// Load and authenticate one of the five canonical dataset bundles.
pub fn load_dataset(baseline_root: &Path, dataset: &str) -> Result<LoadedGat, String> {
    let dataset = canonical_dataset_name(dataset)?;
    let manifest_path = baseline_root.join("resources/resource_manifest.json");
    load_dataset_with_manifest(baseline_root, &manifest_path, &dataset)
}

/// Load a dataset using an explicit baseline-relative manifest path.
pub fn load_dataset_with_manifest(baseline_root: &Path, manifest_path: &Path, dataset: &str,) -> Result<LoadedGat, String> {
    let dataset = canonical_dataset_name(dataset)?;
    let manifest = ResourceManifest::load_from_baseline(baseline_root, manifest_path)?;
    let resources = manifest.authenticate(baseline_root)?;
    let manifest_bytes = fs::read(manifest_path).map_err(|e| format!("read resource manifest {}: {}", manifest_path.display(), e))?;
    let resource_manifest_sha256 = sha256_digest(&manifest_bytes)?;
    load_from_resources(baseline_root, &dataset, &resources, resource_manifest_sha256, manifest.source_snapshot.clone(),) }

/// Authenticate the current resource manifest and parse only the public
/// table/ledger registry required by a verifier.
pub fn load_relation_resources_with_manifest(baseline_root: &Path, manifest_path: &Path, dataset: &str,) -> Result<AuthenticatedRelationResources, String> {
    let dataset = canonical_dataset_name(dataset)?;
    let manifest = ResourceManifest::load_from_baseline(baseline_root, manifest_path)?;
    let resources = manifest.authenticate(baseline_root)?;
    let manifest_bytes = fs::read(manifest_path).map_err(|e| format!("read resource manifest {}: {}", manifest_path.display(), e))?;
    let resource_manifest_sha256 = sha256_digest(&manifest_bytes)?;
    let secure_cfg = resource_path(&resources, &format!("configs/{}_secure.cfg", dataset))?;
    let secure = parse_cfg(&secure_cfg.path)?;
    require_cfg(&secure, &["checkpoint_bundle", "quant_cfg_id", "static_table_id", "degree_bound_id", "static_table_digest", "degree_bound_digest",],)?;
    let checkpoint_manifest = resource_path(&resources, &format!("{}/manifest.json", secure["checkpoint_bundle"].trim_start_matches("resources/canonical/")),)?;
    let quant_cfg = resource_path(&resources, "configs/quantization_scale_chain.cfg")?;
    let checkpoint: CheckpointManifest = parse_json(&checkpoint_manifest.path)?;
    let scales = parse_quant_chain(&quant_cfg.path, &checkpoint)?;
    let quantization = DigestIdentity { id: checkpoint.quant_cfg_id, digest: sha256_digest(&scales.canonical_bytes())?,};
    let (tables, ledger) = load_exact_relation_resources(&resources, &dataset, &secure)?;
    Ok(AuthenticatedRelationResources { tables, ledger, quantization, resource_manifest_sha256,}) }

fn canonical_dataset_name(name: &str) -> Result<String, String> {
    let name = name.trim().to_ascii_lowercase();
    match name.as_str() {
        "cora" | "citeseer" | "pubmed" | "ppi" | "ogbn_arxiv" => Ok(name),
        _ => Err(format!("unsupported canonical GAT dataset: {}", name)),
    }
}

fn load_from_resources(
    baseline_root: &Path,
    dataset: &str,
    resources: &[AuthenticatedResource],
    resource_manifest_sha256: String,
    source_snapshot: ResourceSourceSnapshot,
) -> Result<LoadedGat, String> {
    let secure_cfg = resource_path(resources, &format!("configs/{}_secure.cfg", dataset))?;
    let checkpoint_root = format!("checkpoints/{}_gat", dataset);
    let checkpoint_manifest = resource_path(resources, &format!("{}/manifest.json", checkpoint_root))?;
    let tensors_txt = resource_path(resources, &format!("{}/tensors.txt", checkpoint_root))?;
    let quant_cfg = resource_path(resources, "configs/quantization_scale_chain.cfg")?;
    let meta_path = resource_path(resources, &format!("cache/{}/meta.cfg", dataset))?;
    let edges_path = resource_path(resources, &format!("cache/{}/edges.txt", dataset))?;
    let feature_name = if dataset == "cora" || dataset == "citeseer" || dataset == "pubmed" { "features.txt" } else { "features.npy" };
    let features_path = resource_path(resources, &format!("cache/{}/{}", dataset, feature_name))?;

    let secure = parse_cfg(&secure_cfg.path)?;
    let meta = parse_dataset_meta(&meta_path.path)?;
    if meta.name != dataset {
        return Err(format!("dataset meta name {} disagrees with {}", meta.name, dataset));
    }
    require_cfg(&secure, &["checkpoint_bundle", "quant_cfg_id", "static_table_id", "degree_bound_id", "ezkl_security_profile_id",],)?;
    let checkpoint: CheckpointManifest = parse_json(&checkpoint_manifest.path)?;
    validate_checkpoint(&checkpoint, dataset, &meta, &secure)?;
    let quant = parse_quant_chain(&quant_cfg.path, &checkpoint)?;
    let features = parse_features(&features_path.path, &meta, &quant)?;
    let graph = parse_graph(&edges_path.path, &meta, dataset, resources)?;
    let tensors = parse_tensor_archive(&tensors_txt.path, &checkpoint)?;
    let model = build_model(&checkpoint, &quant, &tensors)?;
    let (tables, ledger) = load_exact_relation_resources(resources, dataset, &secure)?;
    let topology = DigestIdentity { id: graph.topology_id.clone(), digest: graph.topology_digest.clone(),};
    let quantization = DigestIdentity { id: checkpoint.quant_cfg_id.clone(), digest: sha256_digest(&quant.canonical_bytes())?,};
    let ledger_identity = DigestIdentity { id: ledger.ledger_id.clone(), digest: ledger.ledger_digest.clone(),};
    let model_values = model.flattened_params();
    let canonical_model_digest = {
        let mut bytes = b"ezkl-gat-model-v1".to_vec();
        for value in &model_values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        sha256_digest(&bytes)?
    };
    let native_model_hash = native_poseidon_hash_string(&model_values.iter().copied().map(integer_rep_to_felt::<Fr>).collect::<Vec<_>>(),)?;
    let profile = secure.get("ezkl_security_profile_id").cloned().ok_or_else(|| "missing ezkl_security_profile_id".to_string())?;
    let model_binding = ModelBinding { hash: native_model_hash, hash_domain: "ezkl-gat-model-v1".into(), canonical_model_digest,};
    let spec = CanonicalGat {
        schema_version: GAT_SCHEMA_VERSION,
        graph,
        features,
        feature_dim: meta.num_features,
        model,
        scales: quant,
        tables,
        ledger,
        quant_cfg_id: checkpoint.quant_cfg_id,
        ezkl_security_profile_id: profile,
        topology,
        quantization,
        ledger_identity,
        circuit_ref: None,
        model_binding,
        private_params_redacted: false,
    };
    spec.validate_for_compile()?;
    Ok(LoadedGat { spec, dataset: dataset.to_string(), resource_manifest_sha256, baseline_root: baseline_root.to_path_buf(), source_snapshot,}) }

fn resource_path(resources: &[AuthenticatedResource], suffix: &str,) -> Result<AuthenticatedResource, String> {
    resources.iter().find(|r| r.record.path.ends_with(suffix)).cloned().ok_or_else(|| format!("authenticated canonical resource not found: {}", suffix))
}

fn parse_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {}", path.display(), e))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("parse {}: {}", path.display(), e))
}

fn parse_cfg(path: &Path) -> Result<BTreeMap<String, String>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("read {}: {}", path.display(), e))?;
    let mut values = BTreeMap::new();
    for (line_no, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let (key, value) = line.split_once('=').ok_or_else(|| format!("{}:{} is not key=value", path.display(), line_no + 1))?;
        let key = key.trim().to_string();
        if key.is_empty() || values.insert(key.clone(), value.trim().to_string()).is_some() {
            return Err(format!("duplicate/empty config key {}:{}", path.display(), line_no + 1));
        }
    }
    Ok(values)
}

fn require_cfg(cfg: &BTreeMap<String, String>, required: &[&str]) -> Result<(), String> {
    for key in required {
        if !cfg.contains_key(*key) {
            return Err(format!("missing required config key {}", key));
        }
    }
    Ok(())
}

fn parse_dataset_meta(path: &Path) -> Result<DatasetMeta, String> {
    let cfg = parse_cfg(path)?;
    let get = |key: &str| { cfg.get(key).cloned().ok_or_else(|| format!("missing dataset meta key {}", key)) };
    let feature_format = cfg.get("feature_format").cloned();
    let feature_dtype = cfg.get("feature_dtype").cloned();
    let feature_shape =
        cfg.get("feature_shape").map(|value| { value.split(',').map(|x| parse_usize(x.trim(), "feature_shape")).collect::<Result<Vec<_>, _>>() }).transpose()?;
    let feature_order = cfg.get("feature_order").cloned();
    let feature_length = cfg.get("feature_length").map(|x| parse_usize(x, "feature_length")).transpose()?;
    let feature_sha256 = cfg.get("feature_sha256").cloned();
    let feature_fields = [
        feature_format.is_some(),
        feature_dtype.is_some(),
        feature_shape.is_some(),
        feature_order.is_some(),
        feature_length.is_some(),
        feature_sha256.is_some(),
    ];
    if feature_fields.iter().any(|present| *present) && !feature_fields.iter().all(|present| *present) {
        return Err("NumPy feature metadata must provide format/dtype/shape/order/length/sha256 together".into(),);
    }
    Ok(DatasetMeta {
        name: get("name")?,
        num_nodes: parse_usize(&get("num_nodes")?, "num_nodes")?,
        num_features: parse_usize(&get("num_features")?, "num_features")?,
        num_classes: parse_usize(&get("num_classes")?, "num_classes")?,
        graph_count: cfg.get("graph_count").map(|x| parse_usize(x, "graph_count")).transpose()?.unwrap_or(1),
        input_preprocess_id: get("input_preprocess_id")?,
        feature_format,
        feature_dtype,
        feature_shape,
        feature_order,
        feature_length,
        feature_sha256,
    })
}

fn validate_checkpoint(checkpoint: &CheckpointManifest, dataset: &str, meta: &DatasetMeta, secure: &BTreeMap<String, String>,) -> Result<(), String> {
    if checkpoint.dataset != dataset || checkpoint.classes != meta.num_classes || checkpoint.d_in_profile.first().copied() != Some(meta.num_features) {
        return Err("checkpoint shape/topology does not match dataset metadata".into());
    }
    if checkpoint.layers < 2
        || checkpoint.layers != checkpoint.hidden_layers.len() + 1
        || checkpoint.d_in_profile.len() != checkpoint.hidden_layers.len()
        || checkpoint.hidden_profile.len() != checkpoint.hidden_layers.len()
        || checkpoint.output_heads == 0
        || checkpoint.output_head_specs.len() != checkpoint.output_heads
    {
        return Err("checkpoint layer/head schema is not closed".into());
    }

    let expected_tensors = expected_tensor_layout(checkpoint)?;
    if checkpoint.tensor_count != expected_tensors.len() || checkpoint.tensor_index.len() != expected_tensors.len() {
        return Err(format!(
            "checkpoint tensor count {} does not match authenticated architecture (expected {})",
            checkpoint.tensor_count,
            expected_tensors.len()
        ));
    }
    let expected_names = expected_tensors.iter().map(|(name, _)| name.as_str()).collect::<HashSet<_>>();
    let actual_names = checkpoint.tensor_index.keys().map(String::as_str).collect::<HashSet<_>>();
    if actual_names != expected_names {
        return Err("checkpoint tensor roles do not match authenticated architecture".into());
    }
    for (name, shape) in &expected_tensors {
        let tensor = checkpoint.tensor_index.get(name).ok_or_else(|| format!("checkpoint is missing tensor {}", name))?;
        if tensor.shape != *shape {
            return Err(format!("checkpoint tensor {} shape {:?} does not match authenticated shape {:?}", name, tensor.shape, shape));
        }
    }

    let mut hidden_spec_cursor = 0_usize;
    for (layer_idx, layer) in checkpoint.hidden_layers.iter().enumerate() {
        if layer.layer_index != layer_idx
            || checkpoint.d_in_profile[layer_idx] != layer.input_dim
            || checkpoint.hidden_profile[layer_idx].head_count != layer.head_count
            || checkpoint.hidden_profile[layer_idx].head_dim != layer.head_dim
            || layer.head_count == 0
            || layer.head_dim == 0
        {
            return Err(format!("checkpoint hidden layer {} architecture metadata is not closed", layer_idx));
        }
        if layer_idx > 0 {
            let previous = &checkpoint.hidden_layers[layer_idx - 1];
            let expected_input = previous.head_count.checked_mul(previous.head_dim).ok_or("checkpoint hidden dimension overflow")?;
            if layer.input_dim != expected_input {
                return Err(format!("checkpoint hidden layer {} input dimension does not match previous concat", layer_idx));
            }
        }
        for head_idx in 0..layer.head_count {
            let spec = checkpoint.hidden_head_specs.get(hidden_spec_cursor).ok_or("checkpoint hidden head specification count mismatch")?;
            let prefix = format!("hidden/layer{}/head{}", layer_idx, head_idx);
            if spec.layer_index != layer_idx
                || spec.local_head_index != head_idx
                || spec.attn_dst_kernel != format!("{}/attn_dst/kernel", prefix)
                || spec.attn_src_kernel != format!("{}/attn_src/kernel", prefix)
                || spec.seq_kernel != format!("{}/seq/kernel", prefix)
            {
                return Err(format!("checkpoint hidden head specification order/role mismatch at layer {} head {}", layer_idx, head_idx));
            }
            hidden_spec_cursor += 1;
        }
    }
    if hidden_spec_cursor != checkpoint.hidden_head_specs.len() {
        return Err("checkpoint hidden head specification count mismatch".into());
    }
    for (head_idx, spec) in checkpoint.output_head_specs.iter().enumerate() {
        let prefix = format!("output/head{}", head_idx);
        if spec.head_index != head_idx
            || spec.attn_dst_kernel != format!("{}/attn_dst/kernel", prefix)
            || spec.attn_src_kernel != format!("{}/attn_src/kernel", prefix)
            || spec.seq_kernel != format!("{}/seq/kernel", prefix)
            || spec.output_bias != format!("{}/output_bias", prefix)
        {
            return Err(format!("checkpoint output head specification order/role mismatch at head {}", head_idx));
        }
    }
    if checkpoint.family_schema != "multi_layer_multi_head"
        || checkpoint.output_average_rule != "per_head_bias_then_arithmetic_mean"
        || checkpoint.input_preprocess_id != meta.input_preprocess_id
    {
        return Err("checkpoint protocol identity mismatch".into());
    }
    for key in ["quant_cfg_id", "static_table_id", "degree_bound_id", "static_table_digest", "degree_bound_digest",] {
        let expected = match key {
            "quant_cfg_id" => &checkpoint.quant_cfg_id,
            "static_table_id" => &checkpoint.static_table_id,
            "degree_bound_id" => &checkpoint.degree_bound_id,
            "static_table_digest" => &checkpoint.static_table_digest,
            "degree_bound_digest" => &checkpoint.degree_bound_digest,
            _ => unreachable!(),
        };
        if secure.get(key) != Some(expected) {
            return Err(format!("secure config/checkpoint {} mismatch", key));
        }
    }
    Ok(())
}

/// Derive the canonical tensor role order and shape from the authenticated
/// architecture.  Three tensors belong to every hidden head and four to
/// every output head; this yields 10 tensors for the compact 1/1/1 fixture and
/// 56 tensors for the formal 8/8/2 model without weakening either contract.
fn expected_tensor_layout(checkpoint: &CheckpointManifest,) -> Result<Vec<(String, Vec<usize>)>, String> {
    let mut expected = Vec::new();
    for (layer_idx, layer) in checkpoint.hidden_layers.iter().enumerate() {
        for head_idx in 0..layer.head_count {
            let prefix = format!("hidden/layer{}/head{}", layer_idx, head_idx);
            expected.push((format!("{}/attn_dst/kernel", prefix), vec![1, layer.head_dim, 1],));
            expected.push((format!("{}/attn_src/kernel", prefix), vec![1, layer.head_dim, 1],));
            expected.push((format!("{}/seq/kernel", prefix), vec![1, layer.input_dim, layer.head_dim],));
        }
    }
    let output_input_dim = checkpoint
        .hidden_layers
        .last()
        .ok_or("checkpoint has no hidden layers")?
        .head_count
        .checked_mul(checkpoint.hidden_layers.last().ok_or("checkpoint has no hidden layers")?.head_dim,)
        .ok_or("checkpoint output input dimension overflow")?;
    for head_idx in 0..checkpoint.output_heads {
        let prefix = format!("output/head{}", head_idx);
        expected.push((format!("{}/attn_dst/kernel", prefix), vec![1, checkpoint.classes, 1],));
        expected.push((format!("{}/attn_src/kernel", prefix), vec![1, checkpoint.classes, 1],));
        expected.push((format!("{}/seq/kernel", prefix), vec![1, output_input_dim, checkpoint.classes],));
        expected.push((format!("{}/output_bias", prefix), vec![1, checkpoint.classes, 1],));
    }
    Ok(expected)
}

fn parse_usize(value: &str, label: &str) -> Result<usize, String> {
    value.trim().parse::<usize>().map_err(|e| format!("{} is not usize: {}", label, e))
}

fn parse_scale(value: &str, label: &str) -> Result<i32, String> {
    let multiplier = value.trim().parse::<u128>().map_err(|e| format!("{} is not an integer multiplier: {}", label, e))?;
    multiplier_to_scale_exponent(multiplier)
}

fn parse_scale_list(cfg: &BTreeMap<String, String>, key: &str) -> Result<Vec<i32>, String> {
    let value = cfg.get(key).ok_or_else(|| format!("missing scale key {}", key))?;
    value.split(',').map(|x| parse_scale(x, key)).collect()
}

fn parse_quant_chain(path: &Path, checkpoint: &CheckpointManifest) -> Result<ScaleChain, String> {
    let cfg = parse_cfg(path)?;
    if cfg.get("schema").map(String::as_str) != Some("gat_l3_scale_chain") {
        return Err("quantization scale-chain schema mismatch".into());
    }
    let id = cfg.get("quant_cfg_id").ok_or_else(|| "missing quant_cfg_id".to_string())?;
    if id != &checkpoint.quant_cfg_id {
        return Err("quant config/checkpoint id mismatch".into());
    }
    let scalar = |key: &str| -> Result<i32, String> {
        let value = cfg.get(key).ok_or_else(|| format!("missing scale key {}", key))?;
        parse_scale(value, key)
    };
    let hidden_layers = checkpoint.hidden_layers.len();
    let mut scales = ScaleChain {
        s_input: scalar("s_input")?,
        s_delta: scalar("s_delta")?,
        s_exp: scalar("s_exp")?,
        s_h: parse_scale_list(&cfg, "s_h")?,
        s_w_hidden: Vec::new(),
        s_h_prime: Vec::new(),
        s_a_src_hidden: Vec::new(),
        s_a_dst_hidden: Vec::new(),
        s_agg_pre: parse_scale_list(&cfg, "s_agg_pre")?,
        s_agg: parse_scale_list(&cfg, "s_agg")?,
        s_cat: parse_scale_list(&cfg, "s_cat")?,
        s_w_output: parse_scale_list(&cfg, "s_w_output")?,
        s_y_prime: parse_scale_list(&cfg, "s_y_prime")?,
        s_a_src_output: parse_scale_list(&cfg, "s_a_src_output")?,
        s_a_dst_output: parse_scale_list(&cfg, "s_a_dst_output")?,
        s_b_output: parse_scale_list(&cfg, "s_b_output")?,
        s_out_pre: scalar("s_out_pre")?,
        s_out_head: scalar("s_out_head")?,
        s_out: scalar("s_out")?,
    };
    for layer in 0..hidden_layers {
        scales.s_w_hidden.push(parse_scale_list(&cfg, &format!("s_w_hidden_{}", layer))?);
        scales.s_h_prime.push(parse_scale_list(&cfg, &format!("s_h_prime_hidden_{}", layer),)?);
        scales.s_a_src_hidden.push(parse_scale_list(&cfg, &format!("s_a_src_hidden_{}", layer),)?);
        scales.s_a_dst_hidden.push(parse_scale_list(&cfg, &format!("s_a_dst_hidden_{}", layer),)?);
    }
    if cfg.get("round").map(String::as_str) != Some("nearest_ties_away")
        || cfg.get("rescale").map(String::as_str) != Some("none")
        || cfg.get("output_average").map(String::as_str) != Some("signed_round_div")
    {
        return Err("unsupported quantization/rounding policy".into());
    }
    scales.validate(checkpoint.layers, &checkpoint.hidden_layers.iter().map(|x| x.head_count).collect::<Vec<_>>(), checkpoint.output_heads,)?;
    if checkpoint.quantization_scale_chain != scales.canonical_serialization()? {
        return Err("checkpoint quantization_scale_chain is not canonical".into());
    }
    Ok(scales)
}

fn parse_features(path: &Path, meta: &DatasetMeta, scales: &ScaleChain,) -> Result<Vec<i128>, String> {
    if path.extension().and_then(|x| x.to_str()) == Some("npy") {
        let parsed = parse_npy(path, scales.s_input)?;
        let expected_shape = vec![meta.num_nodes, meta.num_features];
        if parsed.shape != expected_shape {
            return Err(format!("feature npy shape {:?} disagrees with metadata", parsed.shape));
        }
        if meta.feature_format.as_deref() != Some("npy")
            || meta.feature_dtype.as_deref() != Some(parsed.dtype.as_str())
            || meta.feature_shape.as_ref() != Some(&parsed.shape)
            || meta.feature_order.as_deref() != Some(parsed.order.as_str())
            || meta.feature_length != Some(parsed.values.len())
        {
            return Err("feature npy contract does not match authenticated metadata".into());
        }
        let digest = sha256::digest_file(path).map_err(|e| format!("digest {}: {}", path.display(), e))?;
        if meta.feature_sha256.as_deref() != Some(digest.as_str()) {
            return Err("feature npy SHA-256 does not match authenticated metadata".into());
        }
        return Ok(parsed.values);
    }
    let text = fs::read_to_string(path).map_err(|e| format!("read {}: {}", path.display(), e))?;
    let mut lines = text.lines();
    let header = lines.next().ok_or_else(|| "sparse feature file is empty".to_string())?;
    let mut header_it = header.split_whitespace();
    let rows = parse_usize(header_it.next().ok_or("missing sparse row count")?, "rows")?;
    let cols = parse_usize(header_it.next().ok_or("missing sparse column count")?, "cols",)?;
    if rows != meta.num_nodes || cols != meta.num_features {
        return Err("sparse feature shape disagrees with metadata".into());
    }
    let mut out = vec![0_i128; rows.checked_mul(cols).ok_or("feature shape overflow")?];
    let mut row_count = 0;
    for (row, line) in lines.enumerate() {
        if row >= rows {
            return Err("sparse feature file has too many rows".into());
        }
        let mut raw = Vec::new();
        for token in line.split_whitespace() {
            let (idx, value) = token.split_once(':').ok_or_else(|| format!("invalid sparse feature token {}", token))?;
            let idx = parse_usize(idx, "feature index")?;
            if idx >= cols {
                return Err(format!("feature index {} outside shape", idx));
            }
            let value = parse_finite(token_value(value)?, "feature")?;
            raw.push((idx, value));
        }
        let norm = if meta.input_preprocess_id == "row_normalize" {
            let sum: f64 = raw.iter().map(|(_, x)| *x).sum();
            if !sum.is_finite() || sum < 0.0 {
                return Err(format!("row {} has invalid normalization sum", row));
            }
            if sum == 0.0 {
                1.0
            } else {
                sum
            }
        } else {
            1.0
        };
        for (idx, value) in raw {
            out[row * cols + idx] = quantize(value / norm, scales.s_input)?;
        }
        row_count += 1;
    }
    if row_count != rows {
        return Err(format!("sparse feature rows {} != {}", row_count, rows));
    }
    Ok(out)
}

fn token_value(value: &str) -> Result<f64, String> {
    value.parse::<f64>().map_err(|e| format!("invalid float {}: {}", value, e))
}

fn parse_finite(value: f64, label: &str) -> Result<f64, String> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!("{} is not finite", label))
    }
}

fn quantize(value: f64, exponent: i32) -> Result<i128, String> {
    let value = parse_finite(value, "quantization value")?;
    let multiplier = (2_f64).powi(exponent);
    let scaled = value * multiplier;
    if !scaled.is_finite() || scaled > i128::MAX as f64 || scaled < i128::MIN as f64 {
        return Err("quantized value outside i128".into());
    }
    let rounded = if scaled >= 0.0 { (scaled + 0.5).floor() } else { (scaled - 0.5).ceil() };
    if rounded > i128::MAX as f64 || rounded < i128::MIN as f64 {
        return Err("rounded value outside i128".into());
    }
    Ok(rounded as i128)
}

fn parse_graph(path: &Path, meta: &DatasetMeta, dataset: &str, resources: &[AuthenticatedResource],) -> Result<CanonicalGraph, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("read {}: {}", path.display(), e))?;
    let node_ptr = if dataset == "ppi" {
        let ptr = resource_path(resources, "cache/ppi/node_ptr.txt")?;
        let ptr_text = fs::read_to_string(&ptr.path).map_err(|e| format!("read {}: {}", ptr.path.display(), e))?;
        ptr_text.split_whitespace().map(|token| parse_usize(token, "node_ptr")).collect::<Result<Vec<_>, _>>()?
    } else {
        vec![0, meta.num_nodes]
    };
    if node_ptr.len() != meta.graph_count + 1 || node_ptr.first() != Some(&0) || node_ptr.last() != Some(&meta.num_nodes) {
        return Err("node_ptr does not match graph metadata".into());
    }
    if node_ptr.windows(2).any(|x| x[0] >= x[1]) {
        return Err("node_ptr must be strictly increasing".into());
    }
    let mut graph_for_node = vec![0_usize; meta.num_nodes];
    for g in 0..meta.graph_count {
        for node in node_ptr[g]..node_ptr[g + 1] {
            graph_for_node[node] = g;
        }
    }
    let mut raw = Vec::<(usize, usize, usize)>::new();
    for (order, line) in text.lines().enumerate() {
        let mut it = line.split_whitespace();
        let first = parse_usize(it.next().ok_or("edge missing source")?, "edge source")?;
        let second = parse_usize(it.next().ok_or("edge missing destination")?, "edge destination",)?;
        let (declared_graph, src, dst) = if dataset == "ppi" {
            let dst = it.next().ok_or("ppi edge missing destination")?;
            if it.next().is_some() {
                return Err(format!("invalid edge at line {}", order + 1));
            }
            (Some(first), second, parse_usize(dst, "edge destination")?)
        } else {
            if it.next().is_some() {
                return Err(format!("invalid edge at line {}", order + 1));
            }
            (None, first, second)
        };
        if src >= meta.num_nodes || dst >= meta.num_nodes {
            return Err(format!("invalid edge at line {}", order + 1));
        }
        let g = graph_for_node[src];
        if declared_graph.is_some_and(|declared| declared != g) || graph_for_node[dst] != g {
            return Err(format!("edge crosses graph boundary at line {}", order + 1));
        }
        raw.push((g, src, dst));
    }
    let mut incoming = HashSet::new();
    for (_, _, dst) in &raw {
        incoming.insert(*dst);
    }
    for node in 0..meta.num_nodes {
        if !incoming.contains(&node) {
            raw.push((graph_for_node[node], node, node));
        }
    }
    raw.sort_by_key(|(g, _, dst)| (*g, *dst));
    let mut src = Vec::with_capacity(raw.len());
    let mut dst = Vec::with_capacity(raw.len());
    let mut edge_ptr = vec![0];
    let mut cursor = 0;
    for g in 0..meta.graph_count {
        while cursor < raw.len() && raw[cursor].0 == g {
            src.push(raw[cursor].1);
            dst.push(raw[cursor].2);
            cursor += 1;
        }
        edge_ptr.push(src.len());
    }
    if cursor != raw.len() {
        return Err("edge graph grouping failed".into());
    }
    let mut graph = CanonicalGraph {
        n_total: meta.num_nodes,
        n: meta.num_nodes,
        e: src.len(),
        g_batch: meta.graph_count,
        node_ids: (0..meta.num_nodes).collect(),
        src,
        dst,
        node_ptr,
        edge_ptr,
        topology_id: format!("topology:{}:{}", dataset, meta.graph_count),
        topology_digest: String::new(),
    };
    graph.refresh_digest()?;
    graph.validate()?;
    Ok(graph)
}

fn parse_tensor_archive(path: &Path, manifest: &CheckpointManifest,) -> Result<BTreeMap<String, ParsedTensor>, String> {
    if manifest.tensor_count != manifest.tensor_index.len() {
        return Err("checkpoint tensor manifest count mismatch".into());
    }
    let mut archive_to_name = BTreeMap::new();
    for (name, meta) in &manifest.tensor_index {
        let expected_archive_key = canonical_tensor_archive_key(name)?;
        if meta.archive_key != expected_archive_key {
            return Err(format!("tensor archive key mismatch for {}: expected {}, got {}", name, expected_archive_key, meta.archive_key));
        }
        if meta.dtype != "float32" {
            return Err(format!("unsupported tensor dtype for {}: {}", name, meta.dtype));
        }
        if meta.shape.is_empty() || meta.shape.iter().any(|dimension| *dimension == 0) {
            return Err(format!("invalid tensor shape for {}", name));
        }
        if archive_to_name.insert(meta.archive_key.clone(), name.clone()).is_some() {
            return Err(format!("tensor archive key collision for {}", meta.archive_key));
        }
    }
    let text = fs::read_to_string(path).map_err(|e| format!("read {}: {}", path.display(), e))?;
    let mut result = BTreeMap::new();
    let mut current_name: Option<String> = None;
    let mut current_shape = Vec::new();
    let mut values = Vec::new();
    let flush = |result: &mut BTreeMap<String, ParsedTensor>, name: &mut Option<String>, shape: &mut Vec<usize>, values: &mut Vec<f64>| -> Result<(), String> {
        if let Some(name) = name.take() {
            let expected = shape.iter().try_fold(1_usize, |acc, x| acc.checked_mul(*x)).ok_or("tensor shape overflow")?;
            if expected != values.len() {
                return Err(format!("tensor {} has {} values, expected {}", name, values.len(), expected));
            }
            let old = result.insert(name.clone(), ParsedTensor { shape: std::mem::take(shape), values: std::mem::take(values),},);
            if old.is_some() {
                return Err(format!("duplicate tensor {}", name));
            }
        }
        Ok(())
    };
    for line in text.lines() {
        if let Some(header) = line.strip_prefix("TENSOR ") {
            flush(&mut result, &mut current_name, &mut current_shape, &mut values,)?;
            let mut it = header.split_whitespace();
            let name = it.next().ok_or("tensor header missing name")?.to_string();
            let rank = parse_usize(it.next().ok_or("tensor header missing rank")?, "tensor rank",)?;
            let mut shape = Vec::with_capacity(rank);
            for _ in 0..rank {
                shape.push(parse_usize(it.next().ok_or("tensor header missing shape")?, "tensor dimension",)?);
            }
            let declared = parse_usize(it.next().ok_or("tensor header missing element count")?, "tensor element count",)?;
            let expected = shape.iter().try_fold(1_usize, |acc, x| acc.checked_mul(*x)).ok_or("tensor shape overflow")?;
            if expected != declared {
                return Err(format!("tensor {} header count mismatch", name));
            }
            current_name = Some(name);
            current_shape = shape;
        } else if !line.trim().is_empty() {
            if current_name.is_none() {
                return Err("tensor values precede first header".into());
            }
            for token in line.split_whitespace() {
                values.push(parse_finite(token.parse::<f64>().map_err(|e| format!("invalid tensor float: {}", e))?, "tensor",)?);
            }
        }
    }
    flush(&mut result, &mut current_name, &mut current_shape, &mut values,)?;
    if result.len() != manifest.tensor_count || result.len() != archive_to_name.len() {
        return Err("tensor archive count mismatch".into());
    }
    for (name, meta) in &manifest.tensor_index {
        let tensor = result.get(name).ok_or_else(|| format!("missing tensor {}", name))?;
        let header_archive_key = canonical_tensor_archive_key(name)?;
        if header_archive_key != meta.archive_key || tensor.shape != meta.shape {
            return Err(format!("tensor metadata mismatch for {}", name));
        }
    }
    Ok(result)
}

fn parse_npy(path: &Path, quantization_exponent: i32) -> Result<ParsedNpy, String> {
    let file = fs::File::open(path).map_err(|e| format!("read {}: {}", path.display(), e))?;
    let mut reader = BufReader::new(file);
    let mut prefix = [0_u8; 8];
    reader.read_exact(&mut prefix).map_err(|e| format!("read NumPy header {}: {}", path.display(), e))?;
    if &prefix[..6] != b"\x93NUMPY" {
        return Err("invalid NumPy magic".into());
    }
    let major = prefix[6];
    let (header_len_u64, data_start_u64): (u64, u64) = if major == 1 {
        let mut length = [0_u8; 2];
        reader.read_exact(&mut length).map_err(|e| format!("read NumPy header length: {}", e))?;
        (u64::from(u16::from_le_bytes(length)), 10)
    } else if major == 2 {
        let mut length = [0_u8; 4];
        reader.read_exact(&mut length).map_err(|e| format!("read NumPy header length: {}", e))?;
        (u64::from(u32::from_le_bytes(length)), 12)
    } else {
        return Err(format!("unsupported NumPy version {}", major));
    };
    if header_len_u64 > 16 * 1024 * 1024 {
        return Err("NumPy header is unreasonably large".into());
    }
    let header_len = usize::try_from(header_len_u64).map_err(|_| "NumPy header does not fit usize")?;
    let mut header_bytes = vec![0_u8; header_len];
    reader.read_exact(&mut header_bytes).map_err(|e| format!("read NumPy header body: {}", e))?;
    let header = std::str::from_utf8(&header_bytes).map_err(|e| format!("NumPy header utf8: {}", e))?;
    if !header.contains("'fortran_order': False") && !header.contains("\"fortran_order\": False") {
        return Err("Fortran-order NumPy arrays are unsupported".into());
    }
    let descr = header.split("'descr':").nth(1).or_else(|| header.split("\"descr\":").nth(1)).ok_or("NumPy descr missing")?;
    let (item_size, dtype) = if descr.contains("f4") {
        (4, "float32")
    } else if descr.contains("f8") {
        (8, "float64")
    } else {
        return Err("NumPy dtype must be float32/float64".into());
    };
    let shape_text = header.split("'shape':").nth(1).or_else(|| header.split("\"shape\":").nth(1)).ok_or("NumPy shape missing")?;
    let open = shape_text.find('(').ok_or("NumPy shape tuple missing")?;
    let close = shape_text[open + 1..].find(')').ok_or("NumPy shape tuple unterminated")? + open + 1;
    let shape: Vec<usize> =
        shape_text[open + 1..close].split(',').filter(|x| !x.trim().is_empty()).map(|x| parse_usize(x.trim(), "NumPy dimension")).collect::<Result<_, _>>()?;
    let count = shape.iter().try_fold(1_usize, |acc, x| acc.checked_mul(*x)).ok_or("NumPy shape overflow")?;
    let expected_bytes = count.checked_mul(item_size).ok_or("NumPy data size overflow")?;
    let expected_file_size = data_start_u64
        .checked_add(header_len_u64)
        .and_then(|x| { u64::try_from(expected_bytes).ok().and_then(|size| x.checked_add(size)) })
        .ok_or("NumPy file size overflow")?;
    let actual_file_size = path.metadata().map_err(|e| format!("stat {}: {}", path.display(), e))?.len();
    if actual_file_size != expected_file_size {
        return Err("NumPy data length mismatch".into());
    }
    let mut values = Vec::with_capacity(count);
    let mut remaining = expected_bytes;
    let chunk_size = (1024 * 1024) - ((1024 * 1024) % item_size);
    let mut chunk = vec![0_u8; chunk_size];
    while remaining > 0 {
        let take = remaining.min(chunk.len());
        reader.read_exact(&mut chunk[..take]).map_err(|e| format!("read NumPy payload {}: {}", path.display(), e))?;
        for value_bytes in chunk[..take].chunks_exact(item_size) {
            let value = if item_size == 4 {
                let bytes: [u8; 4] = value_bytes.try_into().unwrap();
                f32::from_le_bytes(bytes) as f64
            } else {
                let bytes: [u8; 8] = value_bytes.try_into().unwrap();
                f64::from_le_bytes(bytes)
            };
            values.push(quantize(value, quantization_exponent)?);
        }
        remaining -= take;
    }
    let mut trailing = [0_u8; 1];
    if reader.read(&mut trailing).map_err(|e| e.to_string())? != 0 {
        return Err("NumPy data has trailing bytes".into());
    }
    Ok(ParsedNpy { shape, dtype: dtype.to_string(), order: "C".into(), values,}) }

fn build_model(manifest: &CheckpointManifest, scales: &ScaleChain, tensors: &BTreeMap<String, ParsedTensor>,) -> Result<CanonicalModel, String> {
    let mut hidden_layers = Vec::new();
    for (layer_idx, layer_meta) in manifest.hidden_layers.iter().enumerate() {
        let mut heads = Vec::new();
        for head_idx in 0..layer_meta.head_count {
            let prefix = format!("hidden/layer{}/head{}", layer_idx, head_idx);
            let projection = tensor_quantized(tensors, &format!("{}/seq/kernel", prefix), &scales.s_w_hidden[layer_idx][head_idx],)?;
            let a_src = tensor_quantized(tensors, &format!("{}/attn_src/kernel", prefix), &scales.s_a_src_hidden[layer_idx][head_idx],)?;
            let a_dst = tensor_quantized(tensors, &format!("{}/attn_dst/kernel", prefix), &scales.s_a_dst_hidden[layer_idx][head_idx],)?;
            heads.push(HeadWeights { projection, a_src, a_dst, bias: Vec::new(),});
        }
        hidden_layers.push(HiddenLayerSpec { d_in: layer_meta.input_dim, heads, d_head: layer_meta.head_dim,});
    }
    let mut output_heads = Vec::new();
    for head_idx in 0..manifest.output_heads {
        let prefix = format!("output/head{}", head_idx);
        output_heads.push(HeadWeights {
            projection: tensor_quantized(tensors, &format!("{}/seq/kernel", prefix), &scales.s_w_output[head_idx],)?,
            a_src: tensor_quantized(tensors, &format!("{}/attn_src/kernel", prefix), &scales.s_a_src_output[head_idx],)?,
            a_dst: tensor_quantized(tensors, &format!("{}/attn_dst/kernel", prefix), &scales.s_a_dst_output[head_idx],)?,
            bias: tensor_quantized(tensors, &format!("{}/output_bias", prefix), &scales.s_b_output[head_idx],)?,
        });
    }
    let model = CanonicalModel { hidden_layers, output_heads, classes: manifest.classes, model_arch_id: manifest.model_arch_id.clone(),};
    model.validate(manifest.d_in_profile[0])?;
    Ok(model)
}

fn tensor_quantized(tensors: &BTreeMap<String, ParsedTensor>, name: &str, scale: &i32,) -> Result<Vec<i128>, String> {
    let tensor = tensors.get(name).ok_or_else(|| format!("missing tensor {}", name))?;
    tensor.values.iter().map(|x| quantize(*x, *scale)).collect()
}

fn load_exact_relation_resources(
    resources: &[AuthenticatedResource],
    dataset: &str,
    secure: &BTreeMap<String, String>,
) -> Result<(StaticTables, IntegerBoundLedger), String> {
    let table_path = format!("resources/canonical/tables/{}/gat_static_tables.bin", dataset);
    let ledger_path = format!("resources/canonical/ledgers/{}/gat_integer_ledger.json", dataset);
    let table = resources.iter().find(|resource| resource.record.path == table_path);
    let ledger = resources.iter().find(|resource| resource.record.path == ledger_path);
    let mut missing = Vec::new();
    if table.is_none() {
        missing.push(table_path.as_str());
    }
    if ledger.is_none() {
        missing.push(ledger_path.as_str());
    }
    if !missing.is_empty() {
        return Err(format!("unsupported_exact_relation: authenticated exact table/ledger resources are missing; provide {}", missing.join(", ")));
    }
    let tables = parse_binary_tables(&table.unwrap().path)?;
    let ledger: IntegerBoundLedger = parse_json(&ledger.unwrap().path)?;
    tables.validate()?;
    ledger.validate()?;
    let expected_table_id = secure.get("static_table_id").ok_or_else(|| "missing static_table_id in secure config".to_string())?;
    if &tables.static_tables_id != expected_table_id {
        return Err("unsupported_exact_relation: authenticated table registry id disagrees with secure config".into());
    }
    let expected_table_digest = secure.get("static_table_digest").ok_or_else(|| "missing static_table_digest in secure config".to_string())?;
    if &tables.static_tables_digest != expected_table_digest {
        return Err("unsupported_exact_relation: authenticated table digest disagrees with secure config".into(),);
    }
    let expected_ledger_id = secure.get("degree_bound_id").ok_or_else(|| "missing degree_bound_id in secure config".to_string())?;
    if expected_ledger_id == "auto" || &ledger.ledger_id != expected_ledger_id {
        return Err("unsupported_exact_relation: secure config does not bind an authenticated exact integer ledger".into());
    }
    let expected_ledger_digest = secure.get("degree_bound_digest").ok_or_else(|| "missing degree_bound_digest in secure config".to_string())?;
    if &ledger.ledger_digest != expected_ledger_digest {
        return Err("unsupported_exact_relation: authenticated ledger digest disagrees with secure config".into(),);
    }
    Ok((tables, ledger))
}

const TABLE_MAGIC: &[u8] = b"EZKL-GAT-TABLES\x00";
const TABLE_FORMAT_VERSION: u32 = 1;
const TABLE_METADATA_LIMIT: u64 = 16 * 1024 * 1024;

fn read_u32(reader: &mut BufReader<fs::File>, label: &str) -> Result<u32, String> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes).map_err(|e| format!("read {}: {}", label, e))?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut BufReader<fs::File>, label: &str) -> Result<u64, String> {
    let mut bytes = [0_u8; 8];
    reader.read_exact(&mut bytes).map_err(|e| format!("read {}: {}", label, e))?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_bounded_bytes(reader: &mut BufReader<fs::File>, length: u64, label: &str,) -> Result<Vec<u8>, String> {
    if length > TABLE_METADATA_LIMIT {
        return Err(format!("{} exceeds metadata limit", label));
    }
    let length = usize::try_from(length).map_err(|_| format!("{} does not fit usize", label))?;
    let mut bytes = vec![0_u8; length];
    reader.read_exact(&mut bytes).map_err(|e| format!("read {}: {}", label, e))?;
    Ok(bytes)
}

fn archive_canonical_length(metadata: &StaticTableArchiveMeta) -> Result<u64, String> {
    let string_bytes = [metadata.table_id.len(), metadata.function_id.len(), metadata.domain_id.len(), metadata.generator_id.len(),]
        .into_iter()
        .try_fold(0_u64, |total, length| { total.checked_add(8).and_then(|value| value.checked_add(u64::try_from(length).ok()?)) })
        .ok_or_else(|| format!("{} canonical length overflow", metadata.table_id))?;
    let rows = metadata.row_count.checked_mul(32).ok_or_else(|| format!("{} canonical row length overflow", metadata.table_id))?;
    u64::try_from(b"ezkl-gat-exact-table-v1".len())
        .ok()
        .and_then(|prefix| prefix.checked_add(string_bytes))
        .and_then(|value| value.checked_add(4 + 4 + 16 + 16))
        .and_then(|value| value.checked_add(rows))
        .ok_or_else(|| format!("{} canonical length overflow", metadata.table_id))
}

fn parse_binary_tables(path: &Path) -> Result<StaticTables, String> {
    let file = fs::File::open(path).map_err(|e| format!("read {}: {}", path.display(), e))?;
    let mut reader = BufReader::new(file);
    let mut magic = [0_u8; 16];
    reader.read_exact(&mut magic).map_err(|e| format!("read table archive magic: {}", e))?;
    if magic.as_slice() != TABLE_MAGIC {
        return Err("unsupported static-table archive magic".into());
    }
    let version = read_u32(&mut reader, "table archive version")?;
    if version != TABLE_FORMAT_VERSION {
        return Err(format!("unsupported static-table archive version {}", version));
    }
    let table_count = read_u32(&mut reader, "table archive count")?;
    if table_count != 4 {
        return Err(format!("static-table archive must contain four tables, got {}", table_count));
    }
    let registry_length = read_u64(&mut reader, "registry metadata length")?;
    let registry: StaticTableRegistryMeta = serde_json::from_slice(&read_bounded_bytes(&mut reader, registry_length, "registry metadata",)?)
        .map_err(|e| format!("parse registry metadata: {}", e))?;
    if registry.format != "ezkl-gat-static-tables-bin"
        || registry.version != TABLE_FORMAT_VERSION
        || registry.table_order != vec!["lrelu".to_string(), "exp".to_string(), "elu".to_string(), "max_relu".to_string(),]
    {
        return Err("static-table registry metadata identity mismatch".into());
    }
    let mut tables = Vec::new();
    for expected_function in ["lrelu", "exp_neg", "elu", "max_relu"] {
        let metadata_length = read_u64(&mut reader, "table metadata length")?;
        let metadata: StaticTableArchiveMeta =
            serde_json::from_slice(&read_bounded_bytes(&mut reader, metadata_length, "table metadata",)?).map_err(|e| format!("parse table metadata: {}", e))?;
        let row_count = read_u64(&mut reader, "table row count")?;
        if row_count != metadata.row_count {
            return Err(format!("{} row count metadata mismatch", metadata.table_id));
        }
        let expected_count = metadata
            .input_max
            .checked_sub(metadata.input_min)
            .and_then(|x| x.checked_add(1))
            .ok_or_else(|| format!("{} input domain overflow", metadata.table_id))?;
        if u64::try_from(expected_count).ok() != Some(row_count) {
            return Err(format!("{} row count does not cover its domain", metadata.table_id));
        }
        let capacity = usize::try_from(row_count).map_err(|_| format!("{} row count does not fit usize", metadata.table_id))?;
        let mut rows = Vec::with_capacity(capacity);
        for _ in 0..row_count {
            let mut input_bytes = [0_u8; 16];
            let mut output_bytes = [0_u8; 16];
            reader.read_exact(&mut input_bytes).map_err(|e| format!("read {} input row: {}", metadata.table_id, e))?;
            reader.read_exact(&mut output_bytes).map_err(|e| format!("read {} output row: {}", metadata.table_id, e))?;
            rows.push((i128::from_le_bytes(input_bytes), i128::from_le_bytes(output_bytes),));
        }
        if metadata.function_id != expected_function {
            return Err(format!("{} function identity mismatch: {} != {}", metadata.table_id, metadata.function_id, expected_function));
        }
        let canonical_len = metadata.canonical_len;
        if archive_canonical_length(&metadata)? != canonical_len {
            return Err(format!("{} canonical length mismatch", metadata.table_id));
        }
        let table = ExactTable {
            table_id: metadata.table_id,
            input_scale: metadata.input_scale,
            output_scale: metadata.output_scale,
            function_id: metadata.function_id,
            domain_id: metadata.domain_id,
            generator_id: metadata.generator_id,
            rows,
            input_min: metadata.input_min,
            input_max: metadata.input_max,
            digest: metadata.digest,
        };
        tables.push(table);
    }
    let [lrelu, exp, elu, max_relu] = tables.try_into().map_err(|_| "table count mismatch")?;
    let registry = StaticTables {
        lrelu,
        exp,
        elu,
        max_relu,
        static_tables_id: registry.static_tables_id,
        backend_aux_tables_id: registry.backend_aux_tables_id,
        static_tables_digest: registry.static_tables_digest,
    };
    registry.validate()?;
    let mut trailing = [0_u8; 1];
    if reader.read(&mut trailing).map_err(|e| e.to_string())? != 0 {
        return Err("static-table archive has trailing bytes".into());
    }
    Ok(registry)
}

/// Write Rust-generated static-table and integer-ledger metadata beside a run.
pub fn write_generated_metadata(spec: &CanonicalGat, output_dir: &Path) -> Result<(), String> {
    fs::create_dir_all(output_dir).map_err(|e| format!("create {}: {}", output_dir.display(), e))?;
    let tables = serde_json::json!({
        "format": "ezkl-gat-static-tables-metadata-v1",
        "source_format": "gat_static_tables.bin",
        "static_tables_id": spec.tables.static_tables_id,
        "backend_aux_tables_id": spec.tables.backend_aux_tables_id,
        "static_tables_digest": spec.tables.static_tables_digest,
        "tables": [
            {"table_id": spec.tables.lrelu.table_id, "digest": spec.tables.lrelu.digest, "input_min": spec.tables.lrelu.input_min, "input_max": spec.tables.lrelu.input_max},
            {"table_id": spec.tables.exp.table_id, "digest": spec.tables.exp.digest, "input_min": spec.tables.exp.input_min, "input_max": spec.tables.exp.input_max},
            {"table_id": spec.tables.elu.table_id, "digest": spec.tables.elu.digest, "input_min": spec.tables.elu.input_min, "input_max": spec.tables.elu.input_max},
            {"table_id": spec.tables.max_relu.table_id, "digest": spec.tables.max_relu.digest, "input_min": spec.tables.max_relu.input_min, "input_max": spec.tables.max_relu.input_max}
        ]
    });
    let tables = serde_json::to_vec_pretty(&tables).map_err(|e| format!("serialize tables: {}", e))?;
    let ledger = serde_json::to_vec_pretty(&spec.ledger).map_err(|e| format!("serialize ledger: {}", e))?;
    write_atomic_metadata(&output_dir.join("gat_static_tables.json"), &tables)?;
    write_atomic_metadata(&output_dir.join("gat_integer_ledger.json"), &ledger)?;
    Ok(())
}

fn write_atomic_metadata(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path.file_name().and_then(|value| value.to_str()).ok_or_else(|| format!("metadata path has no valid filename: {}", path.display()))?;
    let temporary = parent.join(format!(".{}.{}.tmp", name, std::process::id()));
    let result = (|| {
        let mut file =
            fs::OpenOptions::new().write(true).create_new(true).open(&temporary).map_err(|e| format!("create temporary metadata {}: {}", path.display(), e))?;
        file.write_all(bytes).map_err(|e| format!("write temporary metadata {}: {}", path.display(), e))?;
        file.sync_all().map_err(|e| format!("sync temporary metadata {}: {}", path.display(), e))?;
        drop(file);
        fs::rename(&temporary, path).map_err(|e| format!("install metadata {}: {}", path.display(), e))
    })();
    if temporary.exists() {
        let _ = fs::remove_file(&temporary);
    }
    result?;
    Ok(())
}
