//! Canonical, baseline-local input schema for the GAT comparison adapter.
//!
//! The types in this module are deliberately independent from the main
//! experiment.  They describe the public grouped-edge graph, the fixed point
//! contract, the authenticated lookup tables and the private model tensor
//! layout used by the ezkl graph adapter.

use crate::pfsys::canonical_string_to_fr;
use halo2curves::bn256::Fr as Bn254Fr;
use halo2curves::ff::PrimeField;
use num::{BigUint, One};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::convert::TryFrom;

/// Schema version for canonical ezkl GAT manifests.
pub const GAT_SCHEMA_VERSION: u32 = 1;
/// Model-binding domain for the native public Hashed parameter relation.
pub const MODEL_HASH_DOMAIN: &str = "ezkl-gat-model-v1";

/// Return a collision-resistant digest for a canonical byte encoding.
///
/// The official ezkl build enables the `ezkl` feature (which owns the
/// baseline's existing SHA-256 dependency).  Other feature combinations
/// deliberately fail closed instead of substituting a weaker hash.
pub fn sha256_digest(bytes: &[u8]) -> Result<String, String> {
    #[cfg(feature = "ezkl")]
    {
        Ok(sha256::digest(bytes))
    }
    #[cfg(not(feature = "ezkl"))]
    {
        let _ = bytes;
        Err("canonical GAT digests require the official ezkl SHA-256 feature".into())
    }
}

/// Stream a file through SHA-256 once while reporting byte-level progress.
/// The returned size is the number of bytes actually hashed, not a second
/// metadata or digest pass.
pub fn sha256_file_with_progress(path: &std::path::Path, stage: &str) -> Result<(u64, String), String> {
    #[cfg(feature = "ezkl")]
    {
        use sha2::{Digest, Sha256};
        use std::fs::File;
        use std::io::Read;
        use std::time::{Duration, Instant};

        let mut file = File::open(path).map_err(|error| format!("open {}: {}", path.display(), error))?;
        let total = file
            .metadata()
            .map_err(|error| format!("stat {}: {}", path.display(), error))?
            .len();
        eprintln!("[sha256] {}: started {} (0/{} bytes)", stage, path.display(), total);

        let started = Instant::now();
        let mut last_report = started;
        let mut last_report_bytes = 0_u64;
        let mut processed = 0_u64;
        let mut hasher = Sha256::new();
        let mut buffer = [0_u8; 1024 * 1024];
        loop {
            let read = file
                .read(&mut buffer)
                .map_err(|error| format!("read {}: {}", path.display(), error))?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            processed = processed
                .checked_add(read as u64)
                .ok_or_else(|| format!("size overflow while reading {}", path.display()))?;

            let now = Instant::now();
            if processed.saturating_sub(last_report_bytes) >= 1024 * 1024 * 1024
                || now.duration_since(last_report) >= Duration::from_secs(20)
            {
                eprintln!(
                    "[sha256] {}: progress {} ({}/{}) bytes, {:.1}s",
                    stage,
                    path.display(),
                    processed,
                    total,
                    started.elapsed().as_secs_f64()
                );
                last_report = now;
                last_report_bytes = processed;
            }
        }

        let digest = hex::encode(hasher.finalize());
        eprintln!(
            "[sha256] {}: complete {} ({}/{} bytes, {:.1}s)",
            stage,
            path.display(),
            processed,
            total,
            started.elapsed().as_secs_f64()
        );
        Ok((processed, digest))
    }
    #[cfg(not(feature = "ezkl"))]
    {
        let _ = (path, stage);
        Err("streaming file SHA-256 requires the official ezkl feature".into())
    }
}

/// Encode one canonical logical tensor name using the checkpoint bundle's
/// NPZ key rule.  Logical names remain the identity in the manifest and text
/// dump; this encoding is only the deterministic archive key.  Callers must
/// build the reverse map over the complete manifest and reject collisions so
/// that an alias can never silently select another tensor.
pub fn canonical_tensor_archive_key(name: &str) -> Result<String, String> {
    if name.is_empty() {
        return Err("canonical tensor name must not be empty".into());
    }
    if name.contains('\0') || name.contains('\\') {
        return Err(format!("canonical tensor name contains a forbidden path character: {}", name));
    }
    let encoded = name.replace('/', "__").replace(':', "__");
    if encoded.is_empty() || encoded.contains('/') || encoded.contains(':') {
        return Err(format!("canonical tensor archive key is not encodable: {}", name));
    }
    Ok(encoded)
}

fn append_i128(out: &mut Vec<u8>, value: i128) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn append_usize(out: &mut Vec<u8>, value: usize) {
    out.extend_from_slice(&(value as u64).to_le_bytes());
}

fn append_string(out: &mut Vec<u8>, value: &str) {
    append_usize(out, value.len());
    out.extend_from_slice(value.as_bytes());
}

fn validate_digest(label: &str, digest: &str) -> Result<(), String> {
    if digest.len() != 64 || hex::decode(digest).is_err() {
        return Err(format!("{} must be a lowercase 32-byte hex digest", label));
    }
    if digest.bytes().any(|b| b.is_ascii_uppercase()) {
        return Err(format!("{} must use lowercase hexadecimal", label));
    }
    Ok(())
}

/// Parse one canonical unsigned BN254 modulus representation.
///
/// Native halo2curves exposes `PrimeField::MODULUS` as a lower-case `0x`
/// hexadecimal string, while the baseline resource generator serializes the
/// ledger modulus as canonical decimal.  Accept exactly those two forms and
/// reject signs, whitespace, upper-case hex, and non-canonical leading zeroes
/// before comparing the resulting integer.
pub fn parse_bn254_modulus(value: &str, label: &str) -> Result<BigUint, String> {
    if value.is_empty() || value.trim() != value {
        return Err(format!("{} must not be empty or contain whitespace", label));
    }
    if value.starts_with('+') || value.starts_with('-') {
        return Err(format!("{} must be unsigned", label));
    }
    let (digits, radix) = if let Some(hex) = value.strip_prefix("0x") {
        if hex.is_empty() || hex.bytes().any(|byte| !byte.is_ascii_hexdigit() || byte.is_ascii_uppercase()) {
            return Err(format!("{} must use lower-case 0x hexadecimal", label));
        }
        (hex, 16)
    } else {
        if value.bytes().any(|byte| !byte.is_ascii_digit()) {
            return Err(format!("{} must use canonical decimal or 0x hexadecimal", label));
        }
        (value, 10)
    };
    if digits.len() > 1 && digits.starts_with('0') {
        return Err(format!("{} has non-canonical leading zeroes", label));
    }
    BigUint::parse_bytes(digits.as_bytes(), radix).ok_or_else(|| format!("{} is not a valid unsigned integer", label))
}

/// Parse and authenticate a resource ledger's BN254 Fr modulus.
pub fn parse_authenticated_bn254_modulus(value: &str) -> Result<BigUint, String> {
    let modulus = parse_bn254_modulus(value, "field_modulus")?;
    let native = parse_bn254_modulus(Bn254Fr::MODULUS, "native BN254 Fr modulus")?;
    if modulus != native {
        return Err("field_modulus does not match native BN254 Fr".into());
    }
    Ok(modulus)
}

/// An externally indexed object whose bytes are authenticated by a digest.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DigestIdentity {
    /// Stable object identifier.
    pub id: String,
    /// SHA-256 of the canonical object encoding.
    pub digest: String,
}

impl DigestIdentity {
    /// Validate an externally supplied identity.
    pub fn validate(&self, label: &str) -> Result<(), String> {
        if self.id.is_empty() {
            return Err(format!("{} id must not be empty", label));
        }
        validate_digest(&format!("{} digest", label), &self.digest)
    }
}

/// The native compiled-circuit/settings/VK identity used by the public
/// statement.  These IDs are deliberately opaque: the official setup path is
/// responsible for producing their bytes, while the adapter refuses missing
/// or malformed bindings.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CircuitRef {
    /// Compiled circuit identity.
    pub compiled_circuit_id: String,
    /// SHA-256 identity of the compiled circuit manifest.
    pub compiled_circuit_digest: String,
    /// Settings identity.
    pub settings_id: String,
    /// SHA-256 identity of the settings manifest.
    pub settings_digest: String,
    /// Verification-key identity.
    pub vk_id: String,
    /// SHA-256 identity of the verification-key bytes.
    pub vk_digest: String,
    /// Model-graph identity included in the compiled circuit manifest.
    pub model_graph_id: String,
    /// SHA-256 identity of the serialized graph/model bytes.
    pub model_graph_digest: String,
    /// Native ezkl/Halo2 security profile identity.
    pub ezkl_security_profile_id: String,
}

impl CircuitRef {
    /// Validate the native circuit identity.
    pub fn validate(&self) -> Result<(), String> {
        for (label, value) in [
            ("compiled_circuit_id", &self.compiled_circuit_id),
            ("settings_id", &self.settings_id),
            ("vk_id", &self.vk_id),
            ("model_graph_id", &self.model_graph_id),
            ("ezkl_security_profile_id", &self.ezkl_security_profile_id),
        ] {
            if value.is_empty() {
                return Err(format!("CircuitRef.{} must not be empty", label));
            }
            let lowered = value.to_ascii_lowercase();
            if ["unbound", "pending", "placeholder", "dummy", "fake"].iter().any(|marker| lowered.contains(marker)) {
                return Err(format!("CircuitRef.{} contains an unbound/placeholder identity", label));
            }
        }
        for (label, digest) in [
            ("compiled_circuit_digest", &self.compiled_circuit_digest),
            ("settings_digest", &self.settings_digest),
            ("vk_digest", &self.vk_digest),
            ("model_graph_digest", &self.model_graph_digest),
        ] {
            validate_digest(&format!("CircuitRef.{}", label), digest)?;
        }
        Ok(())
    }
}

/// Public model binding used by ezkl's native Hashed runtime path.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelBinding {
    /// Domain-separated hash of the canonical model parameter closure H(W).
    pub hash: String,
    /// Hash domain/version used by the native module.
    pub hash_domain: String,
    /// SHA-256 identity of the canonical parameter encoding.  This is an
    /// audit binding and is distinct from the native Poseidon H(W) value.
    pub canonical_model_digest: String,
}

impl ModelBinding {
    /// Validate the native model binding.
    pub fn validate(&self) -> Result<(), String> {
        if self.hash_domain.is_empty() {
            return Err("model binding hash_domain must not be empty".into());
        }
        if self.hash.is_empty() {
            return Err("model binding hash must not be empty".into());
        }
        canonical_string_to_fr(&self.hash).map_err(|error| format!("model binding hash: {}", error))?;
        validate_digest("model binding canonical_model_digest", &self.canonical_model_digest,) } }

/// Complete verifier-facing public instance binding.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicInstanceBinding {
    /// Public topology identity.
    pub topology: DigestIdentity,
    /// Main exact-table registry identity.
    pub static_tables: DigestIdentity,
    /// Quantization identity.
    pub quant: DigestIdentity,
    /// Integer-bound ledger identity.
    pub ledger: DigestIdentity,
    /// Native circuit/settings/VK identity.
    pub circuit: CircuitRef,
    /// Public model hash H(W).
    pub model: ModelBinding,
    /// Canonical public feature-table identity.
    pub feature_digest: String,
    /// Canonical public output identity, filled by the verifier/runner.
    pub output_digest: String,
}

impl PublicInstanceBinding {
    /// Validate all public binding fields without reading private witness data.
    pub fn validate(&self) -> Result<(), String> {
        self.topology.validate("topology")?;
        self.static_tables.validate("static_tables")?;
        self.quant.validate("quant")?;
        self.ledger.validate("ledger")?;
        self.circuit.validate()?;
        self.model.validate()?;
        validate_digest("feature_digest", &self.feature_digest)?;
        validate_digest("output_digest", &self.output_digest)
    }
}

/// An exact integer lookup table.  Rows are canonicalized in the order given.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExactTable {
    /// Stable table identity, including generator and scale semantics.
    pub table_id: String,
    /// Integer scale of the lookup input.
    #[serde(default)]
    pub input_scale: i32,
    /// Integer scale of the lookup output.
    #[serde(default)]
    pub output_scale: i32,
    /// Canonical function identity (`lrelu`, `exp_neg`, `elu`, or
    /// `max_relu` for the four registered tables).
    #[serde(default)]
    pub function_id: String,
    /// Authenticated domain/clipping identity.
    #[serde(default)]
    pub domain_id: String,
    /// Authenticated integer/rational row-generator identity.
    #[serde(default)]
    pub generator_id: String,
    /// Canonical `(input, output)` rows.
    pub rows: Vec<(i128, i128)>,
    /// Inclusive input domain authenticated by the manifest.
    pub input_min: i128,
    /// Inclusive input domain authenticated by the manifest.
    pub input_max: i128,
    /// SHA-256 of the table id, domain and canonical contiguous rows.
    pub digest: String,
}

impl ExactTable {
    /// Canonical bytes covered by [`Self::digest`].
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = b"ezkl-gat-exact-table-v1".to_vec();
        append_string(&mut out, &self.table_id);
        out.extend_from_slice(&self.input_scale.to_le_bytes());
        out.extend_from_slice(&self.output_scale.to_le_bytes());
        append_string(&mut out, &self.function_id);
        append_string(&mut out, &self.domain_id);
        append_string(&mut out, &self.generator_id);
        append_i128(&mut out, self.input_min);
        append_i128(&mut out, self.input_max);
        for (input, output) in &self.rows {
            append_i128(&mut out, *input);
            append_i128(&mut out, *output);
        }
        out
    }

    /// Fill the authenticated digest after constructing a canonical table.
    pub fn refresh_digest(&mut self) -> Result<(), String> {
        self.digest = sha256_digest(&self.canonical_bytes())?;
        Ok(())
    }

    /// Validate row ordering, contiguous domain membership and digest.
    pub fn validate(&self, non_negative_input: bool, positive_output: bool) -> Result<(), String> {
        if self.rows.is_empty() {
            return Err(format!("{} must not be empty", self.table_id));
        }
        if self.table_id.is_empty() {
            return Err("exact table id must not be empty".into());
        }
        if self.input_scale < 0 || self.output_scale < 0 {
            return Err(format!("{} input/output scales must be non-negative exact exponents", self.table_id));
        }
        scale_exponent_to_multiplier(self.input_scale)?;
        scale_exponent_to_multiplier(self.output_scale)?;
        for (label, identity) in [("function_id", &self.function_id), ("domain_id", &self.domain_id), ("generator_id", &self.generator_id),] {
            if identity.is_empty() {
                return Err(format!("{} {} must not be empty", self.table_id, label));
            }
        }
        if self.input_min > self.input_max {
            return Err(format!("{} has an inverted input domain", self.table_id));
        }
        let expected_len = self
            .input_max
            .checked_sub(self.input_min)
            .and_then(|x| x.checked_add(1))
            .ok_or_else(|| format!("{} input domain length overflow", self.table_id))?;
        let expected_len = usize::try_from(expected_len).map_err(|_| format!("{} input domain is too large", self.table_id))?;
        if self.rows.len() != expected_len {
            return Err(format!("{} rows are not a complete contiguous domain", self.table_id));
        }
        for (row_idx, (input, output)) in self.rows.iter().enumerate() {
            if *input < self.input_min || *input > self.input_max {
                return Err(format!("{} row input {} is outside [{}, {}]", self.table_id, input, self.input_min, self.input_max));
            }
            if non_negative_input && *input < 0 {
                return Err(format!("{} contains a negative input", self.table_id));
            }
            if positive_output && *output <= 0 {
                return Err(format!("{} contains a non-positive output", self.table_id));
            }
            let expected_input = self
                .input_min
                .checked_add(i128::try_from(row_idx).map_err(|_| format!("{} row index does not fit i128", self.table_id))?,)
                .ok_or_else(|| format!("{} row input overflow", self.table_id))?;
            if *input != expected_input {
                return Err(format!("{} rows must be sorted and contiguous", self.table_id));
            }
        }
        let expected_digest = sha256_digest(&self.canonical_bytes())?;
        if self.digest != expected_digest {
            return Err(format!("{} digest does not match canonical rows", self.table_id));
        }
        Ok(())
    }

    /// Look up an exact row without interpolation or host-side fallback.
    pub fn lookup(&self, input: i128) -> Result<i128, String> {
        self.rows
            .binary_search_by_key(&input, |(key, _)| *key)
            .map(|idx| self.rows[idx].1)
            .map_err(|_| format!("{} has no row for input {}", self.table_id, input))
    }
}

/// Public grouped-edge topology.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalGraph {
    /// Number of rows in the public feature table.
    pub n_total: usize,
    /// Number of nodes in the current proof instance.
    pub n: usize,
    /// Number of valid edges.
    pub e: usize,
    /// Number of graphs in the batch.
    pub g_batch: usize,
    /// Absolute node ids selecting rows from the feature table.
    pub node_ids: Vec<usize>,
    /// Source node for every edge, in grouped stable order.
    pub src: Vec<usize>,
    /// Destination node for every edge, in grouped stable order.
    pub dst: Vec<usize>,
    /// Node prefix boundaries for each graph.
    pub node_ptr: Vec<usize>,
    /// Edge prefix boundaries for each graph.
    pub edge_ptr: Vec<usize>,
    /// Stable topology registry identity.
    pub topology_id: String,
    /// SHA-256 over the canonical graph fields (excluding these identity fields).
    pub topology_digest: String,
}

impl CanonicalGraph {
    /// Canonical bytes for topology identity and circuit wiring binding.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = b"ezkl-gat-topology-v1".to_vec();
        for value in [self.n_total, self.n, self.e, self.g_batch] {
            append_usize(&mut out, value);
        }
        for values in [self.node_ids.as_slice(), self.src.as_slice(), self.dst.as_slice(), self.node_ptr.as_slice(), self.edge_ptr.as_slice(),] {
            append_usize(&mut out, values.len());
            for value in values {
                append_usize(&mut out, *value);
            }
        }
        out
    }

    /// SHA-256 identity used to compare the normalized graph arrays across
    /// native baselines. The encoding streams the already-consumed arrays
    /// without building a second graph-sized byte buffer.
    pub fn comparison_canonical_digest(&self) -> Result<String, String> {
        #[cfg(feature = "ezkl")]
        {
            use sha2::{Digest, Sha256};

            fn update_u64(hasher: &mut Sha256, value: usize, label: &str) -> Result<(), String> {
                let value = u64::try_from(value)
                    .map_err(|_| format!("{} does not fit the shared u64 graph encoding", label))?;
                hasher.update(value.to_le_bytes());
                Ok(())
            }

            let mut hasher = Sha256::new();
            hasher.update(b"comparison-canonical-graph-v1\0");
            for (label, value) in [("N", self.n_total), ("E", self.e), ("G", self.g_batch)] {
                update_u64(&mut hasher, value, label)?;
            }
            for (label, values) in [
                ("src", self.src.as_slice()),
                ("dst", self.dst.as_slice()),
                ("node_ptr", self.node_ptr.as_slice()),
                ("edge_ptr", self.edge_ptr.as_slice()),
            ] {
                update_u64(&mut hasher, values.len(), label)?;
                for value in values {
                    update_u64(&mut hasher, *value, label)?;
                }
            }
            Ok(hex::encode(hasher.finalize()))
        }
        #[cfg(not(feature = "ezkl"))]
        {
            Err("shared canonical graph digest requires the official ezkl SHA-256 feature".into())
        }
    }

    /// Fill the topology digest after constructing a canonical graph.
    pub fn refresh_digest(&mut self) -> Result<(), String> {
        self.topology_digest = sha256_digest(&self.canonical_bytes())?;
        Ok(())
    }

    /// Validate graph, graph-group isolation and stable destination ordering.
    pub fn validate(&self) -> Result<(), String> {
        if self.n_total < self.n || self.n == 0 || self.e == 0 || self.g_batch == 0 {
            return Err("graph requires n_total >= n >= 1, e >= 1 and g_batch >= 1".into());
        }
        if self.node_ids.len() != self.n {
            return Err("node_ids length does not equal n".into());
        }
        if self.topology_id.is_empty() {
            return Err("topology_id must not be empty".into());
        }
        let expected_digest = sha256_digest(&self.canonical_bytes())?;
        if self.topology_digest != expected_digest {
            return Err("topology_digest does not match canonical graph wiring".into());
        }
        let mut seen = HashSet::new();
        for id in &self.node_ids {
            if *id >= self.n_total || !seen.insert(*id) {
                return Err("node_ids must be unique and within n_total".into());
            }
        }
        if self.src.len() != self.e || self.dst.len() != self.e {
            return Err("src/dst lengths must equal e".into());
        }
        if self.node_ptr.len() != self.g_batch + 1
            || self.edge_ptr.len() != self.g_batch + 1
            || self.node_ptr.first() != Some(&0)
            || self.edge_ptr.first() != Some(&0)
            || self.node_ptr.last() != Some(&self.n)
            || self.edge_ptr.last() != Some(&self.e)
        {
            return Err("node_ptr/edge_ptr must have g_batch+1 entries and close at n/e".into());
        }
        for ptr in self.node_ptr.windows(2) {
            if ptr[0] >= ptr[1] {
                return Err("node_ptr must be strictly increasing".into());
            }
        }
        for ptr in self.edge_ptr.windows(2) {
            if ptr[0] >= ptr[1] {
                return Err("edge_ptr must be strictly increasing".into());
            }
        }
        let mut has_self_loop = vec![false; self.n];
        for graph in 0..self.g_batch {
            let node_start = self.node_ptr[graph];
            let node_end = self.node_ptr[graph + 1];
            let edge_start = self.edge_ptr[graph];
            let edge_end = self.edge_ptr[graph + 1];
            let mut previous_destination = None;
            let mut sources_for_destination = HashSet::new();
            for edge in edge_start..edge_end {
                let src = self.src[edge];
                let dst = self.dst[edge];
                if src < node_start || src >= node_end || dst < node_start || dst >= node_end {
                    return Err(format!("edge {} crosses graph boundary", edge));
                }
                if previous_destination != Some(dst) {
                    if previous_destination.is_some_and(|previous| previous > dst) {
                        return Err("edges must be stably sorted by destination within each graph".into());
                    }
                    previous_destination = Some(dst);
                    sources_for_destination.clear();
                }
                if !sources_for_destination.insert(src) {
                    return Err(format!("duplicate directed edge ({}, {})", src, dst));
                }
                if src == dst {
                    has_self_loop[src] = true;
                }
            }
        }
        if let Some(node) = has_self_loop.iter().position(|present| !present) {
            return Err(format!("node {} is missing its required per-node self-loop", node));
        }
        Ok(())
    }

    /// Return the graph containing a node.
    pub fn graph_for_node(&self, node: usize) -> Result<usize, String> {
        self.node_ptr.windows(2).position(|w| w[0] <= node && node < w[1]).ok_or_else(|| format!("node {} is outside node_ptr", node))
    }

    /// Return the graph containing an edge.
    pub fn graph_for_edge(&self, edge: usize) -> Result<usize, String> {
        self.edge_ptr.windows(2).position(|w| w[0] <= edge && edge < w[1]).ok_or_else(|| format!("edge {} is outside edge_ptr", edge))
    }

    /// Return contiguous edge ranges for each destination group.
    pub fn destination_groups(&self) -> Result<Vec<(usize, usize, usize)>, String> {
        self.validate()?;
        Ok(self.destination_groups_validated())
    }

    /// Build contiguous destination ranges after the caller has already
    /// validated this immutable canonical graph.
    pub(crate) fn destination_groups_validated(&self) -> Vec<(usize, usize, usize)> {
        let mut groups = Vec::new();
        for graph in 0..self.g_batch {
            let mut start = self.edge_ptr[graph];
            let end = self.edge_ptr[graph + 1];
            while start < end {
                let dst = self.dst[start];
                let mut next = start + 1;
                while next < end && self.dst[next] == dst {
                    next += 1;
                }
                groups.push((dst, start, next));
                start = next;
            }
        }
        groups
    }
}

/// Public scale chain.  Values are exponents in ezkl's fixed point scale
/// convention, except field-exact alpha/inverse values which are represented
/// by `None` and are not silently rescaled.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScaleChain {
    /// Input feature scale.
    pub s_input: i32,
    /// Stable-softmax score difference scale.
    pub s_delta: i32,
    /// Exponential table output scale.
    pub s_exp: i32,
    /// Hidden input scales by layer.
    pub s_h: Vec<i32>,
    /// Hidden projection scales by layer/head.
    pub s_w_hidden: Vec<Vec<i32>>,
    /// Hidden projection output scales by layer/head.
    pub s_h_prime: Vec<Vec<i32>>,
    /// Hidden source-attention scales by layer/head.
    pub s_a_src_hidden: Vec<Vec<i32>>,
    /// Hidden destination-attention scales by layer/head.
    pub s_a_dst_hidden: Vec<Vec<i32>>,
    /// Hidden aggregation pre-ELU scale by hidden layer.
    pub s_agg_pre: Vec<i32>,
    /// Hidden ELU output scale by hidden layer.
    pub s_agg: Vec<i32>,
    /// Hidden concatenation scale by hidden layer.
    pub s_cat: Vec<i32>,
    /// Output projection scales by head.
    pub s_w_output: Vec<i32>,
    /// Output projection scale by head.
    pub s_y_prime: Vec<i32>,
    /// Output source-attention scales by head.
    pub s_a_src_output: Vec<i32>,
    /// Output destination-attention scales by head.
    pub s_a_dst_output: Vec<i32>,
    /// Output bias scales by head.
    pub s_b_output: Vec<i32>,
    /// Output recovery scale.
    pub s_out_pre: i32,
    /// Common output-head scale.
    pub s_out_head: i32,
    /// Final output scale.
    pub s_out: i32,
}

/// Convert an ezkl log2 exponent into its exact positive power-of-two
/// multiplier.  This intentionally does not use floating point or rounding.
pub fn scale_exponent_to_multiplier(exponent: i32) -> Result<u128, String> {
    if exponent < 0 {
        return Err(format!("scale exponent {} is not a positive power-of-two", exponent));
    }
    let exponent = u32::try_from(exponent).map_err(|_| "invalid scale exponent".to_string())?;
    1_u128.checked_shl(exponent).ok_or_else(|| "scale exponent exceeds exact u128 multiplier range".into())
}

/// Convert an exact positive power-of-two multiplier into an ezkl log2
/// exponent.  Non-powers and zero are rejected instead of being rounded.
pub fn multiplier_to_scale_exponent(multiplier: u128) -> Result<i32, String> {
    if multiplier == 0 || !multiplier.is_power_of_two() {
        return Err(format!("multiplier {} is not a positive power of two", multiplier));
    }
    i32::try_from(multiplier.trailing_zeros()).map_err(|_| "power-of-two exponent does not fit i32".into())
}

impl ScaleChain {
    /// Validate the scale closure required by the protocol.
    pub fn validate(&self, layers: usize, hidden_heads: &[usize], output_heads: usize,) -> Result<(), String> {
        if layers < 2 || self.s_input <= 0 || self.s_delta <= 0 || self.s_exp <= 0 {
            return Err("invalid scale chain or layer count".into());
        }
        if self.s_h.len() != layers
            || self.s_w_hidden.len() != layers - 1
            || self.s_h_prime.len() != layers - 1
            || self.s_a_src_hidden.len() != layers - 1
            || self.s_a_dst_hidden.len() != layers - 1
            || self.s_agg_pre.len() != layers - 1
            || self.s_agg.len() != layers - 1
            || self.s_cat.len() != layers - 1
            || hidden_heads.len() != layers - 1
            || self.s_w_output.len() != output_heads
            || self.s_y_prime.len() != output_heads
            || self.s_a_src_output.len() != output_heads
            || self.s_a_dst_output.len() != output_heads
            || self.s_b_output.len() != output_heads
        {
            return Err("scale chain dimensions do not match model architecture".into());
        }
        let mut all_scales = vec![self.s_input, self.s_delta, self.s_exp, self.s_out_pre, self.s_out_head, self.s_out,];
        all_scales.extend(self.s_h.iter().copied());
        all_scales.extend(self.s_agg_pre.iter().copied());
        all_scales.extend(self.s_agg.iter().copied());
        all_scales.extend(self.s_cat.iter().copied());
        all_scales.extend(self.s_w_output.iter().copied());
        all_scales.extend(self.s_y_prime.iter().copied());
        all_scales.extend(self.s_a_src_output.iter().copied());
        all_scales.extend(self.s_a_dst_output.iter().copied());
        all_scales.extend(self.s_b_output.iter().copied());
        for layer in &self.s_w_hidden {
            all_scales.extend(layer.iter().copied());
        }
        for layer in &self.s_h_prime {
            all_scales.extend(layer.iter().copied());
        }
        for layer in &self.s_a_src_hidden {
            all_scales.extend(layer.iter().copied());
        }
        for layer in &self.s_a_dst_hidden {
            all_scales.extend(layer.iter().copied());
        }
        if all_scales.iter().copied().any(|scale| scale_exponent_to_multiplier(scale).is_err()) {
            return Err("all numeric scales must be exact positive powers of two".into());
        }
        if self.s_h[0] != self.s_input || self.s_cat != self.s_agg || self.s_out != self.s_out_head || self.s_out_pre <= 0 || self.s_out_head <= 0 {
            return Err("input/concat/output scale closure is not canonical".into());
        }
        for layer in 0..layers - 1 {
            if self.s_h[layer + 1] != self.s_cat[layer] {
                return Err(format!("hidden concat transition at layer {} does not close to s_cat", layer));
            }
            if self.s_w_hidden[layer].len() != hidden_heads[layer]
                || self.s_h_prime[layer].len() != hidden_heads[layer]
                || self.s_a_src_hidden[layer].len() != hidden_heads[layer]
                || self.s_a_dst_hidden[layer].len() != hidden_heads[layer]
            {
                return Err(format!("hidden scale shape mismatch at layer {}", layer));
            }
            for head in 0..hidden_heads[layer] {
                if self.s_h_prime[layer][head] != self.s_h[layer] + self.s_w_hidden[layer][head] {
                    return Err(format!("hidden projection scale mismatch at {}:{}", layer, head));
                }
                if self.s_h_prime[layer][head] + self.s_a_src_hidden[layer][head] != self.s_delta
                    || self.s_h_prime[layer][head] + self.s_a_dst_hidden[layer][head] != self.s_delta
                {
                    return Err(format!("hidden attention scale mismatch at {}:{}", layer, head));
                }
            }
            if self.s_agg_pre[layer] <= 0 || self.s_agg[layer] <= 0 || self.s_cat[layer] <= 0 {
                return Err(format!("invalid hidden aggregation scale at layer {}", layer));
            }
        }
        self.shared_elu_scales()?;
        for head in 0..output_heads {
            if self.s_y_prime[head] != self.s_h[layers - 1] + self.s_w_output[head]
                || self.s_y_prime[head] + self.s_a_src_output[head] != self.s_delta
                || self.s_y_prime[head] + self.s_a_dst_output[head] != self.s_delta
                || self.s_b_output[head] != self.s_out_head
            {
                return Err(format!("output scale closure mismatch at head {}", head));
            }
        }
        Ok(())
    }

    /// Return the shared ELU table scales after proving that every hidden
    /// layer uses the same authenticated table domain.
    pub fn shared_elu_scales(&self) -> Result<(i32, i32), String> {
        let (&input, rest_input) = self.s_agg_pre.split_first().ok_or_else(|| "missing hidden aggregation input scales".to_string())?;
        let (&output, rest_output) = self.s_agg.split_first().ok_or_else(|| "missing hidden aggregation output scales".to_string())?;
        if rest_input.iter().any(|value| *value != input) || rest_output.iter().any(|value| *value != output) {
            return Err("hidden layers require one shared authenticated ELU table scale".into());
        }
        Ok((input, output))
    }

    /// Serialize the typed chain in the canonical source format used by the
    /// checkpoint manifests and the main GAT implementation.
    pub fn canonical_serialization(&self) -> Result<String, String> {
        let source_value = |value: i32| scale_exponent_to_multiplier(value).map(|value| value.to_string());
        let vector = |values: &[i32]| { values.iter().copied().map(source_value).collect::<Result<Vec<_>, _>>().map(|values| values.join(",")) };
        let matrix = |values: &[Vec<i32>]| {
            values.iter().map(|row| vector(row).map(|row| format!("[{}]", row))).collect::<Result<Vec<_>, _>>().map(|values| values.join(";"))
        };
        Ok(format!(
            "schema=gat_l3_scale_chain|S_input={}|S_Delta={}|S_exp={}|S_H=[{}]|S_W_hidden=[{}]|S_H_prime_hidden=[{}]|S_a_src_hidden=[{}]|S_a_dst_hidden=[{}]|S_agg_pre=[{}]|S_agg=[{}]|S_cat=[{}]|S_out_pre={}|S_out_head={}|S_out={}|S_W_output=[{}]|S_Y_prime_output=[{}]|S_a_src_output=[{}]|S_a_dst_output=[{}]|S_b_output=[{}]|S_inv=field_exact|S_alpha=field_exact|round=nearest_ties_away|rescale=none|output_average=signed_round_div",
            source_value(self.s_input)?,
            source_value(self.s_delta)?,
            source_value(self.s_exp)?,
            vector(&self.s_h)?,
            matrix(&self.s_w_hidden)?,
            matrix(&self.s_h_prime)?,
            matrix(&self.s_a_src_hidden)?,
            matrix(&self.s_a_dst_hidden)?,
            vector(&self.s_agg_pre)?,
            vector(&self.s_agg)?,
            vector(&self.s_cat)?,
            source_value(self.s_out_pre)?,
            source_value(self.s_out_head)?,
            source_value(self.s_out)?,
            vector(&self.s_w_output)?,
            vector(&self.s_y_prime)?,
            vector(&self.s_a_src_output)?,
            vector(&self.s_a_dst_output)?,
            vector(&self.s_b_output)?,
        ))
    }

    /// Canonical bytes covered by the quantization identity.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = b"ezkl-gat-quant-v1".to_vec();
        for scale in [self.s_input, self.s_delta, self.s_exp] {
            out.extend_from_slice(&scale.to_le_bytes());
        }
        for group in [self.s_h.as_slice()] {
            append_usize(&mut out, group.len());
            for scale in group {
                out.extend_from_slice(&scale.to_le_bytes());
            }
        }
        for layers in [self.s_w_hidden.as_slice(), self.s_h_prime.as_slice(), self.s_a_src_hidden.as_slice(), self.s_a_dst_hidden.as_slice(),] {
            append_usize(&mut out, layers.len());
            for layer in layers {
                append_usize(&mut out, layer.len());
                for scale in layer {
                    out.extend_from_slice(&scale.to_le_bytes());
                }
            }
        }
        for group in [self.s_agg_pre.as_slice(), self.s_agg.as_slice(), self.s_cat.as_slice(),] {
            append_usize(&mut out, group.len());
            for scale in group {
                out.extend_from_slice(&scale.to_le_bytes());
            }
        }
        for scale in [self.s_out_pre, self.s_out_head, self.s_out] {
            out.extend_from_slice(&scale.to_le_bytes());
        }
        for group in
            [self.s_w_output.as_slice(), self.s_y_prime.as_slice(), self.s_a_src_output.as_slice(), self.s_a_dst_output.as_slice(), self.s_b_output.as_slice(),]
        {
            append_usize(&mut out, group.len());
            for scale in group {
                out.extend_from_slice(&scale.to_le_bytes());
            }
        }
        out
    }
}

/// Registered exact tables used by the GAT relation.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaticTables {
    /// LeakyReLU table at score scale.
    pub lrelu: ExactTable,
    /// Stable negative exponential table.
    pub exp: ExactTable,
    /// ELU table at aggregation scale.
    pub elu: ExactTable,
    /// Backend-only ReLU table used by exact max arguments.
    pub max_relu: ExactTable,
    /// Canonical static-table registry identity.
    pub static_tables_id: String,
    /// Identity of the backend-only auxiliary table registry.
    pub backend_aux_tables_id: String,
    /// SHA-256 over all table identities, rows and backend table identity.
    pub static_tables_digest: String,
}

impl StaticTables {
    /// Canonical bytes covered by the complete table registry digest.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = b"ezkl-gat-static-tables-v1".to_vec();
        append_string(&mut out, &self.static_tables_id);
        append_string(&mut out, &self.backend_aux_tables_id);
        for table in [&self.lrelu, &self.exp, &self.elu, &self.max_relu] {
            let bytes = table.canonical_bytes();
            append_usize(&mut out, bytes.len());
            out.extend_from_slice(&bytes);
            append_string(&mut out, &table.digest);
        }
        out
    }

    /// Fill the complete registry digest after constructing canonical tables.
    pub fn refresh_digest(&mut self) -> Result<(), String> {
        self.static_tables_digest = sha256_digest(&self.canonical_bytes())?;
        Ok(())
    }

    /// Validate all table semantics.
    pub fn validate(&self) -> Result<(), String> {
        if self.static_tables_id.is_empty() || self.backend_aux_tables_id.is_empty() {
            return Err("static table registry ids must not be empty".into());
        }
        self.lrelu.validate(false, false)?;
        self.exp.validate(true, true)?;
        self.elu.validate(false, false)?;
        self.max_relu.validate(false, false)?;
        validate_registered_table(&self.lrelu, "lrelu")?;
        validate_registered_table(&self.exp, "exp_neg")?;
        validate_registered_table(&self.elu, "elu")?;
        validate_registered_table(&self.max_relu, "max_relu")?;
        let expected_digest = sha256_digest(&self.canonical_bytes())?;
        if self.static_tables_digest != expected_digest {
            return Err("static_tables_digest does not match canonical tables".into());
        }
        Ok(())
    }
}

fn validate_registered_table(table: &ExactTable, expected_function: &str) -> Result<(), String> {
    if table.function_id != expected_function {
        return Err(format!("table {} has function {}, expected {}", table.table_id, table.function_id, expected_function));
    }
    match expected_function {
        "lrelu" => {
            let mut previous = None;
            for (input, output) in &table.rows {
                if *input >= 0 && *output != *input {
                    return Err(format!("{} positive LeakyReLU row is not identity at {}", table.table_id, input));
                }
                if *input < 0 && (*output < *input || *output > 0) {
                    return Err(format!("{} negative LeakyReLU row is outside the canonical slope range at {}", table.table_id, input));
                }
                if let Some(previous) = previous {
                    if *output < previous {
                        return Err(format!("{} LeakyReLU rows are not monotone", table.table_id));
                    }
                }
                previous = Some(*output);
            }
        }
        "exp_neg" => {
            let mut previous = None;
            for (input, output) in &table.rows {
                if *input < 0 || *output <= 0 {
                    return Err(format!("{} ExpMap rows violate the non-negative/positive domain", table.table_id));
                }
                if let Some(previous) = previous {
                    if *output > previous {
                        return Err(format!("{} ExpMap rows must be non-increasing", table.table_id));
                    }
                }
                previous = Some(*output);
            }
        }
        "elu" => {
            let mut previous = None;
            for (input, output) in &table.rows {
                if *input >= 0 && *output != *input {
                    return Err(format!("{} non-negative ELU row is not identity at {}", table.table_id, input));
                }
                if *input < 0 && *output >= 0 {
                    return Err(format!("{} negative ELU row must remain negative at {}", table.table_id, input));
                }
                if let Some(previous) = previous {
                    if *output < previous {
                        return Err(format!("{} ELU rows are not monotone", table.table_id));
                    }
                }
                previous = Some(*output);
            }
        }
        "max_relu" => {
            for (input, output) in &table.rows {
                let expected = (*input).max(0);
                if *output != expected {
                    return Err(format!("{} max/ReLU row at {} is {}, expected {}", table.table_id, input, output, expected));
                }
            }
        }
        _ => { return Err(format!("unsupported exact table function {}", expected_function)) } }
    Ok(())
}

/// Per-layer/head bound registration.  A single global bound is insufficient
/// because every lookup family has its own score, exponent, aggregation and
/// division domains.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundFamily {
    /// Layer index; `usize::MAX` denotes the output layer.
    pub layer: usize,
    /// Head index within the layer/final output family.
    pub head: usize,
    /// Whether this is an output-head family.
    pub is_output: bool,
    /// Absolute score bound before LeakyReLU.
    pub max_score_abs: i128,
    /// Projection weight absolute bound.
    pub max_weight_abs: i128,
    /// Attention-vector absolute bound.
    pub max_attention_abs: i128,
    /// Output-bias absolute bound (zero for hidden families).
    pub max_bias_abs: i128,
    /// Absolute score-table output bound.
    pub max_table_output_abs: i128,
    /// Stable-softmax difference bound.
    pub max_delta: i128,
    /// Exponential table output bound.
    pub max_exp: i128,
    /// Group denominator bound.
    pub max_sum: i128,
    /// Integer aggregation numerator bound.
    pub max_aggregation_num: i128,
    /// Signed division dividend bound.
    pub max_dividend_abs: i128,
    /// All non-negative division auxiliary bounds.
    pub max_aux_abs: i128,
    /// Maximum positive divisor for this family.
    pub max_divisor: i128,
}

impl BoundFamily {
    fn validate(&self, label: &str, field_half: &BigUint) -> Result<(), String> {
        if self.max_divisor <= 0 {
            return Err(format!("{} max_divisor must be positive", label));
        }
        for (name, value) in [
            ("max_score_abs", self.max_score_abs),
            ("max_weight_abs", self.max_weight_abs),
            ("max_attention_abs", self.max_attention_abs),
            ("max_bias_abs", self.max_bias_abs),
            ("max_table_output_abs", self.max_table_output_abs),
            ("max_delta", self.max_delta),
            ("max_exp", self.max_exp),
            ("max_sum", self.max_sum),
            ("max_aggregation_num", self.max_aggregation_num),
            ("max_dividend_abs", self.max_dividend_abs),
            ("max_aux_abs", self.max_aux_abs),
            ("max_divisor", self.max_divisor),
        ] {
            if value < 0 || BigUint::from(value as u128) >= field_half.clone() {
                return Err(format!("{}.{} does not satisfy the strict BN254 field no-wrap bound", label, name));
            }
        }
        Ok(())
    }
}

/// Public no-wrap and range ledger.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntegerBoundLedger {
    /// Maximum absolute score before the LeakyReLU table.
    pub max_score_abs: i128,
    /// Maximum absolute table output.
    pub max_table_output_abs: i128,
    /// Maximum non-negative exponential difference.
    pub max_delta: i128,
    /// Maximum exponential output.
    pub max_exp: i128,
    /// Maximum group denominator.
    pub max_sum: i128,
    /// Maximum aggregation numerator.
    pub max_aggregation_num: i128,
    /// Maximum absolute signed dividend.
    pub max_dividend_abs: i128,
    /// Maximum quotient/remainder auxiliary value.
    pub max_aux_abs: i128,
    /// Maximum positive divisor across all recovery/alignment relations.
    pub max_divisor: i128,
    /// Backend field modulus used for the ledger certificate.
    pub field_modulus: String,
    /// Stable ledger identity.
    pub ledger_id: String,
    /// SHA-256 over the canonical ledger fields.
    pub ledger_digest: String,
    /// Per hidden/output layer-head bounds.
    pub families: Vec<BoundFamily>,
}

impl IntegerBoundLedger {
    /// Canonical bytes covered by the ledger digest.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = b"ezkl-gat-ledger-v1".to_vec();
        append_string(&mut out, &self.ledger_id);
        append_string(&mut out, &self.field_modulus);
        for value in [
            self.max_score_abs,
            self.max_table_output_abs,
            self.max_delta,
            self.max_exp,
            self.max_sum,
            self.max_aggregation_num,
            self.max_dividend_abs,
            self.max_aux_abs,
            self.max_divisor,
        ] {
            append_i128(&mut out, value);
        }
        for family in &self.families {
            append_usize(&mut out, family.layer);
            append_usize(&mut out, family.head);
            out.push(u8::from(family.is_output));
            for value in [
                family.max_score_abs,
                family.max_weight_abs,
                family.max_attention_abs,
                family.max_bias_abs,
                family.max_table_output_abs,
                family.max_delta,
                family.max_exp,
                family.max_sum,
                family.max_aggregation_num,
                family.max_dividend_abs,
                family.max_aux_abs,
                family.max_divisor,
            ] {
                append_i128(&mut out, value);
            }
        }
        out
    }

    /// Fill the ledger digest after constructing all family bounds.
    pub fn refresh_digest(&mut self) -> Result<(), String> {
        self.ledger_digest = sha256_digest(&self.canonical_bytes())?;
        Ok(())
    }

    /// Validate no-wrap and non-negative auxiliary bounds against BN254 Fr.
    pub fn validate(&self) -> Result<(), String> {
        let modulus = parse_authenticated_bn254_modulus(&self.field_modulus)?;
        if modulus <= BigUint::from(2_u8) {
            return Err("field_modulus is too small".into());
        }
        let half = (&modulus - BigUint::one()) >> 1usize;
        for (name, value) in [
            ("max_score_abs", self.max_score_abs),
            ("max_table_output_abs", self.max_table_output_abs),
            ("max_delta", self.max_delta),
            ("max_exp", self.max_exp),
            ("max_sum", self.max_sum),
            ("max_aggregation_num", self.max_aggregation_num),
            ("max_dividend_abs", self.max_dividend_abs),
            ("max_aux_abs", self.max_aux_abs),
            ("max_divisor", self.max_divisor),
        ] {
            if value < 0 || BigUint::from(value as u128) >= half {
                return Err(format!("{} does not satisfy field no-wrap bound", name));
            }
        }
        if self.max_divisor <= 0 {
            return Err("max_divisor must be positive".into());
        }
        if self.ledger_id.is_empty() {
            return Err("ledger_id must not be empty".into());
        }
        let expected_digest = sha256_digest(&self.canonical_bytes())?;
        if self.ledger_digest != expected_digest {
            return Err("ledger_digest does not match canonical bounds".into());
        }
        if self.families.is_empty() {
            return Err("integer bound ledger requires per-head families".into());
        }
        let mut identities = HashSet::new();
        for (idx, family) in self.families.iter().enumerate() {
            let identity = (family.layer, family.head, family.is_output);
            if !identities.insert(identity) {
                return Err(format!("duplicate bound family at index {}", idx));
            }
            family.validate(&format!("family[{}]", idx), &half)?;
            for (name, family_value, global_value) in [
                ("max_score_abs", family.max_score_abs, self.max_score_abs),
                ("max_table_output_abs", family.max_table_output_abs, self.max_table_output_abs,),("max_delta", family.max_delta, self.max_delta),
                ("max_exp", family.max_exp, self.max_exp),
                ("max_sum", family.max_sum, self.max_sum),
                ("max_aggregation_num", family.max_aggregation_num, self.max_aggregation_num,),("max_dividend_abs", family.max_dividend_abs, self.max_dividend_abs,),("max_aux_abs", family.max_aux_abs, self.max_aux_abs),
            ] {
                if family_value > global_value {
                    return Err(format!("family[{}] {} exceeds global ledger bound", idx, name));
                }
            }
            if family.max_divisor > self.max_divisor {
                return Err(format!("family[{}] divisor bound exceeds global max_divisor", idx));
            }
        }
        Ok(())
    }
}

/// A private projection/attention/output head tensor bundle.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeadWeights {
    /// Row-major projection matrix.
    pub projection: Vec<i128>,
    /// Source attention vector.
    pub a_src: Vec<i128>,
    /// Destination attention vector.
    pub a_dst: Vec<i128>,
    /// Output-only bias.  Hidden heads must leave this empty.
    pub bias: Vec<i128>,
}

/// One hidden layer specification.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HiddenLayerSpec {
    /// Input dimension.
    pub d_in: usize,
    /// Number of hidden heads.
    pub heads: Vec<HeadWeights>,
    /// Per-head output dimension.
    pub d_head: usize,
}

/// Canonical complete model.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalModel {
    /// Hidden attention layers; there must be at least one.
    pub hidden_layers: Vec<HiddenLayerSpec>,
    /// Final output heads.
    pub output_heads: Vec<HeadWeights>,
    /// Number of output classes.
    pub classes: usize,
    /// Canonical model identity.
    pub model_arch_id: String,
}

impl CanonicalModel {
    /// Number of attention layers including the output layer.
    pub fn layer_count(&self) -> usize {
        self.hidden_layers.len() + 1
    }

    /// Validate dimensions and hidden/output bias policy.
    pub fn validate(&self, input_dim: usize) -> Result<(), String> {
        if self.hidden_layers.is_empty() || self.output_heads.is_empty() || self.classes == 0 {
            return Err("GAT requires at least one hidden layer, one output head and one class".into(),);
        }
        if self.model_arch_id.is_empty() {
            return Err("model_arch_id must not be empty".into());
        }
        let mut current_dim = input_dim;
        for (layer_idx, layer) in self.hidden_layers.iter().enumerate() {
            if layer.d_in != current_dim || layer.d_head == 0 || layer.heads.is_empty() {
                return Err(format!("hidden layer {} dimension/head mismatch", layer_idx));
            }
            for head in &layer.heads {
                let projection_len = layer.d_in.checked_mul(layer.d_head).ok_or_else(|| { format!("hidden layer {} projection shape overflow", layer_idx) })?;
                if head.projection.len() != projection_len || head.a_src.len() != layer.d_head || head.a_dst.len() != layer.d_head || !head.bias.is_empty() {
                    return Err(format!("invalid hidden head tensor shape at layer {}", layer_idx));
                }
            }
            current_dim = layer.heads.len().checked_mul(layer.d_head).ok_or_else(|| format!("hidden layer {} output shape overflow", layer_idx))?;
        }
        for head in &self.output_heads {
            let projection_len = current_dim.checked_mul(self.classes).ok_or_else(|| "output projection shape overflow".to_string())?;
            if head.projection.len() != projection_len
                || head.a_src.len() != self.classes
                || head.a_dst.len() != self.classes
                || head.bias.len() != self.classes
            {
                return Err("invalid output head tensor shape".into());
            }
        }
        Ok(())
    }

    /// Flatten the canonical private model parameters in graph constant order.
    pub fn flattened_params(&self) -> Vec<i128> {
        let mut out = Vec::new();
        for layer in &self.hidden_layers {
            for head in &layer.heads {
                out.extend(&head.projection);
                out.extend(&head.a_src);
                out.extend(&head.a_dst);
            }
        }
        for head in &self.output_heads {
            out.extend(&head.projection);
            out.extend(&head.a_src);
            out.extend(&head.a_dst);
            out.extend(&head.bias);
        }
        out
    }
}

/// Complete canonical adapter input.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalGat {
    /// Schema version.
    pub schema_version: u32,
    /// Public graph/topology.
    pub graph: CanonicalGraph,
    /// Public feature table in row-major integer form.
    pub features: Vec<i128>,
    /// Input feature dimension.
    pub feature_dim: usize,
    /// Canonical private model.
    pub model: CanonicalModel,
    /// Fixed point scale chain.
    pub scales: ScaleChain,
    /// Exact GAT/backend tables.
    pub tables: StaticTables,
    /// Public no-wrap ledger.
    pub ledger: IntegerBoundLedger,
    /// Canonical quantization identity.
    pub quant_cfg_id: String,
    /// Backend identity/profile.
    pub ezkl_security_profile_id: String,
    /// Authenticated topology registry identity (must match `graph`).
    pub topology: DigestIdentity,
    /// Authenticated quantization identity (must match `scales`).
    pub quantization: DigestIdentity,
    /// Authenticated integer-bound ledger identity (must match `ledger`).
    pub ledger_identity: DigestIdentity,
    /// Native compiled circuit/settings/VK identity.  It is bound by the
    /// native setup phase; a loader result before setup deliberately carries
    /// no circuit identity.
    pub circuit_ref: Option<CircuitRef>,
    /// Public H(W) binding for the private model closure.
    pub model_binding: ModelBinding,
    /// Whether this serialized graph is the compile artifact with private
    /// model values redacted.  Redaction is tracked separately from the
    /// public model-hash domain so the native statement keeps the same
    /// public identity before and after serialization.
    #[serde(default)]
    pub private_params_redacted: bool,
}

impl CanonicalGat {
    /// Validate every public and private schema contract before graph building.
    pub fn validate(&self) -> Result<(), String> {
        self.validate_inner(true)
    }

    /// Validate the relation before native setup has produced compiled
    /// circuit/settings/VK identities.  This is an internal setup-phase
    /// check; public statements and artifact manifests always use [`Self::validate`].
    pub fn validate_for_compile(&self) -> Result<(), String> {
        self.validate_inner(false)
    }

    fn validate_inner(&self, require_circuit_ref: bool) -> Result<(), String> {
        if self.schema_version != GAT_SCHEMA_VERSION {
            return Err(format!("unsupported GAT schema version {}", self.schema_version));
        }
        if self.quant_cfg_id.is_empty() || self.ezkl_security_profile_id.is_empty() {
            return Err("quant_cfg_id and ezkl_security_profile_id are required".into());
        }
        self.graph.validate()?;
        self.topology.validate("topology")?;
        let expected_topology_digest = sha256_digest(&self.graph.canonical_bytes())?;
        if self.topology.digest != expected_topology_digest || self.topology.id != self.graph.topology_id {
            return Err("topology identity does not match canonical graph wiring".into());
        }
        match &self.circuit_ref {
            Some(circuit_ref) => {
                circuit_ref.validate()?;
                if circuit_ref.ezkl_security_profile_id != self.ezkl_security_profile_id {
                    return Err("CircuitRef security profile disagrees with canonical profile".into(),);
                }
            }
            None if require_circuit_ref => {
                return Err("canonical GAT circuit identity is not bound by native setup".into());
            }
            None => {}
        }
        let expected_feature_len = self.graph.n_total.checked_mul(self.feature_dim).ok_or_else(|| "feature table shape overflow".to_string())?;
        if self.features.len() != expected_feature_len || self.feature_dim == 0 {
            return Err("feature table shape does not match n_total and feature_dim".into());
        }
        self.model.validate(self.feature_dim)?;
        let hidden_heads = self.model.hidden_layers.iter().map(|layer| layer.heads.len()).collect::<Vec<_>>();
        self.scales.validate(self.model.layer_count(), &hidden_heads, self.model.output_heads.len(),)?;
        let expected_quant_digest = sha256_digest(&self.scales.canonical_bytes())?;
        if self.quantization.id != self.quant_cfg_id || self.quantization.digest != expected_quant_digest {
            return Err("quantization identity does not match scale chain".into());
        }
        self.tables.validate()?;
        let (elu_input_scale, elu_output_scale) = self.scales.shared_elu_scales()?;
        for (label, table, input_scale, output_scale) in [
            ("lrelu", &self.tables.lrelu, self.scales.s_delta, self.scales.s_delta,),("exp", &self.tables.exp, self.scales.s_delta, self.scales.s_exp,),("elu", &self.tables.elu, elu_input_scale, elu_output_scale),
            // The auxiliary max/ReLU table operates on canonical integer
            // representatives. Its registered scale still closes to the
            // score scale used by the max witness relation.
            ("max_relu", &self.tables.max_relu, self.scales.s_delta, self.scales.s_delta),
        ] {
            if table.input_scale != input_scale || table.output_scale != output_scale {
                return Err(format!(
                    "{} table scale closure is ({}, {}), expected ({}, {})",
                    label, table.input_scale, table.output_scale, input_scale, output_scale
                ));
            }
        }
        self.validate_table_bounds()?;
        self.ledger.validate()?;
        let expected_ledger_digest = sha256_digest(&self.ledger.canonical_bytes())?;
        if self.ledger_identity.id != self.ledger.ledger_id || self.ledger_identity.digest != expected_ledger_digest {
            return Err("ledger identity does not match bound ledger".into());
        }
        if self.model_binding.hash_domain != MODEL_HASH_DOMAIN {
            return Err("unsupported model binding hash domain".into());
        }
        self.model_binding.validate()?;
        if self.private_params_redacted {
            if self.model.flattened_params().iter().any(|value| *value != 0) {
                return Err("redacted model binding contains non-zero private parameters".into());
            }
        } else if self.model_binding.canonical_model_digest != self.canonical_model_digest()? {
            return Err("canonical model digest does not match private parameter closure".into());
        }
        self.validate_bound_families()?;
        Ok(())
    }

    fn validate_table_bounds(&self) -> Result<(), String> {
        let check = |table: &ExactTable, label: &str, input_bound: i128, output_bound: i128| -> Result<(), String> {
            if input_bound < 0 || output_bound < 0 {
                return Err(format!("{} has a negative ledger bound", label));
            }
            let input_bound = input_bound as u128;
            let output_bound = output_bound as u128;
            if table.rows.iter().any(|(input, output)| { input.unsigned_abs() > input_bound || output.unsigned_abs() > output_bound}){
                return Err(format!("{} row exceeds authenticated ledger bounds", label));
            }
            Ok(())
        };
        check(&self.tables.lrelu, "lrelu", self.ledger.max_score_abs, self.ledger.max_table_output_abs,)?;
        check(&self.tables.exp, "exp", self.ledger.max_delta, self.ledger.max_exp,)?;
        check(&self.tables.elu, "elu", self.ledger.max_dividend_abs, self.ledger.max_table_output_abs,)?;
        check(&self.tables.max_relu, "max_relu", self.ledger.max_aux_abs, self.ledger.max_table_output_abs,) }

    fn validate_bound_families(&self) -> Result<(), String> {
        let expected = self.model.hidden_layers.iter().map(|layer| layer.heads.len()).sum::<usize>() + self.model.output_heads.len();
        if self.ledger.families.len() != expected {
            return Err(format!("bound ledger has {} families, expected {}", self.ledger.families.len(), expected));
        }
        let max_degree = self
            .graph
            .destination_groups_validated()
            .iter()
            .map(|(_, start, end)| end - start)
            .max()
            .and_then(|degree| i128::try_from(degree).ok())
            .ok_or_else(|| "maximum graph degree does not fit the integer ledger".to_string())?;
        // A maximum-score edge in every nonempty destination group has
        // delta = 0, so the exponential row at zero is necessarily consumed.
        // This checks only that concrete reachable row; it does not assume
        // that any table covers the ledger's entire signed interval.
        let exp_zero_output = self
            .tables
            .exp
            .rows
            .iter()
            .find(|(input, _)| *input == 0)
            .map(|(_, output)| *output)
            .ok_or_else(|| "exponential table is missing the reachable zero-delta row".to_string())?;
        let check_aggregation_budget = |family: &BoundFamily, projected_scale: i32, is_output: bool, label: &str| -> Result<(), String> {
            if family.max_exp < exp_zero_output {
                return Err(format!(
                    "{}.max_exp {} is below the required zero-delta exponential output {}",
                    label, family.max_exp, exp_zero_output
                ));
            }
            let required_sum = max_degree
                .checked_mul(family.max_exp)
                .ok_or_else(|| format!("{}.max_degree*max_exp overflows the integer ledger", label))?;
            if family.max_sum < required_sum {
                return Err(format!(
                    "{}.max_sum {} is below max_degree*max_exp {}",
                    label, family.max_sum, required_sum
                ));
            }
            let denominator_multiplier = scale_exponent_to_multiplier(projected_scale)?;
            let denominator_multiplier = i128::try_from(denominator_multiplier)
                .map_err(|_| format!("{}.scaled denominator multiplier exceeds i128", label))?;
            let scaled_denominator = family
                .max_sum
                .checked_mul(denominator_multiplier)
                .ok_or_else(|| format!("{}.scaled denominator bound overflows i128", label))?;
            let required_divisor = if is_output {
                let output_alignment_multiplier = scale_exponent_to_multiplier(self.scales.s_out_pre)?;
                let output_alignment_multiplier = i128::try_from(output_alignment_multiplier)
                    .map_err(|_| format!("{}.output alignment multiplier exceeds i128", label))?;
                scaled_denominator.max(output_alignment_multiplier)
            } else {
                scaled_denominator
            };
            if family.max_divisor < required_divisor {
                return Err(format!(
                    "{}.max_divisor {} is below the required scaled divisor {}",
                    label, family.max_divisor, required_divisor
                ));
            }
            Ok(())
        };
        for (layer, spec) in self.model.hidden_layers.iter().enumerate() {
            for head in 0..spec.heads.len() {
                let family = self
                    .ledger
                    .families
                    .iter()
                    .find(|family| { family.layer == layer && family.head == head && !family.is_output})
                    .ok_or_else(|| format!("missing hidden bound family {}:{}", layer, head))?;
                validate_param_bounds(family, &spec.heads[head], false, &format!("hidden {}:{}", layer, head),)?;
                check_aggregation_budget(
                    family,
                    self.scales.s_h_prime[layer][head],
                    false,
                    &format!("hidden {}:{}", layer, head),
                )?;
            }
        }
        let output_layer = usize::MAX;
        for head in 0..self.model.output_heads.len() {
            let family = self
                .ledger
                .families
                .iter()
                .find(|family| { family.layer == output_layer && family.head == head && family.is_output})
                .ok_or_else(|| format!("missing output bound family {}", head))?;
            validate_param_bounds(family, &self.model.output_heads[head], true, &format!("output {}", head),)?;
            check_aggregation_budget(family, self.scales.s_y_prime[head], true, &format!("output {}", head))?;
        }
        Ok(())
    }

    /// Canonical model parameter bytes for an external identity/manifest.
    pub fn canonical_model_bytes(&self) -> Vec<u8> {
        self.model.flattened_params().into_iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// SHA-256 identity of the canonical model parameter bytes.  The native
    /// Hashed module's Poseidon H(W) remains a separate public field.
    pub fn canonical_model_digest(&self) -> Result<String, String> {
        let mut bytes = b"ezkl-gat-model-v1".to_vec();
        bytes.extend_from_slice(&self.canonical_model_bytes());
        sha256_digest(&bytes)
    }

    /// Construct a verifier-facing public instance binding for a canonical
    /// output digest.  This never accepts or reads private witness data.
    pub fn public_instance_binding(&self, feature_digest: String, output_digest: String,) -> Result<PublicInstanceBinding, String> {
        self.validate()?;
        let binding = PublicInstanceBinding {
            topology: self.topology.clone(),
            static_tables: DigestIdentity { id: self.tables.static_tables_id.clone(), digest: self.tables.static_tables_digest.clone(),},
            quant: self.quantization.clone(),
            ledger: self.ledger_identity.clone(),
            circuit: self.circuit_ref.clone().ok_or_else(|| "canonical GAT circuit identity is not bound".to_string())?,
            model: self.model_binding.clone(),
            feature_digest,
            output_digest,
        };
        binding.validate()?;
        Ok(binding)
    }
}

fn validate_param_bounds(family: &BoundFamily, head: &HeadWeights, is_output: bool, label: &str,) -> Result<(), String> {
    let check = |name: &str, values: &[i128], bound: i128| -> Result<(), String> {
        if bound < 0 {
            return Err(format!("{}.{} bound is negative", label, name));
        }
        let bound = bound as u128;
        if values.iter().any(|value| value.unsigned_abs() > bound) {
            return Err(format!("{}.{} exceeds authenticated bound", label, name));
        }
        Ok(())
    };
    check("projection", &head.projection, family.max_weight_abs)?;
    check("a_src", &head.a_src, family.max_attention_abs)?;
    check("a_dst", &head.a_dst, family.max_attention_abs)?;
    if is_output {
        check("bias", &head.bias, family.max_bias_abs)?;
    } else if !head.bias.is_empty() {
        return Err(format!("{} hidden head has a bias", label));
    }
    Ok(())
}
