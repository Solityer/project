use clap::{Args, Parser, Subcommand};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

#[path = "gat_baseline/manifest.rs"]
mod manifest;
#[path = "gat_baseline/process.rs"]
mod process;

use manifest::{load_dataset_spec, ResolvedDatasetSpec};
use process::{run_command, StepRecord};

#[derive(Parser, Debug)]
#[command(about = "Formal full-dataset GAT baseline pipeline for ezkl")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Export(CommonArgs),
    Pipeline(PipelineArgs),
}

#[derive(Args, Debug, Clone)]
struct CommonArgs {
    #[arg(long, value_parser = ["cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv"])]
    dataset: String,
    #[arg(long, default_value = "python3")]
    python: String,
}

#[derive(Args, Debug, Clone)]
struct PipelineArgs {
    #[command(flatten)]
    common: CommonArgs,
    #[arg(long)]
    skip_build: bool,
    #[arg(long)]
    prove: bool,
    #[arg(long)]
    debug_build: bool,
}

const FORMAL_SETUP_LOGROWS_CAP: u32 = 25;
const FORMAL_EZKL_REPO_DIR: &str = ".ezkl_local";
const FORMAL_NUM_INNER_COL_CANDIDATES: &[usize] = &[2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn json_or_null(path: &Path) -> Value {
    match fs::read_to_string(path) {
        Ok(text) => serde_json::from_str(&text).unwrap_or(Value::Null),
        Err(_) => Value::Null,
    }
}

fn cargo_envs() -> BTreeMap<String, String> {
    let mut envs = BTreeMap::new();
    let root = repo_root();
    let cargo_home = root.join(".cargo_local");
    let rustup_home = root.join(".rustup_local");
    if cargo_home.exists() {
        envs.insert("CARGO_HOME".to_string(), cargo_home.display().to_string());
    }
    if rustup_home.exists() {
        envs.insert("RUSTUP_HOME".to_string(), rustup_home.display().to_string());
    }
    envs
}

fn python_envs() -> BTreeMap<String, String> {
    let mut envs = BTreeMap::new();
    let root = repo_root();
    envs.insert(
        "PYTHONPATH".to_string(),
        format!("{}:.", root.join(".pydeps").display()),
    );
    envs
}

fn cargo_program() -> String {
    let candidate = repo_root().join(".cargo_local/bin/cargo");
    if candidate.exists() {
        candidate.display().to_string()
    } else {
        "cargo".to_string()
    }
}

fn ezkl_build_features() -> &'static str {
    "ezkl,precompute-coset,parallel-poly-read,no-banner"
}

fn ezkl_repo_path() -> PathBuf {
    repo_root().join(FORMAL_EZKL_REPO_DIR)
}

fn ensure_ezkl_local_state() -> Result<(), String> {
    let repo = ezkl_repo_path();
    fs::create_dir_all(repo.join("cache"))
        .map_err(|err| format!("failed to create {}: {err}", repo.join("cache").display()))?;
    fs::create_dir_all(repo.join("srs"))
        .map_err(|err| format!("failed to create {}: {err}", repo.join("srs").display()))?;
    Ok(())
}

fn ezkl_envs() -> Result<BTreeMap<String, String>, String> {
    ensure_ezkl_local_state()?;
    let mut envs = cargo_envs();
    envs.insert(
        "EZKL_REPO_PATH".to_string(),
        ezkl_repo_path().display().to_string(),
    );
    Ok(envs)
}

fn ezkl_binary_path(debug_build: bool) -> PathBuf {
    if debug_build {
        repo_root().join("target/debug/ezkl")
    } else {
        repo_root().join("target/release/ezkl")
    }
}

#[derive(Debug, Deserialize)]
struct StageManifest {
    #[serde(default)]
    public_io: Option<PublicIoPlan>,
    node_count: usize,
    edge_count: usize,
    graph_count: usize,
    num_classes: usize,
    #[serde(default)]
    hidden_layers: Vec<HiddenLayerSpec>,
}

#[derive(Debug, Deserialize)]
struct HiddenLayerSpec {
    input_dim: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct PublicIoPlan {
    public_input_count: usize,
    public_output_count: usize,
    total_public_instances: usize,
    safety_margin: usize,
    #[serde(alias = "minimum_logrows")]
    minimum_public_logrows: u32,
}

#[derive(Debug, Deserialize)]
struct SettingsFile {
    #[serde(default)]
    num_rows: Option<u64>,
    #[serde(default)]
    total_assignments: Option<u64>,
    #[serde(default)]
    num_blinding_factors: Option<u64>,
    #[serde(default)]
    run_args: Option<SettingsRunArgs>,
}

#[derive(Debug, Deserialize)]
struct SettingsRunArgs {
    #[serde(default)]
    logrows: Option<u32>,
    #[serde(default)]
    num_inner_cols: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct SettingsSelectionAttempt {
    phase: String,
    requested_logrows: u32,
    requested_num_inner_cols: usize,
    configured_logrows: Option<u32>,
    configured_num_inner_cols: Option<usize>,
    public_min_logrows: u32,
    row_min_logrows: Option<u32>,
    num_rows: Option<u64>,
    total_assignments: Option<u64>,
    num_blinding_factors: Option<u64>,
    required_logrows: u32,
    fits_target_logrows: bool,
}

#[derive(Debug, Serialize)]
struct SettingsSelectionSummary {
    dataset: String,
    settings_path: String,
    calibration_enabled: bool,
    selection_rule: String,
    public_io: Option<PublicIoPlan>,
    setup_logrows_cap: u32,
    target_logrows: Option<u32>,
    attempts: Vec<SettingsSelectionAttempt>,
    final_logrows: Option<u32>,
    final_num_inner_cols: Option<usize>,
    failure_reason: Option<String>,
}

#[derive(Debug, Clone)]
struct SelectedSettings {
    logrows: u32,
    num_inner_cols: usize,
}

#[derive(Debug)]
struct SettingsSelectionResult {
    steps: Vec<StepRecord>,
    selected: SelectedSettings,
}

#[derive(Debug)]
struct PipelineExecution {
    steps: Vec<StepRecord>,
    error: Option<String>,
}

fn ceil_log2(mut value: u64) -> u32 {
    if value <= 1 {
        return 0;
    }

    value -= 1;
    let mut logrows = 0;
    while value > 0 {
        value >>= 1;
        logrows += 1;
    }
    logrows
}

fn load_stage_manifest(spec: &ResolvedDatasetSpec) -> Result<StageManifest, String> {
    let manifest_path = spec.output_root.join("staged/stage_manifest.json");
    let text = fs::read_to_string(&manifest_path)
        .map_err(|err| format!("failed to read {}: {err}", manifest_path.display()))?;
    serde_json::from_str(&text)
        .map_err(|err| format!("failed to parse {}: {err}", manifest_path.display()))
}

fn minimum_public_logrows(spec: &ResolvedDatasetSpec) -> Result<u32, String> {
    let manifest_path = spec.output_root.join("staged/stage_manifest.json");
    let manifest = load_stage_manifest(spec)?;
    if let Some(public_io) = &manifest.public_io {
        return Ok(u32::max(17, public_io.minimum_public_logrows));
    }

    let first_hidden = manifest
        .hidden_layers
        .first()
        .ok_or_else(|| format!("{} does not declare hidden_layers", manifest_path.display()))?;

    let public_input_count = (manifest.node_count * first_hidden.input_dim)
        + (2 * manifest.edge_count)
        + (2 * (manifest.graph_count + 1));
    let public_output_count = (manifest.node_count * manifest.num_classes) + 4;
    let safety_margin = 64usize;
    let total_public_instances = public_input_count + public_output_count + safety_margin;

    Ok(u32::max(17, ceil_log2(total_public_instances as u64)))
}

fn run_export(spec: &ResolvedDatasetSpec, args: &CommonArgs) -> Result<StepRecord, String> {
    let log_path = spec.output_root.join("export.log");
    let argv = vec![
        "-m".to_string(),
        "scripts.gat_baseline.cli".to_string(),
        "--dataset".to_string(),
        args.dataset.clone(),
    ];
    run_command(
        "export",
        &args.python,
        &argv,
        &python_envs(),
        &repo_root(),
        &log_path,
    )
}

fn maybe_build_ezkl(debug_build: bool, output_root: &Path) -> Result<Option<StepRecord>, String> {
    if ezkl_binary_path(debug_build).exists() {
        return Ok(None);
    }
    let cargo = cargo_program();
    let mut argv = vec![
        "build".to_string(),
        "--bin".to_string(),
        "ezkl".to_string(),
        "--bin".to_string(),
        "gat_baseline".to_string(),
        "--no-default-features".to_string(),
        "--features".to_string(),
        ezkl_build_features().to_string(),
    ];
    if !debug_build {
        argv.push("--release".to_string());
    }
    let step = run_command(
        "build-ezkl",
        &cargo,
        &argv,
        &cargo_envs(),
        &repo_root(),
        &output_root.join("build-ezkl.log"),
    )?;
    Ok(Some(step))
}

fn ezkl_step(
    name: &str,
    debug_build: bool,
    output_root: &Path,
    argv: &[String],
) -> Result<StepRecord, String> {
    ezkl_step_with_log_name(name, name, debug_build, output_root, argv)
}

fn ezkl_step_with_log_name(
    name: &str,
    log_name: &str,
    debug_build: bool,
    output_root: &Path,
    argv: &[String],
) -> Result<StepRecord, String> {
    let binary = ezkl_binary_path(debug_build);
    let envs = ezkl_envs()?;
    run_command(
        name,
        &binary.display().to_string(),
        argv,
        &envs,
        output_root,
        &output_root.join(format!("{log_name}.log")),
    )
}

fn row_logrows_bound(num_rows: u64, num_blinding_factors: u64) -> u32 {
    let required_rows = num_rows
        .saturating_add(num_blinding_factors)
        .saturating_add(64);
    ceil_log2(required_rows)
}

fn inspect_settings(
    settings_path: &Path,
    phase: &str,
    requested_logrows: u32,
    requested_num_inner_cols: usize,
    public_min_logrows: u32,
    target_logrows: u32,
) -> Result<SettingsSelectionAttempt, String> {
    let text = fs::read_to_string(settings_path)
        .map_err(|err| format!("failed to read {}: {err}", settings_path.display()))?;
    let settings: SettingsFile = serde_json::from_str(&text)
        .map_err(|err| format!("failed to parse {}: {err}", settings_path.display()))?;

    let configured_logrows = settings
        .run_args
        .as_ref()
        .and_then(|run_args| run_args.logrows);
    let configured_num_inner_cols = settings
        .run_args
        .as_ref()
        .and_then(|run_args| run_args.num_inner_cols);
    let row_min_logrows = settings
        .num_rows
        .map(|num_rows| row_logrows_bound(num_rows, settings.num_blinding_factors.unwrap_or(0)));
    let required_logrows = row_min_logrows
        .map(|row_min| row_min.max(public_min_logrows))
        .unwrap_or(public_min_logrows);
    let fits_target_logrows = required_logrows <= target_logrows;

    Ok(SettingsSelectionAttempt {
        phase: phase.to_string(),
        requested_logrows,
        requested_num_inner_cols,
        configured_logrows,
        configured_num_inner_cols,
        public_min_logrows,
        row_min_logrows,
        num_rows: settings.num_rows,
        total_assignments: settings.total_assignments,
        num_blinding_factors: settings.num_blinding_factors,
        required_logrows,
        fits_target_logrows,
    })
}

fn formal_target_logrows(public_min_logrows: u32) -> Result<u32, String> {
    if public_min_logrows > FORMAL_SETUP_LOGROWS_CAP {
        return Err(format!(
            "formal public instances require logrows {public_min_logrows}, exceeding the local Halo2/KZG setup cap {}",
            FORMAL_SETUP_LOGROWS_CAP
        ));
    }
    Ok(FORMAL_SETUP_LOGROWS_CAP)
}

fn write_settings_selection(
    spec: &ResolvedDatasetSpec,
    public_io: Option<PublicIoPlan>,
    target_logrows: Option<u32>,
    attempts: &[SettingsSelectionAttempt],
    selected: Option<&SelectedSettings>,
    failure_reason: Option<String>,
) -> Result<(), String> {
    let summary = SettingsSelectionSummary {
        dataset: spec.dataset.clone(),
        settings_path: spec.output_root.join("settings.json").display().to_string(),
        calibration_enabled: false,
        selection_rule: format!(
            "formal proving route fixes public/public/fixed visibility, baseline-local EZKL cache, baseline-local exact-k SRS, and no mock on the proving path; proving logrows is fixed to {} when public_io.minimum_public_logrows <= {}; first probe the largest num_inner_cols candidate to fast-reject blocked datasets, otherwise choose the smallest num_inner_cols in {:?} whose row bound fits that target",
            FORMAL_SETUP_LOGROWS_CAP,
            FORMAL_SETUP_LOGROWS_CAP,
            FORMAL_NUM_INNER_COL_CANDIDATES
        ),
        public_io,
        setup_logrows_cap: FORMAL_SETUP_LOGROWS_CAP,
        target_logrows,
        attempts: attempts.to_vec(),
        final_logrows: selected.map(|value| value.logrows),
        final_num_inner_cols: selected.map(|value| value.num_inner_cols),
        failure_reason,
    };
    let path = spec.output_root.join("settings_selection.json");
    fs::write(
        path,
        serde_json::to_string_pretty(&summary).map_err(|err| err.to_string())?,
    )
    .map_err(|err| err.to_string())
}

fn run_settings_attempt(
    spec: &ResolvedDatasetSpec,
    args: &PipelineArgs,
    settings: &Path,
    public_min_logrows: u32,
    target_logrows: u32,
    num_inner_cols: usize,
) -> Result<(StepRecord, SettingsSelectionAttempt, PathBuf), String> {
    let gen_log_name = format!("gen-settings-k{target_logrows}-n{num_inner_cols}");
    let gen_log_path = spec.output_root.join(format!("{gen_log_name}.log"));
    let step = ezkl_step_with_log_name(
        "gen-settings",
        &gen_log_name,
        args.debug_build,
        &spec.output_root,
        &[
            "gen-settings".to_string(),
            "-M".to_string(),
            spec.output_root.join("network.onnx").display().to_string(),
            "-O".to_string(),
            settings.display().to_string(),
            "-K".to_string(),
            target_logrows.to_string(),
            "-N".to_string(),
            num_inner_cols.to_string(),
            "--input-visibility".to_string(),
            "public".to_string(),
            "--output-visibility".to_string(),
            "public".to_string(),
            "--param-visibility".to_string(),
            "fixed".to_string(),
        ],
    )?;
    let selection = inspect_settings(
        settings,
        "gen-settings",
        target_logrows,
        num_inner_cols,
        public_min_logrows,
        target_logrows,
    )?;
    Ok((step, selection, gen_log_path))
}

fn copy_log_if_present(source: &Path, destination: &Path) -> Result<(), String> {
    if source.exists() {
        fs::copy(source, destination).map_err(|err| {
            format!(
                "failed to copy {} to {}: {err}",
                source.display(),
                destination.display()
            )
        })?;
    }
    Ok(())
}

fn canonical_gen_settings_step(step: &StepRecord, output_root: &Path) -> StepRecord {
    let mut canonical = step.clone();
    canonical.log_path = output_root.join("gen-settings.log").display().to_string();
    canonical
}

fn remove_file_if_present(path: &Path) -> Result<(), String> {
    if path.exists() {
        fs::remove_file(path)
            .map_err(|err| format!("failed to remove {}: {err}", path.display()))?;
    }
    Ok(())
}

fn prune_attempt_logs(output_root: &Path) -> Result<(), String> {
    for entry in fs::read_dir(output_root)
        .map_err(|err| format!("failed to list {}: {err}", output_root.display()))?
    {
        let entry = entry.map_err(|err| err.to_string())?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        let matches_attempt = name.starts_with("gen-settings-k");
        if matches_attempt {
            remove_file_if_present(&path)?;
        }
    }
    Ok(())
}

fn adopt_legacy_srs(
    output_root: &Path,
    logrows: u32,
    formal_srs_path: &Path,
) -> Result<(), String> {
    if formal_srs_path.exists() {
        return Ok(());
    }
    let legacy_path = output_root.join(format!("kzg{logrows}_local.srs"));
    if legacy_path.exists() {
        fs::rename(&legacy_path, formal_srs_path).map_err(|err| {
            format!(
                "failed to move legacy SRS {} to {}: {err}",
                legacy_path.display(),
                formal_srs_path.display()
            )
        })?;
    }
    Ok(())
}

fn select_settings(
    spec: &ResolvedDatasetSpec,
    args: &PipelineArgs,
    settings: &Path,
) -> Result<SettingsSelectionResult, String> {
    let stage_manifest = load_stage_manifest(spec)?;
    let public_min_logrows = minimum_public_logrows(spec)?;
    let public_io = stage_manifest.public_io.clone();
    let target_logrows = match formal_target_logrows(public_min_logrows) {
        Ok(value) => value,
        Err(err) => {
            write_settings_selection(spec, public_io, None, &[], None, Some(err.clone()))?;
            return Err(err);
        }
    };
    let mut attempts = Vec::new();
    let max_num_inner_cols = *FORMAL_NUM_INNER_COL_CANDIDATES
        .last()
        .ok_or_else(|| "FORMAL_NUM_INNER_COL_CANDIDATES is empty".to_string())?;
    let (max_step, max_attempt, max_log_path) = run_settings_attempt(
        spec,
        args,
        settings,
        public_min_logrows,
        target_logrows,
        max_num_inner_cols,
    )?;
    let max_selected_num_inner_cols = max_attempt
        .configured_num_inner_cols
        .unwrap_or(max_num_inner_cols);
    let max_fits_target_logrows = max_attempt.fits_target_logrows;
    let max_required_logrows = max_attempt.required_logrows;
    let max_num_rows = max_attempt.num_rows;
    let max_total_assignments = max_attempt.total_assignments;
    attempts.push(max_attempt);

    if !max_fits_target_logrows {
        let failure_reason = format!(
            "blocked: max num_inner_cols {} still requires logrows {} at k={} (num_rows={}, total_assignments={})",
            max_selected_num_inner_cols,
            max_required_logrows,
            target_logrows,
            max_num_rows
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            max_total_assignments
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        );
        write_settings_selection(
            spec,
            public_io,
            Some(target_logrows),
            &attempts,
            None,
            Some(failure_reason.clone()),
        )?;
        return Err(failure_reason);
    }

    for &num_inner_cols in FORMAL_NUM_INNER_COL_CANDIDATES {
        if num_inner_cols == max_num_inner_cols {
            continue;
        }
        let (step, selection, gen_log_path) = run_settings_attempt(
            spec,
            args,
            settings,
            public_min_logrows,
            target_logrows,
            num_inner_cols,
        )?;
        let selected_num_inner_cols = selection
            .configured_num_inner_cols
            .unwrap_or(num_inner_cols);
        let fits_target_logrows = selection.fits_target_logrows;
        attempts.push(selection);

        if !fits_target_logrows {
            continue;
        }

        let selected = SelectedSettings {
            logrows: target_logrows,
            num_inner_cols: selected_num_inner_cols,
        };
        write_settings_selection(
            spec,
            public_io,
            Some(target_logrows),
            &attempts,
            Some(&selected),
            None,
        )?;
        copy_log_if_present(&gen_log_path, &spec.output_root.join("gen-settings.log"))?;
        remove_file_if_present(&spec.output_root.join("calibrate-settings.log"))?;
        prune_attempt_logs(&spec.output_root)?;
        return Ok(SettingsSelectionResult {
            steps: vec![canonical_gen_settings_step(&step, &spec.output_root)],
            selected,
        });
    }

    let selected = SelectedSettings {
        logrows: target_logrows,
        num_inner_cols: max_selected_num_inner_cols,
    };
    write_settings_selection(
        spec,
        public_io,
        Some(target_logrows),
        &attempts,
        Some(&selected),
        None,
    )?;
    copy_log_if_present(&max_log_path, &spec.output_root.join("gen-settings.log"))?;
    remove_file_if_present(&spec.output_root.join("calibrate-settings.log"))?;
    prune_attempt_logs(&spec.output_root)?;
    Ok(SettingsSelectionResult {
        steps: vec![canonical_gen_settings_step(&max_step, &spec.output_root)],
        selected,
    })
}

fn pipeline(spec: &ResolvedDatasetSpec, args: &PipelineArgs) -> PipelineExecution {
    let mut steps = Vec::new();
    let export_step = match run_export(spec, &args.common) {
        Ok(step) => step,
        Err(err) => {
            return PipelineExecution {
                steps,
                error: Some(err),
            }
        }
    };
    steps.push(export_step);
    if !args.skip_build {
        let build_step = match maybe_build_ezkl(args.debug_build, &spec.output_root) {
            Ok(step) => step,
            Err(err) => {
                return PipelineExecution {
                    steps,
                    error: Some(err),
                }
            }
        };
        if let Some(step) = build_step {
            steps.push(step);
        }
    }

    let network = spec.output_root.join("network.onnx");
    let input = spec.output_root.join("input.json");
    let settings = spec.output_root.join("settings.json");
    let compiled = spec.output_root.join("model.compiled");
    let witness = spec.output_root.join("witness.json");
    let pk = spec.output_root.join("pk.key");
    let vk = spec.output_root.join("vk.key");
    let proof = spec.output_root.join("proof.json");

    let selection = match select_settings(spec, args, &settings) {
        Ok(selection) => selection,
        Err(err) => {
            return PipelineExecution {
                steps,
                error: Some(err),
            }
        }
    };
    let selected_settings = selection.selected;
    let srs = spec
        .output_root
        .join(format!("kzg{}.srs", selected_settings.logrows));
    steps.extend(selection.steps);

    let compile_step = match ezkl_step(
        "compile-circuit",
        args.debug_build,
        &spec.output_root,
        &[
            "compile-circuit".to_string(),
            "-M".to_string(),
            network.display().to_string(),
            "--compiled-circuit".to_string(),
            compiled.display().to_string(),
            "-S".to_string(),
            settings.display().to_string(),
        ],
    ) {
        Ok(step) => step,
        Err(err) => {
            return PipelineExecution {
                steps,
                error: Some(err),
            }
        }
    };
    steps.push(compile_step);

    let witness_step = match ezkl_step(
        "gen-witness",
        args.debug_build,
        &spec.output_root,
        &[
            "gen-witness".to_string(),
            "-D".to_string(),
            format!("@{}", input.display()),
            "-M".to_string(),
            compiled.display().to_string(),
            "-O".to_string(),
            witness.display().to_string(),
        ],
    ) {
        Ok(step) => step,
        Err(err) => {
            return PipelineExecution {
                steps,
                error: Some(err),
            }
        }
    };
    steps.push(witness_step);

    if args.prove {
        if let Err(err) = adopt_legacy_srs(&spec.output_root, selected_settings.logrows, &srs) {
            return PipelineExecution {
                steps,
                error: Some(err),
            };
        }
        if !srs.exists() {
            let gen_srs_step = match ezkl_step(
                "gen-srs",
                args.debug_build,
                &spec.output_root,
                &[
                    "gen-srs".to_string(),
                    "--srs-path".to_string(),
                    srs.display().to_string(),
                    "--logrows".to_string(),
                    selected_settings.logrows.to_string(),
                ],
            ) {
                Ok(step) => step,
                Err(err) => {
                    return PipelineExecution {
                        steps,
                        error: Some(err),
                    }
                }
            };
            steps.push(gen_srs_step);
        }
        let setup_step = match ezkl_step(
            "setup",
            args.debug_build,
            &spec.output_root,
            &[
                "setup".to_string(),
                "-M".to_string(),
                compiled.display().to_string(),
                "-W".to_string(),
                witness.display().to_string(),
                "--srs-path".to_string(),
                srs.display().to_string(),
                "--vk-path".to_string(),
                vk.display().to_string(),
                "--pk-path".to_string(),
                pk.display().to_string(),
            ],
        ) {
            Ok(step) => step,
            Err(err) => {
                return PipelineExecution {
                    steps,
                    error: Some(err),
                }
            }
        };
        steps.push(setup_step);
        let prove_step = match ezkl_step(
            "prove",
            args.debug_build,
            &spec.output_root,
            &[
                "prove".to_string(),
                "-W".to_string(),
                witness.display().to_string(),
                "-M".to_string(),
                compiled.display().to_string(),
                "--pk-path".to_string(),
                pk.display().to_string(),
                "--proof-path".to_string(),
                proof.display().to_string(),
                "--srs-path".to_string(),
                srs.display().to_string(),
                "--check-mode".to_string(),
                "safe".to_string(),
            ],
        ) {
            Ok(step) => step,
            Err(err) => {
                return PipelineExecution {
                    steps,
                    error: Some(err),
                }
            }
        };
        steps.push(prove_step);
        let verify_step = match ezkl_step(
            "verify",
            args.debug_build,
            &spec.output_root,
            &[
                "verify".to_string(),
                "-S".to_string(),
                settings.display().to_string(),
                "--proof-path".to_string(),
                proof.display().to_string(),
                "--vk-path".to_string(),
                vk.display().to_string(),
                "--srs-path".to_string(),
                srs.display().to_string(),
            ],
        ) {
            Ok(step) => step,
            Err(err) => {
                return PipelineExecution {
                    steps,
                    error: Some(err),
                }
            }
        };
        steps.push(verify_step);
    }

    PipelineExecution { steps, error: None }
}

fn write_pipeline_summary(
    spec: &ResolvedDatasetSpec,
    steps: &[StepRecord],
    prove_requested: bool,
    pipeline_error: Option<&str>,
) -> Result<(), String> {
    let reference = json_or_null(&spec.reference_run_manifest_path);
    let model_info = json_or_null(&spec.output_root.join("model_info.json"));
    let validation = json_or_null(&spec.output_root.join("validation.json"));
    let settings_selection = json_or_null(&spec.output_root.join("settings_selection.json"));
    let formal_logrows = settings_selection
        .get("final_logrows")
        .cloned()
        .unwrap_or_else(|| {
            json_or_null(&spec.output_root.join("settings.json"))
                .get("run_args")
                .and_then(|run_args| run_args.get("logrows"))
                .cloned()
                .unwrap_or(Value::Null)
        });
    let summary = json!({
        "dataset": spec.dataset,
        "output_root": spec.output_root,
        "prove_requested": prove_requested,
        "pipeline_error": pipeline_error,
        "formal_logrows": formal_logrows,
        "formal_route": {
            "input_visibility": "public",
            "output_visibility": "public",
            "param_visibility": "fixed",
            "setup_logrows_cap": FORMAL_SETUP_LOGROWS_CAP,
            "num_inner_cols_candidates": FORMAL_NUM_INNER_COL_CANDIDATES,
            "ezkl_repo_path": ezkl_repo_path(),
            "srs_mode": "baseline_local_exact_k_gen_srs",
            "mock_in_proving_path": false
        },
        "reference": {
            "prove_time_ms": reference.get("prove_time_ms").cloned().unwrap_or(Value::Null),
            "verify_time_ms": reference.get("verify_time_ms").cloned().unwrap_or(Value::Null),
            "proof_size_bytes": reference.get("proof_size_bytes").cloned().unwrap_or(Value::Null),
            "node_count": reference.get("node_count").cloned().unwrap_or(Value::Null),
            "edge_count": reference.get("edge_count").cloned().unwrap_or(Value::Null)
        },
        "model_info": model_info,
        "validation": validation,
        "settings_selection": settings_selection,
        "steps": steps
    });
    let summary_path = spec.output_root.join("pipeline_summary.json");
    fs::write(
        summary_path,
        serde_json::to_string_pretty(&summary).map_err(|err| err.to_string())?,
    )
    .map_err(|err| err.to_string())
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Export(args) => {
            let spec = load_dataset_spec(&args.dataset).and_then(|spec| {
                let step = run_export(&spec, &args)?;
                write_pipeline_summary(&spec, &[step], false, None)?;
                Ok(spec)
            });
            spec.map(|_| ())
        }
        Command::Pipeline(args) => {
            let spec = load_dataset_spec(&args.common.dataset);
            spec.and_then(|spec| {
                let execution = pipeline(&spec, &args);
                write_pipeline_summary(
                    &spec,
                    &execution.steps,
                    args.prove,
                    execution.error.as_deref(),
                )?;
                if let Some(err) = execution.error {
                    Err(err)
                } else {
                    Ok(())
                }
            })
        }
    };

    if let Err(err) = result {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
