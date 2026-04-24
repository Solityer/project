use serde::Serialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

#[derive(Debug, Clone, Serialize)]
pub struct StepRecord {
    pub name: String,
    pub command: Vec<String>,
    pub duration_ms: u128,
    pub log_path: String,
}

pub fn run_command(
    name: &str,
    program: &str,
    argv: &[String],
    envs: &BTreeMap<String, String>,
    workdir: &Path,
    log_path: &Path,
) -> Result<StepRecord, String> {
    if let Some(parent) = log_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }

    let mut command = Command::new(program);
    command.args(argv).current_dir(workdir);
    for (key, value) in envs {
        command.env(key, value);
    }

    let started = Instant::now();
    let output = command
        .output()
        .map_err(|err| format!("failed to start {name}: {err}"))?;
    let duration_ms = started.elapsed().as_millis();

    let mut log = String::new();
    log.push_str(&format!("$ {} {}\n", program, argv.join(" ")));
    if !output.stdout.is_empty() {
        log.push_str("--- stdout ---\n");
        log.push_str(&String::from_utf8_lossy(&output.stdout));
        if !log.ends_with('\n') {
            log.push('\n');
        }
    }
    if !output.stderr.is_empty() {
        log.push_str("--- stderr ---\n");
        log.push_str(&String::from_utf8_lossy(&output.stderr));
        if !log.ends_with('\n') {
            log.push('\n');
        }
    }
    fs::write(log_path, log)
        .map_err(|err| format!("failed to write {}: {err}", log_path.display()))?;

    if !output.status.success() {
        return Err(format!(
            "{name} failed with status {} (see {})",
            output
                .status
                .code()
                .map(|code| code.to_string())
                .unwrap_or_else(|| "signal".to_string()),
            log_path.display()
        ));
    }

    Ok(StepRecord {
        name: name.to_string(),
        command: std::iter::once(program.to_string())
            .chain(argv.iter().cloned())
            .collect(),
        duration_ms,
        log_path: log_path.display().to_string(),
    })
}
