//! Minimal command-line entry point for the formal graph-attention route.
//!
//! This target deliberately exposes only the six native GAT operations.  The
//! general ezkl CLI (ONNX/Python/EVM/web commands) remains a separate target
//! and is not part of the formal build graph.

use ezkl::native_gat::{
    graph_gat_compile, graph_gat_gen_settings, graph_gat_gen_witness, graph_gat_prove,
    graph_gat_setup, graph_gat_verify,
};
use std::collections::BTreeMap;
use std::env;
use std::path::PathBuf;

fn usage() -> ! {
    eprintln!(
        "usage: ezkl_formal <graph-gat-(gen-settings|compile|setup|gen-witness|prove|verify)> [options]"
    );
    std::process::exit(2);
}

fn options(args: impl Iterator<Item = String>) -> BTreeMap<String, String> {
    let mut parsed = BTreeMap::new();
    let mut args = args.peekable();
    while let Some(option) = args.next() {
        if !option.starts_with("--") || option.len() <= 2 {
            usage();
        }
        let key = option[2..].to_owned();
        let value = args.next().unwrap_or_else(|| usage());
        if value.starts_with("--") || parsed.insert(key.clone(), value).is_some() {
            eprintln!("invalid or duplicate option: --{key}");
            usage();
        }
    }
    parsed
}

fn required(options: &BTreeMap<String, String>, key: &str) -> PathBuf {
    options.get(key).map(PathBuf::from).unwrap_or_else(|| {
        eprintln!("missing required option: --{key}");
        usage();
    })
}

fn optional_u32(options: &BTreeMap<String, String>, key: &str) -> Result<Option<u32>, String> {
    options
        .get(key)
        .map(|value| {
            value
                .parse::<u32>()
                .map(Some)
                .map_err(|_| format!("--{key} must be an unsigned integer"))
        })
        .unwrap_or(Ok(None))
}

fn required_string(options: &BTreeMap<String, String>, key: &str) -> String {
    options.get(key).cloned().unwrap_or_else(|| {
        eprintln!("missing required option: --{key}");
        usage();
    })
}

fn ensure_options(options: &BTreeMap<String, String>, allowed: &[&str]) -> Result<(), String> {
    if let Some(key) = options
        .keys()
        .find(|key| !allowed.iter().any(|candidate| candidate == key))
    {
        return Err(format!("unsupported option for formal command: --{key}"));
    }
    Ok(())
}

fn run(command: &str, options: &BTreeMap<String, String>) -> Result<String, String> {
    match command {
        "graph-gat-gen-settings" => {
            ensure_options(
                options,
                &[
                    "spec-path",
                    "csr-layout-path",
                    "settings-path",
                    "max-logrows",
                ],
            )?;
            Ok(graph_gat_gen_settings(
                required(options, "spec-path"),
                required(options, "csr-layout-path"),
                required(options, "settings-path"),
                optional_u32(options, "max-logrows")?,
            )
            .map_err(|error| error.to_string())?)
        }
        "graph-gat-compile" => {
            ensure_options(
                options,
                &[
                    "spec-path",
                    "csr-layout-path",
                    "settings-path",
                    "compiled-circuit",
                    "max-logrows",
                ],
            )?;
            Ok(graph_gat_compile(
                required(options, "spec-path"),
                required(options, "csr-layout-path"),
                required(options, "settings-path"),
                required(options, "compiled-circuit"),
                optional_u32(options, "max-logrows")?,
            )
            .map_err(|error| error.to_string())?)
        }
        "graph-gat-setup" => {
            ensure_options(
                options,
                &[
                    "compiled-circuit",
                    "vk-path",
                    "pk-path",
                    "srs-path",
                    "device",
                ],
            )?;
            Ok(graph_gat_setup(
                required(options, "compiled-circuit"),
                required(options, "vk-path"),
                required(options, "pk-path"),
                required(options, "srs-path"),
                required_string(options, "device"),
            )
            .map_err(|error| error.to_string())?)
        }
        "graph-gat-gen-witness" => {
            ensure_options(options, &["compiled-circuit", "output", "public-statement"])?;
            Ok(graph_gat_gen_witness(
                required(options, "compiled-circuit"),
                required(options, "output"),
                required(options, "public-statement"),
            )
            .map_err(|error| error.to_string())?)
        }
        "graph-gat-prove" => {
            ensure_options(
                options,
                &[
                    "compiled-circuit",
                    "witness",
                    "public-statement",
                    "pk-path",
                    "proof-path",
                    "device",
                ],
            )?;
            Ok(graph_gat_prove(
                required(options, "compiled-circuit"),
                required(options, "witness"),
                required(options, "public-statement"),
                required(options, "pk-path"),
                required(options, "proof-path"),
                required_string(options, "device"),
            )
            .map_err(|error| error.to_string())?)
        }
        "graph-gat-verify" => {
            ensure_options(
                options,
                &[
                    "compiled-circuit",
                    "proof-path",
                    "public-statement",
                    "settings-path",
                    "vk-path",
                    "output",
                    "device",
                ],
            )?;
            Ok(graph_gat_verify(
                required(options, "compiled-circuit"),
                required(options, "proof-path"),
                required(options, "public-statement"),
                required(options, "settings-path"),
                required(options, "vk-path"),
                required(options, "output"),
                required_string(options, "device"),
            )
            .map_err(|error| error.to_string())?)
        }
        _ => {
            eprintln!("unsupported formal command: {command}");
            usage();
        }
    }
}

fn main() {
    let mut args = env::args().skip(1);
    let command = args.next().unwrap_or_else(|| usage());
    let options = options(args);
    match run(&command, &options) {
        Ok(result) => println!("{result}"),
        Err(error) => {
            eprintln!("formal {command} failed: {error}");
            std::process::exit(1);
        }
    }
}
