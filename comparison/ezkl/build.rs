use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

const FORMAL_SOURCE_ROOTS: &[&str] = &[
    "src",
    "cpp/gat_ezkl_adapter",
    "configs",
    "scripts",
    "Cargo.lock",
    "Cargo.toml",
    "build.rs",
    "rust-toolchain",
    "vendor",
];

const EXCLUDED_SOURCE_NAMES: &[&str] = &[".git", ".cargo-ok"];

fn formal_source_files(root: &Path) -> Vec<PathBuf> {
    checked_directory(root, "formal baseline root");
    let mut files = Vec::new();
    for relative in FORMAL_SOURCE_ROOTS {
        let path = lexical_source_path(root, relative);
        let present = match fs::symlink_metadata(&path) {
            Ok(_) => true,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
            Err(error) => panic!(
                "cannot inspect formal source root {}: {error}",
                path.display()
            ),
        };
        if !present {
            if *relative == "vendor/mopro-msm" {
                continue;
            }
            panic!("missing formal source root: {}", path.display());
        }
        let metadata = checked_metadata(&path, "formal source root");
        if metadata.is_file() {
            files.push(path);
        } else if metadata.is_dir() {
            collect_files(root, &path, &mut files);
        } else {
            panic!("unsupported formal source entry: {}", path.display());
        }
    }
    files.sort_by(|left, right| {
        left.strip_prefix(root)
            .unwrap()
            .cmp(right.strip_prefix(root).unwrap())
    });
    files.dedup();
    files
}

fn formal_source_id(root: &Path) -> (String, Vec<PathBuf>) {
    let files = formal_source_files(root);
    let mut digest = Sha256::new();
    for path in &files {
        let relative = path
            .strip_prefix(root)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        let bytes = fs::read(path)
            .unwrap_or_else(|_| panic!("cannot read formal source: {}", path.display()));
        digest.update((relative.len() as u64).to_le_bytes());
        digest.update(relative.as_bytes());
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
    }
    (format!("sha256:{:x}", digest.finalize()), files)
}

fn digest_files(root: &Path, files: &mut Vec<PathBuf>) -> String {
    files.sort();
    let mut digest = Sha256::new();
    for path in files.iter() {
        let relative = path
            .strip_prefix(root)
            .expect("source path must be inside the baseline")
            .to_string_lossy()
            .replace('\\', "/");
        let bytes =
            fs::read(path).unwrap_or_else(|_| panic!("cannot read source: {}", path.display()));
        digest.update((relative.len() as u64).to_le_bytes());
        digest.update(relative.as_bytes());
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
    }
    format!("sha256:{:x}", digest.finalize())
}

fn collect_files(root: &Path, current: &Path, out: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(current)
        .unwrap_or_else(|_| panic!("cannot read vendor directory: {}", current.display()));
    for entry in entries {
        let entry = entry.expect("cannot inspect vendor directory entry");
        let path = entry.path();
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| EXCLUDED_SOURCE_NAMES.contains(&name))
        {
            continue;
        }
        let metadata = checked_metadata(&path, "formal source tree entry");
        if metadata.is_dir() {
            collect_files(root, &path, out);
        } else if metadata.is_file() {
            out.push(path);
        } else {
            panic!("vendor tree contains unsupported entry: {}", path.display());
        }
    }
}

fn checked_metadata(path: &Path, label: &str) -> fs::Metadata {
    let metadata = fs::symlink_metadata(path)
        .unwrap_or_else(|_| panic!("cannot inspect {label}: {}", path.display()));
    if metadata.file_type().is_symlink() {
        panic!("{label} contains a symlink: {}", path.display());
    }
    if metadata.file_type().is_file() {
        if metadata.nlink() != 1 {
            panic!(
                "{label} contains a hard-linked regular file: {}",
                path.display()
            );
        }
    } else if !metadata.file_type().is_dir() {
        panic!(
            "{label} contains a special filesystem entry: {}",
            path.display()
        );
    }
    metadata
}

fn checked_directory(path: &Path, label: &str) {
    let metadata = checked_metadata(path, label);
    if !metadata.is_dir() {
        panic!("{label} is not a directory: {}", path.display());
    }
}

fn lexical_source_path(root: &Path, relative: &str) -> PathBuf {
    let mut current = root.to_path_buf();
    checked_directory(root, "formal baseline root");
    for component in Path::new(relative).components() {
        let name = match component {
            std::path::Component::Normal(name) => name,
            _ => panic!("formal source root contains an unsafe lexical component: {relative}"),
        };
        current.push(name);
        // Inspect the lexical component before any canonicalization.  Missing
        // optional roots are handled by formal_source_files.
        match fs::symlink_metadata(&current) {
            Ok(_) => {
                checked_metadata(&current, "formal source path");
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => panic!(
                "cannot inspect formal source path {}: {error}",
                current.display()
            ),
        }
    }
    current
}

fn quoted_value(line: &str, key: &str) -> Option<String> {
    let marker = format!("{key} = \"");
    let start = line.find(&marker)? + marker.len();
    let rest = &line[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_owned())
}

fn validate_git_selector(source: &str, origin: &Path) {
    if source.contains("branch=")
        || source.contains("tag=")
        || source.contains("?branch")
        || source.contains("?tag")
    {
        panic!(
            "floating branch/tag git selector is forbidden in {}: {source}",
            origin.display()
        );
    }
    if let Some(query) = source.split_once('?').map(|(_, value)| value) {
        let query = query.split('#').next().unwrap_or(query);
        let valid_revision = query.len() == 44
            && query.starts_with("rev=")
            && query[4..].chars().all(|ch| ch.is_ascii_hexdigit());
        if !valid_revision {
            panic!(
                "non-revision git query is forbidden in {}: {source}",
                origin.display()
            );
        }
    }
}

fn validate_path_selector(manifest: &Path, value: &str, vendor_root: &Path) {
    let path = manifest
        .parent()
        .expect("manifest must have a parent")
        .join(value);
    let canonical = path
        .canonicalize()
        .unwrap_or_else(|_| panic!("path dependency does not exist: {}", path.display()));
    if !canonical.starts_with(vendor_root) {
        panic!(
            "path dependency escapes authenticated vendor tree: {} -> {}",
            manifest.display(),
            value
        );
    }
}

fn validate_dependency_sources(root: &Path) -> String {
    let vendor_root = root
        .join("vendor")
        .canonicalize()
        .expect("baseline vendor directory is missing");
    let mut manifests = vec![root.join("Cargo.toml")];
    let mut vendor_files = Vec::new();
    collect_files(root, &vendor_root, &mut vendor_files);
    manifests.extend(
        vendor_files
            .iter()
            .filter(|path| path.file_name().is_some_and(|name| name == "Cargo.toml"))
            .cloned(),
    );

    for manifest in manifests {
        let text = fs::read_to_string(&manifest)
            .unwrap_or_else(|_| panic!("cannot read manifest: {}", manifest.display()));
        for line in text.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("branch =") || trimmed.starts_with("tag =") {
                panic!(
                    "branch/tag dependency selector is forbidden in {}: {line}",
                    manifest.display()
                );
            }
            if let Some(source) = quoted_value(line, "git") {
                validate_git_selector(&source, &manifest);
            }
            if let Some(path) = quoted_value(line, "path") {
                validate_path_selector(&manifest, &path, &vendor_root);
            }
        }
    }

    let lock = root.join("Cargo.lock");
    let lock_text = fs::read_to_string(&lock).expect("Cargo.lock is missing");
    for line in lock_text.lines() {
        if let Some(source) = quoted_value(line, "source") {
            if source.starts_with("git+") {
                validate_git_selector(&source, &lock);
            }
        }
    }
    digest_files(root, &mut vendor_files)
}

fn main() {
    let root =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is missing"));
    let vendor_tree_id = validate_dependency_sources(&root);
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/native_gat.rs");
    println!("cargo::rerun-if-changed=configs/formal_operation_registry.json");
    let (source_id, source_files) = formal_source_id(&root);
    let manifest_path = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is missing"))
        .join("formal_source_manifest.txt");
    let manifest = source_files
        .iter()
        .map(|path| {
            path.strip_prefix(&root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(&manifest_path, format!("{manifest}\n"))
        .expect("cannot write formal source manifest");
    println!(
        "cargo::warning=ezkl formal source inventory: {} files, {source_id}",
        source_files.len()
    );
    for path in &source_files {
        let relative = path.strip_prefix(&root).unwrap();
        println!("cargo::rerun-if-changed={}", relative.display());
    }
    println!("cargo::rerun-if-changed=Cargo.toml");
    println!("cargo::rerun-if-changed=Cargo.lock");
    println!("cargo::rerun-if-changed=vendor");
    println!("cargo::rerun-if-env-changed=EZKL_CUPTI_ROOT");
    println!("cargo::rerun-if-env-changed=CUDA_HOME");
    println!("cargo::rerun-if-env-changed=CUDA_PATH");
    println!("cargo::rustc-env=EZKL_FORMAL_SOURCE_ID={source_id}");
    println!("cargo::rustc-env=EZKL_VENDOR_TREE_ID={vendor_tree_id}");

    if env::var_os("CARGO_FEATURE_GPU_ACCELERATED").is_some() {
        let (include, library) = cupti_layout().unwrap_or_else(|| {
            panic!(
                "gpu-accelerated build requires CUPTI headers and libcupti.so; set EZKL_CUPTI_ROOT"
            )
        });
        cc::Build::new()
            .file("src/gpu_telemetry.c")
            .include(&include)
            .flag_if_supported("-std=c11")
            .compile("ezkl_gpu_telemetry");
        println!("cargo::rustc-link-search=native={}", library.display());
        println!("cargo::rustc-link-lib=dylib=cupti");
        println!("cargo::rustc-link-lib=dylib=cuda");
        println!("cargo::rustc-link-lib=dylib=dl");
        println!("cargo::rustc-link-lib=dylib=pthread");
        println!(
            "cargo::rustc-link-arg-bins=-Wl,-rpath,{}",
            library.display()
        );
        println!(
            "cargo::rustc-link-arg-cdylib=-Wl,-rpath,{}",
            library.display()
        );
        println!("cargo::rustc-env=EZKL_CUPTI_CONTRACT=cupti_activity_kernel_scoped_v1");
    }
}

fn cupti_layout() -> Option<(PathBuf, PathBuf)> {
    let mut roots = Vec::new();
    for variable in ["EZKL_CUPTI_ROOT", "CUDA_HOME", "CUDA_PATH"] {
        if let Some(value) = env::var_os(variable) {
            roots.push(PathBuf::from(value));
        }
    }
    roots.extend([
        PathBuf::from("/usr/local/cuda"),
        PathBuf::from("/usr/local/cuda-13.0"),
    ]);
    for root in roots {
        let include = if root.join("include/cupti.h").is_file() {
            root.join("include")
        } else if root.join("targets/sbsa-linux/include/cupti.h").is_file() {
            root.join("targets/sbsa-linux/include")
        } else if root.join("targets/x86_64-linux/include/cupti.h").is_file() {
            root.join("targets/x86_64-linux/include")
        } else {
            continue;
        };
        let candidates = [
            root.join("lib64"),
            root.join("lib"),
            root.join("targets/sbsa-linux/lib"),
            root.join("targets/x86_64-linux/lib"),
        ];
        if let Some(library) = candidates.into_iter().find(|path| {
            path.join("libcupti.so").is_file() || path.join("libcupti.so.13").is_file()
        }) {
            return Some((include, library));
        }
    }
    None
}
