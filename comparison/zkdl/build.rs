use std::env;
use std::path::PathBuf;
use std::process::Command;

fn run(command: &mut Command, label: &str) {
    let status = command
        .status()
        .unwrap_or_else(|error| panic!("failed to execute {label}: {error}"));
    if !status.success() {
        panic!("{label} failed with {status}");
    }
}

fn main() {
    println!("cargo:rerun-if-changed=src/gpu/zkdl_gpu_backend.cu");
    println!("cargo:rerun-if-changed=src/core/bls12-381.cuh");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=ZKDL_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    if env::var_os("CARGO_FEATURE_GPU").is_none() {
        return;
    }

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let object = out_dir.join("zkdl_gpu_backend.o");
    let archive = out_dir.join("libzkdl_gpu_backend.a");
    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".to_owned());
    let arch = env::var("ZKDL_CUDA_ARCH").unwrap_or_else(|_| "native".to_owned());

    run(
        Command::new(nvcc)
            .arg("-std=c++17")
            .arg("-O3")
            .arg("--compiler-options=-fPIC")
            .arg(format!("-arch={arch}"))
            .arg("-c")
            .arg("src/gpu/zkdl_gpu_backend.cu")
            .arg("-o")
            .arg(&object),
        "formal zkDL CUDA provider compilation",
    );
    run(
        Command::new("ar").arg("crus").arg(&archive).arg(&object),
        "formal zkDL CUDA provider archive",
    );

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    // CUDA 13 on AArch64/SBSA keeps libcudart below the target-specific
    // toolkit directory rather than the legacy `lib64` spelling.  Resolve
    // this at build time from the local CUDA installation so the native GPU
    // binary links against the same toolkit used by nvcc; no CPU fallback is
    // introduced when the runtime library is absent.
    let cuda_root = env::var_os("CUDA_HOME")
        .or_else(|| env::var_os("CUDA_PATH"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
    let target = match env::var("CARGO_CFG_TARGET_ARCH").as_deref() {
        Ok("aarch64") => "sbsa-linux",
        Ok("x86_64") => "x86_64-linux",
        _ => "",
    };
    let target_lib = if target.is_empty() {
        PathBuf::new()
    } else {
        cuda_root.join("targets").join(target).join("lib")
    };
    let legacy_lib = cuda_root.join("lib64");
    let cuda_lib = if target_lib.join("libcudart.so").is_file() {
        target_lib
    } else if legacy_lib.join("libcudart.so").is_file() {
        legacy_lib
    } else {
        panic!(
            "CUDA runtime library libcudart.so is missing below {} and {}",
            target_lib.display(),
            legacy_lib.display()
        );
    };
    println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    println!("cargo:rustc-link-lib=static=zkdl_gpu_backend");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
