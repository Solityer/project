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
    println!("cargo:rustc-link-lib=static=zkdl_gpu_backend");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
