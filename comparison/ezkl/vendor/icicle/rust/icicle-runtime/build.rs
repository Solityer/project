use cmake::Config;
use std::{env, path::PathBuf};

fn main() {
    // Construct the path to the deps directory
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR is not set");
    let build_dir = PathBuf::from(format!("{}/../../../", &out_dir));
    let deps_dir = build_dir.join("deps");

    let main_dir = env::current_dir().expect("Failed to get current directory");
    let icicle_src_dir = PathBuf::from(format!("{}/../../../icicle/icicle", main_dir.display()));

    // Keep the arbitrary-path loader out of the public ICICLE device DSO.
    // This bridge is linked into the authenticated Rust executable and its
    // hidden symbol is callable only by the Rust wrapper in this crate.
    cc::Build::new()
        .cpp(true)
        .file(icicle_src_dir.join("src/runtime_internal_loader.cpp"))
        .include(icicle_src_dir.join("include"))
        .include(icicle_src_dir.join("src"))
        .flag_if_supported("-fvisibility=hidden")
        .compile("icicle_authenticated_loader");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!(
        "cargo:rerun-if-changed={}",
        icicle_src_dir
            .join("src/runtime_internal_loader.cpp")
            .display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        icicle_src_dir.join("src/runtime_internal.h").display()
    );

    println!("cargo:rerun-if-env-changed=CXXFLAGS");
    println!("cargo:rerun-if-changed={}", icicle_src_dir.display());

    // Base config
    let mut config = Config::new(format!("{}", icicle_src_dir.display()));
    // Keep generated libraries in Cargo's baseline-local target directory.
    let icicle_install_dir = deps_dir.join("icicle");
    config
        .define("HASH", "OFF")
        .define("CMAKE_INSTALL_PREFIX", &icicle_install_dir);

    // The public ICICLE commit intentionally contains only the CPU frontend.
    // CUDA is a separately staged, authenticated baseline-local runtime
    // resource, so this build never attempts to compile or fetch a backend.

    // Build
    config.build_target("install").build();

    println!(
        "cargo:rustc-link-search={}/lib",
        icicle_install_dir.display()
    );
    println!("cargo:rustc-link-lib=icicle_device");
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,{}/lib",
        icicle_install_dir.display()
    ); // Add RPATH linker arguments
}
