use cmake::Config;
use std::{env, path::PathBuf};

fn main() {
    // Construct the path to the deps directory
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR is not set");
    let build_dir = PathBuf::from(format!("{}/../../../", &out_dir));
    let deps_dir = build_dir.join("deps");

    let main_dir = env::current_dir().expect("Failed to get current directory");
    let icicle_src_dir = PathBuf::from(format!("{}/../../../icicle/icicle", main_dir.display()));

    println!("cargo:rerun-if-env-changed=CXXFLAGS");
    println!("cargo:rerun-if-changed={}", icicle_src_dir.display());

    // Base config
    let mut config = Config::new(format!("{}", icicle_src_dir.display()));
    let icicle_install_dir = deps_dir.join("icicle");
    config
        .define("HASH", "ON")
        .define("CMAKE_INSTALL_PREFIX", &icicle_install_dir);

    // CUDA is a separately staged, authenticated baseline-local runtime
    // resource.  This public ICICLE frontend build must not fetch or compile
    // a backend here.

    // Build
    config.build_target("install").build();

    println!(
        "cargo:rustc-link-search={}/lib",
        icicle_install_dir.display()
    );
    println!("cargo:rustc-link-lib=icicle_hash");
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,{}/lib",
        icicle_install_dir.display()
    ); // Add RPATH linker arguments
}
