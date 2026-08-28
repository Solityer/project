use cmake::Config;
use std::{env, path::PathBuf};

fn main() {
    // Construct the path to the deps directory
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR is not set");
    let build_dir = PathBuf::from(format!("{}/../../../", &out_dir));
    let deps_dir = build_dir.join("deps");

    // Construct the path to icicle source directory
    let main_dir = env::current_dir().expect("Failed to get current directory");
    let icicle_src_dir = PathBuf::from(format!("{}/../../../../icicle/icicle", main_dir.display()));

    println!("cargo:rerun-if-env-changed=CXXFLAGS");
    println!("cargo:rerun-if-changed={}", icicle_src_dir.display());

    // Base config
    let mut config = Config::new(format!("{}", icicle_src_dir.display()));
    let icicle_install_dir = deps_dir.join("icicle");
    config
        .define("CURVE", "bn254")
        .define("FIELD", "bn254")
        .define("HASH", "ON")
        .define("CMAKE_INSTALL_PREFIX", &icicle_install_dir);

    // CUDA is a separately staged, authenticated baseline-local runtime
    // resource.  This public ICICLE frontend build must not fetch or compile
    // a backend here.
    // Optional Features that are default ON (so that default matches any backend)
    if cfg!(feature = "no_g2") {
        config.define("G2", "OFF");
    }
    if cfg!(feature = "no_ecntt") {
        config.define("ECNTT", "OFF");
    }

    // Build
    config.build_target("install").build();

    println!(
        "cargo:rustc-link-search={}/lib",
        icicle_install_dir.display()
    );
    println!("cargo:rustc-link-lib=icicle_field_bn254");
    println!("cargo:rustc-link-lib=icicle_curve_bn254");
    println!("cargo:rustc-link-lib=icicle_hash");
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,{}/lib",
        icicle_install_dir.display()
    ); // Add RPATH linker arguments
}
