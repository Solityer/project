fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=ICICLE_NATIVE_MANIFEST_SHA256");
    if std::env::var_os("CARGO_FEATURE_GPU_ACCELERATED").is_some() {
        let digest = std::env::var("ICICLE_NATIVE_MANIFEST_SHA256").expect("GPU build requires setup-gpu.sh and its baseline-local env.sh");
        assert!(digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit()), "invalid ICICLE native manifest digest");
        println!("cargo:rustc-env=ICICLE_NATIVE_MANIFEST_SHA256={}", digest);
    }
}
