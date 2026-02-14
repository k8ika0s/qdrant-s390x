use std::env;

fn main() {
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH")
        .expect("CARGO_CFG_TARGET_ARCH env-var is not defined or is not UTF-8");

    // Cargo may omit CARGO_CFG_TARGET_FEATURE for some targets.
    // Missing value means "no target features enabled".
    let target_feature = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();

    if target_arch == "aarch64" && target_feature.split(',').any(|feat| feat == "neon") {
        let mut builder = cc::Build::new();
        builder.file("src/spaces/metric_f16/cpp/neon.c");
        builder.flag("-O3");
        builder.flag("-march=armv8.2-a+fp16");
        builder.compile("simd_utils");
    }
}
