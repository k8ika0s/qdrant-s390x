use std::env;

fn main() {
    println!("cargo:rerun-if-changed=cpp");
    let mut builder = cc::Build::new();
    let mut has_simd_sources = false;

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH")
        .expect("CARGO_CFG_TARGET_ARCH env-var is not defined or is not UTF-8");

    // Cargo may omit CARGO_CFG_TARGET_FEATURE for some targets.
    // Missing value means "no target features enabled".
    let target_feature = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();

    if target_arch == "x86_64" {
        builder.file("cpp/sse.c");
        builder.file("cpp/avx2.c");
        has_simd_sources = true;

        if builder.get_compiler().is_like_msvc() {
            builder.flag("/arch:AVX");
            builder.flag("/arch:AVX2");
            builder.flag("/arch:SSE");
            builder.flag("/arch:SSE2");
        } else {
            builder.flag("-march=haswell");
        }

        // O3 optimization level
        builder.flag("-O3");
        // Use popcnt instruction
        builder.flag("-mpopcnt");
    } else if target_arch == "aarch64" && target_feature.split(',').any(|feat| feat == "neon") {
        builder.file("cpp/neon.c");
        builder.flag("-O3");
        has_simd_sources = true;
    }

    if has_simd_sources {
        builder.compile("simd_utils");
    }
}
