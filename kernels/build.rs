fn main() {
    #[cfg(feature = "openmp")]
    {
        cc::Build::new()
            .file("src/openmp_gemv.c")
            .flag("-fopenmp")
            .flag("-O3")
            .flag("-march=native")
            .compile("openmp_gemv");
        println!("cargo:rustc-link-lib=gomp");
        println!("cargo:rerun-if-changed=src/openmp_gemv.c");
    }
    println!("cargo:rerun-if-changed=build.rs");
}
