use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn require_success(command: &mut Command, description: &str) {
    let status = command
        .status()
        .unwrap_or_else(|error| panic!("failed to run {description}: {error}"));
    assert!(status.success(), "{description} exited with {status}");
}

const SOURCES: [&str; 3] = ["i8_expand", "parcomm", "pairing"];

fn main() {
    for source in SOURCES {
        println!("cargo:rerun-if-changed=src/cuda/{source}.cu");
    }
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");

    let cuda_home = env::var_os("CUDA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
    let nvcc = cuda_home.join("bin/nvcc");
    assert!(nvcc.is_file(), "nvcc was not found at {}", nvcc.display());

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is not set"));
    let archive = out_dir.join("libsmartpc_i8_cuda.a");
    let architecture = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_89".to_owned());

    let mut objects = Vec::with_capacity(SOURCES.len());
    for source in SOURCES {
        let object = out_dir.join(format!("{source}.o"));
        require_success(
            Command::new(&nvcc)
                .arg("-std=c++17")
                .arg("-O3")
                .arg("-Xcompiler")
                .arg("-fPIC")
                .arg(format!("-arch={architecture}"))
                .arg("-c")
                .arg(format!("src/cuda/{source}.cu"))
                .arg("-o")
                .arg(&object),
            "nvcc",
        );
        objects.push(object);
    }

    let ar = env::var_os("AR").unwrap_or_else(|| "ar".into());
    let mut archive_command = Command::new(ar);
    archive_command.arg("crus").arg(&archive);
    for object in &objects {
        archive_command.arg(object);
    }
    require_success(&mut archive_command, "static archive creation");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=smartpc_i8_cuda");
    println!(
        "cargo:rustc-link-search=native={}",
        cuda_home.join(Path::new("lib64")).display()
    );
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
