use anyhow::Result;

#[path = "experiment_llama.rs"]
mod experiment_llama;

const OUTPUT_DIRECTORY: &str = "Llama-3.1-8B-BF16/parcomm-bf16-bits-fresh-benchmark";
const OUTPUT_NAME: &str = "llama-3.1-8b-bf16-bits-fresh";

fn main() -> Result<()> {
    let forwarded = std::env::args().skip(1).collect::<Vec<_>>();
    if forwarded
        .iter()
        .any(|argument| argument == "-h" || argument == "--help")
    {
        print_help();
        return Ok(());
    }

    let mut arguments = vec![
        "llama8b-i16".to_owned(),
        "--force-conversion".to_owned(),
        "--conversion-output-dir".to_owned(),
        OUTPUT_DIRECTORY.to_owned(),
        "--conversion-output-name".to_owned(),
        OUTPUT_NAME.to_owned(),
    ];
    arguments.extend(forwarded);
    experiment_llama::run_from_args(arguments)
}

fn print_help() {
    println!(
        "Usage:\n\
         RAYON_NUM_THREADS=64 cargo bench -p batch --bench experiment_llama8B_16bit -- [OPTIONS]\n\
         \n\
         This benchmark copies every Llama 8B BF16 payload bit-for-bit into a fresh 16-bit\n\
         BVA, without floating-point decoding, scaling, rounding, or clipping, before running\n\
         the native ShortInt 16-bit protocol.\n\
         Its default output is isolated under {OUTPUT_DIRECTORY}/{OUTPUT_NAME}.*.\n\
         \n\
         Common options:\n\
           --model-data-dir PATH    Model-data root; defaults to workspace ../model-data\n\
           --input PATH             Override the source BF16 checkpoint directory\n\
           --python PATH            Python executable (default: python3)\n\
           --converter PATH         Converter path (default: preprocessing/modelconvert.py)\n\
           --chunk-mib N            Python conversion chunk size (default: 64)\n\
           --target-log N           Assert the exact logical BVA domain\n\
           --seed N                 Deterministic prover seed\n\
           --verify-input-hash      Verify the generated BVA SHA-256 (default)\n\
           --skip-input-hash        Skip only the generated BVA SHA-256 pass\n\
           --allow-non-64-threads   Permit a local smoke run\n\
           --shape-validity-proof   Include the optional shape validity proof\n\
           --summary-only           Print stages and the final table without detailed metrics\n\
           -h, --help               Print this help"
    );
}
