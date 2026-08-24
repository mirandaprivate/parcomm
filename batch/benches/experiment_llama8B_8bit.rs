use anyhow::Result;

#[path = "experiment_llama70B_8bit.rs"]
mod experiment_llama_stream;

fn main() -> Result<()> {
    experiment_llama_stream::run_8b_i8_from_args(std::env::args().skip(1))
}
