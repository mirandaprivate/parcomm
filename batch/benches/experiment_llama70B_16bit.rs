use anyhow::Result;

#[path = "experiment_llama70B_8bit.rs"]
mod experiment_llama70b_stream;

fn main() -> Result<()> {
    experiment_llama70b_stream::run_i16_from_args(std::env::args().skip(1))
}
