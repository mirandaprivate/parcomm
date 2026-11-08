use mikros_qafold::{ShortBaseField, koalabear::KoalaBear as B, packed_koalabear::PackedPcs};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let log: usize = std::env::args().nth(1).unwrap_or("26".into()).parse()?;
    let threads: usize = std::env::args().nth(2).unwrap_or("6".into()).parse()?;
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global()?;
    let started = Instant::now();
    let pcs = PackedPcs::new(log, 32, 100, [17; 32])?;
    let setup = started.elapsed().as_secs_f64();
    let mut workspace = mikros_qafold::packed_koalabear::PackedCommitWorkspace::default();
    let started = Instant::now();
    let mut records = Vec::new();
    // Distinct blocks within one sequential commitment pass, not repeated trials.
    for block in 0..4usize {
        let source = |index: usize| B::from_u64(
            (index.wrapping_mul(73).wrapping_add(19 + block * 31) as u16) as u64);
        let tick = Instant::now();
        let data = pcs.commit_from_fn_reusing(source, &mut workspace)?;
        let wall = tick.elapsed().as_secs_f64();
        let timing = data.commit_timings();
        let columns = data.encoding_columns().iter().map(|column| {
            let p = &column.stages;
            serde_json::json!({"start_ms":column.start_ms, "end_ms":column.end_ms,
                "wht_ms":p.wht.as_secs_f64()*1000.0, "multiply_ms":p.multiplication.as_secs_f64()*1000.0,
                "copy_ms":p.copy.as_secs_f64()*1000.0, "copy_wht_ms":p.copy_wht.as_secs_f64()*1000.0, "multiply_wht_ms":p.multiplication_wht.as_secs_f64()*1000.0,
                "wht_calls":p.wht_calls, "multiplications":p.multiplications})
        }).collect::<Vec<_>>();
        records.push(serde_json::json!({"block":block,"wall_s":wall,"root":data.commitment().root,
            "packing_ms":timing.packing_ms,"mobius_ms":timing.mobius_ms,
            "allocation_ms":timing.allocation_ms,"encoding_ms":timing.encoding_ms,
            "witness_release_ms":timing.witness_release_ms,"merkle_ms":timing.merkle_ms,
            "columns":columns}));
        data.recycle(&mut workspace);
    }
    drop(workspace);
    println!("{}", serde_json::json!({"log":log,"threads":threads,"setup_s":setup,
        "commit_pass_s":started.elapsed().as_secs_f64(),"blocks":records,
        "column_timers_are_overlapping_worker_attribution":true}));
    Ok(())
}

