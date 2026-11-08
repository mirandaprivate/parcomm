
use parcomm_qafold::{layout::Shape, tbv, transcript::{Transcript,digest}, *};
use std::time::Instant;
fn main() -> Result<()> {
    let exponent: usize = std::env::args().nth(1).unwrap_or("24".into()).parse()?;
    let threads: usize = std::env::args().nth(2).unwrap_or("8".into()).parse()?;
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global()?;
    let shape = Shape { rows:(1usize<<((exponent+1)/2))-3, cols:(1usize<<(exponent/2))-5 };
    let config = Config { chunk_log:exponent-3, packed_log:4, query_rf_bits:100 };
    let start=Instant::now();
    let parameters=Parameters::new(&[shape],config.clone())?;
    let setup=start.elapsed().as_secs_f64();
    let matrix=(0..shape.rows*shape.cols).map(|index| index.wrapping_mul(73).wrapping_add(19) as i8).collect::<Vec<_>>();
    let values=parameters.layout.encode(&[matrix])?;
    let start=Instant::now();
    let committed=parameters.commit(values)?;
    let commit=start.elapsed().as_secs_f64();
    // Identical public points across protocol versions; independent of statement version.
    let mut transcript=Transcript::new();
    transcript.object(b"terminal-benchmark-shape", &shape)?;
    let row=(0..shape.rows.next_power_of_two().ilog2()).map(|_| transcript.challenge(b"row")).collect::<Vec<_>>();
    let col=(0..shape.cols.next_power_of_two().ilog2()).map(|_| transcript.challenge(b"col")).collect::<Vec<_>>();
    let start=Instant::now();
    let claims=tbv::evaluate_claims(&parameters.layout, committed.values(), &[(row,col)])?;
    let claims_time=start.elapsed().as_secs_f64();
    let statement=committed.statement.clone();
    let start=Instant::now();
    let (proof, profile)=parameters.prove(committed,&claims)?;
    let wire=proof.to_bytes()?;
    let opening=start.elapsed().as_secs_f64();
    let verifier_pool=rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let start=Instant::now();
    let verifier=verifier_pool.install(|| Parameters::new(&[shape],config))?;
    let verify_setup=start.elapsed().as_secs_f64();
    let start=Instant::now();
    let verified=verifier_pool.install(|| verifier.verify(&statement,&claims,&Proof::from_bytes(&wire)?))?;
    let verify=start.elapsed().as_secs_f64();
    if let Some(path)=std::env::args().nth(3) { std::fs::write(path,&wire)?; }
    println!("{}",serde_json::json!({"exponent":exponent,"threads":threads,"roots":statement.roots,
      "claims_hash":format!("{:x?}",digest(&serde_json::to_vec(&claims)?)),"openings":proof.openings.len(),
      "setup_s":setup,"commit_s":commit,"claims_s":claims_time,"open_s":opening,"total_s":commit+opening,
      "verify_setup_s":verify_setup,"verify_s":verify,"profile":profile,"verifier_profile":verified,
      "proof_bytes":wire.len(),"verified":true}));
    Ok(())
}
