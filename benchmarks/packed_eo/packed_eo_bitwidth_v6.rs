//! Hidden EO-LiteBullet with native-width a and implicit MLE eq(point) weights.
use std::{io::Write, time::Instant};
use anyhow::{ensure, Result};
use ark_bn254::Fr;
use ark_ff::{PrimeField, UniformRand, Zero, One};
use ark_std::rand::{rngs::StdRng, RngCore, SeedableRng};
use rayon::prelude::*;
use batch::bucketed_ip::{
    bitwidth_aware_claim, bitwidth_eq_claim, bitwidth_eq_dense_reference,
    prove_bitwidth_eq, prove_bitwidth_dense_reference,
    packing_stride_for_bit_width, BitwidthData, PackedNibbles, BITWIDTH_TILE_ENTRIES,
};
use batch::hidden_packed_eo::{
    prove_hidden_bitwidth_reduction, verify_hidden_packed_i8_reduction,
    audit_bitwidth_endpoints, HiddenPackedEoProof,
};
use batch::jolt_dory::{setup_dory_parameters, commit_field_vector_zk};
use sha2::{Digest, Sha256};

const DOMAIN: &[u8] = b"implicit-mle-eq-hidden-eo-v6-lsb-first-no-input-pcs-opening";
fn point_binding(point:&[Fr])->Vec<u8> {
    let mut hash=Sha256::new();hash.update(DOMAIN);
    hash.update((point.len() as u64).to_le_bytes());
    for value in point {for limb in value.into_bigint().0 {hash.update(limb.to_le_bytes());}}
    hash.finalize().to_vec()
}
const PROOF_SEED:u64=0x5a4b5041434b4544;
fn rss() -> u64 {
    std::fs::read_to_string("/proc/self/status").unwrap().lines()
        .find_map(|line| line.strip_prefix("VmHWM:").and_then(|s| s.split_whitespace().next()).and_then(|s| s.parse().ok())).unwrap()
}
fn generate<E: Default+Clone+Send>(n: usize, seed:u64, sample: impl Fn(&mut StdRng)->E + Sync) -> Vec<E> {
    let mut values = vec![E::default(); n];
    values.par_chunks_mut(8192).enumerate().for_each(|(chunk, dst)| {
        let mut rng = StdRng::seed_from_u64(seed ^ chunk as u64);
        for value in dst { *value = sample(&mut rng); }
    });
    values
}
fn run<D:BitwidthData>(values:D,generation:f64,smoke:bool)->Result<()> {
    let n=values.len(); let width=D::BITS;
    let started=Instant::now();
    let pattern=std::env::var("EO_TEST_PATTERN").unwrap_or_else(|_|"mixed".into());
    ensure!(pattern=="mixed"||smoke,"extreme patterns require --smoke");
    let mut point_rng=StdRng::seed_from_u64(0x45515f504f494e54);
    let mut point:Vec<Fr>=(0..n.ilog2()).map(|_|Fr::rand(&mut point_rng)).collect();
    let point_pattern=std::env::var("EO_EQ_PATTERN").unwrap_or_else(|_|"random".into());
    ensure!(point_pattern=="random"||smoke,"special eq points require --smoke");
    match point_pattern.as_str() {
        "random"=>(),"zero"=>point.fill(Fr::zero()),"one"=>point.fill(Fr::one()),
        "alternating"=>{for (index,x) in point.iter_mut().enumerate(){*x=Fr::from((index%2) as u64);}},
        _=>anyhow::bail!("unknown eq point pattern"),
    }
    let binding=point_binding(&point);
    let basis_generation=started.elapsed().as_secs_f64();
    let stride=packing_stride_for_bit_width::<Fr>(n,width)?;
    println!("scope=hidden_implicit_eq_two_vector_eo_reduction_only");
    println!("cross_kernel=eq_lane_projection_radix256");
    println!("test_pattern={pattern} eq_point_pattern={point_pattern}");
    println!("input_pcs_commitment_and_openings_included=false bucket_count=not_applicable tensor_optimization=true basis_storage=implicit_eq point_order=lsb_first");
    println!("field=BN254_scalar field_modulus_bits={} input_bit_width={width} basis_bit_width=254 entries={n} packing_stride={stride}",Fr::MODULUS_BIT_SIZE);
    println!("engine=implicit_eq_byte_planes_v6 tile_entries={BITWIDTH_TILE_ENTRIES} a_storage_bytes={} b_storage_bytes={} prover_threads={} verifier_threads=1 repetitions=1",values.storage_bytes(),point.len()*32,rayon::current_num_threads());
    let planes=if width<=128 {width.div_ceil(8)} else {0};
    println!("a_byte_planes={planes} packing_policy=adaptive projection_byte_products={} first_fold_byte_products={} folded_vector_entries={} predicted_main_live_bytes={}",n*planes,n*planes,2*n/stride,values.storage_bytes()+64*n/stride);
    println!("eq_row_weight_generation_in_prover=true eq_point_binding_sha256={}",binding.iter().map(|v|format!("{v:02x}")).collect::<String>());
    println!("generation_seconds={:.9} a_generation_seconds={generation:.9} b_generation_seconds={basis_generation:.9}",generation+basis_generation);
    let started=Instant::now();
    let claimed=bitwidth_eq_claim(&values,&point);
    println!("claim_preparation_seconds={:.9}",started.elapsed().as_secs_f64());
    let started=Instant::now();
    let (prover,verifier)=setup_dory_parameters(12);
    // Transcript fixture only, not a commitment to either measured input vector.
    let fixture=commit_field_vector_zk(&prover,&[Fr::zero();64])?;
    let matrix_commitment=&fixture.commitment;
    let verifier_pool=rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    println!("setup_seconds={:.9} setup_excluded_from_prover_and_verifier=true",started.elapsed().as_secs_f64());
    std::io::stdout().flush()?;
    let mut rng=StdRng::seed_from_u64(PROOF_SEED);
    let started=Instant::now();
    let (proof,state,_,profile)=prove_hidden_bitwidth_reduction(
        &prover,n,claimed,width,matrix_commitment,&binding,&mut rng,
        |transcript|prove_bitwidth_eq(&values,&point,claimed,transcript))?;
    let prover_seconds=started.elapsed().as_secs_f64();
    let started=Instant::now();let bytes=bincode::serialize(&proof)?;
    let serialization_seconds=started.elapsed().as_secs_f64();
    let started=Instant::now();let decoded:HiddenPackedEoProof=bincode::deserialize(&bytes)?;
    let deserialization_seconds=started.elapsed().as_secs_f64();
    ensure!(proof==decoded,"proof serialization mismatch");
    let started=Instant::now();
    verifier_pool.install(||verify_hidden_packed_i8_reduction(&verifier,matrix_commitment,n,claimed,width,&binding,&decoded))?;
    let verifier_seconds=started.elapsed().as_secs_f64();let peak_rss_kib=rss();
    println!("prover_seconds={prover_seconds:.9}");
    println!("proof_size_bytes={} proof_size_kib={:.9}",bytes.len(),bytes.len() as f64/1024.0);
    println!("verifier_seconds={verifier_seconds:.9}");
    println!("peak_rss_kib={peak_rss_kib} peak_rss_gib={:.9}",peak_rss_kib as f64/1048576.0);
    println!("serialization_seconds={serialization_seconds:.9} deserialization_seconds={deserialization_seconds:.9}");
    println!("scalar_commitments={} proof_sha256={:x}",proof.scalar_commitment_count(),Sha256::digest(&bytes));
    println!("cross_terms_seconds={:.9} first_fold_seconds={:.9} reduced_check_seconds={:.9} binary_rounds_seconds={:.9}",profile.timings.packed_cross_terms_seconds,profile.timings.first_fold_seconds,profile.timings.reduced_check_seconds,profile.timings.binary_rounds_seconds());
    if smoke {
        let basis=bitwidth_eq_dense_reference(&point);
        ensure!(bitwidth_aware_claim(&values,&basis)==claimed,"independent explicit eq claim mismatch");
        let expanded:Vec<Fr>=(0..n).map(|i|values.field(i)).collect();
        ensure!(audit_bitwidth_endpoints(&state,&expanded,&basis,stride),"independent endpoint audit failed");
        let mut reference_rng=StdRng::seed_from_u64(PROOF_SEED);
        let (reference,_,_,_)=prove_hidden_bitwidth_reduction(&prover,n,claimed,width,matrix_commitment,&binding,&mut reference_rng,
            |transcript|prove_bitwidth_dense_reference(&values,&basis,claimed,transcript))?;
        ensure!(reference==proof,"bitwidth/full-field proof mismatch");
        let mut changed=decoded.clone();changed.b_reduced_commitment=changed.a_reduced_commitment;
        ensure!(verifier_pool.install(||verify_hidden_packed_i8_reduction(&verifier,matrix_commitment,n,claimed,width,&binding,&changed)).is_err(),"changed terminal commitment accepted");
        ensure!(verifier_pool.install(||verify_hidden_packed_i8_reduction(&verifier,matrix_commitment,n,claimed+Fr::one(),width,&binding,&decoded)).is_err(),"altered claim accepted");
        let mut wrong_point=point.clone();wrong_point[0]+=Fr::one();
        ensure!(verifier_pool.install(||verify_hidden_packed_i8_reduction(&verifier,matrix_commitment,n,claimed,width,&point_binding(&wrong_point),&decoded)).is_err(),"altered eq point accepted");
        println!("endpoint_audit=true tamper_rejected=true full_field_equivalence_checked=true boundary_values_injected=true explicit_eq_claim_checked=true altered_point_rejected=true");
    }
    println!("accepted=true");Ok(())
}
fn main()->Result<()> {
    let args:Vec<String>=std::env::args().collect();
    ensure!(args.len()>=3,"usage: packed_eo_bitwidth_v6 <4|8|16|32|64|128|full> <log_n> [--smoke]");
    let width=if args[1]=="full"{254}else{args[1].parse()?};let log:usize=args[2].parse()?;
    ensure!((8..=30).contains(&log),"log_n must be 8..30");
    let n=1usize<<log;let smoke=args.iter().any(|x|x=="--smoke");
    ensure!(!smoke||log<=16,"smoke endpoint audit limited to log_n<=16");
    let started=Instant::now();let seed=0x5041434b45445f45;
    let extreme=std::env::var("EO_TEST_PATTERN").unwrap_or_else(|_|"mixed".into());
    ensure!(matches!(extreme.as_str(),"mixed"|"min"|"max"),"invalid test pattern");
    macro_rules! signed_case {
        ($ty:ty,$sample:expr)=>{{
            let mut v:Vec<$ty>=generate(n,seed,$sample);
            if smoke {for row in v.chunks_mut(8){row[0]=<$ty>::MIN;row[1]=<$ty>::MAX;row[2]=0;row[3]=-1;row[4]=1;}}
            if extreme=="min" {v.fill(<$ty>::MIN);} else if extreme=="max" {v.fill(<$ty>::MAX);}
            run(v,started.elapsed().as_secs_f64(),smoke)
        }}
    }
    match width {
        4=>{let mut v=generate(n/2,seed,|r|r.next_u32() as u8);if smoke {for row in v.chunks_mut(8){row[0]=0xf0;row[1]=0x98;}}if extreme=="min"{v.fill(0);}else if extreme=="max"{v.fill(255);}run(PackedNibbles(v),started.elapsed().as_secs_f64(),smoke)},
        8=>signed_case!(i8,|r|r.next_u32() as i8),
        16=>signed_case!(i16,|r|r.next_u32() as i16),
        32=>signed_case!(i32,|r|r.next_u32() as i32),
        64=>signed_case!(i64,|r|r.next_u64() as i64),
        128=>signed_case!(i128,|r|((r.next_u64() as u128)<<64|r.next_u64() as u128) as i128),
        254=>{let mut v=generate(n,seed,Fr::rand);if smoke {for row in v.chunks_mut(8){row[0]=Fr::zero();row[1]=-Fr::one();row[2]=Fr::one();}}if extreme=="min"{v.fill(Fr::zero());}else if extreme=="max"{v.fill(-Fr::one());}run(v,started.elapsed().as_secs_f64(),smoke)},
        _=>anyhow::bail!("unsupported width"),
    }
}
