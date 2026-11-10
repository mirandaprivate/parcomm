// BN254 bitwidth-aware CPU kernels, included in the isolated bucketed_ip module.
// Two explicit dense vectors; no bucket layout or tensor-basis specialization.
// Every width uses the same row tasks, reduction and verifier.
use ark_ff::{Zero as _, One as _};
type BitwidthFr = ark_bn254::Fr;
pub const BITWIDTH_TILE_ENTRIES: usize = 32768;

pub trait BitwidthData: Sync {
    const BITS: usize;
    fn len(&self) -> usize;
    fn storage_bytes(&self) -> usize;
    fn field(&self, index: usize) -> BitwidthFr;
    fn integer(&self, index: usize) -> i128;
    #[inline]
    fn multiply(&self, index: usize, coefficient: BitwidthFr) -> BitwidthFr {
        if Self::BITS <= 64 {
            coefficient.mul_i64::<5>(self.integer(index) as i64)
        } else if Self::BITS <= 128 {
            coefficient.mul_i128::<5, 6>(self.integer(index))
        } else {
            coefficient * self.field(index)
        }
    }
}

macro_rules! integer_bitwidth_data {
    ($ty:ty,$bits:literal) => {
        impl BitwidthData for Vec<$ty> {
            const BITS: usize = $bits;
            fn len(&self)->usize { self.as_slice().len() }
            fn storage_bytes(&self)->usize { self.len()*std::mem::size_of::<$ty>() }
            #[inline] fn integer(&self,index:usize)->i128 { self[index] as i128 }
            #[inline] fn field(&self,index:usize)->BitwidthFr { BitwidthFr::from(self[index]) }
        }
    }
}
integer_bitwidth_data!(i8,8);
integer_bitwidth_data!(i16,16);
integer_bitwidth_data!(i32,32);
integer_bitwidth_data!(i64,64);
integer_bitwidth_data!(i128,128);

pub struct PackedNibbles(pub Vec<u8>);
impl BitwidthData for PackedNibbles {
    const BITS:usize=4;
    fn len(&self)->usize { self.0.len()*2 }
    fn storage_bytes(&self)->usize { self.0.len() }
    #[inline] fn integer(&self,index:usize)->i128 {
        // Each nibble stores (signed_value + 8), so all values are in [-8,7].
        (((self.0[index/2] >> ((index&1)*4)) & 15) as i128)-8
    }
    #[inline] fn field(&self,index:usize)->BitwidthFr { BitwidthFr::from(self.integer(index) as i8) }
}
impl BitwidthData for Vec<BitwidthFr> {
    const BITS:usize=254;
    fn len(&self)->usize { self.as_slice().len() }
    fn storage_bytes(&self)->usize { self.len()*32 }
    #[inline] fn field(&self,index:usize)->BitwidthFr { self[index] }
    fn integer(&self,_index:usize)->i128 { unreachable!("full field values must never be truncated") }
}

/// Accumulate raw Montgomery limbs times signed integers without conversion.
/// Bound: <=32768 summands, |integer|<=2^31, limb<2^64. Thus each signed
/// accumulator has magnitude <2^110, safely inside i128. Never used for 64/128.
#[inline(always)]
fn bitwidth_accumulate(acc: &mut [i128;4], coefficient: &[u64;4], value: i128) {
    for limb in 0..4 { acc[limb] += coefficient[limb] as i128 * value; }
}

/// Normalize a signed sum of raw Montgomery limbs and reduce it once. The
/// input is already in Montgomery representation: do NOT multiply by R again.
#[inline]
fn bitwidth_reduce(acc: &[i128;4]) -> BitwidthFr {
    let mut limbs=[0u64;5];
    let mut carry=0i128;
    for i in 0..4 {
        let value=acc[i]+carry;
        limbs[i]=value as u64;
        carry=value>>64;
    }
    let negative=carry<0;
    limbs[4]=carry as u64;
    if negative {
        let mut carry=true;
        for limb in &mut limbs {
            let (value,next)=(!*limb).overflowing_add(carry as u64);
            *limb=value; carry=next;
        }
    }
    let value=BitwidthFr::from_unchecked_nplus1::<5>(ark_ff::BigInt(limbs));
    if negative { -value } else { value }
}

pub fn bitwidth_aware_claim<D:BitwidthData>(data:&D,dense_basis:&[BitwidthFr])->BitwidthFr {
    // Independent, unoptimized full-field calculation, excluded from prover time.
    assert_eq!(data.len(),dense_basis.len());
    dense_basis.par_chunks(BITWIDTH_TILE_ENTRIES).enumerate().map(|(tile,values)| {
        values.iter().enumerate().map(|(i,x)|data.field(tile*BITWIDTH_TILE_ENTRIES+i)*x).sum::<BitwidthFr>()
    }).sum()
}

pub fn prove_bitwidth_aware<D:BitwidthData>(
    data:&D,dense_basis:&[BitwidthFr],claimed:BitwidthFr,
    transcript:&mut dyn PackedEoChallengeTranscript<BitwidthFr>,
) -> Result<(PackedEoLiteBulletProof<BitwidthFr>,PaddingWorkCounters,StreamingEoProfile),StreamingEoError<std::convert::Infallible>> {
    prove_bitwidth_dense_impl(data,dense_basis,claimed,transcript,false)
}

/// Slow full-field arithmetic with identical messages, used only in small tests.
pub fn prove_bitwidth_dense_reference<D:BitwidthData>(
    data:&D,dense_basis:&[BitwidthFr],claimed:BitwidthFr,
    transcript:&mut dyn PackedEoChallengeTranscript<BitwidthFr>,
) -> Result<(PackedEoLiteBulletProof<BitwidthFr>,PaddingWorkCounters,StreamingEoProfile),StreamingEoError<std::convert::Infallible>> {
    prove_bitwidth_dense_impl(data,dense_basis,claimed,transcript,true)
}

fn prove_bitwidth_dense_impl<D:BitwidthData>(
    data:&D,dense_basis:&[BitwidthFr],claimed:BitwidthFr,
    transcript:&mut dyn PackedEoChallengeTranscript<BitwidthFr>, reference:bool,
) -> Result<(PackedEoLiteBulletProof<BitwidthFr>,PaddingWorkCounters,StreamingEoProfile),StreamingEoError<std::convert::Infallible>> {
    type F=BitwidthFr;
    assert_eq!(data.len(),dense_basis.len());
    assert!(data.len().is_power_of_two());
    let stride=packing_stride_for_bit_width::<F>(data.len(),D::BITS)?;
    let rows=data.len()/stride;
    let tile_rows=BITWIDTH_TILE_ENTRIES/stride;
    let tasks=rows.div_ceil(tile_rows);
    let mut profile=StreamingEoProfile::default();
    let started=Instant::now();
    // Correlation slots are indexed by (basis_lane - witness_lane + stride - 1).
    // The optimized path computes exactly these correlations by integer
    // convolution or deferred radix-digit reduction, from the dense inputs.
    let use_optimized=!reference && D::BITS<=128
        && !(D::BITS==32 && std::env::var("EO_CROSS_KERNEL").as_deref()!=Ok("v4"))
        && std::env::var("EO_CROSS_KERNEL").as_deref()!=Ok("baseline");
    let correlations=if use_optimized {
        optimized_dense_correlations(data,dense_basis,stride)
    } else { (0..tasks).into_par_iter().map(|task|{
        let start=task*tile_rows;
        let end=(start+tile_rows).min(rows);
        if D::BITS<=32 && !reference {
            let mut acc=vec![[0i128;4];2*stride-1];
            let mut integers=vec![0i128;stride];
            for row in start..end {
                let base=row*stride;
                for (lane,value) in integers.iter_mut().enumerate(){*value=data.integer(base+lane);}
                for (lane,coefficient) in dense_basis[base..base+stride].iter().enumerate(){
                    let window=&mut acc[lane..lane+stride];
                    for (sum,value) in window.iter_mut().zip(integers.iter().rev()){
                        bitwidth_accumulate(sum,&coefficient.0.0,*value);
                    }
                }
            }
            acc.iter().map(bitwidth_reduce).collect::<Vec<_>>()
        } else {
            let mut sums=vec![F::zero();2*stride-1];
            for row in start..end {
                let base=row*stride;
                for (lane,coefficient) in dense_basis[base..base+stride].iter().enumerate(){
                    for other in 0..stride {
                        sums[lane+stride-1-other]+=if reference {data.field(base+other)*coefficient}
                            else {data.multiply(base+other,*coefficient)};
                    }
                }
            }
            sums
        }
    }).reduce(||vec![F::zero();2*stride-1],|mut left,right|{
        for (dst,value) in left.iter_mut().zip(right){*dst+=value;}left
    }) };
    let trace=correlations[stride-1];
    if trace!=claimed{return Err(BvaError::ClaimMismatch.into());}
    let packed_positive=correlations[stride..].to_vec();
    let packed_negative=correlations[..stride-1].iter().rev().copied().collect::<Vec<_>>();
    profile.timings.packed_cross_terms_seconds=started.elapsed().as_secs_f64();
    let challenge=transcript.packed_challenge(&packed_positive,&packed_negative).map_err(StreamingEoError::ChallengeTranscript)?;
    let inverse=challenge.inverse().ok_or(BvaError::ZeroChallenge)?;
    let mut powers=vec![F::one();stride];let mut inverses=powers.clone();
    for i in 1..stride{powers[i]=powers[i-1]*challenge;inverses[i]=inverses[i-1]*inverse;}
    let inverse_limbs:Vec<[u64;4]>=inverses.iter().map(|x|x.0.0).collect();
    let started=Instant::now();
    let mut witness=vec![F::zero();rows];
    let mut basis=vec![F::zero();rows];
    witness.par_chunks_mut(tile_rows).zip(basis.par_chunks_mut(tile_rows)).enumerate().for_each(|(task,(output,other_output))|{
        for (offset,(dst,other_dst)) in output.iter_mut().zip(other_output).enumerate(){
            let start=(task*tile_rows+offset)*stride;
            if D::BITS<=32 && !reference {
                let mut acc=[0i128;4];
                for lane in 0..stride { bitwidth_accumulate(&mut acc,&inverse_limbs[lane],data.integer(start+lane)); }
                *dst=bitwidth_reduce(&acc);
            } else {
                *dst=(0..stride).map(|lane|if reference {data.field(start+lane)*inverses[lane]}
                    else {data.multiply(start+lane,inverses[lane])}).sum();
            }
            *other_dst=dense_basis[start..start+stride].iter().zip(&powers).map(|(v,p)|*v*p).sum();
        }
    });
    profile.timings.first_fold_seconds=started.elapsed().as_secs_f64();
    profile.maximum_live_vector_entries=2*rows;
    profile.physical_entries_per_pass=data.len();profile.total_visited_entries=2*data.len();profile.source_passes=2;
    let started=Instant::now();
    let mut reduced_claim=claimed;
    for i in 0..stride-1{reduced_claim+=packed_positive[i]*powers[i+1]+packed_negative[i]*inverses[i+1];}
    let (actual,_)=flat_inner_product(&witness,&basis);
    if actual!=reduced_claim{return Err(BvaError::ClaimMismatch.into());}
    profile.timings.reduced_check_seconds=started.elapsed().as_secs_f64();
    let mut binary_rounds=Vec::new();
    for round in 0..rows.ilog2() as usize {
        let started=Instant::now();
        let (iota,rho,_)=flat_eo_cross_inner_products(&witness,&basis);
        profile.timings.binary_cross_terms_seconds+=started.elapsed().as_secs_f64();
        let started=Instant::now();
        let challenge=transcript.binary_challenge(round,iota,rho).map_err(StreamingEoError::ChallengeTranscript)?;
        let inverse=challenge.inverse().ok_or(BvaError::ZeroChallenge)?;
        profile.timings.binary_challenge_seconds+=started.elapsed().as_secs_f64();
        binary_rounds.push((iota,rho));
        let started=Instant::now();
        flat_eo_fold_pair_in_place(&mut witness,&mut basis,challenge,inverse);
        profile.timings.binary_fold_seconds+=started.elapsed().as_secs_f64();
    }
    transcript.bind_reduced(witness[0],basis[0]).map_err(StreamingEoError::ChallengeTranscript)?;
    Ok((PackedEoLiteBulletProof {packing_stride:stride,packed_positive,packed_negative,binary_rounds,a_reduced:witness[0],b_reduced:basis[0]},PaddingWorkCounters::default(),profile))
}

include!("packed_eo_cross.inc.rs");
include!("packed_eo_bytewise.inc.rs");
