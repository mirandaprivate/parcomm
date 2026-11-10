// Implicit MLE equality weights. Point coordinates are in LSB-first order:
// eq(point,index) = product_j (index[j] ? point[j] : 1-point[j]).
// Only N/stride row coefficients are materialized, then reused as the folded b.

fn eq_factors(point: &[BitwidthFr]) -> Vec<(BitwidthFr, BitwidthFr)> {
    point.iter().map(|&x| (BitwidthFr::one()-x,x)).collect()
}

fn eq_weights_parallel(point: &[BitwidthFr]) -> Vec<BitwidthFr> {
    // Bounded tables, one final allocation; no successive large tensor copies.
    let low_bits=point.len().min(10);
    let low=tensor_coefficients(&eq_factors(&point[..low_bits]));
    let high=tensor_coefficients(&eq_factors(&point[low_bits..]));
    let mut output=vec![BitwidthFr::zero();1usize<<point.len()];
    output.par_chunks_mut(low.len()).zip(high.par_iter()).for_each(|(dst,&hi)| {
        for (value,&lo) in dst.iter_mut().zip(&low) { *value=hi*lo; }
    });
    output
}

/// Independent full-field input-claim preparation, with bounded block storage.
/// The prover recreates its own row weights inside its measured timer.
pub fn bitwidth_eq_claim<D:BitwidthData>(data:&D,point:&[BitwidthFr])->BitwidthFr {
    assert_eq!(data.len(),1usize<<point.len());
    let low_bits=point.len().min(12);
    let low=tensor_coefficients(&eq_factors(&point[..low_bits]));
    let high=tensor_coefficients(&eq_factors(&point[low_bits..]));
    high.par_iter().enumerate().map(|(block,&hi)| {
        let inner=low.iter().enumerate().map(|(offset,&lo)|data.field(block*low.len()+offset)*lo).sum::<BitwidthFr>();
        inner*hi
    }).sum()
}

/// Explicit reference evaluated as a product per index, only for small audits.
pub fn bitwidth_eq_dense_reference(point:&[BitwidthFr])->Vec<BitwidthFr> {
    assert!(point.len()<=16);
    (0..1usize<<point.len()).map(|index|point.iter().enumerate().map(|(bit,&x)|
        if (index>>bit)&1==1 {x} else {BitwidthFr::one()-x}).product()).collect()
}

/// Combine radix-256 signed accumulators before reduction, so the first fold
/// does not need one field reduction per byte. Lower digits are unsigned; the
/// top digit is signed, including i128::MIN without absolute-value overflow.
fn eq_reduce_planes(acc:&[[i128;4]],step:usize,planes:usize)->BitwidthFr {
    if planes==1 {return bitwidth_reduce(&acc[0]);}
    let mut combined=[0i128;7];
    for plane in 0..planes {
        let word=plane/8;let shift=(plane%8)*8;
        for limb in 0..4 {
            let value=acc[plane*step][limb];
            combined[word+limb]+=((value as u64) as i128)<<shift;
            combined[word+limb+1]+=(value>>64)<<shift;
        }
    }
    // At most 32768 coefficients per lane, byte <=255, raw limb <2^64.
    // Plane accumulators <2^87; combined i128 slots <2^124; total <2^400.
    let mut limbs=[0u64;7];let mut carry=0i128;
    for (dst,&value) in limbs.iter_mut().zip(&combined) {
        let value=value+carry;*dst=value as u64;carry=value>>64;
    }
    let negative=carry<0;
    debug_assert!(carry==0 || carry==-1);
    if negative {
        let mut carry=true;
        for limb in &mut limbs {let (v,c)=(!*limb).overflowing_add(carry as u64);*limb=v;carry=c;}
    }
    let mut used=7;while used>4 && limbs[used-1]==0 {used-=1;}
    // The patched Barrett helper requires input <2^(254+64), not every
    // possible 320-bit integer. Start with four limbs if the fifth is too big.
    // Every following shift is safe because result is already reduced <p.
    let count=if used>=5 && limbs[used-1]<(1u64<<62) {5} else {4};
    let mut initial=[0u64;5];initial[..count].copy_from_slice(&limbs[used-count..used]);
    let mut result=BitwidthFr::from_unchecked_nplus1::<5>(ark_ff::BigInt(initial));
    for low in (0..used-count).rev() {
        result=BitwidthFr::from_unchecked_nplus1::<5>(ark_ff::BigInt([
            limbs[low],result.0.0[0],result.0.0[1],result.0.0[2],result.0.0[3]]));
    }
    if negative {-result} else {result}
}

fn eq_lane_projections<D:BitwidthData>(data:&D,row_weights:&[BitwidthFr],stride:usize)->Vec<BitwidthFr> {
    let tile_rows=BITWIDTH_TILE_ENTRIES/stride;
    row_weights.par_chunks(tile_rows).enumerate().map(|(tile,weights)| {
        if D::BITS<=128 {
            let planes=D::BITS.div_ceil(8);
            let mut acc=vec![[0i128;4];planes*stride];
            for (offset,coefficient) in weights.iter().enumerate() {
                let start=(tile*tile_rows+offset)*stride;
                for lane in 0..stride {
                    let value=data.integer(start+lane);
                    for plane in 0..planes {
                        bitwidth_accumulate(&mut acc[plane*stride+lane],&coefficient.0.0,byte_digit(value,plane,planes) as i128);
                    }
                }
            }
            (0..stride).map(|lane|eq_reduce_planes(&acc[lane..],stride,planes)).collect::<Vec<_>>()
        } else {
            let mut sums=vec![BitwidthFr::zero();stride];
            for (offset,&coefficient) in weights.iter().enumerate() {
                let start=(tile*tile_rows+offset)*stride;
                for lane in 0..stride {sums[lane]+=data.field(start+lane)*coefficient;}
            }
            sums
        }
    }).reduce(||vec![BitwidthFr::zero();stride],|mut left,right| {
        for (dst,value) in left.iter_mut().zip(right) {*dst+=value;} left
    })
}

pub fn prove_bitwidth_eq<D:BitwidthData>(
    data:&D,point:&[BitwidthFr],claimed:BitwidthFr,
    transcript:&mut dyn PackedEoChallengeTranscript<BitwidthFr>,
) -> Result<(PackedEoLiteBulletProof<BitwidthFr>,PaddingWorkCounters,StreamingEoProfile),StreamingEoError<std::convert::Infallible>> {
    type F=BitwidthFr;
    assert_eq!(data.len(),1usize<<point.len());
    let stride=packing_stride_for_bit_width::<F>(data.len(),D::BITS)?;
    let rows=data.len()/stride;let lane_bits=stride.ilog2() as usize;
    let tile_rows=BITWIDTH_TILE_ENTRIES/stride;
    let mut profile=StreamingEoProfile::default();
    let started=Instant::now();
    let lane_weights=tensor_coefficients(&eq_factors(&point[..lane_bits]));
    let mut basis=eq_weights_parallel(&point[lane_bits..]);
    let projections=eq_lane_projections(data,&basis,stride);
    let trace=projections.iter().zip(&lane_weights).map(|(x,y)|*x*y).sum::<F>();
    if trace!=claimed {return Err(BvaError::ClaimMismatch.into());}
    let mut packed_positive=vec![F::zero();stride-1];
    let mut packed_negative=packed_positive.clone();
    for distance in 1..stride {for lane in 0..stride-distance {
        packed_positive[distance-1]+=projections[lane]*lane_weights[lane+distance];
        packed_negative[distance-1]+=projections[lane+distance]*lane_weights[lane];
    }}
    profile.timings.packed_cross_terms_seconds=started.elapsed().as_secs_f64();
    let challenge=transcript.packed_challenge(&packed_positive,&packed_negative).map_err(StreamingEoError::ChallengeTranscript)?;
    let inverse=challenge.inverse().ok_or(BvaError::ZeroChallenge)?;
    let mut powers=vec![F::one();stride];let mut inverses=powers.clone();
    for lane in 1..stride {powers[lane]=powers[lane-1]*challenge;inverses[lane]=inverses[lane-1]*inverse;}
    let inverse_limbs:Vec<[u64;4]>=inverses.iter().map(|x|x.0.0).collect();
    let started=Instant::now();
    let factor=lane_weights.iter().zip(&powers).map(|(x,y)|*x*y).sum::<F>();
    let mut witness=vec![F::zero();rows];
    witness.par_chunks_mut(tile_rows).zip(basis.par_chunks_mut(tile_rows)).enumerate().for_each(|(tile,(dst,basis_dst))| {
        for (offset,(value,weight)) in dst.iter_mut().zip(basis_dst).enumerate() {
            let start=(tile*tile_rows+offset)*stride;
            if D::BITS<=128 {
                let planes=D::BITS.div_ceil(8);let mut acc=[[0i128;4];16];
                for lane in 0..stride {
                    let integer=data.integer(start+lane);
                    for plane in 0..planes {
                        bitwidth_accumulate(&mut acc[plane],&inverse_limbs[lane],byte_digit(integer,plane,planes) as i128);
                    }
                }
                *value=eq_reduce_planes(&acc,1,planes);
            } else {*value=(0..stride).map(|lane|data.field(start+lane)*inverses[lane]).sum();}
            *weight*=factor;
        }
    });
    profile.timings.first_fold_seconds=started.elapsed().as_secs_f64();
    profile.maximum_live_vector_entries=2*rows;
    profile.physical_entries_per_pass=data.len();profile.total_visited_entries=2*data.len();profile.source_passes=2;
    let started=Instant::now();let mut reduced_claim=claimed;
    for lane in 0..stride-1 {reduced_claim+=packed_positive[lane]*powers[lane+1]+packed_negative[lane]*inverses[lane+1];}
    let (actual,_)=flat_inner_product(&witness,&basis);
    if actual!=reduced_claim {return Err(BvaError::ClaimMismatch.into());}
    profile.timings.reduced_check_seconds=started.elapsed().as_secs_f64();
    let mut binary_rounds=Vec::new();
    for round in 0..rows.ilog2() as usize {
        let started=Instant::now();let (iota,rho,_)=flat_eo_cross_inner_products(&witness,&basis);
        profile.timings.binary_cross_terms_seconds+=started.elapsed().as_secs_f64();
        let started=Instant::now();let challenge=transcript.binary_challenge(round,iota,rho).map_err(StreamingEoError::ChallengeTranscript)?;
        let inverse=challenge.inverse().ok_or(BvaError::ZeroChallenge)?;
        profile.timings.binary_challenge_seconds+=started.elapsed().as_secs_f64();
        binary_rounds.push((iota,rho));
        let started=Instant::now();flat_eo_fold_pair_in_place(&mut witness,&mut basis,challenge,inverse);
        profile.timings.binary_fold_seconds+=started.elapsed().as_secs_f64();
    }
    transcript.bind_reduced(witness[0],basis[0]).map_err(StreamingEoError::ChallengeTranscript)?;
    Ok((PackedEoLiteBulletProof {packing_stride:stride,packed_positive,packed_negative,binary_rounds,a_reduced:witness[0],b_reduced:basis[0]},PaddingWorkCounters::default(),profile))
}
