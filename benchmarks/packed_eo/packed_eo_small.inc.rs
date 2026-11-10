// Small signed inputs: register-resident leaf outputs, no recursive clearing,
// and bounded groups accumulated in i32 before widening to tile-level i64.
// The correlation polynomial, packing stride and transcript are unchanged.

macro_rules! small_leaf_coefficients {
    ($n:ident, $emit:ident) => {
        $emit!(0; 0);
        $emit!(1; 0,1);
        $emit!(2; 0,1,2);
        if $n == 4 {
            $emit!(3; 0,1,2,3);
            $emit!(4; 1,2,3);
            $emit!(5; 2,3);
            $emit!(6; 3);
        } else {
            $emit!(3; 0,1,2,3);
            $emit!(4; 0,1,2,3,4);
            $emit!(5; 0,1,2,3,4,5);
            $emit!(6; 0,1,2,3,4,5,6);
            $emit!(7; 0,1,2,3,4,5,6,7);
            $emit!(8; 1,2,3,4,5,6,7);
            $emit!(9; 2,3,4,5,6,7);
            $emit!(10; 3,4,5,6,7);
            $emit!(11; 4,5,6,7);
            $emit!(12; 5,6,7);
            $emit!(13; 6,7);
            $emit!(14; 7);
        }
    }
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
unsafe fn small_leaf_avx2<const N:usize>(a:&[i32],b:&[ByteCoefficient],out:&mut [ByteCoefficient]) {
    use std::arch::x86_64::*;
    let n=N;
    for limb in [0usize,8] {
        macro_rules! coefficient {($slot:literal; $($i:literal),+) => {{
            let mut sum=_mm256_setzero_si256();
            $(sum=_mm256_add_epi32(sum,_mm256_mullo_epi32(
                _mm256_set1_epi32(a[$i]),
                _mm256_loadu_si256(b[$slot-$i].as_ptr().add(limb).cast())));)+
            _mm256_storeu_si256(out[$slot].as_mut_ptr().add(limb).cast(),sum);
        }}}
        small_leaf_coefficients!(n,coefficient);
    }
    out[2*N-1]=[0;BYTE_LIMBS];
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx512f")]
unsafe fn small_leaf_avx512<const N:usize>(a:&[i32],b:&[ByteCoefficient],out:&mut [ByteCoefficient]) {
    use std::arch::x86_64::*;
    let n=N;
    macro_rules! coefficient {($slot:literal; $($i:literal),+) => {{
        let mut sum=_mm512_setzero_si512();
        $(sum=_mm512_add_epi32(sum,_mm512_mullo_epi32(
            _mm512_set1_epi32(a[$i]),_mm512_loadu_si512(b[$slot-$i].as_ptr().cast())));)+
        _mm512_storeu_si512(out[$slot].as_mut_ptr().cast(),sum);
    }}}
    small_leaf_coefficients!(n,coefficient);
    out[2*N-1]=[0;BYTE_LIMBS];
}

fn small_convolution<const LEAF:usize>(a:&[i32],b:&[ByteCoefficient],out:&mut [ByteCoefficient],
    scratch_a:&mut [i32],scratch_b:&mut [ByteCoefficient],scratch_product:&mut [ByteCoefficient],leaf:DenseLeaf) {
    let n=a.len();
    if n==LEAF {
        match leaf {
            #[cfg(target_arch="x86_64")]
            DenseLeaf::Avx512=>unsafe{small_leaf_avx512::<LEAF>(a,b,out)},
            #[cfg(target_arch="x86_64")]
            DenseLeaf::Avx2=>unsafe{small_leaf_avx2::<LEAF>(a,b,out)},
            _=>{out.fill([0;BYTE_LIMBS]);byte_leaf_scalar(a,b,out);},
        }
        return;
    }
    assert!(n>LEAF);
    let half=n/2;
    small_convolution::<LEAF>(&a[..half],&b[..half],&mut out[..n],scratch_a,scratch_b,scratch_product,leaf);
    small_convolution::<LEAF>(&a[half..],&b[half..],&mut out[n..],scratch_a,scratch_b,scratch_product,leaf);
    let(sum_a,rest_a)=scratch_a.split_at_mut(half);
    let(sum_b,rest_b)=scratch_b.split_at_mut(half);
    let(middle,rest_product)=scratch_product.split_at_mut(n);
    for i in 0..half {
        sum_a[i]=a[i].wrapping_add(a[half+i]);
        for limb in 0..BYTE_LIMBS {sum_b[i][limb]=b[i][limb].wrapping_add(b[half+i][limb]);}
    }
    small_convolution::<LEAF>(sum_a,sum_b,middle,rest_a,rest_b,rest_product,leaf);
    for i in 0..n {for limb in 0..BYTE_LIMBS {
        middle[i][limb]=middle[i][limb].wrapping_sub(out[i][limb]).wrapping_sub(out[n+i][limb]);
    }}
    for i in 0..n {for limb in 0..BYTE_LIMBS {
        out[half+i][limb]=out[half+i][limb].wrapping_add(middle[i][limb]);
    }}
}

fn small_dense_correlations<D:BitwidthData,const LEAF:usize>(data:&D,basis:&[BitwidthFr],stride:usize)->Vec<BitwidthFr> {
    assert!(D::BITS==4 || D::BITS==8);
    let rows=data.len()/stride;
    let tile_rows=BITWIDTH_TILE_ENTRIES/stride;
    let row_bound=stride as u64*(1u64<<(D::BITS-1))*65535;
    let max_group=(i32::MAX as u64/row_bound) as usize;
    let group=1usize<<max_group.ilog2();
    // At stride 128/64, 4-bit/8-bit can group 32/4 rows, respectively.
    // All final grouped coefficients have magnitude <2^31. Intermediate
    // Karatsuba wraparound is harmless because it computes in Z/(2^32).
    assert!(group as u64*row_bound<=i32::MAX as u64);
    let leaf=dense_leaf_kind();
    (0..rows.div_ceil(tile_rows)).into_par_iter().map_init(|| (
        vec![0i32;stride], vec![[0i32;BYTE_LIMBS];stride],
        vec![[0i32;BYTE_LIMBS];2*stride], vec![[0i32;BYTE_LIMBS];2*stride],
        vec![[0i64;BYTE_LIMBS];2*stride-1], vec![0i32;stride],
        vec![[0i32;BYTE_LIMBS];stride], vec![[0i32;BYTE_LIMBS];2*stride],
    ), |(a,b,out,group_sum,tile_sum,scratch_a,scratch_b,scratch_product),task| {
        // Each Rayon job reuses scratch across tiles. Only persistent tile sums
        // need clearing; the convolution overwrites every output coefficient.
        tile_sum.fill([0;BYTE_LIMBS]);
        let end=((task+1)*tile_rows).min(rows);
        for first in (task*tile_rows..end).step_by(group) {
            for row in first..(first+group).min(end) {
                let start=row*stride;
                for lane in 0..stride {
                    a[lane]=data.integer(start+stride-1-lane) as i32;
                    for limb in 0..BYTE_LIMBS {b[lane][limb]=((basis[start+lane].0.0[limb/4]>>(16*(limb%4)))&65535) as i32;}
                }
                small_convolution::<LEAF>(a,b,out,scratch_a,scratch_b,scratch_product,leaf);
                if row==first {group_sum.copy_from_slice(out);}
                else {for (sum,value) in group_sum.iter_mut().zip(out.iter()) {
                    for limb in 0..BYTE_LIMBS {sum[limb]=sum[limb].wrapping_add(value[limb]);}
                }}
            }
            for (sum,value) in tile_sum.iter_mut().zip(group_sum.iter()) {
                for limb in 0..BYTE_LIMBS {sum[limb]+=value[limb] as i64;}
            }
        }
        tile_sum.iter().map(|value| {
            let mut limbs=[0i128;4];
            for chunk in 0..BYTE_LIMBS {limbs[chunk/4]+=(value[chunk] as i128)<<(16*(chunk%4));}
            bitwidth_reduce(&limbs)
        }).collect::<Vec<_>>()
    }).reduce(||vec![BitwidthFr::zero();2*stride-1],|mut left,right| {
        for (sum,value) in left.iter_mut().zip(right) {*sum+=value;}left
    })
}
