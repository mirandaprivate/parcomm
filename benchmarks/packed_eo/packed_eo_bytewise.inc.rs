// Exact radix-256 arithmetic. Work scales with the number of input bytes:
// 4/8/16/32/64/128 bits -> 1/1/2/4/8/16 polynomial convolutions per packed row.
// Lower digits are unsigned bytes, the top digit is signed. Every digit is
// multiplied against the 16 raw Montgomery halfwords of b before reduction.
// Row convolution is in Z/(2^32); its bounded final coefficients are recovered
// exactly as signed i32 and accumulated in i64 across rows.
const BYTE_LIMBS:usize=16;
type ByteCoefficient=[i32;BYTE_LIMBS];

pub fn dense_cross_kernel_name(bits:usize)->String {
    match std::env::var("EO_CROSS_KERNEL").as_deref(){
        Ok("baseline")=>return "v2_quadratic".into(),
        Ok("v3")=>return v3_dense_cross_kernel_name(bits),
        _=>(),
    }
    if bits>=16 && std::env::var("EO_CROSS_KERNEL").as_deref()!=Ok("v4") {
        return v3_dense_cross_kernel_name(bits);
    }
    if bits>128 {return "full_field_quadratic".into();}
    let leaf=match dense_leaf_kind(){DenseLeaf::Scalar=>"scalar",DenseLeaf::Avx2=>"avx2",DenseLeaf::Avx512=>"avx512"};
    if bits<=8 && std::env::var("EO_CROSS_KERNEL").as_deref()!=Ok("v4") {
        return format!("small{bits}_grouped_i32_leaf{}_{leaf}",small_leaf_size());
    }
    format!("radix256_{}planes_i32_{leaf}",bits.div_ceil(8))
}

fn byte_leaf_scalar(a:&[i32],b:&[ByteCoefficient],out:&mut [ByteCoefficient]) {
    // Intentional ring arithmetic, identical to low-word SIMD instructions.
    for (i,x) in a.iter().enumerate(){for (j,y) in b.iter().enumerate(){
        for limb in 0..BYTE_LIMBS {
            out[i+j][limb]=out[i+j][limb].wrapping_add(x.wrapping_mul(y[limb]));
        }
    }}
}
#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
unsafe fn byte_leaf_avx2(a:&[i32],b:&[ByteCoefficient],out:&mut [ByteCoefficient]) {
    use std::arch::x86_64::*;
    for (i,x) in a.iter().enumerate(){let x=_mm256_set1_epi32(*x);
        for (j,y) in b.iter().enumerate(){for limb in (0..BYTE_LIMBS).step_by(8){
            let y=_mm256_loadu_si256(y.as_ptr().add(limb).cast());
            let dst=out[i+j].as_mut_ptr().add(limb);
            let sum=_mm256_add_epi32(_mm256_loadu_si256(dst.cast()),_mm256_mullo_epi32(x,y));
            _mm256_storeu_si256(dst.cast(),sum);
        }}
    }
}
#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx512f")]
unsafe fn byte_leaf_avx512(a:&[i32],b:&[ByteCoefficient],out:&mut [ByteCoefficient]) {
    use std::arch::x86_64::*;
    for (i,x) in a.iter().enumerate(){let x=_mm512_set1_epi32(*x);
        for (j,y) in b.iter().enumerate(){for limb in (0..BYTE_LIMBS).step_by(16){
            let y=_mm512_loadu_si512(y.as_ptr().add(limb).cast());
            let dst=out[i+j].as_mut_ptr().add(limb);
            let sum=_mm512_add_epi32(_mm512_loadu_si512(dst.cast()),_mm512_mullo_epi32(x,y));
            _mm512_storeu_si512(dst.cast(),sum);
        }}
    }
}

fn byte_convolution(
    a:&[i32],b:&[ByteCoefficient],out:&mut [ByteCoefficient],
    scratch_a:&mut [i32],scratch_b:&mut [ByteCoefficient],scratch_product:&mut [ByteCoefficient],leaf:DenseLeaf,
) {
    let n=a.len();out.fill([0;BYTE_LIMBS]);
    if n<=8 {
        match leaf {
            #[cfg(target_arch="x86_64")]
            DenseLeaf::Avx512=>unsafe{byte_leaf_avx512(a,b,out)},
            #[cfg(target_arch="x86_64")]
            DenseLeaf::Avx2=>unsafe{byte_leaf_avx2(a,b,out)},
            _=>byte_leaf_scalar(a,b,out),
        }
        return;
    }
    let h=n/2;
    byte_convolution(&a[..h],&b[..h],&mut out[..n],scratch_a,scratch_b,scratch_product,leaf);
    byte_convolution(&a[h..],&b[h..],&mut out[n..],scratch_a,scratch_b,scratch_product,leaf);
    let(sum_a,rest_a)=scratch_a.split_at_mut(h);
    let(sum_b,rest_b)=scratch_b.split_at_mut(h);
    let(middle,rest_product)=scratch_product.split_at_mut(n);
    for i in 0..h {sum_a[i]=a[i].wrapping_add(a[h+i]);for limb in 0..BYTE_LIMBS{sum_b[i][limb]=b[i][limb].wrapping_add(b[h+i][limb]);}}
    byte_convolution(sum_a,sum_b,middle,rest_a,rest_b,rest_product,leaf);
    for i in 0..n {for limb in 0..BYTE_LIMBS {middle[i][limb]=middle[i][limb].wrapping_sub(out[i][limb]).wrapping_sub(out[n+i][limb]);}}
    for i in 0..n {for limb in 0..BYTE_LIMBS {out[h+i][limb]=out[h+i][limb].wrapping_add(middle[i][limb]);}}
}

#[inline]
fn byte_digit(value:i128,plane:usize,planes:usize)->i32 {
    if plane+1==planes {(value>>(8*plane)) as i32}
    else {((value as u128>>(8*plane))&255) as i32}
}

fn bytewise_dense_correlations<D:BitwidthData>(data:&D,basis:&[BitwidthFr],stride:usize)->Vec<BitwidthFr> {
    let planes=D::BITS.div_ceil(8);assert!((1..=16).contains(&planes));
    let rows=data.len()/stride;let tile_rows=BITWIDTH_TILE_ENTRIES/stride;
    // Karatsuba is valid over Z/(2^32), including intentional intermediate
    // wraparound. Each FINAL row coefficient has magnitude at most
    // stride*max_digit*65535 < 2^31, so its signed i32 representative is exact.
    let max_digit=if D::BITS<=8 {1u64<<(D::BITS-1)} else {255};
    assert!(stride as u64*max_digit*65535 < (1u64<<31));
    // Only final, recovered row coefficients enter the i64 tile accumulator.
    // Its magnitude is <=32768*255*65535 < 2^40, with no modular wraparound.
    assert!(BITWIDTH_TILE_ENTRIES as u64*255*65535<i64::MAX as u64);
    let leaf=dense_leaf_kind();
    (0..rows.div_ceil(tile_rows)).into_par_iter().map(|task|{
        let mut a=vec![0i32;stride*planes];let mut b=vec![[0i32;BYTE_LIMBS];stride];
        let mut out=vec![[0i32;BYTE_LIMBS];2*stride];
        let mut accumulators=vec![[0i64;BYTE_LIMBS];(2*stride-1)*planes];
        let mut scratch_a=vec![0i32;stride];let mut scratch_b=b.clone();
        let mut scratch_product=out.clone();
        for row in task*tile_rows..((task+1)*tile_rows).min(rows){
            let start=row*stride;
            for lane in 0..stride {
                let value=data.integer(start+stride-1-lane);
                for plane in 0..planes {a[plane*stride+lane]=byte_digit(value,plane,planes);}
                for limb in 0..BYTE_LIMBS {b[lane][limb]=((basis[start+lane].0.0[limb/4]>>(16*(limb%4)))&65535) as i32;}
            }
            for plane in 0..planes {
                byte_convolution(&a[plane*stride..(plane+1)*stride],&b,&mut out,
                    &mut scratch_a,&mut scratch_b,&mut scratch_product,leaf);
                let acc=&mut accumulators[plane*(2*stride-1)..(plane+1)*(2*stride-1)];
                for(sum,value) in acc.iter_mut().zip(&out){
                    for limb in 0..BYTE_LIMBS {sum[limb]+=value[limb] as i64;}
                }
            }
        }
        (0..2*stride-1).map(|slot|{
            let mut result=BitwidthFr::zero();
            for plane in (0..planes).rev(){
                let value=&accumulators[plane*(2*stride-1)+slot];
                let mut limbs=[0i128;4];
                for chunk in 0..BYTE_LIMBS {limbs[chunk/4]+=(value[chunk] as i128)<<(16*(chunk%4));}
                result=result.mul_u64::<5>(256)+bitwidth_reduce(&limbs);
            }
            result
        }).collect::<Vec<_>>()
    }).reduce(||vec![BitwidthFr::zero();2*stride-1],|mut left,right|{
        for(sum,value) in left.iter_mut().zip(right){*sum+=value;}left
    })
}

fn optimized_dense_correlations<D:BitwidthData>(data:&D,basis:&[BitwidthFr],stride:usize)->Vec<BitwidthFr>{
    if std::env::var("EO_CROSS_KERNEL").as_deref()==Ok("v3")
        || (D::BITS>=16 && std::env::var("EO_CROSS_KERNEL").as_deref()!=Ok("v4")) {
        // The outer dispatcher retains the original v3 quadratic 32-bit path.
        assert_ne!(D::BITS,32);
        v3_optimized_dense_correlations(data,basis,stride)
    } else if D::BITS<=8 && std::env::var("EO_CROSS_KERNEL").as_deref()!=Ok("v4") {
        if small_leaf_size()==4 {small_dense_correlations::<D,4>(data,basis,stride)}
        else {small_dense_correlations::<D,8>(data,basis,stride)}
    } else {bytewise_dense_correlations(data,basis,stride)}
}

fn small_leaf_size()->usize {
    match std::env::var("EO_SMALL_LEAF").as_deref() {
        Ok("4")=>4, Ok("8")|Err(_)=>8, _=>panic!("EO_SMALL_LEAF must be 4 or 8"),
    }
}
include!("packed_eo_small.inc.rs");
