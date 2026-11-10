// Exact integer convolution of reversed a with raw Montgomery limbs of b.
// This changes arithmetic only: stride, correlation order and transcript agree
// exactly with the quadratic full-field reference. No floating-point arithmetic.
#[derive(Clone,Copy)]
enum DenseLeaf { Scalar, Avx2, Avx512 }
fn dense_leaf_kind()->DenseLeaf {
    if std::env::var("EO_CROSS_SIMD").as_deref()==Ok("scalar") {return DenseLeaf::Scalar;}
    #[cfg(target_arch="x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512dq") {return DenseLeaf::Avx512;}
        if std::is_x86_feature_detected!("avx2") {return DenseLeaf::Avx2;}
    }
    DenseLeaf::Scalar
}
fn v3_dense_cross_kernel_name(bits:usize)->String {
    if std::env::var("EO_CROSS_KERNEL").as_deref()==Ok("baseline") {return "v2_quadratic".into();}
    if bits==32 {return "bounded_montgomery_i128_v2".into();}
    if bits<=16 {
        let leaf=match dense_leaf_kind(){DenseLeaf::Scalar=>"scalar",DenseLeaf::Avx2=>"avx2",DenseLeaf::Avx512=>"avx512dq"};
        format!("integer_karatsuba_{leaf}")
    } else if bits<=128 {"deferred_32bit_digits".into()} else {"full_field_quadratic".into()}
}

fn dense_leaf_scalar<const L:usize>(a:&[i64],b:&[[i64;L]],out:&mut [[i64;L]]) {
    for (i,x) in a.iter().enumerate(){for (j,y) in b.iter().enumerate(){
        for limb in 0..L {out[i+j][limb]+=x*y[limb];}
    }}
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
unsafe fn dense_leaf_avx2<const L:usize>(a:&[i64],b:&[[i64;L]],out:&mut [[i64;L]]) {
    use std::arch::x86_64::*;
    // Low 64 bits of signed multiplication via three unsigned 32-bit products.
    // All exact mathematical products and sums fit signed i64 (bounds below).
    assert_eq!(L%4,0);
    for (i,x) in a.iter().enumerate(){
        let x=_mm256_set1_epi64x(*x);let xhi=_mm256_srli_epi64::<32>(x);
        for (j,y) in b.iter().enumerate(){for limb in (0..L).step_by(4){
            let y=_mm256_loadu_si256(y.as_ptr().add(limb).cast());
            let yhi=_mm256_srli_epi64::<32>(y);
            let lo=_mm256_mul_epu32(x,y);
            let hi=_mm256_add_epi64(_mm256_mul_epu32(xhi,y),_mm256_mul_epu32(x,yhi));
            let product=_mm256_add_epi64(lo,_mm256_slli_epi64::<32>(hi));
            let dst=out[i+j].as_mut_ptr().add(limb);
            let sum=_mm256_add_epi64(_mm256_loadu_si256(dst.cast()),product);
            _mm256_storeu_si256(dst.cast(),sum);
        }}
    }
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx512f,avx512dq")]
unsafe fn dense_leaf_avx512<const L:usize>(a:&[i64],b:&[[i64;L]],out:&mut [[i64;L]]) {
    use std::arch::x86_64::*;
    assert_eq!(L%8,0);
    for (i,x) in a.iter().enumerate(){
        let x=_mm512_set1_epi64(*x);
        for (j,y) in b.iter().enumerate(){for limb in (0..L).step_by(8){
            let y=_mm512_loadu_si512(y.as_ptr().add(limb).cast());
            let dst=out[i+j].as_mut_ptr().add(limb);
            let sum=_mm512_add_epi64(_mm512_loadu_si512(dst.cast()),_mm512_mullo_epi64(x,y));
            _mm512_storeu_si512(dst.cast(),sum);
        }}
    }
}

fn dense_convolution<const L:usize>(
    a:&[i64],b:&[[i64;L]],out:&mut [[i64;L]],
    scratch_a:&mut [i64],scratch_b:&mut [[i64;L]],scratch_product:&mut [[i64;L]],leaf:DenseLeaf,
) {
    let n=a.len();out.fill([0;L]);
    if n<=8 {
        match leaf {
            #[cfg(target_arch="x86_64")]
            DenseLeaf::Avx512=>unsafe {dense_leaf_avx512(a,b,out)},
            #[cfg(target_arch="x86_64")]
            DenseLeaf::Avx2=>unsafe {dense_leaf_avx2(a,b,out)},
            _=>dense_leaf_scalar(a,b,out),
        }
        return;
    }
    let h=n/2;
    dense_convolution(&a[..h],&b[..h],&mut out[..n],scratch_a,scratch_b,scratch_product,leaf);
    dense_convolution(&a[h..],&b[h..],&mut out[n..],scratch_a,scratch_b,scratch_product,leaf);
    let (sum_a,rest_a)=scratch_a.split_at_mut(h);
    let (sum_b,rest_b)=scratch_b.split_at_mut(h);
    let (middle,rest_product)=scratch_product.split_at_mut(n);
    for i in 0..h {
        sum_a[i]=a[i]+a[h+i];
        for limb in 0..L {sum_b[i][limb]=b[i][limb]+b[h+i][limb];}
    }
    dense_convolution(sum_a,sum_b,middle,rest_a,rest_b,rest_product,leaf);
    // Compute all middle corrections before overlapping output writes.
    for i in 0..n {for limb in 0..L {middle[i][limb]-=out[i][limb]+out[n+i][limb];}}
    for i in 0..n {for limb in 0..L {out[h+i][limb]+=middle[i][limb];}}
}

fn dense_integer_correlations<D:BitwidthData,const L:usize,const CHUNK:usize>(data:&D,basis:&[BitwidthFr],stride:usize)->Vec<BitwidthFr> {
    assert_eq!(L*CHUNK,256);
    assert!(D::BITS<=32 && D::BITS+CHUNK<=48);
    // Tile sums: <=2^15 terms * 2^(bits-1) * (2^CHUNK-1), strictly <2^62.
    // Per-row recursive intermediates are bounded conservatively by
    // 2^(bits-1+CHUNK+3*log2(stride)); exponent <=62 for supported pairs.
    assert!(D::BITS-1+CHUNK+3*stride.ilog2() as usize<=62);
    let rows=data.len()/stride;let tile_rows=BITWIDTH_TILE_ENTRIES/stride;
    let leaf=dense_leaf_kind();
    (0..rows.div_ceil(tile_rows)).into_par_iter().map(|task|{
        let mut a=vec![0i64;stride];let mut b=vec![[0i64;L];stride];
        let mut out=vec![[0i64;L];2*stride];let mut acc=out.clone();
        let mut scratch_a=vec![0i64;stride];let mut scratch_b=vec![[0i64;L];stride];
        let mut scratch_product=vec![[0i64;L];2*stride];
        for row in task*tile_rows..((task+1)*tile_rows).min(rows) {
            let start=row*stride;
            for i in 0..stride {
                a[i]=data.integer(start+stride-1-i) as i64;
                for chunk in 0..L {
                    b[i][chunk]=((basis[start+i].0.0[chunk*CHUNK/64] >> (chunk*CHUNK%64)) & ((1u64<<CHUNK)-1)) as i64;
                }
            }
            dense_convolution(&a,&b,&mut out,&mut scratch_a,&mut scratch_b,&mut scratch_product,leaf);
            for (sum,value) in acc.iter_mut().zip(&out){for limb in 0..L{sum[limb]+=value[limb];}}
        }
        acc[..2*stride-1].iter().map(|value|{
            let mut limbs=[0i128;4];
            for chunk in 0..L {limbs[chunk*CHUNK/64]+=(value[chunk] as i128) << (chunk*CHUNK%64);}
            bitwidth_reduce(&limbs)
        }).collect::<Vec<_>>()
    }).reduce(||vec![BitwidthFr::zero();2*stride-1],|mut left,right|{
        for (sum,value) in left.iter_mut().zip(right){*sum+=value;}left
    })
}

fn dense_wide_correlations<D:BitwidthData>(data:&D,basis:&[BitwidthFr],stride:usize)->Vec<BitwidthFr> {
    let digits=D::BITS/32;assert!(digits==2 || digits==4);
    let rows=data.len()/stride;let tile_rows=BITWIDTH_TILE_ENTRIES/stride;
    // Unsigned lower radix-2^32 digits, signed top digit. Each raw-limb
    // accumulator is < 2^(64+32+15)=2^111, well within signed i128.
    (0..rows.div_ceil(tile_rows)).into_par_iter().map(|task|{
        let mut acc=vec![[0i128;4];(2*stride-1)*digits];
        let mut integers=vec![0i128;stride*digits];
        for row in task*tile_rows..((task+1)*tile_rows).min(rows) {
            let start=row*stride;
            for lane in 0..stride {
                let value=data.integer(start+stride-1-lane);
                for digit in 0..digits {
                    integers[lane*digits+digit]=if digit+1==digits {value>>(32*digit)}
                        else {((value as u128>>(32*digit))&0xffff_ffff) as i128};
                }
            }
            for (lane,coefficient) in basis[start..start+stride].iter().enumerate(){
                for (sum,value) in acc[lane*digits..(lane+stride)*digits].iter_mut().zip(&integers){
                    bitwidth_accumulate(sum,&coefficient.0.0,*value);
                }
            }
        }
        acc.chunks_exact(digits).map(|parts|{
            let mut value=bitwidth_reduce(&parts[digits-1]);
            for part in parts[..digits-1].iter().rev(){value=value.mul_u64::<5>(1u64<<32)+bitwidth_reduce(part);}
            value
        }).collect::<Vec<_>>()
    }).reduce(||vec![BitwidthFr::zero();2*stride-1],|mut left,right|{
        for (sum,value) in left.iter_mut().zip(right){*sum+=value;}left
    })
}

fn v3_optimized_dense_correlations<D:BitwidthData>(data:&D,basis:&[BitwidthFr],stride:usize)->Vec<BitwidthFr> {
    if D::BITS<=16 {dense_integer_correlations::<D,8,32>(data,basis,stride)}
    else {dense_wide_correlations(data,basis,stride)}
}
