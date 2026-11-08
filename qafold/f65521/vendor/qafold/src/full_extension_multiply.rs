//! Four/eight independent octic products in u64 SIMD lanes.
//! Canonical F65521 coefficients, X^8=17, unchanged wire representation.
//! Karatsuba inputs are sums of at most 8 limbs (<2^19); low-u32 SIMD
//! multiplication is exact. Folded polynomial coefficients are <120*p^2+ p
//! (<2^39). Two folds using 2^16=15 (mod p) leave <2p for final reduction.
use super::FullPrime65521Ext8 as F;
use crate::ShortFieldElement;
const _: () = assert!(core::mem::size_of::<F>() == 16);
const _: () = assert!(core::mem::align_of::<F>() == 2);

pub(super) fn multiply(left: &mut [F], right: &[F]) {
    assert_eq!(left.len(), right.len());
    #[cfg(target_arch="x86_64")]
    // SAFETY: CPU feature checks and whole-element bounds precede all vector accesses.
    unsafe {
        if left.len()>=8 && std::is_x86_feature_detected!("avx512f") { return multiply512(left,right); }
        if left.len()>=4 && std::is_x86_feature_detected!("avx2") { return multiply256(left,right); }
    }
    for (left,right) in left.iter_mut().zip(right) { *left *= *right; }
}
pub(super) fn add_scaled(output: &mut [F], input: &[F], scalar: F) {
    assert_eq!(output.len(), input.len());
    #[cfg(target_arch="x86_64")]
    unsafe {
        if output.len()>=8 && std::is_x86_feature_detected!("avx512f") { return add_scaled512(output,input,scalar); }
        if output.len()>=4 && std::is_x86_feature_detected!("avx2") { return add_scaled256(output,input,scalar); }
    }
    for (output,input) in output.iter_mut().zip(input) { *output = input.mul_add(scalar,*output); }
}
pub(super) fn fold_adjacent(values: &mut [F], scalar: F) {
    assert_eq!(values.len()%2,0);
    #[cfg(target_arch="x86_64")]
    unsafe {
        if values.len()>=16 && std::is_x86_feature_detected!("avx512f") { return fold_adjacent512(values,scalar); }
        if values.len()>=8 && std::is_x86_feature_detected!("avx2") { return fold_adjacent256(values,scalar); }
    }
    for index in 0..values.len()/2 { values[index] = values[2*index+1].mul_add(scalar,values[2*index]); }
}
#[cfg(target_arch="x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
unsafe fn load256(input: *const F, adjacent: bool) -> [__m256i;8] {
    unsafe {
        let (low,high)=if !adjacent { let first=_mm256_permute4x64_epi64::<0xd8>(_mm256_loadu_si256(input.cast()));
            let second=_mm256_permute4x64_epi64::<0xd8>(_mm256_loadu_si256(input.add(2).cast()));
            (_mm256_permute2x128_si256::<0x20>(first,second), _mm256_permute2x128_si256::<0x31>(first,second)) } else { let offsets=_mm256_setr_epi64x(0,32,64,96);
            (_mm256_i64gather_epi64::<1>(input.cast(),offsets), _mm256_i64gather_epi64::<1>(input.cast::<u8>().add(8).cast(),offsets)) };
        let mask=_mm256_set1_epi64x(65535);
        [_mm256_and_si256(low,mask),
         _mm256_and_si256(_mm256_srli_epi64::<16>(low),mask),
         _mm256_and_si256(_mm256_srli_epi64::<32>(low),mask),
         _mm256_and_si256(_mm256_srli_epi64::<48>(low),mask),
         _mm256_and_si256(high,mask),
         _mm256_and_si256(_mm256_srli_epi64::<16>(high),mask),
         _mm256_and_si256(_mm256_srli_epi64::<32>(high),mask),
         _mm256_and_si256(_mm256_srli_epi64::<48>(high),mask)]
    }
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
unsafe fn store256(output: *mut F, values: [__m256i;8]) -> () {
    unsafe {
        let low=_mm256_or_si256(_mm256_or_si256(values[0],_mm256_slli_epi64::<16>(values[1])),_mm256_or_si256(_mm256_slli_epi64::<32>(values[2]),_mm256_slli_epi64::<48>(values[3])));
        let high=_mm256_or_si256(_mm256_or_si256(values[4],_mm256_slli_epi64::<16>(values[5])),_mm256_or_si256(_mm256_slli_epi64::<32>(values[6]),_mm256_slli_epi64::<48>(values[7])));
        let even=_mm256_unpacklo_epi64(low,high);
        let odd=_mm256_unpackhi_epi64(low,high);
        _mm256_storeu_si256(output.cast(),_mm256_permute2x128_si256::<0x20>(even,odd));
        _mm256_storeu_si256(output.add(2).cast(),_mm256_permute2x128_si256::<0x31>(even,odd));
    }
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
unsafe fn karatsuba2256(left: [__m256i;2], right: [__m256i;2]) -> [__m256i;3] {
    unsafe {
        let lo=_mm256_mul_epu32(left[0],right[0]);
        let hi=_mm256_mul_epu32(left[1],right[1]);
        let mid=_mm256_sub_epi64(_mm256_sub_epi64(_mm256_mul_epu32(_mm256_add_epi64(left[0],left[1]),_mm256_add_epi64(right[0],right[1])),lo),hi);
        [lo,mid,hi]
    }
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
unsafe fn karatsuba4256(left: [__m256i;4], right: [__m256i;4]) -> [__m256i;7] {
    unsafe {
        let low=karatsuba2256([left[0],left[1]],[right[0],right[1]]);
        let high=karatsuba2256([left[2],left[3]],[right[2],right[3]]);
        let mid=karatsuba2256([_mm256_add_epi64(left[0],left[2]),_mm256_add_epi64(left[1],left[3])],[_mm256_add_epi64(right[0],right[2]),_mm256_add_epi64(right[1],right[3])]);
        let mut result=[_mm256_setzero_si256();7];
        for index in 0..3 {
            result[index]=_mm256_add_epi64(result[index],low[index]);
            result[index+2]=_mm256_add_epi64(result[index+2],_mm256_sub_epi64(_mm256_sub_epi64(mid[index],low[index]),high[index]));
            result[index+4]=_mm256_add_epi64(result[index+4],high[index]);
        }
        result
    }
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
unsafe fn karatsuba8256(left: [__m256i;8], right: [__m256i;8]) -> [__m256i;15] {
    unsafe {
        let low=karatsuba4256([left[0],left[1],left[2],left[3]],[right[0],right[1],right[2],right[3]]);
        let high=karatsuba4256([left[4],left[5],left[6],left[7]],[right[4],right[5],right[6],right[7]]);
        let mid=karatsuba4256([_mm256_add_epi64(left[0],left[4]),_mm256_add_epi64(left[1],left[5]),_mm256_add_epi64(left[2],left[6]),_mm256_add_epi64(left[3],left[7])],[_mm256_add_epi64(right[0],right[4]),_mm256_add_epi64(right[1],right[5]),_mm256_add_epi64(right[2],right[6]),_mm256_add_epi64(right[3],right[7])]);
        let mut result=[_mm256_setzero_si256();15];
        for index in 0..7 {
            result[index]=_mm256_add_epi64(result[index],low[index]);
            result[index+4]=_mm256_add_epi64(result[index+4],_mm256_sub_epi64(_mm256_sub_epi64(mid[index],low[index]),high[index]));
            result[index+8]=_mm256_add_epi64(result[index+8],high[index]);
        }
        result
    }
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
unsafe fn reduce256(value: __m256i) -> __m256i {
    unsafe {
        let mask=_mm256_set1_epi64x(65535);
        let high=_mm256_srli_epi64::<16>(value);
        let value=_mm256_add_epi64(_mm256_and_si256(value,mask),_mm256_sub_epi64(_mm256_slli_epi64::<4>(high),high));
        let high=_mm256_srli_epi64::<16>(value);
        let value=_mm256_add_epi64(_mm256_and_si256(value,mask),_mm256_sub_epi64(_mm256_slli_epi64::<4>(high),high));
        _mm256_min_epu32(value,_mm256_sub_epi64(value,_mm256_set1_epi64x(65521)))
    }
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
unsafe fn product256(left: [__m256i;8], right: [__m256i;8], addend: [__m256i;8]) -> [__m256i;8] {
    unsafe {
        let product=karatsuba8256(left,right);
        core::array::from_fn(|index| {
            let high=if index<7 {product[index+8]} else {_mm256_setzero_si256()};
            reduce256(_mm256_add_epi64(_mm256_add_epi64(product[index],_mm256_add_epi64(_mm256_slli_epi64::<4>(high),high)),addend[index]))
        })
    }
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
unsafe fn multiply256(left: &mut [F], right: &[F]) -> () {
    unsafe {
        let mut index=0;
        while index+4<=left.len() {
            let product=product256(load256(left.as_ptr().add(index),false),load256(right.as_ptr().add(index),false),[_mm256_setzero_si256();8]);
            store256(left.as_mut_ptr().add(index),product);
            index+=4;
        }
        for (left,right) in left[index..].iter_mut().zip(&right[index..]) { *left *= *right; }
    }
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
unsafe fn add_scaled256(output: &mut [F], input: &[F], scalar: F) -> () {
    unsafe {
        let scalar=scalar;
        let coefficients=core::array::from_fn(|index| _mm256_set1_epi64x(scalar.coefficients()[index].as_u16() as i64));
        let mut index=0;
        while index+4<=input.len() {
            let product=product256(load256(input.as_ptr().add(index),false),coefficients,load256(output.as_ptr().add(index),false));
            store256(output.as_mut_ptr().add(index),product);
            index+=4;
        }
        for (output,input) in output[index..].iter_mut().zip(&input[index..]) { *output=input.mul_add(scalar,*output); }
    }
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
unsafe fn fold_adjacent256(values: &mut [F], scalar: F) -> () {
    unsafe {
        let coefficients=core::array::from_fn(|index| _mm256_set1_epi64x(scalar.coefficients()[index].as_u16() as i64));
        let half=values.len()/2;
        let mut index=0;
        while index+4<=half {
            // Load all 2*LANES input elements before writing the shorter output prefix.
            let even=load256(values.as_ptr().add(2*index),true);
            let odd=load256(values.as_ptr().add(2*index+1),true);
            store256(values.as_mut_ptr().add(index),product256(odd,coefficients,even));
            index+=4;
        }
        for index in index..half { values[index]=values[2*index+1].mul_add(scalar,values[2*index]); }
    }
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx512f")]
unsafe fn load512(input: *const F, adjacent: bool) -> [__m512i;8] {
    unsafe {
        let (low,high)=if !adjacent { let first=_mm512_loadu_si512(input.cast());
            let second=_mm512_loadu_si512(input.add(4).cast());
            (_mm512_permutex2var_epi64(first,_mm512_setr_epi64(0,2,4,6,8,10,12,14),second),
             _mm512_permutex2var_epi64(first,_mm512_setr_epi64(1,3,5,7,9,11,13,15),second)) } else { let offsets=_mm512_setr_epi64(0,32,64,96,128,160,192,224);
            (_mm512_i64gather_epi64::<1>(offsets,input.cast()), _mm512_i64gather_epi64::<1>(offsets,input.cast::<u8>().add(8).cast())) };
        let mask=_mm512_set1_epi64(65535);
        [_mm512_and_si512(low,mask),
         _mm512_and_si512(_mm512_srli_epi64::<16>(low),mask),
         _mm512_and_si512(_mm512_srli_epi64::<32>(low),mask),
         _mm512_and_si512(_mm512_srli_epi64::<48>(low),mask),
         _mm512_and_si512(high,mask),
         _mm512_and_si512(_mm512_srli_epi64::<16>(high),mask),
         _mm512_and_si512(_mm512_srli_epi64::<32>(high),mask),
         _mm512_and_si512(_mm512_srli_epi64::<48>(high),mask)]
    }
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx512f")]
unsafe fn store512(output: *mut F, values: [__m512i;8]) -> () {
    unsafe {
        let low=_mm512_or_si512(_mm512_or_si512(values[0],_mm512_slli_epi64::<16>(values[1])),_mm512_or_si512(_mm512_slli_epi64::<32>(values[2]),_mm512_slli_epi64::<48>(values[3])));
        let high=_mm512_or_si512(_mm512_or_si512(values[4],_mm512_slli_epi64::<16>(values[5])),_mm512_or_si512(_mm512_slli_epi64::<32>(values[6]),_mm512_slli_epi64::<48>(values[7])));
        _mm512_storeu_si512(output.cast(),_mm512_permutex2var_epi64(low,_mm512_setr_epi64(0,8,1,9,2,10,3,11),high));
        _mm512_storeu_si512(output.add(4).cast(),_mm512_permutex2var_epi64(low,_mm512_setr_epi64(4,12,5,13,6,14,7,15),high));
    }
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx512f")]
unsafe fn karatsuba2512(left: [__m512i;2], right: [__m512i;2]) -> [__m512i;3] {
    unsafe {
        let lo=_mm512_mul_epu32(left[0],right[0]);
        let hi=_mm512_mul_epu32(left[1],right[1]);
        let mid=_mm512_sub_epi64(_mm512_sub_epi64(_mm512_mul_epu32(_mm512_add_epi64(left[0],left[1]),_mm512_add_epi64(right[0],right[1])),lo),hi);
        [lo,mid,hi]
    }
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx512f")]
unsafe fn karatsuba4512(left: [__m512i;4], right: [__m512i;4]) -> [__m512i;7] {
    unsafe {
        let low=karatsuba2512([left[0],left[1]],[right[0],right[1]]);
        let high=karatsuba2512([left[2],left[3]],[right[2],right[3]]);
        let mid=karatsuba2512([_mm512_add_epi64(left[0],left[2]),_mm512_add_epi64(left[1],left[3])],[_mm512_add_epi64(right[0],right[2]),_mm512_add_epi64(right[1],right[3])]);
        let mut result=[_mm512_setzero_si512();7];
        for index in 0..3 {
            result[index]=_mm512_add_epi64(result[index],low[index]);
            result[index+2]=_mm512_add_epi64(result[index+2],_mm512_sub_epi64(_mm512_sub_epi64(mid[index],low[index]),high[index]));
            result[index+4]=_mm512_add_epi64(result[index+4],high[index]);
        }
        result
    }
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx512f")]
unsafe fn karatsuba8512(left: [__m512i;8], right: [__m512i;8]) -> [__m512i;15] {
    unsafe {
        let low=karatsuba4512([left[0],left[1],left[2],left[3]],[right[0],right[1],right[2],right[3]]);
        let high=karatsuba4512([left[4],left[5],left[6],left[7]],[right[4],right[5],right[6],right[7]]);
        let mid=karatsuba4512([_mm512_add_epi64(left[0],left[4]),_mm512_add_epi64(left[1],left[5]),_mm512_add_epi64(left[2],left[6]),_mm512_add_epi64(left[3],left[7])],[_mm512_add_epi64(right[0],right[4]),_mm512_add_epi64(right[1],right[5]),_mm512_add_epi64(right[2],right[6]),_mm512_add_epi64(right[3],right[7])]);
        let mut result=[_mm512_setzero_si512();15];
        for index in 0..7 {
            result[index]=_mm512_add_epi64(result[index],low[index]);
            result[index+4]=_mm512_add_epi64(result[index+4],_mm512_sub_epi64(_mm512_sub_epi64(mid[index],low[index]),high[index]));
            result[index+8]=_mm512_add_epi64(result[index+8],high[index]);
        }
        result
    }
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx512f")]
unsafe fn reduce512(value: __m512i) -> __m512i {
    unsafe {
        let mask=_mm512_set1_epi64(65535);
        let high=_mm512_srli_epi64::<16>(value);
        let value=_mm512_add_epi64(_mm512_and_si512(value,mask),_mm512_sub_epi64(_mm512_slli_epi64::<4>(high),high));
        let high=_mm512_srli_epi64::<16>(value);
        let value=_mm512_add_epi64(_mm512_and_si512(value,mask),_mm512_sub_epi64(_mm512_slli_epi64::<4>(high),high));
        _mm512_min_epu32(value,_mm512_sub_epi64(value,_mm512_set1_epi64(65521)))
    }
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx512f")]
unsafe fn product512(left: [__m512i;8], right: [__m512i;8], addend: [__m512i;8]) -> [__m512i;8] {
    unsafe {
        let product=karatsuba8512(left,right);
        core::array::from_fn(|index| {
            let high=if index<7 {product[index+8]} else {_mm512_setzero_si512()};
            reduce512(_mm512_add_epi64(_mm512_add_epi64(product[index],_mm512_add_epi64(_mm512_slli_epi64::<4>(high),high)),addend[index]))
        })
    }
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx512f")]
unsafe fn multiply512(left: &mut [F], right: &[F]) -> () {
    unsafe {
        let mut index=0;
        while index+8<=left.len() {
            let product=product512(load512(left.as_ptr().add(index),false),load512(right.as_ptr().add(index),false),[_mm512_setzero_si512();8]);
            store512(left.as_mut_ptr().add(index),product);
            index+=8;
        }
        for (left,right) in left[index..].iter_mut().zip(&right[index..]) { *left *= *right; }
    }
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx512f")]
unsafe fn add_scaled512(output: &mut [F], input: &[F], scalar: F) -> () {
    unsafe {
        let scalar=scalar;
        let coefficients=core::array::from_fn(|index| _mm512_set1_epi64(scalar.coefficients()[index].as_u16() as i64));
        let mut index=0;
        while index+8<=input.len() {
            let product=product512(load512(input.as_ptr().add(index),false),coefficients,load512(output.as_ptr().add(index),false));
            store512(output.as_mut_ptr().add(index),product);
            index+=8;
        }
        for (output,input) in output[index..].iter_mut().zip(&input[index..]) { *output=input.mul_add(scalar,*output); }
    }
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx512f")]
unsafe fn fold_adjacent512(values: &mut [F], scalar: F) -> () {
    unsafe {
        let coefficients=core::array::from_fn(|index| _mm512_set1_epi64(scalar.coefficients()[index].as_u16() as i64));
        let half=values.len()/2;
        let mut index=0;
        while index+8<=half {
            // Load all 2*LANES input elements before writing the shorter output prefix.
            let even=load512(values.as_ptr().add(2*index),true);
            let odd=load512(values.as_ptr().add(2*index+1),true);
            store512(values.as_mut_ptr().add(index),product512(odd,coefficients,even));
            index+=8;
        }
        for index in index..half { values[index]=values[2*index+1].mul_add(scalar,values[2*index]); }
    }
}

#[cfg(all(test,target_arch="x86_64"))]
mod tests {
    use super::*;
    use crate::Prime65521 as P;
    fn reference(left:F,right:F,addend:F)->F {
        let mut result=addend.coefficients().map(|x| x.as_u16() as u64);
        for row in 0..8 { for col in 0..8 {
            result[(row+col)%8] += left.coefficients()[row].as_u16() as u64 * right.coefficients()[col].as_u16() as u64 * if row+col>=8 {17} else {1};
        }}
        F::new(result.map(|x| P::new((x%65521) as u32)))
    }
    fn check(multiply:unsafe fn(&mut [F],&[F]), add:unsafe fn(&mut [F],&[F],F), fold:unsafe fn(&mut [F],F)) {
        let mut seed=719732u64;
        let mut random=|| {seed^=seed<<13; seed^=seed>>7; seed^=seed<<17; P::new((seed%65521) as u32)};
        for length in [0,1,2,3,4,7,8,9,15,16,17,63,64,65,1025] {
            let mut left=(0..length+2).map(|_|F::new(core::array::from_fn(|_|random()))).collect::<Vec<_>>();
            let mut right=(0..length+2).map(|_|F::new(core::array::from_fn(|_|random()))).collect::<Vec<_>>();
            let edges=[0,1,14,15,32760,32761,65519,65520];
            for index in 0..left.len().min(9) {
                left[index]=F::new(core::array::from_fn(|lane|P::new(edges[(index+lane)%8])));
                right[index]=F::new(core::array::from_fn(|lane|P::new(edges[(7+index-lane%8)%8])));
            }
            let scalar=F::new([P::new(65520);8]);
            let mut actual=left.clone();
            unsafe {multiply(&mut actual[1..length+1],&right[1..length+1]);}
            for index in 1..length+1 {assert_eq!(actual[index],reference(left[index],right[index],F::zero()));}
            assert_eq!(actual[0],left[0]); assert_eq!(actual[length+1],left[length+1]);
            let mut actual=left.clone();
            unsafe {add(&mut actual[1..length+1],&right[1..length+1],scalar);}
            for index in 1..length+1 {assert_eq!(actual[index],reference(right[index],scalar,left[index]));}
            let pairs=(0..2*length+2).map(|_|F::new(core::array::from_fn(|_|random()))).collect::<Vec<_>>();
            let mut actual=pairs.clone();
            unsafe {fold(&mut actual[1..2*length+1],scalar);}
            for index in 0..length {assert_eq!(actual[1+index],reference(pairs[2*index+2],scalar,pairs[2*index+1]));}
            assert_eq!(actual[0],pairs[0]); assert_eq!(actual[2*length+1],pairs[2*length+1]);
        }
    }
    #[test]
    fn octic_products_and_folds_match_independent_convolution() {
        if std::is_x86_feature_detected!("avx2") {check(multiply256,add_scaled256,fold_adjacent256);}
        if std::is_x86_feature_detected!("avx512f") {check(multiply512,add_scaled512,fold_adjacent512);}
    }
}
