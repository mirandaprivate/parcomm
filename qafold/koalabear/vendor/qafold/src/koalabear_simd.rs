//! Canonical KoalaBear^4 kernels: no representation or transcript changes.
//! Each 128-bit group is one polynomial [1,X,X^2,X^3], X^4=3.
use super::KoalaBearExt4 as F;
const _: () = assert!(core::mem::size_of::<F>() == 16);
const _: () = assert!(core::mem::align_of::<F>() == 4);

pub(super) fn butterfly(left: &mut [F], right: &mut [F]) {
    assert_eq!(left.len(), right.len());
    #[cfg(target_arch = "x86_64")]
    // SAFETY: the disjoint slices contain canonical u32 coefficients, and CPU
    // features are checked before entering each target-specific kernel.
    unsafe {
        if left.len() >= 4 && std::is_x86_feature_detected!("avx512f") {
            return butterfly512(left, right);
        }
        if left.len() >= 2 && std::is_x86_feature_detected!("avx2") {
            return butterfly256(left, right);
        }
    }
    for (left, right) in left.iter_mut().zip(right) {
        let original = *left;
        *left = original + *right;
        *right = original - *right;
    }
}
pub(super) fn multiply(left: &mut [F], right: &[F]) {
    assert_eq!(left.len(), right.len());
    #[cfg(target_arch = "x86_64")]
    // SAFETY: same representation and CPU checks as butterfly; no alignment assumed.
    unsafe {
        if left.len() >= 4 && std::is_x86_feature_detected!("avx512f") {
            return multiply512(left, right);
        }
        if left.len() >= 2 && std::is_x86_feature_detected!("avx2") {
            return multiply256(left, right);
        }
    }
    for (left, right) in left.iter_mut().zip(right) { *left *= *right; }
}
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add256(left: __m256i, right: __m256i) -> __m256i {
    let sum = _mm256_add_epi32(left, right);
    _mm256_min_epu32(sum, _mm256_sub_epi32(sum, _mm256_set1_epi32(2130706433)))
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sub256(left: __m256i, right: __m256i) -> __m256i {
    let difference = _mm256_sub_epi32(left, right);
    _mm256_min_epu32(difference, _mm256_add_epi32(difference, _mm256_set1_epi32(2130706433)))
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn reduce256(product: __m256i) -> __m256i {
    // REDC with -p^-1 mod 2^32 = 0x7effffff; product < p^2.
    // product + correction*p < 2^64, and the shifted result is < 2p.
    let correction = _mm256_mul_epu32(product, _mm256_set1_epi64x(0x7effffff));
    let reduced = _mm256_srli_epi64::<32>(_mm256_add_epi64(product,
        _mm256_mul_epu32(correction, _mm256_set1_epi64x(2130706433))));
    _mm256_min_epu32(reduced, _mm256_sub_epi32(reduced, _mm256_set1_epi64x(2130706433)))
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn prime_mul256<const PREPARED: bool>(left: __m256i, right: __m256i) -> __m256i {
    unsafe {
        let even = reduce256(_mm256_mul_epu32(left, right));
        let odd = reduce256(_mm256_mul_epu32(_mm256_srli_epi64::<32>(left), _mm256_srli_epi64::<32>(right)));
        // A second REDC by R^2 restores canonical (non-Montgomery) output.
        let (even, odd) = if PREPARED { (even, odd) } else { (
            reduce256(_mm256_mul_epu32(even, _mm256_set1_epi64x(402124772))),
            reduce256(_mm256_mul_epu32(odd, _mm256_set1_epi64x(402124772))),
        ) };
        _mm256_or_si256(even, _mm256_slli_epi64::<32>(odd))
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn extension_mul256<const PREPARED: bool>(left: __m256i, right: __m256i) -> __m256i {
    unsafe {
        let triple = add256(left, add256(left, left));
        let rotated1 = _mm256_shuffle_epi32::<0x93>(left);
        let rotated2 = _mm256_shuffle_epi32::<0x4e>(left);
        let rotated3 = _mm256_shuffle_epi32::<0x39>(left);
        let wrapped1 = _mm256_blend_epi32::<17>(rotated1, _mm256_shuffle_epi32::<0x93>(triple));
        let wrapped2 = _mm256_blend_epi32::<51>(rotated2, _mm256_shuffle_epi32::<0x4e>(triple));
        let wrapped3 = _mm256_blend_epi32::<119>(rotated3, _mm256_shuffle_epi32::<0x39>(triple));
        let product0 = prime_mul256::<PREPARED>(left, _mm256_shuffle_epi32::<0x00>(right));
        let product1 = prime_mul256::<PREPARED>(wrapped1, _mm256_shuffle_epi32::<0x55>(right));
        let product2 = prime_mul256::<PREPARED>(wrapped2, _mm256_shuffle_epi32::<0xaa>(right));
        let product3 = prime_mul256::<PREPARED>(wrapped3, _mm256_shuffle_epi32::<0xff>(right));
        add256(add256(product0, product1), add256(product2, product3))
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn butterfly256(left: &mut [F], right: &mut [F]) {
    unsafe {
        let mut index = 0;
        while index + 2 <= left.len() {
            let original = _mm256_loadu_si256(left.as_ptr().add(index).cast());
            let other = _mm256_loadu_si256(right.as_ptr().add(index).cast());
            _mm256_storeu_si256(left.as_mut_ptr().add(index).cast(), add256(original, other));
            _mm256_storeu_si256(right.as_mut_ptr().add(index).cast(), sub256(original, other));
            index += 2;
        }
        for (left, right) in left[index..].iter_mut().zip(&mut right[index..]) {
            let original = *left; *left = original + *right; *right = original - *right;
        }
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn multiply256(left: &mut [F], right: &[F]) {
    unsafe {
        let mut index = 0;
        while index + 2 <= left.len() {
            let original = _mm256_loadu_si256(left.as_ptr().add(index).cast());
            let other = _mm256_loadu_si256(right.as_ptr().add(index).cast());
            _mm256_storeu_si256(left.as_mut_ptr().add(index).cast(), extension_mul256::<false>(original, other));
            index += 2;
        }
        for (left, right) in left[index..].iter_mut().zip(&right[index..]) { *left *= *right; }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn add512(left: __m512i, right: __m512i) -> __m512i {
    let sum = _mm512_add_epi32(left, right);
    _mm512_min_epu32(sum, _mm512_sub_epi32(sum, _mm512_set1_epi32(2130706433)))
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sub512(left: __m512i, right: __m512i) -> __m512i {
    let difference = _mm512_sub_epi32(left, right);
    _mm512_min_epu32(difference, _mm512_add_epi32(difference, _mm512_set1_epi32(2130706433)))
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn reduce512(product: __m512i) -> __m512i {
    // REDC with -p^-1 mod 2^32 = 0x7effffff; product < p^2.
    // product + correction*p < 2^64, and the shifted result is < 2p.
    let correction = _mm512_mul_epu32(product, _mm512_set1_epi64(0x7effffff));
    let reduced = _mm512_srli_epi64::<32>(_mm512_add_epi64(product,
        _mm512_mul_epu32(correction, _mm512_set1_epi64(2130706433))));
    _mm512_min_epu32(reduced, _mm512_sub_epi32(reduced, _mm512_set1_epi64(2130706433)))
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn prime_mul512<const PREPARED: bool>(left: __m512i, right: __m512i) -> __m512i {
    unsafe {
        let even = reduce512(_mm512_mul_epu32(left, right));
        let odd = reduce512(_mm512_mul_epu32(_mm512_srli_epi64::<32>(left), _mm512_srli_epi64::<32>(right)));
        // A second REDC by R^2 restores canonical (non-Montgomery) output.
        let (even, odd) = if PREPARED { (even, odd) } else { (
            reduce512(_mm512_mul_epu32(even, _mm512_set1_epi64(402124772))),
            reduce512(_mm512_mul_epu32(odd, _mm512_set1_epi64(402124772))),
        ) };
        _mm512_or_si512(even, _mm512_slli_epi64::<32>(odd))
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn extension_mul512<const PREPARED: bool>(left: __m512i, right: __m512i) -> __m512i {
    unsafe {
        let triple = add512(left, add512(left, left));
        let rotated1 = _mm512_shuffle_epi32::<0x93>(left);
        let rotated2 = _mm512_shuffle_epi32::<0x4e>(left);
        let rotated3 = _mm512_shuffle_epi32::<0x39>(left);
        let wrapped1 = _mm512_mask_blend_epi32(4369, rotated1, _mm512_shuffle_epi32::<0x93>(triple));
        let wrapped2 = _mm512_mask_blend_epi32(13107, rotated2, _mm512_shuffle_epi32::<0x4e>(triple));
        let wrapped3 = _mm512_mask_blend_epi32(30583, rotated3, _mm512_shuffle_epi32::<0x39>(triple));
        let product0 = prime_mul512::<PREPARED>(left, _mm512_shuffle_epi32::<0x00>(right));
        let product1 = prime_mul512::<PREPARED>(wrapped1, _mm512_shuffle_epi32::<0x55>(right));
        let product2 = prime_mul512::<PREPARED>(wrapped2, _mm512_shuffle_epi32::<0xaa>(right));
        let product3 = prime_mul512::<PREPARED>(wrapped3, _mm512_shuffle_epi32::<0xff>(right));
        add512(add512(product0, product1), add512(product2, product3))
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn butterfly512(left: &mut [F], right: &mut [F]) {
    unsafe {
        let mut index = 0;
        while index + 4 <= left.len() {
            let original = _mm512_loadu_si512(left.as_ptr().add(index).cast());
            let other = _mm512_loadu_si512(right.as_ptr().add(index).cast());
            _mm512_storeu_si512(left.as_mut_ptr().add(index).cast(), add512(original, other));
            _mm512_storeu_si512(right.as_mut_ptr().add(index).cast(), sub512(original, other));
            index += 4;
        }
        for (left, right) in left[index..].iter_mut().zip(&mut right[index..]) {
            let original = *left; *left = original + *right; *right = original - *right;
        }
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn multiply512(left: &mut [F], right: &[F]) {
    unsafe {
        let mut index = 0;
        while index + 4 <= left.len() {
            let original = _mm512_loadu_si512(left.as_ptr().add(index).cast());
            let other = _mm512_loadu_si512(right.as_ptr().add(index).cast());
            _mm512_storeu_si512(left.as_mut_ptr().add(index).cast(), extension_mul512::<false>(original, other));
            index += 4;
        }
        for (left, right) in left[index..].iter_mut().zip(&right[index..]) { *left *= *right; }
    }
}

/// The prepared input contains canonical residues of alpha * 2^32, not F values.
pub(super) fn multiply_prepared(left: &mut [F], canonical: &[F], prepared: &[u32]) {
    assert_eq!(left.len(), canonical.len());
    assert_eq!(prepared.len(), 4 * left.len());
    #[cfg(target_arch = "x86_64")]
    // SAFETY: vector kernels only read initialized, bounded u32 residues;
    // every load is within its slice and all CPU features are checked here.
    unsafe {
        if left.len() >= 4 && std::is_x86_feature_detected!("avx512f") {
            return multiply_prepared512(left, canonical, prepared);
        }
        if left.len() >= 2 && std::is_x86_feature_detected!("avx2") {
            return multiply_prepared256(left, canonical, prepared);
        }
    }
    for (value, alpha) in left.iter_mut().zip(canonical) { *value *= *alpha; }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn multiply_prepared256(left: &mut [F], canonical: &[F], prepared: &[u32]) {
    unsafe {
        let mut index = 0;
        while index + 2 <= left.len() {
            let original = _mm256_loadu_si256(left.as_ptr().add(index).cast());
            let constant = _mm256_loadu_si256(prepared.as_ptr().add(4 * index).cast());
            _mm256_storeu_si256(left.as_mut_ptr().add(index).cast(), extension_mul256::<true>(original, constant));
            index += 2;
        }
        for (value, alpha) in left[index..].iter_mut().zip(&canonical[index..]) { *value *= *alpha; }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn multiply_prepared512(left: &mut [F], canonical: &[F], prepared: &[u32]) {
    unsafe {
        let mut index = 0;
        while index + 4 <= left.len() {
            let original = _mm512_loadu_si512(left.as_ptr().add(index).cast());
            let constant = _mm512_loadu_si512(prepared.as_ptr().add(4 * index).cast());
            _mm512_storeu_si512(left.as_mut_ptr().add(index).cast(), extension_mul512::<true>(original, constant));
            index += 4;
        }
        for (value, alpha) in left[index..].iter_mut().zip(&canonical[index..]) { *value *= *alpha; }
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::{koalabear::KoalaBear as B, ShortBaseField};
    fn schoolbook(left: F, right: F) -> F {
        let mut output = [0u128; 4];
        for first in 0..4 { for second in 0..4 {
            let degree = first + second;
            output[degree % 4] += left.coefficients()[first].to_canonical_u64() as u128
                * right.coefficients()[second].to_canonical_u64() as u128
                * if degree >= 4 { 3 } else { 1 };
        } }
        F::from_coefficients(output.map(|value| B::from_u64((value % 2130706433) as u64)))
    }
    #[test]
    fn prepared_encoding_multipliers_match_independent_convolution() {
        use crate::ShortFieldElement;
        let mut state = 913721u64;
        for length in [0, 1, 2, 3, 4, 7, 8, 9, 31, 64, 4095, 4096, 4097] {
            let mut random = || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; B::from_u64(state) };
            let left = (0..length + 2).map(|_| F::from_coefficients(core::array::from_fn(|_| random()))).collect::<Vec<_>>();
            let mut right = (0..length + 2).map(|_| F::from_coefficients(core::array::from_fn(|_| random()))).collect::<Vec<_>>();
            let edges = [0, 1, 2, 32767, 2130706431, 2130706432];
            for (index, value) in right.iter_mut().take(6).enumerate() {
                *value = F::from_coefficients(core::array::from_fn(|lane| B::from_u64(edges[(index + lane) % edges.len()])));
            }
            let prepared = F::prepare_encoding_multipliers(&right).unwrap();
            let expected = left.iter().zip(&right).map(|(&left, &right)| schoolbook(left, right)).collect::<Vec<_>>();
            let mut kernels: Vec<unsafe fn(&mut [F], &[F], &[u32])> = Vec::new();
            if std::is_x86_feature_detected!("avx2") { kernels.push(multiply_prepared256); }
            if std::is_x86_feature_detected!("avx512f") { kernels.push(multiply_prepared512); }
            for kernel in kernels {
                let mut actual = left.clone();
                unsafe { kernel(&mut actual[1..length + 1], &right[1..length + 1], &prepared.words[4..4 * (length + 1)]); }
                assert_eq!(&actual[1..length + 1], &expected[1..length + 1]);
                assert_eq!(actual[0], left[0]); assert_eq!(actual[length + 1], left[length + 1]);
            }
            let mut actual = left.clone();
            F::mul_assign_prepared_slice(&mut actual[1..length + 1], &right[1..length + 1], &prepared, 1);
            assert_eq!(&actual[1..length + 1], &expected[1..length + 1]);
        }
    }
    fn check(butterfly: unsafe fn(&mut [F], &mut [F]), multiply: unsafe fn(&mut [F], &[F])) {
        let edges = [0u64, 1, 2, 32767, 1073741824, 2130706431, 2130706432];
        let mut state = 71837u64;
        for length in [0,1,2,3,4,7,8,9,16,31,64,1025] {
            let mut random = || {state ^= state << 13; state ^= state >> 7; state ^= state << 17; B::from_u64(state)};
            let left = (0..length+2).map(|_| F::from_coefficients(core::array::from_fn(|_| random()))).collect::<Vec<_>>();
            let mut right = (0..length+2).map(|_| F::from_coefficients(core::array::from_fn(|_| random()))).collect::<Vec<_>>();
            for (index, value) in right.iter_mut().enumerate().take(edges.len()) {
                *value = F::from_coefficients(core::array::from_fn(|lane| B::from_u64(edges[(index+lane)%edges.len()])));
            }
            let expected_mul = left.iter().zip(&right).map(|(x,y)| *x * *y).collect::<Vec<_>>();
            let expected_sum = left.iter().zip(&right).map(|(x,y)| *x + *y).collect::<Vec<_>>();
            let expected_diff = left.iter().zip(&right).map(|(x,y)| *x - *y).collect::<Vec<_>>();
            let mut actual = left.clone();
            unsafe { multiply(&mut actual[1..length+1], &right[1..length+1]); }
            assert_eq!(&actual[1..length+1], &expected_mul[1..length+1]);
            assert_eq!(actual[0], left[0]); assert_eq!(actual[length+1], left[length+1]);
            let mut actual = left.clone(); let mut other = right.clone();
            unsafe { butterfly(&mut actual[1..length+1], &mut other[1..length+1]); }
            assert_eq!(&actual[1..length+1], &expected_sum[1..length+1]);
            assert_eq!(&other[1..length+1], &expected_diff[1..length+1]);
        }
    }
    #[test]
    fn koalabear_simd_matches_scalar() {
        if std::is_x86_feature_detected!("avx2") { check(butterfly256, multiply256); }
        if std::is_x86_feature_detected!("avx512f") { check(butterfly512, multiply512); }
    }
}
