//! Contiguous coefficient butterflies for F_65521[X]/(X^8-17).
//!
//! Every input/output coefficient is canonical. A 16-bit wrapping sum needs a
//! subtraction of p when the exact sum >= p, including overflow past 2^16.
//! Saturating addition preserves that comparison. A wrapping difference needs
//! addition of p iff a < b. Neither operation uses a lazy field representation.

use super::FullPrime65521Ext8 as F;

// The cast below relies only on the three repr(transparent) wrappers around
// [u16; 8]; their size/alignment are checked independently at compile time.
const _: () = assert!(core::mem::size_of::<F>() == 16);
const _: () = assert!(core::mem::align_of::<F>() == core::mem::align_of::<u16>());

#[inline]
pub(super) fn butterfly(left: &mut [F], right: &mut [F]) {
    assert_eq!(left.len(), right.len());
    #[cfg(target_arch = "x86_64")]
    {
        // Safe slices are disjoint, equally sized, and contain only canonical
        // limbs. Every kernel loads/stores whole elements within these bounds.
        unsafe {
            if left.len() >= 8
                && std::is_x86_feature_detected!("avx512f")
                && std::is_x86_feature_detected!("avx512bw")
            {
                return avx512(left.as_mut_ptr(), right.as_mut_ptr(), left.len());
            }
            if left.len() >= 4 && std::is_x86_feature_detected!("avx2") {
                return avx2(left.as_mut_ptr(), right.as_mut_ptr(), left.len());
            }
            sse2(left.as_mut_ptr(), right.as_mut_ptr(), left.len());
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    for (left, right) in left.iter_mut().zip(right) {
        let a = *left;
        let b = *right;
        *left = a + b;
        *right = a - b;
    }
}

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// SSE2 is available on every x86-64 target, including the one-element case.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn sse2(left: *mut F, right: *mut F, len: usize) {
    unsafe {
        let p = _mm_set1_epi16(65_521u16 as i16);
        let zero = _mm_setzero_si128();
        for i in 0..len {
            let a = _mm_loadu_si128(left.add(i).cast());
            let b = _mm_loadu_si128(right.add(i).cast());
            let sum = _mm_add_epi16(a, b);
            // p - saturating(a+b) saturates to zero exactly when a+b >= p.
            let reduce = _mm_cmpeq_epi16(_mm_subs_epu16(p, _mm_adds_epu16(a, b)), zero);
            let sum = _mm_sub_epi16(sum, _mm_and_si128(reduce, p));
            let no_borrow = _mm_cmpeq_epi16(_mm_subs_epu16(b, a), zero);
            let difference = _mm_add_epi16(_mm_sub_epi16(a, b), _mm_andnot_si128(no_borrow, p));
            _mm_storeu_si128(left.add(i).cast(), sum);
            _mm_storeu_si128(right.add(i).cast(), difference);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2(left: *mut F, right: *mut F, len: usize) {
    unsafe {
        let p = _mm256_set1_epi16(65_521u16 as i16);
        let zero = _mm256_setzero_si256();
        let mut i = 0;
        while i + 2 <= len {
            let a = _mm256_loadu_si256(left.add(i).cast());
            let b = _mm256_loadu_si256(right.add(i).cast());
            let sum = _mm256_add_epi16(a, b);
            let reduce = _mm256_cmpeq_epi16(_mm256_subs_epu16(p, _mm256_adds_epu16(a, b)), zero);
            let sum = _mm256_sub_epi16(sum, _mm256_and_si256(reduce, p));
            let no_borrow = _mm256_cmpeq_epi16(_mm256_subs_epu16(b, a), zero);
            let difference =
                _mm256_add_epi16(_mm256_sub_epi16(a, b), _mm256_andnot_si256(no_borrow, p));
            _mm256_storeu_si256(left.add(i).cast(), sum);
            _mm256_storeu_si256(right.add(i).cast(), difference);
            i += 2;
        }
        sse2(left.add(i), right.add(i), len - i);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn avx512(left: *mut F, right: *mut F, len: usize) {
    unsafe {
        let p = _mm512_set1_epi16(65_521u16 as i16);
        let mut i = 0;
        while i + 4 <= len {
            let a = _mm512_loadu_si512(left.add(i).cast());
            let b = _mm512_loadu_si512(right.add(i).cast());
            let sum = _mm512_add_epi16(a, b);
            let reduce = _mm512_cmp_epu16_mask::<_MM_CMPINT_NLT>(_mm512_adds_epu16(a, b), p);
            let sum = _mm512_mask_sub_epi16(sum, reduce, sum, p);
            let borrow = _mm512_cmp_epu16_mask::<_MM_CMPINT_LT>(a, b);
            let difference = _mm512_sub_epi16(a, b);
            let difference = _mm512_mask_add_epi16(difference, borrow, difference, p);
            _mm512_storeu_si512(left.add(i).cast(), sum);
            _mm512_storeu_si512(right.add(i).cast(), difference);
            i += 4;
        }
        sse2(left.add(i), right.add(i), len - i);
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::Prime65521 as P;

    fn check(kernel: unsafe fn(*mut F, *mut F, usize)) {
        for len in [0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 64, 1025] {
            // Offset by one element to test 16-byte misalignment for YMM/ZMM.
            let mut a = vec![F::new([P::new(0); 8]); len + 2];
            let mut b = a.clone();
            let edge = [0u32, 1, 14, 15, 32_760, 32_761, 65_519, 65_520];
            for (i, (a, b)) in a[1..len + 1].iter_mut().zip(&mut b[1..len + 1]).enumerate() {
                *a = F::new(core::array::from_fn(|j| P::new(edge[(i + j) % 8])));
                *b = F::new(core::array::from_fn(|j| P::new(edge[(i / 8 + j * 3) % 8])));
            }
            let expected: Vec<_> = a.iter().zip(&b).map(|(a, b)| (*a + *b, *a - *b)).collect();
            // Only the requested interior is accessible to the kernel.
            unsafe {
                kernel(a.as_mut_ptr().add(1), b.as_mut_ptr().add(1), len);
            }
            for i in 1..len + 1 {
                assert_eq!((a[i], b[i]), expected[i]);
                assert!(
                    a[i].coefficients()
                        .iter()
                        .chain(b[i].coefficients())
                        .all(|x| x.as_u16() < 65_521)
                );
            }
            assert_eq!(a[0], F::new([P::new(0); 8]));
            assert_eq!(a[len + 1], F::new([P::new(0); 8]));
            assert_eq!(b[0], F::new([P::new(0); 8]));
            assert_eq!(b[len + 1], F::new([P::new(0); 8]));
        }
    }

    #[test]
    fn butterfly_simd_matches_scalar_at_carry_and_borrow_boundaries() {
        check(sse2);
        eprintln!("validated canonical butterfly kernel: SSE2");
        if std::is_x86_feature_detected!("avx2") {
            check(avx2);
            eprintln!("validated canonical butterfly kernel: AVX2");
        }
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
            check(avx512);
            eprintln!("validated canonical butterfly kernel: AVX512BW");
        }
    }
}
