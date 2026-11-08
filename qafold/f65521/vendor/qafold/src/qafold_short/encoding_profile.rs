//! Optional coarse QA encoding instrumentation. The disabled recorder has no
//! storage and performs no clock reads in builds without `profile-encoding`.

#[cfg(feature = "profile-encoding")]
use std::time::{Duration, Instant};

/// Coarse timings and operation counts for one complete base-message encoding.
///
/// `total` is elapsed wall time. The other durations sum the work of parity
/// workers, including any nested parallelism for code index greater than two.
/// Consequently, their sum need not be bounded by `total`; subtract them only
/// when the encoding has no overlapping parity workers (for example, index two).
#[cfg(feature = "profile-encoding")]
#[derive(Clone, Debug, Default)]
pub struct QaEncodeProfile {
    /// Wall time including input checks and profiling bookkeeping.
    pub total: Duration,
    /// WHT calls, including cache-stripe gather/scatter and scratch allocation.
    pub wht: Duration,
    /// The pointwise multiplication of the spectrum by generator coefficients.
    pub multiplication: Duration,
    /// Witness-to-output and systematic-message-to-spectrum copies.
    pub copy: Duration,
    /// Fused copy + complete WHT; not included in the three fields above.
    pub copy_wht: Duration,
    /// Fused prepared multiplication + complete WHT.
    pub multiplication_wht: Duration,
    pub wht_calls: usize,
    pub multiplications: usize,
    pub copied_elements: usize,
}

pub(super) enum EncodeStage {
    Wht,
    Multiplication,
    Copy,
    CopyWht,
    MultiplyWht,
}

#[derive(Default)]
pub(super) struct EncodeRecorder<const ENABLED: bool> {
    #[cfg(feature = "profile-encoding")]
    profile: QaEncodeProfile,
}

impl<const ENABLED: bool> EncodeRecorder<ENABLED> {
    #[inline(always)]
    pub(super) fn measure<T>(
        &mut self,
        stage: EncodeStage,
        elements: usize,
        operation: impl FnOnce() -> T,
    ) -> T {
        #[cfg(feature = "profile-encoding")]
        if ENABLED {
            let started = Instant::now();
            let result = operation();
            let elapsed = started.elapsed();
            match stage {
                EncodeStage::Wht => {
                    self.profile.wht += elapsed;
                    self.profile.wht_calls += 1;
                }
                EncodeStage::Multiplication => {
                    self.profile.multiplication += elapsed;
                    self.profile.multiplications += elements;
                }
                EncodeStage::Copy => {
                    self.profile.copy += elapsed;
                    self.profile.copied_elements += elements;
                }
                EncodeStage::CopyWht => {
                    self.profile.copy_wht += elapsed;
                    self.profile.copied_elements += elements;
                    self.profile.wht_calls += 1;
                }
                EncodeStage::MultiplyWht => {
                    self.profile.multiplication_wht += elapsed;
                    self.profile.multiplications += elements;
                    self.profile.wht_calls += 1;
                }
            }
            return result;
        }
        let _ = (stage, elements);
        operation()
    }

    #[cfg(any(feature = "profile-encoding", not(feature = "parallel")))]
    #[inline(always)]
    pub(super) fn merge(&mut self, child: Self) {
        #[cfg(feature = "profile-encoding")]
        if ENABLED {
            self.profile.wht += child.profile.wht;
            self.profile.multiplication += child.profile.multiplication;
            self.profile.copy += child.profile.copy;
            self.profile.copy_wht += child.profile.copy_wht;
            self.profile.multiplication_wht += child.profile.multiplication_wht;
            self.profile.wht_calls += child.profile.wht_calls;
            self.profile.multiplications += child.profile.multiplications;
            self.profile.copied_elements += child.profile.copied_elements;
        }
        let _ = child;
    }

    #[cfg(feature = "profile-encoding")]
    pub(super) fn into_profile(self) -> QaEncodeProfile {
        self.profile
    }
}

#[cfg(all(test, feature = "profile-encoding"))]
mod tests {
    use super::super::{ShortBaseField, ShortQaCode};
    use crate::{FullPrime65521Ext8, Prime65521};
    use rand::{SeedableRng, rngs::StdRng};

    fn exercise<B: ShortBaseField>() {
        let mut rng = StdRng::seed_from_u64(0x7072_6f66_696c_6551);
        for log_n in [0, 6, 12] {
            let n = 1 << log_n;
            for index in [2, 4] {
                for systematic in [false, true] {
                    let code = if systematic {
                        ShortQaCode::<B>::sample_systematic(n, index, [41; 32]).unwrap()
                    } else {
                        let mut columns = (0..index)
                            .map(|_| {
                                (0..n)
                                    .map(|_| B::sample_uniform(&mut rng))
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        columns[0].fill(B::zero());
                        columns[0][0] = B::from_u64(2);
                        ShortQaCode::<B>::from_group_algebra(columns).unwrap()
                    };
                    let message = (0..n)
                        .map(|_| B::sample_uniform(&mut rng))
                        .collect::<Vec<_>>();
                    let mut expected = vec![B::zero(); n * index];
                    code.encode_base_into(&message, &mut expected).unwrap();
                    let mut actual = vec![B::one(); n * index];
                    let profile = code
                        .encode_base_into_profiled(&message, &mut actual)
                        .unwrap();
                    assert_eq!(actual, expected);
                    assert_eq!(profile.wht_calls, index + usize::from(!systematic));
                    assert_eq!(
                        profile.multiplications,
                        n * (index - usize::from(systematic))
                    );
                    assert_eq!(profile.copied_elements, n * (1 + usize::from(systematic)));
                    // Only this branch has no nested parity worker overlap.
                    if index == 2 && systematic {
                        assert!(
                            profile.total >= profile.wht + profile.multiplication + profile.copy
                        );
                    }
                    assert!(
                        code.encode_base_into_profiled(&message[..n - 1], &mut actual)
                            .is_err()
                    );
                    assert!(
                        code.encode_base_into_profiled(&message, &mut actual[..n * index - 1])
                            .is_err()
                    );
                }
            }
        }
    }

    #[test]
    fn profiled_encoding_matches_normal_path_and_counts() {
        #[cfg(feature = "parallel")]
        for threads in [1, 4] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    exercise::<Prime65521>();
                    exercise::<FullPrime65521Ext8>();
                });
        }
        #[cfg(not(feature = "parallel"))]
        {
            exercise::<Prime65521>();
            exercise::<FullPrime65521Ext8>();
        }
    }

    #[test]
    fn f65521_fused_profile_preserves_operation_counts() {
        use crate::FullPrime65521Ext8;
        use crate::ShortFieldElement;
        let n = 1 << 14;
        let code = ShortQaCode::<FullPrime65521Ext8>::sample_systematic(n, 2, [41; 32]).unwrap();
        let input = (0..n).map(|i| FullPrime65521Ext8::from_u64(i as u64)).collect::<Vec<_>>();
        let mut output = vec![FullPrime65521Ext8::zero(); n * 2];
        let profile = code.encode_base_into_profiled(&input, &mut output).unwrap();
        assert_eq!(profile.wht_calls, 2);
        assert_eq!(profile.multiplications, n);
        assert_eq!(profile.copied_elements, 2 * n);
        assert!(profile.copy_wht > std::time::Duration::ZERO);
        assert!(profile.multiplication_wht > std::time::Duration::ZERO);
        assert_eq!(profile.wht, std::time::Duration::ZERO);
        assert_eq!(profile.multiplication, std::time::Duration::ZERO);
        assert!(profile.total >= profile.copy + profile.copy_wht + profile.multiplication_wht);
    }
}
