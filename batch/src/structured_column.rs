//! Succinct descriptions of the public relation tables.
//!
//! Every verifier-side public operand is a piecewise regular column: a short
//! list of segments whose entries follow a constant, row-index, geometric, or
//! offset-bit-product rule. The verifier evaluates the column MLE at any
//! point in `O(segments * log^2)` field operations instead of folding the
//! full table, so public tables never contribute linear verification work.

use ark_bn254::Fr as ArkFr;
use ark_ff::{One, Zero};
use ark_serialize::CanonicalSerialize;
use jolt_field::Fr;
use sha2::{Digest, Sha256};

/// The entry rule of one segment, indexed by the local row inside the segment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ColumnPattern {
    /// Every entry equals `value`.
    Constant(ArkFr),
    /// Entry `local` equals `start + local`.
    RowIndex { start: ArkFr },
    /// Entry `local` equals `scale * ratio^local`.
    Geometric { scale: ArkFr, ratio: ArkFr },
    /// Entry `local` equals the product of the bit-basis coordinates selected
    /// by the binary digits of `offset0 + local * 2^stride_log`.
    OffsetBitProduct { offset0: u64, stride_log: u32 },
}

/// One run of rows sharing a single entry rule.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ColumnSegment {
    pub len: usize,
    pub pattern: ColumnPattern,
}

/// A power-of-two column described by consecutive regular segments.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StructuredColumn {
    /// Total power-of-two length; segments must cover it exactly.
    pub padded_len: usize,
    pub segments: Vec<ColumnSegment>,
    /// Public coordinates consumed by [`ColumnPattern::OffsetBitProduct`].
    pub bit_basis: Vec<ArkFr>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StructuredColumnError {
    InvalidShape(String),
}

impl std::fmt::Display for StructuredColumnError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidShape(message) => {
                write!(formatter, "invalid structured column: {message}")
            }
        }
    }
}

impl std::error::Error for StructuredColumnError {}

impl StructuredColumn {
    /// A column made of the listed segments followed by an implicit zero tail.
    pub fn with_zero_tail(
        padded_len: usize,
        mut segments: Vec<ColumnSegment>,
        bit_basis: Vec<ArkFr>,
    ) -> Result<Self, StructuredColumnError> {
        let covered: usize = segments.iter().map(|segment| segment.len).sum();
        if covered > padded_len {
            return Err(StructuredColumnError::InvalidShape(format!(
                "segments cover {covered} rows but the column has {padded_len}"
            )));
        }
        if covered < padded_len {
            segments.push(ColumnSegment {
                len: padded_len - covered,
                pattern: ColumnPattern::Constant(ArkFr::zero()),
            });
        }
        let column = Self {
            padded_len,
            segments,
            bit_basis,
        };
        column.validate()?;
        Ok(column)
    }

    pub fn constant(padded_len: usize, value: ArkFr) -> Result<Self, StructuredColumnError> {
        Self::with_zero_tail(
            padded_len,
            vec![ColumnSegment {
                len: padded_len,
                pattern: ColumnPattern::Constant(value),
            }],
            Vec::new(),
        )
    }

    pub fn validate(&self) -> Result<(), StructuredColumnError> {
        if self.padded_len < 2 || !self.padded_len.is_power_of_two() {
            return Err(StructuredColumnError::InvalidShape(format!(
                "column length {} is not a power of two of at least 2",
                self.padded_len
            )));
        }
        let covered: usize = self.segments.iter().map(|segment| segment.len).sum();
        if covered != self.padded_len {
            return Err(StructuredColumnError::InvalidShape(format!(
                "segments cover {covered} of {} rows",
                self.padded_len
            )));
        }
        if self.segments.iter().any(|segment| segment.len == 0) {
            return Err(StructuredColumnError::InvalidShape(
                "empty segments are not allowed".to_owned(),
            ));
        }
        let num_vars = self.padded_len.trailing_zeros();
        for segment in &self.segments {
            if let ColumnPattern::OffsetBitProduct {
                offset0,
                stride_log,
            } = segment.pattern
            {
                let largest = offset0 as u128 + ((segment.len as u128 - 1) << stride_log);
                if largest >= 1_u128 << self.bit_basis.len() {
                    return Err(StructuredColumnError::InvalidShape(format!(
                        "offset {largest} exceeds the {}-coordinate bit basis",
                        self.bit_basis.len()
                    )));
                }
                let _ = num_vars;
            }
        }
        Ok(())
    }

    pub fn num_vars(&self) -> usize {
        self.padded_len.trailing_zeros() as usize
    }

    /// The dense entries, used by the prover and by tests.
    pub fn materialize(&self) -> Vec<ArkFr> {
        let mut values = Vec::with_capacity(self.padded_len);
        for segment in &self.segments {
            match segment.pattern {
                ColumnPattern::Constant(value) => {
                    values.extend(std::iter::repeat_n(value, segment.len));
                }
                ColumnPattern::RowIndex { start } => {
                    let mut current = start;
                    for _ in 0..segment.len {
                        values.push(current);
                        current += ArkFr::one();
                    }
                }
                ColumnPattern::Geometric { scale, ratio } => {
                    let mut current = scale;
                    for _ in 0..segment.len {
                        values.push(current);
                        current *= ratio;
                    }
                }
                ColumnPattern::OffsetBitProduct {
                    offset0,
                    stride_log,
                } => {
                    for local in 0..segment.len as u64 {
                        let offset = offset0 + (local << stride_log);
                        let mut product = ArkFr::one();
                        for (bit, basis) in self.bit_basis.iter().enumerate() {
                            if (offset >> bit) & 1 == 1 {
                                product *= basis;
                            }
                        }
                        values.push(product);
                    }
                }
            }
        }
        values
    }

    /// A transcript digest of the description, shared by both parties.
    pub fn digest(&self) -> Result<[u8; 32], StructuredColumnError> {
        let mut hasher = Sha256::new();
        hasher.update(b"parcomm-structured-column-v1");
        hasher.update((self.padded_len as u64).to_le_bytes());
        hasher.update((self.segments.len() as u64).to_le_bytes());
        let mut encoded = Vec::new();
        let mut absorb =
            |hasher: &mut Sha256, value: &ArkFr| -> Result<(), StructuredColumnError> {
                encoded.clear();
                value
                    .serialize_compressed(&mut encoded)
                    .map_err(|error| StructuredColumnError::InvalidShape(error.to_string()))?;
                hasher.update(&encoded);
                Ok(())
            };
        for segment in &self.segments {
            hasher.update((segment.len as u64).to_le_bytes());
            match &segment.pattern {
                ColumnPattern::Constant(value) => {
                    hasher.update([0_u8]);
                    absorb(&mut hasher, value)?;
                }
                ColumnPattern::RowIndex { start } => {
                    hasher.update([1_u8]);
                    absorb(&mut hasher, start)?;
                }
                ColumnPattern::Geometric { scale, ratio } => {
                    hasher.update([2_u8]);
                    absorb(&mut hasher, scale)?;
                    absorb(&mut hasher, ratio)?;
                }
                ColumnPattern::OffsetBitProduct {
                    offset0,
                    stride_log,
                } => {
                    hasher.update([3_u8]);
                    hasher.update(offset0.to_le_bytes());
                    hasher.update(stride_log.to_le_bytes());
                }
            }
        }
        hasher.update((self.bit_basis.len() as u64).to_le_bytes());
        for basis in &self.bit_basis {
            absorb(&mut hasher, basis)?;
        }
        Ok(hasher.finalize().into())
    }

    /// Evaluate the column MLE at a most-significant-bit-first point.
    ///
    /// Matches folding the dense column with `bind_one` per point coordinate:
    /// `point[0]` binds the most significant row-index bit.
    pub fn mle_eval(&self, point: &[Fr]) -> Result<Fr, StructuredColumnError> {
        self.validate()?;
        if point.len() != self.num_vars() {
            return Err(StructuredColumnError::InvalidShape(format!(
                "point has {} coordinates but the column has {} variables",
                point.len(),
                self.num_vars()
            )));
        }
        let num_vars = self.num_vars();
        // challenge_for_bit[bit] weights row-index bit `bit` (LSB order).
        let challenge_for_bit = (0..num_vars)
            .map(|bit| point[num_vars - 1 - bit])
            .collect::<Vec<_>>();
        let bit_basis = self
            .bit_basis
            .iter()
            .copied()
            .map(Fr::from)
            .collect::<Vec<_>>();
        let mut total = Fr::zero();
        let mut cursor = 0_usize;
        for segment in &self.segments {
            let begin = cursor;
            let end = cursor + segment.len;
            cursor = end;
            if matches!(segment.pattern, ColumnPattern::Constant(value) if value.is_zero()) {
                continue;
            }
            for (block_base, block_log) in aligned_blocks(begin, end) {
                let eq_fixed = fixed_bit_factor(block_base, block_log, &challenge_for_bit);
                let local_base = (block_base - begin) as u64;
                let contribution = match &segment.pattern {
                    ColumnPattern::Constant(value) => Fr::from(*value),
                    ColumnPattern::RowIndex { start } => {
                        let mut weighted_low = Fr::zero();
                        for bit in 0..block_log {
                            weighted_low +=
                                Fr::from(ArkFr::from(1_u64 << bit)) * challenge_for_bit[bit];
                        }
                        Fr::from(*start) + Fr::from(ArkFr::from(local_base)) + weighted_low
                    }
                    ColumnPattern::Geometric { scale, ratio } => {
                        let ratio = Fr::from(*ratio);
                        let mut factor = Fr::from(*scale) * power_u64(ratio, local_base);
                        let mut ratio_power = ratio;
                        for bit in 0..block_log {
                            factor *= Fr::one() - challenge_for_bit[bit]
                                + challenge_for_bit[bit] * ratio_power;
                            ratio_power = ratio_power * ratio_power;
                        }
                        factor
                    }
                    ColumnPattern::OffsetBitProduct {
                        offset0,
                        stride_log,
                    } => offset_bit_product_block(
                        offset0 + (local_base << stride_log),
                        *stride_log,
                        block_log,
                        &challenge_for_bit,
                        &bit_basis,
                    ),
                };
                total += eq_fixed * contribution;
            }
        }
        Ok(total)
    }
}

/// Split `[begin, end)` into maximal aligned dyadic blocks `(base, log_len)`.
fn aligned_blocks(begin: usize, end: usize) -> Vec<(usize, usize)> {
    let mut blocks = Vec::new();
    let mut base = begin;
    while base < end {
        let alignment = if base == 0 {
            usize::BITS - 1
        } else {
            base.trailing_zeros()
        };
        let remaining_log = (end - base).ilog2();
        let block_log = alignment.min(remaining_log) as usize;
        blocks.push((base, block_log));
        base += 1 << block_log;
    }
    blocks
}

/// The eq-polynomial factor over the fixed high bits of one aligned block.
fn fixed_bit_factor(block_base: usize, block_log: usize, challenge_for_bit: &[Fr]) -> Fr {
    let mut factor = Fr::one();
    for (bit, &challenge) in challenge_for_bit.iter().enumerate().skip(block_log) {
        if (block_base >> bit) & 1 == 1 {
            factor *= challenge;
        } else {
            factor *= Fr::one() - challenge;
        }
    }
    factor
}

fn power_u64(base: Fr, mut exponent: u64) -> Fr {
    let mut result = Fr::one();
    let mut square = base;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result *= square;
        }
        square = square * square;
        exponent >>= 1;
    }
    result
}

/// Sum of `eq(low bits) * prod_{bit set in first + low * 2^stride} basis_bit`
/// over the `2^block_log` free low rows of one aligned block.
///
/// A carry-aware digit scan handles the addition `first + (low << stride)`:
/// scanning offset bits `stride..stride + block_log` keeps one carry state,
/// then the two carry outcomes each contribute a fixed high-bit product.
fn offset_bit_product_block(
    first: u64,
    stride: u32,
    block_log: usize,
    challenge_for_bit: &[Fr],
    bit_basis: &[Fr],
) -> Fr {
    // Bits of `first` below the stride are unaffected by the free rows.
    let mut low_fixed = Fr::one();
    for bit in 0..stride.min(bit_basis.len() as u32) {
        if (first >> bit) & 1 == 1 {
            low_fixed *= bit_basis[bit as usize];
        }
    }
    // carry_weight[carry] accumulates the scanned-digit weight.
    let mut carry_weight = [Fr::one(), Fr::zero()];
    for row_bit in 0..block_log {
        let offset_bit = stride as usize + row_bit;
        let first_bit = ((first >> offset_bit) & 1) as u8;
        let select_zero = Fr::one() - challenge_for_bit[row_bit];
        let select_one = challenge_for_bit[row_bit];
        let basis = bit_basis.get(offset_bit).copied().unwrap_or_else(Fr::zero);
        let mut next = [Fr::zero(), Fr::zero()];
        for carry in 0..2_u8 {
            if carry_weight[carry as usize].is_zero() {
                continue;
            }
            for row_digit in 0..2_u8 {
                let sum = first_bit + row_digit + carry;
                let out_bit = sum & 1;
                let out_carry = sum >> 1;
                let select = if row_digit == 1 {
                    select_one
                } else {
                    select_zero
                };
                let factor = if out_bit == 1 { basis } else { Fr::one() };
                next[out_carry as usize] += carry_weight[carry as usize] * select * factor;
            }
        }
        carry_weight = next;
    }
    // The untouched high bits of `first`, adjusted by the outgoing carry.
    let mut result = Fr::zero();
    for carry in 0..2_u64 {
        if carry_weight[carry as usize].is_zero() {
            continue;
        }
        let high = (first >> (stride as usize + block_log)) + carry;
        let mut high_product = Fr::one();
        for bit in 0..u64::BITS as usize {
            if (high >> bit) & 1 == 1 {
                let offset_bit = stride as usize + block_log + bit;
                high_product *= bit_basis.get(offset_bit).copied().unwrap_or_else(Fr::zero);
            }
        }
        result += carry_weight[carry as usize] * high_product;
    }
    low_fixed * result
}

#[cfg(test)]
mod tests {
    use ark_ff::UniformRand;
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    use super::*;

    fn fold_dense(values: &[ArkFr], point: &[Fr]) -> Fr {
        let mut current = values.iter().copied().map(Fr::from).collect::<Vec<_>>();
        for &challenge in point {
            let half = current.len() / 2;
            for index in 0..half {
                let low = current[index];
                let high = current[index + half];
                current[index] = low + challenge * (high - low);
            }
            current.truncate(half);
        }
        current[0]
    }

    fn random_point(rng: &mut StdRng, num_vars: usize) -> Vec<Fr> {
        (0..num_vars).map(|_| Fr::from(ArkFr::rand(rng))).collect()
    }

    #[test]
    fn every_pattern_matches_the_dense_fold() {
        let mut rng = StdRng::seed_from_u64(7);
        let bit_basis = (0..10).map(|_| ArkFr::rand(&mut rng)).collect::<Vec<_>>();
        let ratio = ArkFr::rand(&mut rng);
        for padded_log in [3_usize, 5, 7] {
            let padded_len = 1 << padded_log;
            let cut = padded_len / 3 + 1;
            let second = padded_len / 4 + 1;
            let column = StructuredColumn::with_zero_tail(
                padded_len,
                vec![
                    ColumnSegment {
                        len: cut,
                        pattern: ColumnPattern::RowIndex {
                            start: ArkFr::from(1_u64),
                        },
                    },
                    ColumnSegment {
                        len: second,
                        pattern: ColumnPattern::Geometric {
                            scale: ArkFr::from(3_u64),
                            ratio,
                        },
                    },
                    ColumnSegment {
                        len: 1,
                        pattern: ColumnPattern::Constant(ArkFr::from(11_u64)),
                    },
                ],
                bit_basis.clone(),
            )
            .unwrap();
            let dense = column.materialize();
            assert_eq!(dense.len(), padded_len);
            for _ in 0..4 {
                let point = random_point(&mut rng, padded_log);
                assert_eq!(column.mle_eval(&point).unwrap(), fold_dense(&dense, &point),);
            }
        }
    }

    #[test]
    fn offset_bit_products_cover_unaligned_strided_offsets() {
        let mut rng = StdRng::seed_from_u64(11);
        let bit_basis = (0..12).map(|_| ArkFr::rand(&mut rng)).collect::<Vec<_>>();
        // Mixed segments mirroring bucket offset tables: differing strides and
        // deliberately unaligned starting offsets.
        let column = StructuredColumn::with_zero_tail(
            64,
            vec![
                ColumnSegment {
                    len: 13,
                    pattern: ColumnPattern::OffsetBitProduct {
                        offset0: 320,
                        stride_log: 4,
                    },
                },
                ColumnSegment {
                    len: 22,
                    pattern: ColumnPattern::OffsetBitProduct {
                        offset0: 528,
                        stride_log: 2,
                    },
                },
                ColumnSegment {
                    len: 9,
                    pattern: ColumnPattern::OffsetBitProduct {
                        offset0: 616,
                        stride_log: 0,
                    },
                },
            ],
            bit_basis,
        )
        .unwrap();
        let dense = column.materialize();
        let mut rng = StdRng::seed_from_u64(13);
        for _ in 0..8 {
            let point = random_point(&mut rng, 6);
            assert_eq!(column.mle_eval(&point).unwrap(), fold_dense(&dense, &point),);
        }
    }

    #[test]
    fn digest_distinguishes_descriptions() {
        let constant = StructuredColumn::constant(8, ArkFr::from(5_u64)).unwrap();
        let other = StructuredColumn::constant(8, ArkFr::from(6_u64)).unwrap();
        let longer = StructuredColumn::constant(16, ArkFr::from(5_u64)).unwrap();
        assert_ne!(constant.digest().unwrap(), other.digest().unwrap());
        assert_ne!(constant.digest().unwrap(), longer.digest().unwrap());
    }
}
