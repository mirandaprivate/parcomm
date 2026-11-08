//! Compact tensor queries. Coordinates and factors are low-bit first.
use crate::{Result, check, field::*, layout::Layout, transcript::Transcript};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Claim {
    pub row: Vec<Scalar>,
    pub col: Vec<Scalar>,
    pub value: Scalar,
}
#[derive(Clone, Debug)]
pub struct Term {
    pub offset: usize,
    pub scale: F,
    pub factors: Vec<[F; 2]>,
}
impl Term {
    pub fn len(&self) -> usize {
        1usize << self.factors.len()
    }
    pub fn weight(&self, index: usize) -> F {
        self.factors
            .iter()
            .enumerate()
            .fold(self.scale, |v, (bit, pair)| v * pair[(index >> bit) & 1])
    }
    pub fn evaluation(&self, point_low: &[F]) -> F {
        let local = self
            .factors
            .iter()
            .zip(point_low)
            .fold(self.scale, |v, (pair, &p)| v * (pair[0] + pair[1] * p));
        local
            * monomial_weight(
                self.offset >> self.factors.len(),
                &point_low[self.factors.len()..],
            )
    }
}
pub struct Query {
    pub terms: Vec<Term>,
    pub claim: F,
}
pub fn weights(factors: &[[F; 2]]) -> Vec<F> {
    let mut result = vec![F::one()];
    for &pair in factors {
        let len = result.len();
        result.resize(2 * len, F::zero());
        let (lower, upper) = result.split_at_mut(len);
        let lower_one = pair[0] == F::one();
        let lower_zero = pair[0] == F::zero();
        let upper_one = pair[1] == F::one();
        let upper_zero = pair[1] == F::zero();
        let fill = |(lo, hi): (&mut F, &mut F)| {
            *hi = if upper_zero {
                F::zero()
            } else if upper_one {
                *lo
            } else {
                *lo * pair[1]
            };
            if lower_zero {
                *lo = F::zero();
            } else if !lower_one {
                *lo *= pair[0];
            }
        };
        if len >= 4096 {
            lower.par_iter_mut().zip(upper).for_each(fill);
        } else {
            lower.iter_mut().zip(upper).for_each(fill);
        }
    }
    result
}

/// The two factors fit in cache; their product is consumed without a full table.
fn tiled_weights(factors: &[[F; 2]], scale: F) -> (Vec<F>, Vec<F>) {
    let split = factors.len().min(10);
    let low = weights(&factors[..split]);
    let mut high = weights(&factors[split..]);
    if scale != F::one() {
        high.par_iter_mut().for_each(|value| *value *= scale);
    }
    (low, high)
}

/// Disjoint, aligned intervals covering [start, end).
fn aligned_intervals(mut start: usize, end: usize) -> Vec<(usize, usize)> {
    let mut result = Vec::new();
    while start < end {
        let fit = (end - start).ilog2();
        let bits = if start == 0 {
            fit
        } else {
            fit.min(start.trailing_zeros())
        };
        let len = 1usize << bits;
        result.push((start, len));
        start += len;
    }
    result
}

/// Remove fixed high coordinates while keeping the same global tensor vector.
fn compact_term(mut term: Term) -> Term {
    while let Some(&pair) = term.factors.last() {
        let fixed = if pair[0] == F::zero() {
            1
        } else if pair[1] == F::zero() {
            0
        } else {
            break;
        };
        let bit = term.factors.len() - 1;
        term.scale *= pair[fixed];
        term.offset += fixed << bit;
        term.factors.pop();
    }
    term
}
pub fn validate_claims(layout: &Layout, claims: &[Claim]) -> Result<()> {
    check(claims.len() == layout.shapes.len(), "claim count mismatch")?;
    for (claim, shape) in claims.iter().zip(&layout.shapes) {
        check(
            claim.row.len() == shape.rows.next_power_of_two().ilog2() as usize
                && claim.col.len() == shape.cols.next_power_of_two().ilog2() as usize,
            "point dimension mismatch",
        )?;
        for value in claim
            .row
            .iter()
            .chain(&claim.col)
            .chain(std::iter::once(&claim.value))
        {
            value.field()?;
        }
    }
    Ok(())
}
/// Generates only a benchmark statement; verify never calls this or reads the witness.
pub fn evaluate_claims(
    layout: &Layout,
    values: &[i8],
    points: &[(Vec<F>, Vec<F>)],
) -> Result<Vec<Claim>> {
    check(
        values.len() == layout.active && points.len() == layout.shapes.len(),
        "claim input dimensions",
    )?;
    let mut claims = points
        .iter()
        .map(|(row, col)| Claim {
            row: row.iter().copied().map(Into::into).collect(),
            col: col.iter().copied().map(Into::into).collect(),
            value: F::zero().into(),
        })
        .collect::<Vec<_>>();
    validate_claims(layout, &claims)?;
    for bucket in &layout.buckets {
        let (row, col) = &points[bucket.matrix];
        let local: Vec<[F; 2]> = col[..bucket.cols.ilog2() as usize]
            .iter()
            .chain(&row[..bucket.rows.ilog2() as usize])
            .map(|&p| [F::one(), p])
            .collect();
        let scale =
            monomial_weight(bucket.row_offset, row) * monomial_weight(bucket.col_offset, col);
        let term = Term {
            offset: bucket.offset,
            scale,
            factors: local,
        };
        // Compact row projection avoids a full field-valued expansion of the BVA.
        let partial = project(&term, values, 16);
        let low = weights(&term.factors[..term.factors.len().min(4)]);
        let value = partial
            .iter()
            .zip(low)
            .fold(F::zero(), |sum, (&entry, weight)| sum + entry * weight);
        claims[bucket.matrix].value = (claims[bucket.matrix].value.field()? + value).into();
    }
    Ok(claims)
}
/// Compile matrix queries and a randomized zero-padding check into one TBV.
pub fn compile(layout: &Layout, claims: &[Claim], transcript: &mut Transcript) -> Result<Query> {
    validate_claims(layout, claims)?;
    let batch_point = (0..claims.len().next_power_of_two().ilog2())
        .map(|_| transcript.challenge(b"batch"))
        .collect::<Vec<_>>();
    let coefficients = (0..claims.len())
        .map(|index| equality_weight(index, &batch_point))
        .collect::<Vec<_>>();
    let mut terms = Vec::new();
    let mut claim_sum = F::zero();
    let points = claims
        .iter()
        .map(|claim| {
            Ok((
                claim
                    .row
                    .iter()
                    .map(|v| v.field())
                    .collect::<Result<Vec<_>>>()?,
                claim
                    .col
                    .iter()
                    .map(|v| v.field())
                    .collect::<Result<Vec<_>>>()?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    for (claim, &coefficient) in claims.iter().zip(&coefficients) {
        claim_sum += coefficient * claim.value.field()?;
    }
    for bucket in &layout.buckets {
        let (row, col) = &points[bucket.matrix];
        let scale = coefficients[bucket.matrix]
            * monomial_weight(bucket.row_offset, row)
            * monomial_weight(bucket.col_offset, col);
        let factors = col[..bucket.cols.ilog2() as usize]
            .iter()
            .chain(&row[..bucket.rows.ilog2() as usize])
            .map(|&p| [F::one(), p])
            .collect();
        terms.push(Term {
            offset: bucket.offset,
            scale,
            factors,
        });
    }
    // The same padding predicate, represented directly on its disjoint support.
    // This avoids scanning all physical entries merely to cancel them again.
    let padding_point = (0..layout.logical.ilog2())
        .map(|_| transcript.challenge(b"padding-point"))
        .collect::<Vec<_>>();
    let mixing = transcript.nonzero(b"padding-mix");
    for bucket in &layout.buckets {
        let col_bits = bucket.cols.ilog2() as usize;
        let bits = bucket.len().ilog2() as usize;
        let mut rectangles = Vec::new();
        for (start, len) in aligned_intervals(bucket.physical_rows, bucket.rows) {
            rectangles.push((start, len, 0, bucket.cols));
        }
        for (row_start, row_len) in aligned_intervals(0, bucket.physical_rows) {
            for (col_start, col_len) in aligned_intervals(bucket.physical_cols, bucket.cols) {
                rectangles.push((row_start, row_len, col_start, col_len));
            }
        }
        for (row_start, row_len, col_start, col_len) in rectangles {
            let mut factors = padding_point[..bits]
                .iter()
                .map(|&point| [F::one(), point])
                .collect::<Vec<_>>();
            for bit in col_len.ilog2() as usize..col_bits {
                factors[bit][1 - ((col_start >> bit) & 1)] = F::zero();
            }
            for bit in row_len.ilog2() as usize..bucket.rows.ilog2() as usize {
                factors[col_bits + bit][1 - ((row_start >> bit) & 1)] = F::zero();
            }
            terms.push(compact_term(Term {
                offset: bucket.offset,
                scale: mixing * monomial_weight(bucket.offset >> bits, &padding_point[bits..]),
                factors,
            }));
        }
    }
    for (offset, len) in aligned_intervals(layout.active, layout.logical) {
        let bits = len.ilog2() as usize;
        terms.push(Term {
            offset,
            scale: mixing * monomial_weight(offset >> bits, &padding_point[bits..]),
            factors: padding_point[..bits]
                .iter()
                .map(|&point| [F::one(), point])
                .collect(),
        });
    }
    Ok(Query {
        terms,
        claim: claim_sum,
    })
}

/// Cache-tiled projection. Each tile sums at most 1024 signed values per lane
/// in i64 (KoalaBear/i16: magnitude < 2^56); only global merging needs i64.
pub fn project(term: &Term, values: &[i8], stride: usize) -> Vec<F> {
    let low_bits = (stride.ilog2() as usize).min(term.factors.len());
    let lanes = 1usize << low_bits;
    if term.offset >= values.len() || term.scale == F::zero() {
        return vec![F::zero(); lanes];
    }
    let (low, high) = tiled_weights(&term.factors[low_bits..], term.scale);
    let row_count = (values.len() - term.offset)
        .div_ceil(lanes)
        .min(term.len() / lanes);
    let sums = high[..row_count.div_ceil(low.len())]
        .par_iter()
        .enumerate()
        .map(|(tile, &prefix)| {
            let mut acc = vec![[0i64; 8]; lanes];
            if prefix == F::zero() {
                return acc;
            }
            let rows = low.len().min(row_count - tile * low.len());
            for (row, &suffix) in low[..rows].iter().enumerate() {
                if suffix == F::zero() {
                    continue;
                }
                let weight = prefix * suffix;
                let coefficients: [i64; 8] = std::array::from_fn(|index| {
                    weight.coefficients()[index].to_canonical_u64() as i64
                });
                let index = term.offset + (tile * low.len() + row) * lanes;
                for lane in 0..lanes.min(values.len() - index) {
                    let value = values[index + lane] as i64;
                    for (sum, coefficient) in acc[lane].iter_mut().zip(coefficients) {
                        *sum += value * coefficient;
                    }
                }
            }
            acc
        })
        .fold(
            || vec![[0i64; 8]; lanes],
            |mut total, tile| {
                for (total, tile) in total.iter_mut().zip(tile) {
                    for (total, value) in total.iter_mut().zip(tile) {
                        *total += value as i64;
                    }
                }
                total
            },
        )
        .reduce(
            || vec![[0i64; 8]; lanes],
            |mut left, right| {
                for (left, right) in left.iter_mut().zip(right) {
                    for (left, right) in left.iter_mut().zip(right) {
                        *left += right;
                    }
                }
                left
            },
        );
    sums.iter()
        .map(|sum| {
            F::from_coefficients(sum.map(|value| B::from_u64(value.rem_euclid(65521) as u64)))
        })
        .collect()
}

impl Query {
    pub fn evaluation(&self, point_low: &[F]) -> F {
        self.terms
            .iter()
            .map(|term| term.evaluation(point_low))
            .fold(F::zero(), |x, y| x + y)
    }
    pub fn cross_terms(&self, values: &[i8], stride: usize) -> (Vec<F>, Vec<F>) {
        let mut positive = vec![F::zero(); stride - 1];
        let mut negative = positive.clone();
        for term in &self.terms {
            if term.len() >= stride {
                let projections = project(term, values, stride);
                let low = weights(&term.factors[..stride.ilog2() as usize]);
                for distance in 1..stride {
                    for lane in 0..stride - distance {
                        positive[distance - 1] += projections[lane] * low[lane + distance];
                        negative[distance - 1] += projections[lane + distance] * low[lane];
                    }
                }
            } else {
                let start = term.offset / stride * stride;
                let mut query = vec![F::zero(); stride];
                for index in 0..term.len() {
                    query[term.offset - start + index] = term.weight(index);
                }
                for distance in 1..stride {
                    for lane in 0..stride - distance {
                        positive[distance - 1] += query[lane + distance]
                            .mul_prime(small(values.get(start + lane).copied().unwrap_or(0)));
                        negative[distance - 1] += query[lane].mul_prime(small(
                            values.get(start + lane + distance).copied().unwrap_or(0),
                        ));
                    }
                }
            }
        }
        (positive, negative)
    }
    pub fn fold(&self, logical: usize, stride: usize, challenge: F) -> Vec<F> {
        let mut result = vec![F::zero(); logical / stride];
        let powers = std::iter::successors(Some(F::one()), |p| Some(*p * challenge))
            .take(stride)
            .collect::<Vec<_>>();
        for term in &self.terms {
            if term.len() >= stride {
                let low = weights(&term.factors[..stride.ilog2() as usize]);
                let scale = term.scale
                    * low
                        .iter()
                        .zip(&powers)
                        .fold(F::zero(), |v, (&x, &y)| v + x * y);
                let (low, high) = tiled_weights(&term.factors[stride.ilog2() as usize..], scale);
                let start = term.offset / stride;
                result[start..start + term.len() / stride]
                    .par_chunks_mut(low.len())
                    .zip(high.par_iter())
                    .for_each(|(output, &prefix)| {
                        if prefix != F::zero() {
                            for (value, &suffix) in output.iter_mut().zip(&low) {
                                if suffix != F::zero() {
                                    *value += prefix * suffix;
                                }
                            }
                        }
                    });
            } else {
                for index in 0..term.len() {
                    result[term.offset / stride] +=
                        term.weight(index) * powers[(term.offset + index) % stride];
                }
            }
        }
        result
    }
}
