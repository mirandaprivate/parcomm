//! Packed Laurent reduction followed by ordinary even/odd folds, over F65521^8.
use crate::{Result, check, field::*, tbv::Query, transcript::Transcript};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EoProof {
    pub positive: Vec<Scalar>,
    pub negative: Vec<Scalar>,
    pub rounds: Vec<[Scalar; 2]>,
    pub value: Scalar,
    pub query: Scalar,
}
pub struct Terminal {
    pub value: F,
    pub point_msb: Vec<F>,
    pub scale: F,
}
fn absorb_packed(
    transcript: &mut Transcript,
    positive: &[Scalar],
    negative: &[Scalar],
) -> Result<()> {
    transcript.object(b"packed-positive", &positive)?;
    transcript.object(b"packed-negative", &negative)
}
fn update_claim(claim: F, positive: &[Scalar], negative: &[Scalar], challenge: F) -> Result<F> {
    let inverse = inverse(challenge)?;
    let mut forward = challenge;
    let mut backward = inverse;
    let mut value = claim;
    for (&pos, &neg) in positive.iter().zip(negative) {
        value += pos.field()? * forward + neg.field()? * backward;
        forward *= challenge;
        backward *= inverse;
    }
    Ok(value)
}
fn normalized(challenges: &[F], value: F) -> Result<Terminal> {
    let mut point = Vec::new();
    let mut scale = F::one();
    for &challenge in challenges {
        let monomial = inverse(challenge)?;
        let factor = F::one() + monomial;
        point.push(monomial * inverse(factor)?);
        scale *= factor;
    }
    point.reverse();
    Ok(Terminal {
        point_msb: point,
        value,
        scale,
    })
}
// Each worker folds a disjoint tile. Doubling waves compact its prefixes:
// each wave writes entirely below its read range, with no second full table.
fn fold_in_place(values: &mut Vec<F>, challenge: F) {
    const TILE: usize = 4096;
    let half = values.len() / 2;
    values.par_chunks_mut(TILE).for_each(|tile| {
        F::fold_adjacent_in_place(tile, challenge);
    });
    let mut start = TILE;
    while start < values.len() {
        let end = (start * 2).min(values.len());
        let (lower, upper) = values.split_at_mut(start);
        let output = &mut lower[start / 2..end / 2];
        let input = &upper[..end - start];
        if input.len() >= 65536 {
            output.par_chunks_mut(TILE / 2).zip(input.par_chunks(TILE))
                .for_each(|(output, input)| output.copy_from_slice(&input[..output.len()]));
        } else {
            output.chunks_mut(TILE / 2).zip(input.chunks(TILE))
                .for_each(|(output, input)| output.copy_from_slice(&input[..output.len()]));
        }
        start *= 2;
    }
    values.truncate(half);
}

#[cfg(test)]
mod cpu_tests {
    use super::*;

    #[test]
    fn tiled_fold_and_compaction_match_scalar_at_wave_boundaries() {
        let challenge = F::from_coefficients(std::array::from_fn(|lane| B::from_u64(9137 * lane as u64 + 137)));
        for length in [0, 2, 14, 16, 18, 4094, 4096, 4098, 8190, 8192, 8194, 65534, 65536, 65538, 131072, 131074, 262144] {
            let mut values = (0..length).map(|index| F::from_coefficients(std::array::from_fn(|lane| B::from_u64((index * 739 + lane * 2311) as u64)))).collect::<Vec<_>>();
            let expected = values.chunks_exact(2).map(|pair| pair[0] + pair[1] * challenge).collect::<Vec<_>>();
            fold_in_place(&mut values, challenge);
            assert_eq!(values, expected, "length {length}");
        }
    }
}

pub fn prove(
    values: &[i8],
    logical: usize,
    stride: usize,
    query: &Query,
    transcript: &mut Transcript,
) -> Result<(EoProof, Terminal)> {
    let (positive, negative) = query.cross_terms(values, stride);
    let positive = positive.into_iter().map(Into::into).collect::<Vec<_>>();
    let negative = negative.into_iter().map(Into::into).collect::<Vec<_>>();
    absorb_packed(transcript, &positive, &negative)?;
    let challenge = transcript.fold_challenge(stride.ilog2() as usize);
    let mut challenges = Vec::new();
    let mut power = challenge;
    for _ in 0..stride.ilog2() {
        challenges.push(power);
        power = power * power;
    }
    let inverse = inverse(challenge)?;
    let powers = std::iter::successors(Some(F::one()), |p| Some(*p * inverse))
        .take(stride)
        .collect::<Vec<_>>();
    let mut witness = (0..logical / stride)
        .into_par_iter()
        .map(|row| {
            let start = row * stride;
            let mut acc = [0i64; 8];
            for (lane, power) in powers.iter().enumerate() {
                let value = values.get(start + lane).copied().unwrap_or(0) as i64;
                for (sum, coefficient) in acc.iter_mut().zip(power.coefficients()) {
                    *sum += value * coefficient.to_canonical_u64() as i64;
                }
            }
            F::from_coefficients(acc.map(|value| B::from_u64(value.rem_euclid(65521) as u64)))
        })
        .collect::<Vec<_>>();
    let mut query_values = query.fold(logical, stride, challenge);
    let mut claim = update_claim(query.claim, &positive, &negative, challenge)?;
    check(
        dot(&witness, &query_values) == claim,
        "packed reduction claim mismatch",
    )?;
    let mut rounds = Vec::new();
    while witness.len() > 1 {
        let cross = witness
            .par_chunks(2048)
            .zip(query_values.par_chunks(2048))
            .map(|(values, query)| {
                let mut product = values.to_vec();
                let swapped = query.chunks_exact(2).flat_map(|pair| [pair[1], pair[0]]).collect::<Vec<_>>();
                F::mul_assign_slice(&mut product, &swapped);
                product.chunks_exact(2).fold([F::zero(); 2], |sum, pair| [sum[0] + pair[0], sum[1] + pair[1]])
            })
            .reduce(
                || [F::zero(); 2],
                |left, right| [left[0] + right[0], left[1] + right[1]],
            );
        let round = [cross[0].into(), cross[1].into()];
        transcript.object(b"binary-cross", &round)?;
        let challenge = transcript.fold_challenge(1);
        let inv = crate::field::inverse(challenge)?;
        claim += cross[0] * challenge + cross[1] * inv;
        challenges.push(challenge);
        fold_in_place(&mut witness, inv);
        fold_in_place(&mut query_values, challenge);
        rounds.push(round);
    }
    check(
        claim == witness[0] * query_values[0] && query.evaluation(&challenges) == query_values[0],
        "EO terminal mismatch",
    )?;
    transcript.scalar(b"terminal-value", witness[0]);
    transcript.scalar(b"terminal-query", query_values[0]);
    let terminal = normalized(&challenges, witness[0])?;
    Ok((
        EoProof {
            positive,
            negative,
            rounds,
            value: witness[0].into(),
            query: query_values[0].into(),
        },
        terminal,
    ))
}
pub fn verify(
    proof: &EoProof,
    logical: usize,
    stride: usize,
    query: &Query,
    transcript: &mut Transcript,
) -> Result<Terminal> {
    check(
        proof.positive.len() == stride - 1
            && proof.negative.len() == stride - 1
            && proof.rounds.len() == (logical / stride).ilog2() as usize,
        "EO proof dimensions",
    )?;
    absorb_packed(transcript, &proof.positive, &proof.negative)?;
    let challenge = transcript.fold_challenge(stride.ilog2() as usize);
    let mut claim = update_claim(query.claim, &proof.positive, &proof.negative, challenge)?;
    let mut challenges = Vec::new();
    let mut power = challenge;
    for _ in 0..stride.ilog2() {
        challenges.push(power);
        power = power * power;
    }
    for round in &proof.rounds {
        transcript.object(b"binary-cross", round)?;
        let challenge = transcript.fold_challenge(1);
        claim += round[0].field()? * challenge + round[1].field()? * inverse(challenge)?;
        challenges.push(challenge);
    }
    let value = proof.value.field()?;
    let query_value = proof.query.field()?;
    check(claim == value * query_value, "EO scalar relation rejected")?;
    check(
        query.evaluation(&challenges) == query_value,
        "TBV terminal evaluation rejected",
    )?;
    transcript.scalar(b"terminal-value", value);
    transcript.scalar(b"terminal-query", query_value);
    normalized(&challenges, value)
}
