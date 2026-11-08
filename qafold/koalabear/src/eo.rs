//! Packed Laurent reduction followed by ordinary even/odd folds, over KoalaBear^4.
use crate::{Result, check, field::*, tbv::Query, transcript::Transcript};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EoTimings {
    pub tbv_cross_terms_seconds: f64,
    pub packed_witness_seconds: f64,
    pub folded_tbv_seconds: f64,
    pub consistency_check_seconds: f64,
    pub binary_eo_seconds: f64,
}

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
// Each worker folds within a disjoint tile; ordered copy_within then compacts
// those tile prefixes. No overlapping parallel reads/writes and no second table.
fn fold_in_place(values: &mut Vec<F>, challenge: F) {
    const TILE: usize = 4096;
    let half = values.len() / 2;
    values.par_chunks_mut(TILE).for_each(|tile| {
        for index in 0..tile.len() / 2 {
            tile[index] = tile[2 * index] + tile[2 * index + 1] * challenge;
        }
    });
    for start in (TILE..values.len()).step_by(TILE) {
        let count = TILE.min(values.len() - start) / 2;
        values.copy_within(start..start + count, start / 2);
    }
    values.truncate(half);
}

pub fn prove(
    values: &[i16],
    logical: usize,
    stride: usize,
    query: &Query,
    transcript: &mut Transcript,
) -> Result<(EoProof, Terminal, EoTimings)> {
    let mut timings = EoTimings::default();
    let stage = Instant::now();
    let (positive, negative) = query.cross_terms(values, stride);
    timings.tbv_cross_terms_seconds = stage.elapsed().as_secs_f64();
    eprintln!(
        "stage=tbv_cross_terms seconds={}",
        timings.tbv_cross_terms_seconds
    );
    let stage = Instant::now();
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
            let mut acc = [0i64; 4];
            for (lane, power) in powers.iter().enumerate() {
                let value = values.get(start + lane).copied().unwrap_or(0) as i64;
                for (sum, coefficient) in acc.iter_mut().zip(power.coefficients()) {
                    *sum += value * coefficient.to_canonical_u64() as i64;
                }
            }
            F::from_coefficients(acc.map(|value| B::from_u64(value.rem_euclid(2130706433) as u64)))
        })
        .collect::<Vec<_>>();
    timings.packed_witness_seconds = stage.elapsed().as_secs_f64();
    let stage = Instant::now();
    let mut query_values = query.fold(logical, stride, challenge);
    timings.folded_tbv_seconds = stage.elapsed().as_secs_f64();
    eprintln!("stage=folded_tbv seconds={}", timings.folded_tbv_seconds);
    let stage = Instant::now();
    let mut claim = update_claim(query.claim, &positive, &negative, challenge)?;
    check(
        dot(&witness, &query_values) == claim,
        "packed reduction claim mismatch",
    )?;
    timings.consistency_check_seconds = stage.elapsed().as_secs_f64();
    let stage = Instant::now();
    let mut rounds = Vec::new();
    while witness.len() > 1 {
        let cross = witness
            .par_chunks_exact(2)
            .zip(query_values.par_chunks_exact(2))
            .map(|(values, query)| [values[0] * query[1], values[1] * query[0]])
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
    timings.binary_eo_seconds = stage.elapsed().as_secs_f64();
    Ok((
        EoProof {
            positive,
            negative,
            rounds,
            value: witness[0].into(),
            query: query_values[0].into(),
        },
        terminal,
        timings,
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
