use ark_ec::{
    pairing::{Pairing, PairingOutput},
    ScalarMul, VariableBaseMSM,
};
use ark_ff::{Field, PrimeField, UniformRand};
use ark_poly_commit::smart_pc::{
    data_structures::{NativeFoldMessage, Trans, UniversalParams},
    sub_protocols, utils as pcutils, FiatShamir,
};
use ark_std::{ops::Mul, Zero};
use rayon::prelude::*;
use std::convert::Infallible;
use std::fmt::{Display, Formatter};

use crate::protocols::bucketed_ip::{BucketedVector, BvaLayout, TensorBasisVector};
pub use crate::protocols::bucketed_ip::{BvaFieldSource, BvaI32Source, BvaI8Source};
use crate::utils::heterogeneous::{HeteroBlockMat, HeteroMatrixError};

/// A rewindable source for the physical columns of a heterogeneous stacked
/// matrix. Implementations must visit every stored block column exactly once.
/// Missing blocks, short column suffixes, and unused domain slots are logical
/// zero padding and must not be passed to `visit`.
pub trait StackedI32Source {
    type Error;

    fn domain(&self) -> crate::utils::heterogeneous::PaddedDomain;

    fn for_each_block_column(
        &mut self,
        visit: &mut dyn FnMut(usize, usize, &[i32]),
    ) -> Result<(), Self::Error>;
}

/// Rewindable sparse field source for a BVA whose omitted *real* entries are
/// zero. Entries must be emitted once in strictly increasing
/// `(vector_index, entry_index)` order. This is used for Lasso multiplicity
/// vectors where visiting a multi-million-row public table would defeat the
/// RAM goal even though only a small number of counts are nonzero.
pub trait SparseBvaFieldSource<F: PrimeField> {
    type Error;

    fn layout(&self) -> &BvaLayout;

    fn for_each_nonzero(
        &mut self,
        visit: &mut dyn FnMut(usize, usize, F),
    ) -> Result<(), Self::Error>;
}

impl StackedI32Source for HeteroBlockMat<i32> {
    type Error = Infallible;

    fn domain(&self) -> crate::utils::heterogeneous::PaddedDomain {
        self.domain
    }

    fn for_each_block_column(
        &mut self,
        visit: &mut dyn FnMut(usize, usize, &[i32]),
    ) -> Result<(), Self::Error> {
        for (slot, block) in self.blocks.iter().enumerate() {
            let Some(block) = block else { continue };
            for (column, values) in block.columns.iter().enumerate() {
                visit(slot, column, values);
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StreamingCommitProfile {
    pub physical_entries: usize,
    pub nonzero_entries: usize,
    pub max_live_scalars: usize,
    pub materialized_padding_entries: usize,
    pub padding_scalar_multiplications: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StreamingOpenProfile {
    pub source_passes: usize,
    pub physical_entries: usize,
    pub nonzero_entries: usize,
    pub real_projection_field_multiplications: usize,
    pub materialized_projection_padding_entries: usize,
    pub padding_projection_field_multiplications: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BvaStreamingCommitProfile {
    pub physical_entries: usize,
    pub nonzero_entries: usize,
    pub logical_entries: usize,
    pub prepared_base_entries: usize,
    pub parallel_vectors: usize,
    pub max_live_group_contributions: usize,
    pub materialized_padding_entries: usize,
    pub padding_group_operations: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BvaStreamingInnerProductProfile {
    pub source_passes: usize,
    pub physical_entries: usize,
    pub maximum_live_vector_entries: usize,
    pub real_field_operations: u64,
    pub materialized_padding_entries: usize,
    pub padding_field_operations: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BvaStreamingInnerProduct<F> {
    pub value: F,
    pub profile: BvaStreamingInnerProductProfile,
}

#[derive(Debug)]
pub enum StreamingVirtualPcsError<SourceError> {
    Pcs(VirtualPcsError),
    Source(SourceError),
}

impl<SourceError> From<VirtualPcsError> for StreamingVirtualPcsError<SourceError> {
    fn from(error: VirtualPcsError) -> Self {
        Self::Pcs(error)
    }
}

impl<SourceError: Display> Display for StreamingVirtualPcsError<SourceError> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pcs(error) => Display::fmt(error, formatter),
            Self::Source(error) => Display::fmt(error, formatter),
        }
    }
}

impl<SourceError> std::error::Error for StreamingVirtualPcsError<SourceError> where
    SourceError: std::error::Error + 'static
{
}

pub fn commit_stacked_i32_streaming<E: Pairing, S: StackedI32Source>(
    pp: &UniversalParams<E>,
    source: &mut S,
    hiding_factor: E::ScalarField,
    bit_width: usize,
) -> Result<
    (PairingOutput<E>, Vec<E::G1>, StreamingCommitProfile),
    StreamingVirtualPcsError<S::Error>,
> {
    if bit_width == 0 || bit_width > 32 {
        return Err(VirtualPcsError::InvalidBitWidth(bit_width).into());
    }

    let domain = source.domain();
    let (packed_rows, packed_cols) = packed_stacked_shape(domain);
    if packed_rows > pp.vec_g.len() || packed_cols > pp.vec_h.len() {
        return Err(VirtualPcsError::ParametersTooSmall {
            packed_rows,
            packed_cols,
            q: pp.q,
        }
        .into());
    }

    let original_cols = domain.col_cap;
    let mut tier_one = vec![E::G1::zero(); packed_cols];
    let mut profile = StreamingCommitProfile::default();
    let mut pcs_error = None;
    source
        .for_each_block_column(&mut |slot, column, values| {
            if pcs_error.is_some() {
                return;
            }
            if slot >= domain.slots {
                pcs_error = Some(VirtualPcsError::SourceBlockOutOfDomain {
                    slot,
                    slots: domain.slots,
                });
                return;
            }
            if column >= original_cols {
                pcs_error = Some(VirtualPcsError::SourceColumnOutOfDomain {
                    column,
                    col_cap: original_cols,
                });
                return;
            }
            if values.len() > domain.row_cap {
                pcs_error = Some(VirtualPcsError::SourceColumnExceedsRowCap {
                    rows: values.len(),
                    row_cap: domain.row_cap,
                });
                return;
            }
            for &value in values {
                let magnitude = value.unsigned_abs();
                if bit_width < 32 && magnitude >= (1_u32 << bit_width) {
                    pcs_error = Some(VirtualPcsError::ValueOutOfRange { value, bit_width });
                    return;
                }
            }

            profile.physical_entries += values.len();
            profile.nonzero_entries += values.iter().filter(|&&value| value != 0).count();
            let row_offset = slot * domain.row_cap;
            if packed_cols >= original_cols {
                let row_group = packed_cols / original_cols;
                let offset_mod = row_offset % row_group;
                for row_mod in 0..row_group {
                    let first_local = (row_mod + row_group - offset_mod) % row_group;
                    if first_local >= values.len() {
                        continue;
                    }
                    let scalars = values[first_local..]
                        .iter()
                        .step_by(row_group)
                        .copied()
                        .map(signed_scalar::<E::ScalarField>)
                        .collect::<Vec<_>>();
                    profile.max_live_scalars = profile.max_live_scalars.max(scalars.len());
                    let first_packed_row = (row_offset + first_local) / row_group;
                    let bases = &pp.vec_g[first_packed_row..first_packed_row + scalars.len()];
                    let packed_col = row_mod * original_cols + column;
                    tier_one[packed_col] += pcutils::msm_g1::<E>(bases, &scalars);
                }
            } else {
                let col_group = original_cols / packed_cols;
                let packed_col = column % packed_cols;
                let col_quotient = column / packed_cols;
                let mut bases = Vec::with_capacity(values.len());
                let mut scalars = Vec::with_capacity(values.len());
                for (local_row, &value) in values.iter().enumerate() {
                    let packed_row = (row_offset + local_row) * col_group + col_quotient;
                    bases.push(pp.vec_g[packed_row]);
                    scalars.push(signed_scalar::<E::ScalarField>(value));
                }
                profile.max_live_scalars = profile.max_live_scalars.max(scalars.len());
                tier_one[packed_col] += pcutils::msm_g1::<E>(&bases, &scalars);
            }
        })
        .map_err(StreamingVirtualPcsError::Source)?;
    if let Some(error) = pcs_error {
        return Err(error.into());
    }

    let commitment = pcutils::inner_pairing_product::<E>(&tier_one, &pp.vec_h[..packed_cols])
        + pp.tilde_u.mul(hiding_factor);
    Ok((commitment, tier_one, profile))
}

pub fn commit_bva_i32_streaming<E: Pairing, S: BvaI32Source>(
    pp: &UniversalParams<E>,
    source: &mut S,
    hiding_factor: E::ScalarField,
    bit_width: usize,
) -> Result<
    (PairingOutput<E>, Vec<E::G1>, BvaStreamingCommitProfile),
    StreamingVirtualPcsError<S::Error>,
> {
    if bit_width == 0 || bit_width > 32 {
        return Err(VirtualPcsError::InvalidBitWidth(bit_width).into());
    }
    let layout = source.layout().clone();
    let logical_entries = layout.logical_len();
    let packed_cols = 1 << (logical_entries.ilog2() as usize / 2);
    let packed_rows = logical_entries / packed_cols;
    if packed_rows > pp.vec_g.len() || packed_cols > pp.vec_h.len() {
        return Err(VirtualPcsError::ParametersTooSmall {
            packed_rows,
            packed_cols,
            q: pp.q,
        }
        .into());
    }
    let prepared_bases = pcutils::prepare_base_short_g1::<E>(&pp.vec_g[..packed_rows], bit_width);
    let bucket_offsets = layout.bucket_offsets();
    let mut tier_one = vec![E::G1::zero(); packed_cols];
    let mut visited = vec![false; layout.vector_to_bucket().len()];
    let mut profile = BvaStreamingCommitProfile {
        logical_entries,
        prepared_base_entries: packed_rows * bit_width,
        ..Default::default()
    };
    let mut pcs_error = None;
    source
        .for_each_vector(&mut |vector_index, values| {
            if pcs_error.is_some() {
                return;
            }
            let Some(&bucket_index) = layout.vector_to_bucket().get(vector_index) else {
                pcs_error = Some(VirtualPcsError::BvaVectorIndex {
                    vector_index,
                    vector_count: visited.len(),
                });
                return;
            };
            if visited[vector_index] {
                pcs_error = Some(VirtualPcsError::DuplicateBvaVector(vector_index));
                return;
            }
            visited[vector_index] = true;
            let bucket_size = layout.bucket_sizes()[bucket_index];
            let expected_len = layout.vector_lengths()[vector_index];
            if values.len() != expected_len {
                pcs_error = Some(VirtualPcsError::BvaVectorLength {
                    vector_index,
                    physical_len: values.len(),
                    expected_len,
                    bucket_size,
                });
                return;
            }
            profile.physical_entries += values.len();
            let logical_offset = bucket_offsets[bucket_index];
            if let Some(&value) = values.iter().find(|&&value| {
                let magnitude = value.unsigned_abs();
                bit_width < 32 && magnitude >= (1_u32 << bit_width)
            }) {
                pcs_error = Some(VirtualPcsError::ValueOutOfRange { value, bit_width });
                return;
            }
            if values.len() >= 1_024 {
                let contributions = values
                    .par_iter()
                    .enumerate()
                    .filter_map(|(local_index, &value)| {
                        bva_short_contribution::<E>(
                            &prepared_bases,
                            bit_width,
                            packed_cols,
                            logical_offset + local_index,
                            value,
                        )
                    })
                    .collect::<Vec<_>>();
                profile.parallel_vectors += 1;
                profile.nonzero_entries += contributions.len();
                profile.max_live_group_contributions = profile
                    .max_live_group_contributions
                    .max(contributions.len());
                for (packed_col, contribution) in contributions {
                    tier_one[packed_col] += contribution;
                }
            } else {
                for (local_index, &value) in values.iter().enumerate() {
                    let Some((packed_col, contribution)) = bva_short_contribution::<E>(
                        &prepared_bases,
                        bit_width,
                        packed_cols,
                        logical_offset + local_index,
                        value,
                    ) else {
                        continue;
                    };
                    profile.nonzero_entries += 1;
                    tier_one[packed_col] += contribution;
                }
            }
        })
        .map_err(StreamingVirtualPcsError::Source)?;
    if let Some(error) = pcs_error {
        return Err(error.into());
    }
    if let Some(vector_index) = visited.iter().position(|visited| !visited) {
        return Err(VirtualPcsError::MissingBvaVector(vector_index).into());
    }

    let commitment = pcutils::inner_pairing_product::<E>(&tier_one, &pp.vec_h[..packed_cols])
        + pp.tilde_u.mul(hiding_factor);
    Ok((commitment, tier_one, profile))
}

/// Commits a byte-backed BVA without widening a complete vector. Each `i8`
/// entry is converted only while its group contribution is accumulated.
pub fn commit_bva_i8_streaming<E: Pairing, S: BvaI8Source>(
    pp: &UniversalParams<E>,
    source: &mut S,
    hiding_factor: E::ScalarField,
    bit_width: usize,
) -> Result<
    (PairingOutput<E>, Vec<E::G1>, BvaStreamingCommitProfile),
    StreamingVirtualPcsError<S::Error>,
> {
    if bit_width == 0 || bit_width > 8 {
        return Err(VirtualPcsError::InvalidBitWidth(bit_width).into());
    }
    let layout = source.layout().clone();
    let logical_entries = layout.logical_len();
    let packed_cols = 1 << (logical_entries.ilog2() as usize / 2);
    let packed_rows = logical_entries / packed_cols;
    if packed_rows > pp.vec_g.len() || packed_cols > pp.vec_h.len() {
        return Err(VirtualPcsError::ParametersTooSmall {
            packed_rows,
            packed_cols,
            q: pp.q,
        }
        .into());
    }
    let prepared_bases = pcutils::prepare_base_short_g1::<E>(&pp.vec_g[..packed_rows], bit_width);
    let bucket_offsets = layout.bucket_offsets();
    let mut tier_one = vec![E::G1::zero(); packed_cols];
    let mut visited = vec![false; layout.vector_to_bucket().len()];
    let mut profile = BvaStreamingCommitProfile {
        logical_entries,
        prepared_base_entries: packed_rows * bit_width,
        ..Default::default()
    };
    let mut pcs_error = None;
    source
        .for_each_vector(&mut |vector_index, values| {
            if pcs_error.is_some() {
                return;
            }
            let Some(&bucket_index) = layout.vector_to_bucket().get(vector_index) else {
                pcs_error = Some(VirtualPcsError::BvaVectorIndex {
                    vector_index,
                    vector_count: visited.len(),
                });
                return;
            };
            if visited[vector_index] {
                pcs_error = Some(VirtualPcsError::DuplicateBvaVector(vector_index));
                return;
            }
            visited[vector_index] = true;
            let bucket_size = layout.bucket_sizes()[bucket_index];
            let expected_len = layout.vector_lengths()[vector_index];
            if values.len() != expected_len {
                pcs_error = Some(VirtualPcsError::BvaVectorLength {
                    vector_index,
                    physical_len: values.len(),
                    expected_len,
                    bucket_size,
                });
                return;
            }
            profile.physical_entries += values.len();
            let logical_offset = bucket_offsets[bucket_index];
            if let Some(&value) = values
                .iter()
                .find(|&&value| bit_width < 8 && value.unsigned_abs() >= (1_u8 << bit_width))
            {
                pcs_error = Some(VirtualPcsError::ValueOutOfRange {
                    value: i32::from(value),
                    bit_width,
                });
                return;
            }
            if values.len() >= 1_024 {
                let contributions = values
                    .par_iter()
                    .enumerate()
                    .filter_map(|(local_index, &value)| {
                        bva_short_contribution::<E>(
                            &prepared_bases,
                            bit_width,
                            packed_cols,
                            logical_offset + local_index,
                            i32::from(value),
                        )
                    })
                    .collect::<Vec<_>>();
                profile.parallel_vectors += 1;
                profile.nonzero_entries += contributions.len();
                profile.max_live_group_contributions = profile
                    .max_live_group_contributions
                    .max(contributions.len());
                for (packed_col, contribution) in contributions {
                    tier_one[packed_col] += contribution;
                }
            } else {
                for (local_index, &value) in values.iter().enumerate() {
                    let Some((packed_col, contribution)) = bva_short_contribution::<E>(
                        &prepared_bases,
                        bit_width,
                        packed_cols,
                        logical_offset + local_index,
                        i32::from(value),
                    ) else {
                        continue;
                    };
                    profile.nonzero_entries += 1;
                    tier_one[packed_col] += contribution;
                }
            }
        })
        .map_err(StreamingVirtualPcsError::Source)?;
    if let Some(error) = pcs_error {
        return Err(error.into());
    }
    if let Some(vector_index) = visited.iter().position(|visited| !visited) {
        return Err(VirtualPcsError::MissingBvaVector(vector_index).into());
    }

    let commitment = pcutils::inner_pairing_product::<E>(&tier_one, &pp.vec_h[..packed_cols])
        + pp.tilde_u.mul(hiding_factor);
    Ok((commitment, tier_one, profile))
}

pub fn commit_bva_field_streaming<E, S>(
    pp: &UniversalParams<E>,
    source: &mut S,
    hiding_factor: E::ScalarField,
) -> Result<
    (PairingOutput<E>, Vec<E::G1>, BvaStreamingCommitProfile),
    StreamingVirtualPcsError<S::Error>,
>
where
    E: Pairing,
    S: BvaFieldSource<E::ScalarField>,
{
    let layout = source.layout().clone();
    let logical_entries = layout.logical_len();
    let packed_cols = 1 << (logical_entries.ilog2() as usize / 2);
    let packed_rows = logical_entries / packed_cols;
    if packed_rows > pp.vec_g.len() || packed_cols > pp.vec_h.len() {
        return Err(VirtualPcsError::ParametersTooSmall {
            packed_rows,
            packed_cols,
            q: pp.q,
        }
        .into());
    }
    let bucket_offsets = layout.bucket_offsets();
    let mut tier_one = vec![E::G1::zero(); packed_cols];
    let mut visited = vec![false; layout.vector_to_bucket().len()];
    let mut profile = BvaStreamingCommitProfile {
        logical_entries,
        ..Default::default()
    };
    let mut pcs_error = None;
    source
        .for_each_vector(&mut |vector_index, values| {
            if pcs_error.is_some() {
                return;
            }
            let Some(&bucket_index) = layout.vector_to_bucket().get(vector_index) else {
                pcs_error = Some(VirtualPcsError::BvaVectorIndex {
                    vector_index,
                    vector_count: visited.len(),
                });
                return;
            };
            if visited[vector_index] {
                pcs_error = Some(VirtualPcsError::DuplicateBvaVector(vector_index));
                return;
            }
            visited[vector_index] = true;
            let expected_len = layout.vector_lengths()[vector_index];
            if values.len() != expected_len {
                pcs_error = Some(VirtualPcsError::BvaVectorLength {
                    vector_index,
                    physical_len: values.len(),
                    expected_len,
                    bucket_size: layout.bucket_sizes()[bucket_index],
                });
                return;
            }
            profile.physical_entries += values.len();
            let logical_offset = bucket_offsets[bucket_index];
            for (local_index, value) in values.iter().enumerate() {
                if value.is_zero() {
                    continue;
                }
                let flat_index = logical_offset + local_index;
                let packed_row = flat_index / packed_cols;
                let packed_col = flat_index % packed_cols;
                tier_one[packed_col] += pp.vec_g[packed_row].mul(*value);
                profile.nonzero_entries += 1;
                profile.max_live_group_contributions = profile.max_live_group_contributions.max(1);
            }
        })
        .map_err(StreamingVirtualPcsError::Source)?;
    if let Some(error) = pcs_error {
        return Err(error.into());
    }
    if let Some(vector_index) = visited.iter().position(|visited| !visited) {
        return Err(VirtualPcsError::MissingBvaVector(vector_index).into());
    }
    let commitment = pcutils::inner_pairing_product::<E>(&tier_one, &pp.vec_h[..packed_cols])
        + pp.tilde_u.mul(hiding_factor);
    Ok((commitment, tier_one, profile))
}

/// Parallel Pippenger commitment for a materialized physical BVA. Each packed
/// column gathers only coordinates that belong to real vectors; logical BVA
/// buckets and short-vector suffixes are never allocated as field elements.
pub fn commit_bva_field_materialized_parallel<E: Pairing>(
    pp: &UniversalParams<E>,
    source: &BucketedVector<E::ScalarField>,
    hiding_factor: E::ScalarField,
) -> Result<(PairingOutput<E>, Vec<E::G1>, BvaStreamingCommitProfile), VirtualPcsError> {
    let layout = source.layout();
    let logical_entries = layout.logical_len();
    let packed_cols = 1 << (logical_entries.ilog2() as usize / 2);
    let packed_rows = logical_entries / packed_cols;
    if packed_rows > pp.vec_g.len() || packed_cols > pp.vec_h.len() {
        return Err(VirtualPcsError::ParametersTooSmall {
            packed_rows,
            packed_cols,
            q: pp.q,
        });
    }
    let bucket_offsets = layout.bucket_offsets();
    let prepared_bases = <E::G1 as ScalarMul>::batch_convert_to_mul_base(&pp.vec_g[..packed_rows]);
    let columns = (0..packed_cols)
        .into_par_iter()
        .map(|packed_col| {
            let mut bases = Vec::with_capacity(packed_rows);
            let mut scalars = Vec::with_capacity(packed_rows);
            for (vector_index, values) in source.vectors().iter().enumerate() {
                let bucket = layout.vector_to_bucket()[vector_index];
                let logical_offset = bucket_offsets[bucket];
                let column_at_start = logical_offset % packed_cols;
                let first_local = (packed_col + packed_cols - column_at_start) % packed_cols;
                for local_index in (first_local..values.len()).step_by(packed_cols) {
                    let value = values[local_index];
                    if value.is_zero() {
                        continue;
                    }
                    let packed_row = (logical_offset + local_index) / packed_cols;
                    bases.push(prepared_bases[packed_row]);
                    scalars.push(value);
                }
            }
            let nonzero = scalars.len();
            (
                <E::G1 as VariableBaseMSM>::msm_unchecked(&bases, &scalars),
                nonzero,
            )
        })
        .collect::<Vec<_>>();
    let mut profile = BvaStreamingCommitProfile {
        logical_entries,
        physical_entries: source.physical_entries(),
        prepared_base_entries: prepared_bases.len(),
        ..Default::default()
    };
    let tier_one = columns
        .into_iter()
        .map(|(commitment, nonzero)| {
            profile.nonzero_entries += nonzero;
            profile.max_live_group_contributions =
                profile.max_live_group_contributions.max(nonzero);
            commitment
        })
        .collect::<Vec<_>>();
    let commitment = pcutils::inner_pairing_product::<E>(&tier_one, &pp.vec_h[..packed_cols])
        + pp.tilde_u.mul(hiding_factor);
    Ok((commitment, tier_one, profile))
}

/// Commits a BVA field vector from its sorted nonzero entries. Omitted real
/// coordinates and logical BVA padding are both represented implicitly, so
/// neither a dense table-count vector nor its power-of-two suffix is
/// allocated.
pub fn commit_sparse_bva_field_streaming<E, S>(
    pp: &UniversalParams<E>,
    source: &mut S,
    hiding_factor: E::ScalarField,
) -> Result<
    (PairingOutput<E>, Vec<E::G1>, BvaStreamingCommitProfile),
    StreamingVirtualPcsError<S::Error>,
>
where
    E: Pairing,
    S: SparseBvaFieldSource<E::ScalarField>,
{
    let layout = source.layout().clone();
    let logical_entries = layout.logical_len();
    let packed_cols = 1 << (logical_entries.ilog2() as usize / 2);
    let packed_rows = logical_entries / packed_cols;
    if packed_rows > pp.vec_g.len() || packed_cols > pp.vec_h.len() {
        return Err(VirtualPcsError::ParametersTooSmall {
            packed_rows,
            packed_cols,
            q: pp.q,
        }
        .into());
    }
    let bucket_offsets = layout.bucket_offsets();
    let mut tier_one = vec![E::G1::zero(); packed_cols];
    let mut profile = BvaStreamingCommitProfile {
        logical_entries,
        physical_entries: layout.vector_lengths().iter().sum(),
        ..Default::default()
    };
    let mut previous = None;
    let mut pcs_error = None;
    source
        .for_each_nonzero(&mut |vector_index, entry_index, value| {
            if pcs_error.is_some() {
                return;
            }
            let coordinate = (vector_index, entry_index);
            if previous.is_some_and(|previous| previous >= coordinate) {
                pcs_error = Some(VirtualPcsError::SparseBvaEntryOrder);
                return;
            }
            previous = Some(coordinate);
            let Some(&bucket_index) = layout.vector_to_bucket().get(vector_index) else {
                pcs_error = Some(VirtualPcsError::SparseBvaEntryIndex {
                    vector_index,
                    entry_index,
                });
                return;
            };
            if entry_index >= layout.vector_lengths()[vector_index] {
                pcs_error = Some(VirtualPcsError::SparseBvaEntryIndex {
                    vector_index,
                    entry_index,
                });
                return;
            }
            if value.is_zero() {
                pcs_error = Some(VirtualPcsError::SparseBvaZeroEntry);
                return;
            }
            let flat_index = bucket_offsets[bucket_index] + entry_index;
            let packed_row = flat_index / packed_cols;
            let packed_col = flat_index % packed_cols;
            tier_one[packed_col] += pp.vec_g[packed_row].mul(value);
            profile.nonzero_entries += 1;
            profile.max_live_group_contributions = 1;
        })
        .map_err(StreamingVirtualPcsError::Source)?;
    if let Some(error) = pcs_error {
        return Err(error.into());
    }
    let commitment = pcutils::inner_pairing_product::<E>(&tier_one, &pp.vec_h[..packed_cols])
        + pp.tilde_u.mul(hiding_factor);
    Ok((commitment, tier_one, profile))
}

/// Evaluates a TBV claim directly against a rewindable BVA source. Only one
/// physical source vector is folded at a time; logical padding buckets and
/// short-vector suffixes never become field elements or field operations.
pub fn inner_product_bva_field_streaming<F, S>(
    source: &mut S,
    tbv: &TensorBasisVector<F>,
) -> Result<BvaStreamingInnerProduct<F>, StreamingVirtualPcsError<S::Error>>
where
    F: PrimeField,
    S: BvaFieldSource<F>,
{
    let layout = source.layout().clone();
    if &layout != tbv.layout() {
        return Err(VirtualPcsError::BvaTbvLayoutMismatch.into());
    }
    let mut visited = vec![false; layout.vector_to_bucket().len()];
    let mut value = F::zero();
    let mut profile = BvaStreamingInnerProductProfile {
        source_passes: 1,
        ..Default::default()
    };
    let mut pcs_error = None;
    source
        .for_each_vector(&mut |vector_index, values| {
            if pcs_error.is_some() {
                return;
            }
            let Some(&bucket_index) = layout.vector_to_bucket().get(vector_index) else {
                pcs_error = Some(VirtualPcsError::BvaVectorIndex {
                    vector_index,
                    vector_count: visited.len(),
                });
                return;
            };
            if visited[vector_index] {
                pcs_error = Some(VirtualPcsError::DuplicateBvaVector(vector_index));
                return;
            }
            visited[vector_index] = true;
            let expected_len = layout.vector_lengths()[vector_index];
            if values.len() != expected_len {
                pcs_error = Some(VirtualPcsError::BvaVectorLength {
                    vector_index,
                    physical_len: values.len(),
                    expected_len,
                    bucket_size: layout.bucket_sizes()[bucket_index],
                });
                return;
            }
            let (contribution, counters) = tbv.project_physical_vector(vector_index, values);
            value += contribution;
            profile.physical_entries += values.len();
            profile.maximum_live_vector_entries =
                profile.maximum_live_vector_entries.max(values.len());
            profile.real_field_operations +=
                counters.real_ip_field_ops + counters.real_fold_field_ops;
        })
        .map_err(StreamingVirtualPcsError::Source)?;
    if let Some(error) = pcs_error {
        return Err(error.into());
    }
    if let Some(vector_index) = visited.iter().position(|visited| !visited) {
        return Err(VirtualPcsError::MissingBvaVector(vector_index).into());
    }
    Ok(BvaStreamingInnerProduct { value, profile })
}

/// Evaluates a TBV directly against byte-backed vectors. No full physical
/// vector is widened to field storage.
pub fn inner_product_bva_i8_streaming<F, S>(
    source: &mut S,
    tbv: &TensorBasisVector<F>,
) -> Result<BvaStreamingInnerProduct<F>, StreamingVirtualPcsError<S::Error>>
where
    F: PrimeField + From<i32>,
    S: BvaI8Source,
{
    let layout = source.layout().clone();
    if &layout != tbv.layout() {
        return Err(VirtualPcsError::BvaTbvLayoutMismatch.into());
    }
    let mut visited = vec![false; layout.vector_to_bucket().len()];
    let mut value = F::zero();
    let mut profile = BvaStreamingInnerProductProfile {
        source_passes: 1,
        ..Default::default()
    };
    let mut pcs_error = None;
    source
        .for_each_vector(&mut |vector_index, values| {
            if pcs_error.is_some() {
                return;
            }
            let Some(&bucket_index) = layout.vector_to_bucket().get(vector_index) else {
                pcs_error = Some(VirtualPcsError::BvaVectorIndex {
                    vector_index,
                    vector_count: visited.len(),
                });
                return;
            };
            if visited[vector_index] {
                pcs_error = Some(VirtualPcsError::DuplicateBvaVector(vector_index));
                return;
            }
            visited[vector_index] = true;
            let expected_len = layout.vector_lengths()[vector_index];
            if values.len() != expected_len {
                pcs_error = Some(VirtualPcsError::BvaVectorLength {
                    vector_index,
                    physical_len: values.len(),
                    expected_len,
                    bucket_size: layout.bucket_sizes()[bucket_index],
                });
                return;
            }
            profile.physical_entries += values.len();
            profile.maximum_live_vector_entries =
                profile.maximum_live_vector_entries.max(values.len());
            let contribution = values
                .par_iter()
                .enumerate()
                .try_fold(
                    || (F::zero(), 0_u64),
                    |(mut sum, operations), (entry_index, &entry)| {
                        if entry == 0 {
                            return Ok((sum, operations));
                        }
                        let coefficient = tbv
                            .physical_coefficient_at(vector_index, entry_index)
                            .map_err(|_| ())?;
                        sum += F::from(i32::from(entry)) * coefficient;
                        Ok((sum, operations + 2))
                    },
                )
                .try_reduce(
                    || (F::zero(), 0_u64),
                    |(left_sum, left_operations), (right_sum, right_operations)| {
                        Ok((left_sum + right_sum, left_operations + right_operations))
                    },
                );
            match contribution {
                Ok((sum, operations)) => {
                    value += sum;
                    profile.real_field_operations += operations;
                }
                Err(()) => {
                    pcs_error = Some(VirtualPcsError::BvaVectorIndex {
                        vector_index,
                        vector_count: visited.len(),
                    });
                }
            }
        })
        .map_err(StreamingVirtualPcsError::Source)?;
    if let Some(error) = pcs_error {
        return Err(error.into());
    }
    if let Some(vector_index) = visited.iter().position(|visited| !visited) {
        return Err(VirtualPcsError::MissingBvaVector(vector_index).into());
    }
    Ok(BvaStreamingInnerProduct { value, profile })
}

/// Evaluates a TBV claim against a sparse BVA source. Only nonzero physical
/// coordinates are visited; omitted public-table rows and logical BVA padding
/// are zero by definition and are never materialized or multiplied.
pub fn inner_product_sparse_bva_field_streaming<F, S>(
    source: &mut S,
    tbv: &TensorBasisVector<F>,
) -> Result<BvaStreamingInnerProduct<F>, StreamingVirtualPcsError<S::Error>>
where
    F: PrimeField,
    S: SparseBvaFieldSource<F>,
{
    let layout = source.layout().clone();
    if &layout != tbv.layout() {
        return Err(VirtualPcsError::BvaTbvLayoutMismatch.into());
    }
    let mut value = F::zero();
    let mut profile = BvaStreamingInnerProductProfile {
        source_passes: 1,
        ..Default::default()
    };
    let mut previous = None;
    let mut pcs_error = None;
    source
        .for_each_nonzero(&mut |vector_index, entry_index, entry| {
            if pcs_error.is_some() {
                return;
            }
            let coordinate = (vector_index, entry_index);
            if previous.is_some_and(|previous| previous >= coordinate) {
                pcs_error = Some(VirtualPcsError::SparseBvaEntryOrder);
                return;
            }
            previous = Some(coordinate);
            if entry.is_zero()
                || vector_index >= layout.vector_lengths().len()
                || entry_index >= layout.vector_lengths()[vector_index]
            {
                pcs_error = Some(if entry.is_zero() {
                    VirtualPcsError::SparseBvaZeroEntry
                } else {
                    VirtualPcsError::SparseBvaEntryIndex {
                        vector_index,
                        entry_index,
                    }
                });
                return;
            }
            match tbv.physical_coefficient_at(vector_index, entry_index) {
                Ok(coefficient) => {
                    value += entry * coefficient;
                    profile.physical_entries += 1;
                    profile.real_field_operations += 2;
                    profile.maximum_live_vector_entries = 1;
                }
                Err(_) => {
                    pcs_error = Some(VirtualPcsError::SparseBvaEntryIndex {
                        vector_index,
                        entry_index,
                    });
                }
            }
        })
        .map_err(StreamingVirtualPcsError::Source)?;
    if let Some(error) = pcs_error {
        return Err(error.into());
    }
    Ok(BvaStreamingInnerProduct { value, profile })
}

#[allow(clippy::too_many_arguments)]
/// Opens the packed SmartPC view of a BVA field source at one MLE point. Both
/// source projections visit only physical vector entries; logical buckets and
/// short-vector suffixes remain metadata-only.
pub fn open_bva_field_streaming<E, S>(
    pp: &UniversalParams<E>,
    source: &mut S,
    xl: &[E::ScalarField],
    xr: &[E::ScalarField],
    value_commitment: PairingOutput<E>,
    matrix_commitment: PairingOutput<E>,
    tier_one: &[E::G1],
    value_hiding: E::ScalarField,
    matrix_hiding: E::ScalarField,
) -> Result<(Trans<E>, StreamingOpenProfile), StreamingVirtualPcsError<S::Error>>
where
    E: Pairing,
    S: BvaFieldSource<E::ScalarField>,
{
    let logical_entries = source.layout().logical_len();
    let packed_cols = 1 << (logical_entries.ilog2() as usize / 2);
    let packed_rows = logical_entries / packed_cols;
    check_opening_inputs(
        pp,
        packed_rows,
        packed_cols,
        xl.len(),
        xr.len(),
        tier_one.len(),
    )?;
    let (value_current, left_profile) =
        project_packed_bva_field_streaming(source, xl, ProjectionSide::Left)?;
    let mut profile = left_profile;
    let proof = open_stacked_i32_from_left_projection::<E, StreamingVirtualPcsError<S::Error>>(
        pp,
        packed_rows,
        packed_cols,
        xl,
        xr,
        value_commitment,
        matrix_commitment,
        tier_one,
        value_hiding,
        matrix_hiding,
        value_current,
        |challenges| {
            let (projection, right_profile) =
                project_packed_bva_field_streaming(source, challenges, ProjectionSide::Right)?;
            profile.source_passes += right_profile.source_passes;
            profile.physical_entries += right_profile.physical_entries;
            profile.nonzero_entries += right_profile.nonzero_entries;
            profile.real_projection_field_multiplications +=
                right_profile.real_projection_field_multiplications;
            Ok(projection)
        },
    )?;
    Ok((proof, profile))
}

/// Opens a byte-backed BVA while widening entries only at the individual
/// projection multiplication that consumes them.
#[allow(clippy::too_many_arguments)]
pub fn open_bva_i8_streaming<E, S>(
    pp: &UniversalParams<E>,
    source: &mut S,
    xl: &[E::ScalarField],
    xr: &[E::ScalarField],
    value_commitment: PairingOutput<E>,
    matrix_commitment: PairingOutput<E>,
    tier_one: &[E::G1],
    value_hiding: E::ScalarField,
    matrix_hiding: E::ScalarField,
) -> Result<(Trans<E>, StreamingOpenProfile), StreamingVirtualPcsError<S::Error>>
where
    E: Pairing,
    E::ScalarField: From<i32>,
    S: BvaI8Source,
{
    let logical_entries = source.layout().logical_len();
    let packed_cols = 1 << (logical_entries.ilog2() as usize / 2);
    let packed_rows = logical_entries / packed_cols;
    check_opening_inputs(
        pp,
        packed_rows,
        packed_cols,
        xl.len(),
        xr.len(),
        tier_one.len(),
    )?;
    let (value_current, left_profile) =
        project_packed_bva_i8_streaming(source, xl, ProjectionSide::Left)?;
    let mut profile = left_profile;
    let proof = open_stacked_i32_from_left_projection::<E, StreamingVirtualPcsError<S::Error>>(
        pp,
        packed_rows,
        packed_cols,
        xl,
        xr,
        value_commitment,
        matrix_commitment,
        tier_one,
        value_hiding,
        matrix_hiding,
        value_current,
        |challenges| {
            let (projection, right_profile) =
                project_packed_bva_i8_streaming(source, challenges, ProjectionSide::Right)?;
            profile.source_passes += right_profile.source_passes;
            profile.physical_entries += right_profile.physical_entries;
            profile.nonzero_entries += right_profile.nonzero_entries;
            profile.real_projection_field_multiplications +=
                right_profile.real_projection_field_multiplications;
            Ok(projection)
        },
    )?;
    Ok((proof, profile))
}

#[allow(clippy::too_many_arguments)]
pub fn open_sparse_bva_field_streaming<E, S>(
    pp: &UniversalParams<E>,
    source: &mut S,
    xl: &[E::ScalarField],
    xr: &[E::ScalarField],
    value_commitment: PairingOutput<E>,
    matrix_commitment: PairingOutput<E>,
    tier_one: &[E::G1],
    value_hiding: E::ScalarField,
    matrix_hiding: E::ScalarField,
) -> Result<(Trans<E>, StreamingOpenProfile), StreamingVirtualPcsError<S::Error>>
where
    E: Pairing,
    S: SparseBvaFieldSource<E::ScalarField>,
{
    let logical_entries = source.layout().logical_len();
    let packed_cols = 1 << (logical_entries.ilog2() as usize / 2);
    let packed_rows = logical_entries / packed_cols;
    check_opening_inputs(
        pp,
        packed_rows,
        packed_cols,
        xl.len(),
        xr.len(),
        tier_one.len(),
    )?;
    let (value_current, left_profile) =
        project_packed_sparse_bva_field_streaming(source, xl, ProjectionSide::Left)?;
    let mut profile = left_profile;
    let proof = open_stacked_i32_from_left_projection::<E, StreamingVirtualPcsError<S::Error>>(
        pp,
        packed_rows,
        packed_cols,
        xl,
        xr,
        value_commitment,
        matrix_commitment,
        tier_one,
        value_hiding,
        matrix_hiding,
        value_current,
        |challenges| {
            let (projection, right_profile) = project_packed_sparse_bva_field_streaming(
                source,
                challenges,
                ProjectionSide::Right,
            )?;
            profile.source_passes += right_profile.source_passes;
            profile.physical_entries += right_profile.physical_entries;
            profile.nonzero_entries += right_profile.nonzero_entries;
            profile.real_projection_field_multiplications +=
                right_profile.real_projection_field_multiplications;
            Ok(projection)
        },
    )?;
    Ok((proof, profile))
}

/// Converts the low-bit-first EO challenges into SmartPC's packed
/// high-bit-first `(row, column)` point convention.
pub fn bva_eo_opening_point<F: Copy>(
    logical_entries: usize,
    challenges_inv: &[F],
) -> Result<(Vec<F>, Vec<F>), VirtualPcsError> {
    if logical_entries == 0 || !logical_entries.is_power_of_two() {
        return Err(VirtualPcsError::InvalidBvaLogicalEntries(logical_entries));
    }
    let expected = logical_entries.ilog2() as usize;
    if challenges_inv.len() != expected {
        return Err(VirtualPcsError::ProjectionPointShape {
            expected,
            actual: challenges_inv.len(),
        });
    }
    let packed_cols = 1_usize << (expected / 2);
    let col_log = packed_cols.ilog2() as usize;
    let xr = challenges_inv[..col_log].iter().rev().copied().collect();
    let xl = challenges_inv[col_log..].iter().rev().copied().collect();
    Ok((xl, xr))
}

fn project_packed_bva_field_streaming<F, S>(
    source: &mut S,
    challenges: &[F],
    side: ProjectionSide,
) -> Result<(Vec<F>, StreamingOpenProfile), StreamingVirtualPcsError<S::Error>>
where
    F: PrimeField,
    S: BvaFieldSource<F>,
{
    let layout = source.layout().clone();
    let logical_entries = layout.logical_len();
    let packed_cols = 1 << (logical_entries.ilog2() as usize / 2);
    let packed_rows = logical_entries / packed_cols;
    let (challenge_axis, output_axis) = match side {
        ProjectionSide::Left => (packed_rows, packed_cols),
        ProjectionSide::Right => (packed_cols, packed_rows),
    };
    let expected_challenges = challenge_axis.ilog2() as usize;
    if challenges.len() != expected_challenges {
        return Err(VirtualPcsError::ProjectionPointShape {
            expected: expected_challenges,
            actual: challenges.len(),
        }
        .into());
    }
    let weights = crate::utils::xi::xi_from_challenges(&challenges.to_vec());
    let bucket_offsets = layout.bucket_offsets();
    let mut projection = vec![F::zero(); output_axis];
    let mut visited = vec![false; layout.vector_to_bucket().len()];
    let mut profile = StreamingOpenProfile {
        source_passes: 1,
        ..Default::default()
    };
    let mut pcs_error = None;
    source
        .for_each_vector(&mut |vector_index, values| {
            if pcs_error.is_some() {
                return;
            }
            let Some(&bucket_index) = layout.vector_to_bucket().get(vector_index) else {
                pcs_error = Some(VirtualPcsError::BvaVectorIndex {
                    vector_index,
                    vector_count: visited.len(),
                });
                return;
            };
            if visited[vector_index] {
                pcs_error = Some(VirtualPcsError::DuplicateBvaVector(vector_index));
                return;
            }
            visited[vector_index] = true;
            let expected_len = layout.vector_lengths()[vector_index];
            if values.len() != expected_len {
                pcs_error = Some(VirtualPcsError::BvaVectorLength {
                    vector_index,
                    physical_len: values.len(),
                    expected_len,
                    bucket_size: layout.bucket_sizes()[bucket_index],
                });
                return;
            }
            let logical_offset = bucket_offsets[bucket_index];
            profile.physical_entries += values.len();
            for (local_index, value) in values.iter().enumerate() {
                if value.is_zero() {
                    continue;
                }
                let flat_index = logical_offset + local_index;
                let packed_row = flat_index / packed_cols;
                let packed_col = flat_index % packed_cols;
                match side {
                    ProjectionSide::Left => {
                        projection[packed_col] += *value * weights[packed_row];
                    }
                    ProjectionSide::Right => {
                        projection[packed_row] += *value * weights[packed_col];
                    }
                }
                profile.nonzero_entries += 1;
                profile.real_projection_field_multiplications += 1;
            }
        })
        .map_err(StreamingVirtualPcsError::Source)?;
    if let Some(error) = pcs_error {
        return Err(error.into());
    }
    if let Some(vector_index) = visited.iter().position(|visited| !visited) {
        return Err(VirtualPcsError::MissingBvaVector(vector_index).into());
    }
    Ok((projection, profile))
}

fn project_packed_bva_i8_streaming<F, S>(
    source: &mut S,
    challenges: &[F],
    side: ProjectionSide,
) -> Result<(Vec<F>, StreamingOpenProfile), StreamingVirtualPcsError<S::Error>>
where
    F: PrimeField + From<i32>,
    S: BvaI8Source,
{
    let layout = source.layout().clone();
    let logical_entries = layout.logical_len();
    let packed_cols = 1 << (logical_entries.ilog2() as usize / 2);
    let packed_rows = logical_entries / packed_cols;
    let (challenge_axis, output_axis) = match side {
        ProjectionSide::Left => (packed_rows, packed_cols),
        ProjectionSide::Right => (packed_cols, packed_rows),
    };
    let expected_challenges = challenge_axis.ilog2() as usize;
    if challenges.len() != expected_challenges {
        return Err(VirtualPcsError::ProjectionPointShape {
            expected: expected_challenges,
            actual: challenges.len(),
        }
        .into());
    }
    let weights = crate::utils::xi::xi_from_challenges(&challenges.to_vec());
    let bucket_offsets = layout.bucket_offsets();
    let mut projection = vec![F::zero(); output_axis];
    let mut visited = vec![false; layout.vector_to_bucket().len()];
    let mut profile = StreamingOpenProfile {
        source_passes: 1,
        ..Default::default()
    };
    let mut pcs_error = None;
    source
        .for_each_vector(&mut |vector_index, values| {
            if pcs_error.is_some() {
                return;
            }
            let Some(&bucket_index) = layout.vector_to_bucket().get(vector_index) else {
                pcs_error = Some(VirtualPcsError::BvaVectorIndex {
                    vector_index,
                    vector_count: visited.len(),
                });
                return;
            };
            if visited[vector_index] {
                pcs_error = Some(VirtualPcsError::DuplicateBvaVector(vector_index));
                return;
            }
            visited[vector_index] = true;
            let expected_len = layout.vector_lengths()[vector_index];
            if values.len() != expected_len {
                pcs_error = Some(VirtualPcsError::BvaVectorLength {
                    vector_index,
                    physical_len: values.len(),
                    expected_len,
                    bucket_size: layout.bucket_sizes()[bucket_index],
                });
                return;
            }
            let logical_offset = bucket_offsets[bucket_index];
            profile.physical_entries += values.len();
            for (local_index, &value) in values.iter().enumerate() {
                if value == 0 {
                    continue;
                }
                let flat_index = logical_offset + local_index;
                let packed_row = flat_index / packed_cols;
                let packed_col = flat_index % packed_cols;
                let value = F::from(i32::from(value));
                match side {
                    ProjectionSide::Left => {
                        projection[packed_col] += value * weights[packed_row];
                    }
                    ProjectionSide::Right => {
                        projection[packed_row] += value * weights[packed_col];
                    }
                }
                profile.nonzero_entries += 1;
                profile.real_projection_field_multiplications += 1;
            }
        })
        .map_err(StreamingVirtualPcsError::Source)?;
    if let Some(error) = pcs_error {
        return Err(error.into());
    }
    if let Some(vector_index) = visited.iter().position(|visited| !visited) {
        return Err(VirtualPcsError::MissingBvaVector(vector_index).into());
    }
    Ok((projection, profile))
}

fn project_packed_sparse_bva_field_streaming<F, S>(
    source: &mut S,
    challenges: &[F],
    side: ProjectionSide,
) -> Result<(Vec<F>, StreamingOpenProfile), StreamingVirtualPcsError<S::Error>>
where
    F: PrimeField,
    S: SparseBvaFieldSource<F>,
{
    let layout = source.layout().clone();
    let logical_entries = layout.logical_len();
    let packed_cols = 1 << (logical_entries.ilog2() as usize / 2);
    let packed_rows = logical_entries / packed_cols;
    let (challenge_axis, output_axis) = match side {
        ProjectionSide::Left => (packed_rows, packed_cols),
        ProjectionSide::Right => (packed_cols, packed_rows),
    };
    let expected_challenges = challenge_axis.ilog2() as usize;
    if challenges.len() != expected_challenges {
        return Err(VirtualPcsError::ProjectionPointShape {
            expected: expected_challenges,
            actual: challenges.len(),
        }
        .into());
    }
    let weights = crate::utils::xi::xi_from_challenges(&challenges.to_vec());
    let bucket_offsets = layout.bucket_offsets();
    let mut projection = vec![F::zero(); output_axis];
    let mut profile = StreamingOpenProfile {
        source_passes: 1,
        ..Default::default()
    };
    let mut previous = None;
    let mut pcs_error = None;
    source
        .for_each_nonzero(&mut |vector_index, entry_index, value| {
            if pcs_error.is_some() {
                return;
            }
            let coordinate = (vector_index, entry_index);
            if previous.is_some_and(|previous| previous >= coordinate) {
                pcs_error = Some(VirtualPcsError::SparseBvaEntryOrder);
                return;
            }
            previous = Some(coordinate);
            let Some(&bucket_index) = layout.vector_to_bucket().get(vector_index) else {
                pcs_error = Some(VirtualPcsError::SparseBvaEntryIndex {
                    vector_index,
                    entry_index,
                });
                return;
            };
            if entry_index >= layout.vector_lengths()[vector_index] {
                pcs_error = Some(VirtualPcsError::SparseBvaEntryIndex {
                    vector_index,
                    entry_index,
                });
                return;
            }
            if value.is_zero() {
                pcs_error = Some(VirtualPcsError::SparseBvaZeroEntry);
                return;
            }
            let flat_index = bucket_offsets[bucket_index] + entry_index;
            let packed_row = flat_index / packed_cols;
            let packed_col = flat_index % packed_cols;
            match side {
                ProjectionSide::Left => projection[packed_col] += value * weights[packed_row],
                ProjectionSide::Right => projection[packed_row] += value * weights[packed_col],
            }
            profile.physical_entries += 1;
            profile.nonzero_entries += 1;
            profile.real_projection_field_multiplications += 1;
        })
        .map_err(StreamingVirtualPcsError::Source)?;
    if let Some(error) = pcs_error {
        return Err(error.into());
    }
    Ok((projection, profile))
}

fn bva_short_contribution<E: Pairing>(
    prepared_bases: &[Vec<E::G1>],
    bit_width: usize,
    packed_cols: usize,
    flat_index: usize,
    value: i32,
) -> Option<(usize, E::G1)> {
    let magnitude = value.unsigned_abs();
    if magnitude == 0 {
        return None;
    }
    let packed_row = flat_index / packed_cols;
    let packed_col = flat_index % packed_cols;
    let mut contribution = E::G1::zero();
    for (bit, bases) in prepared_bases.iter().enumerate().take(bit_width) {
        if magnitude & (1_u32 << bit) == 0 {
            continue;
        }
        if value >= 0 {
            contribution += bases[packed_row];
        } else {
            contribution -= bases[packed_row];
        }
    }
    Some((packed_col, contribution))
}

#[allow(clippy::too_many_arguments)]
/// Compatibility opening for a rectangular SmartPC domain. Source projection
/// skips virtual entries, but the legacy SmartPC LR rounds still allocate and
/// fold the packed projection axes. Use the BVA/EO-LiteBullet path when padding
/// must also be absent from the inner-product folding work.
pub fn open_stacked_i32_streaming<E: Pairing, S: StackedI32Source>(
    pp: &UniversalParams<E>,
    source: &mut S,
    xl: &[E::ScalarField],
    xr: &[E::ScalarField],
    value_commitment: PairingOutput<E>,
    matrix_commitment: PairingOutput<E>,
    tier_one: &[E::G1],
    value_hiding: E::ScalarField,
    matrix_hiding: E::ScalarField,
) -> Result<(Trans<E>, StreamingOpenProfile), StreamingVirtualPcsError<S::Error>> {
    let (m, n) = packed_stacked_shape(source.domain());
    check_opening_inputs(pp, m, n, xl.len(), xr.len(), tier_one.len())?;

    let (value_current, left_profile) =
        project_packed_stacked_streaming::<E::ScalarField, S>(source, xl, ProjectionSide::Left)?;
    let mut profile = left_profile;
    let proof = open_stacked_i32_from_left_projection::<E, StreamingVirtualPcsError<S::Error>>(
        pp,
        m,
        n,
        xl,
        xr,
        value_commitment,
        matrix_commitment,
        tier_one,
        value_hiding,
        matrix_hiding,
        value_current,
        |challenges| {
            let (projection, right_profile) = project_packed_stacked_streaming::<E::ScalarField, S>(
                source,
                challenges,
                ProjectionSide::Right,
            )?;
            profile.source_passes += right_profile.source_passes;
            profile.physical_entries += right_profile.physical_entries;
            profile.nonzero_entries += right_profile.nonzero_entries;
            profile.real_projection_field_multiplications +=
                right_profile.real_projection_field_multiplications;
            Ok(projection)
        },
    )?;
    Ok((proof, profile))
}

#[derive(Clone, Copy)]
enum ProjectionSide {
    Left,
    Right,
}

fn project_packed_stacked_streaming<F: PrimeField, S: StackedI32Source>(
    source: &mut S,
    challenges: &[F],
    side: ProjectionSide,
) -> Result<(Vec<F>, StreamingOpenProfile), StreamingVirtualPcsError<S::Error>> {
    let domain = source.domain();
    let (packed_rows, packed_cols) = packed_stacked_shape(domain);
    let (challenge_axis, output_axis) = match side {
        ProjectionSide::Left => (packed_rows, packed_cols),
        ProjectionSide::Right => (packed_cols, packed_rows),
    };
    let expected_challenges = challenge_axis.ilog2() as usize;
    if challenges.len() != expected_challenges {
        return Err(VirtualPcsError::ProjectionPointShape {
            expected: expected_challenges,
            actual: challenges.len(),
        }
        .into());
    }
    let weights = crate::utils::xi::xi_from_challenges(&challenges.to_vec());
    let mut projection = vec![F::zero(); output_axis];
    let mut profile = StreamingOpenProfile {
        source_passes: 1,
        ..Default::default()
    };
    let mut pcs_error = None;
    source
        .for_each_block_column(&mut |slot, column, values| {
            if pcs_error.is_some() {
                return;
            }
            if slot >= domain.slots {
                pcs_error = Some(VirtualPcsError::SourceBlockOutOfDomain {
                    slot,
                    slots: domain.slots,
                });
                return;
            }
            if column >= domain.col_cap {
                pcs_error = Some(VirtualPcsError::SourceColumnOutOfDomain {
                    column,
                    col_cap: domain.col_cap,
                });
                return;
            }
            if values.len() > domain.row_cap {
                pcs_error = Some(VirtualPcsError::SourceColumnExceedsRowCap {
                    rows: values.len(),
                    row_cap: domain.row_cap,
                });
                return;
            }
            profile.physical_entries += values.len();
            let row_offset = slot * domain.row_cap;
            for (local_row, &value) in values.iter().enumerate() {
                if value == 0 {
                    continue;
                }
                let flat_index = (row_offset + local_row) * domain.col_cap + column;
                let packed_row = flat_index / packed_cols;
                let packed_col = flat_index % packed_cols;
                let scalar = signed_scalar::<F>(value);
                match side {
                    ProjectionSide::Left => {
                        projection[packed_col] += scalar * weights[packed_row];
                    }
                    ProjectionSide::Right => {
                        projection[packed_row] += scalar * weights[packed_col];
                    }
                }
                profile.nonzero_entries += 1;
                profile.real_projection_field_multiplications += 1;
            }
        })
        .map_err(StreamingVirtualPcsError::Source)?;
    if let Some(error) = pcs_error {
        return Err(error.into());
    }
    Ok((projection, profile))
}

fn packed_stacked_shape(domain: crate::utils::heterogeneous::PaddedDomain) -> (usize, usize) {
    let rows = domain.slots * domain.row_cap;
    let entries = rows * domain.col_cap;
    let packed_cols = 1 << (entries.ilog2() as usize / 2);
    (entries / packed_cols, packed_cols)
}

fn signed_scalar<F: PrimeField>(value: i32) -> F {
    let magnitude = F::from(value.unsigned_abs() as u64);
    if value < 0 {
        -magnitude
    } else {
        magnitude
    }
}

pub fn commit_stacked_i32<E: Pairing>(
    pp: &UniversalParams<E>,
    matrix: &HeteroBlockMat<i32>,
    hiding_factor: E::ScalarField,
    bit_width: usize,
) -> Result<(PairingOutput<E>, Vec<E::G1>), VirtualPcsError> {
    if bit_width == 0 || bit_width > 32 {
        return Err(VirtualPcsError::InvalidBitWidth(bit_width));
    }

    let (packed_rows, packed_cols) = matrix.packed_stacked_shape();
    if packed_rows > pp.vec_g.len() || packed_cols > pp.vec_h.len() {
        return Err(VirtualPcsError::ParametersTooSmall {
            packed_rows,
            packed_cols,
            q: pp.q,
        });
    }

    let mut value_out_of_range = None;
    matrix.for_each_packed_stacked_entry(|_, _, value| {
        let magnitude = value.unsigned_abs();
        if bit_width < 32 && magnitude >= (1_u32 << bit_width) {
            value_out_of_range = Some(value);
        }
    });
    if let Some(value) = value_out_of_range {
        return Err(VirtualPcsError::ValueOutOfRange { value, bit_width });
    }

    let prepared_bases = pcutils::prepare_base_short_g1::<E>(&pp.vec_g[..packed_rows], bit_width);
    let original_cols = matrix.domain.col_cap;
    let tier_one = if packed_cols < original_cols {
        let mut tier_one = vec![E::G1::zero(); packed_cols];
        matrix.for_each_packed_stacked_entry(|packed_row, packed_col, value| {
            let magnitude = value.unsigned_abs();
            for bit in 0..bit_width {
                if magnitude & (1_u32 << bit) == 0 {
                    continue;
                }
                if value >= 0 {
                    tier_one[packed_col] += prepared_bases[bit][packed_row];
                } else {
                    tier_one[packed_col] -= prepared_bases[bit][packed_row];
                }
            }
        });
        tier_one
    } else {
        let row_group = packed_cols / original_cols;
        (0..packed_cols)
            .into_par_iter()
            .map(|packed_col| {
                let original_col = packed_col % original_cols;
                let row_mod = packed_col / original_cols;
                let mut commitment = E::G1::zero();
                for (slot, block) in matrix.blocks.iter().enumerate() {
                    let Some(block) = block else { continue };
                    let Some(column) = block.columns.get(original_col) else {
                        continue;
                    };
                    let row_offset = slot * matrix.domain.row_cap;
                    let local_row_mod = (row_mod + row_group - row_offset % row_group) % row_group;
                    for local_row in (local_row_mod..column.len()).step_by(row_group) {
                        let value = column[local_row];
                        let magnitude = value.unsigned_abs();
                        if magnitude == 0 {
                            continue;
                        }
                        let global_row = row_offset + local_row;
                        let packed_row = global_row / row_group;
                        for bit in 0..bit_width {
                            if magnitude & (1_u32 << bit) == 0 {
                                continue;
                            }
                            if value >= 0 {
                                commitment += prepared_bases[bit][packed_row];
                            } else {
                                commitment -= prepared_bases[bit][packed_row];
                            }
                        }
                    }
                }
                commitment
            })
            .collect::<Vec<_>>()
    };

    let commitment = pcutils::inner_pairing_product::<E>(&tier_one, &pp.vec_h[..packed_cols])
        + pp.tilde_u.mul(hiding_factor);

    Ok((commitment, tier_one))
}

#[allow(clippy::too_many_arguments)]
pub fn open_stacked_i32<E: Pairing>(
    pp: &UniversalParams<E>,
    matrix: &HeteroBlockMat<i32>,
    xl: &[E::ScalarField],
    xr: &[E::ScalarField],
    value_commitment: PairingOutput<E>,
    matrix_commitment: PairingOutput<E>,
    tier_one: &[E::G1],
    value_hiding: E::ScalarField,
    matrix_hiding: E::ScalarField,
) -> Result<Trans<E>, VirtualPcsError> {
    let (m, n) = matrix.packed_stacked_shape();
    check_opening_inputs(pp, m, n, xl.len(), xr.len(), tier_one.len())?;

    let value_current = matrix.proj_packed_stacked_left::<E::ScalarField>(xl)?;
    open_stacked_i32_from_left_projection::<E, VirtualPcsError>(
        pp,
        m,
        n,
        xl,
        xr,
        value_commitment,
        matrix_commitment,
        tier_one,
        value_hiding,
        matrix_hiding,
        value_current,
        |challenges| {
            matrix
                .proj_packed_stacked_right::<E::ScalarField>(challenges)
                .map_err(Into::into)
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn open_stacked_i32_from_left_projection<E: Pairing, ProjectionError>(
    pp: &UniversalParams<E>,
    m: usize,
    n: usize,
    xl: &[E::ScalarField],
    xr: &[E::ScalarField],
    value_commitment: PairingOutput<E>,
    matrix_commitment: PairingOutput<E>,
    tier_one: &[E::G1],
    value_hiding: E::ScalarField,
    matrix_hiding: E::ScalarField,
    mut value_current: Vec<E::ScalarField>,
    mut project_right: impl FnMut(&[E::ScalarField]) -> Result<Vec<E::ScalarField>, ProjectionError>,
) -> Result<Trans<E>, ProjectionError> {
    let mut fs = FiatShamir::new();
    fs.push(&matrix_commitment);
    fs.push(&value_commitment);
    let xl_vec = xl.to_vec();
    let xr_vec = xr.to_vec();
    fs.push(&xl_vec);
    fs.push(&xr_vec);
    let x = fs.gen_challenge::<E>();

    let log_m = m.ilog2() as usize;
    let log_n = n.ilog2() as usize;
    let rng = &mut ark_std::rand::thread_rng();
    let u_0 = E::pairing(pp.g_0, pp.h_hat);
    let l_vec = ark_poly_commit::smart_pc::xi::xi_from_challenges::<E>(&xl_vec);
    let r_vec = ark_poly_commit::smart_pc::xi::xi_from_challenges::<E>(&xr_vec);
    let mut accumulated_hiding = value_hiding + x * matrix_hiding;

    let mut vec_l_gt = Vec::with_capacity(log_n);
    let mut vec_r_gt = Vec::with_capacity(log_n);
    let mut capital_a_current = tier_one[..n].to_vec();
    let mut h_current = pp.vec_h[..n].to_vec();
    let mut r_current = r_vec[..n].to_vec();
    let mut n_challenges = Vec::with_capacity(log_n);
    let mut n_challenges_inv = Vec::with_capacity(log_n);

    for round in 0..log_n {
        let current_len = n >> round;
        let midpoint = current_len / 2;
        let value_left = value_current[..midpoint].to_vec();
        let value_right = value_current[midpoint..current_len].to_vec();
        let capital_a_left = capital_a_current[..midpoint].to_vec();
        let capital_a_right = capital_a_current[midpoint..current_len].to_vec();
        let r_left = r_current[..midpoint].to_vec();
        let r_right = r_current[midpoint..current_len].to_vec();
        let h_left = h_current[..midpoint].to_vec();
        let h_right = h_current[midpoint..current_len].to_vec();

        let left_hiding = E::ScalarField::rand(rng);
        let right_hiding = E::ScalarField::rand(rng);
        let left = pcutils::inner_pairing_product::<E>(&capital_a_left, &h_right).mul(x)
            + u_0.mul(pcutils::inner_product::<E>(&value_left, &r_right))
            + pp.tilde_u.mul(left_hiding);
        let right = pcutils::inner_pairing_product::<E>(&capital_a_right, &h_left).mul(x)
            + u_0.mul(pcutils::inner_product::<E>(&value_right, &r_left))
            + pp.tilde_u.mul(right_hiding);
        vec_l_gt.push(left);
        vec_r_gt.push(right);
        fs.push(&left);
        fs.push(&right);

        let challenge = fs.gen_nonzero_challenge::<E>();
        let challenge_inv = challenge.inverse().unwrap();
        accumulated_hiding += left_hiding * challenge + right_hiding * challenge_inv;
        n_challenges.push(challenge);
        n_challenges_inv.push(challenge_inv);
        value_current = pcutils::add_vec_zp::<E>(
            &value_left,
            &pcutils::scalar_mul_vec_zp::<E>(&value_right, &challenge_inv),
        );
        capital_a_current = pcutils::add_vec_g1::<E>(
            &capital_a_left,
            &pcutils::scalar_mul_vec_g1::<E>(&capital_a_right, &challenge_inv),
        );
        h_current = pcutils::add_vec_g2::<E>(
            &h_left,
            &pcutils::scalar_mul_vec_g2::<E>(&h_right, &challenge),
        );
        r_current = pcutils::add_vec_zp::<E>(
            &r_left,
            &pcutils::scalar_mul_vec_zp::<E>(&r_right, &challenge),
        );
    }

    let mut a_current = project_right(&n_challenges_inv)?;
    let h_reduced = h_current[0];
    let r_reduced = r_current[0];
    let mut g_current = pp.vec_g[..m].to_vec();
    let mut l_current = l_vec[..m].to_vec();
    let mut m_challenges = Vec::with_capacity(log_m);
    let mut m_challenges_inv = Vec::with_capacity(log_m);
    let mut vec_l_native = Vec::with_capacity(log_m);
    let mut vec_r_native = Vec::with_capacity(log_m);

    for round in 0..log_m {
        let current_len = m >> round;
        let midpoint = current_len / 2;
        let a_left = a_current[..midpoint].to_vec();
        let a_right = a_current[midpoint..current_len].to_vec();
        let l_left = l_current[..midpoint].to_vec();
        let l_right = l_current[midpoint..current_len].to_vec();
        let g_left = g_current[..midpoint].to_vec();
        let g_right = g_current[midpoint..current_len].to_vec();

        let left_cancel = E::ScalarField::rand(rng);
        let left_hiding = E::ScalarField::rand(rng);
        let right_cancel = E::ScalarField::rand(rng);
        let right_hiding = E::ScalarField::rand(rng);
        let left = NativeFoldMessage::<E> {
            g1: pcutils::msm_g1::<E>(&g_right, &a_left).mul(x) + pp.g_0.mul(left_cancel),
            g2: pp
                .h_hat
                .mul(r_reduced * pcutils::inner_product::<E>(&a_left, &l_right))
                - h_reduced.mul(left_cancel)
                + pp.tilde_h.mul(left_hiding),
        };
        let right = NativeFoldMessage::<E> {
            g1: pcutils::msm_g1::<E>(&g_left, &a_right).mul(x) + pp.g_0.mul(right_cancel),
            g2: pp
                .h_hat
                .mul(r_reduced * pcutils::inner_product::<E>(&a_right, &l_left))
                - h_reduced.mul(right_cancel)
                + pp.tilde_h.mul(right_hiding),
        };
        vec_l_native.push(left);
        vec_r_native.push(right);
        fs.push(&left);
        fs.push(&right);

        let challenge = fs.gen_nonzero_challenge::<E>();
        let challenge_inv = challenge.inverse().unwrap();
        accumulated_hiding += left_hiding * challenge + right_hiding * challenge_inv;
        m_challenges.push(challenge);
        m_challenges_inv.push(challenge_inv);
        a_current = pcutils::add_vec_zp::<E>(
            &a_left,
            &pcutils::scalar_mul_vec_zp::<E>(&a_right, &challenge_inv),
        );
        g_current = pcutils::add_vec_g1::<E>(
            &g_left,
            &pcutils::scalar_mul_vec_g1::<E>(&g_right, &challenge),
        );
        l_current = pcutils::add_vec_zp::<E>(
            &l_left,
            &pcutils::scalar_mul_vec_zp::<E>(&l_right, &challenge),
        );
    }

    let a_reduced = a_current[0];
    let g_reduced = g_current[0];
    let xi_l = ark_poly_commit::smart_pc::xi::xi_ip_from_challenges::<E>(&xl_vec, &m_challenges);
    let xi_r = ark_poly_commit::smart_pc::xi::xi_ip_from_challenges::<E>(&xr_vec, &n_challenges);
    let base_rhs = u_0.mul(xi_l * xi_r) + E::pairing(g_reduced.mul(x), h_reduced);
    fs.push(&g_reduced);
    fs.push(&h_reduced);
    let pip_g1_challenge = fs.gen_challenge::<E>();
    let (v_g_prime, w_g) =
        sub_protocols::pip_g1_prove_with_challenge::<E>(pp, &m_challenges, pip_g1_challenge);
    fs.push(&v_g_prime);
    fs.push(&w_g);
    let pip_g2_challenge = fs.gen_challenge::<E>();
    let (v_h_prime, w_h) =
        sub_protocols::pip_g2_prove_with_challenge::<E>(pp, &n_challenges, pip_g2_challenge);
    fs.push(&v_h_prime);
    fs.push(&w_h);
    let (linear_f, linear_z_a, linear_z_t) = sub_protocols::linear_prove::<E, _>(
        base_rhs,
        pp.tilde_u,
        a_reduced,
        accumulated_hiding,
        &mut fs,
        rng,
    );

    Ok(Trans::<E> {
        vec_l_gt,
        vec_r_gt,
        vec_l_native,
        vec_r_native,
        v_g: g_reduced,
        v_h: h_reduced,
        v_g_prime,
        v_h_prime,
        w_g,
        w_h,
        linear_f,
        linear_z_a,
        linear_z_t,
    })
}

fn check_opening_inputs<E: Pairing>(
    pp: &UniversalParams<E>,
    rows: usize,
    cols: usize,
    xl_len: usize,
    xr_len: usize,
    tier_one_len: usize,
) -> Result<(), VirtualPcsError> {
    if rows > pp.vec_g.len() || cols > pp.vec_h.len() {
        return Err(VirtualPcsError::ParametersTooSmall {
            packed_rows: rows,
            packed_cols: cols,
            q: pp.q,
        });
    }
    let expected_xl = rows.ilog2() as usize;
    let expected_xr = cols.ilog2() as usize;
    if xl_len != expected_xl || xr_len != expected_xr {
        return Err(VirtualPcsError::PointShape {
            expected: (expected_xl, expected_xr),
            actual: (xl_len, xr_len),
        });
    }
    if tier_one_len != cols {
        return Err(VirtualPcsError::TierOneLength {
            expected: cols,
            actual: tier_one_len,
        });
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VirtualPcsError {
    InvalidBitWidth(usize),
    InvalidBvaLogicalEntries(usize),
    ParametersTooSmall {
        packed_rows: usize,
        packed_cols: usize,
        q: usize,
    },
    ValueOutOfRange {
        value: i32,
        bit_width: usize,
    },
    PointShape {
        expected: (usize, usize),
        actual: (usize, usize),
    },
    ProjectionPointShape {
        expected: usize,
        actual: usize,
    },
    TierOneLength {
        expected: usize,
        actual: usize,
    },
    SourceBlockOutOfDomain {
        slot: usize,
        slots: usize,
    },
    SourceColumnOutOfDomain {
        column: usize,
        col_cap: usize,
    },
    SourceColumnExceedsRowCap {
        rows: usize,
        row_cap: usize,
    },
    BvaVectorIndex {
        vector_index: usize,
        vector_count: usize,
    },
    DuplicateBvaVector(usize),
    MissingBvaVector(usize),
    BvaVectorLength {
        vector_index: usize,
        physical_len: usize,
        expected_len: usize,
        bucket_size: usize,
    },
    BvaTbvLayoutMismatch,
    SparseBvaEntryIndex {
        vector_index: usize,
        entry_index: usize,
    },
    SparseBvaEntryOrder,
    SparseBvaZeroEntry,
    Matrix(HeteroMatrixError),
}

impl From<HeteroMatrixError> for VirtualPcsError {
    fn from(error: HeteroMatrixError) -> Self {
        Self::Matrix(error)
    }
}

impl Display for VirtualPcsError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for VirtualPcsError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::bucketed_ip::{
        BvaLayout, EoLiteBulletProof, PackedEoLiteBulletProof, TbvBucket, TensorBasisVector,
    };
    use crate::utils::heterogeneous::{HeteroBlockMat, MatrixBlock, MatrixDims, PaddedDomain};
    use crate::utils::sparse::DenseMatCM;
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_poly_commit::smart_pc::SmartPC;

    struct TestBvaSource {
        layout: BvaLayout,
        vectors: Vec<Vec<i32>>,
    }

    struct TestBvaI8Source {
        layout: BvaLayout,
        vectors: Vec<Vec<i8>>,
    }

    impl BvaI32Source for TestBvaSource {
        type Error = Infallible;

        fn layout(&self) -> &BvaLayout {
            &self.layout
        }

        fn for_each_vector(
            &mut self,
            visit: &mut dyn FnMut(usize, &[i32]),
        ) -> Result<(), Self::Error> {
            for (index, vector) in self.vectors.iter().enumerate() {
                visit(index, vector);
            }
            Ok(())
        }
    }

    impl BvaI8Source for TestBvaI8Source {
        type Error = Infallible;

        fn layout(&self) -> &BvaLayout {
            &self.layout
        }

        fn for_each_vector(
            &mut self,
            visit: &mut dyn FnMut(usize, &[i8]),
        ) -> Result<(), Self::Error> {
            for (index, vector) in self.vectors.iter().enumerate() {
                visit(index, vector);
            }
            Ok(())
        }
    }

    struct TestBvaFieldSource {
        layout: BvaLayout,
        vectors: Vec<Vec<Fr>>,
    }

    struct TestSparseBvaFieldSource {
        layout: BvaLayout,
        entries: Vec<(usize, usize, Fr)>,
    }

    impl SparseBvaFieldSource<Fr> for TestSparseBvaFieldSource {
        type Error = Infallible;

        fn layout(&self) -> &BvaLayout {
            &self.layout
        }

        fn for_each_nonzero(
            &mut self,
            visit: &mut dyn FnMut(usize, usize, Fr),
        ) -> Result<(), Self::Error> {
            for &(vector, entry, value) in &self.entries {
                visit(vector, entry, value);
            }
            Ok(())
        }
    }

    impl BvaFieldSource<Fr> for TestBvaFieldSource {
        type Error = Infallible;

        fn layout(&self) -> &BvaLayout {
            &self.layout
        }

        fn for_each_vector(
            &mut self,
            visit: &mut dyn FnMut(usize, &[Fr]),
        ) -> Result<(), Self::Error> {
            for (index, vector) in self.vectors.iter().enumerate() {
                visit(index, vector);
            }
            Ok(())
        }
    }

    #[test]
    fn sparse_bva_commitment_and_opening_skip_implicit_zero_entries() {
        let layout = BvaLayout::from_lengths(&[5]).unwrap();
        let entries = vec![(0, 1, Fr::from(7_u64)), (0, 4, Fr::from(11_u64))];
        let mut sparse = TestSparseBvaFieldSource {
            layout: layout.clone(),
            entries: entries.clone(),
        };
        let mut dense = TestBvaFieldSource {
            layout: layout.clone(),
            vectors: vec![vec![
                Fr::from(0_u64),
                Fr::from(7_u64),
                Fr::from(0_u64),
                Fr::from(0_u64),
                Fr::from(11_u64),
            ]],
        };
        let mut rng = ark_std::test_rng();
        let pp = SmartPC::<Bls12_381>::setup(2, &mut rng).unwrap();
        let matrix_hiding = Fr::from(13_u64);
        let (expected_commitment, _, _) =
            commit_bva_field_streaming::<Bls12_381, _>(&pp, &mut dense, matrix_hiding).unwrap();
        let (commitment, tier_one, profile) =
            commit_sparse_bva_field_streaming::<Bls12_381, _>(&pp, &mut sparse, matrix_hiding)
                .unwrap();
        assert_eq!(commitment, expected_commitment);
        assert_eq!(profile.nonzero_entries, 2);
        assert_eq!(profile.max_live_group_contributions, 1);
        assert_eq!(profile.materialized_padding_entries, 0);
        assert_eq!(profile.padding_group_operations, 0);

        let xl = vec![Fr::from(2_u64), Fr::from(3_u64)];
        let xr = vec![Fr::from(5_u64)];
        let row_weights = crate::utils::xi::xi_from_challenges(&xl);
        let column_weights = crate::utils::xi::xi_from_challenges(&xr);
        let packed_cols = 2_usize;
        let value = entries
            .iter()
            .fold(Fr::from(0_u64), |sum, (_, index, value)| {
                sum + *value
                    * row_weights[index / packed_cols]
                    * column_weights[index % packed_cols]
            });
        let value_hiding = Fr::from(17_u64);
        let value_commitment =
            Bls12_381::pairing(pp.g_0.mul(value), pp.h_hat) + pp.tilde_u.mul(value_hiding);
        let (opening, opening_profile) = open_sparse_bva_field_streaming::<Bls12_381, _>(
            &pp,
            &mut sparse,
            &xl,
            &xr,
            value_commitment,
            commitment,
            &tier_one,
            value_hiding,
            matrix_hiding,
        )
        .unwrap();
        assert!(SmartPC::<Bls12_381>::verify(
            &pp,
            commitment,
            value_commitment,
            &xl,
            &xr,
            &opening,
        )
        .unwrap());
        assert_eq!(opening_profile.nonzero_entries, 4);
        assert_eq!(opening_profile.materialized_projection_padding_entries, 0);
        assert_eq!(opening_profile.padding_projection_field_multiplications, 0);

        let mut unsorted = TestSparseBvaFieldSource {
            layout,
            entries: entries.into_iter().rev().collect(),
        };
        assert!(commit_sparse_bva_field_streaming::<Bls12_381, _>(
            &pp,
            &mut unsorted,
            matrix_hiding,
        )
        .is_err());
    }

    #[test]
    fn virtual_commitment_matches_explicit_zero_padding() {
        let domain = PaddedDomain::new(2, 4, 4).unwrap();
        let first = MatrixBlock::from_columns(
            MatrixDims::new(2, 3).unwrap(),
            vec![vec![1_i32, 2], vec![3, 4], vec![5, 6]],
        )
        .unwrap();
        let second =
            MatrixBlock::from_columns(MatrixDims::new(1, 2).unwrap(), vec![vec![7_i32], vec![8]])
                .unwrap();
        let matrix = HeteroBlockMat::new(domain, vec![Some(first), Some(second)]).unwrap();

        let explicit_stacked = DenseMatCM::<i32, Fr>::from_data(vec![
            vec![1, 2, 0, 0, 7, 0, 0, 0],
            vec![3, 4, 0, 0, 8, 0, 0, 0],
            vec![5, 6, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0],
        ]);
        let explicit_packed = explicit_stacked.to_square_mat();

        let mut rng = ark_std::test_rng();
        let pp = SmartPC::<Bls12_381>::setup(3, &mut rng).unwrap();
        let hiding_factor = Fr::from(42_u64);
        let expected =
            SmartPC::<Bls12_381>::commit_short(&pp, &explicit_packed.data, hiding_factor, 8)
                .unwrap();

        let actual = commit_stacked_i32::<Bls12_381>(&pp, &matrix, hiding_factor, 8).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn streaming_commitment_matches_materialized_source_without_padding_work() {
        let domain = PaddedDomain::new(2, 4, 4).unwrap();
        let first = MatrixBlock::from_columns(
            MatrixDims::new(2, 3).unwrap(),
            vec![vec![1_i32, 2], vec![3, 0], vec![5, 6]],
        )
        .unwrap();
        let second =
            MatrixBlock::from_columns(MatrixDims::new(1, 2).unwrap(), vec![vec![7_i32], vec![8]])
                .unwrap();
        let matrix = HeteroBlockMat::new(domain, vec![Some(first), Some(second)]).unwrap();

        let mut rng = ark_std::test_rng();
        let pp = SmartPC::<Bls12_381>::setup(3, &mut rng).unwrap();
        let hiding_factor = Fr::from(42_u64);
        let expected = commit_stacked_i32::<Bls12_381>(&pp, &matrix, hiding_factor, 8).unwrap();
        let mut streaming_source = matrix.clone();

        let (commitment, tier_one, profile) = commit_stacked_i32_streaming::<Bls12_381, _>(
            &pp,
            &mut streaming_source,
            hiding_factor,
            8,
        )
        .unwrap();

        assert_eq!((commitment, tier_one), expected);
        assert_eq!(profile.physical_entries, 8);
        assert_eq!(profile.nonzero_entries, 7);
        assert_eq!(profile.max_live_scalars, 2);
        assert_eq!(profile.materialized_padding_entries, 0);
        assert_eq!(profile.padding_scalar_multiplications, 0);
    }

    #[test]
    fn bva_streaming_commitment_matches_expanded_vector_without_materializing_padding() {
        let layout = BvaLayout::from_lengths(&[3, 1]).unwrap();
        let vectors = vec![vec![1_i32, 2, 3], vec![4_i32]];
        let mut expanded = vec![0_i32; layout.logical_len()];
        for (index, vector) in vectors.iter().enumerate() {
            let offset = layout.vector_offset(index).unwrap();
            expanded[offset..offset + vector.len()].copy_from_slice(vector);
        }
        let domain = PaddedDomain::new(1, layout.logical_len(), 1).unwrap();
        let block = MatrixBlock::from_columns(
            MatrixDims::new(layout.logical_len(), 1).unwrap(),
            vec![expanded],
        )
        .unwrap();
        let materialized = HeteroBlockMat::new(domain, vec![Some(block)]).unwrap();
        let mut source = TestBvaSource { layout, vectors };

        let mut rng = ark_std::test_rng();
        let pp = SmartPC::<Bls12_381>::setup(2, &mut rng).unwrap();
        let hiding = Fr::from(19_u64);
        let expected = commit_stacked_i32::<Bls12_381>(&pp, &materialized, hiding, 8).unwrap();
        let (commitment, tier_one, profile) =
            commit_bva_i32_streaming::<Bls12_381, _>(&pp, &mut source, hiding, 8).unwrap();

        assert_eq!((commitment, tier_one), expected);
        assert_eq!(profile.physical_entries, 4);
        assert_eq!(profile.logical_entries, 8);
        assert_eq!(profile.prepared_base_entries, 4 * 8);
        assert_eq!(profile.materialized_padding_entries, 0);
        assert_eq!(profile.padding_group_operations, 0);

        let mut shortened = TestBvaSource {
            layout: source.layout.clone(),
            vectors: vec![vec![1_i32, 2], vec![4_i32]],
        };
        let error =
            commit_bva_i32_streaming::<Bls12_381, _>(&pp, &mut shortened, hiding, 8).unwrap_err();
        assert!(matches!(
            error,
            StreamingVirtualPcsError::Pcs(VirtualPcsError::BvaVectorLength {
                vector_index: 0,
                physical_len: 2,
                expected_len: 3,
                bucket_size: 4,
            })
        ));
    }

    #[test]
    fn field_bva_commitment_matches_the_short_integer_path() {
        let layout = BvaLayout::from_lengths(&[3, 1]).unwrap();
        let vectors = vec![vec![1_i32, -2, 3], vec![4_i32]];
        let mut integer_source = TestBvaSource {
            layout: layout.clone(),
            vectors: vectors.clone(),
        };
        let mut field_source = TestBvaFieldSource {
            layout,
            vectors: vectors
                .into_iter()
                .map(|vector| vector.into_iter().map(signed_scalar::<Fr>).collect())
                .collect(),
        };
        let mut rng = ark_std::test_rng();
        let pp = SmartPC::<Bls12_381>::setup(2, &mut rng).unwrap();
        let hiding = Fr::from(29_u64);

        let expected =
            commit_bva_i32_streaming::<Bls12_381, _>(&pp, &mut integer_source, hiding, 8).unwrap();
        let actual =
            commit_bva_field_streaming::<Bls12_381, _>(&pp, &mut field_source, hiding).unwrap();

        assert_eq!((actual.0, actual.1), (expected.0, expected.1));
        assert_eq!(actual.2.physical_entries, 4);
        assert_eq!(actual.2.prepared_base_entries, 0);
        assert_eq!(actual.2.materialized_padding_entries, 0);
    }

    #[test]
    fn parallel_materialized_field_commitment_matches_streaming_without_padding_entries() {
        use crate::protocols::bucketed_ip::BucketedVector;

        let vectors = vec![
            vec![Fr::from(1_u64), Fr::from(2_u64), Fr::from(3_u64)],
            vec![Fr::from(4_u64)],
        ];
        let mut streaming = TestBvaFieldSource {
            layout: BvaLayout::from_lengths(&[3, 1]).unwrap(),
            vectors: vectors.clone(),
        };
        let materialized = BucketedVector::new(vectors).unwrap();
        let mut rng = ark_std::test_rng();
        let pp = SmartPC::<Bls12_381>::setup(2, &mut rng).unwrap();
        let hiding = Fr::from(31_u64);

        let expected =
            commit_bva_field_streaming::<Bls12_381, _>(&pp, &mut streaming, hiding).unwrap();
        let actual =
            commit_bva_field_materialized_parallel::<Bls12_381>(&pp, &materialized, hiding)
                .unwrap();

        assert_eq!((actual.0, actual.1), (expected.0, expected.1));
        assert_eq!(actual.2.physical_entries, 4);
        assert_eq!(actual.2.materialized_padding_entries, 0);
    }

    #[test]
    fn field_bva_streaming_inner_product_matches_owned_bva_without_padding_work() {
        use crate::protocols::bucketed_ip::{BucketedVector, TbvBucket, TensorBasisVector};

        let layout = BvaLayout::from_lengths(&[3, 1]).unwrap();
        let vectors = vec![
            vec![Fr::from(1_u64), Fr::from(2_u64), Fr::from(3_u64)],
            vec![Fr::from(4_u64)],
        ];
        let tbv = TensorBasisVector::new(
            &layout,
            layout
                .bucket_sizes()
                .iter()
                .enumerate()
                .map(|(bucket, size)| {
                    TbvBucket::new(
                        Fr::from((bucket + 5) as u64),
                        (0..size.ilog2())
                            .map(|round| Fr::from(u64::from(round) + 11))
                            .collect(),
                    )
                })
                .collect(),
        )
        .unwrap();
        let expected = BucketedVector::new(vectors.clone())
            .unwrap()
            .inner_product(&tbv)
            .unwrap();
        let mut source = TestBvaFieldSource { layout, vectors };

        let actual = inner_product_bva_field_streaming(&mut source, &tbv).unwrap();

        assert_eq!(actual.value, expected.value);
        assert_eq!(actual.profile.source_passes, 1);
        assert_eq!(actual.profile.physical_entries, 4);
        assert_eq!(actual.profile.maximum_live_vector_entries, 3);
        assert_eq!(actual.profile.materialized_padding_entries, 0);
        assert_eq!(actual.profile.padding_field_operations, 0);
    }

    #[test]
    fn streaming_eo_reduced_value_opens_against_the_same_bva_commitment() {
        let layout = BvaLayout::from_lengths(&[3, 1]).unwrap();
        let vectors = vec![
            vec![Fr::from(1_u64), Fr::from(2_u64), Fr::from(3_u64)],
            vec![Fr::from(4_u64)],
        ];
        let tbv = TensorBasisVector::new(
            &layout,
            layout
                .bucket_sizes()
                .iter()
                .enumerate()
                .map(|(bucket, size)| {
                    TbvBucket::new(
                        Fr::from((bucket + 5) as u64),
                        (0..size.ilog2())
                            .map(|round| Fr::from(u64::from(round) + 11))
                            .collect(),
                    )
                })
                .collect(),
        )
        .unwrap();
        let mut source = TestBvaFieldSource {
            layout: layout.clone(),
            vectors: vectors.clone(),
        };
        let claimed = inner_product_bva_field_streaming(&mut source, &tbv)
            .unwrap()
            .value;
        let (eo_proof, _, _) =
            EoLiteBulletProof::prove_streaming(&mut source, &tbv, claimed).unwrap();
        let eo_verification = eo_proof.verify_with_tbv(&tbv, claimed);
        assert!(eo_verification.accepted);

        let packed_cols = 1 << (layout.logical_len().ilog2() as usize / 2);
        let packed_rows = layout.logical_len() / packed_cols;
        let mut explicit = vec![vec![Fr::zero(); packed_rows]; packed_cols];
        let offsets = layout.bucket_offsets();
        for (vector_index, vector) in vectors.iter().enumerate() {
            let offset = offsets[layout.vector_to_bucket()[vector_index]];
            for (local_index, value) in vector.iter().enumerate() {
                let flat_index = offset + local_index;
                explicit[flat_index % packed_cols][flat_index / packed_cols] = *value;
            }
        }
        let (xl, xr) =
            bva_eo_opening_point(layout.logical_len(), &eo_verification.challenges_inv).unwrap();
        let mut rng = ark_std::test_rng();
        let pp = SmartPC::<Bls12_381>::setup(2, &mut rng).unwrap();
        let matrix_hiding = Fr::from(29_u64);
        let (matrix_commitment, tier_one, _) =
            commit_bva_field_streaming::<Bls12_381, _>(&pp, &mut source, matrix_hiding).unwrap();
        let (value_commitment, value_hiding) = SmartPC::<Bls12_381>::eval(&pp, &explicit, &xl, &xr);
        let expected_value_commitment =
            Bls12_381::pairing(pp.g_0.mul(eo_proof.a_reduced), pp.h_hat)
                + pp.tilde_u.mul(value_hiding);
        assert_eq!(value_commitment, expected_value_commitment);

        let (opening, profile) = open_bva_field_streaming::<Bls12_381, _>(
            &pp,
            &mut source,
            &xl,
            &xr,
            value_commitment,
            matrix_commitment,
            &tier_one,
            value_hiding,
            matrix_hiding,
        )
        .unwrap();

        assert!(SmartPC::<Bls12_381>::verify(
            &pp,
            matrix_commitment,
            value_commitment,
            &xl,
            &xr,
            &opening,
        )
        .unwrap());
        assert_eq!(profile.materialized_projection_padding_entries, 0);
        assert_eq!(profile.padding_projection_field_multiplications, 0);
    }

    #[test]
    fn packed_eo_commitment_and_opening_keep_the_witness_in_i8_storage() {
        let layout = BvaLayout::from_lengths(&[3, 1]).unwrap();
        let vectors = vec![vec![1_i8, -2, 3], vec![4_i8]];
        let tbv = TensorBasisVector::global_tensor(
            &layout,
            &[Fr::from(2_u64), Fr::from(3_u64), Fr::from(5_u64)],
        )
        .unwrap();
        let mut source = TestBvaI8Source {
            layout: layout.clone(),
            vectors,
        };
        let claim = inner_product_bva_i8_streaming(&mut source, &tbv)
            .unwrap()
            .value;
        let (proof, counters, packed_profile) =
            PackedEoLiteBulletProof::prove_i8_streaming(&mut source, &tbv, claim, 8).unwrap();
        let verification = proof.verify_with_tbv(&tbv, claim);
        assert!(verification.accepted);
        assert_eq!(proof.packing_stride, 8);
        assert_eq!(packed_profile.maximum_live_vector_entries, 1);
        assert_eq!(counters.materialized_padding_entries, 0);

        let mut rng = ark_std::test_rng();
        let pp = SmartPC::<Bls12_381>::setup(2, &mut rng).unwrap();
        let matrix_hiding = Fr::from(29_u64);
        let (matrix_commitment, tier_one, commit_profile) =
            commit_bva_i8_streaming::<Bls12_381, _>(&pp, &mut source, matrix_hiding, 8).unwrap();
        let (xl, xr) =
            bva_eo_opening_point(layout.logical_len(), &verification.challenges_inv).unwrap();
        let value_commitment = Bls12_381::pairing(pp.g_0.mul(proof.a_reduced), pp.h_hat);
        let (opening, opening_profile) = open_bva_i8_streaming::<Bls12_381, _>(
            &pp,
            &mut source,
            &xl,
            &xr,
            value_commitment,
            matrix_commitment,
            &tier_one,
            Fr::zero(),
            matrix_hiding,
        )
        .unwrap();

        assert!(SmartPC::<Bls12_381>::verify(
            &pp,
            matrix_commitment,
            value_commitment,
            &xl,
            &xr,
            &opening,
        )
        .unwrap());
        assert_eq!(commit_profile.physical_entries, 4);
        assert_eq!(commit_profile.materialized_padding_entries, 0);
        assert_eq!(opening_profile.materialized_projection_padding_entries, 0);
        assert_eq!(opening_profile.padding_projection_field_multiplications, 0);
    }

    #[test]
    fn virtual_opening_verifies_against_existing_pcs_verifier() {
        let domain = PaddedDomain::new(2, 4, 4).unwrap();
        let first = MatrixBlock::from_columns(
            MatrixDims::new(2, 3).unwrap(),
            vec![vec![1_i32, 2], vec![3, 4], vec![5, 6]],
        )
        .unwrap();
        let second =
            MatrixBlock::from_columns(MatrixDims::new(1, 2).unwrap(), vec![vec![7_i32], vec![8]])
                .unwrap();
        let matrix = HeteroBlockMat::new(domain, vec![Some(first), Some(second)]).unwrap();
        let explicit_stacked = DenseMatCM::<i32, Fr>::from_data(vec![
            vec![1, 2, 0, 0, 7, 0, 0, 0],
            vec![3, 4, 0, 0, 8, 0, 0, 0],
            vec![5, 6, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0],
        ]);
        let explicit_packed = explicit_stacked.to_square_mat();
        let explicit_field: Vec<Vec<Fr>> = explicit_packed
            .data
            .iter()
            .map(|column| column.iter().copied().map(Fr::from).collect())
            .collect();

        let mut rng = ark_std::test_rng();
        let pp = SmartPC::<Bls12_381>::setup(3, &mut rng).unwrap();
        let hiding_factor = Fr::from(42_u64);
        let (commitment, tier_one) =
            commit_stacked_i32::<Bls12_381>(&pp, &matrix, hiding_factor, 8).unwrap();
        let xl = vec![Fr::from(2_u64), Fr::from(3_u64), Fr::from(5_u64)];
        let xr = vec![Fr::from(7_u64), Fr::from(11_u64)];
        let (value_commitment, value_hiding) =
            SmartPC::<Bls12_381>::eval(&pp, &explicit_field, &xl, &xr);

        let proof = open_stacked_i32::<Bls12_381>(
            &pp,
            &matrix,
            &xl,
            &xr,
            value_commitment,
            commitment,
            &tier_one,
            value_hiding,
            hiding_factor,
        )
        .unwrap();

        assert!(
            SmartPC::<Bls12_381>::verify(&pp, commitment, value_commitment, &xl, &xr, &proof,)
                .unwrap()
        );
    }

    #[test]
    fn streaming_opening_verifies_without_padding_projection_work() {
        let domain = PaddedDomain::new(2, 4, 4).unwrap();
        let first = MatrixBlock::from_columns(
            MatrixDims::new(2, 3).unwrap(),
            vec![vec![1_i32, 2], vec![3, 4], vec![5, 6]],
        )
        .unwrap();
        let second =
            MatrixBlock::from_columns(MatrixDims::new(1, 2).unwrap(), vec![vec![7_i32], vec![8]])
                .unwrap();
        let matrix = HeteroBlockMat::new(domain, vec![Some(first), Some(second)]).unwrap();
        let explicit_stacked = DenseMatCM::<i32, Fr>::from_data(vec![
            vec![1, 2, 0, 0, 7, 0, 0, 0],
            vec![3, 4, 0, 0, 8, 0, 0, 0],
            vec![5, 6, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0],
        ]);
        let explicit_packed = explicit_stacked.to_square_mat();
        let explicit_field: Vec<Vec<Fr>> = explicit_packed
            .data
            .iter()
            .map(|column| column.iter().copied().map(Fr::from).collect())
            .collect();

        let mut rng = ark_std::test_rng();
        let pp = SmartPC::<Bls12_381>::setup(3, &mut rng).unwrap();
        let hiding_factor = Fr::from(42_u64);
        let mut source = matrix.clone();
        let (commitment, tier_one, _) =
            commit_stacked_i32_streaming::<Bls12_381, _>(&pp, &mut source, hiding_factor, 8)
                .unwrap();
        let xl = vec![Fr::from(2_u64), Fr::from(3_u64), Fr::from(5_u64)];
        let xr = vec![Fr::from(7_u64), Fr::from(11_u64)];
        let (value_commitment, value_hiding) =
            SmartPC::<Bls12_381>::eval(&pp, &explicit_field, &xl, &xr);

        let (proof, profile) = open_stacked_i32_streaming::<Bls12_381, _>(
            &pp,
            &mut source,
            &xl,
            &xr,
            value_commitment,
            commitment,
            &tier_one,
            value_hiding,
            hiding_factor,
        )
        .unwrap();

        assert!(
            SmartPC::<Bls12_381>::verify(&pp, commitment, value_commitment, &xl, &xr, &proof,)
                .unwrap()
        );
        assert_eq!(profile.source_passes, 2);
        assert_eq!(profile.physical_entries, 16);
        assert_eq!(profile.materialized_projection_padding_entries, 0);
        assert_eq!(profile.padding_projection_field_multiplications, 0);
    }
}
