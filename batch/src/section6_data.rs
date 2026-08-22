//! Prover-side data construction for the Section 6 point compiler.
//!
//! This module follows the fixed-parameter registered-mask variant of the
//! specification: every source matrix is at most `2^m x 2^m` with `m = 15`,
//! and one collection contains at most `2^d` matrices with `d = 10`.
//! Registration stores exactly three Boolean mask matrices:
//!
//! * `D_eta   in {0,1}^{T x (2m+d)}` - fixed offset bits of every bucket
//!   (column offsets, then row offsets) followed by the bits of `tau - 1`;
//! * `D_Z     in {0,1}^{T x 2m}`     - selectors of the varying coordinates;
//! * `D_eta'  in {0,1}^{T x l}`      - bits of the global bucket offsets.
//!
//! No shape-validity relation is constructed anywhere: the masks are treated
//! as registered inputs, and the per-opening argument only proves that the
//! compiled TBV is consistent with them.
//!
//! This module intentionally stops at witness construction. It does not
//! commit any table, append anything to a transcript, or orchestrate LogUp
//! and relation proofs.

use std::fmt::{Display, Formatter};

use crate::{
    bucketed_ip::{BvaLayout, TbvBucket, TensorBasisVector},
    matrix_collection::DenseBvaDescriptor,
};
use ark_ff::{PrimeField, Zero};
use serde::{Deserialize, Serialize};

#[cfg(test)]
use ark_bn254::Fr;

/// Errors raised while constructing Section 6 public and private data.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Section6DataError {
    InvalidLayout(String),
    InvalidMetadata(String),
    InvalidClaim(String),
    InvalidPoint(String),
    LengthOverflow,
    Backend(String),
}

impl Display for Section6DataError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidLayout(message) => {
                write!(formatter, "invalid Section 6 layout: {message}")
            }
            Self::InvalidMetadata(message) => {
                write!(formatter, "invalid private shape metadata: {message}")
            }
            Self::InvalidClaim(message) => write!(formatter, "invalid matrix claim: {message}"),
            Self::InvalidPoint(message) => write!(formatter, "invalid evaluation point: {message}"),
            Self::LengthOverflow => write!(formatter, "Section 6 length overflowed usize"),
            Self::Backend(message) => {
                write!(formatter, "BVA backend rejected Section 6 data: {message}")
            }
        }
    }
}

impl std::error::Error for Section6DataError {}

/// Fixed per-side point width `m`: every source matrix is at most
/// `2^m x 2^m`.
pub const SECTION6_FIXED_POINT_LOG: usize = 15;

/// Fixed source-count width `d`: one collection has at most `2^d` matrices.
pub const SECTION6_FIXED_MATRIX_LOG: usize = 10;

/// Width `2m` shared by `D_Z`, `X`, `X'`, `sigma`, and `Z~`.
pub const SECTION6_FIXED_LOCAL_WIDTH: usize = 2 * SECTION6_FIXED_POINT_LOG;

/// Width `2m + d` of `D_eta` and of the eta prefix matrix `P`.
pub const SECTION6_FIXED_ETA_WIDTH: usize = SECTION6_FIXED_LOCAL_WIDTH + SECTION6_FIXED_MATRIX_LOG;

/// One private per-matrix MLE claim with fixed-width padded points.
///
/// Points are zero-padded to `m` coordinates per side. Because the tensor
/// span is monomial (`weight(i) = prod_{bit set in i} x_bit`), the padded
/// coordinates never contribute to the claimed value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Section6MaskMatrixClaim<F: PrimeField> {
    /// Left (row) point zero-padded to `SECTION6_FIXED_POINT_LOG`.
    pub left_point: Vec<F>,
    /// Right (column) point zero-padded to `SECTION6_FIXED_POINT_LOG`.
    pub right_point: Vec<F>,
    pub value: F,
}

impl<F: PrimeField> Section6MaskMatrixClaim<F> {
    pub fn from_unpadded(
        left_point: &[F],
        right_point: &[F],
        value: F,
    ) -> Result<Self, Section6DataError> {
        if left_point.len() > SECTION6_FIXED_POINT_LOG
            || right_point.len() > SECTION6_FIXED_POINT_LOG
        {
            return Err(Section6DataError::InvalidClaim(format!(
                "claim points must fit {SECTION6_FIXED_POINT_LOG} coordinates per side"
            )));
        }
        let mut left = left_point.to_vec();
        let mut right = right_point.to_vec();
        left.resize(SECTION6_FIXED_POINT_LOG, F::zero());
        right.resize(SECTION6_FIXED_POINT_LOG, F::zero());
        Ok(Self {
            left_point: left,
            right_point: right,
            value,
        })
    }

    pub fn validate(&self) -> Result<(), Section6DataError> {
        if self.left_point.len() != SECTION6_FIXED_POINT_LOG
            || self.right_point.len() != SECTION6_FIXED_POINT_LOG
        {
            return Err(Section6DataError::InvalidClaim(format!(
                "claim points must have exactly {SECTION6_FIXED_POINT_LOG} coordinates per side"
            )));
        }
        Ok(())
    }
}

/// Shape-hiding public layout shared with the verifier.
///
/// The individual bucket sizes, bucket offsets, and source indices stay
/// private; the verifier only learns the counts and the global BVA cover.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Section6PublicLayout {
    /// Number of original matrices `t`.
    pub matrix_count: usize,
    /// Logical BVA length `K` including the implicit final zero suffix.
    pub logical_len: usize,
    /// Number of physically stored entries `N`.
    pub active_len: usize,
    /// Number of real dyadic buckets `T`.
    pub bucket_count: usize,
    /// `l = log_2(K)`, the length of the Packed EO point.
    pub global_log: usize,
    /// Low-bit witness width used by Packed EO-LiteBullet.
    pub bit_width: usize,
}

impl Section6PublicLayout {
    pub fn validate(&self) -> Result<(), Section6DataError> {
        if self.matrix_count == 0 || self.bucket_count == 0 || self.active_len == 0 {
            return Err(Section6DataError::InvalidLayout(
                "matrix, bucket, and entry counts must be nonzero".to_owned(),
            ));
        }
        if self.bit_width == 0 || self.bit_width > 8 {
            return Err(Section6DataError::InvalidLayout(format!(
                "bit_width must be in 1..=8, got {}",
                self.bit_width
            )));
        }
        if !self.logical_len.is_power_of_two()
            || self.logical_len != next_power_of_two(self.active_len)?
            || self.global_log != self.logical_len.ilog2() as usize
        {
            return Err(Section6DataError::InvalidLayout(
                "logical_len must be next_power_of_two(active_len)".to_owned(),
            ));
        }
        if self.bucket_count < self.matrix_count {
            return Err(Section6DataError::InvalidLayout(
                "every matrix must contribute at least one bucket".to_owned(),
            ));
        }
        if self.global_log == 0 {
            return Err(Section6DataError::InvalidLayout(
                "the compiled protocol requires at least two BVA entries".to_owned(),
            ));
        }
        self.validate_fixed_caps()
    }

    /// Enforce the fixed protocol caps `t <= 2^d` and `N <= 2^{2m+d}`.
    pub fn validate_fixed_caps(&self) -> Result<(), Section6DataError> {
        if self.matrix_count > 1 << SECTION6_FIXED_MATRIX_LOG {
            return Err(Section6DataError::InvalidLayout(format!(
                "matrix_count {} exceeds the fixed bound 2^{SECTION6_FIXED_MATRIX_LOG}",
                self.matrix_count
            )));
        }
        if self.global_log > SECTION6_FIXED_ETA_WIDTH {
            return Err(Section6DataError::InvalidLayout(format!(
                "global log {} exceeds the fixed 2^{SECTION6_FIXED_ETA_WIDTH} collection cover",
                self.global_log
            )));
        }
        Ok(())
    }

    /// Power-of-two bucket-row cover shared by every auxiliary matrix.
    pub fn row_cover(&self) -> Result<usize, Section6DataError> {
        next_power_of_two(self.bucket_count)
    }

    /// Width of `Z`, `Z'`, and `D_Z'`: no bucket has more than `min(2m, l)`
    /// varying coordinates.
    pub fn local_width(&self) -> usize {
        SECTION6_FIXED_LOCAL_WIDTH.min(self.global_log)
    }
}

/// The prover-side layout `L = (K, N, T, {k_j})` with the private bucket list.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Section6Layout {
    /// Number of original matrices and point rows.
    pub matrix_count: usize,
    /// Logical BVA length `K` including the implicit final zero suffix.
    pub logical_len: usize,
    /// Number of physically stored entries `N`.
    pub active_len: usize,
    /// Number of real dyadic buckets `T`.
    pub bucket_count: usize,
    /// Non-increasing power-of-two bucket lengths in dense BVA order.
    pub bucket_sizes: Vec<usize>,
    /// `l = log_2(K)`, the length of the Packed EO point.
    pub global_log: usize,
    /// Low-bit witness width used by Packed EO-LiteBullet.
    pub bit_width: usize,
}

impl Section6Layout {
    pub fn new(
        matrix_count: usize,
        logical_len: usize,
        active_len: usize,
        bucket_sizes: Vec<usize>,
        bit_width: usize,
    ) -> Result<Self, Section6DataError> {
        if matrix_count == 0 {
            return Err(Section6DataError::InvalidLayout(
                "matrix_count must be nonzero".to_owned(),
            ));
        }
        if active_len == 0 {
            return Err(Section6DataError::InvalidLayout(
                "active_len must be nonzero".to_owned(),
            ));
        }
        if bucket_sizes.is_empty() {
            return Err(Section6DataError::InvalidLayout(
                "at least one real bucket is required".to_owned(),
            ));
        }
        if bit_width == 0 || bit_width > 8 {
            return Err(Section6DataError::InvalidLayout(format!(
                "bit_width must be in 1..=8, got {bit_width}"
            )));
        }
        if !logical_len.is_power_of_two() {
            return Err(Section6DataError::InvalidLayout(format!(
                "logical_len {logical_len} is not a power of two"
            )));
        }
        let expected_logical = next_power_of_two(active_len)?;
        if logical_len != expected_logical {
            return Err(Section6DataError::InvalidLayout(format!(
                "logical_len must be next_power_of_two(active_len): expected {expected_logical}, got {logical_len}"
            )));
        }

        let mut offset = 0_usize;
        let mut previous = usize::MAX;
        for (bucket, &size) in bucket_sizes.iter().enumerate() {
            if !size.is_power_of_two() {
                return Err(Section6DataError::InvalidLayout(format!(
                    "bucket {bucket} has non-power-of-two size {size}"
                )));
            }
            if size > previous {
                return Err(Section6DataError::InvalidLayout(format!(
                    "bucket sizes increase at bucket {bucket}: {previous} then {size}"
                )));
            }
            if !offset.is_multiple_of(size) {
                return Err(Section6DataError::InvalidLayout(format!(
                    "bucket {bucket} at offset {offset} is not aligned to size {size}"
                )));
            }
            offset = offset
                .checked_add(size)
                .ok_or(Section6DataError::LengthOverflow)?;
            previous = size;
        }
        if offset != active_len {
            return Err(Section6DataError::InvalidLayout(format!(
                "bucket sizes sum to {offset}, expected active_len {active_len}"
            )));
        }

        let backend_layout = BvaLayout::from_lengths(&bucket_sizes)
            .map_err(|error| Section6DataError::Backend(error.to_string()))?;
        let identity = (0..bucket_sizes.len()).collect::<Vec<_>>();
        if backend_layout.logical_len() != logical_len
            || backend_layout.active_len() != active_len
            || backend_layout.bucket_sizes() != bucket_sizes
            || backend_layout.vector_lengths() != bucket_sizes
            || backend_layout.vector_to_bucket() != identity
        {
            return Err(Section6DataError::InvalidLayout(
                "layout is not the canonical decreasing-size BVA order".to_owned(),
            ));
        }

        Ok(Self {
            matrix_count,
            logical_len,
            active_len,
            bucket_count: bucket_sizes.len(),
            bucket_sizes,
            global_log: logical_len.ilog2() as usize,
            bit_width,
        })
    }

    /// Recheck every layout invariant after deserialization or other
    /// construction that did not go through [`Section6Layout::new`].
    pub fn validate(&self) -> Result<(), Section6DataError> {
        let canonical = Self::new(
            self.matrix_count,
            self.logical_len,
            self.active_len,
            self.bucket_sizes.clone(),
            self.bit_width,
        )?;
        if &canonical != self {
            return Err(Section6DataError::InvalidLayout(
                "derived layout fields are inconsistent with the bucket list".to_owned(),
            ));
        }
        Ok(())
    }

    /// The shape-hiding projection shared with the verifier.
    pub fn public(&self) -> Section6PublicLayout {
        Section6PublicLayout {
            matrix_count: self.matrix_count,
            logical_len: self.logical_len,
            active_len: self.active_len,
            bucket_count: self.bucket_count,
            global_log: self.global_log,
            bit_width: self.bit_width,
        }
    }

    /// Enforce the fixed protocol caps on the private bucket list.
    pub fn validate_fixed_caps(&self) -> Result<(), Section6DataError> {
        self.public().validate_fixed_caps()?;
        if let Some(&largest) = self.bucket_sizes.first() {
            if largest > 1 << SECTION6_FIXED_LOCAL_WIDTH {
                return Err(Section6DataError::InvalidLayout(format!(
                    "bucket size {largest} exceeds the fixed 2^{SECTION6_FIXED_LOCAL_WIDTH} matrix cover"
                )));
            }
        }
        Ok(())
    }

    pub fn from_descriptor(
        descriptor: &DenseBvaDescriptor,
        bit_width: usize,
    ) -> Result<Self, Section6DataError> {
        let bucket_sizes = descriptor
            .buckets()
            .iter()
            .map(|bucket| bucket.entry_count())
            .collect::<Vec<_>>();
        let layout = Self::new(
            descriptor.matrix_count(),
            descriptor.logical_len(),
            descriptor.physical_len(),
            bucket_sizes,
            bit_width,
        )?;
        layout.validate_descriptor(descriptor)?;
        Ok(layout)
    }

    pub fn bva_layout(&self) -> Result<BvaLayout, Section6DataError> {
        BvaLayout::from_lengths(&self.bucket_sizes)
            .map_err(|error| Section6DataError::Backend(error.to_string()))
    }

    pub fn bucket_offsets(&self) -> Result<Vec<usize>, Section6DataError> {
        let mut offset = 0_usize;
        self.bucket_sizes
            .iter()
            .map(|&size| {
                let current = offset;
                offset = offset
                    .checked_add(size)
                    .ok_or(Section6DataError::LengthOverflow)?;
                Ok(current)
            })
            .collect()
    }

    /// Power-of-two bucket-row cover shared by every auxiliary matrix.
    pub fn row_cover(&self) -> Result<usize, Section6DataError> {
        next_power_of_two(self.bucket_count)
    }

    pub fn validate_descriptor(
        &self,
        descriptor: &DenseBvaDescriptor,
    ) -> Result<(), Section6DataError> {
        if descriptor.matrix_count() != self.matrix_count
            || descriptor.logical_len() != self.logical_len
            || descriptor.physical_len() != self.active_len
            || descriptor.bucket_count() != self.bucket_count
            || descriptor.layout().bucket_sizes() != self.bucket_sizes
            || descriptor.layout().vector_lengths() != self.bucket_sizes
        {
            return Err(Section6DataError::InvalidLayout(
                "DenseBvaDescriptor does not match the prover Section6Layout".to_owned(),
            ));
        }
        let identity = (0..self.bucket_count).collect::<Vec<_>>();
        if descriptor.layout().vector_to_bucket() != identity {
            return Err(Section6DataError::InvalidLayout(
                "DenseBvaDescriptor does not use identity bucket ordering".to_owned(),
            ));
        }
        for ((bucket, &expected_size), expected_offset) in descriptor
            .buckets()
            .iter()
            .zip(&self.bucket_sizes)
            .zip(self.bucket_offsets()?)
        {
            if bucket.entry_count() != expected_size || bucket.dense_offset() != expected_offset {
                return Err(Section6DataError::InvalidLayout(
                    "DenseBvaDescriptor bucket range differs from the prover layout".to_owned(),
                ));
            }
        }
        Ok(())
    }
}

/// A field vector padded to an explicit power-of-two commitment domain.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PaddedFieldVector<F: PrimeField> {
    pub logical_len: usize,
    pub padded_len: usize,
    pub values: Vec<F>,
}

impl<F: PrimeField> PaddedFieldVector<F> {
    pub fn from_logical(values: Vec<F>) -> Result<Self, Section6DataError> {
        let cover = padded_dimension(values.len())?;
        Self::from_logical_with_cover(values, cover)
    }

    pub fn from_logical_with_cover(
        mut values: Vec<F>,
        cover: usize,
    ) -> Result<Self, Section6DataError> {
        let logical_len = values.len();
        if !cover.is_power_of_two() || cover < logical_len.max(1) {
            return Err(Section6DataError::InvalidLayout(format!(
                "vector cover {cover} does not cover {logical_len} logical entries"
            )));
        }
        values.resize(cover, F::zero());
        Ok(Self {
            logical_len,
            padded_len: cover,
            values,
        })
    }

    pub fn logical_values(&self) -> &[F] {
        &self.values[..self.logical_len]
    }

    pub fn padding_is_zero(&self) -> bool {
        self.values[self.logical_len..].iter().all(Zero::is_zero)
    }
}

/// A column-major logical matrix embedded in a power-of-two rectangular cover.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PaddedFieldMatrix<F: PrimeField> {
    pub logical_rows: usize,
    pub logical_cols: usize,
    pub padded_rows: usize,
    pub padded_cols: usize,
    pub values: Vec<F>,
}

impl<F: PrimeField> PaddedFieldMatrix<F> {
    pub fn from_fn(
        logical_rows: usize,
        logical_cols: usize,
        value: impl FnMut(usize, usize) -> F,
    ) -> Result<Self, Section6DataError> {
        let row_cover = padded_dimension(logical_rows)?;
        Self::from_fn_with_cover(logical_rows, row_cover, logical_cols, value)
    }

    /// Build a matrix over an explicit shared row cover.
    pub fn from_fn_with_cover(
        logical_rows: usize,
        row_cover: usize,
        logical_cols: usize,
        mut value: impl FnMut(usize, usize) -> F,
    ) -> Result<Self, Section6DataError> {
        if !row_cover.is_power_of_two() || row_cover < logical_rows.max(1) {
            return Err(Section6DataError::InvalidLayout(format!(
                "row cover {row_cover} does not cover {logical_rows} logical rows"
            )));
        }
        let padded_cols = padded_dimension(logical_cols)?;
        let padded_len = row_cover
            .checked_mul(padded_cols)
            .ok_or(Section6DataError::LengthOverflow)?;
        let mut values = vec![F::zero(); padded_len];
        for col in 0..logical_cols {
            let col_start = col
                .checked_mul(row_cover)
                .ok_or(Section6DataError::LengthOverflow)?;
            for row in 0..logical_rows {
                values[col_start + row] = value(row, col);
            }
        }
        Ok(Self {
            logical_rows,
            logical_cols,
            padded_rows: row_cover,
            padded_cols,
            values,
        })
    }

    pub fn logical_row(&self, row: usize) -> Option<Vec<F>> {
        if row >= self.logical_rows {
            return None;
        }
        Some(
            (0..self.logical_cols)
                .map(|col| self.values[col * self.padded_rows + row])
                .collect(),
        )
    }

    pub fn logical_value(&self, row: usize, col: usize) -> Option<F> {
        if row >= self.logical_rows || col >= self.logical_cols {
            return None;
        }
        Some(self.values[col * self.padded_rows + row])
    }

    /// One padded column, including its zero-padded row suffix.
    pub fn padded_column(&self, col: usize) -> Option<&[F]> {
        if col >= self.logical_cols {
            return None;
        }
        let start = col.checked_mul(self.padded_rows)?;
        let end = start.checked_add(self.padded_rows)?;
        self.values.get(start..end)
    }

    /// All logical columns as padded column slices.
    pub fn padded_columns(&self) -> Vec<&[F]> {
        (0..self.logical_cols)
            .map(|col| self.padded_column(col).expect("column is in bounds"))
            .collect()
    }

    pub fn padding_is_zero(&self) -> bool {
        (0..self.padded_rows).all(|row| {
            (0..self.padded_cols).all(|col| {
                (row < self.logical_rows && col < self.logical_cols)
                    || self.values[col * self.padded_rows + row].is_zero()
            })
        })
    }
}

/// Persistent private metadata for the deterministic dyadic split.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrivateShapeMetadata {
    /// One-based source matrix identifiers, as specified by Section 6.
    pub tau: Vec<usize>,
    pub alpha_l: Vec<usize>,
    pub alpha_r: Vec<usize>,
    pub gamma_l: Vec<usize>,
    pub gamma_r: Vec<usize>,
}

impl PrivateShapeMetadata {
    pub fn from_descriptor(
        layout: &Section6Layout,
        descriptor: &DenseBvaDescriptor,
    ) -> Result<Self, Section6DataError> {
        layout.validate_descriptor(descriptor)?;
        let mut tau = Vec::with_capacity(layout.bucket_count);
        let mut alpha_l = Vec::with_capacity(layout.bucket_count);
        let mut alpha_r = Vec::with_capacity(layout.bucket_count);
        let mut gamma_l = Vec::with_capacity(layout.bucket_count);
        let mut gamma_r = Vec::with_capacity(layout.bucket_count);
        for (bucket_index, bucket) in descriptor.buckets().iter().enumerate() {
            if !bucket.row_len().is_power_of_two() || !bucket.col_len().is_power_of_two() {
                return Err(Section6DataError::InvalidMetadata(format!(
                    "bucket {bucket_index} is not a dyadic rectangle"
                )));
            }
            let source_shape = descriptor.matrix_shapes()[bucket.matrix()];
            let row_end = bucket
                .row_offset()
                .checked_add(bucket.row_len())
                .ok_or(Section6DataError::LengthOverflow)?;
            let col_end = bucket
                .col_offset()
                .checked_add(bucket.col_len())
                .ok_or(Section6DataError::LengthOverflow)?;
            if row_end > source_shape.0 || col_end > source_shape.1 {
                return Err(Section6DataError::InvalidMetadata(format!(
                    "bucket {bucket_index} lies outside source matrix {}",
                    bucket.matrix()
                )));
            }
            if !bucket.row_offset().is_multiple_of(bucket.row_len())
                || !bucket.col_offset().is_multiple_of(bucket.col_len())
            {
                return Err(Section6DataError::InvalidMetadata(format!(
                    "bucket {bucket_index} is not aligned to its dyadic shape"
                )));
            }
            let row_log = bucket.row_len().ilog2() as usize;
            let col_log = bucket.col_len().ilog2() as usize;
            if row_log + col_log != layout.bucket_sizes[bucket_index].ilog2() as usize {
                return Err(Section6DataError::InvalidMetadata(format!(
                    "bucket {bucket_index} shape logs do not match its public size"
                )));
            }
            tau.push(
                bucket
                    .matrix()
                    .checked_add(1)
                    .ok_or(Section6DataError::LengthOverflow)?,
            );
            alpha_l.push(bucket.row_offset());
            alpha_r.push(bucket.col_offset());
            gamma_l.push(row_log);
            gamma_r.push(col_log);
        }
        Ok(Self {
            tau,
            alpha_l,
            alpha_r,
            gamma_l,
            gamma_r,
        })
    }

    pub fn validate_against_descriptor(
        &self,
        layout: &Section6Layout,
        descriptor: &DenseBvaDescriptor,
    ) -> Result<(), Section6DataError> {
        self.validate_column_lengths()?;
        let expected = Self::from_descriptor(layout, descriptor)?;
        if self != &expected {
            return Err(Section6DataError::InvalidMetadata(
                "metadata does not encode the descriptor's deterministic split".to_owned(),
            ));
        }
        Ok(())
    }

    fn validate_column_lengths(&self) -> Result<(), Section6DataError> {
        let expected = self.tau.len();
        if expected == 0
            || self.alpha_l.len() != expected
            || self.alpha_r.len() != expected
            || self.gamma_l.len() != expected
            || self.gamma_r.len() != expected
        {
            return Err(Section6DataError::InvalidMetadata(
                "all metadata columns must have the same nonzero length".to_owned(),
            ));
        }
        Ok(())
    }

    /// Validate one bucket row against the fixed caps and the layout.
    fn validate_fixed_bucket(
        &self,
        layout: &Section6Layout,
        bucket: usize,
    ) -> Result<(), Section6DataError> {
        let m = SECTION6_FIXED_POINT_LOG;
        let source = self.tau[bucket];
        if source == 0 || source > layout.matrix_count {
            return Err(Section6DataError::InvalidMetadata(format!(
                "bucket {bucket} has source matrix {source}, expected 1..={}",
                layout.matrix_count
            )));
        }
        let row_log = self.gamma_l[bucket];
        let col_log = self.gamma_r[bucket];
        if row_log > m || col_log > m {
            return Err(Section6DataError::InvalidMetadata(format!(
                "bucket {bucket} has local logs ({row_log}, {col_log}) wider than m={m}"
            )));
        }
        let local_log = row_log
            .checked_add(col_log)
            .ok_or(Section6DataError::LengthOverflow)?;
        if local_log != layout.bucket_sizes[bucket].ilog2() as usize {
            return Err(Section6DataError::InvalidMetadata(format!(
                "bucket {bucket} local logs do not match its public size"
            )));
        }
        let row_len = 1_usize << row_log;
        let col_len = 1_usize << col_log;
        let row_end = self.alpha_l[bucket]
            .checked_add(row_len)
            .ok_or(Section6DataError::LengthOverflow)?;
        let col_end = self.alpha_r[bucket]
            .checked_add(col_len)
            .ok_or(Section6DataError::LengthOverflow)?;
        if row_end > 1 << m || col_end > 1 << m {
            return Err(Section6DataError::InvalidMetadata(format!(
                "bucket {bucket} lies outside the fixed 2^{m} x 2^{m} matrix cover"
            )));
        }
        if !self.alpha_l[bucket].is_multiple_of(row_len)
            || !self.alpha_r[bucket].is_multiple_of(col_len)
        {
            return Err(Section6DataError::InvalidMetadata(format!(
                "bucket {bucket} offsets are not aligned to its dyadic shape"
            )));
        }
        Ok(())
    }

    fn validate_fixed(&self, layout: &Section6Layout) -> Result<(), Section6DataError> {
        layout.validate()?;
        layout.validate_fixed_caps()?;
        self.validate_column_lengths()?;
        if self.tau.len() != layout.bucket_count {
            return Err(Section6DataError::InvalidMetadata(format!(
                "metadata row count mismatch: expected {}, got {}",
                layout.bucket_count,
                self.tau.len()
            )));
        }
        for bucket in 0..layout.bucket_count {
            self.validate_fixed_bucket(layout, bucket)?;
        }
        Ok(())
    }

    /// Little-endian `D_eta` bit at `(bucket, coordinate)`: column-offset bits
    /// first, then row-offset bits, then the bits of `tau - 1`.
    fn eta_mask_bit(&self, bucket: usize, coordinate: usize) -> bool {
        let m = SECTION6_FIXED_POINT_LOG;
        if coordinate < m {
            (self.alpha_r[bucket] >> coordinate) & 1 == 1
        } else if coordinate < SECTION6_FIXED_LOCAL_WIDTH {
            (self.alpha_l[bucket] >> (coordinate - m)) & 1 == 1
        } else {
            ((self.tau[bucket] - 1) >> (coordinate - SECTION6_FIXED_LOCAL_WIDTH)) & 1 == 1
        }
    }

    /// `D_Z` bit at `(bucket, coordinate)`: varying column coordinates first,
    /// then varying row coordinates.
    fn local_mask_bit(&self, bucket: usize, coordinate: usize) -> bool {
        let m = SECTION6_FIXED_POINT_LOG;
        if coordinate < m {
            coordinate < self.gamma_r[bucket]
        } else {
            coordinate - m < self.gamma_l[bucket]
        }
    }
}

/// The three registered Boolean mask matrices `D_eta`, `D_Z`, and `D_eta'`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Section6RegisteredMasks<F: PrimeField> {
    /// Fixed-coordinate selector, logical shape `T x (2m+d)`.
    pub d_eta: PaddedFieldMatrix<F>,
    /// Varying-coordinate selector, logical shape `T x 2m`.
    pub d_z: PaddedFieldMatrix<F>,
    /// Global-offset bits, logical shape `T x l`.
    pub d_eta_prime: PaddedFieldMatrix<F>,
}

impl<F: PrimeField> Section6RegisteredMasks<F> {
    /// Construct the three registered masks over the shared bucket-row cover.
    pub fn from_metadata(
        layout: &Section6Layout,
        metadata: &PrivateShapeMetadata,
    ) -> Result<Self, Section6DataError> {
        metadata.validate_fixed(layout)?;
        let rows = layout.bucket_count;
        let cover = layout.row_cover()?;
        let d_eta =
            PaddedFieldMatrix::from_fn_with_cover(rows, cover, SECTION6_FIXED_ETA_WIDTH, |j, q| {
                bool_to_field(metadata.eta_mask_bit(j, q))
            })?;
        let d_z = PaddedFieldMatrix::from_fn_with_cover(
            rows,
            cover,
            SECTION6_FIXED_LOCAL_WIDTH,
            |j, q| bool_to_field(metadata.local_mask_bit(j, q)),
        )?;
        let offsets = layout.bucket_offsets()?;
        let d_eta_prime =
            PaddedFieldMatrix::from_fn_with_cover(rows, cover, layout.global_log, |j, q| {
                bool_to_field((offsets[j] >> q) & 1 == 1)
            })?;
        Ok(Self {
            d_eta,
            d_z,
            d_eta_prime,
        })
    }
}

/// The public per-coordinate batching scalars `x_q = omega^{2^q}`, so that
/// the induced tensor weight of matrix `tau` is `omega^{tau - 1}`.
pub fn section6_batching_point<F: PrimeField>(omega: F) -> Vec<F> {
    let mut powers = Vec::with_capacity(SECTION6_FIXED_MATRIX_LOG);
    let mut current = omega;
    for _ in 0..SECTION6_FIXED_MATRIX_LOG {
        powers.push(current);
        current = current * current;
    }
    powers
}

/// Per-opening point-compilation witness over the shared bucket-row cover.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Section6MaskPointWitness<F: PrimeField> {
    /// Point matrix `X = (X_r | X_l)`, logical `t x 2m`, committed pre-omega.
    pub x: PaddedFieldMatrix<F>,
    /// One-based source identifiers `tau`, logical length `T`.
    pub tau: PaddedFieldVector<F>,
    /// Gathered source points `X' = X_{tau_j}`, logical `T x 2m`.
    pub x_selected: PaddedFieldMatrix<F>,
    /// Running count of `D_Z` ones per row, logical `T x 2m`.
    pub sigma: PaddedFieldMatrix<F>,
    /// Masked ranks `sigma^ = D_Z o sigma`, logical `T x 2m`.
    pub sigma_masked: PaddedFieldMatrix<F>,
    /// Masked coordinates `Z~ = D_Z o X'`, logical `T x 2m`.
    pub z_tilde: PaddedFieldMatrix<F>,
    /// Compacted local points `Z`, logical `T x min(2m, l)`.
    pub z: PaddedFieldMatrix<F>,
    /// Grand-product prefixes `P` of the eta factors, logical `T x (2m+d)`.
    pub eta_prefix: PaddedFieldMatrix<F>,
    /// Compiled TBV scales `eta` (copy of the last prefix column).
    pub eta: PaddedFieldVector<F>,
    /// Public batching scalars `x_q = omega^{2^q}`.
    pub batching_point: Vec<F>,
    /// `c = sum_tau omega^{tau-1} u_tau`.
    pub aggregated_claim: F,
}

impl<F: PrimeField> Section6MaskPointWitness<F> {
    pub fn compile(
        layout: &Section6Layout,
        descriptor: &DenseBvaDescriptor,
        metadata: &PrivateShapeMetadata,
        claims: &[Section6MaskMatrixClaim<F>],
        omega: F,
    ) -> Result<Self, Section6DataError> {
        layout.validate_descriptor(descriptor)?;
        metadata.validate_against_descriptor(layout, descriptor)?;
        metadata.validate_fixed(layout)?;
        validate_fixed_claims(layout, claims)?;
        if omega.is_zero() {
            return Err(Section6DataError::InvalidPoint(
                "omega must be nonzero".to_owned(),
            ));
        }

        let m = SECTION6_FIXED_POINT_LOG;
        let rows = layout.bucket_count;
        let cover = layout.row_cover()?;
        let local_width = layout.public().local_width();

        let x = PaddedFieldMatrix::from_fn_with_cover(
            layout.matrix_count,
            cover,
            SECTION6_FIXED_LOCAL_WIDTH,
            |row, col| {
                if col < m {
                    claims[row].right_point[col]
                } else {
                    claims[row].left_point[col - m]
                }
            },
        )?;
        let tau = PaddedFieldVector::from_logical_with_cover(
            metadata
                .tau
                .iter()
                .map(|&source| field_from_usize(source))
                .collect(),
            cover,
        )?;
        let x_selected = PaddedFieldMatrix::from_fn_with_cover(
            rows,
            cover,
            SECTION6_FIXED_LOCAL_WIDTH,
            |row, col| {
                x.logical_value(metadata.tau[row] - 1, col)
                    .expect("tau is a validated one-based source index")
            },
        )?;

        // Columns are visited in increasing order for every row, so one
        // per-row accumulator yields the running rank.
        let mut rank_rows = vec![0_usize; rows];
        let sigma = PaddedFieldMatrix::from_fn_with_cover(
            rows,
            cover,
            SECTION6_FIXED_LOCAL_WIDTH,
            |row, col| {
                if col == 0 {
                    rank_rows[row] = 0;
                }
                if metadata.local_mask_bit(row, col) {
                    rank_rows[row] += 1;
                }
                field_from_usize(rank_rows[row])
            },
        )?;
        let sigma_masked = PaddedFieldMatrix::from_fn_with_cover(
            rows,
            cover,
            SECTION6_FIXED_LOCAL_WIDTH,
            |row, col| {
                if metadata.local_mask_bit(row, col) {
                    sigma
                        .logical_value(row, col)
                        .expect("sigma covers the local width")
                } else {
                    F::zero()
                }
            },
        )?;
        let z_tilde = PaddedFieldMatrix::from_fn_with_cover(
            rows,
            cover,
            SECTION6_FIXED_LOCAL_WIDTH,
            |row, col| {
                if metadata.local_mask_bit(row, col) {
                    x_selected
                        .logical_value(row, col)
                        .expect("X' covers the local width")
                } else {
                    F::zero()
                }
            },
        )?;
        let z = PaddedFieldMatrix::from_fn_with_cover(rows, cover, local_width, |row, col| {
            let source = metadata.tau[row] - 1;
            if col < metadata.gamma_r[row] {
                claims[source].right_point[col]
            } else if col < metadata.gamma_r[row] + metadata.gamma_l[row] {
                claims[source].left_point[col - metadata.gamma_r[row]]
            } else {
                F::zero()
            }
        })?;

        let batching_point = section6_batching_point(omega);
        let mut prefix_rows = vec![vec![F::zero(); SECTION6_FIXED_ETA_WIDTH]; rows];
        for (row, prefix_row) in prefix_rows.iter_mut().enumerate() {
            let mut running = F::one();
            for col in 0..SECTION6_FIXED_ETA_WIDTH {
                if metadata.eta_mask_bit(row, col) {
                    if col < SECTION6_FIXED_LOCAL_WIDTH {
                        running *= x_selected
                            .logical_value(row, col)
                            .expect("X' covers the eta width prefix");
                    } else {
                        running *= batching_point[col - SECTION6_FIXED_LOCAL_WIDTH];
                    }
                }
                prefix_row[col] = running;
            }
        }
        let eta_prefix = PaddedFieldMatrix::from_fn_with_cover(
            rows,
            cover,
            SECTION6_FIXED_ETA_WIDTH,
            |row, col| prefix_rows[row][col],
        )?;
        let eta = PaddedFieldVector::from_logical_with_cover(
            prefix_rows
                .iter()
                .map(|row| row[SECTION6_FIXED_ETA_WIDTH - 1])
                .collect(),
            cover,
        )?;

        let mut omega_power = F::one();
        let mut aggregated_claim = F::zero();
        for claim in claims {
            aggregated_claim += claim.value * omega_power;
            omega_power *= omega;
        }

        Ok(Self {
            x,
            tau,
            x_selected,
            sigma,
            sigma_masked,
            z_tilde,
            z,
            eta_prefix,
            eta,
            batching_point,
            aggregated_claim,
        })
    }

    /// The last eta-prefix entry of one bucket row (equal to `eta`).
    pub fn prefix_last_column_value(&self, bucket: usize) -> Option<F> {
        self.eta_prefix
            .logical_value(bucket, SECTION6_FIXED_ETA_WIDTH - 1)
    }

    pub fn local_point(&self, bucket: usize, len: usize) -> Result<Vec<F>, Section6DataError> {
        let row = self.z.logical_row(bucket).ok_or_else(|| {
            Section6DataError::InvalidPoint(format!("missing Z row for bucket {bucket}"))
        })?;
        if len > row.len() {
            return Err(Section6DataError::InvalidPoint(format!(
                "bucket {bucket} local point length {len} exceeds Z width {}",
                row.len()
            )));
        }
        Ok(row[..len].to_vec())
    }

    pub fn reconstruct_tbv(
        &self,
        layout: &Section6Layout,
    ) -> Result<TensorBasisVector<F>, Section6DataError> {
        if self.eta.logical_len != layout.bucket_count
            || self.z.logical_rows != layout.bucket_count
        {
            return Err(Section6DataError::InvalidPoint(
                "point witness dimensions do not match Section6Layout".to_owned(),
            ));
        }
        let buckets = layout
            .bucket_sizes
            .iter()
            .enumerate()
            .map(|(bucket, &size)| {
                let local_log = size.ilog2() as usize;
                Ok(TbvBucket::new(
                    self.eta.logical_values()[bucket],
                    self.local_point(bucket, local_log)?,
                ))
            })
            .collect::<Result<Vec<_>, Section6DataError>>()?;
        TensorBasisVector::new(&layout.bva_layout()?, buckets)
            .map_err(|error| Section6DataError::Backend(error.to_string()))
    }
}

/// Prover-only witness for the final TBV evaluation at the Packed EO point.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Section6MaskFinalWitness<F: PrimeField> {
    /// Derived prefix mask `D_Z' = Prefix(D_Z 1)`, logical `T x min(2m, l)`.
    pub d_z_prime: PaddedFieldMatrix<F>,
    /// Grand-product prefixes of the eta' factors, logical `T x l`.
    pub eta_prime_prefix: PaddedFieldMatrix<F>,
    /// Restriction scales `eta'` (copy of the last prefix column).
    pub eta_prime: PaddedFieldVector<F>,
    /// Restriction points `Z' = D_Z' o (1 y^T)`, logical `T x min(2m, l)`.
    pub z_prime: PaddedFieldMatrix<F>,
    /// Product prefixes `Q` with `Q_0 = eta o eta'`, logical `T x (w+1)`.
    pub v_prefix: PaddedFieldMatrix<F>,
    /// `b = sum_j v_j`, the claimed TBV evaluation.
    pub evaluation: F,
}

impl<F: PrimeField> Section6MaskFinalWitness<F> {
    pub fn compile(
        layout: &Section6Layout,
        point: &Section6MaskPointWitness<F>,
        y: &[F],
    ) -> Result<Self, Section6DataError> {
        let rows = layout.bucket_count;
        let cover = layout.row_cover()?;
        let local_width = layout.public().local_width();
        if y.len() != layout.global_log {
            return Err(Section6DataError::InvalidPoint(format!(
                "final point length mismatch: expected {}, got {}",
                layout.global_log,
                y.len()
            )));
        }
        if point.eta.logical_len != rows
            || point.z.logical_rows != rows
            || point.z.logical_cols != local_width
        {
            return Err(Section6DataError::InvalidPoint(
                "final witness inputs do not match the layout".to_owned(),
            ));
        }
        let gamma_prime = layout
            .bucket_sizes
            .iter()
            .map(|size| size.ilog2() as usize)
            .collect::<Vec<_>>();
        if gamma_prime.iter().any(|&log| log > local_width) {
            return Err(Section6DataError::InvalidPoint(
                "a bucket exceeds the compiled local width".to_owned(),
            ));
        }
        let offsets = layout.bucket_offsets()?;

        let d_z_prime =
            PaddedFieldMatrix::from_fn_with_cover(rows, cover, local_width, |row, col| {
                bool_to_field(col < gamma_prime[row])
            })?;
        let z_prime =
            PaddedFieldMatrix::from_fn_with_cover(rows, cover, local_width, |row, col| {
                if col < gamma_prime[row] {
                    y[col]
                } else {
                    F::zero()
                }
            })?;
        let mut eta_prime_rows = vec![vec![F::zero(); layout.global_log]; rows];
        for (row, prefix_row) in eta_prime_rows.iter_mut().enumerate() {
            let mut running = F::one();
            for bit in 0..layout.global_log {
                if (offsets[row] >> bit) & 1 == 1 {
                    running *= y[bit];
                }
                prefix_row[bit] = running;
            }
        }
        let eta_prime_prefix =
            PaddedFieldMatrix::from_fn_with_cover(rows, cover, layout.global_log, |row, col| {
                eta_prime_rows[row][col]
            })?;
        let eta_prime = PaddedFieldVector::from_logical_with_cover(
            eta_prime_rows
                .iter()
                .map(|row| row[layout.global_log - 1])
                .collect(),
            cover,
        )?;

        let mut v_rows = vec![vec![F::zero(); local_width + 1]; rows];
        for (row, v_row) in v_rows.iter_mut().enumerate() {
            let mut running = point.eta.logical_values()[row] * eta_prime.logical_values()[row];
            v_row[0] = running;
            for col in 0..local_width {
                let z = point
                    .z
                    .logical_value(row, col)
                    .expect("validated Z bounds");
                let z_prime_entry = z_prime
                    .logical_value(row, col)
                    .expect("constructed Z' bounds");
                running *= F::one() + z * z_prime_entry;
                v_row[col + 1] = running;
            }
        }
        let v_prefix =
            PaddedFieldMatrix::from_fn_with_cover(rows, cover, local_width + 1, |row, col| {
                v_rows[row][col]
            })?;
        let evaluation = v_rows.iter().map(|row| row[local_width]).sum();
        Ok(Self {
            d_z_prime,
            eta_prime_prefix,
            eta_prime,
            z_prime,
            v_prefix,
            evaluation,
        })
    }
}

/// Validate fixed-width claims against the layout.
pub fn validate_fixed_claims<F: PrimeField>(
    layout: &Section6Layout,
    claims: &[Section6MaskMatrixClaim<F>],
) -> Result<(), Section6DataError> {
    if claims.len() != layout.matrix_count {
        return Err(Section6DataError::InvalidClaim(format!(
            "claim count mismatch: expected {}, got {}",
            layout.matrix_count,
            claims.len()
        )));
    }
    for claim in claims {
        claim.validate()?;
    }
    Ok(())
}

fn padded_dimension(value: usize) -> Result<usize, Section6DataError> {
    if value == 0 {
        Ok(1)
    } else {
        next_power_of_two(value)
    }
}

fn next_power_of_two(value: usize) -> Result<usize, Section6DataError> {
    value
        .checked_next_power_of_two()
        .ok_or(Section6DataError::LengthOverflow)
}

fn bool_to_field<F: PrimeField>(bit: bool) -> F {
    if bit {
        F::one()
    } else {
        F::zero()
    }
}

pub(crate) fn field_from_usize<F: PrimeField>(value: usize) -> F {
    F::from_le_bytes_mod_order(&value.to_le_bytes())
}

#[cfg(test)]
mod tests {
    use ark_ff::One;

    use super::*;
    use crate::matrix_collection::MatrixMleClaim;

    fn field_values(start: u64, len: usize) -> Vec<Fr> {
        (0..len)
            .map(|index| Fr::from(start + index as u64))
            .collect()
    }

    fn asymmetric_fixture() -> (DenseBvaDescriptor, Vec<Section6MaskMatrixClaim<Fr>>) {
        let shapes = [(3, 5), (5, 2), (2, 7)];
        let descriptor = DenseBvaDescriptor::from_shapes(&shapes).unwrap();
        let claims = vec![
            Section6MaskMatrixClaim::from_unpadded(
                &field_values(11, 2),
                &field_values(21, 3),
                Fr::from(31_u64),
            )
            .unwrap(),
            Section6MaskMatrixClaim::from_unpadded(
                &field_values(41, 3),
                &field_values(51, 1),
                Fr::from(61_u64),
            )
            .unwrap(),
            Section6MaskMatrixClaim::from_unpadded(
                &field_values(71, 1),
                &field_values(81, 3),
                Fr::from(91_u64),
            )
            .unwrap(),
        ];
        (descriptor, claims)
    }

    fn oracle_claims(
        shapes: &[(usize, usize)],
        claims: &[Section6MaskMatrixClaim<Fr>],
    ) -> Vec<MatrixMleClaim<Fr>> {
        shapes
            .iter()
            .zip(claims)
            .map(|(&shape, claim)| {
                let left_len = shape.0.next_power_of_two().ilog2() as usize;
                let right_len = shape.1.next_power_of_two().ilog2() as usize;
                MatrixMleClaim::new(
                    shape,
                    claim.left_point[..left_len].to_vec(),
                    claim.right_point[..right_len].to_vec(),
                    claim.value,
                )
                .unwrap()
            })
            .collect()
    }

    #[test]
    fn registered_masks_encode_the_three_boolean_selectors() {
        let (descriptor, _) = asymmetric_fixture();
        let layout = Section6Layout::from_descriptor(&descriptor, 8).unwrap();
        let metadata = PrivateShapeMetadata::from_descriptor(&layout, &descriptor).unwrap();
        let masks = Section6RegisteredMasks::<Fr>::from_metadata(&layout, &metadata).unwrap();
        let m = SECTION6_FIXED_POINT_LOG;
        let one = Fr::one();
        let cover = layout.row_cover().unwrap();
        let offsets = layout.bucket_offsets().unwrap();

        assert_eq!(
            (masks.d_eta.logical_rows, masks.d_eta.logical_cols),
            (layout.bucket_count, SECTION6_FIXED_ETA_WIDTH)
        );
        assert_eq!(
            (masks.d_z.logical_rows, masks.d_z.logical_cols),
            (layout.bucket_count, SECTION6_FIXED_LOCAL_WIDTH)
        );
        assert_eq!(
            (
                masks.d_eta_prime.logical_rows,
                masks.d_eta_prime.logical_cols,
            ),
            (layout.bucket_count, layout.global_log)
        );
        assert!(masks.d_eta.padded_rows == cover && masks.d_z.padded_rows == cover);
        assert!(masks.d_eta.padding_is_zero());
        assert!(masks.d_z.padding_is_zero());
        assert!(masks.d_eta_prime.padding_is_zero());

        for bucket in 0..layout.bucket_count {
            let eta_row = masks.d_eta.logical_row(bucket).unwrap();
            let z_row = masks.d_z.logical_row(bucket).unwrap();
            let prime_row = masks.d_eta_prime.logical_row(bucket).unwrap();
            assert!(eta_row
                .iter()
                .chain(&z_row)
                .chain(&prime_row)
                .all(|entry| entry.is_zero() || *entry == one));

            let decoded_alpha_r = (0..m).fold(0_usize, |value, bit| {
                value | ((eta_row[bit] == one) as usize) << bit
            });
            let decoded_alpha_l = (0..m).fold(0_usize, |value, bit| {
                value | ((eta_row[m + bit] == one) as usize) << bit
            });
            let decoded_tau = (0..SECTION6_FIXED_MATRIX_LOG).fold(0_usize, |value, bit| {
                value | ((eta_row[SECTION6_FIXED_LOCAL_WIDTH + bit] == one) as usize) << bit
            });
            assert_eq!(decoded_alpha_r, metadata.alpha_r[bucket]);
            assert_eq!(decoded_alpha_l, metadata.alpha_l[bucket]);
            assert_eq!(decoded_tau + 1, metadata.tau[bucket]);

            for coordinate in 0..SECTION6_FIXED_LOCAL_WIDTH {
                let expected = if coordinate < m {
                    coordinate < metadata.gamma_r[bucket]
                } else {
                    coordinate - m < metadata.gamma_l[bucket]
                };
                assert_eq!(z_row[coordinate] == one, expected);
                // Alignment keeps the varying coordinates disjoint from the
                // fixed offset bits.
                assert!(!(z_row[coordinate] == one && eta_row[coordinate] == one));
            }

            let decoded_offset = (0..layout.global_log).fold(0_usize, |value, bit| {
                value | ((prime_row[bit] == one) as usize) << bit
            });
            assert_eq!(decoded_offset, offsets[bucket]);
        }
    }

    #[test]
    fn point_witness_matches_dense_descriptor_compiler() {
        let (descriptor, claims) = asymmetric_fixture();
        let layout = Section6Layout::from_descriptor(&descriptor, 8).unwrap();
        let metadata = PrivateShapeMetadata::from_descriptor(&layout, &descriptor).unwrap();
        let omega = Fr::from(7_u64);
        let point =
            Section6MaskPointWitness::compile(&layout, &descriptor, &metadata, &claims, omega)
                .unwrap();
        let oracle = descriptor
            .compile_with_omega(&oracle_claims(&[(3, 5), (5, 2), (2, 7)], &claims), omega, 8)
            .unwrap();

        assert_eq!(point.aggregated_claim, oracle.claimed);
        assert_eq!(point.reconstruct_tbv(&layout).unwrap(), oracle.tbv);

        // eta equals the last prefix column, and the batching point encodes
        // omega^{tau-1} through the eta-mask tau bits.
        for bucket in 0..layout.bucket_count {
            assert_eq!(
                point.eta.logical_values()[bucket],
                point
                    .prefix_last_column_value(bucket)
                    .expect("bucket is in bounds")
            );
        }

        let y = field_values(101, layout.global_log);
        let final_witness = Section6MaskFinalWitness::compile(&layout, &point, &y).unwrap();
        assert_eq!(
            final_witness.evaluation,
            oracle.tbv.fold_to_scalar(&y).unwrap(),
        );
        assert!(final_witness.d_z_prime.padding_is_zero());
        assert!(final_witness.z_prime.padding_is_zero());
        assert!(final_witness.eta_prime_prefix.padding_is_zero());
        assert!(final_witness.v_prefix.padding_is_zero());

        // The v recurrence terminates at the row sum claimed by the packed
        // reduction.
        let width = layout.public().local_width();
        let mut total = Fr::zero();
        for bucket in 0..layout.bucket_count {
            total += final_witness.v_prefix.logical_value(bucket, width).unwrap();
        }
        assert_eq!(total, final_witness.evaluation);
    }

    #[test]
    fn point_witness_rows_follow_fixed_width_conventions() {
        let (descriptor, claims) = asymmetric_fixture();
        let layout = Section6Layout::from_descriptor(&descriptor, 8).unwrap();
        let metadata = PrivateShapeMetadata::from_descriptor(&layout, &descriptor).unwrap();
        let omega = Fr::from(7_u64);
        let point =
            Section6MaskPointWitness::compile(&layout, &descriptor, &metadata, &claims, omega)
                .unwrap();
        let m = SECTION6_FIXED_POINT_LOG;
        let cover = layout.row_cover().unwrap();

        assert_eq!((layout.active_len, layout.logical_len), (39, 64));
        assert_eq!(
            (point.x.logical_rows, point.x.logical_cols),
            (3, SECTION6_FIXED_LOCAL_WIDTH)
        );
        assert_eq!((point.x.padded_rows, point.x.padded_cols), (cover, 32));
        assert!(point.x.padding_is_zero());
        assert!(point.tau.padding_is_zero());
        assert!(point.x_selected.padding_is_zero());
        assert!(point.sigma.padding_is_zero());
        assert!(point.sigma_masked.padding_is_zero());
        assert!(point.z_tilde.padding_is_zero());
        assert!(point.z.padding_is_zero());
        assert!(point.eta_prefix.padding_is_zero());
        assert!(point.eta.padding_is_zero());

        // X' rows are exact copies of the source (X_r || X_l) rows, and the
        // masked ranks enumerate the D_Z selections in column order.
        for bucket in 0..layout.bucket_count {
            let source = metadata.tau[bucket] - 1;
            let row = point.x_selected.logical_row(bucket).unwrap();
            assert_eq!(&row[..m], &claims[source].right_point[..]);
            assert_eq!(&row[m..], &claims[source].left_point[..]);

            let sigma_row = point.sigma.logical_row(bucket).unwrap();
            let masked_row = point.sigma_masked.logical_row(bucket).unwrap();
            let z_tilde_row = point.z_tilde.logical_row(bucket).unwrap();
            let mut rank = 0_usize;
            for coordinate in 0..SECTION6_FIXED_LOCAL_WIDTH {
                let selected = if coordinate < m {
                    coordinate < metadata.gamma_r[bucket]
                } else {
                    coordinate - m < metadata.gamma_l[bucket]
                };
                if selected {
                    rank += 1;
                    assert_eq!(masked_row[coordinate], Fr::from(rank as u64));
                    assert_eq!(z_tilde_row[coordinate], row[coordinate]);
                    // The compacted matrix holds the same value at the rank.
                    assert_eq!(
                        point.z.logical_value(bucket, rank - 1).unwrap(),
                        row[coordinate]
                    );
                } else {
                    assert!(masked_row[coordinate].is_zero());
                    assert!(z_tilde_row[coordinate].is_zero());
                }
                assert_eq!(sigma_row[coordinate], Fr::from(rank as u64));
            }
            let gamma = metadata.gamma_r[bucket] + metadata.gamma_l[bucket];
            for col in gamma..layout.public().local_width() {
                assert!(point.z.logical_value(bucket, col).unwrap().is_zero());
            }
        }

        let interior = descriptor
            .buckets()
            .iter()
            .position(|bucket| {
                bucket.matrix() == 0
                    && bucket.row_offset() == 0
                    && bucket.row_len() == 2
                    && bucket.col_offset() == 0
                    && bucket.col_len() == 4
            })
            .unwrap();
        assert_eq!(metadata.tau[interior], 1);
        assert_eq!(metadata.gamma_l[interior], 1);
        assert_eq!(metadata.gamma_r[interior], 2);
        let mut expected_z = vec![Fr::from(0_u64); layout.public().local_width()];
        expected_z[..3].copy_from_slice(&[Fr::from(21_u64), Fr::from(22_u64), Fr::from(11_u64)]);
        assert_eq!(point.z.logical_row(interior).unwrap(), expected_z);
        assert_eq!(point.eta.logical_values()[interior], Fr::from(1_u64));

        let fixed = descriptor
            .buckets()
            .iter()
            .position(|bucket| {
                bucket.matrix() == 0
                    && bucket.row_offset() == 2
                    && bucket.row_len() == 1
                    && bucket.col_offset() == 4
                    && bucket.col_len() == 1
            })
            .unwrap();
        assert_eq!(metadata.tau[fixed], 1);
        assert_eq!((metadata.alpha_l[fixed], metadata.alpha_r[fixed]), (2, 4));
        assert_eq!((metadata.gamma_l[fixed], metadata.gamma_r[fixed]), (0, 0));
        assert!(point
            .z
            .logical_row(fixed)
            .unwrap()
            .iter()
            .all(Zero::is_zero));
        assert_eq!(
            point.eta.logical_values()[fixed],
            Fr::from(23_u64) * Fr::from(12_u64)
        );
    }

    #[test]
    fn oracle_matches_for_multiple_asymmetric_non_power_of_two_shape_sets() {
        let shape_sets = [
            vec![(6, 3), (1, 9), (7, 1), (3, 6)],
            vec![(9, 5), (2, 11), (5, 3)],
        ];
        for (case, shapes) in shape_sets.iter().enumerate() {
            let descriptor = DenseBvaDescriptor::from_shapes(shapes).unwrap();
            let claims = shapes
                .iter()
                .enumerate()
                .map(|(matrix, &shape)| {
                    let left_len = shape.0.next_power_of_two().ilog2() as usize;
                    let right_len = shape.1.next_power_of_two().ilog2() as usize;
                    let base = 200 + (case * 100 + matrix * 20) as u64;
                    Section6MaskMatrixClaim::from_unpadded(
                        &field_values(base, left_len),
                        &field_values(base + 10, right_len),
                        Fr::from(base + 19),
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            let layout = Section6Layout::from_descriptor(&descriptor, 4).unwrap();
            let metadata = PrivateShapeMetadata::from_descriptor(&layout, &descriptor).unwrap();
            let omega = Fr::from(13_u64 + case as u64);
            let point =
                Section6MaskPointWitness::compile(&layout, &descriptor, &metadata, &claims, omega)
                    .unwrap();
            let oracle = descriptor
                .compile_with_omega(&oracle_claims(shapes, &claims), omega, 4)
                .unwrap();
            assert_eq!(point.aggregated_claim, oracle.claimed);
            assert_eq!(point.reconstruct_tbv(&layout).unwrap(), oracle.tbv);

            let y = field_values(401 + 50 * case as u64, layout.global_log);
            let final_witness = Section6MaskFinalWitness::compile(&layout, &point, &y).unwrap();
            assert_eq!(
                final_witness.evaluation,
                oracle.tbv.fold_to_scalar(&y).unwrap()
            );
        }
    }

    #[test]
    fn fixed_caps_reject_oversized_collections() {
        let (descriptor, _) = asymmetric_fixture();
        let layout = Section6Layout::from_descriptor(&descriptor, 8).unwrap();
        layout.validate_fixed_caps().unwrap();
        let mut public = layout.public();
        public.validate().unwrap();
        public.matrix_count = (1 << SECTION6_FIXED_MATRIX_LOG) + 1;
        assert!(public.validate_fixed_caps().is_err());
    }

    #[test]
    fn public_layout_hides_bucket_sizes_but_keeps_the_covers() {
        let (descriptor, _) = asymmetric_fixture();
        let layout = Section6Layout::from_descriptor(&descriptor, 8).unwrap();
        let public = layout.public();
        public.validate().unwrap();
        assert_eq!(public.bucket_count, layout.bucket_count);
        assert_eq!(public.row_cover().unwrap(), layout.row_cover().unwrap());
        assert_eq!(public.local_width(), layout.global_log.min(30));
        assert_eq!(public.global_log, layout.global_log);
    }

    #[test]
    fn prover_layout_rejects_increasing_or_materialized_suffix_buckets() {
        assert!(Section6Layout::new(1, 8, 6, vec![2, 4], 8).is_err());
        assert!(Section6Layout::new(1, 8, 6, vec![4, 2, 2], 8).is_err());
        assert!(Section6Layout::new(1, 8, 6, vec![4, 2], 0).is_err());
    }

    #[test]
    fn prover_layout_validation_rejects_mutated_derived_fields() {
        let layout = Section6Layout::new(2, 8, 6, vec![4, 2], 8).unwrap();
        layout.validate().unwrap();

        let mut wrong_bucket_count = layout.clone();
        wrong_bucket_count.bucket_count += 1;
        assert!(wrong_bucket_count.validate().is_err());

        let mut wrong_global_log = layout.clone();
        wrong_global_log.global_log += 1;
        assert!(wrong_global_log.validate().is_err());

        let mut increasing = layout;
        increasing.bucket_sizes.swap(0, 1);
        assert!(increasing.validate().is_err());
    }

    #[test]
    fn padded_section6_matrix_uses_column_major_order() {
        let matrix =
            PaddedFieldMatrix::<Fr>::from_fn(2, 3, |row, col| Fr::from((10 * row + col) as u64))
                .unwrap();

        assert_eq!(matrix.padded_rows, 2);
        assert_eq!(matrix.padded_cols, 4);
        assert_eq!(
            matrix.values,
            vec![
                Fr::from(0_u64),
                Fr::from(10_u64),
                Fr::from(1_u64),
                Fr::from(11_u64),
                Fr::from(2_u64),
                Fr::from(12_u64),
                Fr::from(0_u64),
                Fr::from(0_u64),
            ]
        );
        assert_eq!(matrix.logical_value(1, 2), Some(Fr::from(12_u64)));
        assert_eq!(
            matrix.padded_column(2).unwrap(),
            &[Fr::from(2_u64), Fr::from(12_u64)]
        );
        assert!(matrix.padding_is_zero());

        let covered = PaddedFieldMatrix::<Fr>::from_fn_with_cover(2, 8, 3, |row, col| {
            Fr::from((10 * row + col) as u64)
        })
        .unwrap();
        assert_eq!(covered.padded_rows, 8);
        assert_eq!(covered.padded_column(1).unwrap()[1], Fr::from(11_u64));
        assert!(covered.padded_column(1).unwrap()[2..].iter().all(Zero::is_zero));
        assert!(covered.padding_is_zero());
    }
}
