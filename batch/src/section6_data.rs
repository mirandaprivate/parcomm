//! Prover-side data construction for the Section 6 point compiler.
//!
//! This module follows the Boolean-mask shape encoding of the specification.
//! Every point is zero-padded to `m` coordinates per side, where `m` is the
//! public log bound on the matrix dimensions, and the persistent shape
//! encoding consists of two Boolean matrices over the bucket-row cover:
//!
//! * `D_eta in {0,1}^{T x (2m+d)}` - the offset bits of `alpha_r` then
//!   `alpha_l`, followed by the bits of the zero-based source index;
//! * `D_Z   in {0,1}^{T x 2m}`     - the prefix masks `1^{gamma_r} 0..`
//!   then `1^{gamma_l} 0..` selecting the varying coordinates.
//!
//! The public layout additionally determines two derived masks:
//!
//! * `D_eta' in {0,1}^{T x ell}` - the bit decompositions of the global
//!   bucket offsets, with `ell = log_2 K`;
//! * `D_Z'   in {0,1}^{T x 2m}`  - the prefix masks `1^{log_2 k_j} 0..`.
//!
//! The layout `L = (K, N, T, {k_j}, m, d)` is public; only `D_eta` and
//! `D_Z` (that is, which source matrix each bucket reads and where) stay
//! private.
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

/// One private per-matrix MLE claim with points padded to `m = log_2 K`.
///
/// Because the tensor span is monomial (`weight(i) = prod_{bit set in i}
/// x_bit`), the padded coordinates never contribute to the claimed value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Section6MatrixClaim<F: PrimeField> {
    /// Left (row) point zero-padded to `m` coordinates.
    pub left_point: Vec<F>,
    /// Right (column) point zero-padded to `m` coordinates.
    pub right_point: Vec<F>,
    pub value: F,
}

impl<F: PrimeField> Section6MatrixClaim<F> {
    pub fn from_unpadded(
        point_log: usize,
        left_point: &[F],
        right_point: &[F],
        value: F,
    ) -> Result<Self, Section6DataError> {
        if left_point.len() > point_log || right_point.len() > point_log {
            return Err(Section6DataError::InvalidClaim(format!(
                "claim points must fit {point_log} coordinates per side"
            )));
        }
        let mut left = left_point.to_vec();
        let mut right = right_point.to_vec();
        left.resize(point_log, F::zero());
        right.resize(point_log, F::zero());
        Ok(Self {
            left_point: left,
            right_point: right,
            value,
        })
    }

    pub fn validate(&self, point_log: usize) -> Result<(), Section6DataError> {
        if self.left_point.len() != point_log || self.right_point.len() != point_log {
            return Err(Section6DataError::InvalidClaim(format!(
                "claim points must have exactly {point_log} coordinates per side"
            )));
        }
        Ok(())
    }
}

/// The public layout `L = (K, N, T, {k_j})` shared with the verifier.
///
/// The bucket sizes (and hence their global offsets) are public; which
/// source matrix each bucket reads, and at which in-source offset, stays
/// private inside [`PrivateShapeMetadata`].
///
/// The bucket-size list serializes as run-length pairs `(size, count)`, so
/// the encoded statement stays `O(log K)` even for millions of buckets.
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
    #[serde(with = "bucket_sizes_rle")]
    pub bucket_sizes: Vec<usize>,
    /// `ell = log_2(K)`, the Packed EO point length and `D_eta'` width.
    pub global_log: usize,
    /// `m`, the public per-side point width: every matrix fits `2^m x 2^m`.
    pub point_width: usize,
    /// `d`, the public source-index width: at most `2^d` matrices.
    pub tau_width: usize,
    /// Low-bit witness width used by Packed EO-LiteBullet.
    pub bit_width: usize,
}

/// Run-length serialization of the ordered bucket sizes.
mod bucket_sizes_rle {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    /// Hard cap against decompression bombs; real layouts stay far below.
    const MAX_TOTAL_BUCKETS: u64 = 1 << 40;

    pub fn serialize<S: Serializer>(values: &[usize], serializer: S) -> Result<S::Ok, S::Error> {
        let mut runs: Vec<(u64, u64)> = Vec::new();
        for &value in values {
            match runs.last_mut() {
                Some((size, count)) if *size == value as u64 => *count += 1,
                _ => runs.push((value as u64, 1)),
            }
        }
        runs.serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<usize>, D::Error> {
        let runs = Vec::<(u64, u64)>::deserialize(deserializer)?;
        let mut total = 0_u64;
        for &(_, count) in &runs {
            total = total
                .checked_add(count)
                .filter(|&sum| sum <= MAX_TOTAL_BUCKETS)
                .ok_or_else(|| serde::de::Error::custom("bucket run-length overflow"))?;
        }
        let mut values = Vec::with_capacity(total as usize);
        for (size, count) in runs {
            values.extend(std::iter::repeat_n(size as usize, count as usize));
        }
        Ok(values)
    }
}

/// The verifier's view of the layout. Since the bucket-size list is public
/// in the registered-shape variant, this is the full layout.
pub type Section6PublicLayout = Section6Layout;

impl Section6Layout {
    pub fn new(
        matrix_count: usize,
        logical_len: usize,
        active_len: usize,
        bucket_sizes: Vec<usize>,
        point_width: usize,
        tau_width: usize,
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
        if bit_width == 0 || bit_width > 16 {
            return Err(Section6DataError::InvalidLayout(format!(
                "bit_width must be in 1..=16, got {bit_width}"
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
        if logical_len < 2 {
            return Err(Section6DataError::InvalidLayout(
                "the compiled protocol requires at least two BVA entries".to_owned(),
            ));
        }
        if bucket_sizes.len() < matrix_count {
            return Err(Section6DataError::InvalidLayout(
                "every matrix must contribute at least one bucket".to_owned(),
            ));
        }
        if point_width == 0 || point_width >= usize::BITS as usize / 2 {
            return Err(Section6DataError::InvalidLayout(format!(
                "point width m={point_width} is out of range"
            )));
        }
        if tau_width == 0 || tau_width >= usize::BITS as usize {
            return Err(Section6DataError::InvalidLayout(format!(
                "source-index width d={tau_width} is out of range"
            )));
        }
        if matrix_count > 1_usize << tau_width {
            return Err(Section6DataError::InvalidLayout(format!(
                "{matrix_count} matrices exceed the 2^{tau_width} source bound"
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
            if size.ilog2() as usize > 2 * point_width {
                return Err(Section6DataError::InvalidLayout(format!(
                    "bucket {bucket} size {size} exceeds the 2^{{2m}} bound"
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
            point_width,
            tau_width,
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
            self.point_width,
            self.tau_width,
            self.bit_width,
        )?;
        if &canonical != self {
            return Err(Section6DataError::InvalidLayout(
                "derived layout fields are inconsistent with the bucket list".to_owned(),
            ));
        }
        Ok(())
    }

    /// The verifier's view. The registered-shape layout is entirely public.
    pub fn public(&self) -> Section6PublicLayout {
        self.clone()
    }

    pub fn from_descriptor(
        descriptor: &DenseBvaDescriptor,
        bit_width: usize,
    ) -> Result<Self, Section6DataError> {
        let point_width = descriptor
            .matrix_shapes()
            .iter()
            .map(|&(rows, cols)| {
                let side = rows.max(cols).max(2);
                side.next_power_of_two().ilog2() as usize
            })
            .max()
            .unwrap_or(1);
        let tau_width = descriptor.matrix_count().max(2).next_power_of_two().ilog2() as usize;
        Self::from_descriptor_with_widths(descriptor, point_width, tau_width, bit_width)
    }

    /// Build a layout with explicit public width bounds `m` and `d`.
    pub fn from_descriptor_with_widths(
        descriptor: &DenseBvaDescriptor,
        point_width: usize,
        tau_width: usize,
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
            descriptor.layout().active_len(),
            bucket_sizes,
            point_width,
            tau_width,
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

    /// Public local point lengths `gamma'_j = log_2 k_j`.
    pub fn gamma_prime(&self) -> Vec<usize> {
        self.bucket_sizes
            .iter()
            .map(|size| size.ilog2() as usize)
            .collect()
    }

    /// Power-of-two bucket-row cover shared by every auxiliary matrix.
    pub fn row_cover(&self) -> Result<usize, Section6DataError> {
        next_power_of_two(self.bucket_count)
    }

    /// Per-side point width `m`, the public matrix-dimension log bound.
    pub fn point_log(&self) -> usize {
        self.point_width
    }

    /// Source-index width `d`: at most `2^d` matrices.
    pub fn tau_log(&self) -> usize {
        self.tau_width
    }

    /// Width `2m` shared by `D_Z`, `X`, `X'`, `Z`, and `D_Z'`.
    pub fn pair_width(&self) -> usize {
        2 * self.point_width
    }

    /// Width `2m + d` of `D_eta` and the eta grand-product chain.
    pub fn eta_width(&self) -> usize {
        2 * self.point_width + self.tau_width
    }

    /// Width of `Z` and the local restriction: every genuine local point has
    /// `log_2 k_j = gamma_l + gamma_r <= 2m` coordinates.
    pub fn local_width(&self) -> usize {
        2 * self.point_width
    }

    /// Width `ell = log_2 K` of `D_eta'` and the Packed EO point `y`.
    pub fn offset_width(&self) -> usize {
        self.global_log
    }

    pub fn validate_descriptor(
        &self,
        descriptor: &DenseBvaDescriptor,
    ) -> Result<(), Section6DataError> {
        if descriptor.matrix_count() != self.matrix_count
            || descriptor.logical_len() != self.logical_len
            || descriptor.layout().active_len() != self.active_len
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
            let logical_row_end = bucket
                .row_offset()
                .checked_add(bucket.row_len())
                .ok_or(Section6DataError::LengthOverflow)?;
            let logical_col_end = bucket
                .col_offset()
                .checked_add(bucket.col_len())
                .ok_or(Section6DataError::LengthOverflow)?;
            let physical_row_end = bucket
                .row_offset()
                .checked_add(bucket.physical_row_len())
                .ok_or(Section6DataError::LengthOverflow)?;
            let physical_col_end = bucket
                .col_offset()
                .checked_add(bucket.physical_col_len())
                .ok_or(Section6DataError::LengthOverflow)?;
            if bucket.row_offset() >= source_shape.0
                || bucket.col_offset() >= source_shape.1
                || physical_row_end != logical_row_end.min(source_shape.0)
                || physical_col_end != logical_col_end.min(source_shape.1)
            {
                return Err(Section6DataError::InvalidMetadata(format!(
                    "bucket {bucket_index} has an invalid physical intersection with source matrix {}",
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

    /// Validate one bucket row against the dynamic width `m = log_2 K`.
    fn validate_bucket(
        &self,
        layout: &Section6Layout,
        bucket: usize,
    ) -> Result<(), Section6DataError> {
        let m = layout.point_log();
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
                "bucket {bucket} lies outside the 2^{m} x 2^{m} point cover"
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

    pub(crate) fn validate_dynamic(
        &self,
        layout: &Section6Layout,
    ) -> Result<(), Section6DataError> {
        layout.validate()?;
        self.validate_column_lengths()?;
        if self.tau.len() != layout.bucket_count {
            return Err(Section6DataError::InvalidMetadata(format!(
                "metadata row count mismatch: expected {}, got {}",
                layout.bucket_count,
                self.tau.len()
            )));
        }
        for bucket in 0..layout.bucket_count {
            self.validate_bucket(layout, bucket)?;
        }
        Ok(())
    }

    /// Offset bit of `D_eta` at `(bucket, coordinate)`: the bits of `alpha_r`
    /// occupy positions `[0, m)` and the bits of `alpha_l` positions
    /// `[m, 2m)`, matching the column split `X = (X_r | X_l)`.
    pub(crate) fn offset_bit(&self, bucket: usize, coordinate: usize, point_log: usize) -> bool {
        if coordinate < point_log {
            (self.alpha_r[bucket] >> coordinate) & 1 == 1
        } else {
            (self.alpha_l[bucket] >> (coordinate - point_log)) & 1 == 1
        }
    }

    /// Bit of `D_eta` at `(bucket, coordinate)` over the full `2m + d` row:
    /// offset bits first, then the bits of the zero-based source index.
    pub(crate) fn eta_bit(&self, bucket: usize, coordinate: usize, point_log: usize) -> bool {
        if coordinate < 2 * point_log {
            self.offset_bit(bucket, coordinate, point_log)
        } else {
            let bit = coordinate - 2 * point_log;
            ((self.tau[bucket] - 1) >> bit) & 1 == 1
        }
    }

    /// Bit of `D_Z` at `(bucket, coordinate)`: the prefix masks
    /// `1^{gamma_r} 0..` then `1^{gamma_l} 0..`.
    pub(crate) fn z_bit(&self, bucket: usize, coordinate: usize, point_log: usize) -> bool {
        if coordinate < point_log {
            coordinate < self.gamma_r[bucket]
        } else {
            coordinate - point_log < self.gamma_l[bucket]
        }
    }
}

/// A compact column-major Boolean matrix stored one byte per entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BooleanMask {
    pub rows: usize,
    pub cols: usize,
    bits: Vec<u8>,
}

impl BooleanMask {
    pub fn from_fn(
        rows: usize,
        cols: usize,
        mut bit: impl FnMut(usize, usize) -> bool,
    ) -> Result<Self, Section6DataError> {
        let len = rows
            .checked_mul(cols)
            .ok_or(Section6DataError::LengthOverflow)?;
        let mut bits = vec![0_u8; len];
        for col in 0..cols {
            let start = col * rows;
            for row in 0..rows {
                bits[start + row] = u8::from(bit(row, col));
            }
        }
        Ok(Self { rows, cols, bits })
    }

    pub fn bit(&self, row: usize, col: usize) -> bool {
        self.bits[col * self.rows + row] != 0
    }

    /// One column materialized over the shared power-of-two row cover.
    pub fn column_field_values<F: PrimeField>(
        &self,
        col: usize,
        cover: usize,
    ) -> Result<Vec<F>, Section6DataError> {
        if col >= self.cols || cover < self.rows || !cover.is_power_of_two() {
            return Err(Section6DataError::InvalidLayout(format!(
                "mask column {col} does not fit cover {cover}"
            )));
        }
        let start = col * self.rows;
        let mut values = vec![F::zero(); cover];
        for (row, value) in values[..self.rows].iter_mut().enumerate() {
            if self.bits[start + row] != 0 {
                *value = F::one();
            }
        }
        Ok(values)
    }
}

/// The two private registered shape masks `D_eta` and `D_Z`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Section6ShapeMasks {
    /// `T x (2m + d)`: offset bits of `alpha_r`, `alpha_l`, then source bits.
    pub d_eta: BooleanMask,
    /// `T x 2m`: prefix masks `1^{gamma_r} 0..` then `1^{gamma_l} 0..`.
    pub d_z: BooleanMask,
}

impl Section6ShapeMasks {
    pub fn from_metadata(
        layout: &Section6Layout,
        metadata: &PrivateShapeMetadata,
    ) -> Result<Self, Section6DataError> {
        metadata.validate_dynamic(layout)?;
        let rows = layout.bucket_count;
        let m = layout.point_log();
        let d_eta = BooleanMask::from_fn(rows, layout.eta_width(), |row, col| {
            metadata.eta_bit(row, col, m)
        })?;
        let d_z = BooleanMask::from_fn(rows, layout.pair_width(), |row, col| {
            metadata.z_bit(row, col, m)
        })?;
        Ok(Self { d_eta, d_z })
    }
}

/// The two public derived masks `D_eta'` and `D_Z'`, determined by the
/// layout alone: global bucket-offset bits and bucket-size prefix masks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Section6DerivedMasks {
    /// `T x ell`: little-endian bits of the global bucket offsets.
    pub d_eta_prime: BooleanMask,
    /// `T x 2m`: prefix masks `1^{log_2 k_j} 0..`.
    pub d_z_prime: BooleanMask,
}

impl Section6DerivedMasks {
    pub fn from_layout(layout: &Section6Layout) -> Result<Self, Section6DataError> {
        let rows = layout.bucket_count;
        let offsets = layout.bucket_offsets()?;
        let gamma_prime = layout.gamma_prime();
        let d_eta_prime = BooleanMask::from_fn(rows, layout.offset_width(), |row, bit| {
            (offsets[row] >> bit) & 1 == 1
        })?;
        let d_z_prime =
            BooleanMask::from_fn(rows, layout.pair_width(), |row, bit| bit < gamma_prime[row])?;
        Ok(Self {
            d_eta_prime,
            d_z_prime,
        })
    }
}

/// Per-opening point-compilation witness over the shared bucket-row cover.
///
/// Padding rows carry zero on `X'`, `Z`, and `gamma_r`, while the eta chain
/// pads with ones (every masked factor degenerates to `1`); consumers mask
/// padded rows with the public bucket indicator where it matters.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Section6PointWitness<F: PrimeField> {
    /// Gathered source points `X' = X_{tau_j}`, logical `T x 2m`.
    pub x_selected: PaddedFieldMatrix<F>,
    /// Compacted local points `Z`, logical `T x 2m`.
    pub z: PaddedFieldMatrix<F>,
    /// Grand-product prefixes of the eta chain, logical `T x (2m + d - 1)`.
    pub eta_prefix: PaddedFieldMatrix<F>,
    /// Compiled TBV scales `eta` (the closed chain, padding rows one).
    pub eta: PaddedFieldVector<F>,
    /// Column prefix lengths `gamma_r = D_{Z,r} 1`, logical length `T`.
    pub gamma_r: PaddedFieldVector<F>,
    /// `c = sum_tau xi_tau u_tau` for the monomial batching weights.
    pub aggregated_claim: F,
}

impl<F: PrimeField> Section6PointWitness<F> {
    pub fn compile(
        layout: &Section6Layout,
        descriptor: &DenseBvaDescriptor,
        metadata: &PrivateShapeMetadata,
        claims: &[Section6MatrixClaim<F>],
        batch_point: &[F],
    ) -> Result<Self, Section6DataError> {
        layout.validate_descriptor(descriptor)?;
        metadata.validate_against_descriptor(layout, descriptor)?;
        metadata.validate_dynamic(layout)?;
        validate_claims(layout, claims)?;
        if batch_point.len() != layout.tau_log() {
            return Err(Section6DataError::InvalidPoint(format!(
                "batching point must have d = {} coordinates",
                layout.tau_log()
            )));
        }

        let m = layout.point_log();
        let pair = layout.pair_width();
        let eta_width = layout.eta_width();
        let rows = layout.bucket_count;
        let cover = layout.row_cover()?;

        let x_selected = PaddedFieldMatrix::from_fn_with_cover(rows, cover, pair, |row, col| {
            let source = metadata.tau[row] - 1;
            if col < m {
                claims[source].right_point[col]
            } else {
                claims[source].left_point[col - m]
            }
        })?;
        let z = PaddedFieldMatrix::from_fn_with_cover(rows, cover, pair, |row, col| {
            let source = metadata.tau[row] - 1;
            if col < metadata.gamma_r[row] {
                claims[source].right_point[col]
            } else if col < metadata.gamma_r[row] + metadata.gamma_l[row] {
                claims[source].left_point[col - metadata.gamma_r[row]]
            } else {
                F::zero()
            }
        })?;

        // The eta chain multiplies the masked offset coordinates and the
        // masked batching coordinates: factor_h = 1 - D_eta_h + D_eta_h * v_h
        // with v_h = X'_h for h < 2m and v_h = x_{h-2m} afterwards. The chain
        // is built column by column so no per-row buffers are materialized;
        // padding rows stay one across the whole chain.
        let prefix_cols = eta_width - 1;
        let prefix_padded_cols = padded_dimension(prefix_cols)?;
        let prefix_len = cover
            .checked_mul(prefix_padded_cols)
            .ok_or(Section6DataError::LengthOverflow)?;
        let mut prefix_values = vec![F::zero(); prefix_len];
        let chain_factor = |row: usize, col: usize| -> F {
            if row < rows && metadata.eta_bit(row, col, m) {
                if col < pair {
                    x_selected
                        .logical_value(row, col)
                        .expect("X' covers the pair width")
                } else {
                    batch_point[col - pair]
                }
            } else {
                F::one()
            }
        };
        for col in 0..prefix_cols {
            let column_start = col * cover;
            let (before, tail) = prefix_values.split_at_mut(column_start);
            let current = &mut tail[..cover];
            if col == 0 {
                for (row, value) in current.iter_mut().enumerate() {
                    *value = chain_factor(row, 0);
                }
            } else {
                let previous = &before[(col - 1) * cover..col * cover];
                for (row, (value, prev)) in current.iter_mut().zip(previous).enumerate() {
                    *value = *prev * chain_factor(row, col);
                }
            }
        }
        let eta_values = {
            let last_prefix = &prefix_values[(prefix_cols - 1) * cover..prefix_cols * cover];
            last_prefix
                .iter()
                .enumerate()
                .map(|(row, prev)| *prev * chain_factor(row, eta_width - 1))
                .collect::<Vec<_>>()
        };
        let eta_prefix = PaddedFieldMatrix {
            logical_rows: rows,
            logical_cols: prefix_cols,
            padded_rows: cover,
            padded_cols: prefix_padded_cols,
            values: prefix_values,
        };
        let eta = PaddedFieldVector {
            logical_len: rows,
            padded_len: cover,
            values: eta_values,
        };
        let gamma_r = PaddedFieldVector::from_logical_with_cover(
            metadata
                .gamma_r
                .iter()
                .map(|&gamma| field_from_usize(gamma))
                .collect(),
            cover,
        )?;

        let mut aggregated_claim = F::zero();
        for (index, claim) in claims.iter().enumerate() {
            aggregated_claim += claim.value * monomial_weight(batch_point, index);
        }

        Ok(Self {
            x_selected,
            z,
            eta_prefix,
            eta,
            gamma_r,
            aggregated_claim,
        })
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
        if self.eta.logical_len != layout.bucket_count || self.z.logical_rows != layout.bucket_count
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
///
/// The restriction scales `eta'` and the restriction points
/// `Z'_c = y_c [c < gamma'_j]` are committed inside `C_zeta'` and bound to
/// their public piecewise-regular descriptions by plain identities, so the
/// evaluation argument needs no grand-product chain of its own.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Section6FinalWitness<F: PrimeField> {
    /// Restriction scales `eta'_j = prod_{bit of offset_j} y_bit`; padding
    /// rows carry zero.
    pub eta_prime: PaddedFieldVector<F>,
    /// Restriction points `Z'`, logical `T x 2m`; padding rows carry zero.
    pub z_prime: PaddedFieldMatrix<F>,
    /// Product prefixes, logical `T x (2m + 1)`: `P_0 = eta o eta'`, then
    /// `P_{c+1} = P_c o (1 + Z_c o Z'_c)`; the last column is `v`.
    pub p: PaddedFieldMatrix<F>,
    /// `b = sum_j v_j`, the claimed TBV evaluation.
    pub evaluation: F,
}

impl<F: PrimeField> Section6FinalWitness<F> {
    pub fn compile(
        layout: &Section6Layout,
        point: &Section6PointWitness<F>,
        y: &[F],
    ) -> Result<Self, Section6DataError> {
        let rows = layout.bucket_count;
        let cover = layout.row_cover()?;
        let pair = layout.pair_width();
        let ell = layout.offset_width();
        if y.len() != ell {
            return Err(Section6DataError::InvalidPoint(format!(
                "final point length mismatch: expected {ell}, got {}",
                y.len()
            )));
        }
        if point.eta.logical_len != rows
            || point.z.logical_rows != rows
            || point.z.logical_cols != pair
        {
            return Err(Section6DataError::InvalidPoint(
                "final witness inputs do not match the layout".to_owned(),
            ));
        }
        let gamma_prime = layout.gamma_prime();
        if gamma_prime.iter().any(|&log| log > pair) {
            return Err(Section6DataError::InvalidPoint(
                "a bucket exceeds the compiled pair width".to_owned(),
            ));
        }
        let offsets = layout.bucket_offsets()?;

        // eta'_j is the monomial weight of the bucket offset at y.
        let eta_prime = PaddedFieldVector::from_logical_with_cover(
            (0..rows)
                .map(|row| {
                    let mut product = F::one();
                    for (bit, &coordinate) in y.iter().enumerate() {
                        if (offsets[row] >> bit) & 1 == 1 {
                            product *= coordinate;
                        }
                    }
                    product
                })
                .collect(),
            cover,
        )?;
        let z_prime = PaddedFieldMatrix::from_fn_with_cover(rows, cover, pair, |row, col| {
            if col < gamma_prime[row] {
                y[col]
            } else {
                F::zero()
            }
        })?;

        // P chain, built column by column: P_0 = eta o eta' and
        // P_{c+1} = P_c o (1 + Z_c o Z'_c); padding rows stay zero because
        // eta' pads with zero.
        let p_cols = pair + 1;
        let p_padded_cols = padded_dimension(p_cols)?;
        let p_len = cover
            .checked_mul(p_padded_cols)
            .ok_or(Section6DataError::LengthOverflow)?;
        let mut p_values = vec![F::zero(); p_len];
        for col in 0..p_cols {
            let column_start = col * cover;
            let (before, tail) = p_values.split_at_mut(column_start);
            let current = &mut tail[..cover];
            if col == 0 {
                for (row, value) in current.iter_mut().take(rows).enumerate() {
                    *value = point.eta.values[row] * eta_prime.values[row];
                }
            } else {
                let previous = &before[(col - 1) * cover..col * cover];
                for (row, (value, prev)) in current.iter_mut().zip(previous).take(rows).enumerate()
                {
                    let z = point
                        .z
                        .logical_value(row, col - 1)
                        .expect("validated Z bounds");
                    let z_prime_value = if col - 1 < gamma_prime[row] {
                        y[col - 1]
                    } else {
                        F::zero()
                    };
                    *value = *prev * (F::one() + z * z_prime_value);
                }
            }
        }
        let evaluation = p_values[pair * cover..pair * cover + rows]
            .iter()
            .copied()
            .sum();
        let p = PaddedFieldMatrix {
            logical_rows: rows,
            logical_cols: p_cols,
            padded_rows: cover,
            padded_cols: p_padded_cols,
            values: p_values,
        };
        Ok(Self {
            eta_prime,
            z_prime,
            p,
            evaluation,
        })
    }
}

/// Monomial tensor weight of `index` at `point`: the product of the
/// coordinates whose bit is set in `index`.
pub fn monomial_weight<F: PrimeField>(point: &[F], index: usize) -> F {
    let mut weight = F::one();
    for (bit, coordinate) in point.iter().enumerate() {
        if (index >> bit) & 1 == 1 {
            weight *= *coordinate;
        }
    }
    weight
}

/// Validate claims against the dynamic point width of the layout.
pub fn validate_claims<F: PrimeField>(
    layout: &Section6Layout,
    claims: &[Section6MatrixClaim<F>],
) -> Result<(), Section6DataError> {
    if claims.len() != layout.matrix_count {
        return Err(Section6DataError::InvalidClaim(format!(
            "claim count mismatch: expected {}, got {}",
            layout.matrix_count,
            claims.len()
        )));
    }
    for claim in claims {
        claim.validate(layout.point_log())?;
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

pub(crate) fn bool_to_field<F: PrimeField>(bit: bool) -> F {
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
    use ark_ff::{Field, One};

    use super::*;

    fn field_values(start: u64, len: usize) -> Vec<Fr> {
        (0..len)
            .map(|index| Fr::from(start + index as u64))
            .collect()
    }

    fn asymmetric_fixture() -> (
        DenseBvaDescriptor,
        Section6Layout,
        PrivateShapeMetadata,
        Vec<Section6MatrixClaim<Fr>>,
    ) {
        let shapes = [(3, 5), (5, 2), (2, 7)];
        let descriptor = DenseBvaDescriptor::from_shapes(&shapes).unwrap();
        let layout = Section6Layout::from_descriptor(&descriptor, 8).unwrap();
        let metadata = PrivateShapeMetadata::from_descriptor(&layout, &descriptor).unwrap();
        let m = layout.point_log();
        let claims = vec![
            Section6MatrixClaim::from_unpadded(
                m,
                &field_values(11, 2),
                &field_values(21, 3),
                Fr::from(31_u64),
            )
            .unwrap(),
            Section6MatrixClaim::from_unpadded(
                m,
                &field_values(41, 3),
                &field_values(51, 1),
                Fr::from(61_u64),
            )
            .unwrap(),
            Section6MatrixClaim::from_unpadded(
                m,
                &field_values(71, 1),
                &field_values(81, 3),
                Fr::from(91_u64),
            )
            .unwrap(),
        ];
        (descriptor, layout, metadata, claims)
    }

    fn batch_point(layout: &Section6Layout) -> Vec<Fr> {
        (0..layout.tau_log())
            .map(|bit| Fr::from((101 + bit * 3) as u64))
            .collect()
    }

    #[test]
    fn shape_masks_encode_the_metadata_bits() {
        let (_descriptor, layout, metadata, _claims) = asymmetric_fixture();
        let masks = Section6ShapeMasks::from_metadata(&layout, &metadata).unwrap();
        let m = layout.point_log();
        for j in 0..layout.bucket_count {
            let mut alpha_r = 0_usize;
            let mut alpha_l = 0_usize;
            let mut tau = 0_usize;
            for h in 0..m {
                if masks.d_eta.bit(j, h) {
                    alpha_r += 1 << h;
                }
                if masks.d_eta.bit(j, m + h) {
                    alpha_l += 1 << h;
                }
            }
            for h in 0..layout.tau_log() {
                if masks.d_eta.bit(j, 2 * m + h) {
                    tau += 1 << h;
                }
            }
            assert_eq!(alpha_r, metadata.alpha_r[j]);
            assert_eq!(alpha_l, metadata.alpha_l[j]);
            assert_eq!(tau + 1, metadata.tau[j]);
            for h in 0..m {
                assert_eq!(masks.d_z.bit(j, h), h < metadata.gamma_r[j]);
                assert_eq!(masks.d_z.bit(j, m + h), h < metadata.gamma_l[j]);
                // Alignment: offset bits vanish on the varying coordinates.
                if masks.d_z.bit(j, h) {
                    assert!(!masks.d_eta.bit(j, h));
                }
                if masks.d_z.bit(j, m + h) {
                    assert!(!masks.d_eta.bit(j, m + h));
                }
            }
        }
    }

    #[test]
    fn derived_masks_encode_the_public_layout() {
        let (_descriptor, layout, _metadata, _claims) = asymmetric_fixture();
        let derived = Section6DerivedMasks::from_layout(&layout).unwrap();
        let offsets = layout.bucket_offsets().unwrap();
        for j in 0..layout.bucket_count {
            let mut offset = 0_usize;
            for bit in 0..layout.offset_width() {
                if derived.d_eta_prime.bit(j, bit) {
                    offset += 1 << bit;
                }
            }
            assert_eq!(offset, offsets[j]);
            let mut size = 1_usize;
            for bit in 0..layout.pair_width() {
                if derived.d_z_prime.bit(j, bit) {
                    assert!(bit == 0 || derived.d_z_prime.bit(j, bit - 1));
                    size += 1 << bit;
                }
            }
            assert_eq!(size, layout.bucket_sizes[j]);
        }
    }

    #[test]
    fn point_witness_compiles_the_tbv_scales() {
        let (descriptor, layout, metadata, claims) = asymmetric_fixture();
        let point = batch_point(&layout);
        let witness =
            Section6PointWitness::compile(&layout, &descriptor, &metadata, &claims, &point)
                .unwrap();
        let m = layout.point_log();
        for j in 0..layout.bucket_count {
            let source = metadata.tau[j] - 1;
            let mut expected = monomial_weight(&point, source);
            for h in 0..layout.pair_width() {
                if metadata.offset_bit(j, h, m) {
                    let coordinate = if h < m {
                        claims[source].right_point[h]
                    } else {
                        claims[source].left_point[h - m]
                    };
                    expected *= coordinate;
                }
            }
            assert_eq!(witness.eta.logical_values()[j], expected);
            // Z holds the varying coordinates, column side first.
            for c in 0..metadata.gamma_r[j] {
                assert_eq!(
                    witness.z.logical_value(j, c).unwrap(),
                    claims[source].right_point[c]
                );
            }
            for c in 0..metadata.gamma_l[j] {
                assert_eq!(
                    witness.z.logical_value(j, metadata.gamma_r[j] + c).unwrap(),
                    claims[source].left_point[c]
                );
            }
        }
        let expected_claim = claims
            .iter()
            .enumerate()
            .map(|(index, claim)| claim.value * monomial_weight(&point, index))
            .sum::<Fr>();
        assert_eq!(witness.aggregated_claim, expected_claim);
    }

    #[test]
    fn final_witness_matches_the_dense_tensor_oracle() {
        let (descriptor, layout, metadata, claims) = asymmetric_fixture();
        let point = batch_point(&layout);
        let witness =
            Section6PointWitness::compile(&layout, &descriptor, &metadata, &claims, &point)
                .unwrap();
        let y = field_values(101, layout.offset_width());
        let final_witness = Section6FinalWitness::compile(&layout, &witness, &y).unwrap();

        // Oracle: expand the compiled TBV into its dense monomial-basis
        // vector and evaluate sum_i tbv_i prod_{bit set in i} y_bit.
        let offsets = layout.bucket_offsets().unwrap();
        let mut expected = Fr::zero();
        for j in 0..layout.bucket_count {
            let local_log = layout.bucket_sizes[j].ilog2() as usize;
            let local = witness.local_point(j, local_log).unwrap();
            let eta = witness.eta.logical_values()[j];
            for index in 0..layout.bucket_sizes[j] {
                let mut weight = eta;
                for (bit, coordinate) in local.iter().enumerate() {
                    if (index >> bit) & 1 == 1 {
                        weight *= *coordinate;
                    }
                }
                let global = offsets[j] + index;
                for (bit, coordinate) in y.iter().enumerate() {
                    if (global >> bit) & 1 == 1 {
                        weight *= *coordinate;
                    }
                }
                expected += weight;
            }
        }
        assert_eq!(final_witness.evaluation, expected);

        // The committed prefix matrix ends in v and pads with zeros.
        assert_eq!(final_witness.p.logical_cols, layout.local_width() + 1);
        assert!(final_witness.p.padding_is_zero());

        // eta' matches the offset-bit product of the Packed EO point.
        for j in 0..layout.bucket_count {
            assert_eq!(
                final_witness.eta_prime.logical_values()[j],
                monomial_weight(&y, offsets[j])
            );
        }
    }

    #[test]
    fn claims_must_match_the_dynamic_point_width() {
        let (_descriptor, layout, _metadata, mut claims) = asymmetric_fixture();
        claims[0].left_point.push(Fr::one());
        assert!(validate_claims(&layout, &claims).is_err());
    }
}
