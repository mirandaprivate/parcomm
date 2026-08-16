use ark_ff::PrimeField;
use rayon::prelude::*;
use std::fmt::{Display, Formatter};

use super::sparse::{ProjectionMatrix, ShortInt};
use super::xi;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct MatrixDims {
    pub rows: usize,
    pub cols: usize,
}

impl MatrixDims {
    pub fn new(rows: usize, cols: usize) -> Result<Self, HeteroMatrixError> {
        if rows == 0 || cols == 0 {
            return Err(HeteroMatrixError::ZeroDimension);
        }
        Ok(Self { rows, cols })
    }

    pub fn entries(self) -> usize {
        self.rows * self.cols
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PaddedDomain {
    pub slots: usize,
    pub row_cap: usize,
    pub col_cap: usize,
}

impl PaddedDomain {
    pub fn new(slots: usize, row_cap: usize, col_cap: usize) -> Result<Self, HeteroMatrixError> {
        for (axis, size) in [("slots", slots), ("rows", row_cap), ("cols", col_cap)] {
            if size == 0 {
                return Err(HeteroMatrixError::ZeroDimension);
            }
            if !size.is_power_of_two() {
                return Err(HeteroMatrixError::DomainNotPowerOfTwo { axis, size });
            }
        }

        slots
            .checked_mul(row_cap)
            .and_then(|rows| rows.checked_mul(col_cap))
            .ok_or(HeteroMatrixError::DimensionOverflow)?;

        Ok(Self {
            slots,
            row_cap,
            col_cap,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatrixBlock<I> {
    pub dims: MatrixDims,
    pub columns: Vec<Vec<I>>,
}

impl<I> MatrixBlock<I> {
    pub fn from_columns(dims: MatrixDims, columns: Vec<Vec<I>>) -> Result<Self, HeteroMatrixError> {
        let block = Self { dims, columns };
        block.validate_layout()?;
        Ok(block)
    }

    pub fn validate_layout(&self) -> Result<(), HeteroMatrixError> {
        if self.columns.len() != self.dims.cols {
            return Err(HeteroMatrixError::ColumnCount {
                expected: self.dims.cols,
                actual: self.columns.len(),
            });
        }
        for (column, values) in self.columns.iter().enumerate() {
            if values.len() != self.dims.rows {
                return Err(HeteroMatrixError::ColumnHeight {
                    column,
                    expected: self.dims.rows,
                    actual: values.len(),
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HeteroBlockMat<I>
where
    I: ShortInt,
{
    pub domain: PaddedDomain,
    pub blocks: Vec<Option<MatrixBlock<I>>>,
}

impl<I> HeteroBlockMat<I>
where
    I: ShortInt,
{
    pub fn new(
        domain: PaddedDomain,
        mut blocks: Vec<Option<MatrixBlock<I>>>,
    ) -> Result<Self, HeteroMatrixError> {
        if blocks.len() > domain.slots {
            return Err(HeteroMatrixError::TooManyBlocks {
                slots: domain.slots,
                blocks: blocks.len(),
            });
        }
        for (slot, block) in blocks.iter().enumerate() {
            if let Some(block) = block {
                block.validate_layout()?;
                if block.dims.rows > domain.row_cap || block.dims.cols > domain.col_cap {
                    return Err(HeteroMatrixError::BlockExceedsDomain {
                        slot,
                        dims: block.dims,
                        row_cap: domain.row_cap,
                        col_cap: domain.col_cap,
                    });
                }
            }
        }
        blocks.resize_with(domain.slots, || None);
        Ok(Self { domain, blocks })
    }

    pub fn stored_entries(&self) -> usize {
        self.blocks
            .iter()
            .flatten()
            .map(|block| block.dims.entries())
            .sum()
    }

    pub fn stacked_shape(&self) -> (usize, usize) {
        (self.domain.slots * self.domain.row_cap, self.domain.col_cap)
    }

    pub fn diagonal_shape(&self) -> (usize, usize) {
        (
            self.domain.slots * self.domain.row_cap,
            self.domain.slots * self.domain.col_cap,
        )
    }

    pub fn packed_stacked_shape(&self) -> (usize, usize) {
        let (rows, cols) = self.stacked_shape();
        let entries = rows * cols;
        let packed_cols = 1 << (entries.ilog2() as usize / 2);
        let packed_rows = entries / packed_cols;
        (packed_rows, packed_cols)
    }

    pub fn for_each_packed_stacked_entry(&self, mut visit: impl FnMut(usize, usize, I)) {
        let (_, packed_cols) = self.packed_stacked_shape();
        let original_cols = self.domain.col_cap;
        for (slot, block) in self.blocks.iter().enumerate() {
            let Some(block) = block else { continue };
            let row_offset = slot * self.domain.row_cap;
            for (col, values) in block.columns.iter().enumerate() {
                for (row, value) in values.iter().enumerate() {
                    if *value == I::from(false) {
                        continue;
                    }
                    let flat_index = (row_offset + row) * original_cols + col;
                    visit(flat_index / packed_cols, flat_index % packed_cols, *value);
                }
            }
        }
    }

    pub fn proj_packed_stacked_left<F>(&self, xl: &[F]) -> Result<Vec<F>, HeteroMatrixError>
    where
        F: PrimeField + From<I>,
    {
        let (rows, cols) = self.packed_stacked_shape();
        check_challenge_len("packed left", rows, xl.len())?;
        let original_cols = self.domain.col_cap;
        let left_weights = xi::xi_from_challenges(&xl.to_vec());
        if cols < original_cols {
            let mut projection = vec![F::zero(); cols];
            self.for_each_packed_stacked_entry(|row, col, entry| {
                projection[col] += F::from(entry) * left_weights[row];
            });
            return Ok(projection);
        }
        let row_group = cols / original_cols;
        Ok((0..cols)
            .into_par_iter()
            .map(|packed_col| {
                let original_col = packed_col % original_cols;
                let row_mod = packed_col / original_cols;
                let mut value = F::zero();
                for (slot, block) in self.blocks.iter().enumerate() {
                    let Some(block) = block else { continue };
                    let Some(column) = block.columns.get(original_col) else {
                        continue;
                    };
                    let row_offset = slot * self.domain.row_cap;
                    let local_row_mod = (row_mod + row_group - row_offset % row_group) % row_group;
                    for local_row in (local_row_mod..column.len()).step_by(row_group) {
                        let entry = column[local_row];
                        if entry == I::from(false) {
                            continue;
                        }
                        let global_row = row_offset + local_row;
                        value += F::from(entry) * left_weights[global_row / row_group];
                    }
                }
                value
            })
            .collect())
    }

    pub fn proj_packed_stacked_right<F>(&self, xr: &[F]) -> Result<Vec<F>, HeteroMatrixError>
    where
        F: PrimeField + From<I>,
    {
        let (rows, cols) = self.packed_stacked_shape();
        check_challenge_len("packed right", cols, xr.len())?;
        let original_cols = self.domain.col_cap;
        let right_weights = xi::xi_from_challenges(&xr.to_vec());
        if cols < original_cols {
            let mut projection = vec![F::zero(); rows];
            self.for_each_packed_stacked_entry(|row, col, entry| {
                projection[row] += F::from(entry) * right_weights[col];
            });
            return Ok(projection);
        }
        let row_group = cols / original_cols;
        Ok((0..rows)
            .into_par_iter()
            .map(|packed_row| {
                let mut value = F::zero();
                let first_global_row = packed_row * row_group;
                for row_mod in 0..row_group {
                    let global_row = first_global_row + row_mod;
                    let slot = global_row / self.domain.row_cap;
                    let local_row = global_row % self.domain.row_cap;
                    let Some(block) = self.blocks.get(slot).and_then(Option::as_ref) else {
                        continue;
                    };
                    if local_row >= block.dims.rows {
                        continue;
                    }
                    for (column, entries) in block.columns.iter().enumerate() {
                        let entry = entries[local_row];
                        if entry == I::from(false) {
                            continue;
                        }
                        value += F::from(entry) * right_weights[row_mod * original_cols + column];
                    }
                }
                value
            })
            .collect())
    }

    pub fn proj_stacked_left<F>(&self, xl: &[F]) -> Result<Vec<F>, HeteroMatrixError>
    where
        F: PrimeField + From<I>,
    {
        let (rows, cols) = self.stacked_shape();
        check_challenge_len("left", rows, xl.len())?;
        let left_weights = xi::xi_from_challenges(&xl.to_vec());

        Ok((0..cols)
            .into_par_iter()
            .map(|col| {
                let mut projection = F::zero();
                for (slot, block) in self.blocks.iter().enumerate() {
                    let Some(block) = block else { continue };
                    let Some(values) = block.columns.get(col) else {
                        continue;
                    };
                    let row_offset = slot * self.domain.row_cap;
                    for (row, value) in values.iter().enumerate() {
                        if *value == I::from(false) {
                            continue;
                        }
                        projection += F::from(*value) * left_weights[row_offset + row];
                    }
                }
                projection
            })
            .collect())
    }

    pub fn proj_stacked_right<F>(&self, xr: &[F]) -> Result<Vec<F>, HeteroMatrixError>
    where
        F: PrimeField + From<I>,
    {
        let (rows, cols) = self.stacked_shape();
        check_challenge_len("right", cols, xr.len())?;
        let right_weights = xi::xi_from_challenges(&xr.to_vec());

        Ok((0..rows)
            .into_par_iter()
            .map(|global_row| {
                let slot = global_row / self.domain.row_cap;
                let local_row = global_row % self.domain.row_cap;
                let Some(block) = self.blocks.get(slot).and_then(Option::as_ref) else {
                    return F::zero();
                };
                if local_row >= block.dims.rows {
                    return F::zero();
                }
                let mut projection = F::zero();
                for (col, values) in block.columns.iter().enumerate() {
                    let value = values[local_row];
                    if value == I::from(false) {
                        continue;
                    }
                    projection += F::from(value) * right_weights[col];
                }
                projection
            })
            .collect())
    }

    pub fn proj_stacked_lr<F>(&self, xl: &[F], xr: &[F]) -> Result<F, HeteroMatrixError>
    where
        F: PrimeField + From<I>,
    {
        let (rows, cols) = self.stacked_shape();
        check_challenge_len("left", rows, xl.len())?;
        check_challenge_len("right", cols, xr.len())?;
        let mut projection = F::zero();
        let left_weights = xi::xi_from_challenges(&xl.to_vec());
        let right_weights = xi::xi_from_challenges(&xr.to_vec());

        for (slot, block) in self.blocks.iter().enumerate() {
            let Some(block) = block else { continue };
            let row_offset = slot * self.domain.row_cap;
            for (col, values) in block.columns.iter().enumerate() {
                let right_weight = right_weights[col];
                for (row, value) in values.iter().enumerate() {
                    if *value == I::from(false) {
                        continue;
                    }
                    projection += F::from(*value) * left_weights[row_offset + row] * right_weight;
                }
            }
        }

        Ok(projection)
    }

    pub fn proj_diagonal_left<F>(&self, xl: &[F]) -> Result<Vec<F>, HeteroMatrixError>
    where
        F: PrimeField + From<I>,
    {
        let (rows, cols) = self.diagonal_shape();
        check_challenge_len("left", rows, xl.len())?;
        let mut projection = vec![F::zero(); cols];
        let left_weights = xi::xi_from_challenges(&xl.to_vec());

        for (slot, block) in self.blocks.iter().enumerate() {
            let Some(block) = block else { continue };
            let row_offset = slot * self.domain.row_cap;
            let col_offset = slot * self.domain.col_cap;
            for (col, values) in block.columns.iter().enumerate() {
                for (row, value) in values.iter().enumerate() {
                    if *value == I::from(false) {
                        continue;
                    }
                    projection[col_offset + col] +=
                        F::from(*value) * left_weights[row_offset + row];
                }
            }
        }

        Ok(projection)
    }

    pub fn proj_diagonal_right<F>(&self, xr: &[F]) -> Result<Vec<F>, HeteroMatrixError>
    where
        F: PrimeField + From<I>,
    {
        let (rows, cols) = self.diagonal_shape();
        check_challenge_len("right", cols, xr.len())?;
        let mut projection = vec![F::zero(); rows];
        let right_weights = xi::xi_from_challenges(&xr.to_vec());

        for (slot, block) in self.blocks.iter().enumerate() {
            let Some(block) = block else { continue };
            let row_offset = slot * self.domain.row_cap;
            let col_offset = slot * self.domain.col_cap;
            for (col, values) in block.columns.iter().enumerate() {
                let weight = right_weights[col_offset + col];
                for (row, value) in values.iter().enumerate() {
                    if *value == I::from(false) {
                        continue;
                    }
                    projection[row_offset + row] += F::from(*value) * weight;
                }
            }
        }

        Ok(projection)
    }

    pub fn proj_diagonal_lr<F>(&self, xl: &[F], xr: &[F]) -> Result<F, HeteroMatrixError>
    where
        F: PrimeField + From<I>,
    {
        let (rows, cols) = self.diagonal_shape();
        check_challenge_len("left", rows, xl.len())?;
        check_challenge_len("right", cols, xr.len())?;
        let mut projection = F::zero();
        let left_weights = xi::xi_from_challenges(&xl.to_vec());
        let right_weights = xi::xi_from_challenges(&xr.to_vec());

        for (slot, block) in self.blocks.iter().enumerate() {
            let Some(block) = block else { continue };
            let row_offset = slot * self.domain.row_cap;
            let col_offset = slot * self.domain.col_cap;
            for (col, values) in block.columns.iter().enumerate() {
                let right_weight = right_weights[col_offset + col];
                for (row, value) in values.iter().enumerate() {
                    if *value == I::from(false) {
                        continue;
                    }
                    projection += F::from(*value) * left_weights[row_offset + row] * right_weight;
                }
            }
        }

        Ok(projection)
    }
}

pub struct StackedHeteroView<'a, I>
where
    I: ShortInt,
{
    matrix: &'a HeteroBlockMat<I>,
}

impl<'a, I> StackedHeteroView<'a, I>
where
    I: ShortInt,
{
    pub fn new(matrix: &'a HeteroBlockMat<I>) -> Self {
        Self { matrix }
    }
}

impl<I, F> ProjectionMatrix<I, F> for StackedHeteroView<'_, I>
where
    I: ShortInt,
    F: PrimeField + From<I>,
{
    fn projection_shape(&self) -> (usize, usize) {
        self.matrix.stacked_shape()
    }

    fn project_left(&self, xl: &Vec<F>) -> Vec<F> {
        self.matrix.proj_stacked_left(xl).unwrap()
    }

    fn project_right(&self, xr: &Vec<F>) -> Vec<F> {
        self.matrix.proj_stacked_right(xr).unwrap()
    }
}

pub struct DiagonalHeteroView<'a, I>
where
    I: ShortInt,
{
    matrix: &'a HeteroBlockMat<I>,
}

/// A projection-only view of column zero of every padded block, stacked as one
/// vector.  This is the physical vector carried by HD-R1CS vector wires; the
/// remaining logical columns stay implicit zeros.
pub struct ColumnZeroHeteroView<'a, I>
where
    I: ShortInt,
{
    matrix: &'a HeteroBlockMat<I>,
    complement: bool,
}

impl<'a, I> ColumnZeroHeteroView<'a, I>
where
    I: ShortInt,
{
    pub fn new(matrix: &'a HeteroBlockMat<I>) -> Self {
        Self {
            matrix,
            complement: false,
        }
    }

    /// View the vector as `1 - value`.  This is used by the streaming boolean
    /// check and does not allocate the padded all-ones vector.
    pub fn one_minus(matrix: &'a HeteroBlockMat<I>) -> Self {
        Self {
            matrix,
            complement: true,
        }
    }
}

impl<I, F> ProjectionMatrix<I, F> for ColumnZeroHeteroView<'_, I>
where
    I: ShortInt,
    F: PrimeField + From<I>,
{
    fn projection_shape(&self) -> (usize, usize) {
        (self.matrix.stacked_shape().0, 1)
    }

    fn project_left(&self, xl: &Vec<F>) -> Vec<F> {
        let rows = self.matrix.stacked_shape().0;
        let weights = xi::xi_from_challenges(xl);
        debug_assert_eq!(weights.len(), rows);
        let mut selected = F::zero();
        for (slot, block) in self.matrix.blocks.iter().enumerate() {
            let Some(column) = block.as_ref().and_then(|block| block.columns.first()) else {
                continue;
            };
            let row_offset = slot * self.matrix.domain.row_cap;
            for (row, value) in column.iter().enumerate() {
                selected += F::from(*value) * weights[row_offset + row];
            }
        }
        if self.complement {
            vec![weights.into_par_iter().sum::<F>() - selected]
        } else {
            vec![selected]
        }
    }

    fn project_right(&self, xr: &Vec<F>) -> Vec<F> {
        assert!(xr.is_empty(), "column-zero vector has one logical column");
        let mut projection =
            vec![if self.complement { F::one() } else { F::zero() }; self.matrix.stacked_shape().0];
        for (slot, block) in self.matrix.blocks.iter().enumerate() {
            let Some(column) = block.as_ref().and_then(|block| block.columns.first()) else {
                continue;
            };
            let row_offset = slot * self.matrix.domain.row_cap;
            for (row, value) in column.iter().enumerate() {
                projection[row_offset + row] = if self.complement {
                    F::one() - F::from(*value)
                } else {
                    F::from(*value)
                };
            }
        }
        projection
    }
}

/// A block-diagonal matrix whose diagonal entries are column zero of the
/// corresponding stacked vector.  It permits element-wise products without
/// materializing a `(slots * n)^2` diagonal matrix.
pub struct ElementwiseDiagonalHeteroView<'a, I>
where
    I: ShortInt,
{
    matrix: &'a HeteroBlockMat<I>,
}

impl<'a, I> ElementwiseDiagonalHeteroView<'a, I>
where
    I: ShortInt,
{
    pub fn new(matrix: &'a HeteroBlockMat<I>) -> Self {
        Self { matrix }
    }
}

impl<I, F> ProjectionMatrix<I, F> for ElementwiseDiagonalHeteroView<'_, I>
where
    I: ShortInt,
    F: PrimeField + From<I>,
{
    fn projection_shape(&self) -> (usize, usize) {
        let rows = self.matrix.stacked_shape().0;
        (rows, rows)
    }

    fn project_left(&self, xl: &Vec<F>) -> Vec<F> {
        let rows = self.matrix.stacked_shape().0;
        let weights = xi::xi_from_challenges(xl);
        debug_assert_eq!(weights.len(), rows);
        let mut projection = vec![F::zero(); rows];
        for (slot, block) in self.matrix.blocks.iter().enumerate() {
            let Some(column) = block.as_ref().and_then(|block| block.columns.first()) else {
                continue;
            };
            let row_offset = slot * self.matrix.domain.row_cap;
            for (row, value) in column.iter().enumerate() {
                projection[row_offset + row] = F::from(*value) * weights[row_offset + row];
            }
        }
        projection
    }

    fn project_right(&self, xr: &Vec<F>) -> Vec<F> {
        let rows = self.matrix.stacked_shape().0;
        let weights = xi::xi_from_challenges(xr);
        debug_assert_eq!(weights.len(), rows);
        let mut projection = vec![F::zero(); rows];
        for (slot, block) in self.matrix.blocks.iter().enumerate() {
            let Some(column) = block.as_ref().and_then(|block| block.columns.first()) else {
                continue;
            };
            let row_offset = slot * self.matrix.domain.row_cap;
            for (row, value) in column.iter().enumerate() {
                projection[row_offset + row] = F::from(*value) * weights[row_offset + row];
            }
        }
        projection
    }
}

impl<'a, I> DiagonalHeteroView<'a, I>
where
    I: ShortInt,
{
    pub fn new(matrix: &'a HeteroBlockMat<I>) -> Self {
        Self { matrix }
    }
}

impl<I, F> ProjectionMatrix<I, F> for DiagonalHeteroView<'_, I>
where
    I: ShortInt,
    F: PrimeField + From<I>,
{
    fn projection_shape(&self) -> (usize, usize) {
        self.matrix.diagonal_shape()
    }

    fn project_left(&self, xl: &Vec<F>) -> Vec<F> {
        self.matrix.proj_diagonal_left(xl).unwrap()
    }

    fn project_right(&self, xr: &Vec<F>) -> Vec<F> {
        self.matrix.proj_diagonal_right(xr).unwrap()
    }
}

fn check_challenge_len(
    axis: &'static str,
    dimension: usize,
    actual: usize,
) -> Result<(), HeteroMatrixError> {
    let expected = dimension.ilog2() as usize;
    if actual != expected {
        return Err(HeteroMatrixError::ChallengeLength {
            axis,
            expected,
            actual,
        });
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HeteroMatrixError {
    ZeroDimension,
    DomainNotPowerOfTwo {
        axis: &'static str,
        size: usize,
    },
    DimensionOverflow,
    ColumnCount {
        expected: usize,
        actual: usize,
    },
    ColumnHeight {
        column: usize,
        expected: usize,
        actual: usize,
    },
    TooManyBlocks {
        slots: usize,
        blocks: usize,
    },
    BlockExceedsDomain {
        slot: usize,
        dims: MatrixDims,
        row_cap: usize,
        col_cap: usize,
    },
    ChallengeLength {
        axis: &'static str,
        expected: usize,
        actual: usize,
    },
}

impl Display for HeteroMatrixError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for HeteroMatrixError {}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;

    fn f(value: u64) -> Fr {
        Fr::from(value)
    }

    #[test]
    fn stacked_virtual_projection_matches_worked_padding_example() {
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

        let xl = vec![f(2), f(3), f(5)];
        let xr = vec![f(7), f(11)];

        assert_eq!(matrix.stored_entries(), 8);
        assert_eq!(matrix.stacked_shape(), (8, 4));
        assert_eq!(
            matrix.proj_stacked_left(&xl).unwrap(),
            vec![f(25), f(39), f(35), f(0)],
        );
        assert_eq!(
            matrix.proj_stacked_right(&xr).unwrap(),
            vec![f(69), f(88), f(0), f(0), f(95), f(0), f(0), f(0)],
        );
        assert_eq!(matrix.proj_stacked_lr(&xl, &xr).unwrap(), f(699));
    }

    #[test]
    fn diagonal_virtual_projection_matches_worked_padding_example() {
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

        let xl = vec![f(2), f(3), f(5)];
        let xr = vec![f(13), f(17), f(19)];

        assert_eq!(matrix.diagonal_shape(), (8, 8));
        assert_eq!(
            matrix.proj_diagonal_left(&xl).unwrap(),
            vec![f(11), f(23), f(35), f(0), f(14), f(16), f(0), f(0)],
        );
        assert_eq!(
            matrix.proj_diagonal_right(&xr).unwrap(),
            vec![f(143), f(180), f(0), f(0), f(2067), f(0), f(0), f(0)],
        );
        assert_eq!(matrix.proj_diagonal_lr(&xl, &xr).unwrap(), f(5177));
    }

    #[test]
    fn heterogeneous_matrix_rejects_a_malformed_public_block_without_copying_it() {
        let malformed = MatrixBlock {
            dims: MatrixDims::new(2, 2).unwrap(),
            columns: vec![vec![1_i32, 2]],
        };
        let result =
            HeteroBlockMat::new(PaddedDomain::new(1, 2, 2).unwrap(), vec![Some(malformed)]);
        assert_eq!(
            result.unwrap_err(),
            HeteroMatrixError::ColumnCount {
                expected: 2,
                actual: 1,
            }
        );
    }
}
