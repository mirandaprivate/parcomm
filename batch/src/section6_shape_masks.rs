//! Registration commitments for the three persistent Section 6 mask matrices.
//!
//! Registration commits the fixed-coordinate selector `D_eta`, the
//! varying-coordinate selector `D_Z`, and the global-offset bits `D_eta'` as
//! three separate hiding Dory commitments and nothing else. No shape-validity
//! relation is proved anywhere; the masks are fixed committed inputs reused
//! by every opening.

use std::fmt::{Display, Formatter};

use ark_bn254::Fr;
use jolt_dory::DoryCommitment;
use serde::{Deserialize, Serialize};

use crate::{
    jolt_dory::{CommittedDoryFieldVector, DoryProverParameters},
    matrix_collection::DenseBvaDescriptor,
    section6_data::{
        PaddedFieldMatrix, PrivateShapeMetadata, Section6DataError, Section6Layout,
        Section6PublicLayout, Section6RegisteredMasks, SECTION6_FIXED_ETA_WIDTH,
        SECTION6_FIXED_LOCAL_WIDTH,
    },
    section6_final_opening::{
        commit_section6_field_matrix_columns_on_grid_zk, Section6DoryGrid,
        Section6FinalOpeningError,
    },
};

/// Verifier-visible description of the three registered mask commitments.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Section6ShapeMaskStatement {
    /// Number of real bucket rows before zero padding.
    pub bucket_rows: usize,
    /// Shared power-of-two row cover of all three mask matrices.
    pub padded_rows: usize,
    /// Logical width `2m + d` of `D_eta`.
    pub eta_mask_width: usize,
    /// Logical width `2m` of `D_Z`.
    pub local_mask_width: usize,
    /// Logical width `l` of `D_eta'`.
    pub offset_mask_width: usize,
    pub d_eta_commitment: DoryCommitment,
    pub d_z_commitment: DoryCommitment,
    pub d_eta_prime_commitment: DoryCommitment,
}

impl Section6ShapeMaskStatement {
    pub fn physical_commitment_count(&self) -> usize {
        3
    }

    /// Shared power-of-two column cover of one mask matrix.
    pub fn padded_cols(width: usize) -> usize {
        width.max(1).next_power_of_two()
    }

    /// Validate the statement dimensions against the shape-hiding layout.
    pub fn validate(&self, layout: &Section6PublicLayout) -> Result<(), Section6DataError> {
        layout.validate()?;
        let dimensions_match = self.bucket_rows == layout.bucket_count
            && self.padded_rows == layout.row_cover()?
            && self.eta_mask_width == SECTION6_FIXED_ETA_WIDTH
            && self.local_mask_width == SECTION6_FIXED_LOCAL_WIDTH
            && self.offset_mask_width == layout.global_log;
        if !dimensions_match {
            return Err(Section6DataError::InvalidLayout(
                "shape-mask statement dimensions do not match the public layout".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Prover-only registered mask state retained for the per-opening relations.
pub struct CommittedSection6ShapeMasks {
    statement: Section6ShapeMaskStatement,
    layout: Section6Layout,
    metadata: PrivateShapeMetadata,
    masks: Section6RegisteredMasks<Fr>,
    d_eta_columns: Vec<CommittedDoryFieldVector>,
    d_z_columns: Vec<CommittedDoryFieldVector>,
    d_eta_prime_columns: Vec<CommittedDoryFieldVector>,
}

impl CommittedSection6ShapeMasks {
    pub fn statement(&self) -> &Section6ShapeMaskStatement {
        &self.statement
    }

    pub fn layout(&self) -> &Section6Layout {
        &self.layout
    }

    pub fn metadata(&self) -> &PrivateShapeMetadata {
        &self.metadata
    }

    pub fn masks(&self) -> &Section6RegisteredMasks<Fr> {
        &self.masks
    }

    /// One committed view per logical `D_eta` column.
    pub fn d_eta_columns(&self) -> &[CommittedDoryFieldVector] {
        &self.d_eta_columns
    }

    /// One committed view per logical `D_Z` column.
    pub fn d_z_columns(&self) -> &[CommittedDoryFieldVector] {
        &self.d_z_columns
    }

    /// One committed view per logical `D_eta'` column.
    pub fn d_eta_prime_columns(&self) -> &[CommittedDoryFieldVector] {
        &self.d_eta_prime_columns
    }

    pub fn d_eta_column_values(&self, column: usize) -> Option<&[Fr]> {
        self.masks.d_eta.padded_column(column)
    }

    pub fn d_z_column_values(&self, column: usize) -> Option<&[Fr]> {
        self.masks.d_z.padded_column(column)
    }

    pub fn d_eta_prime_column_values(&self, column: usize) -> Option<&[Fr]> {
        self.masks.d_eta_prime.padded_column(column)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Section6ShapeMaskCommitmentError {
    Data(Section6DataError),
    Commitment(Section6FinalOpeningError),
    MissingColumnViews,
}

impl Display for Section6ShapeMaskCommitmentError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Data(error) => Display::fmt(error, formatter),
            Self::Commitment(error) => Display::fmt(error, formatter),
            Self::MissingColumnViews => {
                formatter.write_str("shape commitment returned no column views")
            }
        }
    }
}

impl std::error::Error for Section6ShapeMaskCommitmentError {}

impl From<Section6DataError> for Section6ShapeMaskCommitmentError {
    fn from(error: Section6DataError) -> Self {
        Self::Data(error)
    }
}

impl From<Section6FinalOpeningError> for Section6ShapeMaskCommitmentError {
    fn from(error: Section6FinalOpeningError) -> Self {
        Self::Commitment(error)
    }
}

/// Commit `D_eta`, `D_Z`, and `D_eta'` as three hiding Dory commitments.
pub fn commit_section6_shape_masks_on_grid_zk(
    parameters: &DoryProverParameters,
    grid: &Section6DoryGrid,
    layout: Section6Layout,
    metadata: PrivateShapeMetadata,
) -> Result<CommittedSection6ShapeMasks, Section6ShapeMaskCommitmentError> {
    let masks = Section6RegisteredMasks::<Fr>::from_metadata(&layout, &metadata)?;
    let padded_rows = masks.d_eta.padded_rows;
    if masks.d_z.padded_rows != padded_rows || masks.d_eta_prime.padded_rows != padded_rows {
        return Err(Section6ShapeMaskCommitmentError::Data(
            Section6DataError::InvalidMetadata("mask row covers diverge".to_owned()),
        ));
    }

    let d_eta_columns = commit_padded_matrix(parameters, grid, &masks.d_eta)?;
    let d_z_columns = commit_padded_matrix(parameters, grid, &masks.d_z)?;
    let d_eta_prime_columns = commit_padded_matrix(parameters, grid, &masks.d_eta_prime)?;

    let statement = Section6ShapeMaskStatement {
        bucket_rows: layout.bucket_count,
        padded_rows,
        eta_mask_width: masks.d_eta.logical_cols,
        local_mask_width: masks.d_z.logical_cols,
        offset_mask_width: masks.d_eta_prime.logical_cols,
        d_eta_commitment: first_commitment(&d_eta_columns)?,
        d_z_commitment: first_commitment(&d_z_columns)?,
        d_eta_prime_commitment: first_commitment(&d_eta_prime_columns)?,
    };

    Ok(CommittedSection6ShapeMasks {
        statement,
        layout,
        metadata,
        masks,
        d_eta_columns,
        d_z_columns,
        d_eta_prime_columns,
    })
}

/// Derive the deterministic shape metadata from a dense-BVA descriptor and
/// commit the three mask matrices.
pub fn commit_section6_shape_masks_from_descriptor_on_grid_zk(
    parameters: &DoryProverParameters,
    grid: &Section6DoryGrid,
    layout: Section6Layout,
    descriptor: &DenseBvaDescriptor,
) -> Result<CommittedSection6ShapeMasks, Section6ShapeMaskCommitmentError> {
    let metadata = PrivateShapeMetadata::from_descriptor(&layout, descriptor)?;
    commit_section6_shape_masks_on_grid_zk(parameters, grid, layout, metadata)
}

fn commit_padded_matrix(
    parameters: &DoryProverParameters,
    grid: &Section6DoryGrid,
    matrix: &PaddedFieldMatrix<Fr>,
) -> Result<Vec<CommittedDoryFieldVector>, Section6ShapeMaskCommitmentError> {
    let column_slices = matrix.padded_columns();
    if column_slices.is_empty() {
        return Err(Section6ShapeMaskCommitmentError::MissingColumnViews);
    }
    Ok(commit_section6_field_matrix_columns_on_grid_zk(
        parameters,
        grid,
        &column_slices,
        matrix.padded_cols,
    )?)
}

fn first_commitment(
    columns: &[CommittedDoryFieldVector],
) -> Result<DoryCommitment, Section6ShapeMaskCommitmentError> {
    columns
        .first()
        .map(|column| column.commitment.clone())
        .ok_or(Section6ShapeMaskCommitmentError::MissingColumnViews)
}

#[cfg(test)]
mod tests {
    use ark_ff::Zero;

    use crate::{
        jolt_dory::{commitment_binding_bytes, setup_dory_parameters},
        matrix_collection::DenseBvaDescriptor,
        section6_data::{
            PrivateShapeMetadata, Section6Layout, SECTION6_FIXED_ETA_WIDTH,
            SECTION6_FIXED_LOCAL_WIDTH, SECTION6_FIXED_POINT_LOG,
        },
        section6_final_opening::Section6DoryGrid,
    };

    use super::{
        commit_section6_shape_masks_from_descriptor_on_grid_zk,
        commit_section6_shape_masks_on_grid_zk,
    };

    #[test]
    fn registration_commits_three_mask_matrices() {
        let layout = Section6Layout::new(2, 8, 7, vec![4, 2, 1], 8).unwrap();
        let metadata = PrivateShapeMetadata {
            tau: vec![1, 2, 2],
            alpha_l: vec![0, 0, 2],
            alpha_r: vec![0, 0, 0],
            gamma_l: vec![1, 1, 0],
            gamma_r: vec![1, 0, 0],
        };
        let grid = Section6DoryGrid::new(12).unwrap();
        let (parameters, _) = setup_dory_parameters(grid.num_vars());

        let committed = commit_section6_shape_masks_on_grid_zk(
            &parameters,
            &grid,
            layout.clone(),
            metadata.clone(),
        )
        .unwrap();

        let statement = committed.statement();
        statement.validate(&layout.public()).unwrap();
        assert_eq!(statement.bucket_rows, 3);
        assert_eq!(statement.padded_rows, 4);
        assert_eq!(statement.eta_mask_width, SECTION6_FIXED_ETA_WIDTH);
        assert_eq!(statement.local_mask_width, SECTION6_FIXED_LOCAL_WIDTH);
        assert_eq!(statement.offset_mask_width, layout.global_log);
        assert_eq!(statement.physical_commitment_count(), 3);
        assert_ne!(
            commitment_binding_bytes(&statement.d_eta_commitment),
            commitment_binding_bytes(&statement.d_z_commitment)
        );
        assert_ne!(
            commitment_binding_bytes(&statement.d_eta_commitment),
            commitment_binding_bytes(&statement.d_eta_prime_commitment)
        );

        assert_eq!(committed.d_eta_columns().len(), SECTION6_FIXED_ETA_WIDTH);
        assert_eq!(committed.d_z_columns().len(), SECTION6_FIXED_LOCAL_WIDTH);
        assert_eq!(committed.d_eta_prime_columns().len(), layout.global_log);

        // Bucket 0 is 2x2: its first column and first row coordinates vary.
        let d_z_col_0 = committed.d_z_column_values(0).unwrap();
        assert_eq!(d_z_col_0[0], ark_bn254::Fr::from(1_u64));
        let d_z_row_side = committed
            .d_z_column_values(SECTION6_FIXED_POINT_LOG)
            .unwrap();
        assert_eq!(d_z_row_side[0], ark_bn254::Fr::from(1_u64));
        // Bucket 2 sits at row offset 2 of matrix 2: alpha_l bit 1 is set.
        let bit_column = committed
            .d_eta_column_values(SECTION6_FIXED_POINT_LOG + 1)
            .unwrap();
        assert_eq!(bit_column[2], ark_bn254::Fr::from(1_u64));
        // tau - 1 bits live in the trailing block: bucket 1 and 2 use matrix 2.
        let tau_bit_column = committed
            .d_eta_column_values(SECTION6_FIXED_LOCAL_WIDTH)
            .unwrap();
        assert!(tau_bit_column[0].is_zero());
        assert_eq!(tau_bit_column[1], ark_bn254::Fr::from(1_u64));
        assert_eq!(tau_bit_column[2], ark_bn254::Fr::from(1_u64));
        // Global offsets 0, 4, 6 in little-endian bits.
        let offset_bit_2 = committed.d_eta_prime_column_values(2).unwrap();
        assert!(offset_bit_2[0].is_zero());
        assert_eq!(offset_bit_2[1], ark_bn254::Fr::from(1_u64));
        assert_eq!(offset_bit_2[2], ark_bn254::Fr::from(1_u64));
        let offset_bit_1 = committed.d_eta_prime_column_values(1).unwrap();
        assert!(offset_bit_1[1].is_zero());
        assert_eq!(offset_bit_1[2], ark_bn254::Fr::from(1_u64));

        for column in 0..SECTION6_FIXED_ETA_WIDTH {
            let values = committed.d_eta_column_values(column).unwrap();
            assert_eq!(values.len(), statement.padded_rows);
            assert!(values[statement.bucket_rows..].iter().all(Zero::is_zero));
        }
    }

    #[test]
    fn descriptor_constructor_derives_private_shape_metadata() {
        let descriptor = DenseBvaDescriptor::from_shapes(&[(3, 5), (2, 3)]).unwrap();
        let layout = Section6Layout::from_descriptor(&descriptor, 8).unwrap();
        let expected_metadata =
            PrivateShapeMetadata::from_descriptor(&layout, &descriptor).unwrap();
        let grid = Section6DoryGrid::new(12).unwrap();
        let (parameters, _) = setup_dory_parameters(grid.num_vars());

        let committed = commit_section6_shape_masks_from_descriptor_on_grid_zk(
            &parameters,
            &grid,
            layout.clone(),
            &descriptor,
        )
        .unwrap();

        assert_eq!(committed.layout(), &layout);
        assert_eq!(committed.metadata(), &expected_metadata);
        assert_eq!(committed.statement().physical_commitment_count(), 3);
    }
}
