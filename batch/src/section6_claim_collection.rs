//! Collection adapters for the one terminal Section 6 Dory opening.

use ark_bn254::Fr as ArkFr;
use jolt_crypto::{Bn254G1, JoltGroup};
use jolt_field::Fr;
use jolt_poly::eq_index_msb;

use crate::{
    hidden_relation::{
        matrix_column_point, HiddenRelationProverOpening, HiddenRelationTable,
        HiddenRelationTableStatement, HiddenRelationVerifierOpening, HiddenScalarWitness,
    },
    jolt_dory::CommittedDoryFieldVector,
    section6_final_opening::{
        resolve_section6_hidden_relation_prover_claim,
        resolve_section6_hidden_relation_sparse_matrix_prover_claim,
        resolve_section6_hidden_relation_sparse_matrix_verifier_claim,
        resolve_section6_hidden_relation_verifier_claim,
        resolve_section6_hidden_relation_whole_matrix_prover_claim,
        resolve_section6_hidden_relation_whole_matrix_verifier_claim,
        ResolvedSection6FieldProverClaim, ResolvedSection6FieldVerifierClaim, Section6DoryGrid,
        Section6EvaluationCommitment, Section6FinalOpeningError, Section6FinalOpeningProverClaim,
        Section6FinalOpeningVerifierClaim,
    },
};

enum CollectedEvaluationCommitment {
    Direct(Bn254G1),
    MatrixColumns {
        logical_cols: usize,
        padded_cols: usize,
        selected_columns: Vec<usize>,
        commitments: Vec<Bn254G1>,
    },
}

impl CollectedEvaluationCommitment {
    fn as_commitment(&self) -> Section6EvaluationCommitment<'_> {
        match self {
            Self::Direct(commitment) => Section6EvaluationCommitment::Direct(*commitment),
            Self::MatrixColumns {
                logical_cols,
                padded_cols,
                selected_columns,
                commitments,
            } => Section6EvaluationCommitment::MatrixColumns {
                logical_cols: *logical_cols,
                padded_cols: *padded_cols,
                selected_columns,
                commitments,
            },
        }
    }
}

struct ProverMatrixGroup<'a> {
    representative: HiddenRelationTable<'a>,
    table_indices: Vec<usize>,
    by_column: Vec<Option<usize>>,
    padded_cols: usize,
}

struct VerifierMatrixGroup<'a> {
    representative: HiddenRelationTableStatement<'a>,
    table_indices: Vec<usize>,
    by_column: Vec<Option<usize>>,
    padded_cols: usize,
}

/// Prover-side terminal claims after all algebraic reductions have completed.
pub struct Section6ProverClaimCollection<'a> {
    resolved: Vec<ResolvedSection6FieldProverClaim<'a>>,
    evaluations: Vec<HiddenScalarWitness>,
    evaluation_commitments: Vec<CollectedEvaluationCommitment>,
}

impl<'a> Section6ProverClaimCollection<'a> {
    pub fn new() -> Self {
        Self {
            resolved: Vec::new(),
            evaluations: Vec::new(),
            evaluation_commitments: Vec::new(),
        }
    }

    pub fn push_relation(
        &mut self,
        opening: &HiddenRelationProverOpening<'a>,
        grid: &Section6DoryGrid,
    ) -> Result<(), Section6FinalOpeningError> {
        let tables = opening.hidden_tables();
        if tables.len() != opening.evaluations().len()
            || tables.len() != opening.evaluation_commitments().len()
            || tables.len() != opening.evaluation_blinds().len()
        {
            return Err(Section6FinalOpeningError::ProofShape);
        }
        let groups = prover_matrix_groups(opening)?;
        let mut consumed = vec![false; tables.len()];
        for group in groups {
            for &index in &group.table_indices {
                consumed[index] = true;
            }
            let mut commitments = vec![Bn254G1::identity(); group.padded_cols];
            let selected_columns = group
                .by_column
                .iter()
                .enumerate()
                .filter_map(|(column, table_index)| table_index.map(|_| column))
                .collect::<Vec<_>>();
            for &column in &selected_columns {
                let table_index =
                    group.by_column[column].ok_or(Section6FinalOpeningError::ProofShape)?;
                commitments[column] = opening.evaluation_commitments()[table_index];
            }
            let matrix_commitments = [&group.representative.committed.commitment];
            let column_commitments = [commitments.clone()];
            let column_point = matrix_column_point(
                opening.outer_binding(),
                opening.logical_len(),
                group.padded_cols,
                opening.row_point(),
                &matrix_commitments,
                &column_commitments,
            );
            let weights = selected_columns
                .iter()
                .map(|&column| eq_index_msb(&column_point, column as u128))
                .collect::<Vec<_>>();
            let value = selected_columns
                .iter()
                .zip(&weights)
                .map(|(&column, &weight)| {
                    let table_index = group.by_column[column]
                        .expect("selected columns have relation table indices");
                    opening.evaluations()[table_index] * weight
                })
                .sum::<Fr>();
            let selected_commitments = selected_columns
                .iter()
                .map(|&column| commitments[column])
                .collect::<Vec<_>>();
            let commitment = Bn254G1::msm(&selected_commitments, &weights);
            let blind = selected_columns
                .iter()
                .zip(&weights)
                .map(|(&column, &weight)| {
                    let table_index = group.by_column[column]
                        .expect("selected columns have relation table indices");
                    opening.evaluation_blinds()[table_index] * weight
                })
                .sum::<Fr>();
            let full_point = column_point
                .iter()
                .chain(opening.row_point())
                .copied()
                .collect::<Vec<_>>();
            let resolved = if selected_columns.len() == group.by_column.len() {
                resolve_section6_hidden_relation_whole_matrix_prover_claim(
                    group.representative,
                    &full_point,
                    grid,
                )?
            } else {
                resolve_section6_hidden_relation_sparse_matrix_prover_claim(
                    group.representative,
                    &full_point,
                    &selected_columns,
                    grid,
                )?
            };
            self.resolved.push(resolved);
            self.evaluations.push(HiddenScalarWitness {
                value: ArkFr::from(value),
                commitment,
                blind,
            });
            self.evaluation_commitments
                .push(CollectedEvaluationCommitment::MatrixColumns {
                    logical_cols: group.by_column.len(),
                    padded_cols: group.padded_cols,
                    selected_columns,
                    commitments: selected_commitments,
                });
        }

        for (index, (((table, &value), &commitment), &blind)) in tables
            .iter()
            .copied()
            .zip(opening.evaluations())
            .zip(opening.evaluation_commitments())
            .zip(opening.evaluation_blinds())
            .enumerate()
        {
            if consumed[index] {
                continue;
            }
            self.resolved
                .push(resolve_section6_hidden_relation_prover_claim(
                    table,
                    opening.row_point(),
                    grid,
                )?);
            self.evaluations.push(HiddenScalarWitness {
                value: ArkFr::from(value),
                commitment,
                blind,
            });
            self.evaluation_commitments
                .push(CollectedEvaluationCommitment::Direct(commitment));
        }
        Ok(())
    }

    pub fn push_field(
        &mut self,
        values: &'a [ArkFr],
        committed: &'a CommittedDoryFieldVector,
        point_msb: &[ArkFr],
        evaluation: HiddenScalarWitness,
        grid: &Section6DoryGrid,
    ) -> Result<(), Section6FinalOpeningError> {
        let point = point_msb
            .iter()
            .copied()
            .map(jolt_field::Fr::from)
            .collect::<Vec<_>>();
        self.resolved
            .push(resolve_section6_hidden_relation_prover_claim(
                HiddenRelationTable { values, committed },
                &point,
                grid,
            )?);
        let evaluation_commitment = evaluation.commitment;
        self.evaluations.push(evaluation);
        self.evaluation_commitments
            .push(CollectedEvaluationCommitment::Direct(evaluation_commitment));
        Ok(())
    }

    pub fn claims(&self) -> Vec<Section6FinalOpeningProverClaim<'_>> {
        self.resolved
            .iter()
            .zip(&self.evaluations)
            .zip(&self.evaluation_commitments)
            .map(|((resolved, evaluation), commitment)| {
                resolved.as_claim_with_evaluation_commitment(evaluation, commitment.as_commitment())
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.resolved.len()
    }

    pub fn is_empty(&self) -> bool {
        self.resolved.is_empty()
    }
}

impl Default for Section6ProverClaimCollection<'_> {
    fn default() -> Self {
        Self::new()
    }
}

/// Verifier-side commitment-only terminal claims in the same order.
pub struct Section6VerifierClaimCollection<'a> {
    resolved: Vec<ResolvedSection6FieldVerifierClaim<'a>>,
    evaluation_commitments: Vec<CollectedEvaluationCommitment>,
}

impl<'a> Section6VerifierClaimCollection<'a> {
    pub fn new() -> Self {
        Self {
            resolved: Vec::new(),
            evaluation_commitments: Vec::new(),
        }
    }

    pub fn push_relation(
        &mut self,
        opening: &HiddenRelationVerifierOpening<'a>,
        grid: &Section6DoryGrid,
    ) -> Result<(), Section6FinalOpeningError> {
        let tables = opening.hidden_tables();
        if tables.len() != opening.evaluation_commitments().len() {
            return Err(Section6FinalOpeningError::ProofShape);
        }
        let groups = verifier_matrix_groups(opening)?;
        let mut consumed = vec![false; tables.len()];
        for group in groups {
            for &index in &group.table_indices {
                consumed[index] = true;
            }
            let mut commitments = vec![Bn254G1::identity(); group.padded_cols];
            let selected_columns = group
                .by_column
                .iter()
                .enumerate()
                .filter_map(|(column, table_index)| table_index.map(|_| column))
                .collect::<Vec<_>>();
            for &column in &selected_columns {
                let table_index =
                    group.by_column[column].ok_or(Section6FinalOpeningError::ProofShape)?;
                commitments[column] = opening.evaluation_commitments()[table_index];
            }
            let matrix_commitments = [group.representative.commitment];
            let column_commitments = [commitments.clone()];
            let column_point = matrix_column_point(
                opening.outer_binding(),
                opening.logical_len(),
                group.padded_cols,
                opening.row_point(),
                &matrix_commitments,
                &column_commitments,
            );
            let selected_commitments = selected_columns
                .iter()
                .map(|&column| commitments[column])
                .collect::<Vec<_>>();
            let full_point = column_point
                .iter()
                .chain(opening.row_point())
                .copied()
                .collect::<Vec<_>>();
            let resolved = if selected_columns.len() == group.by_column.len() {
                resolve_section6_hidden_relation_whole_matrix_verifier_claim(
                    group.representative,
                    &full_point,
                    Bn254G1::identity(),
                    grid,
                )?
            } else {
                resolve_section6_hidden_relation_sparse_matrix_verifier_claim(
                    group.representative,
                    &full_point,
                    &selected_columns,
                    Bn254G1::identity(),
                    grid,
                )?
            };
            self.resolved.push(resolved);
            self.evaluation_commitments
                .push(CollectedEvaluationCommitment::MatrixColumns {
                    logical_cols: group.by_column.len(),
                    padded_cols: group.padded_cols,
                    selected_columns,
                    commitments: selected_commitments,
                });
        }

        for (index, (table, &evaluation_commitment)) in tables
            .iter()
            .copied()
            .zip(opening.evaluation_commitments())
            .enumerate()
        {
            if consumed[index] {
                continue;
            }
            self.resolved
                .push(resolve_section6_hidden_relation_verifier_claim(
                    table,
                    opening.row_point(),
                    evaluation_commitment,
                    grid,
                )?);
            self.evaluation_commitments
                .push(CollectedEvaluationCommitment::Direct(evaluation_commitment));
        }
        Ok(())
    }

    pub fn push_field(
        &mut self,
        commitment: &'a jolt_dory::DoryCommitment,
        logical_len: usize,
        point_msb: &[ArkFr],
        evaluation_commitment: jolt_crypto::Bn254G1,
        grid: &Section6DoryGrid,
    ) -> Result<(), Section6FinalOpeningError> {
        let point = point_msb
            .iter()
            .copied()
            .map(jolt_field::Fr::from)
            .collect::<Vec<_>>();
        self.resolved
            .push(resolve_section6_hidden_relation_verifier_claim(
                HiddenRelationTableStatement::vector(commitment, logical_len),
                &point,
                evaluation_commitment,
                grid,
            )?);
        self.evaluation_commitments
            .push(CollectedEvaluationCommitment::Direct(evaluation_commitment));
        Ok(())
    }

    pub fn claims(&self) -> Vec<Section6FinalOpeningVerifierClaim<'_>> {
        self.resolved
            .iter()
            .zip(&self.evaluation_commitments)
            .map(|(resolved, commitment)| {
                resolved.as_claim_with_evaluation_commitment(commitment.as_commitment())
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.resolved.len()
    }

    pub fn is_empty(&self) -> bool {
        self.resolved.is_empty()
    }
}

impl Default for Section6VerifierClaimCollection<'_> {
    fn default() -> Self {
        Self::new()
    }
}

fn prover_matrix_groups<'a>(
    opening: &HiddenRelationProverOpening<'a>,
) -> Result<Vec<ProverMatrixGroup<'a>>, Section6FinalOpeningError> {
    let mut groups = Vec::<ProverMatrixGroup<'a>>::new();
    for (table_index, &table) in opening.hidden_tables().iter().enumerate() {
        let Some(matrix) = &table.committed.matrix else {
            continue;
        };
        if matrix.logical_cols == 0
            || matrix.logical_cols > matrix.padded_cols
            || table.committed.matrix_column >= matrix.logical_cols
            || matrix.padded_rows != opening.logical_len()
        {
            return Err(Section6FinalOpeningError::ProofShape);
        }
        let group_index = groups
            .iter()
            .position(|group| {
                group.representative.committed.commitment == table.committed.commitment
                    && group.representative.committed.logical_len == table.committed.logical_len
                    && group.by_column.len() == matrix.logical_cols
                    && group.padded_cols == matrix.padded_cols
            })
            .unwrap_or_else(|| {
                groups.push(ProverMatrixGroup {
                    representative: table,
                    table_indices: Vec::new(),
                    by_column: vec![None; matrix.logical_cols],
                    padded_cols: matrix.padded_cols,
                });
                groups.len() - 1
            });
        let group = &mut groups[group_index];
        let column = table.committed.matrix_column;
        if let Some(previous) = group.by_column[column] {
            if opening.evaluations()[previous] != opening.evaluations()[table_index]
                || opening.evaluation_commitments()[previous]
                    != opening.evaluation_commitments()[table_index]
                || opening.evaluation_blinds()[previous] != opening.evaluation_blinds()[table_index]
            {
                return Err(Section6FinalOpeningError::ProofShape);
            }
        } else {
            group.by_column[column] = Some(table_index);
        }
        group.table_indices.push(table_index);
    }
    Ok(groups)
}

fn verifier_matrix_groups<'a>(
    opening: &HiddenRelationVerifierOpening<'a>,
) -> Result<Vec<VerifierMatrixGroup<'a>>, Section6FinalOpeningError> {
    let mut groups = Vec::<VerifierMatrixGroup<'a>>::new();
    for (table_index, &table) in opening.hidden_tables().iter().enumerate() {
        let Some(matrix) = table.matrix else {
            continue;
        };
        if matrix.logical_cols == 0
            || matrix.logical_cols > matrix.padded_cols
            || matrix.column >= matrix.logical_cols
            || table.logical_len != opening.logical_len()
        {
            return Err(Section6FinalOpeningError::ProofShape);
        }
        let group_index = groups
            .iter()
            .position(|group| {
                group.representative.commitment == table.commitment
                    && group.representative.logical_len == table.logical_len
                    && group.by_column.len() == matrix.logical_cols
                    && group.padded_cols == matrix.padded_cols
            })
            .unwrap_or_else(|| {
                groups.push(VerifierMatrixGroup {
                    representative: table,
                    table_indices: Vec::new(),
                    by_column: vec![None; matrix.logical_cols],
                    padded_cols: matrix.padded_cols,
                });
                groups.len() - 1
            });
        let group = &mut groups[group_index];
        let column = matrix.column;
        if let Some(previous) = group.by_column[column] {
            if opening.evaluation_commitments()[previous]
                != opening.evaluation_commitments()[table_index]
            {
                return Err(Section6FinalOpeningError::ProofShape);
            }
        } else {
            group.by_column[column] = Some(table_index);
        }
        group.table_indices.push(table_index);
    }
    Ok(groups)
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr as ArkFr;
    use jolt_crypto::{Bn254G1, JoltGroup};
    use jolt_field::{Fr, FromPrimitiveInt};

    use super::Section6VerifierClaimCollection;
    use crate::{
        hidden_relation::{HiddenRelationTableStatement, HiddenRelationVerifierOpening},
        jolt_dory::{setup_dory_parameters, CommittedDoryFieldVector},
        matrix_collection::DenseBvaDescriptor,
        section6_data::Section6Layout,
        section6_final_opening::{
            commit_section6_field_matrix_columns_on_grid_zk, Section6DoryGrid,
            Section6FinalOpeningFunctional,
        },
        validity_final::FinalAuxColumns,
        validity_point::PointAuxColumns,
    };

    fn commit_matrix(
        parameters: &crate::jolt_dory::DoryProverParameters,
        grid: &Section6DoryGrid,
        logical_cols: usize,
        padded_cols: usize,
        seed: u64,
    ) -> Vec<CommittedDoryFieldVector> {
        let columns = (0..logical_cols)
            .map(|column| {
                vec![
                    ArkFr::from(seed + 2 * column as u64),
                    ArkFr::from(seed + 2 * column as u64 + 1),
                ]
            })
            .collect::<Vec<_>>();
        let columns = columns.iter().map(Vec::as_slice).collect::<Vec<_>>();
        commit_section6_field_matrix_columns_on_grid_zk(parameters, grid, &columns, padded_cols)
            .unwrap()
    }

    #[test]
    fn llama_width_metadata_compacts_to_25_terminal_claims() {
        let descriptor = DenseBvaDescriptor::from_shapes(&[(1, 2)]).unwrap();
        let layout = Section6Layout::from_descriptor_with_widths(&descriptor, 17, 9, 8).unwrap();
        let pair = layout.pair_width();
        let eta = layout.eta_width();
        let mask_cols = pair + eta;
        let shape_aux_cols = mask_cols + 2;
        let point_aux_cols = PointAuxColumns::new(pair, eta).count();
        let final_aux_cols = FinalAuxColumns::new(pair).count();

        assert_eq!((pair, eta), (34, 43));
        assert_eq!((mask_cols, shape_aux_cols), (77, 79));
        assert_eq!((point_aux_cols, final_aux_cols), (77, 70));

        let specs = [
            (mask_cols, 128),
            (shape_aux_cols, 128),
            (3, 4),
            (point_aux_cols, 128),
            (pair, 64),
            (pair + 1, 64),
            (final_aux_cols, 128),
            (4, 4),
            (2, 2),
            (4, 4),
            (2, 2),
            (2, 2),
            (2, 2),
            (2, 2),
            (2, 2),
        ];
        let grid = Section6DoryGrid::new(8).unwrap();
        let (parameters, _) = setup_dory_parameters(grid.num_vars());
        let sources = specs
            .into_iter()
            .enumerate()
            .map(|(index, (logical_cols, padded_cols))| {
                commit_matrix(
                    &parameters,
                    &grid,
                    logical_cols,
                    padded_cols,
                    1_000 * (index as u64 + 1),
                )
            })
            .collect::<Vec<_>>();

        let mut collection = Section6VerifierClaimCollection::new();
        let relations = [
            (0, 0..mask_cols),
            (0, mask_cols - 2..mask_cols),
            (0, 0..mask_cols),
            (1, 0..shape_aux_cols),
            (1, shape_aux_cols - 4..shape_aux_cols),
            (1, 0..4),
            (2, 0..3),
            (2, 0..3),
            (3, 0..1),
            (3, 0..point_aux_cols),
            (4, 0..pair),
            (5, 0..pair + 1),
            (6, 0..final_aux_cols),
            (7, 0..4),
            (8, 0..2),
            (9, 0..4),
            (9, 0..4),
            (10, 0..2),
            (10, 0..2),
            (11, 0..2),
            (12, 0..2),
            (13, 0..2),
            (13, 0..2),
            (14, 0..2),
        ];
        for (tag, (source, columns)) in relations.into_iter().enumerate() {
            let tables = columns
                .map(|column| HiddenRelationTableStatement::from(&sources[source][column]))
                .collect::<Vec<_>>();
            let opening = HiddenRelationVerifierOpening::from_test_parts(
                tables.clone(),
                2,
                vec![Fr::from_u64(tag as u64 + 2)],
                vec![Bn254G1::identity(); tables.len()],
                vec![tag as u8],
            )
            .unwrap();
            collection.push_relation(&opening, &grid).unwrap();
        }

        assert_eq!(collection.len(), 24);
        assert_eq!(collection.len() + 1, 25);
    }

    #[test]
    fn sparse_matrix_groups_deduplicate_columns_and_keep_distinct_row_points() {
        let grid = Section6DoryGrid::new(4).unwrap();
        let (parameters, _) = setup_dory_parameters(grid.num_vars());
        let source = commit_matrix(&parameters, &grid, 4, 4, 700);
        let statement = |column| HiddenRelationTableStatement::from(&source[column]);
        let first_tables = vec![statement(0), statement(0), statement(2)];
        let first = HiddenRelationVerifierOpening::from_test_parts(
            first_tables.clone(),
            2,
            vec![Fr::from_u64(13)],
            vec![Bn254G1::identity(); first_tables.len()],
            vec![1],
        )
        .unwrap();
        let second_tables = vec![statement(0), statement(2)];
        let second = HiddenRelationVerifierOpening::from_test_parts(
            second_tables.clone(),
            2,
            vec![Fr::from_u64(17)],
            vec![Bn254G1::identity(); second_tables.len()],
            vec![2],
        )
        .unwrap();

        let mut collection = Section6VerifierClaimCollection::new();
        collection.push_relation(&first, &grid).unwrap();
        collection.push_relation(&second, &grid).unwrap();
        assert_eq!(collection.len(), 2);

        let claims = collection.claims();
        assert_eq!(claims[0].commitment, claims[1].commitment);
        assert_ne!(claims[0].point_msb, claims[1].point_msb);
        for claim in claims {
            let Section6FinalOpeningFunctional::SparseMatrixColumns {
                padded_rows,
                logical_cols,
                padded_cols,
                selected_columns,
            } = claim.functional
            else {
                panic!("partial matrix group must use the sparse functional");
            };
            assert_eq!((padded_rows, logical_cols, padded_cols), (2, 4, 4));
            assert_eq!(selected_columns, &[0, 2]);
        }
    }
}
