//! One tagged LogUp argument for several lookup families.

use std::fmt::{Display, Formatter};

use ark_bn254::Fr;
use ark_ff::Zero;
use ark_std::rand::RngCore;
use sha2::{Digest, Sha256};

use crate::{
    hidden_logup::{
        prove_hidden_logup_openings_batch, prove_hidden_logup_parts_deferred,
        prove_hidden_logup_parts_deferred_on_grid, verify_hidden_logup_openings_batch,
        verify_hidden_logup_parts_deferred, verify_hidden_logup_parts_deferred_on_grid,
        HiddenLogUpError, HiddenLogUpPartsProof, HiddenLogUpPartsReductionProof,
        HiddenLogUpProverOpeningState, HiddenLogUpSide, HiddenLogUpVerifierOpeningState,
        MixedLogUpColumn, MixedLogUpColumnStatement,
    },
    hidden_relation::{HiddenRelationTable, HiddenRelationTableStatement, PublicRelationTable},
    jolt_dory::{DoryProverParameters, DoryVerifierParameters},
    section6_final_opening::Section6DoryGrid,
};

const UNION_DOMAIN: &[u8] = b"parcomm-hidden-logup-union-v1";

/// One prover-side lookup family whose parts share a tuple width.
#[derive(Clone, Copy)]
pub struct HiddenLogUpUnionProverFamily<'a> {
    pub query_parts: &'a [Vec<MixedLogUpColumn<'a>>],
    pub table_parts: &'a [Vec<MixedLogUpColumn<'a>>],
}

/// One verifier-side lookup family whose parts share a tuple width.
#[derive(Clone, Copy)]
pub struct HiddenLogUpUnionVerifierFamily<'a> {
    pub query_parts: &'a [Vec<MixedLogUpColumnStatement<'a>>],
    pub table_parts: &'a [Vec<MixedLogUpColumnStatement<'a>>],
}

#[derive(Debug)]
pub enum HiddenLogUpUnionError {
    EmptyFamilies,
    EmptyFamily {
        family: usize,
        side: HiddenLogUpSide,
    },
    EmptyTuple {
        family: usize,
    },
    PartTupleWidth {
        family: usize,
        side: HiddenLogUpSide,
        part: usize,
        expected: usize,
        actual: usize,
    },
    LogUp(HiddenLogUpError),
}

impl Display for HiddenLogUpUnionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for HiddenLogUpUnionError {}

impl From<HiddenLogUpError> for HiddenLogUpUnionError {
    fn from(error: HiddenLogUpError) -> Self {
        Self::LogUp(error)
    }
}

/// Prove all lookup families with one tagged multipart LogUp proof.
pub fn prove_hidden_logup_union(
    label: &[u8],
    parameters: &DoryProverParameters,
    families: &[HiddenLogUpUnionProverFamily<'_>],
    rng: &mut impl RngCore,
) -> Result<HiddenLogUpPartsProof, HiddenLogUpUnionError> {
    let (reduction, pending) = prove_hidden_logup_union_deferred(label, parameters, families, rng)?;
    let relation_opening = prove_hidden_logup_openings_batch(parameters, &[&pending], rng)?;
    Ok(HiddenLogUpPartsProof {
        auxiliary_commitments: reduction.auxiliary_commitments,
        identity_reduction: reduction.identity_reduction,
        sum_reduction: reduction.sum_reduction,
        relation_opening,
    })
}

/// Reduce all tagged lookup families while deferring terminal PCS openings.
pub fn prove_hidden_logup_union_deferred<'a>(
    label: &[u8],
    parameters: &DoryProverParameters,
    families: &[HiddenLogUpUnionProverFamily<'a>],
    rng: &mut impl RngCore,
) -> Result<
    (
        HiddenLogUpPartsReductionProof,
        HiddenLogUpProverOpeningState<'a>,
    ),
    HiddenLogUpUnionError,
> {
    prove_hidden_logup_union_deferred_with_grid(label, parameters, None, families, rng)
}

/// Reduce all tagged lookup families and place every auxiliary matrix on the
/// shared Section 6 Dory grid.
pub fn prove_hidden_logup_union_deferred_on_grid<'a>(
    label: &[u8],
    parameters: &DoryProverParameters,
    grid: &Section6DoryGrid,
    families: &[HiddenLogUpUnionProverFamily<'a>],
    rng: &mut impl RngCore,
) -> Result<
    (
        HiddenLogUpPartsReductionProof,
        HiddenLogUpProverOpeningState<'a>,
    ),
    HiddenLogUpUnionError,
> {
    prove_hidden_logup_union_deferred_with_grid(label, parameters, Some(grid), families, rng)
}

fn prove_hidden_logup_union_deferred_with_grid<'a>(
    label: &[u8],
    parameters: &DoryProverParameters,
    grid: Option<&Section6DoryGrid>,
    families: &[HiddenLogUpUnionProverFamily<'a>],
    rng: &mut impl RngCore,
) -> Result<
    (
        HiddenLogUpPartsReductionProof,
        HiddenLogUpProverOpeningState<'a>,
    ),
    HiddenLogUpUnionError,
> {
    let shape = union_shape(
        families
            .iter()
            .map(|family| (family.query_parts, family.table_parts)),
        prover_column_len,
    )?;
    let family_tags = family_tag_columns(families.len(), shape.logical_len);
    let zero = vec![Fr::zero(); shape.logical_len];
    let mut query_parts = Vec::new();
    let mut table_parts = Vec::new();
    for (family_index, family) in families.iter().enumerate() {
        append_prover_parts(
            &mut query_parts,
            family.query_parts,
            &family_tags[family_index],
            &zero,
            shape.max_width,
        );
        append_prover_parts(
            &mut table_parts,
            family.table_parts,
            &family_tags[family_index],
            &zero,
            shape.max_width,
        );
    }
    let union_label = union_label(label, &shape);
    let (proof, pending) = match grid {
        Some(grid) => prove_hidden_logup_parts_deferred_on_grid(
            &union_label,
            parameters,
            grid,
            &query_parts,
            &table_parts,
            rng,
        )?,
        None => prove_hidden_logup_parts_deferred(
            &union_label,
            parameters,
            &query_parts,
            &table_parts,
            rng,
        )?,
    };
    let pending = pending.rebind_parts_hidden_tables(
        prover_family_hidden_parts(families, HiddenLogUpSide::Query),
        prover_family_hidden_parts(families, HiddenLogUpSide::Table),
    )?;
    Ok((proof, pending))
}

/// Reduce a union built from temporary family containers, then rebind its
/// terminal claims to explicitly supplied long-lived hidden table views.
pub fn prove_hidden_logup_union_deferred_with_hidden_tables<'a>(
    label: &[u8],
    parameters: &DoryProverParameters,
    families: &[HiddenLogUpUnionProverFamily<'_>],
    query_hidden_parts: Vec<Vec<HiddenRelationTable<'a>>>,
    table_hidden_parts: Vec<Vec<HiddenRelationTable<'a>>>,
    rng: &mut impl RngCore,
) -> Result<
    (
        HiddenLogUpPartsReductionProof,
        HiddenLogUpProverOpeningState<'a>,
    ),
    HiddenLogUpUnionError,
> {
    let (proof, pending) = prove_hidden_logup_union_deferred(label, parameters, families, rng)?;
    Ok((
        proof,
        pending.rebind_parts_hidden_tables(query_hidden_parts, table_hidden_parts)?,
    ))
}

/// Grid-aware union reduction for temporary family containers. The returned
/// terminal state borrows only the explicitly supplied hidden table views.
pub fn prove_hidden_logup_union_deferred_on_grid_with_hidden_tables<'a>(
    label: &[u8],
    parameters: &DoryProverParameters,
    grid: &Section6DoryGrid,
    families: &[HiddenLogUpUnionProverFamily<'_>],
    query_hidden_parts: Vec<Vec<HiddenRelationTable<'a>>>,
    table_hidden_parts: Vec<Vec<HiddenRelationTable<'a>>>,
    rng: &mut impl RngCore,
) -> Result<
    (
        HiddenLogUpPartsReductionProof,
        HiddenLogUpProverOpeningState<'a>,
    ),
    HiddenLogUpUnionError,
> {
    let (proof, pending) =
        prove_hidden_logup_union_deferred_on_grid(label, parameters, grid, families, rng)?;
    Ok((
        proof,
        pending.rebind_parts_hidden_tables(query_hidden_parts, table_hidden_parts)?,
    ))
}

/// Verify all lookup families with one tagged multipart LogUp proof.
pub fn verify_hidden_logup_union(
    label: &[u8],
    parameters: &DoryVerifierParameters,
    families: &[HiddenLogUpUnionVerifierFamily<'_>],
    proof: &HiddenLogUpPartsProof,
) -> Result<(), HiddenLogUpUnionError> {
    let reduction = HiddenLogUpPartsReductionProof {
        auxiliary_commitments: proof.auxiliary_commitments.clone(),
        identity_reduction: proof.identity_reduction.clone(),
        sum_reduction: proof.sum_reduction.clone(),
    };
    let pending = verify_hidden_logup_union_deferred(label, parameters, families, &reduction)?;
    Ok(verify_hidden_logup_openings_batch(
        parameters,
        &[&pending],
        &proof.relation_opening,
    )?)
}

/// Verify all tagged family reductions and return terminal commitment claims.
pub fn verify_hidden_logup_union_deferred<'a>(
    label: &[u8],
    parameters: &DoryVerifierParameters,
    families: &[HiddenLogUpUnionVerifierFamily<'a>],
    proof: &'a HiddenLogUpPartsReductionProof,
) -> Result<HiddenLogUpVerifierOpeningState<'a>, HiddenLogUpUnionError> {
    verify_hidden_logup_union_deferred_with_grid(label, parameters, None, families, proof)
}

/// Verify all tagged lookup reductions whose terminal claims are opened on the
/// shared Section 6 Dory grid.
pub fn verify_hidden_logup_union_deferred_on_grid<'a>(
    label: &[u8],
    parameters: &DoryVerifierParameters,
    grid: &Section6DoryGrid,
    families: &[HiddenLogUpUnionVerifierFamily<'a>],
    proof: &'a HiddenLogUpPartsReductionProof,
) -> Result<HiddenLogUpVerifierOpeningState<'a>, HiddenLogUpUnionError> {
    verify_hidden_logup_union_deferred_with_grid(label, parameters, Some(grid), families, proof)
}

fn verify_hidden_logup_union_deferred_with_grid<'a>(
    label: &[u8],
    parameters: &DoryVerifierParameters,
    grid: Option<&Section6DoryGrid>,
    families: &[HiddenLogUpUnionVerifierFamily<'a>],
    proof: &'a HiddenLogUpPartsReductionProof,
) -> Result<HiddenLogUpVerifierOpeningState<'a>, HiddenLogUpUnionError> {
    let shape = union_shape(
        families
            .iter()
            .map(|family| (family.query_parts, family.table_parts)),
        verifier_column_len,
    )?;
    let family_tags = family_tag_columns(families.len(), shape.logical_len);
    let zero = vec![Fr::zero(); shape.logical_len];
    let mut query_parts = Vec::new();
    let mut table_parts = Vec::new();
    for (family_index, family) in families.iter().enumerate() {
        append_verifier_parts(
            &mut query_parts,
            family.query_parts,
            &family_tags[family_index],
            &zero,
            shape.max_width,
        );
        append_verifier_parts(
            &mut table_parts,
            family.table_parts,
            &family_tags[family_index],
            &zero,
            shape.max_width,
        );
    }
    let union_label = union_label(label, &shape);
    let pending = match grid {
        Some(grid) => verify_hidden_logup_parts_deferred_on_grid(
            &union_label,
            parameters,
            grid,
            &query_parts,
            &table_parts,
            proof,
        )?,
        None => verify_hidden_logup_parts_deferred(
            &union_label,
            parameters,
            &query_parts,
            &table_parts,
            proof,
        )?,
    };
    Ok(pending.rebind_parts_hidden_tables(
        proof,
        verifier_family_hidden_parts(families, HiddenLogUpSide::Query),
        verifier_family_hidden_parts(families, HiddenLogUpSide::Table),
    )?)
}

/// Verify a union built from temporary family containers, then rebind its
/// terminal claims to explicitly supplied long-lived hidden statements.
pub fn verify_hidden_logup_union_deferred_with_hidden_tables<'a>(
    label: &[u8],
    parameters: &DoryVerifierParameters,
    families: &[HiddenLogUpUnionVerifierFamily<'_>],
    proof: &'a HiddenLogUpPartsReductionProof,
    query_hidden_parts: Vec<Vec<HiddenRelationTableStatement<'a>>>,
    table_hidden_parts: Vec<Vec<HiddenRelationTableStatement<'a>>>,
) -> Result<HiddenLogUpVerifierOpeningState<'a>, HiddenLogUpUnionError> {
    let pending = verify_hidden_logup_union_deferred(label, parameters, families, proof)?;
    Ok(pending.rebind_parts_hidden_tables(proof, query_hidden_parts, table_hidden_parts)?)
}

/// Grid-aware union verification for temporary family containers. The
/// returned terminal state borrows only the supplied hidden statements.
pub fn verify_hidden_logup_union_deferred_on_grid_with_hidden_tables<'a>(
    label: &[u8],
    parameters: &DoryVerifierParameters,
    grid: &Section6DoryGrid,
    families: &[HiddenLogUpUnionVerifierFamily<'_>],
    proof: &'a HiddenLogUpPartsReductionProof,
    query_hidden_parts: Vec<Vec<HiddenRelationTableStatement<'a>>>,
    table_hidden_parts: Vec<Vec<HiddenRelationTableStatement<'a>>>,
) -> Result<HiddenLogUpVerifierOpeningState<'a>, HiddenLogUpUnionError> {
    let pending =
        verify_hidden_logup_union_deferred_on_grid(label, parameters, grid, families, proof)?;
    Ok(pending.rebind_parts_hidden_tables(proof, query_hidden_parts, table_hidden_parts)?)
}

struct UnionShape {
    logical_len: usize,
    max_width: usize,
    families: Vec<UnionFamilyShape>,
}

struct UnionFamilyShape {
    query_parts: usize,
    table_parts: usize,
    tuple_width: usize,
}

fn union_shape<'a, Column: 'a>(
    families: impl IntoIterator<Item = (&'a [Vec<Column>], &'a [Vec<Column>])>,
    column_len: impl Fn(&Column) -> usize,
) -> Result<UnionShape, HiddenLogUpUnionError> {
    let mut shapes = Vec::new();
    let mut logical_len = None;
    let mut max_width = 0;
    for (family, (query_parts, table_parts)) in families.into_iter().enumerate() {
        if query_parts.is_empty() {
            return Err(HiddenLogUpUnionError::EmptyFamily {
                family,
                side: HiddenLogUpSide::Query,
            });
        }
        if table_parts.is_empty() {
            return Err(HiddenLogUpUnionError::EmptyFamily {
                family,
                side: HiddenLogUpSide::Table,
            });
        }
        let tuple_width = query_parts[0].len();
        if tuple_width == 0 {
            return Err(HiddenLogUpUnionError::EmptyTuple { family });
        }
        for (side, parts) in [
            (HiddenLogUpSide::Query, query_parts),
            (HiddenLogUpSide::Table, table_parts),
        ] {
            for (part, columns) in parts.iter().enumerate() {
                if columns.len() != tuple_width {
                    return Err(HiddenLogUpUnionError::PartTupleWidth {
                        family,
                        side,
                        part,
                        expected: tuple_width,
                        actual: columns.len(),
                    });
                }
            }
        }
        logical_len.get_or_insert_with(|| column_len(&query_parts[0][0]));
        max_width = max_width.max(tuple_width);
        shapes.push(UnionFamilyShape {
            query_parts: query_parts.len(),
            table_parts: table_parts.len(),
            tuple_width,
        });
    }
    let logical_len = logical_len.ok_or(HiddenLogUpUnionError::EmptyFamilies)?;
    Ok(UnionShape {
        logical_len,
        max_width,
        families: shapes,
    })
}

fn prover_column_len(column: &MixedLogUpColumn<'_>) -> usize {
    match column {
        MixedLogUpColumn::Hidden(table) => table.committed.logical_len,
        MixedLogUpColumn::Public(table) => table.values.len(),
    }
}

fn verifier_column_len(column: &MixedLogUpColumnStatement<'_>) -> usize {
    match column {
        MixedLogUpColumnStatement::Hidden(table) => table.logical_len,
        MixedLogUpColumnStatement::Public(table) => table.values.len(),
    }
}

fn family_tag_columns(family_count: usize, logical_len: usize) -> Vec<Vec<Fr>> {
    (0..family_count)
        .map(|family| vec![Fr::from(family as u64 + 1); logical_len])
        .collect()
}

fn prover_family_hidden_parts<'a>(
    families: &[HiddenLogUpUnionProverFamily<'a>],
    side: HiddenLogUpSide,
) -> Vec<Vec<HiddenRelationTable<'a>>> {
    let mut hidden_parts = Vec::new();
    for family in families {
        let parts = match side {
            HiddenLogUpSide::Query => family.query_parts,
            HiddenLogUpSide::Table => family.table_parts,
        };
        for part in parts {
            hidden_parts.push(
                part.iter()
                    .filter_map(|column| match column {
                        MixedLogUpColumn::Hidden(table) => Some(*table),
                        MixedLogUpColumn::Public(_) => None,
                    })
                    .collect(),
            );
        }
    }
    hidden_parts
}

fn verifier_family_hidden_parts<'a>(
    families: &[HiddenLogUpUnionVerifierFamily<'a>],
    side: HiddenLogUpSide,
) -> Vec<Vec<HiddenRelationTableStatement<'a>>> {
    let mut hidden_parts = Vec::new();
    for family in families {
        let parts = match side {
            HiddenLogUpSide::Query => family.query_parts,
            HiddenLogUpSide::Table => family.table_parts,
        };
        for part in parts {
            hidden_parts.push(
                part.iter()
                    .filter_map(|column| match column {
                        MixedLogUpColumnStatement::Hidden(table) => Some(*table),
                        MixedLogUpColumnStatement::Public(_) => None,
                    })
                    .collect(),
            );
        }
    }
    hidden_parts
}

fn append_prover_parts<'source, 'borrow>(
    output: &mut Vec<Vec<MixedLogUpColumn<'borrow>>>,
    parts: &'borrow [Vec<MixedLogUpColumn<'source>>],
    family_tag: &'borrow [Fr],
    zero: &'borrow [Fr],
    max_width: usize,
) where
    'source: 'borrow,
{
    for part in parts {
        let mut normalized = Vec::with_capacity(max_width + 1);
        normalized.push(MixedLogUpColumn::Public(PublicRelationTable {
            values: family_tag,
        }));
        normalized.extend(part.iter().map(|column| match column {
            MixedLogUpColumn::Hidden(table) => MixedLogUpColumn::Hidden(HiddenRelationTable {
                values: table.values,
                committed: table.committed,
            }),
            MixedLogUpColumn::Public(table) => MixedLogUpColumn::Public(PublicRelationTable {
                values: table.values,
            }),
        }));
        normalized.extend(
            (part.len()..max_width)
                .map(|_| MixedLogUpColumn::Public(PublicRelationTable { values: zero })),
        );
        output.push(normalized);
    }
}

fn append_verifier_parts<'source, 'borrow>(
    output: &mut Vec<Vec<MixedLogUpColumnStatement<'borrow>>>,
    parts: &'borrow [Vec<MixedLogUpColumnStatement<'source>>],
    family_tag: &'borrow [Fr],
    zero: &'borrow [Fr],
    max_width: usize,
) where
    'source: 'borrow,
{
    for part in parts {
        let mut normalized = Vec::with_capacity(max_width + 1);
        normalized.push(MixedLogUpColumnStatement::Public(PublicRelationTable {
            values: family_tag,
        }));
        normalized.extend(part.iter().map(|column| match column {
            MixedLogUpColumnStatement::Hidden(table) => MixedLogUpColumnStatement::Hidden(*table),
            MixedLogUpColumnStatement::Public(table) => MixedLogUpColumnStatement::Public(*table),
        }));
        normalized.extend(
            (part.len()..max_width)
                .map(|_| MixedLogUpColumnStatement::Public(PublicRelationTable { values: zero })),
        );
        output.push(normalized);
    }
}

fn union_label(label: &[u8], shape: &UnionShape) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(UNION_DOMAIN);
    hasher.update((label.len() as u64).to_le_bytes());
    hasher.update(label);
    hasher.update((shape.logical_len as u64).to_le_bytes());
    hasher.update((shape.families.len() as u64).to_le_bytes());
    for (family, family_shape) in shape.families.iter().enumerate() {
        hasher.update((family as u64).to_le_bytes());
        hasher.update((family_shape.query_parts as u64).to_le_bytes());
        hasher.update((family_shape.table_parts as u64).to_le_bytes());
        hasher.update((family_shape.tuple_width as u64).to_le_bytes());
    }
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    use crate::{
        hidden_logup::{MixedLogUpColumn, MixedLogUpColumnStatement},
        hidden_relation::{HiddenRelationTable, HiddenRelationTableStatement, PublicRelationTable},
        jolt_dory::{commit_field_vector_zk, setup_dory_parameters},
        section6_final_opening::{
            resolve_section6_hidden_relation_prover_claim,
            resolve_section6_hidden_relation_verifier_claim, Section6DoryGrid,
        },
    };

    fn field_column(values: &[u64]) -> Vec<Fr> {
        values.iter().copied().map(Fr::from).collect()
    }

    fn prover_public_parts<'a>(columns: &'a [Vec<Fr>]) -> Vec<Vec<MixedLogUpColumn<'a>>> {
        vec![columns
            .iter()
            .map(|values| MixedLogUpColumn::Public(PublicRelationTable { values }))
            .collect()]
    }

    fn verifier_public_parts<'a>(
        columns: &'a [Vec<Fr>],
    ) -> Vec<Vec<MixedLogUpColumnStatement<'a>>> {
        vec![columns
            .iter()
            .map(|values| MixedLogUpColumnStatement::Public(PublicRelationTable { values }))
            .collect()]
    }

    fn prove_union_with_temporary_parts<'a>(
        parameters: &DoryProverParameters,
        query: HiddenRelationTable<'a>,
        table: HiddenRelationTable<'a>,
        rng: &mut impl RngCore,
    ) -> (
        HiddenLogUpPartsReductionProof,
        HiddenLogUpProverOpeningState<'a>,
    ) {
        let public = vec![Fr::from(5_u64); query.values.len()];
        let query_parts = vec![vec![
            MixedLogUpColumn::Public(PublicRelationTable { values: &public }),
            MixedLogUpColumn::Hidden(query),
        ]];
        let table_parts = vec![vec![
            MixedLogUpColumn::Public(PublicRelationTable { values: &public }),
            MixedLogUpColumn::Hidden(table),
        ]];
        let families = [HiddenLogUpUnionProverFamily {
            query_parts: &query_parts,
            table_parts: &table_parts,
        }];
        prove_hidden_logup_union_deferred_with_hidden_tables(
            b"temporary-union-parts",
            parameters,
            &families,
            vec![vec![query]],
            vec![vec![table]],
            rng,
        )
        .unwrap()
    }

    fn verify_union_with_temporary_parts<'a>(
        parameters: &DoryVerifierParameters,
        proof: &'a HiddenLogUpPartsReductionProof,
        query: HiddenRelationTableStatement<'a>,
        table: HiddenRelationTableStatement<'a>,
    ) -> HiddenLogUpVerifierOpeningState<'a> {
        let public = vec![Fr::from(5_u64); query.logical_len];
        let query_parts = vec![vec![
            MixedLogUpColumnStatement::Public(PublicRelationTable { values: &public }),
            MixedLogUpColumnStatement::Hidden(query),
        ]];
        let table_parts = vec![vec![
            MixedLogUpColumnStatement::Public(PublicRelationTable { values: &public }),
            MixedLogUpColumnStatement::Hidden(table),
        ]];
        let families = [HiddenLogUpUnionVerifierFamily {
            query_parts: &query_parts,
            table_parts: &table_parts,
        }];
        verify_hidden_logup_union_deferred_with_hidden_tables(
            b"temporary-union-parts",
            parameters,
            &families,
            proof,
            vec![vec![query]],
            vec![vec![table]],
        )
        .unwrap()
    }

    #[test]
    fn differently_shaped_families_share_one_order_bound_logup_proof() {
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(5);
        let narrow_query_values = vec![field_column(&[1, 2])];
        let narrow_table_values = vec![field_column(&[2, 1])];
        let wide_query_values = vec![field_column(&[3, 4]), field_column(&[5, 6])];
        let wide_table_values = vec![field_column(&[4, 3]), field_column(&[6, 5])];

        let narrow_query = prover_public_parts(&narrow_query_values);
        let narrow_table = prover_public_parts(&narrow_table_values);
        let wide_query = prover_public_parts(&wide_query_values);
        let wide_table = prover_public_parts(&wide_table_values);
        let prover_families = [
            HiddenLogUpUnionProverFamily {
                query_parts: &narrow_query,
                table_parts: &narrow_table,
            },
            HiddenLogUpUnionProverFamily {
                query_parts: &wide_query,
                table_parts: &wide_table,
            },
        ];
        let mut rng = StdRng::seed_from_u64(91);
        let proof: HiddenLogUpPartsProof = prove_hidden_logup_union(
            b"different-width-union",
            &prover_parameters,
            &prover_families,
            &mut rng,
        )
        .unwrap();

        let narrow_query_statement = verifier_public_parts(&narrow_query_values);
        let narrow_table_statement = verifier_public_parts(&narrow_table_values);
        let wide_query_statement = verifier_public_parts(&wide_query_values);
        let wide_table_statement = verifier_public_parts(&wide_table_values);
        let verifier_families = [
            HiddenLogUpUnionVerifierFamily {
                query_parts: &narrow_query_statement,
                table_parts: &narrow_table_statement,
            },
            HiddenLogUpUnionVerifierFamily {
                query_parts: &wide_query_statement,
                table_parts: &wide_table_statement,
            },
        ];
        verify_hidden_logup_union(
            b"different-width-union",
            &verifier_parameters,
            &verifier_families,
            &proof,
        )
        .unwrap();

        let (reduction, pending) = prove_hidden_logup_union_deferred(
            b"different-width-union-deferred",
            &prover_parameters,
            &prover_families,
            &mut rng,
        )
        .unwrap();
        let HiddenLogUpPartsReductionProof {
            auxiliary_commitments: _,
            identity_reduction: _,
            sum_reduction: _,
        } = reduction.clone();
        let opening =
            prove_hidden_logup_openings_batch(&prover_parameters, &[&pending], &mut rng).unwrap();
        let verifier_pending = verify_hidden_logup_union_deferred(
            b"different-width-union-deferred",
            &verifier_parameters,
            &verifier_families,
            &reduction,
        )
        .unwrap();
        verify_hidden_logup_openings_batch(&verifier_parameters, &[&verifier_pending], &opening)
            .unwrap();

        let swapped = [verifier_families[1], verifier_families[0]];
        assert!(verify_hidden_logup_union(
            b"different-width-union",
            &verifier_parameters,
            &swapped,
            &proof,
        )
        .is_err());

        assert!(verify_hidden_logup_union(
            b"different-width-union",
            &verifier_parameters,
            &verifier_families[..1],
            &proof,
        )
        .is_err());

        let zero = field_column(&[0, 0]);
        let widened_narrow_query_values = vec![narrow_query_values[0].clone(), zero.clone()];
        let widened_narrow_table_values = vec![narrow_table_values[0].clone(), zero];
        let widened_narrow_query = verifier_public_parts(&widened_narrow_query_values);
        let widened_narrow_table = verifier_public_parts(&widened_narrow_table_values);
        let width_tampered = [
            HiddenLogUpUnionVerifierFamily {
                query_parts: &widened_narrow_query,
                table_parts: &widened_narrow_table,
            },
            verifier_families[1],
        ];
        assert!(verify_hidden_logup_union(
            b"different-width-union",
            &verifier_parameters,
            &width_tampered,
            &proof,
        )
        .is_err());
    }

    #[test]
    fn deferred_union_auxiliaries_resolve_on_shared_section6_grid() {
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(5);
        let grid = Section6DoryGrid::new(5).unwrap();
        let query_values = vec![field_column(&[1, 2, 3, 4])];
        let table_values = vec![field_column(&[4, 3, 2, 1])];
        let query_parts = prover_public_parts(&query_values);
        let table_parts = prover_public_parts(&table_values);
        let prover_families = [HiddenLogUpUnionProverFamily {
            query_parts: &query_parts,
            table_parts: &table_parts,
        }];
        let mut rng = StdRng::seed_from_u64(97);
        let (reduction, pending) = prove_hidden_logup_union_deferred_on_grid(
            b"shared-grid-union",
            &prover_parameters,
            &grid,
            &prover_families,
            &mut rng,
        )
        .unwrap();

        let query_statements = verifier_public_parts(&query_values);
        let table_statements = verifier_public_parts(&table_values);
        let verifier_families = [HiddenLogUpUnionVerifierFamily {
            query_parts: &query_statements,
            table_parts: &table_statements,
        }];
        let verifier_pending = verify_hidden_logup_union_deferred_on_grid(
            b"shared-grid-union",
            &verifier_parameters,
            &grid,
            &verifier_families,
            &reduction,
        )
        .unwrap();

        let prover_openings = pending.relation_openings().unwrap();
        let verifier_openings = verifier_pending.relation_openings().unwrap();
        for (prover_opening, verifier_opening) in prover_openings.iter().zip(&verifier_openings) {
            for table in prover_opening.hidden_tables() {
                resolve_section6_hidden_relation_prover_claim(
                    *table,
                    prover_opening.row_point(),
                    &grid,
                )
                .unwrap();
            }
            for (table, evaluation_commitment) in verifier_opening
                .hidden_tables()
                .iter()
                .zip(verifier_opening.evaluation_commitments())
            {
                resolve_section6_hidden_relation_verifier_claim(
                    *table,
                    verifier_opening.row_point(),
                    *evaluation_commitment,
                    &grid,
                )
                .unwrap();
            }
        }
    }

    #[test]
    fn family_tags_reject_cross_family_cancellation() {
        let (prover_parameters, _) = setup_dory_parameters(5);
        let first_query_values = vec![field_column(&[1, 1])];
        let first_table_values = vec![field_column(&[2, 2])];
        let second_query_values = vec![field_column(&[2, 2])];
        let second_table_values = vec![field_column(&[1, 1])];

        let mut untagged_query = first_query_values[0]
            .iter()
            .chain(&second_query_values[0])
            .copied()
            .collect::<Vec<_>>();
        let mut untagged_table = first_table_values[0]
            .iter()
            .chain(&second_table_values[0])
            .copied()
            .collect::<Vec<_>>();
        untagged_query.sort_unstable();
        untagged_table.sort_unstable();
        assert_eq!(untagged_query, untagged_table);

        let first_query = prover_public_parts(&first_query_values);
        let first_table = prover_public_parts(&first_table_values);
        let second_query = prover_public_parts(&second_query_values);
        let second_table = prover_public_parts(&second_table_values);
        let families = [
            HiddenLogUpUnionProverFamily {
                query_parts: &first_query,
                table_parts: &first_table,
            },
            HiddenLogUpUnionProverFamily {
                query_parts: &second_query,
                table_parts: &second_table,
            },
        ];
        let mut rng = StdRng::seed_from_u64(93);
        assert!(matches!(
            prove_hidden_logup_union(
                b"cross-family-cancellation",
                &prover_parameters,
                &families,
                &mut rng,
            ),
            Err(HiddenLogUpUnionError::LogUp(
                HiddenLogUpError::MissingQueryTuple { .. }
            ))
        ));
    }

    #[test]
    fn deferred_union_outlives_temporary_public_and_part_containers() {
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(5);
        let table_values = field_column(&[10, 11]);
        let query_values = field_column(&[11, 10]);
        let committed_table = commit_field_vector_zk(&prover_parameters, &table_values).unwrap();
        let committed_query = commit_field_vector_zk(&prover_parameters, &query_values).unwrap();
        let query = HiddenRelationTable {
            values: &query_values,
            committed: &committed_query,
        };
        let table = HiddenRelationTable {
            values: &table_values,
            committed: &committed_table,
        };
        let mut rng = StdRng::seed_from_u64(95);
        let (proof, pending) =
            prove_union_with_temporary_parts(&prover_parameters, query, table, &mut rng);
        let opening =
            prove_hidden_logup_openings_batch(&prover_parameters, &[&pending], &mut rng).unwrap();
        let verifier_pending = verify_union_with_temporary_parts(
            &verifier_parameters,
            &proof,
            HiddenRelationTableStatement::from(&committed_query),
            HiddenRelationTableStatement::from(&committed_table),
        );
        verify_hidden_logup_openings_batch(&verifier_parameters, &[&verifier_pending], &opening)
            .unwrap();
    }
}
