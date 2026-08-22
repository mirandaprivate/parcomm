//! Collection adapters for the one terminal Section 6 Dory opening.

use ark_bn254::Fr as ArkFr;

use crate::{
    hidden_relation::{
        HiddenRelationProverOpening, HiddenRelationTable, HiddenRelationTableStatement,
        HiddenRelationVerifierOpening, HiddenScalarWitness,
    },
    jolt_dory::CommittedDoryFieldVector,
    section6_final_opening::{
        resolve_section6_hidden_relation_prover_claim,
        resolve_section6_hidden_relation_verifier_claim, ResolvedSection6FieldProverClaim,
        ResolvedSection6FieldVerifierClaim, Section6DoryGrid, Section6FinalOpeningError,
        Section6FinalOpeningProverClaim, Section6FinalOpeningVerifierClaim,
    },
};

/// Prover-side terminal claims after all algebraic reductions have completed.
pub struct Section6ProverClaimCollection<'a> {
    resolved: Vec<ResolvedSection6FieldProverClaim<'a>>,
    evaluations: Vec<HiddenScalarWitness>,
}

impl<'a> Section6ProverClaimCollection<'a> {
    pub fn new() -> Self {
        Self {
            resolved: Vec::new(),
            evaluations: Vec::new(),
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
        for (((table, &value), &commitment), &blind) in tables
            .iter()
            .copied()
            .zip(opening.evaluations())
            .zip(opening.evaluation_commitments())
            .zip(opening.evaluation_blinds())
        {
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
        self.evaluations.push(evaluation);
        Ok(())
    }

    pub fn claims(&self) -> Vec<Section6FinalOpeningProverClaim<'_>> {
        self.resolved
            .iter()
            .zip(&self.evaluations)
            .map(|(resolved, evaluation)| resolved.as_claim(evaluation))
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
}

impl<'a> Section6VerifierClaimCollection<'a> {
    pub fn new() -> Self {
        Self {
            resolved: Vec::new(),
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
        for (table, &evaluation_commitment) in
            tables.iter().copied().zip(opening.evaluation_commitments())
        {
            self.resolved
                .push(resolve_section6_hidden_relation_verifier_claim(
                    table,
                    opening.row_point(),
                    evaluation_commitment,
                    grid,
                )?);
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
        Ok(())
    }

    pub fn claims(&self) -> Vec<Section6FinalOpeningVerifierClaim<'_>> {
        self.resolved
            .iter()
            .map(|resolved| resolved.as_claim())
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
