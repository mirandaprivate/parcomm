use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ff::UniformRand;
use ark_poly_commit::smart_pc::{data_structures::Trans as PcsTrans, SmartPC};
use ark_std::marker::PhantomData;
use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};

use crate::data_structures::SRS;
use crate::protocols::{
    diagonal::ZkDiagonal,
    mul::ZkMatMul,
    virtualpcs::{commit_stacked_i32, open_stacked_i32, VirtualPcsError},
};
use crate::utils::{
    fiat_shamir::FiatShamir,
    heterogeneous::{
        DiagonalHeteroView, HeteroBlockMat, HeteroMatrixError, MatrixBlock, MatrixDims,
        PaddedDomain, StackedHeteroView,
    },
    linear,
    zktr::{TranSeq, ZkTranSeq},
};

/// Public dimensions of one independent matrix-multiplication gate.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct MulGateDims {
    pub left: MatrixDims,
    pub right: MatrixDims,
    pub output: MatrixDims,
}

/// Public shape metadata for a heterogeneous batch.
///
/// The outer domains describe the caller's sparse storage layout. The proof
/// protocol deterministically re-buckets populated gates by their exact
/// `(m, k, n)` dimensions, so it never materializes these monolithic domains.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HeteroMulShapeSpec {
    pub left_domain: PaddedDomain,
    pub right_domain: PaddedDomain,
    pub output_domain: PaddedDomain,
    pub gates: Vec<Option<MulGateDims>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MulBucketSpec {
    gate: MulGateDims,
    slots: usize,
}

impl MulBucketSpec {
    fn domains(self) -> Result<(PaddedDomain, PaddedDomain, PaddedDomain), HeteroMulError> {
        Ok((
            PaddedDomain::new(self.slots, self.gate.left.rows, self.gate.left.cols)?,
            PaddedDomain::new(self.slots, self.gate.right.rows, self.gate.right.cols)?,
            PaddedDomain::new(self.slots, self.gate.output.rows, self.gate.output.cols)?,
        ))
    }

    fn padded_entries(self) -> usize {
        self.slots
            * (self.gate.left.entries() + self.gate.right.entries() + self.gate.output.entries())
    }
}

impl HeteroMulShapeSpec {
    fn validate(&self) -> Result<(), HeteroMulError> {
        PaddedDomain::new(
            self.left_domain.slots,
            self.left_domain.row_cap,
            self.left_domain.col_cap,
        )?;
        PaddedDomain::new(
            self.right_domain.slots,
            self.right_domain.row_cap,
            self.right_domain.col_cap,
        )?;
        PaddedDomain::new(
            self.output_domain.slots,
            self.output_domain.row_cap,
            self.output_domain.col_cap,
        )?;

        let slots = self.left_domain.slots;
        if self.right_domain.slots != slots
            || self.output_domain.slots != slots
            || self.gates.len() != slots
        {
            return Err(HeteroMulError::SlotCountMismatch);
        }
        if self.left_domain.row_cap != self.output_domain.row_cap
            || self.left_domain.col_cap != self.right_domain.row_cap
            || self.right_domain.col_cap != self.output_domain.col_cap
        {
            return Err(HeteroMulError::PaddedDomainMismatch);
        }

        let mut populated = 0_usize;
        for (slot, gate) in self.gates.iter().enumerate() {
            let Some(gate) = gate else { continue };
            populated += 1;
            validate_gate_dims(slot, *gate)?;
            for (name, dims, domain) in [
                ("left", gate.left, self.left_domain),
                ("right", gate.right, self.right_domain),
                ("output", gate.output, self.output_domain),
            ] {
                if dims.rows > domain.row_cap || dims.cols > domain.col_cap {
                    return Err(HeteroMulError::ShapeExceedsDomain {
                        slot,
                        matrix: name,
                        dims,
                        domain,
                    });
                }
            }
        }
        if populated == 0 {
            return Err(HeteroMulError::NoGates);
        }
        Ok(())
    }

    fn bucket_specs(&self) -> Result<Vec<MulBucketSpec>, HeteroMulError> {
        self.validate()?;
        let mut counts = BTreeMap::<MulGateDims, usize>::new();
        for gate in self.gates.iter().flatten() {
            *counts.entry(*gate).or_default() += 1;
        }
        Ok(counts
            .into_iter()
            .map(|(gate, count)| MulBucketSpec {
                gate,
                slots: count.next_power_of_two(),
            })
            .collect())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HeteroMulWitness {
    pub left: HeteroBlockMat<i32>,
    pub right: HeteroBlockMat<i32>,
    pub output: HeteroBlockMat<i32>,
    pub shape: HeteroMulShapeSpec,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct MulBucketWitness {
    spec: MulBucketSpec,
    left: HeteroBlockMat<i32>,
    right: HeteroBlockMat<i32>,
    output: HeteroBlockMat<i32>,
}

/// Logical-size counters for capacity planning. These are element counts, not
/// allocator byte measurements: field and group representations are
/// backend-dependent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeteroRamProfile {
    /// Values actually retained by the heterogeneous witness.
    pub stored_witness_entries: usize,
    /// Values retained by one shared-max-dimension, monolithic layout.
    pub monolithic_padded_entries: usize,
    /// Values in all exact-shape buckets, including at most one power-of-two
    /// slot expansion per distinct shape.
    pub bucket_padded_entries: usize,
    /// Largest field-vector projection created while proving one bucket.
    pub max_projection_entries: usize,
    /// Largest row/column axis retained by one virtual SMART-PC opening.
    pub max_pcs_axis: usize,
}

impl HeteroMulWitness {
    pub fn new(
        left: HeteroBlockMat<i32>,
        right: HeteroBlockMat<i32>,
        output: HeteroBlockMat<i32>,
    ) -> Result<Self, HeteroMulError> {
        let slots = left.domain.slots;
        if right.domain.slots != slots || output.domain.slots != slots {
            return Err(HeteroMulError::SlotCountMismatch);
        }

        let mut gates = Vec::with_capacity(slots);
        for slot in 0..slots {
            match (
                left.blocks[slot].as_ref(),
                right.blocks[slot].as_ref(),
                output.blocks[slot].as_ref(),
            ) {
                (None, None, None) => gates.push(None),
                (Some(left), Some(right), Some(output)) => {
                    let gate = MulGateDims {
                        left: left.dims,
                        right: right.dims,
                        output: output.dims,
                    };
                    validate_gate_dims(slot, gate)?;
                    gates.push(Some(gate));
                }
                _ => return Err(HeteroMulError::IncompleteGate { slot }),
            }
        }

        let shape = HeteroMulShapeSpec {
            left_domain: left.domain,
            right_domain: right.domain,
            output_domain: output.domain,
            gates,
        };
        let witness = Self {
            left,
            right,
            output,
            shape,
        };
        witness.validate()?;
        Ok(witness)
    }

    fn validate(&self) -> Result<(), HeteroMulError> {
        self.shape.validate()?;
        if self.left.domain != self.shape.left_domain
            || self.right.domain != self.shape.right_domain
            || self.output.domain != self.shape.output_domain
        {
            return Err(HeteroMulError::WitnessDomainMismatch);
        }
        let slots = self.shape.left_domain.slots;
        if self.left.blocks.len() != slots
            || self.right.blocks.len() != slots
            || self.output.blocks.len() != slots
        {
            return Err(HeteroMulError::SlotCountMismatch);
        }

        for slot in 0..slots {
            let blocks = [
                ("left", self.left.blocks[slot].as_ref()),
                ("right", self.right.blocks[slot].as_ref()),
                ("output", self.output.blocks[slot].as_ref()),
            ];
            match self.shape.gates[slot] {
                None => {
                    if blocks.iter().any(|(_, block)| block.is_some()) {
                        return Err(HeteroMulError::WitnessShapeMismatch {
                            slot,
                            matrix: "dummy",
                        });
                    }
                }
                Some(gate) => {
                    for ((name, block), expected) in blocks
                        .iter()
                        .zip([gate.left, gate.right, gate.output].iter())
                    {
                        let block = block.ok_or(HeteroMulError::IncompleteGate { slot })?;
                        validate_block_layout(block)?;
                        if block.dims != *expected {
                            return Err(HeteroMulError::WitnessShapeMismatch {
                                slot,
                                matrix: name,
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn buckets(&self) -> Result<Vec<MulBucketWitness>, HeteroMulError> {
        self.validate()?;
        type BlockGroups = (
            Vec<Option<MatrixBlock<i32>>>,
            Vec<Option<MatrixBlock<i32>>>,
            Vec<Option<MatrixBlock<i32>>>,
        );
        let mut groups = BTreeMap::<MulGateDims, BlockGroups>::new();
        for (slot, gate) in self.shape.gates.iter().enumerate() {
            let Some(gate) = gate else { continue };
            let group = groups
                .entry(*gate)
                .or_insert_with(|| (Vec::new(), Vec::new(), Vec::new()));
            group
                .0
                .push(Some(self.left.blocks[slot].as_ref().unwrap().clone()));
            group
                .1
                .push(Some(self.right.blocks[slot].as_ref().unwrap().clone()));
            group
                .2
                .push(Some(self.output.blocks[slot].as_ref().unwrap().clone()));
        }

        groups
            .into_iter()
            .map(|(gate, (left, right, output))| {
                let spec = MulBucketSpec {
                    gate,
                    slots: left.len().next_power_of_two(),
                };
                let (left_domain, right_domain, output_domain) = spec.domains()?;
                Ok(MulBucketWitness {
                    spec,
                    left: HeteroBlockMat::new(left_domain, left)?,
                    right: HeteroBlockMat::new(right_domain, right)?,
                    output: HeteroBlockMat::new(output_domain, output)?,
                })
            })
            .collect()
    }

    fn into_buckets(self) -> Result<Vec<MulBucketWitness>, HeteroMulError> {
        self.validate()?;
        type BlockGroups = (
            Vec<Option<MatrixBlock<i32>>>,
            Vec<Option<MatrixBlock<i32>>>,
            Vec<Option<MatrixBlock<i32>>>,
        );
        let HeteroMulWitness {
            left,
            right,
            output,
            shape,
        } = self;
        let mut groups = BTreeMap::<MulGateDims, BlockGroups>::new();
        for (((gate, left), right), output) in shape
            .gates
            .iter()
            .copied()
            .zip(left.blocks.into_iter())
            .zip(right.blocks.into_iter())
            .zip(output.blocks.into_iter())
        {
            let Some(gate) = gate else { continue };
            let group = groups
                .entry(gate)
                .or_insert_with(|| (Vec::new(), Vec::new(), Vec::new()));
            group.0.push(left);
            group.1.push(right);
            group.2.push(output);
        }

        groups
            .into_iter()
            .map(|(gate, (left, right, output))| {
                let spec = MulBucketSpec {
                    gate,
                    slots: left.len().next_power_of_two(),
                };
                let (left_domain, right_domain, output_domain) = spec.domains()?;
                Ok(MulBucketWitness {
                    spec,
                    left: HeteroBlockMat::new(left_domain, left)?,
                    right: HeteroBlockMat::new(right_domain, right)?,
                    output: HeteroBlockMat::new(output_domain, output)?,
                })
            })
            .collect()
    }

    pub fn stored_entries(&self) -> usize {
        self.left.stored_entries() + self.right.stored_entries() + self.output.stored_entries()
    }

    pub fn logical_entries(&self) -> usize {
        [self.left.domain, self.right.domain, self.output.domain]
            .iter()
            .map(|domain| domain.slots * domain.row_cap * domain.col_cap)
            .sum()
    }

    pub fn ram_profile(&self) -> HeteroRamProfile {
        let specs = self.shape.bucket_specs().unwrap_or_default();
        let bucket_padded_entries = specs.iter().map(|spec| spec.padded_entries()).sum();
        let max_projection_entries = specs
            .iter()
            .map(|spec| spec.slots * spec.gate.left.cols)
            .max()
            .unwrap_or(0);
        let max_pcs_axis = specs
            .iter()
            .flat_map(|spec| spec.domains().ok())
            .flat_map(|domains| [domains.0, domains.1, domains.2])
            .flat_map(|domain| {
                let (rows, cols) = packed_stacked_shape(domain);
                [rows, cols]
            })
            .max()
            .unwrap_or(0);

        HeteroRamProfile {
            stored_witness_entries: self.stored_entries(),
            monolithic_padded_entries: self.logical_entries(),
            bucket_padded_entries,
            max_projection_entries,
            max_pcs_axis,
        }
    }
}

pub struct HeteroMulBucketCommitments<E: Pairing> {
    pub left: PairingOutput<E>,
    pub right: PairingOutput<E>,
    pub output: PairingOutput<E>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeteroMulBucketHidings<F> {
    pub left: F,
    pub right: F,
    pub output: F,
}

pub struct HeteroMulInstance<E: Pairing> {
    pub shape: HeteroMulShapeSpec,
    /// Ordered by `MulGateDims`; the order is derived from `shape`.
    pub buckets: Vec<HeteroMulBucketCommitments<E>>,
}

pub struct HeteroMulBucketProof<E: Pairing> {
    pub output_value_commitment: E::G1,
    pub reduction_transcript: TranSeq<E::ScalarField, E::G1>,
    pub left_opening: PcsTrans<E>,
    pub right_opening: PcsTrans<E>,
    pub output_opening: PcsTrans<E>,
}

pub struct HeteroMulProof<E: Pairing> {
    /// One proof per exact-shape bucket, in the same order as the instance.
    pub buckets: Vec<HeteroMulBucketProof<E>>,
}

pub struct HeteroMulProtocol<E: Pairing> {
    _marker: PhantomData<E>,
}

impl<E: Pairing> HeteroMulProtocol<E> {
    pub fn setup(witness: &HeteroMulWitness) -> Result<SRS<E>, HeteroMulError> {
        witness.validate()?;
        Ok(SRS::<E>::setup(required_log_q(&witness.shape)?))
    }

    pub fn prove(
        srs: &SRS<E>,
        witness: &HeteroMulWitness,
        bit_width: usize,
    ) -> Result<(HeteroMulInstance<E>, HeteroMulProof<E>), HeteroMulError> {
        witness.validate()?;
        Self::prove_buckets(
            srs,
            witness.shape.clone(),
            witness.buckets()?,
            bit_width,
            None,
        )
    }

    /// Memory-oriented proving entry point. It consumes the witness and moves
    /// its blocks into exact-shape buckets instead of cloning every matrix.
    pub fn prove_owned(
        srs: &SRS<E>,
        witness: HeteroMulWitness,
        bit_width: usize,
    ) -> Result<(HeteroMulInstance<E>, HeteroMulProof<E>), HeteroMulError> {
        witness.validate()?;
        let shape = witness.shape.clone();
        let buckets = witness.into_buckets()?;
        Self::prove_buckets(srs, shape, buckets, bit_width, None)
    }

    pub fn prove_owned_with_hidings(
        srs: &SRS<E>,
        witness: HeteroMulWitness,
        bit_width: usize,
        bucket_hidings: Vec<HeteroMulBucketHidings<E::ScalarField>>,
    ) -> Result<(HeteroMulInstance<E>, HeteroMulProof<E>), HeteroMulError> {
        witness.validate()?;
        let shape = witness.shape.clone();
        let buckets = witness.into_buckets()?;
        if buckets.len() != bucket_hidings.len() {
            return Err(HeteroMulError::BucketHidingCountMismatch {
                expected: buckets.len(),
                actual: bucket_hidings.len(),
            });
        }
        Self::prove_buckets(srs, shape, buckets, bit_width, Some(bucket_hidings))
    }

    fn prove_buckets(
        srs: &SRS<E>,
        shape: HeteroMulShapeSpec,
        buckets: Vec<MulBucketWitness>,
        bit_width: usize,
        bucket_hidings: Option<Vec<HeteroMulBucketHidings<E::ScalarField>>>,
    ) -> Result<(HeteroMulInstance<E>, HeteroMulProof<E>), HeteroMulError> {
        ensure_srs_capacity(srs, &shape)?;
        let rng = &mut ark_std::rand::thread_rng();
        let mut instance_buckets = Vec::with_capacity(buckets.len());
        let mut proof_buckets = Vec::with_capacity(buckets.len());

        // Each bucket is committed, reduced, and opened before moving to the
        // next one. Temporary PCS bases therefore scale with the largest
        // bucket, not the sum of all heterogeneous domains.
        for (bucket_index, bucket) in buckets.iter().enumerate() {
            let hidings = bucket_hidings
                .as_ref()
                .map(|hidings| hidings[bucket_index])
                .unwrap_or_else(|| HeteroMulBucketHidings {
                    left: E::ScalarField::rand(rng),
                    right: E::ScalarField::rand(rng),
                    output: E::ScalarField::rand(rng),
                });
            let left_hiding = hidings.left;
            let right_hiding = hidings.right;
            let output_hiding = hidings.output;
            let (left_commitment, left_tier_one) =
                commit_stacked_i32::<E>(&srs.pc_pp, &bucket.left, left_hiding, bit_width)?;
            let (right_commitment, right_tier_one) =
                commit_stacked_i32::<E>(&srs.pc_pp, &bucket.right, right_hiding, bit_width)?;
            let (output_commitment, output_tier_one) =
                commit_stacked_i32::<E>(&srs.pc_pp, &bucket.output, output_hiding, bit_width)?;
            let commitments = HeteroMulBucketCommitments {
                left: left_commitment,
                right: right_commitment,
                output: output_commitment,
            };
            let output_point =
                derive_bucket_output_point::<E>(&shape, bucket.spec, bucket_index, &commitments);
            let output_value = bucket
                .output
                .proj_stacked_lr(&output_point.0, &output_point.1)?;
            let output_value_hiding = E::ScalarField::rand(rng);
            let output_value_commitment = srs.zksrs.commit(output_value, output_value_hiding);

            let left_diagonal = DiagonalHeteroView::new(&bucket.left);
            let right_stacked = StackedHeteroView::new(&bucket.right);
            let mut reduction_transcript = ZkTranSeq::new(&srs.zksrs);
            let matmul = ZkMatMul::<
                i32,
                E::ScalarField,
                E::G1,
                DiagonalHeteroView<'_, i32>,
                StackedHeteroView<'_, i32>,
            >::new(
                output_value_commitment,
                output_point.clone(),
                bucket.spec.slots * bucket.spec.gate.left.cols,
            );
            let (
                left_diagonal_value_commitment,
                right_value_commitment,
                left_value_hiding,
                right_value_hiding,
                left_diagonal_point,
                right_point,
            ) = matmul.reduce_prover(
                &srs.zksrs,
                &mut reduction_transcript,
                &left_diagonal,
                &right_stacked,
                output_value_hiding,
            );

            let left_stacked = StackedHeteroView::new(&bucket.left);
            let diagonal =
                ZkDiagonal::<i32, E::ScalarField, E::G1, StackedHeteroView<'_, i32>>::new(
                    left_diagonal_value_commitment,
                    left_diagonal_point,
                    bucket.spec.slots,
                );
            let (left_value_commitment, left_value_hiding, left_point) = diagonal.reduce_prover(
                &srs.zksrs,
                &mut reduction_transcript,
                &left_stacked,
                left_value_hiding,
            );

            let left_packed_point = linear::reshape_points_keep_projection(&left_point);
            let right_packed_point = linear::reshape_points_keep_projection(&right_point);
            let output_packed_point = linear::reshape_points_keep_projection(&output_point);
            let left_opening = open_stacked_i32::<E>(
                &srs.pc_pp,
                &bucket.left,
                &left_packed_point.0,
                &left_packed_point.1,
                E::pairing(left_value_commitment, srs.h_hat),
                left_commitment,
                &left_tier_one,
                left_value_hiding,
                left_hiding,
            )?;
            let right_opening = open_stacked_i32::<E>(
                &srs.pc_pp,
                &bucket.right,
                &right_packed_point.0,
                &right_packed_point.1,
                E::pairing(right_value_commitment, srs.h_hat),
                right_commitment,
                &right_tier_one,
                right_value_hiding,
                right_hiding,
            )?;
            let output_opening = open_stacked_i32::<E>(
                &srs.pc_pp,
                &bucket.output,
                &output_packed_point.0,
                &output_packed_point.1,
                E::pairing(output_value_commitment, srs.h_hat),
                output_commitment,
                &output_tier_one,
                output_value_hiding,
                output_hiding,
            )?;

            instance_buckets.push(commitments);
            proof_buckets.push(HeteroMulBucketProof {
                output_value_commitment,
                reduction_transcript: reduction_transcript.publish_trans(),
                left_opening,
                right_opening,
                output_opening,
            });
        }

        Ok((
            HeteroMulInstance {
                shape,
                buckets: instance_buckets,
            },
            HeteroMulProof {
                buckets: proof_buckets,
            },
        ))
    }

    pub fn verify(
        srs: &SRS<E>,
        instance: &HeteroMulInstance<E>,
        proof: &HeteroMulProof<E>,
    ) -> bool {
        let specs = match instance.shape.bucket_specs() {
            Ok(specs) => specs,
            Err(_) => return false,
        };
        if ensure_srs_capacity(srs, &instance.shape).is_err()
            || instance.buckets.len() != specs.len()
            || proof.buckets.len() != specs.len()
        {
            return false;
        }

        for (bucket_index, ((spec, commitments), bucket_proof)) in specs
            .iter()
            .zip(instance.buckets.iter())
            .zip(proof.buckets.iter())
            .enumerate()
        {
            let output_point =
                derive_bucket_output_point::<E>(&instance.shape, *spec, bucket_index, commitments);
            let mut transcript = bucket_proof.reduction_transcript.clone();
            let matmul = ZkMatMul::<
                i32,
                E::ScalarField,
                E::G1,
                DiagonalHeteroView<'static, i32>,
                StackedHeteroView<'static, i32>,
            >::new(
                bucket_proof.output_value_commitment,
                output_point.clone(),
                spec.slots * spec.gate.left.cols,
            );
            let (
                multiplication_ok,
                left_diagonal_value_commitment,
                right_value_commitment,
                left_diagonal_point,
                right_point,
            ) = matmul.verify_as_subprotocol(&srs.zksrs, &mut transcript);
            if !multiplication_ok {
                return false;
            }

            let diagonal =
                ZkDiagonal::<i32, E::ScalarField, E::G1, StackedHeteroView<'static, i32>>::new(
                    left_diagonal_value_commitment,
                    left_diagonal_point,
                    spec.slots,
                );
            let (diagonal_ok, left_value_commitment, left_point) =
                diagonal.verify_as_subprotocol(&srs.zksrs, &mut transcript);
            if !diagonal_ok || !transcript.check_fiat_shamir() || !transcript.is_fully_consumed() {
                return false;
            }

            let left_point = linear::reshape_points_keep_projection(&left_point);
            let right_point = linear::reshape_points_keep_projection(&right_point);
            let output_point = linear::reshape_points_keep_projection(&output_point);
            if !pcs_proof_matches_point(&bucket_proof.left_opening, &left_point)
                || !pcs_proof_matches_point(&bucket_proof.right_opening, &right_point)
                || !pcs_proof_matches_point(&bucket_proof.output_opening, &output_point)
            {
                return false;
            }
            let left_opening_ok = SmartPC::<E>::verify(
                &srs.pc_pp,
                commitments.left,
                E::pairing(left_value_commitment, srs.h_hat),
                &left_point.0,
                &left_point.1,
                &bucket_proof.left_opening,
            )
            .unwrap_or(false);
            let right_opening_ok = SmartPC::<E>::verify(
                &srs.pc_pp,
                commitments.right,
                E::pairing(right_value_commitment, srs.h_hat),
                &right_point.0,
                &right_point.1,
                &bucket_proof.right_opening,
            )
            .unwrap_or(false);
            let output_opening_ok = SmartPC::<E>::verify(
                &srs.pc_pp,
                commitments.output,
                E::pairing(bucket_proof.output_value_commitment, srs.h_hat),
                &output_point.0,
                &output_point.1,
                &bucket_proof.output_opening,
            )
            .unwrap_or(false);
            if !left_opening_ok || !right_opening_ok || !output_opening_ok {
                return false;
            }
        }
        true
    }
}

fn pcs_proof_matches_point<E: Pairing>(
    proof: &PcsTrans<E>,
    point: &(Vec<E::ScalarField>, Vec<E::ScalarField>),
) -> bool {
    proof.has_rounds(point.0.len(), point.1.len())
}

fn validate_block_layout(block: &MatrixBlock<i32>) -> Result<(), HeteroMulError> {
    block.validate_layout()?;
    Ok(())
}

fn validate_gate_dims(slot: usize, gate: MulGateDims) -> Result<(), HeteroMulError> {
    if gate.left.cols != gate.right.rows
        || gate.output.rows != gate.left.rows
        || gate.output.cols != gate.right.cols
    {
        return Err(HeteroMulError::GateShapeMismatch { slot, gate });
    }
    for (matrix, dims) in [
        ("left", gate.left),
        ("right", gate.right),
        ("output", gate.output),
    ] {
        if !dims.rows.is_power_of_two() || !dims.cols.is_power_of_two() {
            return Err(HeteroMulError::GateDimensionNotPowerOfTwo { slot, matrix, dims });
        }
    }
    Ok(())
}

fn packed_stacked_shape(domain: PaddedDomain) -> (usize, usize) {
    let entries = domain.slots * domain.row_cap * domain.col_cap;
    let packed_cols = 1 << (entries.ilog2() as usize / 2);
    (entries / packed_cols, packed_cols)
}

fn packed_max_dimension(domain: PaddedDomain) -> usize {
    let (rows, cols) = packed_stacked_shape(domain);
    rows.max(cols)
}

fn required_log_q(shape: &HeteroMulShapeSpec) -> Result<usize, HeteroMulError> {
    let mut maximum = 2_usize;
    for spec in shape.bucket_specs()? {
        let domains = spec.domains()?;
        for domain in [domains.0, domains.1, domains.2] {
            maximum = maximum.max(packed_max_dimension(domain));
        }
    }
    Ok(maximum.ilog2() as usize)
}

fn ensure_srs_capacity<E: Pairing>(
    srs: &SRS<E>,
    shape: &HeteroMulShapeSpec,
) -> Result<(), HeteroMulError> {
    let required = 1 << required_log_q(shape)?;
    if srs.pc_pp.q < required {
        return Err(HeteroMulError::SrsTooSmall {
            required,
            actual: srs.pc_pp.q,
        });
    }
    Ok(())
}

fn push_shape_into_fs<E: Pairing>(fs: &mut FiatShamir, shape: &HeteroMulShapeSpec) {
    for domain in [shape.left_domain, shape.right_domain, shape.output_domain] {
        fs.push(&domain.slots);
        fs.push(&domain.row_cap);
        fs.push(&domain.col_cap);
    }
    for gate in &shape.gates {
        match gate {
            None => fs.push(&0_usize),
            Some(gate) => {
                fs.push(&1_usize);
                for dims in [gate.left, gate.right, gate.output] {
                    fs.push(&dims.rows);
                    fs.push(&dims.cols);
                }
            }
        }
    }
}

fn derive_bucket_output_point<E: Pairing>(
    shape: &HeteroMulShapeSpec,
    spec: MulBucketSpec,
    bucket_index: usize,
    commitments: &HeteroMulBucketCommitments<E>,
) -> (Vec<E::ScalarField>, Vec<E::ScalarField>) {
    let mut fs = FiatShamir::new();
    fs.push(&2_usize);
    push_shape_into_fs::<E>(&mut fs, shape);
    fs.push(&bucket_index);
    fs.push(&spec.slots);
    for dims in [spec.gate.left, spec.gate.right, spec.gate.output] {
        fs.push(&dims.rows);
        fs.push(&dims.cols);
    }
    fs.push(&commitments.left);
    fs.push(&commitments.right);
    fs.push(&commitments.output);

    let rows = spec.slots * spec.gate.output.rows;
    let cols = spec.gate.output.cols;
    let left = (0..rows.ilog2())
        .map(|_| fs.gen_challenge::<E::ScalarField>())
        .collect();
    let right = (0..cols.ilog2())
        .map(|_| fs.gen_challenge::<E::ScalarField>())
        .collect();
    (left, right)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HeteroMulError {
    Matrix(HeteroMatrixError),
    Pcs(VirtualPcsError),
    NoGates,
    SlotCountMismatch,
    PaddedDomainMismatch,
    WitnessDomainMismatch,
    IncompleteGate {
        slot: usize,
    },
    WitnessShapeMismatch {
        slot: usize,
        matrix: &'static str,
    },
    GateShapeMismatch {
        slot: usize,
        gate: MulGateDims,
    },
    GateDimensionNotPowerOfTwo {
        slot: usize,
        matrix: &'static str,
        dims: MatrixDims,
    },
    ShapeExceedsDomain {
        slot: usize,
        matrix: &'static str,
        dims: MatrixDims,
        domain: PaddedDomain,
    },
    SrsTooSmall {
        required: usize,
        actual: usize,
    },
    BucketHidingCountMismatch {
        expected: usize,
        actual: usize,
    },
}

impl From<HeteroMatrixError> for HeteroMulError {
    fn from(error: HeteroMatrixError) -> Self {
        Self::Matrix(error)
    }
}

impl From<VirtualPcsError> for HeteroMulError {
    fn from(error: VirtualPcsError) -> Self {
        Self::Pcs(error)
    }
}

impl Display for HeteroMulError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for HeteroMulError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::heterogeneous::{HeteroBlockMat, MatrixBlock, MatrixDims, PaddedDomain};
    use crate::utils::zktr::TranElem;
    use ark_bls12_381::Bls12_381;

    fn witness() -> HeteroMulWitness {
        let left = HeteroBlockMat::new(
            PaddedDomain::new(2, 2, 2).unwrap(),
            vec![
                Some(
                    MatrixBlock::from_columns(
                        MatrixDims::new(2, 2).unwrap(),
                        vec![vec![1_i32, 2], vec![3, 4]],
                    )
                    .unwrap(),
                ),
                Some(
                    MatrixBlock::from_columns(MatrixDims::new(1, 1).unwrap(), vec![vec![2_i32]])
                        .unwrap(),
                ),
            ],
        )
        .unwrap();
        let right = HeteroBlockMat::new(
            PaddedDomain::new(2, 2, 2).unwrap(),
            vec![
                Some(
                    MatrixBlock::from_columns(
                        MatrixDims::new(2, 2).unwrap(),
                        vec![vec![1_i32, 0], vec![0, 1]],
                    )
                    .unwrap(),
                ),
                Some(
                    MatrixBlock::from_columns(
                        MatrixDims::new(1, 2).unwrap(),
                        vec![vec![3_i32], vec![4]],
                    )
                    .unwrap(),
                ),
            ],
        )
        .unwrap();
        let output = HeteroBlockMat::new(
            PaddedDomain::new(2, 2, 2).unwrap(),
            vec![
                Some(
                    MatrixBlock::from_columns(
                        MatrixDims::new(2, 2).unwrap(),
                        vec![vec![1_i32, 2], vec![3, 4]],
                    )
                    .unwrap(),
                ),
                Some(
                    MatrixBlock::from_columns(
                        MatrixDims::new(1, 2).unwrap(),
                        vec![vec![6_i32], vec![8]],
                    )
                    .unwrap(),
                ),
            ],
        )
        .unwrap();
        HeteroMulWitness::new(left, right, output).unwrap()
    }

    #[test]
    fn heterogeneous_multiplications_prove_and_verify_end_to_end() {
        let witness = witness();
        let srs = HeteroMulProtocol::<Bls12_381>::setup(&witness).unwrap();
        let (instance, proof) =
            HeteroMulProtocol::<Bls12_381>::prove_owned(&srs, witness, 8).unwrap();

        assert_eq!(instance.buckets.len(), 2);
        assert!(HeteroMulProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof,
        ));
    }

    #[test]
    fn heterogeneous_multiplication_rejects_incorrect_output() {
        let mut witness = witness();
        witness.output.blocks[0].as_mut().unwrap().columns[0][0] += 1;
        let srs = HeteroMulProtocol::<Bls12_381>::setup(&witness).unwrap();
        let (instance, proof) = HeteroMulProtocol::<Bls12_381>::prove(&srs, &witness, 8).unwrap();

        assert!(!HeteroMulProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof,
        ));
    }

    #[test]
    fn heterogeneous_multiplication_binds_public_shapes() {
        let witness = witness();
        let srs = HeteroMulProtocol::<Bls12_381>::setup(&witness).unwrap();
        let (mut instance, proof) =
            HeteroMulProtocol::<Bls12_381>::prove(&srs, &witness, 8).unwrap();
        instance.shape.gates[1] = Some(MulGateDims {
            left: MatrixDims::new(2, 1).unwrap(),
            right: MatrixDims::new(1, 2).unwrap(),
            output: MatrixDims::new(2, 2).unwrap(),
        });

        assert!(!HeteroMulProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof,
        ));
    }

    #[test]
    fn heterogeneous_multiplication_rejects_trailing_transcript_data() {
        let witness = witness();
        let srs = HeteroMulProtocol::<Bls12_381>::setup(&witness).unwrap();
        let (instance, mut proof) =
            HeteroMulProtocol::<Bls12_381>::prove(&srs, &witness, 8).unwrap();
        proof.buckets[0]
            .reduction_transcript
            .data
            .push(TranElem::Field(<Bls12_381 as Pairing>::ScalarField::from(
                1_u64,
            )));

        assert!(!HeteroMulProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof,
        ));
    }

    #[test]
    fn heterogeneous_multiplication_rejects_truncated_transcript() {
        let witness = witness();
        let srs = HeteroMulProtocol::<Bls12_381>::setup(&witness).unwrap();
        let (instance, mut proof) =
            HeteroMulProtocol::<Bls12_381>::prove(&srs, &witness, 8).unwrap();
        proof.buckets[0].reduction_transcript.data.pop();

        assert!(!HeteroMulProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof,
        ));
    }

    #[test]
    fn ram_profile_separates_stored_values_from_virtual_padding() {
        let witness = witness();
        let profile = witness.ram_profile();

        assert_eq!(profile.stored_witness_entries, 17);
        assert_eq!(profile.monolithic_padded_entries, 24);
        assert_eq!(profile.bucket_padded_entries, 17);
        assert_eq!(profile.max_projection_entries, 2);
        assert_eq!(profile.max_pcs_axis, 2);
    }

    #[test]
    fn top_level_protocol_rejects_non_power_of_two_gate_dimensions() {
        let left = HeteroBlockMat::new(
            PaddedDomain::new(1, 2, 4).unwrap(),
            vec![Some(
                MatrixBlock::from_columns(
                    MatrixDims::new(2, 3).unwrap(),
                    vec![vec![1_i32, 2], vec![3, 4], vec![5, 6]],
                )
                .unwrap(),
            )],
        )
        .unwrap();
        let right = HeteroBlockMat::new(
            PaddedDomain::new(1, 4, 2).unwrap(),
            vec![Some(
                MatrixBlock::from_columns(
                    MatrixDims::new(3, 2).unwrap(),
                    vec![vec![1_i32, 0, 1], vec![0, 1, 1]],
                )
                .unwrap(),
            )],
        )
        .unwrap();
        let output = HeteroBlockMat::new(
            PaddedDomain::new(1, 2, 2).unwrap(),
            vec![Some(
                MatrixBlock::from_columns(
                    MatrixDims::new(2, 2).unwrap(),
                    vec![vec![6_i32, 8], vec![8, 10]],
                )
                .unwrap(),
            )],
        )
        .unwrap();

        assert!(matches!(
            HeteroMulWitness::new(left, right, output),
            Err(HeteroMulError::GateDimensionNotPowerOfTwo { slot: 0, .. }),
        ));
    }
}
