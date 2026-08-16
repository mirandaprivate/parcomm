//! Bounded-RAM grand-product proof over a heterogeneous BVA factor stream.

use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ff::{PrimeField, UniformRand};
use ark_poly_commit::smart_pc::{data_structures::Trans as PcsTrans, SmartPC};
use ark_serialize::{CanonicalSerialize, Compress};
use ark_std::Zero;
use std::convert::Infallible;
use std::fmt::{Display, Formatter};

use crate::data_structures::SRS;
use crate::protocols::bucketed_ip::{
    BucketedVector, BvaError, BvaFieldPairSource, BvaFieldSource, BvaLayout, EoLiteBulletProof,
    EoWitnessPairProof, PaddingWorkCounters, StreamingEoError, StreamingEoProfile, TbvBucket,
    TensorBasisVector,
};
use crate::protocols::virtualpcs::{
    bva_eo_opening_point, commit_bva_field_materialized_parallel, commit_bva_field_streaming,
    inner_product_bva_field_streaming, open_bva_field_streaming, BvaStreamingCommitProfile,
    StreamingOpenProfile, StreamingVirtualPcsError, VirtualPcsError,
};
use crate::utils::fiat_shamir::FiatShamir;

#[derive(Clone, Debug)]
pub struct BvaGrandProductOpening<E: Pairing> {
    pub value_commitment: E::G1,
    pub opening: PcsTrans<E>,
}

#[derive(Clone, Debug)]
pub struct BvaGrandProductHadamardProof<E: Pairing> {
    pub product_claim: E::ScalarField,
    pub eo_proof: EoWitnessPairProof<E::ScalarField>,
    pub factor_opening: BvaGrandProductOpening<E>,
    pub prefix_opening: BvaGrandProductOpening<E>,
    pub product_opening: BvaGrandProductOpening<E>,
}

#[derive(Clone, Debug)]
pub struct BvaGrandProductRecurrenceProof<E: Pairing> {
    pub prefix_claim: E::ScalarField,
    pub product_claim: E::ScalarField,
    pub prefix_eo_proof: EoLiteBulletProof<E::ScalarField>,
    pub product_eo_proof: EoLiteBulletProof<E::ScalarField>,
    pub prefix_opening: BvaGrandProductOpening<E>,
    pub product_opening: BvaGrandProductOpening<E>,
}

/// Proof of the BatchPCS/zkSMART grand-product equations
///
/// `w' = factors * w`, `w[i+1] = w'[i]`, `w[0] = 1`, and
/// `w'[last] = expected_product`, including transitions across physical BVA
/// vectors. The prefix vectors are auxiliary commitments; the caller's factor
/// commitment remains the canonical input to this subprotocol.
#[derive(Clone, Debug)]
pub struct BvaGrandProductProof<E: Pairing> {
    pub prefix_commitment: PairingOutput<E>,
    pub product_commitment: PairingOutput<E>,
    pub hadamard: BvaGrandProductHadamardProof<E>,
    pub recurrence: BvaGrandProductRecurrenceProof<E>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BvaGrandProductProfile<F: PrimeField> {
    pub final_product: F,
    pub source_passes: usize,
    pub total_visited_entries: usize,
    pub maximum_live_vector_entries: usize,
    pub materialized_padding_entries: usize,
    pub padding_field_operations: u64,
    pub padding_group_operations: usize,
}

impl<F: PrimeField> BvaGrandProductProfile<F> {
    fn new(layout: &BvaLayout) -> Self {
        Self {
            final_product: F::one(),
            source_passes: 0,
            total_visited_entries: 0,
            maximum_live_vector_entries: layout
                .vector_lengths()
                .iter()
                .copied()
                .max()
                .unwrap_or(0)
                .saturating_mul(2),
            materialized_padding_entries: 0,
            padding_field_operations: 0,
            padding_group_operations: 0,
        }
    }

    fn absorb_commit(&mut self, current: BvaStreamingCommitProfile) {
        self.source_passes += 1;
        self.total_visited_entries += current.physical_entries;
        self.materialized_padding_entries += current.materialized_padding_entries;
        self.padding_group_operations += current.padding_group_operations;
    }

    fn absorb_open(&mut self, current: StreamingOpenProfile) {
        self.source_passes += current.source_passes;
        self.total_visited_entries += current.physical_entries;
        self.materialized_padding_entries += current.materialized_projection_padding_entries;
        self.padding_field_operations += current.padding_projection_field_multiplications as u64;
    }

    fn absorb_eo(&mut self, counters: PaddingWorkCounters, current: StreamingEoProfile) {
        self.source_passes += current.source_passes;
        self.total_visited_entries += current.total_visited_entries;
        self.maximum_live_vector_entries = self
            .maximum_live_vector_entries
            .max(current.maximum_live_vector_entries);
        self.materialized_padding_entries += current.materialized_padding_entries;
        self.padding_field_operations += counters.padding_ip_field_ops
            + counters.padding_fold_field_ops
            + current.padding_field_operations;
    }
}

#[derive(Debug)]
enum PrefixSourceError<SourceError> {
    Source(SourceError),
    Bva(BvaError),
}

impl<SourceError: Display> Display for PrefixSourceError<SourceError> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Source(error) => Display::fmt(error, formatter),
            Self::Bva(error) => Display::fmt(error, formatter),
        }
    }
}

impl<SourceError> std::error::Error for PrefixSourceError<SourceError> where
    SourceError: std::error::Error + 'static
{
}

#[derive(Clone, Copy)]
enum PrefixView {
    Exclusive,
    Inclusive,
}

struct PrefixProductBvaSource<'a, S, F> {
    source: &'a mut S,
    layout: BvaLayout,
    view: PrefixView,
    final_product: F,
}

impl<'a, S, F: PrimeField> PrefixProductBvaSource<'a, S, F>
where
    S: BvaFieldSource<F>,
{
    fn new(source: &'a mut S, view: PrefixView) -> Self {
        Self {
            layout: source.layout().clone(),
            source,
            view,
            final_product: F::one(),
        }
    }
}

impl<S, F> BvaFieldSource<F> for PrefixProductBvaSource<'_, S, F>
where
    S: BvaFieldSource<F>,
    F: PrimeField,
{
    type Error = PrefixSourceError<S::Error>;

    fn layout(&self) -> &BvaLayout {
        &self.layout
    }

    fn for_each_vector(&mut self, visit: &mut dyn FnMut(usize, &[F])) -> Result<(), Self::Error> {
        let mut prefix = F::one();
        let mut expected_index = 0_usize;
        let mut bva_error = None;
        let view = self.view;
        let lengths = self.layout.vector_lengths().to_vec();
        self.source
            .for_each_vector(&mut |vector_index, factors| {
                if bva_error.is_some() {
                    return;
                }
                if vector_index != expected_index
                    || lengths.get(vector_index).copied() != Some(factors.len())
                {
                    bva_error = Some(BvaError::VectorIndex);
                    return;
                }
                let mut values = Vec::with_capacity(factors.len());
                for factor in factors {
                    match view {
                        PrefixView::Exclusive => {
                            values.push(prefix);
                            prefix *= factor;
                        }
                        PrefixView::Inclusive => {
                            prefix *= factor;
                            values.push(prefix);
                        }
                    }
                }
                visit(vector_index, &values);
                expected_index += 1;
            })
            .map_err(PrefixSourceError::Source)?;
        if let Some(error) = bva_error {
            return Err(PrefixSourceError::Bva(error));
        }
        if expected_index != lengths.len() {
            return Err(PrefixSourceError::Bva(BvaError::MissingVector));
        }
        self.final_product = prefix;
        Ok(())
    }
}

struct FactorWeightedPrefixPairSource<'a, S, F> {
    source: &'a mut S,
    layout: BvaLayout,
    weights: TensorBasisVector<F>,
    final_product: F,
}

impl<'a, S, F: PrimeField> FactorWeightedPrefixPairSource<'a, S, F>
where
    S: BvaFieldSource<F>,
{
    fn new(source: &'a mut S, weights: TensorBasisVector<F>) -> Self {
        Self {
            layout: source.layout().clone(),
            source,
            weights,
            final_product: F::one(),
        }
    }
}

impl<S, F> BvaFieldPairSource<F> for FactorWeightedPrefixPairSource<'_, S, F>
where
    S: BvaFieldSource<F>,
    F: PrimeField,
{
    type Error = PrefixSourceError<S::Error>;

    fn layout(&self) -> &BvaLayout {
        &self.layout
    }

    fn for_each_vector_pair(
        &mut self,
        visit: &mut dyn FnMut(usize, &[F], &[F]),
    ) -> Result<(), Self::Error> {
        let mut prefix = F::one();
        let mut expected_index = 0_usize;
        let mut bva_error = None;
        let lengths = self.layout.vector_lengths().to_vec();
        let weights = &self.weights;
        self.source
            .for_each_vector(&mut |vector_index, factors| {
                if bva_error.is_some() {
                    return;
                }
                if vector_index != expected_index
                    || lengths.get(vector_index).copied() != Some(factors.len())
                {
                    bva_error = Some(BvaError::VectorIndex);
                    return;
                }
                let mut weighted_prefix = Vec::with_capacity(factors.len());
                for factor in factors {
                    weighted_prefix.push(prefix);
                    prefix *= factor;
                }
                if let Err(error) =
                    weights.scale_physical_vector_in_place(vector_index, &mut weighted_prefix)
                {
                    bva_error = Some(error);
                    return;
                }
                visit(vector_index, factors, &weighted_prefix);
                expected_index += 1;
            })
            .map_err(PrefixSourceError::Source)?;
        if let Some(error) = bva_error {
            return Err(PrefixSourceError::Bva(error));
        }
        if expected_index != lengths.len() {
            return Err(PrefixSourceError::Bva(BvaError::MissingVector));
        }
        self.final_product = prefix;
        Ok(())
    }
}

#[derive(Debug)]
pub enum BvaGrandProductError<SourceError> {
    Bva(BvaError),
    VirtualPcs(VirtualPcsError),
    Source(SourceError),
    ProductMismatch,
    ZeroChallenge,
    RelationMismatch,
}

impl<SourceError: Display> Display for BvaGrandProductError<SourceError> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bva(error) => Display::fmt(error, formatter),
            Self::VirtualPcs(error) => Display::fmt(error, formatter),
            Self::Source(error) => Display::fmt(error, formatter),
            Self::ProductMismatch => write!(formatter, "grand-product terminal mismatch"),
            Self::ZeroChallenge => write!(formatter, "grand-product Fiat-Shamir challenge is zero"),
            Self::RelationMismatch => write!(formatter, "grand-product relation mismatch"),
        }
    }
}

impl<SourceError> std::error::Error for BvaGrandProductError<SourceError> where
    SourceError: std::error::Error + 'static
{
}

struct GrandProductChallenges<F> {
    hadamard_point: Vec<F>,
    hadamard_binding: F,
    theta: F,
    vector_weights: Vec<F>,
    boundary_weights: Vec<F>,
    terminal_weight: F,
    prefix_eo_binding: F,
    product_eo_binding: F,
}

impl<E: Pairing> BvaGrandProductProof<E> {
    /// Materialized prover for large instances on machines with sufficient
    /// RAM. Physical factor, prefix, and product entries are retained and EO
    /// folded in linear total work; BVA padding remains metadata-only.
    #[allow(clippy::too_many_arguments)]
    pub fn prove_materialized(
        srs: &SRS<E>,
        layer: usize,
        tag: usize,
        factor_source: &mut BucketedVector<E::ScalarField>,
        factor_commitment: PairingOutput<E>,
        factor_tier_one: &[E::G1],
        factor_hiding: E::ScalarField,
        expected_product: E::ScalarField,
    ) -> Result<(Self, BvaGrandProductProfile<E::ScalarField>), BvaGrandProductError<Infallible>>
    {
        let layout = factor_source.layout().clone();
        let mut profile = BvaGrandProductProfile::new(&layout);
        let (mut prefix_source, mut product_source, terminal) =
            materialize_prefix_products(factor_source).map_err(BvaGrandProductError::Bva)?;
        if terminal != expected_product {
            return Err(BvaGrandProductError::ProductMismatch);
        }
        profile.final_product = terminal;
        profile.total_visited_entries += factor_source.physical_entries();
        let rng = &mut ark_std::rand::thread_rng();

        let prefix_hiding = E::ScalarField::rand(rng);
        let (prefix_commitment, prefix_tier_one, prefix_commit_profile) =
            commit_bva_field_materialized_parallel::<E>(&srs.pc_pp, &prefix_source, prefix_hiding)
                .map_err(BvaGrandProductError::VirtualPcs)?;
        profile.absorb_commit(prefix_commit_profile);

        let product_hiding = E::ScalarField::rand(rng);
        let (product_commitment, product_tier_one, product_commit_profile) =
            commit_bva_field_materialized_parallel::<E>(
                &srs.pc_pp,
                &product_source,
                product_hiding,
            )
            .map_err(BvaGrandProductError::VirtualPcs)?;
        profile.absorb_commit(product_commit_profile);

        let challenges = derive_grand_product_challenges::<E>(
            layer,
            tag,
            &layout,
            factor_commitment,
            prefix_commitment,
            product_commitment,
            expected_product,
        )
        .ok_or(BvaGrandProductError::ZeroChallenge)?;
        let hadamard_tbv = TensorBasisVector::global_tensor(&layout, &challenges.hadamard_point)
            .map_err(BvaGrandProductError::Bva)?;
        let product_claim = product_source
            .inner_product(&hadamard_tbv)
            .map_err(BvaGrandProductError::Bva)?
            .value;
        profile.total_visited_entries += product_source.physical_entries();

        let mut weighted_prefix = prefix_source.clone();
        for (vector_index, values) in weighted_prefix.vectors_mut().iter_mut().enumerate() {
            hadamard_tbv
                .scale_physical_vector_in_place(vector_index, values)
                .map_err(BvaGrandProductError::Bva)?;
        }
        let (hadamard_eo, hadamard_counters) = EoWitnessPairProof::prove_bound(
            factor_source,
            &weighted_prefix,
            product_claim,
            challenges.hadamard_binding,
        )
        .map_err(BvaGrandProductError::Bva)?;
        absorb_owned_eo(
            &mut profile,
            hadamard_counters,
            factor_source.physical_entries(),
            factor_source.physical_entries() + weighted_prefix.physical_entries(),
        );
        drop(weighted_prefix);
        let hadamard_verification = hadamard_eo.verify_bound(
            layout.logical_len(),
            product_claim,
            challenges.hadamard_binding,
        );
        if !hadamard_verification.accepted {
            return Err(BvaGrandProductError::RelationMismatch);
        }
        let factor_opening = open_plain_source_at::<E, _>(
            srs,
            factor_source,
            factor_commitment,
            factor_tier_one,
            factor_hiding,
            &hadamard_verification.challenges_inv,
            hadamard_eo.a_reduced,
            &mut profile,
        )?;
        let weighted_prefix_point = challenges
            .hadamard_point
            .iter()
            .zip(&hadamard_verification.challenges)
            .map(|(weight, eo)| *weight * eo)
            .collect::<Vec<_>>();
        let prefix_opening = open_plain_source_at::<E, _>(
            srs,
            &mut prefix_source,
            prefix_commitment,
            &prefix_tier_one,
            prefix_hiding,
            &weighted_prefix_point,
            hadamard_eo.b_reduced,
            &mut profile,
        )?;
        let product_opening = open_plain_source_at::<E, _>(
            srs,
            &mut product_source,
            product_commitment,
            &product_tier_one,
            product_hiding,
            &challenges.hadamard_point,
            product_claim,
            &mut profile,
        )?;

        let (prefix_tbv, product_tbv, public_offset) = build_recurrence_tbvs(
            &layout,
            challenges.theta,
            &challenges.vector_weights,
            &challenges.boundary_weights,
            challenges.terminal_weight,
            expected_product,
        )
        .map_err(BvaGrandProductError::Bva)?;
        let prefix_claim = prefix_source
            .inner_product(&prefix_tbv)
            .map_err(BvaGrandProductError::Bva)?
            .value;
        let recurrence_product_claim = product_source
            .inner_product(&product_tbv)
            .map_err(BvaGrandProductError::Bva)?
            .value;
        profile.total_visited_entries +=
            prefix_source.physical_entries() + product_source.physical_entries();
        if prefix_claim + recurrence_product_claim + public_offset != E::ScalarField::zero() {
            return Err(BvaGrandProductError::RelationMismatch);
        }

        let (prefix_eo_proof, prefix_eo_counters) = EoLiteBulletProof::prove_bound(
            &prefix_source,
            &prefix_tbv,
            prefix_claim,
            challenges.prefix_eo_binding,
        )
        .map_err(BvaGrandProductError::Bva)?;
        absorb_owned_eo(
            &mut profile,
            prefix_eo_counters,
            prefix_source.physical_entries(),
            prefix_source.physical_entries(),
        );
        let prefix_verification = prefix_eo_proof.verify_with_tbv_bound(
            &prefix_tbv,
            prefix_claim,
            challenges.prefix_eo_binding,
        );
        if !prefix_verification.accepted {
            return Err(BvaGrandProductError::RelationMismatch);
        }
        let recurrence_prefix_opening = open_plain_source_at::<E, _>(
            srs,
            &mut prefix_source,
            prefix_commitment,
            &prefix_tier_one,
            prefix_hiding,
            &prefix_verification.challenges_inv,
            prefix_eo_proof.a_reduced,
            &mut profile,
        )?;

        let (product_eo_proof, product_eo_counters) = EoLiteBulletProof::prove_bound(
            &product_source,
            &product_tbv,
            recurrence_product_claim,
            challenges.product_eo_binding,
        )
        .map_err(BvaGrandProductError::Bva)?;
        absorb_owned_eo(
            &mut profile,
            product_eo_counters,
            product_source.physical_entries(),
            product_source.physical_entries(),
        );
        let product_verification = product_eo_proof.verify_with_tbv_bound(
            &product_tbv,
            recurrence_product_claim,
            challenges.product_eo_binding,
        );
        if !product_verification.accepted {
            return Err(BvaGrandProductError::RelationMismatch);
        }
        let recurrence_product_opening = open_plain_source_at::<E, _>(
            srs,
            &mut product_source,
            product_commitment,
            &product_tier_one,
            product_hiding,
            &product_verification.challenges_inv,
            product_eo_proof.a_reduced,
            &mut profile,
        )?;

        Ok((
            Self {
                prefix_commitment,
                product_commitment,
                hadamard: BvaGrandProductHadamardProof {
                    product_claim,
                    eo_proof: hadamard_eo,
                    factor_opening,
                    prefix_opening,
                    product_opening,
                },
                recurrence: BvaGrandProductRecurrenceProof {
                    prefix_claim,
                    product_claim: recurrence_product_claim,
                    prefix_eo_proof,
                    product_eo_proof,
                    prefix_opening: recurrence_prefix_opening,
                    product_opening: recurrence_product_opening,
                },
            },
            profile,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prove<S: BvaFieldSource<E::ScalarField>>(
        srs: &SRS<E>,
        layer: usize,
        tag: usize,
        factor_source: &mut S,
        factor_commitment: PairingOutput<E>,
        factor_tier_one: &[E::G1],
        factor_hiding: E::ScalarField,
        expected_product: E::ScalarField,
    ) -> Result<(Self, BvaGrandProductProfile<E::ScalarField>), BvaGrandProductError<S::Error>>
    {
        let layout = factor_source.layout().clone();
        let mut profile = BvaGrandProductProfile::new(&layout);
        let rng = &mut ark_std::rand::thread_rng();

        let prefix_hiding = E::ScalarField::rand(rng);
        let (prefix_commitment, prefix_tier_one, prefix_commit_profile, prefix_terminal) = {
            let mut prefix_source =
                PrefixProductBvaSource::new(factor_source, PrefixView::Exclusive);
            let (commitment, tier_one, commit_profile) =
                commit_bva_field_streaming::<E, _>(&srs.pc_pp, &mut prefix_source, prefix_hiding)
                    .map_err(map_prefix_pcs_error)?;
            (
                commitment,
                tier_one,
                commit_profile,
                prefix_source.final_product,
            )
        };
        profile.absorb_commit(prefix_commit_profile);

        let product_hiding = E::ScalarField::rand(rng);
        let (product_commitment, product_tier_one, product_commit_profile, product_terminal) = {
            let mut product_source =
                PrefixProductBvaSource::new(factor_source, PrefixView::Inclusive);
            let (commitment, tier_one, commit_profile) =
                commit_bva_field_streaming::<E, _>(&srs.pc_pp, &mut product_source, product_hiding)
                    .map_err(map_prefix_pcs_error)?;
            (
                commitment,
                tier_one,
                commit_profile,
                product_source.final_product,
            )
        };
        profile.absorb_commit(product_commit_profile);
        if prefix_terminal != expected_product || product_terminal != expected_product {
            return Err(BvaGrandProductError::ProductMismatch);
        }
        profile.final_product = product_terminal;

        let challenges = derive_grand_product_challenges::<E>(
            layer,
            tag,
            &layout,
            factor_commitment,
            prefix_commitment,
            product_commitment,
            expected_product,
        )
        .ok_or(BvaGrandProductError::ZeroChallenge)?;
        let hadamard_tbv = TensorBasisVector::global_tensor(&layout, &challenges.hadamard_point)
            .map_err(BvaGrandProductError::Bva)?;

        let product_claim = {
            let mut source = PrefixProductBvaSource::new(factor_source, PrefixView::Inclusive);
            let value = inner_product_bva_field_streaming(&mut source, &hadamard_tbv)
                .map_err(map_prefix_pcs_error)?;
            absorb_inner_profile(&mut profile, value.profile);
            value.value
        };
        let (hadamard_eo, hadamard_counters, hadamard_profile) = {
            let mut pair_source =
                FactorWeightedPrefixPairSource::new(factor_source, hadamard_tbv.clone());
            EoWitnessPairProof::prove_streaming_bound(
                &mut pair_source,
                product_claim,
                challenges.hadamard_binding,
            )
            .map_err(map_prefix_eo_error)?
        };
        profile.absorb_eo(hadamard_counters, hadamard_profile);
        let hadamard_verification = hadamard_eo.verify_bound(
            layout.logical_len(),
            product_claim,
            challenges.hadamard_binding,
        );
        if !hadamard_verification.accepted {
            return Err(BvaGrandProductError::RelationMismatch);
        }
        let factor_opening = open_plain_source_at::<E, S>(
            srs,
            factor_source,
            factor_commitment,
            factor_tier_one,
            factor_hiding,
            &hadamard_verification.challenges_inv,
            hadamard_eo.a_reduced,
            &mut profile,
        )?;
        let weighted_prefix_point = challenges
            .hadamard_point
            .iter()
            .zip(&hadamard_verification.challenges)
            .map(|(weight, eo)| *weight * eo)
            .collect::<Vec<_>>();
        let prefix_opening = open_prefix_source_at::<E, S>(
            srs,
            factor_source,
            PrefixView::Exclusive,
            prefix_commitment,
            &prefix_tier_one,
            prefix_hiding,
            &weighted_prefix_point,
            hadamard_eo.b_reduced,
            &mut profile,
        )?;
        let product_opening = open_prefix_source_at::<E, S>(
            srs,
            factor_source,
            PrefixView::Inclusive,
            product_commitment,
            &product_tier_one,
            product_hiding,
            &challenges.hadamard_point,
            product_claim,
            &mut profile,
        )?;

        let (prefix_tbv, product_tbv, public_offset) = build_recurrence_tbvs(
            &layout,
            challenges.theta,
            &challenges.vector_weights,
            &challenges.boundary_weights,
            challenges.terminal_weight,
            expected_product,
        )
        .map_err(BvaGrandProductError::Bva)?;
        let prefix_claim = {
            let mut source = PrefixProductBvaSource::new(factor_source, PrefixView::Exclusive);
            let value = inner_product_bva_field_streaming(&mut source, &prefix_tbv)
                .map_err(map_prefix_pcs_error)?;
            absorb_inner_profile(&mut profile, value.profile);
            value.value
        };
        let recurrence_product_claim = {
            let mut source = PrefixProductBvaSource::new(factor_source, PrefixView::Inclusive);
            let value = inner_product_bva_field_streaming(&mut source, &product_tbv)
                .map_err(map_prefix_pcs_error)?;
            absorb_inner_profile(&mut profile, value.profile);
            value.value
        };
        if prefix_claim + recurrence_product_claim + public_offset != E::ScalarField::zero() {
            return Err(BvaGrandProductError::RelationMismatch);
        }

        let (prefix_eo_proof, prefix_eo_counters, prefix_eo_profile) = {
            let mut source = PrefixProductBvaSource::new(factor_source, PrefixView::Exclusive);
            EoLiteBulletProof::prove_streaming_bound(
                &mut source,
                &prefix_tbv,
                prefix_claim,
                challenges.prefix_eo_binding,
            )
            .map_err(map_prefix_eo_error)?
        };
        profile.absorb_eo(prefix_eo_counters, prefix_eo_profile);
        let prefix_verification = prefix_eo_proof.verify_with_tbv_bound(
            &prefix_tbv,
            prefix_claim,
            challenges.prefix_eo_binding,
        );
        if !prefix_verification.accepted {
            return Err(BvaGrandProductError::RelationMismatch);
        }
        let recurrence_prefix_opening = open_prefix_source_at::<E, S>(
            srs,
            factor_source,
            PrefixView::Exclusive,
            prefix_commitment,
            &prefix_tier_one,
            prefix_hiding,
            &prefix_verification.challenges_inv,
            prefix_eo_proof.a_reduced,
            &mut profile,
        )?;

        let (product_eo_proof, product_eo_counters, product_eo_profile) = {
            let mut source = PrefixProductBvaSource::new(factor_source, PrefixView::Inclusive);
            EoLiteBulletProof::prove_streaming_bound(
                &mut source,
                &product_tbv,
                recurrence_product_claim,
                challenges.product_eo_binding,
            )
            .map_err(map_prefix_eo_error)?
        };
        profile.absorb_eo(product_eo_counters, product_eo_profile);
        let product_verification = product_eo_proof.verify_with_tbv_bound(
            &product_tbv,
            recurrence_product_claim,
            challenges.product_eo_binding,
        );
        if !product_verification.accepted {
            return Err(BvaGrandProductError::RelationMismatch);
        }
        let recurrence_product_opening = open_prefix_source_at::<E, S>(
            srs,
            factor_source,
            PrefixView::Inclusive,
            product_commitment,
            &product_tier_one,
            product_hiding,
            &product_verification.challenges_inv,
            product_eo_proof.a_reduced,
            &mut profile,
        )?;

        Ok((
            Self {
                prefix_commitment,
                product_commitment,
                hadamard: BvaGrandProductHadamardProof {
                    product_claim,
                    eo_proof: hadamard_eo,
                    factor_opening,
                    prefix_opening,
                    product_opening,
                },
                recurrence: BvaGrandProductRecurrenceProof {
                    prefix_claim,
                    product_claim: recurrence_product_claim,
                    prefix_eo_proof,
                    product_eo_proof,
                    prefix_opening: recurrence_prefix_opening,
                    product_opening: recurrence_product_opening,
                },
            },
            profile,
        ))
    }

    pub fn verify(
        &self,
        srs: &SRS<E>,
        layer: usize,
        tag: usize,
        layout: &BvaLayout,
        factor_commitment: PairingOutput<E>,
        expected_product: E::ScalarField,
    ) -> bool {
        let Some(challenges) = derive_grand_product_challenges::<E>(
            layer,
            tag,
            layout,
            factor_commitment,
            self.prefix_commitment,
            self.product_commitment,
            expected_product,
        ) else {
            return false;
        };
        let hadamard_verification = self.hadamard.eo_proof.verify_bound(
            layout.logical_len(),
            self.hadamard.product_claim,
            challenges.hadamard_binding,
        );
        if !hadamard_verification.accepted
            || self.hadamard.factor_opening.value_commitment
                != srs
                    .zksrs
                    .commit(self.hadamard.eo_proof.a_reduced, E::ScalarField::zero())
            || self.hadamard.prefix_opening.value_commitment
                != srs
                    .zksrs
                    .commit(self.hadamard.eo_proof.b_reduced, E::ScalarField::zero())
            || self.hadamard.product_opening.value_commitment
                != srs
                    .zksrs
                    .commit(self.hadamard.product_claim, E::ScalarField::zero())
            || !verify_bva_opening(
                srs,
                factor_commitment,
                layout,
                &hadamard_verification.challenges_inv,
                &self.hadamard.factor_opening,
            )
            || !verify_bva_opening(
                srs,
                self.prefix_commitment,
                layout,
                &challenges
                    .hadamard_point
                    .iter()
                    .zip(&hadamard_verification.challenges)
                    .map(|(weight, eo)| *weight * eo)
                    .collect::<Vec<_>>(),
                &self.hadamard.prefix_opening,
            )
            || !verify_bva_opening(
                srs,
                self.product_commitment,
                layout,
                &challenges.hadamard_point,
                &self.hadamard.product_opening,
            )
        {
            return false;
        }

        let Ok((prefix_tbv, product_tbv, public_offset)) = build_recurrence_tbvs(
            layout,
            challenges.theta,
            &challenges.vector_weights,
            &challenges.boundary_weights,
            challenges.terminal_weight,
            expected_product,
        ) else {
            return false;
        };
        if self.recurrence.prefix_claim + self.recurrence.product_claim + public_offset
            != E::ScalarField::zero()
        {
            return false;
        }
        let prefix_verification = self.recurrence.prefix_eo_proof.verify_with_tbv_bound(
            &prefix_tbv,
            self.recurrence.prefix_claim,
            challenges.prefix_eo_binding,
        );
        let product_verification = self.recurrence.product_eo_proof.verify_with_tbv_bound(
            &product_tbv,
            self.recurrence.product_claim,
            challenges.product_eo_binding,
        );
        prefix_verification.accepted
            && product_verification.accepted
            && self.recurrence.prefix_opening.value_commitment
                == srs.zksrs.commit(
                    self.recurrence.prefix_eo_proof.a_reduced,
                    E::ScalarField::zero(),
                )
            && self.recurrence.product_opening.value_commitment
                == srs.zksrs.commit(
                    self.recurrence.product_eo_proof.a_reduced,
                    E::ScalarField::zero(),
                )
            && verify_bva_opening(
                srs,
                self.prefix_commitment,
                layout,
                &prefix_verification.challenges_inv,
                &self.recurrence.prefix_opening,
            )
            && verify_bva_opening(
                srs,
                self.product_commitment,
                layout,
                &product_verification.challenges_inv,
                &self.recurrence.product_opening,
            )
    }

    pub fn serialized_size_bytes(&self) -> usize {
        let opening_size = |opening: &BvaGrandProductOpening<E>| {
            opening.value_commitment.serialized_size(Compress::Yes)
                + opening.opening.serialized_size(Compress::Yes)
        };
        self.prefix_commitment.serialized_size(Compress::Yes)
            + self.product_commitment.serialized_size(Compress::Yes)
            + self.hadamard.product_claim.serialized_size(Compress::Yes)
            + self.hadamard.eo_proof.serialized_size(Compress::Yes)
            + opening_size(&self.hadamard.factor_opening)
            + opening_size(&self.hadamard.prefix_opening)
            + opening_size(&self.hadamard.product_opening)
            + self.recurrence.prefix_claim.serialized_size(Compress::Yes)
            + self.recurrence.product_claim.serialized_size(Compress::Yes)
            + self
                .recurrence
                .prefix_eo_proof
                .serialized_size(Compress::Yes)
            + self
                .recurrence
                .product_eo_proof
                .serialized_size(Compress::Yes)
            + opening_size(&self.recurrence.prefix_opening)
            + opening_size(&self.recurrence.product_opening)
    }
}

fn materialize_prefix_products<F: PrimeField>(
    factors: &BucketedVector<F>,
) -> Result<(BucketedVector<F>, BucketedVector<F>, F), BvaError> {
    let mut prefix = F::one();
    let mut exclusive_vectors = Vec::with_capacity(factors.vectors().len());
    let mut inclusive_vectors = Vec::with_capacity(factors.vectors().len());
    for factor_vector in factors.vectors() {
        let mut exclusive = Vec::with_capacity(factor_vector.len());
        let mut inclusive = Vec::with_capacity(factor_vector.len());
        for factor in factor_vector {
            exclusive.push(prefix);
            prefix *= factor;
            inclusive.push(prefix);
        }
        exclusive_vectors.push(exclusive);
        inclusive_vectors.push(inclusive);
    }
    let exclusive = BucketedVector::new(exclusive_vectors)?;
    let inclusive = BucketedVector::new(inclusive_vectors)?;
    if exclusive.layout() != factors.layout() || inclusive.layout() != factors.layout() {
        return Err(BvaError::LayoutMismatch);
    }
    Ok((exclusive, inclusive, prefix))
}

fn absorb_owned_eo<F: PrimeField>(
    profile: &mut BvaGrandProductProfile<F>,
    counters: PaddingWorkCounters,
    visited_entries: usize,
    maximum_live_entries: usize,
) {
    profile.total_visited_entries += visited_entries;
    profile.maximum_live_vector_entries = profile
        .maximum_live_vector_entries
        .max(maximum_live_entries);
    profile.materialized_padding_entries += counters.materialized_padding_entries as usize;
    profile.padding_field_operations +=
        counters.padding_ip_field_ops + counters.padding_fold_field_ops;
}

#[allow(clippy::too_many_arguments)]
fn derive_grand_product_challenges<E: Pairing>(
    layer: usize,
    tag: usize,
    layout: &BvaLayout,
    factor_commitment: PairingOutput<E>,
    prefix_commitment: PairingOutput<E>,
    product_commitment: PairingOutput<E>,
    expected_product: E::ScalarField,
) -> Option<GrandProductChallenges<E::ScalarField>> {
    let mut fs = FiatShamir::new();
    fs.push(&0x4256_4147_5052_4f44_u64);
    fs.push(&tag);
    fs.push(&layer);
    fs.push(&layout.logical_len());
    fs.push(&factor_commitment);
    fs.push(&prefix_commitment);
    fs.push(&product_commitment);
    fs.push(&expected_product);
    let offsets = layout.bucket_offsets();
    for vector_index in 0..layout.vector_lengths().len() {
        fs.push(&layout.vector_lengths()[vector_index]);
        fs.push(&offsets[layout.vector_to_bucket()[vector_index]]);
    }
    let hadamard_point = (0..layout.logical_len().ilog2())
        .map(|_| fs.gen_challenge::<E::ScalarField>())
        .collect::<Vec<_>>();
    let hadamard_binding = fs.gen_challenge::<E::ScalarField>();
    let theta = fs.gen_challenge::<E::ScalarField>();
    let vector_weights = (0..layout.vector_lengths().len())
        .map(|_| fs.gen_challenge::<E::ScalarField>())
        .collect::<Vec<_>>();
    let boundary_weights = (0..layout.vector_lengths().len())
        .map(|_| fs.gen_challenge::<E::ScalarField>())
        .collect::<Vec<_>>();
    let terminal_weight = fs.gen_challenge::<E::ScalarField>();
    let prefix_eo_binding = fs.gen_challenge::<E::ScalarField>();
    let product_eo_binding = fs.gen_challenge::<E::ScalarField>();
    if hadamard_binding.is_zero()
        || theta.is_zero()
        || terminal_weight.is_zero()
        || prefix_eo_binding.is_zero()
        || product_eo_binding.is_zero()
        || vector_weights.iter().any(|value| value.is_zero())
        || boundary_weights.iter().any(|value| value.is_zero())
    {
        return None;
    }
    Some(GrandProductChallenges {
        hadamard_point,
        hadamard_binding,
        theta,
        vector_weights,
        boundary_weights,
        terminal_weight,
        prefix_eo_binding,
        product_eo_binding,
    })
}

fn build_recurrence_tbvs<F: PrimeField>(
    layout: &BvaLayout,
    theta: F,
    vector_weights: &[F],
    boundary_weights: &[F],
    terminal_weight: F,
    expected_product: F,
) -> Result<(TensorBasisVector<F>, TensorBasisVector<F>, F), BvaError> {
    let vector_count = layout.vector_lengths().len();
    if vector_weights.len() != vector_count || boundary_weights.len() != vector_count {
        return Err(BvaError::VectorIndex);
    }
    let mut prefix_terms = vec![Vec::new(); layout.bucket_sizes().len()];
    let mut product_terms = vec![Vec::new(); layout.bucket_sizes().len()];
    for vector_index in 0..vector_count {
        let length = layout.vector_lengths()[vector_index];
        let bucket = layout.vector_to_bucket()[vector_index];
        let bucket_size = layout.bucket_sizes()[bucket];
        let powers = (0..bucket_size.ilog2() as usize)
            .map(|bit| (F::one(), theta.pow([1_u64 << bit])))
            .collect::<Vec<_>>();
        let point_zero = point_factors::<F>(bucket_size, 0);
        let point_last = point_factors::<F>(bucket_size, length - 1);
        let vector_weight = vector_weights[vector_index];
        prefix_terms[bucket].push((vector_weight, powers.clone()));
        prefix_terms[bucket].push((-vector_weight, point_zero.clone()));
        prefix_terms[bucket].push((boundary_weights[vector_index], point_zero));

        product_terms[bucket].push((-vector_weight * theta, powers));
        product_terms[bucket].push((
            vector_weight * theta.pow([length as u64]),
            point_last.clone(),
        ));
        if vector_index + 1 < vector_count {
            product_terms[bucket].push((-boundary_weights[vector_index + 1], point_last));
        } else {
            product_terms[bucket].push((terminal_weight, point_last));
        }
    }
    let prefix_buckets = prefix_terms
        .into_iter()
        .map(TbvBucket::from_factor_terms)
        .collect::<Vec<_>>();
    let product_buckets = product_terms
        .into_iter()
        .map(TbvBucket::from_factor_terms)
        .collect::<Vec<_>>();
    Ok((
        TensorBasisVector::new(layout, prefix_buckets)?,
        TensorBasisVector::new(layout, product_buckets)?,
        -boundary_weights[0] - terminal_weight * expected_product,
    ))
}

fn point_factors<F: PrimeField>(size: usize, index: usize) -> Vec<(F, F)> {
    (0..size.ilog2() as usize)
        .map(|bit| {
            if (index >> bit) & 1 == 0 {
                (F::one(), F::zero())
            } else {
                (F::zero(), F::one())
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn open_plain_source_at<E, S>(
    srs: &SRS<E>,
    source: &mut S,
    commitment: PairingOutput<E>,
    tier_one: &[E::G1],
    matrix_hiding: E::ScalarField,
    challenges: &[E::ScalarField],
    value: E::ScalarField,
    profile: &mut BvaGrandProductProfile<E::ScalarField>,
) -> Result<BvaGrandProductOpening<E>, BvaGrandProductError<S::Error>>
where
    E: Pairing,
    S: BvaFieldSource<E::ScalarField>,
{
    let value_commitment = srs.zksrs.commit(value, E::ScalarField::zero());
    let (xl, xr) = bva_eo_opening_point(source.layout().logical_len(), challenges)
        .map_err(BvaGrandProductError::VirtualPcs)?;
    let (opening, open_profile) = open_bva_field_streaming::<E, _>(
        &srs.pc_pp,
        source,
        &xl,
        &xr,
        E::pairing(value_commitment, srs.h_hat),
        commitment,
        tier_one,
        E::ScalarField::zero(),
        matrix_hiding,
    )
    .map_err(map_plain_pcs_error)?;
    profile.absorb_open(open_profile);
    Ok(BvaGrandProductOpening {
        value_commitment,
        opening,
    })
}

#[allow(clippy::too_many_arguments)]
fn open_prefix_source_at<E, S>(
    srs: &SRS<E>,
    factor_source: &mut S,
    view: PrefixView,
    commitment: PairingOutput<E>,
    tier_one: &[E::G1],
    matrix_hiding: E::ScalarField,
    challenges: &[E::ScalarField],
    value: E::ScalarField,
    profile: &mut BvaGrandProductProfile<E::ScalarField>,
) -> Result<BvaGrandProductOpening<E>, BvaGrandProductError<S::Error>>
where
    E: Pairing,
    S: BvaFieldSource<E::ScalarField>,
{
    let value_commitment = srs.zksrs.commit(value, E::ScalarField::zero());
    let logical_len = factor_source.layout().logical_len();
    let (xl, xr) =
        bva_eo_opening_point(logical_len, challenges).map_err(BvaGrandProductError::VirtualPcs)?;
    let mut source = PrefixProductBvaSource::new(factor_source, view);
    let (opening, open_profile) = open_bva_field_streaming::<E, _>(
        &srs.pc_pp,
        &mut source,
        &xl,
        &xr,
        E::pairing(value_commitment, srs.h_hat),
        commitment,
        tier_one,
        E::ScalarField::zero(),
        matrix_hiding,
    )
    .map_err(map_prefix_pcs_error)?;
    profile.absorb_open(open_profile);
    Ok(BvaGrandProductOpening {
        value_commitment,
        opening,
    })
}

fn verify_bva_opening<E: Pairing>(
    srs: &SRS<E>,
    commitment: PairingOutput<E>,
    layout: &BvaLayout,
    challenges: &[E::ScalarField],
    proof: &BvaGrandProductOpening<E>,
) -> bool {
    let Ok((xl, xr)) = bva_eo_opening_point(layout.logical_len(), challenges) else {
        return false;
    };
    proof.opening.has_rounds(xl.len(), xr.len())
        && SmartPC::<E>::verify(
            &srs.pc_pp,
            commitment,
            E::pairing(proof.value_commitment, srs.h_hat),
            &xl,
            &xr,
            &proof.opening,
        )
        .unwrap_or(false)
}

fn absorb_inner_profile<F: PrimeField>(
    profile: &mut BvaGrandProductProfile<F>,
    current: crate::protocols::virtualpcs::BvaStreamingInnerProductProfile,
) {
    profile.source_passes += current.source_passes;
    profile.total_visited_entries += current.physical_entries;
    profile.maximum_live_vector_entries = profile
        .maximum_live_vector_entries
        .max(current.maximum_live_vector_entries);
    profile.materialized_padding_entries += current.materialized_padding_entries;
    profile.padding_field_operations += current.padding_field_operations;
}

fn map_prefix_pcs_error<SourceError>(
    error: StreamingVirtualPcsError<PrefixSourceError<SourceError>>,
) -> BvaGrandProductError<SourceError> {
    match error {
        StreamingVirtualPcsError::Pcs(error) => BvaGrandProductError::VirtualPcs(error),
        StreamingVirtualPcsError::Source(PrefixSourceError::Source(error)) => {
            BvaGrandProductError::Source(error)
        }
        StreamingVirtualPcsError::Source(PrefixSourceError::Bva(error)) => {
            BvaGrandProductError::Bva(error)
        }
    }
}

fn map_plain_pcs_error<SourceError>(
    error: StreamingVirtualPcsError<SourceError>,
) -> BvaGrandProductError<SourceError> {
    match error {
        StreamingVirtualPcsError::Pcs(error) => BvaGrandProductError::VirtualPcs(error),
        StreamingVirtualPcsError::Source(error) => BvaGrandProductError::Source(error),
    }
}

fn map_prefix_eo_error<SourceError>(
    error: StreamingEoError<PrefixSourceError<SourceError>>,
) -> BvaGrandProductError<SourceError> {
    match error {
        StreamingEoError::Bva(error) => BvaGrandProductError::Bva(error),
        StreamingEoError::Source(PrefixSourceError::Source(error)) => {
            BvaGrandProductError::Source(error)
        }
        StreamingEoError::Source(PrefixSourceError::Bva(error)) => BvaGrandProductError::Bva(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_structures::SRS;
    use crate::protocols::bucketed_ip::{BucketedVector, BvaFieldSource, BvaLayout};
    use crate::protocols::virtualpcs::commit_bva_field_streaming;
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_ec::pairing::Pairing;

    #[derive(Clone)]
    struct ReplaySource {
        layout: BvaLayout,
        vectors: Vec<Vec<Fr>>,
    }

    impl ReplaySource {
        fn new(vectors: Vec<Vec<Fr>>) -> Self {
            Self {
                layout: BvaLayout::from_lengths(&vectors.iter().map(Vec::len).collect::<Vec<_>>())
                    .unwrap(),
                vectors,
            }
        }
    }

    impl BvaFieldSource<Fr> for ReplaySource {
        type Error = std::convert::Infallible;

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
    fn bva_grand_product_proves_cross_vector_prefix_recurrence_without_padding_work() {
        type E = Bls12_381;
        let srs = SRS::<E>::setup(3);
        let mut source = ReplaySource::new(vec![
            vec![2_u64.into(), 3_u64.into(), 4_u64.into(), 5_u64.into()],
            vec![6_u64.into(), 7_u64.into()],
            vec![8_u64.into()],
        ]);
        let layout = source.layout().clone();
        let factor_hiding = Fr::from(11_u64);
        let (factor_commitment, factor_tier_one, factor_profile) =
            commit_bva_field_streaming::<E, _>(&srs.pc_pp, &mut source, factor_hiding).unwrap();
        assert_eq!(factor_profile.materialized_padding_entries, 0);
        let expected_product = Fr::from(40_320_u64);

        let (mut proof, profile) = BvaGrandProductProof::prove(
            &srs,
            9,
            0x4750,
            &mut source,
            factor_commitment,
            &factor_tier_one,
            factor_hiding,
            expected_product,
        )
        .unwrap();

        assert!(proof.verify(
            &srs,
            9,
            0x4750,
            &layout,
            factor_commitment,
            expected_product,
        ));
        assert_eq!(profile.final_product, expected_product);
        assert_eq!(profile.maximum_live_vector_entries, 8);
        assert_eq!(profile.materialized_padding_entries, 0);
        assert_eq!(profile.padding_field_operations, 0);
        assert_eq!(profile.padding_group_operations, 0);
        assert!(proof.serialized_size_bytes() > 0);

        assert!(!proof.verify(
            &srs,
            9,
            0x4750,
            &layout,
            factor_commitment,
            expected_product + Fr::from(1_u64),
        ));
        let commitment_delta = E::pairing(srs.pc_pp.g_0, srs.h_hat);
        proof.prefix_commitment += commitment_delta;
        assert!(!proof.verify(
            &srs,
            9,
            0x4750,
            &layout,
            factor_commitment,
            expected_product,
        ));
        proof.prefix_commitment -= commitment_delta;

        proof.hadamard.eo_proof.a_reduced += Fr::from(1_u64);
        assert!(!proof.verify(
            &srs,
            9,
            0x4750,
            &layout,
            factor_commitment,
            expected_product,
        ));
        proof.hadamard.eo_proof.a_reduced -= Fr::from(1_u64);

        let reordered_layout = BvaLayout::from_lengths(&[2, 4, 1]).unwrap();
        assert!(!proof.verify(
            &srs,
            9,
            0x4750,
            &reordered_layout,
            factor_commitment,
            expected_product,
        ));
    }

    #[test]
    fn materialized_grand_product_uses_owned_linear_folds_without_padding_work() {
        type E = Bls12_381;
        let srs = SRS::<E>::setup(3);
        let mut factors = BucketedVector::new(vec![
            vec![2_u64.into(), 3_u64.into(), 4_u64.into(), 5_u64.into()],
            vec![6_u64.into(), 7_u64.into()],
            vec![8_u64.into()],
        ])
        .unwrap();
        let layout = factors.layout().clone();
        let factor_hiding = Fr::from(11_u64);
        let (factor_commitment, factor_tier_one, _) =
            commit_bva_field_streaming::<E, _>(&srs.pc_pp, &mut factors, factor_hiding).unwrap();
        let expected_product = Fr::from(40_320_u64);

        let (proof, profile) = BvaGrandProductProof::prove_materialized(
            &srs,
            9,
            0x4750,
            &mut factors,
            factor_commitment,
            &factor_tier_one,
            factor_hiding,
            expected_product,
        )
        .unwrap();

        assert!(proof.verify(
            &srs,
            9,
            0x4750,
            &layout,
            factor_commitment,
            expected_product,
        ));
        assert_eq!(profile.final_product, expected_product);
        assert_eq!(profile.materialized_padding_entries, 0);
        assert_eq!(profile.padding_field_operations, 0);
        assert_eq!(profile.padding_group_operations, 0);
        assert!(profile.source_passes < 20);
    }
}
