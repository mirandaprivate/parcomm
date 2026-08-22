//! GPU-resident prover stages for the packed i8 ParComm flow.
//!
//! The packed witness is uploaded once per device and then stays in device
//! memory for the whole proof, so every stage whose cost grows with the witness
//! reads it from there. Stages whose cost only grows with the packed side
//! lengths (the Dory folds, the pairings, the final openings) stay on the CPU,
//! where Rayon already saturates the available cores.

use std::mem::size_of;
use std::sync::Arc;
use std::time::Instant;

use ark_bls12_381::{Bls12_381, Fr, G1Projective as ArkG1Projective, G2Projective as ArkG2Projective};
use ark_ec::pairing::PairingOutput;
use ark_ec::CurveGroup;
use ark_std::{One, Zero};
use batch::bucketed_ip::{
    BvaI8Source, BvaLayout, PackedEoReducedVectors, PackedLaneProjections, PackedLaneRequest,
    PackedReducedSpec,
};
use icicle_bls12_381::curve::{
    G1Affine as IcicleG1Affine, G1Projective as IcicleG1Projective, ScalarField as IcicleScalarField,
};
use icicle_core::msm::{msm, MSMConfig};
use icicle_core::projective::Projective;
use icicle_runtime::memory::{DeviceVec, HostOrDeviceSlice, HostSlice};
use icicle_runtime::Device;
use poly_commit::smart_pc::utils::inner_pairing_product;
use poly_commit::smart_pc::UniversalParams;
use rayon::prelude::*;

use crate::smart_pc::{ark_g1_to_icicle, icicle_g1_to_ark, GpuSmartPcError};

extern "C" {
    fn parcomm_expand_i8_colmajor(
        input: *const i8,
        cols: usize,
        column_start: usize,
        shard_cols: usize,
        row_start: usize,
        shard_rows: usize,
        output: *mut std::ffi::c_void,
        negative: bool,
    ) -> i32;

    fn parcomm_project_left(
        input: *const i8,
        cols: usize,
        row_start: usize,
        row_end: usize,
        weights: *const std::ffi::c_void,
        magnitudes: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
    ) -> i32;

    fn parcomm_project_right(
        input: *const i8,
        cols: usize,
        row_start: usize,
        row_end: usize,
        weights: *const std::ffi::c_void,
        magnitudes: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
    ) -> i32;

    fn parcomm_cross_terms_lane_partials(
        input: *const i8,
        base: isize,
        window_low: usize,
        window_high: usize,
        inner: usize,
        outer_start: usize,
        outer_count: usize,
        inner_weights: *const std::ffi::c_void,
        magnitudes: *const std::ffi::c_void,
        partial: *mut std::ffi::c_void,
    ) -> i32;

    fn parcomm_copy_device(destination: *mut u8, source: *const u8, bytes: usize) -> i32;

    fn parcomm_pack_reduced(
        input: *const i8,
        base: isize,
        first: usize,
        limit: usize,
        count: usize,
        inverse_powers: *const std::ffi::c_void,
        magnitudes: *const std::ffi::c_void,
        reduced: *mut std::ffi::c_void,
    ) -> i32;

    fn parcomm_eo_project(
        values: *const std::ffi::c_void,
        len: usize,
        inner: usize,
        outer: usize,
        inner_weights: *const std::ffi::c_void,
        even_partial: *mut std::ffi::c_void,
        odd_partial: *mut std::ffi::c_void,
    ) -> i32;

    fn parcomm_eo_fold(
        input: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        old_len: usize,
        challenge: *const std::ffi::c_void,
    ) -> i32;

    fn parcomm_fr_mul_check(
        left: *const std::ffi::c_void,
        right: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        count: usize,
    ) -> i32;

    fn parcomm_fr_add_check(
        left: *const std::ffi::c_void,
        right: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        count: usize,
    ) -> i32;
}

/// Largest magnitude an i8 witness entry can take, plus the zero slot.
const MAGNITUDE_TABLE_LEN: usize = 129;

/// Device scalar scratch budget for one MSM chunk.
const MSM_SCRATCH_BYTES: usize = 256 << 20;

fn icicle_error(error: impl ToString) -> GpuSmartPcError {
    GpuSmartPcError::Icicle(error.to_string())
}

fn cuda_result(code: i32, stage: &'static str) -> Result<(), GpuSmartPcError> {
    if code == 0 {
        Ok(())
    } else {
        Err(GpuSmartPcError::CudaKernel { stage, code })
    }
}

/// Reinterprets arkworks scalars as ICICLE scalars.
///
/// Both types are four little-endian 64-bit limbs holding a Montgomery form,
/// and the device kernels in `parcomm.cu` use the same representation, so no
/// conversion is needed in either direction.
fn ark_scalars_as_icicle(values: &[Fr]) -> &[IcicleScalarField] {
    assert_eq!(size_of::<Fr>(), size_of::<IcicleScalarField>());
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), values.len()) }
}

fn ark_scalars_as_icicle_mut(values: &mut [Fr]) -> &mut [IcicleScalarField] {
    assert_eq!(size_of::<Fr>(), size_of::<IcicleScalarField>());
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), values.len()) }
}

/// Per-device buffers that live for the whole proof.
struct DeviceState {
    device_id: i32,
    /// First stored row this device owns.
    row_start: usize,
    /// Stored rows this device owns.
    row_count: usize,
    /// `row_count * cols` entries, row-major, for the owned rows only.
    witness: DeviceVec<i8>,
    /// Commitment bases for the owned rows.
    bases: DeviceVec<IcicleG1Affine>,
    magnitudes: DeviceVec<IcicleScalarField>,
}

// The buffers are plain device allocations; the owning thread always selects
// its device before touching them.
unsafe impl Send for DeviceState {}
unsafe impl Sync for DeviceState {}

/// Timings for the GPU-resident stages.
#[derive(Clone, Debug, Default)]
pub struct GpuStageProfile {
    pub dense_seconds: f64,
    pub upload_seconds: f64,
    pub commit_seconds: f64,
    pub commit_msm_seconds: f64,
    pub commit_pairing_seconds: f64,
    pub claim_seconds: f64,
    pub cross_terms_seconds: f64,
    pub project_left_seconds: f64,
    pub project_right_seconds: f64,
    pub witness_bytes: usize,
    pub device_count: usize,
}

/// The packed witness held in device memory across every prover stage.
pub struct GpuWitness {
    rows: usize,
    cols: usize,
    /// Row layout of the stored matrix, needed to translate logical indices.
    layout: WitnessLayout,
    devices: Vec<DeviceState>,
    ark_bases: Arc<Vec<ArkG2Projective>>,
    pub profile: GpuStageProfile,
}

/// The packed witness matrix with its all-zero rows dropped.
///
/// A BVA pads every bucket up to a power-of-two cover, and the protocol treats
/// that suffix as implicit zeros rather than data. A row of the matrix that
/// falls entirely inside such a suffix therefore carries nothing, so it is not
/// stored: the matrix keeps its logical column count and its logical row
/// numbering, but only the rows holding witness entries occupy memory.
///
/// For the geometric benchmark shape this halves both the witness and the
/// commitment's multi-scalar multiplications.
pub struct DenseWitness {
    /// Row-major, `layout.live_rows.len() * layout.cols` entries.
    pub values: Vec<i8>,
    pub layout: WitnessLayout,
}

/// Which logical rows the stored matrix keeps, and where each one landed.
pub struct WitnessLayout {
    /// Logical row index of each stored row, ascending.
    pub live_rows: Vec<u32>,
    /// Stored row index of each logical row, meaningless for dropped rows.
    live_rank: Vec<u32>,
    /// Stored entries of each bucket, keyed by its logical offset.
    spans: Vec<(usize, usize)>,
    /// Logical row count, including the rows that were dropped.
    pub rows: usize,
    pub cols: usize,
}

impl WitnessLayout {
    /// Where a logical index sits in the stored matrix.
    ///
    /// Only valid for indices inside a bucket's stored entries, which is where
    /// every caller reads.
    fn compact(&self, logical: usize) -> usize {
        self.live_rank[logical / self.cols] as usize * self.cols + logical % self.cols
    }

    /// Stored entries of the bucket starting at `logical_offset`.
    fn span(&self, logical_offset: usize) -> Result<usize, GpuSmartPcError> {
        self.spans
            .binary_search_by_key(&logical_offset, |&(offset, _)| offset)
            .map(|index| self.spans[index].1)
            .map_err(|_| GpuSmartPcError::BvaSource)
    }
}

/// Expands a byte-backed BVA into the dense row-major matrix the device
/// kernels index, without materializing the implicit zero suffixes.
pub fn dense_packed_witness<S: BvaI8Source>(
    source: &mut S,
) -> Result<DenseWitness, GpuSmartPcError> {
    let layout: BvaLayout = source.layout().clone();
    let logical = layout.logical_len();
    let cols = 1_usize << (logical.ilog2() as usize / 2);
    let rows = logical / cols;

    let offsets = layout.bucket_offsets();
    let buckets = layout.vector_to_bucket().to_vec();
    let lengths = layout.vector_lengths().to_vec();

    // A row is live when any bucket's stored entries reach it.
    let mut live = vec![false; rows];
    let mut spans = Vec::with_capacity(lengths.len());
    for (vector_index, &length) in lengths.iter().enumerate() {
        let start = offsets[buckets[vector_index]];
        spans.push((start, length));
        for row in start / cols..=(start + length - 1) / cols {
            live[row] = true;
        }
    }
    spans.sort_unstable();

    let mut live_rows = Vec::with_capacity(rows);
    let mut live_rank = vec![0_u32; rows];
    for (row, &is_live) in live.iter().enumerate() {
        if is_live {
            live_rank[row] = live_rows.len() as u32;
            live_rows.push(row as u32);
        }
    }

    let layout = WitnessLayout {
        live_rows,
        live_rank,
        spans,
        rows,
        cols,
    };
    let mut values = vec![0_i8; layout.live_rows.len() * cols];

    // A bucket's stored entries span consecutive live rows, so they stay
    // contiguous after the dropped rows are removed.
    let mut error = None;
    source
        .for_each_vector(&mut |vector_index, entries| {
            if error.is_some() {
                return;
            }
            let Some(&bucket) = buckets.get(vector_index) else {
                error = Some(GpuSmartPcError::BvaVectorIndex { vector_index });
                return;
            };
            let start = layout.compact(offsets[bucket]);
            values[start..start + entries.len()].copy_from_slice(entries);
        })
        .map_err(|_| GpuSmartPcError::BvaSource)?;
    if let Some(error) = error {
        return Err(error);
    }
    Ok(DenseWitness { values, layout })
}

impl GpuWitness {
    /// Uploads the packed witness and the shared bases to every device.
    pub fn upload(
        pp: &UniversalParams<Bls12_381>,
        witness: DenseWitness,
        device_ids: &[i32],
    ) -> Result<Self, GpuSmartPcError> {
        let DenseWitness { values, layout } = witness;
        let (rows, cols) = (layout.rows, layout.cols);
        let dense = values.as_slice();
        if rows == 0 || cols == 0 {
            return Err(GpuSmartPcError::EmptyMatrix);
        }
        if dense.len() != layout.live_rows.len() * cols {
            return Err(GpuSmartPcError::I8MatrixLength {
                expected: layout.live_rows.len() * cols,
                actual: dense.len(),
            });
        }
        if pp.vec_g.len() < rows || pp.vec_h.len() < cols {
            return Err(GpuSmartPcError::ParametersTooShort { rows, columns: cols });
        }
        if device_ids.is_empty() {
            return Err(GpuSmartPcError::NoDevices);
        }

        let mut profile = GpuStageProfile {
            witness_bytes: dense.len(),
            device_count: device_ids.len(),
            ..Default::default()
        };

        // Every device holds a band of the stored rows rather than a copy of the
        // whole matrix. Each band spans all columns, so the commitment and the
        // projections split into partial results the host adds up, and every
        // logical range a kernel reads maps onto whole rows.
        let live = layout.live_rows.len();
        let per_device = live.div_ceil(device_ids.len());
        let row_bands = (0..device_ids.len())
            .map(|index| {
                let start = (index * per_device).min(live);
                (start, (start + per_device).min(live))
            })
            .collect::<Vec<_>>();

        let magnitudes = (0..MAGNITUDE_TABLE_LEN as u64)
            .map(Fr::from)
            .collect::<Vec<_>>();

        let upload_start = Instant::now();
        let layout_ref = &layout;
        let devices = std::thread::scope(|scope| {
            let handles = device_ids
                .iter()
                .zip(&row_bands)
                .map(|(&device_id, &(row_start, row_end))| {
                    let magnitudes = &magnitudes;
                    scope.spawn(move || -> Result<DeviceState, GpuSmartPcError> {
                        icicle_runtime::set_device(&Device::new("CUDA", device_id))
                            .map_err(icicle_error)?;

                        let row_count = row_end - row_start;
                        let band = &dense[row_start * cols..row_end * cols];
                        let mut witness =
                            DeviceVec::<i8>::device_malloc(band.len().max(1))
                                .map_err(icicle_error)?;
                        if !band.is_empty() {
                            witness
                                .copy_from_host(HostSlice::from_slice(band))
                                .map_err(icicle_error)?;
                        }

                        let band_bases = ArkG1Projective::normalize_batch(
                            &layout_ref.live_rows[row_start..row_end]
                                .iter()
                                .map(|&row| pp.vec_g[row as usize])
                                .collect::<Vec<_>>(),
                        );
                        let icicle_bases = ark_g1_to_icicle(&band_bases);
                        let mut bases = DeviceVec::<IcicleG1Affine>::device_malloc(
                            icicle_bases.len().max(1),
                        )
                        .map_err(icicle_error)?;
                        if !icicle_bases.is_empty() {
                            bases
                                .copy_from_host(HostSlice::from_slice(&icicle_bases))
                                .map_err(icicle_error)?;
                        }

                        let mut table = DeviceVec::<IcicleScalarField>::device_malloc(
                            magnitudes.len(),
                        )
                        .map_err(icicle_error)?;
                        table
                            .copy_from_host(HostSlice::from_slice(ark_scalars_as_icicle(
                                magnitudes,
                            )))
                            .map_err(icicle_error)?;

                        Ok(DeviceState {
                            device_id,
                            row_start,
                            row_count,
                            witness,
                            bases,
                            magnitudes: table,
                        })
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .zip(device_ids)
                .map(|(handle, &device_id)| {
                    handle
                        .join()
                        .map_err(|_| GpuSmartPcError::WorkerPanic { device_id })?
                })
                .collect::<Result<Vec<_>, GpuSmartPcError>>()
        })?;
        profile.upload_seconds = upload_start.elapsed().as_secs_f64();

        Ok(Self {
            rows,
            cols,
            layout,
            devices,
            ark_bases: Arc::new(pp.vec_h[..cols].to_vec()),
            profile,
        })
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }


    /// Computes the tier-one column commitments on the GPUs and folds them into
    /// the SmartPC commitment with a CPU pairing product.
    pub fn commit(
        &mut self,
        tilde_u: PairingOutput<Bls12_381>,
        hiding_factor: Fr,
    ) -> Result<(PairingOutput<Bls12_381>, Vec<ArkG1Projective>), GpuSmartPcError> {
        let started = Instant::now();
        let cols = self.cols;

        // A device owns a band of the stored rows and every column, so it
        // produces a partial commitment for each column and the bands are added
        // together rather than concatenated.
        let shards = std::thread::scope(|scope| {
            let handles = self
                .devices
                .iter()
                .map(|state| {
                    scope.spawn(move || -> Result<_, GpuSmartPcError> {
                        icicle_runtime::set_device(&Device::new("CUDA", state.device_id))
                            .map_err(icicle_error)?;
                        let rows = state.row_count;
                        if rows == 0 {
                            return Ok((Vec::new(), 0.0));
                        }

                        let chunk = (MSM_SCRATCH_BYTES
                            / (rows * size_of::<IcicleScalarField>()))
                        .max(1)
                        .min(cols);
                        let mut scalars =
                            DeviceVec::<IcicleScalarField>::device_malloc(chunk * rows)
                                .map_err(icicle_error)?;
                        let mut results = DeviceVec::<IcicleG1Projective>::device_malloc(chunk)
                            .map_err(icicle_error)?;
                        let mut host_results = vec![IcicleG1Projective::zero(); chunk];
                        let mut config = MSMConfig::default();
                        config.bitsize = 8;

                        let mut tier_one = Vec::with_capacity(cols);
                        let mut msm_seconds = 0.0_f64;
                        for column_start in (0..cols).step_by(chunk) {
                            let current = chunk.min(cols - column_start);
                            let mut signed = Vec::with_capacity(2);
                            for negative in [false, true] {
                                let stage_start = Instant::now();
                                cuda_result(
                                    unsafe {
                                        parcomm_expand_i8_colmajor(
                                            state.witness.as_ptr(),
                                            cols,
                                            column_start,
                                            current,
                                            0,
                                            rows,
                                            scalars.as_mut_ptr().cast(),
                                            negative,
                                        )
                                    },
                                    "expand_i8_colmajor",
                                )?;
                                msm(
                                    &scalars[..current * rows],
                                    &state.bases,
                                    &config,
                                    &mut results[..current],
                                )
                                .map_err(icicle_error)?;
                                results[..current]
                                    .copy_to_host(HostSlice::from_mut_slice(
                                        &mut host_results[..current],
                                    ))
                                    .map_err(icicle_error)?;
                                msm_seconds += stage_start.elapsed().as_secs_f64();
                                signed.push(
                                    host_results[..current]
                                        .iter()
                                        .map(icicle_g1_to_ark)
                                        .collect::<Vec<_>>(),
                                );
                            }
                            let negative = signed.pop().expect("negative pass");
                            let positive = signed.pop().expect("positive pass");
                            tier_one.extend(
                                positive
                                    .into_iter()
                                    .zip(negative)
                                    .map(|(plus, minus)| plus - minus),
                            );
                        }

                        Ok((tier_one, msm_seconds))
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().map_err(|_| GpuSmartPcError::ScopedJoin)?)
                .collect::<Result<Vec<_>, GpuSmartPcError>>()
        })?;

        let mut tier_one = vec![ArkG1Projective::zero(); cols];
        for (shard, msm_seconds) in &shards {
            if shard.is_empty() {
                continue;
            }
            tier_one
                .par_iter_mut()
                .zip(shard.par_iter())
                .for_each(|(total, value)| *total += value);
            self.profile.commit_msm_seconds = self.profile.commit_msm_seconds.max(*msm_seconds);
        }

        let pairing_start = Instant::now();
        let pairing_product = inner_pairing_product::<Bls12_381>(&tier_one, &self.ark_bases);
        self.profile.commit_pairing_seconds = pairing_start.elapsed().as_secs_f64();
        self.profile.commit_seconds = started.elapsed().as_secs_f64();
        Ok((pairing_product + tilde_u * hiding_factor, tier_one))
    }

    /// `out[col] = sum_row witness[row][col] * xi(xl)[row]`.
    ///
    /// Each device owns a band of rows and returns a partial column vector; the
    /// CPU only adds the few partial vectors together.
    pub fn project_left(&mut self, weights: &[Fr]) -> Result<Vec<Fr>, GpuSmartPcError> {
        let started = Instant::now();
        let result = self.project(weights, ProjectionSide::Left)?;
        self.profile.project_left_seconds += started.elapsed().as_secs_f64();
        Ok(result)
    }

    /// `out[row] = sum_col witness[row][col] * xi(xr)[col]`.
    ///
    /// Each device owns a band of rows and fills exactly that band of the
    /// output, so the CPU only concatenates.
    pub fn project_right(&mut self, weights: &[Fr]) -> Result<Vec<Fr>, GpuSmartPcError> {
        let started = Instant::now();
        let result = self.project(weights, ProjectionSide::Right)?;
        self.profile.project_right_seconds += started.elapsed().as_secs_f64();
        Ok(result)
    }

    /// Evaluates the global tensor claim `sum_i witness[i] * prod_{bit set in i}
    /// challenges[bit]`.
    ///
    /// A row-major packing splits the index bits into a column half and a row
    /// half, so the tensor is the outer product of two small vectors and the
    /// claim collapses to one right projection followed by a length-`rows` dot
    /// product. That reuses the device kernel instead of streaming the witness
    /// again on the CPU.
    pub fn global_tensor_claim(&mut self, challenges: &[Fr]) -> Result<Fr, GpuSmartPcError> {
        let log_cols = self.cols.ilog2() as usize;
        let log_rows = self.rows.ilog2() as usize;
        if challenges.len() != log_cols + log_rows {
            return Err(GpuSmartPcError::ProjectionWeightLength {
                expected: log_cols + log_rows,
                actual: challenges.len(),
            });
        }
        let column_weights = bitwise_tensor(&challenges[..log_cols]);
        let row_weights = bitwise_tensor(&challenges[log_cols..]);

        let started = Instant::now();
        let row_sums = self.project(&column_weights, ProjectionSide::Right)?;
        let claim = row_weights
            .par_iter()
            .zip(row_sums.par_iter())
            .map(|(weight, sum)| *weight * sum)
            .sum();
        self.profile.claim_seconds += started.elapsed().as_secs_f64();
        Ok(claim)
    }

    /// Computes the packed EO cross-term lane projections for every bucket that
    /// covers at least one whole packed row.
    ///
    /// `projections[bucket][term][lane]` is
    /// `sum_k witness[offset + lane + k * stride] * tensor(row_factors)[k]`.
    /// The bucket tensor is split so the device only ever holds its inner half;
    /// the outer half is applied here to the partials the device returns.
    /// Buckets narrower than the stride stay `None` and the CPU handles them,
    /// because they share a packed row and need cross-bucket terms.
    pub fn packed_lane_projections(
        &mut self,
        requests: &[Option<PackedLaneRequest<Fr>>],
    ) -> Result<PackedLaneProjections<Fr>, GpuSmartPcError> {
        let started = Instant::now();
        let mut projections: PackedLaneProjections<Fr> = vec![None; requests.len()];
        for (bucket_index, request) in requests.iter().enumerate() {
            let Some(request) = request else { continue };
            let mut terms = Vec::with_capacity(request.term_row_factors.len());
            for row_factors in &request.term_row_factors {
                terms.push(self.lane_projection(request, row_factors)?);
            }
            projections[bucket_index] = Some(terms);
        }
        self.profile.cross_terms_seconds += started.elapsed().as_secs_f64();
        Ok(projections)
    }

    /// One bucket, one tensor term: the `stride` lane projections.
    fn lane_projection(
        &self,
        request: &PackedLaneRequest<Fr>,
        row_factors: &[(Fr, Fr)],
    ) -> Result<Vec<Fr>, GpuSmartPcError> {
        let rows = request.cover / PACKING_STRIDE;
        if rows == 0 || !rows.is_power_of_two() || row_factors.len() != rows.ilog2() as usize {
            return Err(GpuSmartPcError::ProjectionWeightLength {
                expected: rows.max(1).ilog2() as usize,
                actual: row_factors.len(),
            });
        }
        let inner_log = CROSS_TERMS_INNER_LOG.min(row_factors.len());
        let inner = 1_usize << inner_log;
        let outer = rows / inner;
        let inner_weights = factor_tensor(&row_factors[..inner_log]);
        let outer_weights = factor_tensor(&row_factors[inner_log..]);

        let base = self.layout.compact(request.logical_offset);
        let limit = self.layout.span(request.logical_offset)?;
        let block = inner * PACKING_STRIDE;
        let cols = self.cols;
        // Each device sums over the part of the bucket that lands in its own
        // rows. The window is measured from the bucket's start, so the two
        // devices sharing a block each return a partial sum for it and the host
        // loop below adds them together.
        let bands = self
            .devices
            .iter()
            .map(|state| {
                let low = (state.row_start * cols).saturating_sub(base);
                let high =
                    limit.min(((state.row_start + state.row_count) * cols).saturating_sub(base));
                if low >= high {
                    return (0, 0, 0, 0);
                }
                (low, high, low / block, outer.min(high.div_ceil(block)))
            })
            .collect::<Vec<_>>();

        let partials = std::thread::scope(|scope| {
            let handles = self
                .devices
                .iter()
                .zip(&bands)
                .map(|(state, &(window_low, window_high, outer_start, outer_end))| {
                    let inner_weights = &inner_weights;
                    scope.spawn(move || -> Result<Vec<Fr>, GpuSmartPcError> {
                        if outer_start >= outer_end {
                            return Ok(Vec::new());
                        }
                        icicle_runtime::set_device(&Device::new("CUDA", state.device_id))
                            .map_err(icicle_error)?;
                        let outer_count = outer_end - outer_start;
                        let device_base = base as isize - (state.row_start * cols) as isize;

                        let mut device_weights =
                            DeviceVec::<IcicleScalarField>::device_malloc(inner)
                                .map_err(icicle_error)?;
                        device_weights
                            .copy_from_host(HostSlice::from_slice(ark_scalars_as_icicle(
                                inner_weights,
                            )))
                            .map_err(icicle_error)?;

                        let slots = outer_count * PACKING_STRIDE;
                        let mut device_partial =
                            DeviceVec::<IcicleScalarField>::device_malloc(slots)
                                .map_err(icicle_error)?;
                        cuda_result(
                            unsafe {
                                parcomm_cross_terms_lane_partials(
                                    state.witness.as_ptr(),
                                    device_base,
                                    window_low,
                                    window_high,
                                    inner,
                                    outer_start,
                                    outer_count,
                                    device_weights.as_ptr().cast(),
                                    state.magnitudes.as_ptr().cast(),
                                    device_partial.as_mut_ptr().cast(),
                                )
                            },
                            "cross_terms_lane_partials",
                        )?;

                        let mut host_partial = vec![Fr::zero(); slots];
                        device_partial
                            .copy_to_host(HostSlice::from_mut_slice(ark_scalars_as_icicle_mut(
                                &mut host_partial,
                            )))
                            .map_err(icicle_error)?;
                        Ok(host_partial)
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().map_err(|_| GpuSmartPcError::ScopedJoin)?)
                .collect::<Result<Vec<_>, GpuSmartPcError>>()
        })?;

        let mut lanes = vec![Fr::zero(); PACKING_STRIDE];
        for ((_, _, outer_start, _), partial) in bands.iter().zip(&partials) {
            for (local, chunk) in partial.chunks_exact(PACKING_STRIDE).enumerate() {
                let weight = outer_weights[outer_start + local];
                for (lane, value) in lanes.iter_mut().zip(chunk) {
                    *lane += weight * value;
                }
            }
        }
        Ok(lanes)
    }

    fn project(
        &self,
        weights: &[Fr],
        side: ProjectionSide,
    ) -> Result<Vec<Fr>, GpuSmartPcError> {
        let rows = self.rows;
        let cols = self.cols;
        let live = self.layout.live_rows.len();
        let expected_weights = match side {
            ProjectionSide::Left => rows,
            ProjectionSide::Right => cols,
        };
        if weights.len() != expected_weights {
            return Err(GpuSmartPcError::ProjectionWeightLength {
                expected: expected_weights,
                actual: weights.len(),
            });
        }
        // `project_left` weights one row each, so each device takes the weights
        // of the rows it owns; `project_right` writes one entry per row, so its
        // band output is scattered back to logical rows below.
        let live_weights = match side {
            ProjectionSide::Left => self
                .layout
                .live_rows
                .par_iter()
                .map(|&row| weights[row as usize])
                .collect::<Vec<_>>(),
            ProjectionSide::Right => weights.to_vec(),
        };
        let weights = live_weights.as_slice();
        let _ = live;

        let partials = std::thread::scope(|scope| {
            let handles = self
                .devices
                .iter()
                .map(|state| {
                    scope.spawn(move || -> Result<(usize, usize, Vec<Fr>), GpuSmartPcError> {
                        icicle_runtime::set_device(&Device::new("CUDA", state.device_id))
                            .map_err(icicle_error)?;
                        let row_start = state.row_start;
                        let row_end = row_start + state.row_count;
                        if state.row_count == 0 {
                            return Ok((row_start, row_end, Vec::new()));
                        }
                        let output_len = match side {
                            ProjectionSide::Left => cols,
                            ProjectionSide::Right => state.row_count,
                        };
                        let band_weights = match side {
                            ProjectionSide::Left => &weights[row_start..row_end],
                            ProjectionSide::Right => weights,
                        };

                        let mut device_weights =
                            DeviceVec::<IcicleScalarField>::device_malloc(band_weights.len())
                                .map_err(icicle_error)?;
                        device_weights
                            .copy_from_host(HostSlice::from_slice(ark_scalars_as_icicle(
                                band_weights,
                            )))
                            .map_err(icicle_error)?;

                        // `project_left` accumulates into its output, so the
                        // band buffer starts at zero.
                        let host_zero = vec![Fr::zero(); output_len];
                        let mut device_output =
                            DeviceVec::<IcicleScalarField>::device_malloc(output_len)
                                .map_err(icicle_error)?;
                        device_output
                            .copy_from_host(HostSlice::from_slice(ark_scalars_as_icicle(
                                &host_zero,
                            )))
                            .map_err(icicle_error)?;

                        let code = unsafe {
                            match side {
                                ProjectionSide::Left => parcomm_project_left(
                                    state.witness.as_ptr(),
                                    cols,
                                    0,
                                    state.row_count,
                                    device_weights.as_ptr().cast(),
                                    state.magnitudes.as_ptr().cast(),
                                    device_output.as_mut_ptr().cast(),
                                ),
                                ProjectionSide::Right => parcomm_project_right(
                                    state.witness.as_ptr(),
                                    cols,
                                    0,
                                    state.row_count,
                                    device_weights.as_ptr().cast(),
                                    state.magnitudes.as_ptr().cast(),
                                    device_output.as_mut_ptr().cast(),
                                ),
                            }
                        };
                        cuda_result(code, "projection")?;

                        let mut host_output = vec![Fr::zero(); output_len];
                        device_output
                            .copy_to_host(HostSlice::from_mut_slice(ark_scalars_as_icicle_mut(
                                &mut host_output,
                            )))
                            .map_err(icicle_error)?;
                        Ok((row_start, row_end, host_output))
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().map_err(|_| GpuSmartPcError::ScopedJoin)?)
                .collect::<Result<Vec<_>, GpuSmartPcError>>()
        })?;

        let mut output = vec![
            Fr::zero();
            match side {
                ProjectionSide::Left => cols,
                ProjectionSide::Right => rows,
            }
        ];
        match side {
            ProjectionSide::Left => {
                for (_, _, partial) in &partials {
                    if partial.is_empty() {
                        continue;
                    }
                    output
                        .par_iter_mut()
                        .zip(partial.par_iter())
                        .for_each(|(total, value)| *total += value);
                }
            }
            ProjectionSide::Right => {
                for (row_start, _, partial) in &partials {
                    for (local, value) in partial.iter().enumerate() {
                        output[self.layout.live_rows[row_start + local] as usize] = *value;
                    }
                }
            }
        }
        Ok(output)
    }
}

#[derive(Clone, Copy)]
enum ProjectionSide {
    Left,
    Right,
}

/// Expands `weights[index] = prod_{bit set in index} challenges[bit]`.
fn bitwise_tensor(challenges: &[Fr]) -> Vec<Fr> {
    let mut weights = vec![Fr::one()];
    for challenge in challenges {
        let half = weights.len();
        weights.extend_from_within(..);
        weights[half..]
            .par_iter_mut()
            .for_each(|slot| *slot *= challenge);
    }
    weights
}

/// Expands `weights[index] = prod_bit factors[bit].(left if the bit is clear
/// else right)`, matching the tensor the packed EO basis carries.
fn factor_tensor(factors: &[(Fr, Fr)]) -> Vec<Fr> {
    let mut weights = vec![Fr::one()];
    for &(left, right) in factors {
        let half = weights.len();
        weights.extend_from_within(..);
        let (low, high) = weights.split_at_mut(half);
        low.par_iter_mut().for_each(|slot| *slot *= left);
        high.par_iter_mut().for_each(|slot| *slot *= right);
    }
    weights
}

/// Packing stride of the packed EO reduction, fixed by the field width and
/// mirrored by the device kernel's warp mapping.
pub const PACKING_STRIDE: usize = 32;

/// Index bits the device kernel folds in its inner loop.
///
/// The rest is applied on the host to the returned partials, which keeps the
/// full-length bucket tensor, up to 2^27 entries, off the device.
const CROSS_TERMS_INNER_LOG: usize = 13;

/// One shard of a reduced vector, living in the memory of a single device.
///
/// A fold cannot write over the values it is still reading without a grid-wide
/// barrier, so it goes through scratch space. That scratch is shared by every
/// shard on the device rather than owned per shard, so the folds on a device
/// run one after another and the result is copied back over the front.
struct ResidentVector {
    device_id: i32,
    /// Index of this shard's first entry in the whole vector.
    start: usize,
    len: usize,
    front: DeviceVec<IcicleScalarField>,
}

/// One reduced vector, split into shards over the devices that built them.
///
/// Folding pairs neighbours, so a shard can fold on its own as long as it
/// starts at an even index and holds an even count. Shard boundaries are
/// therefore kept on multiples of `SHARD_ALIGNMENT`, which also lets each shard
/// weight its own entries with the low half of the round tensor. Both
/// guarantees weaken by one bit per round, so the shards are pulled onto a
/// single device once the vector is small enough for that to be free.
struct DistributedVector {
    shards: Vec<ResidentVector>,
    len: usize,
    retired: bool,
}

/// Shard boundaries are multiples of this, which is what keeps folding and the
/// round tensors shard-local for the rounds that matter.
const SHARD_ALIGNMENT: usize = 1 << 13;

/// Vectors at or below this length live on a single device.
///
/// It is also the point where a split vector is pulled back together, well
/// before the alignment above runs out of bits.
const SHARD_MINIMUM_ENTRIES: usize = 1 << 18;

unsafe impl Send for ResidentVector {}
unsafe impl Sync for ResidentVector {}


/// Keeps the packed EO reduced vectors in device memory for the rounds where
/// they are large.
///
/// Each vector is assigned to one device and never leaves it until folding has
/// made it small, so no witness-sized buffer crosses the bus. The rounds only
/// exchange the two field elements the transcript needs.
pub struct GpuReducedVectors<'a> {
    witness: &'a GpuWitness,
    vectors: Vec<DistributedVector>,
    /// One fold scratch buffer per device, sized to the largest shard it holds.
    scratch: Vec<(i32, DeviceVec<IcicleScalarField>)>,
    pub build_seconds: f64,
    pub round_seconds: f64,
}

/// Index bits an EO round kernel folds in its inner loop.
const EO_INNER_LOG: usize = 13;

impl<'a> GpuReducedVectors<'a> {
    pub fn new(witness: &'a GpuWitness) -> Self {
        Self {
            witness,
            vectors: Vec::new(),
            scratch: Vec::new(),
            build_seconds: 0.0,
            round_seconds: 0.0,
        }
    }

    /// Runs `action` on every shard of every live vector, each on its own
    /// device, and groups the results back per vector.
    fn for_each_live<T, Action>(&mut self, action: Action) -> Result<Vec<Option<Vec<T>>>, String>
    where
        T: Send,
        Action: Fn(usize, &mut ResidentVector) -> Result<T, GpuSmartPcError> + Sync,
    {
        let action = &action;
        std::thread::scope(|scope| {
            let handles = self
                .vectors
                .iter_mut()
                .enumerate()
                .map(|(slot, vector)| {
                    scope.spawn(move || -> Result<Option<Vec<T>>, GpuSmartPcError> {
                        if vector.retired {
                            return Ok(None);
                        }
                        std::thread::scope(|shard_scope| {
                            let handles = vector
                                .shards
                                .iter_mut()
                                .map(|shard| {
                                    shard_scope.spawn(move || -> Result<T, GpuSmartPcError> {
                                        icicle_runtime::set_device(&Device::new(
                                            "CUDA",
                                            shard.device_id,
                                        ))
                                        .map_err(icicle_error)?;
                                        action(slot, shard)
                                    })
                                })
                                .collect::<Vec<_>>();
                            handles
                                .into_iter()
                                .map(|handle| {
                                    handle.join().map_err(|_| GpuSmartPcError::ScopedJoin)?
                                })
                                .collect::<Result<Vec<_>, GpuSmartPcError>>()
                        })
                        .map(Some)
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().map_err(|_| GpuSmartPcError::ScopedJoin)?)
                .collect::<Result<Vec<_>, GpuSmartPcError>>()
        })
        .map_err(|error| error.to_string())
    }

    /// Moves a split vector onto one device once it is small enough that the
    /// round no longer needs the extra memory, which restores the alignment the
    /// folds and round tensors rely on.
    fn consolidate(&mut self) -> Result<(), String> {
        for vector in &mut self.vectors {
            if vector.retired || vector.shards.len() < 2 || vector.len > SHARD_MINIMUM_ENTRIES {
                continue;
            }
            let mut host = vec![Fr::zero(); vector.len];
            for shard in &vector.shards {
                if shard.len == 0 {
                    continue;
                }
                icicle_runtime::set_device(&Device::new("CUDA", shard.device_id))
                    .map_err(|error| icicle_error(error).to_string())?;
                let len = shard.len;
                shard.front[..len]
                    .copy_to_host(HostSlice::from_mut_slice(ark_scalars_as_icicle_mut(
                        &mut host[shard.start..shard.start + len],
                    )))
                    .map_err(|error| icicle_error(error).to_string())?;
            }

            let device_id = vector.shards[0].device_id;
            for shard in vector.shards.drain(..) {
                icicle_runtime::set_device(&Device::new("CUDA", shard.device_id))
                    .map_err(|error| icicle_error(error).to_string())?;
                drop(shard);
            }
            icicle_runtime::set_device(&Device::new("CUDA", device_id))
                .map_err(|error| icicle_error(error).to_string())?;
            let mut front = DeviceVec::<IcicleScalarField>::device_malloc(vector.len)
                .map_err(|error| icicle_error(error).to_string())?;
            front
                .copy_from_host(HostSlice::from_slice(ark_scalars_as_icicle(&host)))
                .map_err(|error| icicle_error(error).to_string())?;
            vector.shards = vec![ResidentVector {
                device_id,
                start: 0,
                len: vector.len,
                front,
            }];
        }
        Ok(())
    }
}

impl PackedEoReducedVectors<Fr> for GpuReducedVectors<'_> {
    type Error = String;

    fn build(
        &mut self,
        specs: &[PackedReducedSpec],
        inverse_powers: &[Fr],
    ) -> Result<(), String> {
        let started = Instant::now();
        let device_count = self.witness.devices.len();
        // The longest vectors go to distinct devices first, which matters
        // because bucket lengths halve down the list.
        let mut order = (0..specs.len()).collect::<Vec<_>>();
        order.sort_by_key(|&index| std::cmp::Reverse(specs[index].len));
        let mut assignment = vec![0_usize; specs.len()];
        for (rank, &index) in order.iter().enumerate() {
            assignment[index] = rank % device_count;
        }

        let witness = self.witness;

        let vectors = std::thread::scope(|scope| {
            let handles = specs
                .iter()
                .zip(&assignment)
                .map(|(spec, &device_index)| {
                    scope.spawn(move || build_vector(witness, spec, inverse_powers, device_index))
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().map_err(|_| GpuSmartPcError::ScopedJoin)?)
                .collect::<Result<Vec<_>, GpuSmartPcError>>()
        })
        .map_err(|error| error.to_string())?;

        for vector in self.vectors.drain(..) {
            for shard in vector.shards {
                icicle_runtime::set_device(&Device::new("CUDA", shard.device_id))
                    .map_err(|error| icicle_error(error).to_string())?;
                drop(shard);
            }
        }
        for (device_id, buffer) in self.scratch.drain(..) {
            icicle_runtime::set_device(&Device::new("CUDA", device_id))
                .map_err(|error| icicle_error(error).to_string())?;
            drop(buffer);
        }

        self.vectors = vectors;
        self.scratch = witness
            .devices
            .iter()
            .map(|state| {
                let widest = self
                    .vectors
                    .iter()
                    .flat_map(|vector| vector.shards.iter())
                    .filter(|shard| shard.device_id == state.device_id)
                    .map(|shard| shard.len.div_ceil(2))
                    .max()
                    .unwrap_or(0);
                icicle_runtime::set_device(&Device::new("CUDA", state.device_id))
                    .map_err(|error| icicle_error(error).to_string())?;
                let buffer = DeviceVec::<IcicleScalarField>::device_malloc(widest.max(1))
                    .map_err(|error| icicle_error(error).to_string())?;
                Ok((state.device_id, buffer))
            })
            .collect::<Result<Vec<_>, String>>()?;
        self.build_seconds += started.elapsed().as_secs_f64();
        Ok(())
    }

    fn weighted_sums(&mut self, factors: &[Vec<(Fr, Fr)>]) -> Result<Vec<Fr>, String> {
        // `sum_i v[i] * T(i)` is the even and odd sums of the same tensor added
        // together once the round factor is dropped, so one kernel serves both
        // this check and the rounds.
        let started = Instant::now();
        let sums = self.for_each_live(|slot, shard| {
            let (even, odd) = device_even_odd(shard, &factors[slot][1..])?;
            let (left, right) = factors[slot][0];
            Ok(even * left + odd * right)
        })?;
        self.round_seconds += started.elapsed().as_secs_f64();
        Ok(sums
            .into_iter()
            .map(|shards| shards.map_or_else(Fr::zero, |shards| shards.into_iter().sum()))
            .collect())
    }

    fn even_odd_sums(&mut self, factors: &[Vec<(Fr, Fr)>]) -> Result<Vec<(Fr, Fr)>, String> {
        let started = Instant::now();
        let sums = self.for_each_live(|slot, shard| device_even_odd(shard, &factors[slot]))?;
        self.round_seconds += started.elapsed().as_secs_f64();
        Ok(sums
            .into_iter()
            .map(|shards| {
                shards.map_or((Fr::zero(), Fr::zero()), |shards| {
                    shards.into_iter().fold(
                        (Fr::zero(), Fr::zero()),
                        |(even, odd), (shard_even, shard_odd)| {
                            (even + shard_even, odd + shard_odd)
                        },
                    )
                })
            })
            .collect())
    }

    fn fold(&mut self, challenge: Fr) -> Result<(), String> {
        let started = Instant::now();
        // Shards on the same device share one scratch buffer, so they fold in
        // turn; different devices still fold at the same time.
        let (vectors, scratch) = (&mut self.vectors, &mut self.scratch);
        let mut grouped = scratch
            .iter()
            .map(|_| Vec::<&mut ResidentVector>::new())
            .collect::<Vec<_>>();
        for vector in vectors.iter_mut() {
            if vector.retired {
                continue;
            }
            for shard in vector.shards.iter_mut() {
                let slot = scratch
                    .iter()
                    .position(|(device_id, _)| *device_id == shard.device_id)
                    .ok_or_else(|| "reduced vector shard has no scratch buffer".to_string())?;
                grouped[slot].push(shard);
            }
        }

        std::thread::scope(|scope| {
            let handles = scratch
                .iter_mut()
                .zip(grouped)
                .map(|((device_id, scratch), shards)| {
                    scope.spawn(move || -> Result<(), GpuSmartPcError> {
                        icicle_runtime::set_device(&Device::new("CUDA", *device_id))
                            .map_err(icicle_error)?;
                        for shard in shards {
                            let old_len = shard.len;
                            if old_len == 0 {
                                continue;
                            }
                            cuda_result(
                                unsafe {
                                    parcomm_eo_fold(
                                        shard.front[..old_len].as_ptr().cast(),
                                        scratch.as_mut_ptr().cast(),
                                        old_len,
                                        ark_scalars_as_icicle(std::slice::from_ref(&challenge))
                                            .as_ptr()
                                            .cast(),
                                    )
                                },
                                "eo_fold",
                            )?;
                            let folded = old_len.div_ceil(2);
                            cuda_result(
                                unsafe {
                                    parcomm_copy_device(
                                        shard.front.as_mut_ptr().cast(),
                                        scratch.as_ptr().cast(),
                                        folded * size_of::<IcicleScalarField>(),
                                    )
                                },
                                "copy_device",
                            )?;
                        }
                        Ok(())
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().map_err(|_| GpuSmartPcError::ScopedJoin)?)
                .collect::<Result<Vec<_>, GpuSmartPcError>>()
        })
        .map_err(|error| error.to_string())?;

        for vector in &mut self.vectors {
            if vector.retired {
                continue;
            }
            vector.len = vector.len.div_ceil(2);
            for shard in &mut vector.shards {
                shard.start /= 2;
                shard.len = shard.len.div_ceil(2);
            }
        }
        self.consolidate()?;
        self.round_seconds += started.elapsed().as_secs_f64();
        Ok(())
    }

    fn retire_below(&mut self, threshold: usize) -> Result<Vec<Option<Vec<Fr>>>, String> {
        let lengths = self
            .vectors
            .iter()
            .map(|vector| vector.len)
            .collect::<Vec<_>>();
        let values = self.for_each_live(|slot, shard| {
            if lengths[slot] > threshold || shard.len == 0 {
                return Ok((shard.start, Vec::new()));
            }
            let mut host = vec![Fr::zero(); shard.len];
            let len = shard.len;
            shard.front[..len]
                .copy_to_host(HostSlice::from_mut_slice(ark_scalars_as_icicle_mut(
                    &mut host,
                )))
                .map_err(icicle_error)?;
            Ok((shard.start, host))
        })?;
        let retired = values
            .into_iter()
            .zip(&lengths)
            .map(|(shards, &len)| {
                let shards = shards?;
                if len > threshold {
                    return None;
                }
                let mut whole = vec![Fr::zero(); len];
                for (start, piece) in shards {
                    whole[start..start + piece.len()].copy_from_slice(&piece);
                }
                Some(whole)
            })
            .collect::<Vec<_>>();
        for (vector, values) in self.vectors.iter_mut().zip(&retired) {
            if values.is_some() {
                vector.retired = true;
            }
        }
        Ok(retired)
    }
}

/// Builds one reduced vector, split over the devices that hold the witness
/// rows it sums.
///
/// A packed entry can only be built where its witness bytes live, so the split
/// follows the row bands. The shard boundaries are rounded to `SHARD_ALIGNMENT`
/// instead, and the few entries that fall on the wrong side of a rounded
/// boundary travel through the host; there are fewer than one alignment block
/// of those per boundary.
fn build_vector(
    witness: &GpuWitness,
    spec: &PackedReducedSpec,
    inverse_powers: &[Fr],
    device_index: usize,
) -> Result<DistributedVector, GpuSmartPcError> {
    let cols = witness.cols;
    let base = witness.layout.compact(spec.logical_offset);
    let span = witness.layout.span(spec.logical_offset)?;

    // Entries this device can build, from the witness rows it stores.
    let built = witness
        .devices
        .iter()
        .map(|state| {
            let low = (state.row_start * cols).saturating_sub(base);
            let high = ((state.row_start + state.row_count) * cols).saturating_sub(base);
            (
                spec.len.min(low.div_ceil(PACKING_STRIDE)),
                spec.len.min(high / PACKING_STRIDE),
            )
        })
        .collect::<Vec<_>>();

    // Entries each shard owns. Short vectors stay whole on their assigned
    // device; the rounds are cheap there and it keeps the tensor alignment.
    let owned = if spec.len > SHARD_MINIMUM_ENTRIES {
        let count = witness.devices.len();
        (0..count)
            .map(|index| {
                let bound = |index: usize| {
                    if index == count {
                        spec.len
                    } else {
                        (spec.len / count * index) / SHARD_ALIGNMENT * SHARD_ALIGNMENT
                    }
                };
                (bound(index), bound(index + 1))
            })
            .collect::<Vec<_>>()
    } else {
        witness
            .devices
            .iter()
            .enumerate()
            .map(|(index, _)| {
                if index == device_index {
                    (0, spec.len)
                } else {
                    (0, 0)
                }
            })
            .collect::<Vec<_>>()
    };

    let mut shards = std::thread::scope(|scope| {
        let handles = witness
            .devices
            .iter()
            .zip(&owned)
            .zip(&built)
            .map(|((state, &(start, end)), &(can_build_start, can_build_end))| {
                let owned = owned.as_slice();
                scope.spawn(move || -> Result<(Option<ResidentVector>, Vec<(usize, Vec<Fr>)>), GpuSmartPcError> {
                    icicle_runtime::set_device(&Device::new("CUDA", state.device_id))
                        .map_err(icicle_error)?;
                    let len = end - start;
                    // A device that owns no entries still builds the ones its
                    // rows cover for other shards, but it allocates nothing.
                    let mut front = if len == 0 {
                        None
                    } else {
                        Some(
                            DeviceVec::<IcicleScalarField>::device_malloc(len)
                                .map_err(icicle_error)?,
                        )
                    };
                    let mut powers =
                        DeviceVec::<IcicleScalarField>::device_malloc(inverse_powers.len())
                            .map_err(icicle_error)?;
                    powers
                        .copy_from_host(HostSlice::from_slice(ark_scalars_as_icicle(
                            inverse_powers,
                        )))
                        .map_err(icicle_error)?;
                    let device_base = base as isize - (state.row_start * cols) as isize;

                    // This device builds what it can for every shard: its own
                    // part lands in place, the rest is handed back for the
                    // owning device to take.
                    let mut foreign = Vec::new();
                    for (target_start, target_end) in owned.iter().copied() {
                        let first = can_build_start.max(target_start);
                        let last = can_build_end.min(target_end);
                        if first >= last {
                            continue;
                        }
                        let count = last - first;
                        let mine = len > 0 && target_start == start && target_end == end;
                        if let (true, Some(front)) = (mine, front.as_mut()) {
                            cuda_result(
                                unsafe {
                                    parcomm_pack_reduced(
                                        state.witness.as_ptr(),
                                        device_base,
                                        first,
                                        span,
                                        count,
                                        powers.as_ptr().cast(),
                                        state.magnitudes.as_ptr().cast(),
                                        front[first - start..last - start].as_mut_ptr().cast(),
                                    )
                                },
                                "pack_reduced",
                            )?;
                        } else {
                            let mut piece =
                                DeviceVec::<IcicleScalarField>::device_malloc(count)
                                    .map_err(icicle_error)?;
                            cuda_result(
                                unsafe {
                                    parcomm_pack_reduced(
                                        state.witness.as_ptr(),
                                        device_base,
                                        first,
                                        span,
                                        count,
                                        powers.as_ptr().cast(),
                                        state.magnitudes.as_ptr().cast(),
                                        piece.as_mut_ptr().cast(),
                                    )
                                },
                                "pack_reduced",
                            )?;
                            let mut host = vec![Fr::zero(); count];
                            piece
                                .copy_to_host(HostSlice::from_mut_slice(
                                    ark_scalars_as_icicle_mut(&mut host),
                                ))
                                .map_err(icicle_error)?;
                            foreign.push((first, host));
                        }
                    }

                    Ok((
                        front.map(|front| ResidentVector {
                            device_id: state.device_id,
                            start,
                            len,
                            front,
                        }),
                        foreign,
                    ))
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().map_err(|_| GpuSmartPcError::ScopedJoin)?)
            .collect::<Result<Vec<_>, GpuSmartPcError>>()
    })?;

    let stragglers = shards
        .iter_mut()
        .flat_map(|(_, foreign)| std::mem::take(foreign))
        .collect::<Vec<_>>();
    let mut shards = shards
        .into_iter()
        .filter_map(|(shard, _)| shard)
        .collect::<Vec<_>>();
    for (first, piece) in stragglers {
        let shard = shards
            .iter_mut()
            .find(|shard| first >= shard.start && first < shard.start + shard.len)
            .ok_or(GpuSmartPcError::ScopedJoin)?;
        icicle_runtime::set_device(&Device::new("CUDA", shard.device_id)).map_err(icicle_error)?;
        let offset = first - shard.start;
        shard.front[offset..offset + piece.len()]
            .copy_from_host(HostSlice::from_slice(ark_scalars_as_icicle(&piece)))
            .map_err(icicle_error)?;
    }

    Ok(DistributedVector {
        shards,
        len: spec.len,
        retired: false,
    })
}

/// `(sum_i v[2i] * T(i), sum_i v[2i+1] * T(i))` for one resident vector.
///
/// The tensor is split so the device only holds its inner half; the outer half
/// is applied here to the partials, which is what keeps a second vector-sized
/// weight buffer off the device.
fn device_even_odd(
    vector: &ResidentVector,
    factors: &[(Fr, Fr)],
) -> Result<(Fr, Fr), GpuSmartPcError> {
    let pairs = vector.len.div_ceil(2);
    if pairs == 0 {
        return Ok((Fr::zero(), Fr::zero()));
    }
    // The device applies the tensor's low bits by position within the shard, so
    // it may only take as many bits as the shard's own starting pair is aligned
    // to; the rest is applied here, offset to where the shard sits.
    let start_pair = vector.start / 2;
    let aligned = if start_pair == 0 {
        usize::BITS as usize
    } else {
        start_pair.trailing_zeros() as usize
    };
    let inner_log = EO_INNER_LOG.min(factors.len()).min(aligned);
    let inner = 1_usize << inner_log;
    let outer = pairs.div_ceil(inner);
    let inner_weights = factor_tensor(&factors[..inner_log]);
    let outer_weights = factor_tensor(&factors[inner_log..]);
    let outer_start = start_pair >> inner_log;

    let mut device_weights =
        DeviceVec::<IcicleScalarField>::device_malloc(inner).map_err(icicle_error)?;
    device_weights
        .copy_from_host(HostSlice::from_slice(ark_scalars_as_icicle(&inner_weights)))
        .map_err(icicle_error)?;
    let mut device_even =
        DeviceVec::<IcicleScalarField>::device_malloc(outer).map_err(icicle_error)?;
    let mut device_odd =
        DeviceVec::<IcicleScalarField>::device_malloc(outer).map_err(icicle_error)?;

    let len = vector.len;
    cuda_result(
        unsafe {
            parcomm_eo_project(
                vector.front[..len].as_ptr().cast(),
                len,
                inner,
                outer,
                device_weights.as_ptr().cast(),
                device_even.as_mut_ptr().cast(),
                device_odd.as_mut_ptr().cast(),
            )
        },
        "eo_project",
    )?;

    let mut even_partial = vec![Fr::zero(); outer];
    let mut odd_partial = vec![Fr::zero(); outer];
    device_even
        .copy_to_host(HostSlice::from_mut_slice(ark_scalars_as_icicle_mut(
            &mut even_partial,
        )))
        .map_err(icicle_error)?;
    device_odd
        .copy_to_host(HostSlice::from_mut_slice(ark_scalars_as_icicle_mut(
            &mut odd_partial,
        )))
        .map_err(icicle_error)?;

    let weights = &outer_weights[outer_start.min(outer_weights.len())..];
    let even = weights
        .par_iter()
        .zip(even_partial.par_iter())
        .map(|(weight, partial)| *weight * partial)
        .sum();
    let odd = weights
        .par_iter()
        .zip(odd_partial.par_iter())
        .map(|(weight, partial)| *weight * partial)
        .sum();
    Ok((even, odd))
}

/// Runs the device field arithmetic on `pairs` and returns the products and
/// sums, so a caller can compare them against arkworks.
///
/// The kernels read and write the same Montgomery limbs arkworks uses, which is
/// what lets the projection kernels multiply by a host-built magnitude table.
pub fn device_field_roundtrip(
    device_id: i32,
    left: &[Fr],
    right: &[Fr],
) -> Result<(Vec<Fr>, Vec<Fr>), GpuSmartPcError> {
    assert_eq!(left.len(), right.len());
    icicle_runtime::set_device(&Device::new("CUDA", device_id)).map_err(icicle_error)?;
    let count = left.len();

    let mut device_left =
        DeviceVec::<IcicleScalarField>::device_malloc(count).map_err(icicle_error)?;
    device_left
        .copy_from_host(HostSlice::from_slice(ark_scalars_as_icicle(left)))
        .map_err(icicle_error)?;
    let mut device_right =
        DeviceVec::<IcicleScalarField>::device_malloc(count).map_err(icicle_error)?;
    device_right
        .copy_from_host(HostSlice::from_slice(ark_scalars_as_icicle(right)))
        .map_err(icicle_error)?;
    let mut device_output =
        DeviceVec::<IcicleScalarField>::device_malloc(count).map_err(icicle_error)?;

    let mut products = vec![Fr::zero(); count];
    cuda_result(
        unsafe {
            parcomm_fr_mul_check(
                device_left.as_ptr().cast(),
                device_right.as_ptr().cast(),
                device_output.as_mut_ptr().cast(),
                count,
            )
        },
        "fr_mul_check",
    )?;
    device_output
        .copy_to_host(HostSlice::from_mut_slice(ark_scalars_as_icicle_mut(
            &mut products,
        )))
        .map_err(icicle_error)?;

    let mut sums = vec![Fr::zero(); count];
    cuda_result(
        unsafe {
            parcomm_fr_add_check(
                device_left.as_ptr().cast(),
                device_right.as_ptr().cast(),
                device_output.as_mut_ptr().cast(),
                count,
            )
        },
        "fr_add_check",
    )?;
    device_output
        .copy_to_host(HostSlice::from_mut_slice(ark_scalars_as_icicle_mut(
            &mut sums,
        )))
        .map_err(icicle_error)?;

    Ok((products, sums))
}

#[cfg(test)]
mod tests {
    use ark_ff::UniformRand;
    use ark_std::rand::SeedableRng;

    use super::*;

    #[test]
    fn device_field_arithmetic_matches_arkworks() {
        crate::initialize_cuda_backend().expect("load CUDA backend");
        let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(7);
        let count = 1 << 16;
        let mut left = (0..count).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();
        let mut right = (0..count).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();

        // Boundary values random sampling will not reach, plus the magnitude
        // table the projection kernels index.
        left[0] = Fr::zero();
        right[0] = -Fr::from(1_u64);
        left[1] = -Fr::from(1_u64);
        right[1] = -Fr::from(1_u64);
        for magnitude in 0..MAGNITUDE_TABLE_LEN {
            right[2 + magnitude] = Fr::from(magnitude as u64);
        }

        let (products, sums) = device_field_roundtrip(0, &left, &right).expect("device arithmetic");
        for index in 0..count {
            assert_eq!(products[index], left[index] * right[index], "mul at {index}");
            assert_eq!(sums[index], left[index] + right[index], "add at {index}");
        }
    }
}
