use std::ffi::c_void;
use std::mem::size_of;
use std::sync::{Arc, Barrier};
use std::time::Instant;

use ark_bls12_381::{
    Bls12_381, Fr, G1Projective as ArkG1Projective, G2Projective as ArkG2Projective,
};
use ark_ec::pairing::PairingOutput;
use ark_ec::CurveGroup;
use ark_std::Zero;
use poly_commit::smart_pc::{utils::inner_pairing_product, UniversalParams};
use icicle_bls12_381::curve::{
    G1Affine as IcicleG1Affine, G1Projective as IcicleG1Projective,
    ScalarField as IcicleScalarField,
};
use icicle_core::msm::{msm, MSMConfig};
use icicle_core::projective::Projective;
use icicle_runtime::memory::{DeviceVec, HostOrDeviceSlice, HostSlice};
use icicle_runtime::Device;

use crate::smart_pc::{ark_g1_to_icicle, icicle_g1_to_ark, GpuSmartPcError};

#[link(name = "smartpc_i8_cuda", kind = "static")]
extern "C" {
    fn smartpc_expand_i8_device(
        input: *const i8,
        offset: usize,
        count: usize,
        output: *mut c_void,
        negative: bool,
    ) -> i32;
}

#[derive(Clone, Debug)]
pub struct I8GpuShardProfile {
    pub device_id: i32,
    pub column_start: usize,
    pub column_end: usize,
    pub raw_input_bytes: usize,
    pub scalar_scratch_bytes: usize,
    pub raw_upload_seconds: f64,
    pub scalar_expansion_seconds: f64,
    pub positive_msm_seconds: f64,
    pub negative_msm_seconds: f64,
    pub result_download_seconds: f64,
    pub cpu_pairing_seconds: f64,
    pub total_seconds: f64,
}

#[derive(Clone, Debug)]
pub struct I8GpuCommitmentProfile {
    pub rows: usize,
    pub columns: usize,
    pub total_elements: usize,
    pub chunk_columns: usize,
    pub g1_conversion_seconds: f64,
    pub parallel_stage_seconds: f64,
    pub constant_merge_seconds: f64,
    pub total_seconds: f64,
    pub shards: Vec<I8GpuShardProfile>,
}

#[derive(Clone, Debug)]
pub struct I8GpuCommitment {
    pub commitment: PairingOutput<Bls12_381>,
    pub tier_one: Vec<ArkG1Projective>,
    pub profile: I8GpuCommitmentProfile,
}

struct I8ShardResult {
    start: usize,
    tier_one: Vec<ArkG1Projective>,
    pairing_product: PairingOutput<Bls12_381>,
    profile: I8GpuShardProfile,
}

fn expand_i8(
    input: &DeviceVec<i8>,
    offset: usize,
    count: usize,
    output: &mut DeviceVec<IcicleScalarField>,
    negative: bool,
) -> Result<(), GpuSmartPcError> {
    let rc = unsafe {
        smartpc_expand_i8_device(
            input.as_ptr(),
            offset,
            count,
            output.as_mut_ptr().cast(),
            negative,
        )
    };
    if rc == 0 {
        Ok(())
    } else {
        Err(GpuSmartPcError::CudaI8Expansion(rc))
    }
}

fn i8_worker(
    device_id: i32,
    start: usize,
    end: usize,
    rows: usize,
    chunk_columns: usize,
    matrix: &[i8],
    g1_bases: Arc<Vec<IcicleG1Affine>>,
    g2_bases: Arc<Vec<ArkG2Projective>>,
    ready: Arc<Barrier>,
) -> Result<I8ShardResult, GpuSmartPcError> {
    let worker_start = Instant::now();
    icicle_runtime::set_device(&Device::new("CUDA", device_id))
        .map_err(|error| GpuSmartPcError::Icicle(error.to_string()))?;

    let columns = end - start;
    let shard = &matrix[start * rows..end * rows];
    let raw_upload_start = Instant::now();
    let mut device_input = DeviceVec::<i8>::device_malloc(shard.len())
        .map_err(|error| GpuSmartPcError::Icicle(error.to_string()))?;
    device_input
        .copy_from_host(HostSlice::from_slice(shard))
        .map_err(|error| GpuSmartPcError::Icicle(error.to_string()))?;

    let mut device_g1 = DeviceVec::<IcicleG1Affine>::device_malloc(g1_bases.len())
        .map_err(|error| GpuSmartPcError::Icicle(error.to_string()))?;
    device_g1
        .copy_from_host(HostSlice::from_slice(&g1_bases))
        .map_err(|error| GpuSmartPcError::Icicle(error.to_string()))?;
    let raw_upload_seconds = raw_upload_start.elapsed().as_secs_f64();

    let chunk_columns = chunk_columns.min(columns);
    let max_chunk_scalars = chunk_columns * rows;
    let mut device_scalars = DeviceVec::<IcicleScalarField>::device_malloc(max_chunk_scalars)
        .map_err(|error| GpuSmartPcError::Icicle(error.to_string()))?;
    let mut device_results = DeviceVec::<IcicleG1Projective>::device_malloc(chunk_columns)
        .map_err(|error| GpuSmartPcError::Icicle(error.to_string()))?;
    let mut host_results = vec![IcicleG1Projective::zero(); chunk_columns];

    let mut config = MSMConfig::default();
    config.bitsize = 8;
    ready.wait();

    let mut tier_one = Vec::with_capacity(columns);
    let mut scalar_expansion_seconds = 0.0;
    let mut positive_msm_seconds = 0.0;
    let mut negative_msm_seconds = 0.0;
    let mut result_download_seconds = 0.0;

    for local_column_start in (0..columns).step_by(chunk_columns) {
        let current_columns = chunk_columns.min(columns - local_column_start);
        let scalar_count = current_columns * rows;
        let scalar_offset = local_column_start * rows;

        let expansion_start = Instant::now();
        expand_i8(
            &device_input,
            scalar_offset,
            scalar_count,
            &mut device_scalars,
            false,
        )?;
        scalar_expansion_seconds += expansion_start.elapsed().as_secs_f64();

        let msm_start = Instant::now();
        msm(
            &device_scalars[..scalar_count],
            &device_g1,
            &config,
            &mut device_results[..current_columns],
        )
        .map_err(|error| GpuSmartPcError::Icicle(error.to_string()))?;
        positive_msm_seconds += msm_start.elapsed().as_secs_f64();

        let download_start = Instant::now();
        device_results[..current_columns]
            .copy_to_host(HostSlice::from_mut_slice(
                &mut host_results[..current_columns],
            ))
            .map_err(|error| GpuSmartPcError::Icicle(error.to_string()))?;
        let positive = host_results[..current_columns]
            .iter()
            .map(icicle_g1_to_ark)
            .collect::<Vec<_>>();
        result_download_seconds += download_start.elapsed().as_secs_f64();

        let expansion_start = Instant::now();
        expand_i8(
            &device_input,
            scalar_offset,
            scalar_count,
            &mut device_scalars,
            true,
        )?;
        scalar_expansion_seconds += expansion_start.elapsed().as_secs_f64();

        let msm_start = Instant::now();
        msm(
            &device_scalars[..scalar_count],
            &device_g1,
            &config,
            &mut device_results[..current_columns],
        )
        .map_err(|error| GpuSmartPcError::Icicle(error.to_string()))?;
        negative_msm_seconds += msm_start.elapsed().as_secs_f64();

        let download_start = Instant::now();
        device_results[..current_columns]
            .copy_to_host(HostSlice::from_mut_slice(
                &mut host_results[..current_columns],
            ))
            .map_err(|error| GpuSmartPcError::Icicle(error.to_string()))?;
        tier_one.extend(
            positive
                .into_iter()
                .zip(host_results[..current_columns].iter().map(icicle_g1_to_ark))
                .map(|(positive, negative)| positive - negative),
        );
        result_download_seconds += download_start.elapsed().as_secs_f64();
    }

    let pairing_start = Instant::now();
    let pairing_product = inner_pairing_product::<Bls12_381>(&tier_one, &g2_bases[start..end]);
    let cpu_pairing_seconds = pairing_start.elapsed().as_secs_f64();

    Ok(I8ShardResult {
        start,
        tier_one,
        pairing_product,
        profile: I8GpuShardProfile {
            device_id,
            column_start: start,
            column_end: end,
            raw_input_bytes: shard.len(),
            scalar_scratch_bytes: max_chunk_scalars * size_of::<IcicleScalarField>(),
            raw_upload_seconds,
            scalar_expansion_seconds,
            positive_msm_seconds,
            negative_msm_seconds,
            result_download_seconds,
            cpu_pairing_seconds,
            total_seconds: worker_start.elapsed().as_secs_f64(),
        },
    })
}

pub fn commit_i8_four_gpu(
    pp: &UniversalParams<Bls12_381>,
    matrix: &[i8],
    rows: usize,
    columns: usize,
    hiding_factor: Fr,
    device_ids: [i32; 4],
    chunk_columns: usize,
) -> Result<I8GpuCommitment, GpuSmartPcError> {
    if rows == 0 || columns == 0 {
        return Err(GpuSmartPcError::EmptyMatrix);
    }
    if chunk_columns == 0 {
        return Err(GpuSmartPcError::EmptyI8Chunk);
    }
    let expected = rows
        .checked_mul(columns)
        .ok_or(GpuSmartPcError::I8MatrixLength {
            expected: usize::MAX,
            actual: matrix.len(),
        })?;
    if matrix.len() != expected {
        return Err(GpuSmartPcError::I8MatrixLength {
            expected,
            actual: matrix.len(),
        });
    }
    if columns % device_ids.len() != 0 {
        return Err(GpuSmartPcError::UnevenColumns { columns });
    }
    if pp.vec_g.len() < rows || pp.vec_h.len() < columns {
        return Err(GpuSmartPcError::ParametersTooShort { rows, columns });
    }

    let total_start = Instant::now();
    let g1_conversion_start = Instant::now();
    let ark_g1 = ArkG1Projective::normalize_batch(&pp.vec_g[..rows]);
    let g1_bases = Arc::new(ark_g1_to_icicle(&ark_g1));
    let g1_conversion_seconds = g1_conversion_start.elapsed().as_secs_f64();
    let g2_bases = Arc::new(pp.vec_h[..columns].to_vec());

    let ready = Arc::new(Barrier::new(device_ids.len()));
    let columns_per_device = columns / device_ids.len();
    let parallel_start = Instant::now();
    let mut results = std::thread::scope(|scope| {
        let handles = device_ids
            .into_iter()
            .enumerate()
            .map(|(index, device_id)| {
                let start = index * columns_per_device;
                let end = start + columns_per_device;
                let g1_bases = Arc::clone(&g1_bases);
                let g2_bases = Arc::clone(&g2_bases);
                let ready = Arc::clone(&ready);
                scope.spawn(move || {
                    i8_worker(
                        device_id,
                        start,
                        end,
                        rows,
                        chunk_columns,
                        matrix,
                        g1_bases,
                        g2_bases,
                        ready,
                    )
                })
            })
            .collect::<Vec<_>>();

        handles
            .into_iter()
            .zip(device_ids)
            .map(|(handle, device_id)| {
                handle
                    .join()
                    .map_err(|_| GpuSmartPcError::WorkerPanic { device_id })?
            })
            .collect::<Result<Vec<_>, GpuSmartPcError>>()
    })?;
    let parallel_stage_seconds = parallel_start.elapsed().as_secs_f64();

    results.sort_by_key(|result| result.start);
    let merge_start = Instant::now();
    let mut pairing_product = PairingOutput::<Bls12_381>::zero();
    let mut tier_one = Vec::with_capacity(columns);
    let mut shards = Vec::with_capacity(device_ids.len());
    for result in results {
        pairing_product += result.pairing_product;
        tier_one.extend(result.tier_one);
        shards.push(result.profile);
    }
    let commitment = pairing_product + pp.tilde_u * hiding_factor;
    let constant_merge_seconds = merge_start.elapsed().as_secs_f64();
    let total_seconds = total_start.elapsed().as_secs_f64();

    Ok(I8GpuCommitment {
        commitment,
        tier_one,
        profile: I8GpuCommitmentProfile {
            rows,
            columns,
            total_elements: expected,
            chunk_columns,
            g1_conversion_seconds,
            parallel_stage_seconds,
            constant_merge_seconds,
            total_seconds,
            shards,
        },
    })
}
