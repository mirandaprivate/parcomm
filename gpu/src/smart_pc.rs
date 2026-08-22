use std::mem::size_of;
use std::sync::{Arc, Barrier};
use std::time::Instant;

use ark_bls12_381::{
    Bls12_381, Fq, Fr, G1Affine as ArkG1Affine, G1Projective as ArkG1Projective,
    G2Projective as ArkG2Projective,
};
use ark_ec::pairing::PairingOutput;
use ark_ec::CurveGroup;
use ark_ff::{BigInteger, Field, PrimeField};
use ark_std::Zero;
use poly_commit::smart_pc::{utils::inner_pairing_product, UniversalParams};
use icicle_bls12_381::curve::{
    BaseField as IcicleBaseField, G1Affine as IcicleG1Affine, G1Projective as IcicleG1Projective,
    ScalarField as IcicleScalarField,
};
use icicle_core::bignum::BigNum;
use icicle_core::field::Field as IcicleField;
use icicle_core::msm::{msm, MSMConfig};
use icicle_core::projective::Projective;
use icicle_core::traits::MontgomeryConvertible;
use icicle_runtime::memory::{DeviceVec, HostSlice};
use icicle_runtime::stream::IcicleStream;
use icicle_runtime::Device;
use rayon::prelude::*;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GpuSmartPcError {
    #[error("matrix must contain at least one column and one row")]
    EmptyMatrix,
    #[error("matrix is ragged at column {column}: expected {expected} rows, found {actual}")]
    RaggedMatrix {
        column: usize,
        expected: usize,
        actual: usize,
    },
    #[error("the number of columns ({columns}) must be divisible by four")]
    UnevenColumns { columns: usize },
    #[error("i8 matrix length mismatch: expected {expected}, found {actual}")]
    I8MatrixLength { expected: usize, actual: usize },
    #[error("i8 MSM chunk size must contain at least one column")]
    EmptyI8Chunk,
    #[error("SmartPC parameters are too short: rows={rows}, columns={columns}")]
    ParametersTooShort { rows: usize, columns: usize },
    #[error("ICICLE error: {0}")]
    Icicle(String),
    #[error("CUDA i8 expansion failed with cudaError_t={0}")]
    CudaI8Expansion(i32),
    #[error("CUDA {stage} failed with cudaError_t={code}")]
    CudaKernel { stage: &'static str, code: i32 },
    #[error("GPU worker {device_id} panicked")]
    WorkerPanic { device_id: i32 },
    #[error("a GPU worker panicked")]
    ScopedJoin,
    #[error("at least one CUDA device is required")]
    NoDevices,
    #[error("the BVA source reported vector index {vector_index} outside its layout")]
    BvaVectorIndex { vector_index: usize },
    #[error("the BVA source failed to replay its vectors")]
    BvaSource,
    #[error("projection weight length mismatch: expected {expected}, found {actual}")]
    ProjectionWeightLength { expected: usize, actual: usize },
}

#[derive(Clone, Debug)]
pub struct GpuShardProfile {
    pub device_id: i32,
    pub column_start: usize,
    pub column_end: usize,
    pub scalar_bytes: usize,
    pub upload_seconds: f64,
    pub tier_one_msm_seconds: f64,
    pub tier_one_download_seconds: f64,
    pub cpu_pairing_seconds: f64,
    pub total_seconds: f64,
}

#[derive(Clone, Debug)]
pub struct GpuCommitmentProfile {
    pub rows: usize,
    pub columns: usize,
    pub g1_conversion_seconds: f64,
    pub parallel_stage_seconds: f64,
    pub hybrid_commit_seconds: f64,
    pub constant_merge_seconds: f64,
    pub shards: Vec<GpuShardProfile>,
}

#[derive(Clone, Debug)]
pub struct GpuCommitment {
    pub commitment: PairingOutput<Bls12_381>,
    pub tier_one: Vec<ArkG1Projective>,
    pub profile: GpuCommitmentProfile,
}

struct ShardResult {
    start: usize,
    tier_one: Vec<ArkG1Projective>,
    pairing_product: PairingOutput<Bls12_381>,
    profile: GpuShardProfile,
}

pub fn initialize_cuda_backend() -> Result<(), GpuSmartPcError> {
    icicle_runtime::runtime::load_backend_from_env_or_default()
        .map_err(|error| GpuSmartPcError::Icicle(error.to_string()))
}

fn to_icicle_field<T, I>(value: &T) -> I
where
    T: Field,
    I: IcicleField,
{
    let mut bytes = Vec::new();
    for base_element in value.to_base_prime_field_elements() {
        bytes.extend_from_slice(&base_element.into_bigint().to_bytes_le());
    }
    I::from_bytes_le(&bytes)
}

pub(crate) fn ark_g1_to_icicle(points: &[ArkG1Affine]) -> Vec<IcicleG1Affine> {
    points
        .par_iter()
        .map(|point| IcicleG1Affine {
            x: to_icicle_field::<Fq, IcicleBaseField>(&point.x),
            y: to_icicle_field::<Fq, IcicleBaseField>(&point.y),
        })
        .collect()
}

pub(crate) fn icicle_g1_to_ark(point: &IcicleG1Projective) -> ArkG1Projective {
    let x = Fq::from_random_bytes(&point.x.to_bytes_le()).unwrap();
    let y = Fq::from_random_bytes(&point.y.to_bytes_le()).unwrap();
    let z = Fq::from_random_bytes(&point.z.to_bytes_le()).unwrap();
    if z.is_zero() {
        return ArkG1Projective::zero();
    }
    ArkG1Projective::new_unchecked(x * z, y * z.square(), z)
}

unsafe fn ark_scalars_as_icicle(values: &[Fr]) -> &[IcicleScalarField] {
    assert_eq!(size_of::<Fr>(), size_of::<IcicleScalarField>());
    std::slice::from_raw_parts(values.as_ptr().cast(), values.len())
}

fn worker(
    device_id: i32,
    start: usize,
    end: usize,
    rows: usize,
    matrix: Arc<Vec<Fr>>,
    g1_bases: Arc<Vec<IcicleG1Affine>>,
    g2_bases: Arc<Vec<ArkG2Projective>>,
    ready: Arc<Barrier>,
) -> Result<ShardResult, GpuSmartPcError> {
    let worker_start = Instant::now();
    icicle_runtime::set_device(&Device::new("CUDA", device_id))
        .map_err(|error| GpuSmartPcError::Icicle(error.to_string()))?;

    let columns = end - start;
    let scalar_start = start * rows;
    let scalar_end = end * rows;
    let ark_scalars = &matrix[scalar_start..scalar_end];
    let icicle_scalars = unsafe { ark_scalars_as_icicle(ark_scalars) };

    let upload_start = Instant::now();
    let mut device_scalars = DeviceVec::<IcicleScalarField>::device_malloc(icicle_scalars.len())
        .map_err(|error| GpuSmartPcError::Icicle(error.to_string()))?;
    device_scalars
        .copy_from_host(HostSlice::from_slice(icicle_scalars))
        .map_err(|error| GpuSmartPcError::Icicle(error.to_string()))?;
    IcicleScalarField::from_mont(&mut device_scalars, &IcicleStream::default())
        .map_err(|error| GpuSmartPcError::Icicle(error.to_string()))?;

    let mut device_g1 = DeviceVec::<IcicleG1Affine>::device_malloc(g1_bases.len())
        .map_err(|error| GpuSmartPcError::Icicle(error.to_string()))?;
    device_g1
        .copy_from_host(HostSlice::from_slice(&g1_bases))
        .map_err(|error| GpuSmartPcError::Icicle(error.to_string()))?;

    let mut device_tier_one = DeviceVec::<IcicleG1Projective>::device_malloc(columns)
        .map_err(|error| GpuSmartPcError::Icicle(error.to_string()))?;
    let upload_seconds = upload_start.elapsed().as_secs_f64();

    ready.wait();

    let msm_start = Instant::now();
    msm(
        &device_scalars,
        &device_g1,
        &MSMConfig::default(),
        &mut device_tier_one,
    )
    .map_err(|error| GpuSmartPcError::Icicle(error.to_string()))?;
    let tier_one_msm_seconds = msm_start.elapsed().as_secs_f64();

    let download_start = Instant::now();
    let mut host_tier_one = vec![IcicleG1Projective::zero(); columns];
    device_tier_one
        .copy_to_host(HostSlice::from_mut_slice(&mut host_tier_one))
        .map_err(|error| GpuSmartPcError::Icicle(error.to_string()))?;
    let tier_one: Vec<ArkG1Projective> = host_tier_one.par_iter().map(icicle_g1_to_ark).collect();
    let tier_one_download_seconds = download_start.elapsed().as_secs_f64();

    let pairing_start = Instant::now();
    let pairing_product = inner_pairing_product::<Bls12_381>(&tier_one, &g2_bases[start..end]);
    let cpu_pairing_seconds = pairing_start.elapsed().as_secs_f64();

    Ok(ShardResult {
        start,
        tier_one,
        pairing_product,
        profile: GpuShardProfile {
            device_id,
            column_start: start,
            column_end: end,
            scalar_bytes: icicle_scalars.len() * size_of::<IcicleScalarField>(),
            upload_seconds,
            tier_one_msm_seconds,
            tier_one_download_seconds,
            cpu_pairing_seconds,
            total_seconds: worker_start.elapsed().as_secs_f64(),
        },
    })
}

pub fn commit_full_four_gpu(
    pp: &UniversalParams<Bls12_381>,
    matrix: &[Vec<Fr>],
    hiding_factor: Fr,
    device_ids: [i32; 4],
) -> Result<GpuCommitment, GpuSmartPcError> {
    let rows = matrix
        .first()
        .map(Vec::len)
        .ok_or(GpuSmartPcError::EmptyMatrix)?;
    if rows == 0 {
        return Err(GpuSmartPcError::EmptyMatrix);
    }
    for (column, values) in matrix.iter().enumerate() {
        if values.len() != rows {
            return Err(GpuSmartPcError::RaggedMatrix {
                column,
                expected: rows,
                actual: values.len(),
            });
        }
    }
    let columns = matrix.len();
    if columns % device_ids.len() != 0 {
        return Err(GpuSmartPcError::UnevenColumns { columns });
    }
    if pp.vec_g.len() < rows || pp.vec_h.len() < columns {
        return Err(GpuSmartPcError::ParametersTooShort { rows, columns });
    }

    let hybrid_commit_start = Instant::now();
    let g1_conversion_start = Instant::now();
    let ark_g1 = ArkG1Projective::normalize_batch(&pp.vec_g[..rows]);
    let g1_bases = Arc::new(ark_g1_to_icicle(&ark_g1));
    let g1_conversion_seconds = g1_conversion_start.elapsed().as_secs_f64();

    let g2_bases = Arc::new(pp.vec_h[..columns].to_vec());

    let flat_matrix = Arc::new(matrix.iter().flatten().copied().collect::<Vec<_>>());
    let ready = Arc::new(Barrier::new(device_ids.len()));
    let columns_per_device = columns / device_ids.len();
    let parallel_stage_start = Instant::now();

    let mut results = std::thread::scope(|scope| {
        let handles = device_ids
            .into_iter()
            .enumerate()
            .map(|(index, device_id)| {
                let start = index * columns_per_device;
                let end = start + columns_per_device;
                let matrix = Arc::clone(&flat_matrix);
                let g1_bases = Arc::clone(&g1_bases);
                let g2_bases = Arc::clone(&g2_bases);
                let ready = Arc::clone(&ready);
                scope.spawn(move || {
                    worker(
                        device_id, start, end, rows, matrix, g1_bases, g2_bases, ready,
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
    let parallel_stage_seconds = parallel_stage_start.elapsed().as_secs_f64();

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
    let hybrid_commit_seconds = hybrid_commit_start.elapsed().as_secs_f64();

    Ok(GpuCommitment {
        commitment,
        tier_one,
        profile: GpuCommitmentProfile {
            rows,
            columns,
            g1_conversion_seconds,
            parallel_stage_seconds,
            hybrid_commit_seconds,
            constant_merge_seconds,
            shards,
        },
    })
}
