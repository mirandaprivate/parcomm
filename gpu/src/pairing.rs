//! Batched BLS12-381 Miller loops on the device.
//!
//! An inner pairing product is one multi-Miller loop followed by a single final
//! exponentiation. Only the Miller loops grow with the input, so they run on the
//! device and the lone final exponentiation stays on the host, where arkworks
//! already has a tuned implementation.

use std::mem::size_of;
use std::time::Instant;

use ark_bls12_381::{Bls12_381, Fq, Fq2, Fr, G1Affine as ArkG1Affine, G2Affine as ArkG2Affine};
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ff::{Field, One, PrimeField, Zero};
use icicle_runtime::memory::{DeviceVec, HostOrDeviceSlice, HostSlice};
use icicle_runtime::Device;
use rayon::prelude::*;

use crate::smart_pc::GpuSmartPcError;

/// Limbs in one BLS12-381 base field element.
const BASE_FIELD_LIMBS: usize = 6;

/// Base field elements in the layout the device kernels read.
///
/// The kernels keep the tower in Montgomery form, which is exactly how `ark_ff`
/// stores `Fq`, so the limbs cross the boundary unchanged.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct DeviceFq {
    limbs: [u64; BASE_FIELD_LIMBS],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct DeviceFq2 {
    c0: DeviceFq,
    c1: DeviceFq,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct DeviceG1Affine {
    x: DeviceFq,
    y: DeviceFq,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct DeviceG2Affine {
    x: DeviceFq2,
    y: DeviceFq2,
}

/// The twelve base field coefficients of an Fp12 element, ordered the way the
/// nested tower lays them out in memory.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct DeviceFq12 {
    coefficients: [DeviceFq; 12],
}

fn to_device_fq(value: &Fq) -> DeviceFq {
    DeviceFq { limbs: value.0 .0 }
}

fn from_device_fq(value: &DeviceFq) -> Fq {
    Fq::new_unchecked(ark_ff::BigInt(value.limbs))
}

fn to_device_fq2(value: &Fq2) -> DeviceFq2 {
    DeviceFq2 {
        c0: to_device_fq(&value.c0),
        c1: to_device_fq(&value.c1),
    }
}

/// Rebuilds an arkworks target field element from the device coefficients.
///
/// The tower nests as `Fp12 = Fp6[w]`, `Fp6 = Fp2[v]`, `Fp2 = Fp[u]`, so the
/// twelve base elements arrive as `c0.c0.c0, c0.c0.c1, c0.c1.c0, ...`.
fn from_device_fq12(value: &DeviceFq12) -> <Bls12_381 as Pairing>::TargetField {
    let base = |index: usize| from_device_fq(&value.coefficients[index]);
    let fq2 = |index: usize| Fq2::new(base(2 * index), base(2 * index + 1));
    let fq6 = |index: usize| {
        ark_bls12_381::Fq6::new(fq2(3 * index), fq2(3 * index + 1), fq2(3 * index + 2))
    };
    ark_bls12_381::Fq12::new(fq6(0), fq6(1))
}

fn cuda_result(code: i32, stage: &'static str) -> Result<(), GpuSmartPcError> {
    if code == 0 {
        Ok(())
    } else {
        Err(GpuSmartPcError::CudaKernel { stage, code })
    }
}

fn icicle_error(error: impl ToString) -> GpuSmartPcError {
    GpuSmartPcError::Icicle(error.to_string())
}

/// Reusable device state for batched Miller loops.
///
/// The opening calls this once per half of every reduction round, so the device
/// selection and the buffers are set up once and reused; allocating per call
/// costs more than the Miller loops themselves for the later, smaller rounds.
pub struct GpuPairing {
    left: DeviceVec<DeviceG1Affine>,
    right: DeviceVec<DeviceG2Affine>,
    scratch: DeviceVec<DeviceFq12>,
    two_inv: DeviceFq,
    capacity: usize,
}

/// Below this many pairs the host finishes sooner than a device round trip.
const HOST_PAIRING_THRESHOLD: usize = 512;

impl GpuPairing {
    /// Selects the device and reserves room for `capacity` pairs.
    pub fn new(device_id: i32, capacity: usize) -> Result<Self, GpuSmartPcError> {
        icicle_runtime::set_device(&Device::new("CUDA", device_id)).map_err(icicle_error)?;
        Ok(Self {
            left: DeviceVec::device_malloc(capacity).map_err(icicle_error)?,
            right: DeviceVec::device_malloc(capacity).map_err(icicle_error)?,
            scratch: DeviceVec::device_malloc(capacity).map_err(icicle_error)?,
            two_inv: to_device_fq(&Fq::from(2_u64).inverse().expect("two is invertible")),
            capacity,
        })
    }

    /// Computes `prod_i e(left[i], right[i])`.
    ///
    /// Small inputs and inputs beyond the reserved capacity fall back to
    /// arkworks, so the result is the same either way.
    pub fn inner_pairing_product(
        &mut self,
        left: &[ArkG1Affine],
        right: &[ArkG2Affine],
    ) -> Result<PairingOutput<Bls12_381>, GpuSmartPcError> {
        assert_eq!(left.len(), right.len());
        if left.is_empty() {
            return Ok(PairingOutput::zero());
        }
        let miller = self.multi_miller_loop(left, right)?;
        Ok(finish(miller))
    }

    fn multi_miller_loop(
        &mut self,
        left: &[ArkG1Affine],
        right: &[ArkG2Affine],
    ) -> Result<<Bls12_381 as Pairing>::TargetField, GpuSmartPcError> {
        let count = left.len();
        if count < HOST_PAIRING_THRESHOLD || count > self.capacity {
            return Ok(Bls12_381::multi_miller_loop(
                left.iter().copied(),
                right.iter().copied(),
            )
            .0);
        }

        let host_left = left.par_iter().map(to_device_g1).collect::<Vec<_>>();
        let host_right = right.par_iter().map(to_device_g2).collect::<Vec<_>>();

        self.left[..count]
            .copy_from_host(HostSlice::from_slice(&host_left))
            .map_err(icicle_error)?;
        self.right[..count]
            .copy_from_host(HostSlice::from_slice(&host_right))
            .map_err(icicle_error)?;

        let mut output = DeviceFq12::default();
        cuda_result(
            unsafe {
                parcomm_multi_miller_loop(
                    self.left.as_ptr().cast(),
                    self.right.as_ptr().cast(),
                    count,
                    (&self.two_inv as *const DeviceFq).cast(),
                    self.scratch.as_mut_ptr().cast(),
                    (&mut output as *mut DeviceFq12).cast(),
                )
            },
            "multi_miller_loop",
        )?;

        Ok(from_device_fq12(&output))
    }
}

fn to_device_g1(point: &ArkG1Affine) -> DeviceG1Affine {
    DeviceG1Affine {
        x: to_device_fq(&point.x),
        y: to_device_fq(&point.y),
    }
}

fn to_device_g2(point: &ArkG2Affine) -> DeviceG2Affine {
    DeviceG2Affine {
        x: to_device_fq2(&point.x),
        y: to_device_fq2(&point.y),
    }
}

/// Turns an accumulated Miller loop output into a pairing product.
fn finish(miller: <Bls12_381 as Pairing>::TargetField) -> PairingOutput<Bls12_381> {
    PairingOutput(
        Bls12_381::final_exponentiation(ark_ec::pairing::MillerLoopOutput(miller))
            .expect("BLS12-381 final exponentiation")
            .0,
    )
}

/// Accumulated Miller loop output for every pair, without the final
/// exponentiation.
///
/// This allocates per call and exists for the correctness check; the opening
/// uses [`GpuPairing`] instead.
pub fn multi_miller_loop_gpu(
    device_id: i32,
    left: &[ArkG1Affine],
    right: &[ArkG2Affine],
) -> Result<<Bls12_381 as Pairing>::TargetField, GpuSmartPcError> {
    assert_eq!(left.len(), right.len());
    let count = left.len();
    if count == 0 {
        return Ok(<Bls12_381 as Pairing>::TargetField::one());
    }
    let mut context = GpuPairing::new(device_id, count)?;
    // The check binary compares against arkworks at every size, so the host
    // fallback must not kick in here.
    let host_left = left.par_iter().map(to_device_g1).collect::<Vec<_>>();
    let host_right = right.par_iter().map(to_device_g2).collect::<Vec<_>>();
    context.left[..count]
        .copy_from_host(HostSlice::from_slice(&host_left))
        .map_err(icicle_error)?;
    context.right[..count]
        .copy_from_host(HostSlice::from_slice(&host_right))
        .map_err(icicle_error)?;
    let mut output = DeviceFq12::default();
    cuda_result(
        unsafe {
            parcomm_multi_miller_loop(
                context.left.as_ptr().cast(),
                context.right.as_ptr().cast(),
                count,
                (&context.two_inv as *const DeviceFq).cast(),
                context.scratch.as_mut_ptr().cast(),
                (&mut output as *mut DeviceFq12).cast(),
            )
        },
        "multi_miller_loop",
    )?;
    Ok(from_device_fq12(&output))
}

/// The tier-one G1 vector and the G2 basis, resident on one device.
///
/// The column rounds of the opening pair halves of these two vectors and then
/// fold both by a challenge. Keeping them on the device makes each round a pair
/// of kernel launches: nothing but the two Fp12 results crosses the bus.
pub struct GpuPairingVectors {
    tier_one: DeviceVec<DeviceG1Affine>,
    basis: DeviceVec<DeviceG2Affine>,
    scratch: DeviceVec<DeviceFq12>,
    two_inv: DeviceFq,
    live: usize,
}

impl GpuPairingVectors {
    /// Uploads both vectors, which must already be affine and equally long.
    pub fn new(
        device_id: i32,
        tier_one: &[ArkG1Affine],
        basis: &[ArkG2Affine],
    ) -> Result<Self, GpuSmartPcError> {
        assert_eq!(tier_one.len(), basis.len());
        let live = tier_one.len();
        icicle_runtime::set_device(&Device::new("CUDA", device_id)).map_err(icicle_error)?;

        let host_tier_one = tier_one.par_iter().map(to_device_g1).collect::<Vec<_>>();
        let host_basis = basis.par_iter().map(to_device_g2).collect::<Vec<_>>();

        let mut device_tier_one = DeviceVec::device_malloc(live).map_err(icicle_error)?;
        device_tier_one
            .copy_from_host(HostSlice::from_slice(&host_tier_one))
            .map_err(icicle_error)?;
        let mut device_basis = DeviceVec::device_malloc(live).map_err(icicle_error)?;
        device_basis
            .copy_from_host(HostSlice::from_slice(&host_basis))
            .map_err(icicle_error)?;

        Ok(Self {
            tier_one: device_tier_one,
            basis: device_basis,
            scratch: DeviceVec::device_malloc(live / 2).map_err(icicle_error)?,
            two_inv: to_device_fq(&Fq::from(2_u64).inverse().expect("two is invertible")),
            live,
        })
    }

    /// Pairs the lower half of each vector against the upper half of the other.
    pub fn cross_pairings(
        &mut self,
    ) -> Result<(PairingOutput<Bls12_381>, PairingOutput<Bls12_381>), GpuSmartPcError> {
        let half = self.live / 2;
        let mut lower = DeviceFq12::default();
        let mut upper = DeviceFq12::default();
        cuda_result(
            unsafe {
                parcomm_pairing_cross(
                    self.tier_one.as_ptr().cast(),
                    self.basis.as_ptr().cast(),
                    half,
                    (&self.two_inv as *const DeviceFq).cast(),
                    self.scratch.as_mut_ptr().cast(),
                    (&mut lower as *mut DeviceFq12).cast(),
                    (&mut upper as *mut DeviceFq12).cast(),
                )
            },
            "pairing_cross",
        )?;
        Ok((
            finish(from_device_fq12(&lower)),
            finish(from_device_fq12(&upper)),
        ))
    }

    /// Folds each vector into its lower half, halving the live length.
    pub fn fold(
        &mut self,
        tier_one_scalar: Fr,
        basis_scalar: Fr,
    ) -> Result<(), GpuSmartPcError> {
        let half = self.live / 2;
        // Double-and-add needs the plain integer rather than the Montgomery form.
        let tier_one_bits = tier_one_scalar.into_bigint().0;
        let basis_bits = basis_scalar.into_bigint().0;
        cuda_result(
            unsafe {
                parcomm_pairing_fold(
                    self.tier_one.as_mut_ptr().cast(),
                    self.basis.as_mut_ptr().cast(),
                    half,
                    tier_one_bits.as_ptr(),
                    basis_bits.as_ptr(),
                )
            },
            "pairing_fold",
        )?;
        self.live = half;
        Ok(())
    }

    /// The single G2 point left once the rounds are done.
    pub fn reduced_basis(&self) -> Result<ArkG2Affine, GpuSmartPcError> {
        let mut point = DeviceG2Affine::default();
        cuda_result(
            unsafe {
                parcomm_pairing_read_basis(
                    self.basis.as_ptr().cast(),
                    0,
                    (&mut point as *mut DeviceG2Affine).cast(),
                )
            },
            "pairing_read_basis",
        )?;
        Ok(ArkG2Affine::new_unchecked(
            from_device_fq2(&point.x),
            from_device_fq2(&point.y),
        ))
    }
}

fn from_device_fq2(value: &DeviceFq2) -> Fq2 {
    Fq2::new(from_device_fq(&value.c0), from_device_fq(&value.c1))
}

extern "C" {
    fn parcomm_pairing_cross(
        tier_one: *const std::ffi::c_void,
        basis: *const std::ffi::c_void,
        half: usize,
        two_inv: *const std::ffi::c_void,
        scratch: *mut std::ffi::c_void,
        lower_output: *mut std::ffi::c_void,
        upper_output: *mut std::ffi::c_void,
    ) -> i32;

    fn parcomm_pairing_fold(
        tier_one: *mut std::ffi::c_void,
        basis: *mut std::ffi::c_void,
        half: usize,
        tier_one_scalar: *const u64,
        basis_scalar: *const u64,
    ) -> i32;

    fn parcomm_pairing_read_basis(
        basis: *const std::ffi::c_void,
        index: usize,
        output: *mut std::ffi::c_void,
    ) -> i32;

    fn parcomm_multi_miller_loop(
        left: *const std::ffi::c_void,
        right: *const std::ffi::c_void,
        count: usize,
        two_inv: *const std::ffi::c_void,
        scratch: *mut std::ffi::c_void,
        output: *mut std::ffi::c_void,
    ) -> i32;
}

/// Times the device path against arkworks for the same inputs.
pub fn benchmark_against_host(
    device_id: i32,
    left: &[ArkG1Affine],
    right: &[ArkG2Affine],
) -> Result<(f64, f64, bool), GpuSmartPcError> {
    let device_started = Instant::now();
    let device = multi_miller_loop_gpu(device_id, left, right)?;
    let device_seconds = device_started.elapsed().as_secs_f64();

    let host_started = Instant::now();
    let host = Bls12_381::multi_miller_loop(left.iter().copied(), right.iter().copied()).0;
    let host_seconds = host_started.elapsed().as_secs_f64();

    Ok((device_seconds, host_seconds, device == host))
}

const _: () = assert!(size_of::<DeviceFq>() == BASE_FIELD_LIMBS * 8);
