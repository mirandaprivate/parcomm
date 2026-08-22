pub mod i8_smart_pc;
pub mod pairing;
pub mod parcomm;
pub mod smart_pc;
pub mod transcript;

pub use pairing::{benchmark_against_host, multi_miller_loop_gpu, GpuPairing, GpuPairingVectors};
pub use parcomm::{
    dense_packed_witness, DenseWitness, GpuReducedVectors, GpuStageProfile, GpuWitness,
    WitnessLayout, PACKING_STRIDE,
};

pub use i8_smart_pc::{
    commit_i8_four_gpu, I8GpuCommitment, I8GpuCommitmentProfile, I8GpuShardProfile,
};
pub use smart_pc::{
    commit_full_four_gpu, initialize_cuda_backend, GpuCommitment, GpuCommitmentProfile,
    GpuShardProfile, GpuSmartPcError,
};
pub use transcript::FiatShamir;
