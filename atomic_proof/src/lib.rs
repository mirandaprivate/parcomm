//! Atomic proof library for zero-knowledge proofs
pub mod atomic_protocol;
pub mod protocols;

pub use atomic_protocol::{
    AtomicMatProtocol, AtomicMatProtocolInput, EvalWithPointMat, EvalWithPointVec, MatOp,
};

pub use protocols::hadamard::HadamardReduction;
pub use protocols::mul::MatMulReduction;
pub use protocols::BatchPoints;
pub use protocols::Hadamard;
pub use protocols::LinComb;
pub use protocols::LiteBullet;
pub use protocols::MatMul;
