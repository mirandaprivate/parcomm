//! Full-extension QAFold with eight-to-one packing of F_65521 witnesses.
//!
//! `packed` authenticates the original Boolean MLE through one full-extension
//! QAFold opening and recursive Decide. The separate short-code backend retains
//! its original base-field coding policy. Field size alone is not a security claim.

mod brakefold;
pub mod full_extension;
pub mod packed;
pub mod qafold;
pub mod qafold_short;
pub mod qafold_short_pc;

pub use full_extension::{FullPrime65521Ext8, FullQaFoldPC};
pub use qafold_short::{
    Prime65521, Prime65521Ext8, ShortBaseField, ShortExtensionField, ShortFieldElement,
};
pub use qafold_short_pc::ShortQaFoldPC;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    EncodingError,
    HashingError,
    InvalidCommitment,
    InvalidNumberOfVariables,
    InvalidParameters(String),
    TranscriptError,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for Error {}
